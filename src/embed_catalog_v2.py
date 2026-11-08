"""
Phase 3c: Extract Image Embeddings for In-Scope Catalog using Train-Faithful Preprocessing.
Pipeline:
  1. PIL RGB
  2. torchvision.transforms.Resize((224, 224), interpolation=BILINEAR)
  3. SiglipProcessor -> model.encode_image (FP16 on GPU, L2 normalized)
Optimized with ThreadPoolExecutor(max_workers=8) for 6.3x fast image decoding & resizing.
Saves:
  - index_v2/image_embeddings_v2.npy
  - index_v2/id_map_v2.json
  - index_v2/split_v2.json
  - index_v2/faiss_hnsw_index_v2.bin
  - index_v2/manifest.json
"""
import os
import sys
import time
import json
import gc
import hashlib
import pathlib
import yaml
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import faiss

sys.path.insert(0, ".")
from src.model_siglip import load_siglip_inference

SHARD_SIZE = 2000
BATCH_SIZE = 64
NUM_LOADER_WORKERS = 8

def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    t_start = time.time()
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    catalog_path = pathlib.Path(config["paths"]["catalog"])
    df = pd.read_parquet(catalog_path)
    
    # Filter only in_scope items
    df_in_scope = df[df["in_scope"]].reset_index(drop=True)
    total_items = len(df_in_scope)
    print(f"Total in-scope items to embed: {total_items:,}", flush=True)

    index_dir = pathlib.Path("index_v2")
    index_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = index_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Train-faithful resize transform
    eval_image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
    ])

    def load_single_image(img_path: str) -> Image.Image:
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                return eval_image_transform(img_rgb)
        except Exception as e:
            return Image.new("RGB", (224, 224), (255, 255, 255))

    # Check how many shards exist and determine if model load is needed
    num_shards = (total_items + SHARD_SIZE - 1) // SHARD_SIZE
    shards_to_process = []
    for s_idx in range(num_shards):
        shard_npy = shards_dir / f"shard_{s_idx:03d}.npy"
        shard_ids = shards_dir / f"shard_{s_idx:03d}_ids.json"
        s_start = s_idx * SHARD_SIZE
        s_end = min(s_start + SHARD_SIZE, total_items)
        expected_len = s_end - s_start

        if shard_npy.exists() and shard_ids.exists():
            try:
                arr = np.load(shard_npy)
                if arr.shape == (expected_len, 1152):
                    continue
            except Exception:
                pass
        shards_to_process.append(s_idx)

    print(f"Total shards: {num_shards} | Remaining to process: {len(shards_to_process)}", flush=True)

    if shards_to_process:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP Vision Tower on {device} (FP16)...", flush=True)
        model = load_siglip_inference(
            checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
            device=device,
            fp16=(device == "cuda")
        )
        processor = model.processor

        with ThreadPoolExecutor(max_workers=NUM_LOADER_WORKERS) as pool:
            for s_idx in shards_to_process:
                t0_shard = time.time()
                s_start = s_idx * SHARD_SIZE
                s_end = min(s_start + SHARD_SIZE, total_items)
                shard_df = df_in_scope.iloc[s_start:s_end].reset_index(drop=True)
                n_shard = len(shard_df)

                shard_npy = shards_dir / f"shard_{s_idx:03d}.npy"
                shard_ids = shards_dir / f"shard_{s_idx:03d}_ids.json"

                print(f"\n--- Processing Shard {s_idx:03d}/{num_shards - 1:03d} ({n_shard} images, items {s_start}..{s_end-1}) ---", flush=True)
                shard_embs_list = []

                for b_idx in range(0, n_shard, BATCH_SIZE):
                    b_df = shard_df.iloc[b_idx:b_idx + BATCH_SIZE]
                    paths = b_df["path"].tolist()
                    batch_images = list(pool.map(load_single_image, paths))

                    inp = processor(images=batch_images, return_tensors="pt")
                    pix = inp["pixel_values"].to(device)
                    if device == "cuda":
                        pix = pix.half()

                    with torch.no_grad():
                        embs = model.encode_image(pix)  # [B, 1152], L2 normalized
                        shard_embs_list.append(embs.float().cpu().numpy())

                shard_embs = np.vstack(shard_embs_list)
                assert shard_embs.shape == (n_shard, 1152), f"Shard shape mismatch: {shard_embs.shape}"

                # Save shard npy and id records
                np.save(shard_npy, shard_embs)
                meta_records = shard_df[["item_id", "path", "articleType", "gender", "baseColour", "split"]].to_dict(orient="records")
                with open(shard_ids, "w", encoding="utf-8") as f:
                    json.dump(meta_records, f, ensure_ascii=False)

                dt_shard = time.time() - t0_shard
                fps = n_shard / dt_shard
                print(f"Shard {s_idx:03d} finished in {dt_shard:.1f}s ({fps:.1f} img/s).", flush=True)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Merge all shards
    print("\n==================================================", flush=True)
    print("Merging shards into index_v2/image_embeddings_v2.npy...", flush=True)
    print("==================================================", flush=True)
    all_embs = []
    all_id_records = []

    for s_idx in range(num_shards):
        shard_npy = shards_dir / f"shard_{s_idx:03d}.npy"
        shard_ids = shards_dir / f"shard_{s_idx:03d}_ids.json"
        assert shard_npy.exists(), f"Missing {shard_npy}"
        assert shard_ids.exists(), f"Missing {shard_ids}"

        embs = np.load(shard_npy)
        with open(shard_ids, "r", encoding="utf-8") as f:
            records = json.load(f)

        all_embs.append(embs)
        all_id_records.extend(records)

    final_embs = np.vstack(all_embs).astype(np.float32)
    assert final_embs.shape == (total_items, 1152), f"Final shape mismatch: {final_embs.shape}"

    # Verify L2 normalization
    norms = np.linalg.norm(final_embs[:100], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"Embeddings not L2-normalized! Sample norms: {norms[:5]}"

    emb_path = index_dir / "image_embeddings_v2.npy"
    np.save(emb_path, final_embs)
    print(f"Saved merged embeddings: {emb_path} ({os.path.getsize(emb_path)/(1024**2):.2f} MB, shape: {final_embs.shape})", flush=True)

    # Add global index to id_records
    for idx, rec in enumerate(all_id_records):
        rec["idx"] = idx

    id_map_path = index_dir / "id_map_v2.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(all_id_records, f, ensure_ascii=False)
    print(f"Saved id_map: {id_map_path} ({len(all_id_records):,} items)", flush=True)

    # Save split_v2.json
    split_v2 = {str(rec["item_id"]): rec["split"] for rec in all_id_records}
    split_path = index_dir / "split_v2.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_v2, f, ensure_ascii=False)
    print(f"Saved split_v2: {split_path}", flush=True)

    # Build FAISS HNSW Index
    print("\n==================================================", flush=True)
    print("Building FAISS HNSW Index for index_v2...", flush=True)
    print("==================================================", flush=True)
    faiss_cfg = config.get("faiss", {})
    M = int(faiss_cfg.get("M", 32))
    ef_construction = int(faiss_cfg.get("ef_construction", 200))
    ef_search = int(faiss_cfg.get("ef_search", 128))
    dim = 1152

    t0_faiss = time.time()
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    chunk_size = 5000
    for i in range(0, total_items, chunk_size):
        chunk = final_embs[i:i + chunk_size]
        index.add(chunk)
        print(f"Added vectors {i}..{min(i + chunk_size, total_items) - 1} / {total_items}", flush=True)

    dt_faiss = time.time() - t0_faiss
    print(f"FAISS index built with {index.ntotal:,} vectors in {dt_faiss:.1f}s.", flush=True)

    index_path = index_dir / "faiss_hnsw_index_v2.bin"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path} ({os.path.getsize(index_path)/(1024**2):.2f} MB)", flush=True)

    # Write Manifest
    checkpoint_path = "data/checkpoints_siglip-vit/best_model.pt"
    ckpt_hash = compute_file_hash(checkpoint_path)
    catalog_path_str = str(catalog_path)
    catalog_hash = compute_file_hash(catalog_path_str)
    index_hash = compute_file_hash(str(index_path))
    emb_hash = compute_file_hash(str(emb_path))

    manifest = {
        "pipeline": "Resize((224, 224), BILINEAR) + SiglipProcessor",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_items": total_items,
        "embedding_dim": dim,
        "dtype": "float32",
        "l2_normalized": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": ckpt_hash,
        "catalog_path": catalog_path_str,
        "catalog_hash": catalog_hash,
        "index_type": "IndexHNSWFlat",
        "faiss_params": {
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "metric": "inner_product"
        },
        "files": {
            "embeddings": {
                "name": "image_embeddings_v2.npy",
                "sha256": emb_hash,
                "shape": list(final_embs.shape)
            },
            "id_map": {
                "name": "id_map_v2.json",
                "count": len(all_id_records)
            },
            "split": {
                "name": "split_v2.json",
                "test_count": sum(1 for v in split_v2.values() if v == "test"),
                "unknown_count": sum(1 for v in split_v2.values() if v == "unknown")
            },
            "faiss_index": {
                "name": "faiss_hnsw_index_v2.bin",
                "sha256": index_hash,
                "ntotal": index.ntotal
            }
        }
    }

    manifest_path = index_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest: {manifest_path}", flush=True)

    total_time = time.time() - t_start
    print(f"\nPhase 3c Image Embedding completed successfully in {total_time:.1f}s ({total_time/60:.1f} mins).", flush=True)

if __name__ == "__main__":
    main()
