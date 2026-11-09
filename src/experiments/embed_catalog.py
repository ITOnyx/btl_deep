"""
Phase 2 Step B: Extract Image Embeddings for In-Scope Catalog.
Processes images in shards of 2000 items, supports resume, and merges output.
"""
import os
import sys
import time
import json
import gc
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, ".")
from src.model_siglip import load_siglip_inference

SHARD_SIZE = 2000
BATCH_SIZE = 32

def main():
    t_start = time.time()
    catalog_path = pathlib.Path("data/catalog/catalog.parquet")
    df = pd.read_parquet(catalog_path)
    
    # Filter only in_scope items
    df_in_scope = df[df["in_scope"]].reset_index(drop=True)
    total_items = len(df_in_scope)
    print(f"Total in-scope items to embed: {total_items:,}")

    index_dir = pathlib.Path("index")
    index_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = index_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Total shards: {num_shards} | Remaining to process: {len(shards_to_process)}")

    if shards_to_process:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP Vision Tower on {device} (FP16)...")
        model = load_siglip_inference(
            checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
            device=device,
            fp16=(device == "cuda")
        )
        processor = model.processor

        for s_idx in shards_to_process:
            t0_shard = time.time()
            s_start = s_idx * SHARD_SIZE
            s_end = min(s_start + SHARD_SIZE, total_items)
            shard_df = df_in_scope.iloc[s_start:s_end].reset_index(drop=True)
            n_shard = len(shard_df)

            shard_npy = shards_dir / f"shard_{s_idx:03d}.npy"
            shard_ids = shards_dir / f"shard_{s_idx:03d}_ids.json"

            print(f"\n--- Processing Shard {s_idx:03d}/{num_shards - 1:03d} ({n_shard} images, items {s_start}..{s_end-1}) ---")
            shard_embs_list = []

            for b_idx in range(0, n_shard, BATCH_SIZE):
                b_df = shard_df.iloc[b_idx:b_idx + BATCH_SIZE]
                batch_images = []
                for p in b_df["path"]:
                    try:
                        with Image.open(p) as img:
                            batch_images.append(img.convert("RGB"))
                    except Exception as e:
                        print(f"Warning: failed reading {p}: {e}")
                        batch_images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

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
            print(f"Shard {s_idx:03d} finished in {dt_shard:.1f}s ({fps:.1f} img/s).")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Merge all shards
    print("\n==================================================")
    print("Merging shards into index/image_embeddings.npy...")
    print("==================================================")
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

    emb_path = index_dir / "image_embeddings.npy"
    np.save(emb_path, final_embs)
    print(f"Saved merged embeddings: {emb_path} ({os.path.getsize(emb_path)/(1024**2):.2f} MB, shape: {final_embs.shape})")

    # Add idx to each record
    for idx, rec in enumerate(all_id_records):
        rec["idx"] = idx

    id_map_path = index_dir / "id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(all_id_records, f, indent=2, ensure_ascii=False)
    print(f"Saved ID map: {id_map_path} ({len(all_id_records):,} items)")

    # Save splits.json
    splits_dict = {"test": [], "unknown": [], "train": [], "val": []}
    for rec in all_id_records:
        s = rec.get("split", "unknown")
        if s in splits_dict:
            splits_dict[s].append(rec["idx"])
        else:
            splits_dict[s] = [rec["idx"]]

    splits_path = index_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=2)
    print(f"Saved splits index: {splits_path}")
    print("Splits counts in final index:", {k: len(v) for k, v in splits_dict.items() if len(v) > 0})

    print(f"\nPhase 2 Step B completed successfully in {time.time() - t_start:.1f}s!")

if __name__ == "__main__":
    main()
