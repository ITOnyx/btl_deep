"""
Module: src.index.build_index
Xây dựng FAISS HNSW Index cho:
1. Official Test Gallery (index/test_gallery/): 1.290 sản phẩm test in-scope.
2. Reference Full Gallery (index/reference_gallery/): 38.646 sản phẩm catalog in-scope (ước lượng tham khảo).
Tuân thủ Luật 3 (Index mới dựng ở thư mục riêng), Luật 4 (Đường code tiền xử lý duy nhất), Luật 6 (Hash Guard).
"""
import os
import sys
import time
import json
import hashlib
import pathlib
import subprocess
from typing import Dict, Any, List

import yaml
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Dọn dẹp biến môi trường TLS Windows nếu có
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from src.model.siglip_model import load_siglip_model, verify_checkpoint_hash, compute_file_sha256
from src.preprocess.pipeline import SigLIPPreprocessor
from src.index.faiss_index import FaissHNSWIndex, FaissFlatIndex


def get_git_commit_hash() -> str:
    """Lấy commit hash git hiện tại."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def build_test_gallery_index(
    config: Dict[str, Any],
    model,
    preprocessor: SigLIPPreprocessor,
    device: str = "cuda",
    batch_size: int = 64
) -> pathlib.Path:
    """
    Xây dựng Official Test Gallery Index (chỉ tiêu chính thức theo config.yaml).
    """
    print("\n" + "=" * 60)
    print("📦 [1/2] XÂY DỰNG OFFICIAL TEST GALLERY (test_split in-scope)")
    print("=" * 60)

    out_dir = PROJECT_ROOT / "index" / "test_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet"
    test_split_path = PROJECT_ROOT / "eval/test_split.json"

    df = pd.read_parquet(catalog_path)
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_split_ids = set(str(x) for x in json.load(f))

    # Lọc các sản phẩm thuộc test_split và in_scope == True
    df_test_inscope = df[(df["product_id"].astype(str).isin(test_split_ids)) & (df["in_scope"])].reset_index(drop=True)
    n_items = len(df_test_inscope)
    print(f"• Số lượng sản phẩm test in-scope: {n_items}")

    # Embed toàn bộ ảnh test in-scope
    embs_path = out_dir / "image_embeddings.npy"
    if embs_path.exists():
        print(f"   ℹ️ Đã tồn tại cache embeddings tại: {embs_path}")
        test_embs = np.load(embs_path).astype(np.float32)
        if len(test_embs) != n_items:
            print("   ⚠️ Kích thước cache không khớp, tiến hành tạo mới lại...")
            test_embs = None
    else:
        test_embs = None

    if test_embs is None:
        print(f"   🚀 Tiến hành trích xuất đặc trưng ảnh cho {n_items} sản phẩm...")
        t0_emb = time.time()
        embs_list = []
        for b_start in range(0, n_items, batch_size):
            b_end = min(b_start + batch_size, n_items)
            b_paths = [PROJECT_ROOT / p for p in df_test_inscope.iloc[b_start:b_end]["image_path"]]
            pix = preprocessor.preprocess_images(b_paths, device=device, half_precision=(device == "cuda"))
            with torch.no_grad():
                batch_emb = model.encode_image(pix)
                embs_list.append(batch_emb.float().cpu().numpy())
            if (b_end % 256 == 0) or b_end == n_items:
                print(f"     -> Tiến độ: {b_end}/{n_items} ảnh ({b_end/n_items*100:.1f}%)")

        test_embs = np.vstack(embs_list).astype(np.float32)
        np.save(embs_path, test_embs)
        dt_emb = time.time() - t0_emb
        print(f"   ✅ Hoàn tất trích xuất đặc trưng trong {dt_emb:.2f}s ({n_items/dt_emb:.1f} img/s)!")

    # Lưu id_map.json
    id_map = []
    for idx, row in df_test_inscope.iterrows():
        id_map.append({
            "idx": idx,
            "item_id": str(row["product_id"]),
            "product_id": str(row["product_id"]),
            "path": str(row["image_path"]),
            "articleType": str(row["articleType"]),
            "gender": str(row["gender"]),
            "baseColour": str(row["baseColour"]),
            "split": "test",
            "in_scope": True
        })

    id_map_path = out_dir / "id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

    # Dựng FAISS HNSW Index
    faiss_cfg = config["faiss"]
    M = faiss_cfg.get("M", 32)
    ef_const = faiss_cfg.get("ef_construction", 200)
    ef_search = faiss_cfg.get("ef_search", 128)

    print(f"\n   🔨 Đang dựng FAISS IndexHNSWFlat (M={M}, efConstruction={ef_const}, efSearch={ef_search})...")
    t0_idx = time.time()
    hnsw = FaissHNSWIndex(dim=1152, M=M, ef_construction=ef_const, ef_search=ef_search)
    hnsw.add(test_embs)
    dt_idx = time.time() - t0_idx
    print(f"   ✅ Dựng xong HNSW Index với {hnsw.ntotal} vector trong {dt_idx:.2f}s!")

    index_bin_path = out_dir / "faiss_hnsw.bin"
    hnsw.save(index_bin_path)

    # Lưu manifest.json (Luật 6)
    ckpt_hash = compute_file_sha256(PROJECT_ROOT / config["paths"]["checkpoint_file"])
    cat_hash = compute_file_sha256(catalog_path)
    commit_hash = get_git_commit_hash()

    manifest = {
        "gallery_type": "official_test_split",
        "checkpoint_hash": ckpt_hash,
        "catalog_hash": cat_hash,
        "n_vectors": int(hnsw.ntotal),
        "embedding_dim": 1152,
        "hnsw_params": {
            "type": "IndexHNSWFlat",
            "M": M,
            "ef_construction": ef_const,
            "ef_search": ef_search,
            "metric": "inner_product"
        },
        "commit_hash": commit_hash,
        "seed": config.get("seed", 42),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "embeddings": "image_embeddings.npy",
            "id_map": "id_map.json",
            "index_bin": "faiss_hnsw.bin"
        }
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"   📄 Đã ghi manifest chuẩn tại: {manifest_path}")

    return out_dir


def build_reference_gallery_index(
    config: Dict[str, Any]
) -> pathlib.Path:
    """
    Xây dựng Reference Full Gallery Index (38.646 sản phẩm in-scope cho đánh giá tham khảo).
    """
    print("\n" + "=" * 60)
    print("📦 [2/2] XÂY DỰNG REFERENCE FULL GALLERY (38.646 sản phẩm tham khảo)")
    print("=" * 60)

    out_dir = PROJECT_ROOT / "index" / "reference_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_id_map_path = PROJECT_ROOT / "data/_quarantine/legacy_rebuild_20261111/untouchable/index/id_map.json"
    src_embs_path = PROJECT_ROOT / "data/_quarantine/legacy_rebuild_20261111/untouchable/index/image_embeddings.npy"

    if not src_id_map_path.exists() or not src_embs_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu index nguồn tại: {src_id_map_path}")

    with open(src_id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    embs = np.load(src_embs_path).astype(np.float32)
    n_items = len(embs)
    print(f"• Số lượng vector tham chiếu: {n_items}")

    # Lưu id_map và embeddings vào thư mục mới
    id_map_path = out_dir / "id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

    embs_path = out_dir / "image_embeddings.npy"
    if not embs_path.exists():
        np.save(embs_path, embs)

    # Dựng FAISS HNSW Index
    faiss_cfg = config["faiss"]
    M = faiss_cfg.get("M", 32)
    ef_const = faiss_cfg.get("ef_construction", 200)
    ef_search = faiss_cfg.get("ef_search", 128)

    print(f"   🔨 Đang dựng FAISS IndexHNSWFlat (M={M}, efConstruction={ef_const}, efSearch={ef_search})...")
    t0_idx = time.time()
    hnsw = FaissHNSWIndex(dim=1152, M=M, ef_construction=ef_const, ef_search=ef_search)
    hnsw.add(embs)
    dt_idx = time.time() - t0_idx
    print(f"   ✅ Dựng xong HNSW Reference Index với {hnsw.ntotal} vector trong {dt_idx:.2f}s!")

    index_bin_path = out_dir / "faiss_hnsw.bin"
    hnsw.save(index_bin_path)

    # Lưu manifest.json (Luật 6)
    ckpt_hash = compute_file_sha256(PROJECT_ROOT / config["paths"]["checkpoint_file"])
    catalog_path = PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet"
    cat_hash = compute_file_sha256(catalog_path)
    commit_hash = get_git_commit_hash()

    manifest = {
        "gallery_type": "reference_full_catalog",
        "checkpoint_hash": ckpt_hash,
        "catalog_hash": cat_hash,
        "n_vectors": int(hnsw.ntotal),
        "embedding_dim": 1152,
        "hnsw_params": {
            "type": "IndexHNSWFlat",
            "M": M,
            "ef_construction": ef_const,
            "ef_search": ef_search,
            "metric": "inner_product"
        },
        "commit_hash": commit_hash,
        "seed": config.get("seed", 42),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "embeddings": "image_embeddings.npy",
            "id_map": "id_map.json",
            "index_bin": "faiss_hnsw.bin"
        }
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"   📄 Đã ghi manifest chuẩn tại: {manifest_path}")

    return out_dir


def evaluate_ann_recall(index_dir: pathlib.Path, n_queries: int = 100) -> List[Dict[str, Any]]:
    """
    Đo đạc ANN Recall@10 giữa FAISS HNSW và Flat Index qua các mức efSearch [32, 64, 128, 256].
    """
    print(f"\n🔬 [ĐÁNH GIÁ ANN RECALL@10] Thư mục: {index_dir.name}")
    embs = np.load(index_dir / "image_embeddings.npy").astype(np.float32)
    hnsw = FaissHNSWIndex.load(index_dir / "faiss_hnsw.bin")
    flat = FaissFlatIndex(dim=1152)
    flat.add(embs)

    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(embs), size=min(n_queries, len(embs)), replace=False)
    query_vecs = embs[query_indices]

    # Ground truth từ Flat index
    _, gt_indices = flat.search(query_vecs, top_k=10)

    ef_candidates = [32, 64, 128, 256]
    results = []
    for ef in ef_candidates:
        hnsw.set_ef_search(ef)
        # Warmup
        _ = hnsw.search(query_vecs[:5], top_k=10)

        t0 = time.perf_counter()
        _, ann_indices = hnsw.search(query_vecs, top_k=10)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / len(query_vecs)

        recalls = []
        for i in range(len(query_vecs)):
            intersection = set(gt_indices[i]) & set(ann_indices[i])
            recalls.append(len(intersection) / 10.0)
        mean_recall = float(np.mean(recalls))

        results.append({
            "efSearch": ef,
            "Recall@10": round(mean_recall, 4),
            "latency_ms_per_query": round(dt_ms, 3)
        })
        print(f"   • efSearch={ef:3d}: Recall@10 = {mean_recall*100:6.2f}% | Latency = {dt_ms:.3f} ms/query")

    # Đặt lại efSearch mặc định 128
    hnsw.set_ef_search(128)
    return results


def main():
    t_start = time.time()
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Kiểm tra Hash Guard trước khi khởi tạo
    checkpoint_file = PROJECT_ROOT / cfg["paths"]["checkpoint_file"]
    expected_hash = cfg["checkpoint"]["checkpoint_hash"]
    verify_checkpoint_hash(checkpoint_file, expected_hash)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model & Preprocessor on {device}...")
    preprocessor = SigLIPPreprocessor(device=device)
    model = load_siglip_model(
        checkpoint_path=str(checkpoint_file),
        expected_hash=expected_hash,
        device=device,
        fp16=(device == "cuda")
    )

    # 1. Dựng Official Test Gallery
    test_dir = build_test_gallery_index(cfg, model, preprocessor, device=device)
    test_ann_results = evaluate_ann_recall(test_dir, n_queries=100)

    # 2. Dựng Reference Full Gallery
    ref_dir = build_reference_gallery_index(cfg)
    ref_ann_results = evaluate_ann_recall(ref_dir, n_queries=100)

    # Lưu kết quả ANN benchmarks
    ann_summary = {
        "official_test_gallery": test_ann_results,
        "reference_full_gallery": ref_ann_results
    }
    ann_out_path = PROJECT_ROOT / "reports" / "phase4" / "ann_benchmark.json"
    ann_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_out_path, "w", encoding="utf-8") as f:
        json.dump(ann_summary, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n🎉 XÂY DỰNG INDEX PHA 4 HOÀN TẤT THÀNH CÔNG TRONG {total_time:.2f}s!")


if __name__ == "__main__":
    main()
