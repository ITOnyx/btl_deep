"""
Module: src.eval.sanity_check
Kiểm định bắt buộc Pha 4 theo đúng quy định trong AGENTS.md (Mục E) và config.yaml:
1. 0 ID trùng trong index id_map (max_duplicate_ids == 0)
2. 0 ảnh test lọt vào dữ liệu train (max_test_leak_into_train == 0)
3. Truy vấn lại một ảnh phải trả chính nó ở rank 1 (self_query_rank1 == True, cosine >= 0.9999)
4. Số vector khớp manifest (n_vectors_match_manifest == True)
"""
import os
import sys
import json
import time
import pathlib
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.index.faiss_index import FaissHNSWIndex
from src.model.siglip_model import verify_checkpoint_hash


def run_sanity_checks(index_dir_rel: str = "index/test_gallery") -> Dict[str, Any]:
    """
    Thực hiện kiểm định 4 tiêu chí Sanity Check trên thư mục index chỉ định.
    """
    index_dir = PROJECT_ROOT / index_dir_rel
    print("=" * 70)
    print(f"🔍 BẮT ĐẦU SANITY CHECK PHA 4 TRÊN: {index_dir_rel}")
    print("=" * 70)

    # 1. Đọc config.yaml
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Nạp các tệp của index
    embs_path = index_dir / "image_embeddings.npy"
    id_map_path = index_dir / "id_map.json"
    bin_path = index_dir / "faiss_hnsw.bin"
    manifest_path = index_dir / "manifest.json"

    for p in [embs_path, id_map_path, bin_path, manifest_path]:
        if not p.exists():
            raise FileNotFoundError(f"Thiếu tệp bắt buộc: {p}")

    embs = np.load(embs_path).astype(np.float32)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    hnsw = FaissHNSWIndex.load(bin_path)

    n_embs = len(embs)
    n_id_map = len(id_map)
    n_hnsw = hnsw.ntotal
    n_manifest = manifest["n_vectors"]

    # -------------------------------------------------------------------------
    # TIÊU CHÍ 1: 0 ID trùng (max_duplicate_ids == 0)
    # -------------------------------------------------------------------------
    print("\n[Tiêu chí 1/4] Kiểm tra ID trùng trong id_map...")
    all_pids = [str(item["item_id"]) for item in id_map]
    unique_pids = set(all_pids)
    num_duplicates = len(all_pids) - len(unique_pids)
    print(f"   • Tổng số phần tử: {len(all_pids)} | Số ID độc nhất: {len(unique_pids)}")
    print(f"   • Số lượng ID trùng: {num_duplicates}")
    c1_passed = (num_duplicates == 0)
    assert c1_passed, f"Phát hiện {num_duplicates} ID trùng lặp trong id_map!"
    print("   ✅ Tiêu chí 1 ĐẠT: 0 ID trùng lặp.")

    # -------------------------------------------------------------------------
    # TIÊU CHÍ 2: 0 ảnh test lọt vào dữ liệu train (max_test_leak_into_train == 0)
    # -------------------------------------------------------------------------
    print("\n[Tiêu chí 2/4] Kiểm tra rò rỉ tập test vào dữ liệu train...")
    test_split_path = PROJECT_ROOT / "eval/test_split.json"
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_split_set = set(str(x) for x in json.load(f))

    catalog_path = PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet"
    df_cat = pd.read_parquet(catalog_path)
    train_ids = set(df_cat[df_cat["split"] == "train"]["product_id"].astype(str))

    leakage_set = test_split_set.intersection(train_ids)
    num_leak = len(leakage_set)
    print(f"   • Kích thước tập Test chính thức: {len(test_split_set)}")
    print(f"   • Kích thước tập Train: {len(train_ids)}")
    print(f"   • Số ảnh test rò rỉ vào train: {num_leak}")
    c2_passed = (num_leak == 0)
    assert c2_passed, f"LỖI RÒ RỈ DỮ LIỆU: Có {num_leak} sản phẩm test nằm trong tập train!"
    print("   ✅ Tiêu chí 2 ĐẠT: Không có bất kỳ ảnh test nào lọt vào tập train.")

    # -------------------------------------------------------------------------
    # TIÊU CHÍ 3: Truy vấn lại một ảnh phải trả chính nó ở rank 1 (cosine >= 0.9999)
    # -------------------------------------------------------------------------
    print("\n[Tiêu chí 3/4] Kiểm tra Self-Query Rank 1 (Cosine >= 0.9999)...")
    # Lấy 50 mẫu ngẫu nhiên (seed cố định 42)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(n_embs, size=min(50, n_embs), replace=False)
    sample_query_vecs = embs[sample_indices]

    scores, pred_indices = hnsw.search(sample_query_vecs, top_k=5)

    rank1_match_count = 0
    cosine_values = []
    for i, target_idx in enumerate(sample_indices):
        top1_idx = pred_indices[i][0]
        top1_score = scores[i][0]
        cosine_values.append(float(top1_score))
        if top1_idx == target_idx and top1_score >= 0.9999:
            rank1_match_count += 1

    min_self_cos = min(cosine_values)
    mean_self_cos = float(np.mean(cosine_values))
    print(f"   • Số mẫu thử nghiệm self-query: {len(sample_indices)}")
    print(f"   • Số mẫu trả về đúng chính mình ở Rank 1 (cos >= 0.9999): {rank1_match_count}/{len(sample_indices)}")
    print(f"   • Min Cosine: {min_self_cos:.6f} | Mean Cosine: {mean_self_cos:.6f}")
    c3_passed = (rank1_match_count == len(sample_indices))
    assert c3_passed, f"Chỉ có {rank1_match_count}/{len(sample_indices)} mẫu đạt rank 1 cos >= 0.9999!"
    print("   ✅ Tiêu chí 3 ĐẠT: 100% truy vấn lại trả đúng chính ảnh đó ở Rank 1 với Cosine >= 0.9999.")

    # -------------------------------------------------------------------------
    # TIÊU CHÍ 4: Số vector khớp manifest (n_vectors_match_manifest == True)
    # -------------------------------------------------------------------------
    print("\n[Tiêu chí 4/4] Kiểm tra tính nhất quán số lượng vector với manifest...")
    print(f"   • Vector embeddings shape[0]: {n_embs}")
    print(f"   • Số phần tử trong id_map:    {n_id_map}")
    print(f"   • Tổng số vector trong HNSW:   {n_hnsw}")
    print(f"   • Số vector trong manifest:     {n_manifest}")

    c4_passed = (n_embs == n_id_map == n_hnsw == n_manifest)
    assert c4_passed, (
        f"Lệch số lượng vector: embs={n_embs}, id_map={n_id_map}, "
        f"hnsw={n_hnsw}, manifest={n_manifest}"
    )
    print("   ✅ Tiêu chí 4 ĐẠT: Số lượng vector hoàn toàn đồng nhất trên toàn bộ các tệp.")

    # -------------------------------------------------------------------------
    # KIỂM TRA HASH GUARD (Luật 6)
    # -------------------------------------------------------------------------
    print("\n[Bổ trợ] Kiểm tra Hash Guard (Luật 6) trong manifest...")
    ckpt_hash = manifest["checkpoint_hash"]
    expected_hash = cfg["checkpoint"]["checkpoint_hash"]
    hash_match = (ckpt_hash.lower() == expected_hash.lower())
    assert hash_match, f"Hash checkpoint trong manifest không khớp: {ckpt_hash} != {expected_hash}"
    print("   ✅ Hash Guard trong manifest trùng khớp 100%.")

    all_passed = c1_passed and c2_passed and c3_passed and c4_passed and hash_match

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_dir": index_dir_rel,
        "gallery_type": manifest.get("gallery_type"),
        "status": "PASSED" if all_passed else "FAILED",
        "checks": {
            "c1_zero_duplicate_ids": {
                "passed": c1_passed,
                "num_duplicates": num_duplicates,
                "total_ids": len(all_pids)
            },
            "c2_zero_test_leak_into_train": {
                "passed": c2_passed,
                "num_leak": num_leak,
                "test_set_size": len(test_split_set),
                "train_set_size": len(train_ids)
            },
            "c3_self_query_rank1": {
                "passed": c3_passed,
                "tested_samples": len(sample_indices),
                "rank1_matches": rank1_match_count,
                "min_cosine": min_self_cos,
                "mean_cosine": mean_self_cos,
                "target_threshold": 0.9999
            },
            "c4_vector_count_consistency": {
                "passed": c4_passed,
                "n_embeddings": n_embs,
                "n_id_map": n_id_map,
                "n_hnsw": n_hnsw,
                "n_manifest": n_manifest
            },
            "hash_guard": {
                "passed": hash_match,
                "checkpoint_hash": ckpt_hash,
                "expected_hash": expected_hash
            }
        }
    }

    report_out_path = PROJECT_ROOT / "reports" / "phase4" / f"sanity_check_{pathlib.Path(index_dir_rel).name}.json"
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Đã lưu báo cáo Sanity Check tại: {report_out_path}")
    print("\n" + "=" * 70)
    print("⭐ TOÀN BỘ 4 TIÊU CHÍ SANITY CHECK PHA 4 ĐỀU ĐẠT CHUẨN XUẤT SẮC!")
    print("=" * 70)
    return report


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "index/test_gallery"
    run_sanity_checks(target)
