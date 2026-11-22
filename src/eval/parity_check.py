"""
Module: src.eval.parity_check
Kiểm định tiền xử lý và nạp checkpoint (Pha 3 - Luật 4 & Luật 6).
Tái mã hóa 200 ảnh mẫu đối chứng từ checkpoint đã lưu:
Yêu cầu: Cosine Similarity nhỏ nhất phải >= 0.999. Nếu không, dừng ngay lập tức.
"""
import os
import sys
import json
import time
import argparse
import pathlib
from typing import Dict, Any, List

import yaml
import torch
import numpy as np

# Thiết lập đường dẫn root
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.siglip_model import load_siglip_model, verify_checkpoint_hash
from src.preprocess.pipeline import SigLIPPreprocessor


def run_parity_check(
    config_path: str = "config.yaml",
    n_samples: int = 200,
    batch_size: int = 32,
    output_report_path: str = "reports/phase3/parity_report.json"
) -> Dict[str, Any]:
    """
    Thực hiện kiểm định Parity Check trên n_samples ảnh mẫu.
    """
    t_start = time.time()
    print("=" * 70)
    print("🚀 BẮT ĐẦU PHA 3: KIỂM ĐỊNH CHECKPOINT VÀ TIỀN XỬ LÝ (PARITY CHECK)")
    print("=" * 70)

    # 1. Đọc config.yaml
    full_config_path = PROJECT_ROOT / config_path
    if not full_config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config.yaml tại: {full_config_path}")
    
    with open(full_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_file = PROJECT_ROOT / cfg["paths"]["checkpoint_file"]
    expected_hash = cfg["checkpoint"]["checkpoint_hash"]
    target_min_cosine = 0.999

    print(f"📁 Checkpoint: {checkpoint_file}")
    print(f"🔒 SHA-256 kỳ vọng: {expected_hash}")

    # 2. Kiểm tra Hash Guard (Luật 6)
    print("\n[Bước 1/4] Kiểm tra Hash Guard (Luật 6)...")
    verify_checkpoint_hash(checkpoint_file, expected_hash)
    print("   ✅ Hash Guard: SHA-256 trùng khớp 100%!")

    # 3. Nạp dữ liệu đối chứng (Ground Truth)
    print(f"\n[Bước 2/4] Nạp dữ liệu đối chứng ({n_samples} mẫu)...")
    id_map_path = PROJECT_ROOT / "data/_quarantine/legacy_rebuild_20261111/untouchable/index/id_map.json"
    gt_embs_path = PROJECT_ROOT / "data/_quarantine/legacy_rebuild_20261111/untouchable/index/image_embeddings.npy"

    if not id_map_path.exists() or not gt_embs_path.exists():
        raise FileNotFoundError(f"Thiếu tệp đối chứng tại: {id_map_path} hoặc {gt_embs_path}")

    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    if len(id_map) < n_samples:
        raise ValueError(f"id_map chỉ có {len(id_map)} mẫu, không đủ {n_samples} mẫu để kiểm định!")

    gt_embs = np.load(gt_embs_path, mmap_mode="r")
    gt_subset = np.array(gt_embs[:n_samples], dtype=np.float32)
    # Chuẩn hóa L2 vector đối chứng
    gt_norms = np.linalg.norm(gt_subset, axis=1, keepdims=True)
    gt_subset_norm = gt_subset / np.maximum(gt_norms, 1e-12)

    # 4. Nạp Model và Preprocessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Bước 3/4] Nạp mô hình và Preprocessor (Device: {device})...")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"   • GPU: {gpu_name}")
        print(f"   • VRAM khởi điểm: {vram_allocated_mb:.1f} MB")

    preprocessor = SigLIPPreprocessor(device=device)
    model = load_siglip_model(
        checkpoint_path=str(checkpoint_file),
        expected_hash=expected_hash,
        device=device,
        fp16=(device == "cuda")
    )

    # 5. Tái mã hóa n_samples ảnh mẫu
    print(f"\n[Bước 4/4] Tái mã hóa {n_samples} ảnh mẫu qua Preprocessor và Model...")
    sample_records = id_map[:n_samples]
    all_new_embs = []

    for b_start in range(0, n_samples, batch_size):
        b_end = min(b_start + batch_size, n_samples)
        batch_slice = sample_records[b_start:b_end]
        batch_paths = [PROJECT_ROOT / item["path"] for item in batch_slice]

        # Kiểm tra tồn tại
        for p in batch_paths:
            if not p.exists():
                raise FileNotFoundError(f"Ảnh không tồn tại: {p}")

        pixel_values = preprocessor.preprocess_images(
            images=batch_paths,
            device=device,
            half_precision=(device == "cuda")
        )

        with torch.no_grad():
            embs = model.encode_image(pixel_values)
            all_new_embs.append(embs.float().cpu().numpy())

    all_new_embs = np.vstack(all_new_embs)
    new_norms = np.linalg.norm(all_new_embs, axis=1, keepdims=True)
    all_new_embs_norm = all_new_embs / np.maximum(new_norms, 1e-12)

    # 6. Tính toán Cosine Similarity
    cosines = (all_new_embs_norm * gt_subset_norm).sum(axis=1)

    min_cos = float(cosines.min())
    max_cos = float(cosines.max())
    mean_cos = float(cosines.mean())
    median_cos = float(np.median(cosines))
    std_cos = float(cosines.std())
    passed_threshold = int((cosines >= target_min_cosine).sum())
    total_count = len(cosines)
    pass_rate = float(passed_threshold / total_count)

    # Tìm Top 5 thấp nhất và Top 5 cao nhất
    sorted_indices = np.argsort(cosines)
    bottom_5 = [
        {
            "index": int(idx),
            "item_id": sample_records[idx].get("item_id"),
            "path": sample_records[idx].get("path"),
            "articleType": sample_records[idx].get("articleType"),
            "cosine": float(cosines[idx])
        }
        for idx in sorted_indices[:5]
    ]
    top_5 = [
        {
            "index": int(idx),
            "item_id": sample_records[idx].get("item_id"),
            "path": sample_records[idx].get("path"),
            "articleType": sample_records[idx].get("articleType"),
            "cosine": float(cosines[idx])
        }
        for idx in sorted_indices[-5:][::-1]
    ]

    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ KIỂM ĐỊNH TIỀN XỬ LÝ (PARITY CHECK)")
    print("=" * 70)
    print(f"• Số lượng mẫu kiểm định:     {total_count}")
    print(f"• Tiêu chuẩn bắt buộc:       Cosine nhỏ nhất >= {target_min_cosine:.4f}")
    print(f"• Cosine nhỏ nhất (Min):     {min_cos:.8f}")
    print(f"• Cosine lớn nhất (Max):     {max_cos:.8f}")
    print(f"• Cosine trung bình (Mean):  {mean_cos:.8f}")
    print(f"• Cosine trung vị (Median):  {median_cos:.8f}")
    print(f"• Độ lệch chuẩn (Std):       {std_cos:.8e}")
    print(f"• Tỷ lệ đạt chuẩn:           {passed_threshold}/{total_count} ({pass_rate * 100:.2f}%)")
    print(f"• Thời gian thực hiện:       {total_time:.2f}s")

    is_passed = (min_cos >= target_min_cosine)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASSED" if is_passed else "FAILED",
        "checkpoint": {
            "path": str(checkpoint_file),
            "sha256": expected_hash,
            "hash_guard_verified": True
        },
        "metrics": {
            "num_samples": total_count,
            "min_cosine": min_cos,
            "max_cosine": max_cos,
            "mean_cosine": mean_cos,
            "median_cosine": median_cos,
            "std_cosine": std_cos,
            "passed_threshold_count": passed_threshold,
            "pass_rate": pass_rate,
            "target_threshold": target_min_cosine
        },
        "device_info": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "vram_allocated_mb": float(torch.cuda.memory_allocated(0) / (1024 ** 2)) if device == "cuda" else 0.0
        },
        "bottom_5_samples": bottom_5,
        "top_5_samples": top_5,
        "elapsed_seconds": round(total_time, 2)
    }

    # Xuất báo cáo json
    full_output_path = PROJECT_ROOT / output_report_path
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Đã lưu kết quả chi tiết tại: {full_output_path}")

    if not is_passed:
        print("\n❌ LỖI NGHIÊM TRỌNG: Cosine nhỏ nhất không đạt chuẩn >= 0.999 theo Luật 4!")
        sys.exit(1)

    print("\n✅ KIỂM ĐỊNH TIỀN XỬ LÝ PHA 3 ĐẠT CHUẨN XUẤT SẮC!")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pha 3 Parity Check")
    parser.add_argument("--config", type=str, default="config.yaml", help="Đường dẫn tệp config.yaml")
    parser.add_argument("--samples", type=int, default=200, help="Số lượng ảnh mẫu để kiểm định (mặc định 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch khi encode")
    parser.add_argument("--output", type=str, default="reports/phase3/parity_report.json", help="Đường dẫn lưu report")
    args = parser.parse_args()

    run_parity_check(
        config_path=args.config,
        n_samples=args.samples,
        batch_size=args.batch_size,
        output_report_path=args.output
    )
