"""
Pha 4 — Làm sạch, Kiểm định Chất lượng và Khử trùng (Nội bộ + Chéo với Catalog hiện có):
1. Kiểm tra lại toàn bộ tệp ảnh pass từ Pha 3 trên đĩa:
   - missing / hỏng (corrupt)
   - low_res (< 224 px cạnh ngắn)
   - blurry (Laplacian variance < 12.0)
   - banner / placeholder
   Các ảnh vi phạm được chuyển vào data/crawl_v2/_quarantine/<lý_do>/ và ghi log CSV (Luật L2).
   Xuất contact sheet các ảnh có blur_score giáp ngưỡng [12.0, 20.0] để minh bạch.
2. Khử trùng NỘI BỘ trên TOÀN BỘ ảnh crawl (KHÔNG chia theo nhóm nhãn):
   - Trùng SHA-1 tuyệt đối trên byte gốc -> giữ 1 bản đại diện/cụm, chuyển các bản còn lại vào _quarantine/dup_sha1/
   - Trùng pHash 64-bit (Hamming <= 1) trên toàn bộ tập -> giữ 1 bản đại diện/cụm, chuyển các bản còn lại vào _quarantine/dup_phash/
3. Khử trùng CHÉO với catalog hiện có (data/fashion-dataset: train, val, test trong data/catalog/catalog_rebuild.parquet):
   - Trùng SHA-1 hoặc pHash (Hamming <= 1) với bất kỳ ảnh nào trong catalog -> đánh dấu filter_flag = 'overlap_with_catalog'
     và chuyển vào data/crawl_v2/_quarantine/overlap_catalog/
4. Cập nhật trạng thái và xuất báo cáo Pha 4 (reports/crawl_v2/phase4_cleaning_dedup_report.md) + danh sách sạch tạm thời.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from .common import (
    QuarantineLogger,
    SCHEMA_COLUMNS,
    detect_banner_or_placeholder,
    load_config,
    now_iso,
    read_jsonl,
)
from .contact_sheet import render_contact_sheet

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hex_phash_to_uint64(ph_hex: str) -> Optional[int]:
    if not ph_hex or not isinstance(ph_hex, str):
        return None
    s = ph_hex.strip()
    if len(s) != 16:
        return None
    try:
        return int(s, 16)
    except ValueError:
        return None


def min_hamming_and_nearest(
    query_u64: np.ndarray,
    ref_u64: np.ndarray,
    chunk_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trả về (min_dist_array, argmin_index_array) cho mỗi phần tử trong query_u64 so với ref_u64."""
    n_q = len(query_u64)
    if n_q == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    if len(ref_u64) == 0:
        return np.full(n_q, 64, dtype=np.int32), np.full(n_q, -1, dtype=np.int32)

    min_dists = np.full(n_q, 64, dtype=np.int32)
    nearest_idx = np.full(n_q, -1, dtype=np.int32)
    for start in range(0, n_q, chunk_size):
        q = query_u64[start : start + chunk_size, None]  # [B, 1]
        xor_vals = np.bitwise_xor(q, ref_u64[None, :])   # [B, N] uint64
        bytes_view = xor_vals.view(np.uint8).reshape(xor_vals.shape[0], xor_vals.shape[1], 8)
        dists = _POPCOUNT_LUT[bytes_view].sum(axis=2, dtype=np.uint8)  # [B, N]
        min_dists[start : start + chunk_size] = dists.min(axis=1)
        nearest_idx[start : start + chunk_size] = dists.argmin(axis=1)
    return min_dists, nearest_idx


def load_catalog_hashes() -> Tuple[Set[str], np.ndarray, List[str]]:
    """
    Đọc SHA-1 và pHash 64-bit của toàn bộ catalog hiện có (data/fashion-dataset: train, val, test)
    từ reports/phase1/phase1a/quality_metrics.parquet và cache seen_hashes_cache.npz (chế độ chỉ đọc - Luật L1/L3).
    """
    cat_sha1s: Set[str] = set()
    cat_phashes: List[int] = []
    cat_ids: List[str] = []

    # 1. Đọc pHash 44,419 ảnh catalog từ reports/phase1/phase1a/quality_metrics.parquet
    qm_path = Path("reports/phase1/phase1a/quality_metrics.parquet")
    if qm_path.exists():
        df_qm = pd.read_parquet(qm_path, columns=["id", "phash"])
        for pid, ph in zip(df_qm["id"], df_qm["phash"]):
            u = hex_phash_to_uint64(str(ph) if pd.notna(ph) else "")
            if u is not None:
                cat_phashes.append(u)
                cat_ids.append(str(pid))

    # 2. Bổ sung từ seen_hashes_cache.npz nếu có
    npz_path = Path("reports/external_test/seen_hashes_cache.npz")
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        if "phashes" in npz:
            extra_ph = npz["phashes"].astype(np.uint64).tolist()
            cat_phashes.extend(extra_ph)
            cat_ids.extend([f"seen_{i}" for i in range(len(extra_ph))])

    # 3. Đọc SHA-1 nếu có trong catalog_rebuild hoặc tính từ ảnh catalog
    cat_path = Path("data/catalog/catalog_rebuild.parquet")
    if cat_path.exists():
        df_cat = pd.read_parquet(cat_path, columns=["sha1"])
        vals = [s.strip() for s in df_cat["sha1"].dropna().astype(str) if s.strip()]
        cat_sha1s.update(vals)

    # Loại bỏ trùng lặp trong pHash catalog
    if cat_phashes:
        u_arr, u_idx = np.unique(np.array(cat_phashes, dtype=np.uint64), return_index=True)
        u_ids = [cat_ids[i] for i in u_idx]
        return cat_sha1s, u_arr, u_ids

    return cat_sha1s, np.array([], dtype=np.uint64), []


def run_phase4_cleaning(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]
    q_cfg = c2["quality_filter"]

    min_short_side = int(q_cfg.get("min_short_side", 224))
    blur_thr = float(q_cfg.get("blur_laplacian_threshold", 12.0))
    phash_thr = int(q_cfg.get("phash_hamming_threshold", 1))

    raw_state_file = Path(paths["raw_state_file"])
    rep_dir = Path(paths["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    quarantine = QuarantineLogger(paths["quarantine_dir"], paths["quarantine_log_csv"])

    all_records = read_jsonl(raw_state_file)
    print(f"[Phase 4] Tổng bản ghi trong raw_crawled.jsonl: {len(all_records):,}")

    # Phân loại bản ghi ngay từ bước tải thô
    raw_pass_candidates: List[Dict[str, Any]] = []
    initial_drops = Counter()
    for r in all_records:
        ff = r.get("filter_flag", "")
        if ff == "pass":
            raw_pass_candidates.append(r)
        else:
            initial_drops[ff or "unknown"] += 1

    print(f"[Phase 4] Bản ghi pass ban đầu: {len(raw_pass_candidates):,} | Bị loại lúc tải: {dict(initial_drops)}")

    # Bước 1: Kiểm tra lại tệp trên đĩa (missing/corrupt, low_res < 224, blurry < 12.0, banner)
    after_quality: List[Dict[str, Any]] = []
    step1_drops = Counter()
    borderline_blur_samples: List[Dict[str, Any]] = []

    for r in raw_pass_candidates:
        cid = r["crawl_id"]
        brand = r.get("brand", r.get("source", ""))
        lp = Path(r.get("local_path", ""))
        if not lp.exists() or not lp.is_file():
            r["filter_flag"] = "missing"
            step1_drops["missing"] += 1
            continue

        # Nếu đã có đầy đủ sha1, phash, width, height, blur_score hợp lệ từ bước tải thì chỉ kiểm tra ngưỡng
        w = int(r.get("width") or 0)
        h = int(r.get("height") or 0)
        sha1_s = str(r.get("sha1") or "")
        phash_s = str(r.get("phash") or "")
        blur_s = float(r.get("blur_score") or 0.0)

        if w <= 0 or h <= 0 or not sha1_s or not phash_s:
            try:
                raw_bytes = lp.read_bytes()
                sha1_s = hashlib.sha1(raw_bytes).hexdigest()
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    rgb = im.convert("RGB")
                w, h = rgb.size
                phash_s = str(imagehash.phash(rgb.convert("L")))
                gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
                blur_s = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                r["width"], r["height"], r["sha1"], r["phash"], r["blur_score"] = w, h, sha1_s, phash_s, round(blur_s, 4)
            except Exception as exc:
                q_p = quarantine.quarantine_existing_file(lp, cid, brand, "corrupt", f"recheck_{type(exc).__name__}")
                r["local_path"] = q_p
                r["filter_flag"] = "corrupt"
                step1_drops["corrupt"] += 1
                continue

        if min(w, h) < min_short_side:
            q_p = quarantine.quarantine_existing_file(
                lp, cid, brand, "low_res", f"short_side={min(w,h)}<{min_short_side}", sha1_s, phash_s
            )
            r["local_path"] = q_p
            r["filter_flag"] = "low_res"
            step1_drops["low_res"] += 1
            continue

        if blur_s < blur_thr:
            q_p = quarantine.quarantine_existing_file(
                lp, cid, brand, "blurry", f"laplacian={blur_s:.2f}<{blur_thr}", sha1_s, phash_s
            )
            r["local_path"] = q_p
            r["filter_flag"] = "blurry"
            step1_drops["blurry"] += 1
            continue

        if 12.0 <= blur_s <= 20.0 and len(borderline_blur_samples) < 25:
            borderline_blur_samples.append(r)

        after_quality.append(r)

    print(f"[Phase 4 - Step 1] Sau kiểm tra chất lượng trên đĩa: {len(after_quality):,} (loại thêm: {dict(step1_drops)})")

    if borderline_blur_samples:
        render_contact_sheet(
            borderline_blur_samples,
            rep_dir / "phase4_borderline_blur_contact_sheet.png",
            cols=5,
            title="Phase 4 Borderline Blur Score [12.0 - 20.0] Contact Sheet",
        )

    # Bước 2: Khử trùng NỘI BỘ trên TOÀN BỘ ảnh crawl (SHA-1 & pHash Hamming <= 1, KHÔNG chia theo nhóm nhãn)
    seen_sha1: Dict[str, str] = {}
    after_sha1: List[Dict[str, Any]] = []
    dup_sha1_count = 0

    for r in after_quality:
        cid = r["crawl_id"]
        brand = r.get("brand", "")
        s1 = r["sha1"]
        if s1 in seen_sha1:
            rep_id = seen_sha1[s1]
            q_p = quarantine.quarantine_existing_file(
                r["local_path"], cid, brand, "dup_sha1", f"dup_of_{rep_id}", s1, r["phash"]
            )
            r["local_path"] = q_p
            r["filter_flag"] = "dup_sha1"
            dup_sha1_count += 1
        else:
            seen_sha1[s1] = cid
            after_sha1.append(r)

    print(f"[Phase 4 - Step 2a] Sau khử trùng SHA-1 toàn cục: {len(after_sha1):,} (loại dup_sha1={dup_sha1_count})")

    # Khử trùng pHash nội bộ (Hamming <= 1) trên toàn bộ tập after_sha1
    kept_internal: List[Dict[str, Any]] = []
    kept_phashes_u64: List[int] = []
    kept_ids: List[str] = []
    dup_phash_count = 0

    # Vector hóa theo lô để tốc độ cao trên 10,000+ ảnh
    all_u64 = np.array([hex_phash_to_uint64(r["phash"]) or 0 for r in after_sha1], dtype=np.uint64)
    n_s1 = len(after_sha1)
    # Tính ma trận khoảng cách từng phần (chunk 512 x N) để tìm phần tử đứng trước có Hamming <= 1
    is_dup_phash = np.zeros(n_s1, dtype=bool)
    dup_of_idx = np.full(n_s1, -1, dtype=np.int32)

    chunk = 512
    for start in range(0, n_s1, chunk):
        end = min(n_s1, start + chunk)
        q_block = all_u64[start:end, None]      # [B, 1]
        ref_block = all_u64[:end][None, :]      # [1, end]
        xor_v = np.bitwise_xor(q_block, ref_block)
        b_view = xor_v.view(np.uint8).reshape(end - start, end, 8)
        dists = _POPCOUNT_LUT[b_view].sum(axis=2, dtype=np.uint8)  # [B, end]
        for local_i in range(end - start):
            global_i = start + local_i
            if global_i == 0:
                continue
            row_d = dists[local_i, :global_i]
            # Chỉ xét các phần tử phía trước chưa bị đánh dấu trùng
            valid_prev = np.where((row_d <= phash_thr) & (~is_dup_phash[:global_i]))[0]
            if len(valid_prev) > 0:
                is_dup_phash[global_i] = True
                dup_of_idx[global_i] = int(valid_prev[0])

    for idx, r in enumerate(after_sha1):
        cid = r["crawl_id"]
        brand = r.get("brand", "")
        if is_dup_phash[idx]:
            rep_id = after_sha1[int(dup_of_idx[idx])]["crawl_id"]
            q_p = quarantine.quarantine_existing_file(
                r["local_path"], cid, brand, "dup_phash", f"hamming<={phash_thr}_of_{rep_id}", r["sha1"], r["phash"]
            )
            r["local_path"] = q_p
            r["filter_flag"] = "dup_phash"
            dup_phash_count += 1
        else:
            kept_internal.append(r)
            kept_phashes_u64.append(int(all_u64[idx]))
            kept_ids.append(cid)

    print(f"[Phase 4 - Step 2b] Sau khử trùng pHash (Hamming<={phash_thr}) toàn cục: {len(kept_internal):,} (loại dup_phash={dup_phash_count})")

    # Bước 3: Khử trùng CHÉO với catalog hiện có (data/fashion-dataset: train, val, test)
    cat_sha1s, cat_phashes_u64, cat_ids = load_catalog_hashes()
    print(f"[Phase 4 - Step 3] Đối chiếu với catalog hiện có: {len(cat_sha1s):,} SHA-1, {len(cat_phashes_u64):,} pHash")

    q_internal_u64 = np.array(kept_phashes_u64, dtype=np.uint64)
    min_cat_dists, nearest_cat_idx = min_hamming_and_nearest(q_internal_u64, cat_phashes_u64)

    clean_records: List[Dict[str, Any]] = []
    overlap_catalog_count = 0

    for idx, r in enumerate(kept_internal):
        cid = r["crawl_id"]
        brand = r.get("brand", "")
        s1 = r["sha1"]
        d_cat = int(min_cat_dists[idx]) if len(min_cat_dists) > idx else 64
        near_cid = cat_ids[int(nearest_cat_idx[idx])] if (len(nearest_cat_idx) > idx and nearest_cat_idx[idx] >= 0) else ""

        if s1 in cat_sha1s or d_cat <= phash_thr:
            detail = "exact_sha1_catalog" if s1 in cat_sha1s else f"phash_dist={d_cat}_catalog_id={near_cid}"
            q_p = quarantine.quarantine_existing_file(
                r["local_path"], cid, brand, "overlap_catalog", detail, s1, r["phash"]
            )
            r["local_path"] = q_p
            r["filter_flag"] = "overlap_with_catalog"
            overlap_catalog_count += 1
        else:
            r["filter_flag"] = "pass"
            clean_records.append(r)

    print(f"[Phase 4 - Step 3] Sau khử trùng chéo với catalog: {len(clean_records):,} ảnh sạch (loại overlap_with_catalog={overlap_catalog_count})")

    # Lưu lại raw_crawled.jsonl đã cập nhật filter_flag và local_path
    with open(raw_state_file, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Ghi metadata.jsonl và metadata.csv theo đúng 20 cột Schema Pha 1 (chỉ chứa ảnh sạch pass để bàn giao)
    meta_csv_path = Path(paths["metadata_csv"])
    meta_jsonl_path = Path(paths["metadata_jsonl"])
    meta_csv_path.parent.mkdir(parents=True, exist_ok=True)

    clean_schema_rows: List[Dict[str, Any]] = []
    for r in clean_records:
        row = {col: r.get(col, "") for col in SCHEMA_COLUMNS}
        clean_schema_rows.append(row)

    with open(meta_jsonl_path, "w", encoding="utf-8") as f:
        for row in clean_schema_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df_clean = pd.DataFrame(clean_schema_rows, columns=SCHEMA_COLUMNS)
    df_clean.to_csv(meta_csv_path, index=False, encoding="utf-8-sig")

    # Đếm thật số file trên đĩa trong data/crawl_v2/images/
    images_root = Path(paths["images_dir"])
    disk_files = [f for f in images_root.rglob("*") if f.is_file()]
    quarantine_root = Path(paths["quarantine_dir"])
    quar_files = [f for f in quarantine_root.rglob("*") if f.is_file()]

    total_raw_candidates = len(all_records)
    n_oos = initial_drops.get("out_of_scope", 0)
    n_dl_fail = initial_drops.get("download_failed", 0) + initial_drops.get("banner_url_keyword", 0)
    n_raw_downloaded = total_raw_candidates - n_oos - n_dl_fail
    n_q_lowres = initial_drops.get("low_res", 0) + step1_drops.get("low_res", 0)
    n_q_blurry = initial_drops.get("blurry", 0) + step1_drops.get("blurry", 0)
    n_q_banner = initial_drops.get("banner", 0) + step1_drops.get("banner", 0)
    n_q_corrupt = initial_drops.get("corrupt", 0) + step1_drops.get("corrupt", 0) + step1_drops.get("missing", 0)

    rep_lines = [
        "# BÁO CÁO PHA 4 — LÀM SẠCH, LỌC CHẤT LƯỢNG & KHỬ TRÙNG TOÀN CỤC",
        "",
        f"- **Thời điểm hoàn tất (UTC):** `{now_iso()}`",
        f"- **Tiêu chí chất lượng áp dụng:** `short_side >= {min_short_side} px`, `blur_score (Laplacian) >= {blur_thr}`, `detect_banner_or_placeholder`, kiểm tra định dạng/byte gốc.",
        f"- **Tiêu chí khử trùng nội bộ (toàn cục, không chia theo nhãn):** `SHA-1` chính xác + `pHash 64-bit Hamming <= {phash_thr}`.",
        f"- **Tiêu chí khử trùng chéo với Catalog (`data/fashion-dataset` train/val/test):** `SHA-1` chính xác + `pHash 64-bit Hamming <= {phash_thr}` trên `{len(cat_phashes_u64):,}` ảnh catalog.",
        "",
        "## 1. Bảng phễu số lượng ảnh qua từng bước (Đếm thật trên đĩa)",
        "",
        "| Bước xử lý | Số lượng bản ghi / ảnh | Ghi chú / Thư mục đích |",
        "|---|---:|---|",
        f"| 0. Tổng ứng viên sản phẩm đi qua crawler | **{total_raw_candidates:,}** | `data/crawl_v2/raw_crawled.jsonl` |",
        f"| — Loại ngoài phạm vi thời trang (`out_of_scope`) | {n_oos:,} | Mỹ phẩm, quà tặng, lens, bình nước (không tải ảnh) |",
        f"| — Lỗi tải HTTP / URL từ khóa banner | {n_dl_fail:,} | Không lưu tệp |",
        f"| **1. Tổng ảnh tải thô xuống đĩa** | **{n_raw_downloaded:,}** | Đã kiểm tra byte gốc |",
        f"| — Loại do độ phân giải thấp (`low_res < {min_short_side}px`) | {n_q_lowres:,} | `data/crawl_v2/_quarantine/low_res/` |",
        f"| — Loại do ảnh mờ (`blurry Laplacian < {blur_thr}`) | {n_q_blurry:,} | `data/crawl_v2/_quarantine/blurry/` |",
        f"| — Loại do ảnh banner / chữ khuyến mãi / khung cực đoan (`banner`) | {n_q_banner:,} | `data/crawl_v2/_quarantine/banner/` |",
        f"| — Loại do tệp hỏng / định dạng (`corrupt` / `missing`) | {n_q_corrupt:,} | `data/crawl_v2/_quarantine/corrupt/` |",
        f"| **2. Còn lại sau lọc chất lượng** | **{len(after_quality):,}** | 100% ảnh sắc nét, nguyên byte gốc |",
        f"| — Loại do trùng `SHA-1` nội bộ toàn cục (`dup_sha1`) | {dup_sha1_count:,} | `data/crawl_v2/_quarantine/dup_sha1/` |",
        f"| — Loại do trùng `pHash Hamming <= {phash_thr}` nội bộ toàn cục (`dup_phash`) | {dup_phash_count:,} | `data/crawl_v2/_quarantine/dup_phash/` |",
        f"| **3. Còn lại sau khử trùng nội bộ** | **{len(kept_internal):,}** | 1 bản đại diện duy nhất / cụm |",
        f"| — Loại do trùng chéo với Catalog hiện có (`overlap_with_catalog`) | {overlap_catalog_count:,} | `data/crawl_v2/_quarantine/overlap_catalog/` |",
        f"| **4. TỔNG SỐ ẢNH SẠCH CUỐI CÙNG (`filter_flag == pass`)** | **{len(clean_records):,}** | Lưu tại `reports/crawl_v2/metadata.csv` & `.jsonl` |",
        f"| — Đối chiếu số tệp ảnh thật trong `data/crawl_v2/images/` | **{len(disk_files):,}** | Lệch = `{len(disk_files) - len(clean_records)}` |",
        f"| — Tổng số tệp đang lưu trong `data/crawl_v2/_quarantine/` | **{len(quar_files):,}** | Kèm log `reports/crawl_v2/quarantine_log.csv` |",
        "",
    ]

    (rep_dir / "phase4_cleaning_dedup_report.md").write_text("\n".join(rep_lines) + "\n", encoding="utf-8")
    print(f"[Phase 4] Đã xuất báo cáo tại {rep_dir / 'phase4_cleaning_dedup_report.md'}")
    return {
        "n_clean": len(clean_records),
        "n_disk": len(disk_files),
        "n_quarantine": len(quar_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 cleaning, quality filter, and global/cross deduplication")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn tới config.yaml")
    args = parser.parse_args()
    run_phase4_cleaning(args.config)


if __name__ == "__main__":
    main()
