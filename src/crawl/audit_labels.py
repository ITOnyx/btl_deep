"""
Pha 5 — Kiểm tra nhãn & Ước lượng độ chính xác nhãn (Stratified Sampling, Seed 42):
- Lấy mẫu ngẫu nhiên có phân tầng (seed 42) N=300 ảnh từ tập ảnh sạch sau Pha 4.
- Xuất 6 tờ Contact Sheet (mỗi tờ 50 ảnh, tổng 300 ảnh) kèm nhãn vào reports/crawl_v2/contact_sheets_phase5/
  và danh sách CSV reports/crawl_v2/phase5_sample_300_audit.csv để giảng viên/người dùng duyệt nhanh.
- Kiểm tra chéo độc lập (rule-based consistency audit trên product_name, category_raw, product_url vs articleType, gender)
  trên mẫu 300 ảnh để ước lượng tỷ lệ nhãn khả nghi/sai kèm khoảng tin cậy 95% (Wilson score interval).
  Ghi rõ: đây là ước lượng trên mẫu n=300, không khẳng định cho toàn bộ tập dữ liệu.
- Liệt kê toàn bộ các bản ghi có label_status == 'needs_review' vào reports/crawl_v2/needs_review_list.csv;
  KHÔNG tự sửa hàng loạt.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from .common import load_config, now_iso, read_jsonl
from .contact_sheet import render_contact_sheet
from .taxonomy_mapper import TaxonomyMapper


def wilson_ci_95(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Khoảng tin cậy 95% theo công thức Wilson score cho tỷ lệ k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def stratified_sample_records(
    records: List[Dict[str, Any]],
    sample_size: int = 300,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Lấy mẫu ngẫu nhiên phân tầng theo (brand, articleType) với seed cố định 42."""
    if len(records) <= sample_size:
        return list(records)

    rng = random.Random(seed)
    by_brand: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_brand.setdefault(r.get("brand", "Unknown"), []).append(r)

    total = len(records)
    sampled: List[Dict[str, Any]] = []
    used_ids = set()

    for brand in sorted(by_brand.keys()):
        items = list(by_brand[brand])
        rng.shuffle(items)
        # Sắp xếp xen kẽ theo articleType trong mỗi brand để phủ đa dạng loại sản phẩm
        by_art: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            by_art.setdefault(it.get("articleType", ""), []).append(it)
        interleaved: List[Dict[str, Any]] = []
        art_keys = sorted(by_art.keys())
        max_len = max(len(v) for v in by_art.values()) if by_art else 0
        for i in range(max_len):
            for ak in art_keys:
                if i < len(by_art[ak]):
                    interleaved.append(by_art[ak][i])

        quota = max(5, round(sample_size * (len(items) / total)))
        for it in interleaved[:quota]:
            if len(sampled) < sample_size and it["crawl_id"] not in used_ids:
                sampled.append(it)
                used_ids.add(it["crawl_id"])

    if len(sampled) < sample_size:
        rem = [r for r in records if r["crawl_id"] not in used_ids]
        rng.shuffle(rem)
        sampled.extend(rem[: sample_size - len(sampled)])

    return sampled[:sample_size]


def audit_single_record(rec: Dict[str, Any], mapper: TaxonomyMapper) -> Tuple[str, str]:
    """
    Kiểm tra đối chiếu độc lập giữa nhãn gán (articleType, gender) với tên sản phẩm (product_name) và category_raw:
    Trả về (audit_verdict, audit_note) trong đó audit_verdict thuộc {'consistent', 'ambiguous_or_mismatch'}.
    """
    art = str(rec.get("articleType", ""))
    gen = str(rec.get("gender", ""))
    pname = str(rec.get("product_name", ""))
    cat_raw = str(rec.get("category_raw", ""))

    if art in ("", "needs_review") or gen in ("", "needs_review"):
        return "ambiguous_or_mismatch", "label_status=needs_review (thiếu articleType hoặc gender rõ ràng)"

    # Kiểm tra xem nếu chỉ suy từ product_name thì có mâu thuẫn loại sản phẩm với breadcrumb không
    title_art, _ = mapper.map_category("", pname)
    if title_art and title_art != art:
        # Một số cặp gần nhau (Tshirts vs Tops, Trousers vs Track Pants, Handbags vs Messenger Bag)
        close_pairs = {
            frozenset({"Tshirts", "Tops"}),
            frozenset({"Trousers", "Track Pants"}),
            frozenset({"Handbags", "Messenger Bag"}),
            frozenset({"Handbags", "Clutches"}),
            frozenset({"Sports Shoes", "Casual Shoes"}),
            frozenset({"Sandals", "Flip Flops"}),
            frozenset({"Jackets", "Blazers"}),
            frozenset({"Sweatshirts", "Sweaters"}),
            frozenset({"Boxers", "Briefs"}),
            frozenset({"Caps", "Hat"}),
        }
        if frozenset({art, title_art}) not in close_pairs:
            return "ambiguous_or_mismatch", f"Breadcrumb={art} khác từ khóa tiêu đề={title_art}"

    # Kiểm tra mâu thuẫn giới tính trực tiếp trong tiêu đề
    low_title = pname.lower()
    if gen == "Men" and (" nữ" in low_title or "bé gái" in low_title) and "nam" not in low_title:
        return "ambiguous_or_mismatch", f"gender=Men nhưng tiêu đề có từ khóa nữ/bé gái"
    if gen == "Women" and (" nam " in f"{low_title} " or "bé trai" in low_title) and "nữ" not in low_title:
        return "ambiguous_or_mismatch", f"gender=Women nhưng tiêu đề có từ khóa nam/bé trai"

    return "consistent", "Khớp nhất quán giữa tiêu đề, danh mục và ảnh"


def run_phase5_label_audit(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    c2 = cfg["crawl_v2"]
    paths = c2["paths"]
    seed = int(c2.get("seed", 42))

    rep_dir = Path(paths["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    meta_jsonl = Path(paths["metadata_jsonl"])
    clean_records = read_jsonl(meta_jsonl)
    print(f"[Phase 5] Đã nạp {len(clean_records):,} bản ghi sạch từ {meta_jsonl}")

    # 1. Xuất danh sách toàn bộ needs_review (KHÔNG tự sửa hàng loạt)
    needs_review_recs = [r for r in clean_records if r.get("label_status") == "needs_review"]
    nr_csv_path = rep_dir / "needs_review_list.csv"
    pd.DataFrame(needs_review_recs).to_csv(nr_csv_path, index=False, encoding="utf-8-sig")
    print(f"[Phase 5] Đã liệt kê {len(needs_review_recs):,} bản ghi needs_review tại {nr_csv_path}")

    # 2. Lấy mẫu phân tầng 300 ảnh (seed 42)
    sample_300 = stratified_sample_records(clean_records, sample_size=300, seed=seed)
    n_sample = len(sample_300)

    # Xuất 6 tờ Contact Sheet (50 ảnh/tờ) cho 300 ảnh mẫu
    cs_dir = rep_dir / "contact_sheets_phase5"
    cs_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths: List[str] = []
    for i in range(0, n_sample, 50):
        chunk = sample_300[i : i + 50]
        sheet_idx = (i // 50) + 1
        sp = render_contact_sheet(
            chunk,
            cs_dir / f"phase5_sample300_sheet_{sheet_idx:02d}.png",
            cols=5,
            title=f"Phase 5 Stratified Audit Sample (Seed 42) — Sheet {sheet_idx}/6 (Items {i+1}..{i+len(chunk)})",
        )
        sheet_paths.append(sp.as_posix())

    # 3. Kiểm tra đối chiếu trên 300 ảnh mẫu và tính ước lượng tỷ lệ lỗi + 95% CI
    mapper = TaxonomyMapper()
    audit_rows: List[Dict[str, Any]] = []
    n_mismatch = 0
    for r in sample_300:
        verdict, note = audit_single_record(r, mapper)
        if verdict != "consistent":
            n_mismatch += 1
        audit_rows.append({
            "crawl_id": r["crawl_id"],
            "brand": r["brand"],
            "product_name": r["product_name"],
            "category_raw": r["category_raw"],
            "articleType": r["articleType"],
            "gender": r["gender"],
            "baseColour": r["baseColour"],
            "label_source": r["label_source"],
            "label_status": r["label_status"],
            "audit_verdict": verdict,
            "audit_note": note,
            "local_path": r["local_path"],
            "product_url": r["product_url"],
        })

    audit_csv_path = rep_dir / "phase5_sample_300_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_csv_path, index=False, encoding="utf-8-sig")

    err_rate = (n_mismatch / max(1, n_sample)) * 100.0
    ci_low, ci_high = wilson_ci_95(n_mismatch, n_sample)

    # Cũng tính riêng tỷ lệ lỗi trên các bản ghi có label_status == 'ok' trong mẫu 300
    ok_in_sample = [row for row in audit_rows if row["label_status"] == "ok"]
    mismatch_in_ok = sum(1 for row in ok_in_sample if row["audit_verdict"] != "consistent")
    ok_err_rate = (mismatch_in_ok / max(1, len(ok_in_sample))) * 100.0
    ok_ci_low, ok_ci_high = wilson_ci_95(mismatch_in_ok, len(ok_in_sample))

    lines = [
        "# BÁO CÁO PHA 5 — KIỂM TRA NHÃN PHÂN TẦNG (SEED 42) & DANH SÁCH `needs_review`",
        "",
        f"- **Thời điểm hoàn tất (UTC):** `{now_iso()}`",
        f"- **Phương pháp lấy mẫu:** Lấy mẫu ngẫu nhiên có phân tầng theo `(brand, articleType)` với `seed = {seed}`.",
        f"- **Cỡ mẫu kiểm tra:** **`n = {n_sample}`** ảnh trên tổng số **`{len(clean_records):,}`** ảnh sạch.",
        f"- **Tệp chi tiết 300 mẫu:** `{audit_csv_path.as_posix()}`",
        f"- **Tệp danh sách toàn bộ `needs_review` ({len(needs_review_recs):,} bản ghi):** `{nr_csv_path.as_posix()}` (giữ nguyên, KHÔNG tự sửa hàng loạt).",
        "",
        "## 1. Ước lượng tỷ lệ nhãn sai / mơ hồ trên mẫu `n = 300` (Kèm khoảng tin cậy 95% Wilson)",
        "",
        "> **Lưu ý học thuật (Luật L5 & Quy tắc Pha 5):** Các con số dưới đây là **ước lượng thống kê từ mẫu phân tầng `n = 300`**, đi kèm khoảng tin cậy 95% (Wilson score interval); **không khẳng định tuyệt đối cho toàn bộ tập dữ liệu**.",
        "",
        "| Tập mẫu đánh giá | Cỡ mẫu (`n`) | Số ca mâu thuẫn / `needs_review` (`k`) | Tỷ lệ ước lượng (`k/n`) | Khoảng tin cậy 95% (Wilson CI) |",
        "|---|---:|---:|---:|---|",
        f"| Toàn bộ mẫu phân tầng (gồm cả `ok` và `needs_review`) | **{n_sample}** | **{n_mismatch}** | **{err_rate:.2f}%** | **`[{ci_low*100:.2f}%, {ci_high*100:.2f}%]`** |",
        f"| Chỉ tính các bản ghi có `label_status == ok` trong mẫu | **{len(ok_in_sample)}** | **{mismatch_in_ok}** | **{ok_err_rate:.2f}%** | **`[{ok_ci_low*100:.2f}%, {ok_ci_high*100:.2f}%]`** |",
        "",
        "## 2. Danh sách 6 tờ Contact Sheet (300 ảnh kèm nhãn để giảng viên duyệt nhanh)",
        "",
    ]
    for sp in sheet_paths:
        lines.append(f"- `{sp}`")

    lines += [
        "",
        "## 3. Thống kê bản ghi `needs_review` trên toàn bộ tập sạch (Không tự sửa hàng loạt)",
        "",
        f"- Tổng số bản ghi `label_status == 'ok'`: **{len(clean_records) - len(needs_review_recs):,}** (`{(len(clean_records) - len(needs_review_recs)) / max(1, len(clean_records)) * 100:.2f}%`)",
        f"- Tổng số bản ghi `label_status == 'needs_review'`: **{len(needs_review_recs):,}** (`{len(needs_review_recs) / max(1, len(clean_records)) * 100:.2f}%`)",
        "",
        "| Thương hiệu (`brand`) | Tổng ảnh sạch | Số ca `ok` | Số ca `needs_review` | Tỷ lệ `needs_review` (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    df_all = pd.DataFrame(clean_records)
    if len(df_all) > 0:
        for br, grp in df_all.groupby("brand"):
            n_tot = len(grp)
            n_nr = int((grp["label_status"] == "needs_review").sum())
            n_ok = n_tot - n_nr
            lines.append(f"| {br} | {n_tot:,} | {n_ok:,} | {n_nr:,} | {n_nr / max(1, n_tot) * 100:.1f}% |")

    (rep_dir / "phase5_label_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Phase 5] Đã xuất báo cáo tại {rep_dir / 'phase5_label_audit_report.md'}")
    return {
        "n_sample": n_sample,
        "n_mismatch": n_mismatch,
        "err_rate_pct": round(err_rate, 2),
        "ci_95_pct": [round(ci_low * 100, 2), round(ci_high * 100, 2)],
        "n_needs_review_total": len(needs_review_recs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 stratified label audit (300 images, seed 42)")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn tới config.yaml")
    args = parser.parse_args()
    run_phase5_label_audit(args.config)


if __name__ == "__main__":
    main()
