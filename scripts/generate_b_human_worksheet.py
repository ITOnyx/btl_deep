"""
Việc 1 (PROMPT 6b): Xuất bảng cụm khả dụng trên index/val_gallery/ (chỉ đọc, không nạp model).
Tạo:
- reports/phase5b/b_human_cluster_worksheet.csv
- reports/phase5b/b_human_cluster_worksheet.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_b_human import check_b_human_queries, load_reference_datasets
from src.eval.metrics import is_item_relevant, load_gender_gt_policy


def make_cluster_id(spec: Dict[str, Any]) -> str:
    ats = "|".join(sorted(spec.get("articleType") or []))
    gd = str(spec.get("gender") or "None")
    cls = "|".join(sorted(spec.get("baseColour") or []))
    return f"{ats}__{gd}__{cls}"


def main() -> None:
    val_id_map = json.loads((PROJECT_ROOT / "index/val_gallery/id_map.json").read_text(encoding="utf-8"))
    strict_cats = set(
        load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")["policies"]["hybrid"]["strict_categories"]
    )
    val_df = pd.DataFrame(val_id_map)

    # Đếm spec_cluster_id ở DEV, A, B-synth, và B-human hợp lệ
    dev_data = json.loads((PROJECT_ROOT / "eval/handwritten_queries.json").read_text(encoding="utf-8"))
    a_data = json.loads((PROJECT_ROOT / "eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    b_synth_data = json.loads((PROJECT_ROOT / "eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    b_human_path = PROJECT_ROOT / "eval/val_queries_b_human.json"
    b_human_data = json.loads(b_human_path.read_text(encoding="utf-8"))

    refs = load_reference_datasets(PROJECT_ROOT)
    b_human_rep = check_b_human_queries(b_human_path, val_id_map, strict_cats, refs)
    b_human_valid_items = [
        q for q, r in zip(b_human_data, b_human_rep["rows"])
        if r["n_relevant"] >= 5 and not r["near_duplicates"]
    ]

    cnt_dev = Counter(make_cluster_id(x["spec"]) for x in dev_data)
    cnt_a = Counter(make_cluster_id(x["spec"]) for x in a_data)
    cnt_bs = Counter(make_cluster_id(x["spec"]) for x in b_synth_data)
    cnt_bh = Counter(make_cluster_id(x["spec"]) for x in b_human_valid_items)

    # Sinh mọi tổ hợp 1-slot, 2-slot, 3-slot có mặt trong val_gallery
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for at in sorted(val_df["articleType"].dropna().unique()):
        candidates.append(("1_slot (articleType)", {"articleType": [at], "gender": None, "baseColour": []}))
    for (at, gd), _ in val_df.groupby(["articleType", "gender"]):
        candidates.append(("2_slot (articleType+gender)", {"articleType": [str(at)], "gender": str(gd), "baseColour": []}))
    for (at, cl), _ in val_df.groupby(["articleType", "baseColour"]):
        candidates.append(("2_slot (articleType+baseColour)", {"articleType": [str(at)], "gender": None, "baseColour": [str(cl)]}))
    for (at, gd, cl), _ in val_df.groupby(["articleType", "gender", "baseColour"]):
        candidates.append(("3_slot (articleType+gender+baseColour)", {"articleType": [str(at)], "gender": str(gd), "baseColour": [str(cl)]}))

    rows: List[Dict[str, Any]] = []
    for slot_kind, spec in candidates:
        n_rel = sum(1 for m in val_id_map if is_item_relevant(spec, m, "hybrid", strict_cats))
        if n_rel < 5:
            continue
        cid = make_cluster_id(spec)
        u_dev = cnt_dev[cid]
        u_a = cnt_a[cid]
        u_bs = cnt_bs[cid]
        u_bh = cnt_bh[cid]
        total_used = u_dev + u_a + u_bs + u_bh
        rows.append({
            "slot_kind": slot_kind,
            "spec_cluster_id": cid,
            "articleType": spec["articleType"][0],
            "gender": spec["gender"] if spec["gender"] is not None else "null",
            "baseColour": spec["baseColour"][0] if spec["baseColour"] else "",
            "n_relevant": n_rel,
            "used_DEV": u_dev,
            "used_A": u_a,
            "used_B_synth": u_bs,
            "used_B_human_valid": u_bh,
            "total_used": total_used,
            "chua_ai_dung": total_used == 0,
            "da_dung_ge_3": total_used >= 3,
        })

    rows.sort(key=lambda r: (r["total_used"], -r["n_relevant"], r["spec_cluster_id"]))

    out_csv = PROJECT_ROOT / "reports/phase5b/b_human_cluster_worksheet.csv"
    fieldnames = [
        "slot_kind", "spec_cluster_id", "articleType", "gender", "baseColour",
        "n_relevant", "used_DEV", "used_A", "used_B_synth", "used_B_human_valid",
        "total_used", "chua_ai_dung", "da_dung_ge_3",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_unused = sum(1 for r in rows if r["chua_ai_dung"])
    n_ge3 = sum(1 for r in rows if r["da_dung_ge_3"])
    by_kind = Counter(r["slot_kind"] for r in rows)

    catalog_articles = sorted(val_df["articleType"].dropna().unique())
    catalog_genders = sorted(val_df["gender"].dropna().unique())
    catalog_colours = sorted(val_df["baseColour"].dropna().unique())

    lines: List[str] = [
        "# Bảng Cụm Khả dụng trên `index/val_gallery/` (Hỗ trợ Bổ sung B-human)",
        "",
        "- **Nguồn:** `index/val_gallery/id_map.json` (1.884 sản phẩm val sạch), chính sách GT `hybrid` (`eval/gender_gt_policy.json`).",
        "- **Điều kiện lọc:** Mọi tổ hợp 1-slot (`articleType`), 2-slot (`articleType+gender`, `articleType+baseColour`), 3-slot (`articleType+gender+baseColour`) có `n_relevant >= 5`.",
        "- **Lưu ý slot không nêu:** `gender=null` (không lọc giới tính), `baseColour` rỗng (không lọc màu).",
        f"- **Tổng số cụm khả dụng (`n_relevant >= 5`):** **{len(rows)}** cụm "
        f"(1-slot: {by_kind['1_slot (articleType)']}, 2-slot gender: {by_kind['2_slot (articleType+gender)']}, "
        f"2-slot colour: {by_kind['2_slot (articleType+baseColour)']}, 3-slot: {by_kind['3_slot (articleType+gender+baseColour)']}).",
        f"- **Cụm chưa ai dùng (`total_used == 0`):** **{n_unused}** cụm.",
        f"- **Cụm đã dùng >= 3 lần (`total_used >= 3`):** **{n_ge3}** cụm (cần tránh để hạn chế trùng lặp).",
        f"- **File CSV đầy đủ ({len(rows)} dòng):** `reports/phase5b/b_human_cluster_worksheet.csv`.",
        "",
        "## 1. Nhãn chuẩn trong `index/val_gallery/` (chép nguyên văn vào `spec`)",
        "",
        f"- **`articleType` ({len(catalog_articles)}):** " + ", ".join(f"`{x}`" for x in catalog_articles),
        f"- **`gender` ({len(catalog_genders)} + `null`):** " + ", ".join(f"`{x}`" for x in catalog_genders) + ", `null`",
        f"- **`baseColour` ({len(catalog_colours)} + `[]`):** " + ", ".join(f"`{x}`" for x in catalog_colours) + ", `[]`",
        "",
        "## 2. Top 60 Cụm Chưa Ai Dùng (`total_used == 0`, sắp theo `n_relevant` giảm dần)",
        "",
        "| Loại slot | `articleType` | `gender` | `baseColour` | `n_relevant` | `spec_cluster_id` |",
        "|---|---|---|---|---:|---|",
    ]
    unused_rows = [r for r in rows if r["chua_ai_dung"]]
    unused_rows.sort(key=lambda r: (-r["n_relevant"], r["spec_cluster_id"]))
    for r in unused_rows[:60]:
        bc = r["baseColour"] if r["baseColour"] else "`[]`"
        lines.append(f"| {r['slot_kind']} | `{r['articleType']}` | `{r['gender']}` | {bc} | {r['n_relevant']} | `{r['spec_cluster_id']}` |")

    lines.extend([
        "",
        "## 3. Các Cụm Đã Dùng >= 3 Lần (Nên tránh)",
        "",
        "| `spec_cluster_id` | `n_relevant` | DEV | A | B-synth | B-human (hợp lệ) | Tổng dùng |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    ge3_rows = [r for r in rows if r["da_dung_ge_3"]]
    ge3_rows.sort(key=lambda r: (-r["total_used"], r["spec_cluster_id"]))
    for r in ge3_rows:
        lines.append(
            f"| `{r['spec_cluster_id']}` | {r['n_relevant']} | {r['used_DEV']} | {r['used_A']} | {r['used_B_synth']} | {r['used_B_human_valid']} | {r['total_used']} |"
        )

    out_md = PROJECT_ROOT / "reports/phase5b/b_human_cluster_worksheet.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Worksheet rows: {len(rows)} (unused={n_unused}, used>=3={n_ge3})")
    print(f"CSV SHA-256: {hashlib.sha256(out_csv.read_bytes()).hexdigest()}")
    print(f"MD  SHA-256: {hashlib.sha256(out_md.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
