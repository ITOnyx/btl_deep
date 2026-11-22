"""
Sinh Worksheet cho các tập xác nhận mới (Bước 11.2, KHÔNG viết câu truy vấn, KHÔNG chạy model):
1. B-human-v4 (val_gallery 1.884):
   - reports/phase5c/b_human_v4_worksheet.csv & .md
   - Cụm khả dụng n_relevant >= 5, loại bỏ các cụm đã dùng ở B-human cũ, đánh dấu cụm hiếm.
2. held-out v2 (test_gallery 1.290):
   - reports/phase5c/heldout_v2_worksheet.csv & .md
   - Cụm theo thuộc tính catalog (KHÔNG chạy model), kèm phân bố usage/season từ styles.csv,
     đánh dấu nhóm khớp 3 slot có 5..70% ảnh khớp ràng buộc phụ usage/season và đánh dấu slot mỏng.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_query_set import make_cluster_id
from src.eval.metrics import is_item_relevant, load_gender_gt_policy


def _enumerate_candidates(df: pd.DataFrame) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for at in sorted(df["articleType"].dropna().unique()):
        candidates.append(("1_slot (articleType)", {"articleType": [str(at)], "gender": None, "baseColour": []}))
    for (at, gd), _ in df.groupby(["articleType", "gender"]):
        candidates.append(("2_slot (articleType+gender)", {"articleType": [str(at)], "gender": str(gd), "baseColour": []}))
    for (at, cl), _ in df.groupby(["articleType", "baseColour"]):
        candidates.append(("2_slot (articleType+baseColour)", {"articleType": [str(at)], "gender": None, "baseColour": [str(cl)]}))
    for (at, gd, cl), _ in df.groupby(["articleType", "gender", "baseColour"]):
        candidates.append(("3_slot (articleType+gender+baseColour)", {"articleType": [str(at)], "gender": str(gd), "baseColour": [str(cl)]}))
    return candidates


def generate_b_human_v4_worksheet(
    val_id_map: List[Dict[str, Any]],
    strict_cats: Set[str],
    out_dir: pathlib.Path,
) -> Dict[str, Any]:
    val_df = pd.DataFrame(val_id_map)
    at_counts = val_df["articleType"].value_counts().to_dict()
    col_counts = val_df["baseColour"].value_counts().to_dict()

    dev_data = json.loads((PROJECT_ROOT / "eval/handwritten_queries.json").read_text(encoding="utf-8"))
    a_data = json.loads((PROJECT_ROOT / "eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    bs_data = json.loads((PROJECT_ROOT / "eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    bh_data = json.loads((PROJECT_ROOT / "eval/val_queries_b_human.json").read_text(encoding="utf-8"))

    cnt_dev = Counter(make_cluster_id(x["spec"]) for x in dev_data)
    cnt_a = Counter(make_cluster_id(x["spec"]) for x in a_data)
    cnt_bs = Counter(make_cluster_id(x["spec"]) for x in bs_data)
    used_b_human_old: Set[str] = {make_cluster_id(x["spec"]) for x in bh_data}

    candidates = _enumerate_candidates(val_df)
    rows: List[Dict[str, Any]] = []
    excluded_old_count = 0

    for slot_kind, spec in candidates:
        cid = make_cluster_id(spec)
        n_rel = sum(1 for m in val_id_map if is_item_relevant(spec, m, "hybrid", strict_cats))
        if n_rel < 5:
            continue
        if cid in used_b_human_old:
            excluded_old_count += 1
            continue

        at = spec["articleType"][0]
        gd = spec["gender"] if spec["gender"] is not None else "null"
        cl = spec["baseColour"][0] if spec["baseColour"] else ""

        is_rare = (
            (5 <= n_rel <= 15)
            or (at_counts.get(at, 0) < 20)
            or (bool(cl) and col_counts.get(cl, 0) < 20)
        )
        u_dev = cnt_dev[cid]
        u_a = cnt_a[cid]
        u_bs = cnt_bs[cid]
        total_other = u_dev + u_a + u_bs

        rows.append({
            "slot_kind": slot_kind,
            "spec_cluster_id": cid,
            "articleType": at,
            "gender": gd,
            "baseColour": cl,
            "n_relevant": n_rel,
            "is_rare": is_rare,
            "used_DEV": u_dev,
            "used_A": u_a,
            "used_B_synth": u_bs,
            "total_used_other": total_other,
            "completely_unused": total_other == 0,
        })

    rows.sort(key=lambda r: (r["total_used_other"], not r["is_rare"], -r["n_relevant"], r["spec_cluster_id"]))

    csv_path = out_dir / "b_human_v4_worksheet.csv"
    fieldnames = [
        "slot_kind", "spec_cluster_id", "articleType", "gender", "baseColour",
        "n_relevant", "is_rare", "used_DEV", "used_A", "used_B_synth",
        "total_used_other", "completely_unused",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_rare = sum(1 for r in rows if r["is_rare"])
    n_unused = sum(1 for r in rows if r["completely_unused"])
    by_kind = Counter(r["slot_kind"] for r in rows)

    md_lines = [
        "# Worksheet B-human-v4 (`index/val_gallery/` — 1.884 sản phẩm)",
        "",
        "> **Chỉ dẫn:** Worksheet chỉ liệt kê cụm khả dụng (`n_relevant >= 5`) và **đã loại bỏ** toàn bộ cụm từng dùng trong `B-human` cũ (`eval/val_queries_b_human.json`). Không viết sẵn câu truy vấn.",
        "",
        "## 1. Yêu cầu tập `B-human-v4`",
        "- Số câu hợp lệ: **`>= 60` câu**",
        "- Số cụm `spec_cluster_id` khác nhau: **`>= 30` cụm**",
        "- Số lượng ảnh đúng mỗi câu (`n_relevant`): **`>= 5`** (theo chính sách `hybrid`)",
        "- Câu ngoài lexicon v2.1: **`>= 20` câu**",
        "- Câu không dấu (`dia_class == \"none\"`): **`>= 20` câu**",
        "- Câu lỗi gõ (`typo_type` khác null): **đúng `10` câu**",
        "- Không trùng lặp Jaccard `>= 0.8` với `DEV / A / B-synth / C / B-human cũ` và nội bộ.",
        "",
        "## 2. Thống kê cụm khả dụng",
        f"- Số cụm đã loại vì trùng `B-human` cũ: **{excluded_old_count}**",
        f"- Tổng cụm khả dụng còn lại (`n_relevant >= 5`): **{len(rows)}** "
        f"(1-slot: {by_kind['1_slot (articleType)']}, 2-slot gender: {by_kind['2_slot (articleType+gender)']}, "
        f"2-slot colour: {by_kind['2_slot (articleType+baseColour)']}, 3-slot: {by_kind['3_slot (articleType+gender+baseColour)']})",
        f"- Số cụm hiếm (`is_rare=True`, `n_relevant` 5..15 hoặc `articleType`/`baseColour` < 20 mẫu trong val): **{n_rare}**",
        f"- Số cụm chưa từng xuất hiện ở DEV/A/B-synth: **{n_unused}**",
        f"- File CSV đầy đủ: `reports/phase5c/b_human_v4_worksheet.csv`",
        "",
        "## 3. Mẫu Top 40 Cụm Hiếm Chưa Từng Dùng (`is_rare=True`, `completely_unused=True`)",
        "",
        "| Loại slot | `articleType` | `gender` | `baseColour` | `n_relevant` | `spec_cluster_id` |",
        "|---|---|---|---|---:|---|",
    ]
    rare_unused = [r for r in rows if r["is_rare"] and r["completely_unused"]]
    rare_unused.sort(key=lambda r: (-r["n_relevant"], r["spec_cluster_id"]))
    for r in rare_unused[:40]:
        bc = r["baseColour"] if r["baseColour"] else "`[]`"
        md_lines.append(f"| {r['slot_kind']} | `{r['articleType']}` | `{r['gender']}` | {bc} | {r['n_relevant']} | `{r['spec_cluster_id']}` |")

    md_lines.extend([
        "",
        "## 4. Mẫu Top 40 Cụm Phổ Biến Chưa Từng Dùng (`is_rare=False`, `completely_unused=True`)",
        "",
        "| Loại slot | `articleType` | `gender` | `baseColour` | `n_relevant` | `spec_cluster_id` |",
        "|---|---|---|---|---:|---|",
    ])
    common_unused = [r for r in rows if (not r["is_rare"]) and r["completely_unused"]]
    common_unused.sort(key=lambda r: (-r["n_relevant"], r["spec_cluster_id"]))
    for r in common_unused[:40]:
        bc = r["baseColour"] if r["baseColour"] else "`[]`"
        md_lines.append(f"| {r['slot_kind']} | `{r['articleType']}` | `{r['gender']}` | {bc} | {r['n_relevant']} | `{r['spec_cluster_id']}` |")

    (out_dir / "b_human_v4_worksheet.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "total_clusters": len(rows),
        "excluded_old_b_human": excluded_old_count,
        "rare_clusters": n_rare,
        "completely_unused": n_unused,
    }


def generate_heldout_v2_worksheet(
    test_id_map: List[Dict[str, Any]],
    strict_cats: Set[str],
    out_dir: pathlib.Path,
) -> Dict[str, Any]:
    styles = pd.read_csv(PROJECT_ROOT / "data/fashion-dataset/styles.csv", on_bad_lines="skip")
    styles["id_str"] = styles["id"].astype(str)
    meta_lookup = styles.set_index("id_str")[["usage", "season"]].to_dict(orient="index")

    enriched_map: List[Dict[str, Any]] = []
    for m in test_id_map:
        pid = str(m["product_id"])
        extra = meta_lookup.get(pid, {})
        item = dict(m)
        u = extra.get("usage")
        s = extra.get("season")
        item["usage"] = None if pd.isna(u) else str(u)
        item["season"] = None if pd.isna(s) else str(s)
        enriched_map.append(item)

    test_df = pd.DataFrame(enriched_map)
    at_counts = test_df["articleType"].value_counts().to_dict()
    col_counts = test_df["baseColour"].value_counts().to_dict()

    candidates = _enumerate_candidates(test_df)
    rows: List[Dict[str, Any]] = []

    for slot_kind, spec in candidates:
        rel_items = [m for m in enriched_map if is_item_relevant(spec, m, "hybrid", strict_cats)]
        n_rel = len(rel_items)
        if n_rel < 5:
            continue

        cid = make_cluster_id(spec)
        at = spec["articleType"][0]
        gd = spec["gender"] if spec["gender"] is not None else "null"
        cl = spec["baseColour"][0] if spec["baseColour"] else ""

        is_thin = (
            (5 <= n_rel <= 15)
            or (at_counts.get(at, 0) < 15)
            or (bool(cl) and col_counts.get(cl, 0) < 15)
        )

        usage_cnt = Counter(m["usage"] for m in rel_items if m["usage"])
        season_cnt = Counter(m["season"] for m in rel_items if m["season"])

        usage_5_70 = [
            f"{k}:{v}/{n_rel}({v/n_rel*100:.0f}%)"
            for k, v in usage_cnt.most_common()
            if 0.05 <= (v / n_rel) <= 0.70
        ]
        season_5_70 = [
            f"{k}:{v}/{n_rel}({v/n_rel*100:.0f}%)"
            for k, v in season_cnt.most_common()
            if 0.05 <= (v / n_rel) <= 0.70
        ]
        has_5_70 = bool(usage_5_70 or season_5_70)

        rows.append({
            "slot_kind": slot_kind,
            "spec_cluster_id": cid,
            "articleType": at,
            "gender": gd,
            "baseColour": cl,
            "n_relevant": n_rel,
            "is_thin_slot": is_thin,
            "usage_dist": json.dumps(dict(usage_cnt), ensure_ascii=False),
            "season_dist": json.dumps(dict(season_cnt), ensure_ascii=False),
            "usage_5_70pct": "; ".join(usage_5_70),
            "season_5_70pct": "; ".join(season_5_70),
            "has_usage_season_5_70pct": has_5_70,
        })

    rows.sort(key=lambda r: (not r["has_usage_season_5_70pct"], not r["is_thin_slot"], -r["n_relevant"], r["spec_cluster_id"]))

    csv_path = out_dir / "heldout_v2_worksheet.csv"
    fieldnames = [
        "slot_kind", "spec_cluster_id", "articleType", "gender", "baseColour",
        "n_relevant", "is_thin_slot", "usage_dist", "season_dist",
        "usage_5_70pct", "season_5_70pct", "has_usage_season_5_70pct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    by_kind = Counter(r["slot_kind"] for r in rows)
    n_thin = sum(1 for r in rows if r["is_thin_slot"])
    n_3slot_5_70 = sum(
        1 for r in rows
        if r["slot_kind"].startswith("3_slot") and r["has_usage_season_5_70pct"]
    )

    md_lines = [
        "# Worksheet held-out v2 (`index/test_gallery/` — 1.290 sản phẩm, KHÔNG chạy model)",
        "",
        "> **Tuân thủ Luật C.5 & C.10:** Chỉ thống kê thuộc tính catalog (`id_map.json` + `styles.csv`). KHÔNG nạp/chạy model trên `test_gallery`. KHÔNG viết sẵn câu truy vấn.",
        "",
        "## 1. Yêu cầu tập `held-out v2`",
        "- Số câu truy vấn: **`>= 100` câu**",
        "- Số lượng ảnh đúng mỗi câu (`n_relevant`): **`>= 5`**",
        "- Câu không dấu (`dia_class == \"none\"`): **`>= 40` câu**",
        "- Câu ngoài lexicon v2.1: **`>= 30%`** tổng số câu",
        "- Câu có tín hiệu `usage`/`season` (nhóm khớp 3 slot có **5%..70%** ảnh khớp ràng buộc phụ): **`>= 40` câu**",
        "- Phủ đủ các **slot mỏng** (`is_thin_slot=True`).",
        "",
        "## 2. Thống kê cụm khả dụng trên `test_gallery`",
        f"- Tổng số cụm khả dụng (`n_relevant >= 5`): **{len(rows)}** "
        f"(1-slot: {by_kind['1_slot (articleType)']}, 2-slot gender: {by_kind['2_slot (articleType+gender)']}, "
        f"2-slot colour: {by_kind['2_slot (articleType+baseColour)']}, 3-slot: {by_kind['3_slot (articleType+gender+baseColour)']})",
        f"- Số cụm slot mỏng (`is_thin_slot=True`): **{n_thin}**",
        f"- Số cụm **3-slot** có tín hiệu `usage`/`season` trong khoảng **5%..70%**: **{n_3slot_5_70}**",
        f"- File CSV đầy đủ: `reports/phase5c/heldout_v2_worksheet.csv`",
        "",
        "## 3. Top 50 Nhóm 3-Slot Có Tín Hiệu `usage`/`season` (5%..70%)",
        "",
        "| `articleType` | `gender` | `baseColour` | `n_rel` | `usage` (5..70%) | `season` (5..70%) |",
        "|---|---|---|---:|---|---|",
    ]
    top_3slot = [r for r in rows if r["slot_kind"].startswith("3_slot") and r["has_usage_season_5_70pct"]]
    top_3slot.sort(key=lambda r: (-r["n_relevant"], r["spec_cluster_id"]))
    for r in top_3slot[:50]:
        md_lines.append(
            f"| `{r['articleType']}` | `{r['gender']}` | `{r['baseColour']}` | {r['n_relevant']} | "
            f"{r['usage_5_70pct'] or '-'} | {r['season_5_70pct'] or '-'} |"
        )

    md_lines.extend([
        "",
        "## 4. Top 40 Cụm Slot Mỏng (`is_thin_slot=True`, `n_relevant >= 5`)",
        "",
        "| Loại slot | `articleType` | `gender` | `baseColour` | `n_rel` | `usage/season` (5..70%) |",
        "|---|---|---|---|---:|---|",
    ])
    thin_rows = [r for r in rows if r["is_thin_slot"]]
    thin_rows.sort(key=lambda r: (r["n_relevant"], r["spec_cluster_id"]))
    for r in thin_rows[:40]:
        bc = r["baseColour"] if r["baseColour"] else "`[]`"
        us = "; ".join(x for x in [r["usage_5_70pct"], r["season_5_70pct"]] if x) or "-"
        md_lines.append(f"| {r['slot_kind']} | `{r['articleType']}` | `{r['gender']}` | {bc} | {r['n_relevant']} | {us} |")

    (out_dir / "heldout_v2_worksheet.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "total_clusters": len(rows),
        "thin_slot_clusters": n_thin,
        "three_slot_with_5_70_usage_season": n_3slot_5_70,
    }


def main() -> None:
    out_dir = PROJECT_ROOT / "reports/phase5c"
    out_dir.mkdir(parents=True, exist_ok=True)
    strict_cats = set(
        load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")["policies"]["hybrid"]["strict_categories"]
    )
    val_id_map = json.loads((PROJECT_ROOT / "index/val_gallery/id_map.json").read_text(encoding="utf-8"))
    test_id_map = json.loads((PROJECT_ROOT / "index/test_gallery/id_map.json").read_text(encoding="utf-8"))

    bh_stats = generate_b_human_v4_worksheet(val_id_map, strict_cats, out_dir)
    ho_stats = generate_heldout_v2_worksheet(test_id_map, strict_cats, out_dir)
    print("B-human-v4 worksheet:", bh_stats)
    print("held-out v2 worksheet:", ho_stats)


if __name__ == "__main__":
    main()
