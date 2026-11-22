"""
Dựng bộ đánh giá CIR-eval (Bước 11.2, KHÔNG nạp/chạy model, seed=42):
- eval/cir_eval_val.json (>= 100 cặp trên index/val_gallery/ để tuning)
- eval/cir_eval_test.json (>= 100 cặp trên index/test_gallery/ cho báo cáo cuối, chạy 1 lần)

Quy tắc:
- Sinh thuần túy từ thuộc tính catalog (catalog_rebuild.parquet + id_map.json), KHÔNG lấy từ câu của tập khóa.
- Mỗi cặp gồm: (ảnh tham chiếu, modifier tiếng Việt, tập ảnh đích GT).
- GT = cùng articleType, đổi đúng 1 thuộc tính (baseColour hoặc gender), giữ nguyên các thuộc tính còn lại.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COLOUR_MOD_TEMPLATES = [
    "đổi sang màu {val}",
    "chuyển sang màu {val}",
    "mẫu tương tự màu {val}",
]

GENDER_MOD_TEMPLATES = [
    "đổi sang cho {val}",
    "phiên bản dành cho {val}",
    "mẫu dành cho {val}",
]


def _build_vi_maps(catalog_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    at_vi = {}
    gd_vi = {}
    cl_vi = {}
    for _, row in catalog_df[["articleType", "articleType_vi"]].dropna().drop_duplicates().iterrows():
        at_vi[str(row["articleType"])] = str(row["articleType_vi"]).strip().lower()
    for _, row in catalog_df[["gender", "gender_vi"]].dropna().drop_duplicates().iterrows():
        gd_vi[str(row["gender"])] = str(row["gender_vi"]).strip().lower()
    for _, row in catalog_df[["baseColour", "baseColour_vi"]].dropna().drop_duplicates().iterrows():
        cl_vi[str(row["baseColour"])] = str(row["baseColour_vi"]).strip().lower()
    return at_vi, gd_vi, cl_vi


def generate_cir_pairs_for_gallery(
    id_map: List[Dict[str, Any]],
    split_name: str,
    at_vi_map: Dict[str, str],
    gd_vi_map: Dict[str, str],
    cl_vi_map: Dict[str, str],
    target_n: int = 120,
    min_gt: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    # Nhóm sản phẩm theo (articleType, gender, baseColour)
    by_triplet: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for m in id_map:
        at = str(m.get("articleType") or "")
        gd = str(m.get("gender") or "")
        cl = str(m.get("baseColour") or "")
        if at and gd and cl:
            by_triplet[(at, gd, cl)].append(m)

    # Tập hợp các cặp ứng viên (đổi 1 thuộc tính: baseColour hoặc gender)
    colour_candidates: List[Dict[str, Any]] = []
    gender_candidates: List[Dict[str, Any]] = []

    triplets = sorted(by_triplet.keys())
    for at, gd, cl in triplets:
        refs = by_triplet[(at, gd, cl)]
        # 1) Đổi baseColour (cùng articleType, cùng gender)
        other_colours = sorted(
            {
                c2
                for (a2, g2, c2), items in by_triplet.items()
                if a2 == at and g2 == gd and c2 != cl and len(items) >= min_gt
            }
        )
        for c2 in other_colours:
            gt_items = by_triplet[(at, gd, c2)]
            gt_pids = [str(x.get("product_id") or x.get("item_id")) for x in gt_items]
            ref = rng.choice(refs)
            c2_vi = cl_vi_map.get(c2, c2.lower())
            tpl = rng.choice(COLOUR_MOD_TEMPLATES)
            colour_candidates.append({
                "ref_product_id": str(ref.get("product_id") or ref.get("item_id")),
                "ref_image_path": str(ref.get("path") or ref.get("image_path") or ""),
                "ref_attrs": {"articleType": at, "gender": gd, "baseColour": cl},
                "changed_attr": "baseColour",
                "target_attr_value": c2,
                "target_attrs": {"articleType": at, "gender": gd, "baseColour": c2},
                "modifier_vi": tpl.format(val=c2_vi),
                "gt_product_ids": gt_pids,
                "n_gt": len(gt_pids),
                "source": "catalog_attributes",
            })

        # 2) Đổi gender (cùng articleType, cùng baseColour)
        other_genders = sorted(
            {
                g2
                for (a2, g2, c2), items in by_triplet.items()
                if a2 == at and c2 == cl and g2 != gd and len(items) >= min_gt
            }
        )
        for g2 in other_genders:
            gt_items = by_triplet[(at, g2, cl)]
            gt_pids = [str(x.get("product_id") or x.get("item_id")) for x in gt_items]
            ref = rng.choice(refs)
            g2_vi = gd_vi_map.get(g2, g2.lower())
            tpl = rng.choice(GENDER_MOD_TEMPLATES)
            gender_candidates.append({
                "ref_product_id": str(ref.get("product_id") or ref.get("item_id")),
                "ref_image_path": str(ref.get("path") or ref.get("image_path") or ""),
                "ref_attrs": {"articleType": at, "gender": gd, "baseColour": cl},
                "changed_attr": "gender",
                "target_attr_value": g2,
                "target_attrs": {"articleType": at, "gender": g2, "baseColour": cl},
                "modifier_vi": tpl.format(val=g2_vi),
                "gt_product_ids": gt_pids,
                "n_gt": len(gt_pids),
                "source": "catalog_attributes",
            })

    # Lấy mẫu phân tầng theo articleType, ưu tiên giữ tỷ lệ cân bằng giữa gender và colour
    rng.shuffle(gender_candidates)
    rng.shuffle(colour_candidates)

    selected: List[Dict[str, Any]] = []
    seen_keys = set()

    def _pick_diverse(pool: List[Dict[str, Any]], quota: int) -> None:
        # Vòng 1: tối đa 2 cặp mỗi articleType; Vòng 2: lấy thêm nếu thiếu
        per_at: Dict[str, int] = defaultdict(int)
        for cap in (2, 4, 999):
            for item in pool:
                if len([x for x in selected if x["changed_attr"] == item["changed_attr"]]) >= quota:
                    return
                key = (item["ref_product_id"], item["changed_attr"], item["target_attr_value"])
                if key in seen_keys:
                    continue
                at = item["ref_attrs"]["articleType"]
                if per_at[at] >= cap:
                    continue
                seen_keys.add(key)
                per_at[at] += 1
                selected.append(item)

    n_gender_target = min(len(gender_candidates), target_n // 3)
    _pick_diverse(gender_candidates, n_gender_target)
    n_colour_target = target_n - len(selected)
    _pick_diverse(colour_candidates, n_colour_target)

    # Gán ID chuẩn
    for idx, item in enumerate(selected):
        item["id"] = f"cir_{split_name}_{idx:03d}"

    return {
        "split": split_name,
        "seed": seed,
        "source": "catalog_attributes",
        "min_gt_per_pair": min_gt,
        "n_pairs": len(selected),
        "n_colour_change": sum(1 for x in selected if x["changed_attr"] == "baseColour"),
        "n_gender_change": sum(1 for x in selected if x["changed_attr"] == "gender"),
        "n_unique_article_types": len({x["ref_attrs"]["articleType"] for x in selected}),
        "pairs": selected,
    }


def main() -> None:
    cat = pd.read_parquet(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    at_vi, gd_vi, cl_vi = _build_vi_maps(cat)

    val_id_map = json.loads((PROJECT_ROOT / "index/val_gallery/id_map.json").read_text(encoding="utf-8"))
    test_id_map = json.loads((PROJECT_ROOT / "index/test_gallery/id_map.json").read_text(encoding="utf-8"))

    val_cir = generate_cir_pairs_for_gallery(val_id_map, "val", at_vi, gd_vi, cl_vi, target_n=120, min_gt=3, seed=42)
    test_cir = generate_cir_pairs_for_gallery(test_id_map, "test", at_vi, gd_vi, cl_vi, target_n=120, min_gt=3, seed=42)

    val_out = PROJECT_ROOT / "eval/cir_eval_val.json"
    test_out = PROJECT_ROOT / "eval/cir_eval_test.json"

    val_out.write_text(json.dumps(val_cir, ensure_ascii=False, indent=2), encoding="utf-8")
    test_out.write_text(json.dumps(test_cir, ensure_ascii=False, indent=2), encoding="utf-8")

    for p in (val_out, test_out):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        p.with_suffix(".sha256").write_text(f"{h}  {p.name}\n", encoding="utf-8")

    print("CIR-eval val:", {k: v for k, v in val_cir.items() if k != "pairs"})
    print("CIR-eval test:", {k: v for k, v in test_cir.items() if k != "pairs"})


if __name__ == "__main__":
    main()
