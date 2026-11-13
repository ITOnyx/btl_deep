"""
Module tiêm lỗi parse có seed (Seed=42) độc lập với lexicon cho Tập C (Parse-error probe).
Dùng để đo đường cong tác hại theo mức lỗi (0%, 10%, 20%, 30%) trên val_gallery.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional


# Các cặp articleType dễ nhầm lẫn theo 2b.4
CONFUSABLE_ARTICLE_TYPES: Dict[str, List[str]] = {
    "Casual Shoes": ["Formal Shoes", "Flats", "Sports Shoes"],
    "Formal Shoes": ["Casual Shoes"],
    "Flats": ["Casual Shoes", "Heels"],
    "Heels": ["Flats"],
    "Tops": ["Shirts", "Tshirts"],
    "Shirts": ["Tops", "Tshirts"],
    "Tshirts": ["Tops", "Shirts"],
    "Handbags": ["Clutches", "Backpacks"],
    "Clutches": ["Handbags", "Wallets"],
    "Sweaters": ["Sweatshirts", "Jackets"],
    "Sweatshirts": ["Sweaters"],
    "Jackets": ["Blazers", "Sweaters"],
    "Blazers": ["Jackets"],
}

ALL_COLOURS = [
    "Black", "White", "Blue", "Navy Blue", "Grey", "Red", "Green", "Brown", "Pink", "Silver", "Beige", "Yellow", "Purple"
]


def inject_parse_errors(
    clean_parsed_list: List[Dict[str, Any]],
    error_rate: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Tiêm lỗi parse vào danh sách kết quả parse chuẩn (clean_parsed_list) với tỉ lệ error_rate (vd 0.0, 0.1, 0.2, 0.3).
    Phân bố loại lỗi trong phần bị tiêm (tỉ lệ 30 : 10 : 20 : 10, chuẩn hóa tổng = 70):
      - articleType sai sang lớp dễ nhầm: 30/70 (~42.9%)
      - gender đảo (Men <-> Women): 10/70 (~14.3%)
      - slot bị bỏ (xóa 1 slot bất kỳ có mặt): 20/70 (~28.6%)
      - baseColour sai sang màu khác: 10/70 (~14.3%)
    """
    if error_rate <= 0.0:
        return [copy.deepcopy(p) for p in clean_parsed_list]

    rng = random.Random(seed + int(round(error_rate * 1000)))
    out: List[Dict[str, Any]] = []

    # Trọng số loại lỗi theo 2b.4: 30, 10, 20, 10
    error_types = ["wrong_article", "flip_gender", "drop_slot", "wrong_colour"]
    weights = [30, 10, 20, 10]

    for p in clean_parsed_list:
        item = copy.deepcopy(p)
        if rng.random() < error_rate:
            etype = rng.choices(error_types, weights=weights, k=1)[0]
            if etype == "wrong_article" and item.get("articleType"):
                orig = item["articleType"][0]
                cands = CONFUSABLE_ARTICLE_TYPES.get(orig)
                if cands:
                    item["articleType"] = [rng.choice(cands)]
                else:
                    # Fallback sang loại phổ biến khác
                    fallback = [k for k in CONFUSABLE_ARTICLE_TYPES.keys() if k != orig]
                    item["articleType"] = [rng.choice(fallback)]
            elif etype == "flip_gender" and item.get("gender"):
                g = item["gender"]
                if g == "Men":
                    item["gender"] = "Women"
                elif g == "Women":
                    item["gender"] = "Men"
                elif g == "Boys":
                    item["gender"] = "Girls"
                elif g == "Girls":
                    item["gender"] = "Boys"
                else:
                    item["gender"] = rng.choice(["Men", "Women"])
            elif etype == "drop_slot":
                present = [k for k in ["articleType", "gender", "baseColour"] if item.get(k)]
                if present:
                    drop_k = rng.choice(present)
                    item[drop_k] = None if drop_k == "gender" else []
            elif etype == "wrong_colour" and item.get("baseColour"):
                orig_c = item["baseColour"][0]
                cands_c = [c for c in ALL_COLOURS if c != orig_c]
                item["baseColour"] = [rng.choice(cands_c)]
            else:
                # Nếu slot mục tiêu rỗng, rơi về drop_slot
                present = [k for k in ["articleType", "gender", "baseColour"] if item.get(k)]
                if present:
                    drop_k = rng.choice(present)
                    item[drop_k] = None if drop_k == "gender" else []
        out.append(item)

    return out
