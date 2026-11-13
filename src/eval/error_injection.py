"""
Module tiêm lỗi parse có seed (Seed=42) độc lập với lexicon cho Tập C (Parse-error probe).
Phiên bản 2c.3:
- Lỗi LỒNG NHAU (nested): mỗi truy vấn có u ~ U(0, 1) (seed 42), bị lỗi ở mức r nếu u < r;
  loại lỗi được chọn một lần cho mỗi truy vấn, nhất quán giữa các mức.
- Chỉ tiêm lỗi có hiệu lực: không drop slot đã rỗng; không flip gender null/Unisex vô nghĩa.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

VALID_FLIP_GENDERS = {"Men": "Women", "Women": "Men", "Boys": "Girls", "Girls": "Boys"}


def _build_effective_error_for_item(
    clean_item: Dict[str, Any],
    rng: random.Random,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Chọn ngẫu nhiên MỘT lỗi có hiệu lực (thực sự làm thay đổi parsed) cho clean_item.
    Trả về (mutated_item, error_type). Nếu không có slot nào hợp lệ, trả về (deepcopy, None).
    """
    mutated = copy.deepcopy(clean_item)
    base_weights = {
        "wrong_article": 30,
        "flip_gender": 10,
        "drop_slot": 20,
        "wrong_colour": 10,
    }

    eligible_types: List[str] = []
    eligible_weights: List[int] = []

    if mutated.get("articleType"):
        eligible_types.append("wrong_article")
        eligible_weights.append(base_weights["wrong_article"])

    if mutated.get("gender") in VALID_FLIP_GENDERS:
        eligible_types.append("flip_gender")
        eligible_weights.append(base_weights["flip_gender"])

    present_slots = [k for k in ("articleType", "gender", "baseColour") if mutated.get(k)]
    if present_slots:
        eligible_types.append("drop_slot")
        eligible_weights.append(base_weights["drop_slot"])

    if mutated.get("baseColour"):
        eligible_types.append("wrong_colour")
        eligible_weights.append(base_weights["wrong_colour"])

    if not eligible_types:
        return mutated, None

    etype = rng.choices(eligible_types, weights=eligible_weights, k=1)[0]

    if etype == "wrong_article":
        orig_set = set(mutated["articleType"])
        orig_first = mutated["articleType"][0]
        cands = [c for c in CONFUSABLE_ARTICLE_TYPES.get(orig_first, []) if c not in orig_set]
        if not cands:
            cands = [k for k in CONFUSABLE_ARTICLE_TYPES.keys() if k not in orig_set]
        mutated["articleType"] = [rng.choice(cands)]
    elif etype == "flip_gender":
        mutated["gender"] = VALID_FLIP_GENDERS[mutated["gender"]]
    elif etype == "drop_slot":
        drop_k = rng.choice(present_slots)
        mutated[drop_k] = None if drop_k == "gender" else []
    elif etype == "wrong_colour":
        orig_cols = set(mutated["baseColour"])
        cands_c = [c for c in ALL_COLOURS if c not in orig_cols]
        mutated["baseColour"] = [rng.choice(cands_c)]

    return mutated, etype


def inject_nested_parse_errors(
    clean_parsed_list: List[Dict[str, Any]],
    error_rates: Sequence[float] = (0.0, 0.1, 0.2, 0.3),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sinh các mức tiêm lỗi LỒNG NHAU với cùng seed:
    - Mỗi truy vấn i có u_i ~ U(0, 1) và 1 lỗi cố định (mutated_i, etype_i).
    - Ở mức r, truy vấn i nhận mutated_i nếu u_i < r (và etype_i is not None).
    """
    rng = random.Random(seed)
    per_query_plan: List[Tuple[float, Dict[str, Any], Optional[str]]] = []
    for p in clean_parsed_list:
        u = rng.random()
        mutated, etype = _build_effective_error_for_item(p, rng)
        per_query_plan.append((u, mutated, etype))

    levels: Dict[float, Dict[str, Any]] = {}
    n_total = len(clean_parsed_list)
    for r in error_rates:
        items_out: List[Dict[str, Any]] = []
        n_effective = 0
        error_flags: List[bool] = []
        error_types: List[Optional[str]] = []
        for idx, p in enumerate(clean_parsed_list):
            u, mutated, etype = per_query_plan[idx]
            if r > 0.0 and u < r and etype is not None:
                items_out.append(copy.deepcopy(mutated))
                n_effective += 1
                error_flags.append(True)
                error_types.append(etype)
            else:
                items_out.append(copy.deepcopy(p))
                error_flags.append(False)
                error_types.append(None)

        levels[float(r)] = {
            "parsed": items_out,
            "n_effective_errors": n_effective,
            "actual_error_rate": float(n_effective / n_total) if n_total else 0.0,
            "error_flags": error_flags,
            "error_types": error_types,
        }

    return {
        "seed": seed,
        "u_values": [x[0] for x in per_query_plan],
        "planned_error_types": [x[2] for x in per_query_plan],
        "levels": levels,
    }


def inject_parse_errors(
    clean_parsed_list: List[Dict[str, Any]],
    error_rate: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Wrapper tương thích ngược trả về danh sách parsed ở mức error_rate theo cơ chế lồng nhau.
    """
    res = inject_nested_parse_errors(clean_parsed_list, error_rates=(error_rate,), seed=seed)
    return res["levels"][float(error_rate)]["parsed"]
