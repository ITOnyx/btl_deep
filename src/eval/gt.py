# -*- coding: utf-8 -*-
"""Ground truth resolution and lexicon checking helpers for held-out v2 evaluation."""
from __future__ import annotations

from typing import Any, Dict, List, Set
import pandas as pd

from src.eval.metrics import is_item_relevant, load_gender_gt_policy
from scripts.check_query_set import build_v2_lexicon_tokens, tokenize_norm

_V2_LEXICON_TOKENS: Set[str] | None = None


def is_in_lexicon_v21(query: str) -> bool:
    """Trả về True nếu tất cả các token (không trung tính) của câu đều thuộc lexicon v2.1."""
    global _V2_LEXICON_TOKENS
    if _V2_LEXICON_TOKENS is None:
        _V2_LEXICON_TOKENS = build_v2_lexicon_tokens()
    q_toks = tokenize_norm(query)
    return bool(q_toks and q_toks.issubset(_V2_LEXICON_TOKENS))


def resolve_relevant_ids(spec: Dict[str, Any], catalog: pd.DataFrame) -> List[str]:
    """
    Truy xuất danh sách ID liên quan theo chính sách Ground Truth Hybrid chính thức của dự án.
    spec = {"articleType": str, "gender": Optional[str], "baseColour": Optional[str]}
    """
    policy = load_gender_gt_policy()
    strict_cats = set(policy.get("policies", {}).get("hybrid", {}).get("strict_categories", []))

    spec_norm = {
        "articleType": [spec["articleType"]] if spec.get("articleType") else [],
        "gender": spec.get("gender"),
        "baseColour": [spec["baseColour"]] if spec.get("baseColour") else [],
    }

    matched_ids: List[str] = []
    for _, row in catalog.iterrows():
        meta = {
            "articleType": row.get("articleType"),
            "gender": row.get("gender"),
            "baseColour": row.get("baseColour"),
        }
        if is_item_relevant(spec_norm, meta, "hybrid", strict_cats):
            matched_ids.append(str(row["id"]))
    return matched_ids
