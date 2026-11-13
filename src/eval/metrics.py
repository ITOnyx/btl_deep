"""Retrieval metrics, GT matching, hubness, and bootstrap CI utilities for Phase 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_gender_gt_policy(policy_path: str | Path = "eval/gender_gt_policy.json") -> Dict[str, Any]:
    """Load gender ground-truth matching policy from JSON."""
    with open(policy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_gender_match(
    query_gender: Optional[str],
    item_gender: Optional[str],
    item_article_type: Optional[str],
    policy: str = "hybrid",
    strict_categories: Optional[Sequence[str]] = None,
) -> bool:
    """Check whether item_gender satisfies query_gender under strict/relaxed/hybrid policy."""
    if not query_gender:
        return True
    if not item_gender:
        return False

    qg = str(query_gender).strip()
    ig = str(item_gender).strip()
    if qg == ig:
        return True

    if policy == "strict":
        return False
    if policy == "relaxed":
        return ig == "Unisex"
    if policy == "hybrid":
        if ig != "Unisex":
            return False
        strict_set = set(strict_categories or [])
        if item_article_type and str(item_article_type).strip() in strict_set:
            return False
        return True

    raise ValueError(f"Unknown gender policy: {policy}")


def is_item_relevant(
    spec: Dict[str, Any],
    item_meta: Dict[str, Any],
    gender_policy: str = "hybrid",
    strict_categories: Optional[Sequence[str]] = None,
) -> bool:
    """Check if a catalog item matches a handwritten query's ground-truth spec."""
    # 1. articleType (required if non-empty in spec)
    spec_types = spec.get("articleType") or []
    item_type = str(item_meta.get("articleType", "")).strip()
    if spec_types and item_type not in spec_types:
        return False

    # 2. gender
    spec_gender = spec.get("gender")
    if spec_gender:
        if not is_gender_match(
            query_gender=spec_gender,
            item_gender=item_meta.get("gender"),
            item_article_type=item_type,
            policy=gender_policy,
            strict_categories=strict_categories,
        ):
            return False

    # 3. baseColour (required if non-empty in spec)
    spec_colours = spec.get("baseColour") or []
    if spec_colours:
        item_colour = str(item_meta.get("baseColour", "")).strip()
        if item_colour not in spec_colours:
            return False

    return True


def compute_recall_at_k(relevances: Sequence[int], k: int) -> float:
    """Binary Recall@K (hit@K): 1.0 if at least one relevant item in top-K else 0.0."""
    return 1.0 if any(relevances[:k]) else 0.0


def compute_average_precision(relevances: Sequence[int], total_relevant: Optional[int] = None) -> float:
    """Compute Average Precision (AP) over ranked binary relevance list."""
    hits = 0
    sum_precisions = 0.0
    for rank_idx, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            sum_precisions += hits / rank_idx
    if hits == 0:
        return 0.0
    denom = total_relevant if (total_relevant is not None and total_relevant > 0) else hits
    return sum_precisions / min(denom, len(relevances)) if total_relevant is None else sum_precisions / denom


def compute_retrieval_summary(
    ranked_relevances: List[List[int]],
    total_relevants: Optional[List[int]] = None,
    topk: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute mean Recall@K and mAP across queries."""
    n = len(ranked_relevances)
    if n == 0:
        out = {f"recall@{k}": 0.0 for k in topk}
        out["mAP"] = 0.0
        out["n_queries"] = 0
        return out

    out: Dict[str, float] = {"n_queries": n}
    for k in topk:
        out[f"recall@{k}"] = float(np.mean([compute_recall_at_k(r, k) for r in ranked_relevances]))

    aps = []
    for i, r in enumerate(ranked_relevances):
        tot = total_relevants[i] if total_relevants is not None else None
        aps.append(compute_average_precision(r, total_relevant=tot))
    out["mAP"] = float(np.mean(aps))
    return out


def bootstrap_ci_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap CI for mean(B) - mean(A)."""
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    n = len(arr_a)
    if n == 0:
        return {"mean_a": 0.0, "mean_b": 0.0, "diff_b_minus_a": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    delta = arr_b - arr_a
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(delta[idx])

    alpha = (1.0 - ci) / 2.0
    return {
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "diff_b_minus_a": float(np.mean(delta)),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1.0 - alpha)),
    }


def compute_hubness_stats(
    topk_indices_per_query: Sequence[Sequence[int]],
    gallery_size: int,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Measure hubness from top-K retrieved gallery indices across queries."""
    counts = np.zeros(gallery_size, dtype=np.int64)
    n_queries = len(topk_indices_per_query)
    for idxs in topk_indices_per_query:
        for idx in idxs[:top_k]:
            if 0 <= idx < gallery_size:
                counts[idx] += 1

    nonzero = counts[counts > 0]
    sorted_counts = np.sort(counts)[::-1]
    total_slots = int(np.sum(counts))
    top1_hub_share = float(sorted_counts[0] / total_slots) if total_slots > 0 else 0.0
    top5_hub_share = float(np.sum(sorted_counts[:5]) / total_slots) if total_slots > 0 else 0.0
    top10_hub_share = float(np.sum(sorted_counts[:10]) / total_slots) if total_slots > 0 else 0.0

    # Skewness of appearance counts
    mean_c = float(np.mean(counts))
    std_c = float(np.std(counts))
    if std_c > 1e-12:
        skewness = float(np.mean(((counts - mean_c) / std_c) ** 3))
    else:
        skewness = 0.0

    # Hub threshold: > 2 * expected uniform appearances (n_queries * top_k / gallery_size) or >= 5% of queries
    expected = (n_queries * top_k / gallery_size) if gallery_size > 0 else 0.0
    hub_threshold = max(5, int(np.ceil(expected * 3)))
    n_hubs = int(np.sum(counts >= hub_threshold))

    return {
        "n_queries": n_queries,
        "gallery_size": gallery_size,
        "top_k": top_k,
        "unique_items_retrieved": int(len(nonzero)),
        "coverage_ratio": float(len(nonzero) / gallery_size) if gallery_size > 0 else 0.0,
        "max_appearances": int(sorted_counts[0]) if len(sorted_counts) > 0 else 0,
        "max_query_pct": float(sorted_counts[0] / n_queries) if n_queries > 0 else 0.0,
        "top1_hub_slot_share": top1_hub_share,
        "top5_hub_slot_share": top5_hub_share,
        "top10_hub_slot_share": top10_hub_share,
        "mean_appearances": mean_c,
        "std_appearances": std_c,
        "skewness": skewness,
        "hub_threshold": hub_threshold,
        "n_hubs_above_threshold": n_hubs,
        "top5_counts": [int(x) for x in sorted_counts[:5]],
    }
