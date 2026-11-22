"""
Locked Metric, Bootstrap, Secondary GT, and Conclusion Rule Definitions for Prompt 14 (Step 0.6).
Seed: 42. DO NOT MODIFY after inspecting test results (AGENTS.md Law 7, C.5, C.10).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set
import numpy as np


SEED_LOCKED: int = 42
N_BOOTSTRAP_LOCKED: int = 1000
GALLERY_SIZE_TEST_LOCKED: int = 1290
SLA_LATENCY_P95_MS_LOCKED: float = 200.0
TARGET_RECALL_AT_10_LOCKED: float = 0.85
TARGET_T2I_MAP_LOCKED: float = 0.75


LOCKED_SPEC_METADATA: Dict[str, Any] = {
    "spec_version": "prompt14_step0_6_v1",
    "seed": SEED_LOCKED,
    "n_bootstrap": N_BOOTSTRAP_LOCKED,
    "bootstrap_unit": "query (paired when comparing two configurations)",
    "test_gallery_size": GALLERY_SIZE_TEST_LOCKED,
    "metrics": {
        "recall_at_k_hit_rate": "1.0 if sum(rels[:k]) >= 1 else 0.0 (standard hit-rate@k)",
        "true_recall_at_10": "sum(rels[:10]) / max(1, total_relevant_in_gallery) (supplementary)",
        "mAP": "Average Precision over full ranked list of all 1,290 items in test_gallery, divided by total_relevant_in_gallery",
        "secondary_gt_non_circular": (
            "Within the subset S_primary of gallery items matching the query's primary catalog attributes "
            "(articleType, gender under hybrid policy, baseColour), preserve the system's ranking order and "
            "compute within-group AP (secondary_within_group_mAP) and within-group Hit@5 (secondary_within_group_R@5) "
            "against items in S_primary matching the query's non-indexed usage/season attribute."
        ),
    },
    "conclusion_rules": {
        "latency_p95_lt_200ms": {
            "Đạt chắc chắn": "max(run_p95_ms) < 200.0",
            "Đạt theo trung vị nhưng có run vượt": "median(run_p95_ms) < 200.0 <= max(run_p95_ms)",
            "Chưa đạt": "median(run_p95_ms) >= 200.0",
        },
        "quality_threshold": {
            "Đạt chắc chắn": "ci_low >= threshold",
            "Đạt điểm số nhưng CI chứa ngưỡng": "ci_low < threshold <= point_estimate (or point_estimate >= threshold >= ci_low)",
            "Chưa đạt điểm số, CI chứa ngưỡng": "point_estimate < threshold <= ci_high",
            "Chưa đạt": "ci_high < threshold",
        },
    },
}


def compute_hit_rate_at_k(rels: Sequence[int], k: int) -> float:
    """Standard Hit-Rate Recall@K: 1.0 if at least one relevant item is in top-K, else 0.0."""
    if k <= 0 or not rels:
        return 0.0
    return 1.0 if sum(rels[:k]) > 0 else 0.0


def compute_true_recall_at_k(rels: Sequence[int], k: int, total_relevant: int) -> float:
    """Supplementary True-Recall@K: fraction of all relevant items retrieved in top-K."""
    if k <= 0 or total_relevant <= 0 or not rels:
        return 0.0
    return float(sum(rels[:k])) / float(total_relevant)


def compute_full_ranking_ap(rels: Sequence[int], total_relevant: int) -> float:
    """Average Precision over the full ranked list (all 1,290 gallery items)."""
    if total_relevant <= 0 or not rels:
        return 0.0
    hits = 0.0
    sum_precisions = 0.0
    for idx, rel in enumerate(rels, start=1):
        if rel:
            hits += 1.0
            sum_precisions += hits / float(idx)
    return float(sum_precisions / float(total_relevant))


def compute_secondary_within_group_metrics(
    ranked_indices: Sequence[int],
    primary_gt_set: Set[int],
    secondary_gt_set: Set[int],
) -> Dict[str, float]:
    """
    Non-circular secondary GT (usage/season) metric locked in Step 0.6:
    Restrict ranked_indices to items in primary_gt_set (preserving relative rank order),
    then evaluate AP and Hit@5 for items belonging to secondary_gt_set (primary & usage/season match).
    """
    target_set = primary_gt_set & secondary_gt_set
    if not primary_gt_set or not target_set:
        return {"within_group_ap": 0.0, "within_group_r1": 0.0, "within_group_r5": 0.0}
    sub_rels = [1 if idx in target_set else 0 for idx in ranked_indices if idx in primary_gt_set]
    return {
        "within_group_ap": compute_full_ranking_ap(sub_rels, total_relevant=len(target_set)),
        "within_group_r1": compute_hit_rate_at_k(sub_rels, 1),
        "within_group_r5": compute_hit_rate_at_k(sub_rels, 5),
    }


def query_bootstrap_ci(
    scores: Sequence[float],
    n_resamples: int = N_BOOTSTRAP_LOCKED,
    seed: int = SEED_LOCKED,
) -> Dict[str, float]:
    """Query-level percentile bootstrap (1,000 resamples, seed=42)."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(n),
    }


def paired_query_bootstrap_diff(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = N_BOOTSTRAP_LOCKED,
    seed: int = SEED_LOCKED,
) -> Dict[str, float]:
    """Paired query-level bootstrap CI for (A - B) (1,000 resamples, seed=42)."""
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    if len(arr_a) != len(arr_b) or len(arr_a) == 0:
        raise ValueError("scores_a and scores_b must have identical non-zero length.")
    diffs = arr_a - arr_b
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(n),
    }


def classify_latency_conclusion(
    run_p95s_ms: Sequence[float],
    sla_ms: float = SLA_LATENCY_P95_MS_LOCKED,
) -> str:
    """
    Locked Latency Conclusion Rule (Prompt 14 Part A):
    - 'Đạt chắc chắn': max(p95) < 200
    - 'Đạt theo trung vị nhưng có run vượt': median(p95) < 200 <= max(p95)
    - 'Chưa đạt': median(p95) >= 200
    """
    if not run_p95s_ms:
        return "Không xác minh được"
    arr = np.asarray(run_p95s_ms, dtype=np.float64)
    med = float(np.median(arr))
    mx = float(np.max(arr))
    if mx < sla_ms:
        return "Đạt chắc chắn"
    if med < sla_ms <= mx:
        return "Đạt theo trung vị nhưng có run vượt"
    return "Chưa đạt"


def classify_quality_conclusion(
    point_estimate: float,
    ci_low: float,
    ci_high: float,
    threshold: float,
) -> str:
    """
    Locked Quality Conclusion Rule (Prompt 14 Part B3):
    - 'Đạt chắc chắn': ci_low >= threshold
    - 'Đạt điểm số nhưng CI chứa ngưỡng': point_estimate >= threshold >= ci_low
    - 'Chưa đạt điểm số, CI chứa ngưỡng': point_estimate < threshold <= ci_high
    - 'Chưa đạt': ci_high < threshold
    """
    if ci_low >= threshold:
        return "Đạt chắc chắn"
    if point_estimate >= threshold >= ci_low:
        return "Đạt điểm số nhưng CI chứa ngưỡng"
    if point_estimate < threshold <= ci_high:
        return "Chưa đạt điểm số, CI chứa ngưỡng"
    return "Chưa đạt"
