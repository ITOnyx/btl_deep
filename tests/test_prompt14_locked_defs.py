"""Unit tests for locked Prompt 14 Step 0.6 definitions."""
from __future__ import annotations

from src.eval.locked_final_definitions import (
    classify_latency_conclusion,
    classify_quality_conclusion,
    compute_full_ranking_ap,
    compute_hit_rate_at_k,
    compute_secondary_within_group_metrics,
    compute_true_recall_at_k,
    paired_query_bootstrap_diff,
    query_bootstrap_ci,
)


def test_hit_rate_and_true_recall_and_ap() -> None:
    rels = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    assert compute_hit_rate_at_k(rels, 1) == 0.0
    assert compute_hit_rate_at_k(rels, 2) == 1.0
    assert abs(compute_true_recall_at_k(rels, 10, total_relevant=6) - 0.5) < 1e-9
    expected_ap = ((1.0 / 2.0) + (2.0 / 4.0) + (3.0 / 10.0)) / 6.0
    assert abs(compute_full_ranking_ap(rels, total_relevant=6) - expected_ap) < 1e-9


def test_secondary_within_group_metrics() -> None:
    ranked = [10, 11, 12, 13, 14]
    primary_gt = {11, 12, 14}
    secondary_gt = {12, 14}
    res = compute_secondary_within_group_metrics(ranked, primary_gt, secondary_gt)
    # Within primary_gt order is [11 (0), 12 (1), 14 (1)] -> AP = (1/2 + 2/3)/2 = 7/12
    assert abs(res["within_group_ap"] - (7.0 / 12.0)) < 1e-9
    assert res["within_group_r1"] == 0.0
    assert res["within_group_r5"] == 1.0


def test_conclusion_classifiers_and_bootstrap() -> None:
    assert classify_latency_conclusion([120.0, 150.0, 180.0]) == "Đạt chắc chắn"
    assert classify_latency_conclusion([150.0, 180.0, 210.0]) == "Đạt theo trung vị nhưng có run vượt"
    assert classify_latency_conclusion([205.0, 210.0, 220.0]) == "Chưa đạt"

    assert classify_quality_conclusion(0.88, 0.86, 0.90, 0.85) == "Đạt chắc chắn"
    assert classify_quality_conclusion(0.86, 0.83, 0.89, 0.85) == "Đạt điểm số nhưng CI chứa ngưỡng"
    assert classify_quality_conclusion(0.84, 0.81, 0.86, 0.85) == "Chưa đạt điểm số, CI chứa ngưỡng"
    assert classify_quality_conclusion(0.80, 0.76, 0.83, 0.85) == "Chưa đạt"

    ci = query_bootstrap_ci([0.5, 0.6, 0.7, 0.8], n_resamples=200, seed=42)
    assert 0.5 <= ci["ci_low"] <= ci["mean"] <= ci["ci_high"] <= 0.8
    diff = paired_query_bootstrap_diff([0.6, 0.7, 0.8], [0.5, 0.6, 0.7], n_resamples=200, seed=42)
    assert abs(diff["mean_diff"] - 0.1) < 1e-9
