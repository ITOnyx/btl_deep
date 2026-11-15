"""
Unit tests for Prompt 14 Final Evaluation (FINAL_EVAL_COMPLETED):
- Status is FINAL_EVAL_COMPLETED
- T2I benchmarks on test_queries_heldout_v2.json (151 queries)
- CIR benchmarks on cir_eval_test.json (120 pairs)
- Target Recall@10 >= 0.85 and mAP >= 0.75 achieved by official winner cfg_v5_09
- Hubness analysis and paired differences recorded
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def final_results() -> dict:
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    assert json_path.exists(), "Missing reports/phase5c/final_eval_results.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


def test_final_eval_status(final_results: dict) -> None:
    assert final_results["status"] == "FINAL_EVAL_COMPLETED"
    assert final_results["branch"] == "phase7a"
    assert final_results["seed"] == 42


def test_t2i_benchmarks_and_targets(final_results: dict) -> None:
    t2i = final_results["part_b1_t2i_quality"]
    assert t2i["n_queries"] == 151
    assert t2i["n_gallery"] == 1290

    res_cfg = t2i["results_by_config"]
    assert "model_only" in res_cfg
    assert "cfg_v5_01" in res_cfg
    assert "cfg_v5_09" in res_cfg
    assert "cfg_v5_08b" in res_cfg
    assert "cfg_v5_06" in res_cfg

    # Model-only baseline
    raw_s = res_cfg["model_only"]["summary"]
    assert raw_s["recall@10"]["mean"] < 0.85
    assert raw_s["mAP"]["mean"] < 0.75

    # Official Winner cfg_v5_09: Target Recall@10 >= 0.85 and Target mAP >= 0.75
    win_s = res_cfg["cfg_v5_09"]["summary"]
    assert win_s["recall@10"]["mean"] >= 0.85
    assert win_s["recall@10"]["ci_low"] >= 0.85
    assert res_cfg["cfg_v5_09"]["conclusion_recall_at_10"] == "Đạt chắc chắn"

    assert win_s["mAP"]["mean"] >= 0.75
    assert win_s["mAP"]["ci_low"] >= 0.75
    assert res_cfg["cfg_v5_09"]["conclusion_map"] == "Đạt chắc chắn"


def test_cir_benchmarks(final_results: dict) -> None:
    cir = final_results["part_b2_cir_quality"]
    assert cir["n_pairs"] == 120
    assert cir["n_gallery"] == 1290

    res_cir = cir["results_by_config"]
    assert "cir_v1_01" in res_cir
    assert "cir_v1_05" in res_cir

    raw_cir = res_cir["cir_v1_01"]["primary_50_50"]["summary"]
    win_cir = res_cir["cir_v1_05"]["primary_50_50"]["summary"]

    # Official Winner outperforms baseline significantly
    assert win_cir["recall@10"]["mean"] > raw_cir["recall@10"]["mean"]
    assert win_cir["mAP"]["mean"] > raw_cir["mAP"]["mean"]

    paired = cir["paired_differences"]["cir_v1_05_vs_cir_v1_01"]
    assert paired["delta_map"]["mean_diff"] > 0.15
    assert paired["delta_map"]["ci_low"] > 0.0
    assert paired["delta_r10"]["mean_diff"] > 0.20
    assert paired["delta_r10"]["ci_low"] > 0.0


def test_hubness_and_goal_assessment(final_results: dict) -> None:
    hub = final_results["part_c_hubness_analysis"]
    assert "model_only_without_mean_centering" in hub
    assert "official_winner_cfg_v5_09_without_mean_centering" in hub
    assert "model_only_with_mean_centering" in hub

    goals = final_results["part_c_goal_assessment"]
    assert goals["target_recall_at_10"]["system_winner_conclusion"] == "Đạt chắc chắn"
    assert goals["target_text_to_image_map"]["system_winner_conclusion"] == "Đạt chắc chắn"
    assert "PASS_FULL" in goals["target_latency_p95_under_200ms"]["status"]
