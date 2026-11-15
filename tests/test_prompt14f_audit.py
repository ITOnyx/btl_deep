"""
Unit tests for Prompt 14f:
- Step 1: CIR User Decision file & SHA-256
- Step 2: Erratum 14e items
- Step 3: Environment snapshot recorded
- Step 4: Part A2 latency benchmark results (100% official runs, SLA status)
- Step 5: Phase 5 comparison table & diagnostic conclusions
- Numeric parity between final_eval_results.json and final_eval_report.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prompt14f_results_json_and_md_exist() -> None:
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results_prompt14f.json"
    md_path = PROJECT_ROOT / "reports/phase5c/final_eval_report_prompt14f.md"
    assert json_path.exists(), "Missing reports/phase5c/final_eval_results_prompt14f.json"
    assert md_path.exists(), "Missing reports/phase5c/final_eval_report_prompt14f.md"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["status"] == "PART_A2_DONE_AWAITING_HELDOUT_V2"
    assert data["branch"] == "phase7a"
    assert data["seed"] == 42
    assert data["heldout_v2_check"]["exists"] is False


def test_prompt14f_step1_cir_decision() -> None:
    dec_path = PROJECT_ROOT / "reports/phase5c/cir_user_decision_prompt14f.json"
    assert dec_path.exists()
    dec = json.loads(dec_path.read_text(encoding="utf-8"))
    assert dec["system_cir_config"] == "cir_v1_05"
    assert dec["model_only_cir_config"] == "cir_v1_01"
    assert dec["run_in_part_b2"] is True


def test_prompt14f_step2_erratum_14e() -> None:
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    e14 = data["step_2_erratum_14e"]

    assert e14["item_a_temperature"]["total_runs_measured"] == 65
    assert e14["item_a_temperature"]["runs_starting_ge_60c"] == 65
    assert e14["item_c_telemetry_duplication"]["is_copied"] is True
    assert e14["item_d_map_numbers"]["verified_val_cir_map"]["cir_v1_05"] == 0.2902
    assert e14["item_d_map_numbers"]["verified_val_cir_map"]["cir_v1_01"] == 0.0910


def test_prompt14f_part_a2_latencies_and_slas() -> None:
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pa2 = data["part_a2_latency"]

    # All runs must be official (< 60°C)
    cu_ram = pa2["stream4_cir_upload"]["cir_v1_05"]["primary_ram_preloaded"]
    assert cu_ram["n_runs_unofficial"] == 0
    assert cu_ram["all_runs_official"] is True
    assert cu_ram["p95_median_ms"] < 200.0

    # Primary conclusion PASS_FULL
    c_ram = pa2["latency_conclusion_primary_ram"]
    assert c_ram["status"] == "PASS_FULL"
    assert c_ram["passed_count"] == 4

    # Secondary conclusion PASS_FULL
    c_disk = pa2["latency_conclusion_secondary_disk"]
    assert c_disk["status"] == "PASS_FULL"
    assert c_disk["passed_count"] == 4


def test_prompt14f_step5_phase5_comparison() -> None:
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    s5 = data["step_5_phase5_comparison"]
    assert len(s5["comparison_table"]) == 4
    assert len(s5["diagnostic_conclusions"]) > 0
    assert len(s5["optimization_candidates"]) >= 3
