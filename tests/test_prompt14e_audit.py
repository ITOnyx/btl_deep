"""
Unit tests for Prompt 14e Step 1c (Caption limitation), Step 2 (GPU check & idle baseline),
Part A (4-stream latency benchmark), and Markdown <-> JSON auto-generation & verification.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_14e_paths():
    p_json = PROJECT_ROOT / "reports/phase5c/final_eval_results_prompt14e.json"
    p_md = PROJECT_ROOT / "reports/phase5c/final_eval_report_prompt14e.md"
    if not p_json.exists():
        p_json = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
        p_md = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"
    return p_json, p_md


def test_prompt14e_results_json_and_md_exist_and_hashes_match() -> None:
    json_path, md_path = get_14e_paths()
    assert json_path.exists(), "Missing 14e results json"
    assert md_path.exists(), "Missing 14e report md"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["status"] == "PART_A_DONE_AWAITING_HELDOUT_V2"
    assert data["branch"] == "phase7a"
    assert data["seed"] == 42
    assert data["heldout_v2_check"]["exists"] is False
    assert data["part_b_t2i_test_eval"] is None
    assert data["part_c_cir_test_eval"] is None


def test_prompt14e_md_vs_json_verification_perfect_match() -> None:
    json_path, md_path = get_14e_paths()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    md_text = md_path.read_text(encoding="utf-8")

    ver = data["md_vs_json_verification"]
    assert ver["all_matched"] is True
    assert ver["mismatch_count"] == 0
    assert ver["total_checked_items"] > 50

    # Explicit cross-check of key latency figures in Markdown text
    pa = data["part_a_latency"]
    for row in pa["comparison_table"]:
        expected_cell = f"**`{row['prompt14e_clean_p95_median_ms']:.3f}`**"
        assert expected_cell in md_text, f"Missing {expected_cell} in Markdown"


def test_prompt14e_step1c_caption_recount_and_manual_samples() -> None:
    json_path, _ = get_14e_paths()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    s1c = data["step_1c_caption_limitation"]

    # Check parity with 14d
    parity = s1c["item2_script_recount"]["parity_with_14d"]
    assert parity["val_exact_match"] is True
    assert parity["test_echo_exact_match"] is True
    assert parity["test_1_dot_exact_match"] is True
    assert parity["any_discrepancy_vs_14d"] is False

    # Check 40 samples
    samples = s1c["item3_manual_review_40_captions"]
    assert len(samples["val_samples_20"]) == 20
    assert len(samples["test_samples_20"]) == 20
    assert samples["val_describing_image_count"] == 0
    assert samples["test_describing_image_count"] == 0


def test_prompt14e_step2_gpu_and_way2_authorization_logged() -> None:
    json_path, _ = get_14e_paths()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    s2 = data["step_2_gpu"]

    way2 = s2["way2_authorized_stop_process_log"]
    assert len(way2["actions"]) == 2
    assert way2["actions"][0]["target_pid"] == 16284
    assert way2["actions"][0]["stopped_successfully"] is True
    assert way2["actions"][1]["target_pid"] == 3588
    assert way2["actions"][1]["stopped_successfully"] is True

    # No forbidden graphics processes remaining
    live_chk = s2["live_check_before_part_a"]
    assert live_chk["has_blocking_processes"] is False


def test_prompt14e_part_a_latency_results_and_post_part_a_cir_decision() -> None:
    json_path, _ = get_14e_paths()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pa = data["part_a_latency"]

    # Stream 1 T2I winner (cfg_v5_09) passes SLA (< 200 ms)
    t2i_09 = pa["stream1_text_to_image"]["cfg_v5_09"]
    assert t2i_09["sla_p95_under_200ms"] is True
    assert t2i_09["conclusion_label"] == "Đạt chắc chắn"
    assert t2i_09["p95_median_ms"] < 200.0

    # Stream 3 CIR catalog winner (cir_v1_05) passes SLA (< 200 ms)
    cir_05_cat = pa["stream3_cir_catalog"]["cir_v1_05"]
    assert cir_05_cat["sla_p95_under_200ms"] is True
    assert cir_05_cat["conclusion_label"] == "Đạt chắc chắn"
    assert cir_05_cat["p95_median_ms"] < 200.0

    # Stream 4 CIR upload exceeds 200 ms
    cir_05_up_ram = pa["stream4_cir_upload"]["cir_v1_05"]["primary_ram_preloaded"]
    assert cir_05_up_ram["sla_p95_under_200ms"] is False
    assert cir_05_up_ram["conclusion_label"] == "Chưa đạt"
    assert cir_05_up_ram["p95_median_ms"] > 200.0

    # Post-Part A CIR locking correctly flags user choice required
    cir_lock = pa["post_part_a_cir_locking"]
    assert cir_lock["both_cir_streams_under_200ms_on_clean_gpu"] is False
    assert cir_lock["requires_user_option_choice"] is True
    assert len(cir_lock["options_for_user"]) == 2
