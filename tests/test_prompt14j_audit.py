"""
Tests for Prompt 14j Audit & Verification
- Standard convention CIR results verification
- Subjective adjectives prohibition audit
- Absence of ungrounded comparative claims
- Side-by-side CIR protocol comparison table integrity
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports/phase5c"


def test_prompt14j_standard_cir_results_integrity():
    std_path = REPORTS_DIR / "cir_standard_convention_results.json"
    assert std_path.exists(), "Missing cir_standard_convention_results.json"
    data = json.loads(std_path.read_text(encoding="utf-8"))

    assert data["status"] == "CIR_STANDARD_CONVENTION_EVALUATION_COMPLETED"
    assert data["n_gallery_after_filter"] == 1289
    assert data["seed"] == 42
    assert data["n_bootstrap"] == 1000

    # Check baseline under standard convention
    b_std = data["summary_by_subset"]["overall"]["cir_i2i_baseline"]
    assert b_std["recall@1"]["mean"] == pytest.approx(0.1083, abs=1e-4)
    assert b_std["recall@10"]["mean"] == pytest.approx(0.6000, abs=1e-4)
    assert b_std["mAP"]["mean"] == pytest.approx(0.1877, abs=1e-4)

    # Check paired differences overall
    pairs = data["paired_differences_by_subset"]["overall"]
    assert pairs["cir_v1_01_vs_baseline"]["recall@10"]["mean_diff"] == pytest.approx(-0.3083, abs=1e-4)
    assert pairs["cir_v1_05_vs_baseline"]["recall@10"]["mean_diff"] == pytest.approx(0.0000, abs=1e-4)
    assert pairs["cir_v1_05_vs_baseline"]["recall@10"]["label"] == "Chưa kết luận được (CI chứa hoặc chạm 0)"
    assert pairs["cir_v1_05_vs_cir_v1_01"]["recall@10"]["label"] == "Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01)"


def test_prompt14j_supplements_step_3e_present():
    supp_path = REPORTS_DIR / "final_eval_supplements.json"
    assert supp_path.exists()
    supp = json.loads(supp_path.read_text(encoding="utf-8"))
    assert "step_3e_cir_standard_convention" in supp
    s3e = supp["step_3e_cir_standard_convention"]
    assert s3e["n_gallery"] == 1289
    assert s3e["summary_by_subset"]["overall"]["cir_i2i_baseline"]["recall@10"]["mean"] == pytest.approx(0.6000, abs=1e-4)


def test_prompt14j_corrections_version_v3():
    corr_path = REPORTS_DIR / "final_eval_corrections.json"
    assert corr_path.exists()
    corr = json.loads(corr_path.read_text(encoding="utf-8"))
    assert corr["status"] == "FINAL_EVAL_CORRECTIONS_V1"
    assert corr["version"] == "v3_prompt14j"
    assert "cir_standard_convention_results" in corr
    assert "subjective_adjectives_audit" in corr
    assert corr["subjective_adjectives_audit"]["count_total"] == 0


def test_prompt14j_zero_subjective_adjectives():
    report_text = (REPORTS_DIR / "final_eval_report.md").read_text(encoding="utf-8")
    chat_text = (REPORTS_DIR / "final_summary_for_chat.md").read_text(encoding="utf-8")

    forbidden = ["khổng lồ", "vượt bậc", "vượt trội", "xuất sắc", "hoàn hảo"]
    for w in forbidden:
        assert not re.search(r'\b' + re.escape(w) + r'\b', report_text, re.IGNORECASE), f"Found '{w}' in final_eval_report.md"
        assert not re.search(r'\b' + re.escape(w) + r'\b', chat_text, re.IGNORECASE), f"Found '{w}' in final_summary_for_chat.md"


def test_prompt14j_zero_uninterpolated_placeholders():
    report_text = (REPORTS_DIR / "final_eval_report.md").read_text(encoding="utf-8")
    chat_text = (REPORTS_DIR / "final_summary_for_chat.md").read_text(encoding="utf-8")

    ph_rep = re.findall(r'\[(?:ĐIỀN|TODO|TBD|CHỌN)[^\]]*\]|\{[a-zA-Z_][a-zA-Z0-9_]*\}', report_text)
    ph_chat = re.findall(r'\[(?:ĐIỀN|TODO|TBD|CHỌN)[^\]]*\]|\{[a-zA-Z_][a-zA-Z0-9_]*\}', chat_text)
    assert len(ph_rep) == 0, f"Found placeholders in report: {ph_rep}"
    assert len(ph_chat) == 0, f"Found placeholders in chat: {ph_chat}"
