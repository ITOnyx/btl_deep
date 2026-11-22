"""Unit tests for Prompt 14d audit, nvidia-smi process parser, and Markdown <-> JSON verification."""
from __future__ import annotations

import json

import pytest
from pathlib import Path

from scripts.run_prompt14d_audit_and_report import (
    parse_nvidia_smi_processes,
    render_markdown_from_json,
    verify_markdown_matches_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_nvidia_smi_processes():
    sample_raw = """Sat Nov 14 22:05:42 2026       
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1560    C+G   C:\\Windows\\System32\\ShellHost.exe           N/A      |
|    0   N/A  N/A      3588    C+G   ...4__8wekyb3d8bbwe\\EdgeGameAssist.exe      N/A      |
|    0   N/A  N/A     16284    C+G   ...er\\ldmutiplayer\\dnmultiplayerex.exe      N/A      |
+-----------------------------------------------------------------------------------------+
"""
    procs = parse_nvidia_smi_processes(sample_raw)
    assert len(procs) == 3
    assert [p["pid"] for p in procs] == [1560, 3588, 16284]
    assert procs[1]["type"] == "C+G"


def test_final_eval_report_matches_json_exactly():
    # Test the preserved prompt 14d halted report
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results_halted_prompt14d.json"
    md_path = PROJECT_ROOT / "reports/phase5c/final_eval_report_halted_prompt14d.md"
    if not json_path.exists():
        json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
        md_path = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("prompt") == "Prompt 14e":
        pytest.skip("Active report is Prompt 14e; Prompt 14d test runs on preserved halted file.")
    md_text = md_path.read_text(encoding="utf-8")

    # 1. Verify rendered markdown equals file on disk
    rerendered = render_markdown_from_json(data)
    assert md_text == rerendered

    # 2. Verify all checked items match between Markdown and JSON
    ver = verify_markdown_matches_json(md_text, data)
    assert ver["status"] == "PASS_EXACT_MATCH"
    assert ver["mismatched_items"] == 0
    assert ver["matched_items"] == ver["total_checked_items"]
    assert ver["total_checked_items"] >= 50
