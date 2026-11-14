"""
Kiểm thử cho Pha 5c Prompt 13 (CIR tuning trên cir_eval_val.json):
- Kiểm tra tính toàn vẹn của selection_rule_cir_v1.json (<= 8 cấu hình, khớp SHA-256).
- Kiểm tra CatalogModifierTranslator (C1: dịch thuộc tính từ catalog, giữ nguyên token lạ).
- Kiểm tra kết quả step5c_cir_results.json (khớp SHA-256, đủ chỉ số theo màu/giới, có/không dấu, độ trễ C.9).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.serve.cir_enhancer import CatalogModifierTranslator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_selection_rule_cir_v1_locked() -> None:
    rule_path = PROJECT_ROOT / "reports/phase5c/selection_rule_cir_v1.json"
    sha_path = PROJECT_ROOT / "reports/phase5c/selection_rule_cir_v1.sha256"
    assert rule_path.exists() and sha_path.exists()
    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    assert _sha256(rule_path) == expected_sha

    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    assert rule["seed"] == 42
    assert len(rule["config_grid"]) <= 8
    assert rule["test_dataset_policy"]["forbidden_in_prompt_13"] is True


def test_catalog_modifier_translator_preserves_unknown_tokens() -> None:
    tr = CatalogModifierTranslator(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    out1 = tr.translate("đổi sang màu đen")
    assert "black" in out1
    out2 = tr.translate("mẫu dành cho nữ họa_tiết_xyz")
    assert "women" in out2
    assert "họa_tiết_xyz" in out2


def test_step5c_cir_results_integrity() -> None:
    res_path = PROJECT_ROOT / "reports/phase5c/step5c_cir_results.json"
    sha_path = PROJECT_ROOT / "reports/phase5c/step5c_cir_results.sha256"
    assert res_path.exists() and sha_path.exists()
    expected_sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
    assert _sha256(res_path) == expected_sha

    res = json.loads(res_path.read_text(encoding="utf-8"))
    assert res["winner_config_id"] == "cir_v1_05"
    assert len(res["configs"]) == 8
    w = res["configs"]["cir_v1_05"]["primary_120"]
    assert w["overall"]["mAP"] > res["configs"]["cir_v1_01"]["primary_120"]["overall"]["mAP"]
    assert "baseColour" in w["by_modifier_type"] and "gender" in w["by_modifier_type"]
    assert "full" in w["by_dia_class"] and "none" in w["by_dia_class"]
    assert "cir_catalog" in res["latency_c9"] and "cir_upload" in res["latency_c9"]
