"""Pytest kiểm định các thành phần chuẩn hóa Bước 2b (Pha 5b)."""
import json
import pathlib
import numpy as np

from src.eval.canonical_eval import apply_conservative_tie_break
from src.eval.error_injection import inject_parse_errors

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_conservative_tie_rule_epsilon():
    """Tie-rule conservative (epsilon=1e-5): ảnh rel=0 phải xếp trước rel=1 khi điểm chênh <= 1e-5."""
    # idx 1 là relevant (score 0.900000), idx 2 là non-relevant (score 0.899995, chênh 5e-6 <= 1e-5)
    scored = [(0.900000, 1), (0.899995, 2), (0.850000, 3)]
    rel_map = {1: True, 2: False, 3: True}
    ordered = apply_conservative_tie_break(scored, lambda idx: rel_map[idx], epsilon=1e-5)
    assert ordered == [2, 1, 3]

    # Nếu chênh > 1e-5 (vd 2e-5) thì giữ nguyên thứ tự
    scored2 = [(0.900000, 1), (0.899980, 2)]
    ordered2 = apply_conservative_tie_break(scored2, lambda idx: rel_map[idx], epsilon=1e-5)
    assert ordered2 == [1, 2]


def test_error_injection_reproducible_seed():
    """Hàm tiêm lỗi Tập C tái lập 100% theo seed=42."""
    sample = [
        {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]},
        {"articleType": ["Casual Shoes"], "gender": "Women", "baseColour": ["White"]},
        {"articleType": ["Handbags"], "gender": "Women", "baseColour": ["Brown"]},
    ] * 10
    out1 = inject_parse_errors(sample, error_rate=0.2, seed=42)
    out2 = inject_parse_errors(sample, error_rate=0.2, seed=42)
    assert out1 == out2
    assert out1 != sample  # phải có lỗi được tiêm

    # Mức 0.0 không tiêm lỗi
    out0 = inject_parse_errors(sample, error_rate=0.0, seed=42)
    assert out0 == sample


def test_val_queries_integrity_and_clusters():
    """Kiểm tra Tập A, B-synth, C mới đều có spec_cluster_id và n_relevant >= 5."""
    val_a = json.loads((PROJECT_ROOT / "eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    val_b = json.loads((PROJECT_ROOT / "eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    val_c = json.loads((PROJECT_ROOT / "eval/val_queries_c_error_injection.json").read_text(encoding="utf-8"))

    assert len(val_a) >= 200
    assert len(val_b) >= 190  # 165 + 45 typos = 210
    assert all(x["n_relevant"] >= 5 and "spec_cluster_id" in x for x in val_a)
    assert all(x["n_relevant"] >= 5 and "spec_cluster_id" in x and "dia_class" in x for x in val_b)
    assert len(val_c["queries"]) >= 100
    assert "20%" in val_c["queries"][0]["parsed_by_rate"]
    assert "parsed_real_v1" in val_c["queries"][0]
