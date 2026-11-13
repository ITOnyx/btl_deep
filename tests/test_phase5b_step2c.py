"""Unit tests cho Bước 2c (Pha 5b)."""
import json
import pathlib
import sys
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_b_human import check_b_human_queries, tokenize_norm
from src.eval.canonical_eval import evaluate_t2i_canonical
from src.eval.error_injection import inject_nested_parse_errors


class _DummyEngine:
    def __init__(self, embeddings, id_map):
        self.embeddings = embeddings
        self.id_map = id_map

    def _apply_mean_centering(self, vecs):
        return vecs


def test_tie_rule_rank1_conservative_r1_zero():
    """
    2c.2: Cặp hòa (chênh < epsilon) giữa ảnh đúng và ảnh sai ở rank 1
    => R@1 phải = 0 cho truy vấn đó (xử lý bảo thủ).
    """
    # Tạo 3 item: item 0 là relevant, item 1 là non-relevant, item 2 là non-relevant
    # q_vec = [1, 0]. item 0 có cos = 0.800004, item 1 có cos = 0.800000 (chênh 4e-6 < 1e-5)
    emb = np.array(
        [
            [0.800004, 0.599995],  # item 0 (relevant)
            [0.800000, 0.600000],  # item 1 (non-relevant, tie trong epsilon=1e-5)
            [0.500000, 0.866025],  # item 2
        ],
        dtype=np.float32,
    )
    id_map = [
        {"id": 100, "articleType": "Tshirts", "gender": "Men", "baseColour": "Black"},
        {"id": 101, "articleType": "Jeans", "gender": "Men", "baseColour": "Blue"},
        {"id": 102, "articleType": "Watches", "gender": "Men", "baseColour": "Silver"},
    ]
    engine = _DummyEngine(emb, id_map)
    queries = [
        {
            "id": "q_tie",
            "query": "ao thun nam den",
            "has_diacritics": False,
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]},
        }
    ]
    q_vecs = np.array([[1.0, 0.0]], dtype=np.float32)
    parsed = [{"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}]
    gp = {"policies": {"hybrid": {"strict_categories": ["Tshirts"]}}}

    res = evaluate_t2i_canonical(
        engine=engine,
        queries=queries,
        query_vecs=q_vecs,
        parsed_queries=parsed,
        gender_policy_cfg=gp,
        mode="model_only",
        candidate_pool=3,
        eval_top_k=3,
        force_exact=True,
        tie_rule="conservative",
        tie_epsilon=1e-5,
        policies=("hybrid",),
    )
    assert res["policies"]["hybrid"]["summary"]["recall@1"] == 0.0
    assert res["policies"]["hybrid"]["summary"]["recall@5"] == 1.0


def test_nested_error_injection_2c3():
    """2c.3: Lỗi lồng nhau thực sự và chỉ tiêm lỗi có hiệu lực."""
    clean = [
        {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]},
        {"articleType": ["Watches"], "gender": None, "baseColour": []},
        {"articleType": ["Caps"], "gender": "Unisex", "baseColour": ["White"]},
    ] * 20

    res = inject_nested_parse_errors(clean, error_rates=(0.0, 0.1, 0.2, 0.3), seed=42)
    lv = res["levels"]
    for i in range(len(clean)):
        f0 = lv[0.0]["error_flags"][i]
        f1 = lv[0.1]["error_flags"][i]
        f2 = lv[0.2]["error_flags"][i]
        f3 = lv[0.3]["error_flags"][i]
        assert not f0
        assert (not f1) or f2
        assert (not f2) or f3
        # Không flip gender từ None hay Unisex
        if clean[i]["gender"] in (None, "Unisex"):
            assert lv[0.3]["error_types"][i] != "flip_gender"
        # Nếu cờ lỗi bật thì parsed phải thực sự khác clean
        if f3:
            assert lv[0.3]["parsed"][i] != clean[i]


def test_check_b_human_tool_2c5(tmp_path):
    """2c.5: Công cụ check_b_human phát hiện trùng lặp, OOV và n_relevant < 5."""
    sample = [
        {
            "id": "bh_1",
            "query": "ao thun nam mau den sieu nhe",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]},
        }
    ]
    p = tmp_path / "b_human.json"
    p.write_text(json.dumps(sample), encoding="utf-8")

    val_id_map = [
        {"id": 1, "articleType": "Tshirts", "gender": "Men", "baseColour": "Black"},
        {"id": 2, "articleType": "Tshirts", "gender": "Men", "baseColour": "Black"},
    ]
    refs = {"DEV": [("h001", tokenize_norm("ao thun nam mau den"))]}
    rep = check_b_human_queries(p, val_id_map, {"Tshirts"}, refs, min_queries=60, min_relevant=5)
    assert rep["meets_min_queries"] is False
    assert rep["warnings_low_relevant_lt5"] == 1
    assert "sieu" in rep["rows"][0]["tokens_outside_v1_lexicon"]


def test_val_ab_c_and_rule_v3_integrity():
    """2c.4 & 2c.6: Tập A không dùng 'áo dài' cho Sarees, B-synth wrong_tone != full, rule v3 <= 24."""
    val_a = json.loads((PROJECT_ROOT / "eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    for x in val_a:
        if "Sarees" in x["spec"]["articleType"]:
            assert "áo dài" not in x["query"].lower()

    val_b = json.loads((PROJECT_ROOT / "eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    typos = [x for x in val_b if x.get("is_typo")]
    assert len(typos) >= 75
    for x in typos:
        if x.get("typo_type") == "wrong_tone":
            assert x["dia_class"] != "full"

    rule_v3 = json.loads((PROJECT_ROOT / "reports/phase5b/selection_rule_v3.json").read_text(encoding="utf-8"))
    assert len(rule_v3["config_grid"]) <= 24
    assert any(c["config_id"] == "cfg_02" and c["parser"] == "parser_v2" for c in rule_v3["config_grid"])
