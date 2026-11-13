"""Unit tests cho Bước 11.2 (check_query_set, worksheets, CIR-eval)."""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_query_set import build_v2_lexicon_tokens, check_query_set, make_cluster_id


def test_check_query_set_preset_validation(tmp_path: pathlib.Path):
    gallery = [
        {"item_id": str(i), "articleType": "Tshirts", "gender": "Men", "baseColour": "Black"}
        for i in range(6)
    ]
    queries = [
        {
            "id": "q1",
            "query": "áo thun nam màu đen đẹp tuyệt vời",
            "dia_class": "full",
            "source": "human_written",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]},
        },
        {
            "id": "q2",
            "query": "ao thun nam mau den di choi",
            "dia_class": "none",
            "source": "human_written",
            "typo_type": "missing_char",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]},
        },
    ]
    q_path = tmp_path / "sample_q.json"
    q_path.write_text(json.dumps(queries, ensure_ascii=False), encoding="utf-8")

    rep = check_query_set(
        query_path=q_path,
        gallery_id_map=gallery,
        strict_categories=set(),
        ref_datasets={},
        min_queries=2,
        min_relevant=5,
        preset="b_human_v4",
    )
    assert rep["n_queries"] == 2
    assert rep["warnings_low_relevant_lt5"] == 0
    assert rep["warnings_missing_source"] == 0
    assert rep["dia_class_counts"] == {"full": 1, "none": 1}
    assert rep["n_typo"] == 1
    assert rep["n_outside_v2_lexicon"] == 2
    # Preset b_human_v4 requires >= 60 queries, so passed must be False here
    assert rep["preset_checks"]["passed"] is False


def test_v2_lexicon_tokens_nonempty():
    toks = build_v2_lexicon_tokens()
    assert "ao" in toks
    assert "thun" in toks
    assert "den" in toks


def test_b_human_v4_worksheet_excludes_old_b_human():
    csv_path = PROJECT_ROOT / "reports/phase5c/b_human_v4_worksheet.csv"
    assert csv_path.exists()
    old_bh = json.loads((PROJECT_ROOT / "eval/val_queries_b_human.json").read_text(encoding="utf-8"))
    old_cids = {make_cluster_id(x["spec"]) for x in old_bh}
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    ws_cids = {r["spec_cluster_id"] for r in rows}
    assert len(ws_cids & old_cids) == 0


def test_heldout_v2_worksheet_has_usage_season_and_thin():
    csv_path = PROJECT_ROOT / "reports/phase5c/heldout_v2_worksheet.csv"
    assert csv_path.exists()
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert any(r["has_usage_season_5_70pct"] == "True" for r in rows)
    assert any(r["is_thin_slot"] == "True" for r in rows)


def test_cir_eval_integrity():
    val_p = PROJECT_ROOT / "eval/cir_eval_val.json"
    test_p = PROJECT_ROOT / "eval/cir_eval_test.json"
    assert val_p.exists() and test_p.exists()

    for p in (val_p, test_p):
        sha_p = p.with_suffix(".sha256")
        assert sha_p.exists()
        expected_sha = sha_p.read_text(encoding="utf-8").split()[0]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected_sha

        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["seed"] == 42
        assert data["source"] == "catalog_attributes"
        assert data["n_pairs"] >= 100
        assert len(data["pairs"]) == data["n_pairs"]

        for pair in data["pairs"]:
            assert pair["ref_product_id"] not in pair["gt_product_ids"]
            assert pair["n_gt"] >= 3
            assert pair["changed_attr"] in ("baseColour", "gender")
            assert pair["ref_attrs"]["articleType"] == pair["target_attrs"]["articleType"]
            if pair["changed_attr"] == "baseColour":
                assert pair["ref_attrs"]["gender"] == pair["target_attrs"]["gender"]
                assert pair["ref_attrs"]["baseColour"] != pair["target_attrs"]["baseColour"]
            else:
                assert pair["ref_attrs"]["baseColour"] == pair["target_attrs"]["baseColour"]
                assert pair["ref_attrs"]["gender"] != pair["target_attrs"]["gender"]

    val_data = json.loads(val_p.read_text(encoding="utf-8"))
    test_data = json.loads(test_p.read_text(encoding="utf-8"))
    val_pids = {x["ref_product_id"] for x in val_data["pairs"]} | {
        pid for x in val_data["pairs"] for pid in x["gt_product_ids"]
    }
    test_pids = {x["ref_product_id"] for x in test_data["pairs"]} | {
        pid for x in test_data["pairs"] for pid in x["gt_product_ids"]
    }
    assert len(val_pids & test_pids) == 0
