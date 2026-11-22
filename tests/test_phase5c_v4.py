"""
Unit & regression tests for Phase 5c (v4 inference-layer improvements I1-I7 & Hash Guard).
Runs fast without loading SigLIP weights onto GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.canonical_eval import evaluate_t2i_canonical
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import build_catalog_template, multi_template_encode_texts


class _MockEngine:
    def __init__(self, embeddings: np.ndarray, id_map: list[dict]):
        self.embeddings = embeddings.astype(np.float32)
        self.id_map = id_map

    def _apply_mean_centering(self, vecs: np.ndarray) -> np.ndarray:
        return vecs

    def encode_text_query(self, query: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(query)) % (2**32))
        v = rng.normal(size=(1, self.embeddings.shape[1])).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v


def _make_toy_gallery():
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(6, 16)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    id_map = [
        {"id": 1, "articleType": "Tshirts", "gender": "Men", "baseColour": "Blue"},
        {"id": 2, "articleType": "Tshirts", "gender": "Women", "baseColour": "Red"},
        {"id": 3, "articleType": "Casual Shoes", "gender": "Men", "baseColour": "Black"},
        {"id": 4, "articleType": "Kurtas", "gender": "Women", "baseColour": "Green"},
        {"id": 5, "articleType": "Watches", "gender": "Men", "baseColour": "Silver"},
        {"id": 6, "articleType": "Backpacks", "gender": "Unisex", "baseColour": "Black"},
    ]
    return _MockEngine(embs, id_map), embs, id_map


def test_i1_sigma_scaled_and_tier_soft():
    engine, embs, _ = _make_toy_gallery()
    q_vecs = embs[:1].copy()
    queries = [
        {
            "query": "ao thun nam xanh duong",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Blue"]},
            "relevant_ids": [1],
        }
    ]
    parsed = [{"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Blue"]}]

    res_sigma = evaluate_t2i_canonical(
        engine=engine,
        queries=queries,
        parsed_queries=parsed,
        query_vecs=q_vecs,
        gender_policy_cfg={"policies": {"hybrid": {"strict_categories": []}}},
        mode="system",
        gamma=0.3,
        zscore_cosine=False,
        bonus_mode="sigma_scaled",
    )
    res_tier = evaluate_t2i_canonical(
        engine=engine,
        queries=queries,
        parsed_queries=parsed,
        query_vecs=q_vecs,
        gender_policy_cfg={"policies": {"hybrid": {"strict_categories": []}}},
        mode="system",
        gamma=0.3,
        zscore_cosine=False,
        bonus_mode="tier_soft",
    )
    assert res_sigma["policies"]["hybrid"]["summary"]["recall@1"] == 1.0
    assert res_tier["policies"]["hybrid"]["summary"]["recall@1"] == 1.0


def test_i2_conf_gating_and_parser_v2_conf():
    parser = VietnameseQueryParserV2Conf()
    out = parser.parse_with_confidence("tim mua ao thun nam mau xanh", fuzzy_typo=False)
    assert "slot_confidence" in out
    assert out["slot_confidence"]["articleType"] in {"high", "low", "none"}

    engine, embs, _ = _make_toy_gallery()
    q_vecs = embs[:1].copy()
    queries = [
        {
            "query": "ao thun nam xanh",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Blue"]},
            "relevant_ids": [1],
        }
    ]
    parsed = [{"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Blue"]}]
    confs = [{"articleType": "high", "gender": "low", "baseColour": "high"}]
    res = evaluate_t2i_canonical(
        engine=engine,
        queries=queries,
        parsed_queries=parsed,
        query_vecs=q_vecs,
        gender_policy_cfg={"policies": {"hybrid": {"strict_categories": []}}},
        mode="system",
        gamma=0.3,
        zscore_cosine=True,
        slot_confidences=confs,
    )
    assert res["policies"]["hybrid"]["summary"]["recall@1"] == 1.0


def test_i3_multi_template_encode_texts():
    engine, _, _ = _make_toy_gallery()
    parsed = [{"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Blue"]}]
    tpl = build_catalog_template(parsed[0])
    assert "Tshirts" in tpl or "áo" in tpl or len(tpl) > 0

    vecs = multi_template_encode_texts(engine, ["ao thun nam xanh"], parsed, n_templates=2)
    assert vecs.shape == (1, 16)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_i4_rocchio_prf():
    engine, embs, _ = _make_toy_gallery()
    # Query vector slightly noisy around item 0
    q_vecs = (0.8 * embs[0:1] + 0.2 * embs[1:2]).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    queries = [
        {
            "query": "ao thun nam",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": []},
            "relevant_ids": [1],
        }
    ]
    parsed = [{"articleType": ["Tshirts"], "gender": "Men", "baseColour": []}]
    res = evaluate_t2i_canonical(
        engine=engine,
        queries=queries,
        parsed_queries=parsed,
        query_vecs=q_vecs,
        gender_policy_cfg={"policies": {"hybrid": {"strict_categories": []}}},
        mode="system",
        gamma=0.3,
        zscore_cosine=True,
        rocchio_k=2,
        rocchio_beta=0.2,
    )
    assert res["policies"]["hybrid"]["summary"]["recall@10"] == 1.0


def test_i6_extra_score_matrix():
    engine, embs, _ = _make_toy_gallery()
    q_vecs = embs[:1].copy()
    queries = [
        {
            "query": "ao thun nam",
            "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": []},
            "relevant_ids": [1],
        }
    ]
    parsed = [{"articleType": ["Tshirts"], "gender": "Men", "baseColour": []}]
    extra = np.zeros((1, 6), dtype=np.float32)
    extra[0, 0] = 0.5
    res = evaluate_t2i_canonical(
        engine=engine,
        queries=queries,
        parsed_queries=parsed,
        query_vecs=q_vecs,
        gender_policy_cfg={"policies": {"hybrid": {"strict_categories": []}}},
        mode="system",
        gamma=0.3,
        zscore_cosine=True,
        extra_score_matrix=extra,
        extra_score_weight=0.2,
    )
    assert res["policies"]["hybrid"]["summary"]["recall@1"] == 1.0


def test_i7_fuzzy_typo_guard_5v():
    parser = VietnameseQueryParserV2Conf()
    # Short token 'got' (len 3) must NOT fuzzy-match 'bot'
    res_short = parser.parse_with_confidence("giay got cao cho nu", fuzzy_typo=True)
    assert "Boots" not in res_short["articleType"]

    # Deterministic output
    res1 = parser.parse_with_confidence("ao thun nam mau xanh", fuzzy_typo=True)
    res2 = parser.parse_with_confidence("ao thun nam mau xanh", fuzzy_typo=True)
    assert res1 == res2


def test_val_text_index_manifest_hash_guard():
    manifest_path = Path("index/val_gallery_text_v4/manifest.json")
    if not manifest_path.exists():
        pytest.skip("index/val_gallery_text_v4/manifest.json not yet built")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["checkpoint_hash"] == "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d"
    assert data["n_vectors"] == 1884
    assert data["seed"] == 42
    assert data["internvl2_captions_present"] is False
