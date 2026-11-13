"""Regression tests for Phase 5b Step 3 Preflight (P3)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.canonical_eval import evaluate_t2i_canonical
from src.serve.parser_v1 import VietnameseQueryParser
from src.serve.parser_v2 import VietnameseQueryParserV2
from src.serve.query_rewriter import (
    BM25GalleryScorer,
    dual_encode_texts,
    restore_diacritics_for_query,
)


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_mock_engine(n_items: int = 8, dim: int = 16, mean_center: bool = False):
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(n_items, dim)).astype(np.float32)
    # Add a common bias vector so mean_centering visibly changes rankings/cosines
    vecs += np.array([2.0] + [0.0] * (dim - 1), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    if mean_center:
        vecs = vecs - vecs.mean(axis=0, keepdims=True)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    catalog = [
        {
            "product_id": f"p_{i}",
            "articleType": "Tshirts" if i % 2 == 0 else "Shirts",
            "gender": "Men" if i < 4 else "Women",
            "baseColour": "Black" if i % 3 == 0 else "White",
            "caption_vi": f"áo thun nam màu đen mã {i}" if i % 2 == 0 else f"áo sơ mi nữ màu trắng mã {i}",
        }
        for i in range(n_items)
    ]

    class _MockIndex:
        def __init__(self, v):
            self.vectors = v

        def search(self, qv, top_k=10):
            sims = (self.vectors @ qv).astype(np.float32)
            order = np.argsort(-sims)[:top_k]
            return [(float(sims[idx]), int(idx)) for idx in order]

    return SimpleNamespace(
        hnsw=_MockIndex(vecs),
        id_map=catalog,
        catalog=catalog,
        embeddings=vecs,
        use_mean_centering=mean_center,
        _apply_mean_centering=lambda v: v,
        encode_text_query=lambda t: vecs[0:1],
    )


def test_p3_frozen_hashes_and_hash_guard():
    assert _sha256("src/serve/parser_v1.py").startswith("111002f6")
    assert _sha256("reports/phase5b/selection_rule_v3.json").startswith("fa4e0df3")
    assert _sha256("reports/phase5b/selection_rule_v3.md").startswith("d9e7b56d")
    bh_sha = Path("eval/val_queries_b_human.sha256").read_text(encoding="utf-8").strip().split()[0]
    assert _sha256("eval/val_queries_b_human.validated.json") == bh_sha

    from src.model.siglip_model import verify_checkpoint_hash

    with pytest.raises(Exception):
        verify_checkpoint_hash("best_model.pt", "0" * 64)


def test_p3_flags_zscore_slot_weights_bm25_mean_centering_dual_encode_rewrite():
    eng = _make_mock_engine()
    eng_mc = _make_mock_engine(mean_center=True)
    assert not np.allclose(eng.embeddings, eng_mc.embeddings)

    queries = [
        {"id": "q1", "query": "áo thun nam màu đen", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}},
        {"id": "q2", "query": "áo sơ mi nữ màu trắng", "spec": {"articleType": ["Shirts"], "gender": "Women", "baseColour": ["White"]}},
    ]
    p2 = VietnameseQueryParserV2()
    parsed = [p2.parse(q["query"]) for q in queries]
    q_vecs = np.stack([eng.embeddings[1], eng.embeddings[0]], axis=0)
    gpol = {"policies": {"hybrid": {"strict_categories": []}}}

    # zscore toggle
    r_z0 = evaluate_t2i_canonical(eng, queries, q_vecs, parsed, gpol, gamma=0.5, zscore_cosine=False, candidate_pool=8)
    r_z1 = evaluate_t2i_canonical(eng, queries, q_vecs, parsed, gpol, gamma=0.5, zscore_cosine=True, candidate_pool=8)
    assert "summary" in r_z0["policies"]["hybrid"] and "summary" in r_z1["policies"]["hybrid"]

    # slot_weights uniform vs {0.6, 0.2, 0.2}
    r_sw_u = evaluate_t2i_canonical(eng, queries, q_vecs, parsed, gpol, gamma=0.8, slot_weights=None, candidate_pool=8)
    r_sw_w = evaluate_t2i_canonical(
        eng,
        queries,
        q_vecs,
        parsed,
        gpol,
        gamma=0.8,
        slot_weights={"articleType": 0.6, "gender": 0.2, "baseColour": 0.2},
        candidate_pool=8,
    )
    assert "summary" in r_sw_u["policies"]["hybrid"] and "summary" in r_sw_w["policies"]["hybrid"]

    # bm25 flag
    bm25 = BM25GalleryScorer(eng.catalog)
    bm25_mat = np.vstack([bm25.score_query(q["query"]) for q in queries])
    assert bm25_mat.shape == (2, 8)
    r_bm0 = evaluate_t2i_canonical(eng, queries, q_vecs, parsed, gpol, gamma=0.5, bm25_scores=None, candidate_pool=8)
    r_bm1 = evaluate_t2i_canonical(eng, queries, q_vecs, parsed, gpol, gamma=0.5, bm25_scores=bm25_mat, bm25_weight=0.5, candidate_pool=8)
    assert "summary" in r_bm0["policies"]["hybrid"] and "summary" in r_bm1["policies"]["hybrid"]

    # query_rewrite
    rewritten = restore_diacritics_for_query("ao thun nam mau den")
    assert rewritten != "ao thun nam mau den"

    # dual_encode
    class _DualMock:
        def encode_text_query(self, t):
            v = np.ones((1, 16), dtype=np.float32) * (1.0 if "áo" in t else -1.0)
            v /= np.linalg.norm(v)
            return v

    single = _DualMock().encode_text_query("ao thun nam")
    dual = dual_encode_texts(_DualMock(), ["ao thun nam"])
    assert not np.allclose(single, dual)


def test_p3_all_22_configs_valid_and_runnable():
    rule = json.loads(Path("reports/phase5b/selection_rule_v3.json").read_text(encoding="utf-8"))
    grid = rule["config_grid"]
    assert len(grid) == 22
    assert [c["config_id"] for c in grid] == [f"cfg_{i:02d}" for i in range(1, 23)]

    eng = _make_mock_engine()
    queries = [{"id": "q1", "query": "áo thun nam màu đen", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}}]
    p1 = VietnameseQueryParser()
    p2 = VietnameseQueryParserV2()
    bm25 = BM25GalleryScorer(eng.catalog)
    gpol = {"policies": {"hybrid": {"strict_categories": []}}}

    for cfg in grid:
        parser = p1 if cfg["parser"] == "parser_v1" else p2
        q_text = restore_diacritics_for_query(queries[0]["query"]) if cfg["query_rewrite"] else queries[0]["query"]
        parsed = [parser.parse(q_text)]
        q_vecs = dual_encode_texts(eng, [q_text]) if cfg["dual_encode"] else eng.encode_text_query(q_text)
        sw = None if cfg["slot_weights"] == "uniform" else cfg["slot_weights"]
        bm25_mat = np.vstack([bm25.score_query(q_text)]) if cfg["bm25"] else None
        res = evaluate_t2i_canonical(
            eng,
            queries,
            q_vecs,
            parsed,
            gpol,
            mode="system",
            candidate_pool=8,
            eval_top_k=8,
            gamma=float(cfg["gamma"]),
            slot_weights=sw,
            force_exact=True,
            tie_rule="conservative",
            zscore_cosine=bool(cfg["zscore_cosine"]),
            bm25_scores=bm25_mat,
        )
        assert "summary" in res["policies"]["hybrid"]


def test_p3_provenance_file_exists_and_complete():
    prov_path = Path("reports/phase5b/parser_v2_lexicon_provenance.csv")
    assert prov_path.exists()
    with prov_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 398
