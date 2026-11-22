"""Unit tests for Phase 5 serving, query parsing, metrics, hubness, and Hash Guard."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.metrics import (
    bootstrap_ci_diff,
    compute_average_precision,
    compute_hubness_stats,
    compute_recall_at_k,
    compute_retrieval_summary,
    is_gender_match,
    is_item_relevant,
    load_gender_gt_policy,
)
from src.serve.parser import (
    VietnameseQueryParser,
    compute_soft_attribute_match,
    remove_diacritics,
)
from src.serve.search_engine import ECommerceSearchEngine


def test_remove_diacritics():
    assert remove_diacritics("Áo thun nam trắng") == "ao thun nam trang"
    assert remove_diacritics("Đầm nữ đen") == "dam nu den"


def test_parser_accented_and_unaccented():
    parser = VietnameseQueryParser()

    p1 = parser.parse("áo thun nam trắng")
    assert "Tshirts" in p1["articleType"]
    assert p1["gender"] == "Men"
    assert "White" in p1["baseColour"]

    p2 = parser.parse("ao so mi nu xanh navy")
    assert "Shirts" in p2["articleType"]
    assert p2["gender"] == "Women"
    assert "Navy Blue" in p2["baseColour"]

    p3 = parser.parse("giày thể thao đen")
    assert "Sports Shoes" in p3["articleType"]
    assert p3["gender"] is None
    assert "Black" in p3["baseColour"]


def test_soft_attribute_match_policies():
    parser = VietnameseQueryParser()
    parsed = parser.parse("áo thun nam trắng")
    item_unisex = {"articleType": "Tshirts", "gender": "Unisex", "baseColour": "White"}
    item_men = {"articleType": "Tshirts", "gender": "Men", "baseColour": "White"}

    assert compute_soft_attribute_match(parsed, item_men, gender_policy="hybrid") == pytest.approx(1.0)
    # Tshirts is in default strict_categories ("Tshirts" -> Unisex does not match Men under hybrid/strict)
    assert compute_soft_attribute_match(parsed, item_unisex, gender_policy="hybrid") < 1.0
    assert compute_soft_attribute_match(parsed, item_unisex, gender_policy="relaxed") == pytest.approx(1.0)


def test_gender_gt_policy_and_relevance():
    policy = load_gender_gt_policy("eval/gender_gt_policy.json")
    strict_cats = policy["policies"]["hybrid"]["strict_categories"]

    assert is_gender_match("Men", "Unisex", "Backpacks", policy="hybrid", strict_categories=strict_cats) is True
    assert is_gender_match("Men", "Unisex", "Tshirts", policy="hybrid", strict_categories=strict_cats) is False
    assert is_gender_match("Men", "Unisex", "Tshirts", policy="relaxed", strict_categories=strict_cats) is True
    assert is_gender_match("Men", "Unisex", "Backpacks", policy="strict", strict_categories=strict_cats) is False

    spec = {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["White"]}
    assert is_item_relevant(spec, {"articleType": "Tshirts", "gender": "Men", "baseColour": "White"}, "hybrid", strict_cats) is True
    assert is_item_relevant(spec, {"articleType": "Tshirts", "gender": "Unisex", "baseColour": "White"}, "hybrid", strict_cats) is False
    assert is_item_relevant(spec, {"articleType": "Tshirts", "gender": "Unisex", "baseColour": "White"}, "relaxed", strict_cats) is True


def test_retrieval_metrics_and_bootstrap():
    rels = [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
    assert compute_recall_at_k(rels[0], 1) == 1.0
    assert compute_recall_at_k(rels[1], 1) == 0.0
    assert compute_recall_at_k(rels[1], 5) == 1.0
    assert compute_average_precision(rels[1], total_relevant=1) == pytest.approx(1.0 / 3.0)

    summary = compute_retrieval_summary(rels, total_relevants=[1, 1, 1], topk=(1, 5, 10))
    assert summary["recall@1"] == pytest.approx(1.0 / 3.0)
    assert summary["recall@5"] == pytest.approx(2.0 / 3.0)
    assert summary["mAP"] == pytest.approx((1.0 + 1.0 / 3.0 + 0.0) / 3.0)

    ci = bootstrap_ci_diff([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], n_resamples=1000, seed=42)
    assert ci["diff_b_minus_a"] == pytest.approx(0.1)
    assert ci["ci_low"] == pytest.approx(0.1)
    assert ci["ci_high"] == pytest.approx(0.1)


def test_hubness_stats():
    # 10 queries all returning item 0 at rank 1
    topk_idxs = [[0, 1, 2] for _ in range(10)]
    stats = compute_hubness_stats(topk_idxs, gallery_size=50, top_k=3)
    assert stats["max_appearances"] == 10
    assert stats["max_query_pct"] == pytest.approx(1.0)
    assert stats["unique_items_retrieved"] == 3


def test_search_engine_hash_guard_and_mean_centering(tmp_path: Path):
    index_dir = Path("index/test_gallery")
    if not index_dir.exists():
        pytest.skip("index/test_gallery not present")

    # Test Hash Guard mismatch raises ValueError
    bad_cfg = tmp_path / "bad_config.yaml"
    bad_cfg.write_text(
        'checkpoint:\n  checkpoint_hash: "0000000000000000000000000000000000000000000000000000000000000000"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HASH GUARD"):
        ECommerceSearchEngine(index_dir=str(index_dir), config_path=str(bad_cfg), device="cpu")

    # Test valid initialization without loading model until needed
    engine = ECommerceSearchEngine(index_dir=str(index_dir), config_path="config.yaml", device="cpu", use_mean_centering=True)
    assert engine.embeddings.shape[0] == len(engine.id_map)
    assert engine.gallery_mean is not None
    norms = np.linalg.norm(engine.embeddings[:10], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    centered = engine._apply_mean_centering(engine.embeddings[:10])
    assert np.allclose(np.linalg.norm(centered, axis=1), 1.0, atol=1e-5)
