"""Unit tests cho Bước 12 (selection_rule_v5, template_cache_v5, fallback & equivalence)."""
from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serve.query_rewriter import multi_template_encode_texts
from src.serve.search_engine import ECommerceSearchEngine
from src.serve.template_cache import (
    TemplateCacheV5,
    encode_texts_with_template_cache,
)


def test_selection_rule_v5_constraints():
    rule_path = PROJECT_ROOT / "reports/phase5c/selection_rule_v5.json"
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    assert rule["seed"] == 42
    assert rule["selection_cv"]["folds_sha256"] == "dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25"
    assert rule["i5_text_gallery_policy"]["forbid_title_vi"] is True
    assert rule["i5_text_gallery_policy"]["index_dir"] == "index/val_gallery_text_v5"
    assert rule["i3_cache_policy"]["missing_key_behavior"] == "fallback_to_1_forward_without_error"

    grid = rule["config_grid"]
    assert len(grid) == 12
    assert len(grid) <= 24
    by_id = {c["config_id"]: c for c in grid}
    assert by_id["cfg_v5_02"]["competes_for_winner"] is False
    assert by_id["cfg_v5_03"]["competes_for_winner"] is False
    assert by_id["cfg_v5_08a"]["text_gallery"] == "internvl2_only"
    assert by_id["cfg_v5_08b"]["text_gallery"] == "internvl2_plus_aux"
    assert [by_id[f"cfg_v5_{i:02d}"]["gamma"] for i in (9, 10, 11)] == [1.0, 2.0, 3.0]


def test_template_cache_v5_hash_guard(tmp_path: pathlib.Path):
    src_dir = PROJECT_ROOT / "index/template_cache_v5"
    if not (src_dir / "manifest.json").exists():
        pytest.skip("index/template_cache_v5 chưa được dựng")

    dst_dir = tmp_path / "cache_copy"
    shutil.copytree(src_dir, dst_dir)

    cache = TemplateCacheV5.load(dst_dir)
    assert cache.embeddings.shape[1] == 1152

    with pytest.raises(ValueError, match="checkpoint_hash"):
        TemplateCacheV5.load(dst_dir, expected_checkpoint_hash="deadbeef")

    manifest_p = dst_dir / "manifest.json"
    m = json.loads(manifest_p.read_text(encoding="utf-8"))
    m["cache_hash"] = "0" * 64
    manifest_p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="cache_hash"):
        TemplateCacheV5.load(dst_dir)


def test_template_cache_v5_equivalence_and_fallback():
    src_dir = PROJECT_ROOT / "index/template_cache_v5"
    if not (src_dir / "manifest.json").exists():
        pytest.skip("index/template_cache_v5 chưa được dựng")

    cache = TemplateCacheV5.load(src_dir)
    engine = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)

    # 1) Test fallback when key is missing (must not error)
    missing_parsed = [{"articleType": ["NonExistentTypeXYZ"], "gender": "Men", "baseColour": ["Black"]}]
    vecs_fb, stats_fb = encode_texts_with_template_cache(
        engine, ["ao thun nam mau den"], missing_parsed, cache
    )
    assert vecs_fb.shape == (1, 1152)
    assert stats_fb["n_fallback"] == 1
    assert stats_fb["fallback_rate"] == 1.0
    assert np.isclose(np.linalg.norm(vecs_fb[0]), 1.0, atol=1e-3)

    # 2) Equivalence test on 200 random catalog combinations (cosine >= 0.999)
    rng = random.Random(42)
    all_keys = list(cache.key_map.keys())
    sample_keys = rng.sample(all_keys, min(200, len(all_keys)))

    queries = []
    parsed_list = []
    for k in sample_keys:
        arts_s, gd_s, cols_s = k.split("__")
        pr = {
            "articleType": arts_s.split("|") if arts_s else [],
            "gender": gd_s if gd_s else None,
            "baseColour": cols_s.split("|") if cols_s else [],
        }
        parsed_list.append(pr)
        queries.append(f"sản phẩm {arts_s} {gd_s} {cols_s}".strip())

    vecs_cached, stats_hit = encode_texts_with_template_cache(engine, queries, parsed_list, cache)
    assert stats_hit["n_fallback"] == 0
    vecs_direct = multi_template_encode_texts(engine, queries, parsed_list, n_templates=2)

    cosines = np.sum(vecs_cached * vecs_direct, axis=1)
    min_cos = float(np.min(cosines))
    assert min_cos >= 0.999, f"Min cosine {min_cos:.6f} < 0.999"
