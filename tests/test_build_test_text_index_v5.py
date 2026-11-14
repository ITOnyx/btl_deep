"""Unit tests for `scripts/build_test_text_index_v5.py` and `index/test_gallery_text_v5/` (Prompt 14c Step 1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.build_test_text_index_v5 import (
    LOCKED_BUILD_VAL_SCRIPT_SHA256,
    compare_text_gallery_dirs,
    verify_text_gallery_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_build_val_text_index_v5_untouched():
    """Verify that scripts/build_val_text_index_v5.py and index/val_gallery_text_v5/ were not modified."""
    val_script = PROJECT_ROOT / "scripts" / "build_val_text_index_v5.py"
    assert _sha256_file(val_script) == LOCKED_BUILD_VAL_SCRIPT_SHA256

    val_manifest_path = PROJECT_ROOT / "index" / "val_gallery_text_v5" / "manifest.json"
    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_manifest = json.load(f)
    assert val_manifest["files"]["internvl2_only_embeddings"]["sha256"] == (
        "a1dd96ab2efa293e0748f02934148a547522e94c8124714e1b6f631a1e89c5a2"
    )
    assert val_manifest["files"]["internvl2_plus_aux_embeddings"]["sha256"] == (
        "ea0df85b52dc69d655b1230c55642bfa475f462c5fff1f56c44b1e1481a50f63"
    )
    assert val_manifest["files"]["metadata_aux_embeddings"]["sha256"] == (
        "faf6880e9d3bb7b01544c91b6888e8afc07ea770f964659bcc99c632ffe7b04b"
    )


def test_val_parity_verification_dir():
    """Verify row-by-row cosine >= 0.9999 between index/val_gallery_text_v5/ and index/_verify_val_text_v5/."""
    ref_dir = PROJECT_ROOT / "index" / "val_gallery_text_v5"
    cand_dir = PROJECT_ROOT / "index" / "_verify_val_text_v5"
    assert cand_dir.exists(), "index/_verify_val_text_v5/ must exist"
    parity = compare_text_gallery_dirs(ref_dir, cand_dir)
    assert parity["passed_all_0_9999"] is True
    for fname, stats in parity["files"].items():
        assert stats["n_rows"] == 1884
        assert stats["min_cosine"] >= 0.9999, f"{fname} min_cosine={stats['min_cosine']}"
        assert stats["count_cosine_lt_0_9999"] == 0


def test_test_gallery_text_v5_id_order_shape_l2_and_manifest():
    """Verify id order, shape (1290, 1152), float32, L2 norms, 0 NaNs, 0 duplicate IDs, and manifest."""
    out_dir = PROJECT_ROOT / "index" / "test_gallery_text_v5"
    test_gallery_dir = PROJECT_ROOT / "index" / "test_gallery"
    config_path = PROJECT_ROOT / "config.yaml"

    sanity = verify_text_gallery_index(
        out_dir=out_dir,
        gallery_dir=test_gallery_dir,
        config_path=config_path,
        expected_n_vectors=1290,
    )
    assert sanity["passed"] is True
    assert sanity["checkpoint_hash_match"] is True
    assert sanity["catalog_hash_match"] is True
    assert sanity["n_vectors"] == 1290
    assert sanity["n_duplicate_ids"] == 0
    assert sanity["id_order_matches_gallery_id_map"] is True

    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(test_gallery_dir / "id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)

    expected_ids = [str(m.get("product_id") or m.get("item_id")) for m in id_map]
    assert manifest["id_order"] == expected_ids
    assert len(manifest["id_order"]) == 1290
    assert len(set(manifest["id_order"])) == 1290
    assert manifest["seed"] == 42
    assert manifest["combination_method"] == "merge_v1"
    assert "InternVL2-2B" in manifest["caption_model"]
    assert len(manifest["shard_hashes"]) == 3
    for shard_name in [
        "shard_test_0000.parquet",
        "shard_test_0001.parquet",
        "shard_test_0002.parquet",
    ]:
        assert shard_name in manifest["shard_hashes"]

    for key, fname in [
        ("internvl2_only_embeddings", "internvl2_only_embeddings.npy"),
        ("internvl2_plus_aux_embeddings", "internvl2_plus_aux_embeddings.npy"),
        ("metadata_aux_embeddings", "metadata_aux_embeddings.npy"),
    ]:
        arr = np.load(out_dir / fname)
        assert arr.shape == (1290, 1152)
        assert arr.dtype == np.float32
        assert int(np.isnan(arr).sum() + np.isinf(arr).sum()) == 0
        norms = np.linalg.norm(arr, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)
        assert _sha256_file(out_dir / fname) == manifest["files"][key]["sha256"]
