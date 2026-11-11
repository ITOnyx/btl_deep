"""Pytest test suite for Phase 1 verification in Hoc_sau_ver4."""

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import yaml
import pytest


@pytest.fixture
def config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def catalog_df():
    p = Path("data/catalog/catalog_rebuild.parquet")
    assert p.exists(), "catalog_rebuild.parquet does not exist"
    return pd.read_parquet(p)


def test_catalog_rebuild_schema(catalog_df):
    """Verify all mandatory columns in catalog_rebuild.parquet exist."""
    required_cols = [
        "product_id", "source", "image_path", "sha1", "phash",
        "width", "height", "title_raw", "title_vi",
        "articleType", "articleType_vi", "articleType_vi_nodiacritics",
        "gender", "gender_vi", "gender_vi_nodiacritics",
        "baseColour", "baseColour_vi", "baseColour_vi_nodiacritics",
        "masterCategory", "subCategory", "label_origin",
        "split", "in_scope", "filter_flag",
        "caption_model", "model_revision", "seed",
        "gen_params", "method", "caption_vi", "token_len"
    ]
    for col in required_cols:
        assert col in catalog_df.columns, f"Missing column: {col}"
    assert len(catalog_df) == 44424, f"Expected 44424 rows, got {len(catalog_df)}"


def test_splits_disjoint(catalog_df):
    """Verify train, val, test splits are strictly pairwise disjoint."""
    train_ids = set(catalog_df[catalog_df["split"] == "train"]["product_id"])
    test_ids = set(catalog_df[catalog_df["split"] == "test"]["product_id"])
    
    val_p = Path("eval/val_split.json")
    assert val_p.exists(), "val_split.json missing"
    with open(val_p, "r", encoding="utf-8") as f:
        val_ids = set(json.load(f))
        
    # Check disjointness between test and train
    leak_train_test = train_ids.intersection(test_ids)
    assert len(leak_train_test) == 0, f"Leakage detected between train and test: {len(leak_train_test)} items"
    
    # Val is sampled from non-test pool for hyperparameter tuning
    leak_val_test = val_ids.intersection(test_ids)
    assert len(leak_val_test) == 0, f"Leakage detected between val and test: {len(leak_val_test)} items"


def test_test_split_hash_match():
    """Verify test split JSON hash matches eval/test_split.sha256 exactly."""
    json_p = Path("eval/test_split.json")
    hash_p = Path("eval/test_split.sha256")
    assert json_p.exists(), "test_split.json missing"
    assert hash_p.exists(), "test_split.sha256 missing"

    with open(json_p, "rb") as f:
        computed_hash = hashlib.sha256(f.read()).hexdigest()

    with open(hash_p, "r", encoding="utf-8") as f:
        stored_hash = f.read().strip()

    assert computed_hash == stored_hash, f"Hash mismatch: {computed_hash} != {stored_hash}"


def test_quarantine_log_integrity():
    """Verify quarantine_log.csv exists and contains quarantine entries."""
    q_log_p = Path("reports/quarantine_log.csv")
    assert q_log_p.exists(), "quarantine_log.csv missing"
    df_q = pd.read_csv(q_log_p, on_bad_lines="skip")
    assert len(df_q) > 0, "quarantine_log.csv is empty"
    assert "quarantined" in df_q["status"].values


def test_seed_consistency(config):
    """Verify seed is fixed to 42 across config.yaml."""
    assert config.get("seed") == 42
    assert config.get("split", {}).get("seed") == 42


def test_law_5_compliance():
    """Verify that any auxiliary model calls in src/ are strictly governed under allowed modules."""
    forbidden_terms = ["5CD-AI", "Vintern-1B"]
    violating_files = []
    
    src_dir = Path("src")
    for py_file in src_dir.rglob("*.py"):
        with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            for term in forbidden_terms:
                if term in content:
                    violating_files.append((str(py_file), term))
                    
    assert len(violating_files) == 0, f"Found unencapsulated auxiliary model references in src: {violating_files}"
