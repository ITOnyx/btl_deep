"""Unit Tests for Phase 2: Dense Captioning Modules.
Tests prompts, deterministic merger, pilot sample integrity, and zero test leakage.
"""

import json
from pathlib import Path
import pytest

from src.caption.prompts import PROMPT_VERSION, INTERNVL2_PROMPT_TEMPLATE, VINTERN_PROMPT_TEMPLATE
from src.caption.merger import merge_caption_v1, clean_introductory_phrases, detect_and_resolve_conflicts


def test_prompts_structure():
    """Verify prompt templates comply with Law 4 and Phase 2 requirements."""
    assert PROMPT_VERSION == "v1.0"
    assert "<image>" in INTERNVL2_PROMPT_TEMPLATE
    assert "loại sản phẩm" in INTERNVL2_PROMPT_TEMPLATE
    assert "giới tính" in INTERNVL2_PROMPT_TEMPLATE
    assert "màu sắc" in INTERNVL2_PROMPT_TEMPLATE
    assert "<image>" in VINTERN_PROMPT_TEMPLATE


def test_clean_introductory_phrases():
    """Verify boilerplate prefixes are cleanly stripped."""
    text1 = "Trong hình là một chiếc áo thun nam màu đen cổ tròn."
    cleaned1 = clean_introductory_phrases(text1)
    assert not cleaned1.startswith("Trong hình là")
    assert not cleaned1.startswith("một chiếc")
    assert cleaned1.startswith("áo thun")


def test_detect_and_resolve_conflicts():
    """Verify gender conflicts are properly flagged and sanitized."""
    text = "Áo thun phong cách thể thao dành cho nam năng động."
    cleaned, conflicts = detect_and_resolve_conflicts(text, expected_gender_vi="nữ", expected_colour_vi="đen")
    assert len(conflicts) > 0
    assert "gender_conflict_male_in_female" in conflicts
    assert "dành cho nam" not in cleaned


def test_deterministic_merger():
    """Verify merge_v1 adheres to anchor prefix, conflict resolution, and max 128 tokens."""
    metadata = {
        "articleType_vi": "áo sơ mi",
        "gender_vi": "nam",
        "baseColour_vi": "trắng"
    }
    core_caption = "Bức ảnh chụp một chiếc áo sơ mi nam màu trắng dài tay cổ bẻ thanh lịch."
    res = merge_caption_v1(core_caption=core_caption, metadata=metadata, aux_caption="")
    
    assert res["version"] == "merge_v1"
    assert res["prefix"].lower().startswith("áo sơ mi nam màu trắng")
    assert res["token_len"] <= 128
    assert res["caption"].lower().startswith("áo sơ mi nam màu trắng")


def test_pilot_sample_integrity_and_zero_test_leak():
    """Verify pilot sample has 200 items, exactly 20 manual review items, and 0 test leakage."""
    pilot_path = Path("eval/pilot_200_sample.json")
    assert pilot_path.exists(), "pilot_200_sample.json missing"
    
    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
        
    assert len(pilot_data) == 200
    
    # Check manual review count
    manual_reviews = [x for x in pilot_data if x.get("for_manual_review")]
    assert len(manual_reviews) == 20
    
    # Check split constraint: pilot must strictly come from train
    for item in pilot_data:
        assert item["split"] == "train"
        assert item["in_scope"] is True
        
    # Verify ZERO overlap with locked test_split.json
    test_split_path = Path("eval/test_split.json")
    assert test_split_path.exists(), "test_split.json missing"
    with open(test_split_path, "r", encoding="utf-8") as f:
        test_ids = set(str(x) for x in json.load(f))
        
    pilot_ids = set(str(x["product_id"]) for x in pilot_data)
    overlap = pilot_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Critical violation: Pilot samples leaked into test split: {overlap}"
