"""
Unit tests for Phase 6: Streamlit Application Components & AppSearchService.
Tests:
- Hash Guard rejection on mismatch (mock config/manifest without modifying checkpoint)
- Image validation (format, size limit, corrupted bytes, RGB conversion, in-memory only)
- 4 Search Streams execution (Text->Image, Image->Image, CIR Catalog, CIR Upload)
- Sanity Check: gallery image query returns itself at rank 1 with cosine ≈ 1.0000
- Top-10 output structure (sorted descending, valid gallery IDs)
- Diacritics vs non-diacritics query handling
- Configuration switching (cfg_v5_09, cfg_v5_01, model_only, cir_v1_05, cir_v1_01)
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from app.engine_service import (
    AppSearchService,
    HashGuardError,
    ImageValidationError,
    validate_and_load_image,
    MAX_UPLOAD_BYTES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_hash_guard_rejection_on_mismatch(tmp_path):
    """Verifies that Hash Guard strictly rejects index loading when hash does not match (Rule 6)."""
    # Create a mock index directory with mismatched manifest
    mock_idx = tmp_path / "mock_index"
    mock_idx.mkdir()

    bad_manifest = {
        "checkpoint_hash": "deadbeef1234567890abcdef1234567890abcdef1234567890abcdef12345678",
        "n_vectors": 10,
    }
    (mock_idx / "manifest.json").write_text(json.dumps(bad_manifest), encoding="utf-8")
    (mock_idx / "id_map.json").write_text(json.dumps([]), encoding="utf-8")

    # AppSearchService must raise HashGuardError without modifying any real checkpoint
    with pytest.raises(HashGuardError) as exc_info:
        AppSearchService(index_dir=str(mock_idx))
    assert "HASH GUARD TỪ CHỐI NẠP" in str(exc_info.value)


def test_validate_and_load_image_valid_formats():
    """Verifies in-memory validation and RGB conversion for valid image formats."""
    # Create in-memory valid images
    for fmt, ext, mode in [("JPEG", ".jpg", "RGB"), ("PNG", ".png", "RGBA"), ("WEBP", ".webp", "RGB")]:
        bio = io.BytesIO()
        test_img = Image.new(mode, (100, 100), color=(255, 0, 0) if mode == "RGB" else (255, 0, 0, 128))
        test_img.save(bio, format=fmt)

        loaded = validate_and_load_image(bio.getvalue(), filename=f"test{ext}")
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)


def test_validate_and_load_image_rejects_unsupported_extension():
    """Verifies rejection of unsupported file extensions."""
    bio = io.BytesIO(b"dummy data")
    with pytest.raises(ImageValidationError) as exc:
        validate_and_load_image(bio, filename="malicious.exe")
    assert "không được hỗ trợ" in str(exc.value)


def test_validate_and_load_image_rejects_oversized_file():
    """Verifies rejection of files exceeding 10 MB."""
    oversized_bytes = b"0" * (MAX_UPLOAD_BYTES + 1024)
    with pytest.raises(ImageValidationError) as exc:
        validate_and_load_image(oversized_bytes, filename="large.jpg")
    assert "vượt quá giới hạn" in str(exc.value)


def test_validate_and_load_image_rejects_corrupted_data():
    """Verifies rejection of corrupted image bytes."""
    corrupted_bytes = b"not an image at all"
    with pytest.raises(ImageValidationError) as exc:
        validate_and_load_image(corrupted_bytes, filename="corrupted.jpg")
    assert "Không thể giải mã hình ảnh" in str(exc.value)


@pytest.fixture(scope="module")
def search_service():
    """Initializes and caches the search service for test_gallery."""
    return AppSearchService(index_dir="index/test_gallery", config_path="config.yaml")


def test_system_metadata(search_service):
    """Verifies metadata dictionary structure for sidebar."""
    meta = search_service.get_system_metadata()
    assert meta["checkpoint_hash"] == "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d"
    assert meta["total_items"] == 1290
    assert meta["seed"] == 42


def test_stream1_text_to_image_diacritics_parity(search_service):
    """Verifies Stream 1 returns 10 sorted results for queries with and without diacritics."""
    q_with_dia = "áo sơ mi nam màu xanh"
    q_no_dia = "ao so mi nam mau xanh"

    res_with, parsed_with, ms_with = search_service.search_text_to_image(q_with_dia, config_id="cfg_v5_09", top_k=10)
    res_no, parsed_no, ms_no = search_service.search_text_to_image(q_no_dia, config_id="cfg_v5_09", top_k=10)

    assert len(res_with) == 10
    assert len(res_no) == 10

    # Check scores sorted descending
    scores_with = [r["final_score"] for r in res_with]
    assert scores_with == sorted(scores_with, reverse=True)

    # Check parser parsed attributes
    assert parsed_with.get("gender") == "Men"
    assert "Shirts" in parsed_with.get("articleType", [])
    assert parsed_no.get("gender") == "Men"
    assert "Shirts" in parsed_no.get("articleType", [])


def test_stream1_config_switching(search_service):
    """Verifies switching between cfg_v5_09, cfg_v5_01, and model_only."""
    query = "áo sơ mi nam màu xanh"

    res_09, _, _ = search_service.search_text_to_image(query, config_id="cfg_v5_09", top_k=10)
    res_01, _, _ = search_service.search_text_to_image(query, config_id="cfg_v5_01", top_k=10)
    res_mo, _, _ = search_service.search_text_to_image(query, config_id="model_only", top_k=10)

    assert len(res_09) == 10
    assert len(res_01) == 10
    assert len(res_mo) == 10

    # model_only must have 0 bonus
    for r in res_mo:
        assert r["attribute_bonus"] == 0.0
        assert r["final_score"] == pytest.approx(r["cosine_score"], abs=1e-6)

    # cfg_v5_09 has gamma=1.0, cfg_v5_01 has gamma=1.5
    # For matching items, bonus for cfg_v5_01 should be higher than cfg_v5_09
    if res_01[0]["attribute_bonus"] > 0:
        assert res_01[0]["attribute_bonus"] >= res_09[0]["attribute_bonus"]


def test_stream2_image_to_image_sanity_check(search_service):
    """
    Sanity Check (Rule E, Phase 4/6 requirement):
    Querying a gallery image returns itself at rank 1 with cosine ≈ 1.0000.
    """
    sample_pid = "15970"
    sample_path = PROJECT_ROOT / f"data/fashion-dataset/images/{sample_pid}.jpg"
    assert sample_path.exists(), f"Missing sample image {sample_path}"

    img = Image.open(sample_path).convert("RGB")
    results, elapsed_ms = search_service.search_image_to_image(image_input=img, top_k=10)

    assert len(results) == 10
    assert results[0]["product_id"] == sample_pid, f"Expected {sample_pid} at rank 1, got {results[0]['product_id']}"
    assert results[0]["cosine_score"] == pytest.approx(1.0000, abs=1e-3)
    assert elapsed_ms > 0.0


def test_stream3_cir_catalog(search_service):
    """Verifies Stream 3: CIR from Catalog Reference."""
    ref_pid = "15970"
    mod_text = "nhưng màu đỏ"

    results, parsed, elapsed_ms = search_service.search_cir_catalog(
        product_id=ref_pid,
        modifier_text=mod_text,
        config_id="cir_v1_05",
        top_k=10
    )

    assert len(results) == 10
    assert "Red" in parsed.get("baseColour", [])
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_stream4_cir_upload(search_service):
    """Verifies Stream 4: CIR from Uploaded Image (in-memory)."""
    sample_path = PROJECT_ROOT / "data/fashion-dataset/images/15970.jpg"
    img = Image.open(sample_path).convert("RGB")
    mod_text = "nhưng màu đen"

    results, parsed, elapsed_ms = search_service.search_cir_upload(
        uploaded_image=img,
        modifier_text=mod_text,
        config_id="cir_v1_05",
        top_k=10
    )

    assert len(results) == 10
    assert "Black" in parsed.get("baseColour", [])
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_cir_v1_06_domain_consistency(search_service):
    """Verifies that cir_v1_06 eliminates out-of-domain products (e.g. shoes in t-shirt queries)."""
    sample_pid = "15970"  # Men Navy Blue Shirt (Quần áo > Áo)
    mod_text = "nhưng màu vàng"

    results, parsed, elapsed_ms = search_service.search_cir_catalog(
        product_id=sample_pid,
        modifier_text=mod_text,
        config_id="cir_v1_06",
        top_k=10
    )
    assert len(results) > 0
    for r in results:
        assert r["domain"] == "Quần áo", f"Found non-apparel item in cir_v1_06: {r}"


def test_cfg_v5_10_domain_guard(search_service):
    """Verifies that cfg_v5_10 enforces hard domain guard on Text-to-Image."""
    query = "túi màu đen"
    results, parsed, elapsed_ms = search_service.search_text_to_image(
        query_text=query,
        config_id="cfg_v5_10",
        top_k=10
    )
    assert len(results) > 0
    for r in results:
        assert r["domain"] == "Túi & Ví", f"Found non-bag item in cfg_v5_10: {r}"
