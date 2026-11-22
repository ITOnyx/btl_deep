"""Bẫy lỗi của AppSearchService và giao diện: dữ liệu index hỏng/lệch, đầu vào không hợp lệ, ảnh bất thường.

Mọi test ở đây dừng TRƯỚC khi nạp mô hình SigLIP (nhanh, không cần GPU).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from app.engine_service import (
    AppSearchService,
    ImageValidationError,
    IndexLoadError,
    SearchInputError,
    SearchRuntimeError,
    _check_query_vector,
    _check_request,
    _gpu_guard,
    validate_and_load_image,
)
from src.index.faiss_index import FaissHNSWIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
HASH = CFG["checkpoint"]["checkpoint_hash"]
DIM = int(CFG["checkpoint"].get("embedding_dim", 1152))


def _make_index(d: Path, n: int = 5, *, id_map=None, emb_shape=None, faiss_n=None, write_faiss=True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"checkpoint_hash": HASH}), encoding="utf-8")
    ids = id_map if id_map is not None else [{"item_id": str(i), "path": ""} for i in range(n)]
    (d / "id_map.json").write_text(json.dumps(ids), encoding="utf-8")
    rng = np.random.default_rng(42)
    np.save(d / "image_embeddings.npy", rng.standard_normal(emb_shape or (n, DIM)).astype(np.float32))
    if write_faiss:
        idx = FaissHNSWIndex(dim=DIM)
        idx.add(rng.standard_normal((faiss_n if faiss_n is not None else n, DIM)).astype(np.float32))
        idx.save(d / "faiss_hnsw.bin")
    return d


# ------------------------------------------------------------------ nạp index
def test_corrupt_id_map_json(tmp_path):
    d = _make_index(tmp_path / "idx")
    (d / "id_map.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexLoadError, match="JSON không hợp lệ"):
        AppSearchService(index_dir=str(d))


@pytest.mark.parametrize("id_map, msg", [
    ([], "rỗng"),
    ([{"path": "a"}], "thiếu 'item_id'"),
    ([{"item_id": "1"}, {"item_id": "1"}], "bị trùng"),
])
def test_bad_id_map(tmp_path, id_map, msg):
    d = _make_index(tmp_path / "idx", n=max(len(id_map), 1), id_map=id_map)
    with pytest.raises(IndexLoadError, match=msg):
        AppSearchService(index_dir=str(d))


def test_embeddings_shape_mismatch(tmp_path):
    d = _make_index(tmp_path / "idx", n=5, emb_shape=(4, DIM))
    with pytest.raises(IndexLoadError, match="shape"):
        AppSearchService(index_dir=str(d))


def test_missing_faiss_file(tmp_path):
    d = _make_index(tmp_path / "idx", write_faiss=False)
    with pytest.raises(IndexLoadError, match="faiss_hnsw.bin"):
        AppSearchService(index_dir=str(d))


def test_faiss_count_mismatch(tmp_path):
    d = _make_index(tmp_path / "idx", n=5, faiss_n=3)
    with pytest.raises(IndexLoadError, match="lệch với id_map"):
        AppSearchService(index_dir=str(d))


def test_corrupt_faiss_file(tmp_path):
    d = _make_index(tmp_path / "idx")
    (d / "faiss_hnsw.bin").write_bytes(b"garbage")
    with pytest.raises(IndexLoadError, match="FAISS"):
        AppSearchService(index_dir=str(d))


def test_broken_config_yaml(tmp_path):
    d = _make_index(tmp_path / "idx")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("checkpoint: [unclosed", encoding="utf-8")
    with pytest.raises(IndexLoadError, match="YAML"):
        AppSearchService(index_dir=str(d), config_path=str(cfg))


def test_config_missing_required_key(tmp_path):
    d = _make_index(tmp_path / "idx")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"checkpoint": {"checkpoint_hash": HASH}}), encoding="utf-8")
    with pytest.raises(IndexLoadError, match="paths.checkpoint_file"):
        AppSearchService(index_dir=str(d), config_path=str(cfg))


# ------------------------------------------------------------------ tham số tìm kiếm
@pytest.mark.parametrize("top_k", [0, -1, 101, 2.5, "10", True, None])
def test_check_request_rejects_bad_top_k(top_k):
    with pytest.raises(SearchInputError):
        _check_request("áo", top_k, "cfg_v5_09", ("cfg_v5_09",))


def test_check_request_rules():
    with pytest.raises(SearchInputError, match="không hợp lệ"):
        _check_request("áo", 10, "cfg_v4_01", ("cfg_v5_09",))
    with pytest.raises(SearchInputError, match="quá dài"):
        _check_request("a" * 501, 10, "cfg_v5_09", ("cfg_v5_09",))
    with pytest.raises(SearchInputError, match="chuỗi"):
        _check_request(123, 10, "cfg_v5_09", ("cfg_v5_09",))
    assert _check_request("  áo  ", np.int64(5), "cfg_v5_09", ("cfg_v5_09",)) == "áo"


def _bare_service() -> AppSearchService:
    """Service chưa nạp gì: các kiểm tra đầu vào phải chặn trước khi chạm tới model/index."""
    return object.__new__(AppSearchService)


@pytest.mark.parametrize("call", [
    lambda s: s.search_text_to_image("   ", "cfg_v5_09", 10),
    lambda s: s.search_text_to_image("áo", "khong_co", 10),
    lambda s: s.search_text_to_image("áo", "cfg_v5_09", 0),
    lambda s: s.search_image_to_image(None, 10),
    lambda s: s.search_cir_catalog("15970", "", "cir_v1_05", 10),
    lambda s: s.search_cir_catalog("  ", "màu đỏ", "cir_v1_05", 10),
    lambda s: s.search_cir_catalog("15970", "màu đỏ", "cfg_v5_09", 10),
    lambda s: s.search_cir_upload(None, "màu đỏ", "cir_v1_05", 10),
    lambda s: s.search_cir_upload(Image.new("RGB", (20, 20)), " ", "cir_v1_05", 10),
])
def test_search_input_validation(call):
    with pytest.raises(SearchInputError):
        call(_bare_service())


def test_query_vector_guard():
    with pytest.raises(SearchRuntimeError):
        _check_query_vector(np.full((1, 4), np.nan, dtype=np.float32))
    with pytest.raises(SearchRuntimeError):
        _check_query_vector(np.zeros((1, 4), dtype=np.float32))
    v = np.ones((1, 4), dtype=np.float32)
    assert _check_query_vector(v) is not None


def test_gpu_oom_becomes_friendly_error():
    @_gpu_guard
    def boom():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(SearchRuntimeError, match="GPU hết bộ nhớ"):
        boom()


# ------------------------------------------------------------------ ảnh
def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


def test_huge_dimension_image_rejected_before_decode():
    # PNG đen 9000x10 nén chỉ vài KB nhưng vượt giới hạn cạnh 8000 px
    with pytest.raises(ImageValidationError, match="quá lớn"):
        validate_and_load_image(_png(9000, 10), "big.png")


def test_animated_webp_uses_first_frame():
    frames = [Image.new("RGB", (32, 32), c) for c in ("red", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:])
    img = validate_and_load_image(buf.getvalue(), "anim.webp")
    assert img.mode == "RGB" and img.size == (32, 32)


def test_wrong_extension_content_rejected():
    with pytest.raises(ImageValidationError):
        validate_and_load_image(b"%PDF-1.4 not an image", "fake.jpg")


# ------------------------------------------------------------------ giao diện
def test_product_card_num_handles_bad_scores():
    from app.components.product_card import _num

    assert _num(None) == 0.0
    assert _num(float("nan")) == 0.0
    assert _num(float("inf")) == 0.0
    assert _num("abc") == 0.0
    assert _num("0.5") == 0.5


def test_load_sample_image_handles_missing_and_corrupt(tmp_path):
    from app.components.errors import load_sample_image

    assert load_sample_image(tmp_path / "none.jpg") is None
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert load_sample_image(bad) is None
