"""
Unit tests cho Pha 4:
- FAISS HNSW Flat Index xây dựng và tìm kiếm
- Lưu/nạp tệp nhị phân .bin
- Sanity Check 4 tiêu chí (0 duplicate IDs, 0 leak, rank 1 self-query, vector count match)
"""
import os
import json
import pytest
import numpy as np

from src.index.faiss_index import FaissHNSWIndex, FaissFlatIndex
from src.eval.sanity_check import run_sanity_checks


def test_faiss_hnsw_construction_and_search():
    """Kiểm tra tạo index HNSW, thêm vector và tìm kiếm cosine similarity."""
    dim = 128
    n_vecs = 200
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vecs, dim)).astype(np.float32)

    hnsw = FaissHNSWIndex(dim=dim, M=16, ef_construction=100, ef_search=64)
    hnsw.add(vectors)

    assert hnsw.ntotal == n_vecs

    # Tìm kiếm chính vector đầu tiên
    query = vectors[:5]
    scores, indices = hnsw.search(query, top_k=3)

    assert scores.shape == (5, 3)
    assert indices.shape == (5, 3)
    # Vector thứ i tự truy vấn chính nó phải ra rank 1 với cosine xấp xỉ 1.0
    for i in range(5):
        assert indices[i, 0] == i
        assert scores[i, 0] >= 0.9999


def test_faiss_hnsw_save_and_load(tmp_path):
    """Kiểm tra lưu index ra file bin và nạp lại."""
    dim = 64
    n_vecs = 50
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vecs, dim)).astype(np.float32)

    hnsw = FaissHNSWIndex(dim=dim, M=16, ef_construction=50, ef_search=32)
    hnsw.add(vectors)

    save_file = tmp_path / "test_hnsw.bin"
    hnsw.save(save_file)
    assert save_file.exists()

    loaded = FaissHNSWIndex.load(save_file, ef_search=64)
    assert loaded.ntotal == n_vecs
    assert loaded.ef_search == 64

    # Kiểm tra truy vấn trả về kết quả tương đương
    query = vectors[:2]
    scores_orig, idx_orig = hnsw.search(query, top_k=2)
    scores_load, idx_load = loaded.search(query, top_k=2)

    np.testing.assert_array_equal(idx_orig, idx_load)
    np.testing.assert_allclose(scores_orig, scores_load, atol=1e-5)


def test_faiss_flat_vs_hnsw_recall():
    """Kiểm tra Flat Index và HNSW Index có độ tương đồng cao (Recall@10 >= 0.95)."""
    dim = 128
    n_vecs = 300
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_vecs, dim)).astype(np.float32)

    hnsw = FaissHNSWIndex(dim=dim, M=32, ef_construction=100, ef_search=64)
    hnsw.add(vectors)

    flat = FaissFlatIndex(dim=dim)
    flat.add(vectors)

    query = vectors[:20]
    _, gt_indices = flat.search(query, top_k=10)
    _, hnsw_indices = hnsw.search(query, top_k=10)

    recalls = []
    for i in range(len(query)):
        inter = set(gt_indices[i]) & set(hnsw_indices[i])
        recalls.append(len(inter) / 10.0)

    mean_recall = np.mean(recalls)
    assert mean_recall >= 0.95, f"Recall quá thấp: {mean_recall}"
