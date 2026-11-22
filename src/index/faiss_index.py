"""
Module: src.index.faiss_index
Quản lý FAISS HNSW Index và Flat Index theo đúng config.yaml và AGENTS.md.
Tham số bắt buộc: IndexHNSWFlat, M=32, efConstruction=200, efSearch=128, metric=inner_product.
"""
import os
import pathlib
from typing import Tuple, Optional, Union

import faiss
import numpy as np


class FaissHNSWIndex:
    """
    Wrapper quản lý FAISS HNSW Flat Index chuẩn hóa L2 cho cosine similarity.
    """
    def __init__(
        self,
        dim: int = 1152,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128
    ):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

    def set_ef_search(self, ef_search: int) -> None:
        """Cập nhật độ sâu tìm kiếm efSearch."""
        self.ef_search = ef_search
        self.index.hnsw.efSearch = ef_search

    def add(self, vectors: np.ndarray) -> None:
        """Thêm các vector đã chuẩn hóa L2 vào index."""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        # Đảm bảo chuẩn hóa L2
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tìm kiếm Top-K vector gần nhất theo cosine similarity (inner product trên vector L2-normalized).
        Trả về: (scores, indices) với shape [batch_size, top_k].
        """
        # Sao chép trước khi chuẩn hoá: normalize_L2 sửa tại chỗ, không được ghi đè mảng của người gọi
        # (vd. trên CPU, vector truy vấn CIR có thể là view của ma trận embedding gallery).
        query_vectors = np.array(query_vectors, dtype=np.float32, order="C", copy=True)
        faiss.normalize_L2(query_vectors)
        scores, indices = self.index.search(query_vectors, top_k)
        return scores, indices

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def save(self, file_path: Union[str, pathlib.Path]) -> None:
        """Lưu index nhị phân ra tệp (hỗ trợ đường dẫn Unicode trên Windows)."""
        path_str = str(file_path)
        os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
        buf = faiss.serialize_index(self.index)
        with open(path_str, "wb") as f:
            f.write(buf.tobytes())

    @classmethod
    def load(cls, file_path: Union[str, pathlib.Path], ef_search: int = 128) -> "FaissHNSWIndex":
        """Nạp index từ tệp nhị phân (hỗ trợ đường dẫn Unicode trên Windows)."""
        path_str = str(file_path)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Không tìm thấy index FAISS tại: {path_str}")

        with open(path_str, "rb") as f:
            buf_read = np.frombuffer(f.read(), dtype=np.uint8)

        raw_index = faiss.deserialize_index(buf_read)
        instance = cls(dim=raw_index.d, ef_search=ef_search)
        instance.index = raw_index
        instance.index.hnsw.efSearch = ef_search
        return instance


class FaissFlatIndex:
    """
    FAISS Flat Index (IndexFlatIP) dùng để đo Ground Truth chính xác 100% khi đánh giá Recall-ANN.
    """
    def __init__(self, dim: int = 1152):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors)

        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Sao chép trước khi chuẩn hoá: normalize_L2 sửa tại chỗ, không được ghi đè mảng của người gọi
        # (vd. trên CPU, vector truy vấn CIR có thể là view của ma trận embedding gallery).
        query_vectors = np.array(query_vectors, dtype=np.float32, order="C", copy=True)
        faiss.normalize_L2(query_vectors)
        return self.index.search(query_vectors, top_k)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal
