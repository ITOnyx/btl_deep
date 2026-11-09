"""
Optimized In-Memory Sparse BM25 and Reciprocal Rank Fusion (RRF) (src/lexical.py)
Phase 3e:
- Uses scipy.sparse.csr_matrix for vector-matrix dot products (latency < 2 ms/query).
- Precomputes all BM25 Okapi weights (k1=1.5, b=0.75) for 38,646 documents.
- Incorporates src/vi_to_en_terms.json for bilingual term expansion.
- Fixed RRF (k=60) for merging dense and sparse ranking scores.
"""

import math
import re
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import sys
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.vi_vocab import (
        remove_diacritics,
        GENDER_MAP,
        COLOUR_MAP,
        ARTICLE_TYPE_MAP,
        USAGE_MAP
    )
except ImportError:
    from vi_vocab import (
        remove_diacritics,
        GENDER_MAP,
        COLOUR_MAP,
        ARTICLE_TYPE_MAP,
        USAGE_MAP
    )

class BM25Index:
    def __init__(
        self,
        catalog_path: str = "data/catalog/catalog.parquet",
        terms_map_path: str = "src/vi_to_en_terms.json",
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.k1 = k1
        self.b = b

        # 1. Load bilingual terms expansion mapping
        t_path = Path(terms_map_path)
        self.vi_to_en = {}
        if t_path.exists():
            with open(t_path, "r", encoding="utf-8") as f:
                self.vi_to_en = json.load(f)

        # 2. Load catalog in_scope
        df = pd.read_parquet(catalog_path)
        self.df = df[df["in_scope"] == True].reset_index(drop=True)
        self.N = len(self.df)
        self.item_ids = self.df["item_id"].tolist()
        self.splits = self.df["split"].fillna("unknown").to_numpy()

        # 3. Build sparse BM25 matrix
        self._build_sparse_matrix()

    def _tokenize(self, text: str) -> List[str]:
        """Accent-insensitive lowercase tokenization."""
        clean = remove_diacritics(text)
        tokens = re.findall(r"\b\w+\b", clean)
        return tokens

    def _build_sparse_matrix(self):
        """Constructs scipy.sparse.csr_matrix with precomputed BM25 scores."""
        names = self.df["name"].fillna("").astype(str).tolist()
        genders = self.df["gender"].fillna("").astype(str).tolist()
        art_types = self.df["articleType"].fillna("").astype(str).tolist()
        colours = self.df["baseColour"].fillna("").astype(str).tolist()
        usages = self.df["usage"].fillna("").astype(str).tolist()

        doc_tokens_list = []
        doc_lens = np.zeros(self.N, dtype=np.int32)
        total_tokens = 0
        doc_freqs = Counter()

        for doc_idx in range(self.N):
            en_name = names[doc_idx]
            gender_vi = GENDER_MAP.get(genders[doc_idx], genders[doc_idx])
            art_vi = ARTICLE_TYPE_MAP.get(art_types[doc_idx], art_types[doc_idx])
            col_vi = COLOUR_MAP.get(colours[doc_idx], colours[doc_idx])
            usage_vi = USAGE_MAP.get(usages[doc_idx], usages[doc_idx])

            doc_text = f"{en_name} {gender_vi} {art_vi} {col_vi} {usage_vi}"
            tokens = self._tokenize(doc_text)
            doc_len = len(tokens)
            doc_lens[doc_idx] = doc_len
            total_tokens += doc_len
            doc_tokens_list.append(tokens)

            for term in set(tokens):
                doc_freqs[term] += 1

        self.avg_doc_len = float(total_tokens) / max(1, self.N)

        # Build vocabulary & precompute IDF
        self.vocab = sorted(doc_freqs.keys())
        self.term2id = {term: idx for idx, term in enumerate(self.vocab)}
        self.V = len(self.vocab)

        self.idf = np.zeros(self.V, dtype=np.float32)
        for term, term_idx in self.term2id.items():
            df_val = doc_freqs[term]
            self.idf[term_idx] = math.log(1.0 + (self.N - df_val + 0.5) / (df_val + 0.5))

        # Build sparse matrix coordinates: row=doc_idx, col=term_idx, data=bm25_weight
        row_indices = []
        col_indices = []
        data_values = []

        for doc_idx, tokens in enumerate(doc_tokens_list):
            doc_len = doc_lens[doc_idx]
            counts = Counter(tokens)
            denom_len_factor = self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))

            for term, tf in counts.items():
                term_idx = self.term2id[term]
                idf_val = self.idf[term_idx]
                score = idf_val * (tf * (self.k1 + 1.0)) / (tf + denom_len_factor)

                row_indices.append(doc_idx)
                col_indices.append(term_idx)
                data_values.append(score)

        # Create CSR matrix (N, V)
        self.matrix = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(self.N, self.V),
            dtype=np.float32
        )

    def search(
        self,
        query: str,
        top_k: int = 200,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-fast sparse BM25 retrieval using scipy CSR dot product.
        Includes bilingual term expansion from vi_to_en_terms.json.
        """
        clean_q = remove_diacritics(query)
        tokens = self._tokenize(clean_q)

        # Bilingual expansion: check if any Vietnamese terms match
        expanded_tokens = list(tokens)
        for vi_phrase, en_trans in self.vi_to_en.items():
            # Check with word boundaries
            if re.search(rf"(?:\b|^){re.escape(vi_phrase)}(?:\b|$)", clean_q):
                for en_tok in en_trans.split():
                    expanded_tokens.append(en_tok.lower())

        # Map to vocabulary column indices
        term_cols = [self.term2id[t] for t in set(expanded_tokens) if t in self.term2id]
        if not term_cols:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Sparse vector dot product
        # Create a boolean indicator or weight vector for matched columns
        # To make it fastest, we extract submatrix columns and sum rows
        submatrix = self.matrix[:, term_cols]
        scores = np.asarray(submatrix.sum(axis=1)).flatten() # (N,)

        # Apply candidate mask if provided
        if mask is not None:
            # Only consider docs where mask == True
            cand_indices = np.where(mask)[0]
            if len(cand_indices) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            sub_scores = scores[cand_indices]
            
            # Filter non-zero scores if possible
            nz_mask = (sub_scores > 0.0)
            if np.any(nz_mask):
                cand_indices = cand_indices[nz_mask]
                sub_scores = sub_scores[nz_mask]

            if len(sub_scores) <= top_k:
                order = np.argsort(-sub_scores)
                return cand_indices[order], sub_scores[order]
            else:
                top_part = np.argpartition(-sub_scores, top_k)[:top_k]
                top_part = top_part[np.argsort(-sub_scores[top_part])]
                return cand_indices[top_part], sub_scores[top_part]
        else:
            # Global search
            nz_indices = np.where(scores > 0.0)[0]
            if len(nz_indices) == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            nz_scores = scores[nz_indices]

            if len(nz_scores) <= top_k:
                order = np.argsort(-nz_scores)
                return nz_indices[order], nz_scores[order]
            else:
                top_part = np.argpartition(-nz_scores, top_k)[:top_k]
                top_part = top_part[np.argsort(-nz_scores[top_part])]
                return nz_indices[top_part], nz_scores[top_part]


def rrf_merge(
    dense_indices: np.ndarray,
    sparse_indices: np.ndarray,
    k: int = 60,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reciprocal Rank Fusion (RRF) merging between dense and sparse rankings.
    Formula: RRF(d) = sum_{rankings} (1 / (k + rank)) with fixed k=60.
    """
    rrf_scores = defaultdict(float)

    for rank, doc_idx in enumerate(dense_indices, 1):
        rrf_scores[doc_idx] += 1.0 / (k + rank)

    for rank, doc_idx in enumerate(sparse_indices, 1):
        rrf_scores[doc_idx] += 1.0 / (k + rank)

    sorted_items = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    if not sorted_items:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    top_idxs = np.array([item[0] for item in sorted_items], dtype=np.int64)
    top_scs = np.array([item[1] for item in sorted_items], dtype=np.float32)
    return top_idxs, top_scs

if __name__ == "__main__":
    import time
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    bm25 = BM25Index()
    init_time = (time.perf_counter() - t0) * 1000.0
    print(f"CSR BM25 Index initialized in {init_time:.1f} ms for {bm25.N} items (Vocab: {bm25.V})")

    # Benchmark 200 queries
    queries = [
        "áo thun Puma nam",
        "giày thể thao Nike chạy bộ",
        "quần jean ống đứng nam màu xanh",
        "váy đầm dự tiệc nữ màu đỏ",
        "đồng hồ Casio nam dây bạc",
        "túi xách da nữ màu be",
        "ba lô laptop Wildcraft",
        "cà vạt lụa nam công sở",
        "áo sơ mi kẻ caro nam",
        "quần short thể thao xám"
    ] * 20 # 200 queries

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        _ = bm25.search(q, top_k=200)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    print(f"\nCSR BM25 Latency Benchmark across {len(queries)} queries:")
    print(f"  Mean latency : {np.mean(latencies):.2f} ms")
    print(f"  p50 latency  : {np.percentile(latencies, 50):.2f} ms")
    print(f"  p95 latency  : {np.percentile(latencies, 95):.2f} ms")
    print(f"  p99 latency  : {np.percentile(latencies, 99):.2f} ms")
