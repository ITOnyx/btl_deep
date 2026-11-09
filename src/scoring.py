"""
Scoring and Reranking Module for Phase 3c (SigLIP Visual Search).
Implements:
1. Mode 1: Raw Cosine
2. Mode 2: Mean-Centering with mu_shared and gamma (Cone Effect correction)
3. Mode 3: CSLS (Cross-Domain Similarity Local Scaling) with beta and k
Supports reranking candidate sets (e.g. top-200 from FAISS HNSW).
"""
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

class Scorer:
    def __init__(
        self,
        stats_path: str = "data/checkpoints_siglip-vit/vit_zero_train_stats.npz"
    ):
        p = Path(stats_path)
        if p.exists():
            data = np.load(str(p))
            self.mu_shared = data["mu_shared"].astype(np.float32) # (1, D)
            self.mu_img = data["mu_img"].astype(np.float32)
            self.mu_txt = data["mu_txt"].astype(np.float32)
            self.gamma = float(data["cone_restore_gamma"]) # 0.75
            self.beta = float(data["csls_beta"]) # 0.28
            self.k = int(data["csls_k"]) # 7
            self.val_query_bank = data["val_query_bank"].astype(np.float32) # (2000, D)
            # Normalize bank
            self.val_query_bank = self.val_query_bank / np.linalg.norm(self.val_query_bank, axis=-1, keepdims=True)
            if "gallery_hub_scores" in data:
                self.gallery_hub_scores = data["gallery_hub_scores"].astype(np.float32)
            else:
                self.gallery_hub_scores = None
        else:
            self.mu_shared = None
            self.gamma = 0.75
            self.beta = 0.28
            self.k = 7
            self.val_query_bank = None
            self.gallery_hub_scores = None

    def center_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Applies mean-centering: z' = (z - gamma * mu) / ||z - gamma * mu||."""
        if self.mu_shared is None:
            return vectors
        v = vectors - self.gamma * self.mu_shared
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return v / norms

    def score_mode1_cosine(self, query_vec: np.ndarray, gallery_vecs: np.ndarray) -> np.ndarray:
        """
        Mode 1: Raw Cosine similarity.
        query_vec: (1, D) or (D,)
        gallery_vecs: (N, D)
        returns: (N,) cosine similarities
        """
        q = query_vec.reshape(1, -1)
        q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
        g = gallery_vecs / np.maximum(np.linalg.norm(gallery_vecs, axis=-1, keepdims=True), 1e-12)
        return np.dot(g, q.T).flatten()

    def score_mode2_centered(self, query_vec: np.ndarray, gallery_vecs: np.ndarray) -> np.ndarray:
        """
        Mode 2: Mean-Centered Cosine similarity.
        """
        q = self.center_vectors(query_vec.reshape(1, -1))
        g = self.center_vectors(gallery_vecs)
        return np.dot(g, q.T).flatten()

    def score_mode3_csls(
        self,
        query_vec: np.ndarray,
        gallery_vecs: np.ndarray,
        candidate_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mode 3: CSLS scoring:
        CSLS(q, g) = 2 * cos(q, g) - beta * (r(q) + r(g))
        where r(q) is average similarity of q to its top-k gallery items,
        and r(g) is average similarity of g to its top-k query neighbors in val_query_bank.
        """
        raw_sims = self.score_mode1_cosine(query_vec, gallery_vecs) # (N,)
        
        # r(q): mean of top-k in candidate gallery
        k_val = min(self.k, len(raw_sims))
        topk_g = np.partition(raw_sims, -k_val)[-k_val:]
        r_q = float(np.mean(topk_g))

        # r(g): mean of top-k against val_query_bank
        if self.val_query_bank is not None:
            g = gallery_vecs / np.maximum(np.linalg.norm(gallery_vecs, axis=-1, keepdims=True), 1e-12)
            sims_g_to_bank = np.dot(g, self.val_query_bank.T) # (N, 2000)
            k_bank = min(self.k, sims_g_to_bank.shape[1])
            topk_bank = np.partition(sims_g_to_bank, -k_bank, axis=1)[:, -k_bank:]
            r_g = np.mean(topk_bank, axis=1) # (N,)
        else:
            r_g = np.zeros_like(raw_sims)

        csls_scores = 2.0 * raw_sims - self.beta * (r_q + r_g)
        return csls_scores

    def score_mode4_csls_centered(
        self,
        query_vec: np.ndarray,
        gallery_vecs: np.ndarray,
        candidate_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mode 4: CSLS applied on Mean-Centered representations.
        """
        q_c = self.center_vectors(query_vec.reshape(1, -1))
        g_c = self.center_vectors(gallery_vecs)
        raw_sims = np.dot(g_c, q_c.T).flatten()

        k_val = min(self.k, len(raw_sims))
        topk_g = np.partition(raw_sims, -k_val)[-k_val:]
        r_q = float(np.mean(topk_g))

        if self.val_query_bank is not None:
            bank_c = self.center_vectors(self.val_query_bank)
            sims_bank = np.dot(g_c, bank_c.T)
            k_bank = min(self.k, sims_bank.shape[1])
            topk_bank = np.partition(sims_bank, -k_bank, axis=1)[:, -k_bank:]
            r_g = np.mean(topk_bank, axis=1)
        else:
            r_g = np.zeros_like(raw_sims)

        csls_scores = 2.0 * raw_sims - self.beta * (r_q + r_g)
        return csls_scores

    def rerank_candidates(
        self,
        query_vec: np.ndarray,
        candidate_indices: np.ndarray,
        full_gallery_embeddings: np.ndarray,
        mode: str = "cosine"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reranks a list of candidate indices (e.g. top-200 from HNSW).
        mode: 'cosine' | 'centered' | 'csls' | 'csls_centered'
        returns: (sorted_indices, sorted_scores)
        """
        cand_vecs = full_gallery_embeddings[candidate_indices]
        if mode == "cosine":
            scores = self.score_mode1_cosine(query_vec, cand_vecs)
        elif mode == "centered":
            scores = self.score_mode2_centered(query_vec, cand_vecs)
        elif mode == "csls":
            scores = self.score_mode3_csls(query_vec, cand_vecs, candidate_indices)
        elif mode in ("csls_centered", "csls_center"):
            scores = self.score_mode4_csls_centered(query_vec, cand_vecs, candidate_indices)
        else:
            raise ValueError(f"Unknown scoring mode: {mode}")

        order = np.argsort(-scores)
        return candidate_indices[order], scores[order]
