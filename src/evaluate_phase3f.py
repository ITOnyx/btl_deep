"""
Comprehensive Phase 3f Test Evaluation Script (src/evaluate_phase3f.py)
Audit & Reconciliation Evaluation for SigLIP Visual Search (Run 2, after bug fix).
Strictly inference-only. Uses max_length=128 and GPU-accelerated FP16 tensors.
Covers:
- Q1b: 1,290 paired captions (Configs 0, 1, 2) on G_test & G_full + F2 Pool Decomposition
- Q2: 61 attribute queries (Configs 0..5 on G_test & G_full) [Label: Parser/Filter Synthetic]
- Q3: 100 handwritten dev queries (Configs 0..5 on G_test & G_full) [Label: Development Set]
- Q5: 150 brand queries (Configs 0..5 on G_test & G_full) [Label: Brand Parser/Filter]
- Heldout: 30 independent test queries (Configs 0..5 on G_test & G_full) [Label: Heldout Blind Test]
  + Generates F5 Blind A/B contact sheets and eval/ab_blind_heldout.csv (mapping in eval/ab_mapping.json)
- Q6-test: 120 independent usage/season queries (Configs 0, 1, 2, 3, 5 on G_test) with frozen gamma=0.3, alpha=1.0
- Q4: 1,290 visual queries (self excluded, raw, flip-aug, kNN-voting) on G_test & G_full
- Diagnostics: 20 failed cases + F6 Brand Deep Dive table for 8 brand queries
Saves all metrics to eval/results_3f.json.
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torchvision.transforms as T

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_siglip import load_siglip_inference
from src.scoring import Scorer
from src.attr_index import AttributeIndex
from src.query_parser import QueryParser
from src.lexical import BM25Index, rrf_merge

# -------------------------------------------------------------
# Metric Helpers
# -------------------------------------------------------------
def compute_ap_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    R = len(relevant_ids)
    if R == 0:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    norm_denom = min(R, k)
    return sum_precisions / norm_denom if norm_denom > 0 else 0.0

def compute_precision_and_hit(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> Tuple[float, float]:
    if not retrieved_ids:
        return 0.0, 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / k, (1.0 if hits > 0 else 0.0)

def compute_p_min_r(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    R = len(relevant_ids)
    if R == 0:
        return 0.0
    k_min = min(R, k)
    hits = sum(1 for item_id in retrieved_ids[:k_min] if item_id in relevant_ids)
    return hits / k_min

def bootstrap_ci(scores: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(scores))
    boot_means = [float(np.mean(rng.choice(scores, size=n, replace=True))) for _ in range(n_bootstraps)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper

def compute_exact_rank(score_matrix: np.ndarray, target_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    N = score_matrix.shape[0]
    ranks = np.zeros(N, dtype=np.int32)
    hits_1 = np.zeros(N, dtype=np.float32)
    hits_5 = np.zeros(N, dtype=np.float32)
    hits_10 = np.zeros(N, dtype=np.float32)
    ties_count = 0

    for i in range(N):
        row = score_matrix[i]
        tgt_idx = target_indices[i]
        tgt_score = row[tgt_idx]
        strictly_greater = np.sum(row > tgt_score)
        equal_score = np.sum(np.isclose(row, tgt_score, atol=1e-6))
        if equal_score > 1:
            ties_count += 1
        rank = int(strictly_greater) + 1
        ranks[i] = rank
        if rank <= 1: hits_1[i] = 1.0
        if rank <= 5: hits_5[i] = 1.0
        if rank <= 10: hits_10[i] = 1.0

    return ranks, hits_1, hits_5, hits_10, ties_count


class Phase3fEvaluator:
    def __init__(self, device: str = "cuda"):
        print("==========================================================", flush=True)
        print("Initializing Phase 3f Evaluator (Run 2, after bug fix)...", flush=True)
        print("==========================================================", flush=True)

        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        print(f"Hardware execution device: {self.device}", flush=True)

        # 1. Load Attribute Index
        print("  Loading Attribute Index...", flush=True)
        self.attr_idx = AttributeIndex()
        self.df = self.attr_idx.df
        self.N = self.attr_idx.N
        self.item_ids = [str(x) for x in self.attr_idx.item_ids]
        self.id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        # Precompute string numpy arrays for vectorized attribute matching
        self.df_art = self.df["articleType"].fillna("").astype(str).values
        self.df_gen = self.df["gender"].fillna("").astype(str).values
        self.df_col = self.df["baseColour"].fillna("").astype(str).values
        self.df_brand = self.df["brand"].fillna("").astype(str).values
        self.df_brand_lower = np.array([b.lower() for b in self.df_brand])
        self.item_ids_arr = np.array(self.item_ids)

        # 2. Load BM25 Index
        print("  Loading BM25 CSR Index...", flush=True)
        self.bm25 = BM25Index()

        # 3. Load Query Parser
        print("  Loading Query Parser...", flush=True)
        self.parser = QueryParser()

        # 4. Load Scorer
        print("  Loading Scorer...", flush=True)
        self.scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")

        # 5. Load Embeddings
        print("  Loading Gallery Embeddings (Image & Text)...", flush=True)
        self.emb_img_raw = np.load("index_v2/image_embeddings_v2.npy")
        self.emb_txt_raw = np.load("index_v2/text_embeddings_v2.npy")

        # Center embeddings
        print("  Mean-centering embeddings...", flush=True)
        self.emb_img_c = self.scorer.center_vectors(self.emb_img_raw)
        self.emb_txt_c = self.scorer.center_vectors(self.emb_txt_raw)

        # GPU cache
        if self.device == "cuda":
            print("  Caching FP16 embedding matrices on GPU...", flush=True)
            self.t_emb_img_c = torch.tensor(self.emb_img_c, dtype=torch.float16, device=self.device)
            self.t_emb_txt_c = torch.tensor(self.emb_txt_c, dtype=torch.float16, device=self.device)
        else:
            self.t_emb_img_c = None
            self.t_emb_txt_c = None

        # 6. Load Testset and Gallery Splits
        with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
            self.testset = json.load(f)

        self.g_test_ids = [str(x) for x in self.testset["g_test_ids"]]
        self.g_test_indices = np.array([self.id_to_idx[tid] for tid in self.g_test_ids if tid in self.id_to_idx])
        self.g_test_mask = np.zeros(self.N, dtype=bool)
        self.g_test_mask[self.g_test_indices] = True

        self.g_full_ids = self.item_ids
        self.g_full_indices = np.arange(self.N)
        self.g_full_mask = np.ones(self.N, dtype=bool)

        print(f"  Galleries: G_test={len(self.g_test_indices):,}, G_full={len(self.g_full_indices):,}", flush=True)

        # 7. Load Families
        with open("eval/families.json", "r", encoding="utf-8") as f:
            self.families = json.load(f)
        self.gender_compat = self.families.get("gender_compatibility", {})
        self.art_families = self.families.get("article_type_families", {})
        self.col_families = self.families.get("colour_families", {})

        # 8. Load Model
        print(f"  Loading SigLIP model on {self.device} (FP16)...", flush=True)
        self.model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=self.device, fp16=(self.device=="cuda"))
        self.processor = self.model.processor

        # 9. Frozen hyperparameters from Q6 validation grid
        self.gamma = 0.3
        self.alpha = 1.0  # Selected alpha*=1.0 to save 170MB RAM & eliminate text fusion latency
        self.rrf_k = 60

        print("Phase 3f Evaluator successfully initialized.\n", flush=True)

    def encode_text_queries(self, queries: List[str], batch_size: int = 128) -> np.ndarray:
        """
        Encodes text queries with max_length=128 (matches SigLIP text tower training length).
        """
        embs_list = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                b_texts = queries[i:i+batch_size]
                inp = self.processor(text=b_texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
                feats = self.model.encode_text(inp["input_ids"].to(self.device), None)
                embs_list.append(feats.cpu().numpy())
        embs = np.vstack(embs_list)
        norms = np.maximum(np.linalg.norm(embs, axis=-1, keepdims=True), 1e-12)
        embs = embs / norms
        return self.scorer.center_vectors(embs)

    def search_text_all_configs(
        self,
        query_text: str,
        q_vec_c: np.ndarray,
        g_mask: np.ndarray,
        g_indices: np.ndarray,
        g_ids: List[str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """
        Executes all 6 configurations for a given query vector and string on the specified gallery.
        """
        parsed = self.parser.parse(query_text)
        if "usage" in parsed: parsed["usage"] = None
        if "season" in parsed: parsed["season"] = None

        if self.device == "cuda" and self.t_emb_img_c is not None:
            g_idx_t = torch.tensor(g_indices, dtype=torch.long, device=self.device)
            q_t = torch.tensor(q_vec_c, dtype=torch.float16, device=self.device)
            sims_img_t = torch.matmul(self.t_emb_img_c[g_idx_t], q_t)
            sims_txt_t = torch.matmul(self.t_emb_txt_c[g_idx_t], q_t)
            sims_img = sims_img_t.float().cpu().numpy()
            sims_txt = sims_txt_t.float().cpu().numpy()
        else:
            sims_img = np.dot(self.emb_img_c[g_indices], q_vec_c)
            sims_txt = np.dot(self.emb_txt_c[g_indices], q_vec_c)

        # Config (0): Model Thuan (Dense Baseline)
        top10_c0_local = np.argsort(-sims_img)[:10]
        retrieved_c0 = [g_ids[i] for i in top10_c0_local]

        # Config (1): Hard Filtering
        c1_cand_idxs = self.attr_idx.filter_exact(parsed, mask=g_mask)
        if len(c1_cand_idxs) == 0:
            c1_fallback = True
            retrieved_c1 = retrieved_c0
        else:
            c1_fallback = False
            sims_c1 = np.dot(self.emb_img_c[c1_cand_idxs], q_vec_c)
            top10_c1_local = np.argsort(-sims_c1)[:10]
            retrieved_c1 = [self.item_ids[c1_cand_idxs[i]] for i in top10_c1_local]

        # Config (2): Soft Attribute Scoring (gamma = 0.3)
        bonus_g = self.attr_idx.compute_attribute_soft_bonus(
            parsed, g_indices,
            gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
        )
        scores_c2 = sims_img + bonus_g
        top10_c2_local = np.argsort(-scores_c2)[:10]
        retrieved_c2 = [g_ids[i] for i in top10_c2_local]

        # Config (3): Product Text Fusion (with alpha=1.0, equals dense image score)
        scores_c3 = self.alpha * sims_img + (1.0 - self.alpha) * sims_txt
        top10_c3_local = np.argsort(-scores_c3)[:10]
        retrieved_c3 = [g_ids[i] for i in top10_c3_local]

        # Config (4): Hybrid Dense (0) + BM25 RRF (k=60)
        dense_top100_c4 = [g_indices[i] for i in np.argsort(-sims_img)[:100]]
        sparse_top100_c4, _ = self.bm25.search(query_text, top_k=100, mask=g_mask)
        merged_c4, _ = rrf_merge(dense_top100_c4, sparse_top100_c4, k=self.rrf_k, top_k=10)
        retrieved_c4 = [self.item_ids[i] for i in merged_c4]

        # Config (5): Full System (Relaxation L0..L6 + Soft Bonus + BM25 RRF)
        c5_cand_idxs, tier = self.attr_idx.filter_relaxed(parsed, min_candidates=20, mask=g_mask)
        sims_cand_img = np.dot(self.emb_img_c[c5_cand_idxs], q_vec_c)
        bonus_cand = self.attr_idx.compute_attribute_soft_bonus(
            parsed, c5_cand_idxs,
            gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
        )
        scores_c5_dense = sims_cand_img + bonus_cand
        dense_top100_c5 = c5_cand_idxs[np.argsort(-scores_c5_dense)[:100]]

        c5_cand_mask = np.zeros(self.N, dtype=bool)
        c5_cand_mask[c5_cand_idxs] = True
        sparse_top100_c5, _ = self.bm25.search(query_text, top_k=100, mask=c5_cand_mask)

        merged_c5, _ = rrf_merge(dense_top100_c5, sparse_top100_c5, k=self.rrf_k, top_k=10)
        retrieved_c5 = [self.item_ids[i] for i in merged_c5]

        configs = {
            "config_0": retrieved_c0,
            "config_1": retrieved_c1,
            "config_2": retrieved_c2,
            "config_3": retrieved_c3,
            "config_4": retrieved_c4,
            "config_5": retrieved_c5
        }
        diagnostics = {
            "parsed": parsed,
            "c1_fallback": c1_fallback,
            "c1_cand_count": len(c1_cand_idxs),
            "c5_tier": tier,
            "c5_cand_count": len(c5_cand_idxs)
        }
        return configs, diagnostics

    # -------------------------------------------------------------
    # Q1b: Paired Caption -> Image (1,290 items) + F2 Pool Analysis
    # -------------------------------------------------------------
    def run_q1b(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Evaluating Q1b: Paired Caption -> Image (1,290 items)...", flush=True)
        print("==========================================================", flush=True)

        with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}

        q1_texts = [meta_dict[tid] for tid in self.g_test_ids]
        print(f"Loaded {len(q1_texts)} train-format captions for Q1b.", flush=True)

        print("Encoding Q1b query texts (max_length=128)...", flush=True)
        q1_vecs_c = self.encode_text_queries(q1_texts)

        results_q1b = {}
        for g_name, g_mask, g_indices, g_ids in [
            ("G_test", self.g_test_mask, self.g_test_indices, self.g_test_ids),
            ("G_full", self.g_full_mask, self.g_full_indices, self.g_full_ids)
        ]:
            print(f"\n--- Q1b on {g_name} (N={len(g_ids):,}) ---", flush=True)
            results_q1b[g_name] = {}
            target_indices_in_g = [g_ids.index(tid) for tid in self.g_test_ids]

            # 1. Config (0): Model Thuan (Dense Baseline)
            sims_c0 = np.dot(q1_vecs_c, self.emb_img_c[g_indices].T)
            r_c0, h1_c0, h5_c0, h10_c0, ties_c0 = compute_exact_rank(sims_c0, target_indices_in_g)
            m_c0, c0_l, c0_u = bootstrap_ci(h10_c0)
            results_q1b[g_name]["config_0_model_thuan"] = {
                "recall_at_1": float(np.mean(h1_c0)),
                "recall_at_5": float(np.mean(h5_c0)),
                "recall_at_10": float(np.mean(h10_c0)),
                "recall_at_10_ci95": [c0_l, c0_u],
                "mrr": float(np.mean(1.0 / r_c0)),
                "median_rank": float(np.median(r_c0)),
                "mean_rank": float(np.mean(r_c0)),
                "ties_count": ties_c0
            }
            print(f"  Config (0) Model Thuan : R@10={np.mean(h10_c0):.4f} [{c0_l:.4f}, {c0_u:.4f}], MRR={np.mean(1.0/r_c0):.4f}, MedRank={np.median(r_c0):.1f}")

            # 2. Config (1): Hard Filtering + Pool Decomposition (F2)
            c1_ranks = []
            c1_hits_1 = []
            c1_hits_5 = []
            c1_hits_10 = []
            c1_fallbacks = 0
            c1_pool_sizes = []
            c1_random_expectations = []

            for i in range(len(self.g_test_ids)):
                target_iid = self.g_test_ids[i]
                p = self.parser.parse(q1_texts[i])
                cand_idxs = self.attr_idx.filter_exact(p, mask=g_mask)
                pool_size = len(cand_idxs) if len(cand_idxs) > 0 else len(g_indices)
                c1_pool_sizes.append(pool_size)
                c1_random_expectations.append(min(1.0, 10.0 / pool_size))

                if len(cand_idxs) == 0:
                    c1_fallbacks += 1
                    r = r_c0[i]
                else:
                    target_full_idx = self.id_to_idx[target_iid]
                    sims_cand = np.dot(self.emb_img_c[cand_idxs], q1_vecs_c[i])
                    if target_full_idx in cand_idxs:
                        pos = np.where(cand_idxs == target_full_idx)[0][0]
                        tgt_score = sims_cand[pos]
                        r = int(np.sum(sims_cand > tgt_score)) + 1
                    else:
                        r = len(cand_idxs) + 1 + int(np.sum(sims_c0[i] > sims_c0[i, target_indices_in_g[i]]))

                c1_ranks.append(r)
                c1_hits_1.append(1.0 if r <= 1 else 0.0)
                c1_hits_5.append(1.0 if r <= 5 else 0.0)
                c1_hits_10.append(1.0 if r <= 10 else 0.0)

            c1_ranks = np.array(c1_ranks)
            c1_hits_10 = np.array(c1_hits_10)
            m_c1, c1_l, c1_u = bootstrap_ci(c1_hits_10)

            c1_actual_r10 = float(np.mean(c1_hits_10))
            c1_rand_r10 = float(np.mean(c1_random_expectations))
            c1_true_contrib = c1_actual_r10 - c1_rand_r10

            results_q1b[g_name]["config_1_hard_filter"] = {
                "recall_at_1": float(np.mean(c1_hits_1)),
                "recall_at_5": float(np.mean(c1_hits_5)),
                "recall_at_10": c1_actual_r10,
                "recall_at_10_ci95": [c1_l, c1_u],
                "mrr": float(np.mean(1.0 / c1_ranks)),
                "median_rank": float(np.median(c1_ranks)),
                "mean_rank": float(np.mean(c1_ranks)),
                "fallback_count": c1_fallbacks,
                "pool_analysis": {
                    "pool_min": int(np.min(c1_pool_sizes)),
                    "pool_median": float(np.median(c1_pool_sizes)),
                    "pool_p95": float(np.percentile(c1_pool_sizes, 95)),
                    "pool_max": int(np.max(c1_pool_sizes)),
                    "random_baseline_r10": round(c1_rand_r10, 4),
                    "true_embedding_contribution": round(c1_true_contrib, 4)
                }
            }
            print(f"  Config (1) Hard Filter : R@10={c1_actual_r10:.4f} [{c1_l:.4f}, {c1_u:.4f}], MedRank={np.median(c1_ranks):.1f}")
            print(f"    -> Pool: median={np.median(c1_pool_sizes):.0f}, p95={np.percentile(c1_pool_sizes, 95):.0f} | RandBaseline={c1_rand_r10:.4f} | TrueContrib={c1_true_contrib:+.4f}")

            # 3. Config (2): Soft Attribute Scoring (gamma = 0.3)
            bonuses_q1 = []
            for i in range(len(self.g_test_ids)):
                p = self.parser.parse(q1_texts[i])
                b = self.attr_idx.compute_attribute_soft_bonus(
                    p, g_indices,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                bonuses_q1.append(b)
            bonuses_q1 = np.array(bonuses_q1)

            scores_c2 = sims_c0 + bonuses_q1
            r_c2, h1_c2, h5_c2, h10_c2, ties_c2 = compute_exact_rank(scores_c2, target_indices_in_g)
            m_c2, c2_l, c2_u = bootstrap_ci(h10_c2)

            c2_actual_r10 = float(np.mean(h10_c2))
            c2_rand_r10 = float(min(1.0, 10.0 / len(g_indices)))
            c2_true_contrib = c2_actual_r10 - c2_rand_r10

            results_q1b[g_name]["config_2_soft_scoring"] = {
                "recall_at_1": float(np.mean(h1_c2)),
                "recall_at_5": float(np.mean(h5_c2)),
                "recall_at_10": c2_actual_r10,
                "recall_at_10_ci95": [c2_l, c2_u],
                "mrr": float(np.mean(1.0 / r_c2)),
                "median_rank": float(np.median(r_c2)),
                "mean_rank": float(np.mean(r_c2)),
                "ties_count": ties_c2,
                "pool_analysis": {
                    "pool_size": len(g_indices),
                    "random_baseline_r10": round(c2_rand_r10, 4),
                    "true_embedding_contribution": round(c2_true_contrib, 4)
                }
            }
            print(f"  Config (2) Soft Scoring: R@10={c2_actual_r10:.4f} [{c2_l:.4f}, {c2_u:.4f}], MedRank={np.median(r_c2):.1f}")
            print(f"    -> Pool: {len(g_indices)} | RandBaseline={c2_rand_r10:.4f} | TrueContrib={c2_true_contrib:+.4f}")

        return results_q1b

    # -------------------------------------------------------------
    # Generic Evaluation Loop for Multi-Item Attribute Queries
    # -------------------------------------------------------------
    def _evaluate_query_list(
        self,
        query_items: List[Dict[str, Any]],
        get_gt_fn,
        q_label: str
    ) -> Dict[str, Any]:
        print(f"\n==========================================================", flush=True)
        print(f"Evaluating {q_label} ({len(query_items)} queries)...", flush=True)
        print(f"==========================================================", flush=True)

        q_texts = [q.get("query_text", q.get("query", "")) for q in query_items]
        q_vecs_c = self.encode_text_queries(q_texts)

        results = {}
        for g_name, g_mask, g_indices, g_ids in [
            ("G_test", self.g_test_mask, self.g_test_indices, self.g_test_ids),
            ("G_full", self.g_full_mask, self.g_full_indices, self.g_full_ids)
        ]:
            results[g_name] = {}
            cfg_names = [f"config_{i}" for i in range(6)]
            p10_dict = {cfg: [] for cfg in cfg_names}
            hit10_dict = {cfg: [] for cfg in cfg_names}
            ap10_dict = {cfg: [] for cfg in cfg_names}
            pmin_dict = {cfg: [] for cfg in cfg_names}

            c1_fallbacks = 0
            tier_counts = Counter()
            valid_queries_count = 0

            for q_idx in range(len(query_items)):
                q_obj = query_items[q_idx]
                q_text = q_texts[q_idx]
                q_vec_c = q_vecs_c[q_idx]

                gt = get_gt_fn(q_obj, g_name, g_ids)
                rel_strict = gt.get("strict", set()) if isinstance(gt, dict) else gt

                if len(rel_strict) == 0:
                    continue
                valid_queries_count += 1

                configs, diag = self.search_text_all_configs(q_text, q_vec_c, g_mask, g_indices, g_ids)
                if diag["c1_fallback"]:
                    c1_fallbacks += 1
                tier_counts[diag["c5_tier"]] += 1

                for cfg in cfg_names:
                    top10 = configs[cfg]
                    p, h = compute_precision_and_hit(top10, rel_strict, k=10)
                    ap = compute_ap_at_k(top10, rel_strict, k=10)
                    pmin = compute_p_min_r(top10, rel_strict, k=10)
                    p10_dict[cfg].append(p)
                    hit10_dict[cfg].append(h)
                    ap10_dict[cfg].append(ap)
                    pmin_dict[cfg].append(pmin)

            results[g_name]["metadata"] = {
                "total_queries": len(query_items),
                "evaluated_queries": valid_queries_count,
                "c1_fallback_count": c1_fallbacks,
                "c5_tier_counts": dict(tier_counts)
            }

            for cfg in cfg_names:
                m_p, p_l, p_u = bootstrap_ci(np.array(p10_dict[cfg])) if p10_dict[cfg] else (0,0,0)
                m_ap, ap_l, ap_u = bootstrap_ci(np.array(ap10_dict[cfg])) if ap10_dict[cfg] else (0,0,0)
                cfg_res = {
                    "precision_at_10": float(np.mean(p10_dict[cfg])) if p10_dict[cfg] else 0.0,
                    "precision_at_10_ci95": [p_l, p_u],
                    "hit_at_10": float(np.mean(hit10_dict[cfg])) if hit10_dict[cfg] else 0.0,
                    "ap_at_10": float(np.mean(ap10_dict[cfg])) if ap10_dict[cfg] else 0.0,
                    "ap_at_10_ci95": [ap_l, ap_u],
                    "precision_at_min_r_10": float(np.mean(pmin_dict[cfg])) if pmin_dict[cfg] else 0.0
                }
                results[g_name][cfg] = cfg_res

                tag = "MODEL THUAN (0)" if cfg == "config_0" else f"SYSTEM ({cfg[-1]})"
                print(f"  {tag:<18}: P@10={cfg_res['precision_at_10']:.4f}, Hit@10={cfg_res['hit_at_10']:.4f}, AP@10={cfg_res['ap_at_10']:.4f}, P@min(R,10)={cfg_res['precision_at_min_r_10']:.4f}")

        return results

    # -------------------------------------------------------------
    # Q2: 61 Attribute Queries
    # -------------------------------------------------------------
    def run_q2(self) -> Dict[str, Any]:
        q2_queries = self.testset["q2_attribute_queries"]
        def get_gt_q2(q, g_name, g_ids):
            sub_key = "g_test" if g_name == "G_test" else "g_full"
            if "target_ids_strict" in q and isinstance(q["target_ids_strict"], dict) and sub_key in q["target_ids_strict"]:
                return set(str(x) for x in q["target_ids_strict"][sub_key])

            tgt = q.get("target_attributes", {})
            art = tgt.get("articleType")
            gen = tgt.get("gender")
            col = tgt.get("baseColour")
            compat_gens = set(self.gender_compat.get(gen, [gen, "Unisex"])) if gen else None

            strict_rel = set()
            for tid in g_ids:
                row = self.df.iloc[self.id_to_idx[tid]]
                i_art, i_gen, i_col = row["articleType"], row["gender"], row["baseColour"]
                art_ok = (i_art == art) if art else True
                gen_ok = (i_gen in compat_gens) if compat_gens else True
                col_ok = (i_col == col) if col else True
                if art_ok and gen_ok and col_ok:
                    strict_rel.add(tid)
            return strict_rel

        return self._evaluate_query_list(q2_queries, get_gt_q2, "Q2_Attribute_Queries [Kiểm tra parser/lọc synthetic]")

    # -------------------------------------------------------------
    # Q3: 100 Handwritten Dev Queries
    # -------------------------------------------------------------
    def run_q3(self) -> Dict[str, Any]:
        with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
            q3_queries = json.load(f)

        def get_gt_q3(q, g_name, g_ids):
            spec = q.get("spec", {})
            spec_arts = spec.get("articleType") or []
            spec_gen = spec.get("gender")
            spec_cols = spec.get("colour") or spec.get("baseColour") or []
            compat_gens = self.gender_compat.get(spec_gen, [spec_gen]) if spec_gen else None

            g_mask = self.g_test_mask if g_name == "G_test" else self.g_full_mask
            m_art = np.isin(self.df_art, spec_arts) if spec_arts else True
            m_gen = np.isin(self.df_gen, compat_gens) if compat_gens else True
            m_col = np.isin(self.df_col, spec_cols) if spec_cols else True
            strict_mask = m_art & m_gen & m_col & g_mask
            return set(self.item_ids_arr[strict_mask])

        return self._evaluate_query_list(q3_queries, get_gt_q3, "Q3_Handwritten_Dev_Queries [Tập phát triển, kiểm tra parser/lọc]")

    # -------------------------------------------------------------
    # Q5: 150 Brand Queries
    # -------------------------------------------------------------
    def run_q5(self) -> Dict[str, Any]:
        with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
            q5_data = json.load(f)
        q5_queries = q5_data["queries"]

        def get_gt_q5(q, g_name, g_ids):
            if g_name == "G_test":
                return set(q.get("relevant_ids_test", []))
            else:
                return set(q.get("relevant_ids_full", []))

        return self._evaluate_query_list(q5_queries, get_gt_q5, "Q5_Brand_Queries [Kiểm tra parser/lọc thương hiệu]")

    # -------------------------------------------------------------
    # Heldout: 30 Independent Queries + F5 Blind A/B Generation
    # -------------------------------------------------------------
    def run_heldout(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Evaluating Heldout Testset (30 Independent Queries) [Kiểm định độc lập Blind Test]...", flush=True)
        print("==========================================================", flush=True)

        with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
            heldout_data = json.load(f)

        spec_queries = [q for q in heldout_data if q.get("spec") is not None]
        vague_queries = [q for q in heldout_data if q.get("spec") is None]
        print(f"Loaded {len(spec_queries)} spec queries and {len(vague_queries)} vague queries.", flush=True)

        # 1. Evaluate 26 Spec Queries
        q_texts = [q["query"] for q in spec_queries]
        q_vecs_c = self.encode_text_queries(q_texts)

        results_spec = {}
        for g_name, g_mask, g_indices, g_ids in [
            ("G_test", self.g_test_mask, self.g_test_indices, self.g_test_ids),
            ("G_full", self.g_full_mask, self.g_full_indices, self.g_full_ids)
        ]:
            print(f"\n--- Heldout (26 Spec) on {g_name} (N={len(g_ids):,}) ---", flush=True)
            results_spec[g_name] = {"overall": {}, "by_group": {}}

            cfg_names = [f"config_{i}" for i in range(6)]
            query_metrics = {cfg: [] for cfg in cfg_names}
            group_metrics = defaultdict(lambda: {cfg: [] for cfg in cfg_names})

            for q_idx, q_obj in enumerate(spec_queries):
                q_text = q_obj["query"]
                q_group = q_obj["group"]
                q_vec_c = q_vecs_c[q_idx]

                spec = q_obj["spec"]
                spec_arts = spec.get("articleType") or []
                spec_gen = spec.get("gender")
                spec_cols = spec.get("colour") or spec.get("baseColour") or []
                spec_brand = spec.get("brand")
                compat_gens = self.gender_compat.get(spec_gen, [spec_gen]) if spec_gen else None

                m_art = np.isin(self.df_art, spec_arts) if spec_arts else True
                m_gen = np.isin(self.df_gen, compat_gens) if compat_gens else True
                m_col = np.isin(self.df_col, spec_cols) if spec_cols else True
                m_brand = (self.df_brand_lower == spec_brand.lower()) if spec_brand else True
                strict_mask = m_art & m_gen & m_col & m_brand & g_mask
                rel_strict = set(self.item_ids_arr[strict_mask])

                configs, diag = self.search_text_all_configs(q_text, q_vec_c, g_mask, g_indices, g_ids)

                for cfg in cfg_names:
                    top10 = configs[cfg]
                    p, h = compute_precision_and_hit(top10, rel_strict, k=10)
                    ap = compute_ap_at_k(top10, rel_strict, k=10)
                    pmin = compute_p_min_r(top10, rel_strict, k=10)

                    m_dict = {"p10": p, "hit10": h, "ap10": ap, "pmin": pmin, "rel_count": len(rel_strict)}
                    query_metrics[cfg].append(m_dict)
                    group_metrics[q_group][cfg].append(m_dict)

            # Summarize overall
            for cfg in cfg_names:
                p_arr = np.array([m["p10"] for m in query_metrics[cfg]])
                ap_arr = np.array([m["ap10"] for m in query_metrics[cfg]])
                hit_arr = np.array([m["hit10"] for m in query_metrics[cfg]])
                pmin_arr = np.array([m["pmin"] for m in query_metrics[cfg]])
                m_p, p_l, p_u = bootstrap_ci(p_arr)
                m_ap, ap_l, ap_u = bootstrap_ci(ap_arr)

                results_spec[g_name]["overall"][cfg] = {
                    "precision_at_10": float(np.mean(p_arr)),
                    "precision_at_10_ci95": [p_l, p_u],
                    "hit_at_10": float(np.mean(hit_arr)),
                    "ap_at_10": float(np.mean(ap_arr)),
                    "ap_at_10_ci95": [ap_l, ap_u],
                    "precision_at_min_r_10": float(np.mean(pmin_arr))
                }
                tag = "MODEL THUAN (0)" if cfg == "config_0" else f"SYSTEM ({cfg[-1]})"
                print(f"  {tag:<18}: P@10={float(np.mean(p_arr)):.4f}, Hit@10={float(np.mean(hit_arr)):.4f}, AP@10={float(np.mean(ap_arr)):.4f}, P@min(R,10)={float(np.mean(pmin_arr)):.4f}")

            # By group
            for grp, grp_cfgs in group_metrics.items():
                results_spec[g_name]["by_group"][grp] = {}
                for cfg in cfg_names:
                    p_arr = np.array([m["p10"] for m in grp_cfgs[cfg]])
                    ap_arr = np.array([m["ap10"] for m in grp_cfgs[cfg]])
                    results_spec[g_name]["by_group"][grp][cfg] = {
                        "n_queries": len(p_arr),
                        "precision_at_10": float(np.mean(p_arr)),
                        "ap_at_10": float(np.mean(ap_arr))
                    }

        # 2. Generate F5 Blind A/B Contact Sheets and Mapping for ALL 30 Heldout Queries
        print("\n--- Generating F5 Blind A/B Contact Sheets for 30 Heldout Queries ---", flush=True)
        all_30_texts = [q["query"] for q in heldout_data]
        all_30_vecs_c = self.encode_text_queries(all_30_texts)

        rng = np.random.default_rng(seed=42)
        ab_mapping = {}
        blind_csv_rows = []

        contact_dir = Path("reports/contact_sheets")
        contact_dir.mkdir(parents=True, exist_ok=True)

        for i, q_obj in enumerate(heldout_data):
            qid = q_obj["id"]
            q_text = q_obj["query"]
            q_vec_c = all_30_vecs_c[i]

            configs, diag = self.search_text_all_configs(
                q_text, q_vec_c, self.g_full_mask, self.g_full_indices, self.g_full_ids
            )
            top10_c0 = configs["config_0"]
            top10_c5 = configs["config_5"]

            # Randomize whether Row A is Config 0 or Config 5
            is_c0_first = bool(rng.choice([True, False]))
            row_a_cfg = "config_0" if is_c0_first else "config_5"
            row_b_cfg = "config_5" if is_c0_first else "config_0"

            row_a_ids = top10_c0 if is_c0_first else top10_c5
            row_b_ids = top10_c5 if is_c0_first else top10_c0

            ab_mapping[qid] = {
                "query": q_text,
                "row_A": row_a_cfg,
                "row_B": row_b_cfg
            }

            blind_csv_rows.append({
                "id": qid,
                "query": q_text,
                "better": "", # A / B / equal
                "correct_count_A": "", # 0-10
                "correct_count_B": "", # 0-10
                "notes": ""
            })

            # Render 2-row contact sheet labeled A and B
            sheet_path = contact_dir / f"blind_ab_{qid}.png"
            self._render_blind_ab_contact_sheet(
                qid=qid,
                query_text=q_text,
                row_a_ids=row_a_ids,
                row_b_ids=row_b_ids,
                out_path=str(sheet_path)
            )

        # Save secret mapping to eval/ab_mapping.json
        map_path = Path("eval/ab_mapping.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(ab_mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved secret A/B mapping to {map_path} (NOT disclosed in chat/report)", flush=True)

        # Save CSV template to eval/ab_blind_heldout.csv
        csv_path = Path("eval/ab_blind_heldout.csv")
        pd.DataFrame(blind_csv_rows).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved Blind A/B evaluation template to {csv_path}", flush=True)

        return {
            "spec_queries_evaluation": results_spec
        }

    def _render_blind_ab_contact_sheet(
        self,
        qid: str,
        query_text: str,
        row_a_ids: List[str],
        row_b_ids: List[str],
        out_path: str
    ):
        thumb_size = (140, 180)
        padding = 10
        header_height = 60
        row_label_width = 100
        total_width = row_label_width + 10 * (thumb_size[0] + padding) + padding
        total_height = header_height + 2 * (thumb_size[1] + padding) + padding

        sheet = Image.new("RGB", (total_width, total_height), color=(245, 245, 245))
        draw = ImageDraw.Draw(sheet)

        # Draw header text
        header_text = f"Query {qid}: \"{query_text}\""
        draw.text((padding, 15), header_text, fill=(20, 20, 20))

        # Rows A and B
        rows_data = [("A", row_a_ids), ("B", row_b_ids)]
        for row_idx, (row_label, id_list) in enumerate(rows_data):
            y_base = header_height + row_idx * (thumb_size[1] + padding) + padding
            # Draw row label
            draw.text((padding + 10, y_base + 70), f"Row {row_label}", fill=(0, 50, 150))

            for col_idx, item_id in enumerate(id_list):
                x_base = row_label_width + col_idx * (thumb_size[0] + padding) + padding
                img_path = str(self.df.iloc[self.id_to_idx[item_id]]["path"])
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as im:
                            im_rgb = im.convert("RGB")
                            im_thumb = ImageOps.fit(im_rgb, thumb_size, Image.Resampling.BILINEAR)
                            sheet.paste(im_thumb, (x_base, y_base))
                    except Exception:
                        pass
        sheet.save(out_path)

    # -------------------------------------------------------------
    # Q6-test: Independent Attribute Relevance (120 queries)
    # -------------------------------------------------------------
    def run_q6_test(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Evaluating Q6-test: Independent Attribute Relevance (120 queries)...", flush=True)
        print("Frozen parameters: gamma*=0.3, alpha*=1.0. Gallery: G_test (N=1,290)", flush=True)
        print("==========================================================", flush=True)

        with open("eval/q6_queries.json", "r", encoding="utf-8") as f:
            q6_data = json.load(f)
        test_queries = q6_data["test_queries"]

        q_texts = [q["query_text"] for q in test_queries]
        q_vecs_c = self.encode_text_queries(q_texts)

        results = {}
        for cfg_id in [0, 1, 2, 3, 5]:
            cfg_key = f"config_{cfg_id}"
            p10_list = []
            hit10_list = []
            ap10_list = []
            pmr_list = []

            for q_idx, q in enumerate(test_queries):
                rel_set = set(str(x) for x in q["relevant_ids"])
                cfgs, diag = self.search_text_all_configs(
                    query_text=q["query_text"],
                    q_vec_c=q_vecs_c[q_idx],
                    g_mask=self.g_test_mask,
                    g_indices=self.g_test_indices,
                    g_ids=self.g_test_ids
                )
                retrieved = cfgs[cfg_key]

                p, h = compute_precision_and_hit(retrieved, rel_set, k=10)
                ap = compute_ap_at_k(retrieved, rel_set, k=10)
                pmr = compute_p_min_r(retrieved, rel_set, k=10)

                p10_list.append(p)
                hit10_list.append(h)
                ap10_list.append(ap)
                pmr_list.append(pmr)

            p10_arr = np.array(p10_list)
            hit10_arr = np.array(hit10_list)
            ap10_arr = np.array(ap10_list)
            pmr_arr = np.array(pmr_list)

            m_p10, p10_l, p10_u = bootstrap_ci(p10_arr)
            m_ap, ap_l, ap_u = bootstrap_ci(ap10_arr)

            results[cfg_key] = {
                "precision_at_10": float(np.mean(p10_arr)),
                "precision_at_10_ci95": [p10_l, p10_u],
                "hit_at_10": float(np.mean(hit10_arr)),
                "ap_at_10": float(np.mean(ap10_arr)),
                "ap_at_10_ci95": [ap_l, ap_u],
                "precision_at_min_r_10": float(np.mean(pmr_arr)),
                "n_queries": len(test_queries)
            }
            tag = "MODEL THUAN (0)" if cfg_id == 0 else f"SYSTEM ({cfg_id})"
            print(f"  {tag:<18}: P@10={np.mean(p10_arr):.4f}, Hit@10={np.mean(hit10_arr):.4f}, AP@10={np.mean(ap10_arr):.4f}, P@min(R,10)={np.mean(pmr_arr):.4f}")

        return results

    # -------------------------------------------------------------
    # Q4: Visual Retrieval (1,290 Images, Self Excluded)
    # -------------------------------------------------------------
    def run_q4(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Evaluating Q4: Visual Retrieval (1,290 Test Images, Self Excluded)...", flush=True)
        print("==========================================================", flush=True)

        img_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print("Encoding horizontally flipped test images for augmentation...", flush=True)
        flip_tensors = []
        valid_q4_ids = []
        for tid in self.g_test_ids:
            img_path = str(self.df.iloc[self.id_to_idx[tid]]["path"])
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as im:
                        im_rgb = im.convert("RGB")
                        im_flip = ImageOps.mirror(im_rgb)
                        t = img_transform(im_flip)
                        flip_tensors.append(t)
                        valid_q4_ids.append(tid)
                except Exception:
                    pass

        flip_embs_list = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(flip_tensors), bs):
                batch = torch.stack(flip_tensors[i:i+bs]).to(self.device)
                if self.device == "cuda":
                    batch = batch.half()
                feats = self.model.encode_image(batch)
                flip_embs_list.append(feats.cpu().numpy())
        flip_embs = np.vstack(flip_embs_list)
        flip_embs = flip_embs / np.maximum(np.linalg.norm(flip_embs, axis=-1, keepdims=True), 1e-12)

        test_raw_embs = self.emb_img_raw[self.g_test_indices]
        aug_embs = test_raw_embs + flip_embs
        aug_embs = aug_embs / np.maximum(np.linalg.norm(aug_embs, axis=-1, keepdims=True), 1e-12)
        aug_embs_c = self.scorer.center_vectors(aug_embs)
        raw_embs_c = self.emb_img_c[self.g_test_indices]

        results_q4 = {}
        for g_name, g_mask, g_indices, g_ids in [
            ("G_test", self.g_test_mask, self.g_test_indices, self.g_test_ids),
            ("G_full", self.g_full_mask, self.g_full_indices, self.g_full_ids)
        ]:
            print(f"\n--- Q4 on {g_name} (N={len(g_ids):,}) ---", flush=True)
            results_q4[g_name] = {}
            id_to_gpos = {tid: pos for pos, tid in enumerate(g_ids)}

            modes = ["raw_cosine", "flip_augmented", "knn_voting_rescore"]
            metrics = {
                m: {
                    "cat_gen": {"p10": [], "hit10": [], "ap10": [], "pmin": []},
                    "full_spec": {"p10": [], "hit10": [], "ap10": [], "pmin": []},
                    "family": {"p10": [], "hit10": [], "ap10": [], "pmin": []}
                } for m in modes
            }

            type_correct, gender_correct, colour_correct, all_three_correct = 0, 0, 0, 0

            sims_raw_all = np.dot(raw_embs_c, self.emb_img_c[g_indices].T)
            sims_aug_all = np.dot(aug_embs_c, self.emb_img_c[g_indices].T)

            for i, tid in enumerate(self.g_test_ids):
                idx_tid = self.id_to_idx[tid]
                true_art = self.df_art[idx_tid]
                true_gen = self.df_gen[idx_tid]
                true_col = self.df_col[idx_tid]
                compat_gens = self.gender_compat.get(true_gen, [true_gen, "Unisex"])
                fam_arts = self.art_families.get(true_art, [true_art])
                fam_cols = self.col_families.get(true_col, [true_col])

                m_art = (self.df_art == true_art)
                m_gen = np.isin(self.df_gen, compat_gens)
                m_col = (self.df_col == true_col)
                m_fart = np.isin(self.df_art, fam_arts)
                m_fcol = np.isin(self.df_col, fam_cols)

                m_cat_gen = m_art & m_gen & g_mask
                m_full_spec = m_cat_gen & m_col
                m_family = m_fart & m_gen & m_fcol & g_mask

                m_cat_gen[idx_tid] = False
                m_full_spec[idx_tid] = False
                m_family[idx_tid] = False

                rel_cat_gen = set(self.item_ids_arr[m_cat_gen])
                rel_full_spec = set(self.item_ids_arr[m_full_spec])
                rel_family = set(self.item_ids_arr[m_family])

                self_pos = id_to_gpos[tid]

                # (a) Raw Cosine
                sim_raw = sims_raw_all[i].copy()
                sim_raw[self_pos] = -999.0
                top10_raw = [g_ids[idx] for idx in np.argsort(-sim_raw)[:10]]

                # (b) Flip Augmented
                sim_aug = sims_aug_all[i].copy()
                sim_aug[self_pos] = -999.0
                top10_aug = [g_ids[idx] for idx in np.argsort(-sim_aug)[:10]]

                # (c) k-NN Voting (k=20)
                top20_local_idxs = np.argsort(-sim_raw)[:20]
                top20_global_indices = g_indices[top20_local_idxs]
                top20_arts = self.df_art[top20_global_indices]
                top20_gens = self.df_gen[top20_global_indices]
                top20_cols = self.df_col[top20_global_indices]

                voted_art = Counter(top20_arts).most_common(1)[0][0]
                voted_gen = Counter(top20_gens).most_common(1)[0][0]
                voted_col = Counter(top20_cols).most_common(1)[0][0]

                if voted_art == true_art: type_correct += 1
                if voted_gen == true_gen: gender_correct += 1
                if voted_col == true_col: colour_correct += 1
                if (voted_art == true_art) and (voted_gen == true_gen) and (voted_col == true_col):
                    all_three_correct += 1

                voted_parsed = {"articleType": voted_art, "gender": voted_gen, "baseColour": voted_col, "brand": None}
                bonus_voted = self.attr_idx.compute_attribute_soft_bonus(
                    voted_parsed, g_indices,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                scores_knn = sim_raw + bonus_voted
                scores_knn[self_pos] = -999.0
                top10_knn = [g_ids[idx] for idx in np.argsort(-scores_knn)[:10]]

                for m_key, top10 in [("raw_cosine", top10_raw), ("flip_augmented", top10_aug), ("knn_voting_rescore", top10_knn)]:
                    for rel_set, key in [(rel_cat_gen, "cat_gen"), (rel_full_spec, "full_spec"), (rel_family, "family")]:
                        p, h = compute_precision_and_hit(top10, rel_set, k=10)
                        ap = compute_ap_at_k(top10, rel_set, k=10)
                        pmin = compute_p_min_r(top10, rel_set, k=10)
                        metrics[m_key][key]["p10"].append(p)
                        metrics[m_key][key]["hit10"].append(h)
                        metrics[m_key][key]["ap10"].append(ap)
                        metrics[m_key][key]["pmin"].append(pmin)

            N_q = len(self.g_test_ids)
            results_q4[g_name]["attribute_inference_accuracy"] = {
                "articleType_accuracy": type_correct / N_q,
                "gender_accuracy": gender_correct / N_q,
                "baseColour_accuracy": colour_correct / N_q,
                "all_three_accuracy": all_three_correct / N_q
            }
            print(f"  k-NN (k=20) Inferred Accuracy: Type={type_correct/N_q:.4f}, Gender={gender_correct/N_q:.4f}, Colour={colour_correct/N_q:.4f}, All3={all_three_correct/N_q:.4f}")

            for m_key in modes:
                results_q4[g_name][m_key] = {}
                for key in ["cat_gen", "full_spec", "family"]:
                    p_arr = np.array(metrics[m_key][key]["p10"])
                    ap_arr = np.array(metrics[m_key][key]["ap10"])
                    hit_arr = np.array(metrics[m_key][key]["hit10"])
                    pmin_arr = np.array(metrics[m_key][key]["pmin"])
                    m_p, p_l, p_u = bootstrap_ci(p_arr)
                    m_ap, ap_l, ap_u = bootstrap_ci(ap_arr)
                    results_q4[g_name][m_key][key] = {
                        "precision_at_10": float(np.mean(p_arr)),
                        "precision_at_10_ci95": [p_l, p_u],
                        "hit_at_10": float(np.mean(hit_arr)),
                        "ap_at_10": float(np.mean(ap_arr)),
                        "ap_at_10_ci95": [ap_l, ap_u],
                        "precision_at_min_r_10": float(np.mean(pmin_arr))
                    }
                p_cat = results_q4[g_name][m_key]["cat_gen"]["precision_at_10"]
                ap_cat = results_q4[g_name][m_key]["cat_gen"]["ap_at_10"]
                p_full = results_q4[g_name][m_key]["full_spec"]["precision_at_10"]
                ap_full = results_q4[g_name][m_key]["full_spec"]["ap_at_10"]
                print(f"  Mode {m_key:<18}: SameCat+Gen P@10={p_cat:.4f} AP@10={ap_cat:.4f} | SameFull P@10={p_full:.4f} AP@10={ap_full:.4f}")

        return results_q4

    # -------------------------------------------------------------
    # Diagnostics & F6 Brand Deep Dive
    # -------------------------------------------------------------
    def run_diagnostics(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Extracting Failure Cases & F6 Brand Deep Dive Table...", flush=True)
        print("==========================================================", flush=True)

        with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
            heldout_queries = json.load(f)

        # F6: Detailed Table for the 8 Brand Queries in Heldout
        brand_heldout_queries = [q for q in heldout_queries if q.get("group") == "brand" or (q.get("spec") and q["spec"].get("brand"))]
        print(f"Found {len(brand_heldout_queries)} brand queries in heldout.", flush=True)

        with open("src/brands.json", "r", encoding="utf-8") as f:
            brands_dict = json.load(f)
        known_brands_lower = {b.lower() for b in brands_dict.keys()}

        brand_table = []
        for q_obj in brand_heldout_queries:
            qid = q_obj["id"]
            q_text = q_obj["query"]
            spec = q_obj.get("spec", {})
            spec_brand = spec.get("brand", "")

            in_brands_json = (spec_brand.lower() in known_brands_lower)
            catalog_support = int(np.sum(self.df_brand_lower == spec_brand.lower()))

            parsed = self.parser.parse(q_text)
            parsed_brand = parsed.get("brand")
            parser_correct = (parsed_brand.lower() == spec_brand.lower()) if parsed_brand else False

            # Strict relevance mask in G_full
            m_brand = (self.df_brand_lower == spec_brand.lower())
            spec_arts = spec.get("articleType") or []
            spec_gen = spec.get("gender")
            compat_gens = self.gender_compat.get(spec_gen, [spec_gen]) if spec_gen else None
            m_art = np.isin(self.df_art, spec_arts) if spec_arts else True
            m_gen = np.isin(self.df_gen, compat_gens) if compat_gens else True
            rel_mask = m_brand & m_art & m_gen & self.g_full_mask
            rel_count = int(np.sum(rel_mask))
            rel_set = set(self.item_ids_arr[rel_mask])

            # Search Config 0 and Config 5
            q_vec_c = self.encode_text_queries([q_text])[0]
            cfgs, diag = self.search_text_all_configs(q_text, q_vec_c, self.g_full_mask, self.g_full_indices, self.g_full_ids)
            ap10_c0 = compute_ap_at_k(cfgs["config_0"], rel_set, k=10)
            ap10_c5 = compute_ap_at_k(cfgs["config_5"], rel_set, k=10)
            p10_c5, _ = compute_precision_and_hit(cfgs["config_5"], rel_set, k=10)

            root_cause = "Thành công"
            if rel_count == 0:
                root_cause = f"Catalog không có sản phẩm '{spec_brand}' khớp ({spec_arts}, {spec_gen})"
            elif not in_brands_json:
                root_cause = f"Thương hiệu '{spec_brand}' chưa có trong brands.json"
            elif not parser_correct:
                root_cause = f"Parser không nhận diện được thương hiệu (nhận: {parsed_brand})"
            elif ap10_c5 < 0.5:
                root_cause = "Độ tương đồng ngữ nghĩa thấp hoặc nhiễu từ khóa"

            brand_table.append({
                "id": qid,
                "query": q_text,
                "target_brand": spec_brand,
                "in_brands_json": in_brands_json,
                "catalog_brand_support": catalog_support,
                "parser_recognized_brand": parsed_brand,
                "parser_correct": parser_correct,
                "relevant_count_R": rel_count,
                "ap10_model_c0": round(ap10_c0, 4),
                "ap10_system_c5": round(ap10_c5, 4),
                "p10_system_c5": round(p10_c5, 4),
                "root_cause": root_cause
            })

        print("F6 Brand Deep Dive Table:")
        for b in brand_table:
            print(f"  [{b['id']}] {b['query']:<25} | Brand: {b['target_brand']:<12} | in_json: {str(b['in_brands_json']):<5} | Supp: {b['catalog_brand_support']:>4} | R: {b['relevant_count_R']:>3} | AP@10 (C0): {b['ap10_model_c0']:.2f} | AP@10 (C5): {b['ap10_system_c5']:.2f} | {b['root_cause']}")

        return {
            "f6_brand_deep_dive": brand_table
        }

    # -------------------------------------------------------------
    # Full Execution & Save Results
    # -------------------------------------------------------------
    def run_all(self):
        t_start = time.time()
        q1b_results = self.run_q1b()
        q2_results = self.run_q2()
        q3_results = self.run_q3()
        q5_results = self.run_q5()
        heldout_results = self.run_heldout()
        q6_results = self.run_q6_test()
        q4_results = self.run_q4()
        diag_results = self.run_diagnostics()

        total_sec = time.time() - t_start
        print(f"\nAll Phase 3f evaluations completed in {total_sec:.1f} seconds.", flush=True)

        full_results = {
            "metadata": {
                "phase": "3f",
                "run": "run_2_after_bug_fix",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(total_sec, 2),
                "device": self.device,
                "text_max_length": 128,
                "hyperparameters_frozen": {
                    "gamma": self.gamma,
                    "alpha": self.alpha,
                    "rrf_k": self.rrf_k
                },
                "gallery_sizes": {
                    "G_test": len(self.g_test_ids),
                    "G_full": len(self.g_full_ids)
                }
            },
            "q1b_paired_retrieval": q1b_results,
            "q2_attribute_queries": q2_results,
            "q3_handwritten_dev_queries": q3_results,
            "q5_brand_queries": q5_results,
            "heldout_independent_queries": heldout_results,
            "q6_independent_attribute_queries": q6_results,
            "q4_visual_retrieval_no_self": q4_results,
            "diagnostics": diag_results
        }

        out_json = Path("eval/results_3f.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved comprehensive Phase 3f evaluation metrics to {out_json}", flush=True)

        return full_results

if __name__ == "__main__":
    evaluator = Phase3fEvaluator()
    evaluator.run_all()
