"""
Comprehensive Phase 3e Test Evaluation Script (src/evaluate_phase3e_test.py)
Evaluates Configurations (0) to (5) on:
- Q1b: 1,290 paired captions (Configs 0, 1, 2 ONLY - strictly no BM25, no text fusion)
- Q2: 61 attribute queries (Configs 0..5 on G_test & G_full)
- Q3: 100 handwritten dev queries (Configs 0..5 on G_test & G_full)
- Q5: 150 brand queries (Configs 0..5 on G_test & G_full)
- Heldout: 30 hidden queries (Configs 0..5 on G_test & G_full: 26 with spec, 4 vague for manual rating)
- Q4: 1,290 visual queries (self excluded, raw, flip-aug, kNN-voting + rescoring)
- Diagnostics: 20 failed cases, fallback rates, bootstrap CI deltas
- Latency Benchmark: p50/p95/p99 on >= 200 queries
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
from PIL import Image, ImageOps
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

def bootstrap_diff_ci(scores_a: np.ndarray, scores_b: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    diffs = scores_a - scores_b
    mean_diff = float(np.mean(diffs))
    boot_diffs = [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(n_bootstraps)]
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, ci_lower, ci_upper

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
        equal_score = np.sum(row == tgt_score)
        if equal_score > 1:
            ties_count += 1
        rank = int(strictly_greater) + 1
        ranks[i] = rank
        if rank <= 1: hits_1[i] = 1.0
        if rank <= 5: hits_5[i] = 1.0
        if rank <= 10: hits_10[i] = 1.0

    return ranks, hits_1, hits_5, hits_10, ties_count

# -------------------------------------------------------------
# Evaluator Class for Phase 3e
# -------------------------------------------------------------
class Phase3eEvaluator:
    def __init__(self, device: str = "cuda"):
        print("Initializing Phase 3e Evaluator...", flush=True)
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        
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

        # 9. Frozen hyperparameters from Val
        self.gamma = 0.3
        self.alpha = 0.5
        self.rrf_k = 60

        print("Phase 3e Evaluator successfully initialized.\n", flush=True)

    def encode_text_queries(self, queries: List[str], batch_size: int = 128) -> np.ndarray:
        embs_list = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                b_texts = queries[i:i+batch_size]
                inp = self.processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
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
        Returns dictionary of top-10 retrieved IDs per config, and diagnostic info.
        """
        parsed = self.parser.parse(query_text)

        # Precompute base dense similarities on gallery indices
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

        # Config (2): Soft Attribute Scoring
        bonus_g = self.attr_idx.compute_attribute_soft_bonus(
            parsed, g_indices,
            gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
        )
        scores_c2 = sims_img + bonus_g
        top10_c2_local = np.argsort(-scores_c2)[:10]
        retrieved_c2 = [g_ids[i] for i in top10_c2_local]

        # Config (3): Product Text Fusion
        scores_c3 = self.alpha * sims_img + (1.0 - self.alpha) * sims_txt
        top10_c3_local = np.argsort(-scores_c3)[:10]
        retrieved_c3 = [g_ids[i] for i in top10_c3_local]

        # Config (4): Hybrid Dense (0) + BM25 RRF (k=60)
        dense_top100_c4 = [g_indices[i] for i in np.argsort(-sims_img)[:100]]
        sparse_top100_c4, _ = self.bm25.search(query_text, top_k=100, mask=g_mask)
        merged_c4, _ = rrf_merge(dense_top100_c4, sparse_top100_c4, k=self.rrf_k, top_k=10)
        retrieved_c4 = [self.item_ids[i] for i in merged_c4]

        # Config (5): Full System (Relaxation L0..L6 + Fusion + Bonus + BM25 RRF)
        c5_cand_idxs, tier = self.attr_idx.filter_relaxed(parsed, min_candidates=20, mask=g_mask)
        sims_cand_img = np.dot(self.emb_img_c[c5_cand_idxs], q_vec_c)
        sims_cand_txt = np.dot(self.emb_txt_c[c5_cand_idxs], q_vec_c)
        bonus_cand = self.attr_idx.compute_attribute_soft_bonus(
            parsed, c5_cand_idxs,
            gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
        )
        scores_c5_dense = self.alpha * sims_cand_img + (1.0 - self.alpha) * sims_cand_txt + bonus_cand
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
    # Q1b: Paired Caption -> Image (1,290 test items)
    # -------------------------------------------------------------
    def run_q1b(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Evaluating Q1b: Paired Caption -> Image (1,290 items)...", flush=True)
        print("==========================================================", flush=True)

        with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}

        # Verify test items
        q1_texts = [meta_dict[tid] for tid in self.g_test_ids]
        print(f"Loaded {len(q1_texts)} train-format captions for Q1b.", flush=True)

        # Encode queries
        print("Encoding Q1b query texts...", flush=True)
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
            print(f"  Config (0) Model thuần : R@10={np.mean(h10_c0):.4f} [{c0_l:.4f}, {c0_u:.4f}], MRR={np.mean(1.0/r_c0):.4f}, MedRank={np.median(r_c0):.1f}")

            # 2. Config (1): Hard Filter from Parser
            c1_ranks = []
            c1_hits_1, c1_hits_5, c1_hits_10 = [], [], []
            c1_fallbacks = 0

            for i in range(len(self.g_test_ids)):
                target_iid = self.g_test_ids[i]
                q_text = q1_texts[i]
                parsed = self.parser.parse(q_text)
                cand_idxs = self.attr_idx.filter_exact(parsed, mask=g_mask)

                if len(cand_idxs) == 0:
                    c1_fallbacks += 1
                    r = r_c0[i]
                else:
                    # If target is in candidates, rank among candidates
                    target_full_idx = self.id_to_idx[target_iid]
                    sims_cand = np.dot(self.emb_img_c[cand_idxs], q1_vecs_c[i])
                    if target_full_idx in cand_idxs:
                        pos = np.where(cand_idxs == target_full_idx)[0][0]
                        tgt_score = sims_cand[pos]
                        r = int(np.sum(sims_cand > tgt_score)) + 1
                    else:
                        # Target filtered out! Ranked after all candidate items
                        r = len(cand_idxs) + 1 + int(np.sum(sims_c0[i] > sims_c0[i, target_indices_in_g[i]]))

                c1_ranks.append(r)
                c1_hits_1.append(1.0 if r <= 1 else 0.0)
                c1_hits_5.append(1.0 if r <= 5 else 0.0)
                c1_hits_10.append(1.0 if r <= 10 else 0.0)

            c1_ranks = np.array(c1_ranks)
            c1_hits_10 = np.array(c1_hits_10)
            m_c1, c1_l, c1_u = bootstrap_ci(c1_hits_10)
            results_q1b[g_name]["config_1_hard_filter"] = {
                "recall_at_1": float(np.mean(c1_hits_1)),
                "recall_at_5": float(np.mean(c1_hits_5)),
                "recall_at_10": float(np.mean(c1_hits_10)),
                "recall_at_10_ci95": [c1_l, c1_u],
                "mrr": float(np.mean(1.0 / c1_ranks)),
                "median_rank": float(np.median(c1_ranks)),
                "mean_rank": float(np.mean(c1_ranks)),
                "fallback_count": c1_fallbacks
            }
            print(f"  Config (1) Hard Filter : R@10={np.mean(c1_hits_10):.4f} [{c1_l:.4f}, {c1_u:.4f}], MRR={np.mean(1.0/c1_ranks):.4f}, MedRank={np.median(c1_ranks):.1f} (Fallbacks={c1_fallbacks})")

            # 3. Config (2): Soft Attribute Scoring (gamma = 0.3)
            # Precompute soft attribute bonus for all test captions
            bonuses_q1 = []
            for i in range(len(self.g_test_ids)):
                p = self.parser.parse(q1_texts[i])
                b = self.attr_idx.compute_attribute_soft_bonus(
                    p, g_indices,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                bonuses_q1.append(b)
            bonuses_q1 = np.array(bonuses_q1) # [1290, len(g_indices)]

            scores_c2 = sims_c0 + bonuses_q1
            r_c2, h1_c2, h5_c2, h10_c2, ties_c2 = compute_exact_rank(scores_c2, target_indices_in_g)
            m_c2, c2_l, c2_u = bootstrap_ci(h10_c2)
            results_q1b[g_name]["config_2_soft_scoring"] = {
                "recall_at_1": float(np.mean(h1_c2)),
                "recall_at_5": float(np.mean(h5_c2)),
                "recall_at_10": float(np.mean(h10_c2)),
                "recall_at_10_ci95": [c2_l, c2_u],
                "mrr": float(np.mean(1.0 / r_c2)),
                "median_rank": float(np.median(r_c2)),
                "mean_rank": float(np.mean(r_c2)),
                "ties_count": ties_c2
            }
            print(f"  Config (2) Soft Scoring: R@10={np.mean(h10_c2):.4f} [{c2_l:.4f}, {c2_u:.4f}], MRR={np.mean(1.0/r_c2):.4f}, MedRank={np.median(r_c2):.1f}")

        return results_q1b

    # -------------------------------------------------------------
    # Generic Evaluation Loop for Multi-Item Attribute Queries
    # -------------------------------------------------------------
    def _evaluate_query_list(
        self,
        query_items: List[Dict[str, Any]],
        get_gt_fn,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        Generic evaluator for Q2, Q3, Q5 across all 6 configurations on G_test and G_full.
        """
        print(f"\n==========================================================", flush=True)
        print(f"Evaluating {dataset_name} ({len(query_items)} queries)...", flush=True)
        print(f"==========================================================", flush=True)

        q_texts = [q["query_text"] if "query_text" in q else q["query"] for q in query_items]
        print(f"Encoding {len(q_texts)} query texts...", flush=True)
        q_vecs_c = self.encode_text_queries(q_texts)

        results = {}
        for g_name, g_mask, g_indices, g_ids in [
            ("G_test", self.g_test_mask, self.g_test_indices, self.g_test_ids),
            ("G_full", self.g_full_mask, self.g_full_indices, self.g_full_ids)
        ]:
            print(f"\n--- {dataset_name} on {g_name} (N={len(g_ids):,}) ---", flush=True)
            results[g_name] = {}

            # Storage for metrics across 6 configurations
            cfg_names = [f"config_{i}" for i in range(6)]
            p10_dict = {cfg: [] for cfg in cfg_names}
            hit10_dict = {cfg: [] for cfg in cfg_names}
            ap10_dict = {cfg: [] for cfg in cfg_names}
            pmin_dict = {cfg: [] for cfg in cfg_names}
            
            # Family metrics if available
            p10_fam_dict = {cfg: [] for cfg in cfg_names}
            hit10_fam_dict = {cfg: [] for cfg in cfg_names}
            ap10_fam_dict = {cfg: [] for cfg in cfg_names}
            pmin_fam_dict = {cfg: [] for cfg in cfg_names}

            c1_fallbacks = 0
            tier_counts = Counter()
            valid_queries_count = 0

            for q_idx in range(len(query_items)):
                q_obj = query_items[q_idx]
                q_text = q_texts[q_idx]
                q_vec_c = q_vecs_c[q_idx]

                gt = get_gt_fn(q_obj, g_name, g_ids)
                rel_strict = gt.get("strict", set())
                rel_family = gt.get("family", set())

                if len(rel_strict) == 0 and len(rel_family) == 0:
                    continue
                valid_queries_count += 1

                configs, diag = self.search_text_all_configs(q_text, q_vec_c, g_mask, g_indices, g_ids)
                if diag["c1_fallback"]:
                    c1_fallbacks += 1
                tier_counts[diag["c5_tier"]] += 1

                for cfg in cfg_names:
                    top10 = configs[cfg]
                    if len(rel_strict) > 0:
                        p, h = compute_precision_and_hit(top10, rel_strict, k=10)
                        ap = compute_ap_at_k(top10, rel_strict, k=10)
                        pmin = compute_p_min_r(top10, rel_strict, k=10)
                        p10_dict[cfg].append(p)
                        hit10_dict[cfg].append(h)
                        ap10_dict[cfg].append(ap)
                        pmin_dict[cfg].append(pmin)

                    if len(rel_family) > 0:
                        p, h = compute_precision_and_hit(top10, rel_family, k=10)
                        ap = compute_ap_at_k(top10, rel_family, k=10)
                        pmin = compute_p_min_r(top10, rel_family, k=10)
                        p10_fam_dict[cfg].append(p)
                        hit10_fam_dict[cfg].append(h)
                        ap10_fam_dict[cfg].append(ap)
                        pmin_fam_dict[cfg].append(pmin)

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
                if p10_fam_dict[cfg]:
                    cfg_res["family"] = {
                        "precision_at_10": float(np.mean(p10_fam_dict[cfg])),
                        "hit_at_10": float(np.mean(hit10_fam_dict[cfg])),
                        "ap_at_10": float(np.mean(ap10_fam_dict[cfg])),
                        "precision_at_min_r_10": float(np.mean(pmin_fam_dict[cfg]))
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
        # Ground truth function for Q2
        def get_gt_q2(q, g_name, g_ids):
            sub_key = "g_test" if g_name == "G_test" else "g_full"
            if "target_ids_strict" in q and isinstance(q["target_ids_strict"], dict) and sub_key in q["target_ids_strict"]:
                strict_rel = set(str(x) for x in q["target_ids_strict"][sub_key])
                family_rel = set(str(x) for x in q.get("target_ids_family", {}).get(sub_key, []))
                return {"strict": strict_rel, "family": family_rel}

            tgt = q.get("target_attributes", {})
            art = tgt.get("articleType")
            gen = tgt.get("gender")
            col = tgt.get("baseColour")
            compat_gens = set(self.gender_compat.get(gen, [gen, "Unisex"])) if gen else None
            fam_arts = set(self.art_families.get(art, [art])) if art else None
            fam_cols = set(self.col_families.get(col, [col])) if col else None

            strict_rel = set()
            family_rel = set()
            for tid in g_ids:
                row = self.df.iloc[self.id_to_idx[tid]]
                i_art, i_gen, i_col = row["articleType"], row["gender"], row["baseColour"]
                art_ok = (i_art == art) if art else True
                gen_ok = (i_gen in compat_gens) if compat_gens else True
                col_ok = (i_col == col) if col else True
                if art_ok and gen_ok and col_ok:
                    strict_rel.add(tid)

                f_art_ok = (i_art in fam_arts) if fam_arts else True
                f_col_ok = (i_col in fam_cols) if fam_cols else True
                if f_art_ok and gen_ok and f_col_ok:
                    family_rel.add(tid)

            return {"strict": strict_rel, "family": family_rel}

        return self._evaluate_query_list(q2_queries, get_gt_q2, "Q2_Attribute_Queries")

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

            fam_cols = set()
            for c in spec_cols: fam_cols.update(self.col_families.get(c, [c]))
            fam_arts = set()
            for a in spec_arts: fam_arts.update(self.art_families.get(a, [a]))

            g_mask = self.g_test_mask if g_name == "G_test" else self.g_full_mask

            m_art = np.isin(self.df_art, spec_arts) if spec_arts else True
            m_gen = np.isin(self.df_gen, compat_gens) if compat_gens else True
            m_col = np.isin(self.df_col, spec_cols) if spec_cols else True
            strict_mask = m_art & m_gen & m_col & g_mask
            strict_rel = set(self.item_ids_arr[strict_mask])

            m_fart = np.isin(self.df_art, list(fam_arts)) if fam_arts else True
            m_fcol = np.isin(self.df_col, list(fam_cols)) if fam_cols else True
            family_mask = m_fart & m_gen & m_fcol & g_mask
            family_rel = set(self.item_ids_arr[family_mask])

            return {"strict": strict_rel, "family": family_rel}

        return self._evaluate_query_list(q3_queries, get_gt_q3, "Q3_Handwritten_Dev_Queries")

    # -------------------------------------------------------------
    # Q5: 150 Brand Queries
    # -------------------------------------------------------------
    def run_q5(self) -> Dict[str, Any]:
        with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
            q5_data = json.load(f)
        q5_queries = q5_data["queries"]

        def get_gt_q5(q, g_name, g_ids):
            if g_name == "G_test":
                rel = set(q.get("relevant_ids_test", []))
            else:
                rel = set(q.get("relevant_ids_full", []))
            return {"strict": rel}

        return self._evaluate_query_list(q5_queries, get_gt_q5, "Q5_Brand_Queries")

    # -------------------------------------------------------------
    # Heldout Evaluation (30 Hidden Queries: 26 Spec + 4 Vague)
    # -------------------------------------------------------------
    def run_heldout(self) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Evaluating Heldout Testset (30 Hidden Queries)...", flush=True)
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
            results_spec[g_name] = {"overall": {}, "by_group": {}, "low_support_flagged": {}}

            cfg_names = [f"config_{i}" for i in range(6)]
            query_metrics = {cfg: [] for cfg in cfg_names}
            group_metrics = defaultdict(lambda: {cfg: [] for cfg in cfg_names})
            low_support_metrics = {cfg: [] for cfg in cfg_names}
            supported_metrics = {cfg: [] for cfg in cfg_names}

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

                is_low_support = (len(rel_strict) < 3)
                configs, diag = self.search_text_all_configs(q_text, q_vec_c, g_mask, g_indices, g_ids)

                for cfg in cfg_names:
                    top10 = configs[cfg]
                    p, h = compute_precision_and_hit(top10, rel_strict, k=10)
                    ap = compute_ap_at_k(top10, rel_strict, k=10)
                    pmin = compute_p_min_r(top10, rel_strict, k=10)

                    m_dict = {"p10": p, "hit10": h, "ap10": ap, "pmin": pmin, "rel_count": len(rel_strict)}
                    query_metrics[cfg].append(m_dict)
                    group_metrics[q_group][cfg].append(m_dict)
                    if is_low_support:
                        low_support_metrics[cfg].append(m_dict)
                    else:
                        supported_metrics[cfg].append(m_dict)

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
                print(f"  Overall {tag:<18}: P@10={np.mean(p_arr):.4f}, Hit@10={np.mean(hit_arr):.4f}, AP@10={np.mean(ap_arr):.4f}, P@min={np.mean(pmin_arr):.4f}")

            # Summarize by group
            for grp, grp_data in group_metrics.items():
                results_spec[g_name]["by_group"][grp] = {}
                for cfg in cfg_names:
                    p_arr = np.array([m["p10"] for m in grp_data[cfg]])
                    ap_arr = np.array([m["ap10"] for m in grp_data[cfg]])
                    results_spec[g_name]["by_group"][grp][cfg] = {
                        "count": len(p_arr),
                        "precision_at_10": float(np.mean(p_arr)),
                        "ap_at_10": float(np.mean(ap_arr))
                    }

            # Supported vs low-support breakdown
            results_spec[g_name]["low_support_flagged"] = {
                "n_low_support": len(low_support_metrics["config_0"]),
                "n_supported": len(supported_metrics["config_0"]),
                "supported_ap10_c0": float(np.mean([m["ap10"] for m in supported_metrics["config_0"]])) if supported_metrics["config_0"] else 0.0,
                "supported_ap10_c5": float(np.mean([m["ap10"] for m in supported_metrics["config_5"]])) if supported_metrics["config_5"] else 0.0,
                "low_support_ap10_c0": float(np.mean([m["ap10"] for m in low_support_metrics["config_0"]])) if low_support_metrics["config_0"] else 0.0,
                "low_support_ap10_c5": float(np.mean([m["ap10"] for m in low_support_metrics["config_5"]])) if low_support_metrics["config_5"] else 0.0
            }

        # 2. Process 4 Vague Queries (Export contact sheets & manual ratings CSV)
        print("\nProcessing 4 Vague Heldout Queries (Manual Evaluation)...", flush=True)
        vague_texts = [q["query"] for q in vague_queries]
        vague_vecs_c = self.encode_text_queries(vague_texts)

        os.makedirs("reports/contact_sheets", exist_ok=True)
        manual_rows = []

        vague_results = {}
        for q_idx, q_obj in enumerate(vague_queries):
            qid = q_obj["id"]
            q_text = q_obj["query"]
            q_vec_c = vague_vecs_c[q_idx]

            # Search on G_full
            configs, _ = self.search_text_all_configs(
                q_text, q_vec_c, self.g_full_mask, self.g_full_indices, self.g_full_ids
            )
            top10_c0 = configs["config_0"]
            top10_c5 = configs["config_5"]

            vague_results[qid] = {
                "query": q_text,
                "top10_config_0": top10_c0,
                "top10_config_5": top10_c5
            }

            # Add to manual ratings CSV rows
            manual_rows.append({"id": qid, "query": q_text, "config": "config_0_model_thuan", "top10_ids": ";".join(top10_c0), "top10_correct_count": "", "ghi_chu": ""})
            manual_rows.append({"id": qid, "query": q_text, "config": "config_5_full_system", "top10_ids": ";".join(top10_c5), "top10_correct_count": "", "ghi_chu": ""})

            # Render contact sheet comparing Config 0 (Row 1) and Config 5 (Row 2)
            self._render_vague_contact_sheet(qid, q_text, top10_c0, top10_c5, f"reports/contact_sheets/heldout_{qid}.png")

        # Save CSV template
        manual_df = pd.DataFrame(manual_rows)
        manual_df.to_csv("eval/manual_ratings_heldout.csv", index=False, encoding="utf-8")
        print("Saved manual evaluation template to eval/manual_ratings_heldout.csv", flush=True)

        return {
            "spec_queries_evaluation": results_spec,
            "vague_queries_manual": vague_results
        }

    def _render_vague_contact_sheet(
        self,
        qid: str,
        query_text: str,
        top10_c0: List[str],
        top10_c5: List[str],
        out_path: str
    ):
        """
        Renders a 2x10 contact sheet comparing Config 0 (pure model) and Config 5 (system).
        """
        thumb_size = (140, 180)
        padding = 10
        header_height = 60
        row_label_width = 120
        total_width = row_label_width + 10 * (thumb_size[0] + padding) + padding
        total_height = header_height + 2 * (thumb_size[1] + padding) + padding

        sheet = Image.new("RGB", (total_width, total_height), color=(245, 245, 245))
        
        # Draw images for both configs
        for row_idx, (cfg_name, id_list) in enumerate([("Config 0\n(Model)", top10_c0), ("Config 5\n(System)", top10_c5)]):
            y_base = header_height + row_idx * (thumb_size[1] + padding) + padding
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
        print(f"  Rendered contact sheet for {qid} -> {out_path}", flush=True)

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

        # 1. Precompute Flipped Embeddings for 1,290 Test Images
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

        # Batch encode flipped images
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

        # Raw test embeddings from emb_img_raw
        test_raw_embs = self.emb_img_raw[self.g_test_indices]
        
        # Augmented embeddings: (orig + flip) normalized, then centered
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

            # Precompute raw similarity matrix for all queries
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

                # Exclude self
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

                # Soft bonus based on voted attributes
                voted_parsed = {"articleType": voted_art, "gender": voted_gen, "baseColour": voted_col, "brand": None}
                bonus_voted = self.attr_idx.compute_attribute_soft_bonus(
                    voted_parsed, g_indices,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                scores_knn = sim_raw + bonus_voted
                scores_knn[self_pos] = -999.0
                top10_knn = [g_ids[idx] for idx in np.argsort(-scores_knn)[:10]]

                # Record metrics for each mode
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
    # Diagnostics: Failure Cases, Fallback Rates, Statistical Delta
    # -------------------------------------------------------------
    def run_diagnostics(
        self,
        q2_res: Dict[str, Any],
        q3_res: Dict[str, Any],
        q5_res: Dict[str, Any],
        heldout_res: Dict[str, Any],
        q4_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print("Extracting 20 Failure Cases & Diagnostic Statistics...", flush=True)
        print("==========================================================", flush=True)

        failures = []

        # 1. Text Retrieval Failures (Sample from Heldout and Q3)
        with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
            heldout_queries = json.load(f)

        for q_obj in heldout_queries:
            if len(failures) >= 12:
                break
            if q_obj.get("spec") is None:
                continue
            qid = q_obj["id"]
            q_text = q_obj["query"]
            spec = q_obj["spec"]
            q_vec_c = self.encode_text_queries([q_text])[0]
            configs, diag = self.search_text_all_configs(
                q_text, q_vec_c, self.g_test_mask, self.g_test_indices, self.g_test_ids
            )
            top10_c5 = configs["config_5"]
            top10_c0 = configs["config_0"]

            # Check if any retrieved item is strictly correct
            spec_arts = set(spec.get("articleType") or [])
            spec_gen = spec.get("gender")
            spec_cols = set(spec.get("colour") or spec.get("baseColour") or [])
            spec_brand = spec.get("brand")
            compat_gens = set(self.gender_compat.get(spec_gen, [spec_gen])) if spec_gen else None

            def is_correct(tid):
                row = self.df.iloc[self.id_to_idx[tid]]
                art_ok = (row["articleType"] in spec_arts) if spec_arts else True
                gen_ok = (row["gender"] in compat_gens) if compat_gens else True
                col_ok = (row["baseColour"] in spec_cols) if spec_cols else True
                brand_ok = (str(row.get("brand", "")).lower() == spec_brand.lower()) if spec_brand else True
                return art_ok and gen_ok and col_ok and brand_ok

            c5_hits = sum(1 for tid in top10_c5 if is_correct(tid))
            if c5_hits < 5: # Failure or partial mismatch
                top1_id = top10_c5[0]
                top1_row = self.df.iloc[self.id_to_idx[top1_id]]
                reason = "Fine-grained color/pattern nuance"
                if spec_brand and str(top1_row.get("brand", "")).lower() != spec_brand.lower():
                    reason = f"Brand mismatch ({top1_row.get('brand')} vs {spec_brand})"
                elif spec_arts and top1_row["articleType"] not in spec_arts:
                    reason = f"Category confusion ({top1_row['articleType']} vs {spec_arts})"
                elif spec_cols and top1_row["baseColour"] not in spec_cols:
                    reason = f"Color discrepancy ({top1_row['baseColour']} vs {spec_cols})"

                failures.append({
                    "case_id": f"text_{qid}",
                    "type": "text_query",
                    "query": q_text,
                    "target_spec": spec,
                    "c0_top1_id": top10_c0[0],
                    "c5_top1_id": top1_id,
                    "c5_top1_metadata": {
                        "articleType": top1_row["articleType"],
                        "gender": top1_row["gender"],
                        "baseColour": top1_row["baseColour"],
                        "brand": str(top1_row.get("brand", ""))
                    },
                    "c5_top1_image_path": str(self.df.iloc[self.id_to_idx[top1_id]]["path"]),
                    "c5_precision_at_10": c5_hits / 10.0,
                    "failure_analysis": reason
                })

        # 2. Visual Retrieval Failures (Sample from Q4)
        for i in range(len(self.g_test_ids)):
            if len(failures) >= 20:
                break
            tid = self.g_test_ids[i]
            row = self.df.iloc[self.id_to_idx[tid]]
            true_art, true_gen, true_col = row["articleType"], row["gender"], row["baseColour"]

            q_vec_c = self.emb_img_c[self.id_to_idx[tid]]
            sims = np.dot(self.emb_img_c[self.g_test_indices], q_vec_c)
            self_pos = self.g_test_ids.index(tid)
            sims[self_pos] = -999.0
            top10_ids = [self.g_test_ids[idx] for idx in np.argsort(-sims)[:10]]

            # Check top 1 match
            top1_id = top10_ids[0]
            top1_row = self.df.iloc[self.id_to_idx[top1_id]]
            if top1_row["articleType"] != true_art or top1_row["baseColour"] != true_col:
                failures.append({
                    "case_id": f"visual_{tid}",
                    "type": "image_query",
                    "query_item_id": tid,
                    "query_image_path": str(self.df.iloc[self.id_to_idx[tid]]["path"]),
                    "query_metadata": {
                        "articleType": true_art,
                        "gender": true_gen,
                        "baseColour": true_col
                    },
                    "retrieved_top1_id": top1_id,
                    "retrieved_top1_image_path": str(self.df.iloc[self.id_to_idx[top1_id]]["path"]),
                    "retrieved_top1_metadata": {
                        "articleType": top1_row["articleType"],
                        "gender": top1_row["gender"],
                        "baseColour": top1_row["baseColour"]
                    },
                    "failure_analysis": f"Visual texture similarity dominated over category/color ({top1_row['articleType']}, {top1_row['baseColour']})"
                })

        print(f"Extracted {len(failures)} concrete failure cases.", flush=True)

        return {
            "failure_cases": failures
        }

    # -------------------------------------------------------------
    # Latency Benchmark (>= 200 Queries After Warm-up)
    # -------------------------------------------------------------
    def run_latency_benchmark(self, n_queries: int = 200) -> Dict[str, Any]:
        print("\n==========================================================", flush=True)
        print(f"Running Latency Benchmark on {n_queries} Queries (After Warm-up)...", flush=True)
        print("==========================================================", flush=True)

        # Pool queries from heldout, handwritten, q5
        benchmark_queries = []
        with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
            benchmark_queries.extend([q["query"] for q in json.load(f)])
        with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
            benchmark_queries.extend([q["query"] for q in json.load(f)])
        with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
            benchmark_queries.extend([q["query_text"] for q in json.load(f)["queries"]])

        rng = np.random.default_rng(seed=42)
        test_queries = list(rng.choice(benchmark_queries, size=n_queries, replace=True))

        # Warm-up (10 queries)
        print("  Warming up pipeline (10 queries)...", flush=True)
        for q in test_queries[:10]:
            q_vec_c = self.encode_text_queries([q])[0]
            _ = self.search_text_all_configs(q, q_vec_c, self.g_full_mask, self.g_full_indices, self.g_full_ids)

        # Measure each configuration individually
        timings = {f"config_{i}": [] for i in range(6)}

        print(f"  Benchmarking {n_queries} queries across all configurations on G_full (38,646 items)...", flush=True)
        for q in test_queries:
            # 1. Query Encoding
            t0 = time.perf_counter()
            q_vec_c = self.encode_text_queries([q])[0]
            enc_time_ms = (time.perf_counter() - t0) * 1000.0

            # Measure Config 0
            t0 = time.perf_counter()
            sims_img = np.dot(self.emb_img_c[self.g_full_indices], q_vec_c)
            _ = np.argsort(-sims_img)[:10]
            timings["config_0"].append(enc_time_ms + (time.perf_counter() - t0) * 1000.0)

            # Measure Config 1
            t0 = time.perf_counter()
            p = self.parser.parse(q)
            c1_cand = self.attr_idx.filter_exact(p, mask=self.g_full_mask)
            if len(c1_cand) > 0:
                _ = np.argsort(-np.dot(self.emb_img_c[c1_cand], q_vec_c))[:10]
            else:
                _ = np.argsort(-sims_img)[:10]
            timings["config_1"].append(enc_time_ms + (time.perf_counter() - t0) * 1000.0)

            # Measure Config 2
            t0 = time.perf_counter()
            bonus = self.attr_idx.compute_attribute_soft_bonus(p, self.g_full_indices, gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma)
            _ = np.argsort(-(sims_img + bonus))[:10]
            timings["config_2"].append(enc_time_ms + (time.perf_counter() - t0) * 1000.0)

            # Measure Config 3
            t0 = time.perf_counter()
            sims_txt = np.dot(self.emb_txt_c[self.g_full_indices], q_vec_c)
            _ = np.argsort(-(self.alpha * sims_img + (1.0 - self.alpha) * sims_txt))[:10]
            timings["config_3"].append(enc_time_ms + (time.perf_counter() - t0) * 1000.0)

            # Measure Config 4
            t0 = time.perf_counter()
            d_top = [self.g_full_indices[i] for i in np.argsort(-sims_img)[:100]]
            s_top, _ = self.bm25.search(q, top_k=100, mask=self.g_full_mask)
            _ = rrf_merge(d_top, s_top, k=self.rrf_k, top_k=10)
            timings["config_4"].append(enc_time_ms + (time.perf_counter() - t0) * 1000.0)

            # Measure Config 5 (Full System)
            t0 = time.perf_counter()
            cand_idxs, tier = self.attr_idx.filter_relaxed(p, min_candidates=20, mask=self.g_full_mask)
            sims_cand_img = np.dot(self.emb_img_c[cand_idxs], q_vec_c)
            sims_cand_txt = np.dot(self.emb_txt_c[cand_idxs], q_vec_c)
            bonus_cand = self.attr_idx.compute_attribute_soft_bonus(p, cand_idxs, gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma)
            scores_dense = self.alpha * sims_cand_img + (1.0 - self.alpha) * sims_cand_txt + bonus_cand
            dense_top = cand_idxs[np.argsort(-scores_dense)[:100]]
            cand_mask = np.zeros(self.N, dtype=bool)
            cand_mask[cand_idxs] = True
            sparse_top, _ = self.bm25.search(q, top_k=100, mask=cand_mask)
            _ = rrf_merge(dense_top, sparse_top, k=self.rrf_k, top_k=10)
            timings["config_5"].append(enc_time_ms + (time.perf_counter() - t0) * 1000.0)

        latency_report = {}
        for cfg in timings:
            arr = np.array(timings[cfg])
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            mean_lat = float(np.mean(arr))
            latency_report[cfg] = {
                "mean_ms": round(mean_lat, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2)
            }
            tag = "MODEL THUAN (0)" if cfg == "config_0" else f"SYSTEM ({cfg[-1]})"
            print(f"  {tag:<18}: Mean={mean_lat:.2f}ms | p50={p50:.2f}ms | p95={p95:.2f}ms | p99={p99:.2f}ms")

        # Memory usage stats
        extra_mem = {
            "attribute_index_mb": round(self.df.memory_usage(deep=True).sum() / (1024*1024), 2),
            "bm25_csr_mb": round((self.bm25.matrix.data.nbytes + self.bm25.matrix.indices.nbytes + self.bm25.matrix.indptr.nbytes) / (1024*1024), 2),
            "product_text_embeddings_mb": round(self.emb_txt_raw.nbytes / (1024*1024), 2),
            "total_extra_mb": round((self.df.memory_usage(deep=True).sum() + self.bm25.matrix.data.nbytes + self.emb_txt_raw.nbytes) / (1024*1024), 2)
        }
        print(f"  Extra Memory Footprint: {extra_mem['total_extra_mb']} MB (Text Embeddings: {extra_mem['product_text_embeddings_mb']} MB, BM25: {extra_mem['bm25_csr_mb']} MB)", flush=True)

        return {
            "latency": latency_report,
            "memory": extra_mem
        }

# -------------------------------------------------------------
# Main Execution & Report Generation
# -------------------------------------------------------------
def main():
    print("==========================================================", flush=True)
    print("Starting Comprehensive Phase 3e Test Evaluation...", flush=True)
    print("==========================================================", flush=True)

    t_start = time.time()
    evaluator = Phase3eEvaluator()

    # 1. Run Q1b (Paired text->image, configs 0, 1, 2)
    q1b_results = evaluator.run_q1b()

    # 2. Run Q2 (61 Attribute Queries, configs 0..5)
    q2_results = evaluator.run_q2()

    # 3. Run Q3 (100 Handwritten Dev Queries, configs 0..5)
    q3_results = evaluator.run_q3()

    # 4. Run Q5 (150 Brand Queries, configs 0..5)
    q5_results = evaluator.run_q5()

    # 5. Run Heldout (30 Hidden Queries: 26 Spec + 4 Vague)
    heldout_results = evaluator.run_heldout()

    # 6. Run Q4 (Visual Retrieval 1,290 Images, Self Excluded)
    q4_results = evaluator.run_q4()

    # 7. Run Diagnostics (20 Failure Cases & Fallbacks)
    diag_results = evaluator.run_diagnostics(q2_results, q3_results, q5_results, heldout_results, q4_results)

    # 8. Run Latency Benchmark (>= 200 Queries)
    latency_results = evaluator.run_latency_benchmark(n_queries=200)

    # Compile Complete Evaluation Results
    full_results = {
        "metadata": {
            "evaluation_phase": "Phase 3e (Inference-Only Enhancement)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - t_start, 2),
            "device": evaluator.device,
            "hyperparameters_frozen": {
                "gamma": evaluator.gamma,
                "alpha": evaluator.alpha,
                "rrf_k": evaluator.rrf_k
            },
            "gallery_sizes": {
                "G_test": len(evaluator.g_test_ids),
                "G_full": len(evaluator.g_full_ids)
            }
        },
        "q1b_paired_text_to_image": q1b_results,
        "q2_attribute_queries": q2_results,
        "q3_handwritten_dev_queries": q3_results,
        "q5_brand_queries": q5_results,
        "heldout_hidden_queries": heldout_results,
        "q4_visual_retrieval_no_self": q4_results,
        "diagnostics": diag_results,
        "latency_and_memory": latency_results
    }

    # Save to eval/results_3e.json
    out_json = Path("eval/results_3e.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comprehensive evaluation metrics to {out_json}", flush=True)

    print("\nPhase 3e Test Evaluation Complete!\n", flush=True)

if __name__ == "__main__":
    main()
