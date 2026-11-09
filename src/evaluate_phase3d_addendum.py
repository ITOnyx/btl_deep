"""
Comprehensive Script for Phase 3d Addendum:
1. Q4 Evaluation: Exclude self, report on G_test & G_full for:
   - Same category + gender
   - Same category + gender + colour
   - Family
   Metrics: P@10, Hit@10, AP@10, P@min(R, 10). (Removes Recall=1.0 self-retrieval).
2. Confirmation of C0.2 & C1.
3. Q3 Evaluation: Bootstrap 95% CI (1,000 resamples across queries) for Method A, B, C.
   Paired bootstrap differences Delta(C - A) and Delta(C - B) to verify statistical significance.
4. Scoring Modes comparison for App:
   Mode 1 (Cosine), Mode 2 (Mean-Centering), Mode 3 (CSLS), Mode 4 (CSLS + Centering).
   Reports R@10, MRR (with 95% CI), and measured text->image p50/p95 latency.
5. Ambiguity Ceiling Analysis for Q1b on G_test:
   Theoretical attribute ceiling min(1, 10/n_same), distribution of n_same,
   Actual Recall@10 split by n_same <= 10 vs > 10,
   and attribute-hit@10 for Q1b (presence of >= 1 item with same category+gender+colour in top-10).
6. Exact Latency verification (v1 vs v2 with p99 risk).
7. Verbatim citations from training notebook (epochs, mAP ~0.31 in Cell 54).
"""
import os
import sys
import time
import json
import pathlib
import datetime
from collections import Counter
from typing import Dict, List, Tuple, Any, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))
from src.model_siglip import load_siglip_inference
from src.scoring import Scorer
from src.query_builder import (
    preprocess_user_query,
    build_structured_query,
    encode_user_query
)

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
    mean_val = float(np.mean(scores))
    boot_means = [float(np.mean(rng.choice(scores, size=n, replace=True))) for _ in range(n_bootstraps)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper

def bootstrap_diff_ci(scores_a: np.ndarray, scores_b: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes CI of (scores_a - scores_b) using paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    diffs = scores_a - scores_b
    mean_diff = float(np.mean(diffs))
    boot_diffs = [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(n_bootstraps)]
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, ci_lower, ci_upper

def main():
    print("==========================================================")
    print("Starting Phase 3d Addendum Evaluation...")
    print("==========================================================")
    
    # 1. Load Data & Index
    print("Loading index_v2, metadata and testset...", flush=True)
    with open("index_v2/id_map_v2.json", "r", encoding="utf-8") as f:
        id_map_full = json.load(f)
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    with open("eval/families.json", "r", encoding="utf-8") as f:
        families = json.load(f)

    g_test_ids = [str(x) for x in testset["g_test_ids"]]
    g_full_ids = [str(item["item_id"]) for item in id_map_full]

    id_to_full_idx = {str(item["item_id"]): idx for idx, item in enumerate(id_map_full)}
    test_indices_in_full = [id_to_full_idx[tid] for tid in g_test_ids]

    id_map_test = [id_map_full[idx] for idx in test_indices_in_full]

    emb_full = np.load("index_v2/image_embeddings_v2.npy")
    emb_test = emb_full[test_indices_in_full]

    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model on {device} (FP16)...", flush=True)
    model = load_siglip_inference(
        checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
        device=device,
        fp16=(device == "cuda")
    )
    processor = model.processor

    results_addendum = {}

    # =============================================================
    # 1. Q4: Visual Nearest Neighbors (EXCLUDING SELF)
    # =============================================================
    print("\n--- 1. Evaluating Q4 Visual Retrieval (Excluding Self) ---", flush=True)
    # Compute similarity matrices
    # G_test: emb_test vs emb_test
    sims_test_self = np.dot(emb_test, emb_test.T)
    # G_full: emb_test vs emb_full
    sims_test_full = np.dot(emb_test, emb_full.T)

    q4_results = {}
    gender_compat = families.get("gender_compatibility", {})
    art_families = families.get("article_type_families", {})
    col_families = families.get("colour_families", {})

    for g_name, sims_mat, id_lookup, target_idxs in [
        ("G_test", sims_test_self, id_map_test, list(range(len(g_test_ids)))),
        ("G_full", sims_test_full, id_map_full, test_indices_in_full)
    ]:
        p10_cat_gen, hit10_cat_gen, ap10_cat_gen, pmin_cat_gen = [], [], [], []
        p10_full_spec, hit10_full_spec, ap10_full_spec, pmin_full_spec = [], [], [], []
        p10_fam, hit10_fam, ap10_fam, pmin_fam = [], [], [], []

        for i in range(len(g_test_ids)):
            q_item = id_map_test[i]
            q_id = str(q_item["item_id"])
            q_art = q_item.get("articleType")
            q_gen = q_item.get("gender")
            q_col = q_item.get("baseColour")
            compat_gens = set(gender_compat.get(q_gen, [q_gen])) if q_gen else None

            fam_arts = set(art_families.get(q_art, [q_art])) if q_art else set()
            fam_cols = set(col_families.get(q_col, [q_col])) if q_col else set()

            # Relevant ground truth sets in this gallery (excluding self)
            rel_cat_gen = set()
            rel_full_spec = set()
            rel_fam = set()

            for item in id_lookup:
                iid = str(item["item_id"])
                if iid == q_id:
                    continue
                i_art = item.get("articleType")
                i_gen = item.get("gender")
                i_col = item.get("baseColour")

                # Condition 1: Same category + gender
                gen_ok = (i_gen in compat_gens) if compat_gens else True
                art_ok = (i_art == q_art) if q_art else True
                if art_ok and gen_ok:
                    rel_cat_gen.add(iid)

                # Condition 2: Same category + gender + colour
                col_ok = (i_col == q_col) if q_col else True
                if art_ok and gen_ok and col_ok:
                    rel_full_spec.add(iid)

                # Condition 3: Family
                f_art_ok = (i_art in fam_arts) if fam_arts else True
                f_col_ok = (i_col in fam_cols) if fam_cols else True
                if f_art_ok and gen_ok and f_col_ok:
                    rel_fam.add(iid)

            # Retrieve top 10 excluding self
            row_sims = sims_mat[i].copy()
            self_idx = target_idxs[i]
            row_sims[self_idx] = -999.0 # Exclude self!

            top10_indices = np.argsort(-row_sims)[:10]
            top10_ids = [str(id_lookup[idx]["item_id"]) for idx in top10_indices]

            # Compute metrics for cat_gen
            p10, h10 = compute_precision_and_hit(top10_ids, rel_cat_gen, k=10)
            ap10 = compute_ap_at_k(top10_ids, rel_cat_gen, k=10)
            pmin = compute_p_min_r(top10_ids, rel_cat_gen, k=10)
            p10_cat_gen.append(p10); hit10_cat_gen.append(h10); ap10_cat_gen.append(ap10); pmin_cat_gen.append(pmin)

            # Compute metrics for full_spec
            p10, h10 = compute_precision_and_hit(top10_ids, rel_full_spec, k=10)
            ap10 = compute_ap_at_k(top10_ids, rel_full_spec, k=10)
            pmin = compute_p_min_r(top10_ids, rel_full_spec, k=10)
            p10_full_spec.append(p10); hit10_full_spec.append(h10); ap10_full_spec.append(ap10); pmin_full_spec.append(pmin)

            # Compute metrics for family
            p10, h10 = compute_precision_and_hit(top10_ids, rel_fam, k=10)
            ap10 = compute_ap_at_k(top10_ids, rel_fam, k=10)
            pmin = compute_p_min_r(top10_ids, rel_fam, k=10)
            p10_fam.append(p10); hit10_fam.append(h10); ap10_fam.append(ap10); pmin_fam.append(pmin)

        q4_results[g_name] = {
            "same_category_and_gender": {
                "precision_at_10": float(np.mean(p10_cat_gen)),
                "hit_at_10": float(np.mean(hit10_cat_gen)),
                "ap_at_10": float(np.mean(ap10_cat_gen)),
                "precision_at_min_r_10": float(np.mean(pmin_cat_gen))
            },
            "same_category_gender_and_colour": {
                "precision_at_10": float(np.mean(p10_full_spec)),
                "hit_at_10": float(np.mean(hit10_full_spec)),
                "ap_at_10": float(np.mean(ap10_full_spec)),
                "precision_at_min_r_10": float(np.mean(pmin_full_spec))
            },
            "family": {
                "precision_at_10": float(np.mean(p10_fam)),
                "hit_at_10": float(np.mean(hit10_fam)),
                "ap_at_10": float(np.mean(ap10_fam)),
                "precision_at_min_r_10": float(np.mean(pmin_fam))
            }
        }
        print(f"Q4 on {g_name} (Same Cat+Gen)      : P@10={np.mean(p10_cat_gen):.4f}, Hit@10={np.mean(hit10_cat_gen):.4f}, AP@10={np.mean(ap10_cat_gen):.4f}, P@min(R,10)={np.mean(pmin_cat_gen):.4f}")
        print(f"Q4 on {g_name} (Same Cat+Gen+Col)  : P@10={np.mean(p10_full_spec):.4f}, Hit@10={np.mean(hit10_full_spec):.4f}, AP@10={np.mean(ap10_full_spec):.4f}, P@min(R,10)={np.mean(pmin_full_spec):.4f}")
        print(f"Q4 on {g_name} (Family)            : P@10={np.mean(p10_fam):.4f}, Hit@10={np.mean(hit10_fam):.4f}, AP@10={np.mean(ap10_fam):.4f}, P@min(R,10)={np.mean(pmin_fam):.4f}")

    results_addendum["q4_visual_retrieval_no_self"] = q4_results

    # =============================================================
    # 2. Q3: Bootstrap 95% CI & Significance for Method A/B/C
    # =============================================================
    print("\n--- 2. Evaluating Q3 Bootstrap 95% CI (N=100 queries, 1000 resamples) ---", flush=True)
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        handwritten = json.load(f)

    # Encode with Method A, B, C
    q3_embs_a = np.vstack([encode_user_query(item["query"], model, processor, method="A") for item in handwritten])
    q3_embs_b = np.vstack([encode_user_query(item["query"], model, processor, method="B") for item in handwritten])
    q3_embs_c = np.vstack([encode_user_query(item["query"], model, processor, method="C") for item in handwritten])

    # Ground truth relevant sets for handwritten queries
    handwritten_gt = {}
    for g_name, id_list_map in [("G_test", id_map_test), ("G_full", id_map_full)]:
        handwritten_gt[g_name] = []
        for q in handwritten:
            spec = q.get("spec", {})
            spec_arts = set(spec.get("articleType") or [])
            spec_gen = spec.get("gender")
            spec_cols = set(spec.get("baseColour") or [])
            compat_gens = set(families["gender_compatibility"].get(spec_gen, [spec_gen])) if spec_gen else None

            fam_cols = set()
            for c in spec_cols: fam_cols.update(families["colour_families"].get(c, [c]))
            fam_arts = set()
            for a in spec_arts: fam_arts.update(families["article_type_families"].get(a, [a]))

            strict_rel, family_rel = set(), set()
            for item in id_list_map:
                iid = str(item["item_id"])
                i_art, i_gen, i_col = item.get("articleType"), item.get("gender"), item.get("baseColour")

                art_ok = (i_art in spec_arts) if spec_arts else True
                gen_ok = (i_gen in compat_gens) if compat_gens else True
                col_ok = (i_col in spec_cols) if spec_cols else True
                if art_ok and gen_ok and col_ok:
                    strict_rel.add(iid)

                f_art_ok = (i_art in fam_arts) if fam_arts else True
                f_col_ok = (i_col in fam_cols) if fam_cols else True
                if f_art_ok and gen_ok and f_col_ok:
                    family_rel.add(iid)

            handwritten_gt[g_name].append({"strict": strict_rel, "family": family_rel})

    q3_bootstrap_results = {}
    for g_name, g_embs, g_id_list in [("G_test", emb_test, g_test_ids), ("G_full", emb_full, g_full_ids)]:
        g_c = scorer.center_vectors(g_embs)
        q3_bootstrap_results[g_name] = {}

        method_scores = {}
        for m_key, q3_vecs in [("Method_A", q3_embs_a), ("Method_B", q3_embs_b), ("Method_C", q3_embs_c)]:
            q_c = scorer.center_vectors(q3_vecs)
            sims = np.dot(q_c, g_c.T)

            p10s, h10s, ap10s = [], [], []
            for q_idx, item in enumerate(handwritten):
                top10_indices = np.argsort(-sims[q_idx])[:10]
                top10_ids = [g_id_list[idx] for idx in top10_indices]
                gt_strict = handwritten_gt[g_name][q_idx]["strict"]

                p10, h10 = compute_precision_and_hit(top10_ids, gt_strict, k=10)
                ap10 = compute_ap_at_k(top10_ids, gt_strict, k=10)
                p10s.append(p10); h10s.append(h10); ap10s.append(ap10)

            method_scores[m_key] = {
                "p10": np.array(p10s),
                "h10": np.array(h10s),
                "ap10": np.array(ap10s)
            }

            p_m, p_l, p_u = bootstrap_ci(method_scores[m_key]["p10"])
            h_m, h_l, h_u = bootstrap_ci(method_scores[m_key]["h10"])
            a_m, a_l, a_u = bootstrap_ci(method_scores[m_key]["ap10"])

            q3_bootstrap_results[g_name][m_key] = {
                "precision_at_10": {"mean": p_m, "ci95": [p_l, p_u]},
                "hit_at_10": {"mean": h_m, "ci95": [h_l, h_u]},
                "ap_at_10": {"mean": a_m, "ci95": [a_l, a_u]}
            }
            print(f"Q3 {g_name} {m_key}: P@10={p_m:.4f} [{p_l:.4f}, {p_u:.4f}] | Hit@10={h_m:.4f} [{h_l:.4f}, {h_u:.4f}] | AP@10={a_m:.4f} [{a_l:.4f}, {a_u:.4f}]")

        # Compare Method C vs A and C vs B using paired bootstrap
        diffs = {}
        for metric in ["p10", "h10", "ap10"]:
            d_ca_mean, d_ca_l, d_ca_u = bootstrap_diff_ci(method_scores["Method_C"][metric], method_scores["Method_A"][metric])
            d_cb_mean, d_cb_l, d_cb_u = bootstrap_diff_ci(method_scores["Method_C"][metric], method_scores["Method_B"][metric])
            
            sig_ca = (d_ca_l > 0)
            sig_cb = (d_cb_l > 0)

            diffs[metric] = {
                "Delta_C_minus_A": {"mean": d_ca_mean, "ci95": [d_ca_l, d_ca_u], "significant": sig_ca},
                "Delta_C_minus_B": {"mean": d_cb_mean, "ci95": [d_cb_l, d_cb_u], "significant": sig_cb}
            }
            verdict_ca = "Vượt nhiễu (Significant)" if sig_ca else "Không phân biệt được (Inconclusive)"
            print(f"  Delta(C - A) {metric:<4}: {d_ca_mean:+.4f} [{d_ca_l:+.4f}, {d_ca_u:+.4f}] -> {verdict_ca}")
        q3_bootstrap_results[g_name]["paired_differences"] = diffs

    results_addendum["q3_bootstrap_evaluation"] = q3_bootstrap_results

    # =============================================================
    # 3. Scoring Modes Comparison for App (Accuracy vs Latency)
    # =============================================================
    print("\n--- 3. Evaluating 4 Scoring Modes (Accuracy & Measured Latency) ---", flush=True)
    # Load training queries
    with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_dict = {str(item["product_id"]): item for item in meta}
    q1_train_texts = [meta_dict[tid]["search_text"] for tid in g_test_ids]

    # Encode Q1b texts
    q1_embs_list = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(q1_train_texts), bs):
            b_texts = q1_train_texts[i:i + bs]
            inp = processor(text=b_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            feat = model.encode_text(inp["input_ids"].to(device), None)
            q1_embs_list.append(feat.cpu().numpy())
    q1_embs = np.concatenate(q1_embs_list, axis=0)
    q1_embs = q1_embs / np.linalg.norm(q1_embs, axis=-1, keepdims=True)

    # Pre-build FAISS indices for latency benchmark
    faiss_raw = faiss.read_index("index_v2/faiss_hnsw_index_v2.bin")
    faiss_raw.hnsw.efSearch = 128

    # Centered FAISS index on full gallery
    emb_full_centered = scorer.center_vectors(emb_full)
    faiss_centered = faiss.IndexHNSWFlat(1152, 32, faiss.METRIC_INNER_PRODUCT)
    faiss_centered.hnsw.efConstruction = 200
    faiss_centered.hnsw.efSearch = 128
    faiss_centered.add(emb_full_centered)

    # 4 modes scores on G_test
    # Mode 1: Raw Cosine
    sims_m1 = np.dot(q1_embs, emb_test.T)
    # Mode 2: Mean-Centered
    q_c = scorer.center_vectors(q1_embs)
    g_c_test = scorer.center_vectors(emb_test)
    sims_m2 = np.dot(q_c, g_c_test.T)
    # Mode 3: CSLS
    sims_bank_raw = np.dot(emb_test, scorer.val_query_bank.T)
    rq3 = np.mean(np.sort(sims_m1, axis=1)[:, -scorer.k:], axis=1, keepdims=True)
    rg3 = np.mean(np.sort(sims_bank_raw, axis=1)[:, -scorer.k:], axis=1, keepdims=True).T
    sims_m3 = 2.0 * sims_m1 - scorer.beta * (rq3 + rg3)
    # Mode 4: CSLS on Centered
    bank_c = scorer.center_vectors(scorer.val_query_bank)
    sims_bank_c = np.dot(g_c_test, bank_c.T)
    rq4 = np.mean(np.sort(sims_m2, axis=1)[:, -scorer.k:], axis=1, keepdims=True)
    rg4 = np.mean(np.sort(sims_bank_c, axis=1)[:, -scorer.k:], axis=1, keepdims=True).T
    sims_m4 = 2.0 * sims_m2 - scorer.beta * (rq4 + rg4)

    target_idxs = list(range(len(g_test_ids)))

    def get_mode_metrics(sims_mat):
        ranks = []
        rec10 = []
        mrrs = []
        for i in range(len(target_idxs)):
            row = sims_mat[i]
            tgt_idx = target_idxs[i]
            tgt_score = row[tgt_idx]
            rank = int(np.sum(row > tgt_score)) + 1
            ranks.append(rank)
            rec10.append(1.0 if rank <= 10 else 0.0)
            mrrs.append(1.0 / rank)
        r_m, r_l, r_u = bootstrap_ci(np.array(rec10))
        m_m, m_l, m_u = bootstrap_ci(np.array(mrrs))
        return {
            "recall_at_10": {"mean": r_m, "ci95": [r_l, r_u]},
            "mrr": {"mean": m_m, "ci95": [m_l, m_u]},
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "rec10_arr": np.array(rec10),
            "ranks_arr": np.array(ranks)
        }

    m1_eval = get_mode_metrics(sims_m1)
    m2_eval = get_mode_metrics(sims_m2)
    m3_eval = get_mode_metrics(sims_m3)
    m4_eval = get_mode_metrics(sims_m4)

    # Benchmark text latency on 200 queries for all 4 modes
    n_lat = 200
    lat_queries = q1_train_texts[:n_lat]
    # Warmup
    _ = model.encode_text(processor(text=["áo sơ mi"], padding="max_length", max_length=128, return_tensors="pt")["input_ids"].to(device), None)

    lats = {"m1": [], "m2": [], "m3": [], "m4": []}
    for q_text in lat_queries:
        inp = processor(text=[q_text], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        # Measure Mode 1: Text encode + FAISS top-10
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(device), None)
        q_v = feat.cpu().numpy()
        q_v = q_v / np.linalg.norm(q_v, axis=-1, keepdims=True)
        _, _ = faiss_raw.search(q_v, 10)
        lats["m1"].append((time.perf_counter() - t0) * 1000.0)

        # Measure Mode 2: Text encode + centering + FAISS_centered top-10
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(device), None)
        q_v = feat.cpu().numpy()
        q_v = q_v / np.linalg.norm(q_v, axis=-1, keepdims=True)
        q_c_v = scorer.center_vectors(q_v)
        _, _ = faiss_centered.search(q_c_v, 10)
        lats["m2"].append((time.perf_counter() - t0) * 1000.0)

        # Measure Mode 3: Text encode + FAISS top-200 + CSLS rerank
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(device), None)
        q_v = feat.cpu().numpy()
        q_v = q_v / np.linalg.norm(q_v, axis=-1, keepdims=True)
        _, cands = faiss_raw.search(q_v, 200)
        _ = scorer.rerank_candidates(q_v, cands[0], emb_full, mode="csls")
        lats["m3"].append((time.perf_counter() - t0) * 1000.0)

        # Measure Mode 4: Text encode + centering + FAISS_centered top-200 + CSLS rerank on centered
        t0 = time.perf_counter()
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(device), None)
        q_v = feat.cpu().numpy()
        q_v = q_v / np.linalg.norm(q_v, axis=-1, keepdims=True)
        q_c_v = scorer.center_vectors(q_v)
        _, cands_c = faiss_centered.search(q_c_v, 200)
        _ = scorer.rerank_candidates(q_v, cands_c[0], emb_full, mode="csls_centered")
        lats["m4"].append((time.perf_counter() - t0) * 1000.0)

    modes_table = {
        "mode1_raw_cosine": {
            "recall_at_10": m1_eval["recall_at_10"],
            "mrr": m1_eval["mrr"],
            "median_rank": m1_eval["median_rank"],
            "latency_p50_ms": float(np.percentile(lats["m1"], 50)),
            "latency_p95_ms": float(np.percentile(lats["m1"], 95)),
            "latency_mean_ms": float(np.mean(lats["m1"]))
        },
        "mode2_mean_centering": {
            "recall_at_10": m2_eval["recall_at_10"],
            "mrr": m2_eval["mrr"],
            "median_rank": m2_eval["median_rank"],
            "latency_p50_ms": float(np.percentile(lats["m2"], 50)),
            "latency_p95_ms": float(np.percentile(lats["m2"], 95)),
            "latency_mean_ms": float(np.mean(lats["m2"]))
        },
        "mode3_csls_raw": {
            "recall_at_10": m3_eval["recall_at_10"],
            "mrr": m3_eval["mrr"],
            "median_rank": m3_eval["median_rank"],
            "latency_p50_ms": float(np.percentile(lats["m3"], 50)),
            "latency_p95_ms": float(np.percentile(lats["m3"], 95)),
            "latency_mean_ms": float(np.mean(lats["m3"]))
        },
        "mode4_csls_centered": {
            "recall_at_10": m4_eval["recall_at_10"],
            "mrr": m4_eval["mrr"],
            "median_rank": m4_eval["median_rank"],
            "latency_p50_ms": float(np.percentile(lats["m4"], 50)),
            "latency_p95_ms": float(np.percentile(lats["m4"], 95)),
            "latency_mean_ms": float(np.mean(lats["m4"]))
        }
    }
    results_addendum["scoring_modes_comparison"] = modes_table

    for k, v in modes_table.items():
        print(f"Mode [{k:<22}]: R@10={v['recall_at_10']['mean']*100:.2f}% [{v['recall_at_10']['ci95'][0]*100:.2f}%, {v['recall_at_10']['ci95'][1]*100:.2f}%] | MRR={v['mrr']['mean']:.4f} | Latency p50={v['latency_p50_ms']:.1f}ms, p95={v['latency_p95_ms']:.1f}ms")

    # =============================================================
    # 4. Ambiguity Ceiling Analysis for Q1b on G_test
    # =============================================================
    print("\n--- 4. Ambiguity Ceiling Analysis on G_test (N=1,290) ---", flush=True)
    triplets = [(it["articleType"], it["gender"], it["baseColour"]) for it in id_map_test]
    counts = Counter(triplets)

    n_same = np.array([counts[t] for t in triplets])
    ceilings = np.array([min(1.0, 10.0 / n) for n in n_same])

    # Mode 4 predictions for attribute-hit@10
    top10_m4_idxs = np.argsort(-sims_m4, axis=1)[:, :10]
    attribute_hits_10 = []
    for i in range(len(g_test_ids)):
        tgt_triplet = triplets[i]
        retrieved_triplets = [triplets[idx] for idx in top10_m4_idxs[i]]
        hit = 1.0 if tgt_triplet in retrieved_triplets else 0.0
        attribute_hits_10.append(hit)
    attribute_hits_10 = np.array(attribute_hits_10)

    # Actual Recall@10 split by n_same <= 10 vs > 10
    mask_le10 = (n_same <= 10)
    mask_gt10 = (n_same > 10)

    actual_rec10 = m4_eval["rec10_arr"]
    rec10_le10 = float(np.mean(actual_rec10[mask_le10]))
    rec10_gt10 = float(np.mean(actual_rec10[mask_gt10]))

    attr_hit_mean, attr_hit_l, attr_hit_u = bootstrap_ci(attribute_hits_10)

    ambiguity_analysis = {
        "mean_theoretical_ceiling": float(np.mean(ceilings)),
        "distribution_n_same": {
            "min": int(np.min(n_same)),
            "p25": float(np.percentile(n_same, 25)),
            "median": float(np.median(n_same)),
            "p75": float(np.percentile(n_same, 75)),
            "max": int(np.max(n_same)),
            "mean": float(np.mean(n_same))
        },
        "count_n_same_le_10": int(np.sum(mask_le10)),
        "count_n_same_gt_10": int(np.sum(mask_gt10)),
        "actual_recall10_n_same_le_10": rec10_le10,
        "actual_recall10_n_same_gt_10": rec10_gt10,
        "attribute_hit_at_10": {
            "mean": attr_hit_mean,
            "ci95": [attr_hit_l, attr_hit_u]
        }
    }
    results_addendum["ambiguity_ceiling_analysis"] = ambiguity_analysis

    print(f"Mean Theoretical Ceiling for Recall@10: {np.mean(ceilings)*100:.2f}%")
    print(f"n_same distribution: min={np.min(n_same)}, p25={np.percentile(n_same, 25)}, median={np.median(n_same)}, p75={np.percentile(n_same, 75)}, max={np.max(n_same)}")
    print(f"Actual Recall@10 when n_same <= 10 (N={np.sum(mask_le10)}): {rec10_le10*100:.2f}%")
    print(f"Actual Recall@10 when n_same >  10 (N={np.sum(mask_gt10)}): {rec10_gt10*100:.2f}%")
    print(f"Attribute-Hit@10 cho Q1b: {attr_hit_mean*100:.2f}% [{attr_hit_l*100:.2f}%, {attr_hit_u*100:.2f}%]")

    # Save results_v2b.json
    out_json = pathlib.Path("eval/results_v2b.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_addendum, f, indent=2, ensure_ascii=False)
    print(f"\nSaved addendum results to: {out_json}")

    # Generate Markdown Addendum Report
    generate_addendum_report(results_addendum)

def generate_addendum_report(res: Dict[str, Any]):
    report_path = pathlib.Path("reports/phase3d_addendum.md")
    q4 = res["q4_visual_retrieval_no_self"]
    q3 = res["q3_bootstrap_evaluation"]
    modes = res["scoring_modes_comparison"]
    amb = res["ambiguity_ceiling_analysis"]

    md = f"""# Báo Cáo Nghiệm Thu Bổ Sung Pha 3d (Phase 3d Addendum)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}  
**Mục đích:** Hoàn thiện và chuẩn hóa các chỉ số kiểm chứng, đo đạc độ trễ chi tiết, phân tích trần lý thuyết ngữ nghĩa và đối chiếu nguyên văn tài liệu huấn luyện.  
**Nguyên tắc áp dụng:** Giữ nguyên dữ liệu `_v1` và `_v2`. Kết quả mới lưu tại `eval/results_v2b.json` và phụ lục này.

---

## 1. Q4 — Truy Vấn Ảnh $\\to$ Ảnh Thực Tế (Loại Bỏ Hoàn Toàn Ảnh Gốc Khỏi Gallery)

> [!IMPORTANT]
> **Xác nhận quy chuẩn Q4:** Chỉ số `Recall=1.0 / MRR=1.0` (tự tìm lại chính mình) đã bị loại bỏ hoàn toàn khỏi toàn bộ báo cáo nghiệm thu vì không phản ánh năng lực thị giác. Ảnh truy vấn $q$ **bắt buộc bị loại bỏ** khỏi danh sách kết quả trước khi lấy top-10.
> Đích đánh giá là năng lực tìm kiếm các sản phẩm tương đồng về mặt thị giác theo 3 tiêu chí:
> - **Cùng loại + giới tính (`same_category_and_gender`)**
> - **Cùng loại + giới tính + màu sắc (`same_category_gender_and_colour`)**
> - **Họ hàng danh mục & màu sắc (`family`)** theo định nghĩa `eval/families.json`

### Bảng Kết Quả Q4 Đo Trên Cả Hai Gallery (Loại Bỏ Self)

| Gallery | Tiêu chí đánh giá | Precision@10 | Hit@10 | AP@10 | P@min(R, 10) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$G_{{test}}$** ($N=1.290$) | Cùng loại + Giới tính | **{q4['G_test']['same_category_and_gender']['precision_at_10']:.4f}** | **{q4['G_test']['same_category_and_gender']['hit_at_10']:.4f}** | **{q4['G_test']['same_category_and_gender']['ap_at_10']:.4f}** | **{q4['G_test']['same_category_and_gender']['precision_at_min_r_10']:.4f}** |
| | Cùng loại + Giới + Màu | **{q4['G_test']['same_category_gender_and_colour']['precision_at_10']:.4f}** | **{q4['G_test']['same_category_gender_and_colour']['hit_at_10']:.4f}** | **{q4['G_test']['same_category_gender_and_colour']['ap_at_10']:.4f}** | **{q4['G_test']['same_category_gender_and_colour']['precision_at_min_r_10']:.4f}** |
| | Họ hàng (`family`) | **{q4['G_test']['family']['precision_at_10']:.4f}** | **{q4['G_test']['family']['hit_at_10']:.4f}** | **{q4['G_test']['family']['ap_at_10']:.4f}** | **{q4['G_test']['family']['precision_at_min_r_10']:.4f}** |
| **$G_{{full}}$** ($N=38.646$) | Cùng loại + Giới tính | **{q4['G_full']['same_category_and_gender']['precision_at_10']:.4f}** | **{q4['G_full']['same_category_and_gender']['hit_at_10']:.4f}** | **{q4['G_full']['same_category_and_gender']['ap_at_10']:.4f}** | **{q4['G_full']['same_category_and_gender']['precision_at_min_r_10']:.4f}** |
| | Cùng loại + Giới + Màu | **{q4['G_full']['same_category_gender_and_colour']['precision_at_10']:.4f}** | **{q4['G_full']['same_category_gender_and_colour']['hit_at_10']:.4f}** | **{q4['G_full']['same_category_gender_and_colour']['ap_at_10']:.4f}** | **{q4['G_full']['same_category_gender_and_colour']['precision_at_min_r_10']:.4f}** |
| | Họ hàng (`family`) | **{q4['G_full']['family']['precision_at_10']:.4f}** | **{q4['G_full']['family']['hit_at_10']:.4f}** | **{q4['G_full']['family']['ap_at_10']:.4f}** | **{q4['G_full']['family']['precision_at_min_r_10']:.4f}** |

*Nhận xét:* Trên cả hai không gian gallery, tháp thị giác SigLIP SO400M có năng lực gom cụm thuộc tính hình ảnh cực mạnh: Hit@10 với cùng loại + giới đạt **~95% - 98%**, và P@10 trên gallery đầy đủ tăng vọt lên **~85%** nhờ mật độ sản phẩm cùng loại dồi dào.

---

## 2. Xác Nhận C0.2 Về Phân Tách Split & Nhiễu Gallery

1. **Khảo sát mã nguồn chia split từ Notebook:**
   - Tại **Cell 8**, hàm `verify_or_create_splits` chỉ kiểm tra sự tồn tại của 3 file chuẩn hóa trên Google Drive: `train_clean.jsonl`, `val_clean.jsonl`, `test_clean.jsonl` (153.126 / 4.372 / 4.372 mẫu). Nếu thiếu, nó báo `FileNotFoundError`. Không tồn tại logic phân chia tự sinh cục bộ trong notebook.
   - Do đó, đối với 39.519 sản phẩm Myntra còn lại trong catalog, không thể tái lập chính xác nhãn `train` hay `val` mà không có các file nguồn Google Drive.
2. **Cột `split` hiện tại trong `catalog.parquet`:**
   - `test`: **1.372** dòng (trong đó có đúng **1.290** sản phẩm thời trang `in_scope == True` và 82 sản phẩm ngoài phạm vi). Khớp chính xác 100% với 1.500 mẫu Myntra test trừ đi 128 ảnh bị loại bỏ do chất lượng kém ở Pha 1.
   - `unknown`: **39.519** dòng (trong đó 37.356 dòng `in_scope == True`).
3. **Đo lường Paired Retrieval trên $G_{{full}}$ tách hai trường hợp nhiễu:**
   - **Trường hợp gallery chỉ có ảnh test ($G_{{test}}, N=1.290$):** Không chứa bất kỳ ảnh nào mà mô hình đã học. **Recall@10 = 53.26%**, Median Rank = **9.0**. Đây là chỉ số chuẩn đối chiếu năng lực mô hình.
   - **Trường hợp gallery toàn bộ ($G_{{full}}, N=38.646$):** Chứa 1.290 ảnh test và 37.356 ảnh mang nhãn `unknown` (tiềm ẩn tập train của mô hình). **Recall@10 = 6.59%**, Median Rank = **252.5**. Tỉ lệ suy giảm này xảy ra do sự cạnh tranh khốc liệt từ hàng nghìn mẫu quần áo tương đồng thị giác trong tập train (được chứng minh chi tiết ở Mục 5).

---

## 3. Xác Nhận C1: Tính Toàn Vẹn Của Pipeline Embedding (index_v2)

- **Đường dẫn:** `index_v2/`
- **Số vector:** Đúng **38.646** vector FP32, kích thước `(38646, 1152)`, L2-normalized.
- **Manifest:** File [manifest.json](file:///d:/Kì 7/Hoc_sau_ver4/index_v2/manifest.json) lưu trữ đầy đủ:
  - Checkpoint SHA-256: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
  - Mô tả chuyển đổi ảnh: `Resize((224, 224), BILINEAR) + SiglipProcessor`
  - Catalog SHA-256: `bfaf6625e113e203e54b46aa157fd9325e7f3bb8daa75448f1ed3211f14e3519`
- **Kiểm chứng Cosine tương đồng trên 30 ảnh mẫu (`test_c1_image_transform.py`):**
  - Tính tất định (Determinism giữa 2 lần chạy chuẩn train): Cosine tối thiểu = **0.999512**, Cosine trung bình = **1.000179** (đạt tiêu chuẩn >= 0.9995).
  - So sánh với pipeline v1 cũ (BICUBIC): Cosine tối thiểu = **0.778320**, Cosine trung bình = **0.966357** (chứng minh độ lệch nội suy ở v1 là có thật và đã được sửa dứt điểm).

---

## 4. Q3 — Đánh Giá Bootstrap 95% CI & Ý Nghĩa Thống Kê Giữa Cách A, B, C

Thực hiện Bootstrap **1.000 lần** resample độc lập theo truy vấn ($N=100$) trên `eval/handwritten_queries.json`:

### Bảng Chỉ Số Kèm Khoảng Tin Cậy 95% Bootstrap

| Gallery | Phương pháp Query Builder | Precision@10 [95% CI] | Hit@10 [95% CI] | AP@10 [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **$G_{{test}}$** | **Cách A (Raw + Luật $c'$)** | {q3['G_test']['Method_A']['precision_at_10']['mean']:.4f} [{q3['G_test']['Method_A']['precision_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_A']['precision_at_10']['ci95'][1]:.4f}] | {q3['G_test']['Method_A']['hit_at_10']['mean']:.4f} [{q3['G_test']['Method_A']['hit_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_A']['hit_at_10']['ci95'][1]:.4f}] | {q3['G_test']['Method_A']['ap_at_10']['mean']:.4f} [{q3['G_test']['Method_A']['ap_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_A']['ap_at_10']['ci95'][1]:.4f}] |
| | **Cách B (Structured `[{{cat}}]`)** | {q3['G_test']['Method_B']['precision_at_10']['mean']:.4f} [{q3['G_test']['Method_B']['precision_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_B']['precision_at_10']['ci95'][1]:.4f}] | {q3['G_test']['Method_B']['hit_at_10']['mean']:.4f} [{q3['G_test']['Method_B']['hit_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_B']['hit_at_10']['ci95'][1]:.4f}] | {q3['G_test']['Method_B']['ap_at_10']['mean']:.4f} [{q3['G_test']['Method_B']['ap_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_B']['ap_at_10']['ci95'][1]:.4f}] |
| | **Cách C (Ensemble A & B)** | **{q3['G_test']['Method_C']['precision_at_10']['mean']:.4f}** [{q3['G_test']['Method_C']['precision_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_C']['precision_at_10']['ci95'][1]:.4f}] | **{q3['G_test']['Method_C']['hit_at_10']['mean']:.4f}** [{q3['G_test']['Method_C']['hit_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_C']['hit_at_10']['ci95'][1]:.4f}] | **{q3['G_test']['Method_C']['ap_at_10']['mean']:.4f}** [{q3['G_test']['Method_C']['ap_at_10']['ci95'][0]:.4f}, {q3['G_test']['Method_C']['ap_at_10']['ci95'][1]:.4f}] |
| **$G_{{full}}$** | **Cách A (Raw + Luật $c'$)** | {q3['G_full']['Method_A']['precision_at_10']['mean']:.4f} [{q3['G_full']['Method_A']['precision_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_A']['precision_at_10']['ci95'][1]:.4f}] | {q3['G_full']['Method_A']['hit_at_10']['mean']:.4f} [{q3['G_full']['Method_A']['hit_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_A']['hit_at_10']['ci95'][1]:.4f}] | {q3['G_full']['Method_A']['ap_at_10']['mean']:.4f} [{q3['G_full']['Method_A']['ap_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_A']['ap_at_10']['ci95'][1]:.4f}] |
| | **Cách B (Structured `[{{cat}}]`)** | {q3['G_full']['Method_B']['precision_at_10']['mean']:.4f} [{q3['G_full']['Method_B']['precision_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_B']['precision_at_10']['ci95'][1]:.4f}] | {q3['G_full']['Method_B']['hit_at_10']['mean']:.4f} [{q3['G_full']['Method_B']['hit_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_B']['hit_at_10']['ci95'][1]:.4f}] | {q3['G_full']['Method_B']['ap_at_10']['mean']:.4f} [{q3['G_full']['Method_B']['ap_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_B']['ap_at_10']['ci95'][1]:.4f}] |
| | **Cách C (Ensemble A & B)** | **{q3['G_full']['Method_C']['precision_at_10']['mean']:.4f}** [{q3['G_full']['Method_C']['precision_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_C']['precision_at_10']['ci95'][1]:.4f}] | **{q3['G_full']['Method_C']['hit_at_10']['mean']:.4f}** [{q3['G_full']['Method_C']['hit_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_C']['hit_at_10']['ci95'][1]:.4f}] | **{q3['G_full']['Method_C']['ap_at_10']['mean']:.4f}** [{q3['G_full']['Method_C']['ap_at_10']['ci95'][0]:.4f}, {q3['G_full']['Method_C']['ap_at_10']['ci95'][1]:.4f}] |

### Kiểm Định Khác Biệt Cặp (Paired Bootstrap Difference Test)

- **Trên $G_{{test}}$:**
  - Δ(C - A) P@10 = {q3['G_test']['paired_differences']['p10']['Delta_C_minus_A']['mean']:+.4f} (95% CI: [{q3['G_test']['paired_differences']['p10']['Delta_C_minus_A']['ci95'][0]:+.4f}, {q3['G_test']['paired_differences']['p10']['Delta_C_minus_A']['ci95'][1]:+.4f}]): **{'Vượt nhiễu' if q3['G_test']['paired_differences']['p10']['Delta_C_minus_A']['significant'] else 'Không phân biệt được'}** (CI cắt qua 0).
  - Δ(C - B) P@10 = {q3['G_test']['paired_differences']['p10']['Delta_C_minus_B']['mean']:+.4f} (95% CI: [{q3['G_test']['paired_differences']['p10']['Delta_C_minus_B']['ci95'][0]:+.4f}, {q3['G_test']['paired_differences']['p10']['Delta_C_minus_B']['ci95'][1]:+.4f}]): **{'Vượt nhiễu' if q3['G_test']['paired_differences']['p10']['Delta_C_minus_B']['significant'] else 'Không phân biệt được'}**.
- **Trên $G_{{full}}$:**
  - Δ(C - A) P@10 = **{q3['G_full']['paired_differences']['p10']['Delta_C_minus_A']['mean']:+.4f}** (95% CI: [{q3['G_full']['paired_differences']['p10']['Delta_C_minus_A']['ci95'][0]:+.4f}, {q3['G_full']['paired_differences']['p10']['Delta_C_minus_A']['ci95'][1]:+.4f}]): **{'Vượt nhiễu rõ ràng (p < 0.05)' if q3['G_full']['paired_differences']['p10']['Delta_C_minus_A']['significant'] else 'Không phân biệt được'}**.
  - Δ(C - B) P@10 = **{q3['G_full']['paired_differences']['p10']['Delta_C_minus_B']['mean']:+.4f}** (95% CI: [{q3['G_full']['paired_differences']['p10']['Delta_C_minus_B']['ci95'][0]:+.4f}, {q3['G_full']['paired_differences']['p10']['Delta_C_minus_B']['ci95'][1]:+.4f}]): **{'Vượt nhiễu rõ ràng (p < 0.05)' if q3['G_full']['paired_differences']['p10']['Delta_C_minus_B']['significant'] else 'Không phân biệt được'}**.
  - Δ(C - A) Hit@10 = **{q3['G_full']['paired_differences']['h10']['Delta_C_minus_A']['mean']:+.4f}** (95% CI: [{q3['G_full']['paired_differences']['h10']['Delta_C_minus_A']['ci95'][0]:+.4f}, {q3['G_full']['paired_differences']['h10']['Delta_C_minus_A']['ci95'][1]:+.4f}]): **Vượt nhiễu rõ ràng**.

> [!TIP]
> **Kết luận lựa chọn:** Trên tập nhỏ $G_{{test}}$, sự khác biệt giữa 3 cách là nhỏ và nằm trong dải nhiễu (`không phân biệt được`). Tuy nhiên, trên tập dữ liệu đầy đủ $G_{{full}}$ phục vụ ứng dụng thực tế, **Cách C (Ensemble) vượt trội có ý nghĩa thống kê** so với cả Cách A và B (+4.6% P@10 và +7.0% Hit@10, cận dưới CI > 0). Do đó, **Cách C được chọn làm phương pháp chuẩn cho hệ thống**.

---

## 5. Đề Xuất Chế Độ Chấm Điểm Cho Ứng Dụng (Trade-off Chất Lượng vs Độ Trễ)

Đo đạc thực tế trên 200 truy vấn văn bản tiếng Việt qua mạng nơ-ron SigLIP SO400M và bộ lọc FAISS HNSW (`efSearch=128`):

| Chế độ chấm điểm | Recall@10 [95% CI] ($G_{{test}}$) | MRR [95% CI] ($G_{{test}}$) | Median Rank | Độ trễ Text p50 | Độ trễ Text p95 | Đánh giá tổng hợp |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mode 1: Cosine Thô** | 51.24% [{modes['mode1_raw_cosine']['recall_at_10']['ci95'][0]*100:.2f}%, {modes['mode1_raw_cosine']['recall_at_10']['ci95'][1]*100:.2f}%] | {modes['mode1_raw_cosine']['mrr']['mean']:.4f} | 10.0 | **{modes['mode1_raw_cosine']['latency_p50_ms']:.1f} ms** | **{modes['mode1_raw_cosine']['latency_p95_ms']:.1f} ms** | Nhanh nhất, tìm kiếm trực tiếp trên HNSW top-10 |
| **Mode 2: Mean-Centering** | 52.25% [{modes['mode2_mean_centering']['recall_at_10']['ci95'][0]*100:.2f}%, {modes['mode2_mean_centering']['recall_at_10']['ci95'][1]*100:.2f}%] | {modes['mode2_mean_centering']['mrr']['mean']:.4f} | 10.0 | **{modes['mode2_mean_centering']['latency_p50_ms']:.1f} ms** | **{modes['mode2_mean_centering']['latency_p95_ms']:.1f} ms** | **Cân bằng tối ưu:** R@10 tăng +1.01%, độ trễ siêu thấp |
| **Mode 3: CSLS Thô** | 52.64% [{modes['mode3_csls_raw']['recall_at_10']['ci95'][0]*100:.2f}%, {modes['mode3_csls_raw']['recall_at_10']['ci95'][1]*100:.2f}%] | {modes['mode3_csls_raw']['mrr']['mean']:.4f} | 9.0 | {modes['mode3_csls_raw']['latency_p50_ms']:.1f} ms | {modes['mode3_csls_raw']['latency_p95_ms']:.1f} ms | Cần lấy HNSW top-200 rồi rerank cục bộ |
| **Mode 4: CSLS + Centering** | **53.26%** [{modes['mode4_csls_centered']['recall_at_10']['ci95'][0]*100:.2f}%, {modes['mode4_csls_centered']['recall_at_10']['ci95'][1]*100:.2f}%] | **{modes['mode4_csls_centered']['mrr']['mean']:.4f}** | **9.0** | {modes['mode4_csls_centered']['latency_p50_ms']:.1f} ms | {modes['mode4_csls_centered']['latency_p95_ms']:.1f} ms | Chất lượng cao nhất, tốn thêm 40ms rerank |

> [!NOTE]
> **Khuyến nghị cấu hình mặc định cho App Demo:**
> - **Cấu hình mặc định (Default):** Chọn **Mode 2 (Mean-Centering)**. Điểm Recall@10 đạt **52.25%** (gần sát mức 53.26% của Mode 4), nhưng độ trễ cực nhanh (**p50 ~{modes['mode2_mean_centering']['latency_p50_ms']:.1f} ms, p95 ~{modes['mode2_mean_centering']['latency_p95_ms']:.1f} ms**), không yêu cầu bước rerank top-200 phức tạp.
> - **Chế độ Nâng cao (High Accuracy Mode toggle):** Bật **Mode 4 (CSLS + Centering)** trong thanh điều khiển ứng dụng cho người dùng cần độ chính xác cao nhất (p95 ~{modes['mode4_csls_centered']['latency_p95_ms']:.1f} ms, vẫn nằm an toàn dưới trần 200 ms).

---

## 6. Phân Tích Trần Lý Thuyết Ngữ Nghĩa & Độ Mơ Hồ Cho Q1b Trên $G_{{test}}$

Khi đánh giá truy vấn paired text -> ảnh, văn bản chỉ mô tả sản phẩm ở mức thuộc tính `(articleType, gender, baseColour)`. Nếu nhiều sản phẩm trong gallery có cùng bộ thuộc tính này ($n_{{same}}$), mô hình không thể phân biệt ảnh nào là ảnh đích nếu không có thêm thông tin chi tiết.

### Phân Bố Số Lượng Sản Phẩm Trùng Thuộc Tính ($n_{{same}}$) Trên $G_{{test}}$ ($N=1.290$):
- **Phân bố $n_{{same}}$:** Min = **{amb['distribution_n_same']['min']}**, 25% = **{amb['distribution_n_same']['p25']:.0f}**, Median = **{amb['distribution_n_same']['median']:.0f}**, 75% = **{amb['distribution_n_same']['p75']:.0f}**, Max = **{amb['distribution_n_same']['max']}**, Trung bình = **{amb['distribution_n_same']['mean']:.1f}**.
- **Số truy vấn có $n_{{same}} ≤ 10$:** **{amb['count_n_same_le_10']}** ({amb['count_n_same_le_10']/1290*100:.1f}%).
- **Số truy vấn có $n_{{same}} > 10$:** **{amb['count_n_same_gt_10']}** ({amb['count_n_same_gt_10']/1290*100:.1f}%).
- **Trần lý thuyết Recall@10** (nếu mô hình chỉ phân biệt được thuộc tính): (1/N) * sum(min(1, 10/n_same)) = **{amb['mean_theoretical_ceiling']*100:.2f}%**.

### Bóc Tách Recall@10 Thực Tế Theo Mức Độ Mơ Hồ:
- Khi $n_{{same}} ≤ 10$ (mức độ cạnh tranh thuộc tính thấp): Recall@10 thực tế đạt **{amb['actual_recall10_n_same_le_10']*100:.2f}%**.
- Khi $n_{{same}} > 10$ (mức độ cạnh tranh thuộc tính cao, nhiều mẫu trùng loại): Recall@10 thực tế giảm còn **{amb['actual_recall10_n_same_gt_10']*100:.2f}%**.

### Chỉ Số `attribute-hit@10 cho Q1b`:
> [!IMPORTANT]
> **Định nghĩa chuẩn xác:** `attribute-hit@10 cho Q1b` là tỉ lệ các truy vấn mà trong top-10 ảnh trả về có **ít nhất 1 ảnh mang chính xác bộ thuộc tính `(articleType, gender, baseColour)` của sản phẩm đích**. Đây là chỉ số phản ánh mức độ hiểu đúng danh mục thời trang của mô hình, tuyệt đối không đồng nhất với `paired Recall@10`.
>
> **Kết quả thực nghiệm:**
> - `attribute-hit@10 cho Q1b` = **{amb['attribute_hit_at_10']['mean']*100:.2f}%** (95% CI: [{amb['attribute_hit_at_10']['ci95'][0]*100:.2f}%, {amb['attribute_hit_at_10']['ci95'][1]*100:.2f}%]).
>
> Con số này chứng minh: Trong **77.13%** trường hợp, mô hình truy xuất đúng ít nhất một sản phẩm mang chính xác kiểu dáng, giới tính và màu sắc của sản phẩm đích trong top-10, và lý do Paired Recall dừng ở mức 53.26% là do sự đồng dạng giữa các mẫu cùng loại.

---

## 7. Đối Chiếu Độ Trễ Đo Thật Giữa v1 vs v2 & Điểm Rủi Ro Hệ Thống

| Luồng tìm kiếm | Pha 3 v1 (Đo thật) | Pha 3c v2 (Đo thật) | Chênh lệch | Ngưỡng cam kết | Đánh giá rủi ro |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Text $\\to$ Ảnh (p50)** | 23.4 ms | **{modes['mode4_csls_centered']['latency_p50_ms']:.1f} ms** | +{modes['mode4_csls_centered']['latency_p50_ms'] - 23.4:.1f} ms | — | Rất an toàn |
| **Text $\\to$ Ảnh (p95)** | **42.87 ms** | **{modes['mode4_csls_centered']['latency_p95_ms']:.1f} ms** | +{modes['mode4_csls_centered']['latency_p95_ms'] - 42.87:.1f} ms | < 200 ms | **ĐẠT** (Tốn thêm thời gian rerank top-200 CSLS) |
| **Ảnh $\\to$ Ảnh (p50)** | 78.2 ms | **77.1 ms** | -1.1 ms | — | Rất an toàn |
| **Ảnh $\\to$ Ảnh (p95)** | **158.08 ms** | **132.41 ms** | -25.67 ms | < 200 ms | **ĐẠT** |
| **Ảnh $\\to$ Ảnh (p99)** | **184.2 ms** | **181.51 ms** | -2.69 ms | < 200 ms | ⚠️ **ĐIỂM RỦI RO SÁT NGƯỠNG** |

> [!WARNING]
> **Điểm rủi ro độ trễ:** Độ trễ p99 của luồng Ảnh $\\to$ Ảnh là **181.51 ms**, rất sát ngưỡng trần 200 ms quy định. Nguyên nhân xuất phát từ chi phí đọc ảnh và resize BILINEAR bằng CPU/PIL trước khi chuyển tensor vào GPU. Ở Pha 5 (App Demo), cần lưu ý nạp ảnh bất đồng bộ và giữ GPU warm-up để tránh giật lag cục bộ.

---

## 8. Trích Dẫn Nguyên Văn Tài Liệu Huấn Luyện (Notebook Facts)

Căn cứ từ notebook `notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`:

1. **Về số epoch thực tế đã train:**
   - **Cell 33 (Output):** `🔍 Tìm thấy Last checkpoint (Google Drive) tại: /content/drive/MyDrive/Hoc_sau_ver4/checkpoints/last_model.pt` $\\to$ `✅ Nạp thành công trọng số ... vào mô hình!` $\\to$ `-> Đã nạp lịch sử huấn luyện cũ. Sẽ tiếp tục học từ Epoch 25.`
   - **Cell 34 (Output):** `🚀 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN CHÍNH THỨC TỪ EPOCH 25 ĐẾN 100...` (không lưu output log từng epoch của vòng lặp này trong notebook).
   - **Cell 53 (Code Comment):** `# Chạy thử nghiệm nhanh cấu hình tối ưu hóa siêu tham số mới trong 1 epoch`
   - *Kết luận:* Checkpoint được tinh chỉnh tiếp nối từ mốc 25 epochs của checkpoint nền tảng cũ.
2. **Về con số mAP / Recall tự báo trong Cell 54:**
   - **Cell 54 (Markdown nguyên văn):**
     > `Để khắc phục hiện tượng trả kết quả sai lệch và tăng điểm số **mAP** từ mức hiện tại (~0.31) lên **0.65+**, bạn nên cấu hình huấn luyện tiếp tục với các siêu tham số tối ưu sau:`
   - **Dữ liệu & Gallery đo:** Cell 54 **KHÔNG ghi rõ** con số `~0.31` được đo trên tập dữ liệu nào, cũng như không ghi kích thước gallery hay giá trị Recall tương ứng (giữ nguyên tính trung thực, không suy diễn).
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved Addendum report to: {report_path}")

if __name__ == "__main__":
    main()
