"""
Phase 3 Evaluation Engine (src/evaluate_phase3.py)
Implements all requirements for Phase 3 Phần B:
- Q1 Paired text -> image (Recall@1/5/10, MRR, Median Rank, Bootstrap 95% CI) on G_test and G_full
- Q2 Attribute queries in 3 text modes: (a) accented, (b) raw unaccented, (c) restored unaccented
- Q3 Handwritten queries in 3 text modes (strict and family relevance)
- Q4 Image -> image retrieval (leave-one-out, rank-1 excluded)
- Subgroup breakdown by masterCategory and articleType
- Comparison of efSearch = 128 vs efSearch = 64
- Filtered retrieval evaluation (parse_query, Precision@10, relaxation rate)
- Error analysis (20 error cases, top 15 category confusions, color confusions, accessories)
- Contact sheets for 30 handwritten queries + manual ratings template
- End-to-end latency benchmark (p50, p95, p99)
- Outputs: eval/results.json, reports/phase3_eval.md, updated reports/run_manifest.json
"""
import os
import sys
import json
import time
import pathlib
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import faiss
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
from src.vi_vocab import remove_diacritics
from src.vi_restore import restore_diacritics
from src.query_parser import parse_query
from src.model_siglip import load_siglip_inference
from src.search import compute_file_sha256, verify_manifest


def bootstrap_ci(scores: np.ndarray, n_bootstraps: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    mean_val = float(np.mean(scores))
    boot_means = []
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    for _ in range(n_bootstraps):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(sample))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper


def compute_ap_at_k(retrieved_ids: List[str], relevant_ids: set, k: int = 10) -> float:
    """Computes Average Precision at K normalized by min(R, K)."""
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


def compute_precision_and_hit(retrieved_ids: List[str], relevant_ids: set, k: int = 10) -> Tuple[float, float]:
    """Computes Precision@K and Hit@K."""
    if not retrieved_ids:
        return 0.0, 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    p_at_k = hits / k
    hit_at_k = 1.0 if hits > 0 else 0.0
    return p_at_k, hit_at_k


class Phase3Evaluator:
    def __init__(self, config_path: str = "config.yaml"):
        print("=== Initializing Phase 3 Evaluator ===")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Hardware execution device: {self.device}")

        # 1. Load Manifest & Verify Integrity
        self.manifest = verify_manifest("index")
        print("Index manifest verified.")

        # 2. Load Embeddings and ID Map
        print("Loading precomputed image embeddings and metadata...")
        self.embeddings_full = np.load("index/image_embeddings.npy")  # [38646, 1152]
        with open("index/id_map.json", "r", encoding="utf-8") as f:
            self.id_map_full = json.load(f)

        with open("index/splits.json", "r", encoding="utf-8") as f:
            self.splits = json.load(f)

        self.test_indices = self.splits["test"]
        self.embeddings_test = self.embeddings_full[self.test_indices]  # [1290, 1152]
        self.id_map_test = [self.id_map_full[idx] for idx in self.test_indices]
        self.test_item_ids = [str(item["item_id"]) for item in self.id_map_test]
        self.test_id_to_idx = {item_id: i for i, item_id in enumerate(self.test_item_ids)}

        print(f"Gallery loaded: G_test={len(self.id_map_test):,}, G_full={len(self.id_map_full):,}")

        # 3. Load FAISS HNSW Index
        print("Loading FAISS HNSW index...")
        self.index_hnsw = faiss.read_index("index/faiss_hnsw_index.bin")
        self.index_hnsw.hnsw.efSearch = 128

        # 4. Load Evaluation Datasets & Mappings
        with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
            self.testset = json.load(f)

        with open("eval/families.json", "r", encoding="utf-8") as f:
            self.families = json.load(f)

        with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
            self.handwritten_queries = json.load(f)

        # 5. Load SigLIP Model (Inference Mode)
        print("Loading SigLIP model for text query encoding...")
        self.model = load_siglip_inference(
            checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
            device=self.device,
            fp16=(self.device == "cuda")
        )
        self.processor = self.model.processor
        print("Evaluator initialization completed successfully.\n")

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encodes a list of text queries using SigLIP text tower."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inp = self.processor(
                text=batch,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inp["input_ids"].to(self.device)
            attention_mask = inp.get("attention_mask").to(self.device) if "attention_mask" in inp else None

            with torch.no_grad():
                emb = self.model.encode_text(input_ids, attention_mask)
            all_vecs.append(emb.float().cpu().numpy())

        vecs = np.concatenate(all_vecs, axis=0)
        faiss.normalize_L2(vecs)
        return vecs

    def search_g_test_exact(self, query_vecs: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Exact search on G_test using inner product."""
        # query_vecs: [Q, 1152], self.embeddings_test: [1290, 1152]
        scores = np.matmul(query_vecs, self.embeddings_test.T)  # [Q, 1290]
        # Sort descending
        top_k = min(top_k, scores.shape[1])
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, indices, axis=1)
        return sorted_scores, indices

    def search_g_full_hnsw(self, query_vecs: np.ndarray, top_k: int = 100, ef_search: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """HNSW search on G_full."""
        self.index_hnsw.hnsw.efSearch = ef_search
        scores, indices = self.index_hnsw.search(query_vecs, top_k)
        return scores, indices

    def evaluate_q1_paired(self, ef_search: int = 128) -> Dict[str, Any]:
        """Evaluates Q1 Paired Retrieval on G_test and G_full."""
        print(f"\n--- Running Q1 Paired Retrieval Evaluation (efSearch={ef_search}) ---")
        q1_items = self.testset["q1_paired_queries"]
        texts = [q["query_text"] for q in q1_items]
        target_ids = [q["target_id"] for q in q1_items]

        print(f"Encoding {len(texts)} Q1 paired queries...")
        query_vecs = self.encode_texts(texts)

        # 1. Search on G_test (Exact)
        scores_test, indices_test = self.search_g_test_exact(query_vecs, top_k=100)
        
        # 2. Search on G_full (HNSW)
        scores_full, indices_full = self.search_g_full_hnsw(query_vecs, top_k=100, ef_search=ef_search)

        def compute_paired_metrics(indices_matrix, id_lookup, gallery_name):
            ranks = []
            recalls_1 = []
            recalls_5 = []
            recalls_10 = []
            mrrs = []
            item_metrics = []

            for i, target_id in enumerate(target_ids):
                row_indices = indices_matrix[i]
                retrieved_ids = [str(id_lookup[idx]["item_id"]) for idx in row_indices if idx >= 0]
                
                try:
                    rank = retrieved_ids.index(target_id) + 1
                except ValueError:
                    rank = 999999  # > top_k

                ranks.append(rank)
                r1 = 1.0 if rank == 1 else 0.0
                r5 = 1.0 if rank <= 5 else 0.0
                r10 = 1.0 if rank <= 10 else 0.0
                mrr = 1.0 / rank if rank <= 100 else 0.0

                recalls_1.append(r1)
                recalls_5.append(r5)
                recalls_10.append(r10)
                mrrs.append(mrr)

                # Store per-item metric for subgroup breakdown
                cat_item = id_lookup[self.test_indices[i] if gallery_name == "G_full" else i]
                item_metrics.append({
                    "item_id": target_id,
                    "masterCategory": cat_item.get("masterCategory", "Unknown"),
                    "articleType": cat_item.get("articleType", "Unknown"),
                    "gender": cat_item.get("gender", "Unknown"),
                    "rank": rank,
                    "r10": r10,
                    "mrr": mrr
                })

            recalls_10 = np.array(recalls_10)
            mrrs = np.array(mrrs)

            r1_mean = float(np.mean(recalls_1))
            r5_mean = float(np.mean(recalls_5))
            r10_mean, r10_low, r10_high = bootstrap_ci(recalls_10)
            mrr_mean, mrr_low, mrr_high = bootstrap_ci(mrrs)
            med_rank = float(np.median(ranks))

            return {
                "n_queries": len(target_ids),
                "recall@1": r1_mean,
                "recall@5": r5_mean,
                "recall@10": r10_mean,
                "recall@10_ci": [r10_low, r10_high],
                "mrr": mrr_mean,
                "mrr_ci": [mrr_low, mrr_high],
                "median_rank": med_rank,
                "item_metrics": item_metrics
            }

        res_test = compute_paired_metrics(indices_test, self.id_map_test, "G_test")
        res_full = compute_paired_metrics(indices_full, self.id_map_full, "G_full")

        print(f"G_test Paired Results -> Recall@10: {res_test['recall@10']:.4f} (95% CI: [{res_test['recall@10_ci'][0]:.4f}, {res_test['recall@10_ci'][1]:.4f}]), MRR: {res_test['mrr']:.4f}, Median Rank: {res_test['median_rank']}")
        print(f"G_full Paired Results -> Recall@10: {res_full['recall@10']:.4f} (95% CI: [{res_full['recall@10_ci'][0]:.4f}, {res_full['recall@10_ci'][1]:.4f}]), MRR: {res_full['mrr']:.4f}, Median Rank: {res_full['median_rank']}")

        return {
            "g_test": res_test,
            "g_full": res_full
        }

    def evaluate_attribute_queries(
        self,
        query_list: List[Dict[str, Any]],
        query_type_label: str = "Q2",
        ef_search: int = 128
    ) -> Dict[str, Any]:
        """
        Evaluates attribute queries (Q2 or Q3) in 3 text modes:
        (a) accented, (b) raw unaccented, (c) restored unaccented
        across strict and family relevance on both G_test and G_full.
        """
        print(f"\n--- Running {query_type_label} Attribute Queries Evaluation (Total: {len(query_list)}) ---")

        # Prepare 3 query variants
        queries_accented = []
        queries_unaccented = []
        queries_restored = []

        for q in query_list:
            raw_text = q["query"] if "query" in q else q["query_text"]
            # Mode A: Accented (or original)
            queries_accented.append(raw_text)
            # Mode B: Raw unaccented
            unaccented_text = remove_diacritics(raw_text)
            queries_unaccented.append(unaccented_text)
            # Mode C: Restored
            restored_text = restore_diacritics(unaccented_text)
            queries_restored.append(restored_text)

        modes = [
            ("accented", queries_accented),
            ("unaccented", queries_unaccented),
            ("restored", queries_restored)
        ]

        mode_results = {}

        for mode_name, texts in modes:
            print(f"Encoding {query_type_label} in mode '{mode_name}'...")
            vecs = self.encode_texts(texts)

            # Search G_test
            _, indices_test = self.search_g_test_exact(vecs, top_k=20)
            # Search G_full
            _, indices_full = self.search_g_full_hnsw(vecs, top_k=20, ef_search=ef_search)

            def eval_relevance(indices_matrix, id_lookup, gallery_key):
                strict_p10, strict_hit10, strict_ap10 = [], [], []
                family_p10, family_hit10, family_ap10 = [], [], []

                for i, q in enumerate(query_list):
                    retrieved_ids = [str(id_lookup[idx]["item_id"]) for idx in indices_matrix[i] if idx >= 0]

                    if "target_ids_strict" in q:
                        strict_relevant = set(q["target_ids_strict"][gallery_key])
                        family_relevant = set(q["target_ids_family"][gallery_key])
                    else:
                        # Q3 handwritten query: derive relevance from query spec and families
                        spec = q["spec"]
                        spec_arts = set(spec.get("articleType") or [])
                        spec_gen = spec.get("gender")
                        spec_cols = set(spec.get("baseColour") or [])

                        compat_gens = set(self.families["gender_compatibility"].get(spec_gen, [spec_gen])) if spec_gen else None

                        # Strict relevant items in id_lookup
                        strict_relevant = set()
                        family_relevant = set()

                        # Build family sets
                        fam_cols = set()
                        for c in spec_cols:
                            fam_cols.update(self.families["colour_families"].get(c, [c]))

                        fam_arts = set()
                        for a in spec_arts:
                            fam_arts.update(self.families["article_type_families"].get(a, [a]))

                        for item in id_lookup:
                            iid = str(item["item_id"])
                            i_art = item.get("articleType")
                            i_gen = item.get("gender")
                            i_col = item.get("baseColour")

                            # Strict check
                            art_match = (i_art in spec_arts) if spec_arts else True
                            gen_match = (i_gen in compat_gens) if compat_gens else True
                            col_match = (i_col in spec_cols) if spec_cols else True
                            if art_match and gen_match and col_match:
                                strict_relevant.add(iid)

                            # Family check
                            f_art_match = (i_art in fam_arts) if fam_arts else True
                            f_col_match = (i_col in fam_cols) if fam_cols else True
                            if f_art_match and gen_match and f_col_match:
                                family_relevant.add(iid)

                    # Compute metrics
                    sp10, shit10 = compute_precision_and_hit(retrieved_ids, strict_relevant, k=10)
                    sap10 = compute_ap_at_k(retrieved_ids, strict_relevant, k=10)
                    strict_p10.append(sp10)
                    strict_hit10.append(shit10)
                    strict_ap10.append(sap10)

                    fp10, fhit10 = compute_precision_and_hit(retrieved_ids, family_relevant, k=10)
                    fap10 = compute_ap_at_k(retrieved_ids, family_relevant, k=10)
                    family_p10.append(fp10)
                    family_hit10.append(fhit10)
                    family_ap10.append(fap10)

                return {
                    "strict": {
                        "precision@10": float(np.mean(strict_p10)),
                        "hit@10": float(np.mean(strict_hit10)),
                        "ap@10": float(np.mean(strict_ap10))
                    },
                    "family": {
                        "precision@10": float(np.mean(family_p10)),
                        "hit@10": float(np.mean(family_hit10)),
                        "ap@10": float(np.mean(family_ap10))
                    }
                }

            mode_results[mode_name] = {
                "g_test": eval_relevance(indices_test, self.id_map_test, "g_test"),
                "g_full": eval_relevance(indices_full, self.id_map_full, "g_full")
            }

            p10_s = mode_results[mode_name]["g_test"]["strict"]["precision@10"]
            p10_f = mode_results[mode_name]["g_test"]["family"]["precision@10"]
            print(f"  Mode [{mode_name:<11}] G_test -> Strict P@10: {p10_s:.4f} | Family P@10: {p10_f:.4f}")

        return mode_results

    def evaluate_q4_image_to_image(self) -> Dict[str, Any]:
        """
        Evaluates Q4 Image-to-Image retrieval on G_test.
        Excludes rank 1 (self) to evaluate true visual nearest-neighbor retrieval.
        """
        print("\n--- Running Q4 Image-to-Image Retrieval Evaluation (1,290 Test Images) ---")
        # Test embeddings query test embeddings
        scores, indices = self.search_g_test_exact(self.embeddings_test, top_k=25)

        strict_p10, strict_hit10, strict_ap10 = [], [], []
        color_p10, color_hit10, color_ap10 = [], [], []

        gender_compat = self.families["gender_compatibility"]

        for i in range(len(self.id_map_test)):
            query_item = self.id_map_test[i]
            q_id = str(query_item["item_id"])
            q_art = query_item.get("articleType")
            q_gen = query_item.get("gender")
            q_col = query_item.get("baseColour")
            compat_gens = set(gender_compat.get(q_gen, [q_gen]))

            # Exclude self from retrieved list
            row_indices = indices[i]
            retrieved_ids = [str(self.id_map_test[idx]["item_id"]) for idx in row_indices if idx >= 0]
            if retrieved_ids and retrieved_ids[0] == q_id:
                retrieved_ids = retrieved_ids[1:11]
            else:
                retrieved_ids = [rid for rid in retrieved_ids if rid != q_id][:10]

            # Relevance 1: Same articleType + compatible gender
            strict_relevant = {
                str(it["item_id"]) for it in self.id_map_test
                if str(it["item_id"]) != q_id and it.get("articleType") == q_art and it.get("gender") in compat_gens
            }

            # Relevance 2: Same articleType + compatible gender + same baseColour
            color_relevant = {
                str(it["item_id"]) for it in self.id_map_test
                if str(it["item_id"]) != q_id and it.get("articleType") == q_art and it.get("gender") in compat_gens and it.get("baseColour") == q_col
            }

            sp10, shit10 = compute_precision_and_hit(retrieved_ids, strict_relevant, k=10)
            sap10 = compute_ap_at_k(retrieved_ids, strict_relevant, k=10)
            strict_p10.append(sp10)
            strict_hit10.append(shit10)
            strict_ap10.append(sap10)

            cp10, chit10 = compute_precision_and_hit(retrieved_ids, color_relevant, k=10)
            cap10 = compute_ap_at_k(retrieved_ids, color_relevant, k=10)
            color_p10.append(cp10)
            color_hit10.append(chit10)
            color_ap10.append(cap10)

        res = {
            "n_queries": len(self.id_map_test),
            "same_category_and_gender": {
                "precision@10": float(np.mean(strict_p10)),
                "hit@10": float(np.mean(strict_hit10)),
                "ap@10": float(np.mean(strict_ap10))
            },
            "same_category_gender_and_colour": {
                "precision@10": float(np.mean(color_p10)),
                "hit@10": float(np.mean(color_hit10)),
                "ap@10": float(np.mean(color_ap10))
            }
        }
        print(f"Q4 Results (excl. self) -> Same Category P@10: {res['same_category_and_gender']['precision@10']:.4f}, Same Color P@10: {res['same_category_gender_and_colour']['precision@10']:.4f}")
        return res

    def evaluate_filtered_retrieval(self, query_list: List[Dict[str, Any]], query_type: str = "Q2") -> Dict[str, Any]:
        """Evaluates retrieval WITH attribute post-filtering."""
        print(f"\n--- Running Filtered Retrieval Evaluation for {query_type} ---")
        texts = [q["query"] if "query" in q else q["query_text"] for q in query_list]
        vecs = self.encode_texts(texts)

        # Retrieve top 200 candidates per query from G_full
        _, indices_matrix = self.search_g_full_hnsw(vecs, top_k=200, ef_search=128)

        gender_compat = self.families["gender_compatibility"]

        unfiltered_p10 = []
        filtered_p10 = []
        relaxed_count = 0

        for i, q in enumerate(query_list):
            parsed = parse_query(texts[i])
            parsed_arts = parsed.get("articleType")
            parsed_gen = parsed.get("gender")
            parsed_cols = parsed.get("colour")

            cand_items = [self.id_map_full[idx] for idx in indices_matrix[i] if idx >= 0]
            unfiltered_ids = [str(it["item_id"]) for it in cand_items[:10]]

            # Determine ground truth relevant items
            if "target_ids_strict" in q:
                strict_relevant = set(q["target_ids_strict"]["g_full"])
            else:
                spec = q["spec"]
                spec_arts = set(spec.get("articleType") or [])
                spec_gen = spec.get("gender")
                spec_cols = set(spec.get("baseColour") or [])
                compat_gens = set(gender_compat.get(spec_gen, [spec_gen])) if spec_gen else None

                strict_relevant = set()
                for it in self.id_map_full:
                    iid = str(it["item_id"])
                    art_m = (it.get("articleType") in spec_arts) if spec_arts else True
                    gen_m = (it.get("gender") in compat_gens) if compat_gens else True
                    col_m = (it.get("baseColour") in spec_cols) if spec_cols else True
                    if art_m and gen_m and col_m:
                        strict_relevant.add(iid)

            up10, _ = compute_precision_and_hit(unfiltered_ids, strict_relevant, k=10)
            unfiltered_p10.append(up10)

            # Filter candidates
            def filter_candidates(items, use_art=True, use_gen=True, use_col=True):
                filtered = []
                for it in items:
                    if use_art and parsed_arts and it.get("articleType") not in parsed_arts:
                        continue
                    if use_gen and parsed_gen:
                        compats = gender_compat.get(parsed_gen, [parsed_gen])
                        if it.get("gender") not in compats:
                            continue
                    if use_col and parsed_cols and it.get("baseColour") not in parsed_cols:
                        continue
                    filtered.append(str(it["item_id"]))
                    if len(filtered) >= 10:
                        break
                return filtered

            filtered_ids = filter_candidates(cand_items, True, True, True)
            if len(filtered_ids) < 10:
                relaxed_count += 1
                # Relax colour constraint first
                filtered_ids = filter_candidates(cand_items, True, True, False)
                if len(filtered_ids) < 10:
                    # Relax gender
                    filtered_ids = filter_candidates(cand_items, True, False, False)
                    if len(filtered_ids) < 10:
                        filtered_ids = [str(it["item_id"]) for it in cand_items[:10]]

            fp10, _ = compute_precision_and_hit(filtered_ids, strict_relevant, k=10)
            filtered_p10.append(fp10)

        res = {
            "n_queries": len(query_list),
            "unfiltered_precision@10": float(np.mean(unfiltered_p10)),
            "filtered_precision@10": float(np.mean(filtered_p10)),
            "relaxation_rate": float(relaxed_count / len(query_list))
        }
        print(f"Filtered Results ({query_type}) -> Unfiltered P@10: {res['unfiltered_precision@10']:.4f} | Filtered P@10: {res['filtered_precision@10']:.4f} | Relaxation Rate: {res['relaxation_rate']*100:.1f}%")
        return res

    def perform_error_analysis(self, q1_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes top 20 error cases, category confusions, and close-up accessories."""
        print("\n--- Running Error Analysis & Confusion Diagnostics ---")
        # 1. 20 error cases (paired rank > 10)
        sorted_by_rank = sorted(q1_metrics, key=lambda x: x["rank"], reverse=True)
        top20_errors = sorted_by_rank[:20]

        # 2. Confusion Analysis on Q1 predictions vs ground truth
        # Using top-1 retrieved items in Q1
        cat_confusions = Counter()
        color_confusions = Counter()

        # Let's inspect category confusions on Q1
        q1_items = self.testset["q1_paired_queries"]
        texts = [q["query_text"] for q in q1_items]
        vecs = self.encode_texts(texts)
        _, indices_test = self.search_g_test_exact(vecs, top_k=5)

        accessories_types = {"Watches", "Sunglasses", "Caps", "Belts", "Handbags", "Wallets", "Earrings", "Pendant"}
        acc_total = 0
        acc_errors = 0

        for i, q in enumerate(q1_items):
            true_item = self.id_map_test[i]
            top1_item = self.id_map_test[indices_test[i][0]]

            true_art = true_item.get("articleType")
            top1_art = top1_item.get("articleType")
            true_col = true_item.get("baseColour")
            top1_col = top1_item.get("baseColour")

            if true_art in accessories_types:
                acc_total += 1
                if indices_test[i][0] != i:
                    acc_errors += 1

            if true_art != top1_art:
                cat_confusions[(true_art, top1_art)] += 1

            if true_col != top1_col:
                color_confusions[(true_col, top1_col)] += 1

        top15_cat_conf = [
            {"true": k[0], "predicted": k[1], "count": v}
            for k, v in cat_confusions.most_common(15)
        ]

        top10_col_conf = [
            {"true": k[0], "predicted": k[1], "count": v}
            for k, v in color_confusions.most_common(10)
        ]

        acc_error_rate = acc_errors / acc_total if acc_total > 0 else 0.0

        print(f"Top 15 Category Confusions: {len(top15_cat_conf)} pairs identified.")
        print(f"Accessories error rate: {acc_errors}/{acc_total} ({acc_error_rate*100:.1f}%)")

        return {
            "top20_errors": top20_errors,
            "top15_category_confusions": top15_cat_conf,
            "top10_color_confusions": top10_col_conf,
            "accessories_error_rate": acc_error_rate,
            "accessories_count": acc_total
        }

    def generate_contact_sheets_for_handwritten(self, n_sheets: int = 30):
        """Generates contact sheets for 30 handwritten queries and rating template."""
        print(f"\n--- Generating Contact Sheets for {n_sheets} Handwritten Queries ---")
        selected_queries = self.handwritten_queries[:n_sheets]
        texts = [q["query"] for q in selected_queries]
        vecs = self.encode_texts(texts)

        _, indices_full = self.search_g_full_hnsw(vecs, top_k=10, ef_search=128)

        out_dir = pathlib.Path("reports/contact_sheets")
        out_dir.mkdir(parents=True, exist_ok=True)

        template_rows = []

        thumb_w, thumb_h = 100, 130
        grid_w = thumb_w * 10 + 40
        grid_h = thumb_h + 90

        for i, q in enumerate(selected_queries):
            qid = q["id"]
            query_str = q["query"]
            cat = q["category"]
            row_indices = indices_full[i]

            sheet = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(sheet)

            # Draw header
            header_text = f"[{qid}] Category: {cat} | Query: {query_str}"
            draw.text((15, 10), header_text, fill=(0, 0, 0))

            top10_ids = []

            for col_idx, idx in enumerate(row_indices):
                if idx < 0 or idx >= len(self.id_map_full):
                    continue
                item = self.id_map_full[idx]
                top10_ids.append(str(item["item_id"]))
                img_path = pathlib.Path(item["path"])

                x_offset = 20 + col_idx * thumb_w
                y_offset = 40

                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            img_thumb = img.convert("RGB").resize((thumb_w - 6, thumb_h - 26))
                            sheet.paste(img_thumb, (x_offset, y_offset))
                    except Exception:
                        draw.rectangle([x_offset, y_offset, x_offset + thumb_w - 6, y_offset + thumb_h - 26], fill=(220, 220, 220))
                else:
                    draw.rectangle([x_offset, y_offset, x_offset + thumb_w - 6, y_offset + thumb_h - 26], fill=(220, 220, 220))

                # Draw label
                lbl = f"{item.get('articleType', '')[:8]}\n{item.get('baseColour', '')[:8]}"
                draw.text((x_offset, y_offset + thumb_h - 24), lbl, fill=(50, 50, 50))

            sheet_path = out_dir / f"hw_{qid}.png"
            sheet.save(sheet_path)

            template_rows.append({
                "query_id": qid,
                "category": cat,
                "has_diacritics": q["has_diacritics"],
                "query_text": query_str,
                "top10_retrieved_ids": ";".join(top10_ids),
                "num_relevant_out_of_10": "",  # To be manually scored
                "user_notes": ""
            })

        df_template = pd.DataFrame(template_rows)
        template_csv_path = pathlib.Path("eval/manual_ratings_template.csv")
        df_template.to_csv(template_csv_path, index=False, encoding="utf-8")
        print(f"Saved {n_sheets} contact sheets to {out_dir} and template to {template_csv_path}")

    def benchmark_latency(self, n_queries: int = 200) -> Dict[str, Any]:
        """Measures end-to-end latency p50, p95, p99 after warmup."""
        print(f"\n--- Running End-to-End Latency Benchmark ({n_queries} queries) ---")
        # 1. Text Search Latency
        test_queries = [q["query_text"] for q in self.testset["q1_paired_queries"][:n_queries + 20]]
        
        # Warmup 20 queries
        print("Warming up text retrieval pipeline...")
        for q_text in test_queries[:20]:
            _ = parse_query(q_text)
            vec = self.encode_texts([q_text])
            _, _ = self.search_g_full_hnsw(vec, top_k=10, ef_search=128)

        print("Measuring text retrieval latency...")
        text_times = []
        for q_text in test_queries[20:20 + n_queries]:
            t0 = time.perf_counter()
            _ = parse_query(q_text)
            vec = self.encode_texts([q_text])
            _, _ = self.search_g_full_hnsw(vec, top_k=10, ef_search=128)
            text_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. Image Search Latency
        sample_img_paths = [self.id_map_test[i]["path"] for i in range(20 + n_queries)]
        print("Warming up image retrieval pipeline...")
        for img_path in sample_img_paths[:20]:
            with Image.open(img_path) as img:
                inp = self.processor(images=[img.convert("RGB")], return_tensors="pt")
                pix = inp["pixel_values"].to(self.device)
                if self.device == "cuda":
                    pix = pix.half()
                with torch.no_grad():
                    emb = self.model.encode_image(pix).float().cpu().numpy()
                faiss.normalize_L2(emb)
                _, _ = self.search_g_full_hnsw(emb, top_k=10, ef_search=128)

        print("Measuring image retrieval latency...")
        image_times = []
        for img_path in sample_img_paths[20:20 + n_queries]:
            t0 = time.perf_counter()
            with Image.open(img_path) as img:
                inp = self.processor(images=[img.convert("RGB")], return_tensors="pt")
                pix = inp["pixel_values"].to(self.device)
                if self.device == "cuda":
                    pix = pix.half()
                with torch.no_grad():
                    emb = self.model.encode_image(pix).float().cpu().numpy()
                faiss.normalize_L2(emb)
                _, _ = self.search_g_full_hnsw(emb, top_k=10, ef_search=128)
            image_times.append((time.perf_counter() - t0) * 1000.0)

        res = {
            "text_to_image": {
                "n_queries": n_queries,
                "p50_ms": float(np.percentile(text_times, 50)),
                "p95_ms": float(np.percentile(text_times, 95)),
                "p99_ms": float(np.percentile(text_times, 99)),
                "mean_ms": float(np.mean(text_times))
            },
            "image_to_image": {
                "n_queries": n_queries,
                "p50_ms": float(np.percentile(image_times, 50)),
                "p95_ms": float(np.percentile(image_times, 95)),
                "p99_ms": float(np.percentile(image_times, 99)),
                "mean_ms": float(np.mean(image_times))
            }
        }
        print(f"Latency Text->Image: p50={res['text_to_image']['p50_ms']:.2f} ms | p95={res['text_to_image']['p95_ms']:.2f} ms | p99={res['text_to_image']['p99_ms']:.2f} ms")
        print(f"Latency Image->Image: p50={res['image_to_image']['p50_ms']:.2f} ms | p95={res['image_to_image']['p95_ms']:.2f} ms | p99={res['image_to_image']['p99_ms']:.2f} ms")
        return res


def run_all_eval():
    evaluator = Phase3Evaluator()

    # 1. Q1 Paired (efSearch = 128)
    q1_128 = evaluator.evaluate_q1_paired(ef_search=128)
    # Also evaluate efSearch = 64 on G_full for comparison
    q1_64 = evaluator.evaluate_q1_paired(ef_search=64)

    # 2. Q2 Attribute Queries (61 queries)
    q2_results = evaluator.evaluate_attribute_queries(
        evaluator.testset["q2_attribute_queries"],
        query_type_label="Q2",
        ef_search=128
    )

    # 3. Q3 Handwritten Queries (100 queries)
    q3_results = evaluator.evaluate_attribute_queries(
        evaluator.handwritten_queries,
        query_type_label="Q3",
        ef_search=128
    )

    # 4. Q4 Image-to-Image (1,290 test images)
    q4_results = evaluator.evaluate_q4_image_to_image()

    # 5. Filtered Retrieval Evaluation
    filtered_q2 = evaluator.evaluate_filtered_retrieval(evaluator.testset["q2_attribute_queries"], "Q2")
    filtered_q3 = evaluator.evaluate_filtered_retrieval(evaluator.handwritten_queries, "Q3")

    # 6. Error Analysis
    error_analysis = evaluator.perform_error_analysis(q1_128["g_test"]["item_metrics"])

    # 7. Contact Sheets for 30 Handwritten Queries
    evaluator.generate_contact_sheets_for_handwritten(30)

    # 8. Latency Benchmark
    latency_results = evaluator.benchmark_latency(200)

    # Subgroup breakdown on Q1 (G_test)
    df_items = pd.DataFrame(q1_128["g_test"]["item_metrics"])
    master_breakdown = df_items.groupby("masterCategory").agg(
        n=("r10", "count"),
        recall10=("r10", "mean"),
        mrr=("mrr", "mean")
    ).reset_index().to_dict("records")

    article_breakdown = df_items.groupby("articleType").agg(
        n=("r10", "count"),
        recall10=("r10", "mean"),
        mrr=("mrr", "mean")
    ).reset_index()
    # Sort by count desc
    article_breakdown = article_breakdown.sort_values(by="n", ascending=False).to_dict("records")

    # Compile Final Structured Results
    results_all = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "q1_paired": {
            "g_test": {
                "recall@1": q1_128["g_test"]["recall@1"],
                "recall@5": q1_128["g_test"]["recall@5"],
                "recall@10": q1_128["g_test"]["recall@10"],
                "recall@10_ci": q1_128["g_test"]["recall@10_ci"],
                "mrr": q1_128["g_test"]["mrr"],
                "mrr_ci": q1_128["g_test"]["mrr_ci"],
                "median_rank": q1_128["g_test"]["median_rank"]
            },
            "g_full_ef128": {
                "recall@1": q1_128["g_full"]["recall@1"],
                "recall@5": q1_128["g_full"]["recall@5"],
                "recall@10": q1_128["g_full"]["recall@10"],
                "recall@10_ci": q1_128["g_full"]["recall@10_ci"],
                "mrr": q1_128["g_full"]["mrr"],
                "mrr_ci": q1_128["g_full"]["mrr_ci"],
                "median_rank": q1_128["g_full"]["median_rank"]
            },
            "g_full_ef64": {
                "recall@1": q1_64["g_full"]["recall@1"],
                "recall@5": q1_64["g_full"]["recall@5"],
                "recall@10": q1_64["g_full"]["recall@10"],
                "recall@10_ci": q1_64["g_full"]["recall@10_ci"],
                "mrr": q1_64["g_full"]["mrr"],
                "mrr_ci": q1_64["g_full"]["mrr_ci"],
                "median_rank": q1_64["g_full"]["median_rank"]
            }
        },
        "q2_attributes": q2_results,
        "q3_handwritten": q3_results,
        "q4_image_to_image": q4_results,
        "filtered_retrieval": {
            "q2": filtered_q2,
            "q3": filtered_q3
        },
        "error_analysis": {
            "top20_errors": error_analysis["top20_errors"],
            "top15_category_confusions": error_analysis["top15_category_confusions"],
            "top10_color_confusions": error_analysis["top10_color_confusions"],
            "accessories_error_rate": error_analysis["accessories_error_rate"],
            "accessories_count": error_analysis["accessories_count"]
        },
        "subgroup_breakdown": {
            "master_category": master_breakdown,
            "article_type": article_breakdown
        },
        "latency": latency_results
    }

    # Save eval/results.json
    results_path = pathlib.Path("eval/results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"\nSaved structured results to {results_path}")

    # Generate Markdown Report reports/phase3_eval.md
    generate_markdown_report(results_all)


def generate_markdown_report(res: Dict[str, Any]):
    print("Generating comprehensive Markdown Report reports/phase3_eval.md...")
    q1_t = res["q1_paired"]["g_test"]
    q1_f128 = res["q1_paired"]["g_full_ef128"]
    q1_f64 = res["q1_paired"]["g_full_ef64"]

    q2_t = res["q2_attributes"]["accented"]["g_test"]
    q3_t = res["q3_handwritten"]["accented"]["g_test"]
    q4 = res["q4_image_to_image"]
    lat_txt = res["latency"]["text_to_image"]
    lat_img = res["latency"]["image_to_image"]

    # Target compliance
    pass_recall = q1_t["recall@10"] >= 0.85
    pass_mrr = q1_t["mrr"] >= 0.75
    pass_lat_txt = lat_txt["p95_ms"] < 200.0
    pass_lat_img = lat_img["p95_ms"] < 200.0

    lines = [
        "# Báo Cáo Nghiệm Thu Pha 3: Đánh Giá Cố Định (Fixed Evaluation)",
        f"**Thời gian thực hiện**: {res['timestamp']}",
        "**Checkpoint SigLIP**: `data/checkpoints_siglip-vit/best_model.pt`",
        "**Index FAISS**: `index/faiss_hnsw_index.bin` ($M=32, efConstruction=200$)",
        "",
        "---",
        "",
        "## 1. Bảng Đối Chiếu Chỉ Tiêu Dự Án (Nghiệm Thu)",
        "",
        "| Chỉ tiêu | Mục tiêu | Kết quả thực tế ($G_{test}$) | Kết quả $G_{full}$ (Lạc quan) | Trạng thái |",
        "| :--- | :--- | :--- | :--- | :---: |",
        f"| **Recall@10** (Text $\\to$ Ảnh paired) | $\\ge 0.85$ | **{q1_t['recall@10']:.4f}** (95% CI: [{q1_t['recall@10_ci'][0]:.4f}, {q1_t['recall@10_ci'][1]:.4f}]) | {q1_f128['recall@10']:.4f} (CI: [{q1_f128['recall@10_ci'][0]:.4f}, {q1_f128['recall@10_ci'][1]:.4f}]) | {'✅ ĐẠT' if pass_recall else '❌ KHÔNG ĐẠT'} |",
        f"| **mAP / MRR** (Text $\\to$ Ảnh paired) | $\\ge 0.75$ | **{q1_t['mrr']:.4f}** (95% CI: [{q1_t['mrr_ci'][0]:.4f}, {q1_t['mrr_ci'][1]:.4f}]) | {q1_f128['mrr']:.4f} (CI: [{q1_f128['mrr_ci'][0]:.4f}, {q1_f128['mrr_ci'][1]:.4f}]) | {'✅ ĐẠT' if pass_mrr else '❌ KHÔNG ĐẠT'} |",
        f"| **Độ trễ Text $\\to$ Ảnh** (p95) | $< 200$ ms | **{lat_txt['p95_ms']:.2f} ms** (p50: {lat_txt['p50_ms']:.2f} ms) | — | {'✅ ĐẠT' if pass_lat_txt else '❌ KHÔNG ĐẠT'} |",
        f"| **Độ trễ Ảnh $\\to$ Ảnh** (p95) | $< 200$ ms | **{lat_img['p95_ms']:.2f} ms** (p50: {lat_img['p50_ms']:.2f} ms) | — | {'✅ ĐẠT' if pass_lat_img else '❌ KHÔNG ĐẠT'} |",
        "",
        "> **Ghi chú về Ground Rule Đánh giá**:",
        "> - Checkpoint đã train trên 161k mẫu bao gồm toàn bộ phần lớn ảnh Myntra Kaggle.",
        "> - Theo quy tắc chống rò rỉ tại `AGENTS.md`, số liệu nghiệm thu CHỈ tính trên tập con test độc lập $G_{test}$ ($n=1.290$).",
        "> - Số liệu trên $G_{full}$ ($n=38.646$) chứa các ảnh trong tập train nên được ghi nhận là lạc quan (để tham khảo năng lực phục vụ thực tế của index).",
        "",
        "---",
        "",
        "## 2. Chi Tiết Kết Quả Paired Retrieval (Q1)",
        "",
        "| Bộ Gallery | efSearch | Recall@1 | Recall@5 | Recall@10 | MRR (= mAP) | Median Rank |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **$G_{{test}}$ ($n=1.290$)** | Exact | {q1_t['recall@1']:.4f} | {q1_t['recall@5']:.4f} | {q1_t['recall@10']:.4f} | {q1_t['mrr']:.4f} | {q1_t['median_rank']:.1f} |",
        f"| **$G_{{full}}$ ($n=38.646$)** | 128 | {q1_f128['recall@1']:.4f} | {q1_f128['recall@5']:.4f} | {q1_f128['recall@10']:.4f} | {q1_f128['mrr']:.4f} | {q1_f128['median_rank']:.1f} |",
        f"| **$G_{{full}}$ ($n=38.646$)** | 64 | {q1_f64['recall@1']:.4f} | {q1_f64['recall@5']:.4f} | {q1_f64['recall@10']:.4f} | {q1_f64['mrr']:.4f} | {q1_f64['median_rank']:.1f} |",
        "",
        "---",
        "",
        "## 3. Đánh Giá Truy Vấn Thuộc Tính Tự Nhiên (Q2) & Ba Chế Độ Văn Bản",
        "",
        "| Chế độ văn bản | Mức độ | Precision@10 ($G_{test}$) | Hit@10 ($G_{test}$) | AP@10 ($G_{test}$) | P@10 ($G_{full}$ - Lạc quan) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for mode in ["accented", "unaccented", "restored"]:
        m_t = res["q2_attributes"][mode]["g_test"]
        m_f = res["q2_attributes"][mode]["g_full"]
        lbl = "(a) Có dấu" if mode == "accented" else ("(b) Không dấu thô" if mode == "unaccented" else "(c) Khôi phục dấu")
        lines.append(f"| **{lbl}** | Strict | {m_t['strict']['precision@10']:.4f} | {m_t['strict']['hit@10']:.4f} | {m_t['strict']['ap@10']:.4f} | {m_f['strict']['precision@10']:.4f} |")
        lines.append(f"| | Family | {m_t['family']['precision@10']:.4f} | {m_t['family']['hit@10']:.4f} | {m_t['family']['ap@10']:.4f} | {m_f['family']['precision@10']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Đánh Giá Bộ 100 Truy Vấn Viết Tay (Q3)",
        "",
        "| Chế độ văn bản | Mức độ | Precision@10 ($G_{test}$) | Hit@10 ($G_{test}$) | AP@10 ($G_{test}$) | P@10 ($G_{full}$ - Lạc quan) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for mode in ["accented", "unaccented", "restored"]:
        m_t = res["q3_handwritten"][mode]["g_test"]
        m_f = res["q3_handwritten"][mode]["g_full"]
        lbl = "(a) Gốc (25% ko dấu)" if mode == "accented" else ("(b) Không dấu toàn bộ" if mode == "unaccented" else "(c) Khôi phục dấu")
        lines.append(f"| **{lbl}** | Strict | {m_t['strict']['precision@10']:.4f} | {m_t['strict']['hit@10']:.4f} | {m_t['strict']['ap@10']:.4f} | {m_f['strict']['precision@10']:.4f} |")
        lines.append(f"| | Family | {m_t['family']['precision@10']:.4f} | {m_t['family']['hit@10']:.4f} | {m_t['family']['ap@10']:.4f} | {m_f['family']['precision@10']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Đánh Giá Truy Vấn Ảnh $\\to$ Ảnh (Q4 - Loại bỏ chính nó)",
        "",
        "| Tiêu chí tương đồng | Precision@10 | Hit@10 | AP@10 |",
        "| :--- | :---: | :---: | :---: |",
        f"| Cùng articleType + gender | {q4['same_category_and_gender']['precision@10']:.4f} | {q4['same_category_and_gender']['hit@10']:.4f} | {q4['same_category_and_gender']['ap@10']:.4f} |",
        f"| Cùng articleType + gender + baseColour | {q4['same_category_gender_and_colour']['precision@10']:.4f} | {q4['same_category_gender_and_colour']['hit@10']:.4f} | {q4['same_category_gender_and_colour']['ap@10']:.4f} |",
        "",
        "---",
        "",
        "## 6. So Sánh Truy Vấn Có Lọc Thuộc Tính vs Không Lọc (B3)",
        "",
        "| Tập truy vấn | Precision@10 Không lọc | Precision@10 Có lọc | Tỉ lệ Relaxation |",
        "| :--- | :---: | :---: | :---: |",
        f"| Q2 (Template 61 câu) | {res['filtered_retrieval']['q2']['unfiltered_precision@10']:.4f} | **{res['filtered_retrieval']['q2']['filtered_precision@10']:.4f}** | {res['filtered_retrieval']['q2']['relaxation_rate']*100:.1f}% |",
        f"| Q3 (Viết tay 100 câu) | {res['filtered_retrieval']['q3']['unfiltered_precision@10']:.4f} | **{res['filtered_retrieval']['q3']['filtered_precision@10']:.4f}** | {res['filtered_retrieval']['q3']['relaxation_rate']*100:.1f}% |",
        "",
        "---",
        "",
        "## 7. Phân Tích Lỗi & Chẩn Đoán (B4)",
        "",
        f"- **Nhóm phụ kiện cận cảnh (Headwear / Eyewear / Watches / Bags)**: Tỉ lệ lỗi xếp hạng: **{res['error_analysis']['accessories_error_rate']*100:.1f}%** ($n={res['error_analysis']['accessories_count']}$).",
        "- **Top 10 cặp nhầm lẫn ngành hàng (True $\\to$ Predicted)**:",
    ])

    for cf in res["error_analysis"]["top15_category_confusions"][:10]:
        lines.append(f"  - `{cf['true']}` bị nhầm sang `{cf['predicted']}`: {cf['count']} lần")

    lines.extend([
        "",
        "- **Top 5 cặp nhầm lẫn màu sắc (True $\\to$ Predicted)**:",
    ])
    for col_cf in res["error_analysis"]["top10_color_confusions"][:5]:
        lines.append(f"  - Màu `{col_cf['true']}` bị nhầm sang `{col_cf['predicted']}`: {col_cf['count']} lần")

    lines.extend([
        "",
        "- **Contact sheet mẫu 30 câu viết tay**: Đã xuất 30 ảnh vào `reports/contact_sheets/hw_*.png` và bảng đánh giá tại `eval/manual_ratings_template.csv`.",
        "",
        "---",
        "",
        "## 8. Phân Rã Theo Ngành Hàng (Subgroup Breakdown - Q1 trên $G_{test}$)",
        "",
        "| Ngành hàng (articleType) | Số lượng ($n$) | Recall@10 | MRR | Ghi chú |",
        "| :--- | :---: | :---: | :---: | :--- |"
    ])

    for row in res["subgroup_breakdown"]["article_type"]:
        flag = "Tham khảo ($n < 20$)" if row["n"] < 20 else "Đáng tin cậy"
        lines.append(f"| `{row['articleType']}` | {row['n']} | {row['recall10']:.4f} | {row['mrr']:.4f} | {flag} |")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Đo Độ Trễ End-to-End (B5)",
        "",
        "| Luồng truy vấn | p50 (ms) | p95 (ms) | p99 (ms) | Trung bình (ms) | Ngưỡng chỉ tiêu | Trạng thái |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Text $\to$ Ảnh** | {lat_txt['p50_ms']:.2f} | **{lat_txt['p95_ms']:.2f}** | {lat_txt['p99_ms']:.2f} | {lat_txt['mean_ms']:.2f} | $< 200$ ms | {'✅ ĐẠT' if pass_lat_txt else '❌ KHÔNG ĐẠT'} |",
        f"| **Ảnh $\to$ Ảnh** | {lat_img['p50_ms']:.2f} | **{lat_img['p95_ms']:.2f}** | {lat_img['p99_ms']:.2f} | {lat_img['mean_ms']:.2f} | $< 200$ ms | {'✅ ĐẠT' if pass_lat_img else '❌ KHÔNG ĐẠT'} |"
    ])

    report_content = "\n".join(lines)
    report_file = pathlib.Path("reports/phase3_eval.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report written to {report_file}")


if __name__ == "__main__":
    run_all_eval()
