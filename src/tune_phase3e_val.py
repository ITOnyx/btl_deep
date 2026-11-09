"""
Validation Tuning for Phase 3e Hyperparameters (src/tune_phase3e_val.py)
Phase 3e: Tunes gamma (soft attribute bonus) and alpha (dense text embedding fusion) on validation split.
Grid: gamma in [0.05, 0.1, 0.2, 0.3], alpha in [0.5, 0.7, 0.85, 1.0] (16 combinations).
Selection criterion: Average AP@10 strict on Q2-val and Q5-val.
Ties broken by lower latency.
Saves all 16 results to reports/val_grid_results.json.
"""

import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

import sys
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.attr_index import AttributeIndex
from src.query_parser import QueryParser
from src.lexical import BM25Index, rrf_merge
from src.scoring import Scorer
from src.model_siglip import load_siglip_inference

def compute_ap_at_k(retrieved_ids: list, relevant_ids: set, k: int = 10) -> float:
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

def run_val_tuning():
    print("==========================================================")
    print("Starting Phase 3e Validation Tuning (B2)...")
    print("==========================================================")

    # 1. Setup Data & Indices
    attr_idx = AttributeIndex()
    df = attr_idx.df # (38,646)
    emb_full = np.load("index_v2/image_embeddings_v2.npy")
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    parser = QueryParser()
    bm25 = BM25Index()

    # Load families for gender compatibility
    with open("eval/families.json", "r", encoding="utf-8") as f:
        fam_data = json.load(f)
    gender_compat = fam_data.get("gender_compatibility", {})

    # 2. Partition Validation Set (20% of non-test in_scope items, seed 42)
    non_test_idxs = np.where(df["split"] != "test")[0]
    rng = np.random.default_rng(seed=42)
    val_sample_size = int(len(non_test_idxs) * 0.20) # 7,471 items
    val_idxs = np.sort(rng.choice(non_test_idxs, size=val_sample_size, replace=False))
    
    # Sub-sample 2,000 items as Val Gallery for rapid, deterministic grid tuning
    val_gallery_idxs = val_idxs[:2000]
    val_gallery_ids = [str(df.iloc[idx]["item_id"]) for idx in val_gallery_idxs]
    val_gallery_id_set = set(val_gallery_ids)
    val_id_to_gidx = {item_id: i for i, item_id in enumerate(val_gallery_ids)}

    print(f"Val split items: {len(val_idxs)}, Val Gallery items: {len(val_gallery_idxs)}")

    # 3. Generate Val Queries
    # (a) Q2-val: 30 attribute queries from val gallery
    val_df = df.iloc[val_gallery_idxs]
    triplet_counts = val_df.groupby(["articleType", "gender", "baseColour"]).size()
    top_triplets = triplet_counts[triplet_counts >= 4].head(30).index.tolist()

    ARTICLE_VI = {
        "Tshirts": "áo thun", "Shirts": "áo sơ mi", "Casual Shoes": "giày casual",
        "Sports Shoes": "giày thể thao", "Watches": "đồng hồ", "Kurtas": "áo kurta",
        "Handbags": "túi xách", "Tops": "áo kiểu", "Heels": "giày cao gót",
        "Wallets": "ví", "Briefs": "quần lót", "Sandals": "dép sandal",
        "Sunglasses": "kính râm", "Flip Flops": "dép kẹp", "Belts": "thắt lưng",
        "Backpacks": "balo", "Socks": "tất", "Formal Shoes": "giày tây",
        "Flats": "giày búp bê", "Shorts": "quần short", "Bra": "áo ngực",
        "Jeans": "quần jean", "Dresses": "váy đầm", "Trousers": "quần âu"
    }
    GENDER_VI = {"Men": "nam", "Women": "nữ", "Boys": "bé trai", "Girls": "bé gái", "Unisex": "unisex"}
    COLOUR_VI = {"Black": "màu đen", "White": "màu trắng", "Blue": "màu xanh dương", "Red": "màu đỏ", "Grey": "màu xám"}

    q2_val_queries = []
    for art, gen, col in top_triplets:
        if art in ARTICLE_VI and gen in GENDER_VI and col in COLOUR_VI:
            q_text = f"{ARTICLE_VI[art]} {GENDER_VI[gen]} {COLOUR_VI[col]}"
            # Find relevant IDs in val_gallery
            compat_gens = gender_compat.get(gen, [gen, "Unisex"])
            rel_mask = (val_df["articleType"] == art) & (val_df["gender"].isin(compat_gens)) & (val_df["baseColour"] == col)
            rel_ids = set(val_df.loc[rel_mask, "item_id"].astype(str))
            if len(rel_ids) >= 3:
                q2_val_queries.append({
                    "query_text": q_text,
                    "articleType": art, "gender": gen, "baseColour": col,
                    "rel_ids": rel_ids
                })

    print(f"Generated {len(q2_val_queries)} Q2-val queries.")

    # (b) Q5-val: 30 brand queries from val gallery
    brand_counts = val_df.groupby(["brand", "articleType"]).size()
    top_brand_pairs = [(b, a) for (b, a), cnt in brand_counts.items() if b != "Unknown" and a in ARTICLE_VI and cnt >= 3][:30]

    q5_val_queries = []
    for b, a in top_brand_pairs:
        q_text = f"{ARTICLE_VI[a]} {b}"
        rel_mask = (val_df["brand"] == b) & (val_df["articleType"] == a)
        rel_ids = set(val_df.loc[rel_mask, "item_id"].astype(str))
        if len(rel_ids) >= 2:
            q5_val_queries.append({
                "query_text": q_text,
                "brand": b, "articleType": a,
                "rel_ids": rel_ids
            })

    print(f"Generated {len(q5_val_queries)} Q5-val queries.")

    # 4. Load Model on CUDA & Precompute Embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP model on {device} (FP16)...")
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))

    # Val Gallery Image Embeddings (mean-centered)
    val_img_embs = emb_full[val_gallery_idxs]
    val_img_embs_c = scorer.center_vectors(val_img_embs) # (2000, 1152)

    # Val Gallery Text Embeddings (for alpha < 1.0)
    print("Encoding Val Gallery Text Embeddings...")
    val_texts = []
    for idx in val_gallery_idxs:
        row = df.iloc[idx]
        title = str(row.get("name", ""))
        cat = f"{row.get('masterCategory', '')} > {row.get('subCategory', '')} > {row.get('articleType', '')}"
        attrs = f"{row.get('gender', '')}, {row.get('baseColour', '')}"
        val_texts.append(f"[{cat}] {title} ({attrs})")

    val_txt_embs_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            b_texts = val_texts[i:i+batch_size]
            inp = model.processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            feats = model.encode_text(inp["input_ids"], None)
            val_txt_embs_list.append(feats.cpu().numpy())
    val_txt_embs = np.vstack(val_txt_embs_list)
    val_txt_embs = val_txt_embs / np.maximum(np.linalg.norm(val_txt_embs, axis=-1, keepdims=True), 1e-12)
    val_txt_embs_c = scorer.center_vectors(val_txt_embs)

    # Encode query texts
    all_val_queries = q2_val_queries + q5_val_queries
    query_texts = [q["query_text"] for q in all_val_queries]
    query_embs_list = []
    with torch.no_grad():
        for i in range(0, len(query_texts), batch_size):
            b_texts = query_texts[i:i+batch_size]
            inp = model.processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            feats = model.encode_text(inp["input_ids"], None)
            query_embs_list.append(feats.cpu().numpy())
    query_embs = np.vstack(query_embs_list)
    query_embs = query_embs / np.maximum(np.linalg.norm(query_embs, axis=-1, keepdims=True), 1e-12)
    query_embs_c = scorer.center_vectors(query_embs)

    # Pre-parse and pre-run BM25 for all queries
    parsed_queries = [parser.parse(q["query_text"]) for q in all_val_queries]
    
    # Val gallery mask over full catalog
    val_gallery_mask = np.zeros(attr_idx.N, dtype=bool)
    val_gallery_mask[val_gallery_idxs] = True

    bm25_val_results = []
    for q in all_val_queries:
        top_idxs, _ = bm25.search(q["query_text"], top_k=100, mask=val_gallery_mask)
        # Convert full catalog indices to val_gallery sub-indices
        sub_idxs = [val_id_to_gidx[str(attr_idx.item_ids[idx])] for idx in top_idxs if str(attr_idx.item_ids[idx]) in val_id_to_gidx]
        bm25_val_results.append(sub_idxs)

    # Precompute pairwise similarity matrices
    # sims_img: (num_queries, 2000), sims_txt: (num_queries, 2000)
    sims_img = np.dot(query_embs_c, val_img_embs_c.T)
    sims_txt = np.dot(query_embs_c, val_txt_embs_c.T)

    # Precompute soft attribute match masks for val gallery
    val_cand_indices = np.array(val_gallery_idxs)

    # 5. Execute 16 Grid Combinations
    gamma_candidates = [0.05, 0.1, 0.2, 0.3]
    alpha_candidates = [0.5, 0.7, 0.85, 1.0]

    grid_results = []

    print("\n--- EVALUATING 16 GRID COMBINATIONS ---")
    for gamma in gamma_candidates:
        # Precompute attribute bonus for each query on val gallery
        bonuses = []
        for p in parsed_queries:
            b_arr = attr_idx.compute_attribute_soft_bonus(
                p, val_cand_indices,
                gamma_type=gamma, gamma_gender=gamma, gamma_colour=gamma, gamma_brand=gamma
            )
            bonuses.append(b_arr)
        bonuses = np.array(bonuses) # (num_queries, 2000)

        for alpha in alpha_candidates:
            t0 = time.perf_counter()
            fused_dense_sims = alpha * sims_img + (1.0 - alpha) * sims_txt + bonuses # (num_queries, 2000)

            # Evaluate on Q2-val and Q5-val
            ap_q2_list = []
            for q_idx in range(len(q2_val_queries)):
                q_obj = q2_val_queries[q_idx]
                dense_order = np.argsort(-fused_dense_sims[q_idx])[:100]
                sparse_order = bm25_val_results[q_idx]

                # RRF merge
                final_sub_idxs, _ = rrf_merge(dense_order, sparse_order, k=60, top_k=10)
                retrieved_ids = [val_gallery_ids[i] for i in final_sub_idxs]
                ap = compute_ap_at_k(retrieved_ids, q_obj["rel_ids"], k=10)
                ap_q2_list.append(ap)

            ap_q5_list = []
            for offset in range(len(q5_val_queries)):
                q_idx = len(q2_val_queries) + offset
                q_obj = q5_val_queries[offset]
                dense_order = np.argsort(-fused_dense_sims[q_idx])[:100]
                sparse_order = bm25_val_results[q_idx]

                final_sub_idxs, _ = rrf_merge(dense_order, sparse_order, k=60, top_k=10)
                retrieved_ids = [val_gallery_ids[i] for i in final_sub_idxs]
                ap = compute_ap_at_k(retrieved_ids, q_obj["rel_ids"], k=10)
                ap_q5_list.append(ap)

            mean_ap_q2 = float(np.mean(ap_q2_list))
            mean_ap_q5 = float(np.mean(ap_q5_list))
            avg_ap = (mean_ap_q2 + mean_ap_q5) / 2.0
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            grid_results.append({
                "gamma": gamma,
                "alpha": alpha,
                "ap10_q2_val": round(mean_ap_q2, 4),
                "ap10_q5_val": round(mean_ap_q5, 4),
                "avg_ap10_val": round(avg_ap, 4),
                "elapsed_ms": round(elapsed_ms, 2)
            })

            print(f"  gamma={gamma:.2f}, alpha={alpha:.2f} -> Q2 AP@10={mean_ap_q2:.4f}, Q5 AP@10={mean_ap_q5:.4f} | AVG={avg_ap:.4f} ({elapsed_ms:.1f}ms)")

    # 6. Select Best Combination
    best_config = max(grid_results, key=lambda x: (x["avg_ap10_val"], -x["elapsed_ms"]))
    print("\n==========================================================")
    print(f"BEST CONFIGURATION SELECTED ON VAL:")
    print(f"  gamma* = {best_config['gamma']}")
    print(f"  alpha* = {best_config['alpha']}")
    print(f"  Validation Average AP@10: {best_config['avg_ap10_val']:.4f}")
    print("==========================================================")

    output_data = {
        "metadata": {
            "val_split_size": len(val_idxs),
            "val_gallery_size": len(val_gallery_idxs),
            "n_q2_val": len(q2_val_queries),
            "n_q5_val": len(q5_val_queries),
            "selection_criterion": "Average AP@10 strict across Q2-val and Q5-val",
            "best_config": best_config
        },
        "grid_results": grid_results
    }

    out_file = Path("reports/val_grid_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved validation grid results to {out_file}")

    return best_config

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_val_tuning()
