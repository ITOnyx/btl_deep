"""
Phase 3f: Grid Tuning on Q6-val (Independent Attribute Relevance)
Evaluates 20 combinations:
gamma in [0.0, 0.05, 0.1, 0.2, 0.3]
alpha in [0.5, 0.7, 0.85, 1.0]
Gallery: 37,356 in-scope val items (split == 'unknown')
Selection criterion: AP@10 (normalized by min(R, 10))
Rule: If alpha=1.0 is within 1.0 point of best, prefer alpha*=1.0 to save 170MB RAM.
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.attr_index import AttributeIndex
from src.query_parser import QueryParser
from src.scoring import Scorer
from src.model_siglip import load_siglip_inference

def compute_ap_at_k(retrieved_ids: list, relevant_id_set: set, k: int = 10) -> float:
    R = len(relevant_id_set)
    if R == 0:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant_id_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    norm_denom = min(R, k)
    return sum_precisions / norm_denom if norm_denom > 0 else 0.0

def compute_p_at_k(retrieved_ids: list, relevant_id_set: set, k: int = 10) -> float:
    hits = sum(1 for item_id in retrieved_ids[:k] if item_id in relevant_id_set)
    return hits / k

def compute_hit_at_k(retrieved_ids: list, relevant_id_set: set, k: int = 10) -> float:
    for item_id in retrieved_ids[:k]:
        if item_id in relevant_id_set:
            return 1.0
    return 0.0

def run_q6_val_tuning():
    print("=====================================================================")
    print("Starting Phase 3f Grid Tuning on Q6-val (Independent Attribute)...")
    print("=====================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data & Attributes
    attr_idx = AttributeIndex()
    df = attr_idx.df
    parser = QueryParser()
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    
    # Mask for Val Gallery: split == "unknown" and in_scope == True
    val_mask = (df["split"] == "unknown").to_numpy()
    val_indices = np.where(val_mask)[0]
    print(f"Total catalog items: {len(df)}, Val gallery items: {len(val_indices)}")
    
    # 2. Load Embeddings and Mean-Center
    print("Loading image and text embeddings...")
    raw_img = np.load("index_v2/image_embeddings_v2.npy")
    raw_txt = np.load("index_v2/text_embeddings_v2.npy")
    
    emb_img_c = scorer.center_vectors(raw_img)
    emb_txt_c = scorer.center_vectors(raw_txt)
    
    # 3. Load Q6 Val Queries
    with open("eval/q6_queries.json", "r", encoding="utf-8") as f:
        q6_data = json.load(f)
    val_queries = q6_data["val_queries"]
    print(f"Loaded {len(val_queries)} Q6-val queries.")
    
    # 4. Pre-encode all queries
    print("Pre-encoding 120 queries with SigLIP text tower...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    model = load_siglip_inference(
        checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
        device=device
    )
    
    q_texts = [q["query_text"] for q in val_queries]
    with torch.no_grad():
        inp = processor(text=q_texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        raw_q_embs = model.encode_text(inp["input_ids"].to(device), None)
        # Normalize
        raw_q_embs = torch.nn.functional.normalize(raw_q_embs, p=2, dim=-1)
    raw_q_embs = raw_q_embs.cpu().numpy()
    q_embs_c = scorer.center_vectors(raw_q_embs)
    print(f"Encoded queries shape: {q_embs_c.shape}")
    
    # Free model VRAM before grid search
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 5. Move Embeddings to Torch Tensor for Fast Dot-Products
    t_emb_img = torch.tensor(emb_img_c, dtype=torch.float32, device=device)
    t_emb_txt = torch.tensor(emb_txt_c, dtype=torch.float32, device=device)
    t_q_embs = torch.tensor(q_embs_c, dtype=torch.float32, device=device)
    
    # Pre-parse and get candidate indices & base bonuses for each query
    print("Preparing candidate pools and attribute bonuses for all queries...")
    parsed_queries = []
    cand_info = []
    
    for i, q in enumerate(val_queries):
        parsed = parser.parse(q["query_text"])
        # Ensure usage and season are ignored in filter
        parsed["usage"] = None
        parsed["season"] = None
        parsed_queries.append(parsed)
        
        # Candidates in val_mask
        cand_idxs, tier = attr_idx.filter_relaxed(parsed, min_candidates=20, mask=val_mask)
        cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=device)
        
        # Raw attribute bonus (gamma = 1.0)
        # compute_attribute_soft_bonus: sum of matches for type, gender, colour, brand
        bonus_1 = attr_idx.compute_attribute_soft_bonus(
            parsed, cand_idxs,
            gamma_type=1.0, gamma_gender=1.0, gamma_colour=1.0, gamma_brand=1.0
        )
        bonus_t = torch.tensor(bonus_1, dtype=torch.float32, device=device)
        
        # Precompute dot-products for candidates
        q_v = t_q_embs[i:i+1] # (1, 1152)
        sim_img = torch.matmul(t_emb_img[cand_t], q_v.T).squeeze(1) # (len_cand,)
        sim_txt = torch.matmul(t_emb_txt[cand_t], q_v.T).squeeze(1) # (len_cand,)
        
        cand_info.append({
            "cand_idxs": cand_idxs,
            "cand_t": cand_t,
            "sim_img": sim_img,
            "sim_txt": sim_txt,
            "bonus_t": bonus_t,
            "relevant_set": set(str(x) for x in q["relevant_ids"])
        })
        
    # 6. Grid Search
    gamma_candidates = [0.0, 0.05, 0.1, 0.2, 0.3]
    alpha_candidates = [0.5, 0.7, 0.85, 1.0]
    
    print("\nEvaluating Grid: 5 gamma values x 4 alpha values = 20 combinations...")
    grid_results = []
    
    for gamma in gamma_candidates:
        for alpha in alpha_candidates:
            ap_list = []
            p10_list = []
            hit10_list = []
            
            for i in range(len(val_queries)):
                info = cand_info[i]
                sim_img = info["sim_img"]
                sim_txt = info["sim_txt"]
                bonus_t = info["bonus_t"]
                cand_idxs = info["cand_idxs"]
                rel_set = info["relevant_set"]
                
                # Dense scoring
                if alpha == 1.0:
                    dense_score = sim_img
                else:
                    dense_score = alpha * sim_img + (1.0 - alpha) * sim_txt
                    
                if gamma > 0:
                    score = dense_score + gamma * bonus_t
                else:
                    score = dense_score
                    
                k_val = min(10, len(score))
                top_vals, top_cand_pos = torch.topk(score, k=k_val)
                top_cand_pos = top_cand_pos.cpu().numpy()
                retrieved_item_ids = [str(df.iloc[cand_idxs[p]]["item_id"]) for p in top_cand_pos]
                
                ap = compute_ap_at_k(retrieved_item_ids, rel_set, k=10)
                p10 = compute_p_at_k(retrieved_item_ids, rel_set, k=10)
                hit10 = compute_hit_at_k(retrieved_item_ids, rel_set, k=10)
                
                ap_list.append(ap)
                p10_list.append(p10)
                hit10_list.append(hit10)
                
            mean_ap = float(np.mean(ap_list))
            mean_p10 = float(np.mean(p10_list))
            mean_hit10 = float(np.mean(hit10_list))
            
            grid_results.append({
                "gamma": gamma,
                "alpha": alpha,
                "ap10": round(mean_ap, 4),
                "p10": round(mean_p10, 4),
                "hit10": round(mean_hit10, 4)
            })
            print(f"gamma={gamma:<4} | alpha={alpha:<4} -> AP@10: {mean_ap:.4f} | P@10: {mean_p10:.4f} | Hit@10: {mean_hit10:.4f}")
            
    # 7. Selection
    # Sort by AP@10 descending
    grid_results.sort(key=lambda x: x["ap10"], reverse=True)
    best_overall = grid_results[0]
    
    # Check if alpha=1.0 is competitive (within 1.0 point / 0.0100 AP@10 of best)
    alpha_1_candidates = [r for r in grid_results if r["alpha"] == 1.0]
    best_alpha_1 = alpha_1_candidates[0] if alpha_1_candidates else None
    
    print("\n------------------------------------------------------------")
    print(f"Top 3 Combinations:")
    for rank, r in enumerate(grid_results[:3], start=1):
        print(f"  Rank {rank}: gamma={r['gamma']}, alpha={r['alpha']} -> AP@10={r['ap10']:.4f}, P@10={r['p10']:.4f}")
        
    print(f"\nBest with alpha=1.0: gamma={best_alpha_1['gamma']}, alpha=1.0 -> AP@10={best_alpha_1['ap10']:.4f}")
    
    # Rule: If alpha=1.0 is within 0.0100 (1.0 percentage point) of best, choose alpha=1.0
    gap = best_overall["ap10"] - best_alpha_1["ap10"]
    if gap <= 0.0100:
        chosen = best_alpha_1
        print(f"Selection: Preferring alpha=1.0! (Gap = {gap:.4f} <= 0.0100). Saves 170MB RAM and eliminates text fusion latency.")
    else:
        chosen = best_overall
        print(f"Selection: Choosing best overall (gamma={chosen['gamma']}, alpha={chosen['alpha']}) because gap = {gap:.4f} > 0.0100.")
        
    out_data = {
        "dataset": "Q6_val_grid_results",
        "gallery_size": len(val_indices),
        "num_queries": len(val_queries),
        "best_overall": best_overall,
        "best_alpha_1": best_alpha_1,
        "chosen_parameters": {
            "gamma": chosen["gamma"],
            "alpha": chosen["alpha"],
            "ap10": chosen["ap10"],
            "p10": chosen["p10"],
            "hit10": chosen["hit10"],
            "gap_vs_best": round(gap, 4)
        },
        "all_combinations": grid_results
    }
    
    out_file = Path("reports/q6_val_grid_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved all 20 grid results to {out_file}")
    
    return chosen

if __name__ == "__main__":
    run_q6_val_tuning()
