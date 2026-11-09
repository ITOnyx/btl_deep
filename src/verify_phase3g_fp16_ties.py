"""
Phase 3g: FP16 & Ties Analysis for Config 0 (Q1b) (C.1 - C.3)
Compares FP32 vs FP16 precision on G_test and G_full for Q1b (1,290 captions).
Evaluates optimistic, pessimistic, and average tie-breaking.
Saves results to reports/phase3g_fp16_ties.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.scoring import Scorer
from src.model_siglip import load_siglip_inference

def compute_ranks_detailed(score_matrix: np.ndarray, target_indices: List[int], atol: float = 1e-6):
    N = score_matrix.shape[0]
    
    ranks_opt = np.zeros(N, dtype=np.int32)
    ranks_pess = np.zeros(N, dtype=np.int32)
    ranks_avg = np.zeros(N, dtype=np.float32)
    
    ties_count = 0
    total_tied_items = 0
    
    for i in range(N):
        row = score_matrix[i]
        tgt_idx = target_indices[i]
        tgt_score = row[tgt_idx]
        
        strictly_greater = np.sum(row > tgt_score)
        equal_score = np.sum(np.isclose(row, tgt_score, atol=atol))
        
        if equal_score > 1:
            ties_count += 1
            total_tied_items += int(equal_score - 1)
            
        r_opt = int(strictly_greater) + 1
        r_pess = int(strictly_greater) + int(equal_score)
        r_avg = float(strictly_greater) + float(equal_score + 1) / 2.0
        
        ranks_opt[i] = r_opt
        ranks_pess[i] = r_pess
        ranks_avg[i] = r_avg
        
    def get_metrics(ranks):
        h1 = float(np.mean(ranks <= 1))
        h5 = float(np.mean(ranks <= 5))
        h10 = float(np.mean(ranks <= 10))
        mrr = float(np.mean(1.0 / ranks))
        med = float(np.median(ranks))
        mean_r = float(np.mean(ranks))
        return {
            "recall_at_1": round(h1, 4),
            "recall_at_5": round(h5, 4),
            "recall_at_10": round(h10, 4),
            "mrr": round(mrr, 4),
            "median_rank": round(med, 2),
            "mean_rank": round(mean_r, 2)
        }
        
    return {
        "optimistic": get_metrics(ranks_opt),
        "pessimistic": get_metrics(ranks_pess),
        "average": get_metrics(ranks_avg),
        "ties_count": ties_count,
        "total_tied_competitors": total_tied_items
    }

def run_fp16_ties_analysis():
    print("=== Phase 3g: FP16 & Ties Analysis (Q1b Config 0) ===")
    
    # 1. Load catalog & splits
    catalog_path = Path("data/catalog/catalog.parquet")
    df = pd.read_parquet(catalog_path)
    in_scope = df[df["in_scope"] == True].reset_index(drop=True)
    item_ids = [str(x) for x in in_scope["item_id"]]
    
    test_mask = (in_scope["split"] == "test").to_numpy()
    g_test_indices = np.where(test_mask)[0]
    g_test_ids = [item_ids[i] for i in g_test_indices]
    
    g_full_indices = np.arange(len(in_scope))
    g_full_ids = item_ids
    
    target_indices_test = list(range(len(g_test_ids)))
    target_indices_full = [g_full_ids.index(tid) for tid in g_test_ids]
    
    # 2. Load Gallery Embeddings
    print("Loading gallery embeddings...")
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    emb_img_raw = np.load("index_v2/image_embeddings_v2.npy")
    emb_img_c = scorer.center_vectors(emb_img_raw)
    
    # 3. Load Q1b Captions
    with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}
    q1_texts = [meta_dict[tid] for tid in g_test_ids]
    print(f"Loaded {len(q1_texts)} train-format captions for Q1b.")
    
    # 4. Encode texts (max_length=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding queries with SigLIP on {device}...")
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))
    processor = model.processor
    
    q_vecs_list = []
    batch_size = 64
    for i in range(0, len(q1_texts), batch_size):
        batch = q1_texts[i:i+batch_size]
        inp = processor(text=batch, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            feat = model.encode_text(inp["input_ids"].to(device), None)
            vec = feat.cpu().numpy()
        norms = np.linalg.norm(vec, axis=-1, keepdims=True)
        vec = vec / np.maximum(norms, 1e-12)
        q_vecs_list.append(vec)
    q_vecs_raw = np.vstack(q_vecs_list)
    q_vecs_c = scorer.center_vectors(q_vecs_raw)
    
    out_results = {}
    
    for g_name, g_indices, target_indices in [
        ("G_test", g_test_indices, target_indices_test),
        ("G_full", g_full_indices, target_indices_full)
    ]:
        print(f"\n--- Analyzing {g_name} (N={len(g_indices):,}) ---")
        g_emb = emb_img_c[g_indices]
        
        # FP32 evaluation
        print("Running FP32 evaluation...")
        sims_fp32 = np.dot(q_vecs_c.astype(np.float32), g_emb.astype(np.float32).T)
        res_fp32 = compute_ranks_detailed(sims_fp32, target_indices, atol=1e-6)
        
        # FP16 evaluation
        print("Running FP16 evaluation...")
        # Simulate torch FP16 / numpy float16 dot product
        if device == "cuda":
            t_q = torch.tensor(q_vecs_c, dtype=torch.float16, device=device)
            t_g = torch.tensor(g_emb, dtype=torch.float16, device=device)
            sims_fp16 = torch.matmul(t_q, t_g.T).cpu().numpy()
        else:
            sims_fp16 = np.dot(q_vecs_c.astype(np.float16), g_emb.astype(np.float16).T)
        res_fp16 = compute_ranks_detailed(sims_fp16, target_indices, atol=1e-4)
        
        out_results[g_name] = {
            "fp32": res_fp32,
            "fp16": res_fp16,
            "comparison": {
                "r10_diff_optimistic": round(res_fp16["optimistic"]["recall_at_10"] - res_fp32["optimistic"]["recall_at_10"], 4),
                "r10_diff_pessimistic": round(res_fp16["pessimistic"]["recall_at_10"] - res_fp32["pessimistic"]["recall_at_10"], 4),
                "ties_count_fp32": res_fp32["ties_count"],
                "ties_count_fp16": res_fp16["ties_count"],
                "ties_increase": res_fp16["ties_count"] - res_fp32["ties_count"]
            }
        }
        
        print(f"[{g_name}] FP32: R@10={res_fp32['optimistic']['recall_at_10']}, MRR={res_fp32['optimistic']['mrr']}, Ties={res_fp32['ties_count']}")
        print(f"[{g_name}] FP16: R@10={res_fp16['optimistic']['recall_at_10']}, MRR={res_fp16['optimistic']['mrr']}, Ties={res_fp16['ties_count']}")
        print(f"[{g_name}] Pessimistic FP16: R@10={res_fp16['pessimistic']['recall_at_10']}, MRR={res_fp16['pessimistic']['mrr']}")
        
    out_payload = {
        "metadata": {
            "task": "Q1b Config 0 FP32 vs FP16 precision & tie-breaking audit",
            "queries_count": len(q1_texts),
            "max_length": 128,
            "centering": "mu_shared",
            "current_pipeline_mode": "optimistic (strictly_greater + 1)"
        },
        "results": out_results
    }
    
    out_file = Path("reports/phase3g_fp16_ties.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved FP16 & Ties analysis to {out_file}")

if __name__ == "__main__":
    run_fp16_ties_analysis()
