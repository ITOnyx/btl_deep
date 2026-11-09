"""
Phase 3g: Re-evaluate Heldout & Q5 with brands_v2.json (A.3 & A.4)
Compares:
- brands v1 (frozen 3f) vs brands v2 (aliases, post-heldout inspection).
- Evaluates Heldout brand group (x017..x024) and overall spec queries on G_full (Config 5).
- Evaluates Q5 (150 brand queries) on G_full (Config 5).
- Recalculates F6 table, low support count, and mean AP@10.
Saves results to reports/phase3g_brands_v2_eval.json.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.scoring import Scorer
from src.attr_index import AttributeIndex
from src.query_parser import QueryParser
from src.lexical import BM25Index, rrf_merge

def compute_ap_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    R = len(relevant_ids)
    if R == 0 or not retrieved_ids:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1)
    norm_denom = min(R, k)
    return sum_precisions / norm_denom if norm_denom > 0 else 0.0

def compute_p_and_hit(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> Tuple[float, float]:
    if not retrieved_ids:
        return 0.0, 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / k, (1.0 if hits > 0 else 0.0)

def sha256_file(path_str: str) -> str:
    with open(path_str, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_evaluation():
    print("=== Phase 3g: Re-evaluating with brands_v2.json ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Check hashes
    v1_hash = sha256_file("src/brands.json")
    v2_hash = sha256_file("src/brands_v2.json")
    print(f"brands.json (v1) SHA-256   : {v1_hash}")
    print(f"brands_v2.json (v2) SHA-256: {v2_hash}")
    
    # 1. Load catalog & AttributeIndex
    attr_idx_v1 = AttributeIndex(brands_path="src/brands.json")
    attr_idx_v2 = AttributeIndex(brands_path="src/brands_v2.json")
    df = attr_idx_v1.df
    N = attr_idx_v1.N
    item_ids = [str(x) for x in attr_idx_v1.item_ids]
    id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    # 2. Parsers
    parser_v1 = QueryParser(brands_path="src/brands.json")
    parser_v2 = QueryParser(brands_path="src/brands_v2.json")
    
    # 3. Load Gallery Embeddings (centered)
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    emb_img_raw = np.load("index_v2/image_embeddings_v2.npy")
    emb_img_c = scorer.center_vectors(emb_img_raw)
    
    if device == "cuda":
        t_emb_img_c = torch.tensor(emb_img_c, dtype=torch.float16, device=device)
    else:
        t_emb_img_c = None
        
    # 4. Model for text encoding
    from src.model_siglip import load_siglip_inference
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))
    processor = model.processor
    
    def encode_text(text: str) -> np.ndarray:
        with torch.no_grad():
            inp = processor(text=[text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            feat = model.encode_text(inp["input_ids"].to(device), None)
            vec = feat.cpu().numpy()[0]
        norm = max(float(np.linalg.norm(vec)), 1e-12)
        vec = vec / norm
        return scorer.center_vectors(vec[np.newaxis, :])[0]
        
    bm25 = BM25Index()
    
    # Search function for Config 5
    def search_config_5(query_text: str, q_vec: np.ndarray, parser: QueryParser, attr_idx: AttributeIndex) -> List[str]:
        parsed = parser.parse(query_text)
        cand_idxs, _ = attr_idx.filter_relaxed(parsed, min_candidates=20)
        
        # Dense + soft bonus
        if device == "cuda" and t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=device)
            q_t = torch.tensor(q_vec, dtype=torch.float16, device=device)
            sims_t = torch.matmul(t_emb_img_c[cand_t], q_t)
            bonus = attr_idx.compute_attribute_soft_bonus(
                parsed, cand_idxs, gamma_type=0.3, gamma_gender=0.3, gamma_colour=0.3, gamma_brand=0.3
            )
            dense_scores_t = sims_t + torch.tensor(bonus, dtype=torch.float16, device=device)
            k_top = min(100, len(cand_idxs))
            top_pos = torch.topk(dense_scores_t, k=k_top)[1].cpu().numpy()
            dense_order = np.array(cand_idxs)[top_pos]
        else:
            sims = np.dot(emb_img_c[cand_idxs], q_vec)
            bonus = attr_idx.compute_attribute_soft_bonus(
                parsed, cand_idxs, gamma_type=0.3, gamma_gender=0.3, gamma_colour=0.3, gamma_brand=0.3
            )
            dense_scores = sims + bonus
            k_top = min(100, len(cand_idxs))
            dense_order = np.array(cand_idxs)[np.argsort(-dense_scores)[:k_top]]
            
        # BM25
        cand_mask = np.zeros(N, dtype=bool)
        cand_mask[cand_idxs] = True
        sparse_order, _ = bm25.search(query_text, top_k=100, mask=cand_mask)
        
        # RRF merge
        final_idxs, _ = rrf_merge(dense_order, sparse_order, k=60, top_k=10)
        return [item_ids[i] for i in final_idxs]
        
    # Load families for ground truth spec
    with open("eval/families.json", "r", encoding="utf-8") as f:
        fam_data = json.load(f)
    gender_compat = fam_data.get("gender_compatibility", {})
    
    # -------------------------------------------------------------
    # 5. Evaluate Heldout Queries (G_full)
    # -------------------------------------------------------------
    print("\n--- Evaluating Heldout Queries (G_full, Config 5) ---")
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        heldout_all = json.load(f)
        
    spec_queries = [q for q in heldout_all if q.get("spec") is not None]
    print(f"Total spec queries in heldout: {len(spec_queries)}")
    
    # Load alias map for canonical ground truth
    brand_alias_map = {
        "benetton": "United Colors of Benetton",
        "adidas": "ADIDAS"
    }
    
    heldout_brand_results = []
    
    h_overall_ap_v1, h_overall_p_v1 = [], []
    h_overall_ap_v2, h_overall_p_v2 = [], []
    
    h_brand_ap_v1, h_brand_p_v1 = [], []
    h_brand_ap_v2, h_brand_p_v2 = [], []
    
    low_support_v1 = 0
    low_support_v2 = 0
    
    for q in spec_queries:
        qid = q["id"]
        q_text = q["query"]
        q_grp = q.get("group", "")
        spec = q["spec"]
        
        spec_arts = spec.get("articleType") or []
        spec_gen = spec.get("gender")
        spec_cols = spec.get("colour") or spec.get("baseColour") or []
        spec_brand = spec.get("brand")
        
        if isinstance(spec_arts, str): spec_arts = [spec_arts]
        if isinstance(spec_cols, str): spec_cols = [spec_cols]
        
        compat_gens = gender_compat.get(spec_gen, [spec_gen, "Unisex"]) if spec_gen else None
        
        # Strict ground truth on G_full
        m = np.ones(N, dtype=bool)
        if spec_arts: m &= df["articleType"].isin(spec_arts)
        if compat_gens: m &= df["gender"].isin(compat_gens)
        if spec_cols: m &= df["baseColour"].isin(spec_cols)
        
        # In v1 vs v2: Benetton in v1 was missing alias, in v2 canonical is United Colors of Benetton
        if spec_brand:
            # Map canonical
            canon_b = brand_alias_map.get(spec_brand.lower(), spec_brand)
            m_v2 = m & df["brand"].isin([spec_brand, canon_b])
            # For v1, exact match spec_brand (which caused x018 to have 0 if catalog was labeled differently)
            m_v1 = m & df["brand"].isin([spec_brand])
        else:
            m_v1 = m
            m_v2 = m
            
        gt_v1 = set(np.array(item_ids)[m_v1])
        gt_v2 = set(np.array(item_ids)[m_v2])
        
        # Count low support (< 3)
        if len(gt_v1) < 3: low_support_v1 += 1
        if len(gt_v2) < 3: low_support_v2 += 1
        
        # Encode query text
        q_vec = encode_text(q_text)
        
        # Search with v1
        ret_v1 = search_config_5(q_text, q_vec, parser_v1, attr_idx_v1)
        ap_v1 = compute_ap_at_k(ret_v1, gt_v2, k=10) # compare against real ground truth
        p_v1, _ = compute_p_and_hit(ret_v1, gt_v2, k=10)
        
        # Search with v2
        ret_v2 = search_config_5(q_text, q_vec, parser_v2, attr_idx_v2)
        ap_v2 = compute_ap_at_k(ret_v2, gt_v2, k=10)
        p_v2, _ = compute_p_and_hit(ret_v2, gt_v2, k=10)
        
        h_overall_ap_v1.append(ap_v1)
        h_overall_p_v1.append(p_v1)
        h_overall_ap_v2.append(ap_v2)
        h_overall_p_v2.append(p_v2)
        
        if q_grp == "brand":
            h_brand_ap_v1.append(ap_v1)
            h_brand_p_v1.append(p_v1)
            h_brand_ap_v2.append(ap_v2)
            h_brand_p_v2.append(p_v2)
            
            heldout_brand_results.append({
                "id": qid,
                "query": q_text,
                "brand_spec": spec_brand,
                "r_strict_g_full": len(gt_v2),
                "v1_parsed_brand": parser_v1.parse(q_text).get("brand"),
                "v2_parsed_brand": parser_v2.parse(q_text).get("brand"),
                "v1_ap10": round(ap_v1, 4),
                "v1_p10": round(p_v1, 4),
                "v2_ap10": round(ap_v2, 4),
                "v2_p10": round(p_v2, 4),
                "status": "CẢI THIỆN" if ap_v2 > ap_v1 else ("KHÔNG ĐỔI" if ap_v2 == ap_v1 else "GIẢM")
            })
            
    # Calculate means
    mean_brand_ap_v1 = float(np.mean(h_brand_ap_v1))
    mean_brand_p_v1 = float(np.mean(h_brand_p_v1))
    mean_brand_ap_v2 = float(np.mean(h_brand_ap_v2))
    mean_brand_p_v2 = float(np.mean(h_brand_p_v2))
    
    # Valid brands mean (excluding Lavie R=0, i.e. 7 valid brands x018..x024)
    valid_7_ap_v1 = float(np.mean([r["v1_ap10"] for r in heldout_brand_results if r["r_strict_g_full"] > 0]))
    valid_7_ap_v2 = float(np.mean([r["v2_ap10"] for r in heldout_brand_results if r["r_strict_g_full"] > 0]))
    
    mean_spec_ap_v1 = float(np.mean(h_overall_ap_v1))
    mean_spec_p_v1 = float(np.mean(h_overall_p_v1))
    mean_spec_ap_v2 = float(np.mean(h_overall_ap_v2))
    mean_spec_p_v2 = float(np.mean(h_overall_p_v2))
    
    print("\n--- Heldout Brand Group Results (8 queries) ---")
    for r in heldout_brand_results:
        print(f"[{r['id']}] {r['query']:<24} (R={r['r_strict_g_full']:>4}) | v1: AP@10={r['v1_ap10']:.4f}, P@10={r['v1_p10']:.4f} -> v2: AP@10={r['v2_ap10']:.4f}, P@10={r['v2_p10']:.4f} | {r['status']}")
    print(f"Mean Brand AP@10 (all 8)  : v1={mean_brand_ap_v1:.4f} -> v2={mean_brand_ap_v2:.4f} (Delta: {mean_brand_ap_v2 - mean_brand_ap_v1:+.4f})")
    print(f"Mean Brand P@10 (all 8)   : v1={mean_brand_p_v1:.4f} -> v2={mean_brand_p_v2:.4f} (Delta: {mean_brand_p_v2 - mean_brand_p_v1:+.4f})")
    print(f"Mean Valid Brand AP@10 (7): v1={valid_7_ap_v1:.4f} -> v2={valid_7_ap_v2:.4f} (Delta: {valid_7_ap_v2 - valid_7_ap_v1:+.4f})")
    print(f"Mean Spec AP@10 (overall) : v1={mean_spec_ap_v1:.4f} -> v2={mean_spec_ap_v2:.4f} (Delta: {mean_spec_ap_v2 - mean_spec_ap_v1:+.4f})")
    print(f"Low support queries on G_full: v1={low_support_v1} -> v2={low_support_v2}")
    
    # -------------------------------------------------------------
    # 6. Evaluate Q5 (150 brand queries) on G_full
    # -------------------------------------------------------------
    print("\n--- Evaluating Q5 (150 brand queries on G_full) ---")
    with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
        q5_data = json.load(f)
    q5_queries = q5_data["queries"]
    
    q5_ap_v1, q5_p_v1 = [], []
    q5_ap_v2, q5_p_v2 = [], []
    
    for q in q5_queries:
        q_text = q["query_text"]
        gt = set(q.get("relevant_ids_full", []))
        q_vec = encode_text(q_text)
        
        ret_v1 = search_config_5(q_text, q_vec, parser_v1, attr_idx_v1)
        ap1 = compute_ap_at_k(ret_v1, gt, k=10)
        p1, _ = compute_p_and_hit(ret_v1, gt, k=10)
        q5_ap_v1.append(ap1)
        q5_p_v1.append(p1)
        
        ret_v2 = search_config_5(q_text, q_vec, parser_v2, attr_idx_v2)
        ap2 = compute_ap_at_k(ret_v2, gt, k=10)
        p2, _ = compute_p_and_hit(ret_v2, gt, k=10)
        q5_ap_v2.append(ap2)
        q5_p_v2.append(p2)
        
    mean_q5_ap_v1 = float(np.mean(q5_ap_v1))
    mean_q5_p_v1 = float(np.mean(q5_p_v1))
    mean_q5_ap_v2 = float(np.mean(q5_ap_v2))
    mean_q5_p_v2 = float(np.mean(q5_p_v2))
    
    print(f"Q5 G_full Config 5 AP@10: v1={mean_q5_ap_v1:.4f} -> v2={mean_q5_ap_v2:.4f}")
    print(f"Q5 G_full Config 5 P@10 : v1={mean_q5_p_v1:.4f} -> v2={mean_q5_p_v2:.4f}")
    
    # Save to reports/phase3g_brands_v2_eval.json
    out_payload = {
        "metadata": {
            "phase": "3g",
            "task": "Evaluation of brands_v2 with approved aliases on Heldout & Q5",
            "brands_v1_sha256": v1_hash,
            "brands_v2_sha256": v2_hash,
            "note": "v2 là sửa hậu nghiệm sau khi phát hiện heldout x018 thiếu alias; heldout không còn là tập ẩn hoàn toàn cho thay đổi này."
        },
        "heldout_brand_queries_detail": heldout_brand_results,
        "metrics_summary": {
            "heldout_brand_group": {
                "v1_frozen_3f": {
                    "ap10_all_8": round(mean_brand_ap_v1, 4),
                    "p10_all_8": round(mean_brand_p_v1, 4),
                    "ap10_valid_7": round(valid_7_ap_v1, 4),
                    "valid_brands_count": "7/8 (chỉ Lavie R=0)"
                },
                "v2_approved_aliases": {
                    "ap10_all_8": round(mean_brand_ap_v2, 4),
                    "p10_all_8": round(mean_brand_p_v2, 4),
                    "ap10_valid_7": round(valid_7_ap_v2, 4),
                    "valid_brands_count": "7/8 (chỉ Lavie R=0)"
                },
                "delta_ap10_all_8": round(mean_brand_ap_v2 - mean_brand_ap_v1, 4),
                "delta_ap10_valid_7": round(valid_7_ap_v2 - valid_7_ap_v1, 4)
            },
            "heldout_overall_spec": {
                "v1_frozen_3f": {
                    "ap10": round(mean_spec_ap_v1, 4),
                    "p10": round(mean_spec_p_v1, 4)
                },
                "v2_approved_aliases": {
                    "ap10": round(mean_spec_ap_v2, 4),
                    "p10": round(mean_spec_p_v2, 4)
                },
                "delta_ap10": round(mean_spec_ap_v2 - mean_spec_ap_v1, 4)
            },
            "q5_brand_queries": {
                "v1_frozen_3f": {
                    "ap10": round(mean_q5_ap_v1, 4),
                    "p10": round(mean_q5_p_v1, 4)
                },
                "v2_approved_aliases": {
                    "ap10": round(mean_q5_ap_v2, 4),
                    "p10": round(mean_q5_p_v2, 4)
                },
                "delta_ap10": round(mean_q5_ap_v2 - mean_q5_ap_v1, 4)
            },
            "low_support_queries_g_full": {
                "v1_frozen_3f": low_support_v1,
                "v2_approved_aliases": low_support_v2,
                "note": "Ở v1 có 2 câu hỗ trợ thấp (x017 Lavie: 0 sp, x018 Benetton: 0 sp do thiếu alias). Ở v2 chỉ còn duy nhất x017 (Lavie) là 0 sp."
            }
        }
    }
    
    out_file = Path("reports/phase3g_brands_v2_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved evaluation to {out_file}")

if __name__ == "__main__":
    run_evaluation()
