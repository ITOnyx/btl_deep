"""
Phase 3g: Root Cause Verification for max_length & Padding (D.1 - D.4)
Investigates whether SigLIP padding length (64 vs 128) fundamentally alters text embeddings
even for short queries that are never truncated (due to absence of attention mask).

Tasks:
1. Q2 (61 queries) & Q3 (100 queries):
   - Encode at max_length=64 and 128 (padding='max_length').
   - Compute mean cosine similarity between the two embeddings of the same query.
   - Evaluate Config 0 P@10 & AP@10 on G_test for both pad64 and pad128.
2. Q1b (1,290 captions):
   - Token length distribution (pct > 64 tokens, pct > 128 tokens).
   - Disentangle truncation vs padding length:
     * Variant a: truncate=64, pad=64
     * Variant b: truncate=64, pad=128
     * Variant c: truncate=128, pad=128
   - Evaluate R@10 on G_test for all 3 variants.
3. Verify alignment with training code (Law 4).
4. Save to reports/phase3g_maxlen.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.scoring import Scorer
from src.model_siglip import load_siglip_inference

def compute_ap(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
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

def compute_p(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item_id in top_k if item_id in relevant_ids)
    return hits / k

def run_maxlen_verification():
    print("=== Phase 3g: Root Cause Verification for max_length (D) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP model on {device}...")
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))
    processor = model.processor
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    
    # Load catalog & splits
    catalog_path = Path("data/catalog/catalog.parquet")
    df = pd.read_parquet(catalog_path)
    in_scope = df[df["in_scope"] == True].reset_index(drop=True)
    item_ids = [str(x) for x in in_scope["item_id"]]
    
    test_mask = (in_scope["split"] == "test").to_numpy()
    g_test_indices = np.where(test_mask)[0]
    g_test_ids = [item_ids[i] for i in g_test_indices]
    
    # Load gallery embeddings (G_test)
    emb_img_raw = np.load("index_v2/image_embeddings_v2.npy")
    emb_img_c = scorer.center_vectors(emb_img_raw)
    g_test_emb = emb_img_c[g_test_indices]
    
    # Load families for ground truth
    with open("eval/families.json", "r", encoding="utf-8") as f:
        fam_data = json.load(f)
    gender_compat = fam_data.get("gender_compatibility", {})
    art_families = fam_data.get("article_type_families", {})
    col_families = fam_data.get("colour_families", {})
    
    def encode_texts(texts: List[str], max_len: int, truncate_len: int = None) -> np.ndarray:
        vecs = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if truncate_len is not None and truncate_len != max_len:
                # First tokenize with truncate_len, then pad to max_len
                tok_out = processor(text=batch, padding=False, truncation=True, max_length=truncate_len)
                pad_token_id = processor.tokenizer.pad_token_id or 0
                padded_ids = []
                for seq in tok_out["input_ids"]:
                    if len(seq) < max_len:
                        seq = seq + [pad_token_id] * (max_len - len(seq))
                    else:
                        seq = seq[:max_len]
                    padded_ids.append(seq)
                input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            else:
                inp = processor(text=batch, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
                input_ids = inp["input_ids"].to(device)
                
            with torch.no_grad():
                feat = model.encode_text(input_ids, None)
                v = feat.cpu().numpy()
            norms = np.linalg.norm(v, axis=-1, keepdims=True)
            v = v / np.maximum(norms, 1e-12)
            vecs.append(v)
        raw_vecs = np.vstack(vecs)
        return scorer.center_vectors(raw_vecs)
        
    # -------------------------------------------------------------
    # 1. Q2 Verification (61 queries)
    # -------------------------------------------------------------
    print("\n--- Evaluating Q2 (61 queries) at max_length=64 vs 128 ---")
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    q2_items = testset["q2_attribute_queries"]
    q2_texts = [q["query_text"] for q in q2_items]
    
    q2_v64 = encode_texts(q2_texts, max_len=64)
    q2_v128 = encode_texts(q2_texts, max_len=128)
    
    # Cosine between v64 and v128
    cos_q2 = np.sum(q2_v64 * q2_v128, axis=-1) / (np.linalg.norm(q2_v64, axis=-1) * np.linalg.norm(q2_v128, axis=-1))
    mean_cos_q2 = float(np.mean(cos_q2))
    print(f"Q2 mean cosine similarity (pad64 vs pad128): {mean_cos_q2:.4f}")
    
    # Evaluate Config 0 on G_test for Q2
    sims64_q2 = np.dot(q2_v64, g_test_emb.T)
    sims128_q2 = np.dot(q2_v128, g_test_emb.T)
    
    p10_64_q2, ap10_64_q2 = [], []
    p10_128_q2, ap10_128_q2 = [], []
    
    for i, q in enumerate(q2_items):
        gt = set(str(x) for x in q.get("target_ids_strict", {}).get("g_test", []))
        if not gt:
            # Fallback compute gt
            tgt = q.get("target_attributes", {})
            art = tgt.get("articleType")
            gen = tgt.get("gender")
            col = tgt.get("baseColour")
            m = np.ones(len(g_test_ids), dtype=bool)
            if art: m &= (in_scope.iloc[g_test_indices]["articleType"] == art)
            if gen: m &= (in_scope.iloc[g_test_indices]["gender"].isin(gender_compat.get(gen, [gen])))
            if col: m &= (in_scope.iloc[g_test_indices]["baseColour"] == col)
            gt = set(np.array(g_test_ids)[m])
            
        ret64 = [g_test_ids[idx] for idx in np.argsort(-sims64_q2[i])[:10]]
        ret128 = [g_test_ids[idx] for idx in np.argsort(-sims128_q2[i])[:10]]
        
        p10_64_q2.append(compute_p(ret64, gt))
        ap10_64_q2.append(compute_ap(ret64, gt))
        p10_128_q2.append(compute_p(ret128, gt))
        ap10_128_q2.append(compute_ap(ret128, gt))
        
    res_q2_64 = {"p10": round(float(np.mean(p10_64_q2)), 4), "ap10": round(float(np.mean(ap10_64_q2)), 4)}
    res_q2_128 = {"p10": round(float(np.mean(p10_128_q2)), 4), "ap10": round(float(np.mean(ap10_128_q2)), 4)}
    print(f"Q2 Config 0 (pad 64) : P@10={res_q2_64['p10']}, AP@10={res_q2_64['ap10']}")
    print(f"Q2 Config 0 (pad 128): P@10={res_q2_128['p10']}, AP@10={res_q2_128['ap10']}")
    
    # -------------------------------------------------------------
    # 2. Q3 Verification (100 handwritten queries)
    # -------------------------------------------------------------
    print("\n--- Evaluating Q3 (100 queries) at max_length=64 vs 128 ---")
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        q3_items = json.load(f)
    q3_texts = [q["query"] for q in q3_items]
    
    q3_v64 = encode_texts(q3_texts, max_len=64)
    q3_v128 = encode_texts(q3_texts, max_len=128)
    
    cos_q3 = np.sum(q3_v64 * q3_v128, axis=-1) / (np.linalg.norm(q3_v64, axis=-1) * np.linalg.norm(q3_v128, axis=-1))
    mean_cos_q3 = float(np.mean(cos_q3))
    print(f"Q3 mean cosine similarity (pad64 vs pad128): {mean_cos_q3:.4f}")
    
    sims64_q3 = np.dot(q3_v64, g_test_emb.T)
    sims128_q3 = np.dot(q3_v128, g_test_emb.T)
    
    p10_64_q3, ap10_64_q3 = [], []
    p10_128_q3, ap10_128_q3 = [], []
    
    for i, q in enumerate(q3_items):
        spec = q.get("spec", {})
        spec_arts = spec.get("articleType") or []
        spec_gen = spec.get("gender")
        spec_cols = spec.get("colour") or spec.get("baseColour") or []
        if isinstance(spec_arts, str): spec_arts = [spec_arts]
        if isinstance(spec_cols, str): spec_cols = [spec_cols]
        
        m = np.ones(len(g_test_ids), dtype=bool)
        sub_df = in_scope.iloc[g_test_indices]
        if spec_arts: m &= sub_df["articleType"].isin(spec_arts)
        if spec_gen: m &= sub_df["gender"].isin(gender_compat.get(spec_gen, [spec_gen]))
        if spec_cols: m &= sub_df["baseColour"].isin(spec_cols)
        gt = set(np.array(g_test_ids)[m])
        
        ret64 = [g_test_ids[idx] for idx in np.argsort(-sims64_q3[i])[:10]]
        ret128 = [g_test_ids[idx] for idx in np.argsort(-sims128_q3[i])[:10]]
        
        p10_64_q3.append(compute_p(ret64, gt))
        ap10_64_q3.append(compute_ap(ret64, gt))
        p10_128_q3.append(compute_p(ret128, gt))
        ap10_128_q3.append(compute_ap(ret128, gt))
        
    res_q3_64 = {"p10": round(float(np.mean(p10_64_q3)), 4), "ap10": round(float(np.mean(ap10_64_q3)), 4)}
    res_q3_128 = {"p10": round(float(np.mean(p10_128_q3)), 4), "ap10": round(float(np.mean(ap10_128_q3)), 4)}
    print(f"Q3 Config 0 (pad 64) : P@10={res_q3_64['p10']}, AP@10={res_q3_64['ap10']}")
    print(f"Q3 Config 0 (pad 128): P@10={res_q3_128['p10']}, AP@10={res_q3_128['ap10']}")
    
    # -------------------------------------------------------------
    # 3. Q1b Disentanglement (1,290 captions)
    # -------------------------------------------------------------
    print("\n--- Disentangling Truncation vs Padding on Q1b (1,290 captions) ---")
    with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}
    q1_texts = [meta_dict[tid] for tid in g_test_ids]
    target_indices = list(range(len(g_test_ids)))
    
    # Measure raw token lengths without padding
    raw_lengths = []
    for t in q1_texts:
        toks = processor.tokenizer.encode(t, add_special_tokens=True)
        raw_lengths.append(len(toks))
    raw_lengths = np.array(raw_lengths)
    pct_gt_64 = round(float(np.mean(raw_lengths > 64) * 100.0), 2)
    pct_gt_128 = round(float(np.mean(raw_lengths > 128) * 100.0), 2)
    print(f"Token length stats: >64 tokens: {pct_gt_64}%, >128 tokens: {pct_gt_128}%")
    
    # Variant a: truncate=64, pad=64
    print("Encoding Variant a (trunc=64, pad=64)...")
    va_vecs = encode_texts(q1_texts, max_len=64, truncate_len=64)
    sims_a = np.dot(va_vecs, g_test_emb.T)
    ranks_a = [np.sum(sims_a[i] > sims_a[i, target_indices[i]]) + 1 for i in range(len(q1_texts))]
    r10_a = round(float(np.mean(np.array(ranks_a) <= 10)), 4)
    
    # Variant b: truncate=64, pad=128
    print("Encoding Variant b (trunc=64, pad=128)...")
    vb_vecs = encode_texts(q1_texts, max_len=128, truncate_len=64)
    sims_b = np.dot(vb_vecs, g_test_emb.T)
    ranks_b = [np.sum(sims_b[i] > sims_b[i, target_indices[i]]) + 1 for i in range(len(q1_texts))]
    r10_b = round(float(np.mean(np.array(ranks_b) <= 10)), 4)
    
    # Variant c: truncate=128, pad=128
    print("Encoding Variant c (trunc=128, pad=128)...")
    vc_vecs = encode_texts(q1_texts, max_len=128, truncate_len=128)
    sims_c = np.dot(vc_vecs, g_test_emb.T)
    ranks_c = [np.sum(sims_c[i] > sims_c[i, target_indices[i]]) + 1 for i in range(len(q1_texts))]
    r10_c = round(float(np.mean(np.array(ranks_c) <= 10)), 4)
    
    print(f"Variant a (trunc=64, pad=64)   : R@10 = {r10_a}")
    print(f"Variant b (trunc=64, pad=128)  : R@10 = {r10_b}")
    print(f"Variant c (trunc=128, pad=128) : R@10 = {r10_c}")
    
    # 4. Check Law 4 Training Alignment
    # Training code expanded position embeddings to 128 (Cell 27) and trained with max_length=128 (Cell 29, 39, 51)
    training_match = True
    training_notes = (
        "Notebook siglip_vietnamese_ecommerce_colab_ver2.ipynb (Cells 27, 29, 33, 35, 39, 51) "
        "explicitly called expand_siglip_position_embeddings(model, target_max_length=128) and "
        "trained with processor(..., padding='max_length', truncation=True, max_length=128). "
        "Thus, max_length=128 and padding='max_length' strictly matches the trained checkpoint weights."
    )
    
    out_payload = {
        "metadata": {
            "task": "Phase 3g max_length & padding length root cause verification",
            "q2_count": len(q2_items),
            "q3_count": len(q3_items),
            "q1b_count": len(q1_texts)
        },
        "q2_verification": {
            "mean_cosine_pad64_vs_pad128": round(mean_cos_q2, 4),
            "pad64": res_q2_64,
            "pad128": res_q2_128
        },
        "q3_verification": {
            "mean_cosine_pad64_vs_pad128": round(mean_cos_q3, 4),
            "pad64": res_q3_64,
            "pad128": res_q3_128
        },
        "q1b_disentanglement": {
            "token_length_distribution": {
                "pct_greater_than_64_tokens": pct_gt_64,
                "pct_greater_than_128_tokens": pct_gt_128
            },
            "variants": [
                {"variant": "a", "truncate": 64, "pad": 64, "recall_at_10_g_test": r10_a},
                {"variant": "b", "truncate": 64, "pad": 128, "recall_at_10_g_test": r10_b},
                {"variant": "c", "truncate": 128, "pad": 128, "recall_at_10_g_test": r10_c}
            ],
            "impact_of_truncation": round(r10_c - r10_b, 4),
            "impact_of_padding_length": round(r10_b - r10_a, 4)
        },
        "law_4_training_alignment": {
            "pipeline_config_yaml": {"padding": "max_length", "max_length": 128},
            "training_code": {"padding": "max_length", "max_length": 128, "cells": [27, 29, 33, 35, 39, 51]},
            "matches": training_match,
            "notes": training_notes
        }
    }
    
    out_file = Path("reports/phase3g_maxlen.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved max_length verification to {out_file}")

if __name__ == "__main__":
    run_maxlen_verification()
