import sys
import json
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Any, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.model_siglip import load_siglip_inference
from src.scoring import Scorer

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

def main():
    print("=== RUNNING TRUNCATE 64 + PAD 128 ON Q2, Q3, Q5, AND HELDOUT ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP model on {device}...")
    model_wrapper = load_siglip_inference(
        checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
        device=device,
        fp16=True
    )
    processor = model_wrapper.processor
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    # Load Scorer and gallery
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)

    g_test_ids = [str(i) for i in testset["g_test_ids"]]
    g_full_ids = [str(i) for i in testset["g_full_ids"]]

    # Catalog & mappings for ground truth
    from src.attr_index import AttributeIndex
    attr_idx = AttributeIndex(brands_path="src/brands_v2.json")
    df = attr_idx.df
    N = len(df)
    item_ids = [str(x) for x in attr_idx.item_ids]
    id_to_df_idx = {iid: idx for idx, iid in enumerate(item_ids)}

    df_art = df["articleType"].values
    df_gen = df["gender"].values
    df_col = df["baseColour"].values
    df_brand_lower = np.array([b.lower() for b in df["brand"].fillna("").astype(str).values])
    item_ids_arr = np.array(item_ids)

    # Align raw embeddings (38,646) to df (38,644)
    raw_id_map = json.load(open("index_v2/id_map_v2.json", "r", encoding="utf-8"))
    raw_id_map_ids = [str(x["item_id"] if isinstance(x, dict) else x) for x in raw_id_map]
    raw_id_to_idx = {iid: idx for idx, iid in enumerate(raw_id_map_ids)}

    g_test_raw_indices = np.array([raw_id_to_idx[tid] for tid in g_test_ids], dtype=np.int64)
    g_full_raw_indices = np.array([raw_id_to_idx[tid] for tid in g_full_ids if tid in raw_id_to_idx], dtype=np.int64)

    raw_img_embs = np.load("index_v2/image_embeddings_v2.npy", mmap_mode="r")
    raw_g_test = np.array(raw_img_embs[g_test_raw_indices], dtype=np.float32)
    raw_g_full = np.array(raw_img_embs[g_full_raw_indices], dtype=np.float32)

    norm_g_test = raw_g_test / np.maximum(np.linalg.norm(raw_g_test, axis=-1, keepdims=True), 1e-12)
    norm_g_full = raw_g_full / np.maximum(np.linalg.norm(raw_g_full, axis=-1, keepdims=True), 1e-12)

    g_test_img_c = scorer.center_vectors(norm_g_test)
    g_full_img_c = scorer.center_vectors(norm_g_full)

    # Gallery masks over df (38,644)
    g_test_df_indices = np.array([id_to_df_idx[tid] for tid in g_test_ids if tid in id_to_df_idx], dtype=np.int64)
    g_full_df_indices = np.array([id_to_df_idx[tid] for tid in g_full_ids if tid in id_to_df_idx], dtype=np.int64)

    g_test_mask = np.zeros(N, dtype=bool)
    g_test_mask[g_test_df_indices] = True
    g_full_mask = np.zeros(N, dtype=bool)
    g_full_mask[g_full_df_indices] = True

    with open("eval/families.json", "r", encoding="utf-8") as f:
        fams = json.load(f)
    art_families = fams.get("article_type_families", {})
    col_families = fams.get("colour_families", {})
    gender_compat = fams.get("gender_compatibility", {
        "Men": ["Men", "Boys", "Unisex"],
        "Women": ["Women", "Girls", "Unisex"],
        "Boys": ["Boys", "Men", "Unisex"],
        "Girls": ["Girls", "Women", "Unisex"],
        "Unisex": ["Unisex", "Men", "Women", "Boys", "Girls"]
    })

    batch_size = 128

    def encode_texts(texts: List[str], mode: str) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b_texts = texts[i:i+batch_size]
                if mode == "condition_b_trunc64_pad128":
                    inp64 = processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    ids64 = inp64["input_ids"]
                    pad_ext = torch.full((ids64.size(0), 64), pad_token_id, dtype=ids64.dtype)
                    input_ids = torch.cat([ids64, pad_ext], dim=1).to(device)
                elif mode == "condition_c_len64":
                    inp64 = processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    input_ids = inp64["input_ids"].to(device)
                elif mode == "condition_a_len128":
                    inp128 = processor(text=b_texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
                    input_ids = inp128["input_ids"].to(device)
                feats = model_wrapper.encode_text(input_ids, None)
                all_embs.append(feats.cpu().numpy())
        raw = np.vstack(all_embs)
        norms = np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
        return scorer.center_vectors(raw / norms)

    def eval_suite(dataset_name: str, query_list: List[Dict[str, Any]], get_gt_fn):
        q_texts = [q.get("query_text") or q.get("query") for q in query_list]
        print(f"\n========================================================")
        print(f"Dataset: {dataset_name} ({len(q_texts)} queries)")
        print(f"========================================================")

        modes = [
            ("Condition C (Length 64 - Pha 3e/3f test)", "condition_c_len64"),
            ("Condition B (Trunc 64 + Pad 128)", "condition_b_trunc64_pad128"),
            ("Condition A (Length 128 - Train baseline)", "condition_a_len128")
        ]

        # Precompute GT once for both galleries
        gt_test = [get_gt_fn(q, "G_test", g_test_ids).get("strict", set()) for q in query_list]
        gt_full = [get_gt_fn(q, "G_full", g_full_ids).get("strict", set()) for q in query_list]

        results = {}
        for mode_name, mode_key in modes:
            q_vecs = encode_texts(q_texts, mode_key)
            sims_test = np.dot(q_vecs, g_test_img_c.T)
            sims_full = np.dot(q_vecs, g_full_img_c.T)

            # Evaluate on G_test and G_full for Model thuan (Config 0)
            res_mode = {}
            for g_name, sims, g_ids, gt_list in [
                ("G_test", sims_test, g_test_ids, gt_test),
                ("G_full", sims_full, g_full_ids, gt_full)
            ]:
                p10_list, hit10_list, ap10_list = [], [], []
                for q_idx in range(len(query_list)):
                    rel = gt_list[q_idx]
                    if len(rel) == 0:
                        continue
                    top10_idx = np.argsort(-sims[q_idx])[:10]
                    retrieved = [g_ids[i] for i in top10_idx]
                    p, h = compute_precision_and_hit(retrieved, rel, k=10)
                    ap = compute_ap_at_k(retrieved, rel, k=10)
                    p10_list.append(p)
                    hit10_list.append(h)
                    ap10_list.append(ap)

                p10 = float(np.mean(p10_list)) if p10_list else 0.0
                h10 = float(np.mean(hit10_list)) if hit10_list else 0.0
                ap10 = float(np.mean(ap10_list)) if ap10_list else 0.0
                res_mode[g_name] = {"P@10": p10, "Hit@10": h10, "AP@10": ap10, "n_queries": len(p10_list)}
            
            print(f"  {mode_name}:", flush=True)
            print(f"    G_test: P@10={res_mode['G_test']['P@10']:.4f}, Hit@10={res_mode['G_test']['Hit@10']:.4f}, AP@10={res_mode['G_test']['AP@10']:.4f}", flush=True)
            print(f"    G_full: P@10={res_mode['G_full']['P@10']:.4f}, Hit@10={res_mode['G_full']['Hit@10']:.4f}, AP@10={res_mode['G_full']['AP@10']:.4f}", flush=True)
            results[mode_key] = res_mode
        return results

    # 1. Q2 GT function (Vectorized numpy for high performance)
    def get_gt_q2(q, g_name, g_ids):
        tgt = q.get("target_attributes", {})
        art, gen, col = tgt.get("articleType"), tgt.get("gender"), tgt.get("baseColour")
        compat_gens = gender_compat.get(gen, [gen, "Unisex"]) if gen else None
        g_mask = g_test_mask if g_name == "G_test" else g_full_mask
        m_art = (df_art == art) if art else True
        m_gen = np.isin(df_gen, compat_gens) if compat_gens else True
        m_col = (df_col == col) if col else True
        strict_mask = m_art & m_gen & m_col & g_mask
        return {"strict": set(item_ids_arr[strict_mask])}

    # 2. Q3 GT function
    def get_gt_q3(q, g_name, g_ids):
        spec = q.get("spec", {})
        spec_arts = spec.get("articleType") or []
        spec_gen = spec.get("gender")
        spec_cols = spec.get("colour") or spec.get("baseColour") or []
        compat_gens = gender_compat.get(spec_gen, [spec_gen]) if spec_gen else None
        g_mask = g_test_mask if g_name == "G_test" else g_full_mask
        m_art = np.isin(df_art, spec_arts) if spec_arts else True
        m_gen = np.isin(df_gen, compat_gens) if compat_gens else True
        m_col = np.isin(df_col, spec_cols) if spec_cols else True
        strict_mask = m_art & m_gen & m_col & g_mask
        return {"strict": set(item_ids_arr[strict_mask])}

    # 3. Q5 GT function
    def get_gt_q5(q, g_name, g_ids):
        rel_key = "relevant_ids_test" if g_name == "G_test" else "relevant_ids_full"
        return {"strict": set(q.get(rel_key, []))}

    # 4. Heldout GT function (26 spec queries)
    def get_gt_heldout(q, g_name, g_ids):
        spec = q.get("spec", {})
        if not spec:
            return {"strict": set()}
        spec_arts = spec.get("articleType") or []
        spec_gen = spec.get("gender")
        spec_cols = spec.get("colour") or spec.get("baseColour") or []
        spec_brand = spec.get("brand")
        compat_gens = gender_compat.get(spec_gen, [spec_gen, "Unisex"]) if spec_gen else None
        g_mask = g_test_mask if g_name == "G_test" else g_full_mask
        m_art = np.isin(df_art, spec_arts) if spec_arts else True
        m_gen = np.isin(df_gen, compat_gens) if compat_gens else True
        m_col = np.isin(df_col, spec_cols) if spec_cols else True
        m_brd = (df_brand_lower == spec_brand.lower()) if spec_brand else True
        strict_mask = m_art & m_gen & m_col & m_brd & g_mask
        return {"strict": set(item_ids_arr[strict_mask])}

    # Run evaluations
    all_res = {}
    
    # Q2
    q2_queries = testset["q2_attribute_queries"]
    all_res["Q2"] = eval_suite("Q2_Attribute_Queries", q2_queries, get_gt_q2)

    # Q3
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        q3_queries = json.load(f)
    all_res["Q3"] = eval_suite("Q3_Handwritten_Queries", q3_queries, get_gt_q3)

    # Q5
    with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
        q5_data = json.load(f)
    all_res["Q5"] = eval_suite("Q5_Brand_Queries", q5_data["queries"], get_gt_q5)

    # Heldout
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        heldout_data = json.load(f)
    heldout_spec = [q for q in heldout_data if q.get("spec") is not None]
    all_res["Heldout_26Spec"] = eval_suite("Heldout_26Spec_Queries", heldout_spec, get_gt_heldout)

    out_path = Path("reports/phase4_trunc64_pad128_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"\nAll evaluations saved to {out_path} successfully.")

if __name__ == "__main__":
    main()
