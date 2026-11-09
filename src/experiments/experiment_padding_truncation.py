"""
Experiment: Padding & Truncation Isolation on Train-Format Q1b Captions
Replicating Phase 3f exact pipeline:
- Query text: search_text from product_metadata.json: "[{category}] {title} ({attrs})"
- Model: best_model.pt (SigLIP ViT)
- Gallery: G_test (1,290 items) & G_full (38,646 items) with mu_shared centering
- Evaluation: compute_exact_rank matching evaluate_phase3f.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
from typing import List, Tuple
from src.model_siglip import load_siglip_inference
from src.scoring import Scorer

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

def main():
    print("=================================================================")
    print("RUNNING PADDING & TRUNCATION EXPERIMENT ON TRAIN-FORMAT CAPTIONS...")
    print("=================================================================")

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

    # 1. Load testset and metadata
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    g_test_ids = [str(i) for i in testset["g_test_ids"]]
    g_full_ids = [str(i) for i in testset["g_full_ids"]]

    with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}

    q1_texts = [meta_dict[tid] for tid in g_test_ids]
    print(f"Loaded {len(q1_texts)} train-format captions for Q1b.")
    print("Sample query 0:", q1_texts[0][:100], "...")

    # 2. Load index gallery & Scorer
    print("Loading Scorer from data/checkpoints_siglip-vit/vit_zero_train_stats.npz...")
    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")

    id_map = json.load(open("index_v2/id_map_v2.json", "r", encoding="utf-8"))
    id_to_idx = {str(item["item_id"] if isinstance(item, dict) else item): idx for idx, item in enumerate(id_map)}

    g_test_indices = np.array([id_to_idx[tid] for tid in g_test_ids], dtype=np.int64)
    g_full_indices = np.array([id_to_idx[tid] for tid in g_full_ids], dtype=np.int64)

    raw_img_embs = np.load("index_v2/image_embeddings_v2.npy", mmap_mode="r")

    # Gallery centered exactly like evaluate_phase3f.py:
    # First normalize raw image embeddings if not already normalized
    raw_g_test = np.array(raw_img_embs[g_test_indices], dtype=np.float32)
    raw_g_full = np.array(raw_img_embs[g_full_indices], dtype=np.float32)
    norm_g_test = raw_g_test / np.maximum(np.linalg.norm(raw_g_test, axis=-1, keepdims=True), 1e-12)
    norm_g_full = raw_g_full / np.maximum(np.linalg.norm(raw_g_full, axis=-1, keepdims=True), 1e-12)

    g_test_img_c = scorer.center_vectors(norm_g_test)
    g_full_img_c = scorer.center_vectors(norm_g_full)

    target_indices_test = [g_test_ids.index(tid) for tid in g_test_ids]
    target_indices_full = [g_full_ids.index(tid) for tid in g_test_ids]

    batch_size = 128

    def encode_condition(mode):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(q1_texts), batch_size):
                b_texts = q1_texts[i:i+batch_size]
                if mode == "full_128_pad_128":
                    inp = processor(text=b_texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
                    input_ids = inp["input_ids"].to(device)
                elif mode == "truncate_64_pad_128":
                    inp64 = processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    ids64 = inp64["input_ids"]
                    pad_ext = torch.full((ids64.size(0), 64), pad_token_id, dtype=ids64.dtype)
                    input_ids = torch.cat([ids64, pad_ext], dim=1).to(device)
                elif mode == "truncate_64_pad_64":
                    inp64 = processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    input_ids = inp64["input_ids"].to(device)

                feats = model_wrapper.encode_text(input_ids, None)
                all_embs.append(feats.cpu().numpy())

        raw_text_embs = np.vstack(all_embs)
        # L2 norm before centering
        norms = np.maximum(np.linalg.norm(raw_text_embs, axis=-1, keepdims=True), 1e-12)
        normed_raw = raw_text_embs / norms
        # Center with scorer
        return scorer.center_vectors(normed_raw)

    conditions = [
        ("Condition A: Full 128, Pad 128 (Colab 3f Baseline)", "full_128_pad_128"),
        ("Condition B: Truncate 64, then Pad to 128", "truncate_64_pad_128"),
        ("Condition C: Truncate 64, Pad 64 (Short Sequence)", "truncate_64_pad_64")
    ]

    results = {}

    for name, mode in conditions:
        print(f"\nEncoding: {name}...")
        q_vecs_c = encode_condition(mode)

        # Evaluate on G_test
        sims_test = np.dot(q_vecs_c, g_test_img_c.T)
        r_test, h1_t, h5_t, h10_t, ties_t = compute_exact_rank(sims_test, target_indices_test)

        # Evaluate on G_full
        sims_full = np.dot(q_vecs_c, g_full_img_c.T)
        r_full, h1_f, h5_f, h10_f, ties_f = compute_exact_rank(sims_full, target_indices_full)

        res_entry = {
            "name": name,
            "G_test": {
                "recall_at_1": round(float(np.mean(h1_t)), 4),
                "recall_at_5": round(float(np.mean(h5_t)), 4),
                "recall_at_10": round(float(np.mean(h10_t)), 4),
                "mrr": round(float(np.mean(1.0 / r_test)), 4),
                "mean_rank": round(float(np.mean(r_test)), 2),
                "median_rank": round(float(np.median(r_test)), 1),
                "ties_count": int(ties_t)
            },
            "G_full": {
                "recall_at_1": round(float(np.mean(h1_f)), 4),
                "recall_at_5": round(float(np.mean(h5_f)), 4),
                "recall_at_10": round(float(np.mean(h10_f)), 4),
                "mrr": round(float(np.mean(1.0 / r_full)), 4),
                "mean_rank": round(float(np.mean(r_full)), 2),
                "median_rank": round(float(np.median(r_full)), 1),
                "ties_count": int(ties_f)
            }
        }
        results[mode] = res_entry

        print(f"  {name} Results:")
        print(f"    G_test: R@1={res_entry['G_test']['recall_at_1']}, R@5={res_entry['G_test']['recall_at_5']}, R@10={res_entry['G_test']['recall_at_10']}, MRR={res_entry['G_test']['mrr']}, MeanRank={res_entry['G_test']['mean_rank']}")
        print(f"    G_full: R@1={res_entry['G_full']['recall_at_1']}, R@5={res_entry['G_full']['recall_at_5']}, R@10={res_entry['G_full']['recall_at_10']}, MRR={res_entry['G_full']['mrr']}, MeanRank={res_entry['G_full']['mean_rank']}")

    out_file = Path("reports/phase3g_padding_truncation_experiment.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Isolation of truncation vs padding length on Train-Format Captions (matches 3f baseline)",
            "query_format": "[{category}] {title} ({attrs})",
            "queries_count": len(q1_texts),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved exact replicated experiment results to {out_file}.")

if __name__ == "__main__":
    main()
