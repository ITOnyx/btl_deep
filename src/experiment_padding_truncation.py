"""
Experiment: Truncate 64 then Pad 128 vs Full 128 Pad 128 vs Pad 64 (Phase 3g Rev4)
Evaluates on all 1,290 paired test queries (Q1b) on G_test to isolate:
1. Information loss due to truncation (Tokens 65..128 removed)
2. Sequence length / padding artifacts (Seq len 128 vs Seq len 64)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
import pandas as pd
from src.model_siglip import load_siglip_inference

def main():
    print("=================================================================")
    print("STARTING PADDING & TRUNCATION EXPERIMENT (1,290 QUERIES)...")
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

    # Load 1,290 Q1b queries
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    q1_queries = testset["q1_paired_queries"]
    print(f"Loaded {len(q1_queries)} queries.")

    # Load gallery embeddings and mu_shared
    split_info = json.load(open("index_v2/split_v2.json", "r", encoding="utf-8"))
    id_map = json.load(open("index_v2/id_map_v2.json", "r", encoding="utf-8"))
    id_to_idx = {str(item["item_id"] if isinstance(item, dict) else item): idx for idx, item in enumerate(id_map)}

    # G_test indices
    g_test_ids = [str(i) for i in testset["g_test_ids"]]
    g_test_indices = np.array([id_to_idx[i] for i in g_test_ids if i in id_to_idx], dtype=np.int64)
    g_test_id_list = [str(id_map[idx]["item_id"] if isinstance(id_map[idx], dict) else id_map[idx]) for idx in g_test_indices]
    g_test_id_to_rank_pos = {str(iid): pos for pos, iid in enumerate(g_test_id_list)}

    # Load image embeddings
    raw_img_embs = np.load("index_v2/image_embeddings_v2.npy", mmap_mode="r")
    g_test_img_embs = np.array(raw_img_embs[g_test_indices], dtype=np.float32)

    # Load mu_shared
    mu_path = Path("index_v2/mu_shared.npy")
    if mu_path.exists():
        mu_shared = np.load(mu_path).astype(np.float32)
        print("Loaded mu_shared for centering.")
    else:
        mu_shared = np.zeros((1152,), dtype=np.float32)

    # Centered gallery
    g_test_centered = g_test_img_embs - mu_shared
    g_test_norms = np.linalg.norm(g_test_centered, axis=1, keepdims=True)
    g_test_normed = g_test_centered / np.maximum(g_test_norms, 1e-12)

    target_positions = []
    for q in q1_queries:
        tid = str(q.get("target_id", q.get("item_id")))
        target_positions.append(g_test_id_to_rank_pos[tid])
    target_positions = np.array(target_positions, dtype=np.int64)

    query_texts = [q["query_text"] for q in q1_queries]
    batch_size = 64

    def encode_condition(mode):
        all_embs = []
        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i+batch_size]
            if mode == "full_128_pad_128":
                inp = processor(
                    text=batch_texts,
                    padding="max_length",
                    max_length=128,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inp["input_ids"].to(device)
            elif mode == "truncate_64_pad_128":
                # Tokenize max_length 64 with truncation
                inp64 = processor(
                    text=batch_texts,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt"
                )
                ids64 = inp64["input_ids"] # [B, 64]
                # Pad remaining 64 tokens with pad_token_id
                pad_extension = torch.full((ids64.size(0), 64), pad_token_id, dtype=ids64.dtype)
                input_ids = torch.cat([ids64, pad_extension], dim=1).to(device) # [B, 128]
            elif mode == "truncate_64_pad_64":
                inp64 = processor(
                    text=batch_texts,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inp64["input_ids"].to(device) # [B, 64]

            with torch.no_grad():
                embs = model_wrapper.encode_text(input_ids)
                all_embs.append(embs.float().cpu().numpy())
        return np.vstack(all_embs)

    def evaluate_embs(text_embs):
        # Center and norm
        txt_centered = text_embs - mu_shared
        txt_norms = np.linalg.norm(txt_centered, axis=1, keepdims=True)
        txt_normed = txt_centered / np.maximum(txt_norms, 1e-12)

        # Matmul on CPU FP32
        sims = np.dot(txt_normed, g_test_normed.T) # [1290, 1290]

        ranks = []
        for i in range(len(query_texts)):
            tgt_idx = target_positions[i]
            tgt_sim = sims[i, tgt_idx]
            # Optimistic rank: strictly greater + 1
            strictly_greater = np.sum(sims[i] > tgt_sim)
            rank = strictly_greater + 1
            ranks.append(rank)

        ranks = np.array(ranks)
        r1 = float(np.mean(ranks <= 1))
        r5 = float(np.mean(ranks <= 5))
        r10 = float(np.mean(ranks <= 10))
        mrr = float(np.mean(1.0 / ranks))
        mean_rank = float(np.mean(ranks))
        median_rank = float(np.median(ranks))

        return {
            "recall_at_1": round(r1, 4),
            "recall_at_5": round(r5, 4),
            "recall_at_10": round(r10, 4),
            "mrr": round(mrr, 4),
            "mean_rank": round(mean_rank, 2),
            "median_rank": round(median_rank, 1)
        }

    conditions = [
        ("Condition A: Full 128, Pad 128 (Colab Baseline)", "full_128_pad_128"),
        ("Condition B: Truncate 64, then Pad to 128", "truncate_64_pad_128"),
        ("Condition C: Truncate 64, Pad 64 (Short Sequence)", "truncate_64_pad_64")
    ]

    results = {}
    for name, mode in conditions:
        print(f"\nEncoding & evaluating: {name}...")
        embs = encode_condition(mode)
        metrics = evaluate_embs(embs)
        results[mode] = {
            "name": name,
            "metrics": metrics
        }
        print(f"Results for {name}:")
        print(f"  R@1: {metrics['recall_at_1']}, R@5: {metrics['recall_at_5']}, R@10: {metrics['recall_at_10']}, MRR: {metrics['mrr']}, Mean Rank: {metrics['mean_rank']}")

    out_file = Path("reports/phase3g_padding_truncation_experiment.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Isolation of truncation vs padding length in SigLIP text encoding",
            "queries_count": len(q1_queries),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved experiment results to {out_file}.")

if __name__ == "__main__":
    main()
