"""
Experiment: Truncate 64 then Pad 128 vs Full 128 Pad 128 vs Pad 64 (Phase 3g Rev5)
Evaluates on all 1,290 paired test queries (Q1b) on G_test and G_full using train-format caption:
"[{category}] {title} ({attrs})" (from data/checkpoints_siglip-vit/product_metadata.json).

Isolates:
1. Information loss due to truncation (Tokens 65..128 removed)
2. Sequence length / padding artifacts (Seq len 128 vs Seq len 64)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
from src.model_siglip import load_siglip_inference
from src.scoring import Scorer

def main():
    print("=================================================================")
    print("STARTING PADDING & TRUNCATION EXPERIMENT (1,290 QUERIES)...")
    print("=================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP model on {device}...")
    model_wrapper = load_siglip_inference(
        checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
        device=device,
        fp16=(device == "cuda")
    )
    processor = model_wrapper.processor
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 1

    # Load 1,290 train-format Q1b queries
    with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}

    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    g_test_ids = [str(x) for x in testset["g_test_ids"]]
    q1_texts = [meta_dict[tid] for tid in g_test_ids]
    print(f"Loaded {len(q1_texts)} train-format captions for Q1b.")

    # Load id_map_v2 and align embeddings
    id_map = json.load(open("index_v2/id_map_v2.json", "r", encoding="utf-8"))
    id_to_idx = {str(item["item_id"] if isinstance(item, dict) else item): idx for idx, item in enumerate(id_map)}

    g_test_indices = np.array([id_to_idx[tid] for tid in g_test_ids], dtype=np.int64)

    scorer = Scorer("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    raw_img_embs = np.load("index_v2/image_embeddings_v2.npy")
    emb_img_c = scorer.center_vectors(raw_img_embs)
    g_test_emb = emb_img_c[g_test_indices]

    batch_size = 64

    def encode_condition(mode):
        all_embs = []
        for i in range(0, len(q1_texts), batch_size):
            batch_texts = q1_texts[i:i+batch_size]
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
                tok64 = processor(
                    text=batch_texts,
                    padding=False,
                    max_length=64,
                    truncation=True
                )
                padded_ids = []
                for seq in tok64["input_ids"]:
                    if len(seq) < 128:
                        seq = seq + [pad_token_id] * (128 - len(seq))
                    else:
                        seq = seq[:128]
                    padded_ids.append(seq)
                input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            elif mode == "truncate_64_pad_64":
                inp = processor(
                    text=batch_texts,
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inp["input_ids"].to(device)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            with torch.no_grad():
                feats = model_wrapper.encode_text(input_ids, None)
                v = feats.cpu().numpy()
            norms = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            all_embs.append(v / norms)

        raw_vecs = np.vstack(all_embs)
        return scorer.center_vectors(raw_vecs)

    conditions = [
        ("full_128_pad_128", "Condition A: Full 128, Pad 128 (Colab Baseline)"),
        ("truncate_64_pad_128", "Condition B: Truncate 64, then Pad to 128"),
        ("truncate_64_pad_64", "Condition C: Truncate 64, Pad 64 (Short Sequence)")
    ]

    results = {}
    target_full = [id_to_idx[tid] for tid in g_test_ids]

    for cond_key, cond_name in conditions:
        print(f"\nEvaluating {cond_name}...")
        q_embs_c = encode_condition(cond_key)

        # G_test evaluation
        sims_test = np.dot(q_embs_c, g_test_emb.T)
        ranks_test = [int(np.sum(sims_test[i] > sims_test[i, i])) + 1 for i in range(len(q1_texts))]
        r_test_arr = np.array(ranks_test)

        # G_full evaluation
        sims_full = np.dot(q_embs_c, emb_img_c.T)
        ranks_full = [int(np.sum(sims_full[i] > sims_full[i, target_full[i]])) + 1 for i in range(len(q1_texts))]
        r_full_arr = np.array(ranks_full)

        results[cond_key] = {
            "name": cond_name,
            "g_test": {
                "recall_at_1": round(float(np.mean(r_test_arr <= 1)), 4),
                "recall_at_5": round(float(np.mean(r_test_arr <= 5)), 4),
                "recall_at_10": round(float(np.mean(r_test_arr <= 10)), 4),
                "mrr": round(float(np.mean(1.0 / r_test_arr)), 4),
                "mean_rank": round(float(np.mean(r_test_arr)), 2),
                "median_rank": round(float(np.median(r_test_arr)), 1)
            },
            "g_full": {
                "recall_at_1": round(float(np.mean(r_full_arr <= 1)), 4),
                "recall_at_5": round(float(np.mean(r_full_arr <= 5)), 4),
                "recall_at_10": round(float(np.mean(r_full_arr <= 10)), 4),
                "mrr": round(float(np.mean(1.0 / r_full_arr)), 4),
                "mean_rank": round(float(np.mean(r_full_arr)), 2),
                "median_rank": round(float(np.median(r_full_arr)), 1)
            }
        }
        print(f"  G_test -> R@10={results[cond_key]['g_test']['recall_at_10']}, MRR={results[cond_key]['g_test']['mrr']}, MeanRank={results[cond_key]['g_test']['mean_rank']}")
        print(f"  G_full -> R@10={results[cond_key]['g_full']['recall_at_10']}, MRR={results[cond_key]['g_full']['mrr']}, MeanRank={results[cond_key]['g_full']['mean_rank']}")

    out_file = Path("reports/phase3g_padding_truncation_experiment.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "task": "Isolation of truncation vs padding length in SigLIP text encoding",
            "queries_count": len(q1_texts),
            "caption_format": "[{category}] {title} ({attrs})",
            "results": results
        }, f, indent=2)

    print(f"\nSaved results to {out_file}.")

if __name__ == "__main__":
    main()
