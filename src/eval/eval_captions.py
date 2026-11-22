"""Phase 5: Paired Caption Retrieval Comparison (Config A vs Config B vs Config C).

Law 5 compliance:
- Config A: InternVL2-2B only (caption_A)
- Config B: InternVL2-2B + Auxiliary + Metadata merge_v1 (caption_B)
- Config C: Auxiliary reference only (caption_C)
- Reports 1,000-resample Bootstrap 95% CI for Config B vs Config A (seed=42).
- Note (Law D.5): SigLIP checkpoint was not trained on InternVL2 captions.
"""

from __future__ import annotations

import os

os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.eval.metrics import (
    bootstrap_ci_diff,
    compute_average_precision,
    compute_recall_at_k,
    compute_retrieval_summary,
)
from src.serve.search_engine import ECommerceSearchEngine


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def encode_texts_batched(
    engine: ECommerceSearchEngine,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode a list of texts using the single canonical SigLIPPreprocessor + SigLIP model."""
    vecs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = engine.preprocessor.preprocess_texts(batch, device=engine.device)
        emb = engine.model.encode_text(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        vecs.append(emb.float().detach().cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


def evaluate_paired_retrieval(
    query_vecs: np.ndarray,
    gallery_vecs: np.ndarray,
    topk: tuple[int, ...] = (1, 5, 10),
) -> Dict[str, Any]:
    """Compute paired T2I metrics where item i is the unique ground-truth for query i."""
    # Exact cosine similarity (N x N) for N=1,290 is fast and exact
    sim_matrix = query_vecs @ gallery_vecs.T
    n = sim_matrix.shape[0]
    max_k = max(max(topk), 100)

    ranked_relevances: List[List[int]] = []
    per_query_r1: List[float] = []
    per_query_r5: List[float] = []
    per_query_r10: List[float] = []
    per_query_ap: List[float] = []

    for i in range(n):
        row = sim_matrix[i]
        # Partial sort for top max_k
        top_idx = np.argpartition(-row, max_k)[:max_k]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        rels = [1 if idx == i else 0 for idx in top_idx]
        ranked_relevances.append(rels)

        per_query_r1.append(compute_recall_at_k(rels, 1))
        per_query_r5.append(compute_recall_at_k(rels, 5))
        per_query_r10.append(compute_recall_at_k(rels, 10))
        # For AP with 1 GT item, if not in top max_k, compute exact rank
        if 1 in rels:
            per_query_ap.append(compute_average_precision(rels, total_relevant=1))
        else:
            rank = int(np.sum(row > row[i])) + 1
            per_query_ap.append(1.0 / rank)

    summary = compute_retrieval_summary(ranked_relevances, total_relevants=[1] * n, topk=topk)
    summary["mAP"] = float(np.mean(per_query_ap))
    return {
        "summary": summary,
        "per_query": {
            "recall@1": per_query_r1,
            "recall@5": per_query_r5,
            "recall@10": per_query_r10,
            "ap": per_query_ap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Caption Config A/B/C Paired Retrieval Evaluation")
    parser.add_argument("--index-dir", default="index/test_gallery")
    parser.add_argument("--captions-glob", default="data/catalog/captions/shard_test_*.parquet")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/phase5/caption_eval_ab.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = ECommerceSearchEngine(index_dir=args.index_dir, config_path=args.config)
    id_map = pd.DataFrame(
        [{"item_id": str(item["item_id"]), "index_pos": i} for i, item in enumerate(engine.id_map)]
    )

    shard_paths = sorted(Path().glob(args.captions_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No caption shards matched: {args.captions_glob}")
    cap_df = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    cap_df["item_id"] = cap_df["product_id"].astype(str)
    cap_df = cap_df.drop_duplicates(subset=["item_id"], keep="first")

    merged = id_map.merge(cap_df, on="item_id", how="inner").sort_values("index_pos").reset_index(drop=True)
    if len(merged) != len(id_map):
        raise RuntimeError(f"Caption join mismatch: got {len(merged)} vs gallery {len(id_map)}")

    gallery_vecs = engine.embeddings[merged["index_pos"].to_numpy()]

    configs = {
        "config_A_internvl2_only": merged["caption_A"].fillna("").astype(str).tolist(),
        "config_B_internvl2_plus_aux": merged["caption_B"].fillna("").astype(str).tolist(),
        "config_C_vintern_ref": merged["caption_C"].fillna("").astype(str).tolist(),
    }

    eval_results: Dict[str, Any] = {}
    for name, texts in configs.items():
        print(f"Encoding {name} ({len(texts)} captions)...")
        q_vecs = encode_texts_batched(engine, texts, batch_size=args.batch_size)
        eval_results[name] = evaluate_paired_retrieval(q_vecs, gallery_vecs, topk=(1, 5, 10))

    # Bootstrap 95% CI comparing Config B vs Config A
    pq_a = eval_results["config_A_internvl2_only"]["per_query"]
    pq_b = eval_results["config_B_internvl2_plus_aux"]["per_query"]
    bootstrap_b_vs_a = {}
    for metric_key in ("recall@1", "recall@5", "recall@10", "ap"):
        bootstrap_b_vs_a[metric_key] = bootstrap_ci_diff(
            pq_a[metric_key],
            pq_b[metric_key],
            n_resamples=1000,
            ci=0.95,
            seed=args.seed,
        )

    out_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "seed": args.seed,
            "index_dir": args.index_dir,
            "n_items": int(len(merged)),
            "note_law_d5": "SigLIP checkpoint was not trained on InternVL2 captions; paired retrieval scores with new captions cannot be compared directly to historical training-time metrics.",
        },
        "summaries": {k: v["summary"] for k, v in eval_results.items()},
        "bootstrap_95ci_B_vs_A": bootstrap_b_vs_a,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved caption eval to {out_path}")
    print(json.dumps(out_payload["summaries"], indent=2))
    print("Bootstrap B vs A:", json.dumps(bootstrap_b_vs_a, indent=2))


if __name__ == "__main__":
    main()
