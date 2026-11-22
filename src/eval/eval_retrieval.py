"""Phase 5: Official Test Gallery & Reference Gallery Retrieval + Hubness Evaluation.

Compliance:
- Official metrics strictly on Official Test Gallery (index/test_gallery).
- Reference Full Gallery (index/reference_gallery) explicitly labeled 'ước lượng lạc quan (tham khảo)'.
- Separates 'Model thuần' (model_only) and 'Hệ thống' (system, gamma=0.3).
- Evaluates Before and After Mean-Centering + Hubness stats (C.8).
- Reports hybrid (official), strict, and relaxed gender GT policies.
- Breakdowns by thin_slot (category), colour, gender, has_diacritics.
- 1,000x Bootstrap 95% CI for system vs model_only.
- Also evaluates Image-to-Image (200 sampled test items, seed=42).
"""

from __future__ import annotations

import os

os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.metrics import (
    bootstrap_ci_diff,
    compute_average_precision,
    compute_hubness_stats,
    compute_recall_at_k,
    compute_retrieval_summary,
    is_item_relevant,
    load_gender_gt_policy,
)
from src.serve.search_engine import ECommerceSearchEngine


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _evaluate_t2i_gallery(
    engine: ECommerceSearchEngine,
    queries: List[Dict[str, Any]],
    query_vecs: np.ndarray,
    parsed_queries: List[Dict[str, Any]],
    gender_policy_cfg: Dict[str, Any],
    mode: str,
    eval_top_k: int = 100,
    compute_breakdowns: bool = False,
) -> Dict[str, Any]:
    """Evaluate Text-to-Image queries on an initialized ECommerceSearchEngine."""
    strict_cats = gender_policy_cfg["policies"]["hybrid"]["strict_categories"]
    policies = ["hybrid", "strict", "relaxed"]
    id_map_records = engine.id_map

    # Precompute search results once per query (independent of GT gender policy)
    ranked_indices_list: List[List[int]] = []
    for i, q in enumerate(queries):
        q_vec = engine._apply_mean_centering(query_vecs[i : i + 1])
        results = engine.search_by_vector(
            query_vec=q_vec,
            top_k=eval_top_k,
            parsed_query=parsed_queries[i],
            mode=mode,
            candidate_pool=max(100, eval_top_k),
        )
        ranked_indices_list.append([r["index_pos"] for r in results])

    hubness = compute_hubness_stats(
        [idxs[:10] for idxs in ranked_indices_list],
        gallery_size=len(id_map_records),
        top_k=10,
    )

    policy_outputs: Dict[str, Any] = {}
    for pol in policies:
        ranked_rels: List[List[int]] = []
        tot_rels: List[int] = []
        per_q_r1: List[float] = []
        per_q_r5: List[float] = []
        per_q_r10: List[float] = []
        per_q_ap: List[float] = []
        valid_mask: List[bool] = []

        for i, q in enumerate(queries):
            spec = q["spec"]
            # Count exact total relevant items in gallery
            total_rel = sum(
                1 for meta in id_map_records if is_item_relevant(spec, meta, pol, strict_cats)
            )
            tot_rels.append(total_rel)
            valid_mask.append(total_rel > 0)

            rels = [
                1 if is_item_relevant(spec, id_map_records[idx], pol, strict_cats) else 0
                for idx in ranked_indices_list[i]
            ]
            ranked_rels.append(rels)
            per_q_r1.append(compute_recall_at_k(rels, 1))
            per_q_r5.append(compute_recall_at_k(rels, 5))
            per_q_r10.append(compute_recall_at_k(rels, 10))
            per_q_ap.append(compute_average_precision(rels, total_relevant=total_rel) if total_rel > 0 else 0.0)

        # Summary over queries with >=1 GT in gallery (and also record total queries)
        valid_idxs = [i for i, ok in enumerate(valid_mask) if ok]
        summary_valid = compute_retrieval_summary(
            [ranked_rels[i] for i in valid_idxs],
            total_relevants=[tot_rels[i] for i in valid_idxs],
            topk=(1, 5, 10),
        )
        summary_valid["n_queries_total"] = len(queries)
        summary_valid["n_queries_with_gt"] = len(valid_idxs)
        policy_outputs[pol] = {
            "summary": summary_valid,
            "per_query_valid": {
                "indices": valid_idxs,
                "recall@1": [per_q_r1[i] for i in valid_idxs],
                "recall@5": [per_q_r5[i] for i in valid_idxs],
                "recall@10": [per_q_r10[i] for i in valid_idxs],
                "ap": [per_q_ap[i] for i in valid_idxs],
            },
        }

        if compute_breakdowns and pol == "hybrid":
            breakdowns: Dict[str, Dict[str, Any]] = {
                "by_thin_slot": {},
                "by_has_diacritics": {},
                "by_gender": {},
                "by_colour": {},
            }
            groups: Dict[str, Dict[str, List[int]]] = {
                "by_thin_slot": defaultdict(list),
                "by_has_diacritics": defaultdict(list),
                "by_gender": defaultdict(list),
                "by_colour": defaultdict(list),
            }
            for i in valid_idxs:
                q = queries[i]
                groups["by_thin_slot"][str(q.get("category", "unknown"))].append(i)
                groups["by_has_diacritics"][str(q.get("has_diacritics", True))].append(i)
                groups["by_gender"][str(q["spec"].get("gender") or "Unspecified")].append(i)
                colours = q["spec"].get("baseColour") or ["Unspecified"]
                for c in colours:
                    groups["by_colour"][str(c)].append(i)

            for b_name, g_map in groups.items():
                for g_key, idxs in sorted(g_map.items()):
                    breakdowns[b_name][g_key] = compute_retrieval_summary(
                        [ranked_rels[i] for i in idxs],
                        total_relevants=[tot_rels[i] for i in idxs],
                        topk=(1, 5, 10),
                    )
            policy_outputs[pol]["breakdowns"] = breakdowns

    return {
        "policies": policy_outputs,
        "hubness_top10": hubness,
    }


def _evaluate_i2i_gallery(
    engine: ECommerceSearchEngine,
    n_queries: int = 200,
    seed: int = 42,
    eval_top_k: int = 100,
) -> Dict[str, Any]:
    """Evaluate Image-to-Image retrieval on 200 sampled gallery items (excluding self)."""
    rng = np.random.default_rng(seed)
    n_gallery = len(engine.id_map)
    sample_size = min(n_queries, n_gallery)
    sampled_idxs = rng.choice(n_gallery, size=sample_size, replace=False)

    records = engine.id_map
    ranked_rels: List[List[int]] = []
    tot_rels: List[int] = []
    topk_for_hubness: List[List[int]] = []

    for q_idx in sampled_idxs:
        q_meta = records[int(q_idx)]
        q_type = q_meta.get("articleType")
        q_gender = q_meta.get("gender")
        q_colour = q_meta.get("baseColour")

        # Count other items matching articleType + gender + baseColour
        total_rel = sum(
            1
            for j, m in enumerate(records)
            if j != int(q_idx)
            and m.get("articleType") == q_type
            and m.get("gender") == q_gender
            and m.get("baseColour") == q_colour
        )
        if total_rel == 0:
            continue

        q_vec = engine._apply_mean_centering(engine.embeddings[int(q_idx) : int(q_idx) + 1])
        results = engine.search_by_vector(
            query_vec=q_vec,
            top_k=eval_top_k + 1,
            mode="model_only",
        )
        filtered = [r for r in results if r["index_pos"] != int(q_idx)][:eval_top_k]
        topk_for_hubness.append([r["index_pos"] for r in filtered[:10]])

        rels = [
            1
            if (
                r.get("articleType") == q_type
                and r.get("gender") == q_gender
                and r.get("baseColour") == q_colour
            )
            else 0
            for r in filtered
        ]
        ranked_rels.append(rels)
        tot_rels.append(total_rel)

    summary = compute_retrieval_summary(ranked_rels, total_relevants=tot_rels, topk=(1, 5, 10))
    summary["n_sampled"] = int(sample_size)
    summary["n_queries_with_gt"] = len(ranked_rels)
    hubness = compute_hubness_stats(topk_for_hubness, gallery_size=n_gallery, top_k=10)
    return {"summary": summary, "hubness_top10": hubness}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Retrieval & Hubness Benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--queries", default="eval/handwritten_queries.json")
    parser.add_argument("--gender-policy", default="eval/gender_gt_policy.json")
    parser.add_argument("--official-index", default="index/test_gallery")
    parser.add_argument("--reference-index", default="index/reference_gallery")
    parser.add_argument("--output", default="reports/phase5/retrieval_eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)
    gender_policy_cfg = load_gender_gt_policy(args.gender_policy)

    # 1. Encode all 100 handwritten queries ONCE using the canonical SigLIP pipeline
    print(f"Loading Official Test Gallery ({args.official_index}) without mean-centering...")
    engine_off_raw = ECommerceSearchEngine(
        index_dir=args.official_index,
        config_path=args.config,
        use_mean_centering=False,
    )
    print(f"Encoding {len(queries)} unseen handwritten queries...")
    query_vecs = np.vstack([engine_off_raw.encode_text_query(q["query"]) for q in queries])
    parsed_queries = [engine_off_raw.parser.parse(q["query"]) for q in queries]

    # Official Test Gallery — Before Mean-Centering
    off_raw_model = _evaluate_t2i_gallery(
        engine_off_raw, queries, query_vecs, parsed_queries, gender_policy_cfg,
        mode="model_only", compute_breakdowns=True,
    )
    off_raw_sys = _evaluate_t2i_gallery(
        engine_off_raw, queries, query_vecs, parsed_queries, gender_policy_cfg,
        mode="system", compute_breakdowns=True,
    )
    off_raw_i2i = _evaluate_i2i_gallery(engine_off_raw, n_queries=200, seed=args.seed)

    # Official Test Gallery — After Mean-Centering
    print(f"Loading Official Test Gallery ({args.official_index}) WITH mean-centering...")
    engine_off_mc = ECommerceSearchEngine(
        index_dir=args.official_index,
        config_path=args.config,
        use_mean_centering=True,
    )
    off_mc_model = _evaluate_t2i_gallery(
        engine_off_mc, queries, query_vecs, parsed_queries, gender_policy_cfg,
        mode="model_only", compute_breakdowns=True,
    )
    off_mc_sys = _evaluate_t2i_gallery(
        engine_off_mc, queries, query_vecs, parsed_queries, gender_policy_cfg,
        mode="system", compute_breakdowns=True,
    )
    off_mc_i2i = _evaluate_i2i_gallery(engine_off_mc, n_queries=200, seed=args.seed)

    # Bootstrap 95% CI: System vs Model-only on Official Test Gallery (Before & After MC, hybrid)
    bootstrap_sys_vs_model_raw = {}
    bootstrap_sys_vs_model_mc = {}
    for m_key in ("recall@1", "recall@5", "recall@10", "ap"):
        bootstrap_sys_vs_model_raw[m_key] = bootstrap_ci_diff(
            off_raw_model["policies"]["hybrid"]["per_query_valid"][m_key],
            off_raw_sys["policies"]["hybrid"]["per_query_valid"][m_key],
            n_resamples=1000,
            seed=args.seed,
        )
        bootstrap_sys_vs_model_mc[m_key] = bootstrap_ci_diff(
            off_mc_model["policies"]["hybrid"]["per_query_valid"][m_key],
            off_mc_sys["policies"]["hybrid"]["per_query_valid"][m_key],
            n_resamples=1000,
            seed=args.seed,
        )

    # Strip bulky per_query arrays before saving JSON
    def _strip_per_query(res: Dict[str, Any]) -> Dict[str, Any]:
        for pol_data in res["policies"].values():
            pol_data.pop("per_query_valid", None)
        return res

    # 2. Reference Full Gallery (ước lượng lạc quan - tham khảo)
    ref_results: Dict[str, Any] = {}
    if Path(args.reference_index).exists():
        print(f"Loading Reference Full Gallery ({args.reference_index})...")
        engine_ref_raw = ECommerceSearchEngine(
            index_dir=args.reference_index,
            config_path=args.config,
            use_mean_centering=False,
        )
        ref_raw_model = _strip_per_query(
            _evaluate_t2i_gallery(
                engine_ref_raw, queries, query_vecs, parsed_queries, gender_policy_cfg,
                mode="model_only", compute_breakdowns=False,
            )
        )
        ref_raw_sys = _strip_per_query(
            _evaluate_t2i_gallery(
                engine_ref_raw, queries, query_vecs, parsed_queries, gender_policy_cfg,
                mode="system", compute_breakdowns=False,
            )
        )
        ref_results = {
            "label": "ước lượng lạc quan (tham khảo)",
            "gallery_size": int(len(engine_ref_raw.id_map)),
            "before_mean_centering": {
                "model_only": ref_raw_model,
                "system": ref_raw_sys,
            },
        }

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "seed": args.seed,
            "queries_file": args.queries,
            "n_queries": len(queries),
            "gamma": engine_off_raw.gamma,
            "ef_search": engine_off_raw.ef_search,
        },
        "official_test_gallery": {
            "gallery_size": int(len(engine_off_raw.id_map)),
            "before_mean_centering": {
                "t2i_model_only": _strip_per_query(off_raw_model),
                "t2i_system": _strip_per_query(off_raw_sys),
                "i2i": off_raw_i2i,
                "bootstrap_95ci_system_vs_model": bootstrap_sys_vs_model_raw,
            },
            "after_mean_centering": {
                "t2i_model_only": _strip_per_query(off_mc_model),
                "t2i_system": _strip_per_query(off_mc_sys),
                "i2i": off_mc_i2i,
                "bootstrap_95ci_system_vs_model": bootstrap_sys_vs_model_mc,
            },
        },
        "reference_full_gallery": ref_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved retrieval eval to {out_path}")
    print("Official Test Gallery (Hybrid, Before MC) Model-only:", json.dumps(payload["official_test_gallery"]["before_mean_centering"]["t2i_model_only"]["policies"]["hybrid"]["summary"], indent=2))
    print("Official Test Gallery (Hybrid, Before MC) System:", json.dumps(payload["official_test_gallery"]["before_mean_centering"]["t2i_system"]["policies"]["hybrid"]["summary"], indent=2))
    print("Official Test Gallery (Hybrid, After MC) System:", json.dumps(payload["official_test_gallery"]["after_mean_centering"]["t2i_system"]["policies"]["hybrid"]["summary"], indent=2))


if __name__ == "__main__":
    main()
