"""
Compute CIR evaluations under standard convention:
Reference image is excluded from candidate ranking for all rows
(cir_i2i_baseline, cir_v1_01, cir_v1_05), ranking 1,289 items.

Prompt 14j Step 1.
Seed: 42, Bootstrap: 1,000 resamples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.locked_final_definitions import (
    N_BOOTSTRAP_LOCKED,
    SEED_LOCKED,
    compute_full_ranking_ap,
    compute_hit_rate_at_k,
    compute_true_recall_at_k,
    paired_query_bootstrap_diff,
    query_bootstrap_ci,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_paired_diff_label(ci_low: float, ci_high: float, name_a: str, name_b: str) -> str:
    """Mechanical paired difference classification:
    - CI excludes 0 (both bounds > 0): significant difference, name_a > name_b
    - CI excludes 0 (both bounds < 0): significant difference, name_b > name_a
    - CI contains or touches 0: inconclusive
    """
    if ci_low > 0.0:
        return f"Khác biệt có ý nghĩa ({name_a} hơn {name_b})"
    elif ci_high < 0.0:
        return f"Khác biệt có ý nghĩa ({name_b} hơn {name_a})"
    else:
        return "Chưa kết luận được (CI chứa hoặc chạm 0)"


def compute_metrics_bundle(
    r1: List[float],
    r5: List[float],
    r10: List[float],
    tr10: List[float],
    ap: List[float],
    n_bootstrap: int = N_BOOTSTRAP_LOCKED,
    seed: int = SEED_LOCKED,
) -> Dict[str, Any]:
    n = len(ap)
    note = f"Cảnh báo n nhỏ: n={n} < 20, phương sai bootstrap lớn" if n < 20 else None
    return {
        "n_queries": n,
        "small_n_warning": note,
        "recall@1": query_bootstrap_ci(r1, n_resamples=n_bootstrap, seed=seed),
        "recall@5": query_bootstrap_ci(r5, n_resamples=n_bootstrap, seed=seed),
        "recall@10": query_bootstrap_ci(r10, n_resamples=n_bootstrap, seed=seed),
        "true_recall@10": query_bootstrap_ci(tr10, n_resamples=n_bootstrap, seed=seed),
        "mAP": query_bootstrap_ci(ap, n_resamples=n_bootstrap, seed=seed),
    }


def compute_paired_bundle(
    scores_a: Dict[str, List[float]],
    scores_b: Dict[str, List[float]],
    name_a: str,
    name_b: str,
    n_bootstrap: int = N_BOOTSTRAP_LOCKED,
    seed: int = SEED_LOCKED,
) -> Dict[str, Any]:
    n = len(scores_a["ap"])
    metrics_map = {
        "recall@1": ("r1", paired_query_bootstrap_diff(scores_a["r1"], scores_b["r1"], n_bootstrap, seed)),
        "recall@5": ("r5", paired_query_bootstrap_diff(scores_a["r5"], scores_b["r5"], n_bootstrap, seed)),
        "recall@10": ("r10", paired_query_bootstrap_diff(scores_a["r10"], scores_b["r10"], n_bootstrap, seed)),
        "mAP": ("ap", paired_query_bootstrap_diff(scores_a["ap"], scores_b["ap"], n_bootstrap, seed)),
    }
    res: Dict[str, Any] = {
        "n_pairs": n,
        "small_n_warning": f"Cảnh báo n nhỏ: n={n} < 20, phương sai bootstrap lớn" if n < 20 else None,
        "comparison": f"{name_a} vs {name_b}",
    }
    for m_key, (raw_k, diff_d) in metrics_map.items():
        lbl = classify_paired_diff_label(diff_d["ci_low"], diff_d["ci_high"], name_a, name_b)
        res[m_key] = {
            "mean_diff": diff_d["mean_diff"],
            "ci_low": diff_d["ci_low"],
            "ci_high": diff_d["ci_high"],
            "n": diff_d["n"],
            "label": lbl,
        }
    return res


def main() -> int:
    print("================================================================================")
    print("COMPUTING CIR EVALUATIONS UNDER STANDARD CONVENTION (PROMPT 14j STEP 1)")
    print("Reference Image Excluded from Candidates (N=1,289)")
    print("================================================================================")

    cir_test_path = PROJECT_ROOT / "eval/cir_eval_test.json"
    with open(cir_test_path, "r", encoding="utf-8") as f:
        cir_test = json.load(f)
    pairs_map = {p["id"]: p for p in cir_test["pairs"]}

    rankings_dir = PROJECT_ROOT / "reports/phase5c/rankings"
    f1_path = rankings_dir / "cir_v1_01_primary.json"
    f5_path = rankings_dir / "cir_v1_05_primary.json"
    fb_path = rankings_dir / "cir_i2i_baseline.json"

    data_01 = json.loads(f1_path.read_text(encoding="utf-8"))
    data_05 = json.loads(f5_path.read_text(encoding="utf-8"))
    data_b = json.loads(fb_path.read_text(encoding="utf-8"))

    # Compute per-query metrics excluding reference image
    datasets = {
        "cir_i2i_baseline": data_b,
        "cir_v1_01": data_01,
        "cir_v1_05": data_05,
    }

    per_query_results: Dict[str, Dict[str, List[float]]] = {}
    pair_metadata: List[Dict[str, Any]] = []

    for name, dset in datasets.items():
        r1_list, r5_list, r10_list, tr10_list, ap_list = [], [], [], [], []
        for row in dset:
            pair_id = row["pair_id"]
            p_info = pairs_map[pair_id]
            ref_pid = str(p_info["ref_product_id"])
            gt_set: Set[str] = {str(x) for x in p_info["gt_product_ids"]}

            # Exclude ref_pid from candidate ranking
            ranked_filtered = [pid for pid in row["ranked_product_ids"] if str(pid) != ref_pid]
            assert len(ranked_filtered) == 1289, f"Expected 1289 items, got {len(ranked_filtered)}"

            rels = [1 if str(pid) in gt_set else 0 for pid in ranked_filtered]
            r1 = compute_hit_rate_at_k(rels, 1)
            r5 = compute_hit_rate_at_k(rels, 5)
            r10 = compute_hit_rate_at_k(rels, 10)
            tr10 = compute_true_recall_at_k(rels, 10, total_relevant=len(gt_set))
            ap = compute_full_ranking_ap(rels, total_relevant=len(gt_set))

            r1_list.append(r1)
            r5_list.append(r5)
            r10_list.append(r10)
            tr10_list.append(tr10)
            ap_list.append(ap)

            if name == "cir_v1_05":
                pair_metadata.append({
                    "pair_id": pair_id,
                    "changed_attr": p_info["changed_attr"],
                    "ref_product_id": ref_pid,
                })

        per_query_results[name] = {
            "r1": r1_list,
            "r5": r5_list,
            "r10": r10_list,
            "tr10": tr10_list,
            "ap": ap_list,
        }

    # Stratify subsets: overall, baseColour, gender
    attrs = [p["changed_attr"] for p in pair_metadata]
    idx_overall = list(range(len(attrs)))
    idx_colour = [i for i, a in enumerate(attrs) if a == "baseColour"]
    idx_gender = [i for i, a in enumerate(attrs) if a == "gender"]

    subsets_idx = {
        "overall": idx_overall,
        "baseColour": idx_colour,
        "gender": idx_gender,
    }

    summary_tables: Dict[str, Any] = {}
    paired_tables: Dict[str, Any] = {}

    for sub_name, idxs in subsets_idx.items():
        summary_tables[sub_name] = {}
        sub_scores: Dict[str, Dict[str, List[float]]] = {}

        for cfg_name in ["cir_i2i_baseline", "cir_v1_01", "cir_v1_05"]:
            full_s = per_query_results[cfg_name]
            s_dict = {
                "r1": [full_s["r1"][i] for i in idxs],
                "r5": [full_s["r5"][i] for i in idxs],
                "r10": [full_s["r10"][i] for i in idxs],
                "tr10": [full_s["tr10"][i] for i in idxs],
                "ap": [full_s["ap"][i] for i in idxs],
            }
            sub_scores[cfg_name] = s_dict
            summary_tables[sub_name][cfg_name] = compute_metrics_bundle(
                s_dict["r1"], s_dict["r5"], s_dict["r10"], s_dict["tr10"], s_dict["ap"]
            )

        paired_tables[sub_name] = {
            "cir_v1_01_vs_baseline": compute_paired_bundle(
                sub_scores["cir_v1_01"], sub_scores["cir_i2i_baseline"], "cir_v1_01", "cir_i2i_baseline"
            ),
            "cir_v1_05_vs_baseline": compute_paired_bundle(
                sub_scores["cir_v1_05"], sub_scores["cir_i2i_baseline"], "cir_v1_05", "cir_i2i_baseline"
            ),
            "cir_v1_05_vs_cir_v1_01": compute_paired_bundle(
                sub_scores["cir_v1_05"], sub_scores["cir_v1_01"], "cir_v1_05", "cir_v1_01"
            ),
        }

    # Print summary
    print("\n--- Summary Table (Standard Convention: Excluding Ref Image, N=1,289) ---")
    for sub_name in ["overall", "baseColour", "gender"]:
        print(f"\n[Subset: {sub_name} (n={len(subsets_idx[sub_name])})]")
        for cfg_name in ["cir_i2i_baseline", "cir_v1_01", "cir_v1_05"]:
            b = summary_tables[sub_name][cfg_name]
            print(f"  {cfg_name:20s}: R@1={b['recall@1']['mean']:.4f} [{b['recall@1']['ci_low']:.4f}, {b['recall@1']['ci_high']:.4f}], R@5={b['recall@5']['mean']:.4f}, R@10={b['recall@10']['mean']:.4f} [{b['recall@10']['ci_low']:.4f}, {b['recall@10']['ci_high']:.4f}], mAP={b['mAP']['mean']:.4f} [{b['mAP']['ci_low']:.4f}, {b['mAP']['ci_high']:.4f}]")

    print("\n--- Paired Bootstrap Differences (Standard Convention) ---")
    for sub_name in ["overall", "baseColour", "gender"]:
        print(f"\n[Subset: {sub_name}]")
        for comp_name, p_bundle in paired_tables[sub_name].items():
            print(f"  {comp_name:25s}:")
            for m in ["recall@1", "recall@5", "recall@10", "mAP"]:
                md = p_bundle[m]
                print(f"    {m:10s}: Δ = {md['mean_diff']:+.4f} [{md['ci_low']:+.4f}, {md['ci_high']:+.4f}] -> {md['label']}")

    # Save to dedicated json file
    payload = {
        "status": "CIR_STANDARD_CONVENTION_EVALUATION_COMPLETED",
        "description": "CIR metrics under standard convention: reference image excluded from candidate pool (N=1,289)",
        "protocol": "standard_convention_ref_excluded",
        "n_gallery_after_filter": 1289,
        "seed": SEED_LOCKED,
        "n_bootstrap": N_BOOTSTRAP_LOCKED,
        "summary_by_subset": summary_tables,
        "paired_differences_by_subset": paired_tables,
    }

    out_path = PROJECT_ROOT / "reports/phase5c/cir_standard_convention_results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    h_out = sha256_file(out_path)
    (PROJECT_ROOT / "reports/phase5c/cir_standard_convention_results.sha256").write_text(
        f"{h_out}  reports/phase5c/cir_standard_convention_results.json\n", encoding="utf-8"
    )
    print(f"\n[Saved JSON] {out_path} (SHA-256: {h_out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
