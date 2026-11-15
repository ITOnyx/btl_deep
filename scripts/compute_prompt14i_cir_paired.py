"""
Compute CIR paired comparisons with reference-only baseline and across subsets (Prompt 14i Step 1).

Label: "BỔ SUNG HOÀN THIỆN KẾ HOẠCH (tính sau khi xem kết quả chính)"
Seed: 42, Bootstrap: 1,000 iterations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.locked_final_definitions import (
    N_BOOTSTRAP_LOCKED,
    SEED_LOCKED,
    paired_query_bootstrap_diff,
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


def compute_paired_bundle(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    name_a: str,
    name_b: str,
    n_bootstrap: int = N_BOOTSTRAP_LOCKED,
    seed: int = SEED_LOCKED,
) -> Dict[str, Any]:
    n = len(records_a)
    assert len(records_b) == n, f"Length mismatch: {len(records_a)} vs {len(records_b)}"

    metrics_map = {
        "recall@1": ("r1", paired_query_bootstrap_diff([r["r1"] for r in records_a], [r["r1"] for r in records_b], n_bootstrap, seed)),
        "recall@5": ("r5", paired_query_bootstrap_diff([r["r5"] for r in records_a], [r["r5"] for r in records_b], n_bootstrap, seed)),
        "recall@10": ("r10", paired_query_bootstrap_diff([r["r10"] for r in records_a], [r["r10"] for r in records_b], n_bootstrap, seed)),
        "mAP": ("ap", paired_query_bootstrap_diff([r["ap"] for r in records_a], [r["ap"] for r in records_b], n_bootstrap, seed)),
    }

    res: Dict[str, Any] = {
        "n_pairs": n,
        "small_n_warning": f"Cảnh báo n nhỏ: n={n} < 20, phương sai bootstrap lớn" if n < 20 else None,
        "comparison": f"{name_a} vs {name_b}",
    }

    for m_key, (raw_key, diff_dict) in metrics_map.items():
        lbl = classify_paired_diff_label(diff_dict["ci_low"], diff_dict["ci_high"], name_a, name_b)
        res[m_key] = {
            "mean_diff": diff_dict["mean_diff"],
            "ci_low": diff_dict["ci_low"],
            "ci_high": diff_dict["ci_high"],
            "n": diff_dict["n"],
            "label": lbl,
        }
    return res


def main() -> int:
    rankings_dir = PROJECT_ROOT / "reports/phase5c/rankings"
    f1_path = rankings_dir / "cir_v1_01_primary.json"
    f5_path = rankings_dir / "cir_v1_05_primary.json"
    fb_path = rankings_dir / "cir_i2i_baseline.json"

    assert f1_path.exists(), f"Missing {f1_path}"
    assert f5_path.exists(), f"Missing {f5_path}"
    assert fb_path.exists(), f"Missing {fb_path}"

    with open(f1_path, "r", encoding="utf-8") as f:
        data_01 = json.load(f)
    with open(f5_path, "r", encoding="utf-8") as f:
        data_05 = json.load(f)
    with open(fb_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    n_total = len(data_01)
    assert n_total == 120 and len(data_05) == 120 and len(data_b) == 120

    # Verify identical pair ordering
    pids_01 = [r["pair_id"] for r in data_01]
    pids_05 = [r["pair_id"] for r in data_05]
    pids_b = [r["pair_id"] for r in data_b]
    assert pids_01 == pids_05 == pids_b, "Ordering mismatch between CIR ranking files!"

    # Stratification subsets
    subsets = {
        "overall": (data_01, data_05, data_b),
        "baseColour": (
            [r for r in data_01 if r["changed_attr"] == "baseColour"],
            [r for r in data_05 if r["changed_attr"] == "baseColour"],
            [r for r in data_b if r["changed_attr"] == "baseColour"],
        ),
        "gender": (
            [r for r in data_01 if r["changed_attr"] == "gender"],
            [r for r in data_05 if r["changed_attr"] == "gender"],
            [r for r in data_b if r["changed_attr"] == "gender"],
        ),
    }

    print("================================================================================")
    print("CIR PAIRED STATISTICAL COMPARISONS (Prompt 14i Step 1)")
    print("================================================================================")

    paired_results: Dict[str, Any] = {}

    for sub_name, (sub_01, sub_05, sub_b) in subsets.items():
        n = len(sub_01)
        print(f"\n--- Subset: {sub_name} (n={n}) ---")
        
        bundle_01_vs_b = compute_paired_bundle(sub_01, sub_b, "cir_v1_01", "cir_i2i_baseline")
        bundle_05_vs_b = compute_paired_bundle(sub_05, sub_b, "cir_v1_05", "cir_i2i_baseline")
        bundle_05_vs_01 = compute_paired_bundle(sub_05, sub_01, "cir_v1_05", "cir_v1_01")

        paired_results[sub_name] = {
            "n_pairs": n,
            "cir_v1_01_vs_baseline": bundle_01_vs_b,
            "cir_v1_05_vs_baseline": bundle_05_vs_b,
            "cir_v1_05_vs_cir_v1_01": bundle_05_vs_01,
        }

        for comp_name, b_data in [
            ("cir_v1_01 vs baseline", bundle_01_vs_b),
            ("cir_v1_05 vs baseline", bundle_05_vs_b),
            ("cir_v1_05 vs cir_v1_01", bundle_05_vs_01),
        ]:
            print(f"  [{comp_name}]")
            for m in ["recall@1", "recall@5", "recall@10", "mAP"]:
                md = b_data[m]
                print(f"    {m:10s}: diff = {md['mean_diff']:+.4f} [{md['ci_low']:+.4f}, {md['ci_high']:+.4f}] -> {md['label']}")

    # Save to a dedicated json file
    out_path = PROJECT_ROOT / "reports/phase5c/cir_paired_baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(paired_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved CIR paired baseline results to {out_path.name}")
    print(f"SHA-256: {sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
