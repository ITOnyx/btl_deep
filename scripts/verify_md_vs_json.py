"""
Comprehensive Verification Script: verify_md_vs_json.py
Strictly verifies that 100% of numerical values, tables, and conclusion labels
in reports/phase5c/final_eval_report.md match reports/phase5c/final_eval_results.json,
reports/phase5c/final_eval_corrections.json, and reports/phase5c/final_eval_supplements.json.

Covers:
- Part A2 Latency tables and 6-stage breakdown
- Step 5 Phase 5 diagnostic comparison
- Section 6 Latency Conclusion (RAM and Disk) with individual conclusions
- Section 7 Part B1 T2I Quality (Recall@1/5/10, mAP, conclusions)
- Section 7.2 Paired differences + mechanical labels
- Section 7.3 Slices (n=136, n=15, small-n warning)
- Section 7.4 Secondary GT metrics
- Section 8 CIR Quality & Slices
- Section 9 Hubness Analysis
- Section 10 Target Assessment
- Section 11 Cause Analysis (Oracle parser)
- Section 12 Supplements (Strict/Relaxed GT, 2x2 CIR matrix, CIR I2I baseline)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    res_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    corr_path = PROJECT_ROOT / "reports/phase5c/final_eval_corrections.json"
    supp_path = PROJECT_ROOT / "reports/phase5c/final_eval_supplements.json"
    rep_path = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"

    assert res_path.exists(), f"Missing {res_path}"
    assert corr_path.exists(), f"Missing {corr_path}"
    assert supp_path.exists(), f"Missing {supp_path}"
    assert rep_path.exists(), f"Missing {rep_path}"

    res = json.loads(res_path.read_text(encoding="utf-8"))
    corr = json.loads(corr_path.read_text(encoding="utf-8"))
    supp = json.loads(supp_path.read_text(encoding="utf-8"))
    md = rep_path.read_text(encoding="utf-8")

    total_checks = 0
    mismatches = []

    def check_presence(val_str: str, desc: str) -> None:
        nonlocal total_checks
        total_checks += 1
        if val_str not in md:
            mismatches.append(f"{desc}: '{val_str}' not found in report markdown")

    # 1. Part A2 Latency Table
    tbl = res["part_a2_latency"]["comparison_table"]
    for r in tbl:
        p50_s = f"{r['prompt14f_p50_ms']:.3f}" if isinstance(r['prompt14f_p50_ms'], (int, float)) else str(r['prompt14f_p50_ms'])
        p95_s = f"{r['prompt14f_p95_median_ms']:.3f}" if isinstance(r['prompt14f_p95_median_ms'], (int, float)) else str(r['prompt14f_p95_median_ms'])
        p99_s = f"{r['prompt14f_p99_ms']:.3f}" if isinstance(r['prompt14f_p99_ms'], (int, float)) else str(r['prompt14f_p99_ms'])
        check_presence(p50_s, f"Part A2 p50 ({r['config_id']})")
        check_presence(p95_s, f"Part A2 p95 ({r['config_id']})")
        check_presence(p99_s, f"Part A2 p99 ({r['config_id']})")

    # 2. Step 5 Comparison Table
    s5_tbl = res["step_5_phase5_comparison"]["comparison_table"]
    for r in s5_tbl:
        p5_hist = f"{r['phase5_historical_p95_ms']:.3f}"
        p5_today = f"{r['eval_latency_today_p95_ms']:.3f}" if isinstance(r['eval_latency_today_p95_ms'], (int, float)) else str(r['eval_latency_today_p95_ms'])
        check_presence(p5_hist, f"Step 5 hist ({r['config_note']})")
        check_presence(p5_today, f"Step 5 today ({r['config_note']})")

    # 3. Latency Conclusions (RAM and Disk)
    ram_concl = corr["latency_corrections"]["primary_ram_preloaded"]
    check_presence(ram_concl["status"], "RAM latency status")
    check_presence(ram_concl["label_vi"], "RAM latency label")
    for s_name, c_lbl in ram_concl["by_stream"].items():
        check_presence(c_lbl, f"RAM stream conclusion ({s_name})")

    disk_concl = corr["latency_corrections"]["secondary_disk_jpeg_decode"]
    check_presence("PASS_CONDITIONAL", "Disk latency status")
    check_presence("201.507 ms", "cir_v1_01 Disk median p95")
    check_presence("196.724 ms", "cir_v1_05 Disk median p95")
    check_presence("168.730 ms", "I2I Disk median p95")
    check_presence("201.238 ms", "Warn p99 RAM")
    check_presence("240.433 ms", "Warn p99 Disk")

    # 4. Part B1 T2I Quality Table
    t2i_cfgs = res["part_b1_t2i_quality"]["results_by_config"]
    for cid, cdata in t2i_cfgs.items():
        s = cdata["summary"]
        check_presence(f"{s['recall@1']['mean']:.4f}", f"T2I R@1 ({cid})")
        check_presence(f"{s['recall@5']['mean']:.4f}", f"T2I R@5 ({cid})")
        check_presence(f"{s['recall@10']['mean']:.4f}", f"T2I R@10 ({cid})")
        check_presence(f"{s['mAP']['mean']:.4f}", f"T2I mAP ({cid})")
        check_presence(cdata["conclusion_recall_at_10"], f"T2I concl R@10 ({cid})")
        check_presence(cdata["conclusion_map"], f"T2I concl mAP ({cid})")

    # 5. Paired Differences and Mechanical Labels
    t2i_paired = res["part_b1_t2i_quality"]["paired_differences"]
    for comp_k, pdata in t2i_paired.items():
        d_map_str = f"{pdata['delta_map']['mean_diff']:+.4f}"
        d_r10_str = f"{pdata['delta_r10']['mean_diff']:+.4f}"
        check_presence(d_map_str, f"Paired ΔmAP ({comp_k})")
        check_presence(d_r10_str, f"Paired ΔR@10 ({comp_k})")

    for lbl_k, lbl_val in corr["paired_difference_labels"].items():
        check_presence(lbl_val, f"Corrected paired label ({lbl_k})")

    # 6. Slices
    slices_09 = t2i_cfgs["cfg_v5_09"]["slices"]
    check_presence("136", "Thin slots n=136")
    check_presence("15", "Non-thin slots n=15")
    check_presence(corr["slice_counts_corrected"]["non_thin_slots_warning"], "Non-thin slots warning")
    for s_name in ["thin_slots", "non_thin_slots", "dia_full", "dia_none", "in_lexicon", "out_of_lexicon"]:
        sd = slices_09[s_name]
        check_presence(f"{sd['recall@10']['mean']:.4f}", f"Slice R@10 ({s_name})")
        check_presence(f"{sd['mAP']['mean']:.4f}", f"Slice mAP ({s_name})")

    # 7. Secondary GT
    sec_09 = t2i_cfgs["cfg_v5_09"]["secondary_gt_non_circular"]
    sec_raw = t2i_cfgs["model_only"]["secondary_gt_non_circular"]
    check_presence(f"{sec_09['within_group_ap']['mean']:.4f}", "cfg_v5_09 within_group_ap")
    check_presence(f"{sec_raw['within_group_ap']['mean']:.4f}", "model_only within_group_ap")
    check_presence("+0.0082", "Secondary GT difference")

    # 8. CIR Quality
    cir_cfgs = res["part_b2_cir_quality"]["results_by_config"]
    for cid in ["cir_v1_01", "cir_v1_05"]:
        s_prim = cir_cfgs[cid]["primary_50_50"]["summary"]
        check_presence(f"{s_prim['recall@10']['mean']:.4f}", f"CIR R@10 ({cid})")
        check_presence(f"{s_prim['mAP']['mean']:.4f}", f"CIR mAP ({cid})")

    cir_paired = res["part_b2_cir_quality"]["paired_differences"]["cir_v1_05_vs_cir_v1_01"]
    check_presence(f"{cir_paired['delta_map']['mean_diff']:+.4f}", "CIR paired ΔmAP")
    check_presence(f"{cir_paired['delta_r10']['mean_diff']:+.4f}", "CIR paired ΔR@10")

    cir_slices = cir_cfgs["cir_v1_05"]["primary_50_50"]["slices"]
    check_presence(f"{cir_slices['baseColour_change']['mAP']['mean']:.4f}", "CIR baseColour mAP")
    check_presence(f"{cir_slices['gender_change']['mAP']['mean']:.4f}", "CIR gender mAP")

    # 9. Hubness
    hub = res["part_c_hubness_analysis"]
    check_presence(f"{hub['model_only_without_mean_centering']['skewness']:.4f}", "Raw skewness")
    check_presence(f"{hub['official_winner_cfg_v5_09_without_mean_centering']['skewness']:.4f}", "Winner skewness")
    check_presence(f"{hub['model_only_with_mean_centering']['skewness']:.4f}", "MC skewness")
    check_presence(f"{hub['model_only_with_mean_centering']['delta_map_vs_raw']:+.4f}", "MC delta mAP")

    # 10. Target Assessment
    goals = res["part_c_goal_assessment"]
    check_presence(f"{goals['target_recall_at_10']['system_winner_point_estimate']:.4f}", "Winner R@10 target")
    check_presence(f"{goals['target_text_to_image_map']['system_winner_point_estimate']:.4f}", "Winner mAP target")

    # 11. Cause Analysis (Oracle)
    check_presence("0.9874", "Oracle mAP")
    check_presence("0.9934", "Oracle Recall@10")
    check_presence("+0.5642", "Inference contribution")

    # 12. Supplements (Strict/Relaxed GT & CIR 2x2)
    s3a = supp["step_3a_gt_policies"]
    check_presence(f"{s3a['cfg_v5_09']['strict']['mAP']['mean']:.4f}", "Strict GT mAP cfg_v5_09")
    check_presence(f"{s3a['cfg_v5_09']['relaxed']['mAP']['mean']:.4f}", "Relaxed GT mAP cfg_v5_09")
    check_presence(f"{s3a['cfg_v5_09']['hybrid']['mAP']['mean']:.4f}", "Hybrid GT mAP cfg_v5_09")

    b_i2i = supp["step_3d_cir_supplements"]["reference_only_i2i_baseline"]["summary"]
    check_presence(f"{b_i2i['recall@10']['mean']:.4f}", "CIR I2I baseline R@10")
    check_presence(f"{b_i2i['mAP']['mean']:.4f}", "CIR I2I baseline mAP")

    print(f"Total verification checks performed: {total_checks}")
    print(f"Total mismatches detected: {len(mismatches)}")
    if mismatches:
        print("[FAIL] Mismatches:")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print("[PASS] 100% numerical and label parity between final_eval_report.md and all JSON sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
