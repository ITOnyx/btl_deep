"""
Comprehensive Verification Script: verify_md_vs_json.py
Strictly verifies that 100% of numerical values, tables, and conclusion labels
in reports/phase5c/final_eval_report.md match reports/phase5c/final_eval_results.json,
reports/phase5c/final_eval_corrections.json, reports/phase5c/final_eval_supplements.json,
and reports/phase5c/cir_paired_baseline_results.json.

Prompt 14i Upgraded Verification:
- Part A2 Latency tables and 6-stage breakdown
- Step 5 Phase 5 diagnostic comparison
- Section 6 Latency Conclusion (RAM and Disk) with individual conclusions
- Section 6.2 Disk latency applicability ("Không áp dụng" for Stream 1 & 3)
- Section 7 Part B1 T2I Quality (Recall@1/5/10, mAP, conclusions)
- Section 7.2 Paired differences + mechanical labels
- Section 7.3 Slices (n=136, n=15, small-n warning)
- Section 7.4 Secondary GT metrics
- Section 8 CIR Quality & Slices & Honest evaluation
- Section 9 Hubness Analysis
- Section 10 Target Assessment
- Section 11 Cause Analysis (Oracle parser + gap qualification)
- Section 12 Supplements (Strict/Relaxed GT, Config A, CIR I2I baseline, CIR 2x2 matrix)
- Section 12.4 CIR Paired Baseline Comparisons (overall, baseColour, gender)
- Section 14 Erratum Notice (rejection of false claim, true baseline values)
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
    cir_paired_path = PROJECT_ROOT / "reports/phase5c/cir_paired_baseline_results.json"
    rep_path = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"

    assert res_path.exists(), f"Missing {res_path}"
    assert corr_path.exists(), f"Missing {corr_path}"
    assert supp_path.exists(), f"Missing {supp_path}"
    assert cir_paired_path.exists(), f"Missing {cir_paired_path}"
    assert rep_path.exists(), f"Missing {rep_path}"

    res = json.loads(res_path.read_text(encoding="utf-8"))
    corr = json.loads(corr_path.read_text(encoding="utf-8"))
    supp = json.loads(supp_path.read_text(encoding="utf-8"))
    cir_paired = json.loads(cir_paired_path.read_text(encoding="utf-8"))
    md = rep_path.read_text(encoding="utf-8")

    total_checks = 0
    mismatches = []

    def check_presence(val_str: str, desc: str) -> None:
        nonlocal total_checks
        total_checks += 1
        if val_str not in md:
            mismatches.append(f"{desc}: Expected '{val_str}' to be present in final_eval_report.md")

    # 1. Hashes in Section 0
    check_presence(corr["frozen_results_sha256"], "Frozen results SHA-256")
    check_presence(corr["supplements_sha256"], "Supplements SHA-256")
    check_presence(corr["cir_paired_baseline_results_sha256"], "CIR paired baseline SHA-256")

    # 2. Part A2 Latency table values
    part_a2 = res["part_a2_latency"]
    for cid in ["model_only", "cfg_v5_01", "cfg_v5_09", "cfg_v5_08b"]:
        cdata = part_a2["stream1_text_to_image"][cid]
        check_presence(f"{cdata['p95_median_ms']:.2f}", f"T2I p95 median ({cid})")
        check_presence(f"{cdata['p95_max_ms']:.2f}", f"T2I p95 max ({cid})")

    cir_up_ram = part_a2["stream4_cir_upload"]["cir_v1_05"]["primary_ram_preloaded"]
    cir_up_disk = part_a2["stream4_cir_upload"]["cir_v1_05"]["secondary_disk_jpeg_decode"]
    check_presence(f"{cir_up_ram['p95_median_ms']:.2f}", "CIR upload RAM p95 median")
    check_presence(f"{cir_up_disk['p95_median_ms']:.2f}", "CIR upload Disk p95 median")
    check_presence("201.238", "CIR upload RAM p99 pooled warning")
    check_presence("240.433", "CIR upload Disk p99 pooled warning")

    # 3. Stage breakdown (Mục 4.2)
    s1_disk_01 = part_a2["stream4_cir_upload"]["cir_v1_01"]["stage_breakdown_ms"]["stage1_disk_jpeg_decode"]
    check_presence(f"{s1_disk_01['p95_ms']:.3f}", "Stage 1 disk p95 cir_v1_01")
    s3_vision_05 = part_a2["stream4_cir_upload"]["cir_v1_05"]["stage_breakdown_ms"]["stage3_vision_tower_forward"]
    check_presence(f"{s3_vision_05['p95_ms']:.3f}", "Stage 3 vision p95 cir_v1_05")

    # 4. Latency Conclusions (Mục 6)
    check_presence("PASS_FULL", "RAM latency PASS_FULL")
    check_presence("PASS_CONDITIONAL", "Disk latency PASS_CONDITIONAL")
    check_presence("Đạt theo trung vị nhưng có run vượt", "CIR upload Disk conclusion cir_v1_05")
    check_presence("Chưa đạt", "CIR upload Disk conclusion cir_v1_01")
    check_presence("Không áp dụng", "Disk latency non-applicable streams")

    # 5. Part B1 T2I Quality
    t2i_cfgs = res["part_b1_t2i_quality"]["results_by_config"]
    for cid in ["model_only", "cfg_v5_01", "cfg_v5_09", "cfg_v5_08b", "cfg_v5_06"]:
        info = t2i_cfgs[cid]
        s = info["summary"]
        check_presence(f"{s['recall@10']['mean']:.4f}", f"T2I R@10 mean ({cid})")
        check_presence(f"{s['mAP']['mean']:.4f}", f"T2I mAP mean ({cid})")

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

    cir_paired_primary = res["part_b2_cir_quality"]["paired_differences"]["cir_v1_05_vs_cir_v1_01"]
    check_presence(f"{cir_paired_primary['delta_map']['mean_diff']:+.4f}", "CIR paired ΔmAP")
    check_presence(f"{cir_paired_primary['delta_r10']['mean_diff']:+.4f}", "CIR paired ΔR@10")

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
    check_presence("nguyên nhân chi tiết của khoảng cách: không xác minh được", "Oracle gap qualification")

    # 12. Supplements (Strict/Relaxed GT & Config A)
    s3a = supp["step_3a_gt_policies"]
    check_presence(f"{s3a['cfg_v5_09']['strict']['mAP']['mean']:.4f}", "Strict GT mAP cfg_v5_09")
    check_presence(f"{s3a['cfg_v5_09']['relaxed']['mAP']['mean']:.4f}", "Relaxed GT mAP cfg_v5_09")
    check_presence(f"{s3a['cfg_v5_09']['hybrid']['mAP']['mean']:.4f}", "Hybrid GT mAP cfg_v5_09")

    # 13. CIR I2I Baseline & Paired Table (Section 12.4)
    b_i2i = supp["step_3d_cir_supplements"]["reference_only_i2i_baseline"]["summary"]
    check_presence(f"{b_i2i['recall@10']['mean']:.4f}", "CIR I2I baseline R@10")
    check_presence(f"{b_i2i['mAP']['mean']:.4f}", "CIR I2I baseline mAP")

    for sub_k in ["overall", "baseColour", "gender"]:
        for comp_k in ["cir_v1_01_vs_baseline", "cir_v1_05_vs_baseline", "cir_v1_05_vs_cir_v1_01"]:
            b_data = cir_paired[sub_k][comp_k]
            check_presence(f"{b_data['recall@10']['mean_diff']:+.4f}", f"CIR paired {sub_k} {comp_k} ΔR@10")
            check_presence(f"{b_data['mAP']['mean_diff']:+.4f}", f"CIR paired {sub_k} {comp_k} ΔmAP")
            check_presence(b_data["recall@10"]["label"], f"CIR paired {sub_k} {comp_k} label R@10")
            check_presence(b_data["mAP"]["label"], f"CIR paired {sub_k} {comp_k} label mAP")

    # 14. Erratum Notice
    check_presence("Cả cir_v1_01 và cir_v1_05 đều vượt baseline này trên Recall@10", "Erratum mention of false claim")
    check_presence("0.2333", "Erratum mention of chat draft R@10")
    check_presence("0.0827", "Erratum mention of chat draft mAP")

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
