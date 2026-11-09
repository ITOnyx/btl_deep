"""
Analyze Blind A/B Evaluation Results (Phase 3g Rev3):
- Decodes sealed ab_mapping.json.
- Computes Win / Loss / Tie for System (Config 5) vs Baseline (Config 0).
- Performs two-sided Sign Test and 10,000 bootstrap resamples for 95% CI.
- Breaks down by query groups: Overall (30q), Vague (4q: x025-x028), Brand (8q: x011-x018), Standard Spec (18q).
- Compares Human Ratings with Automated Metadata Metrics.
- Identifies any queries where System C5 got 0 correct items.
- Outputs reports/phase3g_blind_ab_results.json.
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

def main():
    print("=====================================================================")
    print("DECODING BLIND A/B EVALUATION (Phase 3g Rev3)...")
    print("=====================================================================")

    # 1. Load human ratings
    df_blind = pd.read_csv("eval/ab_blind_heldout.csv")
    print(f"Loaded {len(df_blind)} rows from eval/ab_blind_heldout.csv.")

    # 2. Load sealed mapping
    with open("eval/_sealed/ab_mapping.json", "r", encoding="utf-8") as f:
        ab_mapping = json.load(f)

    # 3. Load metadata inspection
    with open("reports/phase3g_blind_metadata_inspection.json", "r", encoding="utf-8") as f:
        meta_insp = json.load(f)["queries_inspection"]
    meta_dict = {item["query_id"]: item for item in meta_insp}

    decoded_rows = []
    
    for _, row in df_blind.iterrows():
        qid = str(row["id"])
        qt = str(row["query"])
        better = str(row["better"]).strip().upper()
        cnt_a = int(row["correct_count_A"]) if pd.notnull(row["correct_count_A"]) else 0
        cnt_b = int(row["correct_count_B"]) if pd.notnull(row["correct_count_B"]) else 0
        notes = str(row["notes"]) if pd.notnull(row["notes"]) else ""

        map_entry = ab_mapping[qid]
        a_model = map_entry["A"]
        b_model = map_entry["B"]

        # Identify which row is C5 (System) and which is C0 (Model Thuan)
        if "System" in a_model or "Config 5" in a_model:
            c5_row = "A"
            c0_row = "B"
            c5_cnt = cnt_a
            c0_cnt = cnt_b
        else:
            c5_row = "B"
            c0_row = "A"
            c5_cnt = cnt_b
            c0_cnt = cnt_a

        # Determine winner
        if better == c5_row:
            winner = "Config 5 (System)"
            outcome = "WIN" # System wins
        elif better == c0_row:
            winner = "Config 0 (Model Thuan)"
            outcome = "LOSS" # System loses
        elif better in ["TIE", "EQUAL", "="]:
            winner = "TIE"
            outcome = "TIE"
        else:
            # Infer from counts if ambiguous
            if c5_cnt > c0_cnt:
                winner = "Config 5 (System)"
                outcome = "WIN"
            elif c0_cnt > c5_cnt:
                winner = "Config 0 (Model Thuan)"
                outcome = "LOSS"
            else:
                winner = "TIE"
                outcome = "TIE"

        # Metadata comparison
        meta_item = meta_dict.get(qid, {})
        has_spec = meta_item.get("has_spec", False)
        # Note: in meta_inspection: Row_1 = Row A, Row_2 = Row B
        if c5_row == "A":
            meta_c5_hits = meta_item.get("Row_1_correct_count")
            meta_c0_hits = meta_item.get("Row_2_correct_count")
        else:
            meta_c5_hits = meta_item.get("Row_2_correct_count")
            meta_c0_hits = meta_item.get("Row_1_correct_count")

        decoded_rows.append({
            "qid": qid,
            "query": qt,
            "user_better": better,
            "row_A_model": a_model,
            "row_B_model": b_model,
            "c5_row": c5_row,
            "c0_row": c0_row,
            "c5_correct": c5_cnt,
            "c0_correct": c0_cnt,
            "outcome": outcome,
            "winner": winner,
            "notes": notes,
            "has_spec": has_spec,
            "meta_c5_hits": meta_c5_hits,
            "meta_c0_hits": meta_c0_hits,
            "diff_c5_human_vs_meta": (c5_cnt - meta_c5_hits) if (has_spec and meta_c5_hits is not None) else None,
            "diff_c0_human_vs_meta": (c0_cnt - meta_c0_hits) if (has_spec and meta_c0_hits is not None) else None
        })

    df_res = pd.DataFrame(decoded_rows)

    # 4. Statistical Analysis
    def run_stats(sub_df, name):
        n = len(sub_df)
        wins = sum(sub_df["outcome"] == "WIN")
        losses = sum(sub_df["outcome"] == "LOSS")
        ties = sum(sub_df["outcome"] == "TIE")
        
        # Sign test excluding ties
        n_effective = wins + losses
        if n_effective > 0:
            # Two-sided binomtest (H0: p = 0.5)
            binom_res = stats.binomtest(wins, n_effective, p=0.5, alternative="two-sided")
            p_val = binom_res.pvalue
        else:
            p_val = 1.0

        # Mean human precision P@10
        c5_p10_mean = sub_df["c5_correct"].mean() / 10.0
        c0_p10_mean = sub_df["c0_correct"].mean() / 10.0
        delta_p10_mean = c5_p10_mean - c0_p10_mean

        # Bootstrap 95% CI for delta P@10 (10,000 resamples)
        rng = np.random.RandomState(42)
        boot_deltas = []
        c5_arr = sub_df["c5_correct"].to_numpy() / 10.0
        c0_arr = sub_df["c0_correct"].to_numpy() / 10.0
        diff_arr = c5_arr - c0_arr

        for _ in range(10000):
            idx = rng.randint(0, n, size=n)
            boot_deltas.append(np.mean(diff_arr[idx]))
        
        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))

        # Bootstrap Win Rate CI
        boot_winrates = []
        for _ in range(10000):
            idx = rng.randint(0, n, size=n)
            sample_wins = sum(sub_df["outcome"].iloc[idx] == "WIN")
            boot_winrates.append(sample_wins / n)
        winrate_ci_lower = float(np.percentile(boot_winrates, 2.5))
        winrate_ci_upper = float(np.percentile(boot_winrates, 97.5))

        return {
            "group_name": name,
            "total_queries": n,
            "c5_wins": int(wins),
            "c0_wins": int(losses),
            "ties": int(ties),
            "win_rate": round(wins / n, 4),
            "win_rate_ci95": [round(winrate_ci_lower, 4), round(winrate_ci_upper, 4)],
            "sign_test_two_sided_p_value": float(p_val),
            "is_statistically_significant_p05": bool(p_val < 0.05),
            "human_p10_mean_c5": round(c5_p10_mean, 4),
            "human_p10_mean_c0": round(c0_p10_mean, 4),
            "delta_p10_mean": round(delta_p10_mean, 4),
            "delta_p10_ci95": [round(ci_lower, 4), round(ci_upper, 4)]
        }

    # Groups:
    # 1. Overall 30 queries
    stat_overall = run_stats(df_res, "Overall (All 30 Queries)")

    # 2. Vague queries (x025 - x028)
    vague_ids = ["x025", "x026", "x027", "x028"]
    stat_vague = run_stats(df_res[df_res["qid"].isin(vague_ids)], "Vague / Style Queries (x025-x028, N=4)")

    # 3. Brand queries (x011 - x018)
    brand_ids = [f"x{i:03d}" for i in range(11, 19)]
    stat_brand = run_stats(df_res[df_res["qid"].isin(brand_ids)], "Brand Queries (x011-x018, N=8)")

    # 4. Standard Attribute queries (26 spec excluding brand = 18 queries)
    standard_ids = [f"x{i:03d}" for i in range(1, 11)] + [f"x{i:03d}" for i in range(19, 25)] + ["x029", "x030"]
    stat_standard = run_stats(df_res[df_res["qid"].isin(standard_ids)], "Standard Attribute Queries (N=18)")

    # 5. All 26 Spec queries
    stat_spec26 = run_stats(df_res[~df_res["qid"].isin(vague_ids)], "Spec Ground Truth Queries (N=26)")

    # 5. Queries where System C5 got 0 correct items
    c5_zero_queries = df_res[df_res["c5_correct"] == 0].to_dict(orient="records")

    # 6. Discrepancies between Human Rating and Metadata
    discrepancies = []
    for r in decoded_rows:
        if r["has_spec"]:
            diff_c5 = r["diff_c5_human_vs_meta"]
            diff_c0 = r["diff_c0_human_vs_meta"]
            if abs(diff_c5) >= 2 or abs(diff_c0) >= 2:
                discrepancies.append({
                    "qid": r["qid"],
                    "query": r["query"],
                    "human_c5": r["c5_correct"],
                    "meta_c5": r["meta_c5_hits"],
                    "diff_c5": diff_c5,
                    "human_c0": r["c0_correct"],
                    "meta_c0": r["meta_c0_hits"],
                    "diff_c0": diff_c0,
                    "notes": r["notes"]
                })

    out_data = {
        "summary": "Full Decoded Blind A/B Evaluation Analysis",
        "statistics": {
            "overall_30q": stat_overall,
            "spec_26q": stat_spec26,
            "vague_4q": stat_vague,
            "brand_8q": stat_brand,
            "standard_18q": stat_standard
        },
        "system_c5_zero_correct_queries": c5_zero_queries,
        "discrepancies_human_vs_metadata": discrepancies,
        "decoded_details": decoded_rows
    }

    out_file = Path("reports/phase3g_blind_ab_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved decoded analysis to {out_file}.")
    print("\n================== KEY HEADLINE RESULTS ==================")
    print(f"Overall (N=30): C5 Wins = {stat_overall['c5_wins']}, C0 Wins = {stat_overall['c0_wins']}, Ties = {stat_overall['ties']}")
    print(f"Win Rate: {stat_overall['win_rate']*100:.1f}% [95% CI: {stat_overall['win_rate_ci95'][0]*100:.1f}% - {stat_overall['win_rate_ci95'][1]*100:.1f}%]")
    print(f"Sign Test p-value: {stat_overall['sign_test_two_sided_p_value']:.4e} (Significant: {stat_overall['is_statistically_significant_p05']})")
    print(f"Human P@10: C5 = {stat_overall['human_p10_mean_c5']:.4f} vs C0 = {stat_overall['human_p10_mean_c0']:.4f} (Delta = +{stat_overall['delta_p10_mean']:.4f})")
    print(f"Delta P@10 95% CI: [{stat_overall['delta_p10_ci95'][0]:.4f}, {stat_overall['delta_p10_ci95'][1]:.4f}]")
    print(f"Queries where C5 got 0 correct: {len(c5_zero_queries)}")

if __name__ == "__main__":
    main()
