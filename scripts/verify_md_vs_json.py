import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    res_path = root / "reports/phase5c/final_eval_results.json"
    rep_path = root / "reports/phase5c/final_eval_report.md"

    d = json.loads(res_path.read_text(encoding="utf-8"))
    md = rep_path.read_text(encoding="utf-8")

    tbl = d["part_a2_latency"]["comparison_table"]
    missing = []
    for r in tbl:
        p50_s = f"{r['prompt14f_p50_ms']:.3f}" if isinstance(r['prompt14f_p50_ms'], (int, float)) else str(r['prompt14f_p50_ms'])
        p95_s = f"{r['prompt14f_p95_median_ms']:.3f}" if isinstance(r['prompt14f_p95_median_ms'], (int, float)) else str(r['prompt14f_p95_median_ms'])
        p99_s = f"{r['prompt14f_p99_ms']:.3f}" if isinstance(r['prompt14f_p99_ms'], (int, float)) else str(r['prompt14f_p99_ms'])
        if p50_s not in md:
            missing.append(("p50", r["config_id"], p50_s))
        if p95_s not in md:
            missing.append(("p95", r["config_id"], p95_s))
        if p99_s not in md:
            missing.append(("p99", r["config_id"], p99_s))

    # check step 5
    s5_tbl = d["step_5_phase5_comparison"]["comparison_table"]
    for r in s5_tbl:
        p5_hist = f"{r['phase5_historical_p95_ms']:.3f}"
        p5_today = f"{r['eval_latency_today_p95_ms']:.3f}" if isinstance(r['eval_latency_today_p95_ms'], (int, float)) else str(r['eval_latency_today_p95_ms'])
        if p5_hist not in md:
            missing.append(("step5_hist", r["config_note"], p5_hist))
        if p5_today not in md:
            missing.append(("step5_today", r["config_note"], p5_today))

    print(f"Verified {len(tbl)} latency rows + {len(s5_tbl)} Step 5 rows.")
    if missing:
        print(f"FAILED: {len(missing)} missing items: {missing}")
    else:
        print("PASS: 100% numeric parity between final_eval_results.json and final_eval_report.md.")

if __name__ == "__main__":
    main()
