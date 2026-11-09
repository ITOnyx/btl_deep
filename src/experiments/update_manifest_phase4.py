import sys
import json
import hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def get_sha256(path_str):
    p = Path(path_str)
    if p.exists():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return None

def main():
    manifest_path = Path("reports/run_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    print("=== UPDATING RUN_MANIFEST.JSON ACCORDING TO TASK 6 ===")

    # 1. Fix maxlen_padding_status in first block (phase3g_errata)
    if "phase3g_errata" in d and "summary" in d["phase3g_errata"]:
        old_val = d["phase3g_errata"]["summary"].get("maxlen_padding_status")
        d["phase3g_errata"]["summary"]["maxlen_padding_status"] = "EXACT_3F_BASELINE_REPRODUCED_PADDING128_CONFIRMED"
        print(f"1. Updated phase3g_errata.summary.maxlen_padding_status: '{old_val}' -> '{d['phase3g_errata']['summary']['maxlen_padding_status']}'")

    # 2. Add note on phase3g_errata_rev3.evaluated_by ("đã đính chính ở Rev4")
    if "phase3g_errata_rev3" in d:
        if "blind_ab_evaluation" in d["phase3g_errata_rev3"]:
            d["phase3g_errata_rev3"]["blind_ab_evaluation"]["evaluated_by"] = "USER (human ground truth) [đã đính chính ở Rev4: do Claude AI single rater chấm, không phải User]"
            d["phase3g_errata_rev3"]["blind_ab_evaluation"]["note_correction"] = "đã đính chính ở Rev4"
            print(f"2. Updated phase3g_errata_rev3.blind_ab_evaluation.evaluated_by: {d['phase3g_errata_rev3']['blind_ab_evaluation']['evaluated_by']}")

    # 3. Remove degenerate win_rate_ci95 [1, 1] in rev4 (keep Clopper-Pearson)
    if "phase3g_errata_rev4" in d and "blind_ab_evaluation" in d["phase3g_errata_rev4"]:
        stats = d["phase3g_errata_rev4"]["blind_ab_evaluation"].get("statistics", {})
        for grp_key, grp_data in stats.items():
            if isinstance(grp_data, dict) and "win_rate_ci95" in grp_data:
                ci = grp_data["win_rate_ci95"]
                if ci == [1.0, 1.0] or ci == [1, 1]:
                    del grp_data["win_rate_ci95"]
                    print(f"3. Removed degenerate win_rate_ci95 [1, 1] from rev4 {grp_key}")

    # 4. Check sha256 of reports/phase3f_errata.md (Rev5) and update in phase3g_rev5
    errata_sha = get_sha256("reports/phase3f_errata.md")
    print(f"4. SHA256 of reports/phase3f_errata.md: {errata_sha}")

    if "phase3g_rev5" not in d:
        d["phase3g_rev5"] = {}

    d["phase3g_rev5"]["status"] = "RECONCILED_REV5_FINAL"
    d["phase3g_rev5"]["errata_rev5_sha256"] = errata_sha
    if "files" not in d["phase3g_rev5"]:
        d["phase3g_rev5"]["files"] = {}
    d["phase3g_rev5"]["files"]["reports/phase3f_errata.md"] = errata_sha

    # 5. Update Phase 4 block with all new files and smoke test metadata
    phase4_files = [
        "app/engine.py",
        "app/main.py",
        "src/search_pipeline.py",
        "src/test_app_service.py",
        "src/test_search_pipeline.py",
        "src/evaluate_cir_20queries.py",
        "src/experiment_trunc64_pad128_eval.py",
        "src/run_phase4_smoke_test.py",
        "reports/phase4_smoke.json",
        "reports/phase4_cir_20queries_eval.json",
        "reports/phase4_trunc64_pad128_eval.json",
        "README_run.md"
    ]

    p4_file_hashes = {}
    for f_rel in phase4_files:
        sha = get_sha256(f_rel)
        if sha:
            p4_file_hashes[f_rel] = sha

    if "phase4" not in d:
        d["phase4"] = {}

    d["phase4"]["status"] = "COMPLETED_REVISED"
    d["phase4"]["files"] = p4_file_hashes
    d["phase4"]["smoke_test"] = {
        "verified_file": "reports/phase4_smoke.json",
        "text_queries_tested": 5,
        "image_queries_tested": 3,
        "cir_queries_tested": 3,
        "status": "PASSED"
    }
    d["phase4"]["cir_evaluation"] = {
        "verified_file": "reports/phase4_cir_20queries_eval.json",
        "queries_count": 20,
        "hit_at_1": 0.15,
        "hit_at_10": 0.35,
        "precision_at_10": 0.16,
        "status": "EXPERIMENTAL_UNCOMMITTED_SLA"
    }
    d["phase4"]["trunc64_pad128_experiment"] = {
        "verified_file": "reports/phase4_trunc64_pad128_eval.json",
        "status": "EVALUATED_ON_Q2_Q3_Q5_HELDOUT",
        "conclusion": "Condition B and Condition A are identical on queries <= 64 tokens; pad 128 systematically boosts dense retrieval by matching train positional length"
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    print("Manifest successfully updated with all Task 6 items.")

if __name__ == "__main__":
    main()
