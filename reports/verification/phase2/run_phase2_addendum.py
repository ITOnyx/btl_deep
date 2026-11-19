"""Phase 2 Addendum Verification Script:
A. Quantify leakage impact (test-train, test-val, val-train, test-test internal) by SHA-1 and pHash (Hamming=0 and Hamming=1),
   attribute comparison, and count of affected held-out v2 queries (GT hybrid) without scoring.
   Exports reports/verification/phase2/leak_pairs.csv.
B. Trace 1500 -> 1372 -> 1290 chain, 22 skipped styles.csv rows vs 22 extra disk images, empty sha1/phash columns, and 4 WIP quarantine_log rows.
"""
import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "reports" / "verification" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_ALT = PROJECT_ROOT / "reports" / "verification" / "phase_2"
OUT_DIR_ALT.mkdir(parents=True, exist_ok=True)

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def file_sha1(path: pathlib.Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def find_phash_pairs(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    max_hamming: int = 1,
    same_set: bool = False,
) -> list[dict]:
    """Find all pairs (a, b) with pHash Hamming distance <= max_hamming."""
    a_pids = df_a["pid_str"].to_numpy()
    b_pids = df_b["pid_str"].to_numpy()
    a_ph_str = df_a["eff_phash"].to_numpy()
    b_ph_str = df_b["eff_phash"].to_numpy()
    a_ph_u64 = np.array([int(x, 16) for x in a_ph_str], dtype=np.uint64)
    b_ph_u64 = np.array([int(x, 16) for x in b_ph_str], dtype=np.uint64)

    a_art = df_a["articleType"].to_numpy()
    b_art = df_b["articleType"].to_numpy()
    a_gen = df_a["gender"].to_numpy()
    b_gen = df_b["gender"].to_numpy()
    a_col = df_a["baseColour"].to_numpy()
    b_col = df_b["baseColour"].to_numpy()
    a_paths = df_a["image_path"].to_numpy()
    b_paths = df_b["image_path"].to_numpy()
    a_split = df_a["split"].to_numpy()
    b_split = df_b["split"].to_numpy()

    sha1_cache: dict[str, str] = {}

    def get_sha1(pid: str, rel_path: str) -> str:
        if pid not in sha1_cache:
            sha1_cache[pid] = file_sha1(PROJECT_ROOT / str(rel_path))
        return sha1_cache[pid]

    pairs = []
    for i in range(len(df_a)):
        xor_vals = np.bitwise_xor(a_ph_u64[i], b_ph_u64)
        bytes_view = xor_vals.view(np.uint8).reshape(-1, 8)
        dists = _POPCOUNT_LUT[bytes_view].sum(axis=1, dtype=np.uint8)
        idxs = np.where(dists <= max_hamming)[0]
        for j in idxs:
            if same_set and j <= i:
                continue
            h_dist = int(dists[j])
            sha_a = get_sha1(str(a_pids[i]), str(a_paths[i]))
            sha_b = get_sha1(str(b_pids[j]), str(b_paths[j]))
            is_sha1_eq = sha_a == sha_b
            if is_sha1_eq:
                mtype = "sha1_exact"
            elif h_dist == 0:
                mtype = "phash_hamming_0"
            else:
                mtype = f"phash_hamming_{h_dist}"

            same_art = str(a_art[i]) == str(b_art[j])
            same_gen = str(a_gen[i]) == str(b_gen[j])
            same_col = str(a_col[i]) == str(b_col[j])

            pairs.append({
                "id_a": str(a_pids[i]),
                "id_b": str(b_pids[j]),
                "match_type": mtype,
                "sha1_equal": is_sha1_eq,
                "hamming_dist": h_dist,
                "phash_a": str(a_ph_str[i]),
                "phash_b": str(b_ph_str[j]),
                "sha1_a": sha_a,
                "sha1_b": sha_b,
                "articleType_a": str(a_art[i]),
                "articleType_b": str(b_art[j]),
                "same_articleType": same_art,
                "gender_a": str(a_gen[i]),
                "gender_b": str(b_gen[j]),
                "same_gender": same_gen,
                "baseColour_a": str(a_col[i]),
                "baseColour_b": str(b_col[j]),
                "same_baseColour": same_col,
                "same_all_3_attrs": bool(same_art and same_gen and same_col),
                "split_a": str(a_split[i]),
                "split_b": str(b_split[j]),
            })
    return pairs


def summarize_pairs(pairs: list[dict]) -> dict:
    p_sha1 = [p for p in pairs if p["sha1_equal"]]
    p_h0 = [p for p in pairs if p["hamming_dist"] == 0]
    p_h1 = [p for p in pairs if p["hamming_dist"] == 1]

    def attr_breakdown(sub: list[dict]) -> dict:
        return {
            "total_pairs": len(sub),
            "distinct_id_a": len({p["id_a"] for p in sub}),
            "distinct_id_b": len({p["id_b"] for p in sub}),
            "distinct_phash_a": len({p["phash_a"] for p in sub}),
            "same_all_3_attrs": sum(1 for p in sub if p["same_all_3_attrs"]),
            "diff_at_least_1_attr": sum(1 for p in sub if not p["same_all_3_attrs"]),
            "same_articleType_count": sum(1 for p in sub if p["same_articleType"]),
            "diff_articleType_count": sum(1 for p in sub if not p["same_articleType"]),
            "same_gender_count": sum(1 for p in sub if p["same_gender"]),
            "diff_gender_count": sum(1 for p in sub if not p["same_gender"]),
            "same_baseColour_count": sum(1 for p in sub if p["same_baseColour"]),
            "diff_baseColour_count": sum(1 for p in sub if not p["same_baseColour"]),
            "diff_only_baseColour": sum(
                1 for p in sub if p["same_articleType"] and p["same_gender"] and not p["same_baseColour"]
            ),
            "diff_only_articleType": sum(
                1 for p in sub if not p["same_articleType"] and p["same_gender"] and p["same_baseColour"]
            ),
            "diff_only_gender": sum(
                1 for p in sub if p["same_articleType"] and not p["same_gender"] and p["same_baseColour"]
            ),
        }

    return {
        "sha1_exact": attr_breakdown(p_sha1),
        "phash_hamming_0_incl_sha1": attr_breakdown(p_h0),
        "phash_hamming_1_only": attr_breakdown(p_h1),
        "phash_hamming_le1_total": attr_breakdown(pairs),
    }


def main() -> None:
    df = pd.read_parquet(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    df["pid_str"] = df["product_id"].astype(str)
    qm = pd.read_parquet(PROJECT_ROOT / "reports/phase1/phase1a/quality_metrics.parquet")
    qm_map = dict(zip(qm["id"].astype(str), qm["phash"].fillna("")))
    df["eff_phash"] = [qm_map.get(p, "") for p in df["pid_str"]]

    val_ids = {str(x) for x in json.loads((PROJECT_ROOT / "eval/val_split.json").read_text(encoding="utf-8"))}
    test_ids = {str(x) for x in json.loads((PROJECT_ROOT / "eval/test_split.json").read_text(encoding="utf-8"))}

    clean = df[(df["filter_flag"] == "pass") & (df["in_scope"] == True)].copy()
    test_clean = clean[clean["split"] == "test"].copy()
    train_all_clean = clean[clean["split"] == "train"].copy()
    val_clean = clean[clean["pid_str"].isin(val_ids)].copy()
    train_nonval_clean = train_all_clean[~train_all_clean["pid_str"].isin(val_ids)].copy()

    # 1. Compute pairs across splits
    pairs_test_vs_train_all = find_phash_pairs(test_clean, train_all_clean, max_hamming=1, same_set=False)
    pairs_test_vs_val = find_phash_pairs(test_clean, val_clean, max_hamming=1, same_set=False)
    pairs_val_vs_train_nonval = find_phash_pairs(val_clean, train_nonval_clean, max_hamming=1, same_set=False)
    pairs_test_internal = find_phash_pairs(test_clean, test_clean, max_hamming=1, same_set=True)

    # Export leak_pairs.csv for test vs train_all (with train_in_val column)
    csv_path = OUT_DIR / "leak_pairs.csv"
    fieldnames = [
        "test_id",
        "train_id",
        "match_type",
        "hamming_dist",
        "sha1_equal",
        "test_articleType",
        "train_articleType",
        "same_articleType",
        "test_gender",
        "train_gender",
        "same_gender",
        "test_baseColour",
        "train_baseColour",
        "same_baseColour",
        "same_all_3_attrs",
        "test_split",
        "train_split",
        "train_in_val_split",
        "test_phash",
        "train_phash",
        "test_sha1",
        "train_sha1",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pairs_test_vs_train_all:
            writer.writerow({
                "test_id": p["id_a"],
                "train_id": p["id_b"],
                "match_type": p["match_type"],
                "hamming_dist": p["hamming_dist"],
                "sha1_equal": p["sha1_equal"],
                "test_articleType": p["articleType_a"],
                "train_articleType": p["articleType_b"],
                "same_articleType": p["same_articleType"],
                "test_gender": p["gender_a"],
                "train_gender": p["gender_b"],
                "same_gender": p["same_gender"],
                "test_baseColour": p["baseColour_a"],
                "train_baseColour": p["baseColour_b"],
                "same_baseColour": p["same_baseColour"],
                "same_all_3_attrs": p["same_all_3_attrs"],
                "test_split": p["split_a"],
                "train_split": p["split_b"],
                "train_in_val_split": p["id_b"] in val_ids,
                "test_phash": p["phash_a"],
                "train_phash": p["phash_b"],
                "test_sha1": p["sha1_a"],
                "train_sha1": p["sha1_b"],
            })
    # Copy to phase_2 as well
    (OUT_DIR_ALT / "leak_pairs.csv").write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    # 3. Count affected queries in held-out v2 (NO scoring, ONLY GT set intersection)
    heldout_v2 = json.loads((PROJECT_ROOT / "eval/test_queries_heldout_v2.json").read_text(encoding="utf-8"))
    queries = heldout_v2["queries"] if isinstance(heldout_v2, dict) and "queries" in heldout_v2 else heldout_v2

    affected_test_sha1 = {p["id_a"] for p in pairs_test_vs_train_all if p["sha1_equal"]}
    affected_test_h0 = {p["id_a"] for p in pairs_test_vs_train_all if p["hamming_dist"] == 0}
    affected_test_hle1 = {p["id_a"] for p in pairs_test_vs_train_all if p["hamming_dist"] <= 1}

    # Check both precomputed relevant_ids in heldout_v2 AND hybrid policy via is_item_relevant
    from src.eval.metrics import load_gender_gt_policy, is_item_relevant
    policy = load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")
    strict_cats = policy.get("strict_gender_categories", [])
    test_id_map = json.loads((PROJECT_ROOT / "index/test_gallery/id_map.json").read_text(encoding="utf-8"))

    q_hit_sha1 = 0
    q_hit_h0 = 0
    q_hit_hle1 = 0
    q_hit_sha1_ids = []
    q_hit_h0_ids = []

    for q in queries:
        qid = q.get("query_id", q.get("id", ""))
        if "relevant_ids" in q and q["relevant_ids"]:
            gt_ids = {str(x) for x in q["relevant_ids"]}
        else:
            spec = q.get("ground_truth", q.get("spec", q))
            gt_ids = {
                str(item["item_id"])
                for item in test_id_map
                if is_item_relevant(spec, item, gender_policy="hybrid", strict_categories=strict_cats)
            }
        if gt_ids & affected_test_sha1:
            q_hit_sha1 += 1
            q_hit_sha1_ids.append(qid)
        if gt_ids & affected_test_h0:
            q_hit_h0 += 1
            q_hit_h0_ids.append(qid)
        if gt_ids & affected_test_hle1:
            q_hit_hle1 += 1

    # B.7: Check the 22 skipped lines in styles.csv vs 22 extra images on disk
    styles_path = PROJECT_ROOT / "data/fashion-dataset/styles.csv"
    bad_line_ids = []
    all_csv_ids = []
    with open(styles_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split(",")
        n_cols = len(header)
        for line_no, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split(",")
            pid = parts[0].strip()
            all_csv_ids.append(pid)
            if len(parts) != n_cols:
                bad_line_ids.append({"line_no": line_no, "id": pid, "n_fields": len(parts), "raw_preview": line.strip()[:100]})

    cat_ids = set(df["pid_str"])
    src_imgs = {p.stem for p in (PROJECT_ROOT / "data/fashion-dataset/images").glob("*.jpg")}
    q_imgs = {
        p.stem
        for p in (PROJECT_ROOT / "data/_quarantine").rglob("*.jpg")
        if p.parent.name in ("blurry", "duplicate", "banner", "low_res")
    }
    extra_disk_img_ids = sorted((src_imgs | q_imgs) - cat_ids)
    bad_line_id_set = {x["id"] for x in bad_line_ids}

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "A1_A2_test_vs_train_all": summarize_pairs(pairs_test_vs_train_all),
        "A3_heldout_v2_gt_impact_counts_only": {
            "total_queries": len(queries),
            "distinct_affected_test_pids_sha1": sorted(affected_test_sha1),
            "distinct_affected_test_pids_phash_h0_count": len(affected_test_h0),
            "distinct_affected_test_pids_phash_hle1_count": len(affected_test_hle1),
            "queries_with_ge1_sha1_affected_in_gt": q_hit_sha1,
            "queries_with_ge1_sha1_query_ids": q_hit_sha1_ids,
            "queries_with_ge1_phash_h0_affected_in_gt": q_hit_h0,
            "queries_with_ge1_phash_hle1_affected_in_gt": q_hit_hle1,
        },
        "A4_other_split_pairs": {
            "test_vs_val_clean": summarize_pairs(pairs_test_vs_val),
            "val_vs_train_nonval_clean": summarize_pairs(pairs_val_vs_train_nonval),
            "test_gallery_internal_duplicates": summarize_pairs(pairs_test_internal),
            "note_val_subset_of_train": "val_clean (1,884) is a 100% subset of train_all_clean (37,354); above val_vs_train_nonval compares val_clean (1,884) against train_all_clean \\ val_clean (35,470).",
        },
        "B6_test_split_chain": {
            "step1_checkpoint_unseen_in_catalog_rebuild": int((df["split"] == "test").sum()),
            "step1_filter_failed_in_test": df[df["split"] == "test"]["filter_flag"].value_counts().to_dict(),
            "step2_pass_written_to_test_split_json": len(test_ids),
            "step2_out_of_scope_in_test_pass": int(((df["split"] == "test") & (df["filter_flag"] == "pass") & (~df["in_scope"])).sum()),
            "step3_in_scope_built_into_test_gallery": len(test_clean),
        },
        "B7_styles_csv_22_diff": {
            "styles_csv_total_data_lines": len(all_csv_ids),
            "catalog_rebuild_rows": len(df),
            "bad_lines_skipped_by_pandas_count": len(bad_line_ids),
            "extra_images_on_disk_not_in_catalog_count": len(extra_disk_img_ids),
            "exact_same_22_ids": sorted(bad_line_id_set) == extra_disk_img_ids,
            "bad_line_ids": sorted(bad_line_id_set),
            "bad_lines_sample": bad_line_ids[:5],
        },
        "B9_wip_quarantine_4_rows": {
            "found_in_data_crawl_rejected": [
                str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                for p in sorted((PROJECT_ROOT / "data/crawl/_rejected").rglob("*.jpg"))
            ]
            if (PROJECT_ROOT / "data/crawl/_rejected").exists()
            else [],
        },
    }

    (OUT_DIR / "phase2_addendum_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR_ALT / "phase2_addendum_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
