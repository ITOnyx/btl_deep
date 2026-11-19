"""Phase 2 Independent Verification Script: Data inventory, quarantine integrity, leakage (ID + SHA1 + pHash), reference_gallery diff, train caption status."""
import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "reports" / "verification" / "phase_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    res: dict = {
        "phase": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
    }

    # --- Item 7: Counts in metadata, disk images, filter_flag, in_scope, splits ---
    cat_path = PROJECT_ROOT / "data" / "catalog" / "catalog_rebuild.parquet"
    df = pd.read_parquet(cat_path)
    df["pid_str"] = df["product_id"].astype(str)
    cat_sha256 = sha256_file(cat_path)

    styles_path = PROJECT_ROOT / "data" / "fashion-dataset" / "styles.csv"
    with open(styles_path, "r", encoding="utf-8", errors="replace") as f:
        styles_lines = sum(1 for _ in f) - 1  # minus header

    qm_path = PROJECT_ROOT / "reports" / "phase1" / "phase1a" / "quality_metrics.parquet"
    df_qm = pd.read_parquet(qm_path) if qm_path.exists() else None

    fashion_img_dir = PROJECT_ROOT / "data" / "fashion-dataset" / "images"
    fashion_imgs_on_disk = list(fashion_img_dir.glob("*.jpg")) if fashion_img_dir.exists() else []

    quarantine_dir = PROJECT_ROOT / "data" / "_quarantine"
    quarantine_all_files = [p for p in quarantine_dir.rglob("*") if p.is_file()] if quarantine_dir.exists() else []
    quarantine_jpgs = [p for p in quarantine_all_files if p.suffix.lower() == ".jpg" and p.parent.name in ("blurry", "duplicate", "banner", "low_res")]

    filter_counts = df["filter_flag"].value_counts().to_dict()
    in_scope_total = int((df["in_scope"] == True).sum())
    clean_in_scope = df[(df["filter_flag"] == "pass") & (df["in_scope"] == True)]
    n_clean_in_scope = len(clean_in_scope)

    split_all_counts = df["split"].value_counts().to_dict()
    split_pass_counts = df[df["filter_flag"] == "pass"]["split"].value_counts().to_dict()
    split_clean_in_scope_counts = clean_in_scope["split"].value_counts().to_dict()

    val_split_path = PROJECT_ROOT / "eval" / "val_split.json"
    test_split_path = PROJECT_ROOT / "eval" / "test_split.json"
    val_json = json.loads(val_split_path.read_text(encoding="utf-8"))
    test_json = json.loads(test_split_path.read_text(encoding="utf-8"))

    val_ids_list = [str(x) for x in (val_json["product_ids"] if isinstance(val_json, dict) and "product_ids" in val_json else val_json)]
    test_ids_list = [str(x) for x in (test_json["product_ids"] if isinstance(test_json, dict) and "product_ids" in test_json else test_json)]
    val_ids_set = set(val_ids_list)
    test_ids_set = set(test_ids_list)

    df["pid_str"] = df["product_id"].astype(str)
    clean_in_scope_pids = set(clean_in_scope["product_id"].astype(str))

    val_in_clean_in_scope = len(val_ids_set & clean_in_scope_pids)
    test_in_clean_in_scope = len(test_ids_set & clean_in_scope_pids)

    # Also check Shopping100k & external_test counts mentioned in data_report.md
    shopping100k_dir = PROJECT_ROOT / "data" / "Shopping100k" / "Images"
    shopping100k_count = sum(1 for _ in shopping100k_dir.rglob("*.jpg")) if shopping100k_dir.exists() else None

    res["7_inventory_and_splits"] = {
        "catalog_rebuild_sha256": cat_sha256,
        "catalog_rows": len(df),
        "catalog_cols": len(df.columns),
        "styles_csv_rows": styles_lines,
        "quality_metrics_rows": len(df_qm) if df_qm is not None else None,
        "fashion_images_remaining_in_source_dir": len(fashion_imgs_on_disk),
        "quarantined_quality_jpgs": len(quarantine_jpgs),
        "total_actual_images_source_plus_quarantine": len(fashion_imgs_on_disk) + len(quarantine_jpgs),
        "filter_flag_counts": {k: int(v) for k, v in filter_counts.items()},
        "in_scope_true_total": in_scope_total,
        "in_scope_and_pass": n_clean_in_scope,
        "catalog_split_column_all": {k: int(v) for k, v in split_all_counts.items()},
        "catalog_split_column_pass": {k: int(v) for k, v in split_pass_counts.items()},
        "catalog_split_column_clean_in_scope": {k: int(v) for k, v in split_clean_in_scope_counts.items()},
        "val_split_json_total": len(val_ids_list),
        "val_split_json_clean_in_scope": val_in_clean_in_scope,
        "test_split_json_total": len(test_ids_list),
        "test_split_json_clean_in_scope": test_in_clean_in_scope,
        "shopping100k_jpg_count": shopping100k_count,
    }

    # --- Item 8: Zero leakage (product_id, sha1, pHash) & articleType coverage ---
    train_df_all = df[df["split"] == "train"]
    test_df_all = df[df["split"] == "test"]
    train_clean = clean_in_scope[clean_in_scope["split"] == "train"]
    test_clean = clean_in_scope[clean_in_scope["split"] == "test"]
    val_clean = clean_in_scope[clean_in_scope["pid_str"].isin(val_ids_set)]
    train_excl_val_clean = train_clean[~train_clean["pid_str"].isin(val_ids_set)]

    train_pids_all = set(train_df_all["pid_str"])
    test_pids_all = set(test_df_all["pid_str"])

    # Note: investigate how val_split.json relates to catalog_rebuild.parquet['split']
    val_split_col_dist = df[df["pid_str"].isin(val_ids_set)]["split"].value_counts().to_dict()

    # SHA1 and pHash checks between test and train (and test vs val)
    # Check both catalog_rebuild.parquet columns AND quality_metrics.parquet columns!
    cat_nonempty_sha1 = int((df["sha1"].fillna("") != "").sum())
    cat_nonempty_phash = int((df["phash"].fillna("") != "").sum())

    if df_qm is not None:
        df_qm["pid_str"] = df_qm["id"].astype(str)
        qm_sha1_map = dict(zip(df_qm["pid_str"], df_qm["sha1"].fillna(""))) if "sha1" in df_qm.columns else {}
        qm_phash_map = dict(zip(df_qm["pid_str"], df_qm["phash"].fillna("")))
    else:
        qm_sha1_map = {}
        qm_phash_map = {}

    df["eff_sha1"] = [s if s else qm_sha1_map.get(p, "") for p, s in zip(df["pid_str"], df["sha1"].fillna(""))]
    df["eff_phash"] = [s if s else qm_phash_map.get(p, "") for p, s in zip(df["pid_str"], df["phash"].fillna(""))]
    clean_in_scope = df[(df["filter_flag"] == "pass") & (df["in_scope"] == True)]
    train_clean = clean_in_scope[clean_in_scope["split"] == "train"]
    test_clean = clean_in_scope[clean_in_scope["split"] == "test"]
    val_clean = clean_in_scope[clean_in_scope["pid_str"].isin(val_ids_set)]

    train_sha1_clean = {s for s in train_clean["eff_sha1"] if s}
    test_sha1_clean = {s for s in test_clean["eff_sha1"] if s}
    val_sha1_clean = {s for s in val_clean["eff_sha1"] if s}

    train_phash_clean = {s for s in train_clean["eff_phash"] if s}
    test_phash_clean = {s for s in test_clean["eff_phash"] if s}
    val_phash_clean = {s for s in val_clean["eff_phash"] if s}

    # Also check ALL rows (even before in_scope filter, among pass images)
    train_pass = df[(df["split"] == "train") & (df["filter_flag"] == "pass")]
    test_pass = df[(df["split"] == "test") & (df["filter_flag"] == "pass")]
    sha1_overlap_pass_all = len({s for s in train_pass["eff_sha1"] if s} & {s for s in test_pass["eff_sha1"] if s})
    phash_exact_overlap_pass_all = len({s for s in train_pass["eff_phash"] if s} & {s for s in test_pass["eff_phash"] if s})

    # Compute Hamming distance <= 1 between test_clean pHashes and train_clean pHashes
    train_valid_p = train_clean[train_clean["eff_phash"] != ""]
    test_valid_p = test_clean[test_clean["eff_phash"] != ""]

    def hex_to_uint64(hex_series: pd.Series) -> np.ndarray:
        return np.array([int(x, 16) for x in hex_series], dtype=np.uint64)

    train_p_int = hex_to_uint64(train_valid_p["eff_phash"])
    test_p_int = hex_to_uint64(test_valid_p["eff_phash"])
    train_attrs = list(zip(train_valid_p["articleType"], train_valid_p["gender"], train_valid_p["baseColour"]))
    test_attrs = list(zip(test_valid_p["articleType"], test_valid_p["gender"], test_valid_p["baseColour"]))
    train_pids_arr = train_valid_p["pid_str"].to_numpy()
    test_pids_arr = test_valid_p["pid_str"].to_numpy()

    hamming_le1_any_attr_pairs = []
    hamming_le1_same_attr_pairs = []
    for i, tp in enumerate(test_p_int):
        xor = np.bitwise_xor(train_p_int, tp)
        # bit count for uint64
        # unpackbits on uint8 view
        bits = np.unpackbits(xor.view(np.uint8)).reshape(-1, 64).sum(axis=1)
        le1_idx = np.where(bits <= 1)[0]
        for j in le1_idx:
            pair_info = {
                "test_pid": str(test_pids_arr[i]),
                "train_pid": str(train_pids_arr[j]),
                "hamming": int(bits[j]),
                "test_attr": test_attrs[i],
                "train_attr": train_attrs[j],
            }
            hamming_le1_any_attr_pairs.append(pair_info)
            if test_attrs[i] == train_attrs[j]:
                hamming_le1_same_attr_pairs.append(pair_info)

    # ArticleType coverage
    total_article_types_clean = clean_in_scope["articleType"].nunique()
    test_article_types_clean = test_clean["articleType"].nunique()

    res["8_leakage_and_coverage"] = {
        "catalog_nonempty_sha1_rows": cat_nonempty_sha1,
        "catalog_nonempty_phash_rows": cat_nonempty_phash,
        "effective_nonempty_sha1_train_clean": len(train_valid_p),
        "effective_nonempty_phash_test_clean": len(test_valid_p),
        "pid_overlap_train_test_all": len(train_pids_all & test_pids_all),
        "pid_overlap_val_test": len(val_ids_set & test_ids_set),
        "pid_overlap_train_col_vs_val_json": len(train_pids_all & val_ids_set),
        "val_json_split_column_distribution_in_catalog": {k: int(v) for k, v in val_split_col_dist.items()},
        "train_clean_excluding_val_count": len(train_excl_val_clean),
        "sha1_overlap_test_vs_train_clean": len(test_sha1_clean & train_sha1_clean),
        "sha1_overlap_test_vs_val_clean": len(test_sha1_clean & val_sha1_clean),
        "sha1_overlap_test_vs_train_all_pass": sha1_overlap_pass_all,
        "phash_exact_overlap_test_vs_train_clean": len(test_phash_clean & train_phash_clean),
        "phash_exact_overlap_test_vs_val_clean": len(test_phash_clean & val_phash_clean),
        "phash_exact_overlap_test_vs_train_all_pass": phash_exact_overlap_pass_all,
        "phash_hamming_le1_test_vs_train_same_attr_count": len(hamming_le1_same_attr_pairs),
        "phash_hamming_le1_test_vs_train_any_attr_count": len(hamming_le1_any_attr_pairs),
        "phash_hamming_le1_any_attr_examples": hamming_le1_any_attr_pairs[:10],
        "article_type_clean_total": int(total_article_types_clean),
        "article_type_test_clean": int(test_article_types_clean),
    }

    # --- Item 9: Quarantine integrity ---
    qlog_path = PROJECT_ROOT / "reports" / "quarantine_log.csv"
    qlog_df = pd.read_csv(qlog_path)
    status_counts = qlog_df["status"].value_counts().to_dict() if "status" in qlog_df.columns else {}
    active_q = qlog_df[qlog_df["status"] == "quarantined"] if "status" in qlog_df.columns else qlog_df
    reason_counts = active_q["reason"].value_counts().to_dict()

    # Check every active quarantined row exists on disk
    missing_from_quarantine = []
    logged_quarantine_paths = set()
    for _, row in active_q.iterrows():
        qp = PROJECT_ROOT / str(row["quarantine_path"])
        logged_quarantine_paths.add(qp.resolve())
        if not qp.exists():
            missing_from_quarantine.append(str(row["quarantine_path"]))

    # Check every file on disk in data/_quarantine is in logged_quarantine_paths
    disk_quarantine_paths = {p.resolve() for p in quarantine_all_files}
    unlogged_files_in_quarantine = [
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for p in sorted(disk_quarantine_paths - logged_quarantine_paths)
    ]

    # Check every 'pass' image in catalog exists on disk
    missing_pass_images = []
    for p_rel in df[df["filter_flag"] == "pass"]["image_path"]:
        if not (PROJECT_ROOT / str(p_rel)).exists():
            missing_pass_images.append(str(p_rel))

    res["9_quarantine_integrity"] = {
        "quarantine_log_total_rows": len(qlog_df),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "active_quarantined_by_reason": {str(k): int(v) for k, v in reason_counts.items()},
        "disk_quarantine_total_files": len(disk_quarantine_paths),
        "active_logged_unique_paths": len(logged_quarantine_paths),
        "missing_from_quarantine_count": len(missing_from_quarantine),
        "unlogged_files_in_quarantine_count": len(unlogged_files_in_quarantine),
        "unlogged_files_in_quarantine_sample": unlogged_files_in_quarantine[:20],
        "pass_images_missing_on_disk_count": len(missing_pass_images),
    }

    # --- Item 10: reference_gallery 38,646 vs 38,644 (2 Key chain) & test_gallery unaffected ---
    ref_id_map = json.loads((PROJECT_ROOT / "index" / "reference_gallery" / "id_map.json").read_text(encoding="utf-8"))
    test_id_map = json.loads((PROJECT_ROOT / "index" / "test_gallery" / "id_map.json").read_text(encoding="utf-8"))
    ref_pids = {str(x["item_id"]) for x in ref_id_map}
    test_gallery_pids = {str(x["item_id"]) for x in test_id_map}
    test_clean_pids = set(test_clean["pid_str"])

    extra_in_ref = sorted(ref_pids - clean_in_scope_pids)
    missing_in_ref = sorted(clean_in_scope_pids - ref_pids)
    extra_rows_info = df[df["pid_str"].isin(extra_in_ref)][
        ["product_id", "articleType", "masterCategory", "filter_flag", "in_scope", "split"]
    ].to_dict(orient="records")

    res["10_reference_gallery_diff"] = {
        "reference_gallery_len": len(ref_id_map),
        "clean_in_scope_len": len(clean_in_scope_pids),
        "extra_in_reference_gallery_pids": extra_in_ref,
        "missing_in_reference_gallery_pids": missing_in_ref,
        "extra_rows_details": extra_rows_info,
        "test_gallery_len": len(test_id_map),
        "test_gallery_exact_match_test_clean": test_gallery_pids == test_clean_pids,
        "extra_pids_in_test_gallery": sorted(test_gallery_pids & set(extra_in_ref)),
    }

    # --- Item 11: Dense caption status for 37,354 train images ---
    caption_cols = [c for c in df.columns if "caption" in c.lower()]
    train_caption_non_null = {
        col: int(train_clean[col].notna().sum()) for col in caption_cols
    }
    captions_dir = PROJECT_ROOT / "data" / "catalog" / "captions"
    shard_files = sorted(captions_dir.glob("*.parquet")) if captions_dir.exists() else []
    shard_summary = {}
    all_captioned_pids = set()
    for sf in shard_files:
        sdf = pd.read_parquet(sf)
        pids = set(sdf["product_id"].astype(str)) if "product_id" in sdf.columns else set()
        all_captioned_pids |= pids
        shard_summary[sf.name] = {
            "rows": len(sdf),
            "overlap_with_train_clean": len(pids & (set(train_clean["pid_str"]) - val_ids_set)),
            "overlap_with_val_json": len(pids & val_ids_set),
            "overlap_with_test_json": len(pids & test_ids_set),
        }

    res["11_train_caption_status"] = {
        "catalog_caption_columns": caption_cols,
        "train_clean_non_null_in_catalog": train_caption_non_null,
        "caption_shards_on_disk": shard_summary,
        "train_excluding_val_captioned_in_shards": len(all_captioned_pids & (set(train_clean["pid_str"]) - val_ids_set)),
    }

    (OUT_DIR / "phase2_results.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
