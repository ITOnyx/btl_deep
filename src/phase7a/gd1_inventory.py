"""Phase 7a GĐ1 — Inventory & Cross-Reference Audit for merged_dataset.

Reads config.yaml for all paths/thresholds, scans merged_dataset/annotations/all.parquet
and image files on disk (resumable via shard parquets), compares against catalog.parquet
(ID, path, SHA-1), audits build_merged_dataset.py filters vs config.yaml, and assigns
conservative `seen_in_train` labels.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.phase7a.common import ensure_utf8_stdio, load_config, resolve_phase7a_paths
from src.phase7a.img_workers import hash_catalog_image_worker, inspect_merged_image_worker

REJECTS_COLUMNS = [
    "uid",
    "source",
    "orig_id",
    "image_path",
    "stage",
    "reason",
    "detail",
    "short_side",
    "blur_score",
    "banner_flag",
    "phash",
    "duplicate_of",
]


def _run_sharded_pool(
    items: List[Tuple[str, str]],
    worker_fn: Any,
    shards_dir: Path,
    prefix: str,
    shard_size: int = 5000,
    max_workers: int | None = None,
    use_threads: bool = False,
) -> pd.DataFrame:
    """Execute worker_fn across items in resumable parquet shards using one persistent pool."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    n_workers = max_workers or min(12, max(2, os.cpu_count() or 4))
    n_shards = max(1, (len(items) + shard_size - 1) // shard_size)
    shard_frames: List[Optional[pd.DataFrame]] = [None] * n_shards
    todo_indices: List[int] = []

    for s_idx in range(n_shards):
        shard_path = shards_dir / f"{prefix}_shard_{s_idx:04d}.parquet"
        start = s_idx * shard_size
        end = min(len(items), start + shard_size)
        expected_n = end - start
        if shard_path.exists():
            df_s = pd.read_parquet(shard_path)
            if len(df_s) == expected_n:
                shard_frames[s_idx] = df_s
                continue
        todo_indices.append(s_idx)

    if todo_indices:
        pool_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        with pool_cls(max_workers=n_workers) as ex:
            for s_idx in todo_indices:
                start = s_idx * shard_size
                end = min(len(items), start + shard_size)
                batch = items[start:end]
                shard_path = shards_dir / f"{prefix}_shard_{s_idx:04d}.parquet"
                if use_threads:
                    rows = list(ex.map(worker_fn, batch))
                else:
                    rows = list(ex.map(worker_fn, batch, chunksize=64))
                df_s = pd.DataFrame(rows)
                df_s.to_parquet(shard_path, index=False)
                shard_frames[s_idx] = df_s
                print(f"  [{prefix}] shard {s_idx + 1}/{n_shards} saved ({len(df_s):,} rows)", flush=True)

    return pd.concat([f for f in shard_frames if f is not None], ignore_index=True)


def _is_missing_series(s: pd.Series) -> pd.Series:
    """Detect None/NaN as well as empty or sentinel string placeholders."""
    na_mask = s.isna()
    s_str = s.fillna("").astype(str).str.strip().str.lower()
    sentinels = {"", "nan", "none", "na", "null", "unknown", "{}"}
    return na_mask | s_str.isin(sentinels)


def _resolve_dense_caption_uids(
    checkpoint_dir: Path,
    df_all: pd.DataFrame,
) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
    """Independently map dense_captions_catalog.json (4,372 test items) to merged UIDs."""
    dc_path = checkpoint_dir / "dense_captions_catalog.json"
    pm_old_path = checkpoint_dir / "product_metadata_old.json"

    dc = json.loads(dc_path.read_text(encoding="utf-8"))
    pm_old = json.loads(pm_old_path.read_text(encoding="utf-8")) if pm_old_path.exists() else []

    # Build Shopping100k stem/full lookup from merged df_all
    sh_df = df_all[df_all["source"] == "shopping100k"]
    sh_by_full: Set[str] = set(sh_df["uid"].tolist())
    sh_by_stem_gender: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    sh_by_stem: Dict[str, List[str]] = defaultdict(list)

    for uid, orig_id in zip(sh_df["uid"], sh_df["orig_id"]):
        parts = Path(str(orig_id)).with_suffix("").parts
        stem = parts[-1].lower()
        g_folder = parts[0].lower() if parts else ""
        sh_by_stem_gender[(stem, g_folder)].append(uid)
        sh_by_stem[stem].append(uid)

    pm_by_pid = {str(x["product_id"]): x for x in pm_old}
    resolved_fashion: Set[str] = set()
    resolved_sh100k: Set[str] = set()
    external_tiki_shopee: Counter = Counter()
    unresolved: List[str] = []

    for pid in dc.keys():
        pid_s = str(pid)
        if pid_s.isdigit():
            resolved_fashion.add(f"fashion_{pid_s}")
            continue
        if pid_s.startswith("myntra_"):
            resolved_fashion.add(f"fashion_{pid_s[len('myntra_'):]}")
            continue
        if pid_s.startswith("tiki_") or pid_s.startswith("shopee_"):
            external_tiki_shopee[pid_s.split("_", 1)[0]] += 1
            continue
        if pid_s.startswith("shop100k_"):
            slug = pid_s[len("shop100k_"):].lower()
            cand = f"shopping100k_{slug}"
            if cand in sh_by_full:
                resolved_sh100k.add(cand)
                continue
            attrs = (pm_by_pid.get(pid_s) or {}).get("attributes") or {}
            g_raw = str(attrs.get("gender", "")).lower()
            g_folder = (
                "female"
                if ("nữ" in g_raw or "female" in g_raw or "women" in g_raw)
                else ("male" if ("nam" in g_raw or "male" in g_raw or "men" in g_raw) else "")
            )
            cands = sh_by_stem_gender.get((slug, g_folder), [])
            if len(cands) == 1:
                resolved_sh100k.add(cands[0])
                continue
            cands2 = sh_by_stem.get(slug, [])
            if len(cands2) == 1:
                resolved_sh100k.add(cands2[0])
                continue
            unresolved.append(pid_s)
        else:
            unresolved.append(pid_s)

    dc_all_uids = resolved_fashion | resolved_sh100k
    meta = {
        "dense_captions_total": len(dc),
        "resolved_fashion": len(resolved_fashion),
        "resolved_shopping100k_unique_uids": len(resolved_sh100k),
        "external_unresolvable_counts": dict(external_tiki_shopee),
        "unresolved_in_source": len(unresolved),
    }
    return dc_all_uids, resolved_fashion, meta


def _audit_scope_consistency(cfg: Dict[str, Any], df_all: pd.DataFrame) -> Dict[str, Any]:
    """Check whether all.parquet['in_scope'] matches config.yaml scope exclusions."""
    scope_cfg = cfg.get("scope", {})
    excl_master = set(scope_cfg.get("exclude_master_categories", []))
    excl_art = set(scope_cfg.get("exclude_article_types", []))
    incl_exc = set(scope_cfg.get("include_article_type_exceptions", []))

    expected = []
    for cat, art in zip(df_all["category"], df_all["article_type"]):
        ok = True
        if cat in excl_master and art not in incl_exc:
            ok = False
        if art in excl_art:
            ok = False
        expected.append(ok)
    exp_arr = np.array(expected, dtype=bool)
    actual_arr = df_all["in_scope"].to_numpy(dtype=bool)
    mismatch = int((exp_arr != actual_arr).sum())
    return {
        "in_scope_true": int(actual_arr.sum()),
        "in_scope_false": int((~actual_arr).sum()),
        "config_scope_mismatch_count": mismatch,
    }


def _inspect_build_filters(cfg: Dict[str, Any], build_script: Path) -> Dict[str, Any]:
    """Compare config.yaml image_filter vs scripts/build_merged_dataset.py behavior."""
    text = build_script.read_text(encoding="utf-8")
    img_cfg = cfg.get("image_filter", {})
    return {
        "config_min_short_side": img_cfg.get("min_short_side"),
        "build_min_short_side": 100 if "short_side\"] < 100" in text else None,
        "config_blur_threshold": img_cfg.get("blur_laplacian_var_threshold"),
        "build_uses_laplacian_blur": "Laplacian" in text,
        "config_detect_banner": img_cfg.get("detect_banner"),
        "build_uses_banner_detector": "banner_flag" in text or "Canny" in text,
        "config_phash_threshold": img_cfg.get("phash_hamming_threshold"),
        "build_dedup_drop_rule": "md5_exact + (phash_hamming==0 & same article_type+gender+color)",
        "build_label_group_cluster_rule": "pHash<=4 Star/UnionFind (at_gen+color<=2 or at_gen<=1, max_sz=12)",
        "config_dedup_match_attributes": img_cfg.get("dedup_match_attributes"),
        "config_dedup_transitive": img_cfg.get("dedup_transitive"),
    }


def run_gd1(
    config_path: str | Path | None = None,
    merged_dir_override: str | Path | None = None,
    shard_size: int = 5000,
    max_workers: int | None = None,
) -> None:
    """Execute Phase 7a GĐ1 inventory and cross-reference audit."""
    t0 = time.time()
    ensure_utf8_stdio()
    cfg = load_config(config_path)
    paths = resolve_phase7a_paths(cfg, merged_dir_override=merged_dir_override)

    merged_dir = paths["merged_dir"]
    eval_p7 = paths["eval_p7"]
    reports_p7 = paths["reports_p7"]
    shards_dir = paths["shards_dir"]
    eval_p7.mkdir(parents=True, exist_ok=True)
    reports_p7.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHA 7a — GĐ1: KIỂM KÊ & ĐỐI CHIẾU merged_dataset")
    print("=" * 72)
    print(f"Nguồn merged_dataset (chỉ đọc): {merged_dir}")
    print(f"Catalog hiện tại (chỉ đọc)    : {paths['catalog_path']}")

    all_parquet = merged_dir / "annotations" / "all.parquet"
    splits_path = merged_dir / "annotations" / "splits.json"
    rejected_csv = merged_dir / "reports" / "rejected.csv"
    duplicates_csv = merged_dir / "reports" / "duplicates.csv"

    df_all = pd.read_parquet(all_parquet)
    cat_df = pd.read_parquet(paths["catalog_path"])
    splits_json = json.loads(splits_path.read_text(encoding="utf-8"))

    # 1. Source, split, metadata missingness audit
    print("[1/5] Thống kê nguồn, split và tỷ lệ thiếu metadata...")
    src_counts = df_all["source"].value_counts().to_dict()
    split_counts = df_all["split"].value_counts().to_dict()
    src_split_cross = (
        df_all.groupby(["source", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    leg_by_src = (
        df_all.groupby("source")["legacy_test"].sum().astype(int).to_dict()
    )

    meta_cols_to_check = [
        "title",
        "title_vi",
        "brand",
        "category",
        "sub_category",
        "article_type",
        "gender",
        "color",
        "usage",
        "season",
        "captions",
        "attributes",
        "label_group",
    ]
    missing_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for src_name in sorted(df_all["source"].unique()):
        sub = df_all[df_all["source"] == src_name]
        n_sub = len(sub)
        src_stats: Dict[str, Dict[str, Any]] = {}
        for col in meta_cols_to_check:
            if col not in sub.columns:
                src_stats[col] = {"present": False, "missing": n_sub, "pct": 100.0}
                continue
            miss_n = int(_is_missing_series(sub[col]).sum())
            src_stats[col] = {
                "present": True,
                "missing": miss_n,
                "pct": round(miss_n / max(1, n_sub) * 100.0, 2),
            }
        missing_stats[src_name] = src_stats

    # 2. Scan all merged_dataset images via PIL + SHA1 (resumable shards)
    print(f"[2/5] Kiểm tra đọc PIL & tính SHA-1 cho {len(df_all):,} ảnh merged_dataset...")
    merged_tasks = [
        (str(uid), str(merged_dir / rel_p))
        for uid, rel_p in zip(df_all["uid"], df_all["image_path"])
    ]
    df_img = _run_sharded_pool(
        merged_tasks,
        inspect_merged_image_worker,
        shards_dir,
        prefix="gd1_merged",
        shard_size=shard_size,
        max_workers=max_workers or 10,
        use_threads=True,
    )

    # Save any unreadable images to eval/phase7/candidate_rejects.csv without stopping
    unreadable_df = df_img[~df_img["readable"]].copy()
    reject_rows: List[Dict[str, Any]] = []
    if len(unreadable_df) > 0:
        unr_merged = unreadable_df.merge(
            df_all[["uid", "source", "orig_id", "image_path"]], on="uid", how="left"
        )
        for r in unr_merged.itertuples(index=False):
            reject_rows.append(
                {
                    "uid": r.uid,
                    "source": r.source,
                    "orig_id": r.orig_id,
                    "image_path": r.image_path,
                    "stage": "gd1",
                    "reason": "unreadable",
                    "detail": r.read_error,
                    "short_side": 0,
                    "blur_score": np.nan,
                    "banner_flag": False,
                    "phash": "",
                    "duplicate_of": "",
                }
            )
    rejects_path = eval_p7 / "candidate_rejects.csv"
    pd.DataFrame(reject_rows, columns=REJECTS_COLUMNS).to_csv(
        rejects_path, index=False, encoding="utf-8"
    )
    print(f"  - Ảnh lỗi không đọc được (unreadable): {len(reject_rows):,}")

    # 3. Compute catalog SHA1 & cross-reference
    print(f"[3/5] Đối chiếu với catalog hiện tại ({len(cat_df):,} dòng, {int(cat_df['in_scope'].sum()):,} in_scope)...")
    repo_root = paths["repo_root"]
    cat_tasks = [
        (str(iid), str((repo_root / p_str) if not Path(p_str).is_absolute() else Path(p_str)))
        for iid, p_str in zip(cat_df["item_id"], cat_df["path"])
    ]
    df_cat_hash = _run_sharded_pool(
        cat_tasks,
        hash_catalog_image_worker,
        shards_dir,
        prefix="gd1_catalog",
        shard_size=shard_size,
        max_workers=max_workers or 16,
        use_threads=True,
    )

    cat_meta = cat_df[["item_id", "path", "in_scope", "split"]].copy()
    cat_meta["item_id"] = cat_meta["item_id"].astype(str)
    cat_meta = cat_meta.merge(df_cat_hash[["item_id", "file_exists", "sha1"]], on="item_id", how="left")

    cat_all_ids: Set[str] = set(cat_meta["item_id"].tolist())
    cat_inscope_ids: Set[str] = set(cat_meta[cat_meta["in_scope"]]["item_id"].tolist())
    cat_all_paths: Set[str] = set(cat_meta["path"].astype(str).tolist())
    cat_inscope_paths: Set[str] = set(cat_meta[cat_meta["in_scope"]]["path"].astype(str).tolist())
    cat_split_map: Dict[str, str] = dict(zip(cat_meta["item_id"], cat_meta["split"]))

    sha1_to_cat_all: Dict[str, str] = {}
    sha1_to_cat_inscope: Dict[str, str] = {}
    for r in cat_meta.itertuples(index=False):
        if r.sha1:
            sha1_to_cat_all.setdefault(r.sha1, r.item_id)
            if r.in_scope:
                sha1_to_cat_inscope.setdefault(r.sha1, r.item_id)

    # Combine df_all with PIL + SHA1 columns cleanly
    df_inv = df_all.merge(
        df_img[
            [
                "uid",
                "file_exists",
                "file_size",
                "sha1",
                "readable",
                "read_error",
                "pil_format",
                "pil_mode",
                "pil_width",
                "pil_height",
                "exif_orientation",
                "has_exif_rotation",
                "has_alpha",
                "is_cmyk",
                "is_grayscale",
            ]
        ],
        on="uid",
        how="left",
    )

    # Candidate original paths before merge (fashion-dataset/<id>.jpg vs Shopping100k/Images/<rel>)
    orig_rel_paths = [
        f"data/fashion-dataset/images/{oid}.jpg" if src == "fashion" else f"data/Shopping100k/Images/{oid}"
        for src, oid in zip(df_inv["source"], df_inv["orig_id"])
    ]
    df_inv["orig_rel_path"] = orig_rel_paths

    in_cat_id = []
    in_cat_inscope_id = []
    in_cat_path = []
    in_cat_sha1 = []
    in_cat_inscope_sha1 = []
    matched_cat_id = []

    for src, oid, img_p, orig_p, s1 in zip(
        df_inv["source"], df_inv["orig_id"], df_inv["image_path"], df_inv["orig_rel_path"], df_inv["sha1"]
    ):
        oid_s = str(oid)
        by_id = (src == "fashion") and (oid_s in cat_all_ids)
        by_in_id = (src == "fashion") and (oid_s in cat_inscope_ids)
        by_path = (img_p in cat_all_paths) or (orig_p in cat_all_paths)
        c_sha_all = sha1_to_cat_all.get(s1, "") if s1 else ""
        c_sha_in = sha1_to_cat_inscope.get(s1, "") if s1 else ""

        in_cat_id.append(by_id)
        in_cat_inscope_id.append(by_in_id)
        in_cat_path.append(by_path)
        in_cat_sha1.append(bool(c_sha_all))
        in_cat_inscope_sha1.append(bool(c_sha_in))
        matched_cat_id.append(oid_s if by_id else (c_sha_all or ""))

    df_inv["in_catalog_by_id"] = in_cat_id
    df_inv["in_catalog_inscope_by_id"] = in_cat_inscope_id
    df_inv["in_catalog_by_path"] = in_cat_path
    df_inv["in_catalog_by_sha1"] = in_cat_sha1
    df_inv["in_catalog_inscope_by_sha1"] = in_cat_inscope_sha1
    df_inv["in_catalog_any"] = (
        df_inv["in_catalog_by_id"] | df_inv["in_catalog_by_path"] | df_inv["in_catalog_by_sha1"]
    )
    df_inv["catalog_match_item_id"] = matched_cat_id

    # Diagnose the 150 fashion items not in catalog_by_id
    quarantine_map: Dict[str, str] = {}
    q_dir = paths["repo_root"] / cfg["paths"]["quarantine"]
    if q_dir.exists():
        for qp in q_dir.rglob("*.jpg"):
            quarantine_map[qp.stem] = qp.parent.name

    styles_csv_path = paths["repo_root"] / cfg["paths"]["kaggle_dir"] / "styles.csv"
    styles_std_ids: Set[str] = set()
    if styles_csv_path.exists():
        df_st_std = pd.read_csv(styles_csv_path, on_bad_lines="skip", usecols=["id"])
        styles_std_ids = set(df_st_std["id"].astype(str).tolist())

    f_not_cat = df_inv[(df_inv["source"] == "fashion") & (~df_inv["in_catalog_by_id"])]
    f_not_cat_reasons: Counter = Counter()
    for oid, by_sha in zip(f_not_cat["orig_id"], f_not_cat["in_catalog_by_sha1"]):
        oid_s = str(oid)
        q_r = quarantine_map.get(oid_s)
        if q_r:
            tag = f"quarantine_{q_r}"
        elif oid_s not in styles_std_ids:
            tag = "styles_csv_bad_line_rescued"
        else:
            tag = "other_not_in_catalog"
        if by_sha:
            tag += "+sha1_dup_in_catalog"
        f_not_cat_reasons[tag] += 1

    # 4. Audit build_merged_dataset.py vs config.yaml + scope check
    print("[4/5] Đối chiếu bộ lọc build_merged_dataset.py vs config.yaml...")
    filter_audit = _inspect_build_filters(cfg, paths["build_script"])
    scope_audit = _audit_scope_consistency(cfg, df_inv)

    # 5. Assign seen_in_train in {no, yes, unknown}
    print("[5/5] Xác định seen_in_train đối chiếu dense_captions_catalog.json, splits.json, catalog split...")
    dc_uids, dc_fashion_uids, dc_meta = _resolve_dense_caption_uids(paths["checkpoint_dir"], df_inv)
    splits_leg_uids: Set[str] = set(splits_json.get("legacy_test_uids", []))
    cat_test_ids: Set[str] = {k for k, v in cat_split_map.items() if v == "test"}

    # Check if any non-legacy item shares identical SHA1 with a confirmed test item
    test_sha1s: Set[str] = set(
        df_inv.loc[df_inv["uid"].isin(dc_uids | splits_leg_uids) & (df_inv["sha1"] != ""), "sha1"].tolist()
    )

    seen_vals: List[str] = []
    seen_reasons: List[str] = []
    for uid, src, oid, s1 in zip(df_inv["uid"], df_inv["source"], df_inv["orig_id"], df_inv["sha1"]):
        oid_s = str(oid)
        in_dc = uid in dc_uids
        in_leg = uid in splits_leg_uids
        in_cat_test = (src == "fashion") and (oid_s in cat_test_ids)

        if in_dc or in_leg or in_cat_test:
            seen_vals.append("no")
            if in_dc and in_leg and in_cat_test:
                seen_reasons.append("confirmed_test:dense_captions+splits_legacy_test+catalog_test")
            elif in_dc and in_leg:
                seen_reasons.append("confirmed_test:dense_captions+splits_legacy_test")
            else:
                flags = []
                if in_dc:
                    flags.append("dense_captions")
                if in_leg:
                    flags.append("splits_legacy_test")
                if in_cat_test:
                    flags.append("catalog_test")
                seen_reasons.append("partial_test:" + "+".join(flags))
        elif s1 and s1 in test_sha1s:
            seen_vals.append("unknown")
            seen_reasons.append("sha1_matches_legacy_test_item_not_in_id_list")
        else:
            seen_vals.append("unknown")
            seen_reasons.append("no_colab_161k_train_val_id_manifest_on_disk")

    df_inv["seen_in_train"] = seen_vals
    df_inv["seen_in_train_reason"] = seen_reasons
    seen_summary = df_inv.groupby(["source", "seen_in_train"]).size().to_dict()
    seen_reason_summary = df_inv["seen_in_train_reason"].value_counts().to_dict()

    # Save eval/phase7/inventory.parquet
    inv_path = eval_p7 / "inventory.parquet"
    df_inv.to_parquet(inv_path, index=False)
    print(f"Đã lưu inventory: {inv_path} ({len(df_inv):,} dòng)")

    # Collect PIL stats for report
    pil_modes = df_inv.groupby(["source", "pil_mode"]).size().to_dict()
    pil_formats = df_inv["pil_format"].value_counts().to_dict()
    exif_rot_cnt = df_inv.groupby("source")["has_exif_rotation"].sum().astype(int).to_dict()
    exif_orient_dist = df_inv["exif_orientation"].value_counts().to_dict()
    alpha_cnt = df_inv.groupby("source")["has_alpha"].sum().astype(int).to_dict()
    cmyk_cnt = df_inv.groupby("source")["is_cmyk"].sum().astype(int).to_dict()
    gray_cnt = df_inv.groupby("source")["is_grayscale"].sum().astype(int).to_dict()
    wh_mismatch = int(((df_inv["width"] != df_inv["pil_width"]) | (df_inv["height"] != df_inv["pil_height"])).sum())

    rej_src_df = pd.read_csv(rejected_csv) if rejected_csv.exists() else pd.DataFrame()
    dup_src_df = pd.read_csv(duplicates_csv) if duplicates_csv.exists() else pd.DataFrame()

    # Compute catalog overlap summary by source
    f_df = df_inv[df_inv["source"] == "fashion"]
    s_df = df_inv[df_inv["source"] == "shopping100k"]

    f_dup_all_id_or_sha = int((f_df["in_catalog_by_id"] | f_df["in_catalog_by_sha1"]).sum())
    f_dup_id = int(f_df["in_catalog_by_id"].sum())
    f_dup_sha_only = int((~f_df["in_catalog_by_id"] & f_df["in_catalog_by_sha1"]).sum())
    f_new = len(f_df) - f_dup_all_id_or_sha

    s_dup_sha = int(s_df["in_catalog_by_sha1"].sum())
    s_new = len(s_df) - s_dup_sha

    # Also check how many catalog items exist that are NOT in merged_dataset
    cat_in_merged_id = int(cat_meta["item_id"].isin(set(f_df["orig_id"].astype(str))).sum())
    merged_all_sha1s = set(df_inv.loc[df_inv["sha1"] != "", "sha1"].tolist())
    cat_in_merged_sha = int(cat_meta["sha1"].isin(merged_all_sha1s).sum())

    # Write reports/phase7/gd1_inventory.md
    report_lines = [
        "# Pha 7a — GĐ1: Báo cáo Kiểm kê & Đối chiếu `merged_dataset`",
        "",
        f"- **Thời gian chạy:** {time.time() - t0:.1f}s | **Seed:** `{cfg.get('eval', {}).get('seed', 42)}`",
        f"- **Nguồn đọc (chỉ đọc):** `{merged_dir}`",
        f"- **Catalog hiện tại (chỉ đọc):** `{paths['catalog_path']}` ({len(cat_df):,} dòng, {int(cat_df['in_scope'].sum()):,} `in_scope`)",
        f"- **Dữ liệu xuất:** `eval/phase7/inventory.parquet` ({len(df_inv):,} dòng x {df_inv.shape[1]} cột), `eval/phase7/candidate_rejects.csv` ({len(reject_rows)} dòng)",
        "",
        "---",
        "",
        "## 1. Số lượng theo nguồn, split và tỷ lệ thiếu metadata (`all.parquet`)",
        "",
        "Lưu ý cấu trúc thực tế trên đĩa: thư mục `images/` trong `merged_dataset` đã được làm phẳng thành `images/<uid>.jpg` (`139.139` ảnh, không chia thư mục con `images/fashion/` hay `images/shopping100k/`).",
        "",
        "### 1.1 Phân bố theo Nguồn × Split & `legacy_test`",
        "",
        "| Nguồn (`source`) | `train` | `val` | `test` | Tổng | Trong đó `legacy_test` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s_name in ["fashion", "shopping100k"]:
        sp_map = src_split_cross.get(s_name, {})
        tr_n = int(sp_map.get("train", 0))
        va_n = int(sp_map.get("val", 0))
        te_n = int(sp_map.get("test", 0))
        tot_n = int(src_counts.get(s_name, 0))
        leg_n = int(leg_by_src.get(s_name, 0))
        report_lines.append(f"| `{s_name}` | {tr_n:,} | {va_n:,} | {te_n:,} | **{tot_n:,}** | {leg_n:,} |")
    report_lines.append(
        f"| **Tổng** | {int(split_counts.get('train', 0)):,} | {int(split_counts.get('val', 0)):,} | {int(split_counts.get('test', 0)):,} | **{len(df_inv):,}** | **{int(df_inv['legacy_test'].sum()):,}** |"
    )

    report_lines.extend([
        "",
        "### 1.2 Cột metadata thực tế & Tỷ lệ thiếu theo nguồn",
        "",
        "(Tính cả `NaN`/`None` và chuỗi rỗng/sentinel như `\"\"`, `\"NA\"`, `\"Unknown\"`, `\"{}\"`; lưu ý schema `all.parquet` dùng tên cột `article_type`, `color`, `title` tương ứng với `articleType`, `baseColour`, `name` ở `catalog.parquet`.)",
        "",
        "| Cột (`all.parquet`) | Vai trò | Thiếu `fashion` (40.953) | % | Thiếu `shopping100k` (98.186) | % |",
        "|---|---|---:|---:|---:|---:|",
    ])
    role_map = {
        "title": "Tên tiếng Anh (gốc/tổng hợp)",
        "title_vi": "Tên tiếng Việt (tổng hợp/cũ)",
        "brand": "Thương hiệu",
        "category": "Danh mục cấp 1 (masterCategory)",
        "sub_category": "Danh mục cấp 2 (subCategory)",
        "article_type": "Loại sản phẩm (articleType)",
        "gender": "Giới tính",
        "color": "Màu sắc (baseColour)",
        "usage": "Mục đích sử dụng (usage)",
        "season": "Mùa (season)",
        "captions": "JSON caption đa góc nhìn",
        "attributes": "JSON thuộc tính chi tiết",
        "label_group": "Cụm biến thể",
    }
    for col in meta_cols_to_check:
        f_st = missing_stats.get("fashion", {}).get(col, {})
        s_st = missing_stats.get("shopping100k", {}).get(col, {})
        report_lines.append(
            f"| `{col}` | {role_map.get(col, '')} | {f_st.get('missing', 0):,} | {f_st.get('pct', 0):.2f}% | {s_st.get('missing', 0):,} | {s_st.get('pct', 0):.2f}% |"
        )

    report_lines.extend([
        "",
        f"Nhận xét thiếu metadata ở `shopping100k`: `brand` thiếu {missing_stats['shopping100k']['brand']['pct']:.2f}%, `season` thiếu {missing_stats['shopping100k']['season']['pct']:.2f}% (phần có nhãn do `build_merged_dataset.py` suy diễn quy tắc cho nhóm mùa đông/hè), `color` thiếu {missing_stats['shopping100k']['color']['pct']:.2f}% (`color.txt` có nhãn `-1`).",
        "",
        "---",
        "",
        "## 2. Kiểm tra đọc ảnh bằng PIL & thuộc tính pixel",
        "",
        f"- **Tổng file tồn tại trên đĩa:** {int(df_inv['file_exists'].sum()):,} / {len(df_inv):,}",
        f"- **Đọc giải mã pixel PIL thành công (`im.load()`):** **{int(df_img['readable'].sum()):,} / {len(df_inv):,}** | Lỗi `unreadable`: **{len(reject_rows)}** (đã ghi `eval/phase7/candidate_rejects.csv`)",
        f"- **Định dạng ảnh (`pil_format`):** {pil_formats}",
        f"- **Mode ảnh theo nguồn:** { {f'{k[0]}:{k[1]}': int(v) for k, v in pil_modes.items()} }",
        f"- **EXIF Orientation khác 1 (`has_exif_rotation`):** {exif_rot_cnt} (phân bố tag: {exif_orient_dist})",
        f"- **Kênh Alpha (`has_alpha`):** {alpha_cnt} | **CMYK:** {cmyk_cnt} | **Ảnh xám (`L/1/I/F/LA`):** {gray_cnt}",
        f"- **Sai lệch `(pil_width, pil_height)` so với `all.parquet`:** {wh_mismatch} ảnh",
        "",
        "---",
        "",
        "## 3. Đối chiếu trùng lặp với `catalog.parquet` (40.891 dòng / 38.644 `in_scope`)",
        "",
        "| Nguồn (`merged_dataset`) | Tổng trong merged | Trùng ID catalog (40.891) | Trùng thêm chỉ qua SHA-1 | Tổng TRÙNG catalog (ID ∪ SHA-1) | Thực sự MỚI so với 40.891 | Trong đó trùng `in_scope` (38.644) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `fashion` | {len(f_df):,} | {f_dup_id:,} | {f_dup_sha_only:,} | **{f_dup_all_id_or_sha:,}** | **{f_new:,}** | {int((f_df['in_catalog_inscope_by_id'] | f_df['in_catalog_inscope_by_sha1']).sum()):,} |",
        f"| `shopping100k` | {len(s_df):,} | 0 | {s_dup_sha:,} | **{s_dup_sha:,}** | **{s_new:,}** | {int(s_df['in_catalog_inscope_by_sha1'].sum()):,} |",
        f"| **Tổng** | **{len(df_inv):,}** | {f_dup_id:,} | {f_dup_sha_only + s_dup_sha:,} | **{f_dup_all_id_or_sha + s_dup_sha:,}** | **{f_new + s_new:,}** | {int((df_inv['in_catalog_inscope_by_id'] | df_inv['in_catalog_inscope_by_sha1']).sum()):,} |",
        "",
        f"- **Đối chiếu theo đường dẫn gốc (`orig_rel_path` vs `catalog.path`):** {int(df_inv['in_catalog_by_path'].sum()):,} dòng khớp `data/fashion-dataset/images/<id>.jpg`.",
        f"- **Chiều ngược lại (`catalog.parquet` → `merged_dataset`):** {cat_in_merged_id:,} / {len(cat_df):,} ID của catalog có mặt trong `merged_dataset` (và {cat_in_merged_sha:,} khớp SHA-1); {len(cat_df) - cat_in_merged_id:,} ID catalog vắng mặt trong `merged_dataset` (do `build_merged_dataset.py` khử trùng MD5 / `pHash=0` hoặc `pHash` cụm đa nguồn).",
        f"- **Giải thích {len(f_not_cat):,} ảnh `fashion` KHÔNG trùng ID trong `catalog.parquet`:** {dict(f_not_cat_reasons)} (gồm các ảnh bị loại ở Pha 1 vào `_quarantine` nhưng được `build_merged_dataset.py` cứu do thuộc 1.500 `legacy_test`, các dòng lỗi dấu phẩy trong `styles.csv` được cứu bằng `split(',', 9)`, và các trường hợp trùng SHA-1 với ID khác trong catalog).",
        "",
        "---",
        "",
        "## 4. Kiểm toán bộ lọc của `scripts/build_merged_dataset.py` so với `config.yaml`",
        "",
        "| Tiêu chí | `config.yaml` (Chuẩn dự án) | `scripts/build_merged_dataset.py` (Thực tế đã chạy) | Sai khác & Rủi ro còn lại |",
        "|---|---|---|---|",
        f"| `min_short_side` | **{filter_audit['config_min_short_side']} px** | **`{filter_audit['build_min_short_side']} px`** (và bỏ qua nếu là `legacy_test`) | Shopping100k và `legacy_test` có thể chứa ảnh `100..223 px` (< 224 px). |",
        f"| Độ mờ (`blur`) | **`blur_laplacian_var_threshold = {filter_audit['config_blur_threshold']}`** | **Không tính (`{filter_audit['build_uses_laplacian_blur']}`)**; `fashion` chỉ kế thừa `_quarantine/blurry` (trừ `legacy_test`) | **98.186 ảnh Shopping100k chưa từng được kiểm tra độ mờ Laplacian!** |",
        f"| Banner / viền chữ | **`detect_banner = {filter_audit['config_detect_banner']}`** | **Không chạy (`{filter_audit['build_uses_banner_detector']}`)**; `fashion` chỉ kế thừa `_quarantine/banner` (trừ `legacy_test`) | **98.186 ảnh Shopping100k chưa từng chạy bộ phát hiện banner/chữ.** |",
        f"| Khử trùng lặp (`pHash`) | `pHash <= {filter_audit['config_phash_threshold']}`, cùng `{filter_audit['config_dedup_match_attributes']}`, `transitive={filter_audit['config_dedup_transitive']}`, ưu tiên `short_side -> blur -> id` | Chỉ xóa **MD5 giống hệt** và **`pHash == 0` cùng thuộc tính**; còn **`pHash == 1`** chỉ gom `label_group`, KHÔNG loại | Ảnh gần trùng `pHash == 1` vẫn còn trong `merged_dataset`; không dùng `blur_score` để chọn đại diện. |",
        f"| Phạm vi (`in_scope`) | `exclude_master_categories`, `exclude_article_types`, giữ `Wristbands` | Đọc từ `config.yaml`, nhưng giữ cả **`{scope_audit['in_scope_false']:,}` dòng `in_scope=False`** vào dataset (`mismatch={scope_audit['config_scope_mismatch_count']}`) | Phải lọc lại `in_scope == True` & 2 Keychain (`43230`, `42066`) ở GĐ3. |",
        "",
        f"Thống kê báo cáo gốc của `merged_dataset`: `rejected.csv` có **{len(rej_src_df):,}** dòng (`{rej_src_df['reason'].value_counts().to_dict() if not rej_src_df.empty else {}}`), `duplicates.csv` có **{len(dup_src_df):,}** dòng.",
        "",
        "---",
        "",
        "## 5. Gắn nhãn `seen_in_train` (`no` / `yes` / `unknown`)",
        "",
        "### Căn cứ từng giá trị",
        f"- **`no` (chắc chắn KHÔNG nằm trong tập train của checkpoint hiện tại):** Các UID đối chiếu khớp được với tập Test gốc **4.372 sản phẩm** trong `data/checkpoints_siglip-vit/dense_captions_catalog.json` (tương ứng `product_metadata_old.json` gồm `1.500 myntra + 1.500 shop100k + 1.372 tiki`), khớp `splits.json['legacy_test_uids']` (`{len(splits_leg_uids):,}` UID = `1.500 fashion + 1.471 shopping100k` duy nhất), và `catalog.parquet['split'] == 'test'` (`1.372 fashion`). Ghi chú: `dense_captions_catalog.json` ánh xạ được `1.500 fashion` + `1.471 shopping100k` duy nhất (`17` mã `shop100k` trỏ trùng ảnh vật lý, còn `1.372 tiki` + `12 shopee` là nguồn ngoài).",
        f"- **`yes` (chắc chắn nằm trong tập train của checkpoint hiện tại):** **`0` ảnh.** Lý do: Notebook huấn luyện (`notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`) huấn luyện trên **161.870 ảnh** (`Train = 153.126 | Val = 4.372 | Test = 4.372`), nhưng danh sách ID của 153.126 ảnh `train` và 4.372 ảnh `val` **không tồn tại trên đĩa** (ngay cả `catalog.parquet` cũng chỉ gắn được `test = 1.372`, còn lại `39.519` gắn `unknown`). Ngoài ra `splits.json['train'/'val'/'test']` (111.311 / 13.914 / 13.914) là phân chia mới do `build_merged_dataset.py` tạo *sau này* — không phải phân chia lúc train checkpoint đang chạy (`10.943` ảnh `test` mới trong `splits.json` KHÔNG thuộc 4.372 test gốc). Theo nguyên tắc không chắc thì không đoán, không gắn `yes` bừa.",
        "- **`unknown`:** Toàn bộ các ảnh không nằm trong danh sách 4.372 test gốc (về mặt xác suất phần lớn nằm trong 153.126 ảnh train của checkpoint nhưng không có manifest ID trên đĩa để khẳng định từng ID là train vs val hay bị lọc trước Colab).",
        "",
        "| Nguồn | `seen_in_train = no` | `seen_in_train = yes` | `seen_in_train = unknown` | Tổng |",
        "|---|---:|---:|---:|---:|",
        f"| `fashion` | {int(seen_summary.get(('fashion', 'no'), 0)):,} | 0 | {int(seen_summary.get(('fashion', 'unknown'), 0)):,} | {len(f_df):,} |",
        f"| `shopping100k` | {int(seen_summary.get(('shopping100k', 'no'), 0)):,} | 0 | {int(seen_summary.get(('shopping100k', 'unknown'), 0)):,} | {len(s_df):,} |",
        f"| **Tổng** | **{int(seen_summary.get(('fashion', 'no'), 0) + seen_summary.get(('shopping100k', 'no'), 0)):,}** | **0** | **{int(seen_summary.get(('fashion', 'unknown'), 0) + seen_summary.get(('shopping100k', 'unknown'), 0)):,}** | **{len(df_inv):,}** |",
        "",
        f"Chi tiết lý do `seen_in_train_reason`: `{seen_reason_summary}`.",
        "",
        "---",
        "",
        "## 6. Kết luận GĐ1 & Định hướng GĐ2",
        "",
        f"1. **Phần lớn `fashion` ({f_dup_id:,} / {len(f_df):,} theo ID; {f_dup_all_id_or_sha:,} theo ID ∪ SHA-1) là ảnh ĐÃ CÓ trong `catalog.parquet`**; nguồn bổ sung thực sự mới chủ yếu là **`shopping100k` ({s_new:,} / {len(s_df):,} ảnh)**. Số ảnh đọc/giải mã PIL thành công: **{int(df_img['readable'].sum()):,} / {len(df_inv):,}** (`{len(reject_rows)}` lỗi `unreadable`).",
        "2. `shopping100k` **chưa từng đi qua bộ lọc chất lượng** (`min_short_side=224`, `Laplacian blur >= 12.0`, `banner_flag`, `pHash <= 1`). Bước GĐ2 sẽ đo toàn bộ các chỉ số chất lượng này trên toàn bộ ứng viên và xuất contact sheet để người dùng duyệt ngưỡng trước khi lọc chính thức ở GĐ3.",
    ])

    report_path = reports_p7 / "gd1_inventory.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Đã ghi báo cáo: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a GD1 inventory and audit")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_gd1(
        config_path=args.config,
        merged_dir_override=args.merged_dir,
        shard_size=args.shard_size,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
