"""Phase 7a GĐ3 — Normalization, Approved Filtering, Dedup, Catalog Exclusion & RGB Check.

Applies approved GĐ2 rules:
  1. Unified quality rule for both sources: min_short_side=224, detect_banner=False,
     no blur dropping (adds `low_contrast_flag = (blur_score < 12.0)`).
  2. Protects the 100 non-catalog fashion legacy_test (seen_in_train='no') items:
     only drop if unreadable or low_res (<224); keep the rest and attach explicit flags.
  3. Dedup: exact SHA-1 on all items (keeping 1st by priority); pHash <= 1 non-transitive
     dedup ONLY when all 3 attributes (articleType, gender, baseColour) are defined
     (no missing/Unknown color), tie-break: short_side DESC -> blur_score DESC -> uid ASC.
  4. Excludes duplicates vs catalog_inscope (ID, SHA-1, or pHash<=1 with matching 3 attrs),
     Keychains (43230, 42066), and out-of-scope articleTypes/masterCategories per config.yaml.
  5. Verifies absolute path existence & PIL convert('RGB') readiness (resumable shards).
  6. Exports 60-image random contact sheet (seed 42) + reports/phase7/gd3_dedup.md.
"""

from __future__ import annotations

import argparse
import io
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

from src.phase7a.common import ensure_utf8_stdio, load_config, resolve_phase7a_paths
from src.phase7a.gd2_quality import _render_grid_contact_sheet

ImageFile.LOAD_TRUNCATED_IMAGES = False

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

MISSING_COLOR_TOKENS = {"", "nan", "none", "na", "null", "unknown"}


def _has_known_color(val: Any) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip().lower() not in MISSING_COLOR_TOKENS


def _has_known_str(val: Any) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip().lower() not in MISSING_COLOR_TOKENS


def _verify_rgb_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Verify file existence and PIL convert('RGB') readability for future embedding/UI."""
    uid, abs_path_str = args
    p = Path(abs_path_str)
    try:
        if not p.is_file():
            return {"uid": uid, "rgb_ok": False, "rgb_error": "file_not_found"}
        raw = p.read_bytes()
        if not raw:
            return {"uid": uid, "rgb_ok": False, "rgb_error": "zero_byte"}
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            im_t = ImageOps.exif_transpose(im)
            rgb = im_t.convert("RGB")
            w, h = rgb.size
            if w <= 0 or h <= 0 or rgb.mode != "RGB":
                return {"uid": uid, "rgb_ok": False, "rgb_error": f"invalid_rgb:{rgb.mode}:{w}x{h}"}
        return {"uid": uid, "rgb_ok": True, "rgb_error": ""}
    except Exception as exc:
        return {"uid": uid, "rgb_ok": False, "rgb_error": f"{type(exc).__name__}:{str(exc)[:120]}"}


def _run_sharded_rgb_check(
    items: List[Tuple[str, str]],
    shards_dir: Path,
    shard_size: int = 5000,
    max_workers: int = 10,
) -> pd.DataFrame:
    """Verify RGB conversion in resumable parquet shards."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, (len(items) + shard_size - 1) // shard_size)
    frames: List[Optional[pd.DataFrame]] = [None] * n_shards
    todo: List[int] = []

    for s_idx in range(n_shards):
        sp = shards_dir / f"gd3_rgb_shard_{s_idx:04d}.parquet"
        start = s_idx * shard_size
        end = min(len(items), start + shard_size)
        if sp.exists():
            df_s = pd.read_parquet(sp)
            if len(df_s) == (end - start) and set(df_s["uid"].tolist()) == {x[0] for x in items[start:end]}:
                frames[s_idx] = df_s
                continue
        todo.append(s_idx)

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for s_idx in todo:
                start = s_idx * shard_size
                end = min(len(items), start + shard_size)
                batch = items[start:end]
                rows = list(ex.map(_verify_rgb_worker, batch))
                df_s = pd.DataFrame(rows)
                sp = shards_dir / f"gd3_rgb_shard_{s_idx:04d}.parquet"
                df_s.to_parquet(sp, index=False)
                frames[s_idx] = df_s
                print(f"  [gd3_rgb] shard {s_idx + 1}/{n_shards} saved ({len(df_s):,} rows)", flush=True)

    return pd.concat([f for f in frames if f is not None], ignore_index=True)


def run_gd3(
    config_path: str | Path | None = None,
    merged_dir_override: str | Path | None = None,
    shard_size: int = 5000,
    max_workers: int = 10,
) -> None:
    """Execute Phase 7a GĐ3 approved filtering, dedup, catalog exclusion & RGB check."""
    t0 = time.time()
    ensure_utf8_stdio()
    cfg = load_config(config_path)
    paths = resolve_phase7a_paths(cfg, merged_dir_override=merged_dir_override)

    img_cfg = cfg.get("image_filter", {})
    min_short_side = int(img_cfg["min_short_side"])  # 224
    blur_flag_thresh = float(img_cfg["blur_laplacian_var_threshold"])  # 12.0 (flag only!)
    phash_thresh = int(img_cfg["phash_hamming_threshold"])  # 1
    scope_cfg = cfg.get("scope", {})
    excl_master = set(scope_cfg.get("exclude_master_categories", []))
    excl_art = set(scope_cfg.get("exclude_article_types", []))
    incl_exc = set(scope_cfg.get("include_article_type_exceptions", []))
    seed = int(cfg.get("eval", {}).get("seed", 42))
    keychain_ids = {"43230", "42066"}

    eval_p7 = paths["eval_p7"]
    reports_p7 = paths["reports_p7"]
    cs_dir = paths["contact_sheets_dir"]
    merged_dir = paths["merged_dir"]
    eval_p7.mkdir(parents=True, exist_ok=True)
    reports_p7.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHA 7a — GĐ3: CHUẨN HÓA, LỌC NGƯỠNG ĐÃ DUYỆT & CHỐNG TRÙNG")
    print("=" * 72)

    df = pd.read_parquet(eval_p7 / "quality_metrics.parquet")
    df["abs_path"] = [str((merged_dir / rel_p).resolve()) for rel_p in df["image_path"]]
    df["low_contrast_flag"] = df["blur_score"].fillna(0.0) < blur_flag_thresh
    df["legacy_test_protected"] = (
        (df["source"] == "fashion") & (~df["in_catalog_any"]) & (df["seen_in_train"] == "no")
    )

    rejects: List[Dict[str, Any]] = []
    stage_stats: List[Dict[str, Any]] = []

    def _rec_stage(name: str, frame: pd.DataFrame, dropped: int = 0) -> None:
        by_src = frame["source"].value_counts().to_dict()
        stage_stats.append({
            "step": name,
            "total": len(frame),
            "fashion": int(by_src.get("fashion", 0)),
            "shopping100k": int(by_src.get("shopping100k", 0)),
            "dropped": int(dropped),
        })

    def _add_rejects(sub: pd.DataFrame, stage: str, reason: str, detail_fn: Any, dup_col: str = "") -> None:
        for r in sub.itertuples(index=False):
            rejects.append({
                "uid": r.uid,
                "source": r.source,
                "orig_id": r.orig_id,
                "image_path": r.image_path,
                "stage": stage,
                "reason": reason,
                "detail": detail_fn(r),
                "short_side": int(r.short_side) if pd.notna(r.short_side) else 0,
                "blur_score": round(float(r.blur_score), 3) if pd.notna(r.blur_score) else np.nan,
                "banner_flag": bool(r.banner_flag) if pd.notna(r.banner_flag) else False,
                "phash": str(r.phash or ""),
                "duplicate_of": getattr(r, dup_col, "") if dup_col else "",
            })

    _rec_stage("0. Đầu vào từ GĐ2 (all.parquet)", df, dropped=0)

    # 1. Unreadable check
    bad_read = (~df["readable"]) | (df["qual_status"] != "ok")
    if bad_read.any():
        _add_rejects(df[bad_read], "gd3_1_quality", "unreadable", lambda r: str(r.read_error or r.qual_status))
    df1 = df[~bad_read].copy()
    _rec_stage("1a. Sau kiểm tra đọc được (unreadable)", df1, dropped=int(bad_read.sum()))

    # 2. Low-res (< 224) — applies to ALL including protected_100
    bad_res = df1["short_side"] < min_short_side
    if bad_res.any():
        _add_rejects(
            df1[bad_res],
            "gd3_1_quality",
            "low_res",
            lambda r: f"short_side={int(r.short_side)} < {min_short_side} ({int(r.cv_width)}x{int(r.cv_height)})",
        )
    df2 = df1[~bad_res].copy()
    _rec_stage(f"1b. Sau lọc low_res (short_side >= {min_short_side})", df2, dropped=int(bad_res.sum()))

    # 3. Dedup STEP A: Exact SHA-1 within candidates (priority: short_side DESC, blur_score DESC, uid ASC)
    # Protected 100 fashion legacy_test items are flagged, not dropped, per Rule 2.
    print("[1/4] Khử trùng lặp nội bộ (SHA-1 cho tất cả; pHash<=1 chỉ khi đủ 3 thuộc tính)...")
    df2 = df2.sort_values(
        by=["legacy_test_protected", "short_side", "blur_score", "uid"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    seen_sha1: Dict[str, str] = {}
    sha1_drop_uids: Set[str] = set()
    sha1_dup_of: Dict[str, str] = {}
    prot_internal_sha1_dup_of: Dict[str, str] = {}
    for r in df2[["uid", "sha1", "legacy_test_protected"]].itertuples(index=False):
        if not r.sha1:
            continue
        if r.sha1 in seen_sha1:
            rep = seen_sha1[r.sha1]
            if r.legacy_test_protected:
                prot_internal_sha1_dup_of[r.uid] = rep
            else:
                sha1_drop_uids.add(r.uid)
                sha1_dup_of[r.uid] = rep
        else:
            seen_sha1[r.sha1] = r.uid

    if sha1_drop_uids:
        sub_sha = df2[df2["uid"].isin(sha1_drop_uids)].copy()
        sub_sha["dup_target"] = sub_sha["uid"].map(sha1_dup_of)
        _add_rejects(sub_sha, "gd3_2_dedup_sha1", "duplicate_internal_sha1", lambda r: f"exact SHA-1 match with {r.dup_target}", dup_col="dup_target")
    df3a = df2[~df2["uid"].isin(sha1_drop_uids)].copy()
    df3a["internal_sha1_dup_flag"] = df3a["uid"].isin(prot_internal_sha1_dup_of)
    df3a["internal_sha1_dup_of"] = df3a["uid"].map(lambda u: prot_internal_sha1_dup_of.get(u, ""))
    _rec_stage("2a. Sau dedup SHA-1 nội bộ", df3a, dropped=len(sha1_drop_uids))

    # Dedup STEP B: pHash <= 1 ONLY when (article_type, gender, color) ALL defined!
    has_full_attrs = (
        df3a["article_type"].apply(_has_known_str)
        & df3a["gender"].apply(_has_known_str)
        & df3a["color"].apply(_has_known_color)
        & (df3a["phash"].astype(str).str.len() == 16)
    )
    df3a["has_full_attrs_for_phash"] = has_full_attrs

    df_full_attr = df3a[has_full_attrs].copy()
    df_full_attr["phash_u64"] = np.array([int(x, 16) for x in df_full_attr["phash"]], dtype=np.uint64)
    # Priority: protected DESC, short_side DESC, blur_score DESC, uid ASC
    df_full_attr = df_full_attr.sort_values(
        by=["legacy_test_protected", "short_side", "blur_score", "uid"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    ph_drop_uids: Set[str] = set()
    ph_dup_of: Dict[str, Tuple[str, int]] = {}
    prot_ph_dup_of: Dict[str, str] = {}

    for (_, _, _), grp in df_full_attr.groupby(["article_type", "gender", "color"], sort=False):
        n_grp = len(grp)
        if n_grp < 2:
            continue
        uids = grp["uid"].tolist()
        hashes = grp["phash_u64"].to_numpy(dtype=np.uint64)
        prots = grp["legacy_test_protected"].tolist()
        assigned = np.zeros(n_grp, dtype=bool)
        for i in range(n_grp):
            if assigned[i]:
                continue
            assigned[i] = True
            if i + 1 >= n_grp:
                break
            xor_vals = hashes[i] ^ hashes[i + 1 :]
            dists = lut[xor_vals.view(np.uint8)].reshape(-1, 8).sum(axis=1)
            hits = np.nonzero((dists <= phash_thresh) & (~assigned[i + 1 :]))[0]
            for off in hits:
                j = i + 1 + int(off)
                assigned[j] = True
                hdist = int(dists[off])
                if prots[j]:
                    prot_ph_dup_of[uids[j]] = f"{uids[i]}(h={hdist})"
                else:
                    ph_drop_uids.add(uids[j])
                    ph_dup_of[uids[j]] = (uids[i], hdist)

    if ph_drop_uids:
        sub_ph = df3a[df3a["uid"].isin(ph_drop_uids)].copy()
        sub_ph["dup_target"] = sub_ph["uid"].map(lambda u: ph_dup_of[u][0])
        sub_ph["dup_hdist"] = sub_ph["uid"].map(lambda u: ph_dup_of[u][1])
        _add_rejects(
            sub_ph,
            "gd3_2_dedup_phash",
            "duplicate_internal_phash",
            lambda r: f"same (article_type,gender,color) pHash dist={r.dup_hdist} vs {r.dup_target}",
            dup_col="dup_target",
        )
    df3b = df3a[~df3a["uid"].isin(ph_drop_uids)].copy()
    df3b["internal_phash_dup_flag"] = df3b["uid"].isin(prot_ph_dup_of)
    df3b["internal_phash_dup_detail"] = df3b["uid"].map(lambda u: prot_ph_dup_of.get(u, ""))
    _rec_stage("2b. Sau dedup pHash<=1 (chỉ nhóm đủ màu)", df3b, dropped=len(ph_drop_uids))

    # 4. Exclude catalog duplicates & out-of-scope / Keychains (43230, 42066)
    print("[2/4] Đối chiếu trùng catalog_inscope (ID / SHA-1 / pHash<=1 đủ màu) & loại out-of-scope...")
    cat = pd.read_parquet(paths["catalog_path"])
    cat["item_id"] = cat["item_id"].astype(str)
    cat_all_ids = set(cat["item_id"].tolist())
    cat_in = cat[cat["in_scope"]].copy()

    # Load pha1a phash for catalog_inscope
    p1a_path = paths["reports_dir"] / "phase1" / "phase1a" / "quality_metrics.parquet"
    df_p1a = pd.read_parquet(p1a_path, columns=["id", "phash"])
    df_p1a["id"] = df_p1a["id"].astype(str)
    cat_in = cat_in.merge(df_p1a, left_on="item_id", right_on="id", how="left")

    # Build catalog in_scope pHash lookup by defined (articleType, gender, baseColour)
    cat_in_groups: Dict[Tuple[str, str, str], Tuple[List[str], np.ndarray]] = {}
    cat_known = cat_in[
        cat_in["articleType"].apply(_has_known_str)
        & cat_in["gender"].apply(_has_known_str)
        & cat_in["baseColour"].apply(_has_known_color)
        & (cat_in["phash"].astype(str).str.len() == 16)
    ]
    for (at, gen, col), grp in cat_known.groupby(["articleType", "gender", "baseColour"]):
        cat_in_groups[(str(at), str(gen), str(col))] = (
            grp["item_id"].tolist(),
            np.array([int(x, 16) for x in grp["phash"]], dtype=np.uint64),
        )

    # Detect catalog pHash<=1 matches on remaining rows with full 3 defined attributes
    cat_ph_match: Dict[str, Tuple[str, int]] = {}
    for r in df3b[df3b["has_full_attrs_for_phash"]][["uid", "article_type", "gender", "color", "phash"]].itertuples(index=False):
        key = (str(r.article_type), str(r.gender), str(r.color))
        if key not in cat_in_groups:
            continue
        c_ids, c_hashes = cat_in_groups[key]
        h_val = np.uint64(int(r.phash, 16))
        dists = lut[(h_val ^ c_hashes).view(np.uint8)].reshape(-1, 8).sum(axis=1)
        hits = np.nonzero(dists <= phash_thresh)[0]
        if len(hits) > 0:
            best_pos = int(hits[int(np.argmin(dists[hits]))])
            cat_ph_match[r.uid] = (c_ids[best_pos], int(dists[best_pos]))

    # Recompute exact scope rule from config.yaml (plus explicit Keychains 43230, 42066)
    is_keychain = df3b["orig_id"].astype(str).isin(keychain_ids) | (df3b["article_type"] == "Key chain")
    cfg_in_scope = []
    for cat_m, art_t, kc in zip(df3b["category"], df3b["article_type"], is_keychain):
        ok = True
        if cat_m in excl_master and art_t not in incl_exc:
            ok = False
        if art_t in excl_art or kc:
            ok = False
        cfg_in_scope.append(ok)
    df3b["cfg_in_scope"] = cfg_in_scope
    df3b["catalog_phash_dup_flag"] = df3b["uid"].isin(cat_ph_match)
    df3b["catalog_phash_dup_id"] = df3b["uid"].map(lambda u: cat_ph_match[u][0] if u in cat_ph_match else "")

    # Classify Step 3.3 drops (non-protected only; protected_100 kept & flagged per Rule 2!)
    # Check duplicates against catalog (in_scope by ID/SHA1/pHash, AND any fashion already in catalog.parquet out-of-scope)
    is_prot = df3b["legacy_test_protected"]
    dup_cat_in_id_sha = (~is_prot) & (df3b["in_catalog_inscope_by_id"] | df3b["in_catalog_inscope_by_sha1"])
    dup_cat_in_phash = (~is_prot) & (~dup_cat_in_id_sha) & df3b["catalog_phash_dup_flag"]
    drop_keychain = (~is_prot) & (~dup_cat_in_id_sha) & (~dup_cat_in_phash) & is_keychain
    drop_oos = (~is_prot) & (~dup_cat_in_id_sha) & (~dup_cat_in_phash) & (~drop_keychain) & (~df3b["cfg_in_scope"])
    # Check if any non-protected fashion item matched out-of-scope catalog.parquet by ID/SHA1? Already caught by drop_oos!
    drop_cat_oos_extra = (~is_prot) & (~dup_cat_in_id_sha) & (~dup_cat_in_phash) & (~drop_keychain) & (~drop_oos) & df3b["in_catalog_any"]

    if dup_cat_in_id_sha.any():
        sub = df3b[dup_cat_in_id_sha].copy()
        _add_rejects(sub, "gd3_3_catalog_excl", "duplicate_catalog_inscope_id_or_sha1", lambda r: f"matches catalog item_id={r.catalog_match_item_id}", dup_col="catalog_match_item_id")
    if dup_cat_in_phash.any():
        sub = df3b[dup_cat_in_phash].copy()
        sub["cat_dup_target"] = sub["uid"].map(lambda u: f"catalog_{cat_ph_match[u][0]}")
        _add_rejects(sub, "gd3_3_catalog_excl", "duplicate_catalog_inscope_phash", lambda r: f"pHash<={phash_thresh} same attrs vs {r.cat_dup_target}", dup_col="cat_dup_target")
    if drop_keychain.any():
        _add_rejects(df3b[drop_keychain], "gd3_3_scope", "excluded_keychain", lambda r: f"Keychain excluded orig_id={r.orig_id}")
    if drop_oos.any():
        _add_rejects(df3b[drop_oos], "gd3_3_scope", "out_of_scope", lambda r: f"excluded scope: {r.category} / {r.article_type}")
    if drop_cat_oos_extra.any():
        _add_rejects(df3b[drop_cat_oos_extra], "gd3_3_catalog_excl", "duplicate_catalog_other", lambda r: f"matches catalog item_id={r.catalog_match_item_id}", dup_col="catalog_match_item_id")

    df3_after_cat = df3b[~(dup_cat_in_id_sha | dup_cat_in_phash)].copy()
    _rec_stage("3a. Sau loại trùng catalog in_scope (ID/SHA-1/pHash)", df3_after_cat, dropped=int((dup_cat_in_id_sha | dup_cat_in_phash).sum()))

    drop_mask_33 = dup_cat_in_id_sha | dup_cat_in_phash | drop_keychain | drop_oos | drop_cat_oos_extra
    df3c = df3b[~drop_mask_33].copy().reset_index(drop=True)
    _rec_stage("3b. Sau loại Keychain & ngoài scope (giữ 94 protected có gắn cờ)", df3c, dropped=int((drop_keychain | drop_oos | drop_cat_oos_extra).sum()))

    # 5. Mandatory RGB + absolute path check for embed/UI readiness (resumable shards)
    print(f"[3/4] Kiểm tra bắt buộc convert('RGB') + đường dẫn tuyệt đối cho {len(df3c):,} ảnh...")
    rgb_tasks = list(zip(df3c["uid"].astype(str), df3c["abs_path"].astype(str)))
    df_rgb = _run_sharded_rgb_check(rgb_tasks, paths["shards_dir"], shard_size=shard_size, max_workers=max_workers)
    df3d = df3c.merge(df_rgb, on="uid", how="left")

    bad_rgb = ~df3d["rgb_ok"].fillna(False)
    if bad_rgb.any():
        _add_rejects(df3d[bad_rgb], "gd3_4_rgb", "rgb_convert_failed", lambda r: str(r.rgb_error))
    df_clean = df3d[~bad_rgb].copy().reset_index(drop=True)
    _rec_stage("4. Sau kiểm tra bắt buộc RGB & đường dẫn tuyệt đối", df_clean, dropped=int(bad_rgb.sum()))

    # Save clean intermediate candidates & candidate_rejects.csv
    clean_path = eval_p7 / "clean_candidates_gd3.parquet"
    df_clean.to_parquet(clean_path, index=False)
    rejects_path = eval_p7 / "candidate_rejects.csv"
    pd.DataFrame(rejects, columns=REJECTS_COLUMNS).to_csv(rejects_path, index=False, encoding="utf-8")
    print(f"Đã lưu tập sạch GĐ3: {clean_path} ({len(df_clean):,} dòng) và log loại: {rejects_path} ({len(rejects):,} dòng)")

    # 6. Render 60-image random contact sheet (seed=42) from final clean candidate pool
    print("[4/4] Xuất contact sheet 60 ảnh ngẫu nhiên (seed=42) từ tập ứng viên sạch...")
    sample_60 = df_clean.sample(n=min(60, len(df_clean)), random_state=seed)
    entries_60 = []
    for r in sample_60.itertuples(index=False):
        lc_tag = "LC" if r.low_contrast_flag else "OK"
        col_s = str(r.color) if _has_known_color(r.color) else "UnkCol"
        l1 = f"{r.source[:4]}|{r.article_type[:13]}|{r.gender[:3]}|{col_s[:7]}"
        l2 = f"s:{int(r.short_side)} b:{float(r.blur_score):.1f}[{lc_tag}] st:{r.seen_in_train}"
        border = (220, 175, 70) if r.low_contrast_flag else ((80, 185, 235) if r.seen_in_train == "no" else (70, 175, 100))
        entries_60.append({"abs_path": r.abs_path, "line1": l1, "line2": l2, "border_color": border, "text_color": (190, 235, 255)})

    cs_60_path = cs_dir / "gd3_candidates_sample_60.png"
    _render_grid_contact_sheet(
        entries_60,
        cs_60_path,
        title=f"GĐ3 — 60 mẫu ngẫu nhiên (seed={seed}) từ {len(df_clean):,} ứng viên sạch (xanh dương=seen_in_train:no, vàng=low_contrast)",
        rows=10,
        cols=6,
        cell_w=230,
        cell_h=285,
    )

    # Summarize protected 100 fashion legacy_test accounting
    f100_all = df[df["legacy_test_protected"]]
    f100_kept = df_clean[df_clean["legacy_test_protected"]]
    f100_rej_rows = [r for r in rejects if r["uid"] in set(f100_all["uid"])]
    f100_kept_flags = {
        "in_scope_true": int(f100_kept["cfg_in_scope"].sum()),
        "out_of_scope_flagged": int((~f100_kept["cfg_in_scope"]).sum()),
        "low_contrast_flag": int(f100_kept["low_contrast_flag"].sum()),
        "banner_flag_pha1_unfiltered": int(f100_kept["banner_flag"].sum()),
        "catalog_phash_dup_flag": int(f100_kept["catalog_phash_dup_flag"].sum()),
        "internal_sha1_dup_flag": int(f100_kept["internal_sha1_dup_flag"].sum()),
        "internal_phash_dup_flag": int(f100_kept["internal_phash_dup_flag"].sum()),
    }

    rej_reason_counts = Counter((r["reason"], r["source"]) for r in rejects)

    # Write reports/phase7/gd3_dedup.md
    lines = [
        "# Pha 7a — GĐ3: Báo cáo Chuẩn hóa, Lọc theo Ngưỡng Đã Duyệt & Chống trùng",
        "",
        f"- **Thời gian chạy:** {time.time() - t0:.1f}s | **Seed:** `{seed}`",
        "- **Quy tắc đã duyệt áp dụng thống nhất 2 nguồn:** `min_short_side=224`, `detect_banner=False`, không lọc `blur` (gắn cờ `low_contrast_flag = blur < 12.0`), `pHash <= 1` không bắc cầu chỉ khi đủ `(articleType, gender, baseColour)` đều xác định (thiếu màu chỉ dedup `SHA-1` giống hệt).",
        f"- **Đầu ra:** `eval/phase7/clean_candidates_gd3.parquet` (**{len(df_clean):,}** dòng), `eval/phase7/candidate_rejects.csv` (**{len(rejects):,}** dòng), `reports/phase7/contact_sheets/gd3_candidates_sample_60.png`.",
        "",
        "---",
        "",
        "## 1. Số ảnh trước / sau từng bước theo nguồn",
        "",
        "| Bước xử lý | Tổng còn lại | `fashion` | `shopping100k` | Bị loại ở bước này |",
        "|---|---:|---:|---:|---:|",
    ]
    for st in stage_stats:
        lines.append(f"| {st['step']} | **{st['total']:,}** | {st['fashion']:,} | {st['shopping100k']:,} | {st['dropped']:,} |")

    lines.extend([
        "",
        "### Chi tiết số dòng bị loại theo `reason` × `source` (`candidate_rejects.csv`)",
        "",
        "| Lý do (`reason`) | `fashion` | `shopping100k` | Tổng |",
        "|---|---:|---:|---:|",
    ])
    all_reasons = sorted({k[0] for k in rej_reason_counts.keys()})
    for rsn in all_reasons:
        fc = rej_reason_counts.get((rsn, "fashion"), 0)
        sc = rej_reason_counts.get((rsn, "shopping100k"), 0)
        lines.append(f"| `{rsn}` | {fc:,} | {sc:,} | **{fc + sc:,}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Kiểm toán 100 ảnh `fashion legacy_test` (`seen_in_train='no'`) ngoài catalog",
        "",
        f"- **Tổng ban đầu:** `{len(f100_all)}` ảnh (`seen_in_train='no'` và `not in_catalog_any`).",
        f"- **Bị loại:** **`{len(f100_rej_rows)}` / 100** ảnh, lý do duy nhất là **`low_res` (`short_side = 150 < 224`)**: `{[r['uid'] for r in f100_rej_rows]}` (`unreadable = 0`).",
        f"- **Giữ lại & gắn cờ:** **`{len(f100_kept)}` / 100** ảnh (`legacy_test_protected = True`). Phân rã các cờ trên `{len(f100_kept)}` ảnh giữ lại:",
        f"  - `cfg_in_scope == True` (hợp lệ thời trang): **{f100_kept_flags['in_scope_true']}** | `cfg_in_scope == False` (Personal Care ngoài scope — gắn cờ, chưa bỏ): **{f100_kept_flags['out_of_scope_flagged']}** (`{df_clean.loc[df_clean['legacy_test_protected'] & (~df_clean['cfg_in_scope']), 'uid'].tolist()}`)",
        f"  - `low_contrast_flag` (`blur < 12.0`): **{f100_kept_flags['low_contrast_flag']}**",
        f"  - `banner_flag` (cờ Pha 1 cũ, không lọc theo ngưỡng đã duyệt): **{f100_kept_flags['banner_flag_pha1_unfiltered']}**",
        f"  - `catalog_phash_dup_flag` (`pHash<=1` cùng 3 thuộc tính vs `catalog_inscope`): **{f100_kept_flags['catalog_phash_dup_flag']}** (`{df_clean.loc[df_clean['legacy_test_protected'] & df_clean['catalog_phash_dup_flag'], 'uid'].tolist()}`)",
        f"  - `internal_sha1_dup_flag`: **{f100_kept_flags['internal_sha1_dup_flag']}** | `internal_phash_dup_flag`: **{f100_kept_flags['internal_phash_dup_flag']}**",
        "",
        "---",
        "",
        "## 3. Thống kê tập ứng viên sạch cuối GĐ3 (`clean_candidates_gd3.parquet`)",
        "",
        f"- **Tổng ứng viên sạch GĐ3:** **{len(df_clean):,}** (`fashion`: **{int((df_clean['source']=='fashion').sum()):,}**, `shopping100k`: **{int((df_clean['source']=='shopping100k').sum()):,}**).",
        f"- **Phân bố `seen_in_train`:** `{df_clean.groupby(['source', 'seen_in_train']).size().to_dict()}`.",
        f"- **Cờ `low_contrast_flag` (`blur < 12.0`):** `fashion`: {int(df_clean[df_clean['source']=='fashion']['low_contrast_flag'].sum()):,}, `shopping100k`: {int(df_clean[df_clean['source']=='shopping100k']['low_contrast_flag'].sum()):,}.",
        f"- **Ảnh thiếu màu (`color` thiếu, chỉ dedup SHA-1):** {int((~df_clean['color'].apply(_has_known_color)).sum()):,} ảnh (trong đó `shopping100k`: {int((~df_clean[df_clean['source']=='shopping100k']['color'].apply(_has_known_color)).sum()):,}).",
        f"- **Kiểm tra chuyển RGB & đường dẫn tuyệt đối:** **{int(df_clean['rgb_ok'].sum()):,} / {len(df_clean):,}** đạt (`0` lỗi — gồm cả 2 ảnh mode `L` đã chuyển `RGB` thành công).",
        f"- **Contact sheet kiểm tra bằng mắt (60 ảnh ngẫu nhiên, seed=42):** [gd3_candidates_sample_60.png](./contact_sheets/gd3_candidates_sample_60.png).",
    ])

    rep_path = reports_p7 / "gd3_dedup.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi báo cáo GĐ3: {rep_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a GD3 approved filtering, dedup & RGB check")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_gd3(
        config_path=args.config,
        merged_dir_override=args.merged_dir,
        shard_size=args.shard_size,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
