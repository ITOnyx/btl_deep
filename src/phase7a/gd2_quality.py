"""Phase 7a GĐ2 — Image Quality Assessment, Contact Sheets, and Threshold Review.

Computes short_side, Laplacian blur variance, Pha-1 banner heuristic (including sub-flags),
and 64-bit pHash on all readable images in eval/phase7/inventory.parquet using config.yaml
thresholds, renders contact sheets for human review, and stops BEFORE finalizing thresholds.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.phase7a.common import ensure_utf8_stdio, load_config, resolve_phase7a_paths
from src.phase7a.quality_workers import compute_quality_metrics_worker, init_quality_worker


def _load_fonts() -> Tuple[Any, Any]:
    """Load Unicode-capable Windows fonts with safe fallback."""
    for f_reg, f_bold in [
        ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]:
        if Path(f_reg).exists():
            try:
                font = ImageFont.truetype(f_reg, 12)
                tfont = ImageFont.truetype(f_bold if Path(f_bold).exists() else f_reg, 18)
                return font, tfont
            except Exception:
                pass
    df = ImageFont.load_default()
    return df, df


def _render_grid_contact_sheet(
    entries: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    rows: int = 6,
    cols: int = 6,
    cell_w: int = 230,
    cell_h: int = 285,
) -> Optional[Path]:
    """Render a rows x cols contact sheet image with 2-line captions."""
    if not entries:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font = _load_fonts()
    header_h = 46
    label_h = 40
    margin = 8
    n_use = min(len(entries), rows * cols)
    eff_rows = max(1, (n_use + cols - 1) // cols)

    sheet_w = cols * cell_w + (cols + 1) * margin
    sheet_h = header_h + eff_rows * cell_h + (eff_rows + 1) * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(28, 28, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin + 6, 12), title, fill=(242, 242, 246), font=title_font)

    box_h = cell_h - label_h
    for idx in range(n_use):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_w + margin)
        y = header_h + margin + r * (cell_h + margin)
        ent = entries[idx]
        border = ent.get("border_color", (78, 78, 92))
        draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(44, 44, 52), outline=border, width=2)

        p = Path(ent["abs_path"])
        try:
            with Image.open(p) as im:
                thumb = im.convert("RGB")
                thumb.thumbnail((cell_w - 6, box_h - 6), Image.Resampling.LANCZOS)
                px = x + (cell_w - thumb.width) // 2
                py = y + (box_h - thumb.height) // 2
                sheet.paste(thumb, (px, py))
        except Exception:
            draw.text((x + 12, y + box_h // 2), "ERR", fill=(255, 100, 100), font=font)

        l1 = str(ent.get("line1", ""))[:36]
        l2 = str(ent.get("line2", ""))[:36]
        draw.text((x + 5, y + box_h + 2), l1, fill=(225, 225, 232), font=font)
        draw.text((x + 5, y + box_h + 19), l2, fill=ent.get("text_color", (180, 220, 255)), font=font)

    sheet.save(out_path, format="PNG")
    return out_path


def _render_pair_contact_sheet(
    pairs: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    max_pairs: int = 12,
) -> Optional[Path]:
    """Render side-by-side [Representative | Duplicate] pairs (2 pairs per row)."""
    if not pairs:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font = _load_fonts()
    use_pairs = pairs[:max_pairs]
    pair_cols = 2
    pair_rows = max(1, (len(use_pairs) + pair_cols - 1) // pair_cols)
    img_w, img_h = 190, 210
    label_h = 44
    gap = 6
    margin = 10
    header_h = 46
    box_w = img_w * 2 + gap + 12
    box_h = img_h + label_h + 10

    sheet_w = pair_cols * box_w + (pair_cols + 1) * margin
    sheet_h = header_h + pair_rows * box_h + (pair_rows + 1) * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(28, 28, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin + 4, 12), title, fill=(242, 242, 246), font=title_font)

    for idx, pr in enumerate(use_pairs):
        r = idx // pair_cols
        c = idx % pair_cols
        bx = margin + c * (box_w + margin)
        by = header_h + margin + r * (box_h + margin)
        draw.rectangle([bx, by, bx + box_w, by + box_h], fill=(42, 42, 50), outline=(95, 95, 115), width=2)

        for side_idx, p_key in enumerate(["rep_path", "dup_path"]):
            ix = bx + 6 + side_idx * (img_w + gap)
            iy = by + 6
            try:
                with Image.open(Path(pr[p_key])) as im:
                    thumb = im.convert("RGB")
                    thumb.thumbnail((img_w, img_h), Image.Resampling.LANCZOS)
                    sheet.paste(thumb, (ix + (img_w - thumb.width) // 2, iy + (img_h - thumb.height) // 2))
            except Exception:
                draw.text((ix + 20, iy + img_h // 2), "ERR", fill=(255, 100, 100), font=font)

        draw.text((bx + 8, by + img_h + 8), pr["cap1"][:58], fill=(170, 240, 180), font=font)
        draw.text((bx + 8, by + img_h + 25), pr["cap2"][:58], fill=(255, 200, 160), font=font)

    sheet.save(out_path, format="PNG")
    return out_path


def _run_sharded_quality(
    items: List[Tuple[str, str]],
    shards_dir: Path,
    shard_size: int = 2000,
    max_workers: int = 8,
    use_process_pool: bool = True,
) -> pd.DataFrame:
    """Compute quality metrics in resumable parquet shards using one persistent pool."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, (len(items) + shard_size - 1) // shard_size)
    frames: List[Optional[pd.DataFrame]] = [None] * n_shards
    todo: List[int] = []

    for s_idx in range(n_shards):
        sp = shards_dir / f"gd2_qual_shard_{s_idx:04d}.parquet"
        start = s_idx * shard_size
        end = min(len(items), start + shard_size)
        if sp.exists():
            df_s = pd.read_parquet(sp)
            if len(df_s) == (end - start):
                frames[s_idx] = df_s
                continue
        todo.append(s_idx)

    if todo:
        if use_process_pool:
            with ProcessPoolExecutor(max_workers=max_workers, initializer=init_quality_worker) as ex:
                for s_idx in todo:
                    start = s_idx * shard_size
                    end = min(len(items), start + shard_size)
                    batch = items[start:end]
                    rows = list(ex.map(compute_quality_metrics_worker, batch, chunksize=32))
                    df_s = pd.DataFrame(rows)
                    sp = shards_dir / f"gd2_qual_shard_{s_idx:04d}.parquet"
                    df_s.to_parquet(sp, index=False)
                    frames[s_idx] = df_s
                    print(f"  [gd2_qual] shard {s_idx + 1}/{n_shards} saved ({len(df_s):,} rows)", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for s_idx in todo:
                    start = s_idx * shard_size
                    end = min(len(items), start + shard_size)
                    batch = items[start:end]
                    rows = list(ex.map(compute_quality_metrics_worker, batch))
                    df_s = pd.DataFrame(rows)
                    sp = shards_dir / f"gd2_qual_shard_{s_idx:04d}.parquet"
                    df_s.to_parquet(sp, index=False)
                    frames[s_idx] = df_s
                    print(f"  [gd2_qual] shard {s_idx + 1}/{n_shards} saved ({len(df_s):,} rows)", flush=True)

    return pd.concat([f for f in frames if f is not None], ignore_index=True)


def _find_non_transitive_phash_dups(
    df_pool: pd.DataFrame,
    match_cols: List[str],
    phash_thresh: int,
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """Dry-run non-transitive dedup (priority: short_side DESC, blur_score DESC, uid ASC)."""
    col_map = {"articleType": "article_type", "gender": "gender", "baseColour": "color"}
    mapped_cols = [col_map.get(c, c) for c in match_cols]

    sub = df_pool[df_pool["phash"].astype(str).str.len() == 16].copy()
    for mc in mapped_cols:
        sub[mc] = sub[mc].fillna("Unknown").astype(str)
    sub["phash_u64"] = np.array([int(x, 16) for x in sub["phash"]], dtype=np.uint64)
    sub = sub.sort_values(
        by=["short_side", "blur_score", "uid"], ascending=[False, False, True]
    ).reset_index(drop=True)

    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    dup_uids: Set[str] = set()
    pair_records: List[Dict[str, Any]] = []

    for key_vals, grp in sub.groupby(mapped_cols, sort=False):
        n_grp = len(grp)
        if n_grp < 2:
            continue
        uids = grp["uid"].tolist()
        hashes = grp["phash_u64"].to_numpy(dtype=np.uint64)
        srcs = grp["source"].tolist()
        sides = grp["short_side"].tolist()
        blurs = grp["blur_score"].tolist()
        paths = grp["abs_path"].tolist()

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
                dup_uids.add(uids[j])
                pair_records.append(
                    {
                        "rep_uid": uids[i],
                        "dup_uid": uids[j],
                        "rep_source": srcs[i],
                        "dup_source": srcs[j],
                        "rep_path": paths[i],
                        "dup_path": paths[j],
                        "hdist": int(dists[off]),
                        "attrs": "/".join(str(x) for x in (key_vals if isinstance(key_vals, tuple) else (key_vals,))),
                        "rep_side": int(sides[i]),
                        "dup_side": int(sides[j]),
                        "rep_blur": round(float(blurs[i]), 1),
                        "dup_blur": round(float(blurs[j]), 1),
                    }
                )
    return dup_uids, pair_records


def run_gd2(
    config_path: str | Path | None = None,
    merged_dir_override: str | Path | None = None,
    shard_size: int = 2000,
    max_workers: int = 8,
    use_process_pool: bool = True,
) -> None:
    """Compute quality metrics, render sample contact sheets, and output GĐ2 review table."""
    t0 = time.time()
    ensure_utf8_stdio()
    cfg = load_config(config_path)
    paths = resolve_phase7a_paths(cfg, merged_dir_override=merged_dir_override)

    img_cfg = cfg.get("image_filter", {})
    min_short_side = int(img_cfg["min_short_side"])
    blur_thresh = float(img_cfg["blur_laplacian_var_threshold"])
    phash_thresh = int(img_cfg["phash_hamming_threshold"])
    detect_banner = bool(img_cfg["detect_banner"])
    dedup_attrs = list(img_cfg["dedup_match_attributes"])
    seed = int(cfg.get("eval", {}).get("seed", 42))

    eval_p7 = paths["eval_p7"]
    reports_p7 = paths["reports_p7"]
    cs_dir = paths["contact_sheets_dir"]
    merged_dir = paths["merged_dir"]
    cs_dir.mkdir(parents=True, exist_ok=True)
    reports_p7.mkdir(parents=True, exist_ok=True)

    inv_path = eval_p7 / "inventory.parquet"
    df_inv = pd.read_parquet(inv_path)
    df_inv["abs_path"] = [str(merged_dir / rel_p) for rel_p in df_inv["image_path"]]

    print("=" * 72)
    print("PHA 7a — GĐ2: ĐO CHẤT LƯỢNG ẢNH & XUẤT CONTACT SHEET DUYỆT NGƯỠNG")
    print("=" * 72)
    print(f"Ngưỡng config.yaml: min_short_side={min_short_side}, blur>={blur_thresh}, detect_banner={detect_banner}, pHash<={phash_thresh}")

    readable_df = df_inv[df_inv["readable"]].copy()
    tasks = list(zip(readable_df["uid"].astype(str), readable_df["abs_path"].astype(str)))
    print(f"[1/4] Tính short_side, Laplacian blur, banner (thuật toán Pha 1), pHash cho {len(tasks):,} ảnh...")
    df_q = _run_sharded_quality(
        tasks,
        paths["shards_dir"],
        shard_size=shard_size,
        max_workers=max_workers,
        use_process_pool=use_process_pool,
    )

    # Rename source phash before merging freshly computed GĐ2 quality metrics
    if "phash" in df_inv.columns:
        df_inv = df_inv.rename(columns={"phash": "src_phash"})
    qm_df = df_inv.merge(df_q, on="uid", how="left")
    qm_path = eval_p7 / "quality_metrics.parquet"
    qm_df.to_parquet(qm_path, index=False)
    print(f"Đã lưu metrics chất lượng: {qm_path} ({len(qm_df):,} dòng)")

    # 2. Evaluate config.yaml filter cascade per source (dry-run)
    print("[2/4] Mô phỏng phân tầng lọc theo ngưỡng config.yaml hiện hành...")
    valid = qm_df[qm_df["readable"] & (qm_df["qual_status"] == "ok")].copy()
    valid["fail_low_res"] = valid["short_side"] < min_short_side
    valid["fail_blurry"] = (~valid["fail_low_res"]) & (valid["blur_score"] < blur_thresh)
    valid["fail_banner"] = (~valid["fail_low_res"]) & (~valid["fail_blurry"]) & (valid["banner_flag"] if detect_banner else False)
    valid["pass_quality_cfg"] = (~valid["fail_low_res"]) & (~valid["fail_blurry"]) & (~valid["fail_banner"])

    clean_pool = valid[valid["pass_quality_cfg"]].copy()
    dup_uids, dup_pairs = _find_non_transitive_phash_dups(clean_pool, dedup_attrs, phash_thresh)
    valid["fail_dup"] = valid["uid"].isin(dup_uids)
    valid["keep_cfg"] = valid["pass_quality_cfg"] & (~valid["fail_dup"])

    # Compute raw (non-cascaded) flag rates too for transparent diagnosis
    valid["raw_low_res"] = valid["short_side"] < min_short_side
    valid["raw_blurry"] = valid["blur_score"] < blur_thresh
    valid["raw_banner"] = valid["banner_flag"]

    # Subsets: fashion (all), fashion (new vs catalog), shopping100k (all), shopping100k (in_scope)
    summary_rows = []
    for label, mask in [
        ("fashion (tất cả)", valid["source"] == "fashion"),
        ("fashion (chưa có trong catalog)", (valid["source"] == "fashion") & (~valid["in_catalog_any"])),
        ("shopping100k (tất cả)", valid["source"] == "shopping100k"),
        ("shopping100k (new & in_scope)", (valid["source"] == "shopping100k") & (~valid["in_catalog_by_sha1"]) & valid["in_scope"]),
        ("Tổng merged readable", pd.Series(True, index=valid.index)),
    ]:
        sub = valid[mask]
        n = len(sub)
        n_lr = int(sub["fail_low_res"].sum())
        n_bl = int(sub["fail_blurry"].sum())
        n_bn = int(sub["fail_banner"].sum())
        n_dp = int(sub["fail_dup"].sum())
        n_kp = int(sub["keep_cfg"].sum())
        rej = n - n_kp
        summary_rows.append({
            "subset": label,
            "total": n,
            "low_res": n_lr,
            "blurry": n_bl,
            "banner": n_bn,
            "duplicate": n_dp,
            "rejected_total": rej,
            "rejected_pct": round(rej / max(1, n) * 100.0, 2),
            "kept": n_kp,
            "kept_pct": round(n_kp / max(1, n) * 100.0, 2),
            "raw_low_res": int(sub["raw_low_res"].sum()),
            "raw_blurry": int(sub["raw_blurry"].sum()),
            "raw_banner": int(sub["raw_banner"].sum()),
        })

    # Percentiles & banner sub-flag diagnostics by source
    src_diag: Dict[str, Any] = {}
    for src_name in ["fashion", "shopping100k"]:
        sub = valid[valid["source"] == src_name]
        ss = sub["short_side"].to_numpy()
        bs = sub["blur_score"].to_numpy()
        src_diag[src_name] = {
            "n": len(sub),
            "short_side_pcts": {k: float(np.percentile(ss, p)) for k, p in [("min", 0), ("p1", 1), ("p5", 5), ("p10", 10), ("p25", 25), ("p50", 50), ("p95", 95), ("max", 100)]},
            "short_side_buckets": {
                "lt_180": int((ss < 180).sum()),
                "180_223": int(((ss >= 180) & (ss < 224)).sum()),
                "224_255": int(((ss >= 224) & (ss < 256)).sum()),
                "ge_256": int((ss >= 256).sum()),
            },
            "blur_pcts": {k: round(float(np.percentile(bs, p)), 2) for k, p in [("min", 0), ("p1", 1), ("p5", 5), ("p10", 10), ("p25", 25), ("p50", 50), ("p95", 95)]},
            "banner_subflags": {
                "is_top_solid": int(sub["is_top_solid"].sum()),
                "is_bot_solid": int(sub["is_bot_solid"].sum()),
                "is_top_text": int(sub["is_top_text"].sum()),
                "is_side_frame": int(sub["is_side_frame"].sum()),
                "any_banner": int(sub["banner_flag"].sum()),
            },
        }

    # Sensitivity sweeps for shopping100k (to prepare clear options for user decision)
    s_sub = valid[valid["source"] == "shopping100k"]
    sweeps = []
    for ss_thr, bl_thr, bn_mode in [
        (224, 12.0, "pha1_all"),
        (180, 12.0, "pha1_all"),
        (180, 12.0, "text_only (bo top/bot solid)"),
        (160, 8.0, "text_only (bo top/bot solid)"),
        (224, 12.0, "off (detect_banner=False)"),
        (150, 10.0, "off (detect_banner=False)"),
    ]:
        ok_ss = s_sub["short_side"] >= ss_thr
        ok_bl = s_sub["blur_score"] >= bl_thr
        if bn_mode == "pha1_all":
            ok_bn = ~s_sub["banner_flag"]
        elif bn_mode.startswith("text_only"):
            ok_bn = ~(s_sub["is_top_text"] | s_sub["is_side_frame"])
        else:
            ok_bn = pd.Series(True, index=s_sub.index)
        kept_n = int((ok_ss & ok_bl & ok_bn).sum())
        sweeps.append({
            "min_short_side": ss_thr,
            "blur_thresh": bl_thr,
            "banner_rule": bn_mode,
            "kept": kept_n,
            "dropped": len(s_sub) - kept_n,
            "kept_pct": round(kept_n / max(1, len(s_sub)) * 100.0, 2),
        })

    # 3. Generate Contact Sheets (each group x each source)
    print("[3/4] Tạo contact sheets mẫu theo từng nhóm x nguồn...")
    rng = np.random.default_rng(seed)
    rendered_sheets: List[str] = []

    def _to_entries(df_slice: pd.DataFrame, label_fn: Any, border=(78, 78, 92), tcol=(180, 220, 255)) -> List[Dict[str, Any]]:
        out = []
        for r in df_slice.itertuples(index=False):
            l1, l2 = label_fn(r)
            out.append({"abs_path": r.abs_path, "line1": l1, "line2": l2, "border_color": border, "text_color": tcol})
        return out

    for src_name in ["fashion", "shopping100k"]:
        df_s = valid[valid["source"] == src_name].copy()

        # A. Clear Pass
        pass_pool = df_s[df_s["keep_cfg"] & (df_s["blur_score"] >= max(25.0, blur_thresh * 2))]
        if len(pass_pool) == 0:
            pass_pool = df_s[df_s["keep_cfg"]]
        pass_samp = pass_pool.sample(n=min(36, len(pass_pool)), random_state=seed)
        p_out = _render_grid_contact_sheet(
            _to_entries(pass_samp, lambda r: (f"{r.uid[-22:]}", f"side:{r.short_side} blur:{r.blur_score:.1f}"), border=(70, 170, 95), tcol=(175, 245, 190)),
            cs_dir / f"{src_name}_pass_clear.png",
            f"[{src_name}] Đạt rõ ({len(pass_pool):,}/{len(df_s):,}) — side>={min_short_side}, blur>={blur_thresh}, no banner",
        )
        if p_out:
            rendered_sheets.append(p_out.name)

        # B. Low-res: rejected (<224) + borderline (closest below + just above 224)
        lr_rej = df_s[df_s["short_side"] < min_short_side]
        if len(lr_rej) > 0:
            s_low = lr_rej.sample(n=min(18, len(lr_rej)), random_state=seed)
            s_top = lr_rej.sort_values("short_side", ascending=False).head(18)
            lr_samp = pd.concat([s_low, s_top]).drop_duplicates("uid").head(36)
            p_out = _render_grid_contact_sheet(
                _to_entries(lr_samp, lambda r: (f"{r.uid[-22:]}", f"side:{r.short_side} ({r.cv_width}x{r.cv_height})"), border=(200, 80, 80), tcol=(255, 165, 165)),
                cs_dir / f"{src_name}_low_res_rejected.png",
                f"[{src_name}] Bị loại low_res (<{min_short_side}px): {len(lr_rej):,} ảnh",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

        under_b = df_s[(df_s["short_side"] >= max(100, min_short_side - 50)) & (df_s["short_side"] < min_short_side)].sort_values("short_side", ascending=False).head(18)
        over_b = df_s[(df_s["short_side"] >= min_short_side) & (df_s["short_side"] <= min_short_side + 45)].sort_values("short_side", ascending=True).head(18)
        lr_bord = pd.concat([under_b, over_b])
        if len(lr_bord) > 0:
            p_out = _render_grid_contact_sheet(
                _to_entries(lr_bord, lambda r: (f"{r.uid[-22:]}", f"side:{r.short_side} {'PASS' if r.short_side >= min_short_side else 'REJ'}"), border=(210, 175, 70), tcol=(255, 225, 130)),
                cs_dir / f"{src_name}_low_res_borderline.png",
                f"[{src_name}] Sát ngưỡng low_res (~{min_short_side}px: dưới & trên)",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

        # C. Blurry: rejected + borderline
        bl_rej = df_s[df_s["blur_score"] < blur_thresh]
        if len(bl_rej) > 0:
            bl_samp = bl_rej.sort_values("blur_score", ascending=True).head(18)
            bl_near = bl_rej.sort_values("blur_score", ascending=False).head(18)
            bl_comb = pd.concat([bl_samp, bl_near]).drop_duplicates("uid").head(36)
            p_out = _render_grid_contact_sheet(
                _to_entries(bl_comb, lambda r: (f"{r.uid[-22:]}", f"blur:{r.blur_score:.2f} side:{r.short_side}"), border=(200, 80, 80), tcol=(255, 165, 165)),
                cs_dir / f"{src_name}_blurry_rejected.png",
                f"[{src_name}] Bị loại blurry (blur < {blur_thresh}): {len(bl_rej):,} ảnh",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

        bl_below = df_s[(df_s["blur_score"] >= blur_thresh * 0.7) & (df_s["blur_score"] < blur_thresh)]
        bl_above = df_s[(df_s["blur_score"] >= blur_thresh) & (df_s["blur_score"] <= blur_thresh * 1.6)]
        if len(bl_below) > 0 or len(bl_above) > 0:
            b1 = bl_below.sample(n=min(18, len(bl_below)), random_state=seed) if len(bl_below) > 0 else bl_below
            b2 = bl_above.sample(n=min(18, len(bl_above)), random_state=seed) if len(bl_above) > 0 else bl_above
            bl_bcomb = pd.concat([b1, b2])
            p_out = _render_grid_contact_sheet(
                _to_entries(bl_bcomb, lambda r: (f"{r.uid[-22:]}", f"blur:{r.blur_score:.2f} {'PASS' if r.blur_score >= blur_thresh else 'REJ'}"), border=(210, 175, 70), tcol=(255, 225, 130)),
                cs_dir / f"{src_name}_blurry_borderline.png",
                f"[{src_name}] Sát ngưỡng blur ({blur_thresh * 0.7:.1f} .. {blur_thresh * 1.6:.1f})",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

        # D. Banner: rejected (stratified by sub-flag) + borderline clean
        bn_rej = df_s[df_s["banner_flag"]]
        if len(bn_rej) > 0:
            chunks = []
            for subf in ["is_top_text", "is_side_frame", "is_top_solid", "is_bot_solid"]:
                part = bn_rej[bn_rej[subf]]
                if len(part) > 0:
                    chunks.append(part.sample(n=min(9, len(part)), random_state=seed))
            bn_samp = pd.concat(chunks).drop_duplicates("uid")
            if len(bn_samp) < 36 and len(bn_rej) > len(bn_samp):
                rem = bn_rej[~bn_rej["uid"].isin(set(bn_samp["uid"]))]
                bn_samp = pd.concat([bn_samp, rem.sample(n=min(36 - len(bn_samp), len(rem)), random_state=seed)])
            def _bn_lbl(r):
                tags = []
                if r.is_top_solid: tags.append("Ts")
                if r.is_bot_solid: tags.append("Bs")
                if r.is_top_text: tags.append("Tt")
                if r.is_side_frame: tags.append("Sf")
                return (f"{r.uid[-22:]}", f"{'+'.join(tags)} Te:{r.top_edge_density:.3f} Se:{r.side_edge_density:.3f}")
            p_out = _render_grid_contact_sheet(
                _to_entries(bn_samp.head(36), _bn_lbl, border=(200, 80, 80), tcol=(255, 165, 165)),
                cs_dir / f"{src_name}_banner_rejected.png",
                f"[{src_name}] Bị gắn cờ banner ({len(bn_rej):,} ảnh) — Ts/Bs:solid, Tt:top_text, Sf:side_frame",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

        bn_bord = df_s[
            (~df_s["banner_flag"])
            & (
                (df_s["top_edge_density"] >= 0.015)
                | (df_s["side_edge_density"] >= 0.020)
                | ((df_s["top_mean"] < 242.0) & (df_s["top_std"] < 12.0))
            )
        ]
        if len(bn_bord) > 0:
            bn_bsamp = bn_bord.sample(n=min(36, len(bn_bord)), random_state=seed)
            p_out = _render_grid_contact_sheet(
                _to_entries(bn_bsamp, lambda r: (f"{r.uid[-22:]}", f"PASS Te:{r.top_edge_density:.3f} Se:{r.side_edge_density:.3f} Tm:{r.top_mean:.0f}"), border=(210, 175, 70), tcol=(255, 225, 130)),
                cs_dir / f"{src_name}_banner_borderline.png",
                f"[{src_name}] Sát ngưỡng banner nhưng PASS ({len(bn_bord):,} ảnh)",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

        # E. Duplicate pairs for source
        src_pairs = [pr for pr in dup_pairs if pr["dup_source"] == src_name and pr["rep_source"] == src_name]
        if src_pairs:
            # Sample deterministically if many
            if len(src_pairs) > 12:
                idxs = sorted(rng.choice(len(src_pairs), size=12, replace=False).tolist())
                use_p = [src_pairs[i] for i in idxs]
            else:
                use_p = src_pairs
            pair_cards = [
                {
                    "rep_path": pr["rep_path"],
                    "dup_path": pr["dup_path"],
                    "cap1": f"Rep:{pr['rep_uid'][-18:]} ({pr['rep_side']}, b={pr['rep_blur']})",
                    "cap2": f"Dup:{pr['dup_uid'][-18:]} h={pr['hdist']} | {pr['attrs']}",
                }
                for pr in use_p
            ]
            p_out = _render_pair_contact_sheet(
                pair_cards,
                cs_dir / f"{src_name}_duplicates_pairs.png",
                f"[{src_name}] Cặp trùng pHash<={phash_thresh} cùng thuộc tính ({len(src_pairs):,} cặp)",
            )
            if p_out:
                rendered_sheets.append(p_out.name)

    # Cross-source duplicates if any
    cross_pairs = [pr for pr in dup_pairs if pr["rep_source"] != pr["dup_source"]]
    if cross_pairs:
        idxs = sorted(rng.choice(len(cross_pairs), size=min(12, len(cross_pairs)), replace=False).tolist())
        pair_cards = [
            {
                "rep_path": cross_pairs[i]["rep_path"],
                "dup_path": cross_pairs[i]["dup_path"],
                "cap1": f"Rep({cross_pairs[i]['rep_source'][:4]}):{cross_pairs[i]['rep_uid'][-16:]} s={cross_pairs[i]['rep_side']}",
                "cap2": f"Dup({cross_pairs[i]['dup_source'][:4]}):{cross_pairs[i]['dup_uid'][-16:]} h={cross_pairs[i]['hdist']}",
            }
            for i in idxs
        ]
        p_out = _render_pair_contact_sheet(
            pair_cards,
            cs_dir / "cross_source_duplicates_pairs.png",
            f"Trùng xuyên nguồn fashion vs shopping100k pHash<={phash_thresh} ({len(cross_pairs):,} cặp)",
        )
        if p_out:
            rendered_sheets.append(p_out.name)

    # 4. Write reports/phase7/gd2_quality_review.md
    print("[4/4] Ghi báo cáo GĐ2 (chưa chốt ngưỡng, chờ duyệt)...")
    f_diag = src_diag["fashion"]
    s_diag = src_diag["shopping100k"]
    s_cfg_row = [r for r in summary_rows if r["subset"] == "shopping100k (tất cả)"][0]

    lines = [
        "# Pha 7a — GĐ2: Kiểm tra Chất lượng Ảnh & Contact Sheets (Chờ duyệt ngưỡng)",
        "",
        f"- **Thời gian chạy:** {time.time() - t0:.1f}s | **Seed:** `{seed}`",
        f"- **Ngưỡng `config.yaml` đang thử nghiệm (CHƯA áp dụng chốt):** `min_short_side={min_short_side}`, `blur_laplacian_var_threshold={blur_thresh}`, `detect_banner={detect_banner}`, `phash_hamming_threshold={phash_thresh}`, `dedup_match_attributes={dedup_attrs}`, `dedup_transitive={img_cfg.get('dedup_transitive')}`.",
        f"- **File metrics:** `eval/phase7/quality_metrics.parquet` ({len(qm_df):,} dòng)",
        "",
        "---",
        "",
        "## 1. Bảng giữ/loại theo ngưỡng `config.yaml` hiện hành (mô phỏng tầng nối tiếp)",
        "",
        "Thứ tự tầng lọc: `low_res (short_side < 224)` → `blurry (blur < 12.0)` → `banner (thuật toán Pha 1)` → `duplicate (cùng article_type+gender+color & pHash <= 1)`.",
        "",
        "| Tập kiểm tra | Tổng ảnh | Loại `low_res` | Loại `blurry` | Loại `banner` | Loại `duplicate` | Tổng loại | Tỷ lệ loại | Giữ lại | Tỷ lệ giữ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['subset']} | {r['total']:,} | {r['low_res']:,} | {r['blurry']:,} | {r['banner']:,} | {r['duplicate']:,} | **{r['rejected_total']:,}** | **{r['rejected_pct']:.2f}%** | **{r['kept']:,}** | **{r['kept_pct']:.2f}%** |"
        )

    lines.extend([
        "",
        f"- Cặp trùng pHash <= {phash_thresh} cùng thuộc tính trong pool sạch: tổng **{len(dup_pairs):,}** cặp (trong đó xuyên nguồn `fashion` ↔ `shopping100k`: **{len(cross_pairs):,}** cặp).",
        "",
        "---",
        "",
        "## 2. Phân bố chỉ số gốc & phân rã nguyên nhân theo nguồn (không chồng tầng)",
        "",
        "| Nguồn | `short_side` min / p5 / p25 / p50 / p95 | `<180` | `180–223` | `224–255` | `>=256` | `blur` p1 / p5 / p10 / p50 |",
        "|---|---|---:|---:|---:|---:|---|",
        f"| `fashion` ({f_diag['n']:,}) | {f_diag['short_side_pcts']['min']:.0f} / {f_diag['short_side_pcts']['p5']:.0f} / {f_diag['short_side_pcts']['p25']:.0f} / {f_diag['short_side_pcts']['p50']:.0f} / {f_diag['short_side_pcts']['p95']:.0f} | {f_diag['short_side_buckets']['lt_180']:,} | {f_diag['short_side_buckets']['180_223']:,} | {f_diag['short_side_buckets']['224_255']:,} | {f_diag['short_side_buckets']['ge_256']:,} | {f_diag['blur_pcts']['p1']:.2f} / {f_diag['blur_pcts']['p5']:.2f} / {f_diag['blur_pcts']['p10']:.2f} / {f_diag['blur_pcts']['p50']:.2f} |",
        f"| `shopping100k` ({s_diag['n']:,}) | {s_diag['short_side_pcts']['min']:.0f} / {s_diag['short_side_pcts']['p5']:.0f} / {s_diag['short_side_pcts']['p25']:.0f} / {s_diag['short_side_pcts']['p50']:.0f} / {s_diag['short_side_pcts']['p95']:.0f} | {s_diag['short_side_buckets']['lt_180']:,} | {s_diag['short_side_buckets']['180_223']:,} | {s_diag['short_side_buckets']['224_255']:,} | {s_diag['short_side_buckets']['ge_256']:,} | {s_diag['blur_pcts']['p1']:.2f} / {s_diag['blur_pcts']['p5']:.2f} / {s_diag['blur_pcts']['p10']:.2f} / {s_diag['blur_pcts']['p50']:.2f} |",
        "",
        "### Phân rã 4 điều kiện cờ Banner (Pha 1)",
        "",
        "| Nguồn | `is_top_solid` (`mean<230 & std<10`) | `is_bot_solid` (`mean<230 & std<10`) | `is_top_text` (`top_edge>0.025`) | `is_side_frame` (`side_edge>0.030`) | Tổng `banner_flag=True` |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `fashion` | {f_diag['banner_subflags']['is_top_solid']:,} | {f_diag['banner_subflags']['is_bot_solid']:,} | {f_diag['banner_subflags']['is_top_text']:,} | {f_diag['banner_subflags']['is_side_frame']:,} | **{f_diag['banner_subflags']['any_banner']:,}** ({f_diag['banner_subflags']['any_banner']/max(1,f_diag['n'])*100:.2f}%) |",
        f"| `shopping100k` | {s_diag['banner_subflags']['is_top_solid']:,} | {s_diag['banner_subflags']['is_bot_solid']:,} | {s_diag['banner_subflags']['is_top_text']:,} | {s_diag['banner_subflags']['is_side_frame']:,} | **{s_diag['banner_subflags']['any_banner']:,}** ({s_diag['banner_subflags']['any_banner']/max(1,s_diag['n'])*100:.2f}%) |",
        "",
        "---",
        "",
        "## 3. Khảo sát phương án ngưỡng cho `shopping100k` (để người dùng quyết định)",
        "",
        "| Phương án `min_short_side` | Ngưỡng `blur` | Luật banner | Ảnh `shopping100k` giữ (trước dedup) | Bị loại | Tỷ lệ giữ |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    for sw in sweeps:
        lines.append(
            f"| {sw['min_short_side']} | {sw['blur_thresh']} | {sw['banner_rule']} | {sw['kept']:,} | {sw['dropped']:,} | {sw['kept_pct']:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Danh sách Contact Sheets đã xuất (`reports/phase7/contact_sheets/`)",
        "",
    ])
    for fn in sorted(rendered_sheets):
        lines.append(f"- `{fn}`")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Điểm dừng [DỪNG] chờ duyệt ngưỡng",
        "",
        f"Theo ngưỡng `config.yaml` hiện hành, `shopping100k` bị loại **{s_cfg_row['rejected_pct']:.2f}%** (giữ **{s_cfg_row['kept']:,} / {s_cfg_row['total']:,}**). ",
        "Xin bạn xem contact sheets (`_low_res_*`, `_blurry_*`, `_banner_*`, `_pass_clear`, `_duplicates_pairs`) và xác nhận áp dụng giữ nguyên ngưỡng `config.yaml` cho cả 2 nguồn hay chọn ngưỡng riêng cho `shopping100k` trước khi chạy GĐ3.",
    ])

    rep_path = reports_p7 / "gd2_quality_review.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi báo cáo GĐ2: {rep_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a GD2 image quality & contact sheets")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", action="store_true", help="Use ThreadPoolExecutor instead of ProcessPoolExecutor")
    args = parser.parse_args()
    run_gd2(
        config_path=args.config,
        merged_dir_override=args.merged_dir,
        shard_size=args.shard_size,
        max_workers=args.workers,
        use_process_pool=not args.threads,
    )


if __name__ == "__main__":
    main()
