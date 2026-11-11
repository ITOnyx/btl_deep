"""Phase 7a GD5 — Export candidates_v3.parquet, Phase 7a Final Report, and Manifest Update.

Generates:
  - data/catalog/candidates_v3.parquet  (98,289 clean candidates, NEVER overwrites catalog.parquet)
  - reports/phase7/BAO_CAO_PHA7A.md     (comprehensive Phase 7a Vietnamese report & Phase 7b stats)
  - reports/run_manifest.json           (appends phase7a SHA-256 checksums preserving prior phases)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.phase7a.color_estimator import load_colour_family_map
from src.phase7a.common import ensure_utf8_stdio, load_config, resolve_phase7a_paths


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest using chunked streaming reads."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Visually audited pixel accuracy findings across gd5_color_visual_eval_100.csv (seed=42)
# Each category summarizes direct visual inspection of _tmp_color_eval_100_p1/p2.png:
#   56/100 agreed with source attribute label & visually matched garment
#   12/100 pixel prediction was visually RIGHT where source label was wrong (#1,#8,#18,#30,#33,#52,#65,#80,#85,#92,#93,#100)
#   32/100 estimator error (#15,#16,#17,#21,#25,#38,#41,#44,#45,#49,#58,#59,#60,#63,#64,#66,#68,#70,#71,#73,#74,#76,#81,#86,#88,#90,#94,#96,#97,#99 + #3,#13)
VISUAL_100_PIXEL_RIGHT_WHEN_DIFF = {1, 8, 18, 30, 33, 52, 65, 80, 85, 92, 93, 100}
VISUAL_100_FAILURE_MODE = {
    16: "pale_to_white", 17: "pale_to_white", 38: "pale_to_white", 41: "pale_to_white",
    58: "pale_to_white", 59: "pale_to_white", 70: "pale_to_white", 73: "pale_to_white",
    88: "pale_to_white", 94: "pale_to_white", 96: "pale_to_white",
    15: "white_studio_drift", 44: "white_studio_drift", 60: "white_studio_drift",
    90: "white_studio_drift", 99: "white_studio_drift",
    45: "pink_mauve_to_red", 71: "pink_mauve_to_red",
    64: "dark_navy_to_purple",
    13: "warm_tone_confusion", 21: "warm_tone_confusion", 49: "warm_tone_confusion",
    74: "warm_tone_confusion", 86: "warm_tone_confusion", 97: "warm_tone_confusion",
    3: "pattern_multi_green", 25: "pattern_multi_green", 63: "pattern_multi_green",
    66: "pattern_multi_green", 68: "pattern_multi_green", 76: "pattern_multi_green",
    81: "pattern_multi_green",
}


def run_gd5(config_path: str | Path | None = None, merged_dir_override: str | Path | None = None) -> None:
    """Export candidates_v3.parquet, BAO_CAO_PHA7A.md, and update run_manifest.json."""
    t0 = time.time()
    ensure_utf8_stdio()
    cfg = load_config(config_path)
    paths = resolve_phase7a_paths(cfg, merged_dir_override=merged_dir_override)

    repo_root = paths["repo_root"]
    eval_p7 = paths["eval_p7"]
    reports_p7 = paths["reports_p7"]
    candidates_v3_path = paths["candidates_out"]

    print("=" * 72)
    print("PHA 7a — GĐ5: XUẤT candidates_v3.parquet + BÁO CÁO PHA 7a + MANIFEST")
    print("=" * 72)

    df_inv = pd.read_parquet(eval_p7 / "inventory.parquet")
    df_clean = pd.read_parquet(eval_p7 / "clean_candidates_gd3.parquet")
    df_rej = pd.read_csv(eval_p7 / "candidate_rejects.csv")
    df_gd4 = pd.read_parquet(eval_p7 / "candidates_mapped_gd4.parquet")
    fam_map = load_colour_family_map(str(paths["eval_dir"] / "families.json"))

    # 1. Build clean candidates_v3.parquet (strictly never modifying catalog.parquet)
    df_v3 = pd.DataFrame({
        "item_id": df_gd4["uid"].astype(str),
        "orig_id": df_gd4["orig_id"].astype(str),
        "image_path": df_gd4["image_path"].astype(str),
        "abs_path": df_gd4["abs_path"].astype(str),
        "source": df_gd4["source"].astype(str),
        "split_src": df_gd4["split"].astype(str),
        "seen_in_train": df_gd4["seen_in_train"].astype(str),
        "articleType": df_gd4["articleType"].astype(str),
        "gender": df_gd4["gender"].astype(str),
        "baseColour": df_gd4["baseColour"].astype(str),
        "color_label": df_gd4["color_label"].astype(str),
        "color_pixel": df_gd4["color_pixel"].astype(str),
        "color_conf": df_gd4["color_conf"].astype(str),
        "color_source": df_gd4["color_source"].astype(str),
        "cat_conf": df_gd4["cat_conf"].astype(str),
        "gender_conf": df_gd4["gender_conf"].astype(str),
        "attr_confidence": df_gd4["attr_confidence"].astype(str),
        "mapping_reason": df_gd4["mapping_reason"].astype(str),
        "rule_id": df_gd4["split_rule_id"].astype(str),
        "pending_lexicon": df_gd4["pending_lexicon"].astype(bool),
        "low_contrast_flag": df_gd4["low_contrast_flag"].astype(bool),
        "banner_flag": df_gd4["banner_flag"].astype(bool),
        "short_side": df_gd4["short_side"].astype(int),
        "blur_score": df_gd4["blur_score"].astype(float),
        "serving_eligible": df_gd4["serving_eligible"].astype(bool),
        "eval_ext_eligible": df_gd4["eval_ext_eligible"].astype(bool),
        "bm25_eligible": df_gd4["bm25_eligible"].astype(bool),
        "title_vi": df_gd4["title_vi"].astype(str),
        "title_vi_display": df_gd4["title_vi_display"].astype(str),
        "meta_source": df_gd4["meta_source"].astype(str),
        "proposed_index_tier": df_gd4["proposed_index_tier"].astype(str),
        "cfg_in_scope": df_gd4["cfg_in_scope"].astype(bool),
        "catalog_phash_dup_flag": df_gd4["catalog_phash_dup_flag"].astype(bool),
        "internal_sha1_dup_flag": df_gd4["internal_sha1_dup_flag"].astype(bool),
        "internal_phash_dup_flag": df_gd4["internal_phash_dup_flag"].astype(bool),
        "legacy_test_protected": df_gd4["legacy_test_protected"].astype(bool),
        "sha1": df_gd4["sha1"].astype(str),
        "phash": df_gd4["phash"].astype(str),
    })

    # Invariant assertions before writing
    assert len(df_v3) == 98289, f"Expected 98289 rows, got {len(df_v3)}"
    assert df_v3["item_id"].is_unique, "item_id must be unique"
    assert int(((df_v3["color_source"] == "image_estimated") & (df_v3["baseColour"] != "Unknown")).sum()) == 0
    assert int(((df_v3["bm25_eligible"]) & (df_v3["attr_confidence"] != "high")).sum()) == 0
    assert int(((df_v3["pending_lexicon"]) & (df_v3["attr_confidence"] == "high")).sum()) == 0

    candidates_v3_path.parent.mkdir(parents=True, exist_ok=True)
    df_v3.to_parquet(candidates_v3_path, index=False)
    print(f"Đã xuất {candidates_v3_path} ({len(df_v3):,} dòng x {df_v3.shape[1]} cột)")

    # 2. Load index_v2 & dense_captions_catalog.json for Phase 7b analytics (Req 4 & 7)
    id_map = json.loads((repo_root / "index_v2" / "id_map_v2.json").read_text(encoding="utf-8"))
    v2_clean = [r for r in id_map if str(r["item_id"]) not in {"43230", "42066"}]
    v2_df = pd.DataFrame(v2_clean)
    v2_slot_cnt = Counter(zip(v2_df["articleType"], v2_df["gender"]))

    dc_raw = json.loads((paths["checkpoint_dir"] / "dense_captions_catalog.json").read_text(encoding="utf-8"))
    dc_ids = set(str(k) for k in dc_raw.keys())

    sh = df_v3[df_v3["source"] == "shopping100k"].copy()
    sh_high = sh[sh["attr_confidence"] == "high"]
    sh_lab = sh[sh["color_source"] == "attributes"].copy()
    sh_lab["lbl_fam"] = sh_lab["color_label"].map(lambda c: fam_map.get(c, c))
    sh_lab["px_fam"] = sh_lab["color_pixel"].map(lambda c: fam_map.get(c, c))
    sh_lab["match"] = sh_lab["lbl_fam"] == sh_lab["px_fam"]

    # Req 4(b): 100 visual eval stats
    s100_path = eval_p7 / "gd5_color_visual_eval_100.csv"
    df_s100 = pd.read_csv(s100_path)
    s100_rows = []
    for idx, r in enumerate(df_s100.itertuples(index=False), start=1):
        vis_ok = bool(r.match) or (idx in VISUAL_100_PIXEL_RIGHT_WHEN_DIFF)
        s100_rows.append({
            "idx": idx,
            "lbl_fam": r.lbl_fam,
            "px_fam": r.px_fam,
            "auto_match": bool(r.match),
            "visual_px_ok": vis_ok,
            "failure_mode": VISUAL_100_FAILURE_MODE.get(idx, "ok" if vis_ok else "other"),
        })
    df_vis100 = pd.DataFrame(s100_rows)
    df_vis100.to_csv(eval_p7 / "gd5_color_visual_eval_100_annotated.csv", index=False, encoding="utf-8")

    # Req 4(c): high + White risk count
    sh_high_white = int((sh_high["baseColour"] == "White").sum())
    all_high_white = int(((df_v3["serving_eligible"]) & (df_v3["attr_confidence"] == "high") & (df_v3["baseColour"] == "White")).sum())
    all_high_total = int(((df_v3["serving_eligible"]) & (df_v3["attr_confidence"] == "high")).sum())

    # Req 7(c): eval_ext_eligible & dense_captions_catalog match
    ext_df = df_v3[df_v3["eval_ext_eligible"]].copy()
    ext_df["in_dense_captions"] = ext_df["orig_id"].isin(dc_ids) | ext_df["item_id"].isin(dc_ids)
    ext_summary = ext_df.groupby(["source", "in_dense_captions"]).size().to_dict()

    # Req 7(a) & (b): Slot analysis (all serving_eligible candidates) vs index_v2
    elig = df_v3[df_v3["serving_eligible"]].copy()
    slot_conf = elig.groupby(["articleType", "gender", "attr_confidence"]).size().unstack(fill_value=0)
    for col in ("high", "medium", "low"):
        if col not in slot_conf.columns:
            slot_conf[col] = 0
    slot_pend = elig.groupby(["articleType", "gender"])["pending_lexicon"].any()
    slot_med_colok = elig[(elig["attr_confidence"] == "medium") & (elig["color_conf"] == "high")].groupby(["articleType", "gender"]).size()

    all_slots = sorted(set(v2_slot_cnt.keys()) | set(slot_conf.index))
    thin_v2_rows: List[Dict[str, Any]] = []
    for slot in sorted(k for k, v in v2_slot_cnt.items() if v < 300):
        v2_n = v2_slot_cnt[slot]
        if slot in slot_conf.index:
            h_n = int(slot_conf.loc[slot, "high"])
            m_n = int(slot_conf.loc[slot, "medium"])
            l_n = int(slot_conf.loc[slot, "low"])
            pend = bool(slot_pend.get(slot, False))
            mc_n = int(slot_med_colok.get(slot, 0))
            if (h_n + m_n + l_n) > 0:
                thin_v2_rows.append({
                    "articleType": slot[0], "gender": slot[1], "v2_cnt": v2_n,
                    "high_add": h_n, "med_add": m_n, "low_add": l_n,
                    "med_color_ok": mc_n, "pending_lexicon": pend,
                })

    # Compute exact SHA-256 hashes (we write manifest after report is finalized too)
    inv_src_cnt = df_inv["source"].value_counts().to_dict()
    rej_cnt = df_rej["reason"].value_counts().to_dict()
    clean_src_cnt = df_clean["source"].value_counts().to_dict()
    v3_conf_sh = sh["attr_confidence"].value_counts().to_dict()
    v3_conf_fa = df_v3[(df_v3["source"] == "fashion") & (df_v3["serving_eligible"])]["attr_confidence"].value_counts().to_dict()

    bm25_true = int(df_v3["bm25_eligible"].sum())
    bm25_sh = int(((df_v3["source"] == "shopping100k") & df_v3["bm25_eligible"]).sum())
    bm25_fa = int(((df_v3["source"] == "fashion") & df_v3["bm25_eligible"]).sum())
    no_color_disp = int(((df_v3["color_conf"] != "high") & (df_v3["serving_eligible"])).sum())
    miss_color_n = int((df_v3["color_source"] == "image_estimated").sum())
    pend_lex_n = int(df_v3["pending_lexicon"].sum())

    lines_bm25_str = f"- **Hiển thị & BM25:** `bm25_eligible=True` chỉ bật cho **`{bm25_true:,}`** ảnh (`{bm25_sh:,} shopping100k + {bm25_fa:,} fashion`, đúng bằng tập `serving_eligible & attr_confidence == 'high'`). Cột `title_vi_display` chỉ chèn màu khi `color_conf == 'high'`; bỏ màu ở toàn bộ **{no_color_disp:,}** ảnh phục vụ còn lại (`color_conf != 'high'`)."

    # Build reports/phase7/BAO_CAO_PHA7A.md
    lines: List[str] = [
        "# BÁO CÁO PHA 7a — TIỀN XỬ LÝ & TẠO TẬP ỨNG VIÊN `candidates_v3.parquet`",
        "",
        f"- **Ngày hoàn tất:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Seed:** `{cfg.get('eval', {}).get('seed', 42)}` | **Nhánh git:** `phase7a`",
        f"- **Đầu ra chính:** [`data/catalog/candidates_v3.parquet`](../../data/catalog/candidates_v3.parquet) (**{len(df_v3):,}** dòng × **{df_v3.shape[1]}** cột), [`eval/phase7/s100k_mapping.csv`](../../eval/phase7/s100k_mapping.csv) (**98,174** dòng).",
        "- **Tuân thủ bất biến:** KHÔNG ghi đè `catalog.parquet`, KHÔNG sửa `src/lexicon_vi.json`, KHÔNG embed/dựng index, KHÔNG đụng `_build/`, KHÔNG sửa/crop ảnh gốc.",
        "",
        "---",
        "",
        "## 1. Phễu số lượng ảnh qua từng giai đoạn (GĐ1 → GĐ5)",
        "",
        "| Giai đoạn | Tổng ảnh | `fashion` | `shopping100k` | Ghi chú |",
        "|---|---:|---:|---:|---|",
        f"| Đầu vào `merged_dataset/annotations/all.parquet` (GĐ1) | **{len(df_inv):,}** | {inv_src_cnt.get('fashion', 0):,} | {inv_src_cnt.get('shopping100k', 0):,} | 100% ảnh đọc được (0 unreadable, 2 ảnh xám `L`, 0 EXIF xoay) |",
        f"| Loại ở GĐ3 (`candidate_rejects.csv`) | **-{len(df_rej):,}** | -{inv_src_cnt.get('fashion', 0) - clean_src_cnt.get('fashion', 0):,} | -{inv_src_cnt.get('shopping100k', 0) - clean_src_cnt.get('shopping100k', 0):,} | Bảo vệ 94/100 ảnh `fashion legacy_test` (chỉ loại 6 ảnh `low_res=150`) |",
        f"| **Tập sạch GĐ3 → GĐ5 (`candidates_v3.parquet`)** | **{len(df_v3):,}** | **{clean_src_cnt.get('fashion', 0):,}** | **{clean_src_cnt.get('shopping100k', 0):,}** | Tách cờ `serving_eligible`, `eval_ext_eligible`, `bm25_eligible`, `pending_lexicon` |",
        f"| ↳ `serving_eligible = True` | **{int(df_v3['serving_eligible'].sum()):,}** | 105 | 98,174 | Loại 10 ảnh `fashion` khỏi phục vụ (6 ngoài scope + 4 trùng pHash catalog; giữ phục vụ đánh giá) |",
        f"| ↳ `eval_ext_eligible = True` (`seen_in_train='no'`) | **{int(df_v3['eval_ext_eligible'].sum()):,}** | 94 | 1,468 | Đánh giá mở rộng Pha 7b (chi tiết overlap dense captions ở mục 5c) |",
        "",
        "### Chi tiết lý do loại ở GĐ3 (`candidate_rejects.csv`: 40.850 ảnh)",
        "",
        "| Lý do loại | Số ảnh |",
        "|---|---:|",
    ]
    for rsn, cnt in sorted(rej_cnt.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{rsn}` | {cnt:,} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Phân bố `attr_confidence` cuối & các quy tắc đã kiểm chứng / hạ cấp",
        "",
        "| Nguồn | `high` (`AttributeIndex+dense`) | `medium` (`dense_only`) | `low` (`dense_only`) | Tổng |",
        "|---|---:|---:|---:|---:|",
        f"| `shopping100k` | **{v3_conf_sh.get('high', 0):,}** ({v3_conf_sh.get('high', 0)/len(sh)*100:.2f}%) | **{v3_conf_sh.get('medium', 0):,}** ({v3_conf_sh.get('medium', 0)/len(sh)*100:.2f}%) | **{v3_conf_sh.get('low', 0):,}** ({v3_conf_sh.get('low', 0)/len(sh)*100:.2f}%) | **{len(sh):,}** |",
        f"| `fashion` (`serving_eligible`) | **{v3_conf_fa.get('high', 0):,}** | **{v3_conf_fa.get('medium', 0):,}** | **{v3_conf_fa.get('low', 0):,}** | **{sum(v3_conf_fa.values()):,}** (+10 eval_only) |",
        f"| **Toàn tập phục vụ (`serving_eligible=True`)** | **{all_high_total:,}** | **{int((df_v3['serving_eligible'] & (df_v3['attr_confidence']=='medium')).sum()):,}** | **{int((df_v3['serving_eligible'] & (df_v3['attr_confidence']=='low')).sum()):,}** | **{int(df_v3['serving_eligible'].sum()):,}** |",
        "",
        "### Bảng kiểm chứng tiếp xúc (contact sheet 40 ảnh) & quyết định cập nhật quy tắc",
        "",
        "| Quy tắc / Nhóm | Tổng ảnh gốc | Kết quả kiểm chứng bằng mắt (40 mẫu/hết) | Tỷ lệ sai ước lượng | Hành động áp dụng ở GĐ4/GĐ5 | Sheet kiểm chứng |",
        "|---|---:|---|---:|---|---|",
        "| `F4_polo` | 324 | 40/40 đều là áo polo nữ chuẩn | **0.0%** (0/40) | **Giữ `high`** | `gd4_verify_F4_polo.png` |",
        "| `M4_polo` | 2,048 | 40/40 đều là áo polo nam chuẩn | **0.0%** (0/40) | **Giữ `high`** | `gd4_verify_M4_polo.png` |",
        "| `F5_polo` | 47 | 35/40 là áo cổ lọ / cổ cao / áo kiểu không tay | **87.5%** (35/40) | **Hạ `high → medium`** (`dense_only`) | `gd4_verify_F5_polo.png` |",
        "| `M5_polo` | 33 | 17 cổ lọ/hoodie + 1 áo cổ tròn, chỉ 15 polo | **54.5%** (18/33) | **Hạ `high → medium`** (`dense_only`) | `gd4_verify_M5_polo.png` |",
        "| `F5_classic_short_tee` | 1,545 | 6/40 là áo kiểu ren/bèo/cowl (+ áo hở vai trên `shirts_tops_40`) | **15.0%** (6/40) | **Hạ `high → medium`** (`dense_only`) | `gd4_verify_F5_classic_short_tee.png` |",
        "| `Swimwear` (`long`/`belt`) | 73 | 40/40 đều là áo choàng tắm / kimono robe | **100.0%** (40/40) | **Tách sang `Bath Robe`** (164 tổng), `medium` + `pending_lexicon=True` | `gd4_verify_Swimwear_long_or_belt.png` |",
        "| `J_sweater` (`fastening=Zip`) | 506 | 35/40 là áo khoác thể thao / áo khoác nỉ kéo khóa | **87.5%** (35/40) | **Tách `J_sweater_zip`, hạ `medium → low`** | `gd4_verify_J_sweater_zip.png` |",
        f"| `TracksuitBottoms_pending_lex` & nhóm thiếu từ điển | {pend_lex_n:,} | `Track Pants` (3.634) trộn jogger + legging yoga; `Swimwear` (5.119), `Jumpsuit` (1.240), `Tracksuits` (213), `Bath Robe` (164) chưa mở khóa `lexicon_vi.json` | — | **Gắn `pending_lexicon=True`**, giữ **`attr_confidence <= medium`**, `dense_only` | `gd4_sample_200_part*.png` |",
        "",
        lines_bm25_str,
        "",
        "---",
        "",
        "## 3. Phân tích kiểm chứng màu pixel (Mục 4a, 4b, 4c)",
        "",
        f"- **Bảo toàn nhãn gốc:** toàn bộ **{miss_color_n:,}** ảnh thiếu màu (`color == -1`) vẫn giữ `baseColour = 'Unknown'` (`color_source = 'image_estimated'`), `color_conf = 'low'`, tuyệt đối KHÔNG tự điền vào `baseColour`.",
        f"- **(4c) Rủi ro cùng sai ở màu `White` trong tập `high`:** **`{sh_high_white:,} / {len(sh_high):,}`** ({sh_high_white/len(sh_high)*100:.2f}%) ảnh `shopping100k high` mang nhãn `baseColour = 'White'` (+ 14 ảnh `fashion` = **{all_high_white:,}**). Vì ảnh sáng/màu pastel nhạt dễ được `color_pixel` ước lượng thành `White`, nếu nhãn gốc Shopping100k cũng ghi nhầm `White` thì cặp `(label=White, pixel=White)` sẽ cùng lọt `color_conf=high` (cần giới hạn cap và lấy mẫu kiểm lại ở Pha 7b).",
        "",
        "### (4a) Phân bố `baseColour` (`shopping100k high` vs `shopping100k 98.174` vs `index_v2 38.644`)",
        "",
        "| `baseColour` | `s100k high` (33.101) | % `high` | `s100k all` (98.174) | % `all` | `index_v2` (38.644) | % `v2` |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    c_high = sh_high["baseColour"].value_counts()
    c_all = sh["baseColour"].value_counts()
    c_v2 = v2_df["baseColour"].value_counts()
    all_cols_sorted = sorted(set(c_high.index) | set(c_all.index), key=lambda c: (-int(c_all.get(c, 0)), c))
    for cname in all_cols_sorted:
        hn = int(c_high.get(cname, 0))
        an = int(c_all.get(cname, 0))
        vn = int(c_v2.get(cname, 0))
        lines.append(
            f"| `{cname}` | {hn:,} | {hn/len(sh_high)*100:.2f}% | {an:,} | {an/len(sh)*100:.2f}% | {vn:,} | {vn/len(v2_df)*100:.2f}% |"
        )

    lines.extend([
        "",
        "### (4b) Độ đúng của `color_pixel` theo họ màu (toàn tập 80.289 có nhãn + chấm mắt 100 mẫu `seed=42`)",
        "",
        "- **Trên 100 ảnh mẫu phân tầng theo họ màu (`gd5_color_visual_eval_100.csv`):**",
        f"  - Khớp tự động `color_label == color_pixel` (cùng họ): **{int(df_vis100['auto_match'].sum())}/100 (56.0%)**",
        f"  - Thêm **12/100** ca lệch mà mắt kiểm tra thấy `color_pixel` **ĐÚNG** do nhãn gốc Shopping100k sai (`#1, #8, #18, #30, #33, #52, #65, #80, #85, #92, #93, #100`) → **Tỷ lệ ước lượng `color_pixel` đúng thực tế = {int(df_vis100['visual_px_ok'].sum())}/100 ({df_vis100['visual_px_ok'].mean()*100:.1f}%)**.",
        "  - **32/100 lỗi của `color_pixel`:** màu nhạt/pastel ra `White` (**11/32**: #16, #17, #38, #41, #58, #59, #70, #73, #88, #94, #96); đồ đa sắc/họa tiết/xanh lá tối (**7/32**: #3, #25, #63, #66, #68, #76, #81); cam/nâu/be/xám ranh giới (**6/32**: #13, #21, #49, #74, #86, #97); đồ trắng dính bóng/chữ ra `Grey/Charcoal/Navy` (**5/32**: #15, #44, #60, #90, #99); hồng đậm/tím mauve ra `Red` (**2/32**: #45, #71); navy họa tiết đậm ra `Purple` (**1/32**: #64).",
        "",
        "| Họ màu (`eval/families.json`) | Tổng ảnh có nhãn | Khớp nhãn-pixel toàn tập | Mẫu 100 (khớp tự động) | Mẫu 100 (`color_pixel` đúng thật) | Đánh giá độ tin cậy `color_pixel` |",
        "|---|---:|---:|---:|---:|---|",
    ])
    fam_full_stats = sh_lab.groupby("lbl_fam")["match"].agg(["count", "sum", "mean"])
    for fname in sorted(fam_full_stats.index, key=lambda fn: fam_full_stats.loc[fn, "mean"]):
        fc = int(fam_full_stats.loc[fname, "count"])
        fs = int(fam_full_stats.loc[fname, "sum"])
        fm = float(fam_full_stats.loc[fname, "mean"]) * 100.0
        sub100 = df_vis100[df_vis100["lbl_fam"] == fname]
        a100 = int(sub100["auto_match"].sum())
        v100 = int(sub100["visual_px_ok"].sum())
        n100 = len(sub100)
        tag = "**KÉM**" if fm < 35.0 else ("Trung bình" if fm < 70.0 else "Tốt")
        lines.append(f"| `{fname}` | {fc:,} | {fs:,} ({fm:.1f}%) | {a100}/{n100} | {v100}/{n100} ({v100/max(1,n100)*100:.0f}%) | {tag} |")

    lines.extend([
        "",
        "- **Các họ màu ước lượng kém cần lưu ý:** `Brown` (khớp 6,1% — áo nâu nhạt/be dễ ra White/Orange hoặc đen tối), `Multi` (11,3% — hay bị chọn màu nền chủ đạo), `Orange` (20,2% — dễ lệch `Yellow`/`Red`/`Beige`), `Pink` (20,7% — hồng nhạt ra `White`, hồng đậm ra `Red`), `Purple` (20,8% — ranh giới `Navy`/`Black`/`Red`), `Beige` (25,8% — nhạt ra `White`), `Grey` (28,3% — xám sáng ra `White`, xám đậm ra `Charcoal`), `Green` (32,5% — rêu/xanh đậm lệch `Grey`/`Black`/`Brown`).",
        "",
        "---",
        "",
        "## 4. Hạn chế còn lại & Đề xuất cho Pha 7b",
        "",
        "1. **Lệch phân bố nặng (Bias nguồn & màu):** Riêng Shopping100k `high` (`33,101` ảnh) tập trung 75%+ vào `Black` (`10,602`), `Blue/Navy` (`10,956`), `White` (`3,987`), và dồn rất mạnh vào vài slot (`(Dresses, Women) +6,522`, `(Tshirts, Men) +5,027`). Nếu nhúng toàn bộ 33k ảnh vào index, không gian vector và prior truy vấn text→ảnh sẽ lệch mạnh sang ảnh studio châu Âu (ghost-mannequin) của Shopping100k.",
        "2. **Độ tin cậy màu & nhãn gốc:** Shopping100k có ~12–15% nhãn màu gốc sai (ví dụ `Navy Blue` trên áo hồng đậm, `Beige` trên áo trắng) và `3,987` ảnh `high + White` mang rủi ro *double-agreement bias* (`label=White` + `pixel=White`).",
        "3. **Đề xuất chọn lọc cho Pha 7b:** **KHÔNG đưa toàn bộ 33.203 ảnh `high` vào index.** Áp dụng **cap theo từng slot `(articleType × gender)`** (ví dụ: ưu tiên lấp các slot `index_v2 < 300` đến ngưỡng mục tiêu `300–400 sp/slot`; các slot đã dày như `(Tshirts, Men) = 5,089` hoặc `(Shirts, Men) = 2,597` chỉ lấy mẫu nhỏ `<= 200–300` đa dạng màu hoặc bằng `0` cho `AttributeIndex`, đồng thời rà nhanh slot `White`). Với 5 nhóm `pending_lexicon=True`, hoàn thiện `lexicon_vi.json` + test parser trước khi mở khóa vào `AttributeIndex`.",
        "",
        "---",
        "",
        "## 5. Thống kê phục vụ Pha 7b (Mục 7a, 7b, 7c)",
        "",
        "### (7c) Tập đánh giá mở rộng `eval_ext_eligible` (`seen_in_train == 'no'`) & đối chiếu ID với `dense_captions_catalog.json`",
        "",
        f"- `fashion` (`eval_ext_eligible=True`): **{ext_summary.get(('fashion', True), 0) + ext_summary.get(('fashion', False), 0):,}** ảnh — có ID trong `dense_captions_catalog.json`: **{ext_summary.get(('fashion', True), 0):,}** (`0` thiếu).",
        f"- `shopping100k` (`eval_ext_eligible=True`): **{ext_summary.get(('shopping100k', True), 0) + ext_summary.get(('shopping100k', False), 0):,}** ảnh — có ID trong `dense_captions_catalog.json`: **{ext_summary.get(('shopping100k', True), 0):,}** (tất cả 1.468 ảnh Shopping100k chưa từng có caption trong file train/test gốc).",
        "",
        "### (7b) Các slot mỏng trong `index_v2` (`< 300` sản phẩm) có ứng viên Pha 7a bổ sung (gồm cả `pending_lexicon`)",
        "",
        "| `(articleType, gender)` | `index_v2` | `+ high` | `+ medium` (`color=high`) | `+ low` | `pending_lexicon` | Ghi chú Pha 7b |",
        "|---|---:|---:|---:|---:|---|---|",
    ])
    for row in sorted(thin_v2_rows, key=lambda r: (r["v2_cnt"], -(r["high_add"] + r["med_color_ok"]))):
        note = "Chờ lexicon_vi ở Pha 7b" if row["pending_lexicon"] else ("Có ảnh `high` lấp trực tiếp" if row["high_add"] > 0 else "Chỉ có `medium`/`low` (`dense_only`)")
        lines.append(
            f"| `({row['articleType']}, {row['gender']})` | {row['v2_cnt']:,} | +{row['high_add']:,} | +{row['med_add']:,} ({row['med_color_ok']:,} màu ok) | +{row['low_add']:,} | {'True' if row['pending_lexicon'] else 'False'} | {note} |"
        )

    lines.extend([
        "",
        "### (7a) Bảng đầy đủ theo từng slot `(articleType × gender)` đối chiếu `index_v2` (tất cả `serving_eligible` có ứng viên)",
        "",
        "| `(articleType, gender)` | `index_v2` | `high` | `medium` | `low` | Tổng ứng viên |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for slot in sorted(slot_conf.index, key=lambda s: (-int(slot_conf.loc[s, "high"]), -(int(slot_conf.loc[s, "medium"]) + int(slot_conf.loc[s, "low"])))):
        v2_n = v2_slot_cnt.get(slot, 0)
        h_n = int(slot_conf.loc[slot, "high"])
        m_n = int(slot_conf.loc[slot, "medium"])
        l_n = int(slot_conf.loc[slot, "low"])
        lines.append(f"| `({slot[0]}, {slot[1]})` | {v2_n:,} | {h_n:,} | {m_n:,} | {l_n:,} | **{h_n + m_n + l_n:,}** |")

    rep_path = reports_p7 / "BAO_CAO_PHA7A.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi báo cáo cuối Pha 7a: {rep_path}")

    # 3. Update reports/run_manifest.json (preserving all existing blocks)
    manifest_path = repo_root / "reports" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    hashes = {
        "candidates_v3_parquet": sha256_file(candidates_v3_path),
        "s100k_mapping_csv": sha256_file(eval_p7 / "s100k_mapping.csv"),
        "candidates_mapped_gd4_parquet": sha256_file(eval_p7 / "candidates_mapped_gd4.parquet"),
        "clean_candidates_gd3_parquet": sha256_file(eval_p7 / "clean_candidates_gd3.parquet"),
        "bao_cao_pha7a_md": sha256_file(rep_path),
    }

    manifest["phase7a"] = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+07:00"),
        "status": "COMPLETED",
        "seed": int(cfg.get("eval", {}).get("seed", 42)),
        "counts": {
            "inventory_total": int(len(df_inv)),
            "rejected_gd3_total": int(len(df_rej)),
            "candidates_v3_total": int(len(df_v3)),
            "serving_eligible_true": int(df_v3["serving_eligible"].sum()),
            "serving_eligible_false": int((~df_v3["serving_eligible"]).sum()),
            "eval_ext_eligible_true": int(df_v3["eval_ext_eligible"].sum()),
            "bm25_eligible_true": int(df_v3["bm25_eligible"].sum()),
            "pending_lexicon_true": int(df_v3["pending_lexicon"].sum()),
            "shopping100k_attr_confidence": {k: int(v) for k, v in v3_conf_sh.items()},
            "fashion_serving_attr_confidence": {k: int(v) for k, v in v3_conf_fa.items()},
            "shopping100k_high_white": sh_high_white,
        },
        "files_sha256": hashes,
    }
    root_hashes = manifest.setdefault("root_hashes", {})
    root_hashes["candidates_v3_parquet"] = hashes["candidates_v3_parquet"]
    root_hashes["s100k_mapping_csv"] = hashes["s100k_mapping_csv"]
    root_hashes["bao_cao_pha7a_md"] = hashes["bao_cao_pha7a_md"]

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã cập nhật manifest sha256 -> {manifest_path}")
    print(f"SHA256 candidates_v3.parquet: {hashes['candidates_v3_parquet']}")
    print(f"SHA256 s100k_mapping.csv    : {hashes['s100k_mapping_csv']}")
    print(f"Hoàn tất GĐ5 trong {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a GD5 candidates_v3 export and manifest update")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    args = parser.parse_args()
    run_gd5(config_path=args.config, merged_dir_override=args.merged_dir)


if __name__ == "__main__":
    main()
