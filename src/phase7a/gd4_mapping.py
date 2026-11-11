"""Phase 7a GD4 — Shopping100k Metadata Mapping, Attribute Splitting & Pixel Color Cross-Check.

Processes all 98,289 items in eval/phase7/clean_candidates_gd3.parquet (115 fashion + 98,174
Shopping100k) without modifying source images or existing catalog/index artifacts:
  1. Compiles raw counts across all 12 Shopping100k Attributes/*.txt files.
  2. Splits mixed Shopping100k groups (shirts_tops, 6_Jumpers, 10_Swimming) using detailed
     attributes (collar, fastening, sleevelen, neckline, fabric) into standard articleTypes,
     assigning unresolved shirt/top items to 'Tops/Shirts-ambiguous' (attr_confidence='low').
  3. Runs sharded resumable pixel dominant color estimation (color_estimator.py) across all
     98,174 Shopping100k images to cross-check raw color labels via eval/families.json without
     overwriting baseColour, flagging image_estimated for color=-1 items.
  4. Generates derived Vietnamese titles (generate_vietnamese_title) gated by bm25_eligible.
  5. Sets serving_eligible (False for 6 out-of-scope + 4 catalog-pHash-dup fashion items)
     and eval_ext_eligible (seen_in_train == 'no': 94 fashion + 1,468 shopping100k).
  6. Exports s100k_mapping.csv, candidates_mapped_gd4.parquet, stratified & diagnostic contact
     sheets (~40 images/sheet), index_v2 <30 slot comparison, and reports/phase7/gd4_mapping.md.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src.phase7a.color_estimator import (
    load_colour_family_map,
    predict_color_pixel_worker,
)
from src.phase7a.common import ensure_utf8_stdio, load_config, resolve_phase7a_paths
from src.phase7a.gd2_quality import _load_fonts
from src.vi_vocab import ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP, generate_vietnamese_title

CONF_RANK = {"low": 0, "medium": 1, "high": 2}
RANK_TO_CONF = {0: "low", 1: "medium", 2: "high"}

# Complete 12 Shopping100k Attribute Codebooks from data/Shopping100k/Readme
S100K_CODEBOOKS: Dict[str, Dict[int, str]] = {
    "category": {
        -1: "MISSING(-1)", 0: "Coat", 1: "Jacket", 2: "Suit", 3: "Shirt", 4: "T-shirt",
        5: "Jumper", 6: "Shorts", 7: "Trouser", 8: "Jean", 9: "Swimming", 10: "Jumpsuit",
        11: "Pyjamas", 12: "Tracksuit Bottoms", 13: "Tracksuit", 14: "Skirt", 15: "Dress",
    },
    "collar": {
        -1: "MISSING(-1)", 0: "Buttondown", 1: "Cutaway", 2: "High", 3: "Hood", 4: "Kent",
        5: "Lapel", 6: "Lined", 7: "Mandarin", 8: "Polo", 9: "Round", 10: "Shawl",
        11: "Turndown", 12: "V-neck", 13: "Peter Pan", 14: "Volant", 15: "Shirt", 16: "Chin",
    },
    "color": {
        -1: "MISSING(-1)", 0: "Maroon", 1: "Black", 2: "White", 3: "Charcoal",
        4: "Indigo", 5: "Navy", 6: "Pink", 7: "Light blue", 8: "Blue", 9: "Beige",
        10: "Orange", 11: "Yellow", 12: "Red", 13: "Purple", 14: "Brown",
        15: "Olive", 16: "Green", 17: "Gray", 18: "Multi",
    },
    "fabric": {
        -1: "MISSING(-1)", 0: "Canvas", 1: "Crocheted", 2: "Denim", 3: "Fleece",
        4: "Hardshell", 5: "Jersey", 6: "Jersey Lace", 7: "Lace", 8: "Mesh",
        9: "Mesh Jersey", 10: "Rib", 11: "Softshell", 12: "Sweat", 13: "Leather",
    },
    "fastening": {
        -1: "MISSING(-1)", 0: "Zip", 1: "Back closure", 2: "Belt", 3: "Button",
        4: "Covered", 5: "Hook", 6: "Laces", 7: "No closure", 8: "Front", 9: "Others",
    },
    "fit": {
        -1: "MISSING(-1)", 0: "Bootcut", 1: "Flared", 2: "High waist", 3: "Jeggings",
        4: "Large", 5: "Loose", 6: "Low", 7: "Oversize", 8: "Regular", 9: "Skinny",
        10: "Slim", 11: "Small", 12: "Straight", 13: "Tailered", 14: "Tapered",
    },
    "gender": {-1: "MISSING(-1)", 0: "Male", 1: "Female"},
    "neckline": {
        -1: "MISSING(-1)", 0: "Boat", 1: "Backless", 2: "Cache-coeur", 3: "Henley",
        4: "Low v-neck", 5: "Low round", 6: "Off-the-shoulder", 7: "Round",
        8: "Square", 9: "V-neck", 10: "Envelope",
    },
    "pattern": {
        -1: "MISSING(-1)", 0: "Animal", 1: "Burnout", 2: "Camouflage", 3: "Checked",
        4: "Colour gradient", 5: "Colourful", 6: "Floral", 7: "Herringbone", 8: "Marl",
        9: "Paisley", 10: "Photo", 11: "Pinstriped", 12: "Plain", 13: "Polka dot",
        14: "Print", 15: "Striped",
    },
    "pocket": {
        -1: "MISSING(-1)", 0: "Side", 1: "Cargo", 2: "Flap", 3: "Napoleon",
        4: "Belt", 5: "Sleeve", 6: "Zip",
    },
    "sleevelen": {
        -1: "MISSING(-1)", 0: "3/4", 1: "Spaghetti", 2: "Sleeveless", 3: "Elbow",
        4: "Extra Long", 5: "Extra Short", 6: "Long", 7: "Short", 8: "Strapless",
    },
    "sport": {
        -1: "MISSING(-1)", 0: "Skii", 1: "Basketball", 2: "Football", 3: "Golf",
        4: "Handball", 5: "Hiking", 6: "Climbing", 7: "Running", 8: "Snowboard",
        9: "Swim", 10: "Tennis", 11: "Training", 12: "Yoga", 13: "Urban", 14: "Dance",
    },
}


def _min_conf(*confs: str) -> str:
    return RANK_TO_CONF[min(CONF_RANK[c] for c in confs)]


def _map_s100k_article_type(
    raw_cat: str,
    g_folder: str,
    sub_folder: str,
    collar: str,
    fabric: str,
    fastening: str,
    neckline: str,
    sleevelen: str,
    lex_ats: set[str],
    vi_vocab_ats: set[str],
    vi_map_ats: set[str],
    cat_ats: set[str],
) -> Tuple[str, str, str, str]:
    """Return (mapped_articleType, cat_conf, rule_id, reason) using detailed attributes."""
    # 1. Mixed shirt / blouse / tee folders split via collar, fastening, sleevelen, neckline, fabric
    if raw_cat == "Shirt" and g_folder == "Male":
        if collar == "Polo":
            return ("Tshirts", "high", "M4_polo", "Male/4_Shirts nhung collar=Polo -> Tshirts (ao polo)")
        if collar in {"Kent", "Buttondown", "Cutaway", "Turndown", "Shirt"} or fastening == "Button":
            return ("Shirts", "high", "M4_shirt_or_btn", f"Male/4_Shirts co/nut chuan so mi (collar={collar}, fastening={fastening})")
        if collar == "Mandarin":
            return ("Shirts", "medium", "M4_mandarin_nobtn", "Male/4_Shirts collar=Mandarin khong ro nut -> Shirts (medium)")
        if collar == "MISSING(-1)" and fastening == "MISSING(-1)" and fabric in {"Jersey", "Mesh", "Mesh Jersey", "Rib"}:
            return ("Tshirts", "medium", "M4_jersey_tee", f"Male/4_Shirts khong co/nut nhung fabric={fabric} -> Tshirts (medium)")
        return ("Tops/Shirts-ambiguous", "low", "M4_ambig", f"Male/4_Shirts thieu can cu co/nut/vai (collar={collar}, fabric={fabric}) -> Tops/Shirts-ambiguous")

    if raw_cat == "Shirt" and g_folder == "Female":
        if collar == "Polo":
            return ("Tshirts", "high", "F4_polo", "Female/4_Shirts_blouses collar=Polo -> Tshirts")
        if collar in {"Shirt", "Buttondown", "Kent", "Turndown", "Cutaway"}:
            if fastening == "Button":
                return ("Shirts", "high", "F4_shirt_btn", f"Female/4_Shirts_blouses collar={collar} + fastening=Button -> Shirts")
            return ("Shirts", "medium", "F4_shirt_nobtn", f"Female/4_Shirts_blouses collar={collar} khong fastening=Button -> Shirts (medium)")
        if (
            sleevelen in {"Spaghetti", "Strapless"}
            or neckline in {"Off-the-shoulder", "Backless", "Cache-coeur", "Boat"}
            or collar in {"Volant", "Peter Pan"}
            or fastening == "Back closure"
            or fabric in {"Lace", "Jersey Lace", "Crocheted"}
            or (sleevelen == "Sleeveless" and fastening != "Button")
        ):
            return ("Tops", "medium", "F4_feminine_top_cue", f"Female/4_Shirts_blouses co dau hieu ao kieu/blouse ({sleevelen}/{neckline}/{collar}/{fabric}) -> Tops (medium)")
        return ("Tops/Shirts-ambiguous", "low", "F4_ambig", f"Female/4_Shirts_blouses khong tach duoc Shirts vs Tops (collar={collar}, fabric={fabric}, fastening={fastening})")

    if raw_cat == "T-shirt" and g_folder == "Female":
        if collar == "Polo":
            return ("Tshirts", "medium", "F5_polo", "Female/5_T_shirts_tops collar=Polo ha xuong medium (kiem chung 35/40 = 87.5% la co lo/co cao)")
        if collar == "Hood" and sleevelen in {"Long", "Extra Long", "3/4"}:
            return ("Sweatshirts", "medium", "F5_hoodie", f"Female/5_T_shirts_tops collar=Hood + sleeve={sleevelen} -> Sweatshirts (medium)")
        if (
            sleevelen in {"Spaghetti", "Strapless"}
            or neckline in {"Off-the-shoulder", "Backless", "Cache-coeur", "Boat"}
            or collar in {"Volant", "Peter Pan"}
            or fastening == "Back closure"
            or fabric in {"Lace", "Jersey Lace", "Crocheted"}
        ):
            return ("Tops", "high", "F5_strong_top", f"Female/5_T_shirts_tops dau hieu Tops ro ({sleevelen}/{neckline}/{collar}/{fabric}) -> Tops")
        if sleevelen == "Sleeveless":
            return ("Tops", "medium", "F5_sleeveless_top", "Female/5_T_shirts_tops sleevelen=Sleeveless -> Tops (medium)")
        if fabric == "Jersey" and sleevelen in {"Short", "Extra Short"} and collar in {"MISSING(-1)", "Round", "V-neck"} and fastening == "MISSING(-1)":
            return ("Tshirts", "medium", "F5_classic_short_tee", f"Female/5_T_shirts_tops Jersey tay {sleevelen} ha xuong medium (kiem chung 6/40 = 15.0% ao kieu/cowl/lace + ao ho vai)")
        if fabric in {"Jersey", "Mesh", "Mesh Jersey", "Rib"} and collar in {"MISSING(-1)", "Round", "V-neck"} and fastening == "MISSING(-1)":
            return ("Tshirts", "medium", "F5_other_jersey_tee", f"Female/5_T_shirts_tops {fabric} co {collar} tay {sleevelen} -> Tshirts (medium)")
        if fabric == "Sweat" and collar != "Hood" and sleevelen in {"Short", "Extra Short"}:
            return ("Tshirts", "medium", "F5_sweat_short_tee", "Female/5_T_shirts_tops fabric=Sweat tay ngan khong mu -> Tshirts (medium)")
        return ("Tops/Shirts-ambiguous", "low", "F5_ambig", f"Female/5_T_shirts_tops thieu can cu tach Tops vs Tshirts (fabric={fabric}, sleeve={sleevelen}, collar={collar})")

    if raw_cat == "T-shirt" and g_folder == "Male":
        if collar == "Hood" and fabric == "Sweat":
            return ("Sweatshirts", "medium", "M5_hoodie", "Male/5_T_shirts collar=Hood + fabric=Sweat -> Sweatshirts (medium)")
        if collar == "Polo":
            return ("Tshirts", "medium", "M5_polo", "Male/5_T_shirts collar=Polo ha xuong medium (kiem chung 18/33 = 54.5% la co lo/hoodie/crew)")
        if fabric in {"Jersey", "Mesh", "Mesh Jersey", "Rib"} and sleevelen in {"Short", "Extra Short", "MISSING(-1)"}:
            return ("Tshirts", "high", "M5_classic_tee", f"Male/5_T_shirts {fabric} tay {sleevelen} -> Tshirts")
        return ("Tshirts", "medium", "M5_other_tee", f"Male/5_T_shirts (fabric={fabric}, sleeve={sleevelen}) -> Tshirts (medium)")

    # 2. Jumper & Swimming sub-splits
    if raw_cat == "Jumper":
        if collar == "Hood":
            return ("Sweatshirts", "high", "J_hoodie", "6_Jumpers collar=Hood -> Sweatshirts (ao ni co mu)")
        if fabric == "Sweat":
            return ("Sweatshirts", "medium", "J_sweat", "6_Jumpers fabric=Sweat khong ro mu -> Sweatshirts (medium)")
        if fastening == "Zip":
            return ("Sweaters", "low", "J_sweater_zip", "6_Jumpers fastening=Zip -> ha xuong low (kiem chung 35/40 = 87.5% ao khoac the thao/zip)")
        return ("Sweaters", "medium", "J_sweater", "6_Jumpers con lai -> Sweaters (medium vi mot so hoodie thieu nhan Hood)")

    if raw_cat == "Swimming":
        if collar in {"Shawl", "Hood"} or sleevelen in {"Long", "Extra Long", "3/4"} or fastening == "Belt":
            return ("Bath Robe", "medium", "Swim_bathrobe", "10_Swimming (Shawl/Hood hoac tay dai/Belt) -> Bath Robe (pending_lexicon)")
        return ("Swimwear", "medium", "Swim_swimwear", "10_Swimming con lai -> Swimwear (pending_lexicon, gop do boi nam/nu)")

    # 3. Direct 1-to-1 apparel categories
    direct_map = {
        "Coat": "Coats",
        "Jacket": "Jackets",
        "Suit": "Suits",
        "Shorts": "Shorts",
        "Trouser": "Trousers",
        "Jean": "Jeans",
        "Jumpsuit": "Jumpsuit",
        "Pyjamas": "Night suits",
        "Tracksuit Bottoms": "Track Pants",
        "Tracksuit": "Tracksuits",
        "Skirt": "Skirts",
        "Dress": "Dresses",
    }
    target = direct_map[raw_cat]
    if raw_cat == "Coat":
        return (target, "low", "Coat_out_of_schema", "Coats khong co trong catalog(0), lexicon_vi, vi_vocab va vi_map")
    if raw_cat == "Suit":
        return (target, "medium", "Suit_partial_schema", "Suits co trong vi_vocab nhung catalog v2=0 va thieu trong lexicon_vi/vi_map")
    if raw_cat in ("Jumpsuit", "Tracksuit"):
        return (target, "medium", f"{raw_cat}_no_lex", f"{target} (pending_lexicon=True, dense_only cho Pha 7b)")
    if raw_cat == "Tracksuit Bottoms":
        return (target, "medium", "TracksuitBottoms_pending_lex", "Tracksuit Bottoms -> Track Pants (pending_lexicon=True, giu medium/dense_only do tron jogger & legging cho Pha 7b)")

    in_all_4 = (target in lex_ats) and (target in vi_vocab_ats) and (target in vi_map_ats) and (target in cat_ats)
    if in_all_4:
        return (target, "high", f"{raw_cat}_1to1", f"Khop 1-1 {raw_cat} -> {target} tren du 4 bo chuan")
    return (target, "medium", f"{raw_cat}_partial", f"{target} thieu trong mot hoac nhieu tu dien chuan")


def _map_s100k_color_label(raw_color: str) -> Tuple[str, str]:
    """Return (mapped_color_label, note) from raw Shopping100k color string."""
    if raw_color == "MISSING(-1)":
        return ("Unknown", "color.txt = -1 (thieu nhan mau goc)")
    alias = {
        "Navy": "Navy Blue",
        "Gray": "Grey",
        "Light blue": "Blue",
        "Indigo": "Blue",
    }
    target = alias.get(raw_color, raw_color)
    note = f"'{raw_color}' -> '{target}'"
    if raw_color in ("Light blue", "Indigo"):
        note += " (gom vao Blue theo 47 baseColour & Blue family)"
    return (target, note)


def _render_gd4_sheet(
    entries: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    rows: int = 5,
    cols: int = 8,
    cell_w: int = 265,
    cell_h: int = 300,
    max_chars: int = 40,
) -> Path:
    """Render clean, readable contact sheet without truncating GĐ4 multi-attribute labels."""
    pad_top = 56
    w = cols * cell_w
    h = pad_top + rows * cell_h
    canvas = Image.new("RGB", (w, h), (22, 24, 30))
    draw = ImageDraw.Draw(canvas)
    f_sub, f_title = _load_fonts()
    draw.text((16, 16), title, fill=(242, 244, 248), font=f_title)

    img_box_h = cell_h - 60
    for idx, entry in enumerate(entries[: rows * cols]):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, pad_top + r * cell_h
        border = entry.get("border_color", (80, 90, 110))
        draw.rectangle([x0 + 3, y0 + 3, x0 + cell_w - 4, y0 + cell_h - 4], outline=border, width=2)
        try:
            with Image.open(entry["abs_path"]) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                rgb.thumbnail((cell_w - 16, img_box_h - 10))
                ix = x0 + (cell_w - rgb.width) // 2
                iy = y0 + 8 + (img_box_h - 10 - rgb.height) // 2
                canvas.paste(rgb, (ix, iy))
        except Exception:
            draw.text((x0 + 16, y0 + 40), "IMAGE ERR", fill=(235, 90, 90), font=f_sub)

        t1 = str(entry.get("line1", ""))[:max_chars]
        t2 = str(entry.get("line2", ""))[:max_chars]
        draw.text((x0 + 8, y0 + img_box_h + 2), t1, fill=entry.get("text_color", (225, 238, 250)), font=f_sub)
        draw.text((x0 + 8, y0 + img_box_h + 22), t2, fill=(195, 204, 218), font=f_sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def _stratified_sample_200(df_s: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Select 200 Shopping100k samples stratified across all mapped articleTypes."""
    rng = np.random.default_rng(seed)
    n_target = min(200, len(df_s))
    groups = df_s.groupby("articleType", sort=True)
    counts = {k: len(g) for k, g in groups}
    total = sum(counts.values())

    alloc: Dict[str, int] = {k: min(cnt, 4) for k, cnt in counts.items()}
    rem = n_target - sum(alloc.values())
    if rem > 0:
        fracs = {k: (counts[k] / total) * rem for k in counts}
        extras = {k: min(counts[k] - alloc[k], int(np.floor(fracs[k]))) for k in counts}
        for k, ex in extras.items():
            alloc[k] += ex
        left = n_target - sum(alloc.values())
        for k in sorted(counts, key=lambda k: (fracs[k] - np.floor(fracs[k]), counts[k]), reverse=True):
            if left <= 0:
                break
            if alloc[k] < counts[k]:
                alloc[k] += 1
                left -= 1

    parts = [groups.get_group(k).sample(n=min(alloc[k], counts[k]), random_state=seed) for k in sorted(alloc) if alloc[k] > 0]
    out = pd.concat(parts, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def _compute_sharded_pixel_colors(
    df_sh: pd.DataFrame,
    shards_dir: Path,
    centroids_path: Path,
    shard_size: int = 5000,
    workers: int = 10,
) -> Dict[str, Tuple[str, str]]:
    """Run resumable sharded pixel color estimation across all Shopping100k items."""
    shards_dir.mkdir(parents=True, exist_ok=True)
    rows_input = list(zip(df_sh["uid"].astype(str), df_sh["abs_path"].astype(str)))
    n_total = len(rows_input)
    n_shards = (n_total + shard_size - 1) // shard_size
    c_path_str = str(centroids_path)

    result_map: Dict[str, Tuple[str, str]] = {}
    t_start = time.time()
    for s_idx in range(n_shards):
        shard_path = shards_dir / f"gd4_color_shard_{s_idx:03d}.parquet"
        chunk = rows_input[s_idx * shard_size : (s_idx + 1) * shard_size]
        if shard_path.exists():
            sdf = pd.read_parquet(shard_path)
            if len(sdf) == len(chunk) and {"uid", "color_pixel", "px_mode"}.issubset(sdf.columns):
                for r in sdf.itertuples(index=False):
                    result_map[str(r.uid)] = (str(r.color_pixel), str(r.px_mode))
                print(f"  [color shard {s_idx + 1}/{n_shards}] loaded cached ({len(sdf):,} rows)")
                continue

        tasks = [(uid, ap, c_path_str) for uid, ap in chunk]
        shard_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(predict_color_pixel_worker, t) for t in tasks]
            for fut in as_completed(futs):
                shard_rows.append(fut.result())

        sdf = pd.DataFrame(shard_rows)
        sdf.to_parquet(shard_path, index=False)
        for r in sdf.itertuples(index=False):
            result_map[str(r.uid)] = (str(r.color_pixel), str(r.px_mode))
        elapsed = time.time() - t_start
        print(f"  [color shard {s_idx + 1}/{n_shards}] computed {len(sdf):,} rows ({elapsed:.1f}s elapsed)")

    return result_map


def _build_index_v2_comparison(
    repo_root: Path,
    df_all: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare high/medium candidate (articleType x gender) against index_v2 slots (<30)."""
    id_map = json.loads((repo_root / "index_v2" / "id_map_v2.json").read_text(encoding="utf-8"))
    v2_clean = [r for r in id_map if str(r["item_id"]) not in {"43230", "42066"}]
    v2_slots = Counter((str(r["articleType"]), str(r["gender"])) for r in v2_clean)

    eligible = df_all[df_all["serving_eligible"]].copy()
    high_all = eligible[eligible["attr_confidence"] == "high"]
    high_sh = high_all[high_all["source"] == "shopping100k"]
    med_sh = eligible[(eligible["source"] == "shopping100k") & (eligible["attr_confidence"] == "medium") & (eligible["color_conf"] == "high")]

    high_all_slots = Counter(zip(high_all["articleType"], high_all["gender"]))
    high_sh_slots = Counter(zip(high_sh["articleType"], high_sh["gender"]))
    med_sh_colorok_slots = Counter(zip(med_sh["articleType"], med_sh["gender"]))

    lt30_v2 = {k: v for k, v in v2_slots.items() if v < 30}
    filled_by_high_all = []
    filled_by_high_sh = []
    potential_if_lex = []

    for slot, v2_cnt in sorted(lt30_v2.items(), key=lambda kv: (kv[1], kv[0])):
        ha = high_all_slots.get(slot, 0)
        hs = high_sh_slots.get(slot, 0)
        ms = med_sh_colorok_slots.get(slot, 0)
        if ha > 0:
            filled_by_high_all.append({"articleType": slot[0], "gender": slot[1], "v2_cnt": v2_cnt, "high_all_add": ha, "high_sh_add": hs, "new_total": v2_cnt + ha})
        if hs > 0:
            filled_by_high_sh.append({"articleType": slot[0], "gender": slot[1], "v2_cnt": v2_cnt, "high_sh_add": hs})
        if ms > 0:
            potential_if_lex.append({"articleType": slot[0], "gender": slot[1], "v2_cnt": v2_cnt, "med_color_ok_add": ms, "new_total_if_promoted": v2_cnt + ms})

    # Zero-shot in v2 but available in Shopping100k (like Suits, Coats, Bath Robe Men, Swimwear Men)
    zero_v2_s100k = []
    for slot in sorted(set(high_all_slots) | set(med_sh_colorok_slots)):
        if slot not in v2_slots:
            zero_v2_s100k.append({
                "articleType": slot[0],
                "gender": slot[1],
                "high_add": high_all_slots.get(slot, 0),
                "med_color_ok_add": med_sh_colorok_slots.get(slot, 0),
            })

    return {
        "v2_total": len(v2_clean),
        "v2_slot_count": len(v2_slots),
        "v2_lt30_count": len(lt30_v2),
        "v2_slots": v2_slots,
        "high_all_slots": high_all_slots,
        "high_sh_slots": high_sh_slots,
        "filled_by_high_all": filled_by_high_all,
        "filled_by_high_sh": filled_by_high_sh,
        "potential_if_lex": potential_if_lex,
        "zero_v2_s100k": zero_v2_s100k,
    }


def run_gd4(
    config_path: str | Path | None = None,
    merged_dir_override: str | Path | None = None,
    workers: int = 10,
    shard_size: int = 5000,
) -> None:
    """Execute full Phase 7a GD4 pipeline and write report awaiting user approval."""
    t0 = time.time()
    ensure_utf8_stdio()
    cfg = load_config(config_path)
    paths = resolve_phase7a_paths(cfg, merged_dir_override=merged_dir_override)

    seed = int(cfg.get("eval", {}).get("seed", 42))
    eval_p7 = paths["eval_p7"]
    reports_p7 = paths["reports_p7"]
    cs_dir = paths["contact_sheets_dir"]
    shards_dir = paths["shards_dir"]
    eval_p7.mkdir(parents=True, exist_ok=True)
    reports_p7.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHA 7a — GĐ4: ÁNH XẠ SHOPPING100K + TÁCH ATTRIBUTES + KIỂM CHỨNG MÀU")
    print("=" * 72)

    # Remove stale temporary _diag_* sheets if present
    for stale in cs_dir.glob("_diag_*.png"):
        stale.unlink(missing_ok=True)

    lex = json.loads((paths["repo_root"] / "src" / "lexicon_vi.json").read_text(encoding="utf-8"))
    vi_map = json.loads((paths["eval_dir"] / "text2img_test" / "vi_map.json").read_text(encoding="utf-8"))
    cat = pd.read_parquet(paths["catalog_path"])
    fam_map = load_colour_family_map(str(paths["eval_dir"] / "families.json"))

    lex_ats = set(lex["article_types"].keys())
    vi_vocab_ats = set(ARTICLE_TYPE_MAP.keys())
    vi_map_ats = set(vi_map["articleType"].keys())
    cat_ats = set(cat["articleType"].dropna().unique().tolist())

    lex_cols = set(lex["colours"].keys())
    vi_vocab_cols = set(COLOUR_MAP.keys())
    vi_map_cols = set(vi_map["baseColour"].keys())
    cat_cols = set(cat["baseColour"].dropna().unique().tolist())

    df_clean = pd.read_parquet(eval_p7 / "clean_candidates_gd3.parquet")
    df_sh = df_clean[df_clean["source"] == "shopping100k"].copy().reset_index(drop=True)
    df_fa = df_clean[df_clean["source"] == "fashion"].copy().reset_index(drop=True)
    print(f"Ứng viên sạch GĐ3: tổng={len(df_clean):,} (shopping100k={len(df_sh):,}, fashion={len(df_fa):,})")

    # Req 1: Parse ALL 12 Shopping100k Attributes/*.txt files for the 98,174 clean items
    rp_set = set(df_sh["orig_id"].astype(str))
    sh_attr_dir = paths["repo_root"] / "data" / "Shopping100k" / "Attributes"
    attr_names = list(S100K_CODEBOOKS.keys())
    raw_attrs: Dict[str, Dict[str, int]] = {rp: {} for rp in rp_set}
    for an in attr_names:
        for ln in (sh_attr_dir / f"{an}.txt").read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rp, code_str = ln.strip().split()
            if rp in raw_attrs:
                raw_attrs[rp][an] = int(code_str)

    # Save raw attribute statistics across all 12 attributes
    raw_stat_rows: List[Dict[str, Any]] = []
    for an in attr_names:
        cb = S100K_CODEBOOKS[an]
        cnt = Counter(raw_attrs[rp].get(an, -1) for rp in rp_set)
        for code_val, n_val in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            raw_stat_rows.append({
                "attribute": an,
                "raw_code": code_val,
                "raw_label": cb.get(code_val, f"UNKNOWN({code_val})"),
                "count": n_val,
                "pct": round(n_val / len(rp_set) * 100.0, 2),
            })
    df_raw_stats = pd.DataFrame(raw_stat_rows)
    df_raw_stats.to_csv(eval_p7 / "s100k_raw_attribute_stats.csv", index=False, encoding="utf-8")
    print(f"Đã thống kê 12 thuộc tính thô -> eval/phase7/s100k_raw_attribute_stats.csv ({len(df_raw_stats)} giá trị)")

    # Ensure catalog_color_centroids_lab.json exists and run sharded pixel color estimation
    centroids_path = eval_p7 / "catalog_color_centroids_lab.json"
    print("Đang ước lượng màu pixel độc lập (sharded + resumable)...")
    px_map = _compute_sharded_pixel_colors(
        df_sh,
        shards_dir=shards_dir,
        centroids_path=centroids_path,
        shard_size=shard_size,
        workers=workers,
    )

    # Req 2, 3, 4, 5, 6: Build full Shopping100k mapped table
    s100k_rows: List[Dict[str, Any]] = []
    gender_mismatch_cnt = 0
    rule_counter: Dict[Tuple[str, str, str, str], List[Any]] = {}

    for r in df_sh.itertuples(index=False):
        rp = str(r.orig_id)
        parts = rp.split("/")
        g_folder = parts[0]
        sub_folder = parts[1] if len(parts) > 1 else ""
        a = raw_attrs[rp]

        dec = {an: S100K_CODEBOOKS[an].get(a.get(an, -1), "MISSING(-1)") for an in attr_names}
        raw_cat = dec["category"]
        raw_col = dec["color"]
        raw_gen_code = a.get("gender", -1)

        # Gender mapping
        mapped_gen = "Women" if g_folder == "Female" else "Men"
        expected_code = 1 if g_folder == "Female" else 0
        if raw_gen_code != expected_code:
            gender_mismatch_cnt += 1
        gen_conf = "high" if (mapped_gen in GENDER_MAP and raw_gen_code == expected_code) else "low"
        gen_reason = f"Thu muc '{g_folder}' -> '{mapped_gen}' (gender.txt={raw_gen_code})"
        rc_key = ("gender", f"{g_folder} (code={raw_gen_code})", mapped_gen, gen_conf)
        rule_counter.setdefault(rc_key, [0, gen_reason])[0] += 1

        # Category + attribute splitting
        mapped_at, cat_conf, rule_id, cat_reason = _map_s100k_article_type(
            raw_cat=raw_cat,
            g_folder=g_folder,
            sub_folder=sub_folder,
            collar=dec["collar"],
            fabric=dec["fabric"],
            fastening=dec["fastening"],
            neckline=dec["neckline"],
            sleevelen=dec["sleevelen"],
            lex_ats=lex_ats,
            vi_vocab_ats=vi_vocab_ats,
            vi_map_ats=vi_map_ats,
            cat_ats=cat_ats,
        )
        at_key = ("articleType", f"{rule_id}: {raw_cat} | {g_folder}/{sub_folder}", mapped_at, cat_conf)
        rule_counter.setdefault(at_key, [0, cat_reason])[0] += 1

        # Color mapping & pixel cross-check (NEVER overwrite baseColour!)
        color_label, col_note = _map_s100k_color_label(raw_col)
        color_pixel, px_mode = px_map.get(str(r.uid), ("Unknown", "missing"))
        if raw_col == "MISSING(-1)":
            color_conf = "low"
            color_source = "image_estimated"
            col_reason = f"Thieu nhan color=-1; ghi color_pixel='{color_pixel}' (image_estimated), khong dua vao AttributeIndex"
            base_colour = "Unknown"
        else:
            color_source = "attributes"
            base_colour = color_label
            fam_lbl = fam_map.get(color_label, color_label)
            fam_px = fam_map.get(color_pixel, color_pixel)
            if fam_lbl == fam_px:
                color_conf = "high"
                col_reason = f"{col_note}; khop ho mau pixel '{color_pixel}' ({fam_lbl})"
            else:
                color_conf = "low"
                col_reason = f"LECH MAU: nhan='{color_label}' ({fam_lbl}) vs pixel='{color_pixel}' ({fam_px})"
        col_rkey = ("baseColour", f"{raw_col} -> {color_label} ({color_source})", f"color_conf={color_conf}", color_conf)
        rule_counter.setdefault(col_rkey, [0, col_note])[0] += 1

        attr_conf = _min_conf(cat_conf, gen_conf, color_conf)
        reasons = []
        if cat_conf != "high":
            reasons.append(f"[type:{cat_conf}] {cat_reason}")
        if color_conf != "high":
            reasons.append(f"[color:{color_conf}] {col_reason}")
        if not reasons:
            reasons.append("Khop du loai, gioi va kiem chung mau pixel cung ho")

        # Req 5: Temporary Vietnamese title (bm25_eligible only for attr_confidence=='high')
        # and title_vi_display (insert colour ONLY when color_conf=='high').
        at_for_title = None if mapped_at == "Tops/Shirts-ambiguous" else mapped_at
        col_for_title = None if base_colour == "Unknown" else base_colour
        col_for_display = base_colour if (color_conf == "high" and base_colour != "Unknown") else None
        title_vi_drv = generate_vietnamese_title(
            name=str(r.title or ""),
            article_type=at_for_title,
            gender=mapped_gen,
            base_colour=col_for_title,
            brand=None,
        )
        title_vi_disp = generate_vietnamese_title(
            name=str(r.title or ""),
            article_type=at_for_title,
            gender=mapped_gen,
            base_colour=col_for_display,
            brand=None,
        )
        pending_lex = bool(
            mapped_at in {"Swimwear", "Jumpsuit", "Bath Robe", "Tracksuits"}
            or raw_cat == "Tracksuit Bottoms"
        )

        serving_eligible = True
        eval_ext_eligible = (str(r.seen_in_train) == "no")
        bm25_eligible = serving_eligible and (attr_conf == "high")
        proposed_tier = "attribute_index+dense" if attr_conf == "high" else "dense_only"

        s100k_rows.append({
            "uid": str(r.uid),
            "orig_id": rp,
            "source": "shopping100k",
            "abs_path": str(r.abs_path),
            "source_category_raw": f"{raw_cat} | {g_folder}/{sub_folder} (collar={dec['collar']}, fabric={dec['fabric']}, fast={dec['fastening']}, neck={dec['neckline']}, slv={dec['sleevelen']})",
            "raw_category": raw_cat,
            "gender_folder": g_folder,
            "sub_folder": sub_folder,
            "raw_collar": dec["collar"],
            "raw_fabric": dec["fabric"],
            "raw_fastening": dec["fastening"],
            "raw_neckline": dec["neckline"],
            "raw_sleevelen": dec["sleevelen"],
            "raw_gender_code": raw_gen_code,
            "raw_color": raw_col,
            "split_rule_id": rule_id,
            "articleType": mapped_at,
            "gender": mapped_gen,
            "baseColour": base_colour,
            "color_label": color_label,
            "color_pixel": color_pixel,
            "color_conf": color_conf,
            "color_source": color_source,
            "cat_conf": cat_conf,
            "gender_conf": gen_conf,
            "attr_confidence": attr_conf,
            "pending_lexicon": pending_lex,
            "mapping_reason": " ; ".join(reasons),
            "meta_source": "derived",
            "title_vi": title_vi_drv,
            "title_vi_display": title_vi_disp,
            "title_vi_source": "derived",
            "bm25_eligible": bm25_eligible,
            "serving_eligible": serving_eligible,
            "eval_ext_eligible": eval_ext_eligible,
            "proposed_index_tier": proposed_tier,
            "px_mode": px_mode,
        })

    df_s_map = pd.DataFrame(s100k_rows)
    s100k_csv_path = eval_p7 / "s100k_mapping.csv"
    df_s_map.to_csv(s100k_csv_path, index=False, encoding="utf-8")
    print(f"Đã ghi {s100k_csv_path} ({len(df_s_map):,} dòng)")

    # Save rule summary table
    df_rules = pd.DataFrame([
        {
            "field": k[0],
            "raw_key": k[1],
            "target_value": k[2],
            "confidence": k[3],
            "count": v[0],
            "reason": v[1],
        }
        for k, v in sorted(rule_counter.items(), key=lambda kv: (kv[0][0], -kv[1][0], kv[0][1]))
    ])
    df_rules.to_csv(eval_p7 / "s100k_mapping_rules.csv", index=False, encoding="utf-8")

    # Build Fashion 115 rows with uniform schema
    fash_rows: List[Dict[str, Any]] = []
    for r in df_fa.itertuples(index=False):
        at = str(r.article_type or "").strip()
        gn = str(r.gender or "").strip()
        co = str(r.color or "").strip()
        co_valid = bool(co) and co not in {"Unknown", "NA", "nan", "None"}
        co_label = co if co_valid else "Unknown"
        in_scope = bool(r.cfg_in_scope)
        is_cat_phash_dup = bool(r.catalog_phash_dup_flag)
        serving_eligible = in_scope and (not is_cat_phash_dup)
        eval_ext_eligible = (str(r.seen_in_train) == "no")

        cat_conf = "high" if (at and at in cat_ats and at in lex_ats) else ("medium" if at else "low")
        gen_conf = "high" if gn in GENDER_MAP else "low"
        col_conf = "high" if (co_valid and co_label in cat_cols) else "low"
        attr_conf = _min_conf(cat_conf, gen_conf, col_conf)

        if not in_scope:
            reason = f"Ngoai scope (cfg_in_scope=False: {r.category}/{r.sub_category}/{at}) -> serving_eligible=False, eval_only"
        elif is_cat_phash_dup:
            reason = f"Trung pHash<=1 voi catalog ({r.catalog_phash_dup_id}) -> serving_eligible=False, eval_only"
        else:
            reason = f"Du lieu Fashion goc hop le (attr_confidence={attr_conf})"

        t_vi = str(r.title_vi or "").strip()
        t_src = "original_fashion"
        if not t_vi:
            t_vi = generate_vietnamese_title(str(r.title or ""), at, gn, None if co_label == "Unknown" else co_label, str(r.brand or ""))
            t_src = "derived"
        t_vi_disp = t_vi if col_conf == "high" else generate_vietnamese_title(str(r.title or ""), at, gn, None, str(r.brand or ""))
        pending_lex_f = bool(at in {"Swimwear", "Jumpsuit", "Bath Robe", "Tracksuits"})

        proposed_tier = "excluded_eval_only" if not serving_eligible else ("attribute_index+dense" if attr_conf == "high" else "dense_only")
        fash_rows.append({
            "uid": str(r.uid),
            "orig_id": str(r.orig_id),
            "source": "fashion",
            "abs_path": str(r.abs_path),
            "source_category_raw": f"{r.category}/{r.sub_category}/{at}",
            "raw_category": at,
            "gender_folder": "",
            "sub_folder": str(r.sub_category or ""),
            "raw_collar": "",
            "raw_fabric": "",
            "raw_fastening": "",
            "raw_neckline": "",
            "raw_sleevelen": "",
            "raw_gender_code": -1,
            "raw_color": co,
            "split_rule_id": "fashion_original",
            "articleType": at,
            "gender": gn,
            "baseColour": co_label,
            "color_label": co_label,
            "color_pixel": "",
            "color_conf": col_conf,
            "color_source": "fashion_metadata",
            "cat_conf": cat_conf,
            "gender_conf": gen_conf,
            "attr_confidence": attr_conf,
            "pending_lexicon": pending_lex_f,
            "mapping_reason": reason,
            "meta_source": "original_fashion",
            "title_vi": t_vi,
            "title_vi_display": t_vi_disp,
            "title_vi_source": t_src,
            "bm25_eligible": bool(serving_eligible) and (attr_conf == "high"),
            "serving_eligible": serving_eligible,
            "eval_ext_eligible": eval_ext_eligible,
            "proposed_index_tier": proposed_tier,
            "px_mode": "not_run",
        })

    df_f_map = pd.DataFrame(fash_rows)
    df_mapped_all = pd.concat([df_f_map, df_s_map], ignore_index=True)

    # Merge clean GĐ3 quality & lineage columns onto full 98,289 table
    gd3_keep = [
        "uid", "image_path", "split", "seen_in_train", "seen_in_train_reason",
        "short_side", "blur_score", "low_contrast_flag", "banner_flag",
        "cfg_in_scope", "catalog_phash_dup_flag", "catalog_phash_dup_id",
        "internal_sha1_dup_flag", "internal_phash_dup_flag", "legacy_test_protected",
        "sha1", "phash", "title", "brand", "season", "usage", "label_group",
    ]
    df_gd4_full = df_clean[gd3_keep].merge(df_mapped_all, on="uid", how="inner")
    assert len(df_gd4_full) == len(df_clean), f"Row mismatch: {len(df_gd4_full)} != {len(df_clean)}"

    gd4_parquet = eval_p7 / "candidates_mapped_gd4.parquet"
    df_gd4_full.to_parquet(gd4_parquet, index=False)
    print(f"Đã ghi toàn tập 98.289 ảnh -> {gd4_parquet}")

    # Req 7: Contact sheets (~40/sheet, seed=42)
    conf_colors = {"high": (70, 180, 100), "medium": (220, 175, 70), "low": (210, 85, 85)}

    # 7a. Stratified 200 across Shopping100k articleTypes -> 5 sheets x 40
    s200 = _stratified_sample_200(df_s_map, seed=seed)
    s200.to_csv(eval_p7 / "gd4_sample_200.csv", index=False, encoding="utf-8")
    sheets_200 = []
    for p_idx in range(5):
        chunk = s200.iloc[p_idx * 40 : (p_idx + 1) * 40]
        if len(chunk) == 0:
            continue
        entries = []
        for i_loc, r in enumerate(chunk.itertuples(index=False), start=p_idx * 40 + 1):
            entries.append({
                "abs_path": r.abs_path,
                "line1": f"#{i_loc} {r.articleType}|{r.gender}|{r.baseColour}",
                "line2": f"[{r.attr_confidence.upper()}] px:{r.color_pixel} | {r.split_rule_id}",
                "border_color": conf_colors.get(r.attr_confidence, (120, 120, 140)),
                "text_color": (215, 240, 255),
            })
        p_out = cs_dir / f"gd4_sample_200_part{p_idx + 1}.png"
        _render_gd4_sheet(
            entries, p_out,
            title=f"GD4 Stratified 200 Sample (seed={seed}) - Part {p_idx + 1}/5 (Green=HIGH, Yellow=MED, Red=LOW)",
            rows=5, cols=8,
        )
        sheets_200.append(p_out.name)

    # 7b. shirts_tops split diagnostic sheet (40 stratified across M4/F4/F5 outcomes)
    st_pool = df_s_map[df_s_map["raw_category"].isin(["Shirt", "T-shirt"])].copy()
    st_groups = ["Shirts", "Tops", "Tshirts", "Sweatshirts", "Tops/Shirts-ambiguous"]
    st_parts = []
    for g_name in st_groups:
        sub = st_pool[st_pool["articleType"] == g_name]
        if len(sub) > 0:
            take = 8 if g_name != "Sweatshirts" else min(8, len(sub))
            st_parts.append(sub.sample(n=min(take, len(sub)), random_state=seed))
    st_df = pd.concat(st_parts, ignore_index=True)
    if len(st_df) < 40:
        extra = st_pool[~st_pool["uid"].isin(set(st_df["uid"]))].sample(n=40 - len(st_df), random_state=seed)
        st_df = pd.concat([st_df, extra], ignore_index=True)
    st_df = st_df.iloc[:40]
    _render_gd4_sheet(
        [
            {
                "abs_path": r.abs_path,
                "line1": f"{r.articleType} ({r.gender[0]}) [{r.cat_conf.upper()}]",
                "line2": f"{r.split_rule_id}: c={r.raw_collar[:7]} f={r.raw_fastening[:6]}",
                "border_color": conf_colors.get(r.cat_conf, (120, 120, 140)),
            }
            for r in st_df.itertuples(index=False)
        ],
        cs_dir / "gd4_shirts_tops_40.png",
        title=f"GD4 shirts_tops Split Review (40 samples, seed={seed})",
        rows=5, cols=8,
    )

    # 7c. Color mismatch sheet (40 random where color_source=='attributes' and color_conf=='low')
    cmis_pool = df_s_map[(df_s_map["color_source"] == "attributes") & (df_s_map["color_conf"] == "low")]
    cmis_40 = cmis_pool.sample(n=min(40, len(cmis_pool)), random_state=seed)
    _render_gd4_sheet(
        [
            {
                "abs_path": r.abs_path,
                "line1": f"lbl:{r.color_label} vs px:{r.color_pixel}",
                "line2": f"{r.articleType} | {r.orig_id.split('/')[-1]}",
                "border_color": (215, 95, 70),
            }
            for r in cmis_40.itertuples(index=False)
        ],
        cs_dir / "gd4_color_mismatch_40.png",
        title=f"GD4 Color Mismatch Audit (color_conf=LOW, n={len(cmis_pool):,}, seed={seed})",
        rows=5, cols=8,
    )

    # 7d. Color missing (-1) sheet (40 random where color_source=='image_estimated')
    cmiss_pool = df_s_map[df_s_map["color_source"] == "image_estimated"]
    cmiss_40 = cmiss_pool.sample(n=min(40, len(cmiss_pool)), random_state=seed)
    _render_gd4_sheet(
        [
            {
                "abs_path": r.abs_path,
                "line1": f"lbl:Unknown(-1) -> px:{r.color_pixel}",
                "line2": f"{r.articleType} | {r.gender} | {r.cat_conf.upper()}",
                "border_color": (210, 170, 70),
            }
            for r in cmiss_40.itertuples(index=False)
        ],
        cs_dir / "gd4_color_missing_40.png",
        title=f"GD4 Missing Color (-1) Pixel Est Review (n={len(cmiss_pool):,}, seed={seed})",
        rows=5, cols=8,
    )

    # Req 8 & 9: Comparison vs index_v2 (<30 slots) and Markdown Report
    v2_cmp = _build_index_v2_comparison(paths["repo_root"], df_gd4_full)

    # Summaries for report
    sh_conf = df_s_map["attr_confidence"].value_counts().to_dict()
    sh_col_src = df_s_map["color_source"].value_counts().to_dict()
    sh_col_labeled = df_s_map[df_s_map["color_source"] == "attributes"]
    col_match_cnt = int((sh_col_labeled["color_conf"] == "high").sum())
    col_mis_cnt = int((sh_col_labeled["color_conf"] == "low").sum())
    at_conf_cross = pd.crosstab(df_s_map["articleType"], df_s_map["attr_confidence"])

    serv_cnt = df_gd4_full["serving_eligible"].value_counts().to_dict()
    eval_ext_df = df_gd4_full[df_gd4_full["eval_ext_eligible"]]
    eval_ext_src = eval_ext_df["source"].value_counts().to_dict()
    bm25_cnt = df_gd4_full.groupby("source")["bm25_eligible"].sum().to_dict()

    lines = [
        "# Pha 7a — GĐ4: Ánh xạ Metadata Shopping100k, Tách Lớp & Kiểm Chứng Màu (Chờ duyệt)",
        "",
        f"- **Thời gian chạy:** {time.time() - t0:.1f}s | **Seed:** `{seed}` | **Đầu vào:** `clean_candidates_gd3.parquet` (**{len(df_gd4_full):,}** ảnh = `98,174 shopping100k` + `115 fashion`)",
        f"- **Output chính:** `eval/phase7/candidates_mapped_gd4.parquet` ({len(df_gd4_full):,} dòng), `eval/phase7/s100k_mapping.csv` ({len(df_s_map):,} dòng), `eval/phase7/s100k_raw_attribute_stats.csv` (12 thuộc tính đầy đủ).",
        f"- **Kiểm tra `gender` (`Female/Male` folder vs `gender.txt`):** lệch = **{gender_mismatch_cnt}**.",
        "",
        "---",
        "",
        "## 1. Thống kê giá trị thô Shopping100k (98.174 ảnh sạch — tóm tắt trường chính, chi tiết 12 thuộc tính trong `s100k_raw_attribute_stats.csv`)",
        "",
        "| Thuộc tính | Giá trị khác `-1` | Số ảnh `-1` (thiếu) | Phân bố chi tiết (`nhãn: số_ảnh`) |",
        "|---|---:|---:|---|",
    ]
    for an in ["category", "gender", "color", "collar", "fabric", "fastening", "neckline", "sleevelen", "fit", "pattern", "pocket", "sport"]:
        sub = df_raw_stats[df_raw_stats["attribute"] == an]
        n_miss = int(sub[sub["raw_code"] == -1]["count"].sum())
        non_miss = sub[sub["raw_code"] != -1]
        top_str = ", ".join(f"`{r.raw_label}`:{r.count:,}" for r in non_miss.head(8).itertuples(index=False))
        if len(non_miss) > 8:
            top_str += f", ... (+{len(non_miss) - 8} giá trị)"
        lines.append(f"| `{an}` | {len(df_s_map) - n_miss:,} | {n_miss:,} | {top_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Quy tắc tách nhóm `shirts_tops`, `6_Jumpers`, `10_Swimming` & kiểm chứng màu pixel",
        "",
        "### 2a. Phân bố `attr_confidence` trên 98.174 Shopping100k & toàn tập 98.289 ảnh",
        "",
        "| Tập | `high` (`AttributeIndex+dense`) | `medium` (`dense_only`) | `low` (`dense_only`) | Tổng |",
        "|---|---:|---:|---:|---:|",
        f"| `shopping100k` | **{sh_conf.get('high', 0):,}** ({sh_conf.get('high', 0)/len(df_s_map)*100:.2f}%) | **{sh_conf.get('medium', 0):,}** ({sh_conf.get('medium', 0)/len(df_s_map)*100:.2f}%) | **{sh_conf.get('low', 0):,}** ({sh_conf.get('low', 0)/len(df_s_map)*100:.2f}%) | **{len(df_s_map):,}** |",
    ])
    fa_conf = df_f_map[df_f_map["serving_eligible"]]["attr_confidence"].value_counts().to_dict()
    lines.append(
        f"| `fashion` (serving_eligible) | **{fa_conf.get('high', 0):,}** | **{fa_conf.get('medium', 0):,}** | **{fa_conf.get('low', 0):,}** | **{sum(fa_conf.values()):,}** (+10 eval_only) |"
    )

    lines.extend([
        "",
        "### 2b. Kết quả kiểm chứng chéo màu (`color_label` vs `color_pixel` qua `eval/families.json`)",
        "",
        f"- Có nhãn màu gốc (`color >= 0`, `color_source='attributes'`): **{len(sh_col_labeled):,}** ảnh:",
        f"  - Khớp họ màu (`color_conf='high'`): **{col_match_cnt:,}** ({col_match_cnt/len(sh_col_labeled)*100:.2f}%)",
        f"  - Lệch họ màu (`color_conf='low'`): **{col_mis_cnt:,}** ({col_mis_cnt/len(sh_col_labeled)*100:.2f}%) — giữ nguyên `baseColour=color_label`, ghi `color_pixel` để kiểm chứng",
        f"- Thiếu nhãn gốc (`color == -1`, `color_source='image_estimated'`): **{sh_col_src.get('image_estimated', 0):,}** ảnh — giữ `baseColour='Unknown'`, `color_conf='low'`, ghi `color_pixel`, KHÔNG đưa vào `AttributeIndex`.",
        "",
        "### 2c. Bảng phân rã theo `articleType` (`shopping100k`, 98.174 ảnh)",
        "",
        "| `articleType` | `high` | `medium` | `low` | Tổng | `color_conf=high` | Ghi chú |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for at_name in sorted(at_conf_cross.index):
        sub_at = df_s_map[df_s_map["articleType"] == at_name]
        h_n = int(at_conf_cross.loc[at_name].get("high", 0))
        m_n = int(at_conf_cross.loc[at_name].get("medium", 0))
        l_n = int(at_conf_cross.loc[at_name].get("low", 0))
        ch_n = int((sub_at["color_conf"] == "high").sum())
        note = "Đề xuất dense-only (ngoài schema / mơ hồ)" if at_name in {"Coats", "Tops/Shirts-ambiguous"} else ("Chờ từ khóa lexicon_vi" if at_name in {"Bath Robe", "Jumpsuit", "Suits", "Swimwear", "Tracksuits"} else "Đã kiểm định 4 bộ từ điển")
        lines.append(f"| `{at_name}` | {h_n:,} | {m_n:,} | {l_n:,} | **{len(sub_at):,}** | {ch_n:,} | {note} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Tách quy tắc chi tiết `shirts_tops`, `6_Jumpers`, `10_Swimming` (`s100k_mapping_rules.csv`)",
        "",
        "| `rule_id` | Nhóm ánh xạ | `cat_conf` | Số ảnh | Cơ sở thuộc tính |",
        "|---|---|---|---:|---|",
    ])
    split_agg = df_s_map.groupby(["split_rule_id", "articleType", "cat_conf"]).size().reset_index(name="cnt")
    for r in split_agg.sort_values(["split_rule_id"]).itertuples(index=False):
        lines.append(f"| `{r.split_rule_id}` | `{r.articleType}` | **{r.cat_conf}** | {r.cnt:,} | Xem chi tiết trong `s100k_mapping_rules.csv` |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Cờ phạm vi, Đánh giá & BM25 (tất cả 98.289 ảnh)",
        "",
        f"- `serving_eligible = True`: **{serv_cnt.get(True, 0):,}** (`98,174 shopping100k` + `105 fashion`); `False`: **{serv_cnt.get(False, 0):,}** (`6` fashion Personal Care ngoài scope + `4` fashion trùng catalog pHash<=1).",
        f"- `eval_ext_eligible = True` (`seen_in_train == 'no'`): **{len(eval_ext_df):,}** (`fashion={eval_ext_src.get('fashion', 0):,}`, `shopping100k={eval_ext_src.get('shopping100k', 0):,}`).",
        f"- `bm25_eligible = True` (chỉ sinh BM25 từ `title_vi` khi đủ độ tin cậy): `shopping100k={int(bm25_cnt.get('shopping100k', 0)):,}`, `fashion={int(bm25_cnt.get('fashion', 0)):,}`.",
        "",
        "---",
        "",
        f"## 5. Đối chiếu `(articleType × gender)` phần `high` với `index_v2` ({v2_cmp['v2_total']:,} ảnh, trừ 2 Keychain; {v2_cmp['v2_lt30_count']} slot <30)",
        "",
        f"- Số slot hiếm `<30` trong `index_v2` được phần `attr_confidence='high'` lấp thêm: **Shopping100k = {len(v2_cmp['filled_by_high_sh'])}**, **Tổng (gồm Fashion) = {len(v2_cmp['filled_by_high_all'])}**.",
    ])
    if v2_cmp["filled_by_high_all"]:
        lines.append("")
        lines.append("| Slot `(articleType, gender)` | `index_v2` | Thêm (`high`) | Tổng sau bổ sung |")
        lines.append("|---|---:|---:|---:|")
        for row in v2_cmp["filled_by_high_all"]:
            lines.append(f"| `({row['articleType']}, {row['gender']})` | {row['v2_cnt']} | +{row['high_all_add']} | **{row['new_total']}** |")

    lines.extend([
        "",
        f"- **Slot hiếm `<30` tiềm năng nếu bổ sung từ khóa vào `lexicon_vi.json` (`attr_confidence` hiện giữ `medium` do thiếu từ khóa parser, nhưng `color_conf='high'`):**",
        "",
        "| Slot `(articleType, gender)` | `index_v2` | Ảnh Shopping100k (`medium` + màu `high`) | Tổng nếu mở khóa `lexicon_vi` |",
        "|---|---:|---:|---:|",
    ])
    for row in v2_cmp["potential_if_lex"]:
        lines.append(f"| `({row['articleType']}, {row['gender']})` | {row['v2_cnt']} | +{row['med_color_ok_add']:,} | **{row['new_total_if_promoted']:,}** |")

    lines.extend([
        "",
        "### Bảng `(articleType × gender)` phần `high` Shopping100k đối chiếu `index_v2`",
        "",
        "| `(articleType, gender)` | `index_v2` | `shopping100k (high)` | `fashion (high)` | Tổng sau gộp nếu chọn |",
        "|---|---:|---:|---:|---:|",
    ])
    for slot in sorted(v2_cmp["high_all_slots"], key=lambda s: -v2_cmp["high_all_slots"][s]):
        v2_n = v2_cmp["v2_slots"].get(slot, 0)
        sh_n = v2_cmp["high_sh_slots"].get(slot, 0)
        fa_n = v2_cmp["high_all_slots"][slot] - sh_n
        lines.append(f"| `({slot[0]}, {slot[1]})` | {v2_n:,} | +{sh_n:,} | +{fa_n:,} | **{v2_n + sh_n + fa_n:,}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Contact Sheets kiểm chứng bằng mắt (40 ảnh/sheet, `seed=42`) — [DỪNG] chờ duyệt",
        "",
        "Tại `reports/phase7/contact_sheets/`:",
    ])
    for png_name in [*sheets_200, "gd4_shirts_tops_40.png", "gd4_color_mismatch_40.png", "gd4_color_missing_40.png"]:
        lines.append(f"- `{png_name}`")

    rep_path = reports_p7 / "gd4_mapping.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã hoàn tất GĐ4 trong {time.time() - t0:.1f}s -> {rep_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a GD4 Shopping100k metadata mapping & color verification")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=5000)
    args = parser.parse_args()
    run_gd4(
        config_path=args.config,
        merged_dir_override=args.merged_dir,
        workers=args.workers,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
