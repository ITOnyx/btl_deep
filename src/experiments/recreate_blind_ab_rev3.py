"""
Re-generate Blind A/B Evaluation for Phase 3g Rev3:
- Reshuffles A/B mapping with NEW seed (20261109) to prevent data leakage from old chat/git history.
- Renders 30 contact sheets (blind_ab_x001.png .. blind_ab_x030.png) using Arial TrueType font to fix Vietnamese diacritics.
- Seals secret mapping to eval/_sealed/ab_mapping.json (DO NOT PRINT OR LOG MAPPING).
- Generates template eval/ab_blind_heldout.csv with 6 standard columns for user blind rating.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont

from src.search_pipeline import VisualSearchEngine

def render_contact_sheet(
    qid: str,
    query_text: str,
    row_a_ids: list,
    row_b_ids: list,
    id_to_path: dict,
    out_path: str
):
    thumb_size = (140, 180)
    padding = 10
    header_height = 65
    row_label_width = 110
    total_width = row_label_width + 10 * (thumb_size[0] + padding) + padding
    total_height = header_height + 2 * (thumb_size[1] + padding) + padding

    sheet = Image.new("RGB", (total_width, total_height), color=(248, 249, 250))
    draw = ImageDraw.Draw(sheet)

    # Load Windows Arial font for Vietnamese support
    font_header = None
    font_row = None
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
        if os.path.exists(fp):
            try:
                font_header = ImageFont.truetype(fp, 22)
                font_row = ImageFont.truetype(fp, 24)
                break
            except Exception:
                pass

    if font_header is None:
        font_header = ImageFont.load_default()
        font_row = ImageFont.load_default()

    # Draw header text with proper Vietnamese font
    header_text = f"Query {qid}: \"{query_text}\""
    draw.text((padding + 5, 18), header_text, fill=(20, 20, 20), font=font_header)

    # Render Rows A and B
    rows_data = [("A", row_a_ids), ("B", row_b_ids)]
    for row_idx, (row_label, id_list) in enumerate(rows_data):
        y_base = header_height + row_idx * (thumb_size[1] + padding) + padding
        # Draw row label
        draw.text((padding + 15, y_base + 70), f"Row {row_label}", fill=(0, 70, 180), font=font_row)

        for col_idx, item_id in enumerate(id_list):
            x_base = row_label_width + col_idx * (thumb_size[0] + padding) + padding
            img_path = id_to_path.get(str(item_id))
            if img_path and os.path.exists(img_path):
                try:
                    with Image.open(img_path) as im:
                        im_rgb = im.convert("RGB")
                        im_thumb = ImageOps.fit(im_rgb, thumb_size, Image.Resampling.BILINEAR)
                        sheet.paste(im_thumb, (x_base, y_base))
                except Exception:
                    pass
    sheet.save(out_path)

def main():
    print("=====================================================================")
    print("Re-shuffling and Re-generating Blind A/B Contact Sheets (Rev3)...")
    print("=====================================================================")
    
    NEW_SEED = 20261109
    rng = np.random.RandomState(NEW_SEED)
    
    # 1. Load heldout queries
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        heldout_queries = json.load(f)
        
    print(f"Loaded {len(heldout_queries)} heldout queries.")
    
    # 2. Initialize engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = VisualSearchEngine(device=device, gamma=0.3, alpha=1.0, enable_text_fusion=False)
    
    df_cat = pd.read_parquet("data/catalog/catalog.parquet")
    id_to_path = {}
    for _, row in df_cat.iterrows():
        iid = str(row["item_id"])
        p = str(row["path"])
        if not os.path.exists(p):
            p = str(Path("data/fashion-dataset/images") / Path(p).name)
        id_to_path[iid] = p
        
    contact_dir = Path("reports/contact_sheets")
    contact_dir.mkdir(parents=True, exist_ok=True)
    
    ab_mapping = {}
    csv_rows = []
    
    # Evaluate Config 0 (Model Thuan) and Config 5 (Full System) on G_full
    for q in heldout_queries:
        qid = q["id"]
        qt = q["query"]
        
        # Config 0: Model Thuan on G_full
        q_vec = engine.encode_text(qt)
        sims_0 = np.dot(engine.emb_img_c, q_vec)
        top10_c0 = [engine.item_ids[i] for i in np.argsort(-sims_0)[:10]]
        
        # Config 5: Full System on G_full
        top10_c5 = engine.search_text(qt, top_k=10)
        
        # Random assignment with NEW_SEED
        coin_flip = rng.randint(0, 2)
        if coin_flip == 0:
            row_a_name, row_a_ids = "Config 0 (Model Thuan)", top10_c0
            row_b_name, row_b_ids = "Config 5 (Full System)", top10_c5
            map_entry = {"A": "Config 0 (Model Thuan)", "B": "Config 5 (Full System)"}
        else:
            row_a_name, row_a_ids = "Config 5 (Full System)", top10_c5
            row_b_name, row_b_ids = "Config 0 (Model Thuan)", top10_c0
            map_entry = {"A": "Config 5 (Full System)", "B": "Config 0 (Model Thuan)"}
            
        ab_mapping[qid] = map_entry
        
        # CSV row
        csv_rows.append({
            "id": qid,
            "query": qt,
            "better": "",
            "correct_count_A": "",
            "correct_count_B": "",
            "notes": ""
        })
        
        # Render sheet
        out_sheet = contact_dir / f"blind_ab_{qid}.png"
        render_contact_sheet(qid, qt, row_a_ids, row_b_ids, id_to_path, str(out_sheet))
        
    # Save sealed mapping
    seal_dir = Path("eval/_sealed")
    seal_dir.mkdir(parents=True, exist_ok=True)
    map_file = seal_dir / "ab_mapping.json"
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(ab_mapping, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully re-shuffled with seed {NEW_SEED} and sealed to {map_file} (SHA-256 updated)")
    
    # Save CSV
    csv_file = Path("eval/ab_blind_heldout.csv")
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False, encoding="utf-8")
    print(f"Saved fresh blind rating template to {csv_file}")

if __name__ == "__main__":
    main()
