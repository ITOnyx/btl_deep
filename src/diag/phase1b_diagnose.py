"""
Phase 1b Step 1: Diagnose Deduplication Anomaly & Generate Diagnostic Contact Sheets
SigLIP E-commerce Visual Search (Hoc_sau_ver4)
"""

import os
import sys
import time
import yaml
import pathlib
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_image_path(item_id):
    """Check where the image actually lives (fashion-dataset/images or quarantine/duplicate)."""
    p1 = pathlib.Path(f"data/fashion-dataset/images/{item_id}.jpg")
    if p1.exists():
        return p1
    p2 = pathlib.Path(f"data/_quarantine/duplicate/{item_id}.jpg")
    if p2.exists():
        return p2
    # Check other quarantine folders
    for reason in ["blurry", "low_res", "banner", "corrupt"]:
        p3 = pathlib.Path(f"data/_quarantine/{reason}/{item_id}.jpg")
        if p3.exists():
            return p3
    return None


def render_cluster_sheet(cluster_rows, output_path, title="Duplicate Diagnosis Sheet"):
    """
    Renders contact sheet where each row is a cluster.
    cluster_rows: list of dicts:
      {
         'cluster_id': id,
         'rep_item': dict(id, colour, article, gender, ...),
         'dup_items': list of dict(id, colour, article, gender, ...)
      }
    """
    font_path = "C:/Windows/Fonts/arial.ttf"
    bold_path = "C:/Windows/Fonts/arialbd.ttf"
    try:
        font = ImageFont.truetype(font_path, 11)
        font_bold = ImageFont.truetype(bold_path, 12)
        title_font = ImageFont.truetype(bold_path, 18)
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        title_font = ImageFont.load_default()

    num_rows = len(cluster_rows)
    max_cols = 6  # 1 rep + 5 dups
    cell_w = 230
    cell_h = 280
    header_h = 44
    label_h = 40
    margin = 8

    sheet_w = max_cols * cell_w + (max_cols + 1) * margin
    sheet_h = header_h + num_rows * cell_h + (num_rows + 1) * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(sheet)

    # Title
    draw.text((margin + 10, 12), title, fill=(240, 240, 240), font=title_font)
    img_box_h = cell_h - label_h

    for r_idx, crow in enumerate(cluster_rows):
        y = header_h + margin + r_idx * (cell_h + margin)
        
        rep = crow["rep_item"]
        dups = crow["dup_items"]
        row_items = [("KEPT", rep)] + [("QUAR", d) for d in dups[:max_cols - 1]]

        for c_idx in range(max_cols):
            x = margin + c_idx * (cell_w + margin)
            if c_idx < len(row_items):
                status_tag, item = row_items[c_idx]
                item_id = str(item["id"])
                colour = str(item.get("baseColour", "Unknown"))
                article = str(item.get("articleType", "Unknown"))[:14]
                gender = str(item.get("gender", "Unknown"))

                is_rep = (status_tag == "KEPT")
                border_color = (60, 180, 80) if is_rep else (180, 70, 70)
                bg_color = (40, 50, 45) if is_rep else (48, 42, 42)

                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=bg_color, outline=border_color, width=2 if is_rep else 1)

                img_p = find_image_path(item_id)
                if img_p and img_p.exists():
                    try:
                        with Image.open(img_p) as orig_img:
                            thumb = orig_img.convert("RGB")
                            thumb.thumbnail((cell_w - 6, img_box_h - 6), Image.Resampling.LANCZOS)
                            tw, th = thumb.size
                            tx = x + (cell_w - tw) // 2
                            ty = y + (img_box_h - th) // 2
                            sheet.paste(thumb, (tx, ty))
                    except Exception as e:
                        draw.text((x + 10, y + img_box_h // 2), f"Err: {str(e)[:10]}", fill=(255, 100, 100), font=font)
                else:
                    draw.text((x + 10, y + img_box_h // 2), "File Missing", fill=(255, 100, 100), font=font)

                # Label box
                label_y = y + img_box_h
                label_bg = (20, 35, 25) if is_rep else (30, 20, 20)
                draw.rectangle([x, label_y, x + cell_w, y + cell_h], fill=label_bg)
                tag_str = f"[{status_tag}] ID: {item_id}"
                detail_str = f"{colour} | {gender}"
                type_str = f"{article}"
                draw.text((x + 4, label_y + 2), tag_str, fill=(100, 255, 120) if is_rep else (255, 130, 130), font=font_bold)
                draw.text((x + 4, label_y + 16), detail_str, fill=(230, 230, 230), font=font)
                draw.text((x + 4, label_y + 27), type_str, fill=(180, 200, 220), font=font)
            else:
                # Empty slot
                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(35, 35, 40), outline=(50, 50, 55))

    sheet.save(output_path, "PNG")
    print(f"[Contact Sheet] Saved: {output_path}")


def main():
    t_start = time.time()
    config = load_config("config.yaml")
    seed = config["eval"].get("seed", 42)
    np.random.seed(seed)

    reports_dir = pathlib.Path("reports")
    contact_sheets_dir = reports_dir / "contact_sheets"
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    metrics_parquet = reports_dir / "quality_metrics.parquet"
    styles_csv = pathlib.Path("data/fashion-dataset/styles.csv")
    quarantine_csv = reports_dir / "quarantine_log.csv"
    catalog_parquet = pathlib.Path("data/catalog/catalog.parquet")

    # Load data
    df_metrics = pd.read_parquet(metrics_parquet)
    df_metrics["id"] = df_metrics["id"].astype(str)

    df_styles = pd.read_csv(styles_csv, on_bad_lines="skip")
    df_styles["id_str"] = df_styles["id"].astype(str)
    df_styles = df_styles.drop(columns=["id"])

    df = df_metrics.merge(df_styles, left_on="id", right_on="id_str", how="inner")
    df_q = pd.read_csv(quarantine_csv)
    df_cat = pd.read_parquet(catalog_parquet)

    print("==================================================")
    print("STEP 1: CLUSTER STATISTICS & DIAGNOSIS")
    print("==================================================")
    dup_q = df_q[df_q["reason"] == "duplicate"].copy()
    dup_q_ids = set(dup_q["item_id"].astype(str))
    print(f"Total quarantined as duplicate in quarantine_log.csv: {len(dup_q_ids)}")

    # Groups formed in Phase 1
    clusters = df[df["dup_group_id"] != -1].groupby("dup_group_id")
    n_clusters = len(clusters)
    cluster_sizes = clusters.size().values

    p50 = np.median(cluster_sizes)
    p95 = np.percentile(cluster_sizes, 95)
    c_min = np.min(cluster_sizes)
    c_max = np.max(cluster_sizes)

    print(f"Total clusters: {n_clusters}")
    print(f"Cluster sizes: min={c_min}, median={p50:.1f}, p95={p95:.1f}, max={c_max}")

    # Top 10 largest clusters
    top10_series = clusters.size().sort_values(ascending=False).head(10)
    print("\nTop 10 largest clusters:")
    top10_info = []
    for rank, (gid, size) in enumerate(top10_series.items(), 1):
        grp = clusters.get_group(gid)
        n_col = grp["baseColour"].nunique()
        n_art = grp["articleType"].nunique()
        n_gen = grp["gender"].nunique()
        top10_info.append({
            "rank": rank,
            "group_id": gid,
            "size": size,
            "n_colors": n_col,
            "n_articles": n_art,
            "n_genders": n_gen,
        })
        print(f" #{rank} Group {gid}: size={size}, {n_col} colours, {n_art} articleTypes, {n_gen} genders")

    # In each cluster, analyze differences vs representative
    # Representative was the one with max short_side, max blur_score, min id among surviving
    print("\nAnalyzing within-cluster diversity...")
    clusters_with_diff_color = 0
    clusters_with_diff_article = 0
    clusters_with_diff_gender = 0
    clusters_with_any_diff = 0

    wrongly_quarantined_count = 0
    wrongly_quarantined_diff_color = 0
    wrongly_quarantined_diff_article = 0
    wrongly_quarantined_diff_gender = 0

    cluster_data_list = []

    for gid, grp in clusters:
        # Determine representative (same logic as phase 1)
        # Note: in Phase 1, rep was chosen from items not filtered by low_res/blurry/banner
        # Filter out those that had low_res/blurry/banner
        valid_in_grp = grp[
            (grp["short_side"] >= 224) &
            (grp["blur_score"] >= 12.0) &
            (~grp["banner_flag"])
        ]
        if len(valid_in_grp) == 0:
            continue

        sorted_grp = valid_in_grp.sort_values(
            by=["short_side", "blur_score", "id"],
            ascending=[False, False, True]
        )
        rep = sorted_grp.iloc[0]
        rep_color = rep["baseColour"]
        rep_article = rep["articleType"]
        rep_gender = rep["gender"]

        # Check duplicates in this cluster that were quarantined as duplicate
        dups = sorted_grp.iloc[1:]
        
        has_diff_color = (dups["baseColour"] != rep_color).any()
        has_diff_article = (dups["articleType"] != rep_article).any()
        has_diff_gender = (dups["gender"] != rep_gender).any()
        has_any_diff = has_diff_color or has_diff_article or has_diff_gender

        if has_diff_color:
            clusters_with_diff_color += 1
        if has_diff_article:
            clusters_with_diff_article += 1
        if has_diff_gender:
            clusters_with_diff_gender += 1
        if has_any_diff:
            clusters_with_any_diff += 1

        for _, d in dups.iterrows():
            d_id = str(d["id"])
            if d_id in dup_q_ids:
                is_diff_c = (d["baseColour"] != rep_color)
                is_diff_a = (d["articleType"] != rep_article)
                is_diff_g = (d["gender"] != rep_gender)
                if is_diff_c or is_diff_a or is_diff_g:
                    wrongly_quarantined_count += 1
                    if is_diff_c:
                        wrongly_quarantined_diff_color += 1
                    if is_diff_a:
                        wrongly_quarantined_diff_article += 1
                    if is_diff_g:
                        wrongly_quarantined_diff_gender += 1

        cluster_data_list.append({
            "cluster_id": gid,
            "size": len(grp),
            "rep_item": rep.to_dict(),
            "dup_items": [r.to_dict() for _, r in dups.iterrows()],
            "has_diff_color": has_diff_color,
            "has_any_diff": has_any_diff,
        })

    total_valid_clusters = len(cluster_data_list)
    print(f"\n--- DIVERSITY METRICS ({total_valid_clusters} clusters) ---")
    print(f"Clusters with different baseColour: {clusters_with_diff_color} ({clusters_with_diff_color/total_valid_clusters*100:.1f}%)")
    print(f"Clusters with different articleType: {clusters_with_diff_article} ({clusters_with_diff_article/total_valid_clusters*100:.1f}%)")
    print(f"Clusters with different gender: {clusters_with_diff_gender} ({clusters_with_diff_gender/total_valid_clusters*100:.1f}%)")
    print(f"Clusters with ANY difference: {clusters_with_any_diff} ({clusters_with_any_diff/total_valid_clusters*100:.1f}%)")

    print(f"\n--- WRONGLY QUARANTINED ESTIMATE (out of {len(dup_q_ids)} quarantined as duplicate) ---")
    print(f"Images quarantined that differ in colour, articleType, or gender from kept representative: {wrongly_quarantined_count} ({wrongly_quarantined_count/len(dup_q_ids)*100:.1f}%)")
    print(f" - Differ in baseColour: {wrongly_quarantined_diff_color}")
    print(f" - Differ in articleType: {wrongly_quarantined_diff_article}")
    print(f" - Differ in gender: {wrongly_quarantined_diff_gender}")

    # Impact on catalog categories
    print("\n--- IMPACT ON CATEGORIES (articleType x gender < 30) ---")
    # Current catalog
    current_groups = df_cat.groupby(["articleType", "gender"]).size().reset_index(name="count")
    n_under30_current = int((current_groups["count"] < 30).sum())

    # Simulated restored catalog:
    # Add back all wrongly quarantined items to catalog
    # Let's see: items currently in catalog + wrongly quarantined items
    simulated_cat = pd.concat([
        df_cat[["articleType", "gender"]],
        df[(df["id"].isin(dup_q_ids)) & (df["id"].apply(lambda x: True))][["articleType", "gender"]].head(wrongly_quarantined_count)
    ])
    # More precisely: take df_styles for all wrongly quarantined IDs
    wrong_ids_set = set()
    for c in cluster_data_list:
        rep_c = c["rep_item"]["baseColour"]
        rep_a = c["rep_item"]["articleType"]
        rep_g = c["rep_item"]["gender"]
        for d in c["dup_items"]:
            if str(d["id"]) in dup_q_ids:
                if d["baseColour"] != rep_c or d["articleType"] != rep_a or d["gender"] != rep_g:
                    wrong_ids_set.add(str(d["id"]))

    simulated_cat_df = pd.concat([
        df_cat[["articleType", "gender"]],
        df_styles[df_styles["id_str"].isin(wrong_ids_set)][["articleType", "gender"]]
    ])
    sim_groups = simulated_cat_df.groupby(["articleType", "gender"]).size().reset_index(name="count")
    n_under30_sim = int((sim_groups["count"] < 30).sum())

    print(f"articleType x gender < 30 items currently: {n_under30_current}")
    print(f"articleType x gender < 30 items if wrongly quarantined restored: {n_under30_sim} (reduction of {n_under30_current - n_under30_sim} depleted groups)")

    # Step 4: Contact sheets
    print("\n==================================================")
    print("STEP 4: GENERATING DIAGNOSTIC CONTACT SHEETS")
    print("==================================================")
    # Sort cluster_data_list by size descending
    clusters_by_size = sorted(cluster_data_list, key=lambda c: c["size"], reverse=True)

    # 1. 10 Largest clusters: 2 sheets (6 clusters + 4 clusters)
    top10_clusters = clusters_by_size[:10]
    render_cluster_sheet(
        top10_clusters[:6],
        contact_sheets_dir / "dup_check_top_1.png",
        title="Top Largest Duplicate Clusters (Sheet 1/2: Clusters #1 - #6)"
    )
    render_cluster_sheet(
        top10_clusters[6:10],
        contact_sheets_dir / "dup_check_top_2.png",
        title="Top Largest Duplicate Clusters (Sheet 2/2: Clusters #7 - #10)"
    )

    # 2. 30 Random clusters: 5 sheets (6 clusters each)
    # Filter for clusters that have > 1 item in dups to show meaningful rows
    multi_dup_clusters = [c for c in cluster_data_list if len(c["dup_items"]) >= 1]
    import random
    random.seed(seed)
    sampled_30 = random.sample(multi_dup_clusters, min(30, len(multi_dup_clusters)))

    for s_idx in range(5):
        chunk = sampled_30[s_idx * 6 : (s_idx + 1) * 6]
        sheet_path = contact_sheets_dir / f"dup_check_rand_{s_idx + 1}.png"
        render_cluster_sheet(
            chunk,
            sheet_path,
            title=f"Random Duplicate Clusters Sample (Sheet {s_idx + 1}/5)"
        )

    print("\nAll 7 diagnostic contact sheets generated successfully in reports/contact_sheets/")
    print(f"Time taken: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
