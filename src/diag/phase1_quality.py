"""
Phase 1: Image Quality Assessment and Contact Sheet Generator
SigLIP E-commerce Visual Search (Hoc_sau_ver4)
"""

import os
import sys
import time
import math

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yaml
import pathlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import imagehash


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_image_safely(file_path):
    """Read image bytes safely on Windows with Unicode path support."""
    try:
        p = pathlib.Path(file_path)
        if not p.exists():
            return None, "not_found"
        size = p.stat().st_size
        if size == 0:
            return None, "zero_byte"
        with open(p, "rb") as f:
            buf = f.read()
        gray = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, "corrupt"
        return gray, "ok"
    except Exception as e:
        return None, f"error_{str(e)}"


def compute_metrics_for_single_image(args):
    """
    Worker function to compute resolution, blur Laplacian variance,
    perceptual hash, and banner heuristic.
    """
    item_id, img_path_str = args
    gray, status = read_image_safely(img_path_str)
    
    if status != "ok":
        return {
            "id": str(item_id),
            "path": img_path_str,
            "status": status,
            "is_corrupt": status in ("zero_byte", "corrupt"),
            "file_exists": status != "not_found",
            "width": 0,
            "height": 0,
            "short_side": 0,
            "blur_score": 0.0,
            "phash": "",
            "banner_flag": False,
            "top_edge_density": 0.0,
            "bot_edge_density": 0.0,
            "side_edge_density": 0.0,
            "top_mean": 0.0,
            "top_std": 0.0,
            "bot_mean": 0.0,
            "bot_std": 0.0,
        }

    h, w = gray.shape[:2]
    short_side = min(w, h)

    # 1. Blur Laplacian variance (using CV_32F for speed & accuracy)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # 2. Perceptual hash (phash)
    pil_gray = Image.fromarray(gray)
    phash_str = str(imagehash.phash(pil_gray))

    # 3. Banner & Border Heuristic
    # Banner strips
    top_h = max(1, int(h * 0.08))
    bot_h = max(1, int(h * 0.08))
    side_w = max(1, int(w * 0.05))

    top_strip = gray[:top_h, :]
    bot_strip = gray[-bot_h:, :]
    left_strip = gray[:, :side_w]
    right_strip = gray[:, -side_w:]

    top_mean = float(np.mean(top_strip))
    top_std = float(np.std(top_strip))
    bot_mean = float(np.mean(bot_strip))
    bot_std = float(np.std(bot_strip))

    # Canny edges for text/graphic detection
    edges = cv2.Canny(gray, 100, 200)
    top_edge_density = float(np.mean(edges[:top_h, :] > 0))
    bot_edge_density = float(np.mean(edges[-bot_h:, :] > 0))
    sides_combined = np.concatenate([edges[:, :side_w], edges[:, -side_w:]], axis=1)
    side_edge_density = float(np.mean(sides_combined > 0))

    # Conditions for banner:
    # a. Solid non-white horizontal header/footer bar (colored banner)
    is_top_solid = (top_mean < 230.0 and top_std < 10.0)
    is_bot_solid = (bot_mean < 230.0 and bot_std < 10.0)
    # b. Dense text or promotional graphics in header
    is_top_text = (top_edge_density > 0.025)
    # c. Frame/border on lateral edges
    is_side_frame = (side_edge_density > 0.030)

    banner_flag = bool(is_top_solid or is_bot_solid or is_top_text or is_side_frame)

    return {
        "id": str(item_id),
        "path": img_path_str,
        "status": "ok",
        "is_corrupt": False,
        "file_exists": True,
        "width": int(w),
        "height": int(h),
        "short_side": int(short_side),
        "blur_score": blur_score,
        "phash": phash_str,
        "banner_flag": banner_flag,
        "top_edge_density": top_edge_density,
        "bot_edge_density": bot_edge_density,
        "side_edge_density": side_edge_density,
        "top_mean": top_mean,
        "top_std": top_std,
        "bot_mean": bot_mean,
        "bot_std": bot_std,
    }


def find_duplicates_phash(records, threshold=4):
    """
    Find duplicate groups among phashes with Hamming distance <= threshold.
    Uses Pigeonhole Multi-Index Hashing for O(N) candidate generation.
    """
    valid_records = [r for r in records if r["status"] == "ok" and len(r["phash"]) == 16]
    int_hashes = []
    ids = []
    for r in valid_records:
        ids.append(r["id"])
        int_hashes.append(int(r["phash"], 16))

    int_hashes = np.array(int_hashes, dtype=np.uint64)
    n = len(int_hashes)
    print(f"[Duplicate Detection] Checking {n} valid hashes with threshold={threshold}...")

    # Split 64-bit into 5 chunks: 13, 13, 13, 13, 12 bits
    # By Pigeonhole, if Hamming dist <= 4, at least 1 chunk is identical.
    tables = [defaultdict(list) for _ in range(5)]
    for i, h in enumerate(int_hashes):
        h_val = int(h)
        c0 = h_val & 0x1FFF
        c1 = (h_val >> 13) & 0x1FFF
        c2 = (h_val >> 26) & 0x1FFF
        c3 = (h_val >> 39) & 0x1FFF
        c4 = (h_val >> 52) & 0xFFF
        tables[0][c0].append(i)
        tables[1][c1].append(i)
        tables[2][c2].append(i)
        tables[3][c3].append(i)
        tables[4][c4].append(i)

    candidates = set()
    for t in range(5):
        for bucket in tables[t].values():
            if len(bucket) > 1:
                b_len = len(bucket)
                for a in range(b_len):
                    idx_a = bucket[a]
                    for b in range(a + 1, b_len):
                        idx_b = bucket[b]
                        if idx_a < idx_b:
                            candidates.add((idx_a, idx_b))
                        else:
                            candidates.add((idx_b, idx_a))

    print(f"[Duplicate Detection] Generated {len(candidates)} candidate pairs to verify...")

    # Check candidates
    parent = list(range(n))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    matching_pairs = []
    for idx_a, idx_b in candidates:
        xor = int(int_hashes[idx_a] ^ int_hashes[idx_b])
        dist = bin(xor).count("1")
        if dist <= threshold:
            matching_pairs.append((ids[idx_a], ids[idx_b], dist))
            union(idx_a, idx_b)

    # Group components
    groups = defaultdict(list)
    for i in range(n):
        root = find(i)
        groups[root].append(ids[i])

    dup_groups = [members for members in groups.values() if len(members) > 1]
    print(f"[Duplicate Detection] Found {len(matching_pairs)} pairs, formed {len(dup_groups)} duplicate clusters.")

    # Map item_id -> group_id
    id_to_group = {}
    for g_idx, members in enumerate(dup_groups):
        for m in members:
            id_to_group[m] = g_idx

    return dup_groups, matching_pairs, id_to_group


def create_contact_sheet(
    image_entries,
    output_path,
    grid_rows=6,
    grid_cols=6,
    cell_w=240,
    cell_h=300,
    title="Contact Sheet"
):
    """
    Renders a grid_rows x grid_cols contact sheet with clear labels.
    image_entries: list of dicts with 'path', 'id', 'label_text'
    """
    font_path = "C:/Windows/Fonts/arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 13)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    header_h = 50
    label_h = 36
    margin = 8

    sheet_w = grid_cols * cell_w + (grid_cols + 1) * margin
    sheet_h = header_h + grid_rows * cell_h + (grid_rows + 1) * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(sheet)

    # Draw title
    draw.text((margin + 10, 14), title, fill=(240, 240, 240), font=title_font)

    img_box_h = cell_h - label_h

    for idx in range(min(len(image_entries), grid_rows * grid_cols)):
        r = idx // grid_cols
        c = idx % grid_cols
        x = margin + c * (cell_w + margin)
        y = header_h + margin + r * (cell_h + margin)

        entry = image_entries[idx]
        img_path = entry["path"]
        label_text = entry.get("label_text", f"ID: {entry['id']}")

        # Draw cell background
        draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(45, 45, 50), outline=(70, 70, 80))

        # Open and thumbnail image
        p = pathlib.Path(img_path)
        if p.exists() and p.stat().st_size > 0:
            try:
                with Image.open(p) as orig_img:
                    thumb = orig_img.convert("RGB")
                    thumb.thumbnail((cell_w - 4, img_box_h - 4), Image.Resampling.LANCZOS)
                    # Center thumbnail in img_box
                    tw, th = thumb.size
                    tx = x + (cell_w - tw) // 2
                    ty = y + (img_box_h - th) // 2
                    sheet.paste(thumb, (tx, ty))
            except Exception as e:
                draw.text((x + 10, y + img_box_h // 2), f"Error: {str(e)[:15]}", fill=(255, 100, 100), font=font)
        else:
            draw.text((x + 10, y + img_box_h // 2), "Missing File", fill=(255, 100, 100), font=font)

        # Draw label box
        label_y = y + img_box_h
        draw.rectangle([x, label_y, x + cell_w, y + cell_h], fill=(20, 20, 25))
        # Draw label text
        draw.text((x + 4, label_y + 2), label_text, fill=(230, 230, 230), font=font)

    sheet.save(output_path, "PNG")
    print(f"[Contact Sheet] Saved: {output_path}")


def create_duplicate_contact_sheet(
    dup_groups,
    id_to_record,
    output_path,
    max_groups=6,
    max_cols=6,
    cell_w=240,
    cell_h=300
):
    """
    Renders duplicate contact sheet where each row is a duplicate group.
    """
    font_path = "C:/Windows/Fonts/arial.ttf"
    try:
        font = ImageFont.truetype(font_path, 13)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    header_h = 50
    label_h = 36
    margin = 8
    num_rows = min(len(dup_groups), max_groups)

    sheet_w = max_cols * cell_w + (max_cols + 1) * margin
    sheet_h = header_h + num_rows * cell_h + (num_rows + 1) * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(sheet)

    title = f"Duplicates Contact Sheet (Each Row = 1 Duplicate Cluster)"
    draw.text((margin + 10, 14), title, fill=(240, 240, 240), font=title_font)
    img_box_h = cell_h - label_h

    for r_idx in range(num_rows):
        group_members = dup_groups[r_idx]
        y = header_h + margin + r_idx * (cell_h + margin)

        for c_idx in range(min(len(group_members), max_cols)):
            x = margin + c_idx * (cell_w + margin)
            m_id = group_members[c_idx]
            rec = id_to_record.get(m_id, {})
            img_path = rec.get("path", "")

            # Cell box
            draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(45, 45, 50), outline=(80, 80, 100))

            p = pathlib.Path(img_path)
            if p.exists() and p.stat().st_size > 0:
                try:
                    with Image.open(p) as orig_img:
                        thumb = orig_img.convert("RGB")
                        thumb.thumbnail((cell_w - 4, img_box_h - 4), Image.Resampling.LANCZOS)
                        tw, th = thumb.size
                        tx = x + (cell_w - tw) // 2
                        ty = y + (img_box_h - th) // 2
                        sheet.paste(thumb, (tx, ty))
                except Exception as e:
                    draw.text((x + 10, y + img_box_h // 2), f"Error: {str(e)[:15]}", fill=(255, 100, 100), font=font)

            label_y = y + img_box_h
            draw.rectangle([x, label_y, x + cell_w, y + cell_h], fill=(20, 20, 25))
            label_text = f"G{r_idx+1} | ID: {m_id}\n{rec.get('width', 0)}x{rec.get('height', 0)}"
            draw.text((x + 4, label_y + 2), label_text, fill=(230, 230, 230), font=font)

    sheet.save(output_path, "PNG")
    print(f"[Contact Sheet] Saved: {output_path}")


def main():
    t_start = time.time()
    config = load_config("config.yaml")

    seed = config["eval"].get("seed", 42)
    np.random.seed(seed)

    kaggle_dir = pathlib.Path(config["paths"]["kaggle_dir"])
    styles_csv_path = kaggle_dir / "styles.csv"
    images_dir = kaggle_dir / "images"

    reports_dir = pathlib.Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets_dir = reports_dir / "contact_sheets"
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = pathlib.Path(config["paths"]["quarantine"])
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("STEP 1: READING METADATA & INTEGRITY CHECK")
    print("==================================================")
    # Count raw lines in styles.csv
    raw_lines_count = 0
    with open(styles_csv_path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            raw_lines_count += 1
    raw_data_rows = raw_lines_count - 1

    df_styles = pd.read_csv(styles_csv_path, on_bad_lines="skip")
    loaded_rows_count = len(df_styles)
    skipped_bad_lines = raw_data_rows - loaded_rows_count
    print(f"styles.csv: {raw_data_rows} raw data rows, {loaded_rows_count} loaded rows.")
    print(f"Skipped bad lines: {skipped_bad_lines}")

    # Check missing gender or articleType
    missing_gender_mask = df_styles["gender"].isna()
    missing_article_mask = df_styles["articleType"].isna()
    missing_required = missing_gender_mask | missing_article_mask
    n_missing_required = int(missing_required.sum())
    print(f"Rows missing articleType or gender: {n_missing_required}")
    if n_missing_required > 0:
        df_styles = df_styles[~missing_required].copy()

    # Match with images on disk
    df_styles["id_str"] = df_styles["id"].astype(str)
    all_image_files = {p.stem: p for p in images_dir.glob("*.jpg")}
    print(f"Total .jpg files in {images_dir}: {len(all_image_files)}")

    # Check existence
    df_styles["img_exists"] = df_styles["id_str"].apply(lambda x: x in all_image_files)
    missing_images = df_styles[~df_styles["img_exists"]]
    print(f"Rows in styles.csv whose image file DOES NOT exist: {len(missing_images)}")

    df_valid = df_styles[df_styles["img_exists"]].copy()
    tasks = [(row["id_str"], str(all_image_files[row["id_str"]])) for _, row in df_valid.iterrows()]

    print("\n==================================================")
    print("STEP 2 & 3: MEASURING RESOLUTION, BLUR, PHASH, BANNER")
    print("==================================================")
    metrics_cache_path = reports_dir / "quality_metrics.parquet"
    cached_metrics = {}

    if metrics_cache_path.exists():
        print(f"Found existing {metrics_cache_path}, loading cache...")
        df_cached = pd.read_parquet(metrics_cache_path)
        for _, r in df_cached.iterrows():
            cached_metrics[str(r["id"])] = r.to_dict()
        print(f"Loaded {len(cached_metrics)} records from cache.")

    remaining_tasks = [t for t in tasks if t[0] not in cached_metrics]
    print(f"Total tasks: {len(tasks)}, Remaining to compute: {len(remaining_tasks)}")

    new_results = []
    if remaining_tasks:
        n_workers = min(10, os.cpu_count() or 4)
        print(f"Running metric computation using {n_workers} processes...")
        chunk_size = 500
        total_tasks = len(remaining_tasks)
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i in range(0, total_tasks, chunk_size):
                batch = remaining_tasks[i:i + chunk_size]
                batch_res = list(executor.map(compute_metrics_for_single_image, batch))
                new_results.extend(batch_res)
                for item in batch_res:
                    cached_metrics[item["id"]] = item
                
                # Checkpoint save every 2000 images for resume capability
                if len(new_results) % 2000 < chunk_size or i + chunk_size >= total_tasks:
                    df_checkpoint = pd.DataFrame(list(cached_metrics.values()))
                    df_checkpoint.to_parquet(metrics_cache_path, index=False)
                    elapsed = time.time() - t_start
                    processed_so_far = len(cached_metrics)
                    print(f"[{processed_so_far}/{len(tasks)}] Checkpointed to {metrics_cache_path} ({elapsed:.1f}s elapsed)")

    records = [cached_metrics[t[0]] for t in tasks]
    df_metrics = pd.DataFrame(records)

    # Resolution statistics (Bước 2)
    valid_res = df_metrics[df_metrics["status"] == "ok"]
    w_series = valid_res["width"]
    h_series = valid_res["height"]
    short_series = valid_res["short_side"]

    min_short_conf = config["image_filter"].get("min_short_side", 224)
    n_under_min_side = int((short_series < min_short_conf).sum())

    print("\n--- RESOLUTION REPORT (Bước 2) ---")
    print(f"Width:  min={w_series.min()}, median={w_series.median()}, max={w_series.max()}")
    print(f"Height: min={h_series.min()}, median={h_series.median()}, max={h_series.max()}")
    print(f"Short side: min={short_series.min()}, median={short_series.median()}, max={short_series.max()}")
    print(f"Images with short_side < {min_short_conf}: {n_under_min_side} ({n_under_min_side/len(valid_res)*100:.2f}%)")

    # Step 2 Check: small dataset abort
    median_h = h_series.median()
    median_w = w_series.median()
    if median_w <= 80 and median_h <= 100:
        print("ERROR: Dataset is 60x80 small version! Aborting as requested.")
        sys.exit(1)
    else:
        print(f"Verified: Full resolution dataset confirmed (median: {int(median_w)}x{int(median_h)}).")

    # Check corrupt files
    corrupt_files = df_metrics[df_metrics["is_corrupt"]]
    print(f"\nCorrupt files found: {len(corrupt_files)}")
    if len(corrupt_files) > 0:
        quarantine_log_path = reports_dir / "quarantine_log.csv"
        q_records = []
        for _, c_row in corrupt_files.iterrows():
            q_records.append({
                "item_id": c_row["id"],
                "original_path": c_row["path"],
                "reason": "corrupt",
                "metric_value": c_row["status"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            })
        df_q = pd.DataFrame(q_records)
        df_q.to_csv(quarantine_log_path, index=False, encoding="utf-8")
        print(f"Logged {len(q_records)} corrupt files to {quarantine_log_path}")

    # Step 3: Duplicate detection via pHash
    phash_thresh = config["image_filter"].get("phash_hamming_threshold", 4)
    dup_groups, matching_pairs, id_to_group = find_duplicates_phash(records, threshold=phash_thresh)

    df_metrics["dup_group_id"] = df_metrics["id"].apply(lambda x: id_to_group.get(x, -1))
    df_metrics.to_parquet(metrics_cache_path, index=False)
    print(f"Saved updated metrics with duplicate groups to {metrics_cache_path}")

    # Blur score distribution
    blur_scores = valid_res["blur_score"].values
    p1 = np.percentile(blur_scores, 1)
    p5 = np.percentile(blur_scores, 5)
    p10 = np.percentile(blur_scores, 10)
    p20 = np.percentile(blur_scores, 20)
    p50 = np.percentile(blur_scores, 50)
    p75 = np.percentile(blur_scores, 75)
    p95 = np.percentile(blur_scores, 95)

    print("\n--- BLUR SCORE DISTRIBUTION (Laplacian Variance) ---")
    print(f"Min: {np.min(blur_scores):.2f}")
    print(f"1%:  {p1:.2f}")
    print(f"5%:  {p5:.2f}")
    print(f"10%: {p10:.2f}")
    print(f"20%: {p20:.2f}")
    print(f"50% (Median): {p50:.2f}")
    print(f"75%: {p75:.2f}")
    print(f"95%: {p95:.2f}")
    print(f"Max: {np.max(blur_scores):.2f}")

    # Banner flag distribution
    banner_flagged_df = df_metrics[df_metrics["banner_flag"]]
    print(f"\n--- BANNER FLAG DISTRIBUTION ---")
    print(f"Banner flagged: {len(banner_flagged_df)} / {len(df_metrics)} ({len(banner_flagged_df)/len(df_metrics)*100:.2f}%)")

    # Step 4: Contact sheets generation
    print("\n==================================================")
    print("STEP 4: GENERATING CONTACT SHEETS FOR REVIEW")
    print("==================================================")
    
    # 1. Blur contact sheets (blur_low.png, blur_mid.png, blur_high.png)
    # 5% percentile
    low_blur_candidates = df_metrics[(df_metrics["blur_score"] >= p1) & (df_metrics["blur_score"] <= p10)]
    low_sample = low_blur_candidates.sample(n=min(36, len(low_blur_candidates)), random_state=seed)
    low_entries = [
        {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | Blur:{r['blur_score']:.1f}"}
        for _, r in low_sample.iterrows()
    ]
    create_contact_sheet(
        low_entries,
        contact_sheets_dir / "blur_low.png",
        title=f"Blur Review: Low Percentile (~5%, Range: {p1:.1f} - {p10:.1f})"
    )

    # 20% percentile
    mid_blur_candidates = df_metrics[(df_metrics["blur_score"] >= p10) & (df_metrics["blur_score"] <= p50)]
    mid_sample = mid_blur_candidates.sample(n=min(36, len(mid_blur_candidates)), random_state=seed)
    mid_entries = [
        {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | Blur:{r['blur_score']:.1f}"}
        for _, r in mid_sample.iterrows()
    ]
    create_contact_sheet(
        mid_entries,
        contact_sheets_dir / "blur_mid.png",
        title=f"Blur Review: Mid Percentile (~20%, Range: {p10:.1f} - {p50:.1f})"
    )

    # 50% percentile
    high_blur_candidates = df_metrics[(df_metrics["blur_score"] >= p50) & (df_metrics["blur_score"] <= p75)]
    high_sample = high_blur_candidates.sample(n=min(36, len(high_blur_candidates)), random_state=seed)
    high_entries = [
        {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | Blur:{r['blur_score']:.1f}"}
        for _, r in high_sample.iterrows()
    ]
    create_contact_sheet(
        high_entries,
        contact_sheets_dir / "blur_high.png",
        title=f"Blur Review: High Percentile (~50%, Range: {p50:.1f} - {p75:.1f})"
    )

    # 2. small_side.png
    # Pick images near 224: below 224 and above 224
    under_224 = df_metrics[df_metrics["short_side"] < 224].sort_values("short_side", ascending=False)
    over_224 = df_metrics[df_metrics["short_side"] >= 224].sort_values("short_side", ascending=True)
    
    n_under = min(18, len(under_224))
    n_over = 36 - n_under
    sample_small = pd.concat([under_224.head(n_under), over_224.head(n_over)])
    small_entries = [
        {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | Side:{r['short_side']} ({r['width']}x{r['height']})"}
        for _, r in sample_small.iterrows()
    ]
    create_contact_sheet(
        small_entries,
        contact_sheets_dir / "small_side.png",
        title="Resolution Review: Short Side Near 224 (Sorted)"
    )

    # 3. banner_flagged.png and banner_not_flagged.png
    if len(banner_flagged_df) > 0:
        banner_sample = banner_flagged_df.sample(n=min(36, len(banner_flagged_df)), random_state=seed)
        banner_entries = [
            {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | T_Edg:{r['top_edge_density']:.3f} | S_Edg:{r['side_edge_density']:.3f}"}
            for _, r in banner_sample.iterrows()
        ]
        create_contact_sheet(
            banner_entries,
            contact_sheets_dir / "banner_flagged.png",
            title=f"Banner Flagged Samples ({len(banner_flagged_df)} total detected)"
        )

    clean_banner_df = df_metrics[~df_metrics["banner_flag"]]
    clean_sample = clean_banner_df.sample(n=min(36, len(clean_banner_df)), random_state=seed)
    clean_entries = [
        {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | Clean | T_Edg:{r['top_edge_density']:.3f}"}
        for _, r in clean_sample.iterrows()
    ]
    create_contact_sheet(
        clean_entries,
        contact_sheets_dir / "banner_not_flagged.png",
        title="Banner NOT Flagged Samples (Normal Clean Images)"
    )

    # 4. duplicates.png
    id_to_record_map = {r["id"]: r for r in records}
    create_duplicate_contact_sheet(
        dup_groups,
        id_to_record_map,
        contact_sheets_dir / "duplicates.png",
        max_groups=6,
        max_cols=6
    )

    # 5. risk_categories.png
    # sample Accessories/Headwear/Eyewear/Watches/Bags
    # merge with styles metadata
    df_merged = df_metrics.merge(df_styles[["id_str", "masterCategory", "subCategory", "articleType"]], left_on="id", right_on="id_str", how="inner")
    risk_subcats = ["Accessories", "Headwear", "Eyewear", "Watches", "Bags"]
    risk_df = df_merged[df_merged["subCategory"].isin(risk_subcats) | df_merged["masterCategory"].isin(["Accessories"])]
    print(f"Risk categories (Accessories/Headwear/Eyewear/Watches/Bags): {len(risk_df)} items found.")
    
    if len(risk_df) > 0:
        sample_risk = risk_df.sample(n=min(36, len(risk_df)), random_state=seed)
        risk_entries = [
            {"path": r["path"], "id": r["id"], "label_text": f"ID:{r['id']} | {r['subCategory']} - {r['articleType'][:10]}"}
            for _, r in sample_risk.iterrows()
        ]
        create_contact_sheet(
            risk_entries,
            contact_sheets_dir / "risk_categories.png",
            title="Risk Categories: Accessories / Headwear / Eyewear / Watches / Bags"
        )

    print("\nAll contact sheets generated in reports/contact_sheets/")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
