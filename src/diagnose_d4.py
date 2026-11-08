"""
Diagnosis D4: Controlled Experiment
Compares 1,000 sampled items from split == 'unknown' vs 1,290 items from split == 'test'
across 4 caption formats (+ training rich text format):
  (1) caption v1 (formula from dense_captions_catalog.json)
  (2) caption title-first (moves 'Tên gốc: {name}' to the front)
  (3) caption title-only (English name from productDisplayName / catalog)
  (4) caption concise Vietnamese ('{loại} {giới} màu {màu}')
  (5) caption training rich-text ('[{category}] {title} ({attrs})')
Evaluates on:
  - Separate Gallery (1,000 for unknown, 1,290 for test)
  - Full Gallery G_full (38,646)
"""
import sys
import json
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model_siglip import load_siglip_inference
from src.vi_vocab import ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP, USAGE_MAP

SEASON_MAP = {
    "Summer": "mùa hè thoáng mát",
    "Fall": "mùa thu",
    "Winter": "mùa đông ấm áp",
    "Spring": "mùa xuân"
}

USAGE_PHRASE_MAP = {
    "Casual": "dạo phố thường ngày",
    "Sports": "thể thao vận động",
    "Ethnic": "truyền thống lễ hội",
    "Formal": "công sở lịch sự",
    "Smart Casual": "thanh lịch trẻ trung",
    "Party": "tiệc tùng sang trọng",
    "Travel": "du lịch dã ngoại"
}

GENDER_PHRASE_MAP = {
    "Men": "nam",
    "Women": "nữ",
    "Boys": "bé trai",
    "Girls": "bé gái",
    "Unisex": "unisex nam nữ"
}

def clean_color_phrase(c_str):
    if not c_str or c_str == "nan":
        return "màu sắc trang nhã"
    c_lower = c_str.lower()
    if not c_lower.startswith("màu"):
        return f"màu {c_lower}"
    return c_lower

def build_captions_for_item(row, dense_cap=None):
    name = str(row["name"]).strip()
    art = str(row["articleType"])
    gen = str(row["gender"])
    col = str(row["baseColour"])
    usage = str(row["usage"])
    season = str(row.get("season", "Summer"))

    art_vi = ARTICLE_TYPE_MAP.get(art, art).lower()
    gen_vi = GENDER_PHRASE_MAP.get(gen, gen.lower())
    col_vi = clean_color_phrase(COLOUR_MAP.get(col, col))
    usage_vi = USAGE_PHRASE_MAP.get(usage, "thường ngày")
    season_vi = SEASON_MAP.get(season, "mùa hè thoáng mát")

    # (1) Format 1: synthetic formula
    if dense_cap:
        c1 = dense_cap
    else:
        c1 = f"Sản phẩm {art_vi} dành cho {gen_vi}, thiết kế {col_vi} phù hợp {usage_vi} ({season_vi}). Tên gốc: {name}."

    # (2) Format 2: title first
    if "Tên gốc:" in c1:
        prefix, title_part = c1.split("Tên gốc:", 1)
        prefix = prefix.strip()
        title_part = title_part.strip().rstrip(".")
        c2 = f"Tên gốc: {title_part}. {prefix}"
    else:
        c2 = f"Tên gốc: {name}. Sản phẩm {art_vi} dành cho {gen_vi}, thiết kế {col_vi} phù hợp {usage_vi} ({season_vi})."

    # (3) Format 3: title only
    c3 = name

    # (4) Format 4: concise Vietnamese
    c4 = f"{art_vi} {gen_vi} {col_vi}".strip()

    # (5) Format 5: training rich text
    cat = f"{row.get('masterCategory', '')} > {row.get('subCategory', '')} > {art}"
    c5 = f"[{cat}] {name} ({gen_vi}, {col_vi}, {art_vi})"

    return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

def evaluate_retrieval(txt_embs, gal_embs, target_indices):
    """
    txt_embs: (N, dim)
    gal_embs: (M, dim)
    target_indices: array of length N with the index of the target in gal_embs
    """
    sim = torch.matmul(txt_embs, gal_embs.t()) # (N, M)
    N, M = sim.shape
    ranks = []
    r1, r5, r10 = 0, 0, 0
    for i in range(N):
        tar_idx = target_indices[i]
        scores = sim[i]
        tar_score = scores[tar_idx].item()
        rank = (scores > tar_score).sum().item() + 1
        ranks.append(rank)
        if rank == 1:
            r1 += 1
        if rank <= 5:
            r5 += 1
        if rank <= 10:
            r10 += 1
    ranks = np.array(ranks)
    return {
        "r1": r1 / N,
        "r5": r5 / N,
        "r10": r10 / N,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks))
    }

def run_d4():
    print("=== D4: CONTROLLED EXPERIMENT ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    catalog = pd.read_parquet("data/catalog/catalog.parquet")
    styles = pd.read_csv("data/fashion-dataset/styles.csv", on_bad_lines="skip")
    styles["id"] = styles["id"].astype(str)
    styles_season = styles.set_index("id")["season"].to_dict()

    # Merge season into catalog
    catalog["season"] = catalog["item_id"].astype(str).map(styles_season).fillna("Summer")

    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    with open("index/id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)
    with open("data/checkpoints_siglip-vit/dense_captions_catalog.json", "r", encoding="utf-8") as f:
        dense_captions = json.load(f)

    # 1. Prepare Test Set (1,290 items)
    g_test_ids = [str(x) for x in testset["g_test_ids"]]
    test_rows = catalog.set_index(catalog["item_id"].astype(str)).loc[g_test_ids].reset_index(drop=True)

    # 2. Prepare Sampled Unknown Set (1,000 items, seed 42)
    unknown_cat = catalog[(catalog["split"] == "unknown") & (catalog["in_scope"])]
    unknown_sample = unknown_cat.sample(n=1000, random_state=42).reset_index(drop=True)
    g_unknown_ids = unknown_sample["item_id"].astype(str).tolist()

    print(f"Test items: {len(test_rows)} | Unknown sampled items: {len(unknown_sample)}")

    # 3. Build captions for both sets
    test_caps = [build_captions_for_item(row, dense_captions.get(str(row["item_id"]))) for _, row in test_rows.iterrows()]
    unknown_caps = [build_captions_for_item(row) for _, row in unknown_sample.iterrows()]

    # 4. Load full gallery embeddings (G_full: 38,646)
    img_embs_np = np.load("index/image_embeddings.npy")
    img_embs_full = torch.tensor(img_embs_np, dtype=torch.float32, device=device)
    img_embs_full = F.normalize(img_embs_full, p=2, dim=-1)

    id_to_full_idx = {str(item["item_id"]): idx for idx, item in enumerate(id_map)}
    
    test_full_indices = [id_to_full_idx[x] for x in g_test_ids]
    unknown_full_indices = [id_to_full_idx[x] for x in g_unknown_ids]

    img_embs_test = img_embs_full[test_full_indices] # (1290, 1152)
    img_embs_unknown = img_embs_full[unknown_full_indices] # (1000, 1152)

    # Load Model for text encoding
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))

    def encode_texts(text_list):
        embs = []
        batch_size = 64
        for i in range(0, len(text_list), batch_size):
            b = text_list[i:i+batch_size]
            inp = model.processor(text=b, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            input_ids = inp["input_ids"].to(device)
            mask = inp.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)
            with torch.no_grad():
                e = model.encode_text(input_ids, mask)
                embs.append(e.float())
        res = torch.cat(embs, dim=0)
        return F.normalize(res, p=2, dim=-1)

    formats = [
        ("c1", "Format 1: Synthetic v1 (formula)"),
        ("c2", "Format 2: Title first ('Tên gốc: {name}...')"),
        ("c3", "Format 3: Title only (English name)"),
        ("c4", "Format 4: Concise Vietnamese ('{loại} {giới} màu {màu}')"),
        ("c5", "Format 5: Training Rich Text ('[{cat}] {name} ({attrs})')")
    ]

    results_table = {}

    for split_name, rows, caps, gal_own, own_indices, full_indices in [
        ("Unknown (Train/Known)", unknown_sample, unknown_caps, img_embs_unknown, np.arange(len(unknown_sample)), unknown_full_indices),
        ("Test (Unseen)", test_rows, test_caps, img_embs_test, np.arange(len(test_rows)), test_full_indices)
    ]:
        print(f"\n==================== EVALUATING: {split_name} ====================")
        results_table[split_name] = {}
        for fmt_key, fmt_label in formats:
            print(f"--- Encoding {fmt_label} (N={len(caps)}) ---")
            texts = [c[fmt_key] for c in caps]
            txt_embs = encode_texts(texts)

            # Evaluate on own gallery
            res_own = evaluate_retrieval(txt_embs, gal_own, own_indices)
            # Evaluate on full gallery
            res_full = evaluate_retrieval(txt_embs, img_embs_full, full_indices)

            results_table[split_name][fmt_key] = {
                "label": fmt_label,
                "own": res_own,
                "full": res_full
            }

            print(f"  {fmt_label}:")
            print(f"    Gallery Riêng (N={len(rows)}): Recall@10={res_own['r10']*100:.2f}% | R@5={res_own['r5']*100:.2f}% | R@1={res_own['r1']*100:.2f}% | Median={res_own['median_rank']:.1f}")
            print(f"    Gallery Đầy Đủ (N=38,646): Recall@10={res_full['r10']*100:.2f}% | R@5={res_full['r5']*100:.2f}% | R@1={res_full['r1']*100:.2f}% | Median={res_full['median_rank']:.1f}")

    with open("reports/diagnose_d4.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2, ensure_ascii=False)
    print("\nSaved reports/diagnose_d4.json successfully.")

if __name__ == "__main__":
    run_d4()
