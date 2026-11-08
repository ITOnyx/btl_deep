import sys
import json
import random
import pandas as pd
import pathlib

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.vi_vocab import (
    remove_diacritics,
    ARTICLE_TYPE_MAP,
    COLOUR_MAP,
    GENDER_MAP
)

def audit_d1():
    cat = pd.read_parquet("data/catalog/catalog.parquet")
    test_cat = cat[(cat["split"] == "test") & (cat["in_scope"] == True)].copy()
    print(f"Total test in-scope items: {len(test_cat)}")

    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)

    q1_queries = {q["item_id"]: q["query_text"] for q in testset["q1_paired_queries"]}

    # Check match for each item
    matches = []
    audit_rows = []

    for idx, row in test_cat.iterrows():
        item_id = str(row["item_id"])
        art = str(row["articleType"])
        gen = str(row["gender"])
        col = str(row["baseColour"])
        cap = q1_queries.get(item_id, "")
        norm_cap = remove_diacritics(cap.lower())

        # Check gender match
        vi_gen = GENDER_MAP.get(gen, gen).lower()
        norm_gen = remove_diacritics(vi_gen)
        # Men accepts 'nam', Women accepts 'nu', Boys accepts 'be trai'
        gen_match = norm_gen in norm_cap or (gen == "Men" and "nam" in norm_cap) or (gen == "Women" and "nu" in norm_cap) or (gen == "Unisex" and ("unisex" in norm_cap or "nam nu" in norm_cap))

        # Check color match
        vi_col = COLOUR_MAP.get(col, col).lower()
        norm_col = remove_diacritics(vi_col)
        # simplify color names
        first_col_word = norm_col.split()[0] if norm_col else ""
        col_match = norm_col in norm_cap or (first_col_word in norm_cap if len(first_col_word) >= 3 else False)

        # Check articleType match
        vi_art = ARTICLE_TYPE_MAP.get(art, art).lower()
        norm_art = remove_diacritics(vi_art)
        first_art_phrase = norm_art.split("(")[0].strip()
        art_match = first_art_phrase in norm_cap or remove_diacritics(art.lower()) in norm_cap

        is_match = gen_match and col_match and art_match
        matches.append({
            "item_id": item_id,
            "gen_match": gen_match,
            "col_match": col_match,
            "art_match": art_match,
            "full_match": is_match
        })

        audit_rows.append({
            "item_id": item_id,
            "articleType": art,
            "gender": gen,
            "baseColour": col,
            "caption": cap,
            "path": row["path"],
            "full_match": is_match
        })

    df_matches = pd.DataFrame(matches)
    full_match_rate = df_matches["full_match"].mean()
    gen_rate = df_matches["gen_match"].mean()
    col_rate = df_matches["col_match"].mean()
    art_rate = df_matches["art_match"].mean()

    print(f"\n--- D1 METADATA AUDIT RESULTS ---")
    print(f"Full Match (Gender + Color + ArticleType): {full_match_rate*100:.2f}% ({df_matches['full_match'].sum()}/{len(df_matches)})")
    print(f"Gender Match Rate:      {gen_rate*100:.2f}%")
    print(f"Color Match Rate:       {col_rate*100:.2f}%")
    print(f"ArticleType Match Rate: {art_rate*100:.2f}%")

    # Sample 10 random rows (seed 42)
    random.seed(42)
    sample_indices = random.sample(range(len(audit_rows)), 10)
    print("\n--- 10 RANDOM AUDIT ROWS (Seed 42) ---")
    for rank, idx in enumerate(sample_indices, 1):
        r = audit_rows[idx]
        print(f"\n[{rank}] Item ID: {r['item_id']} | Type: {r['articleType']} | Gender: {r['gender']} | Color: {r['baseColour']}")
        print(f"    Path:    {r['path']}")
        print(f"    Caption: {r['caption']}")
        print(f"    Match?   {'✅ YES' if r['full_match'] else '❌ NO'}")

if __name__ == "__main__":
    audit_d1()
