import sys
import json
import pandas as pd
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    cat = pd.read_parquet("data/catalog/catalog.parquet")
    g_test = cat[(cat["split"] == "test") & (cat["in_scope"] == True)]
    g_full = cat[cat["in_scope"] == True]

    print(f"G_test length: {len(g_test)}")
    print(f"G_full length: {len(g_full)}")

    # Load captions
    with open("data/checkpoints_siglip-vit/dense_captions_catalog.json", "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    # Q1: Check 1290 captions
    g_test_captions = [captions_dict[str(row["item_id"])] for _, row in g_test.iterrows()]
    caption_counts = Counter(g_test_captions)
    duplicates = {cap: c for cap, c in caption_counts.items() if c > 1}
    print(f"\n--- Q1 Paired Captions Check ---")
    print(f"Total test items: {len(g_test_captions)}")
    print(f"Unique captions: {len(caption_counts)}")
    print(f"Duplicate captions count: {len(duplicates)}")
    if duplicates:
        print("Sample duplicated captions:")
        for cap, c in list(duplicates.items())[:5]:
            matching_ids = g_test[g_test["item_id"].astype(str).map(lambda x: captions_dict.get(x) == cap)]["item_id"].tolist()
            print(f"  Count {c} (IDs: {matching_ids}): {cap[:80]}...")

    # Q2: Combinations of (articleType, gender, baseColour)
    print("\n--- Q2 Attribute Combinations ---")
    combo_counts = g_test.groupby(["articleType", "gender", "baseColour"]).size().reset_index(name="count")
    combo_ge5 = combo_counts[combo_counts["count"] >= 5].sort_values(by="count", ascending=False)
    print(f"Combinations with >= 5 items in G_test: {len(combo_ge5)}")
    print(combo_ge5.head(25))

    # Print all 61 combos
    full_groups = g_full.groupby(["articleType", "gender", "baseColour"]).size().reset_index(name="full_count")
    merged = pd.merge(combo_ge5, full_groups, on=["articleType", "gender", "baseColour"], how="left")
    print(f"\nTotal combos >= 5 in G_test: {len(merged)}")
    for idx, r in merged.iterrows():
        print(f"{r['articleType']:<15} | {r['gender']:<7} | {r['baseColour']:<12} | G_test: {r['count']:<3} | G_full: {r['full_count']:<4}")

if __name__ == "__main__":
    main()
