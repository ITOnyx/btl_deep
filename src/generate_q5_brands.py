"""
Generate Q5: Brand-specific Fashion Queries (src/generate_q5_brands.py)
Phase 3e: Brand queries for brands with >= 20 in-scope products x articleTypes with >= 5 products of that brand.
Creates Vietnamese queries, maximum 150 queries, seed 42.
Relevance: same brand + same articleType (+ compatible gender if query specifies gender).
Saves to eval/q5_brand.json with SHA-256 hash.
"""

import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

import sys
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from src.attr_index import AttributeIndex
    from src.vi_vocab import (
        remove_diacritics,
        GENDER_MAP,
        COLOUR_MAP,
        ARTICLE_TYPE_MAP
    )
except ImportError:
    from attr_index import AttributeIndex
    from vi_vocab import (
        remove_diacritics,
        GENDER_MAP,
        COLOUR_MAP,
        ARTICLE_TYPE_MAP
    )

def generate_q5():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # 1. Load AttributeIndex & catalog
    attr_idx = AttributeIndex()
    df = attr_idx.df # in_scope items

    # 2. Load families for gender compatibility
    with open("eval/families.json", "r", encoding="utf-8") as f:
        fam_data = json.load(f)
    gender_compat = fam_data.get("gender_compatibility", {
        "Men": ["Men", "Unisex"],
        "Women": ["Women", "Unisex"],
        "Boys": ["Boys", "Unisex"],
        "Girls": ["Girls", "Unisex"],
        "Unisex": ["Men", "Women", "Unisex"]
    })

    # 3. Load brands (>= 20 products)
    with open("src/brands.json", "r", encoding="utf-8") as f:
        brands_dict = json.load(f)

    # Map Vietnamese phrases for popular article types
    ARTICLE_VI = {
        "Tshirts": "áo thun",
        "Shirts": "áo sơ mi",
        "Casual Shoes": "giày casual",
        "Sports Shoes": "giày thể thao",
        "Watches": "đồng hồ",
        "Kurtas": "áo kurta",
        "Handbags": "túi xách",
        "Tops": "áo kiểu",
        "Heels": "giày cao gót",
        "Wallets": "ví",
        "Briefs": "quần lót",
        "Sandals": "dép sandal",
        "Sunglasses": "kính râm",
        "Flip Flops": "dép kẹp",
        "Belts": "thắt lưng",
        "Backpacks": "balo",
        "Socks": "tất",
        "Formal Shoes": "giày tây",
        "Flats": "giày búp bê",
        "Shorts": "quần short",
        "Bra": "áo ngực",
        "Jeans": "quần jean",
        "Sarees": "saree",
        "Dresses": "váy đầm",
        "Earrings": "bông tai",
        "Trousers": "quần âu",
        "Clutches": "ví cầm tay",
        "Sweatshirts": "áo nỉ",
        "Track Pants": "quần thể thao",
        "Sweaters": "áo len",
        "Caps": "mũ lưỡi trai",
        "Jackets": "áo khoác",
        "Kurtis": "áo kurti",
        "Tunics": "áo tunic",
        "Ties": "cà vạt",
        "Pendant": "mặt dây chuyền",
        "Trunk": "quần lót trunk",
        "Scarves": "khăn quàng",
        "Leggings": "quần legging",
        "Skirts": "chân váy",
        "Necklace and Chains": "dây chuyền",
        "Laptop Bag": "túi laptop",
        "Duffel Bag": "túi trống",
        "Boxers": "quần boxer"
    }

    GENDER_VI = {
        "Men": "nam",
        "Women": "nữ",
        "Boys": "bé trai",
        "Girls": "bé gái",
        "Unisex": "unisex"
    }

    # 4. Find all eligible (brand, articleType) pairs with >= 5 products
    brand_art_groups = defaultdict(list)
    for idx, row in df.iterrows():
        b = row["brand"]
        art = row["articleType"]
        if b != "Unknown" and b in brands_dict and art in ARTICLE_VI:
            brand_art_groups[(b, art)].append(idx)

    eligible_pairs = [(b, art) for (b, art), idxs in brand_art_groups.items() if len(idxs) >= 5]
    print(f"Total eligible (brand, articleType) pairs with >= 5 items: {len(eligible_pairs)}")

    # Sort pairs for deterministic shuffling
    eligible_pairs.sort()
    random.shuffle(eligible_pairs)

    queries = []
    query_id_counter = 1

    for brand, art in eligible_pairs:
        if len(queries) >= 150:
            break

        idxs = brand_art_groups[(brand, art)]
        sub_df = df.iloc[idxs]
        gender_counts = sub_df["gender"].value_counts()
        
        # Decide whether to specify gender
        # If one gender is dominant (>= 70% of brand-art items), specify it half the time
        top_gender = gender_counts.index[0]
        top_gender_ratio = gender_counts.iloc[0] / len(sub_df)
        
        include_gender = (top_gender in GENDER_VI) and (top_gender_ratio >= 0.7) and (random.random() < 0.65)
        gen = top_gender if include_gender else None

        # Format Vietnamese query text
        art_vi = ARTICLE_VI[art]
        if gen:
            gen_vi = GENDER_VI[gen]
            # Pattern 1: "{art_vi} {brand} {gen_vi}"
            # Pattern 2: "{brand} {art_vi} {gen_vi}"
            if random.random() < 0.5:
                q_text = f"{art_vi} {brand} {gen_vi}"
            else:
                q_text = f"{brand} {art_vi} {gen_vi}"
        else:
            # Pattern 1: "{art_vi} {brand}"
            # Pattern 2: "{brand} {art_vi}"
            if random.random() < 0.5:
                q_text = f"{art_vi} {brand}"
            else:
                q_text = f"{brand} {art_vi}"

        # Find relevant item IDs in G_full and G_test
        # Relevance: same brand + same articleType (+ compatible gender if gen is specified)
        if gen:
            compat_gens = set(gender_compat.get(gen, [gen, "Unisex"]))
            rel_mask_full = (df["brand"] == brand) & (df["articleType"] == art) & (df["gender"].isin(compat_gens))
        else:
            rel_mask_full = (df["brand"] == brand) & (df["articleType"] == art)

        rel_mask_test = rel_mask_full & (df["split"] == "test")

        rel_ids_full = df.loc[rel_mask_full, "item_id"].tolist()
        rel_ids_test = df.loc[rel_mask_test, "item_id"].tolist()

        if len(rel_ids_full) < 5:
            continue

        q_obj = {
            "query_id": f"q5_{query_id_counter:03d}",
            "query_text": q_text,
            "brand": brand,
            "articleType": art,
            "gender": gen,
            "n_relevant_full": len(rel_ids_full),
            "n_relevant_test": len(rel_ids_test),
            "relevant_ids_full": rel_ids_full,
            "relevant_ids_test": rel_ids_test
        }
        queries.append(q_obj)
        query_id_counter += 1

    print(f"Generated {len(queries)} Q5 brand queries.")

    out_data = {
        "metadata": {
            "dataset": "Q5_Brand_Queries",
            "seed": seed,
            "total_queries": len(queries),
            "min_products_per_brand": 20,
            "min_products_per_brand_article": 5,
            "relevance_rule": "brand == Q.brand and articleType == Q.articleType and (gender in compat(Q.gender) if Q.gender else True)"
        },
        "queries": queries
    }

    out_file = Path("eval/q5_brand.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    with open(out_file, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"Saved Q5 brand queries to {out_file}")
    print(f"SHA-256: {sha256}")

    # Print sample queries
    print("\nSample 10 Q5 queries:")
    for q in queries[:10]:
        print(f"  [{q['query_id']}] '{q['query_text']}' -> Brand={q['brand']}, Type={q['articleType']}, Gen={q['gender']} (Rel full: {q['n_relevant_full']}, test: {q['n_relevant_test']})")

    return sha256

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    generate_q5()
