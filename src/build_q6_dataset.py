"""
Build Q6 Dataset (Independent Attribute Relevance)
Usage and Season based queries where AttributeIndex does NOT filter or award soft bonuses.
"""
import json
import hashlib
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.query_parser import QueryParser

def build_q6_dataset():
    seed = 42
    np.random.seed(seed)
    
    cat = pd.read_parquet("data/catalog/catalog.parquet")
    styles = pd.read_csv("data/fashion-dataset/styles.csv", on_bad_lines="skip")
    cat["item_id"] = cat["item_id"].astype(str)
    styles["item_id"] = styles["id"].astype(str)
    cat = cat.merge(styles[["item_id", "season"]], on="item_id", how="left")
    
    in_scope = cat[cat["in_scope"] == True]
    test_df = in_scope[in_scope["split"] == "test"]
    val_df = in_scope[in_scope["split"] == "unknown"]
    
    parser = QueryParser()
    
    # Safe prefixes and keyword mappings
    generic_prefixes = ["quần áo", "phong cách", "sản phẩm", "mẫu", "outfit"]
    
    usage_map = {
        "Casual": ["dạo phố", "thường ngày", "hàng ngày"],
        "Sports": ["thể thao", "tập gym", "vận động"],
        "Formal": ["công sở", "đi làm", "lịch sự"],
        "Ethnic": ["truyền thống", "dân tộc"],
        "Party": ["dự tiệc", "dạ tiệc"]
    }
    
    season_map = {
        "Summer": ["mùa hè", "ngày hè"],
        "Winter": ["mùa lạnh", "ngày lạnh"],
        "Fall": ["mùa thu", "ngày thu"],
        "Spring": ["mùa xuân", "ngày xuân"]
    }
    
    gender_map = {
        "Men": "nam",
        "Women": "nữ",
        "Boys": "bé trai",
        "Girls": "bé gái",
        "Unisex": "unisex"
    }
    
    article_type_map = {
        "Tshirts": ["áo thun", "áo phông"],
        "Shirts": ["áo sơ mi"],
        "Jeans": ["quần jean", "quần bò"],
        "Casual Shoes": ["giày casual", "giày dạo phố"],
        "Sports Shoes": ["giày thể thao"],
        "Formal Shoes": ["giày tây", "giày công sở"],
        "Handbags": ["túi xách"],
        "Backpacks": ["balo"],
        "Dresses": ["đầm", "váy liền"],
        "Kurtas": ["áo kurta"],
        "Tops": ["áo kiểu"],
        "Heels": ["giày cao gót"],
        "Flats": ["giày búp bê"],
        "Sandals": ["dép xăng đan", "sandal"],
        "Wallets": ["ví da", "ví cầm tay"],
        "Sunglasses": ["kính mát", "kính râm"],
        "Shorts": ["quần short", "quần đùi"],
        "Track Pants": ["quần thể thao"],
        "Socks": ["tất vớ", "vớ"],
        "Belts": ["thắt lưng", "dây nịt"],
        "Bra": ["áo lót", "áo bra"],
        "Briefs": ["quần lót"],
        "Clutches": ["ví cầm tay"]
    }
    
    def generate_candidates_for_split(df, split_name):
        candidates = []
        
        # 1. Gender + Usage
        for (g, u), sub in df.groupby(["gender", "usage"]):
            if len(sub) >= 10 and g in gender_map and u in usage_map:
                g_str = gender_map[g]
                for pfx in generic_prefixes:
                    for u_str in usage_map[u]:
                        q_text = f"{pfx} {u_str} {g_str}"
                        candidates.append({
                            "query_text": q_text,
                            "type": "gender_usage",
                            "spec": {"gender": g, "usage": u},
                            "relevant_ids": sub["item_id"].tolist()
                        })
        
        # 2. Gender + Season
        for (g, s), sub in df.groupby(["gender", "season"]):
            if len(sub) >= 10 and g in gender_map and s in season_map:
                g_str = gender_map[g]
                for pfx in generic_prefixes:
                    for s_str in season_map[s]:
                        q_text = f"{pfx} {s_str} {g_str}"
                        candidates.append({
                            "query_text": q_text,
                            "type": "gender_season",
                            "spec": {"gender": g, "season": s},
                            "relevant_ids": sub["item_id"].tolist()
                        })
                        
        # 3. ArticleType + Usage
        for (a, u), sub in df.groupby(["articleType", "usage"]):
            if len(sub) >= 10 and a in article_type_map and u in usage_map:
                for a_str in article_type_map[a]:
                    for u_str in usage_map[u]:
                        q_text = f"{a_str} {u_str}"
                        candidates.append({
                            "query_text": q_text,
                            "type": "article_usage",
                            "spec": {"articleType": a, "usage": u},
                            "relevant_ids": sub["item_id"].tolist()
                        })
                        
        # 4. ArticleType + Season
        for (a, s), sub in df.groupby(["articleType", "season"]):
            if len(sub) >= 10 and a in article_type_map and s in season_map:
                for a_str in article_type_map[a]:
                    for s_str in season_map[s]:
                        q_text = f"{a_str} {s_str}"
                        candidates.append({
                            "query_text": q_text,
                            "type": "article_season",
                            "spec": {"articleType": a, "season": s},
                            "relevant_ids": sub["item_id"].tolist()
                        })
                        
        # 5. ArticleType + Gender + Usage
        for (a, g, u), sub in df.groupby(["articleType", "gender", "usage"]):
            if len(sub) >= 10 and a in article_type_map and g in gender_map and u in usage_map:
                g_str = gender_map[g]
                for a_str in article_type_map[a]:
                    for u_str in usage_map[u]:
                        q_text = f"{a_str} {g_str} {u_str}"
                        candidates.append({
                            "query_text": q_text,
                            "type": "article_gender_usage",
                            "spec": {"articleType": a, "gender": g, "usage": u},
                            "relevant_ids": sub["item_id"].tolist()
                        })
                        
        # 6. ArticleType + Gender + Season
        for (a, g, s), sub in df.groupby(["articleType", "gender", "season"]):
            if len(sub) >= 10 and a in article_type_map and g in gender_map and s in season_map:
                g_str = gender_map[g]
                for a_str in article_type_map[a]:
                    for s_str in season_map[s]:
                        q_text = f"{a_str} {g_str} {s_str}"
                        candidates.append({
                            "query_text": q_text,
                            "type": "article_gender_season",
                            "spec": {"articleType": a, "gender": g, "season": s},
                            "relevant_ids": sub["item_id"].tolist()
                        })
        
        # Deduplicate by query_text
        unique_cand = {}
        for c in candidates:
            # Check parser baseColour is None to avoid accidental color match
            p = parser.parse(c["query_text"])
            if p.get("baseColour") is None:
                if c["query_text"] not in unique_cand:
                    unique_cand[c["query_text"]] = c
                    
        cand_list = list(unique_cand.values())
        print(f"[{split_name}] Generated {len(cand_list)} valid unique candidate queries.")
        
        # Shuffle with fixed seed and select 120 queries
        rng = np.random.RandomState(seed)
        idx_perm = rng.permutation(len(cand_list))
        selected = [cand_list[i] for i in idx_perm[:120]]
        
        # Assign IDs
        for i, q in enumerate(selected):
            q["query_id"] = f"q6_{split_name}_{i+1:03d}"
            q["n_relevant"] = len(q["relevant_ids"])
            
        return selected

    val_queries = generate_candidates_for_split(val_df, "val")
    test_queries = generate_candidates_for_split(test_df, "test")
    
    out_data = {
        "metadata": {
            "dataset": "Q6_Independent_Attribute_Queries",
            "seed": seed,
            "description": "Queries using usage and season attributes independent of AttributeIndex filters/bonuses",
            "n_val": len(val_queries),
            "n_test": len(test_queries),
            "min_relevant_per_query": 10
        },
        "val_queries": val_queries,
        "test_queries": test_queries
    }
    
    out_path = Path("eval/q6_queries.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        
    # Calculate sha256
    with open(out_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Saved {len(val_queries)} val queries and {len(test_queries)} test queries to {out_path}")
    print(f"SHA-256: {file_hash}")
    return file_hash

if __name__ == "__main__":
    build_q6_dataset()
