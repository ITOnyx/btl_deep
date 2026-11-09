import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

print("=== TASK T6: INDEPENDENT CATALOG & INDEX VERIFICATION ===")

# 1. Shape, columns, duplicate IDs, total rows
catalog_path = "data/catalog/catalog.parquet"
df = pd.read_parquet(catalog_path)
print(f"1. Catalog Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
dup_ids = df["item_id"].duplicated().sum()
print(f"   Duplicate item_id count: {dup_ids}")
assert df.shape[0] == 40891, f"Expected 40,891 rows, got {df.shape[0]}"
assert dup_ids == 0, f"Expected 0 duplicates, got {dup_ids}"

# 2. 200 random IDs (seed 42): match (item_id, articleType, name) between catalog.parquet and index metadata
with open("index_v2/id_map_v2.json", "r", encoding="utf-8") as f:
    id_map = json.load(f)

# id_map_v2 has list of dicts with item_id, articleType, gender, baseColour, etc.
# Build map from item_id -> dict in id_map
id_map_dict = {}
for entry in id_map:
    iid = str(entry["item_id"] if isinstance(entry, dict) else entry)
    id_map_dict[iid] = entry

rng = np.random.RandomState(42)
sample_200_ids = rng.choice(df["item_id"].astype(str).values, size=200, replace=False)

mismatches = []
for sid in sample_200_ids:
    row = df[df["item_id"].astype(str) == sid].iloc[0]
    cat_art = str(row["articleType"])
    cat_name = str(row["name"])
    
    if sid in id_map_dict:
        idx_entry = id_map_dict[sid]
        if isinstance(idx_entry, dict):
            idx_art = str(idx_entry.get("articleType", ""))
            idx_name = str(idx_entry.get("name", idx_entry.get("productDisplayName", "")))
            if idx_art != cat_art or (idx_name and idx_name != cat_name):
                mismatches.append({"id": sid, "cat_art": cat_art, "idx_art": idx_art, "cat_name": cat_name, "idx_name": idx_name})

print(f"2. 200 Random IDs Check: {200 - len(mismatches)}/200 matched (Mismatches: {len(mismatches)})")

# 3. ID 1855, 43230, 42066
row_1855 = df[df["item_id"].astype(str) == "1855"].iloc[0]
print(f"3. ID 1855: name='{row_1855['name']}', articleType='{row_1855['articleType']}'")

kc_ids = ["43230", "42066"]
kc_in_catalog = df[df["item_id"].astype(str).isin(kc_ids)]
print(f"   Key chains in catalog.parquet (total): {len(kc_in_catalog)}")
for _, row in kc_in_catalog.iterrows():
    print(f"     ID {row['item_id']}: articleType='{row['articleType']}', in_scope={row['in_scope']}")

from src.attr_index import AttributeIndex
attr_idx = AttributeIndex(brands_path="src/brands_v2.json")
gallery_item_ids = set(str(x) for x in attr_idx.item_ids)
print(f"   Gallery (df in AttributeIndex, size {len(gallery_item_ids)}) contains 43230: {'43230' in gallery_item_ids}")
print(f"   Gallery (df in AttributeIndex, size {len(gallery_item_ids)}) contains 42066: {'42066' in gallery_item_ids}")

# 4. x018 Benetton: G_full and G_test counts
with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    testset = json.load(f)
g_test_ids = set(str(i) for i in testset["g_test_ids"])

# In attr_idx.df (the gallery of 38,644 in-scope items):
df_gallery = attr_idx.df
m_benetton = df_gallery["brand"].fillna("").astype(str).str.lower().str.contains("benetton")
m_x018 = m_benetton & (df_gallery["articleType"] == "Tshirts") & (df_gallery["gender"] == "Men")
r_full_x018 = m_x018.sum()
r_test_x018 = sum(1 for idx, b in enumerate(m_x018) if b and str(df_gallery.iloc[idx]["item_id"]) in g_test_ids)
print(f"4. x018 Benetton Ground Truth (in_scope): R_full = {r_full_x018}, R_test = {r_test_x018}")

# 5. 20 Failure cases in errata K.2
with open("eval/results_3e.json", "r", encoding="utf-8") as f:
    r3e = json.load(f)
fail_cases = r3e.get("diagnostics", {}).get("failure_cases", [])
print(f"5. Real failure cases count in results_3e.json: {len(fail_cases)}")

all_ids = set()
for c in fail_cases:
    for k in ["c0_top1_id", "c5_top1_id", "query_item_id", "retrieved_top1_id"]:
        if k in c:
            all_ids.add(str(c[k]))

cat_ids = set(df["item_id"].astype(str))
missing = [i for i in all_ids if i not in cat_ids]
print(f"   Total unique IDs in failure cases: {len(all_ids)}")
print(f"   Missing IDs in catalog: {len(missing)} (Expected: 0)")

mismatches_meta = []
for i, c in enumerate(fail_cases, 1):
    cid = c["case_id"]
    if c["type"] == "text_query":
        top1 = str(c["c5_top1_id"])
        row = df[df["item_id"].astype(str) == top1].iloc[0]
        meta = c["c5_top1_metadata"]
        if row["articleType"] != meta["articleType"] or row["gender"] != meta["gender"] or row["baseColour"] != meta["baseColour"]:
            mismatches_meta.append((cid, top1, "meta mismatch"))
        print(f"   {i:2d}. {cid:<10} ID {top1:<6}: art={row['articleType']}, gen={row['gender']}, col={row['baseColour']}")
    else:
        qid = str(c["query_item_id"])
        rid = str(c["retrieved_top1_id"])
        rq = df[df["item_id"].astype(str) == qid].iloc[0]
        rr = df[df["item_id"].astype(str) == rid].iloc[0]
        qm = c["query_metadata"]
        rm = c["retrieved_top1_metadata"]
        if rq["articleType"] != qm["articleType"] or rq["baseColour"] != qm["baseColour"]:
            mismatches_meta.append((cid, qid, "query meta mismatch"))
        if rr["articleType"] != rm["articleType"] or rr["baseColour"] != rm["baseColour"]:
            mismatches_meta.append((cid, rid, "ret meta mismatch"))
        print(f"   {i:2d}. {cid:<12} Query {qid:<6} ({rq['articleType']}, {rq['baseColour']}) -> Top1 {rid:<6} ({rr['articleType']}, {rr['baseColour']})")

print(f"   Metadata mismatches count: {len(mismatches_meta)}")
