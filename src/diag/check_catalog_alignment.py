import json
import pandas as pd
import numpy as np
from pathlib import Path

# Load catalog
df = pd.read_parquet("data/catalog/catalog.parquet")
df_in_scope = df[df['in_scope'] == True].reset_index(drop=True)
print(f"Total catalog rows: {len(df)}, in_scope rows: {len(df_in_scope)}")

# Check id_map_v2.json
with open("index_v2/id_map_v2.json", "r", encoding="utf-8") as f:
    id_map = json.load(f)
print(f"id_map length: {len(id_map)}")

id_map_ids = [str(x["item_id"]) for x in id_map]
catalog_in_scope_ids = [str(x) for x in df_in_scope["item_id"]]

print(f"id_map_ids[:5]: {id_map_ids[:5]}")
print(f"catalog_in_scope_ids[:5]: {catalog_in_scope_ids[:5]}")

# Are the IDs in the same order?
in_scope_set = set(catalog_in_scope_ids)
valid_indices = [i for i, iid in enumerate(id_map_ids) if iid in in_scope_set]
filtered_id_map_ids = [id_map_ids[i] for i in valid_indices]

print(f"Are filtered_id_map_ids == catalog_in_scope_ids? {filtered_id_map_ids == catalog_in_scope_ids}")

if filtered_id_map_ids != catalog_in_scope_ids:
    print("MISMATCH IN ORDER!")
    for idx in range(min(10, len(filtered_id_map_ids))):
        print(f"Index {idx}: id_map has {filtered_id_map_ids[idx]}, catalog_in_scope has {catalog_in_scope_ids[idx]}")
else:
    print("ORDER MATCHES PERFECTLY!")
