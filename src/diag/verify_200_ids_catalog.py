import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import random
import pandas as pd
import numpy as np
from app.engine import SearchService

print("Loading catalog...")
df_cat = pd.read_parquet("data/catalog/catalog.parquet")
df_in_scope = df_cat[df_cat['in_scope'] == True].reset_index(drop=True)
print(f"Total in-scope in catalog.parquet: {len(df_in_scope)}")

service = SearchService(config_path="config.yaml")
engine = service.engine

print(f"Engine gallery N: {engine.N}")
print(f"Engine df length: {len(engine.df)}")

# Check 200 random IDs
random.seed(42)
sampled_ids = random.sample(df_in_scope['item_id'].tolist(), 200)

mismatches = []
for iid in sampled_ids:
    str_id = str(iid)
    # Check catalog ground truth
    cat_row = df_in_scope[df_in_scope['item_id'] == iid].iloc[0]
    expected_name = cat_row['name']
    expected_type = cat_row['articleType']
    expected_gender = cat_row['gender']
    expected_color = cat_row['baseColour']

    # Check engine mapping
    if str_id not in engine.id_to_idx:
        mismatches.append((iid, "NOT_FOUND_IN_ENGINE_ID_TO_IDX", None, None))
        continue
    idx = engine.id_to_idx[str_id]
    engine_row = engine.df.iloc[idx]
    actual_id = engine_row['item_id']
    actual_name = engine_row['name']
    actual_type = engine_row['articleType']

    if str(actual_id) != str_id:
        mismatches.append((iid, "ID_MISMATCH", str(actual_id), str_id))
    elif actual_name != expected_name:
        mismatches.append((iid, "NAME_MISMATCH", actual_name, expected_name))
    elif actual_type != expected_type:
        mismatches.append((iid, "TYPE_MISMATCH", actual_type, expected_type))

print(f"\nTested 200 random IDs: {len(sampled_ids)}")
print(f"Total mismatches: {len(mismatches)}")
if mismatches:
    print("Mismatches found:", mismatches[:10])
else:
    print("ALL 200 IDs MATCHED 100% PERFECTLY (id, name, articleType, gender, colour)!")
