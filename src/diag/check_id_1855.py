import pandas as pd
import json

df = pd.read_parquet("data/catalog/catalog.parquet")
print(f"Catalog shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Look for item_id 1855
row_id = df[df['item_id'].astype(str) == '1855']
print("\nRow where item_id == '1855':")
print(row_id[['item_id', 'name', 'articleType', 'gender', 'baseColour', 'masterCategory']])

# Look for iloc 1855
print("\nRow at iloc[1855]:")
print(df.iloc[1855][['item_id', 'name', 'articleType', 'gender', 'baseColour', 'masterCategory']])


# Check what smoke test output was
with open("reports/phase4/phase4b/phase4_smoke.json", "r", encoding="utf-8") as f:
    smoke = json.load(f)
print("\nSmoke test keys:", list(smoke.keys()))
for test_key in smoke:
    res = smoke[test_key]
    if isinstance(res, list):
        for item in res:
            if '1855' in str(item):
                print(f"Found in {test_key}:", item)
    elif isinstance(res, dict):
        if '1855' in str(res):
            print(f"Found in {test_key}:", res)
