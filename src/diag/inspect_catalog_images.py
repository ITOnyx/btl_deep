import pandas as pd
df = pd.read_parquet("data/catalog/catalog.parquet")
print("Columns in catalog:", list(df.columns))
print("Sample path:", df["path"].head(5).tolist())
import os
for p in df["path"].head(5):
    print(f"Exists {p}?", os.path.exists(p))
