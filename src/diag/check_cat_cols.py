import pandas as pd
df = pd.read_parquet("data/catalog/catalog.parquet")
print(list(df.columns))
