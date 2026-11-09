import pandas as pd

df = pd.read_parquet("data/catalog/catalog.parquet")
print(df['split'].value_counts())
print("\nIn-scope by split:")
print(df[df['in_scope'] == True]['split'].value_counts())
