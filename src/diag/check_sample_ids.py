import pandas as pd

df = pd.read_parquet("data/catalog/catalog.parquet")
for iid in [15970, 39386, 1855, 59263]:
    sub = df[df['item_id'].astype(str) == str(iid)]
    if len(sub) > 0:
        row = sub.iloc[0]
        print(f"ID {iid}: Name='{row['name']}', Type='{row['articleType']}', Gender='{row['gender']}', Colour='{row['baseColour']}'")
    else:
        print(f"ID {iid}: Not found")
