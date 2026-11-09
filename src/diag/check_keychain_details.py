import pandas as pd
from pathlib import Path

# 1. In catalog.parquet
cat = pd.read_parquet('data/catalog/catalog.parquet')
kc_cat = cat[cat['articleType'] == 'Key chain']
print("--- KEY CHAIN IN CATALOG.PARQUET ---")
print(f"Total Key chain in catalog: {len(kc_cat)}")
for idx, r in kc_cat.iterrows():
    print(f"  ID: {r['item_id']}, in_scope: {r['in_scope']}, split: {r['split']}, name: {r['name']}")

# Table breakdown
print("\nBreakdown by in_scope & split in catalog:")
print(kc_cat.groupby(['in_scope', 'split']).size())

# 2. In quarantine_log.csv
q_path = Path('reports/quarantine_log.csv')
if q_path.exists():
    df_q = pd.read_csv(q_path)
    print(f"\nquarantine_log columns: {df_q.columns.tolist()}")
    if 'articleType' in df_q.columns:
        kc_q = df_q[df_q['articleType'] == 'Key chain']
        print(f"Total Key chain in quarantine_log.csv: {len(kc_q)}")
        if 'reason' in kc_q.columns:
            print(kc_q['reason'].value_counts())
    elif 'id' in df_q.columns:
        # join with raw styles to check articleType
        pass

# 3. Check original raw styles.csv
raw_p = Path('data/fashion-dataset/styles.csv')
if raw_p.exists():
    df_raw = pd.read_csv(raw_p, on_bad_lines='skip')
    raw_kc = df_raw[df_raw['articleType'] == 'Key chain']
    print(f"\nTotal Key chain in raw fashion-dataset/styles.csv: {len(raw_kc)}")
    print(f"List of raw Key chain IDs ({len(raw_kc)}): {raw_kc['id'].tolist()}")
