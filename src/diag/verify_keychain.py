import json
from pathlib import Path
import pandas as pd

# 1. Catalog check
cat = pd.read_parquet('data/catalog/catalog.parquet')
in_scope_cat = cat[cat['in_scope'] == True]
print(f"Catalog total rows: {len(cat)}")
print(f"Catalog in_scope count: {len(in_scope_cat)}")

kc_ids = [43230, 42066]
for kid in kc_ids:
    row = cat[cat['item_id'] == kid]
    if len(row) > 0:
        print(f"Item {kid}: articleType={row['articleType'].values[0]}, in_scope={row['in_scope'].values[0]}")
    else:
        print(f"Item {kid} not found")

# 2. Check all Key chains in catalog
all_kc = cat[cat['articleType'] == 'Key chain']
print(f"Total 'Key chain' in catalog.parquet: {len(all_kc)}")
print(f"In-scope 'Key chain': {(all_kc['in_scope'] == True).sum()}")
print(f"Out-of-scope 'Key chain': {(all_kc['in_scope'] == False).sum()}")

# 3. index_v2/manifest.json check
idx_m = json.load(open('index_v2/manifest.json', encoding='utf-8'))
print("index_v2/manifest.json:")
for k, v in idx_m.items():
    print(f"  {k}: {v}")

# 4. Check embeddings shape
import numpy as np
emb_path = Path('index_v2/embeddings.npy')
if emb_path.exists():
    # memory map to inspect shape without loading whole array
    emb = np.load(emb_path, mmap_mode='r')
    print(f"index_v2/embeddings.npy shape: {emb.shape}, dtype: {emb.dtype}")

# 5. Check phase2a_findings.md
p2a_path = Path('reports/phase2a_findings.md')
if p2a_path.exists():
    for line in p2a_path.read_text(encoding='utf-8').split('\n'):
        if 'key chain' in line.lower() or 'keychain' in line.lower() or ('35' in line and 'scope' in line.lower()):
            print(f"phase2a_findings: {line.strip()}")
