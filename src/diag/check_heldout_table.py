import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.attr_index import AttributeIndex

attr_idx = AttributeIndex(brands_path='src/brands_v2.json')
df = attr_idx.df
N = len(df)
item_ids = [str(x) for x in attr_idx.item_ids]
id_to_df_idx = {iid: idx for idx, iid in enumerate(item_ids)}

with open('eval/testset_v1.json', 'r', encoding='utf-8') as f:
    testset = json.load(f)

g_test_ids = [str(i) for i in testset['g_test_ids']]
g_full_ids = [str(i) for i in testset['g_full_ids']]

g_test_df_indices = np.array([id_to_df_idx[tid] for tid in g_test_ids if tid in id_to_df_idx], dtype=np.int64)
g_full_df_indices = np.array([id_to_df_idx[tid] for tid in g_full_ids if tid in id_to_df_idx], dtype=np.int64)

g_test_mask = np.zeros(N, dtype=bool)
g_test_mask[g_test_df_indices] = True
g_full_mask = np.zeros(N, dtype=bool)
g_full_mask[g_full_df_indices] = True

with open('eval/heldout_queries.json', 'r', encoding='utf-8') as f:
    heldout_queries = json.load(f)

print(f"Total heldout queries in file: {len(heldout_queries)}")
print(f"{'qid':<6} | {'query':<35} | {'R_full':<7} | {'R_test':<7} | {'valid_full':<10} | {'valid_test':<10}")
print("-" * 85)

valid_full_count = 0
valid_test_count = 0
results = []

df_brand_lower = df['brand'].fillna('').astype(str).str.lower().values

for q in heldout_queries:
    qid = q.get('id', q.get('query_id', 'unknown'))
    q_text = q['query']
    spec = q.get('spec') or {}
    
    def get_rel(mask):
        m = np.ones(N, dtype=bool)
        if spec.get('articleType'):
            m &= np.isin(df['articleType'].values, spec['articleType'])
        if spec.get('gender'):
            m &= (df['gender'].values == spec['gender'])
        colors = spec.get('colour') or spec.get('baseColour')
        if colors:
            m &= np.isin(df['baseColour'].values, colors)
        if spec.get('brand'):
            b = spec['brand'].lower()
            if b == 'benetton':
                m &= df['brand'].fillna('').str.lower().str.contains('benetton').values
            else:
                m &= (df_brand_lower == b)
        m &= mask
        return set(df.loc[m, 'item_id'].astype(str))
    
    rel_full = get_rel(g_full_mask)
    rel_test = get_rel(g_test_mask)
    r_full = len(rel_full)
    r_test = len(rel_test)
    vf = r_full > 0
    vt = r_test > 0
    if vf: valid_full_count += 1
    if vt: valid_test_count += 1
    results.append({
        'qid': qid, 'query': q_text, 'spec': spec,
        'r_full': r_full, 'r_test': r_test,
        'valid_full': vf, 'valid_test': vt
    })
    print(f"{qid:<6} | {q_text:<35} | {r_full:<7} | {r_test:<7} | {str(vf):<10} | {str(vt):<10}")

print("-" * 85)
print(f"Total valid: G_full = {valid_full_count}, G_test = {valid_test_count}")
