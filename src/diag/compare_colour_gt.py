import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.attr_index import AttributeIndex

def main():
    attr_idx = AttributeIndex(brands_path='src/brands_v2.json')
    df = attr_idx.df
    N = len(df)
    g_test_mask = (df['split'] == 'test').values

    with open('eval/heldout_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)

    headers = [
        "qid", "Câu truy vấn", "Thuộc tính spec",
        "R_full CŨ", "R_full MỚI", "R_test CŨ", "R_test MỚI", "Lệch?"
    ]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---'] * len(headers))}|")

    diff_queries = []
    for q in queries:
        qid = q.get('id') or q.get('qid')
        query = q['query']
        spec = q.get('spec')
        if spec is None:
            continue

        # 1. Old GT (spec.get('baseColour'), no brand or flawed brand)
        m_old_full = np.ones(N, dtype=bool)
        m_old_test = g_test_mask.copy()
        if spec.get('articleType'):
            m_old_full &= np.isin(df['articleType'].values, spec['articleType'])
            m_old_test &= np.isin(df['articleType'].values, spec['articleType'])
        if spec.get('gender'):
            m_old_full &= (df['gender'].values == spec['gender'])
            m_old_test &= (df['gender'].values == spec['gender'])
        if spec.get('baseColour'):
            m_old_full &= np.isin(df['baseColour'].values, spec['baseColour'])
            m_old_test &= np.isin(df['baseColour'].values, spec['baseColour'])
        r_old_full = int(m_old_full.sum())
        r_old_test = int(m_old_test.sum())

        # 2. New GT (spec.get('colour') or spec.get('baseColour'), plus spec.get('brand') with alias)
        cols = spec.get('colour') or spec.get('baseColour')
        b = spec.get('brand')
        m_new_full = np.ones(N, dtype=bool)
        m_new_test = g_test_mask.copy()
        if spec.get('articleType'):
            m_new_full &= np.isin(df['articleType'].values, spec['articleType'])
            m_new_test &= np.isin(df['articleType'].values, spec['articleType'])
        if spec.get('gender'):
            m_new_full &= (df['gender'].values == spec['gender'])
            m_new_test &= (df['gender'].values == spec['gender'])
        if cols:
            m_new_full &= np.isin(df['baseColour'].values, cols)
            m_new_test &= np.isin(df['baseColour'].values, cols)
        if b:
            b_target = 'United Colors of Benetton' if b == 'Benetton' else b
            m_new_full &= (df['brand'].str.lower() == b_target.lower())
            m_new_test &= (df['brand'].str.lower() == b_target.lower())
        r_new_full = int(m_new_full.sum())
        r_new_test = int(m_new_test.sum())

        is_diff = (r_old_full != r_new_full) or (r_old_test != r_new_test)
        if is_diff:
            diff_queries.append(qid)

        spec_desc = []
        if cols:
            spec_desc.append(f"Màu: {cols}")
        if b:
            spec_desc.append(f"Brand: {b}")
        if not spec_desc:
            spec_desc.append("Loại + Giới tính")
        desc_str = "; ".join(spec_desc)

        diff_label = "**CÓ LỆCH**" if is_diff else "Khớp"
        print(f"| `{qid}` | {query} | {desc_str} | {r_old_full} | {r_new_full} | {r_old_test} | {r_new_test} | {diff_label} |")

    print(f"\nTổng số câu spec bị lệch: {len(diff_queries)} / 26 câu ({', '.join(diff_queries)})")

if __name__ == '__main__':
    main()
