import sys
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def main():
    df = pd.read_parquet('data/catalog/catalog.parquet')
    with open('eval/heldout_queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Use in_scope gallery (38,644 items)
    in_scope_mask = (df['in_scope'] == True).values
    df_scope = df[in_scope_mask].reset_index(drop=True)
    N = len(df_scope)
    g_test_mask = (df_scope['split'] == 'test').values
    g_full_mask = np.ones(N, dtype=bool)

    headers = [
        "qid", "Câu truy vấn", "Màu sắc spec",
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

        # 1. Old GT (spec.get('baseColour'))
        m_old_full = np.ones(N, dtype=bool)
        m_old_test = g_test_mask.copy()
        if spec.get('articleType'):
            m_old_full &= np.isin(df_scope['articleType'].values, spec['articleType'])
            m_old_test &= np.isin(df_scope['articleType'].values, spec['articleType'])
        if spec.get('gender'):
            m_old_full &= (df_scope['gender'].values == spec['gender'])
            m_old_test &= (df_scope['gender'].values == spec['gender'])
        if spec.get('baseColour'):
            m_old_full &= np.isin(df_scope['baseColour'].values, spec['baseColour'])
            m_old_test &= np.isin(df_scope['baseColour'].values, spec['baseColour'])
        r_old_full = int(m_old_full.sum())
        r_old_test = int(m_old_test.sum())

        # 2. New GT (spec.get('colour') or spec.get('baseColour'))
        cols = spec.get('colour') or spec.get('baseColour')
        m_new_full = np.ones(N, dtype=bool)
        m_new_test = g_test_mask.copy()
        if spec.get('articleType'):
            m_new_full &= np.isin(df_scope['articleType'].values, spec['articleType'])
            m_new_test &= np.isin(df_scope['articleType'].values, spec['articleType'])
        if spec.get('gender'):
            m_new_full &= (df_scope['gender'].values == spec['gender'])
            m_new_test &= (df_scope['gender'].values == spec['gender'])
        if cols:
            m_new_full &= np.isin(df_scope['baseColour'].values, cols)
            m_new_test &= np.isin(df_scope['baseColour'].values, cols)
        r_new_full = int(m_new_full.sum())
        r_new_test = int(m_new_test.sum())

        is_diff = (r_old_full != r_new_full) or (r_old_test != r_new_test)
        if is_diff:
            diff_queries.append(qid)
        col_str = str(cols) if cols else "—"
        diff_label = "**CÓ LỆCH**" if is_diff else "Không"
        print(f"| `{qid}` | {query} | `{col_str}` | {r_old_full} | {r_new_full} | {r_old_test} | {r_new_test} | {diff_label} |")

    print(f"\nTổng số câu spec bị ảnh hưởng bởi bug 'baseColour' vs 'colour': {len(diff_queries)} / 26 câu ({', '.join(diff_queries)})")

if __name__ == '__main__':
    main()
