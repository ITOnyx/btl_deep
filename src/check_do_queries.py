import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import pandas as pd
from src.query_parser import QueryParser
from src.query_parser_v2 import QueryParserV2

p1 = QueryParser()
p2 = QueryParserV2()

def check_queries(query_list, set_name):
    matches = []
    for q in query_list:
        q_str = str(q).strip()
        words = q_str.lower().split()
        has_do = any(w in ['đồ', 'đỏ', 'đo', 'đố', 'độ', 'do', 'dò', 'dó', 'dọ'] or w.startswith('đồ') for w in words)
        if has_do:
            r1 = p1.parse(q_str)
            r2 = p2.parse(q_str)
            diff = (r1['baseColour'] != r2['baseColour']) or (r1['articleType'] != r2['articleType'])
            matches.append({
                'set': set_name,
                'query': q_str,
                'v1_colour': r1.get('baseColour'),
                'v2_colour': r2.get('baseColour'),
                'v1_type': r1.get('articleType'),
                'v2_type': r2.get('articleType'),
                'v1_gender': r1.get('gender'),
                'v2_gender': r2.get('gender'),
                'v1_usage': r1.get('usage'),
                'v2_usage': r2.get('usage'),
                'changed': diff
            })
    return matches

results = []

# 1. Heldout 30
if os.path.exists('eval/ab_blind_heldout.csv'):
    df_heldout = pd.read_csv('eval/ab_blind_heldout.csv')
    results.extend(check_queries(df_heldout['query'].tolist(), 'Heldout 30'))

# 2. Q3 handwritten (100)
if os.path.exists('eval/handwritten_queries.json'):
    q3_data = json.load(open('eval/handwritten_queries.json', encoding='utf-8'))
    if isinstance(q3_data, list):
        q3_queries = [item['query'] if isinstance(item, dict) else item for item in q3_data]
    else:
        q3_queries = list(q3_data.keys())
    results.extend(check_queries(q3_queries, 'Q3 Handwritten'))

# 3. Q2 attribute queries (61)
if os.path.exists('eval/testset_v1.json'):
    ts = json.load(open('eval/testset_v1.json', encoding='utf-8'))
    q2_queries = [item['query_text'] for item in ts.get('q2_attribute_queries', [])]
    results.extend(check_queries(q2_queries, 'Q2 Attribute'))

# 4. Q5 brand queries (150)
if os.path.exists('eval/q5_brand.json'):
    q5_data = json.load(open('eval/q5_brand.json', encoding='utf-8'))
    q5_queries = [item.get('query_text', item.get('query', str(item))) for item in q5_data.get('queries', [])]
    results.extend(check_queries(q5_queries, 'Q5 Brand'))

print(f"Total queries evaluated: {len(results)}")
changed = [r for r in results if r['changed']]
print(f"Queries with changes between V1 and V2: {len(changed)}")
for c in changed:
    print(f"[{c['set']}] '{c['query']}': V1 Colour={c['v1_colour']} -> V2 Colour={c['v2_colour']}")

out_file = Path('reports/phase3g_do_bug_investigation.json')
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump({
        'summary': 'Investigation of do -> Red parser bug and verification of QueryParserV2',
        'root_cause': 'In QueryParser V1, diacritics removal mapped both grave-accented Vietnamese word \"đồ\" (apparel/clothing) and question-accented word \"đỏ\" (Red) to unaccented token \"do\", triggering false-positive baseColour=\"Red\" match in lexicon_vi.json.',
        'fixed_in_parser_v2': True,
        'total_queries_with_do_or_red': len(results),
        'total_changed_queries': len(changed),
        'changed_queries': changed,
        'all_queries': results
    }, f, indent=2, ensure_ascii=False)

print(f"Saved report to {out_file}.")
