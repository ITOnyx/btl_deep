import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('eval/results_3e.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

fc = d.get('diagnostics', {}).get('failure_cases', [])
print(f"Total failure cases in JSON: {len(fc)}")
for i, c in enumerate(fc):
    cid = c.get('case_id')
    qtype = c.get('type')
    query = c.get('query')
    target = c.get('target_spec')
    c0 = c.get('c0_top1_id')
    c5 = c.get('c5_top1_id')
    meta = c.get('c5_top1_metadata')
    p10 = c.get('c5_precision_at_10')
    analysis = c.get('failure_analysis')
    print(f"{i+1}. [{cid}] ({qtype}) Query: \"{query}\" | Target: {target} | C0 Top1: {c0} | C5 Top1: {c5} (Meta: {meta}) | P@10: {p10} | Note: {analysis}")
