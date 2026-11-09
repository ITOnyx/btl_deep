import json

with open("reports/phase3/phase3g/phase3g_brands_v2_eval.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("heldout_brand_queries_detail:")
for q in d.get("heldout_brand_queries_detail", []):
    print(f"qid={q.get('qid')} brand={q.get('brand')} R={q.get('relevant_count')} AP10_v1={q.get('ap10_v1')} AP10_v2={q.get('ap10_v2')}")
