import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open("reports/phase3/phase3g/phase3g_brands_v2_eval.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for item in d.get("heldout_brand_queries_detail", []):
    print(json.dumps(item, ensure_ascii=False))
