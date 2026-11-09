import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)
print("q2 count:", len(t1.get("q2_attribute_queries", [])))
print("q2 sample:", t1.get("q2_attribute_queries", [])[0])

with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
    q3 = json.load(f)
print("\nq3 count:", len(q3))
print("q3 sample:", q3[0])

with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
    q5 = json.load(f)
print("\nq5 count:", len(q5))
print("q5 sample:", q5[0])

with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
    h = json.load(f)
print("\nheldout count:", len(h))
print("heldout sample:", h[0])
