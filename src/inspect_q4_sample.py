import json
with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    d = json.load(f)
q4_sample = d["q4_image_queries"][0]
print("Keys in q4 sample:", list(q4_sample.keys()))
for k, v in q4_sample.items():
    if isinstance(v, list) and len(v) > 5:
        print(f"  {k}: list of length {len(v)}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    else:
        print(f"  {k}: {v}")
