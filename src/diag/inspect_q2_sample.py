import json
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    d = json.load(f)
q2_sample = d["q2_attribute_queries"][0]
print("Keys in q2 sample:", list(q2_sample.keys()))
print("target_attributes:", q2_sample["target_attributes"])
print("len strict:", len(q2_sample["target_ids_strict"]))
print("len family:", len(q2_sample["target_ids_family"]))
