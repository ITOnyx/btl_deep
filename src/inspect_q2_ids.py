import json
with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    d = json.load(f)
q2_sample = d["q2_attribute_queries"][0]
print("target_ids_strict:", q2_sample["target_ids_strict"])
print("type of first item:", type(q2_sample["target_ids_strict"][0]))
g_test_sample = d["g_test_ids"][:5]
print("g_test sample:", g_test_sample)
print("type of first g_test:", type(g_test_sample[0]))

# Check if target_ids_strict are in g_test_ids
tids = set(str(x) for x in q2_sample["target_ids_strict"])
g_test_set = set(str(x) for x in d["g_test_ids"])
print("Intersection with g_test:", tids.intersection(g_test_set))
