import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

meta_dict = {str(item["product_id"]): item for item in meta}

with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    testset = json.load(f)

g_test_ids = [str(x) for x in testset["g_test_ids"]]
print(f"Total g_test_ids: {len(g_test_ids)}")

found = [tid for tid in g_test_ids if tid in meta_dict]
print(f"Found in product_metadata.json: {len(found)} / {len(g_test_ids)}")

has_search_text = [tid for tid in found if meta_dict[tid].get("search_text")]
print(f"Has search_text: {len(has_search_text)} / {len(found)}")

print("\nSample 10 search_text from product_metadata.json for test items:")
for tid in g_test_ids[:10]:
    if tid in meta_dict:
        print(f"ID {tid}: {meta_dict[tid]['search_text']}")
