import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

splits = set()
for item in meta:
    for k in ["split", "set", "subset", "is_val", "is_test", "is_train"]:
        if k in item:
            splits.add((k, item[k]))
print("Found in product_metadata.json:", splits)
