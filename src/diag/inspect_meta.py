import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"Total items in product_metadata.json: {len(meta)}")
print("Sample item 0:")
print(json.dumps(meta[0], indent=2, ensure_ascii=False))

# check if search_text or captions or attributes exist
keys = list(meta[0].keys())
print("Keys:", keys)
