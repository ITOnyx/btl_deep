import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/checkpoints_siglip-vit/dense_captions_catalog.json", "r", encoding="utf-8") as f:
    dense_caps = json.load(f)

print(f"Type: {type(dense_caps)}")
if isinstance(dense_caps, dict):
    print("Dict keys sample:", list(dense_caps.keys())[:10])
    first_key = list(dense_caps.keys())[0]
    print("First val:", dense_caps[first_key])
elif isinstance(dense_caps, list):
    print("List len:", len(dense_caps))
    print("First elem:", dense_caps[0])
