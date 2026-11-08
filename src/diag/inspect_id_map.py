import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("index/id_map.json", "r", encoding="utf-8") as f:
    id_map = json.load(f)

print(f"id_map type: {type(id_map)}, len: {len(id_map)}")
print(f"First 5 elements: {id_map[:5]}")
print(f"Type of elements: {[type(x) for x in id_map[:5]]}")
print(f"Is 15970 in id_map: {15970 in id_map or '15970' in id_map}")
