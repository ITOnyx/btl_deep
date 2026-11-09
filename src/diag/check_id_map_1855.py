import json

with open("index_v2/id_map_v2.json", "r", encoding="utf-8") as f:
    id_map = json.load(f)

print(f"id_map[1855]: {id_map[1855]}")
