import json
from pathlib import Path

for p in [Path("index/id_map.json"), Path("index_v1/id_map.json")]:
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            print(f"{p}[1855]: {d[1855] if len(d) > 1855 else 'too short'}")
