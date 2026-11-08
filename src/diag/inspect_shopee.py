import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

try:
    with open('notebooks/shopee-v03.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"shopee-v03.ipynb cells: {len(nb.get('cells', []))}")
    kws = ["stats", "csls", "mu_shared", "cone", "hub"]
    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        matched = [k for k in kws if k in src.lower()]
        if matched:
            print(f"Cell {i}: {matched}")
except Exception as e:
    print(f"Error reading shopee-v03: {e}")
