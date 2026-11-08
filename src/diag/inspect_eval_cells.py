import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(30, len(nb['cells'])):
    cell = nb['cells'][idx]
    ctype = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    first_lines = '\n'.join([line for line in src.split('\n')[:3] if line.strip()])
    print(f"Cell {idx:2d} ({ctype}): {first_lines[:120]}")
