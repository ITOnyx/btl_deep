import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'max_eval_per_source' in src:
        print(f"=== CELL {idx} ===")
        print(src)
