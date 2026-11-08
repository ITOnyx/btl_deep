import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

keywords = ['vit_zero', 'csls', 'cone_restore', 'stats', 'rerank', 're-rank', 'hub', 'beta', 'eval_image_transform', 'clean_record', 'split', 'mu_shared', 'mu_img']

matches = {}
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    matched = [kw for kw in keywords if kw in src.lower()]
    if matched:
        matches[i] = matched

for idx, kws in matches.items():
    print(f"Cell {idx}: {kws}")
