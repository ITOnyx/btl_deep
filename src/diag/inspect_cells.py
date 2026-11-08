import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [8, 13]:
    print(f"\n==================== CELL {idx} ====================")
    print(''.join(nb['cells'][idx]['source']))
