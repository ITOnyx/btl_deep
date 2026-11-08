import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open('notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][13]['source'])
lines = src.split('\n')
for i, line in enumerate(lines[120:220]):
    print(f"{i+121:3d}: {line}")
