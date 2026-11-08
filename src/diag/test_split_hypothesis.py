import pandas as pd
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

# Load 1372 known test IDs
with open("data/checkpoints_siglip-vit/dense_captions_catalog.json", "r", encoding="utf-8") as f:
    dense_caps = json.load(f)

myntra_test_ids = set()
for pid in dense_caps.keys():
    if pid.isdigit() and len(pid) <= 6:
        myntra_test_ids.add(int(pid))

print(f"Known Myntra test IDs: {len(myntra_test_ids)}")

# Check styles.csv
df = pd.read_csv("data/fashion-dataset/styles.csv", on_bad_lines='skip')
print(f"Total rows in styles.csv: {len(df)}")

all_myntra_ids = set(df['id'].values)
print(f"How many of known test IDs are in styles.csv: {len(myntra_test_ids.intersection(all_myntra_ids))}")

for test_size in [0.03, 0.031, 0.03125, 0.05, 0.1]:
    for seed in [42, 0, 1, 1234]:
        train_ids, test_ids = train_test_split(df['id'].values, test_size=test_size, random_state=seed)
        overlap = len(myntra_test_ids.intersection(set(test_ids)))
        if overlap > 100:
            print(f"Test size {test_size}, seed {seed}: overlap {overlap} / {len(myntra_test_ids)}")
