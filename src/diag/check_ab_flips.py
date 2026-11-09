import json
import numpy as np

seed = 20261109
rng = np.random.RandomState(seed)
generated_flips = [int(rng.randint(0, 2)) for _ in range(30)]

print(f"Generated 30 coin flips with RandomState({seed}):")
print(generated_flips)
print(f"Length: {len(generated_flips)}")
print(f"Count of 1: {generated_flips.count(1)}, Count of 0: {generated_flips.count(0)}")

# Check eval/_sealed/ab_mapping.json
with open("eval/_sealed/ab_mapping.json", "r", encoding="utf-8") as f:
    ab_map = json.load(f)

print("\n--- ab_mapping.json inspection ---")
print(f"Number of queries in ab_mapping.json: {len(ab_map)}")
mapping_flips = []
for qid in sorted(ab_map.keys(), key=lambda x: int(x.replace('x', '')) if x.replace('x', '').isdigit() else x):
    entry = ab_map[qid]
    # Check which row was Config 5 vs Config 0
    row_a = entry.get("row_a", entry.get("A", ""))
    row_b = entry.get("row_b", entry.get("B", ""))
    # In recreate_blind_ab_rev3.py:
    # if coin_flip == 0: row_a = Config 0, row_b = Config 5
    # else: row_a = Config 5, row_b = Config 0
    # So if row_a is Config 5 -> flip was 1; if row_a is Config 0 -> flip was 0
    if "Config 5" in row_a:
        mapping_flips.append(1)
    elif "Config 0" in row_a:
        mapping_flips.append(0)
    else:
        mapping_flips.append(entry.get("coin_flip", None))

print(f"Mapping flips: {mapping_flips}")
print(f"Length: {len(mapping_flips)}")
print(f"Does generated_flips == mapping_flips? {generated_flips == mapping_flips}")

# Now let's check what was written in phase3f_errata.md:
# `[1, 1, 1, 1, 1, 1 | 0, 0, 0, 0, 0 | 1, 1, 1, 1 | 0, 0 | 1, 1 | 0, 0, 0, 0, 0 | 1 | 0 | 1, 1, 1]`
errata_runs = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
print(f"\nErrata sequence in phase3f_errata.md has length: {len(errata_runs)}")
print(f"Errata sequence elements: {errata_runs}")
