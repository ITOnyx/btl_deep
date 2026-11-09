import sys
sys.path.insert(0, ".")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import json
import numpy as np
from src.attr_index import AttributeIndex

attr_idx = AttributeIndex(brands_path="src/brands_v2.json")
df = attr_idx.df
with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
    heldout = json.load(f)
heldout_spec = [q for q in heldout if q.get("spec") is not None]

with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    testset = json.load(f)

g_test_ids = set(str(i) for i in testset["g_test_ids"])
g_full_ids = set(str(i) for i in testset["g_full_ids"])

item_ids = [str(x) for x in attr_idx.item_ids]
id_to_df_idx = {iid: idx for idx, iid in enumerate(item_ids)}

df_art = df["articleType"].values
df_gen = df["gender"].values
df_col = df["baseColour"].values
df_brand_lower = np.array([b.lower() for b in df["brand"].fillna("").astype(str).values])
item_ids_arr = np.array(item_ids)
N = len(df)

g_test_mask = np.zeros(N, dtype=bool)
for tid in g_test_ids:
    if tid in id_to_df_idx:
        g_test_mask[id_to_df_idx[tid]] = True

g_full_mask = np.zeros(N, dtype=bool)
for tid in g_full_ids:
    if tid in id_to_df_idx:
        g_full_mask[id_to_df_idx[tid]] = True

with open("eval/families.json", "r", encoding="utf-8") as f:
    fams = json.load(f)
gender_compat = fams.get("gender_compatibility", {
    "Men": ["Men", "Boys", "Unisex"],
    "Women": ["Women", "Girls", "Unisex"],
    "Boys": ["Boys", "Men", "Unisex"],
    "Girls": ["Girls", "Women", "Unisex"],
    "Unisex": ["Unisex", "Men", "Women", "Boys", "Girls"]
})

for g_name, g_mask, g_id_set in [("G_test", g_test_mask, g_test_ids), ("G_full", g_full_mask, g_full_ids)]:
    skipped = []
    print(f"\n=======================================================")
    print(f"=== {g_name} ===")
    print(f"=======================================================")
    for q in heldout_spec:
        spec = q.get("spec", {})
        spec_arts = spec.get("articleType") or []
        spec_gen = spec.get("gender")
        spec_cols = spec.get("colour") or spec.get("baseColour") or []
        spec_brand = spec.get("brand")
        compat_gens = gender_compat.get(spec_gen, [spec_gen, "Unisex"]) if spec_gen else None
        m_art = np.isin(df_art, spec_arts) if spec_arts else True
        m_gen = np.isin(df_gen, compat_gens) if compat_gens else True
        m_col = np.isin(df_col, spec_cols) if spec_cols else True
        m_brd = (df_brand_lower == spec_brand.lower()) if spec_brand else True
        strict_mask = m_art & m_gen & m_col & m_brd & g_mask
        n_rel = int(np.sum(strict_mask))
        if n_rel == 0:
            skipped.append((q["id"], q["query"], spec))
        print(f"{q['id']}: '{q['query']}' -> {n_rel} items")
    print(f"\nTotal valid queries: {len(heldout_spec) - len(skipped)}/{len(heldout_spec)}")
    print(f"Skipped in {g_name} (n_rel == 0):")
    for s in skipped:
        print("  ", s)
