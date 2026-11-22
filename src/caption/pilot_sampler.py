"""Sample 200 pilot images stratified by articleType x gender from train split."""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def sample_pilot_200(
    catalog_path="data/catalog/catalog_rebuild.parquet",
    output_json="eval/pilot_200_sample.json",
    n_samples=200,
    seed=42
):
    df = pd.read_parquet(catalog_path)
    # Strictly from train and in_scope with pass status (ZERO TEST LEAKAGE)
    train_pool = df[(df["split"] == "train") & (df["filter_flag"] == "pass") & df["in_scope"]].copy()
    print(f"Available train pool for pilot: {len(train_pool)} items")

    train_pool["strata"] = train_pool["articleType"] + "::" + train_pool["gender"]
    strata_counts = train_pool["strata"].value_counts()
    
    rng = np.random.default_rng(seed=seed)
    
    # Proportional stratified sampling
    total_pool = len(train_pool)
    sampled_indices = []

    # Allocate at least 1 sample per major stratum
    for stratum, count in strata_counts.items():
        if count >= 10:
            target = max(1, int(round((count / total_pool) * n_samples)))
            stratum_idxs = train_pool[train_pool["strata"] == stratum].index.values
            chosen = rng.choice(stratum_idxs, size=min(target, len(stratum_idxs)), replace=False)
            sampled_indices.extend(chosen)

    # Adjust to exactly n_samples
    sampled_indices = list(set(sampled_indices))
    if len(sampled_indices) < n_samples:
        remaining = [i for i in train_pool.index.values if i not in sampled_indices]
        needed = n_samples - len(sampled_indices)
        extra = rng.choice(remaining, size=needed, replace=False)
        sampled_indices.extend(extra)
    elif len(sampled_indices) > n_samples:
        sampled_indices = list(rng.choice(sampled_indices, size=n_samples, replace=False))

    sampled_df = train_pool.loc[sampled_indices].sort_values("product_id")
    
    # Select 20 items for manual review across diverse categories
    manual_20_indices = rng.choice(sampled_df.index.values, size=20, replace=False)
    sampled_df["for_manual_review"] = sampled_df.index.isin(manual_20_indices)

    # Export records
    pilot_records = sampled_df.to_dict(orient="records")
    
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pilot_records, f, indent=2, ensure_ascii=False)

    print(f"✅ Sampled {len(pilot_records)} pilot items to {output_json} (Seed={seed})")
    print(f"   Manual review items: {sampled_df['for_manual_review'].sum()}")
    return pilot_records


if __name__ == "__main__":
    sample_pilot_200()
