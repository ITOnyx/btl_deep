"""
Update catalog.parquet with `in_scope` and `split` columns.
"""
import json
import pathlib
import pandas as pd

def main():
    catalog_path = pathlib.Path("data/catalog/catalog.parquet")
    df = pd.read_parquet(catalog_path)
    print(f"Loaded catalog: {len(df)} items.")

    # 1. Update in_scope from reports/scope_proposal.csv
    scope_path = pathlib.Path("reports/scope_proposal.csv")
    df_scope = pd.read_csv(scope_path)
    scope_map = {
        (r["articleType"], r["masterCategory"], r["gender"]): bool(r["in_scope"])
        for _, r in df_scope.iterrows()
    }

    in_scope_vals = [
        scope_map.get((r["articleType"], r["masterCategory"], r["gender"]), False)
        for _, r in df.iterrows()
    ]
    df["in_scope"] = in_scope_vals

    n_in = df["in_scope"].sum()
    n_out = (~df["in_scope"]).sum()
    print(f"in_scope: True = {n_in:,} | False = {n_out:,}")
    assert n_in == 38646, f"Expected 38,646 in_scope, got {n_in}"
    assert n_out == 2245, f"Expected 2,245 out_of_scope, got {n_out}"

    # 2. Update split from dense_captions_catalog.json
    dense_path = pathlib.Path("data/checkpoints_siglip-vit/dense_captions_catalog.json")
    with open(dense_path, "r", encoding="utf-8") as f:
        dense_caps = json.load(f)
    test_ids = set(dense_caps.keys())
    print(f"Loaded {len(test_ids)} test IDs from dense_captions_catalog.json.")

    cat_ids = set(df["item_id"].astype(str))
    common_test = test_ids & cat_ids

    split_vals = []
    for _, r in df.iterrows():
        pid = str(r["item_id"])
        if pid in test_ids:
            split_vals.append("test")
        else:
            split_vals.append("unknown")

    df["split"] = split_vals
    split_counts = df["split"].value_counts().to_dict()
    print("Catalog split distribution:", split_counts)

    # In-scope split distribution
    in_scope_splits = df[df["in_scope"]]["split"].value_counts().to_dict()
    print("In-scope split distribution:", in_scope_splits)

    # Verify integrity
    assert len(df) == 40891, f"Catalog size changed! {len(df)}"
    assert not df["in_scope"].isna().any(), "Found NaN in in_scope"
    assert not df["split"].isna().any(), "Found NaN in split"

    # Save to catalog.parquet
    df.to_parquet(catalog_path, index=False)
    print(f"Saved updated catalog to {catalog_path}.")

if __name__ == "__main__":
    main()
