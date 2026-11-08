import json
import pandas as pd
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    json_path = pathlib.Path("data/checkpoints_siglip-vit/dense_captions_catalog.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total keys in dense_captions_catalog.json: {len(data)}")
    first_keys = list(data.keys())[:3]
    for k in first_keys:
        print(f"--- Key: {k} ---")
        print(data[k])

    # Check "Tên gốc" in captions
    has_ten_goc = sum(1 for v in data.values() if "Tên gốc" in v)
    print(f"\nCaptions containing 'Tên gốc': {has_ten_goc} / {len(data)}")

    # Check catalog
    cat = pd.read_parquet("data/catalog/catalog.parquet")
    print(f"Catalog total rows: {len(cat)}")
    test_in_scope = cat[(cat["split"] == "test") & (cat["in_scope"] == True)]
    print(f"Test in-scope count: {len(test_in_scope)}")

    test_in_scope_ids = set(test_in_scope["item_id"].astype(str))
    json_keys = set(data.keys())

    missing_captions = test_in_scope_ids - json_keys
    print(f"Test in-scope IDs missing from dense_captions_catalog.json: {len(missing_captions)}")

    # Check Myntra styles.csv
    styles = pd.read_csv("data/fashion-dataset/styles.csv", on_bad_lines="skip")
    styles_ids = set(styles["id"].astype(str))
    test_in_styles = json_keys.intersection(styles_ids)
    print(f"Test IDs in json that exist in raw styles.csv: {len(test_in_styles)}")

    cat_ids = set(cat["item_id"].astype(str))
    test_myntra_in_cat = test_in_styles.intersection(cat_ids)
    print(f"Test Myntra IDs present in catalog: {len(test_myntra_in_cat)}")
    test_myntra_removed_cleaning = test_in_styles - cat_ids
    print(f"Test Myntra IDs removed during Phase 1 cleaning: {len(test_myntra_removed_cleaning)}")
    if len(test_myntra_removed_cleaning) > 0:
        print("Sample removed Myntra IDs:", list(test_myntra_removed_cleaning)[:10])

        # Why were they removed?
        # Check quarantine_log
        quar_path = pathlib.Path("reports/quarantine_log.csv")
        if quar_path.exists():
            quar = pd.read_csv(quar_path)
            quar_ids = set(quar["item_id"].astype(str))
            matched_quar = test_myntra_removed_cleaning.intersection(quar_ids)
            print(f"Of removed test Myntra IDs, present in quarantine_log: {len(matched_quar)}")
            if len(matched_quar) > 0:
                quar_matched = quar[quar["item_id"].astype(str).isin(matched_quar)]
                print(quar_matched[["reason", "status"]].value_counts())

        # Check missing metadata in styles.csv
        removed_styles = styles[styles["id"].astype(str).isin(test_myntra_removed_cleaning)]
        print(f"Removed styles missing articleType/gender: {len(removed_styles[removed_styles['articleType'].isna() | removed_styles['gender'].isna()])}")

if __name__ == "__main__":
    main()
