"""
Inspect Heldout Queries Schema, Relevance Support, and Parser Coverage (src/inspect_heldout.py)
Phase 3e B0:
- Verifies unique IDs and spec validity against catalog.parquet.
- Counts strict and family relevant items in G_test (N=1,290) and G_full (N=38,646).
- Flags low_support (< 3 items in G_test).
- Reports parser coverage on heldout queries (one-time observation).
"""

import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.attr_index import AttributeIndex
from src.query_parser import QueryParser

def inspect_heldout():
    # 1. Load AttributeIndex & catalog
    attr_idx = AttributeIndex()
    df = attr_idx.df
    test_mask = (df["split"] == "test").to_numpy()

    # Load families
    with open("eval/families.json", "r", encoding="utf-8") as f:
        fam_data = json.load(f)
    gender_compat = fam_data.get("gender_compatibility", {})
    art_families = fam_data.get("article_type_families", {})
    col_families = fam_data.get("colour_families", {})

    # Map color to family
    color_to_fam = {}
    for fam_name, cols in col_families.items():
        for c in cols:
            color_to_fam[c.lower()] = fam_name

    # Load heldout
    heldout_path = Path("eval/heldout_queries.json")
    with open(heldout_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Total heldout queries: {len(queries)}")

    # 2. Check unique IDs
    ids = [q["id"] for q in queries]
    if len(ids) != len(set(ids)):
        print("ERROR: Duplicate IDs found in heldout_queries.json!")
    else:
        print("PASS: All 30 query IDs are unique.")

    # 3. Check validity and count relevance support
    support_report = []
    low_support_count = 0
    valid_spec_count = 0
    null_spec_count = 0

    catalog_art_types = set(df["articleType"].dropna().unique())
    catalog_genders = set(df["gender"].dropna().unique())
    catalog_colours = set(df["baseColour"].dropna().unique())
    catalog_brands = set(df["brand"].dropna().unique())

    # Brand alias map for heldout note
    # e.g., 'Adidas' -> 'ADIDAS', 'Benetton' -> 'United Colors of Benetton'
    brand_alias = {
        "adidas": "ADIDAS",
        "benetton": "United Colors of Benetton"
    }

    print("\n--- HELDOUT SPEC VALIDATION & SUPPORT IN G_test / G_full ---")
    for q in queries:
        qid = q["id"]
        group = q.get("group", "")
        q_text = q["query"]
        spec = q.get("spec")

        if spec is None:
            null_spec_count += 1
            support_report.append({
                "id": qid,
                "group": group,
                "query": q_text,
                "spec": None,
                "strict_full": 0,
                "strict_test": 0,
                "family_full": 0,
                "family_test": 0,
                "low_support": True,
                "note": "spec_is_null"
            })
            print(f"[{qid}] '{q_text}' (group={group}): spec is NULL (Manual rating only)")
            continue

        valid_spec_count += 1
        spec_arts = spec.get("articleType", [])
        if isinstance(spec_arts, str):
            spec_arts = [spec_arts]
        spec_gen = spec.get("gender")
        spec_cols = spec.get("colour", [])
        if isinstance(spec_cols, str):
            spec_cols = [spec_cols]
        spec_brand = spec.get("brand")

        # Validate against catalog values
        invalid_warnings = []
        for a in spec_arts:
            if a not in catalog_art_types:
                invalid_warnings.append(f"articleType '{a}' not in catalog")
        if spec_gen and spec_gen not in catalog_genders:
            invalid_warnings.append(f"gender '{spec_gen}' not in catalog")
        for c in spec_cols:
            if c not in catalog_colours:
                invalid_warnings.append(f"colour '{c}' not in catalog")

        canonical_b = None
        if spec_brand:
            canonical_b = brand_alias.get(spec_brand.lower(), spec_brand)
            if canonical_b not in catalog_brands and spec_brand not in catalog_brands:
                invalid_warnings.append(f"brand '{spec_brand}' not in catalog")

        # Build strict mask
        strict_mask = np.ones(len(df), dtype=bool)
        if spec_arts:
            strict_mask &= df["articleType"].isin(spec_arts)
        if spec_gen:
            compat_gens = gender_compat.get(spec_gen, [spec_gen, "Unisex"])
            strict_mask &= df["gender"].isin(compat_gens)
        if spec_cols:
            strict_mask &= df["baseColour"].isin(spec_cols)
        if spec_brand:
            brand_options = [spec_brand]
            if canonical_b:
                brand_options.append(canonical_b)
            strict_mask &= df["brand"].isin(brand_options)

        strict_full = int(np.count_nonzero(strict_mask))
        strict_test = int(np.count_nonzero(strict_mask & test_mask))

        # Build family mask
        fam_arts = set(spec_arts)
        for a in spec_arts:
            fam_arts.update(art_families.get(a, [a]))

        fam_cols = set(spec_cols)
        for c in spec_cols:
            fam_name = color_to_fam.get(c.lower())
            if fam_name and fam_name in col_families:
                fam_cols.update(col_families[fam_name])

        family_mask = np.ones(len(df), dtype=bool)
        if fam_arts:
            family_mask &= df["articleType"].isin(fam_arts)
        if spec_gen:
            compat_gens = gender_compat.get(spec_gen, [spec_gen, "Unisex"])
            family_mask &= df["gender"].isin(compat_gens)
        if fam_cols:
            family_mask &= df["baseColour"].isin(fam_cols)
        if spec_brand:
            brand_options = [spec_brand]
            if canonical_b:
                brand_options.append(canonical_b)
            family_mask &= df["brand"].isin(brand_options)

        family_full = int(np.count_nonzero(family_mask))
        family_test = int(np.count_nonzero(family_mask & test_mask))

        is_low_support = (strict_test < 3)
        if is_low_support:
            low_support_count += 1

        warn_str = f" | Warnings: {invalid_warnings}" if invalid_warnings else ""
        flag_str = " [LOW SUPPORT in G_test (<3)]" if is_low_support else ""
        print(f"[{qid}] '{q_text}' (group={group}): Strict(test={strict_test}, full={strict_full}) | Family(test={family_test}, full={family_full}){flag_str}{warn_str}")

        support_report.append({
            "id": qid,
            "group": group,
            "query": q_text,
            "spec": spec,
            "strict_full": strict_full,
            "strict_test": strict_test,
            "family_full": family_full,
            "family_test": family_test,
            "low_support": is_low_support,
            "warnings": invalid_warnings
        })

    print(f"\nSummary:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Valid specs: {valid_spec_count}")
    print(f"  Null specs (Manual ratings): {null_spec_count}")
    print(f"  Queries with low_support in G_test (< 3 items): {low_support_count} ({low_support_count/len(queries)*100:.1f}%)")

    # 4. Parser Coverage on Heldout (One-time informational reporting)
    parser = QueryParser()
    correct_type = 0
    correct_gender = 0
    correct_colour = 0
    correct_brand = 0
    evaluated_count = 0

    print("\n--- PARSER COVERAGE ON HELDOUT (OBSERVATIONAL ONLY) ---")
    for q in queries:
        spec = q.get("spec")
        if not spec:
            continue
        evaluated_count += 1
        parsed = parser.parse(q["query"])

        # Type
        spec_arts = spec.get("articleType", [])
        if isinstance(spec_arts, str):
            spec_arts = [spec_arts]
        if parsed["articleType"] in spec_arts:
            correct_type += 1

        # Gender
        spec_gen = spec.get("gender")
        if parsed["gender"] == spec_gen or (not spec_gen and parsed["gender"] is None):
            correct_gender += 1

        # Colour
        spec_cols = spec.get("colour", [])
        if isinstance(spec_cols, str):
            spec_cols = [spec_cols]
        if parsed["baseColour"] in spec_cols or (not spec_cols and parsed["baseColour"] is None):
            correct_colour += 1

        # Brand
        spec_b = spec.get("brand")
        if spec_b:
            canon_b = brand_alias.get(spec_b.lower(), spec_b)
            if parsed["brand"] in (spec_b, canon_b):
                correct_brand += 1
        elif parsed["brand"] is None:
            correct_brand += 1

    print(f"Coverage on 26 heldout queries with spec:")
    print(f"  ArticleType : {correct_type}/{evaluated_count} ({correct_type/evaluated_count*100:.1f}%)")
    print(f"  Gender      : {correct_gender}/{evaluated_count} ({correct_gender/evaluated_count*100:.1f}%)")
    print(f"  Colour      : {correct_colour}/{evaluated_count} ({correct_colour/evaluated_count*100:.1f}%)")
    print(f"  Brand       : {correct_brand}/{evaluated_count} ({correct_brand/evaluated_count*100:.1f}%)")

    out_json = Path("reports/heldout_inspection_b0.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(support_report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved inspection results to {out_json}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    inspect_heldout()
