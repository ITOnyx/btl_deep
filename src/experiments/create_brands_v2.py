"""
Phase 3g: Generate src/brands_v2.json after user approval.
Preserves src/brands.json untouched (Law 3).
Includes all 234 canonical brands + approved aliases from reports/brand_alias_proposal.csv.
"""

import json
import csv
from pathlib import Path

def create_brands_v2():
    # 1. Load original brands.json
    with open("src/brands.json", "r", encoding="utf-8") as f:
        brands_v1 = json.load(f)
        
    # 2. Load approved alias proposals
    with open("reports/brand_alias_proposal.csv", "r", encoding="utf-8-sig") as f:
        proposals = list(csv.DictReader(f))
        
    canonical_brands = dict(brands_v1)
    aliases = {}
    
    for row in proposals:
        alias = row["alias"].strip()
        canon = row["canonical_brand"].strip()
        p_type = row.get("type", "alias")
        
        if p_type == "brand_moi":
            # Add new independent canonical brand
            canonical_brands[canon] = int(row["canonical_catalog_products"])
        else:
            aliases[alias] = canon
            
    brands_v2_payload = {
        "metadata": {
            "version": "2.0",
            "phase": "3g",
            "description": "Brands dictionary with aliases post-heldout inspection",
            "total_canonical_brands": len(canonical_brands),
            "total_aliases": len(aliases)
        },
        "canonical_brands": canonical_brands,
        "aliases": aliases
    }
    
    v2_path = Path("src/brands_v2.json")
    with open(v2_path, "w", encoding="utf-8") as f:
        json.dump(brands_v2_payload, f, ensure_ascii=False, indent=2)
        
    print(f"Created {v2_path} with {len(canonical_brands)} canonical brands and {len(aliases)} aliases.")

if __name__ == "__main__":
    create_brands_v2()
