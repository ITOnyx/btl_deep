"""
Phase 3g: Brand Audit and Alias Proposal (A.1 & A.2)
1. Audits x017 (Lavie) and x018 (Benetton) in catalog.parquet.
   Outputs reports/brand_audit_3g.json.
2. Scans all 234 brands in src/brands.json, q5_brand.json, heldout_queries.json
   and proposes short names / aliases commonly typed by users.
   Outputs reports/brand_alias_proposal.csv.
"""

import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

import sys
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.attr_index import AttributeIndex
from src.query_parser import QueryParser

def run_brand_audit_and_proposal():
    print("=== Phase 3g: Brand Audit and Alias Proposal ===")
    
    # 1. Audit x017 (Lavie) and x018 (Benetton)
    catalog_path = Path("data/catalog/catalog.parquet")
    df = pd.read_parquet(catalog_path)
    parser = QueryParser()
    attr_idx = AttributeIndex()
    in_scope = attr_idx.df.copy()
    names = in_scope["name"].fillna("").astype(str).tolist()
    
    with open("eval/families.json", "r", encoding="utf-8") as f:
        fam_data = json.load(f)
    gender_compat = fam_data.get("gender_compatibility", {})
    
    queries_to_audit = [
        {
            "id": "x017",
            "brand_query": "Lavie",
            "query_text": "túi xách Lavie nữ",
            "articleType": "Handbags",
            "gender": "Women"
        },
        {
            "id": "x018",
            "brand_query": "Benetton",
            "query_text": "áo thun Benetton nam",
            "articleType": "Tshirts",
            "gender": "Men"
        }
    ]
    
    audit_results = []
    
    for q in queries_to_audit:
        qid = q["id"]
        brand_q = q["brand_query"]
        query_text = q["query_text"]
        art = q["articleType"]
        gen = q["gender"]
        
        # Exact word match in catalog product names (case-insensitive)
        pat_exact = re.compile(rf"\b{re.escape(brand_q)}\b", re.IGNORECASE)
        exact_matches = [n for n in names if pat_exact.search(n)]
        
        # Substring match (case-insensitive)
        substr_matches = [n for n in names if brand_q.lower() in n.lower()]
        
        # Parser output
        parsed = parser.parse(query_text)
        
        # R (articleType + gender compatibility) without brand constraint
        compat_gens = gender_compat.get(gen, [gen, "Unisex"])
        mask_art_gen = (in_scope["articleType"] == art) & (in_scope["gender"].isin(compat_gens))
        r_art_gen_full = int(mask_art_gen.sum())
        r_art_gen_test = int((mask_art_gen & (in_scope["split"] == "test")).sum())
        
        # Check canonical brand in catalog
        # For Benetton -> United Colors of Benetton
        ucb_matches = [n for n in names if re.search(r"\bUnited Colors of Benetton\b", n, re.IGNORECASE)]
        
        # R with canonical brand
        if brand_q.lower() == "benetton":
            mask_brand_canon = mask_art_gen & in_scope["brand"].isin(["United Colors of Benetton", "Benetton"])
            r_strict_full = int(mask_brand_canon.sum())
            r_strict_test = int((mask_brand_canon & (in_scope["split"] == "test")).sum())
        else:
            mask_brand_canon = mask_art_gen & in_scope["brand"].isin([brand_q])
            r_strict_full = int(mask_brand_canon.sum())
            r_strict_test = int((mask_brand_canon & (in_scope["split"] == "test")).sum())
            
        res = {
            "id": qid,
            "query": query_text,
            "brand_in_query": brand_q,
            "exact_matches_in_catalog": len(exact_matches),
            "substring_matches_in_catalog": len(substr_matches),
            "r_art_gen_full": r_art_gen_full,
            "r_art_gen_test": r_art_gen_test,
            "r_strict_full_with_brand": r_strict_full,
            "r_strict_test_with_brand": r_strict_test,
            "parser_recognized_brand": parsed.get("brand"),
            "parser_confidence": parsed.get("confidence"),
            "unparsed_tokens": parsed.get("unparsed_tokens", []),
            "note": "Brand absent from catalog" if len(substr_matches) == 0 else f"Brand alias missing (Canonical: United Colors of Benetton, {len(ucb_matches)} products)"
        }
        audit_results.append(res)
        
    audit_file = Path("reports/brand_audit_3g.json")
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, ensure_ascii=False, indent=2)
    print(f"Saved brand audit to {audit_file}")
    
    # 2. Propose brand aliases across 234 brands in brands.json + q5 + heldout
    with open("src/brands.json", "r", encoding="utf-8") as f:
        known_brands = json.load(f)
        
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        heldout_queries = json.load(f)
        
    with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
        q5_data = json.load(f)
        
    # User-facing alias proposals
    # Check known patterns:
    # 1. Benetton -> United Colors of Benetton
    # 2. UCB -> United Colors of Benetton
    # 3. Gini & Jony / Gini Jony -> Gini and Jony
    # 4. Reid Taylor / Reid and Taylor -> Reid & Taylor
    # 5. US Polo / US Polo Assn / U.S. Polo -> U.S. Polo Assn.
    # 6. Lee Cooper (check if in catalog)
    # 7. FCUK / French Connection
    # 8. Levis / Levi's
    # 9. Adidas / ADIDAS
    # 10. Marks & Spencer / M&S
    # 11. Tommy / Tommy Hilfiger
    # 12. CK / Calvin Klein
    # 13. Q&Q / Q and Q
    # 14. D&G / Dolce & Gabbana
    
    alias_proposals = []
    
    # Hand-curated / domain specific proposals with strong rationale
    standard_rules = [
        # (alias, canonical, rationale, prop_type)
        ("Benetton", "United Colors of Benetton", "Tên thông dụng phổ biến nhất của United Colors of Benetton (Heldout x018)", "alias"),
        ("UCB", "United Colors of Benetton", "Viết tắt 3 chữ cái phổ biến của United Colors of Benetton", "alias"),
        ("Adidas", "ADIDAS", "Chuẩn hóa viết hoa/thường (brands.json viết ADIDAS)", "alias"),
        ("Gini & Jony", "Gini and Jony", "Ký tự & thay cho and (dạng hiển thị nhãn hiệu chính thức)", "alias"),
        ("Gini Jony", "Gini and Jony", "Dạng gõ tắt không ký tự nối", "alias"),
        ("Reid Taylor", "Reid & Taylor", "Bỏ ký tự & khi người dùng gõ phím", "alias"),
        ("Reid and Taylor", "Reid & Taylor", "Dạng chữ 'and' thay cho '&'", "alias"),
        ("US Polo", "U.S. Polo Assn.", "Tên gọi tắt cực kỳ phổ biến của U.S. Polo Assn.", "alias"),
        ("US Polo Assn", "U.S. Polo Assn.", "Bỏ dấu chấm phân cách", "alias"),
        ("U.S. Polo", "U.S. Polo Assn.", "Bỏ chữ Assn.", "alias"),
        ("FCUK", "French Connection", "Tên thương hiệu viết tắt nổi tiếng toàn cầu của French Connection", "alias"),
        ("Levis", "Levi's", "Bỏ dấu nháy đơn khi gõ tiếng Việt / mobile", "alias"),
        ("Q and Q", "Q&Q", "Thay ký tự & bằng and", "alias"),
        ("QQ", "Q&Q", "Dạng gõ liền bỏ &", "alias"),
        ("Lee Cooper", "Lee Cooper", "Brand độc lập (133 sp trong catalog, >=20 sp), hiện thiếu trong brands.json", "brand_moi"),
        ("Wrangler Jeans", "Wrangler", "Kèm articleType vào brand khi tìm", "alias"),
        ("sOliver", "s.Oliver", "Bỏ dấu chấm viết tắt của s.Oliver", "alias"),
        ("MrMen", "Mr.Men", "Bỏ dấu chấm viết tắt của Mr.Men", "alias"),
        ("Modacc", "Mod'acc", "Bỏ dấu nháy đơn của Mod'acc", "alias"),
        ("Tantra Kids", "Tantra Kid's", "Bỏ dấu nháy đơn của Tantra Kid's", "alias")
    ]
    
    # Verify each canonical in catalog and known_brands
    for alias, canon, reason, prop_type in standard_rules:
        # Check canonical exists in known_brands or catalog
        in_catalog_count = len([n for n in names if re.search(rf"\b{re.escape(canon)}\b", n, re.IGNORECASE)])
        alias_in_catalog_count = len([n for n in names if re.search(rf"\b{re.escape(alias)}\b", n, re.IGNORECASE)])
        
        status = "ĐỀ XUẤT ÁP DỤNG"
        if prop_type == "brand_moi":
            status = "ĐỀ XUẤT THÊM BRAND ĐỘC LẬP (133 sp)"
                
        alias_proposals.append({
            "alias": alias,
            "canonical_brand": canon,
            "type": prop_type,
            "canonical_in_brands_json": canon in known_brands,
            "canonical_catalog_products": in_catalog_count,
            "alias_catalog_products": alias_in_catalog_count,
            "status": status,
            "reason": reason
        })
        
    # Also scan all 234 brands for punctuation variations
    for b_name in known_brands.keys():
        # Check '&'
        if " & " in b_name:
            alt1 = b_name.replace(" & ", " and ")
            alt2 = b_name.replace(" & ", " ")
            for alt in [alt1, alt2]:
                if alt != b_name and not any(p["alias"].lower() == alt.lower() for p in alias_proposals):
                    alias_proposals.append({
                        "alias": alt,
                        "canonical_brand": b_name,
                        "type": "alias_dau_cau",
                        "canonical_in_brands_json": True,
                        "canonical_catalog_products": known_brands[b_name],
                        "alias_catalog_products": len([n for n in names if re.search(rf"\b{re.escape(alt)}\b", n, re.IGNORECASE)]),
                        "status": "ĐỀ XUẤT ÁP DỤNG",
                        "reason": f"Biến thể & / and của {b_name}"
                    })
        # Check '.'
        if "." in b_name:
            alt = b_name.replace(".", "").strip()
            if alt != b_name and not any(p["alias"].lower() == alt.lower() for p in alias_proposals):
                alias_proposals.append({
                    "alias": alt,
                    "canonical_brand": b_name,
                    "type": "alias_dau_cau",
                    "canonical_in_brands_json": True,
                    "canonical_catalog_products": known_brands[b_name],
                    "alias_catalog_products": len([n for n in names if re.search(rf"\b{re.escape(alt)}\b", n, re.IGNORECASE)]),
                    "status": "ĐỀ XUẤT ÁP DỤNG",
                    "reason": f"Bỏ dấu chấm viết tắt của {b_name}"
                })
        # Check "'"
        if "'" in b_name:
            alt = b_name.replace("'", "").strip()
            if alt != b_name and not any(p["alias"].lower() == alt.lower() for p in alias_proposals):
                alias_proposals.append({
                    "alias": alt,
                    "canonical_brand": b_name,
                    "type": "alias_dau_cau",
                    "canonical_in_brands_json": True,
                    "canonical_catalog_products": known_brands[b_name],
                    "alias_catalog_products": len([n for n in names if re.search(rf"\b{re.escape(alt)}\b", n, re.IGNORECASE)]),
                    "status": "ĐỀ XUẤT ÁP DỤNG",
                    "reason": f"Bỏ dấu nháy đơn của {b_name}"
                })
                
    df_prop = pd.DataFrame(alias_proposals)
    prop_file = Path("reports/brand_alias_proposal.csv")
    df_prop.to_csv(prop_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_prop)} brand alias proposals to {prop_file}")

if __name__ == "__main__":
    run_brand_audit_and_proposal()
