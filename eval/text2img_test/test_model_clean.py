import sys
import re
import json

sys.stdout.reconfigure(encoding="utf-8")

with open("eval/text2img_test/vi_map.json", "r", encoding="utf-8") as f:
    VI_MAP = json.load(f)

def clean_extracted_model(raw_name: str, brand: str, gender: str, base_colour: str, article_type: str) -> str:
    """
    Extracts the intermediate model/edition name preserving original phrasing like 'puma shield tee'.
    """
    name = (raw_name or "").strip()
    if not name:
        return ""

    tokens = name.split()
    if not tokens:
        return ""

    # Remove initial brand token if it matches
    start_idx = 0
    brand_tokens = brand.split()
    if len(tokens) >= len(brand_tokens) and " ".join(tokens[:len(brand_tokens)]).lower() == brand.lower():
        start_idx = len(brand_tokens)
    elif len(tokens) > 0 and tokens[0].lower() == brand.lower():
        start_idx = 1

    # Remove gender if right after brand
    if start_idx < len(tokens) and gender and tokens[start_idx].lower() == gender.lower():
        start_idx += 1

    # Check end tokens for articleType and baseColour
    end_idx = len(tokens)

    # Try matching articleType at the end
    art_tokens = article_type.lower().split()
    # Also handle plurals/singulars e.g. Tshirts/Tshirt/T-shirt, Handbags/Handbag
    if end_idx > start_idx:
        last_tok = tokens[end_idx - 1].lower()
        if any(last_tok == at or last_tok == at.rstrip('s') or last_tok.rstrip('s') == at.rstrip('s') for at in art_tokens):
            end_idx -= 1

    # Try matching baseColour before articleType at the end
    if base_colour and end_idx > start_idx:
        col_tokens = base_colour.lower().split()
        num_col = len(col_tokens)
        if end_idx - num_col >= start_idx:
            sub = [t.lower() for t in tokens[end_idx - num_col:end_idx]]
            if sub == col_tokens:
                end_idx -= num_col
            elif sub[-1] == col_tokens[-1]:
                end_idx -= 1

    middle_tokens = tokens[start_idx:end_idx]
    middle_str = " ".join(middle_tokens).strip()

    # If the middle is just repeating gender or color or type, ignore it
    test_str = middle_str.lower()
    ignore_set = {
        (gender or "").lower(),
        (base_colour or "").lower(),
        (article_type or "").lower(),
        (brand or "").lower()
    }
    if test_str in ignore_set or len(test_str) < 2:
        return ""

    return middle_str

def format_title_v2(prod: dict) -> str:
    raw_name = str(prod.get("name") or "").strip()
    art_raw = str(prod.get("articleType") or "").strip()
    gen_raw = str(prod.get("gender") or "").strip()
    col_raw = str(prod.get("baseColour") or "").strip()
    explicit_brand = prod.get("brand")

    # 1. Brand
    brand_raw = explicit_brand or (raw_name.split()[0] if raw_name else "")
    brand_norm = VI_MAP["brand"].get(brand_raw.lower(), brand_raw.title() if len(brand_raw) > 3 else brand_raw)

    # 2. Type
    vi_type = VI_MAP["articleType"].get(art_raw, art_raw)
    if not vi_type or vi_type.lower() in ("none", "nan", "unknown"):
        vi_type = "Sản phẩm thời trang"

    # 3. Gender
    vi_gen = VI_MAP["gender"].get(gen_raw, "")
    if vi_gen.lower() in ("none", "nan", "unknown"):
        vi_gen = ""

    # 4. Colour
    vi_col = VI_MAP["baseColour"].get(col_raw, "")
    col_str = f"màu {vi_col}" if vi_col and vi_col.lower() not in ("none", "nan", "unknown", "") else ""

    # Model
    model = clean_extracted_model(raw_name, brand_raw, gen_raw, col_raw, art_raw)

    parts = [vi_type]
    if vi_gen:
        parts.append(vi_gen)
    if brand_norm and brand_norm.lower() not in ("none", "nan", "unknown", ""):
        parts.append(brand_norm)
    if col_str:
        parts.append(col_str)

    title = " ".join(parts).strip()
    if model and model.lower() not in title.lower():
        title = f"{title} – {model}"

    return title

# Test cases
test_samples = [
    {"name": "Murcia Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "Murcia"},
    {"name": "ADIDAS Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "ADIDAS"},
    {"name": "Puma Men puma shield tee Yellow Tshirts", "articleType": "Tshirts", "gender": "Men", "baseColour": "Yellow", "brand": "Puma"},
    {"name": "Turtle Check Men Navy Blue Shirt", "articleType": "Shirts", "gender": "Men", "baseColour": "Navy Blue", "brand": "Turtle"},
    {"name": "Nike Men Black Air Zoom Running Shoes", "articleType": "Sports Shoes", "gender": "Men", "baseColour": "Black", "brand": "Nike"},
    {"name": "Lavie Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "Lavie"},
]

for s in test_samples:
    print(f"Gốc:  \"{s['name']}\"")
    print(f"-> VI: \"{format_title_v2(s)}\"\n")
