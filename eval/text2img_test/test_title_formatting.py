import sys
import json
import re
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Load vi_map.json
with open("eval/text2img_test/vi_map.json", "r", encoding="utf-8") as f:
    VI_MAP = json.load(f)

# Load known brand names from src/brands_v2.json if available
BRANDS_PATH = Path("src/brands_v2.json")
KNOWN_BRANDS = set()
if BRANDS_PATH.exists():
    with open(BRANDS_PATH, "r", encoding="utf-8") as f:
        KNOWN_BRANDS = set(json.load(f))

def extract_brand_and_model(raw_name: str, article_type: str, gender: str, base_colour: str, explicit_brand: str = None) -> tuple:
    """
    Extracts brand and model/edition string from catalog name.
    """
    name_str = (raw_name or "").strip()
    if not name_str:
        return "", ""

    tokens = name_str.split()
    if not tokens:
        return "", ""

    # 1. Determine Brand
    brand_candidate = ""
    if explicit_brand and str(explicit_brand).lower() not in ("none", "nan", "unknown", ""):
        brand_candidate = str(explicit_brand).strip()
    else:
        # Check first 2 tokens for multi-word brand (e.g. United Colors of Benetton, Tommy Hilfiger)
        first_4 = " ".join(tokens[:4]).lower()
        first_3 = " ".join(tokens[:3]).lower()
        first_2 = " ".join(tokens[:2]).lower()
        first_1 = tokens[0].lower()

        if "united colors of benetton" in first_4:
            brand_candidate = "United Colors of Benetton"
        elif first_2 in [b.lower() for b in KNOWN_BRANDS if len(b.split()) >= 2]:
            brand_candidate = " ".join(tokens[:2])
        elif first_1 in [b.lower() for b in KNOWN_BRANDS]:
            brand_candidate = tokens[0]
        else:
            # Fallback: First token is brand
            brand_candidate = tokens[0]

    # Normalize Brand Casing
    brand_lower = brand_candidate.lower()
    normalized_brand = VI_MAP["brand"].get(brand_lower, brand_candidate)
    if normalized_brand == brand_candidate:
        if brand_candidate.isupper() and len(brand_candidate) > 3:
            normalized_brand = brand_candidate.title()
        elif brand_candidate.islower():
            normalized_brand = brand_candidate.title()

    # 2. Extract Model / Special edition
    # Identify words to remove: brand, gender, color words, articleType words
    stop_words = set()
    for w in brand_candidate.lower().split():
        stop_words.add(w)
    if gender:
        stop_words.add(gender.lower())
        if gender.lower() == "women":
            stop_words.add("woman")
        elif gender.lower() == "men":
            stop_words.add("man")
    if base_colour:
        for w in base_colour.lower().split():
            stop_words.add(w)
    if article_type:
        for w in article_type.lower().split():
            stop_words.add(w)
            if w.endswith("s"):
                stop_words.add(w[:-1]) # e.g. Tshirts -> tshirt
            stop_words.add(w + "s")

    # Common repetition words
    stop_words.update(["tee", "t-shirt", "tshirt", "shirt", "shoe", "shoes", "bag", "handbag", "watch", "watches", "colour", "color", "pack", "set", "of"])

    # Collect remaining tokens in order
    model_tokens = []
    for tok in tokens:
        cleaned_tok = re.sub(r"[^\w\-\.]", "", tok).strip()
        if not cleaned_tok:
            continue
        if cleaned_tok.lower() not in stop_words and len(cleaned_tok) > 1:
            model_tokens.append(cleaned_tok)

    # If the remaining model tokens just repeat brand or colors, drop them
    model_str = " ".join(model_tokens).strip()
    if model_str.lower() in [brand_candidate.lower(), base_colour.lower() if base_colour else "", article_type.lower() if article_type else ""]:
        model_str = ""

    return normalized_brand, model_str

def format_vietnamese_product_title(prod: dict) -> str:
    """
    Builds the official Vietnamese eCommerce product title from catalog attributes.
    Formula: {Loại} {giới tính} {Thương hiệu} màu {màu} [– {dòng/mẫu}]
    """
    raw_name = str(prod.get("name") or "").strip()
    art_raw = str(prod.get("articleType") or "").strip()
    gen_raw = str(prod.get("gender") or "").strip()
    col_raw = str(prod.get("baseColour") or "").strip()
    explicit_brand = prod.get("brand")

    # 1. Component: Loại (ArticleType)
    vi_type = VI_MAP["articleType"].get(art_raw, art_raw)
    if not vi_type or vi_type.lower() in ("none", "nan", "unknown"):
        vi_type = "Sản phẩm thời trang"

    # 2. Component: Giới tính (Gender)
    vi_gen = VI_MAP["gender"].get(gen_raw, "")
    if vi_gen.lower() in ("none", "nan", "unknown"):
        vi_gen = ""

    # 3. Component: Thương hiệu & Dòng mẫu
    brand, model = extract_brand_and_model(raw_name, art_raw, gen_raw, col_raw, explicit_brand)

    # 4. Component: Màu sắc (BaseColour)
    vi_col = VI_MAP["baseColour"].get(col_raw, "")
    if vi_col.lower() in ("none", "nan", "unknown", ""):
        col_str = ""
    else:
        col_str = f"màu {vi_col}"

    # Assemble base formula: {Loại} {giới tính} {Thương hiệu} {màu}
    base_parts = [vi_type]
    if vi_gen:
        base_parts.append(vi_gen)
    if brand:
        base_parts.append(brand)
    if col_str:
        base_parts.append(col_str)

    main_title = " ".join(base_parts).strip()

    # Append model if present and meaningful
    if model and len(model) >= 2:
        # Don't append if model is already contained in main_title
        if model.lower() not in main_title.lower():
            main_title = f"{main_title} – {model}"

    return main_title

# Test samples from user prompt
samples = [
    {"name": "Murcia Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "Murcia"},
    {"name": "ADIDAS Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "ADIDAS"},
    {"name": "Puma Men puma shield tee Yellow Tshirts", "articleType": "Tshirts", "gender": "Men", "baseColour": "Yellow", "brand": "Puma"},
    {"name": "Turtle Check Men Navy Blue Shirt", "articleType": "Shirts", "gender": "Men", "baseColour": "Navy Blue", "brand": "Turtle"},
    {"name": "Nike Men Black Air Zoom Running Shoes", "articleType": "Sports Shoes", "gender": "Men", "baseColour": "Black", "brand": "Nike"},
    {"name": "Lavie Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "Lavie"},
    {"name": "Fastrack Men Black Watch", "articleType": "Watches", "gender": "Men", "baseColour": "Black", "brand": "Fastrack"},
    {"name": "Generic Product Without Color", "articleType": "Backpacks", "gender": "Unisex", "baseColour": "Unknown", "brand": ""},
]

print("\n--- TEST KẾT QUẢ ĐỊNH DẠNG TÊN TIẾNG VIỆT ---")
for s in samples:
    vi_name = format_vietnamese_product_title(s)
    print(f"Gốc:  \"{s['name']}\"")
    print(f"-> VI: \"{vi_name}\"\n")
