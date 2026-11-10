"""
CIR Modifier Normalizer: Vietnamese -> English Structured Query Translation for CIR Combiner.
Builds on QueryParserV2 attributes and vi_to_en_terms.json dictionary.
"""
import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

ROOT_DIR = Path(__file__).resolve().parent.parent

# Load vi_to_en_terms
VI_TO_EN_TERMS_PATH = ROOT_DIR / "src" / "vi_to_en_terms.json"
if VI_TO_EN_TERMS_PATH.exists():
    with open(VI_TO_EN_TERMS_PATH, "r", encoding="utf-8") as f:
        VI_TO_EN_TERMS = json.load(f)
else:
    VI_TO_EN_TERMS = {}

# Common filler words in Vietnamese CIR queries
FILLER_WORDS = {
    "mau", "màu", "kieu", "kiểu", "cho", "dang", "dáng", "chiec", "chiếc",
    "cai", "cái", "bo", "bộ", "doi", "đổi", "sang", "thanh", "thành",
    "chuyen", "chuyển", "tim", "tìm", "muon", "muốn", "nhu", "như",
    "nay", "này", "do", "đó", "co", "có", "khong", "không", "giup", "giúp",
    "toi", "tôi", "em", "ban", "bạn", "hang", "hãng", "hieu", "hiệu",
    "loai", "loại", "tong", "tông", "sac", "sắc", "gam", "phoi", "phối",
    "tuyet", "tuyền", "dep", "đẹp", "re", "rẻ", "nhat", "nhất"
}

# Gender map to English
GENDER_TO_EN = {
    "Men": "men",
    "Women": "women",
    "Boys": "boys",
    "Girls": "girls",
    "Unisex": "unisex"
}

# English common fashion words detection
ENGLISH_FASHION_WORDS = {
    "black", "white", "blue", "navy", "red", "green", "yellow", "pink", "purple",
    "grey", "gray", "brown", "orange", "beige", "silver", "gold", "teal", "maroon",
    "shirt", "shirts", "tshirt", "tshirts", "t-shirt", "shoe", "shoes", "dress", "dresses",
    "skirt", "skirts", "top", "tops", "jacket", "jackets", "sweater", "sweaters",
    "trouser", "trousers", "pant", "pants", "short", "shorts", "bag", "bags",
    "handbag", "handbags", "backpack", "backpacks", "watch", "watches", "wallet", "wallets",
    "men", "women", "kids", "boys", "girls", "unisex", "casual", "formal", "sports",
    "striped", "checked", "printed", "floral", "slim", "fit", "cotton", "denim", "leather"
}

def is_already_english(text: str) -> bool:
    """Checks if query is already primarily English fashion text."""
    clean = re.sub(r"[^\w\s\-]", " ", text.lower()).strip()
    tokens = clean.split()
    if not tokens:
        return False
    # If contains accented Vietnamese characters, definitely not English
    if re.search(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", clean):
        return False
    en_matches = sum(1 for t in tokens if t in ENGLISH_FASHION_WORDS)
    return (en_matches / len(tokens)) >= 0.5

def normalize_cir_modifier(
    modifier_text: str,
    parsed_spec: Dict[str, Any],
    ref_type: Optional[str] = None,
    ref_gender: Optional[str] = None
) -> Tuple[str, bool, List[str]]:
    """
    Translates/normalizes Vietnamese CIR modifier text into optimal English prompt for CIR Combiner.
    
    Returns:
        (english_prompt, is_fallback, untranslated_tokens)
    """
    raw_text = (modifier_text or "").strip()
    if not raw_text:
        return ("", False, [])

    # If text is already in English, preserve directly
    if is_already_english(raw_text):
        return (raw_text, False, [])

    # Extract parsed attributes from QueryParserV2
    col_en = parsed_spec.get("baseColour") # e.g. "Navy Blue"
    art_en = parsed_spec.get("articleType") or ref_type # e.g. "Shirts"
    gen_en = parsed_spec.get("gender") # e.g. "Women"
    brand_en = parsed_spec.get("brand") # e.g. "Nike"

    # Normalize tokens for term matching
    tokens = re.findall(r"[\w\-]+", raw_text.lower())
    clean_tokens = [t for t in tokens if t not in FILLER_WORDS]

    # Style / detail terms from vi_to_en_terms
    style_terms = []
    text_joined = " " + " ".join(clean_tokens) + " "
    for vi_k, en_v in sorted(VI_TO_EN_TERMS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(vi_k) + r"\b"
        if re.search(pattern, text_joined):
            style_terms.append(en_v)
            text_joined = re.sub(pattern, " ", text_joined)

    # Detect remaining untranslated tokens
    rem_tokens = text_joined.split()
    untranslated = []
    for t in rem_tokens:
        if t not in FILLER_WORDS and len(t) > 1:
            untranslated.append(t)

    # Build P2 optimal English prompt: [brand] [color] [style] [gender] [type]
    prompt_parts = []
    if brand_en:
        prompt_parts.append(str(brand_en).title())
    if col_en:
        prompt_parts.append(str(col_en).lower())
    if style_terms:
        prompt_parts.extend(style_terms)
    if gen_en:
        prompt_parts.append(GENDER_TO_EN.get(gen_en, str(gen_en).lower()))
    if art_en:
        # Singularize / clean common types
        t_str = str(art_en).lower()
        if t_str.endswith("s") and t_str not in ("dress", "sunglasses", "trousers", "shorts", "jeans"):
            t_str = t_str[:-1]
        prompt_parts.append(t_str)

    if not prompt_parts:
        # Nothing could be understood/translated
        return ("", True, untranslated)

    en_prompt = " ".join(prompt_parts).strip()
    return (en_prompt, False, untranslated)
