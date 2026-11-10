"""
CIR Modifier Normalizer & Intent Classifier (src/cir_normalizer.py)
Provides:
1. General Vietnamese typo cleanup & fuzzy color matching (using lexicon_vi.json).
2. Diacritics-aware context disambiguation (avoiding 'đậm'->Dresses, 'có mũ'->Caps, etc.).
3. Two-state CIR intent classification:
   - State A (Attribute Shift): Keep reference item's articleType.
   - State B (Category Mutation): Modifier requests a new articleType.
4. Optimal P2 English prompt translation ("{color} {type}") for CIR Combiner.
"""
import re
import json
import unicodedata
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

# Load Vietnamese Lexicon for fuzzy color typo recovery & disambiguation
LEXICON_VI_PATH = ROOT_DIR / "src" / "lexicon_vi.json"
if LEXICON_VI_PATH.exists():
    with open(LEXICON_VI_PATH, "r", encoding="utf-8") as f:
        LEXICON_VI = json.load(f)
else:
    LEXICON_VI = {"colours": {}, "article_types": {}, "genders": {}}

# Common filler & emotional adjective words in Vietnamese CIR queries
FILLER_WORDS = {
    "mau", "màu", "kieu", "kiểu", "cho", "dang", "dáng", "chiec", "chiếc",
    "cai", "cái", "bo", "bộ", "doi", "đổi", "sang", "thanh", "thành",
    "chuyen", "chuyển", "tim", "tìm", "muon", "muốn", "nhu", "như",
    "nay", "này", "do", "đó", "co", "có", "khong", "không", "giup", "giúp",
    "toi", "tôi", "em", "ban", "bạn", "hang", "hãng", "hieu", "hiệu",
    "loai", "loại", "tong", "tông", "sac", "sắc", "gam", "phoi", "phối",
    "tuyet", "tuyền", "dep", "đẹp", "re", "rẻ", "nhat", "nhất",
    # Descriptive / intensity / conversational fillers
    "dam", "đậm", "nhat", "nhạt", "tuoi", "tươi", "tram", "trầm",
    "ruc", "rực", "ro", "rỡ", "nhe", "nhẹ", "nhang", "nhàng",
    "thanh", "lich", "lịch", "lam", "lãm", "dien", "điển", "ban", "bản",
    "tinh", "khoi", "khôi", "huyen", "huyền", "bi", "bí", "noi", "nổi", "bat", "bật",
    "tre", "trẻ", "trung", "su", "sự", "trang", "nha", "nhã", "sang", "trong", "trọng",
    "am", "ấm", "ap", "áp", "manh", "mạnh", "me", "mẽ", "ca", "cá", "tinh", "tính",
    "diu", "dịu", "dang", "dàng", "nang", "năng", "dong", "động", "toi", "tối", "gian", "giản",
    "mau_sac", "phom", "form", "nhung", "nhưng", "ma", "mà", "thi", "thì", "tot", "tốt",
    "gium", "giùm", "voi", "với", "cua", "của", "day", "đây", "ay", "ấy",
    "xem", "hay", "hãy", "vua", "vừa", "tuong", "tương", "tu", "tự", "giong", "giống", "vay", "vậy",
    "cung", "cùng", "nu", "nữ", "tinh", "basic", "pastel", "doi_mau"
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
    "tan", "cream", "olive", "peach", "lavender", "turquoise", "burgundy", "khaki",
    "shirt", "shirts", "tshirt", "tshirts", "t-shirt", "shoe", "shoes", "dress", "dresses",
    "skirt", "skirts", "top", "tops", "jacket", "jackets", "sweater", "sweaters",
    "trouser", "trousers", "pant", "pants", "short", "shorts", "bag", "bags",
    "handbag", "handbags", "backpack", "backpacks", "watch", "watches", "wallet", "wallets",
    "men", "women", "kids", "boys", "girls", "unisex", "casual", "formal", "sports",
    "striped", "checked", "printed", "floral", "slim", "fit", "cotton", "denim", "leather"
}

_VN_ACCENT_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE
)

# Color-context trigger words in Vietnamese (used safely for typo recovery & 'đậm' disambiguation)
_COLOUR_ANCHOR_WORDS = {"mau", "màu", "tong", "tông", "sac", "sắc", "gam"}
_COLOR_MOD_PREFIXES = {
    "xanh", "navy", "do", "đỏ", "den", "đen", "trang", "trắng", "vang", "vàng",
    "hong", "hồng", "nau", "nâu", "xam", "xám", "tim", "tím", "lam", "luc", "lục",
    "be", "cam", "ghi"
}


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    nfd = unicodedata.normalize("NFD", text)
    no_mn = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return no_mn.replace("đ", "d").replace("Đ", "D").lower()


def _damerau_levenshtein_le1(a: str, b: str) -> bool:
    """Fast check if Damerau-Levenshtein edit distance between a and b is <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return (j == i + 1) and (a[i] == b[j]) and (a[j] == b[i])
        return False
    # Insertion or deletion
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


def _build_color_lexicon_index() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Builds:
    1) raw entries sorted by phrase word count desc.
    2) specific multi-word shade entries (>= 2 non-anchor color tokens, e.g. 'xanh than', 'vang dong', 'do ruou')
       to rescue shades shadowed by generic 'mau <base>' in QueryParser.
    """
    raw_entries = []
    specific_entries = []
    seen_spec = set()
    for canon, phrases in LEXICON_VI.get("colours", {}).items():
        for p in phrases:
            clean_p = re.sub(r"\s+", " ", _strip_accents(p)).strip()
            if not clean_p:
                continue
            raw_entries.append((clean_p, canon))
            core_words = [w for w in clean_p.split() if w not in _COLOUR_ANCHOR_WORDS]
            if len(core_words) >= 2:
                spec_phrase = " ".join(core_words)
                if (spec_phrase, canon) not in seen_spec:
                    seen_spec.add((spec_phrase, canon))
                    specific_entries.append((spec_phrase, canon))

    raw_entries.sort(key=lambda x: (len(x[0].split()), len(x[0])), reverse=True)
    specific_entries.sort(key=lambda x: (len(x[0].split()), len(x[0])), reverse=True)
    return raw_entries, specific_entries


_COLOR_LEXICON_ENTRIES, _SPECIFIC_COLOR_ENTRIES = _build_color_lexicon_index()


def _dedup_repeated_chars(token: str) -> str:
    """Collapses unnatural repeated characters in Vietnamese typos (e.g. đenn->đen, navvy->navy)."""
    if token.lower() in ENGLISH_FASHION_WORDS:
        return token
    return re.sub(r"(.)\1+", r"\1", token)


def recover_color_from_typos(text: str) -> Optional[str]:
    """
    General Vietnamese typo recovery for colour phrases without hardcoding specific heldout strings.
    Handles:
    1) Repeated-keystroke typos ('đenn'->'đen', 'trắnng'->'trắng', 'navvy'->'navy').
    2) Single-character typos ('xang navy', 'màu trắg', 'màu đn', 'màu xah dương') via edit distance <= 1.
    """
    if not text or not text.strip():
        return None

    clean_no_acc = re.sub(r"[^\w\s]", " ", _strip_accents(text)).strip()
    raw_tokens = clean_no_acc.split()
    if not raw_tokens:
        return None

    # Step 1: Collapse repeated letters per token
    dedup_tokens = [_dedup_repeated_chars(t) for t in raw_tokens]
    dedup_joined = " " + " ".join(dedup_tokens) + " "

    # Exact match on deduped text against color lexicon (longest phrase first)
    for phrase, canon in _COLOR_LEXICON_ENTRIES:
        if re.search(rf"\b{re.escape(phrase)}\b", dedup_joined):
            return canon

    # Step 2: Fuzzy edit-distance <= 1 match on color phrases
    # Look at bigrams & trigrams first, then unigrams only after color anchor words ('mau', 'tong', 'sac', 'gam')
    n = len(dedup_tokens)
    for ngram_len in (3, 2):
        for i in range(n - ngram_len + 1):
            cand = " ".join(dedup_tokens[i : i + ngram_len])
            if len(cand) < 5:
                continue
            for phrase, canon in _COLOR_LEXICON_ENTRIES:
                if len(phrase.split()) == ngram_len and len(phrase) >= 5:
                    # Prefer candidates sharing first char to avoid false hits
                    if cand[0] == phrase[0] and _damerau_levenshtein_le1(cand, phrase):
                        return canon

    # Unigrams: check after color anchor or long tokens (>= 4 chars) or length >= 2 right after 'mau'/'tong'/'sac'/'gam'
    for i, tok in enumerate(dedup_tokens):
        is_after_anchor = (i > 0 and dedup_tokens[i - 1] in _COLOUR_ANCHOR_WORDS)
        min_len = 2 if is_after_anchor else 4
        if len(tok) < min_len:
            continue
        for phrase, canon in _COLOR_LEXICON_ENTRIES:
            if " " in phrase:
                continue
            if len(phrase) < min_len:
                continue
            if (tok[0] == phrase[0] or is_after_anchor) and _damerau_levenshtein_le1(tok, phrase):
                return canon

    return None


def is_already_english(text: str) -> bool:
    """Checks if query is already primarily English fashion text."""
    clean = re.sub(r"[^\w\s\-]", " ", text.lower()).strip()
    tokens = clean.split()
    if not tokens:
        return False
    if _VN_ACCENT_RE.search(clean):
        return False
    en_matches = sum(1 for t in tokens if t in ENGLISH_FASHION_WORDS)
    return (en_matches / len(tokens)) >= 0.5


def sanitize_parsed_modifier(
    modifier_text: str,
    parsed_spec: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Applies general Vietnamese diacritics-aware disambiguation and typo recovery on top of
    QueryParserV2 output without modifying QueryParserV2 itself (preserving Tab 1 zero-regression).

    Returns:
        (clean_spec, extra_style_terms, diag_info)
    """
    clean_spec = dict(parsed_spec or {})
    extra_style_terms: List[str] = []
    diag_info: Dict[str, Any] = {
        "disambiguated_type_dropped": None,
        "typo_color_recovered": None
    }

    raw_text = (modifier_text or "").strip()
    if not raw_text:
        return clean_spec, extra_style_terms, diag_info

    raw_lower = raw_text.lower()
    has_accents = bool(_VN_ACCENT_RE.search(raw_lower))
    no_acc = _strip_accents(raw_lower)

    parsed_art = clean_spec.get("articleType")

    # 1. General Diacritics-aware ArticleType Disambiguation
    if parsed_art == "Dresses":
        if has_accents:
            if not re.search(r"\b(đầm|váy|dresses?|váy đầm)\b", raw_lower):
                clean_spec["articleType"] = None
                diag_info["disambiguated_type_dropped"] = "Dresses (no accented dress/skirt noun; e.g. 'đậm')"
        else:
            # Unaccented: drop 'dam' if preceded by color/shade words and not 'vay dam' / 'sang dam' / 'thanh dam'
            if re.search(r"\b(" + "|".join(_COLOR_MOD_PREFIXES) + r")\s+dam\b", no_acc) and not re.search(
                r"\b(vay\s+dam|sang\s+dam|thanh\s+dam|mac\s+dam)\b", no_acc
            ):
                clean_spec["articleType"] = None
                diag_info["disambiguated_type_dropped"] = "Dresses (unaccented 'dam' modifying color)"

    elif parsed_art == "Caps":
        is_hood_or_lime = bool(
            re.search(r"\b(có\s+mũ|co\s+mu|kèm\s+mũ|kem\s+mu|mũ\s+len|mu\s+len|có\s+nón|co\s+non|xanh\s+non|nõn)\b", raw_lower)
        )
        explicit_cap_switch = bool(
            re.search(r"\b(sang\s+mũ|thành\s+mũ|mũ\s+lưỡi\s+trai|nón\s+kết|caps?)\b", raw_lower)
        )
        if is_hood_or_lime and not explicit_cap_switch:
            clean_spec["articleType"] = None
            diag_info["disambiguated_type_dropped"] = "Caps (hooded detail / lime modifier)"
            if re.search(r"\b(có\s+mũ|co\s+mu|mũ\s+len|mu\s+len)\b", raw_lower):
                extra_style_terms.append("hooded")

    elif parsed_art == "Wallets" and has_accents:
        if not re.search(r"\b(ví|bóp|wallets?)\b", raw_lower):
            clean_spec["articleType"] = None
            diag_info["disambiguated_type_dropped"] = "Wallets (homonym e.g. 'vì')"

    elif parsed_art == "Ring" and has_accents:
        if not re.search(r"\b(nhẫn|rings?)\b", raw_lower):
            clean_spec["articleType"] = None
            diag_info["disambiguated_type_dropped"] = "Ring (homonym e.g. 'nhấn'/'nhắn')"

    elif parsed_art == "Socks" and has_accents:
        if not re.search(r"\b(tất|vớ|socks?)\b", raw_lower) or re.search(r"\btất\s+cả\b", raw_lower):
            if not re.search(r"\b(vớ|tất\s+chân|đôi\s+tất|socks?)\b", raw_lower):
                clean_spec["articleType"] = None
                diag_info["disambiguated_type_dropped"] = "Socks (non-sock context e.g. 'tất cả')"

    # 2. Refine / Recover colour:
    # (a) Generic 'mau <base>' (2 words) in QueryParser can shadow specific 2-word shades like 'xanh than', 'vang dong'.
    cur_col = clean_spec.get("baseColour")
    if not is_already_english(raw_text):
        tokens_no_acc = re.sub(r"[^\w\s]", " ", no_acc).split()
        no_anchor_tokens = [t for t in tokens_no_acc if t not in _COLOUR_ANCHOR_WORDS]
        no_anchor_joined = " " + " ".join(no_anchor_tokens) + " "
        best_specific_col = None
        for phrase, canon in _SPECIFIC_COLOR_ENTRIES:
            if re.search(rf"\b{re.escape(phrase)}\b", no_anchor_joined):
                best_specific_col = canon
                break

        if best_specific_col and best_specific_col != cur_col:
            # Upgrade when a generic color or shadowed shade was missed (avoid overriding leather 'da' as 'da bo' unless 'da bo' itself)
            if cur_col is None or cur_col in ("Blue", "Yellow", "Red", "Pink", "Brown", "Green", "Grey", "Purple", "Black", "White", "Orange"):
                clean_spec["baseColour"] = best_specific_col
                diag_info["typo_color_recovered"] = f"{cur_col} -> {best_specific_col} (specific shade)"
                cur_col = best_specific_col

        # (b) Typo recovery if still None or 'Blue' when 'navvy' is present
        needs_typo_check = (cur_col is None) or (
            cur_col == "Blue" and re.search(r"\bnav+y\b", no_acc) is not None
        )
        if needs_typo_check:
            recovered = recover_color_from_typos(raw_text)
            if recovered:
                clean_spec["baseColour"] = recovered
                diag_info["typo_color_recovered"] = f"{cur_col} -> {recovered}"

    return clean_spec, extra_style_terms, diag_info


def classify_cir_intent(
    modifier_text: str,
    clean_spec: Dict[str, Any],
    ref_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classifies user modifier intent into two clear CIR states:
    - State A ("A_attribute_shift"): Modifier adjusts attributes (colour/gender/brand/style) or
      repeats the original ref_type -> Target category stays locked to `ref_type`.
    - State B ("B_category_mutation"): Modifier explicitly introduces a DIFFERENT valid articleType
      (e.g. "chuyển thành chân váy", "đổi sang áo khoác") -> Target category switches to `mod_type`.
    """
    mod_type = clean_spec.get("articleType")
    if mod_type and (not ref_type or mod_type != ref_type):
        return {
            "intent_state": "B_category_mutation",
            "target_type": mod_type,
            "detected_mod_type": mod_type,
            "ref_type": ref_type,
            "reason": f"Modifier yêu cầu đổi sang loại mới: {mod_type}"
        }
    return {
        "intent_state": "A_attribute_shift",
        "target_type": ref_type,
        "detected_mod_type": mod_type,
        "ref_type": ref_type,
        "reason": f"Giữ nguyên loại gốc: {ref_type}" if ref_type else "Chưa xác định loại gốc"
    }


def normalize_cir_modifier(
    modifier_text: str,
    parsed_spec: Dict[str, Any],
    ref_type: Optional[str] = None,
    ref_gender: Optional[str] = None,
    extra_style_terms: Optional[List[str]] = None
) -> Tuple[str, bool, List[str]]:
    """
    Translates/normalizes Vietnamese CIR modifier text into optimal P2 English prompt for CIR Combiner:
    Format: "[brand] [color] [style] [gender] [type]"

    Returns:
        (english_prompt, is_fallback, untranslated_tokens)
    """
    raw_text = (modifier_text or "").strip()
    if not raw_text:
        return ("", False, [])

    # If text is already in English, preserve directly
    if is_already_english(raw_text):
        return (raw_text, False, [])

    # Extract parsed attributes
    col_en = parsed_spec.get("baseColour")
    mod_art_en = parsed_spec.get("articleType")
    art_en = mod_art_en or ref_type
    gen_en = parsed_spec.get("gender")
    brand_en = parsed_spec.get("brand")

    # Normalize tokens for term matching
    tokens = re.findall(r"[\w\-]+", raw_text.lower())
    clean_tokens = [t for t in tokens if t not in FILLER_WORDS]

    # Style / detail terms from vi_to_en_terms
    style_terms = list(extra_style_terms or [])
    text_joined = " " + " ".join(clean_tokens) + " "
    for vi_k, en_v in sorted(VI_TO_EN_TERMS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(vi_k) + r"\b"
        if re.search(pattern, text_joined):
            if en_v not in style_terms:
                style_terms.append(en_v)
            text_joined = re.sub(pattern, " ", text_joined)

    # Also match accentless keys of vi_to_en_terms cleanly if accented didn't hit
    text_joined_no_acc = " " + _strip_accents(text_joined) + " "
    for vi_k, en_v in sorted(VI_TO_EN_TERMS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(vi_k) + r"\b"
        if re.search(pattern, text_joined_no_acc):
            if en_v not in style_terms:
                style_terms.append(en_v)
            text_joined_no_acc = re.sub(pattern, " ", text_joined_no_acc)

    # Detect remaining untranslated tokens (after removing fillers)
    rem_tokens = text_joined.split()
    untranslated = []
    for t in rem_tokens:
        t_no_acc = _strip_accents(t)
        if t not in FILLER_WORDS and t_no_acc not in FILLER_WORDS and len(t) > 1:
            untranslated.append(t)

    # Check if ANY meaningful modification attribute was recognized from the modifier text
    has_recognized_mod = bool(col_en or mod_art_en or gen_en or brand_en or style_terms)

    if not has_recognized_mod:
        # Modifier contains no understandable fashion modification (e.g. "màu tàng hình", "xyzabc123")
        return ("", True, untranslated)

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
        t_str = str(art_en).lower()
        if t_str.endswith("s") and t_str not in ("dress", "sunglasses", "trousers", "shorts", "jeans"):
            t_str = t_str[:-1]
        prompt_parts.append(t_str)

    if not prompt_parts:
        return ("", True, untranslated)

    en_prompt = " ".join(prompt_parts).strip()
    return (en_prompt, False, untranslated)
