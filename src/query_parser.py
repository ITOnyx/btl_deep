"""
Comprehensive Fashion Query Parser (src/query_parser.py)
Phase 3e: Rule-based, accent-insensitive, longest-match-first parser for Vietnamese fashion queries.

Extracts:
- articleType: canonical English articleType or None
- gender: Boys, Girls, Men, Women, Unisex or None
- baseColour: canonical English baseColour or None
- colour_family: color family from eval/families.json or None
- usage: Casual, Sports, Formal, Ethnic, Party, Smart Casual, Travel or None
- season: Summer, Winter, Spring, Fall or None
- brand: matched brand from src/brands.json or None
- unparsed_tokens: list of remaining tokens
- confidence: parsing confidence score [0.0, 1.0]
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

def remove_diacritics(text: str) -> str:
    """Removes Vietnamese tone marks and accents for search normalization."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    without_combining = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    res = without_combining.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", res).lower().strip()


class QueryParser:
    def __init__(
        self,
        lexicon_path: str = "src/lexicon_vi.json",
        brands_path: str = "src/brands.json",
        families_path: str = "eval/families.json"
    ):
        # 1. Load Lexicon
        lex_p = Path(lexicon_path)
        if lex_p.exists():
            with open(lex_p, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
        else:
            self.lexicon = {"genders": {}, "colours": {}, "article_types": {}, "usages": {}, "seasons": {}}

        # 2. Load Brands
        b_p = Path(brands_path)
        if b_p.exists():
            with open(b_p, "r", encoding="utf-8") as f:
                self.brands_dict = json.load(f)
        else:
            self.brands_dict = {}

        # 3. Load Color Families
        fam_p = Path(families_path)
        self.color_to_family = {}
        if fam_p.exists():
            with open(fam_p, "r", encoding="utf-8") as f:
                fam_data = json.load(f)
                for fam_name, color_list in fam_data.get("colour_families", {}).items():
                    for col in color_list:
                        self.color_to_family[col.lower()] = fam_name

        # Build longest-match-first match tables: (regex, canonical_value)
        brand_mapping = {}
        if isinstance(self.brands_dict, dict) and "canonical_brands" in self.brands_dict:
            canonicals = self.brands_dict.get("canonical_brands", {})
            aliases = self.brands_dict.get("aliases", {})
            for b in canonicals.keys():
                brand_mapping[b] = [b, remove_diacritics(b)]
            for alias, canon in aliases.items():
                if canon not in brand_mapping:
                    brand_mapping[canon] = [canon, remove_diacritics(canon)]
                brand_mapping[canon].extend([alias, remove_diacritics(alias)])
        else:
            brand_mapping = {b: [b, remove_diacritics(b)] for b in self.brands_dict.keys()}

        self.brand_patterns = self._compile_phrase_patterns(brand_mapping)
        self.gender_patterns = self._compile_phrase_patterns(self.lexicon.get("genders", {}))
        self.colour_patterns = self._compile_phrase_patterns(self.lexicon.get("colours", {}))
        self.article_patterns = self._compile_phrase_patterns(self.lexicon.get("article_types", {}))
        self.usage_patterns = self._compile_phrase_patterns(self.lexicon.get("usages", {}))
        self.season_patterns = self._compile_phrase_patterns(self.lexicon.get("seasons", {}))

    def _compile_phrase_patterns(self, mapping: Dict[str, List[str]]) -> List[Tuple[re.Pattern, str, str]]:
        """
        Creates regex patterns sorted by phrase length descending (longest match first).
        Returns list of (compiled_pattern, canonical_key, raw_phrase)
        """
        entries = []
        for canonical, phrases in mapping.items():
            for p in phrases:
                clean_p = remove_diacritics(p)
                if clean_p:
                    entries.append((clean_p, canonical))

        # Deduplicate and sort by length descending (longest phrase first)
        seen = set()
        unique_entries = []
        for phrase, canonical in entries:
            key = (phrase, canonical)
            if key not in seen:
                seen.add(key)
                unique_entries.append((phrase, canonical))

        unique_entries.sort(key=lambda x: len(x[0].split()), reverse=True)

        compiled = []
        for phrase, canonical in unique_entries:
            # Word boundary regex matching
            escaped = re.escape(phrase)
            pattern = re.compile(rf"(?:\b|^){escaped}(?:\b|$)", re.IGNORECASE)
            compiled.append((pattern, canonical, phrase))

        return compiled

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parses a natural text query into structured fields.
        """
        raw_clean = remove_diacritics(query)
        current_text = f" {raw_clean} "
        
        parsed_brand = None
        parsed_gender = None
        parsed_article_type = None
        parsed_colour = None
        parsed_usage = None
        parsed_season = None

        # Helper to blank out matched substring
        def mask_match(match_obj, text):
            start, end = match_obj.span()
            return text[:start] + " " * (end - start) + text[end:]

        # 1. Match Brand (longest match first)
        for pat, canonical, phrase in self.brand_patterns:
            m = pat.search(current_text)
            if m:
                parsed_brand = canonical
                current_text = mask_match(m, current_text)
                break

        # 2. Match Article Type (longest match first)
        for pat, canonical, phrase in self.article_patterns:
            m = pat.search(current_text)
            if m:
                parsed_article_type = canonical
                current_text = mask_match(m, current_text)
                break

        # 3. Match Gender (longest match first)
        for pat, canonical, phrase in self.gender_patterns:
            m = pat.search(current_text)
            if m:
                parsed_gender = canonical
                current_text = mask_match(m, current_text)
                break

        # 4. Match Colour (longest match first)
        for pat, canonical, phrase in self.colour_patterns:
            m = pat.search(current_text)
            if m:
                parsed_colour = canonical
                current_text = mask_match(m, current_text)
                break

        # 5. Match Usage
        for pat, canonical, phrase in self.usage_patterns:
            m = pat.search(current_text)
            if m:
                parsed_usage = canonical
                current_text = mask_match(m, current_text)
                break

        # 6. Match Season
        for pat, canonical, phrase in self.season_patterns:
            m = pat.search(current_text)
            if m:
                parsed_season = canonical
                current_text = mask_match(m, current_text)
                break

        # Map colour family
        colour_family = None
        if parsed_colour:
            colour_family = self.color_to_family.get(parsed_colour.lower(), parsed_colour)

        # Unparsed tokens
        clean_remaining = re.sub(r"\s+", " ", current_text).strip()
        unparsed_tokens = [t for t in clean_remaining.split() if t]

        # Confidence calculation
        total_tokens = len(raw_clean.split())
        if total_tokens == 0:
            confidence = 0.0
        else:
            matched_tokens = max(0, total_tokens - len(unparsed_tokens))
            coverage_ratio = matched_tokens / total_tokens
            
            # Weighted importance of key fields
            field_weight = 0.0
            if parsed_article_type:
                field_weight += 0.45
            if parsed_gender:
                field_weight += 0.20
            if parsed_colour:
                field_weight += 0.20
            if parsed_brand:
                field_weight += 0.15
                
            confidence = round(0.4 * coverage_ratio + 0.6 * min(1.0, field_weight), 4)
            if not parsed_article_type and not parsed_gender and not parsed_colour and not parsed_brand:
                confidence = 0.0

        return {
            "raw_query": query,
            "articleType": parsed_article_type,
            "gender": parsed_gender,
            "baseColour": parsed_colour,
            "colour_family": colour_family,
            "usage": parsed_usage,
            "season": parsed_season,
            "brand": parsed_brand,
            "unparsed_tokens": unparsed_tokens,
            "confidence": confidence
        }

# Module-level instance for simple calling
_default_parser = None

def parse_query(query: str) -> Dict[str, Any]:
    global _default_parser
    if _default_parser is None:
        _default_parser = QueryParser()
    return _default_parser.parse(query)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = QueryParser()
    test_queries = [
        "áo thun Puma nam màu đen",
        "giày thể thao Nike chạy bộ màu xanh navy",
        "váy đầm dự tiệc nữ màu đỏ",
        "đồng hồ Casio nam dây bạc công sở",
        "quần jean nữ ống rộng màu xanh nhạt",
        "United Colors of Benetton áo sơ mi nam trắng",
        "túi xách da nữ Lino Perros màu be",
        "random nonsense từ lạ hoắc xyz123"
    ]
    print("\n--- TEST QUERY PARSER SAMPLES ---")
    for q in test_queries:
        res = parser.parse(q)
        print(f"\nQuery: '{q}'")
        print(f"  articleType  : {res['articleType']}")
        print(f"  gender       : {res['gender']}")
        print(f"  baseColour   : {res['baseColour']} (family: {res['colour_family']})")
        print(f"  brand        : {res['brand']}")
        print(f"  usage/season : {res['usage']} / {res['season']}")
        print(f"  unparsed     : {res['unparsed_tokens']}")
        print(f"  confidence   : {res['confidence']}")
