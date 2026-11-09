"""
QueryParser V2 (src/query_parser_v2.py)
Phase 3g Rev5: Post-hoc parser with comprehensive disambiguation for 'đồ' / 'do' (apparel) vs 'đỏ' / 'do' (Red).

Rules applied:
1. Diacritics-aware check for accented text:
   - If a token has grave accent 'đồ' (e.g. 'đồ đi biển', 'đồ tập gym', 'đồ mặc ở nhà', 'đựng đồ'),
     it represents apparel/item and is STRICTLY PREVENTED from matching 'Red'.
2. Unaccented text rule:
   - When the input is unaccented (token is 'do'):
     'do' is ONLY treated as the color Red if:
     a) It directly follows 'mau' / 'màu' (e.g., 'mau do', 'màu do'), OR
     b) It directly follows a known fashion product type noun (e.g., 'ao do', 'vay do', 'dam do',
        'giay do', 'quan do', 'tui do', 'ao thun do', 'so mi do', 'dep do').
   - If 'do' appears at the start of a query followed by non-apparel nouns/verbs (e.g., 'do tap gym',
     'do di bien', 'do mac o nha', 'do the thao', 'do boi', 'do lot'), it is apparel, NOT Red!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import json
import unicodedata
from typing import Dict, List, Optional, Any, Tuple

from src.query_parser import QueryParser, remove_diacritics

# Common product type prefixes in Vietnamese (both accented and unaccented)
PRODUCT_TYPE_PREFIXES = [
    "ao", "áo", "quan", "quần", "vay", "váy", "dam", "đầm", "giay", "giày",
    "dep", "dép", "tui", "túi", "balo", "ba lô", "that lung", "thắt lưng",
    "vi", "ví", "mu", "mũ", "non", "nón", "khan", "khăn", "ca vat", "cà vạt",
    "ao thun", "áo thun", "so mi", "sơ mi", "khoac", "khoác", "hoodie", "polo",
    "short", "blazer", "vest", "chan vay", "chân váy", "sneaker", "sandal", "heels", "dong ho", "đồng hồ"
]

class QueryParserV2(QueryParser):
    def __init__(
        self,
        lexicon_path: str = "src/lexicon_vi.json",
        brands_path: str = "src/brands_v2.json",
        families_path: str = "eval/families.json"
    ):
        super().__init__(lexicon_path=lexicon_path, brands_path=brands_path, families_path=families_path)

    def parse(self, query: str) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "raw_query": query,
                "articleType": None,
                "gender": None,
                "baseColour": None,
                "colour_family": None,
                "usage": None,
                "season": None,
                "brand": None,
                "unparsed_tokens": [],
                "confidence": 0.0
            }

        raw_query = query.strip()
        raw_lower = raw_query.lower()
        res = super().parse(raw_query)

        # Check if 'Red' was assigned
        if res.get("baseColour") == "Red":
            # 1. Accented check: If query contains 'đồ' (grave accent) and NOT 'đỏ' / 'màu đỏ'
            words_raw = re.findall(r"\b\w+\b", raw_lower, flags=re.UNICODE)
            has_do_grave = any(w in ["đồ", "đồ_bộ", "đựng_đồ"] or w == "đồ" for w in words_raw)
            true_red_indicators_accented = ["đỏ", "màu đỏ", "đỏ tươi", "đỏ thẫm", "red", "scarlet", "crimson"]
            has_explicit_red = any(re.search(rf"\b{ind}\b", raw_lower) for ind in true_red_indicators_accented)

            if has_do_grave and not has_explicit_red:
                res["baseColour"] = None
                res["colour_family"] = None

            # 2. Unaccented check: If query has unaccented 'do'
            unaccented_query = remove_diacritics(raw_query)
            if "do" in unaccented_query.split():
                # 'do' is only valid as Red if it follows 'mau' / 'màu' OR a product type prefix
                is_valid_red_context = False

                # Case a: follows 'mau' / 'màu'
                if re.search(r"\b(mau|màu)\s+do\b", unaccented_query):
                    is_valid_red_context = True

                # Case b: follows product type prefix
                if not is_valid_red_context:
                    for prefix in PRODUCT_TYPE_PREFIXES:
                        p_norm = remove_diacritics(prefix)
                        if re.search(rf"\b{re.escape(p_norm)}\s+do\b", unaccented_query):
                            is_valid_red_context = True
                            break

                # If query explicitly contains genuine red in English or accented Vietnamese
                if has_explicit_red:
                    is_valid_red_context = True

                if not is_valid_red_context:
                    res["baseColour"] = None
                    res["colour_family"] = None

            # Recalculate confidence if colour was stripped
            if res.get("baseColour") is None:
                active_fields = [res.get("articleType"), res.get("gender"), res.get("usage"), res.get("season"), res.get("brand")]
                matched_cnt = sum(1 for f in active_fields if f is not None)
                res["confidence"] = round(min(1.0, matched_cnt * 0.25), 4)

        return res

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = QueryParserV2()
    test_queries = [
        ("đồ đi biển", None),
        ("do di bien", None),
        ("đồ tập gym", None),
        ("do tap gym", None),
        ("đồ mặc ở nhà cho nữ", None),
        ("do mac o nha cho nu", None),
        ("túi xách nữ màu đỏ", "Red"),
        ("tui xach nu mau do", "Red"),
        ("ao do nam", "Red"),
        ("vay do nu", "Red"),
        ("giay do nam", "Red"),
        ("túi trống thể thao tập gym đựng đồ", None),
        ("tui trong the thao tap gym dung do", None)
    ]
    print("Testing QueryParserV2 with Accented and Unaccented queries:")
    all_pass = True
    for tq, expected_col in test_queries:
        p = parser.parse(tq)
        actual = p["baseColour"]
        passed = (actual == expected_col)
        all_pass = all_pass and passed
        print(f"  '{tq}' -> Colour: {actual} (Expected: {expected_col}) [{'PASS' if passed else 'FAIL'}]")
    print(f"\nOverall Test Status: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
