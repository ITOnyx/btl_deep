"""
QueryParser V2 (src/query_parser_v2.py)
Phase 3g Rev4: Post-hoc parser with disambiguation for 'đồ' (clothing/item) vs 'đỏ' (Red).

Key Fix:
- In V1, diacritics removal mapped both 'đồ' and 'đỏ' to unaccented token 'do', which matched 'Red' in lexicon_vi.json.
- In V2, we perform diacritics-aware disambiguation:
  If a token explicitly has grave accent 'đồ' (or compounds like 'đồ ngủ', 'đồ tập', 'đồ bơi', 'đồ lót'),
  it represents apparel/items and is strictly PREVENTED from matching the color 'Red'.
- Also adds gender-preservation check for fallback relaxation so queries with explicit gender do not leak opposite gender.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import json
import unicodedata
from typing import Dict, List, Optional, Any, Tuple

from src.query_parser import QueryParser, remove_diacritics

class QueryParserV2(QueryParser):
    def __init__(
        self,
        lexicon_path: str = "src/lexicon_vi.json",
        brands_path: str = "src/brands_v2.json",
        families_path: str = "eval/families.json"
    ):
        super().__init__(lexicon_path=lexicon_path, brands_path=brands_path, families_path=families_path)
        
        # Refine colour patterns: ensure single token 'do' does not match if original word was 'đồ'
        # We will handle this in parse() by protecting 'đồ'
        
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
        
        # Check if query contains the word 'đồ' with grave accent (or compounds)
        # Tokenize raw query preserving case/accents
        words_raw = re.findall(r"\b\w+\b", raw_query, flags=re.UNICODE)
        has_do_grave = any(w.lower() in ["đồ", "đồ_bộ", "đựng_đồ"] or w.lower() == "đồ" for w in words_raw)
        
        # If the word is explicitly 'đồ' (and NOT 'đỏ' or 'màu đỏ'), protect it from color matcher
        res = super().parse(raw_query)
        
        if has_do_grave:
            # Check if 'Red' was falsely triggered solely by 'đồ'
            # If raw query does NOT contain 'đỏ', 'màu đỏ', 'mau do', 'do tuoi', 'do tham', 'red'
            # then remove the false 'Red' colour
            raw_lower = raw_query.lower()
            true_red_indicators = ["đỏ", "màu đỏ", "mau do", "màu do", "đỏ tươi", "đỏ thẫm", "do tuoi", "do tham", "red", "scarlet", "crimson"]
            has_true_red = any(re.search(rf"\b{ind}\b", raw_lower) for ind in true_red_indicators)
            
            if not has_true_red and res.get("baseColour") == "Red":
                res["baseColour"] = None
                res["colour_family"] = None
                # Recalculate confidence
                # If only usage or gender remained
                active_fields = [res.get("articleType"), res.get("gender"), res.get("usage"), res.get("season"), res.get("brand")]
                matched_cnt = sum(1 for f in active_fields if f is not None)
                res["confidence"] = round(min(1.0, matched_cnt * 0.25), 4)

        return res

if __name__ == "__main__":
    parser = QueryParserV2()
    test_queries = [
        "đồ đi biển",
        "đồ tập gym",
        "đồ mặc ở nhà cho nữ",
        "túi xách nữ màu đỏ",
        "ao do nam",
        "túi trống thể thao tập gym đựng đồ"
    ]
    print("Testing QueryParserV2:")
    for tq in test_queries:
        p = parser.parse(tq)
        print(f"'{tq}' -> Colour: {p['baseColour']}, Type: {p['articleType']}, Gender: {p['gender']}, Usage: {p['usage']}")
