"""
QueryParser V2 (src/query_parser_v2.py)
Phase 3g Rev5: Post-hoc parser with disambiguation for 'đồ' (clothing/item) vs 'đỏ' (Red),
supporting both ACCENTED and UNACCENTED ("không dấu") queries.

Rules:
1. Accented input: If query contains 'đồ' (grave accent), it represents apparel/items and is strictly PREVENTED
   from matching the color 'Red' (unless an explicit 'đỏ' / 'red' indicator is also present).
2. Unaccented input: The token 'do' ONLY matches color 'Red' if it stands immediately after:
   - A color marker: "mau" / "màu", "sac" / "sắc", "tone", "tong" / "tông", OR
   - A recognized product type name (tên loại SP), e.g. "ao", "quan", "vay", "dam", "giay", "dep", "tui", "dong ho", etc.
   If 'do' stands alone or after words like "dung" ("dung do"), "di" ("do di bien"), "tap" ("do tap gym"), it is NOT color Red.
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
        
        # Build comprehensive list of product type phrases (clean, unaccented, sorted by length desc)
        sp_types = [
            "ao thun", "ao so mi", "ao khoac", "ao len", "ao hoodie", "ao gio", "ao polo", "ao phong",
            "ao", "quan jean", "quan tay", "quan kaki", "quan short", "quan dui", "quan bo", "quan",
            "vay", "dam", "giay the thao", "giay cao got", "giay tay", "giay sandal", "giay luoi", "giay",
            "dep", "tui xach", "tui deo cheo", "tui", "balo", "vi", "dong ho", "that lung", "that_lung",
            "non", "mu", "kinh", "khan", "vo", "tat", "chan vay", "blazer", "cardigan", "sweater", "hoodie",
            "jacket", "polo", "tshirt", "shirt", "dress", "skirt", "jeans", "pants", "shoes", "bag", "croptop"
        ]
        # Also add all article phrases from lexicon
        for val_list in self.lexicon.get("article_types", {}).values():
            for p in val_list:
                sp_types.append(p)
                
        self.sp_types_clean = sorted(
            list(set(remove_diacritics(x).lower().strip() for x in sp_types if x.strip())),
            key=lambda x: len(x),
            reverse=True
        )
        self.color_markers = {"mau", "sac", "tone", "tong"}

    def _is_do_color(self, raw_query: str) -> bool:
        """
        Determines whether 'do' in the query is intended as color Red.
        """
        raw_lower = raw_query.lower()
        # 1. If explicit 'đỏ' or foreign red words present
        true_red_indicators = ["đỏ", "màu đỏ", "mau do", "màu do", "đỏ tươi", "đỏ thẫm", "do tuoi", "do tham", "red", "scarlet", "crimson"]
        if any(re.search(rf"\b{re.escape(ind)}\b", raw_lower) for ind in true_red_indicators):
            # But if there's explicit 'đồ' without explicit 'đỏ', check
            words_raw = re.findall(r"\b\w+\b", raw_lower, flags=re.UNICODE)
            has_do_grave = "đồ" in words_raw or any(w.startswith("đồ") for w in words_raw)
            has_do_hoi = "đỏ" in words_raw or any(w.startswith("đỏ") for w in words_raw)
            if has_do_grave and not has_do_hoi and not any(w in ["red", "scarlet", "crimson"] for w in words_raw):
                return False
            return True

        # 2. Check unaccented / token 'do'
        clean_q = remove_diacritics(raw_query).lower()
        clean_words = clean_q.split()
        if "do" not in clean_words:
            return False

        # If input explicitly has 'đồ' (grave accent), it's apparel
        words_raw = re.findall(r"\b\w+\b", raw_lower, flags=re.UNICODE)
        if "đồ" in words_raw or any(w.startswith("đồ") for w in words_raw):
            return False

        # For unaccented 'do', check preceding context
        for m in re.finditer(r"\bdo\b", clean_q):
            start = m.start()
            prefix = clean_q[:start].strip()
            if not prefix:
                continue
            prefix_words = prefix.split()
            last_word = prefix_words[-1]
            if last_word in self.color_markers:
                return True
            for sp in self.sp_types_clean:
                if prefix.endswith(sp):
                    return True
        return False

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
        res = super().parse(raw_query)

        # Apply do/đồ/đỏ disambiguation rule
        if res.get("baseColour") == "Red":
            if not self._is_do_color(raw_query):
                res["baseColour"] = None
                res["colour_family"] = None
                active_fields = [res.get("articleType"), res.get("gender"), res.get("usage"), res.get("season"), res.get("brand")]
                matched_cnt = sum(1 for f in active_fields if f is not None)
                res["confidence"] = round(min(1.0, matched_cnt * 0.25), 4)

        return res

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    parser = QueryParserV2()
    test_queries = [
        "đồ đi biển",
        "đồ tập gym",
        "đồ mặc ở nhà cho nữ",
        "túi xách nữ màu đỏ",
        "ao do nam",
        "do tap gym",
        "do di bien",
        "ao thun mau do",
        "túi trống thể thao tập gym đựng đồ",
        "tui trong the thao tap gym dung do"
    ]
    print("Testing QueryParserV2:")
    for tq in test_queries:
        p = parser.parse(tq)
        print(f"'{tq}' -> Colour: {p['baseColour']}, Type: {p['articleType']}, Gender: {p['gender']}, Usage: {p['usage']}")
