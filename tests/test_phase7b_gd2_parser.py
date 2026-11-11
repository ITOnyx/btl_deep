import sys
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.query_parser_v2 import QueryParserV2

@pytest.fixture(scope="module")
def parser():
    # Use candidate lexicon if present, else fallback to src/lexicon_vi.json
    cand_path = ROOT_DIR / "eval" / "phase7" / "lexicon_vi_candidate.json"
    lex_path = str(cand_path) if cand_path.exists() else "src/lexicon_vi.json"
    return QueryParserV2(lexicon_path=lex_path)

def test_swimwear_keywords(parser):
    test_cases = [
        ("đồ bơi", "Swimwear", None),
        ("do boi", "Swimwear", None),
        ("bikini", "Swimwear", None),
        ("áo tắm", "Swimwear", None),
        ("ao tam", "Swimwear", None),
        ("đồ tắm", "Swimwear", None),
        ("do tam", "Swimwear", None),
        ("quần bơi nam", "Swimwear", "Men"),
        ("quan boi nam", "Swimwear", "Men"),
        ("đồ bơi một mảnh nữ", "Swimwear", "Women"),
        ("do boi hai manh", "Swimwear", None),
        ("đồ bơi màu đỏ", "Swimwear", None, "Red"),
        ("do boi mau do", "Swimwear", None, "Red"),
        ("swimsuit nữ màu xanh", "Swimwear", "Women", "Blue"),
    ]
    for tc in test_cases:
        query = tc[0]
        expected_at = tc[1]
        expected_g = tc[2]
        expected_c = tc[3] if len(tc) > 3 else None
        
        parsed = parser.parse(query)
        assert parsed["articleType"] == expected_at, f"Failed for '{query}': expected AT {expected_at}, got {parsed['articleType']}"
        if expected_g:
            assert parsed["gender"] == expected_g, f"Failed for '{query}': expected Gender {expected_g}, got {parsed['gender']}"
        if expected_c:
            assert parsed["baseColour"] == expected_c, f"Failed for '{query}': expected Colour {expected_c}, got {parsed['baseColour']}"

def test_jumpsuit_keywords(parser):
    test_cases = [
        ("jumpsuit", "Jumpsuit"),
        ("jump suit", "Jumpsuit"),
        ("áo liền quần", "Jumpsuit"),
        ("ao lien quan", "Jumpsuit"),
        ("đồ bay nữ", "Jumpsuit"),
        ("do bay", "Jumpsuit"),
        ("bộ jumpsuit", "Jumpsuit"),
        ("bo jumpsuit", "Jumpsuit"),
        ("áo liền quần jumpsuit nữ thanh lịch", "Jumpsuit"),
        ("bo lien than", "Jumpsuit"),
        ("romper", "Jumpsuit"),
        ("playsuit", "Jumpsuit"),
    ]
    for query, expected_at in test_cases:
        parsed = parser.parse(query)
        assert parsed["articleType"] == expected_at, f"Failed for '{query}': expected AT {expected_at}, got {parsed['articleType']}"

def test_bath_robe_keywords(parser):
    test_cases = [
        ("áo choàng tắm", "Bath Robe"),
        ("ao choang tam", "Bath Robe"),
        ("áo choàng ngủ", "Bath Robe"),
        ("ao choang ngu", "Bath Robe"),
        ("bathrobe", "Bath Robe"),
        ("bath robe", "Bath Robe"),
        ("áo choàng tắm khách sạn", "Bath Robe"),
        ("ao choang tam khach san", "Bath Robe"),
        ("áo choàng bông", "Bath Robe"),
    ]
    for query, expected_at in test_cases:
        parsed = parser.parse(query)
        assert parsed["articleType"] == expected_at, f"Failed for '{query}': expected AT {expected_at}, got {parsed['articleType']}"

def test_tracksuits_keywords(parser):
    test_cases = [
        ("bộ đồ thể thao", "Tracksuits"),
        ("bo do the thao", "Tracksuits"),
        ("bộ thể thao", "Tracksuits"),
        ("bo the thao", "Tracksuits"),
        ("bộ nỉ thể thao", "Tracksuits"),
        ("bo ni the thao", "Tracksuits"),
        ("tracksuit nam", "Tracksuits"),
        ("tracksuits", "Tracksuits"),
        ("set đồ thể thao", "Tracksuits"),
        ("bộ quần áo thể thao", "Tracksuits"),
        ("bo quan ao the thao", "Tracksuits"),
    ]
    for query, expected_at in test_cases:
        parsed = parser.parse(query)
        assert parsed["articleType"] == expected_at, f"Failed for '{query}': expected AT {expected_at}, got {parsed['articleType']}"

def test_track_pants_keywords(parser):
    test_cases = [
        ("quần thể thao", "Track Pants"),
        ("quan the thao", "Track Pants"),
        ("quần jogger", "Track Pants"),
        ("quan jogger", "Track Pants"),
        ("tracksuit bottoms", "Track Pants"),
        ("track pants", "Track Pants"),
        ("quan track pants", "Track Pants"),
        ("quần bo gấu", "Track Pants"),
        ("sweatpants", "Track Pants"),
        ("quần jogger thể thao bo gấu màu xám", "Track Pants"),
        ("quan jogger the thao nam mau xam", "Track Pants"),
    ]
    for query, expected_at in test_cases:
        parsed = parser.parse(query)
        assert parsed["articleType"] == expected_at, f"Failed for '{query}': expected AT {expected_at}, got {parsed['articleType']}"

def test_no_cross_collisions(parser):
    # Verify existing categories do not get misclassified
    assert parser.parse("áo thun nam")["articleType"] == "Tshirts"
    assert parser.parse("ao thun nam")["articleType"] == "Tshirts"
    assert parser.parse("áo sơ mi nữ")["articleType"] == "Shirts"
    assert parser.parse("khăn choàng")["articleType"] == "Scarves"
    assert parser.parse("giày thể thao nam")["articleType"] == "Sports Shoes"
    assert parser.parse("quần lót tam giác")["articleType"] == "Briefs"
    assert parser.parse("đồ thể thao")["articleType"] is None  # Generic apparel -> Usage: Sports
    assert parser.parse("đồ thể thao")["usage"] == "Sports"
