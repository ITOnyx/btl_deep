"""
Unit tests for CIR Modifier Normalizer
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.cir_normalizer import normalize_cir_modifier, is_already_english
from src.query_parser_v2 import QueryParserV2

def test_cir_normalizer():
    parser = QueryParserV2()
    print("=== RUNNING CIR NORMALIZER UNIT TESTS ===")

    # Case 1: Color only with ref_type="Shirts"
    p1 = parser.parse("màu xanh navy")
    en1, fb1, un1 = normalize_cir_modifier("màu xanh navy", p1, ref_type="Shirts")
    assert en1 == "navy blue shirt", f"Failed: {en1}"
    assert not fb1

    # Case 2: Gender shift
    p2 = parser.parse("cho nữ")
    en2, fb2, un2 = normalize_cir_modifier("cho nữ", p2, ref_type="Shirts")
    assert en2 == "women shirt", f"Failed: {en2}"
    assert not fb2

    # Case 3: Brand shift
    p3 = parser.parse("của Nike")
    en3, fb3, un3 = normalize_cir_modifier("của Nike", p3, ref_type="Tshirts")
    assert en3 == "Nike tshirt", f"Failed: {en3}"
    assert not fb3

    # Case 4: English text preservation
    p4 = parser.parse("navy blue shirt")
    en4, fb4, un4 = normalize_cir_modifier("navy blue shirt", p4, ref_type="Shirts")
    assert en4 == "navy blue shirt", f"Failed: {en4}"
    assert is_already_english("navy blue shirt")
    assert not fb4

    # Case 5: Empty text
    en5, fb5, un5 = normalize_cir_modifier("", {}, ref_type="Shirts")
    assert en5 == ""
    assert not fb5

    # Case 6: Pure gibberish / untranslated
    p6 = parser.parse("xyzabc123 bất thường")
    en6, fb6, un6 = normalize_cir_modifier("xyzabc123 bất thường", p6, ref_type=None)
    assert fb6 == True, f"Expected fallback, got {en6}"
    assert "xyzabc123" in un6

    print("🟢 All unit tests for normalize_cir_modifier passed 100%!")

if __name__ == "__main__":
    test_cir_normalizer()
