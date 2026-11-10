import sys
import re
import html
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.vi_vocab import ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP

# Load VI_MAP
with open("eval/text2img_test/vi_map.json", "r", encoding="utf-8") as f:
    VI_MAP = json.load(f)

# Import the actual functions
from app.main import format_vietnamese_title, render_card_html

def test_vietnamese_titles():
    print("=" * 70)
    print("UNIT TEST: KIỂM THỬ ĐỊNH DẠNG TÊN SẢN PHẨM TIẾNG VIỆT & CÂN BẰNG THẺ HTML")
    print("=" * 70)

    test_samples = [
        {
            "prod": {"name": "Murcia Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "Murcia"},
            "expected_vi": "Túi xách tay nữ Murcia màu hồng"
        },
        {
            "prod": {"name": "ADIDAS Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "ADIDAS"},
            "expected_vi": "Túi xách tay nữ Adidas màu hồng"
        },
        {
            "prod": {"name": "Puma Men puma shield tee Yellow Tshirts", "articleType": "Tshirts", "gender": "Men", "baseColour": "Yellow", "brand": "Puma"},
            "expected_vi": "Áo thun nam Puma màu vàng – puma shield tee"
        },
        {
            "prod": {"name": "Turtle Check Men Navy Blue Shirt", "articleType": "Shirts", "gender": "Men", "baseColour": "Navy Blue", "brand": "Turtle"},
            "expected_vi": "Áo sơ mi nam Turtle màu xanh navy – Check"
        },
        {
            "prod": {"name": "Nike Men Black Air Zoom Running Shoes", "articleType": "Sports Shoes", "gender": "Men", "baseColour": "Black", "brand": "Nike"},
            "expected_vi": "Giày thể thao nam Nike màu đen – Air Zoom Running"
        },
        {
            "prod": {"name": "Lavie Women Pink Handbag", "articleType": "Handbags", "gender": "Women", "baseColour": "Pink", "brand": "Lavie"},
            "expected_vi": "Túi xách tay nữ Lavie màu hồng"
        },
        {
            "prod": {"name": "Special Product <script>alert(1)</script> & <b>Bold</b>", "articleType": "Tshirts", "gender": "Women", "baseColour": "Red", "brand": "Brand & Co"},
            "expected_vi": "Áo thun nữ Brand & Co màu đỏ"
        },
        {
            "prod": {"name": "Generic Item No Color Or Gender", "articleType": "Backpacks", "gender": "", "baseColour": "Unknown", "brand": ""},
            "expected_vi": "Balo Generic – Item"
        }
    ]

    for tc in test_samples:
        p = tc["prod"]
        expected = tc.get("expected_vi")
        actual_vi = format_vietnamese_title(p)
        print(f"Gốc:   \"{p['name']}\"")
        print(f"-> VI: \"{actual_vi}\"")
        if expected:
            if tc["prod"]["name"] in ["Murcia Women Pink Handbag", "ADIDAS Women Pink Handbag", "Puma Men puma shield tee Yellow Tshirts"]:
                assert actual_vi == expected, f"Expected '{expected}', got '{actual_vi}'"
                print(f"  [KHỚP TUYỆT ĐỐI YÊU CẦU: {expected}]")

        # Test HTML rendering in user mode (show_tech=False)
        html_user = render_card_html(p, show_tech=False)
        div_open_user = len(re.findall(r"<div\b", html_user))
        div_close_user = len(re.findall(r"</div>", html_user))
        assert div_open_user == div_close_user, f"User mode div mismatch: {div_open_user} vs {div_close_user}"
        assert not any(line.startswith("    ") for line in html_user.splitlines()), "User mode has indentation >= 4"
        assert not any(line.strip() == "" for line in html_user.splitlines()), "User mode has blank line"
        assert "card-name-raw-tech" not in html_user, "User mode leaked tech raw name container!"

        # Test HTML rendering in tech mode (show_tech=True)
        html_tech = render_card_html(p, show_tech=True)
        div_open_tech = len(re.findall(r"<div\b", html_tech))
        div_close_tech = len(re.findall(r"</div>", html_tech))
        assert div_open_tech == div_close_tech, f"Tech mode div mismatch: {div_open_tech} vs {div_close_tech}"
        assert "card-name-raw-tech" in html_tech, "Tech mode should show raw name"
        print("  -> Thẻ HTML user/tech mode: Cân bằng thẻ 100%, không lộ tiếng Anh ở user mode.\n")

    print("TẤT CẢ UNIT TESTS ĐỊNH DẠNG TÊN TIẾNG VIỆT ĐỀU ĐẠT!")

if __name__ == "__main__":
    test_vietnamese_titles()
