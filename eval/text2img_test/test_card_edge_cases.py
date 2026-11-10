import sys
import re
import html
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.main import render_card_html

def test_edge_cases():
    print("=" * 60)
    print("UNIT TEST: render_card_html VỚI CÁC CA BIÊN")
    print("=" * 60)

    test_cases = [
        {
            "desc": "Tên rất dài (> 200 ký tự)",
            "prod": {
                "rank": 1,
                "item_id": 99901,
                "name": "Super Long English Product Title " * 10,
                "name_vi": "Tên Sản Phẩm Cực Dài Tiếng Việt Dành Cho Thử Nghiệm Giới Hạn Hiển Thị Giao Diện Người Dùng " * 5,
                "brand": "LongBrandNameTest",
                "articleType": "Shirts",
                "baseColour": "Navy Blue",
                "gender": "Men",
                "score": 0.05,
                "image_path": "data/fashion-dataset/images/15970.jpg"
            }
        },
        {
            "desc": "Tên chứa ký tự nhạy cảm HTML: <b>, <script>, &, ', \"",
            "prod": {
                "rank": 2,
                "item_id": 99902,
                "name": "Item <script>alert('xss')</script> & <b>Bold</b> \"Quotes\"",
                "name_vi": "Sản phẩm <script> & <b>In đậm</b> và 'dấu nháy đơn' \"nháy kép\"",
                "brand": "Brand & Co. <LTD>",
                "articleType": "Tshirts",
                "baseColour": "Red",
                "gender": "Women",
                "score": 0.042,
                "image_path": ""
            }
        },
        {
            "desc": "Thương hiệu rỗng / None / Unknown",
            "prod": {
                "rank": 3,
                "item_id": 99903,
                "name": "Generic Item No Brand",
                "name_vi": "Áo Thun Không Thương Hiệu",
                "brand": None,
                "articleType": "Tshirts",
                "baseColour": "White",
                "gender": "Unisex",
                "score": 0.015,
                "image_path": ""
            }
        },
        {
            "desc": "Thiếu ảnh / file ảnh không tồn tại",
            "prod": {
                "rank": 4,
                "item_id": 99904,
                "name": "Missing Image Product",
                "name_vi": "Sản Phẩm Không Có Ảnh",
                "brand": "Puma",
                "articleType": "Shoes",
                "baseColour": "Black",
                "gender": "Men",
                "score": 0.038,
                "image_path": "non_existent_folder/missing_image_12345.jpg"
            }
        },
        {
            "desc": "Chế độ kỹ thuật bật (show_tech=True)",
            "prod": {
                "rank": 5,
                "item_id": 15970,
                "name": "Turtle Check Men Navy Blue Shirt",
                "name_vi": "Áo Sơ Mi Nam Kẻ Ca-Rô Xanh Navy",
                "brand": "Turtle",
                "articleType": "Shirts",
                "baseColour": "Navy Blue",
                "gender": "Men",
                "score": 0.031,
                "image_path": "data/fashion-dataset/images/15970.jpg"
            },
            "show_tech": True
        }
    ]

    for tc in test_cases:
        desc = tc["desc"]
        prod = tc["prod"]
        show_tech = tc.get("show_tech", False)

        out_html = render_card_html(prod, show_tech=show_tech)

        # 1. Check div open and close balance
        div_opens = len(re.findall(r"<div\b", out_html))
        div_closes = len(re.findall(r"</div>", out_html))
        assert div_opens == div_closes, f"[{desc}] Div mismatch: {div_opens} open vs {div_closes} close"

        # 2. Check no leading spaces >= 4
        for line_no, line in enumerate(out_html.splitlines()):
            assert not line.startswith("    "), f"[{desc}] Line {line_no} has leading indentation >= 4 spaces: {line[:20]}"
            assert line.strip() != "", f"[{desc}] Blank line at {line_no}"

        # 3. Check raw XSS tags are escaped
        assert "<script>" not in out_html
        assert "<b>Bold</b>" not in out_html

        print(f"PASSED: {desc} (Length: {len(out_html)}, <div>: {div_opens}/{div_closes})")

    print("\nTẤT CẢ CÁC CA BIÊN ĐÃ ĐẠT CHUẨN!")

if __name__ == "__main__":
    test_edge_cases()
