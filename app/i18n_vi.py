"""
Internationalization (i18n) module for Vietnamese localization of fashion products.
Translates attributes (gender, color, articleType/subCategory) and generates natural Vietnamese product names.
Does NOT modify original metadata or inference data structures.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

GENDER_VI: Dict[str, str] = {
    "Boys": "bé trai",
    "Girls": "bé gái",
    "Men": "nam",
    "Unisex": "unisex",
    "Women": "nữ",
}

COLOR_VI: Dict[str, str] = {
    "Beige": "be",
    "Black": "đen",
    "Blue": "xanh dương",
    "Bronze": "đồng",
    "Brown": "nâu",
    "Burgundy": "đỏ rượu",
    "Charcoal": "than chì",
    "Coffee Brown": "nâu cà phê",
    "Copper": "đồng đỏ",
    "Cream": "kem",
    "Fluorescent Green": "xanh dạ quang",
    "Gold": "vàng kim",
    "Green": "xanh lá",
    "Grey": "xám",
    "Grey Melange": "xám tiêu",
    "Khaki": "kaki",
    "Lavender": "oải hương",
    "Lime Green": "xanh chanh",
    "Magenta": "hồng cánh sen",
    "Maroon": "đỏ đô",
    "Mauve": "tím nhạt",
    "Metallic": "ánh kim",
    "Multi": "nhiều màu",
    "Mushroom Brown": "nâu nấm",
    "Mustard": "vàng mù tạt",
    "Navy Blue": "xanh navy",
    "Nude": "màu da",
    "Off White": "trắng ngà",
    "Olive": "xanh rêu",
    "Orange": "cam",
    "Peach": "màu đào",
    "Pink": "hồng",
    "Purple": "tím",
    "Red": "đỏ",
    "Rose": "hồng hoa hồng",
    "Rust": "nâu đỏ",
    "Sea Green": "xanh ngọc biển",
    "Silver": "bạc",
    "Skin": "màu da",
    "Steel": "màu thép",
    "Tan": "nâu da bò",
    "Taupe": "nâu xám",
    "Teal": "xanh mòng két",
    "Turquoise Blue": "xanh ngọc",
    "Unknown": "không xác định",
    "White": "trắng",
    "Yellow": "vàng",
    "nan": "không xác định",
    "NaN": "không xác định",
}

ARTICLE_VI: Dict[str, str] = {
    "Accessories": "Phụ kiện",
    "Accessory Gift Set": "Bộ quà tặng phụ kiện",
    "Apparel Set": "Bộ trang phục",
    "Baby Dolls": "Váy ngủ babydoll",
    "Backpacks": "Balo",
    "Bags": "Túi xách",
    "Bangle": "Vòng tay kiềng",
    "Basketballs": "Bóng rổ",
    "Bath Robe": "Áo choàng tắm",
    "Bath and Body": "Sản phẩm tắm gội",
    "Beauty Accessories": "Phụ kiện làm đẹp",
    "Beauty Accessory": "Phụ kiện làm đẹp",
    "Belts": "Thắt lưng",
    "Blazers": "Áo blazer",
    "Body Lotion": "Sữa dưỡng thể",
    "Body Wash and Scrub": "Sữa tắm và tẩy tế bào chết",
    "Booties": "Giày bốt ngắn",
    "Bottomwear": "Trang phục phần dưới",
    "Boxers": "Quần boxer",
    "Bra": "Áo lót",
    "Bracelet": "Vòng đeo tay",
    "Briefs": "Quần lót tam giác",
    "Camisoles": "Áo hai dây",
    "Capris": "Quần lửng capri",
    "Caps": "Mũ lưỡi trai",
    "Casual Shoes": "Giày thường",
    "Churidar": "Quần churidar",
    "Clothing Set": "Bộ quần áo",
    "Clutches": "Ví cầm tay",
    "Compact": "Phấn phủ nén",
    "Concealer": "Kem che khuyết điểm",
    "Cufflinks": "Khuy măng sét",
    "Cushion Covers": "Vỏ gối tựa",
    "Deodorant": "Lăn khử mùi",
    "Dress": "Váy liền",
    "Dresses": "Váy liền",
    "Duffel Bag": "Túi trống thể thao",
    "Dupatta": "Khăn choàng dupatta",
    "Earrings": "Bông tai",
    "Eye Cream": "Kem dưỡng mắt",
    "Eyes": "Sản phẩm trang điểm mắt",
    "Eyeshadow": "Phấn mắt",
    "Eyewear": "Kính mắt",
    "Face Moisturisers": "Kem dưỡng ẩm mặt",
    "Face Scrub and Exfoliator": "Tẩy da chết mặt",
    "Face Serum and Gel": "Serum dưỡng mặt",
    "Face Wash and Cleanser": "Sữa rửa mặt",
    "Flats": "Giày búp bê",
    "Flip Flops": "Dép xỏ ngón",
    "Footballs": "Quả bóng đá",
    "Formal Shoes": "Giày tây công sở",
    "Foundation and Primer": "Kem nền và kem lót",
    "Fragrance": "Nước hoa",
    "Fragrance Gift Set": "Bộ quà tặng nước hoa",
    "Free Gifts": "Quà tặng kèm",
    "Gloves": "Găng tay",
    "Hair": "Sản phẩm chăm sóc tóc",
    "Hair Accessory": "Phụ kiện tóc",
    "Hair Colour": "Thuốc nhuộm tóc",
    "Handbags": "Túi xách",
    "Hat": "Mũ rộng vành",
    "Headband": "Băng đô cài tóc",
    "Headwear": "Mũ nón",
    "Heels": "Giày cao gót",
    "Highlighter and Blush": "Phấn bắt sáng và má hồng",
    "Home Furnishing": "Đồ trang trí nội thất",
    "Innerwear": "Đồ lót",
    "Innerwear Vests": "Áo may ô lót",
    "Ipad": "Bao da iPad",
    "Jackets": "Áo khoác",
    "Jeans": "Quần jean",
    "Jeggings": "Quần jegging",
    "Jewellery": "Trang sức",
    "Jewellery Set": "Bộ trang sức",
    "Jumpsuit": "Bộ jumpsuit liền quần",
    "Kajal and Eyeliner": "Bút kẻ mắt",
    "Key chain": "Móc khóa",
    "Kurta Sets": "Bộ áo kurta",
    "Kurtas": "Áo kurta",
    "Kurtis": "Áo kurti",
    "Laptop Bag": "Túi đựng laptop",
    "Leggings": "Quần legging",
    "Lehenga Choli": "Trang phục lehenga choli",
    "Lip Care": "Dưỡng môi",
    "Lip Gloss": "Son bóng",
    "Lip Liner": "Chì kẻ môi",
    "Lip Plumper": "Son làm đầy môi",
    "Lips": "Sản phẩm trang điểm môi",
    "Lipstick": "Son môi",
    "Lounge Pants": "Quần mặc nhà",
    "Lounge Shorts": "Quần đùi mặc nhà",
    "Lounge Tshirts": "Áo thun mặc nhà",
    "Loungewear and Nightwear": "Đồ ngủ và đồ mặc nhà",
    "Makeup": "Trang điểm",
    "Makeup Remover": "Nước tẩy trang",
    "Mascara": "Chuốt mi mascara",
    "Mask and Peel": "Mặt nạ dưỡng da",
    "Mens Grooming Kit": "Bộ chăm sóc cho nam",
    "Messenger Bag": "Túi đeo chéo",
    "Mobile Pouch": "Túi đựng điện thoại",
    "Mufflers": "Khăn quàng cổ len",
    "Nail Essentials": "Dụng cụ làm móng",
    "Nail Polish": "Sơn móng tay",
    "Nails": "Sản phẩm làm móng",
    "Necklace and Chains": "Dây chuyền vòng cổ",
    "Nehru Jackets": "Áo khoác cổ đứng Nehru",
    "Night suits": "Bộ đồ ngủ",
    "Nightdress": "Váy ngủ",
    "Patiala": "Quần patiala",
    "Pendant": "Mặt dây chuyền",
    "Perfume and Body Mist": "Nước hoa và xịt thơm",
    "Perfumes": "Nước hoa",
    "Rain Jacket": "Áo mưa chống thấm",
    "Rain Trousers": "Quần mưa chống thấm",
    "Ring": "Nhẫn",
    "Robe": "Áo choàng",
    "Rompers": "Bộ romper liền quần ngắn",
    "Rucksacks": "Balo dã ngoại",
    "Salwar": "Quần salwar",
    "Salwar and Dupatta": "Bộ salwar và khăn",
    "Sandal": "Dép sandal",
    "Sandals": "Dép sandal",
    "Saree": "Trang phục saree",
    "Sarees": "Trang phục saree",
    "Scarves": "Khăn quàng",
    "Shapewear": "Đồ lót định hình",
    "Shirts": "Áo sơ mi",
    "Shoe Accessories": "Phụ kiện giày",
    "Shoe Laces": "Dây giày",
    "Shoes": "Giày",
    "Shorts": "Quần short",
    "Shrug": "Áo khoác lửng shrug",
    "Skin": "Chăm sóc da",
    "Skin Care": "Chăm sóc da",
    "Skirts": "Chân váy",
    "Socks": "Tất vớ",
    "Sports Accessories": "Phụ kiện thể thao",
    "Sports Equipment": "Dụng cụ thể thao",
    "Sports Sandals": "Sandal thể thao",
    "Sports Shoes": "Giày thể thao",
    "Stockings": "Quần tất",
    "Stoles": "Khăn choàng vai",
    "Suits": "Bộ comple vest",
    "Sunglasses": "Kính mát",
    "Sunscreen": "Kem chống nắng",
    "Suspenders": "Dây đeo quần chữ Y",
    "Sweaters": "Áo len",
    "Sweatshirts": "Áo nỉ chui đầu",
    "Swimwear": "Đồ bơi",
    "Tablet Sleeve": "Túi chống sốc máy tính bảng",
    "Ties": "Cà vạt",
    "Ties and Cufflinks": "Bộ cà vạt và măng sét",
    "Tights": "Quần bó sát",
    "Toner": "Nước hoa hồng toner",
    "Tops": "Áo",
    "Topwear": "Trang phục phần trên",
    "Track Pants": "Quần thể thao dài",
    "Tracksuits": "Bộ đồ thể thao",
    "Travel Accessory": "Phụ kiện du lịch",
    "Trolley Bag": "Vali kéo",
    "Trousers": "Quần dài",
    "Trunk": "Quần lót nam dạng đùi",
    "Tshirts": "Áo thun",
    "Tunics": "Áo tunic dáng dài",
    "Umbrellas": "Ô dù",
    "Vouchers": "Phiếu quà tặng",
    "Waist Pouch": "Túi bao tử đeo hông",
    "Waistcoat": "Áo gile",
    "Wallets": "Ví bóp",
    "Watches": "Đồng hồ",
    "Water Bottle": "Bình nước thể thao",
    "Wristbands": "Băng đeo cổ tay",
}


def translate_gender(gender: Optional[str]) -> str:
    """Translates gender to Vietnamese, fallback to original if not found."""
    if not gender:
        return ""
    g = str(gender).strip()
    return GENDER_VI.get(g, g)


def translate_color(color: Optional[str]) -> str:
    """Translates baseColour to Vietnamese, fallback to original if not found."""
    if not color:
        return ""
    c = str(color).strip()
    return COLOR_VI.get(c, c)


def translate_article_type(article_type: Optional[str]) -> str:
    """Translates articleType to Vietnamese, fallback to original if not found."""
    if not article_type:
        return ""
    a = str(article_type).strip()
    return ARTICLE_VI.get(a, a)


def product_name_vi(item: Dict[str, Any]) -> str:
    """
    Constructs a natural Vietnamese product name following grammar:
    "{Loại} {giới} màu {màu}"
    Examples:
      - Women Red Handbags -> "Túi xách nữ màu đỏ"
      - Women Purple Handbags -> "Túi xách nữ màu tím"
      - Men Blue Shirts -> "Áo sơ mi nam màu xanh dương"
    Falls back gracefully if fields are missing or unknown.
    """
    article_raw = item.get("articleType") or item.get("subCategory")
    gender_raw = item.get("gender")
    color_raw = item.get("baseColour")

    # If missing articleType, check if we can fallback to title
    if not article_raw:
        orig_title = str(item.get("title", "")).strip()
        if orig_title:
            return orig_title
        pid = item.get("product_id", item.get("item_id", "N/A"))
        return f"Sản phẩm {pid}"

    # Translate individual components
    article_vi = translate_article_type(article_raw)
    gender_vi = translate_gender(gender_raw)
    color_vi = translate_color(color_raw)

    parts = [article_vi]

    # Append gender if available and not unknown/nan
    if gender_vi and gender_vi.lower() not in {"không xác định", "nan", "none"}:
        parts.append(gender_vi.lower())

    # Append color if available and meaningful
    if color_vi and color_vi.lower() not in {"không xác định", "nan", "none"}:
        if color_vi.lower() in {"nhiều màu", "ánh kim"}:
            parts.append(color_vi.lower())
        else:
            parts.append(f"màu {color_vi.lower()}")

    name = " ".join(parts).strip()
    # Capitalize first letter properly
    if name:
        name = name[0].upper() + name[1:]
    return name
