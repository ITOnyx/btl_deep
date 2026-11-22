"""Vietnamese label dictionaries and normalization utilities for fashion e-commerce catalog."""

import re
import unicodedata

# 1. Gender mapping
GENDER_MAP = {
    "Men": "nam",
    "Women": "nữ",
    "Boys": "bé trai",
    "Girls": "bé gái",
    "Unisex": "nam nữ"
}

# 2. Base colour mapping (English -> Vietnamese)
COLOUR_MAP = {
    "Black": "đen",
    "White": "trắng",
    "Blue": "xanh dương",
    "Navy Blue": "xanh navy",
    "Red": "đỏ",
    "Grey": "xám",
    "Gray": "xám",
    "Brown": "nâu",
    "Green": "xanh lá",
    "Pink": "hồng",
    "Purple": "tím",
    "Yellow": "vàng",
    "Beige": "be",
    "Orange": "cam",
    "Maroon": "đỏ đô",
    "Gold": "vàng kim",
    "Silver": "bạc",
    "Olive": "xanh rêu",
    "Teal": "xanh cổ vịt",
    "Khaki": "kaki",
    "Cream": "kem",
    "Multi": "nhiều màu",
    "Charcoal": "than chì",
    "Mustard": "vàng mù tạt",
    "Peach": "hồng đào",
    "Burgundy": "đỏ rượu",
    "Lavender": "hoa oải hương",
    "Turquoise": "xanh ngọc",
    "Magenta": "đỏ tươi",
    "Copper": "đồng",
    "Metallic": "ánh kim",
    "Rust": "màu rỉ sét",
    "Bronze": "đồng thanh",
    "Mauve": "tím cẩm quỳ",
    "Coffee Brown": "nâu cà phê",
    "Rose": "hồng phấn",
    "Tan": "nâu vàng",
    "Nude": "màu da",
    "Taupe": "nâu xám",
    "Off White": "trắng ngà",
    "Sea Green": "xanh biển",
    "Lime Green": "xanh chanh",
    "Fluorescent Green": "xanh dạ quang",
    "Skin": "màu da"
}

# 3. ArticleType mapping (English -> Vietnamese)
ARTICLE_TYPE_MAP = {
    "Tshirts": "áo thun",
    "Shirts": "áo sơ mi",
    "Casual Shoes": "giày thường",
    "Watches": "đồng hồ",
    "Sports Shoes": "giày thể thao",
    "Kurtas": "áo kurta",
    "Tops": "áo kiểu",
    "Handbags": "túi xách",
    "Heels": "giày cao gót",
    "Sunglasses": "kính râm",
    "Wallets": "ví tiền",
    "Flip Flops": "dép xỏ ngón",
    "Sandals": "giày xăng đan",
    "Briefs": "quần lót nam",
    "Belts": "thắt lưng",
    "Backpacks": "balo",
    "Socks": "tất",
    "Trousers": "quần tây",
    "Jeans": "quần jean",
    "Shorts": "quần short",
    "Flats": "giày búp bê",
    "Dresses": "đầm váy",
    "Formal Shoes": "giày tây công sở",
    "Bra": "áo ngực",
    "Track Pants": "quần thể thao dài",
    "Jackets": "áo khoác",
    "Sweatshirts": "áo nỉ dài tay",
    "Sweaters": "áo len",
    "Tracksuits": "bộ đồ thể thao",
    "Skirts": "chân váy",
    "Blazers": "áo blazer",
    "Kurtis": "áo kurti",
    "Clutches": "ví cầm tay",
    "Scarves": "khăn choàng cổ",
    "Ties": "cà vạt",
    "Stoles": "khăn quàng vai",
    "Caps": "mũ lưỡi trai",
    "Leggings": "quần bó legging",
    "Nightdress": "váy ngủ",
    "Night suits": "đồ ngủ bộ",
    "Pendant": "mặt dây chuyền",
    "Earrings": "bông tai",
    "Ring": "nhẫn",
    "Cufflinks": "khuy măng sét",
    "Necklace and Chains": "dây chuyền",
    "Bangle": "vòng tay bản to",
    "Bracelet": "lắc tay",
    "Wristbands": "vòng đeo tay thể thao",
    "Laptop Bag": "túi đựng laptop",
    "Duffel Bag": "túi du lịch",
    "Messenger Bag": "túi đeo chéo",
    "Sports Sandal": "dép thể thao quai hậu",
    "Booties": "giày bốt ngắn",
    "Bath Robe": "áo choàng tắm",
    "Jumpsuit": "bộ jumpsuit",
    "Robe": "áo choàng",
    "Waistcoat": "áo gile",
    "Shrug": "áo khoác lửng",
    "Stockings": "vớ da chân",
    "Suspenders": "dây đai quần"
}

def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics while maintaining lower case."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", text).strip().lower()

def translate_gender(gender_en: str) -> str:
    return GENDER_MAP.get(str(gender_en).strip(), "khác")

def translate_colour(colour_en: str) -> str:
    return COLOUR_MAP.get(str(colour_en).strip(), str(colour_en).strip().lower())

def translate_article_type(type_en: str) -> str:
    return ARTICLE_TYPE_MAP.get(str(type_en).strip(), str(type_en).strip().lower())

def build_normalized_title_vi(article_type_en: str, gender_en: str, colour_en: str, raw_title: str) -> str:
    """Build standardized, high-relevance Vietnamese title for search and caption prefix."""
    type_vi = translate_article_type(article_type_en)
    gender_vi = translate_gender(gender_en)
    colour_vi = translate_colour(colour_en)
    
    # Template: [Loại] [Giới] màu [Màu] - [Tên gốc]
    prefix = f"{type_vi} {gender_vi} màu {colour_vi}".capitalize()
    if raw_title and str(raw_title).strip() and str(raw_title).strip() != "nan":
        clean_raw = str(raw_title).strip()
        return f"{prefix} ({clean_raw})"
    return prefix
