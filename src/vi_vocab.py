"""
Vietnamese Fashion Vocabulary & Metadata Mapping
Provides English -> Vietnamese translations (with and without diacritics)
for gender, articleType, baseColour, and usage.
"""

import unicodedata
import re

def remove_diacritics(text: str) -> str:
    """Removes Vietnamese tone marks and accents for search normalization."""
    if not text:
        return ""
    # Normalize unicode to decomposed form
    normalized = unicodedata.normalize("NFD", text)
    # Remove combining characters
    without_combining = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    # Replace special Vietnamese characters (d/D)
    res = without_combining.replace("đ", "d").replace("Đ", "D")
    return res.lower().strip()


# Gender Mapping
GENDER_MAP = {
    "Men": "Nam",
    "Women": "Nữ",
    "Boys": "Bé trai",
    "Girls": "Bé gái",
    "Unisex": "Unisex (Dùng chung)",
}

# Base Colour Mapping
COLOUR_MAP = {
    "Beige": "Màu be",
    "Black": "Đen",
    "Blue": "Xanh dương",
    "Bronze": "Màu đồng",
    "Brown": "Nâu",
    "Burgundy": "Đỏ rượu burgundy",
    "Charcoal": "Xám than",
    "Coffee Brown": "Nâu cà phê",
    "Copper": "Đồng đỏ",
    "Cream": "Màu kem",
    "Fluorescent Green": "Xanh dạ quang",
    "Gold": "Vàng kim",
    "Green": "Xanh lá",
    "Grey": "Xám",
    "Grey Melange": "Xám tiêu melange",
    "Khaki": "Kaki",
    "Lavender": "Hoa oải hương tím nhạt",
    "Lime Green": "Xanh chanh",
    "Magenta": "Đỏ tươi cánh sen magenta",
    "Maroon": "Đỏ hạt dẻ",
    "Mauve": "Tím hoa cà mauve",
    "Metallic": "Ánh kim",
    "Multi": "Nhiều màu",
    "Mushroom Brown": "Nâu nấm",
    "Mustard": "Vàng mù tạt",
    "Navy Blue": "Xanh navy (xanh than)",
    "Nude": "Màu nude (da)",
    "Off White": "Trắng ngà",
    "Olive": "Xanh ô liu",
    "Orange": "Cam",
    "Peach": "Màu đào",
    "Pink": "Hồng",
    "Purple": "Tím",
    "Red": "Đỏ",
    "Rose": "Hồng cánh hồng",
    "Rust": "Đỏ rỉ sét",
    "Sea Green": "Xanh nước biển",
    "Silver": "Bạc",
    "Skin": "Màu da",
    "Steel": "Xám thép",
    "Tan": "Nâu rám nắng tan",
    "Taupe": "Nâu xám taupe",
    "Teal": "Xanh mòng két teal",
    "Turquoise Blue": "Xanh ngọc lam",
    "White": "Trắng",
    "Yellow": "Vàng",
}

# Usage Mapping
USAGE_MAP = {
    "Casual": "Thường ngày (Casual)",
    "Ethnic": "Truyền thống (Ethnic)",
    "Formal": "Công sở / Trang trọng (Formal)",
    "Home": "Mặc nhà",
    "Party": "Dự tiệc (Party)",
    "Smart Casual": "Lịch thiệp thường ngày (Smart Casual)",
    "Sports": "Thể thao",
    "Travel": "Du lịch",
}

# Article Type Mapping (All 143 categories)
ARTICLE_TYPE_MAP = {
    "Accessory Gift Set": "Bộ quà tặng phụ kiện",
    "Baby Dolls": "Váy ngủ babydoll",
    "Backpacks": "Balo",
    "Bangle": "Vòng tay bản cứng",
    "Basketballs": "Quả bóng rổ",
    "Bath Robe": "Áo choàng tắm",
    "Beauty Accessory": "Phụ kiện làm đẹp",
    "Belts": "Thắt lưng (Dây nịt)",
    "Blazers": "Áo blazer",
    "Body Lotion": "Sữa dưỡng thể",
    "Body Wash and Scrub": "Sữa tắm tẩy tế bào chết",
    "Booties": "Giày bốt cổ thấp",
    "Boxers": "Quần lót boxer",
    "Bra": "Áo ngực",
    "Bracelet": "Vòng tay (Lắc tay)",
    "Briefs": "Quần lót tam giác",
    "Camisoles": "Áo hai dây camisole",
    "Capris": "Quần lửng capri",
    "Caps": "Mũ lưỡi trai",
    "Casual Shoes": "Giày thường ngày",
    "Churidar": "Quần ống túm churidar",
    "Clothing Set": "Set trang phục bộ",
    "Clutches": "Ví cầm tay clutch",
    "Compact": "Phấn phủ dạng nén",
    "Concealer": "Kem che khuyết điểm",
    "Cufflinks": "Khuy măng sét",
    "Cushion Covers": "Vỏ gối tựa",
    "Deodorant": "Lăn xịt khử mùi",
    "Dresses": "Váy liền (Đầm)",
    "Duffel Bag": "Túi trống du lịch",
    "Dupatta": "Khăn choàng dupatta",
    "Earrings": "Bông tai (Hoa tai)",
    "Eye Cream": "Kem dưỡng mắt",
    "Eyeshadow": "Phấn mắt",
    "Face Moisturisers": "Kem dưỡng ẩm da mặt",
    "Face Scrub and Exfoliator": "Tẩy da chết mặt",
    "Face Serum and Gel": "Serum dưỡng da mặt",
    "Face Wash and Cleanser": "Sữa rửa mặt",
    "Flats": "Giày búp bê đế bệt",
    "Flip Flops": "Dép xỏ ngón (Dép kẹp)",
    "Footballs": "Quả bóng đá",
    "Formal Shoes": "Giày tây công sở",
    "Foundation and Primer": "Kem nền và kem lót",
    "Fragrance Gift Set": "Bộ quà tặng nước hoa",
    "Free Gifts": "Quà tặng kèm",
    "Gloves": "Găng tay",
    "Hair Accessory": "Phụ kiện tóc",
    "Hair Colour": "Thuốc nhuộm tóc",
    "Handbags": "Túi xách tay",
    "Hat": "Mũ vành / Mũ rộng vành",
    "Headband": "Băng đô cài tóc",
    "Heels": "Giày cao gót",
    "Highlighter and Blush": "Phấn bắt sáng và má hồng",
    "Innerwear Vests": "Áo may ô lót trong",
    "Ipad": "Bao da ốp iPad",
    "Jackets": "Áo khoác jacket",
    "Jeans": "Quần jean (Quần bò)",
    "Jeggings": "Quần jegging",
    "Jewellery Set": "Bộ trang sức",
    "Jumpsuit": "Đồ bay jumpsuit",
    "Kajal and Eyeliner": "Bút kẻ mắt eyeliner",
    "Key chain": "Móc khóa",
    "Kurta Sets": "Bộ áo kurta",
    "Kurtas": "Áo kurta truyền thống",
    "Kurtis": "Áo kurti ngắn",
    "Laptop Bag": "Túi đựng laptop",
    "Leggings": "Quần legging",
    "Lehenga Choli": "Trang phục lehenga choli",
    "Lip Care": "Dưỡng môi",
    "Lip Gloss": "Son bóng",
    "Lip Liner": "Chì kẻ viền môi",
    "Lip Plumper": "Son làm căng môi",
    "Lipstick": "Son môi",
    "Lounge Pants": "Quần dài mặc nhà",
    "Lounge Shorts": "Quần đùi mặc nhà",
    "Lounge Tshirts": "Áo thun mặc nhà",
    "Makeup Remover": "Nước tẩy trang",
    "Mascara": "Chuốt mi mascara",
    "Mask and Peel": "Mặt nạ lột",
    "Mens Grooming Kit": "Bộ chăm sóc nam giới",
    "Messenger Bag": "Túi đeo chéo messenger",
    "Mobile Pouch": "Túi đeo điện thoại",
    "Mufflers": "Khăn quàng cổ len",
    "Nail Essentials": "Dụng cụ chăm sóc móng",
    "Nail Polish": "Sơn móng tay",
    "Necklace and Chains": "Dây chuyền và vòng cổ",
    "Nehru Jackets": "Áo khoác cổ đứng Nehru",
    "Night suits": "Bộ đồ ngủ",
    "Nightdress": "Váy ngủ",
    "Patiala": "Quần patiala thụng",
    "Pendant": "Mặt dây chuyền",
    "Perfume and Body Mist": "Nước hoa và xịt toàn thân",
    "Rain Jacket": "Áo mưa",
    "Rain Trousers": "Quần đi mưa",
    "Ring": "Nhẫn",
    "Robe": "Áo choàng",
    "Rompers": "Bộ áo liền quần romper",
    "Rucksacks": "Balo phượt",
    "Salwar": "Quần salwar",
    "Salwar and Dupatta": "Bộ quần salwar và khăn dupatta",
    "Sandals": "Giày xăng-đan (Sandal)",
    "Sarees": "Trang phục saree truyền thống",
    "Scarves": "Khăn quàng cổ",
    "Shapewear": "Đồ lót định hình",
    "Shirts": "Áo sơ mi",
    "Shoe Accessories": "Phụ kiện giày",
    "Shoe Laces": "Dây giày",
    "Shorts": "Quần soóc (Quần ngắn)",
    "Shrug": "Áo khoác lửng shrug",
    "Skirts": "Chân váy",
    "Socks": "Tất (Vớ)",
    "Sports Sandals": "Sandal thể thao",
    "Sports Shoes": "Giày thể thao",
    "Stockings": "Quần tất dài",
    "Stoles": "Khăn choàng vai",
    "Suits": "Bộ âu phục (Vest / Suit)",
    "Sunglasses": "Kính râm (Kính mát)",
    "Sunscreen": "Kem chống nắng",
    "Suspenders": "Dây đeo quần chữ Y",
    "Sweaters": "Áo len chui đầu",
    "Sweatshirts": "Áo nỉ chui đầu",
    "Swimwear": "Đồ bơi",
    "Tablet Sleeve": "Túi chống sốc máy tính bảng",
    "Ties": "Cà vạt",
    "Ties and Cufflinks": "Bộ cà vạt và khuy măng sét",
    "Tights": "Quần bó ôm tights",
    "Toner": "Nước hoa hồng toner",
    "Tops": "Áo kiểu nữ",
    "Track Pants": "Quần thể thao bo gấu",
    "Tracksuits": "Bộ quần áo thể thao",
    "Travel Accessory": "Phụ kiện du lịch",
    "Trolley Bag": "Vali kéo",
    "Trousers": "Quần âu (Quần tây)",
    "Trunk": "Quần lót đùi nam (Trunk)",
    "Tshirts": "Áo thun (Áo phông)",
    "Tunics": "Áo tunic dáng dài",
    "Umbrellas": "Ô (Dù)",
    "Waist Pouch": "Túi đeo bao tử / Túi đeo hông",
    "Waistcoat": "Áo gile (Áo ghi-lê)",
    "Wallets": "Ví tiền (Bóp tiền)",
    "Watches": "Đồng hồ đeo tay",
    "Water Bottle": "Bình đựng nước",
    "Wristbands": "Băng đeo cổ tay",
}

# Precompute non-diacritic versions for rapid search indexing
GENDER_MAP_NO_DIACRITICS = {k: remove_diacritics(v) for k, v in GENDER_MAP.items()}
COLOUR_MAP_NO_DIACRITICS = {k: remove_diacritics(v) for k, v in COLOUR_MAP.items()}
USAGE_MAP_NO_DIACRITICS = {k: remove_diacritics(v) for k, v in USAGE_MAP.items()}
ARTICLE_TYPE_MAP_NO_DIACRITICS = {k: remove_diacritics(v) for k, v in ARTICLE_TYPE_MAP.items()}


def translate_gender(en_gender: str, with_diacritics: bool = True) -> str:
    m = GENDER_MAP if with_diacritics else GENDER_MAP_NO_DIACRITICS
    return m.get(en_gender, en_gender)


def translate_colour(en_colour: str, with_diacritics: bool = True) -> str:
    m = COLOUR_MAP if with_diacritics else COLOUR_MAP_NO_DIACRITICS
    return m.get(en_colour, en_colour)


def translate_article_type(en_type: str, with_diacritics: bool = True) -> str:
    m = ARTICLE_TYPE_MAP if with_diacritics else ARTICLE_TYPE_MAP_NO_DIACRITICS
    return m.get(en_type, en_type)


def translate_usage(en_usage: str, with_diacritics: bool = True) -> str:
    m = USAGE_MAP if with_diacritics else USAGE_MAP_NO_DIACRITICS
    return m.get(en_usage, en_usage)
