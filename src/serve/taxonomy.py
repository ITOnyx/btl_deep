"""
Module: src.serve.taxonomy
Provides a 3-tier product taxonomy derived from real catalog metadata:
- Tier 1: Lĩnh vực (Domain) - e.g., Quần áo, Giày dép, Túi & Ví, Đồng hồ, Trang sức, Kính mắt, Phụ kiện thời trang, Chăm sóc cá nhân, Đồ thể thao, Khác
- Tier 2: Nhóm (Group / subCategory) - e.g., Áo, Quần & Chân váy, Đầm, Giày, Sandal, Túi & Balo, Ví, Đồng hồ...
- Tier 3: Loại (Type / articleType) - 172 unique catalog types
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from src.serve.vn_text import normalize_query, span_compatible, strip_keep_len

def _remove_diacritics(text: str) -> str:
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


# Definition of 172 article types into 10 domains and sub-groups
TAXONOMY_RULES = {
    # Domain 1: Quần áo
    "Quần áo": {
        "Áo": [
            "Tshirts", "Shirts", "Tops", "Topwear", "Jackets", "Sweaters",
            "Sweatshirts", "Kurtas", "Kurtis", "Tunics", "Blazers", "Waistcoat",
            "Nehru Jackets", "Shrug", "Rain Jacket"
        ],
        "Quần & Chân váy": [
            "Jeans", "Trousers", "Shorts", "Track Pants", "Skirts", "Capris",
            "Leggings", "Jeggings", "Churidar", "Salwar", "Patiala", "Bottomwear",
            "Rain Trousers"
        ],
        "Đầm & Trang phục truyền thống": [
            "Dresses", "Dress", "Jumpsuit", "Rompers", "Saree", "Sarees",
            "Lehenga Choli", "Salwar and Dupatta"
        ],
        "Đồ lót & Đồ mặc nhà": [
            "Bra", "Briefs", "Boxers", "Trunk", "Innerwear", "Innerwear Vests",
            "Camisoles", "Night suits", "Nightdress", "Baby Dolls", "Bath Robe",
            "Robe", "Shapewear", "Swimwear", "Loungewear and Nightwear",
            "Lounge Pants", "Lounge Shorts", "Lounge Tshirts"
        ],
        "Bộ trang phục": [
            "Apparel Set", "Clothing Set", "Kurta Sets", "Suits", "Tracksuits"
        ],
    },
    # Domain 2: Giày dép
    "Giày dép": {
        "Giày": [
            "Casual Shoes", "Sports Shoes", "Formal Shoes", "Heels", "Flats",
            "Shoes", "Booties"
        ],
        "Sandal": [
            "Sandals", "Sandal", "Sports Sandals"
        ],
        "Dép xỏ ngón": [
            "Flip Flops"
        ],
    },
    # Domain 3: Túi & Ví
    "Túi & Ví": {
        "Túi & Balo": [
            "Handbags", "Backpacks", "Clutches", "Laptop Bag", "Duffel Bag",
            "Messenger Bag", "Rucksacks", "Trolley Bag", "Waist Pouch",
            "Mobile Pouch", "Bags"
        ],
        "Ví": [
            "Wallets"
        ],
    },
    # Domain 4: Đồng hồ
    "Đồng hồ": {
        "Đồng hồ": ["Watches"]
    },
    # Domain 5: Trang sức
    "Trang sức": {
        "Trang sức": [
            "Bangle", "Bracelet", "Earrings", "Jewellery Set",
            "Necklace and Chains", "Pendant", "Ring", "Jewellery"
        ]
    },
    # Domain 6: Kính mắt
    "Kính mắt": {
        "Kính mắt": ["Sunglasses", "Eyewear"]
    },
    # Domain 7: Phụ kiện thời trang
    "Phụ kiện thời trang": {
        "Mũ nón": ["Caps", "Hat", "Headband", "Headwear"],
        "Khăn & Cà vạt": ["Ties", "Ties and Cufflinks", "Scarves", "Mufflers", "Stoles", "Dupatta"],
        "Phụ kiện khác": [
            "Belts", "Socks", "Stockings", "Tights", "Gloves", "Cufflinks",
            "Key chain", "Shoe Accessories", "Shoe Laces", "Suspenders",
            "Umbrellas", "Accessories", "Accessory Gift Set", "Travel Accessory"
        ]
    },
    # Domain 8: Chăm sóc cá nhân
    "Chăm sóc cá nhân": {
        "Nước hoa & Khử mùi": [
            "Perfume and Body Mist", "Perfumes", "Deodorant", "Fragrance", "Fragrance Gift Set"
        ],
        "Chăm sóc da & Cơ thể": [
            "Face Moisturisers", "Face Scrub and Exfoliator", "Face Serum and Gel",
            "Face Wash and Cleanser", "Body Lotion", "Body Wash and Scrub",
            "Eye Cream", "Sunscreen", "Toner", "Mask and Peel", "Skin", "Skin Care", "Bath and Body"
        ],
        "Trang điểm & Móng": [
            "Lipstick", "Lip Gloss", "Lip Liner", "Lip Care", "Lip Plumper", "Lips",
            "Kajal and Eyeliner", "Mascara", "Eyeshadow", "Eyes", "Foundation and Primer",
            "Compact", "Concealer", "Highlighter and Blush", "Makeup", "Makeup Remover",
            "Nail Polish", "Nail Essentials", "Nails", "Beauty Accessories",
            "Beauty Accessory", "Mens Grooming Kit"
        ],
        "Chăm sóc tóc": ["Hair", "Hair Accessory", "Hair Colour"]
    },
    # Domain 9: Đồ thể thao
    "Đồ thể thao": {
        "Dụng cụ thể thao": [
            "Basketballs", "Footballs", "Sports Accessories", "Sports Equipment",
            "Wristbands", "Water Bottle"
        ]
    },
    # Domain 10: Khác
    "Khác": {
        "Khác": [
            "Home Furnishing", "Cushion Covers", "Ipad", "Tablet Sleeve",
            "Free Gifts", "Vouchers"
        ]
    }
}

# Inverted index: articleType -> (domain, group, articleType)
ARTICLE_TO_TAXONOMY: Dict[str, Tuple[str, str, str]] = {}
for domain, groups in TAXONOMY_RULES.items():
    for group, types in groups.items():
        for t in types:
            ARTICLE_TO_TAXONOMY[t] = (domain, group, t)


# Broad query keywords mapping to Domain and Group
QUERY_TAXONOMY_KEYWORDS: List[Tuple[re.Pattern, str, Optional[str], Optional[str], int, str]] = []
_raw_keywords = [
    # Từ khóa viết CÓ DẤU (dạng chuẩn). Khi khớp, mỗi từ trong truy vấn phải trùng
    # đúng dạng có dấu hoặc được gõ không dấu: tránh 'đậm' -> 'đầm', 'vì' -> 'ví'.
    # Quần áo
    ("áo thun", "Quần áo", "Áo", "Tshirts"),
    ("áo sơ mi", "Quần áo", "Áo", "Shirts"),
    ("áo khoác", "Quần áo", "Áo", "Jackets"),
    ("áo len", "Quần áo", "Áo", "Sweaters"),
    ("áo hoodie", "Quần áo", "Áo", "Sweatshirts"),
    ("áo nỉ", "Quần áo", "Áo", "Sweatshirts"),
    ("áo gile", "Quần áo", "Áo", "Waistcoat"),
    ("áo", "Quần áo", "Áo", None),
    ("quần jean", "Quần áo", "Quần & Chân váy", "Jeans"),
    ("quần bò", "Quần áo", "Quần & Chân váy", "Jeans"),
    ("quần tây", "Quần áo", "Quần & Chân váy", "Trousers"),
    ("quần short", "Quần áo", "Quần & Chân váy", "Shorts"),
    ("quần đùi", "Quần áo", "Quần & Chân váy", "Shorts"),
    ("quần thể thao", "Quần áo", "Quần & Chân váy", "Track Pants"),
    ("quần", "Quần áo", "Quần & Chân váy", None),
    ("chân váy", "Quần áo", "Quần & Chân váy", "Skirts"),
    ("váy liền", "Quần áo", "Đầm & Trang phục truyền thống", "Dresses"),
    ("đầm", "Quần áo", "Đầm & Trang phục truyền thống", "Dresses"),
    ("váy", "Quần áo", "Đầm & Trang phục truyền thống", "Dresses"),
    ("áo ngực", "Quần áo", "Đồ lót & Đồ mặc nhà", "Bra"),
    ("áo lót", "Quần áo", "Đồ lót & Đồ mặc nhà", "Bra"),
    ("quần lót", "Quần áo", "Đồ lót & Đồ mặc nhà", "Briefs"),
    ("đồ lót", "Quần áo", "Đồ lót & Đồ mặc nhà", None),
    ("đồ ngủ", "Quần áo", "Đồ lót & Đồ mặc nhà", None),
    ("đồ bơi", "Quần áo", "Đồ lót & Đồ mặc nhà", "Swimwear"),
    ("quần áo", "Quần áo", None, None),

    # Giày dép
    ("giày thể thao", "Giày dép", "Giày", "Sports Shoes"),
    ("giày chạy bộ", "Giày dép", "Giày", "Sports Shoes"),
    ("giày sneaker", "Giày dép", "Giày", "Casual Shoes"),
    ("giay snacker", "Giày dép", "Giày", "Casual Shoes"),  # lỗi chính tả hay gặp
    ("sneaker", "Giày dép", "Giày", "Casual Shoes"),
    ("giày thường", "Giày dép", "Giày", "Casual Shoes"),
    ("giày tây", "Giày dép", "Giày", "Formal Shoes"),
    ("giày công sở", "Giày dép", "Giày", "Formal Shoes"),
    ("giày cao gót", "Giày dép", "Giày", "Heels"),
    ("giày búp bê", "Giày dép", "Giày", "Flats"),
    ("giày", "Giày dép", "Giày", None),
    ("sandal", "Giày dép", "Sandal", "Sandals"),
    ("xăng đan", "Giày dép", "Sandal", "Sandals"),
    ("dép xăng đan", "Giày dép", "Sandal", "Sandals"),
    ("dép xỏ ngón", "Giày dép", "Dép xỏ ngón", "Flip Flops"),
    ("dép tông", "Giày dép", "Dép xỏ ngón", "Flip Flops"),
    ("dép kẹp", "Giày dép", "Dép xỏ ngón", "Flip Flops"),
    ("dép", "Giày dép", None, None),

    # Túi & Ví
    ("túi xách", "Túi & Ví", "Túi & Balo", "Handbags"),
    ("túi đeo chéo", "Túi & Ví", "Túi & Balo", "Handbags"),
    ("túi laptop", "Túi & Ví", "Túi & Balo", "Laptop Bag"),
    ("ba lô", "Túi & Ví", "Túi & Balo", "Backpacks"),
    ("balo", "Túi & Ví", "Túi & Balo", "Backpacks"),
    ("túi", "Túi & Ví", "Túi & Balo", "Handbags"),
    ("ví da", "Túi & Ví", "Ví", "Wallets"),
    ("ví cầm tay", "Túi & Ví", "Túi & Balo", "Clutches"),
    ("ví tiền", "Túi & Ví", "Ví", "Wallets"),
    ("bóp", "Túi & Ví", "Ví", "Wallets"),
    ("ví", "Túi & Ví", "Ví", "Wallets"),
    ("clutch", "Túi & Ví", "Túi & Balo", "Clutches"),

    # Đồng hồ
    ("đồng hồ", "Đồng hồ", "Đồng hồ", "Watches"),
    ("watch", "Đồng hồ", "Đồng hồ", "Watches"),

    # Trang sức
    ("trang sức", "Trang sức", "Trang sức", None),
    ("vòng tay", "Trang sức", "Trang sức", "Bracelet"),
    ("lắc tay", "Trang sức", "Trang sức", "Bracelet"),
    ("nhẫn", "Trang sức", "Trang sức", "Ring"),
    ("bông tai", "Trang sức", "Trang sức", "Earrings"),
    ("khuyên tai", "Trang sức", "Trang sức", "Earrings"),
    ("dây chuyền", "Trang sức", "Trang sức", "Necklace and Chains"),
    ("vòng cổ", "Trang sức", "Trang sức", "Necklace and Chains"),
    ("mặt dây chuyền", "Trang sức", "Trang sức", "Pendant"),

    # Kính mắt
    ("kính mát", "Kính mắt", "Kính mắt", "Sunglasses"),
    ("kính râm", "Kính mắt", "Kính mắt", "Sunglasses"),
    ("mắt kính", "Kính mắt", "Kính mắt", "Sunglasses"),
    ("kính", "Kính mắt", "Kính mắt", "Sunglasses"),

    # Phụ kiện thời trang
    ("thắt lưng", "Phụ kiện thời trang", "Phụ kiện khác", "Belts"),
    ("dây nịt", "Phụ kiện thời trang", "Phụ kiện khác", "Belts"),
    ("cà vạt", "Phụ kiện thời trang", "Khăn & Cà vạt", "Ties"),
    ("caravat", "Phụ kiện thời trang", "Khăn & Cà vạt", "Ties"),
    ("khăn choàng", "Phụ kiện thời trang", "Khăn & Cà vạt", "Scarves"),
    ("khăn quàng", "Phụ kiện thời trang", "Khăn & Cà vạt", "Scarves"),
    ("mũ lưỡi trai", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("nón kết", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("mũ", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("nón", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("tất", "Phụ kiện thời trang", "Phụ kiện khác", "Socks"),
    ("vớ", "Phụ kiện thời trang", "Phụ kiện khác", "Socks"),
    ("găng tay", "Phụ kiện thời trang", "Phụ kiện khác", "Gloves"),
    ("ô dù", "Phụ kiện thời trang", "Phụ kiện khác", "Umbrellas"),
    ("dù", "Phụ kiện thời trang", "Phụ kiện khác", "Umbrellas"),

    # Chăm sóc cá nhân & Mỹ phẩm
    ("nước hoa", "Chăm sóc cá nhân", "Nước hoa & Khử mùi", "Perfumes"),
    ("dầu thơm", "Chăm sóc cá nhân", "Nước hoa & Khử mùi", "Perfumes"),
    ("xịt toàn thân", "Chăm sóc cá nhân", "Nước hoa & Khử mùi", "Deodorant"),
    ("son môi", "Chăm sóc cá nhân", "Trang điểm & Móng", "Lipstick"),
    ("son", "Chăm sóc cá nhân", "Trang điểm & Móng", "Lipstick"),
    ("phấn mắt", "Chăm sóc cá nhân", "Trang điểm & Móng", "Eyeshadow"),
    ("mascara", "Chăm sóc cá nhân", "Trang điểm & Móng", "Mascara"),
    ("kem nền", "Chăm sóc cá nhân", "Trang điểm & Móng", "Foundation and Primer"),
    ("sữa dưỡng thể", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Body Lotion"),
    ("sữa tắm", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Body Wash and Scrub"),
    ("kem dưỡng", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Face Moisturisers"),
    ("sữa rửa mặt", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Face Wash and Cleanser"),
    ("kem chống nắng", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Sunscreen"),
    ("sơn móng tay", "Chăm sóc cá nhân", "Trang điểm & Móng", "Nail Polish"),
    ("dầu gội", "Chăm sóc cá nhân", "Chăm sóc tóc", "Hair"),
    ("mỹ phẩm", "Chăm sóc cá nhân", None, None),

    # Đồ thể thao
    ("bóng đá", "Đồ thể thao", "Dụng cụ thể thao", "Footballs"),
    ("bóng rổ", "Đồ thể thao", "Dụng cụ thể thao", "Basketballs"),
    ("bình nước", "Đồ thể thao", "Dụng cụ thể thao", "Water Bottle"),
]

# Cụm phi-sản phẩm che đi trước khi khớp: (dạng có dấu, chỉ số từ mơ hồ hoặc None).
# Chỉ che khi span gốc tương thích dấu với cụm (vd. 'đầm màu' KHÔNG bị coi là 'đậm màu').
# Với cụm có từ mơ hồ (chỉ số != None), từ đó phải được gõ CÓ dấu mới che: 'dam mau' / 'dam màu'
# (= đầm màu?) và 'do dam' / 'đỏ dam' (= đồ đầm?) được giữ lại, ưu tiên sản phẩm.
_PROTECTED_NON_PRODUCT = [
    ("tất cả", None), ("đậm màu", 0), ("màu đậm", None), ("đỏ đậm", 1),
    ("xanh đậm", None), ("đen đậm", None), ("nâu đậm", None), ("tím đậm", None),
    ("xám đậm", None), ("vàng đậm", None), ("hồng đậm", None), ("cam đậm", None),
]
_PROTECTED_PATTERNS = [
    (re.compile(rf"(?<!\w){re.escape(_remove_diacritics(ph))}(?!\w)"), ph, amb_idx)
    for ph, amb_idx in _PROTECTED_NON_PRODUCT
]


for kw, dom, grp, art in _raw_keywords:
    kw_acc = unicodedata.normalize("NFC", re.sub(r"\s+", " ", kw.lower()).strip())
    kw_norm = _remove_diacritics(kw_acc)
    pat = re.compile(rf"(?<!\w){re.escape(kw_norm)}(?!\w)")
    QUERY_TAXONOMY_KEYWORDS.append((pat, dom, grp, art, len(kw_norm), kw_acc))

# Longest match first
QUERY_TAXONOMY_KEYWORDS.sort(key=lambda x: x[4], reverse=True)


def get_taxonomy(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extracts 3-tier taxonomy (domain, group, type) for any catalog item.
    Falls back gracefully if metadata is sparse.
    """
    art = item.get("articleType") or item.get("subCategory")
    if not art:
        return ("Khác", "Khác", "Khác")
    art_str = str(art).strip()
    return ARTICLE_TO_TAXONOMY.get(art_str, ("Khác", "Khác", art_str))


def match_query_taxonomy(query_text: str, parsed_attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Determines target domain, group, and articleType from user query or modifier text.
    Returns:
      {
        "domain": Optional[str],
        "group": Optional[str],
        "articleType": Optional[str],
        "explicit_type": bool
      }
    """
    # 1. If parsed_attributes already extracted a specific articleType:
    if parsed_attributes and parsed_attributes.get("articleType"):
        art = parsed_attributes["articleType"][0]
        if art in ARTICLE_TO_TAXONOMY:
            dom, grp, _ = ARTICLE_TO_TAXONOMY[art]
            return {
                "domain": dom,
                "group": grp,
                "articleType": art,
                "explicit_type": True
            }

    # 2. Match against broad taxonomy keywords (e.g. "áo", "giày", "túi", "quần")
    q_low = normalize_query(query_text)
    q_norm = strip_keep_len(q_low)
    for ppat, ph_acc, amb_idx in _PROTECTED_PATTERNS:
        for m in ppat.finditer(q_norm):
            span = q_low[m.start():m.end()]
            if not span_compatible(span, ph_acc):
                continue
            if amb_idx is not None:
                amb_tok = span.split()[amb_idx]
                if amb_tok == strip_keep_len(amb_tok):  # từ mơ hồ gõ không dấu -> không che
                    continue
            q_norm = q_norm[:m.start()] + " " * (m.end() - m.start()) + q_norm[m.end():]
    for pat, dom, grp, art, _, kw_acc in QUERY_TAXONOMY_KEYWORDS:
        if any(span_compatible(q_low[m.start():m.end()], kw_acc) for m in pat.finditer(q_norm)):
            return {
                "domain": dom,
                "group": grp,
                "articleType": art,
                "explicit_type": (art is not None)
            }

    return {
        "domain": None,
        "group": None,
        "articleType": None,
        "explicit_type": False
    }
