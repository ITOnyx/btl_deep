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
QUERY_TAXONOMY_KEYWORDS: List[Tuple[re.Pattern, str, Optional[str], Optional[str], int]] = []
_raw_keywords = [
    # Quần áo
    ("ao thun", "Quần áo", "Áo", "Tshirts"),
    ("ao so mi", "Quần áo", "Áo", "Shirts"),
    ("ao khoac", "Quần áo", "Áo", "Jackets"),
    ("ao len", "Quần áo", "Áo", "Sweaters"),
    ("ao hoodie", "Quần áo", "Áo", "Sweatshirts"),
    ("ao ni", "Quần áo", "Áo", "Sweatshirts"),
    ("ao gile", "Quần áo", "Áo", "Waistcoat"),
    ("ao", "Quần áo", "Áo", None),
    ("quan jean", "Quần áo", "Quần & Chân váy", "Jeans"),
    ("quan bo", "Quần áo", "Quần & Chân váy", "Jeans"),
    ("quan tay", "Quần áo", "Quần & Chân váy", "Trousers"),
    ("quan short", "Quần áo", "Quần & Chân váy", "Shorts"),
    ("quan dui", "Quần áo", "Quần & Chân váy", "Shorts"),
    ("quan the thao", "Quần áo", "Quần & Chân váy", "Track Pants"),
    ("quan", "Quần áo", "Quần & Chân váy", None),
    ("chan vay", "Quần áo", "Quần & Chân váy", "Skirts"),
    ("vay lien", "Quần áo", "Đầm & Trang phục truyền thống", "Dresses"),
    ("dam", "Quần áo", "Đầm & Trang phục truyền thống", "Dresses"),
    ("vay", "Quần áo", "Đầm & Trang phục truyền thống", "Dresses"),
    ("ao nguc", "Quần áo", "Đồ lót & Đồ mặc nhà", "Bra"),
    ("ao lot", "Quần áo", "Đồ lót & Đồ mặc nhà", "Bra"),
    ("quan lot", "Quần áo", "Đồ lót & Đồ mặc nhà", "Briefs"),
    ("do lot", "Quần áo", "Đồ lót & Đồ mặc nhà", None),
    ("do ngu", "Quần áo", "Đồ lót & Đồ mặc nhà", None),
    ("do boi", "Quần áo", "Đồ lót & Đồ mặc nhà", "Swimwear"),
    ("quan ao", "Quần áo", None, None),

    # Giày dép
    ("giay the thao", "Giày dép", "Giày", "Sports Shoes"),
    ("giay chay bo", "Giày dép", "Giày", "Sports Shoes"),
    ("giay snacker", "Giày dép", "Giày", "Casual Shoes"),
    ("sneaker", "Giày dép", "Giày", "Casual Shoes"),
    ("giay thuong", "Giày dép", "Giày", "Casual Shoes"),
    ("giay tay", "Giày dép", "Giày", "Formal Shoes"),
    ("giay cong so", "Giày dép", "Giày", "Formal Shoes"),
    ("giay cao got", "Giày dép", "Giày", "Heels"),
    ("giay bup be", "Giày dép", "Giày", "Flats"),
    ("giay", "Giày dép", "Giày", None),
    ("sandal", "Giày dép", "Sandal", "Sandals"),
    ("xang dan", "Giày dép", "Sandal", "Sandals"),
    ("dep xang dan", "Giày dép", "Sandal", "Sandals"),
    ("dep xo ngon", "Giày dép", "Dép xỏ ngón", "Flip Flops"),
    ("dep tong", "Giày dép", "Dép xỏ ngón", "Flip Flops"),
    ("dep kep", "Giày dép", "Dép xỏ ngón", "Flip Flops"),
    ("dep", "Giày dép", None, None),

    # Túi & Ví
    ("tui xach", "Túi & Ví", "Túi & Balo", "Handbags"),
    ("tui deo cheo", "Túi & Ví", "Túi & Balo", "Handbags"),
    ("tui laptop", "Túi & Ví", "Túi & Balo", "Laptop Bag"),
    ("ba lo", "Túi & Ví", "Túi & Balo", "Backpacks"),
    ("balo", "Túi & Ví", "Túi & Balo", "Backpacks"),
    ("tui", "Túi & Ví", "Túi & Balo", "Handbags"),
    ("vi da", "Túi & Ví", "Ví", "Wallets"),
    ("vi cam tay", "Túi & Ví", "Túi & Balo", "Clutches"),
    ("vi tien", "Túi & Ví", "Ví", "Wallets"),
    ("bop", "Túi & Ví", "Ví", "Wallets"),
    ("vi", "Túi & Ví", "Ví", "Wallets"),
    ("clutch", "Túi & Ví", "Túi & Balo", "Clutches"),

    # Đồng hồ
    ("dong ho", "Đồng hồ", "Đồng hồ", "Watches"),
    ("watch", "Đồng hồ", "Đồng hồ", "Watches"),

    # Trang sức
    ("trang suc", "Trang sức", "Trang sức", None),
    ("vong tay", "Trang sức", "Trang sức", "Bracelet"),
    ("lac tay", "Trang sức", "Trang sức", "Bracelet"),
    ("nhan", "Trang sức", "Trang sức", "Ring"),
    ("bong tai", "Trang sức", "Trang sức", "Earrings"),
    ("khuyen tai", "Trang sức", "Trang sức", "Earrings"),
    ("day chuyen", "Trang sức", "Trang sức", "Necklace and Chains"),
    ("vong co", "Trang sức", "Trang sức", "Necklace and Chains"),
    ("mat day chuyen", "Trang sức", "Trang sức", "Pendant"),

    # Kính mắt
    ("kinh mat", "Kính mắt", "Kính mắt", "Sunglasses"),
    ("kinh ram", "Kính mắt", "Kính mắt", "Sunglasses"),
    ("mat kinh", "Kính mắt", "Kính mắt", "Sunglasses"),
    ("kinh", "Kính mắt", "Kính mắt", "Sunglasses"),

    # Phụ kiện thời trang
    ("that lung", "Phụ kiện thời trang", "Phụ kiện khác", "Belts"),
    ("day nit", "Phụ kiện thời trang", "Phụ kiện khác", "Belts"),
    ("ca vat", "Phụ kiện thời trang", "Khăn & Cà vạt", "Ties"),
    ("caravat", "Phụ kiện thời trang", "Khăn & Cà vạt", "Ties"),
    ("khan choang", "Phụ kiện thời trang", "Khăn & Cà vạt", "Scarves"),
    ("khan quang", "Phụ kiện thời trang", "Khăn & Cà vạt", "Scarves"),
    ("mu luoi trai", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("non ket", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("mu", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("non", "Phụ kiện thời trang", "Mũ nón", "Caps"),
    ("tat", "Phụ kiện thời trang", "Phụ kiện khác", "Socks"),
    ("vo", "Phụ kiện thời trang", "Phụ kiện khác", "Socks"),
    ("gang tay", "Phụ kiện thời trang", "Phụ kiện khác", "Gloves"),
    ("o du", "Phụ kiện thời trang", "Phụ kiện khác", "Umbrellas"),
    ("du", "Phụ kiện thời trang", "Phụ kiện khác", "Umbrellas"),

    # Chăm sóc cá nhân & Mỹ phẩm
    ("nuoc hoa", "Chăm sóc cá nhân", "Nước hoa & Khử mùi", "Perfumes"),
    ("dau thom", "Chăm sóc cá nhân", "Nước hoa & Khử mùi", "Perfumes"),
    ("xit toan than", "Chăm sóc cá nhân", "Nước hoa & Khử mùi", "Deodorant"),
    ("son moi", "Chăm sóc cá nhân", "Trang điểm & Móng", "Lipstick"),
    ("son", "Chăm sóc cá nhân", "Trang điểm & Móng", "Lipstick"),
    ("phan mat", "Chăm sóc cá nhân", "Trang điểm & Móng", "Eyeshadow"),
    ("mascara", "Chăm sóc cá nhân", "Trang điểm & Móng", "Mascara"),
    ("kem nen", "Chăm sóc cá nhân", "Trang điểm & Móng", "Foundation and Primer"),
    ("sua duong the", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Body Lotion"),
    ("sua tam", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Body Wash and Scrub"),
    ("kem duong", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Face Moisturisers"),
    ("sua rua mat", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Face Wash and Cleanser"),
    ("kem chong nang", "Chăm sóc cá nhân", "Chăm sóc da & Cơ thể", "Sunscreen"),
    ("son mong tay", "Chăm sóc cá nhân", "Trang điểm & Móng", "Nail Polish"),
    ("dau goi", "Chăm sóc cá nhân", "Chăm sóc tóc", "Hair"),
    ("my pham", "Chăm sóc cá nhân", None, None),

    # Đồ thể thao
    ("bong da", "Đồ thể thao", "Dụng cụ thể thao", "Footballs"),
    ("bong ro", "Đồ thể thao", "Dụng cụ thể thao", "Basketballs"),
    ("binh nuoc", "Đồ thể thao", "Dụng cụ thể thao", "Water Bottle"),
]

for kw, dom, grp, art in _raw_keywords:
    kw_norm = _remove_diacritics(kw)
    pat = re.compile(rf"(?<!\w){re.escape(kw_norm)}(?!\w)")
    QUERY_TAXONOMY_KEYWORDS.append((pat, dom, grp, art, len(kw_norm)))

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
    q_norm = _remove_diacritics(query_text)
    for pat, dom, grp, art, _ in QUERY_TAXONOMY_KEYWORDS:
        if pat.search(q_norm):
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
