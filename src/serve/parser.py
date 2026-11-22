"""
Module: src.serve.parser
Bộ phân tích truy vấn tiếng Việt (có dấu & không dấu) thành các thuộc tính có cấu trúc:
- articleType (loại sản phẩm)
- gender (giới tính)
- baseColour (màu sắc)
Dùng cho tầng Hệ thống (System Scoring & Soft Bonus) trong Pha 5 và Ứng dụng Streamlit Pha 6.
"""
import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Any


def remove_diacritics(text: str) -> str:
    """Chuyển chuỗi tiếng Việt có dấu thành không dấu, chữ thường."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


# Từ điển ánh xạ chuẩn (đã bỏ dấu) -> Danh sách nhãn gốc trong catalog
ARTICLE_TYPE_LEXICON: Dict[str, List[str]] = {
    "Tshirts": ["ao thun", "ao phong", "ao polo", "t-shirt", "tshirt", "polo"],
    "Shirts": ["ao so mi", "so mi", "shirt"],
    "Tops": ["ao kieu", "ao nu", "ao croptop", "crop top", "ao tre vai"],
    "Jackets": ["ao khoac", "jacket", "ao gio", "ao da", "ao vest", "blazer", "ao blazer"],
    "Sweatshirts": ["ao ni", "ao hoodie", "hoodie", "sweatshirt"],
    "Sweaters": ["ao len", "sweater"],
    "Waistcoat": ["ao gile", "gile"],
    "Kurtas": ["ao kurta", "kurta"],
    "Jeans": ["quan jean", "quan bo", "quan jeans", "jeans"],
    "Trousers": ["quan tay", "quan au", "quan dai", "trousers", "quan vai"],
    "Shorts": ["quan short", "quan dui", "quan sooc", "shorts"],
    "Track Pants": ["quan the thao", "quan jogger", "track pants", "quan ni"],
    "Dresses": ["dam", "vay lien", "vay dam", "vay du tiec", "vay", "dress"],
    "Skirts": ["chan vay", "vay ngan", "vay xoe", "skirt"],
    "Sports Shoes": ["giay the thao", "sneaker", "sneakers", "giay chay bo", "sports shoes"],
    "Casual Shoes": ["giay thuong", "giay luoi", "giay vai", "casual shoes", "giay sneaker", "giay bata"],
    "Formal Shoes": ["giay tay", "giay da cong so", "giay cong so", "formal shoes", "giay da nam"],
    "Sandals": ["giay xang dan", "xang dan", "sandal", "sandals", "dep quai hau"],
    "Flip Flops": ["dep xo ngon", "dep tong", "dep kep", "dep le", "flip flops"],
    "Heels": ["giay cao got", "cao got", "heels"],
    "Flats": ["giay bup be", "giay bet", "flats"],
    "Watches": ["dong ho", "watch", "watches"],
    "Handbags": ["tui xach", "tui deo cheo", "handbag", "tui nu", "tui cam tay", "tui deo vai"],
    "Backpacks": ["ba lo", "balo", "cap sach", "backpack"],
    "Laptop Bag": ["tui dung laptop", "cap laptop", "tui laptop"],
    "Wallets": ["vi da", "vi cam tay", "vi tien", "bop", "vi", "wallet"],
    "Belts": ["that lung", "day nit", "belt"],
    "Sunglasses": ["kinh ram", "kinh mat", "mat kinh", "sunglasses"],
    "Caps": ["mu luoi trai", "non ket", "mu ket", "non luoi trai", "mu", "non"],
    "Socks": ["tat", "vo", "socks"],
    "Ties": ["ca vat", "caravat", "tie"],
    "Scarves": ["khan choang", "khan quang", "scarf"],
    "Bra": ["ao nguc", "ao lot", "bra"],
    "Briefs": ["quan lot nam", "quan sip", "briefs"],
    "Innerwear Vests": ["ao ba lo", "ao may o"],
    "Bracelet": ["lac tay", "vong tay", "bracelet"],
    "Pendant": ["mat day chuyen", "day chuyen", "vong co"],
    "Perfume and Body Mist": ["nuoc hoa", "dau thom"],
}

GENDER_LEXICON: Dict[str, List[str]] = {
    "Boys": ["be trai", "tre em nam", "nhoc nam"],
    "Girls": ["be gai", "tre em nu", "nhoc nu"],
    "Unisex": ["unisex", "nam nu", "ca nam va nu"],
    "Men": ["nam", "dan ong", "phai manh", "con trai"],
    "Women": ["nu", "phu nu", "con gai", "phai dep", "chi em"],
}

COLOUR_LEXICON: Dict[str, List[str]] = {
    "Navy Blue": ["xanh navy", "xanh tim than", "xanh den", "navy"],
    "Off White": ["trang nga", "trang sua", "mau kem", "kem"],
    "Coffee Brown": ["nau ca phe", "nau dam"],
    "Teal": ["xanh co vit", "xanh ngoc"],
    "Charcoal": ["than chi", "xam dam", "xam long chuot"],
    "Maroon": ["do do", "do tham", "do ruou", "do man", "maroon"],
    "Gold": ["vang kim", "vang dong", "gold"],
    "White": ["trang", "white"],
    "Black": ["den", "black"],
    "Blue": ["xanh duong", "xanh bien", "xanh da troi", "xanh lam", "blue", "xanh"],
    "Green": ["xanh la", "xanh luc", "xanh reu", "green"],
    "Red": ["do", "red", "do tuoi"],
    "Pink": ["hong", "pink", "hong phan"],
    "Yellow": ["vang", "yellow", "vang tuoi"],
    "Orange": ["cam", "orange"],
    "Purple": ["tim", "purple", "tim than"],
    "Brown": ["nau", "brown", "nau dat"],
    "Grey": ["xam", "ghi", "grey", "gray"],
    "Beige": ["mau be", "be", "beige"],
    "Silver": ["bac", "silver"],
    "Khaki": ["kaki", "khaki"],
}


class VietnameseQueryParser:
    """
    Phân tích truy vấn tiếng Việt (có dấu & không dấu) thành các slot thuộc tính chuẩn.
    Sử dụng chiến lược Longest-Match-First để đảm bảo độ chính xác cao.
    """
    def __init__(self):
        self.article_patterns = self._compile_patterns(ARTICLE_TYPE_LEXICON)
        self.gender_patterns = self._compile_patterns(GENDER_LEXICON)
        self.colour_patterns = self._compile_patterns(COLOUR_LEXICON)

    def _compile_patterns(self, lexicon: Dict[str, List[str]]) -> List[Tuple[re.Pattern, str, int]]:
        patterns = []
        for label, keywords in lexicon.items():
            for kw in keywords:
                kw_norm = remove_diacritics(kw)
                pat = re.compile(rf"(?<!\w){re.escape(kw_norm)}(?!\w)")
                patterns.append((pat, label, len(kw_norm)))
        # Sắp xếp ưu tiên cụm từ dài nhất trước
        patterns.sort(key=lambda x: x[2], reverse=True)
        return patterns

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Trích xuất thuộc tính từ câu truy vấn.
        Trả về dict gồm:
          - articleType: List[str]
          - gender: Optional[str]
          - baseColour: List[str]
        """
        q_norm = remove_diacritics(query)
        masked_q = q_norm

        found_articles: List[str] = []
        for pat, label, _ in self.article_patterns:
            if pat.search(masked_q):
                if label not in found_articles:
                    found_articles.append(label)
                masked_q = pat.sub(" ", masked_q)

        # Kiểm tra giới tính trên chuỗi gốc đã bỏ dấu (hoặc chuỗi chưa mask hết)
        found_gender: Optional[str] = None
        for pat, label, _ in self.gender_patterns:
            if pat.search(q_norm):
                found_gender = label
                break

        found_colours: List[str] = []
        masked_c = q_norm
        for pat, label, _ in self.colour_patterns:
            if pat.search(masked_c):
                if label not in found_colours:
                    found_colours.append(label)
                masked_c = pat.sub(" ", masked_c)

        return {
            "articleType": found_articles,
            "gender": found_gender,
            "baseColour": found_colours,
            "normalized_query": q_norm
        }


def compute_soft_attribute_match(
    parsed_query: Dict[str, Any],
    item_meta: Dict[str, Any],
    gender_policy: str = "hybrid",
    strict_categories: Optional[Set[str]] = None
) -> float:
    """
    Tính tỷ lệ khớp thuộc tính [0.0, 1.0] giữa câu truy vấn đã phân tích và metadata của sản phẩm.
    Tuân thủ gender_gt_policy.json.
    """
    if strict_categories is None:
        strict_categories = {
            "Apparel", "Shirts", "Tshirts", "Jeans", "Dresses", "Skirts",
            "Sweaters", "Sweatshirts", "Jackets", "Bra", "Swimwear",
            "Nightdress", "Innerwear"
        }

    q_articles = parsed_query.get("articleType", [])
    q_gender = parsed_query.get("gender")
    q_colours = parsed_query.get("baseColour", [])

    total_slots = 0
    matched_slots = 0.0

    # 1. Khớp loại sản phẩm
    if q_articles:
        total_slots += 1
        item_art = item_meta.get("articleType", "")
        if item_art in q_articles:
            matched_slots += 1.0
        # Nới lỏng nhẹ cho các nhóm gần nhau (ví dụ Sports Shoes vs Casual Shoes)
        elif (item_art in ["Sports Shoes", "Casual Shoes"]) and any(a in ["Sports Shoes", "Casual Shoes"] for a in q_articles):
            matched_slots += 0.5

    # 2. Khớp giới tính
    if q_gender:
        total_slots += 1
        item_gender = item_meta.get("gender", "")
        item_art = item_meta.get("articleType", "")
        if item_gender == q_gender:
            matched_slots += 1.0
        elif item_gender == "Unisex" and q_gender in ["Men", "Women", "Boys", "Girls"]:
            if gender_policy == "relaxed":
                matched_slots += 1.0
            elif gender_policy == "hybrid" and item_art not in strict_categories:
                matched_slots += 1.0

    # 3. Khớp màu sắc
    if q_colours:
        total_slots += 1
        item_col = item_meta.get("baseColour", "")
        if item_col in q_colours:
            matched_slots += 1.0
        elif item_col == "Navy Blue" and "Blue" in q_colours:
            matched_slots += 0.7
        elif item_col == "Blue" and "Navy Blue" in q_colours:
            matched_slots += 0.5
        elif item_col == "Off White" and "White" in q_colours:
            matched_slots += 0.8

    if total_slots == 0:
        return 0.0
    return matched_slots / total_slots
