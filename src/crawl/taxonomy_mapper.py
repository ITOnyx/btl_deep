"""
Bộ ánh xạ nhãn xác định (deterministic) từ metadata trang nguồn (breadcrumb, collection, tiêu đề, tags)
sang articleType, gender, baseColour chuẩn theo catalog hiện có (vi_map.json & src/serve/taxonomy.json).
TUYỆT ĐỐI KHÔNG dùng mô hình SigLIP hay VLM để gán nhãn.
Trả về label_source chuẩn ('breadcrumb' hoặc 'title') và label_status ('ok' hoặc 'needs_review').
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _norm(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"\s+", " ", s)


def _strip_accents(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFD", str(text).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d")
    return re.sub(r"\s+", " ", s).strip()


OUT_OF_SCOPE_KEYWORDS = [
    "nước hoa",
    "xịt thơm",
    "sữa tắm",
    "dầu gội",
    "mỹ phẩm",
    "chăm sóc da",
    "trang điểm",
    "juno beauty",
    "xịt khóa nền",
    "son môi",
    "son kem",
    "son thỏi",
    "lip mousse",
    "lip tint",
    "setting spray",
    "kem chống nắng",
    "kem dưỡng",
    "tẩy trang",
    "phấn nước",
    "mascara",
    "kẻ mắt",
    "chì kẻ mày",
    "má hồng",
    "bông mút",
    "cọ trang điểm",
    "sữa rửa mặt",
    "nước tẩy trang",
    "mặt nạ",
    "serum",
    "tinh chất",
    "dưỡng ẩm",
    "dưỡng môi",
    "dưỡng tóc",
    "xịt dưỡng tóc",
    "chăm sóc cơ thể",
    "bình nước",
    "ly giữ nhiệt",
    "bình giữ nhiệt",
    "cốc",
    "gối",
    "chăn",
    "ga giường",
    "khăn tắm",
    "thảm",
    "voucher",
    "phiếu quà tặng",
    "thẻ quà tặng",
    "hộp quà",
    "túi giấy",
    "móc khóa",
    "keychain",
    "vợt cầu lông",
    "vợt tennis",
    "vợt pickleball",
    "quả bóng đá",
    "trái bóng",
    "bóng rổ",
    "tạ tay",
    "dây nhảy",
    "kính áp tròng",
    "nước ngâm lens",
    "nhỏ mắt",
]

CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    # --- Footwear ---
    ("Heels", ["giày cao gót", "cao gót", "guốc", "slingback", "gót nhọn", "gót vuông", "gót trụ", "giày mũi nhọn gót"]),
    ("Flats", ["giày búp bê", "búp bê", "giày lười nữ", "giày mọi nữ", "mary jane", "loafer nữ", "flat"]),
    ("Booties", ["giày bốt", "giày boot", "boots", "boot cổ", "chelsea boot", "ankle boot"]),
    ("Formal Shoes", ["giày tây", "giày da nam", "giày công sở", "oxford", "derby", "giày lười da", "loafer nam"]),
    ("Sports Shoes", ["giày thể thao", "giày chạy bộ", "giày bóng đá", "giày cầu lông", "giày tập gym", "hunter", "running", "sneaker thể thao", "giày đá bóng"]),
    ("Sandals", ["giày sandal", "sandal", "xăng đan", "dép quai hậu", "dép sandal"]),
    ("Flip Flops", ["dép xỏ ngón", "dép kẹp", "dép lào", "dép đi biển", "dép đúc", "dép lê", "dép quai ngang", "dép nam", "dép nữ", "dép trẻ em", "dép bé", "dép đi trong nhà", "dép sục", "dép eva", "dép da"]),
    ("Casual Shoes", ["giày sneaker", "sneaker", "giày thường", "giày vải", "giày casual", "giày mọi", "giày lười", "giày thời trang", "giày tập đi"]),
    # --- Bags & Wallets ---
    ("Laptop Bag", ["túi đựng laptop", "túi chống sốc", "cặp laptop", "túi laptop"]),
    ("Backpacks", ["balo", "ba lô", "backpack"]),
    ("Clutches", ["ví cầm tay", "clutch", "túi cầm tay", "túi dự tiệc"]),
    ("Wallets", ["ví da", "ví tiền", "ví nam", "ví nữ", "ví ngắn", "ví dài", "ví đựng thẻ", "bóp da", "bóp tiền", "wallet", "card holder", "ví ngang", "ví đứng"]),
    ("Waist Pouch", ["túi bao tử", "túi đeo hông", "túi đeo bụng", "belt bag", "fanny pack"]),
    ("Duffel Bag", ["túi trống", "túi du lịch", "túi thể thao", "duffel"]),
    ("Messenger Bag", ["túi đeo chéo", "túi messenger", "túi đưa thư", "crossbody", "túi đeo ngực"]),
    ("Trolley Bag", ["vali", "va li"]),
    ("Travel Accessory", ["phụ kiện du lịch", "gối cổ", "bao trùm vali", "tag hành lý", "khóa số vali", "khóa vali", "túi bọc vali"]),
    ("Umbrellas", ["ô dù", "dù cầm tay", "dù che mưa"]),
    ("Handbags", ["túi xách", "túi tote", "túi đeo vai", "túi kẹp nách", "túi bucket", "túi hobo", "túi công sở", "túi thời trang", "handbag"]),
    # --- Accessories ---
    ("Sunglasses", ["kính mát", "kính râm", "mắt kính", "kính thời trang", "sunglasses", "gọng kính", "kính cận", "eyewear"]),
    ("Watches", ["đồng hồ", "watch", "g-shock", "baby-g", "edifice", "seiko", "citizen", "orient", "tissot", "casio", "daniel wellington", "fossil", "olym pianus", "bentley", "carnival", "obligado"]),
    ("Belts", ["thắt lưng", "dây nịt", "dây lưng", "belt"]),
    ("Socks", ["tất", "vớ", "socks"]),
    ("Caps", ["mũ lưỡi trai", "nón lưỡi trai", "nón kết", "mũ kết", "nón snapback", "nón bóng chày", "cap", "mũ thể thao"]),
    ("Hat", ["mũ bucket", "nón bucket", "nón rộng vành", "mũ rộng vành", "nón len", "mũ len", "mũ nồi", "nón beret", "mũ cói", "nón cói", "mũ nón", "mũ bé", "nón bé"]),
    ("Scarves", ["khăn quàng", "khăn choàng", "khăn lụa", "khăn cổ", "khăn len", "scarf"]),
    ("Ties", ["cà vạt", "nơ cổ", "caravat", "tie"]),
    ("Gloves", ["găng tay", "bao tay"]),
    ("Earrings", ["bông tai", "khuyên tai", "hoa tai"]),
    ("Necklace and Chains", ["dây chuyền", "vòng cổ"]),
    ("Bracelet", ["vòng tay", "lắc tay", "charm"]),
    ("Ring", ["nhẫn"]),
    ("Shoe Accessories", ["lót giày", "dây giày", "vệ sinh giày", "phụ kiện giày"]),
    # --- Dresses & Sets ---
    ("Jumpsuit", ["jumpsuit", "đồ bay", "áo liền quần", "set jumpsuit"]),
    ("Rompers", ["romper", "yếm short", "body liền", "bodysuit"]),
    ("Dresses", ["đầm", "váy liền", "váy đầm", "váy maxi", "váy midi", "váy xòe", "váy ôm", "váy suông", "váy dự tiệc", "váy công sở", "váy bé gái", "áo dài", "dress"]),
    ("Skirts", ["chân váy", "váy ngắn", "váy chữ a", "váy xếp ly", "váy bút chì", "skirt"]),
    ("Suits", ["bộ vest", "bộ suit", "com lê", "comple", "suit nam"]),
    ("Tracksuits", ["bộ thể thao", "đồ bộ thể thao", "tracksuit"]),
    ("Clothing Set", ["set bộ", "bộ trang phục", "đồ bộ", "set áo", "bộ quần áo", "bộ thun", "bộ lửng", "bộ cộc tay", "bộ dài tay"]),
    # --- Underwear / Swimwear / Loungewear ---
    ("Swimwear", ["đồ bơi", "áo bơi", "quần bơi", "bikini", "swimwear"]),
    ("Bra", ["áo ngực", "áo bra", "bra thể thao", "áo lót nữ"]),
    ("Boxers", ["quần lót boxer", "quần boxer", "boxer brief", "sịp đùi", "quần lót đùi", "boxer"]),
    ("Briefs", ["quần lót tam giác", "quần sịp", "sịp tam giác", "quần lót nam", "quần lót nữ", "brief", "quần lót"]),
    ("Trunk", ["quần lót trunk", "trunk"]),
    ("Innerwear Vests", ["áo ba lỗ mặc trong", "áo lót ba lỗ", "áo may ô", "áo singlet mặc trong"]),
    ("Night suits", ["đồ ngủ", "bộ ngủ", "pijama", "pyjama", "đồ mặc nhà"]),
    ("Lounge Pants", ["quần mặc nhà", "quần ngủ"]),
    # --- Tops & Outerwear ---
    ("Blazers", ["áo blazer", "blazer", "áo vest", "vest nữ", "vest nam"]),
    ("Waistcoat", ["áo gile", "gile", "ghi lê", "waistcoat"]),
    ("Sweatshirts", ["áo hoodie", "hoodie", "áo nỉ", "sweatshirt", "áo sweater nỉ"]),
    ("Sweaters", ["áo len", "áo dệt kim", "áo cardigan", "cardigan", "sweater", "áo cổ lọ len"]),
    ("Rain Jacket", ["áo mưa"]),
    ("Jackets", ["áo khoác", "jacket", "bomber", "varsity", "áo gió", "áo phao", "măng tô", "trench coat", "áo chống nắng", "coat", "parka"]),
    ("Tshirts", ["áo thun", "áo phông", "áo polo", "polo", "áo t-shirt", "t-shirt", "tshirt", "áo ba lỗ", "tank top", "singlet", "áo thể thao", "áo dài tay thun", "áo ngắn tay"]),
    ("Shirts", ["áo sơ mi", "sơ mi", "shirt"]),
    ("Tops", ["áo kiểu", "áo croptop", "crop top", "áo hai dây", "áo ống", "áo trễ vai", "áo peplum", "áo blouse", "blouse", "áo sát nách nữ", "áo nữ"]),
    # --- Bottoms ---
    ("Jeans", ["quần jean", "quần bò", "quần denim", "jeans"]),
    ("Leggings", ["quần legging", "legging"]),
    ("Shorts", ["quần short", "quần sooc", "quần sọt", "quần đùi", "quần lửng", "shorts"]),
    ("Track Pants", ["quần jogger", "jogger", "quần thể thao", "quần nỉ", "track pants", "quần chạy bộ dài"]),
    ("Trousers", ["quần tây", "quần âu", "quần kaki", "quần chino", "quần vải", "quần ống rộng", "quần ống suông", "quần dài", "quần công sở", "quần baggy", "quần cargo", "quần túi hộp", "trousers"]),
]


class TaxonomyMapper:
    def __init__(
        self,
        vi_map_path: str | Path = "data/_quarantine/legacy_rebuild_20261111/eval/eval/text2img_test/vi_map.json",
        taxonomy_path: str | Path = "src/serve/taxonomy.json",
    ):
        with open(vi_map_path, "r", encoding="utf-8") as f:
            self.vi_map: Dict[str, Any] = json.load(f)
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            self.taxonomy: Dict[str, Any] = json.load(f)

        self.valid_articles = set(self.vi_map.get("articleType", {}).keys())
        self.article_to_tax = self.taxonomy.get("article_to_taxonomy", {})
        self.excluded_domains = {"Chăm sóc cá nhân", "Đồ thể thao", "Khác"}

        self.vi_to_article: Dict[str, str] = {}
        for eng, vi in self.vi_map.get("articleType", {}).items():
            self.vi_to_article[_norm(vi)] = eng
            self.vi_to_article[_norm(eng)] = eng

        self.color_to_base: Dict[str, str] = {}
        for eng, vi in self.vi_map.get("baseColour", {}).items():
            if eng == "Unknown":
                continue
            self.color_to_base[_norm(eng)] = eng
            self.color_to_base[_strip_accents(eng)] = eng
            if vi:
                self.color_to_base[_norm(vi)] = eng
                self.color_to_base[_strip_accents(vi)] = eng

        extra_colors = {
            "xanh den": "Navy Blue",
            "xanh reu": "Olive",
            "reu": "Olive",
            "xanh ngoc": "Turquoise Blue",
            "xanh bien": "Sea Green",
            "xanh coban": "Blue",
            "xanh da troi": "Blue",
            "xanh duong": "Blue",
            "xanh nhat": "Blue",
            "xanh dam": "Navy Blue",
            "xanh la": "Green",
            "xanh": "Blue",
            "bo": "Blue",
            "kem": "Cream",
            "be": "Beige",
            "da": "Skin",
            "ghi": "Grey",
            "xam nhat": "Grey",
            "xam dam": "Charcoal",
            "xam tieu": "Grey Melange",
            "xam": "Grey",
            "do do": "Maroon",
            "do ruou": "Burgundy",
            "do": "Red",
            "hong phan": "Pink",
            "hong dat": "Rose",
            "hong": "Pink",
            "vang bo": "Mustard",
            "vang dong": "Gold",
            "vang kim": "Gold",
            "vang": "Yellow",
            "nau dat": "Brown",
            "nau tay": "Tan",
            "nau ca phe": "Coffee Brown",
            "nau": "Brown",
            "trang sua": "Off White",
            "trang kem": "Off White",
            "trang": "White",
            "den": "Black",
            "tim": "Purple",
            "cam": "Orange",
            "bac": "Silver",
            "gray": "Grey",
            "navy": "Navy Blue",
            "camel": "Tan",
            "ivory": "Off White",
            "wine": "Burgundy",
            "champagne": "Gold",
            "gun metal": "Charcoal",
            "charcoal": "Charcoal",
            "light blue": "Blue",
            "dark blue": "Navy Blue",
        }
        for k, v in extra_colors.items():
            self.color_to_base.setdefault(k, v)

    def map_color(self, color_raw: str, title_raw: str = "") -> str:
        for candidate in (color_raw, title_raw):
            if not candidate:
                continue
            c_n = _norm(candidate)
            c_a = _strip_accents(candidate)
            if c_n in self.color_to_base:
                return self.color_to_base[c_n]
            if c_a in self.color_to_base:
                return self.color_to_base[c_a]
            for k in sorted(self.color_to_base.keys(), key=len, reverse=True):
                if len(k) >= 2 and re.search(rf"(^|[\s/,\-_()])({re.escape(k)})([\s/,\-_()]|$)", c_a):
                    return self.color_to_base[k]
        return ""

    def is_out_of_scope(self, category_raw: str, title_raw: str) -> bool:
        combined = _norm(f"{category_raw} {title_raw}")
        return any(kw in combined for kw in OUT_OF_SCOPE_KEYWORDS)

    def _is_in_fashion_scope(self, article_type: str) -> bool:
        tax = self.article_to_tax.get(article_type)
        if tax and tax[0] in self.excluded_domains:
            return False
        return article_type in self.valid_articles

    def map_category(self, category_raw: str, title_raw: str = "") -> Tuple[str, str]:
        """
        Trả về (articleType, label_source) với label_source thuộc {'breadcrumb', 'title', 'unmapped'}.
        """
        cat_n = _norm(category_raw)
        title_n = _norm(title_raw)

        if cat_n:
            parts = [p.strip() for p in re.split(r"[>/|,;]", cat_n) if p.strip()]
            for part in reversed(parts):
                if part in self.vi_to_article:
                    art = self.vi_to_article[part]
                    if self._is_in_fashion_scope(art):
                        return art, "breadcrumb"
            for art, keywords in CATEGORY_RULES:
                if any(kw in cat_n for kw in keywords):
                    if self._is_in_fashion_scope(art):
                        return art, "breadcrumb"
            cat_a = _strip_accents(cat_n)
            for art, keywords in CATEGORY_RULES:
                if any(_strip_accents(kw) in cat_a for kw in keywords if len(_strip_accents(kw)) >= 4):
                    if self._is_in_fashion_scope(art):
                        return art, "breadcrumb"

        if title_n:
            for art, keywords in CATEGORY_RULES:
                if any(kw in title_n for kw in keywords):
                    if self._is_in_fashion_scope(art):
                        return art, "title"
            title_a = _strip_accents(title_n)
            for art, keywords in CATEGORY_RULES:
                if any(_strip_accents(kw) in title_a for kw in keywords if len(_strip_accents(kw)) >= 4):
                    if self._is_in_fashion_scope(art):
                        return art, "title"

        return "", "unmapped"

    def map_gender(
        self,
        category_raw: str,
        title_raw: str,
        tags_str: str = "",
        site_default_gender: str = "",
    ) -> str:
        def _detect(txt: str) -> str:
            t = _norm(txt)
            if not t:
                return ""
            if any(k in t for k in ["bé trai", "be trai", "boys", "nam nhí", "trai"]):
                return "Boys"
            if any(k in t for k in ["bé gái", "be gai", "girls", "nữ nhí", "gái"]):
                return "Girls"
            if any(k in t for k in ["unisex", "nam nữ", "nam/nữ", "cả nam và nữ", "đôi"]):
                return "Unisex"
            has_men = bool(re.search(r"(^|[\s>/,|(\-_])(nam|men|mens|man|gent|gents)([\s>/,|)\-_]|$)", t))
            has_women = bool(re.search(r"(^|[\s>/,|(\-_])(nữ|nu|women|womens|woman|ladies|lady|đầm|váy|cao gót|búp bê)([\s>/,|)\-_]|$)", t))
            if has_men and has_women:
                return "Unisex"
            if has_women:
                return "Women"
            if has_men:
                return "Men"
            return ""

        g_primary = _detect(f"{category_raw} {title_raw}")
        if g_primary:
            return g_primary

        g_tags = _detect(tags_str)
        if g_tags:
            return g_tags

        if site_default_gender in ("Men", "Women", "Unisex", "Boys", "Girls"):
            return site_default_gender

        return ""
