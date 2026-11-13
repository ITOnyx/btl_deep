"""
Module: src.serve.parser_v2 (PARSER_VERSION = "v2.1")
Bộ phân tích truy vấn tiếng Việt phiên bản 2.1 (Bước 3b, Pha 5b — sửa sau audit P1/P2).
Tách biệt hoàn toàn khỏi src/serve/parser_v1.py (parser_v1 giữ nguyên đóng băng).

Lưu ý minh bạch (Prompt 8C):
- parser_v2.1 được chỉnh sau khi audit trên Tập A (do đó Tập A không còn "chưa thấy").
- Chỉ thực hiện 2 chỉnh sửa cấu trúc & gỡ rò rỉ theo Prompt 8A/8B:
  (A) Ưu tiên cụm Unisex bao trùm ("nam nữ", "nữ nam", "nam và nữ", "nữ và nam", ...)
      trước khi khớp Men/Women ngắn hơn.
  (B) Gỡ 18 mục từ commit ec87034 không truy được nguồn trực tiếp trước B-human.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from src.serve.parser_v1 import compute_soft_attribute_match, remove_diacritics

PARSER_VERSION = "v2.1"


def _has_diacritics(text: str) -> bool:
    return remove_diacritics(text) != re.sub(r"\s+", " ", text.lower()).strip()


def _strip_diacritics_keep_len(text: str) -> str:
    """Bỏ dấu tiếng Việt giữ nguyên độ dài & vị trí khoảng trắng để khớp offset."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# Các cụm cố định dễ gây dương tính giả ở dạng bỏ dấu (từ homograph_table_2c1c.md)
# Cần mask trước khi trích màu sắc / giới tính đơn âm.
PROTECTED_NON_COLOUR_PHRASES: List[str] = [
    "tìm mua", "tim mua",
    "cần tìm", "can tim",
    "đồ ngủ", "do ngu", "bộ đồ ngủ", "bo do ngu",
    "đựng đồ", "dung do", "bộ đồ", "bo do",
    "cầm tay", "cam tay",
    "búp bê", "bup be",
    "trái tim", "trai tim",
    "thời trang", "thoi trang",
    "trang phục", "trang phuc",
    "trang điểm", "trang diem",
    "đeo hông", "deo hong",
    "đậm màu", "dam mau",
    "xanh đậm", "xanh dam",
    "hàng năm", "hang nam",
    "đi đến", "di den",
    "da thật", "da bò", "da bo", "dây da", "day da",
    "đính đá", "dinh da", "đá bóng", "da bong",
]

# Từ khóa màu đơn âm thuộc bảng homograph 2c.1c (không tự kích hoạt khi bỏ dấu)
HOMOGRAPH_COLOUR_TOKENS: Set[str] = {
    "tim", "do", "cam", "be", "den", "trang", "hong", "vang", "bac", "xam", "nau", "kem", "ghi",
}

# Bảng ánh xạ có dấu + không dấu chuẩn, truy nguyên từ catalog_rebuild.parquet
ARTICLE_TYPE_LEXICON_V2: Dict[str, List[str]] = {
    "Tshirts": ["áo thun", "áo phông", "áo polo", "t-shirt", "tshirt", "polo"],
    "Shirts": ["áo sơ mi", "sơ mi", "shirt"],
    "Tops": ["áo kiểu", "áo croptop", "crop top", "áo trễ vai", "áo cánh", "áo điệu"],
    "Jackets": ["áo khoác", "áo gió", "áo da", "áo vest", "jacket"],
    "Blazers": ["áo blazer", "blazer", "blazers"],
    "Rain Jacket": ["rain jacket"],
    "Sweatshirts": ["áo nỉ", "áo hoodie", "hoodie", "sweatshirt"],
    "Sweaters": ["áo len", "sweater"],
    "Waistcoat": ["áo gile", "gile"],
    "Kurtas": ["áo kurta", "kurta"],
    "Kurtis": ["áo kurti", "kurti"],
    "Kurta Sets": ["kurta set", "kurta sets"],
    "Tunics": ["áo tunic", "tunic", "tunics"],
    "Sarees": ["sari", "saree", "sarees"],
    "Dupatta": ["khăn dupatta", "dupatta"],
    "Jeans": ["quần jean", "quần bò", "quần jeans", "jeans"],
    "Trousers": ["quần tây", "quần âu", "quần dài", "quần vải", "quần kaki", "trousers"],
    "Shorts": ["quần short", "quần đùi", "quần sooc", "shorts"],
    "Capris": ["capri", "capris"],
    "Track Pants": ["quần thể thao", "quần jogger", "track pants", "quần nỉ"],
    "Leggings": ["quần legging", "legging", "leggings"],
    "Tights": ["tights"],
    "Dresses": ["đầm", "váy liền", "váy đầm", "váy dự tiệc", "váy", "dress"],
    "Skirts": ["chân váy", "váy ngắn", "váy xòe", "skirt"],
    "Jumpsuit": ["áo liền quần", "jumpsuit"],
    "Nightdress": ["váy ngủ", "nightdress"],
    "Night suits": ["bộ đồ ngủ", "đồ ngủ", "night suits"],
    "Sports Shoes": ["giày thể thao", "giày chạy bộ", "giày đá bóng", "giày tập gym", "sports shoes"],
    "Casual Shoes": ["giày thường", "giày lười", "giày vải", "giày sneaker", "sneaker", "sneakers", "giày bata", "casual shoes"],
    "Formal Shoes": ["giày tây", "giày da công sở", "giày công sở", "formal shoes"],
    "Sandals": ["giày xăng đan", "xăng đan", "sandal", "sandals", "dép quai hậu"],
    "Sports Sandals": ["sports sandals"],
    "Flip Flops": ["dép xỏ ngón", "dép tông", "dép kẹp", "dép lê", "flip flops"],
    "Heels": ["giày cao gót", "cao gót", "guốc cao gót", "heels"],
    "Flats": ["giày búp bê", "giày bệt", "giày đế bằng", "flats"],
    "Booties": ["giày bốt", "bốt cổ thấp", "booties"],
    "Watches": ["đồng hồ đeo tay", "đồng hồ", "watch", "watches"],
    "Handbags": ["túi xách tay", "túi xách", "túi đeo vai", "túi công sở", "túi nữ", "handbag"],
    "Messenger Bag": ["túi đeo chéo", "messenger bag"],
    "Duffel Bag": ["túi trống", "duffel bag"],
    "Waist Pouch": ["túi bao tử", "túi đeo hông", "waist pouch"],
    "Backpacks": ["ba lô", "balo", "cặp sách", "backpack"],
    "Rucksacks": ["balo du lịch", "rucksack"],
    "Laptop Bag": ["túi đựng laptop", "cặp laptop", "túi laptop", "balo laptop"],
    "Wallets": ["ví da", "ví cầm tay", "ví tiền", "bóp tiền", "bóp da", "bóp", "ví", "wallet"],
    "Clutches": ["ví dự tiệc", "clutch", "clutches"],
    "Belts": ["thắt lưng", "dây nịt", "dây lưng", "belt"],
    "Sunglasses": ["kính râm", "kính mát", "mắt kính", "sunglasses"],
    "Caps": ["mũ lưỡi trai", "nón kết", "mũ kết", "nón lưỡi trai", "mũ", "nón"],
    "Socks": ["tất", "vớ", "socks"],
    "Ties": ["cà vạt", "caravat", "tie"],
    "Scarves": ["khăn choàng", "khăn quàng", "scarf"],
    "Mufflers": ["khăn quàng cổ", "muffler"],
    "Bra": ["áo ngực", "áo lót", "bra"],
    "Briefs": ["quần lót nam", "quần sịp", "briefs"],
    "Boxers": ["quần lót boxer", "quần boxer", "boxer"],
    "Trunk": ["quần lót trunk", "trunk"],
    "Innerwear Vests": ["áo ba lỗ", "áo may ô"],
    "Earrings": ["bông tai", "hoa tai", "khuyên tai", "earrings"],
    "Bracelet": ["lắc tay", "vòng tay", "bracelet"],
    "Necklace and Chains": ["dây chuyền", "vòng cổ"],
    "Pendant": ["mặt dây chuyền"],
    "Perfume and Body Mist": ["nước hoa", "dầu thơm"],
}

GENDER_LEXICON_V2: Dict[str, List[str]] = {
    "Boys": ["bé trai", "trẻ em nam", "nhóc nam"],
    "Girls": ["bé gái", "trẻ em nữ", "nhóc nữ"],
    "Unisex": ["unisex", "nam nữ", "nữ nam", "nam và nữ", "nữ và nam", "cả nam và nữ"],
    "Men": ["nam giới", "của nam", "cho nam", "đàn ông", "phái mạnh", "con trai", "nam"],
    "Women": ["nữ giới", "của nữ", "cho nữ", "phụ nữ", "con gái", "phái đẹp", "chị em", "nữ"],
}

COLOUR_LEXICON_V2: Dict[str, List[str]] = {
    "Navy Blue": ["xanh navy", "xanh tím than", "xanh đen", "xanh than", "navy"],
    "Off White": ["trắng ngà", "trắng sữa", "màu kem", "kem"],
    "Coffee Brown": ["nâu cà phê", "nâu đậm"],
    "Teal": ["xanh cổ vịt", "xanh ngọc"],
    "Turquoise Blue": ["turquoise"],
    "Olive": ["xanh ô liu", "olive"],
    "Steel": ["màu thép", "thép không gỉ", "steel"],
    "Charcoal": ["than chì", "xám đậm", "xám lông chuột", "lông chuột"],
    "Grey Melange": ["xám tiêu", "grey melange"],
    "Maroon": ["đỏ đô", "đỏ thẫm", "đỏ rượu", "đỏ mận", "maroon"],
    "Burgundy": ["burgundy"],
    "Gold": ["vàng kim", "vàng đồng", "vàng hồng", "gold"],
    "Lavender": ["lavender"],
    "White": ["trắng tinh", "trắng", "white"],
    "Black": ["đen tuyền", "đen", "black"],
    "Blue": ["xanh dương", "xanh biển", "xanh da trời", "xanh lam", "xanh đậm", "blue"],
    "Green": ["xanh lá", "xanh lục", "xanh rêu", "green"],
    "Red": ["đỏ tươi", "đỏ", "red"],
    "Pink": ["hồng phấn", "hồng", "pink"],
    "Yellow": ["vàng tươi", "vàng", "yellow"],
    "Orange": ["màu cam", "cam", "orange"],
    "Purple": ["màu tím", "tím", "purple"],
    "Brown": ["nâu đất", "nâu bò", "nâu", "brown"],
    "Grey": ["xám ghi", "xám", "ghi", "grey", "gray"],
    "Beige": ["màu kem be", "màu be", "be", "beige"],
    "Silver": ["bạc sáng", "ánh bạc", "bạc", "silver"],
    "Khaki": ["kaki", "khaki"],
}


def _compile_dual_patterns(
    lexicon: Dict[str, List[str]],
) -> Tuple[List[Tuple[re.Pattern, str, str, int]], List[Tuple[re.Pattern, str, str, int]]]:
    """Biên dịch cả pattern có dấu (lowercase) và pattern bỏ dấu, sắp theo độ dài giảm dần."""
    acc_pats: List[Tuple[re.Pattern, str, str, int]] = []
    unacc_pats: List[Tuple[re.Pattern, str, str, int]] = []
    seen_acc: Set[Tuple[str, str]] = set()
    seen_unacc: Set[Tuple[str, str]] = set()

    for label, keywords in lexicon.items():
        for kw in keywords:
            kw_low = re.sub(r"\s+", " ", kw.lower()).strip()
            kw_norm = remove_diacritics(kw_low)
            if (label, kw_low) not in seen_acc:
                seen_acc.add((label, kw_low))
                acc_pats.append((re.compile(rf"(?<!\w){re.escape(kw_low)}(?!\w)"), label, kw_low, len(kw_low)))
            if (label, kw_norm) not in seen_unacc:
                seen_unacc.add((label, kw_norm))
                unacc_pats.append((re.compile(rf"(?<!\w){re.escape(kw_norm)}(?!\w)"), label, kw_norm, len(kw_norm)))

    acc_pats.sort(key=lambda x: x[3], reverse=True)
    unacc_pats.sort(key=lambda x: x[3], reverse=True)
    return acc_pats, unacc_pats


class VietnameseQueryParserV2:
    """
    Parser v2 (Bước 3b):
    - Ưu tiên khớp có dấu khi truy vấn có dấu.
    - Bảo vệ cụm cố định từ bảng homograph 2c.1c trước khi trích màu/giới tính.
    - Chống dương tính giả màu đơn âm bỏ dấu bằng tín hiệu ngữ cảnh.
    """

    def __init__(self) -> None:
        self.article_acc, self.article_unacc = _compile_dual_patterns(ARTICLE_TYPE_LEXICON_V2)
        self.gender_acc, self.gender_unacc = _compile_dual_patterns(GENDER_LEXICON_V2)
        self.gender_unisex_acc = [p for p in self.gender_acc if p[1] == "Unisex"]
        self.gender_other_acc = [p for p in self.gender_acc if p[1] != "Unisex"]
        self.gender_unisex_unacc = [p for p in self.gender_unacc if p[1] == "Unisex"]
        self.gender_other_unacc = [p for p in self.gender_unacc if p[1] != "Unisex"]
        self.colour_acc, self.colour_unacc = _compile_dual_patterns(COLOUR_LEXICON_V2)

        # Biên dịch cụm cố định bảo vệ (longest-match-first)
        prot_acc, prot_unacc = [], []
        for ph in PROTECTED_NON_COLOUR_PHRASES:
            ph_low = re.sub(r"\s+", " ", ph.lower()).strip()
            ph_norm = remove_diacritics(ph_low)
            prot_acc.append((re.compile(rf"(?<!\w){re.escape(ph_low)}(?!\w)"), len(ph_low)))
            prot_unacc.append((re.compile(rf"(?<!\w){re.escape(ph_norm)}(?!\w)"), len(ph_norm)))
        prot_acc.sort(key=lambda x: x[1], reverse=True)
        prot_unacc.sort(key=lambda x: x[1], reverse=True)
        self.protected_acc = [p[0] for p in prot_acc]
        self.protected_unacc = [p[0] for p in prot_unacc]

    @staticmethod
    def _span_all_unaccented(orig_text: str, start: int, end: int) -> bool:
        span = orig_text[start:end]
        return remove_diacritics(span) == span.lower()

    def parse(self, query: str) -> Dict[str, Any]:
        q_low = unicodedata.normalize("NFC", re.sub(r"\s+", " ", (query or "").lower()).strip())
        q_norm = remove_diacritics(q_low)
        has_dia = _has_diacritics(q_low)

        # 1. Trích articleType (ưu tiên có dấu nếu query có dấu)
        masked_acc = q_low
        found_articles: List[str] = []
        last_noun_end = -1

        if has_dia:
            for pat, label, _, _ in self.article_acc:
                for m in pat.finditer(masked_acc):
                    if label not in found_articles:
                        found_articles.append(label)
                    last_noun_end = max(last_noun_end, m.end())
                masked_acc = pat.sub(lambda m: " " * (m.end() - m.start()), masked_acc)

            # Fallback cho phần thực sự không dấu (vd: trộn dấu 'giày sneaker')
            masked_norm = _strip_diacritics_keep_len(masked_acc)
            for pat, label, _, _ in self.article_unacc:
                for m in pat.finditer(masked_norm):
                    if self._span_all_unaccented(masked_acc, m.start(), m.end()):
                        if label not in found_articles:
                            found_articles.append(label)
                        last_noun_end = max(last_noun_end, m.end())
                        masked_acc = masked_acc[:m.start()] + (" " * (m.end() - m.start())) + masked_acc[m.end():]
                        masked_norm = _strip_diacritics_keep_len(masked_acc)
        else:
            masked_norm = q_norm
            # Mask cụm phi-sản phẩm như 'đậm màu'/'dam mau' trước khi khớp 'đầm'/'dam'
            prot_art_mask = masked_norm
            for pat in self.protected_unacc:
                if "dam mau" in pat.pattern:
                    prot_art_mask = pat.sub(lambda m: " " * (m.end() - m.start()), prot_art_mask)
            for pat, label, _, _ in self.article_unacc:
                for m in pat.finditer(prot_art_mask):
                    if label not in found_articles:
                        found_articles.append(label)
                    last_noun_end = max(last_noun_end, m.end())
                prot_art_mask = pat.sub(lambda m: " " * (m.end() - m.start()), prot_art_mask)

        # 2. Trích gender (ưu tiên cụm Unisex bao trùm trước khi khớp Men/Women ngắn hơn)
        found_gender: Optional[str] = None
        gender_spans: List[Tuple[int, int]] = []
        if has_dia:
            # Mask 'hàng năm'/'hang nam' trước
            g_search = q_low
            for pat in self.protected_acc:
                if "hang nam" in pat.pattern or "hàng năm" in pat.pattern:
                    g_search = pat.sub(lambda m: " " * (m.end() - m.start()), g_search)
            g_norm = _strip_diacritics_keep_len(g_search)
            for pat, label, _, _ in self.gender_unisex_acc:
                m = pat.search(g_search)
                if m:
                    found_gender = label
                    gender_spans.append((m.start(), m.end()))
                    last_noun_end = max(last_noun_end, m.end())
                    break
            if found_gender is None:
                for pat, label, _, _ in self.gender_unisex_unacc:
                    m = pat.search(g_norm)
                    if m and self._span_all_unaccented(g_search, m.start(), m.end()):
                        found_gender = label
                        gender_spans.append((m.start(), m.end()))
                        last_noun_end = max(last_noun_end, m.end())
                        break
            if found_gender is None:
                for pat, label, _, _ in self.gender_other_acc:
                    m = pat.search(g_search)
                    if m:
                        found_gender = label
                        gender_spans.append((m.start(), m.end()))
                        last_noun_end = max(last_noun_end, m.end())
                        break
            if found_gender is None:
                for pat, label, _, _ in self.gender_other_unacc:
                    m = pat.search(g_norm)
                    if m and self._span_all_unaccented(g_search, m.start(), m.end()):
                        found_gender = label
                        gender_spans.append((m.start(), m.end()))
                        last_noun_end = max(last_noun_end, m.end())
                        break
        else:
            g_norm = q_norm
            for pat in self.protected_unacc:
                if "hang nam" in pat.pattern:
                    g_norm = pat.sub(lambda m: " " * (m.end() - m.start()), g_norm)
            for pat, label, _, _ in self.gender_unisex_unacc:
                m = pat.search(g_norm)
                if m:
                    found_gender = label
                    gender_spans.append((m.start(), m.end()))
                    last_noun_end = max(last_noun_end, m.end())
                    break
            if found_gender is None:
                for pat, label, _, _ in self.gender_other_unacc:
                    m = pat.search(g_norm)
                    if m:
                        found_gender = label
                        gender_spans.append((m.start(), m.end()))
                        last_noun_end = max(last_noun_end, m.end())
                        break

        # 3. Trích baseColour (mask cụm bảo vệ + span gender trước!)
        found_colours: List[str] = []
        c_acc = q_low
        for s, e in gender_spans:
            c_acc = c_acc[:s] + (" " * (e - s)) + c_acc[e:]
        for pat in self.protected_acc:
            c_acc = pat.sub(lambda m: " " * (m.end() - m.start()), c_acc)

        if has_dia:
            for pat, label, kw_low, _ in self.colour_acc:
                # Chỉ khớp trực tiếp nếu từ khóa có dấu hoặc không phải đơn âm homograph
                kw_is_homograph_unacc = (kw_low in HOMOGRAPH_COLOUR_TOKENS)
                if kw_is_homograph_unacc:
                    continue
                for m in pat.finditer(c_acc):
                    if label not in found_colours:
                        found_colours.append(label)
                c_acc = pat.sub(lambda m: " " * (m.end() - m.start()), c_acc)

        # Kiểm tra phần không dấu còn lại (cho query không dấu hoặc query trộn)
        c_norm = _strip_diacritics_keep_len(c_acc)
        for pat in self.protected_unacc:
            c_norm = pat.sub(lambda m: " " * (m.end() - m.start()), c_norm)

        for pat, label, kw_norm, _ in self.colour_unacc:
            for m in pat.finditer(c_norm):
                # Nếu query có dấu, bắt buộc span phải thực sự không dấu (không bao giờ lấy 'tìm' -> 'tim')
                if has_dia and not self._span_all_unaccented(c_acc, m.start(), m.end()):
                    continue
                if kw_norm in HOMOGRAPH_COLOUR_TOKENS:
                    # Tín hiệu ngữ cảnh bắt buộc cho đơn âm homograph bỏ dấu:
                    # (a) đứng ngay sau 'mau ' / 'màu '
                    prefix = c_norm[max(0, m.start() - 5):m.start()]
                    has_mau_prefix = prefix.endswith("mau ")
                    # (b) đứng sau cụm danh từ loại sản phẩm / giới tính đã tìm thấy
                    after_noun = (last_noun_end >= 0) and (m.start() >= last_noun_end)
                    if not (has_mau_prefix or after_noun):
                        continue
                if label not in found_colours:
                    found_colours.append(label)
                c_norm = c_norm[:m.start()] + (" " * (m.end() - m.start())) + c_norm[m.end():]

        return {
            "articleType": found_articles,
            "gender": found_gender,
            "baseColour": found_colours,
            "normalized_query": q_norm,
        }
