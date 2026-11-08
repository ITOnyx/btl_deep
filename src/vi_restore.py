"""
Vietnamese Diacritic Restoration Module (src/vi_restore.py)
Provides rule-based and frequency-based restoration of Vietnamese tones/diacritics
for fashion e-commerce search queries.

Constraint from AGENTS.md / Phase 3 prompt:
- Builds dictionary mapping unaccented words/phrases -> most common accented form.
- Built from vi_vocab and caption vocabulary (prioritizing trigrams/bigrams, then unigrams; unknown words left unchanged).
- Dictionary only stores word/phrase mappings, never product IDs or full sentences.
"""
import os
import re
import json
import pathlib
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Optional

from src.vi_vocab import (
    remove_diacritics,
    ARTICLE_TYPE_MAP,
    COLOUR_MAP,
    GENDER_MAP,
    USAGE_MAP
)

# Canonical unigram corrections for high-frequency fashion keywords
CANONICAL_UNIGRAMS = {
    "nam": "nam",
    "nu": "nữ",
    "den": "đen",
    "trang": "trắng",
    "do": "đỏ",
    "vang": "vàng",
    "tim": "tím",
    "hong": "hồng",
    "nau": "nâu",
    "xam": "xám",
    "bac": "bạc",
    "kem": "kem",
    "xanh": "xanh",
    "cam": "cam",
    "be": "be",
    "ao": "áo",
    "quan": "quần",
    "giay": "giày",
    "dep": "dép",
    "tui": "túi",
    "vi": "ví",
    "vay": "váy",
    "dam": "đầm",
    "mu": "mũ",
    "non": "nón",
    "kinh": "kính",
    "mau": "màu",
    "day": "dây",
    "nit": "nịt",
    "bop": "bóp",
    "bo": "bò",
    "phong": "phông",
    "ngan": "ngắn",
    "dai": "dài",
    "tay": "tay",
    "co": "cổ",
    "tron": "tròn",
    "be": "bẻ",
    "rong": "rộng",
    "om": "ôm",
    "loe": "loe",
    "rach": "rách",
    "khoa": "khóa",
    "keo": "kéo",
    "vai": "vải",
    "da": "da",
    "mat": "mặt",
    "vuong": "vuông",
    "the": "thể",
    "thao": "thao",
    "cong": "công",
    "so": "sở",
    "di": "đi",
    "lam": "làm",
    "choi": "chơi",
    "hoc": "học",
    "bien": "biển",
    "nha": "nhà",
    "ngu": "ngủ",
    "chay": "chạy",
    "chong": "chống",
    "nuoc": "nước",
    "soc": "sốc",
    "mua": "mùa",
    "he": "hè",
    "dong": "đông",
    "thu": "thu",
    "xuan": "xuân",
    "du": "dự",
    "tiec": "tiệc",
    "dao": "dạo",
    "pho": "phố",
    "chua": "chữ",
    "xoe": "xòe",
    "suong": "suông",
    "quai": "quai",
    "manh": "mảnh",
    "hau": "hậu",
    "kep": "kẹp",
    "xo": "xỏ",
    "ngon": "ngón",
    "mui": "mũi",
    "nhon": "nhọn",
    "bet": "bệt",
    "luoi": "lười",
    "that": "thắt",
    "lung": "lưng",
    "bong": "bông",
    "tai": "tai",
    "hoa": "hoa",
    "chuyen": "chuyền",
    "nhan": "nhẫn",
    "lac": "lắc",
    "vong": "vòng",
    "khoi": "khỏi",
    "don": "đơn",
    "gian": "giản"
}

# Domain phrases
DOMAIN_PHRASES = [
    # Quần áo
    "áo thun", "áo sơ mi", "áo khoác", "áo len", "áo nỉ", "áo kiểu", "áo phông", "áo gile", "áo vest",
    "áo ba lỗ", "áo hai dây", "áo sát nách", "áo hoodie", "áo blazer", "áo croptop",
    "quần jean", "quần bò", "quần tây", "quần âu", "quần kaki", "quần short", "quần đùi", "quần jogger",
    "quần ống rộng", "quần ống đứng", "quần ống loe", "quần legging", "quần lót", "quần bơi",
    "chân váy", "chân váy chữ a", "chân váy xòe", "chân váy bút chì", "chân váy ngắn", "chân váy dài",
    "váy đầm", "váy liền", "đầm dự tiệc", "đầm suông", "đầm xòe", "đầm ôm", "bộ đồ ngủ", "đồ bay jumpsuit",
    # Giày dép
    "giày thể thao", "giày sneaker", "giày chạy bộ", "giày đá bóng", "giày tập gym", "giày thường ngày",
    "giày tây", "giày công sở", "giày da bóng", "giày lười", "giày búp bê", "giày cao gót", "giày sandal",
    "giày bốt", "dép quai hậu", "dép kẹp", "dép xỏ ngón", "dép tông", "dép quai ngang", "dép quai dán",
    "tất vớ", "vớ cổ ngắn", "vớ cổ cao", "tất thể thao",
    # Túi balo
    "balo laptop", "balo du lịch", "balo học sinh", "balo phượt", "balo thể thao",
    "túi xách", "túi xách tay", "túi đeo chéo", "túi bao tử", "túi đeo hông", "túi trống", "túi canvas",
    "ví da", "ví cầm tay", "bóp tiền", "ví dự tiệc", "ví đựng thẻ", "ví mini",
    # Đồng hồ phụ kiện
    "đồng hồ đeo tay", "đồng hồ kim loại", "đồng hồ dây da", "đồng hồ thông minh", "đồng hồ chống nước",
    "kính râm", "kính mát", "mắt kính", "thắt lưng", "dây nịt", "cà vạt", "khuy măng sét",
    "mũ lưỡi trai", "nón kết", "mũ len", "khăn quàng cổ", "bông tai", "hoa tai", "dây chuyền", "vòng tay",
    # Màu sắc & Thuộc tính
    "màu trắng", "màu đen", "màu đỏ", "màu xanh dương", "màu xanh navy", "màu xanh lá", "màu xanh rêu",
    "màu vàng", "màu tím", "màu hồng", "màu cam", "màu nâu", "màu xám", "màu bạc", "màu be", "màu kem",
    "xanh dương", "xanh navy", "xanh lá", "xanh ngọc", "đỏ đô", "đỏ rượu", "vàng mù tạt", "xám tiêu",
    "da bò", "da thật", "da mềm", "thép không gỉ", "cổ tròn", "cổ bẻ", "cổ tim", "dài tay", "ngắn tay",
    "cộc tay", "dáng ôm", "form rộng", "rách gối", "mũi nhọn", "đế bệt", "quai dán", "quai mảnh",
    # Ngữ cảnh
    "đi làm", "công sở", "thể thao", "mùa hè", "mùa đông", "dự tiệc", "dạo phố", "chạy bộ",
    "đi chơi", "tập gym", "đi biển", "chống nước", "chống sốc", "thoáng mát", "ấm áp"
]


class DiacriticRestorer:
    def __init__(self, captions_path: Optional[str] = "data/checkpoints_siglip-vit/dense_captions_catalog.json"):
        self.unigram_map: Dict[str, str] = dict(CANONICAL_UNIGRAMS)
        self.bigram_map: Dict[str, str] = {}
        self.trigram_map: Dict[str, str] = {}

        self._build_maps(captions_path)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+", text.lower())

    def _build_maps(self, captions_path: Optional[str]):
        corpus = list(DOMAIN_PHRASES)

        # Add vi_vocab phrases
        for m in [ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP, USAGE_MAP]:
            for v in m.values():
                clean_v = re.sub(r"\(.*?\)", "", v).strip()
                corpus.append(clean_v)

        # Add captions if available
        if captions_path and pathlib.Path(captions_path).exists():
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = json.load(f)
            for text in captions.values():
                for p in re.split(r"[,.()\n;:?!]+", text):
                    if p.strip():
                        corpus.append(p.strip())

        uni_counter = defaultdict(Counter)
        bi_counter = defaultdict(Counter)
        tri_counter = defaultdict(Counter)

        for phrase in corpus:
            tokens = self._tokenize(phrase)
            for t in tokens:
                uni_counter[remove_diacritics(t)][t] += 1
            for i in range(len(tokens) - 1):
                bi = " ".join(tokens[i:i+2])
                bi_counter[remove_diacritics(bi)][bi] += 1
            for i in range(len(tokens) - 2):
                tri = " ".join(tokens[i:i+3])
                tri_counter[remove_diacritics(tri)][tri] += 1

        # Populate most frequent mappings
        for norm_tri, c in tri_counter.items():
            self.trigram_map[norm_tri] = c.most_common(1)[0][0]

        for norm_bi, c in bi_counter.items():
            self.bigram_map[norm_bi] = c.most_common(1)[0][0]

        # Clean spurious bigrams/trigrams (e.g., cosmetic phrases like 'nám da', 'tàn nhang')
        for k in list(self.trigram_map.keys()):
            if "nám" in self.trigram_map[k]:
                self.trigram_map[k] = self.trigram_map[k].replace("nám", "nam")
        for k in list(self.bigram_map.keys()):
            if "nám" in self.bigram_map[k]:
                self.bigram_map[k] = self.bigram_map[k].replace("nám", "nam")

    def restore(self, text: str) -> str:
        """
        Restores diacritics using greedy longest-match (trigram -> bigram -> unigram).
        Leaves unrecognized tokens unchanged.
        """
        tokens = self._tokenize(text)
        out = []
        i = 0
        n = len(tokens)

        while i < n:
            # 1. Trigram match
            if i + 3 <= n:
                key3 = " ".join(remove_diacritics(t) for t in tokens[i:i+3])
                if key3 in self.trigram_map:
                    out.append(self.trigram_map[key3])
                    i += 3
                    continue

            # 2. Bigram match
            if i + 2 <= n:
                key2 = " ".join(remove_diacritics(t) for t in tokens[i:i+2])
                if key2 in self.bigram_map:
                    out.append(self.bigram_map[key2])
                    i += 2
                    continue

            # 3. Unigram match
            key1 = remove_diacritics(tokens[i])
            out.append(self.unigram_map.get(key1, tokens[i]))
            i += 1

        return " ".join(out)


# Singleton restorer
_RESTORER = None

def get_restorer() -> DiacriticRestorer:
    global _RESTORER
    if _RESTORER is None:
        _RESTORER = DiacriticRestorer()
    return _RESTORER

def restore_diacritics(text: str) -> str:
    """Public helper to restore diacritics in a query string."""
    return get_restorer().restore(text)
