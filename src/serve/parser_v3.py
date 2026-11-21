"""
Module: src.serve.parser_v3
Parser v3 = Parser v2.1 (đóng băng, không sửa) + bổ sung từ vựng loại sản phẩm (inference-only, Luật 1).
Dùng cho cfg_v5_11 (DEV SET: từ vựng bổ sung được rút ra sau khi xem lỗi trên tập dev).
"""
from typing import Dict, List

from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.parser_v2 import (
    ARTICLE_TYPE_LEXICON_V2,
    HOMOGRAPH_COLOUR_TOKENS,
    remove_diacritics,
    VietnameseQueryParserV2,
    _compile_dual_patterns,
)

PARSER_VERSION = "v3.0-dev"

ARTICLE_TYPE_EXTRA: Dict[str, List[str]] = {
    "Sports Shoes": ["giày thể dục", "giày đá banh"],
    "Tshirts": ["áo cộc tay", "áo cộc"],
    "Formal Shoes": ["giày chú rể"],
    "Trousers": ["quần công sở"],
    "Necklace and Chains": ["dây chuyền"],
    "Briefs": ["quần lót"],
    "Night suits": ["bộ pijama", "pijama", "pyjama"],
    "Flip Flops": ["dép lào"],
    "Heels": ["giày gót nhọn", "gót nhọn"],
    "Casual Shoes": ["giày đi chơi", "giày mọi", "giày sneaker"],
}


# Cụm chuyển nhãn: "ví cầm tay" là Clutches trong catalog (không phải Wallets)
ARTICLE_TYPE_MOVE = {"ví cầm tay": ("Wallets", "Clutches")}


def build_lexicon() -> Dict[str, List[str]]:
    lex = {k: list(v) for k, v in ARTICLE_TYPE_LEXICON_V2.items()}
    for phrase, (src, dst) in ARTICLE_TYPE_MOVE.items():
        if phrase in lex.get(src, []):
            lex[src].remove(phrase)
        lex.setdefault(dst, [])
        if phrase not in lex[dst]:
            lex[dst].append(phrase)
    for k, v in ARTICLE_TYPE_EXTRA.items():
        cur = lex.setdefault(k, [])
        cur.extend(x for x in v if x not in cur)
    return lex


class VietnameseQueryParserV3(VietnameseQueryParserV2Conf):
    """v2.1 + từ vựng bổ sung + khớp mờ (sai 1 ký tự) khi không nhận ra loại/màu (DEV SET)."""

    def __init__(self, fuzzy: bool = True) -> None:
        super().__init__()
        self.fuzzy = fuzzy
        lex = build_lexicon()
        self.article_acc, self.article_unacc = _compile_dual_patterns(lex)
        self._fuzzy_article_kws = []
        for label, kws in lex.items():
            for kw in kws:
                kwn = remove_diacritics(kw)
                if len(kwn) >= 7 and kwn not in HOMOGRAPH_COLOUR_TOKENS:
                    self._fuzzy_article_kws.append((kwn, label))

    def parse(self, query: str):
        res = VietnameseQueryParserV2.parse(self, query)
        if not res["articleType"]:
            # Truy vấn trộn có dấu/không dấu ("quan đùi", "giày chú rể" gõ thiếu dấu): khớp lại trên chuỗi bỏ dấu
            masked = res["normalized_query"]
            for pat, label, _, _ in self.article_unacc:
                if pat.search(masked):
                    if label not in res["articleType"]:
                        res["articleType"].append(label)
                    masked = pat.sub(" ", masked)
        if self.fuzzy:
            q_norm = res["normalized_query"]
            if not res["articleType"]:
                res["articleType"] = self._fuzzy_match_phrases(q_norm, self._fuzzy_article_kws)
            if not res["baseColour"]:
                res["baseColour"] = self._fuzzy_match_phrases(q_norm, self._fuzzy_colour_kws)
        return res
