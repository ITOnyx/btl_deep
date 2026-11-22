"""
Module: src.serve.parser_v2_conf (Pha 5c — I2 & I7)
Bọc ngoài src/serve/parser_v2.py (v2.1 đóng băng, KHÔNG sửa bản gốc):
- I2: Gắn mức tin cậy ('high' | 'low' | 'none') cho từng slot (articleType, gender, baseColour).
- I7: Khớp mờ chịu lỗi gõ (Damerau-Levenshtein <= 1) cho từ khóa >= 5 ký tự khi không khớp chính xác,
      chặn tuyệt đối các từ homograph ngắn.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple

from src.serve.parser_v1 import remove_diacritics
from src.serve.parser_v2 import (
    ARTICLE_TYPE_LEXICON_V2,
    COLOUR_LEXICON_V2,
    HOMOGRAPH_COLOUR_TOKENS,
    VietnameseQueryParserV2,
    _has_diacritics,
)


def _damerau_levenshtein_le1(a: str, b: str) -> bool:
    """Trả về True nếu khoảng cách Damerau-Levenshtein giữa a và b <= 1 (nhanh, không cấp phát ma trận)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) == 2:
            i, j = diffs
            return (j == i + 1) and (a[i] == b[j]) and (a[j] == b[i])
        return False
    # Chèn / xóa 1 ký tự
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    used_skip = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        elif not used_skip:
            used_skip = True
            j += 1
        else:
            return False
    return True


class VietnameseQueryParserV2Conf(VietnameseQueryParserV2):
    """Mở rộng parser_v2.1 với mức tin cậy slot (I2) và khớp mờ chịu lỗi gõ (I7)."""

    def __init__(self) -> None:
        super().__init__()
        # Chỉ lập chỉ mục từ khóa >= 7 ký tự (không dấu) và không thuộc homograph ngắn cho I7
        # (tránh các cặp 5 ký tự dễ nhầm như 'shirt'/'short')
        self._fuzzy_article_kws: List[Tuple[str, str]] = []
        for label, kws in ARTICLE_TYPE_LEXICON_V2.items():
            for kw in kws:
                kwn = remove_diacritics(kw)
                if len(kwn) >= 7 and kwn not in HOMOGRAPH_COLOUR_TOKENS:
                    self._fuzzy_article_kws.append((kwn, label))

        self._fuzzy_colour_kws: List[Tuple[str, str]] = []
        for label, kws in COLOUR_LEXICON_V2.items():
            for kw in kws:
                kwn = remove_diacritics(kw)
                if len(kwn) >= 7 and kwn not in HOMOGRAPH_COLOUR_TOKENS:
                    self._fuzzy_colour_kws.append((kwn, label))

    def parse_with_confidence(self, query: str, fuzzy_typo: bool = False) -> Dict[str, Any]:
        """Phân tích truy vấn, trả về kết quả chuẩn + slot_confidence ('high'|'low'|'none')."""
        res = self.parse(query)
        q_low = unicodedata.normalize("NFC", re.sub(r"\s+", " ", (query or "").lower()).strip())
        q_norm = res["normalized_query"]
        has_dia = _has_diacritics(q_low)

        if fuzzy_typo:
            if not res["articleType"]:
                res["articleType"] = self._fuzzy_match_phrases(q_norm, self._fuzzy_article_kws)
            if not res["baseColour"]:
                res["baseColour"] = self._fuzzy_match_phrases(q_norm, self._fuzzy_colour_kws)

        # Đánh giá độ tin cậy từng slot (I2)
        conf: Dict[str, str] = {"articleType": "none", "gender": "none", "baseColour": "none"}

        if res["articleType"]:
            # High nếu truy vấn có dấu hoặc khớp từ khóa dài (>= 2 từ)
            matched_multi = any(
                len(kw.split()) >= 2 and remove_diacritics(kw) in q_norm
                for lbl in res["articleType"]
                for kw in ARTICLE_TYPE_LEXICON_V2.get(lbl, [])
            )
            conf["articleType"] = "high" if (has_dia or matched_multi) else "low"

        if res["gender"]:
            # High nếu có dấu hoặc Unisex hoặc cụm >= 2 từ ("cho nam", "be trai", ...)
            g_lbl = res["gender"]
            if has_dia or g_lbl == "Unisex" or any(
                kw in q_norm for kw in ("cho nam", "cho nu", "cua nam", "cua nu", "be trai", "be gai", "nam gioi", "nu gioi")
            ):
                conf["gender"] = "high"
            else:
                conf["gender"] = "low"

        if res["baseColour"]:
            # High nếu có dấu hoặc có tiền tố "mau " hoặc từ khóa >= 2 âm tiết
            has_mau_ctx = "mau " in q_norm
            matched_multi_col = any(
                len(kw.split()) >= 2 and remove_diacritics(kw) in q_norm
                for lbl in res["baseColour"]
                for kw in COLOUR_LEXICON_V2.get(lbl, [])
            )
            conf["baseColour"] = "high" if (has_dia or has_mau_ctx or matched_multi_col) else "low"

        res["slot_confidence"] = conf
        return res

    @staticmethod
    def _fuzzy_match_phrases(q_norm: str, kw_list: List[Tuple[str, str]]) -> List[str]:
        tokens = re.findall(r"\w+", q_norm)
        if not tokens:
            return []
        found: List[str] = []
        for kwn, label in kw_list:
            kw_toks = kwn.split()
            n_kw = len(kw_toks)
            if n_kw == 1:
                for tok in tokens:
                    if len(tok) >= 7 and _damerau_levenshtein_le1(tok, kwn):
                        if label not in found:
                            found.append(label)
                        break
            else:
                if len(tokens) >= n_kw:
                    for i in range(len(tokens) - n_kw + 1):
                        cand_toks = tokens[i : i + n_kw]
                        # Chỉ cho phép tối đa 1 token khác nhau, và token khác nhau đó phải dài >= 4
                        diff_idxs = [k for k in range(n_kw) if cand_toks[k] != kw_toks[k]]
                        if len(diff_idxs) == 1:
                            dk = diff_idxs[0]
                            if (
                                len(cand_toks[dk]) >= 4
                                and len(kw_toks[dk]) >= 4
                                and _damerau_levenshtein_le1(cand_toks[dk], kw_toks[dk])
                            ):
                                if label not in found:
                                    found.append(label)
                                break
        return found
