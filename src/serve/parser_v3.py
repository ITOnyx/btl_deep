"""
Module: src.serve.parser_v3
Parser v3 = Parser v2.1 (đóng băng, không sửa) + bổ sung từ vựng loại sản phẩm (inference-only, Luật 1).
Dùng cho cfg_v5_11 (DEV SET: từ vựng bổ sung được rút ra sau khi xem lỗi trên tập dev).
"""
import re
import unicodedata
from typing import Dict, List, Set, Tuple

from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.parser_v2 import (
    ARTICLE_TYPE_LEXICON_V2,
    HOMOGRAPH_COLOUR_TOKENS,
    PROTECTED_NON_COLOUR_PHRASES,
    remove_diacritics,
    VietnameseQueryParserV2,
    _compile_dual_patterns,
    _has_diacritics,
)
from src.serve.vn_text import normalize_query, span_compatible, strip_keep_len

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


# "xanh đậm" bị v2 che (PROTECTED_NON_COLOUR_PHRASES) nên không ra màu; v3 coi là Blue.
_DARK_BLUE_RE = re.compile(r"(?<!\w)xanh dam(?!\w)")
# "tất cả" (= all) không phải "tất" (Socks)
_TAT_CA_RE = re.compile(r"(?<!\w)tat ca(?!\w)")


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
        # (label, kw bỏ dấu) -> các dạng có dấu, để kiểm tra span khi truy vấn có dấu
        self._kw_accented: Dict[Tuple[str, str], Set[str]] = {}
        for label, kws in lex.items():
            for kw in kws:
                kw_low = unicodedata.normalize("NFC", re.sub(r"\s+", " ", kw.lower()).strip())
                self._kw_accented.setdefault((label, remove_diacritics(kw_low)), set()).add(kw_low)
        # Cụm bảo vệ của v2 theo dạng bỏ dấu -> các dạng có dấu (nếu có), để chỉ che khi tương thích dấu
        prot: Dict[str, Set[str]] = {}
        for ph in PROTECTED_NON_COLOUR_PHRASES:
            ph_low = unicodedata.normalize("NFC", re.sub(r"\s+", " ", ph.lower()).strip())
            prot.setdefault(remove_diacritics(ph_low), set()).add(ph_low)
        self._protected_acc = [
            (re.compile(rf"(?<!\w){re.escape(norm)}(?!\w)"), {f for f in forms if f != norm} or forms)
            for norm, forms in prot.items()
        ]
        # (label, màu bỏ dấu) -> các dạng có dấu, để xác định vị trí màu trong câu
        self._colour_accented: Dict[Tuple[str, str], Set[str]] = {}
        for _, label, kw_low, _ in self.colour_acc:
            self._colour_accented.setdefault((label, remove_diacritics(kw_low)), set()).add(kw_low)
        self._fuzzy_article_kws = []
        for label, kws in lex.items():
            for kw in kws:
                kwn = remove_diacritics(kw)
                if len(kwn) >= 7 and kwn not in HOMOGRAPH_COLOUR_TOKENS:
                    self._fuzzy_article_kws.append((kwn, label))

    def _span_matches_kw(self, orig_span: str, label: str, kw_norm: str) -> bool:
        return any(span_compatible(orig_span, kw_acc) for kw_acc in self._kw_accented.get((label, kw_norm), ()))

    @staticmethod
    def _mask_phrase(q_low: str, q_strip: str, phrase_re, phrase_acc: str):
        """Che các span khớp phrase (tương thích dấu). Trả về (chuỗi đã che, vị trí đầu tiên hoặc -1)."""
        first = -1
        for m in phrase_re.finditer(q_strip):
            if span_compatible(q_low[m.start():m.end()], phrase_acc):
                first = m.start() if first < 0 else first
                q_strip = q_strip[:m.start()] + " " * (m.end() - m.start()) + q_strip[m.end():]
        return q_strip, first

    def _has_label_kw(self, q_low: str, masked: str, label: str) -> bool:
        """Còn bằng chứng cho `label` trong chuỗi đã che không: khớp từ khóa dài trước (như v2),
        từng pattern một, có kiểm tra dấu. Vd. 'chan vay' là Skirts nên 'vay' trong đó không tính
        là Dresses; 'váy đậm' không khớp 'váy đầm' nhưng vẫn khớp 'váy'."""
        for pat, lab, kw_norm, _ in self.article_unacc:
            for m in pat.finditer(masked):
                if self._span_matches_kw(q_low[m.start():m.end()], lab, kw_norm):
                    if lab == label:
                        return True
                    masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
        return False

    def _colour_pos(self, q_low: str, label: str) -> int:
        """Vị trí đầu tiên của màu `label` trong câu: bỏ qua cụm bảo vệ của v2 ('bộ đồ', 'đi đến'...)
        và các span không tương thích dấu ('đồ' không phải 'đỏ', 'đến' không phải 'đen')."""
        q_strip = strip_keep_len(q_low)
        for ppat, forms in self._protected_acc:
            for m in ppat.finditer(q_strip):
                if any(span_compatible(q_low[m.start():m.end()], f) for f in forms):
                    q_strip = q_strip[:m.start()] + " " * (m.end() - m.start()) + q_strip[m.end():]
        best = len(q_low) + 1
        for pat, lab, kw_norm, _ in self.colour_unacc:
            if lab != label:
                continue
            accs = self._colour_accented.get((lab, kw_norm), {kw_norm})
            for m in pat.finditer(q_strip):
                if m.start() < best and any(span_compatible(q_low[m.start():m.end()], k) for k in accs):
                    best = m.start()
        return best

    def _post_fix(self, q_low: str, res: Dict) -> str:
        """Sửa các cụm v2 hiểu sai; trả về chuỗi bỏ dấu đã che các cụm đó (dùng cho bước khớp sau).
        - 'xanh đậm'/'xanh dam' -> màu Blue (v2 che cụm này nên không nhận màu), và bỏ 'Dresses'
          nếu chữ 'dam' duy nhất trong truy vấn nằm trong cụm này.
        - 'tất cả' -> bỏ 'Socks' nếu chữ 'tất' duy nhất nằm trong cụm này."""
        q_strip = strip_keep_len(q_low)
        masked, blue_at = self._mask_phrase(q_low, q_strip, _DARK_BLUE_RE, "xanh đậm")
        if blue_at >= 0:
            if "Blue" not in res["baseColour"]:
                # Sắp lại toàn bộ danh sách màu theo vị trí xuất hiện (màu đầu tiên là màu đích khi xếp hạng)
                pos = {c: self._colour_pos(q_low, c) for c in res["baseColour"]}
                pos["Blue"] = min(pos.get("Blue", blue_at), blue_at)
                res["baseColour"] = sorted(res["baseColour"] + ["Blue"], key=lambda c: pos[c])
            if "Dresses" in res["articleType"] and not self._has_label_kw(q_low, masked, "Dresses"):
                res["articleType"].remove("Dresses")
        masked, tat_at = self._mask_phrase(q_low, masked, _TAT_CA_RE, "tất cả")
        if tat_at >= 0 and "Socks" in res["articleType"] and not self._has_label_kw(q_low, masked, "Socks"):
            res["articleType"].remove("Socks")
        return masked

    def parse(self, query: str):
        res = VietnameseQueryParserV2.parse(self, query)
        q_low = normalize_query(query)
        masked = self._post_fix(q_low, res)
        if not res["articleType"]:
            # Truy vấn trộn có dấu/không dấu ("quan đùi", "giày chú rể" gõ thiếu dấu): khớp lại trên chuỗi bỏ dấu.
            # Khi truy vấn có dấu, chỉ nhận span tương thích dấu thanh với từ khóa:
            # tránh 'đậm' -> 'đầm' (Dresses), 'vì' -> 'ví' (Wallets), vẫn nhận 'giầy', 'khóac'.
            has_dia = _has_diacritics(q_low)
            for pat, label, kw_norm, _ in self.article_unacc:
                for m in pat.finditer(masked):
                    if has_dia and not self._span_matches_kw(q_low[m.start():m.end()], label, kw_norm):
                        continue
                    if label not in res["articleType"]:
                        res["articleType"].append(label)
                    masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
        if self.fuzzy:
            # Không để khớp mờ đọc "xanh dam" thành "vay dam" (Dresses) hay "tat ca" thành loại khác
            q_norm = _TAT_CA_RE.sub(" ", _DARK_BLUE_RE.sub(" ", res["normalized_query"]))
            if not res["articleType"]:
                res["articleType"] = self._fuzzy_match_phrases(q_norm, self._fuzzy_article_kws)
            if not res["baseColour"]:
                res["baseColour"] = self._fuzzy_match_phrases(q_norm, self._fuzzy_colour_kws)
        return res
