"""
Module: src.serve.vn_text
Tiện ích so khớp tiếng Việt dùng chung cho taxonomy.py và parser_v3.py:
- strip_keep_len: bỏ dấu nhưng giữ đúng độ dài chuỗi (offset khớp với chuỗi gốc).
- token_compatible: một từ người dùng gõ có "tương thích" với dạng có dấu của từ khóa không.

Quy tắc tương thích: hai từ phải cùng dạng bỏ dấu, và từ người dùng gõ KHÔNG nằm trong danh sách
từ đồng dạng đã biết của từ khóa (HOMOGRAPH_DENY), vd. 'đậm' không phải 'đầm', 'vì' không phải 'ví',
'đẹp' không phải 'dép'. Ngoài danh sách đó, mọi cách gõ sai dấu/thiếu dấu vẫn được chấp nhận
('ảo' -> 'áo', 'tủi' -> 'túi', 'giầy' -> 'giày', 'quần đui' -> 'quần đùi').
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Set

# Từ khóa (có dấu) -> các từ CÙNG dạng bỏ dấu nhưng khác nghĩa, hay gặp trong truy vấn.
HOMOGRAPH_DENY: Dict[str, Set[str]] = {
    # loại sản phẩm
    "đầm": {"đậm", "đâm", "dầm", "dâm"},
    "ví": {"vì", "vị", "vỉ", "vĩ"},
    "tất": {"tắt", "tạt", "tát", "tật"},
    "vớ": {"vô", "vở", "vỡ", "vợ", "vờ"},
    "nhẫn": {"nhãn", "nhắn", "nhân", "nhận", "nhàn", "nhạn"},
    "mũ": {"mù", "mụ"},
    "dù": {"dư", "dự", "dụ", "dữ", "đủ", "đú", "đù"},
    "giày": {"giấy", "giây"},
    "dép": {"đẹp"},
    "váy": {"vậy", "vây", "vẫy", "vấy", "vầy", "vạy"},   # 'giống vậy' không phải 'giống váy'
    "túi": {"tụi"},                                       # 'tụi mình'
    "quần": {"quân", "quấn", "quận", "quẩn"},             # 'khăn quấn', 'quân đội'
    # màu (dùng khi xác định vị trí màu trong câu)
    "đen": {"đến", "đền", "đèn"},
    "đỏ": {"đồ", "đó", "đọ", "độ", "đổ", "đô"},
    "tím": {"tìm"},
    "hồng": {"hông"},
    "nâu": {"nấu"},
}
# Các dạng màu có trong HOMOGRAPH_DENY: không được nới theo ngữ cảnh (xem span_compatible).
COLOUR_HOMOGRAPH_FORMS: Set[str] = {"đen", "đỏ", "tím", "hồng", "nâu"}
# Đối xứng: nếu 'đậm' không phải 'đầm' thì 'đầm' cũng không phải 'đậm' (dùng cho cụm bảo vệ
# như 'xanh đậm', 'bộ đồ', 'tìm mua' — 'đồ' != 'đỏ', 'tìm' != 'tím').
for _kw, _others in list(HOMOGRAPH_DENY.items()):
    for _o in _others:
        HOMOGRAPH_DENY.setdefault(_o, set()).add(_kw)


def normalize_query(text: str) -> str:
    """Lowercase + NFC + gộp khoảng trắng."""
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", (text or "").lower()).strip())


def strip_keep_len(text: str) -> str:
    """Bỏ dấu từng ký tự, đầu ra luôn cùng độ dài đầu vào (kể cả '…', '½')."""
    out = []
    for ch in text:
        if ch in ("đ", "Đ"):
            out.append("d")
            continue
        base = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
        out.append(base.lower() if len(base) == 1 else (ch.lower()[:1] or " "))
    return "".join(out)


def token_compatible(user_tok: str, kw_tok: str) -> bool:
    if user_tok == kw_tok:
        return True
    if strip_keep_len(user_tok) != strip_keep_len(kw_tok):
        return False
    return user_tok not in HOMOGRAPH_DENY.get(kw_tok, ())


def span_compatible(user_span: str, kw_accented: str, allow_context: bool = False) -> bool:
    """allow_context=True (chỉ dùng cho từ khóa LOẠI SẢN PHẨM): trong cụm từ khóa nhiều chữ, cho phép
    đúng một từ thuộc danh sách đồng dạng nếu các chữ còn lại khớp — ngữ cảnh của cụm đủ rõ
    ('vì da' -> 'ví da', 'đẹp tông' -> 'dép tông'). Cụm bảo vệ và màu luôn so chặt.
    Người gọi phải ưu tiên khớp chặt trước, chỉ dùng khớp nới khi không có từ khóa nào khớp chặt
    (tránh 'áo ... vì da' bị đọc thành 'ví da')."""
    u, k = user_span.split(), kw_accented.split()
    if len(u) != len(k):
        return False
    bad = [i for i, (a, b) in enumerate(zip(u, k)) if not token_compatible(a, b)]
    if not bad:
        return True
    if not allow_context or len(k) < 2 or len(bad) > 1:
        return False
    # Từ màu người dùng gõ không bao giờ bị "nới" thành một chữ của cụm sản phẩm
    # ('túi đeo hồng' không phải 'túi đeo hông', 'áo đỏ ngủ' không phải 'đồ ngủ').
    i = bad[0]
    return u[i] not in COLOUR_HOMOGRAPH_FORMS and k[i] not in COLOUR_HOMOGRAPH_FORMS
