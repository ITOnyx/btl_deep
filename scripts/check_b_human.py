"""
Công cụ kiểm tra Tập B-human (Bước 2c.5 & 6b Việc 2, KHÔNG chạy model).
Đọc eval/val_queries_b_human.json, báo cho từng câu:
- n_relevant trên index/val_gallery/ (cảnh báo n < 5)
- dia_class tự phát hiện (dùng danh sách từ trung tính không dấu chuẩn tiếng Việt, nguồn độc lập)
- CẢNH BÁO nếu dia_class khai báo khác dia_class phát hiện (KHÔNG ghi đè, KHÔNG loại câu)
- trùng/gần trùng với DEV / A / B-synth / C (Jaccard token >= 0.8)
- các token nằm ngoài lexicon v1 (CHỈ BÁO CÁO, không vá lexicon theo đó)
- Kiểm tra số lượng tối thiểu >= 60 câu.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import is_item_relevant, load_gender_gt_policy
from src.serve.parser_v1 import (
    ARTICLE_TYPE_LEXICON,
    COLOUR_LEXICON,
    GENDER_LEXICON,
    remove_diacritics,
)

# Danh sách âm tiết tiếng Việt thanh ngang nguyên âm trơn (không cần dấu trong chính tả chuẩn)
# và từ mượn/thuật ngữ tiếng Anh thông dụng. Nguồn: hệ thống âm tiết tiếng Việt chuẩn / Viet74K.
# KHÔNG trích từ DEV, val hay B-human.
NEUTRAL_NO_DIACRITIC_WORDS: Set[str] = {
    # Từ chức năng / đại từ / động từ / tính từ thanh ngang không dấu trong tiếng Việt chuẩn
    "ai", "anh", "ba", "ban", "bao", "bay", "be", "bi", "bia", "bien", "bo", "bon", "bong",
    "bu", "bua", "ca", "cac", "cam", "can", "cao", "cat", "cau", "cay", "cha", "chan",
    "chao", "chau", "che", "chi", "chia", "chieu", "chim", "chin", "cho", "choi", "chon",
    "chu", "chua", "chung", "co", "con", "cong", "cu", "cung", "da", "dai", "dam", "dan",
    "dang", "dao", "dau", "day", "de", "dem", "di", "do", "don", "dong", "du", "dua",
    "dung", "duoi", "em", "eo", "ga", "gai", "gan", "gang", "giao", "giay", "gio", "giu",
    "go", "gon", "gong", "ha", "hai", "ham", "han", "hang", "hao", "hat", "hay", "he",
    "hen", "heo", "hien", "hieu", "hinh", "hoa", "hoc", "hoi", "hom", "hong", "hu", "hua",
    "hung", "huong", "im", "in", "ke", "kem", "ken", "keo", "kep", "khai", "kham", "khan",
    "khang", "khe", "khen", "khi", "kho", "khoa", "khoan", "khoe", "khong", "khu", "khung",
    "ki", "kia", "kiem", "kien", "kieu", "kim", "kinh", "la", "lai", "lam", "lan", "lang",
    "lao", "lau", "le", "len", "leo", "li", "lien", "linh", "lo", "loa", "loai", "lon",
    "long", "lu", "lua", "lui", "lung", "ly", "ma", "mai", "man", "mang", "mat", "may",
    "me", "men", "meo", "mi", "mien", "minh", "mo", "moi", "mon", "mong", "mu", "mua",
    "mui", "mung", "muoi", "na", "nai", "nam", "nan", "nang", "nao", "nau", "nay", "ne",
    "nem", "nen", "neo", "neu", "nga", "ngai", "ngam", "ngan", "ngang", "ngao", "ngay",
    "nghe", "nghi", "nghia", "ngo", "ngon", "ngu", "nguy", "nguyen", "nha", "nhac", "nhai",
    "nham", "nhan", "nhang", "nhao", "nhat", "nhau", "nhe", "nhi", "nhieu", "nhin", "nho",
    "nhom", "nhon", "nhu", "nhung", "ni", "ninh", "no", "noi", "nom", "non", "nong", "nu",
    "nua", "nui", "nuoi", "o", "oi", "om", "on", "ong", "pa", "pi", "pin", "po", "qua",
    "quai", "quan", "quang", "quay", "que", "quen", "qui", "quy", "ra", "rai", "ran", "rang",
    "rao", "rau", "re", "ren", "reo", "ri", "rieng", "ro", "roi", "rong", "ru", "rung",
    "sa", "sai", "sam", "san", "sang", "sao", "sau", "say", "se", "sen", "si", "sieu",
    "sinh", "so", "soi", "som", "son", "song", "su", "sua", "sui", "sum", "sung", "suon",
    "suong", "suy", "ta", "tai", "tam", "tan", "tang", "tao", "tau", "tay", "te", "ten",
    "tha", "thai", "tham", "than", "thang", "thao", "thay", "the", "them", "theo", "thi",
    "thien", "thieu", "thin", "tho", "thoa", "thoi", "thom", "thon", "thong", "thu", "thua",
    "thuan", "thung", "thuong", "thuy", "ti", "tia", "tien", "tieu", "tim", "tin", "tinh",
    "to", "toa", "toan", "toi", "tom", "ton", "tong", "tra", "trai", "tram", "tran", "trang",
    "trao", "trau", "tre", "tren", "treo", "tri", "trieu", "trinh", "tro", "troi", "tron",
    "trong", "tru", "trua", "truc", "trung", "truong", "truy", "tu", "tua", "tui", "tung",
    "tuoi", "tuong", "tuy", "ty", "u", "ua", "ui", "um", "ung", "uy", "va", "vai", "van",
    "vang", "vao", "vay", "ve", "ven", "vi", "vien", "vinh", "vo", "voi", "vom", "von",
    "vong", "vu", "vua", "vui", "vung", "vuon", "vuong", "xa", "xai", "xam", "xan", "xang",
    "xao", "xay", "xe", "xem", "xen", "xi", "xin", "xinh", "xo", "xoa", "xoai", "xom",
    "xong", "xu", "xua", "xui", "xum", "xung", "xuong", "xuyen", "y", "yen", "yeu",
    # Thực thể thời trang / tiếng Việt không dấu chuẩn
    "thun", "ghi", "kaki", "balo", "caro",
    # Từ mượn tiếng Anh phổ biến trong TMĐT thời trang
    "tshirt", "tshirts", "shirt", "shirts", "top", "tops", "jean", "jeans", "short", "shorts",
    "skirt", "skirts", "dress", "dresses", "jacket", "jackets", "sweater", "sweaters",
    "sweatshirt", "sweatshirts", "hoodie", "blazer", "blazers", "polo", "boxer", "boxers",
    "kurta", "kurtas", "sari", "saree", "sarees", "legging", "leggings", "jogger", "joggers",
    "sneaker", "sneakers", "sandal", "sandals", "slip", "flats", "heels", "boots", "boot",
    "watch", "watches", "wallet", "wallets", "clutch", "clutches", "handbag", "handbags",
    "backpack", "backpacks", "laptop", "ipad", "belt", "belts", "cap", "caps", "tie", "ties",
    "sock", "socks", "scarf", "scarves", "bra", "unisex", "navy", "beige", "steel", "silver",
    "gold", "maroon", "teal", "charcoal", "olive", "khaki", "denim", "cotton", "form",
    "style", "basic", "vintage", "oversize", "croptop", "gym", "aerobic", "tennis", "golf",
    "set", "combo", "full", "size", "cm", "mm",
}
# Lưu ý: loại bỏ khỏi tập trung tính những âm tiết mà trong ngữ cảnh thời trang tiếng Việt
# gần như LUÔN cần dấu thanh/ngữ nghĩa (nếu xuất hiện không dấu thì chính là gõ thiếu dấu):
_REQUIRE_DIACRITIC_IN_FASHION = {
    "ao", "quan", "giay", "dep", "tui", "kinh", "dong", "ho", "mu", "non", "vi", "bop",
    "tat", "vo", "dam", "vay", "mau", "den", "trang", "do", "tim", "nau", "hong", "vang",
    "xam", "bac", "nu", "trai", "gai", "the", "thao", "cong", "so", "di", "hoc", "choi",
    "lam", "mac", "nha", "ngu", "tiec", "bien", "chay", "bo", "da", "bong", "got", "bet",
    "cao", "thap", "rong", "ngan", "dai", "rong", "ong", "suong", "xoe", "om", "sat",
    "tron", "co", "tay", "vai", "lung", "mat", "day", "khoa", "nho", "lon", "moi", "cu",
    "dep", "xinh", "sang", "chanh", "nhe", "ben", "re", "xin", "am", "mat", "muon", "can",
    "tim", "hinh", "in", "ke", "soc", "theu", "dinh", "chong", "nuoc", "so", "mi", "lua",
    "ni", "vai", "du", "lich", "ngoai", "hang", "ngay", "phut", "nam", "gioi", "be",
}
NEUTRAL_NO_DIACRITIC_WORDS = NEUTRAL_NO_DIACRITIC_WORDS - _REQUIRE_DIACRITIC_IN_FASHION
# Giữ lại các từ thực sự KHÔNG có dấu trong tiếng Việt chuẩn (như mua, cho, nam, thun, xanh, cam, kem, ghi, len, ...)
NEUTRAL_NO_DIACRITIC_WORDS.update({
    "mua", "cho", "nam", "thun", "xanh", "cam", "kem", "ghi", "len", "kaki", "balo", "caro",
    "minh", "xinh", "xin", "form", "tay", "chan", "vai", "eo", "hai", "ba", "sau",
    "mi", "da", "cao", "trai", "be", "thao",
})


def tokenize_norm(text: str) -> Set[str]:
    norm = remove_diacritics(text)
    return {tok for tok in re.findall(r"\w+", norm) if tok}


def detect_dia_class_auto(text: str) -> str:
    """
    Quy tắc Bước 6b Việc 2:
    - 'none': nếu không có ký tự dấu nào trong toàn bộ câu.
    - 'full': nếu mọi âm tiết/token chữ cái không thuộc danh sách trung tính đều có dấu.
    - 'partial': ngược lại (trộn có dấu và từ thiếu dấu).
    """
    raw_tokens = re.findall(r"[^\s]+", text.strip())
    alpha_tokens = [re.sub(r"^\W+|\W+$", "", t) for t in raw_tokens]
    alpha_tokens = [t for t in alpha_tokens if any(ch.isalpha() for ch in t)]
    if not alpha_tokens:
        return "none"

    has_acc = [remove_diacritics(t) != t.lower() for t in alpha_tokens]
    if not any(has_acc):
        return "none"

    for tok, accented in zip(alpha_tokens, has_acc):
        if accented:
            continue
        low = tok.lower()
        if low.isdigit():
            continue
        if low not in NEUTRAL_NO_DIACRITIC_WORDS:
            return "partial"
    return "full"


def build_v1_lexicon_tokens() -> Set[str]:
    tokens: Set[str] = set()
    for lex in (ARTICLE_TYPE_LEXICON, GENDER_LEXICON, COLOUR_LEXICON):
        for kw_list in lex.values():
            for kw in kw_list:
                tokens.update(tokenize_norm(kw))
    return tokens


def jaccard_sim(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def check_b_human_queries(
    b_human_path: pathlib.Path,
    val_id_map: List[Dict[str, Any]],
    strict_categories: Set[str],
    ref_datasets: Dict[str, List[Tuple[str, Set[str]]]],
    min_queries: int = 60,
    min_relevant: int = 5,
    jaccard_threshold: float = 0.8,
) -> Dict[str, Any]:
    queries = json.loads(b_human_path.read_text(encoding="utf-8"))
    v1_tokens = build_v1_lexicon_tokens()

    rows: List[Dict[str, Any]] = []
    warnings_low_rel = 0
    warnings_dup = 0
    warnings_dia_mismatch = 0

    for idx, q in enumerate(queries):
        qid = q.get("id", f"row_{idx}")
        text = q.get("query", "")
        spec = q.get("spec", {"articleType": [], "gender": None, "baseColour": []})

        n_rel = sum(1 for m in val_id_map if is_item_relevant(spec, m, "hybrid", strict_categories))
        if n_rel < min_relevant:
            warnings_low_rel += 1

        dia_auto = detect_dia_class_auto(text)
        dia_declared = q.get("dia_class")
        dia_mismatch = (dia_declared is not None) and (dia_declared != dia_auto)
        if dia_mismatch:
            warnings_dia_mismatch += 1

        q_toks = tokenize_norm(text)

        near_dups: List[Dict[str, Any]] = []
        for ds_name, ref_items in ref_datasets.items():
            for ref_id, ref_toks in ref_items:
                sim = jaccard_sim(q_toks, ref_toks)
                if sim >= jaccard_threshold:
                    near_dups.append({"dataset": ds_name, "id": ref_id, "jaccard": round(sim, 4)})
        if near_dups:
            warnings_dup += 1

        oov_tokens = sorted(q_toks - v1_tokens)
        rows.append({
            "id": qid,
            "query": text,
            "n_relevant": n_rel,
            "low_relevant_warning": n_rel < min_relevant,
            "dia_class_declared": dia_declared,
            "dia_class_auto": dia_auto,
            "dia_class_mismatch_warning": dia_mismatch,
            "near_duplicates": near_dups,
            "tokens_outside_v1_lexicon": oov_tokens,
        })

    return {
        "file": str(b_human_path),
        "n_queries": len(queries),
        "meets_min_queries": len(queries) >= min_queries,
        "min_queries_required": min_queries,
        "warnings_low_relevant_lt5": warnings_low_rel,
        "warnings_near_duplicate_ge08": warnings_dup,
        "warnings_dia_class_mismatch": warnings_dia_mismatch,
        "rows": rows,
    }


def load_reference_datasets(root: pathlib.Path) -> Dict[str, List[Tuple[str, Set[str]]]]:
    refs: Dict[str, List[Tuple[str, Set[str]]]] = {}
    for name, rel_path, key in [
        ("DEV", "eval/handwritten_queries.json", None),
        ("A", "eval/val_queries_a_synthetic.json", None),
        ("B-synth", "eval/val_queries_b_synth.json", None),
        ("C", "eval/val_queries_c_error_injection.json", "queries"),
    ]:
        p = root / rel_path
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data[key] if key else data
        refs[name] = [(x.get("id", str(i)), tokenize_norm(x.get("query", ""))) for i, x in enumerate(items)]
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra B-human (Bước 2c.5 & 6b, không chạy model).")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(PROJECT_ROOT / "eval/val_queries_b_human.json"),
        help="Đường dẫn tới file val_queries_b_human.json",
    )
    parser.add_argument("--out", default=None, help="Ghi báo cáo JSON ra file nếu muốn")
    args = parser.parse_args()

    b_human_path = pathlib.Path(args.path)
    val_id_map = json.loads((PROJECT_ROOT / "index/val_gallery/id_map.json").read_text(encoding="utf-8"))
    strict_cats = set(
        load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")["policies"]["hybrid"]["strict_categories"]
    )
    refs = load_reference_datasets(PROJECT_ROOT)

    report = check_b_human_queries(b_human_path, val_id_map, strict_cats, refs)

    print(f"File: {report['file']}")
    print(f"Số câu: {report['n_queries']} (tối thiểu {report['min_queries_required']} -> {'ĐẠT' if report['meets_min_queries'] else 'CHƯA ĐẠT'})")
    print(f"Cảnh báo n_relevant < 5: {report['warnings_low_relevant_lt5']}")
    print(f"Cảnh báo trùng/gần trùng (Jaccard >= 0.8): {report['warnings_near_duplicate_ge08']}")
    print(f"Cảnh báo lệch dia_class (chỉ cảnh báo, không loại): {report['warnings_dia_class_mismatch']}")
    for r in report["rows"]:
        flags = []
        if r["low_relevant_warning"]:
            flags.append(f"LOW_REL({r['n_relevant']})")
        if r["near_duplicates"]:
            flags.append(f"DUP({len(r['near_duplicates'])})")
        if r["dia_class_mismatch_warning"]:
            flags.append(f"DIA_WARN({r['dia_class_declared']}!={r['dia_class_auto']})")
        flag_str = f" [{' '.join(flags)}]" if flags else ""
        print(f"  {r['id']}: n_rel={r['n_relevant']}, dia={r['dia_class_auto']}, oov={r['tokens_outside_v1_lexicon']}{flag_str}")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if report["meets_min_queries"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
