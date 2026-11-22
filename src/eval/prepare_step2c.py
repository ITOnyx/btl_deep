"""
Chuẩn bị Bước 2c (Pha 5b):
- 2c.4: Sửa Tập A (sarees -> sari Ấn Độ / sari an do) và Tập B-synth (kiểm toán nhãn đa nghĩa,
        tính lại dia_class cho typo, thêm >= 30 câu typo Telex/VNI/dính phím/sai thanh, seed=42).
        Ghi changelog từng câu vào reports/phase5b/val_ab_changelog.md.
- 2c.3: Dựng lại Tập C lồng nhau (nested errors, chỉ tiêm lỗi có hiệu lực, thêm hàng 'real_parse_v1').
- 2c.1b: Kiểm toán parser_v1 trên DEV, A, B-synth (tỉ lệ dương tính giả theo slot & token kích hoạt,
         kiểm tra 'tìm mua'/'tim mua' -> Purple).
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.error_injection import inject_nested_parse_errors
from src.eval.metrics import is_item_relevant, load_gender_gt_policy
from src.serve.parser_v1 import VietnameseQueryParser, remove_diacritics


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cluster_id(spec: Dict[str, Any]) -> str:
    ats = "|".join(sorted(spec.get("articleType") or []))
    gd = str(spec.get("gender") or "None")
    cls = "|".join(sorted(spec.get("baseColour") or []))
    return f"{ats}__{gd}__{cls}"


def detect_dia_class(text: str, typo_type: Optional[str] = None) -> str:
    """
    Phân loại dấu tiếng Việt:
    - none: không có token nào có dấu
    - partial: có dấu nhưng trộn từ không dấu hoặc bị lỗi thanh/dấu (wrong_tone)
    - full: có dấu đầy đủ, không thuộc wrong_tone
    """
    tokens = text.split()
    if not tokens:
        return "none"
    has_acc_tokens = [remove_diacritics(t) != t for t in tokens]
    if not any(has_acc_tokens):
        return "none"
    if typo_type == "wrong_tone":
        return "partial"
    unaccented_markers = {
        "mau", "ao", "quan", "giay", "dep", "tui", "kinh", "dong", "ho",
        "nu", "trang", "den", "xanh", "nau", "xam", "hong", "vang", "tim",
        "bac", "cho", "di", "lam", "choi",
    }
    if any(t.lower() in unaccented_markers for t in tokens):
        return "partial"
    return "full"


# Bảng kiểm toán từ đồng nghĩa -> articleType đa trị (2c.4a)
# Đối chiếu với DEV (h048, h066/h067, h075/h076) và ngữ nghĩa tiếng Việt
AMBIGUOUS_SYNONYM_MAP: List[Tuple[str, List[str], str]] = [
    ("giày thể thao nhẹ", ["Casual Shoes", "Sports Shoes"], "Đa nghĩa Casual/Sports Shoes (khớp DEV h048)"),
    ("giay the thao nhe", ["Casual Shoes", "Sports Shoes"], "Đa nghĩa Casual/Sports Shoes (khớp DEV h048)"),
    ("giày sneaker thường", ["Casual Shoes", "Sports Shoes"], "Đa nghĩa Casual/Sports Shoes (khớp DEV h048)"),
    ("giay sneaker thuong", ["Casual Shoes", "Sports Shoes"], "Đa nghĩa Casual/Sports Shoes (khớp DEV h048)"),
    ("giày mọi", ["Flats", "Casual Shoes"], "Đa nghĩa Flats/Casual Shoes (giày lười đế bệt)"),
    ("giay moi", ["Flats", "Casual Shoes"], "Đa nghĩa Flats/Casual Shoes (giày lười đế bệt)"),
    ("giày đi tiệc", ["Heels", "Flats", "Formal Shoes"], "Đa nghĩa Heels/Flats/Formal Shoes"),
    ("giay di tiec", ["Heels", "Flats", "Formal Shoes"], "Đa nghĩa Heels/Flats/Formal Shoes"),
    ("dép quai ngang", ["Sandals", "Flip Flops"], "Đa nghĩa Sandals/Flip Flops (dép lê/quai ngang)"),
    ("dep quai ngang", ["Sandals", "Flip Flops"], "Đa nghĩa Sandals/Flip Flops (dép lê/quai ngang)"),
    ("áo ấm", ["Jackets", "Sweaters", "Sweatshirts"], "Đa nghĩa Jackets/Sweaters/Sweatshirts"),
    ("ao am", ["Jackets", "Sweaters", "Sweatshirts"], "Đa nghĩa Jackets/Sweaters/Sweatshirts"),
    ("ba lô laptop", ["Backpacks", "Laptop Bag"], "Đa nghĩa Backpacks/Laptop Bag (khớp DEV h066/h067)"),
    ("ba lo laptop", ["Backpacks", "Laptop Bag"], "Đa nghĩa Backpacks/Laptop Bag (khớp DEV h066/h067)"),
    ("bóp cầm tay", ["Wallets", "Clutches"], "Đa nghĩa Wallets/Clutches (khớp DEV h075/h076)"),
    ("bop cam tay", ["Wallets", "Clutches"], "Đa nghĩa Wallets/Clutches (khớp DEV h075/h076)"),
    ("váy xòe", ["Skirts", "Dresses"], "Đa nghĩa Skirts/Dresses"),
    ("vay xoe", ["Skirts", "Dresses"], "Đa nghĩa Skirts/Dresses"),
    ("áo sweater nỉ", ["Sweatshirts", "Sweaters"], "Đa nghĩa Sweatshirts/Sweaters"),
    ("ao sweater ni", ["Sweatshirts", "Sweaters"], "Đa nghĩa Sweatshirts/Sweaters"),
]


# Bảng chuyển đổi Telex / VNI / sai dấu thanh có seed cho 2c.4b
TELEX_REPLACEMENTS = [
    ("áo", "aso"), ("quần", "quaanf"), ("giày", "giayf"), ("trắng", "trawngs"),
    ("đen", "dden"), ("xanh", "xanh"), ("vàng", "vangf"), ("hồng", "hoongf"),
    ("nâu", "naau"), ("túi", "tuis"), ("đồng", "ddoongf"), ("kính", "kinhs"),
    ("mũ", "mux"), ("dép", "deps"), ("lưng", "luwng"), ("thun", "thun"),
    ("màu", "mauf"), ("nam", "nam"), ("nữ", "nux"), ("đẹp", "ddepj"),
]

VNI_REPLACEMENTS = [
    ("áo", "ao1"), ("quần", "qua6n2"), ("giày", "giay2"), ("trắng", "tra8ng1"),
    ("đen", "d9en"), ("vàng", "vang2"), ("hồng", "ho6ng2"), ("nâu", "na6u"),
    ("túi", "tui1"), ("đồng", "d9o6ng2"), ("kính", "kinh1"), ("mũ", "mu4"),
    ("dép", "d9ep1"), ("màu", "mau2"), ("nữ", "nu4"), ("đỏ", "d9o3"),
]

WRONG_TONE_REPLACEMENTS = [
    ("trắng", "trẳng"), ("đỏ", "đó"), ("áo", "ảo"), ("giày", "giảy"),
    ("quần", "quẩn"), ("hồng", "hổng"), ("vàng", "vảng"), ("túi", "tủi"),
    ("kính", "kỉnh"), ("dép", "dẹp"), ("nâu", "nẩu"), ("màu", "mảu"),
]


def generate_extra_typo(text: str, typo_mode: str, rng: random.Random) -> Tuple[str, str]:
    """Sinh thêm typo theo 4 kiểu yêu cầu ở 2c.4b: telex_raw, vni_raw, sticky_keys, wrong_tone."""
    if typo_mode == "telex_raw":
        for src, dst in TELEX_REPLACEMENTS:
            if src in text:
                return text.replace(src, dst, 1), "telex_raw"
        # fallback nếu câu không dấu: chèn phím telex vào 1 từ
        toks = text.split()
        for i, t in enumerate(toks):
            if len(t) >= 3:
                toks[i] = t + rng.choice(["s", "f", "r", "x"])
                return " ".join(toks), "telex_raw"
    elif typo_mode == "vni_raw":
        for src, dst in VNI_REPLACEMENTS:
            if src in text:
                return text.replace(src, dst, 1), "vni_raw"
        toks = text.split()
        for i, t in enumerate(toks):
            if len(t) >= 3:
                toks[i] = t + rng.choice(["1", "2", "3", "5"])
                return " ".join(toks), "vni_raw"
    elif typo_mode == "wrong_tone":
        for src, dst in WRONG_TONE_REPLACEMENTS:
            if src in text:
                return text.replace(src, dst, 1), "wrong_tone"
        toks = text.split()
        acc_idx = [i for i, t in enumerate(toks) if remove_diacritics(t) != t]
        if acc_idx:
            i = rng.choice(acc_idx)
            toks[i] = remove_diacritics(toks[i])
            return " ".join(toks), "wrong_tone"
    elif typo_mode == "sticky_keys":
        toks = text.split()
        if len(toks) >= 2:
            pos = rng.randint(0, len(toks) - 2)
            merged = toks[pos] + toks[pos + 1]
            new_toks = toks[:pos] + [merged] + toks[pos + 2:]
            return " ".join(new_toks), "sticky_keys"
    return text, "none"


def audit_parser_v1(
    name: str,
    queries: List[Dict[str, Any]],
    parser: VietnameseQueryParser,
) -> Dict[str, Any]:
    """
    Kiểm toán parser_v1 (2c.1b):
    - Dương tính giả (FP) theo slot: parser trả về giá trị khi spec KHÔNG có (empty/None)
      HOẶC trả về giá trị không giao với spec.
    - Thống kê theo token kích hoạt & kiểm tra 'tìm mua'/'tim mua' -> Purple.
    """
    slot_fp_counts = Counter()
    slot_fp_empty_spec = Counter()
    slot_fp_mismatch_spec = Counter()
    slot_pred_nonempty = Counter()
    colour_fp_tokens = Counter()
    tim_mua_purple_ids: List[str] = []

    for q in queries:
        text = q["query"]
        spec = q["spec"]
        parsed = parser.parse(text)
        q_norm = parsed["normalized_query"]

        # articleType
        p_art = set(parsed.get("articleType") or [])
        s_art = set(spec.get("articleType") or [])
        if p_art:
            slot_pred_nonempty["articleType"] += 1
            if not s_art:
                slot_fp_counts["articleType"] += 1
                slot_fp_empty_spec["articleType"] += 1
            elif not (p_art & s_art):
                slot_fp_counts["articleType"] += 1
                slot_fp_mismatch_spec["articleType"] += 1

        # gender
        p_gd = parsed.get("gender")
        s_gd = spec.get("gender")
        if p_gd:
            slot_pred_nonempty["gender"] += 1
            if not s_gd:
                slot_fp_counts["gender"] += 1
                slot_fp_empty_spec["gender"] += 1
            elif p_gd != s_gd:
                slot_fp_counts["gender"] += 1
                slot_fp_mismatch_spec["gender"] += 1

        # baseColour
        p_col = set(parsed.get("baseColour") or [])
        s_col = set(spec.get("baseColour") or [])
        if p_col:
            slot_pred_nonempty["baseColour"] += 1
            fp_cols = p_col - s_col
            if fp_cols:
                slot_fp_counts["baseColour"] += 1
                if not s_col:
                    slot_fp_empty_spec["baseColour"] += 1
                else:
                    slot_fp_mismatch_spec["baseColour"] += 1
                # Ghi nhận token kích hoạt
                toks = set(q_norm.split())
                for c_lbl in fp_cols:
                    colour_fp_tokens[f"{c_lbl} ({'tim_mua' if 'tim mua' in q_norm else ','.join(sorted(toks & {'tim','do','cam','be','den','da','trang','xanh','nau','hong','vang','bac','xam','kem'}))})"] += 1

        if ("tim mua" in q_norm) and ("Purple" in p_col) and ("Purple" not in s_col):
            tim_mua_purple_ids.append(q["id"])

    n = len(queries)
    return {
        "dataset": name,
        "n_queries": n,
        "slot_pred_nonempty": dict(slot_pred_nonempty),
        "slot_fp_counts": dict(slot_fp_counts),
        "slot_fp_rate_over_queries": {k: round(slot_fp_counts[k] / n, 4) if n else 0.0 for k in ("articleType", "gender", "baseColour")},
        "slot_fp_rate_over_preds": {
            k: round(slot_fp_counts[k] / slot_pred_nonempty[k], 4) if slot_pred_nonempty[k] else 0.0
            for k in ("articleType", "gender", "baseColour")
        },
        "slot_fp_empty_spec": dict(slot_fp_empty_spec),
        "slot_fp_mismatch_spec": dict(slot_fp_mismatch_spec),
        "colour_fp_tokens": dict(colour_fp_tokens.most_common(15)),
        "tim_mua_purple_count": len(tim_mua_purple_ids),
        "tim_mua_purple_ids": tim_mua_purple_ids,
    }


def main():
    rng = random.Random(42)
    val_id_map = json.load(open(PROJECT_ROOT / "index/val_gallery/id_map.json", encoding="utf-8"))
    strict_cats = load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")["policies"]["hybrid"]["strict_categories"]

    changelog_lines: List[str] = [
        "# Changelog Sửa Tập A và Tập B-synth (Bước 2c.4)",
        "",
    ]

    # =========================================================================
    # 2c.4c: Sửa Tập A ("áo dài truyền thống sari" -> "sari Ấn Độ" / "sari an do")
    # =========================================================================
    p_a = PROJECT_ROOT / "eval/val_queries_a_synthetic.json"
    val_a = json.load(open(p_a, encoding="utf-8"))
    changelog_lines.append("## 1. Tập A (`eval/val_queries_a_synthetic.json`)")
    changelog_lines.append("")
    a_changed = 0
    for item in val_a:
        q = item["query"]
        if "áo dài truyền thống sari" in q:
            old_q = q
            new_term = "sari Ấn Độ" if item.get("has_diacritics", True) else "sari an do"
            item["query"] = q.replace("áo dài truyền thống sari", new_term)
            a_changed += 1
            changelog_lines.append(f"- `{item['id']}`: `{old_q}` -> `{item['query']}` (bỏ cụm 'áo dài' gây nhầm văn hóa)")
    with open(p_a, "w", encoding="utf-8") as f:
        json.dump(val_a, f, indent=2, ensure_ascii=False)
    sha_a = file_sha256(p_a)
    changelog_lines.append(f"\nTổng số câu sửa ở Tập A: **{a_changed}** | SHA-256 mới: `{sha_a}`\n")

    # =========================================================================
    # 2c.4a & 2c.4b: Sửa Tập B-synth
    # =========================================================================
    p_b = PROJECT_ROOT / "eval/val_queries_b_synth.json"
    val_b = json.load(open(p_b, encoding="utf-8"))

    changelog_lines.append("## 2. Tập B-synth (`eval/val_queries_b_synth.json`) — Kiểm toán nhãn đa nghĩa (2c.4a)")
    changelog_lines.append("")
    b_ambig_changed = 0
    b_dia_changed = 0

    for item in val_b:
        q_lower = item["query"].lower()
        q_norm = remove_diacritics(q_lower)
        # Flag is_typo rõ ràng
        item["is_typo"] = bool(item.get("typo_type"))

        # Kiểm tra từ đồng nghĩa đa nghĩa
        matched_multi: Optional[Tuple[str, List[str], str]] = None
        for syn, multi_ats, reason in AMBIGUOUS_SYNONYM_MAP:
            syn_norm = remove_diacritics(syn)
            if syn_norm in q_norm:
                matched_multi = (syn, multi_ats, reason)
                break

        if matched_multi is not None:
            syn, multi_ats, reason = matched_multi
            old_ats = list(item["spec"]["articleType"])
            if set(old_ats) != set(multi_ats):
                item["spec"]["articleType"] = list(multi_ats)
                item["n_relevant"] = sum(
                    1 for m in val_id_map if is_item_relevant(item["spec"], m, "hybrid", strict_cats)
                )
                item["spec_cluster_id"] = make_cluster_id(item["spec"])
                b_ambig_changed += 1
                changelog_lines.append(
                    f"- `{item['id']}` (`{item['query']}`): `articleType` `{old_ats}` -> `{multi_ats}` ({reason}; `n_relevant={item['n_relevant']}`)"
                )

        # Tính lại dia_class chuẩn cho typo (wrong_tone không được là 'full')
        new_dia = detect_dia_class(item["query"], typo_type=item.get("typo_type"))
        if new_dia != item.get("dia_class"):
            old_dia = item.get("dia_class")
            item["dia_class"] = new_dia
            item["has_diacritics"] = (new_dia != "none")
            b_dia_changed += 1
            changelog_lines.append(
                f"- `{item['id']}` (`typo_type={item.get('typo_type')}`): `dia_class` `{old_dia}` -> `{new_dia}`"
            )

    # 2c.4b: Thêm >= 30 câu typo Telex / VNI / sticky_keys / wrong_tone (thêm 35 câu)
    clean_pool = [x for x in val_b if not x.get("is_typo")]
    seen_queries = {x["query"] for x in val_b}
    typo_modes = ["telex_raw", "vni_raw", "sticky_keys", "wrong_tone"]
    extra_added = 0
    ptr = 0
    changelog_lines.append("\n## 3. Tập B-synth — Bổ sung 35 câu typo Telex/VNI/dính phím/sai thanh (2c.4b)\n")
    while extra_added < 35 and ptr < 1000:
        base_item = clean_pool[(ptr * 7 + 3) % len(clean_pool)]
        mode = typo_modes[extra_added % len(typo_modes)]
        ptr += 1
        # Nếu telex/vni/wrong_tone ưu tiên câu có dấu để chuyển đổi rõ ràng
        if mode in ("telex_raw", "vni_raw", "wrong_tone") and not base_item.get("has_diacritics"):
            continue
        q_typo, t_actual = generate_extra_typo(base_item["query"], mode, rng)
        if q_typo == base_item["query"] or q_typo in seen_queries or t_actual == "none":
            continue
        seen_queries.add(q_typo)
        new_item = copy.deepcopy(base_item)
        new_id = f"val_b_synth_{len(val_b):03d}"
        new_item["id"] = new_id
        new_item["query"] = q_typo
        new_item["is_typo"] = True
        new_item["typo_type"] = t_actual
        new_dia = detect_dia_class(q_typo, typo_type=t_actual)
        new_item["dia_class"] = new_dia
        new_item["has_diacritics"] = (new_dia != "none")
        new_item["source"] = f"typo_from_{base_item['id']}"
        val_b.append(new_item)
        extra_added += 1
        changelog_lines.append(f"- `{new_id}` (`{t_actual}`, `dia_class={new_dia}`): `{base_item['query']}` -> `{q_typo}`")

    with open(p_b, "w", encoding="utf-8") as f:
        json.dump(val_b, f, indent=2, ensure_ascii=False)
    sha_b = file_sha256(p_b)

    n_b_clean = sum(1 for x in val_b if not x.get("is_typo"))
    n_b_typo = sum(1 for x in val_b if x.get("is_typo"))
    n_clusters_b = len(set(x["spec_cluster_id"] for x in val_b))
    n_clusters_b_clean = len(set(x["spec_cluster_id"] for x in val_b if not x.get("is_typo")))

    changelog_lines.append(
        f"\nTổng kết B-synth: **{len(val_b)}** câu (**{n_b_clean}** sạch, **{n_b_typo}** typo), "
        f"**{n_clusters_b}** `spec_cluster_id` (**{n_clusters_b_clean}** trên câu sạch), "
        f"sửa nhãn đa nghĩa: **{b_ambig_changed}**, sửa `dia_class`: **{b_dia_changed}**, thêm typo mới: **{extra_added}** | SHA-256: `{sha_b}`"
    )

    p_chg = PROJECT_ROOT / "reports/phase5b/val_ab_changelog.md"
    p_chg.write_text("\n".join(changelog_lines), encoding="utf-8")

    # =========================================================================
    # 2c.3: Dựng lại Tập C lồng nhau + hiệu lực + real_parse_v1
    # =========================================================================
    parser_v1 = VietnameseQueryParser()
    # Chọn 160 câu gốc cân bằng từ A (80 câu) và B-synth sạch (80 câu) với seed=42
    rng_c = random.Random(42)
    a_pool = copy.deepcopy(val_a)
    b_clean_pool = [copy.deepcopy(x) for x in val_b if not x.get("is_typo")]
    rng_c.shuffle(a_pool)
    rng_c.shuffle(b_clean_pool)
    base_c = a_pool[:80] + b_clean_pool[:80]

    oracle_parsed = [
        {
            "articleType": list(x["spec"].get("articleType") or []),
            "gender": x["spec"].get("gender"),
            "baseColour": list(x["spec"].get("baseColour") or []),
        }
        for x in base_c
    ]
    real_parsed_v1 = [
        {
            "articleType": p["articleType"],
            "gender": p["gender"],
            "baseColour": p["baseColour"],
        }
        for p in (parser_v1.parse(x["query"]) for x in base_c)
    ]

    nested_res = inject_nested_parse_errors(oracle_parsed, error_rates=(0.0, 0.1, 0.2, 0.3), seed=42)
    c_payload = {
        "description": "Tập C (2c.3): Lỗi parse LỒNG NHAU, chỉ tiêm lỗi có hiệu lực, có hàng tham chiếu real_parse_v1.",
        "seed": 42,
        "n_queries": len(base_c),
        "sources": {
            "0.0": "oracle_spec (không tiêm lỗi)",
            "0.1": "nested_effective_injection (u < 0.1, seed=42)",
            "0.2": "nested_effective_injection (u < 0.2, seed=42)",
            "0.3": "nested_effective_injection (u < 0.3, seed=42)",
            "real_parse_v1": "đầu ra thật của parser_v1 trên 160 câu gốc (không tiêm lỗi)",
        },
        "actual_error_rates": {
            f"{int(r * 100)}%": round(info["actual_error_rate"], 4)
            for r, info in nested_res["levels"].items()
        },
        "n_effective_errors": {
            f"{int(r * 100)}%": info["n_effective_errors"]
            for r, info in nested_res["levels"].items()
        },
        "queries": [
            {
                "id": f"val_c_{i:03d}",
                "source_id": x["id"],
                "query": x["query"],
                "has_diacritics": x.get("has_diacritics", True),
                "dia_class": x.get("dia_class", "full"),
                "n_relevant": x["n_relevant"],
                "spec": x["spec"],
                "spec_cluster_id": x["spec_cluster_id"],
                "u_nested": round(nested_res["u_values"][i], 6),
                "planned_error_type": nested_res["planned_error_types"][i],
                "parsed_real_v1": real_parsed_v1[i],
                "parsed_by_rate": {
                    f"{int(r * 100)}%": nested_res["levels"][r]["parsed"][i]
                    for r in (0.0, 0.1, 0.2, 0.3)
                },
            }
            for i, x in enumerate(base_c)
        ],
    }
    p_c = PROJECT_ROOT / "eval/val_queries_c_error_injection.json"
    with open(p_c, "w", encoding="utf-8") as f:
        json.dump(c_payload, f, indent=2, ensure_ascii=False)
    sha_c = file_sha256(p_c)

    # =========================================================================
    # 2c.1b: Kiểm toán parser_v1 trên DEV, Tập A, Tập B-synth
    # =========================================================================
    dev_queries = json.load(open(PROJECT_ROOT / "eval/handwritten_queries.json", encoding="utf-8"))
    audits = [
        audit_parser_v1("DEV (handwritten_queries)", dev_queries, parser_v1),
        audit_parser_v1("Val A (synthetic)", val_a, parser_v1),
        audit_parser_v1("Val B-synth (all)", val_b, parser_v1),
        audit_parser_v1("Val B-synth (clean only)", [x for x in val_b if not x.get("is_typo")], parser_v1),
    ]
    p_audit_json = PROJECT_ROOT / "reports/phase5b/parser_v1_audit.json"
    with open(p_audit_json, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, ensure_ascii=False)

    print("=== 2c.4 & 2c.3 & 2c.1b DONE ===")
    print(f"A: {len(val_a)} queries, {len(set(x['spec_cluster_id'] for x in val_a))} clusters | SHA-256: {sha_a}")
    print(f"B-synth: {len(val_b)} queries ({n_b_clean} clean, {n_b_typo} typo), {n_clusters_b} clusters ({n_clusters_b_clean} clean) | SHA-256: {sha_b}")
    print(f"C: {len(base_c)} queries, actual_error_rates={c_payload['actual_error_rates']}, n_eff={c_payload['n_effective_errors']} | SHA-256: {sha_c}")
    for au in audits:
        print(f"Audit {au['dataset']}: FP_over_queries={au['slot_fp_rate_over_queries']}, FP_over_preds={au['slot_fp_rate_over_preds']}, tim_mua_purple={au['tim_mua_purple_count']} ({au['tim_mua_purple_ids'][:5]})")


if __name__ == "__main__":
    main()
