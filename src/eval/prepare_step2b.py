"""
Chuẩn bị Bước 2b (Pha 5b):
- 2b.2: Cập nhật Tập A (gắn spec_cluster_id, sửa từ chưa dịch, ghi catalog_vi_gaps.md).
- 2b.3: Sửa và tách Tập B thành val_queries_b_synth.json (không mâu thuẫn nhãn, không giới tính trong từ đồng nghĩa,
        thêm >= 45 câu lỗi gõ thật có seed=42, dia_class, cờ overlaps_dev13_vocab) và tạo file trống val_queries_b_human.json.
- 2b.4: Đổi tên C cũ -> val_queries_c_old_missing_probe.json, tạo val_queries_c_error_injection.json (tiêm lỗi seed=42).
- 2b.5: Khóa selection_rule_v2.json và selection_rule_v2.md.
"""
import os
import sys
import json
import copy
import random
import hashlib
import pathlib
import unicodedata
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import load_gender_gt_policy, is_item_relevant
from src.eval.error_injection import inject_parse_errors
from src.serve.parser import remove_diacritics


# Cụm từ trùng từ vựng với 13 câu thất bại của DEV SET (để gắn cờ overlaps_dev13_vocab)
DEV13_VOCAB_PATTERNS = [
    "boxer", "quan lot", "jumpsuit", "ao lien quan", "da bong", "san co",
    "tui deo cheo", "tui trong", "day nit", "mat vuong", "day kim loai",
    "quan jean", "quan tay", "vi da", "dong ho"
]


def make_cluster_id(spec: dict) -> str:
    ats = "|".join(sorted(spec.get("articleType") or []))
    gd = str(spec.get("gender") or "None")
    cls = "|".join(sorted(spec.get("baseColour") or []))
    return f"{ats}__{gd}__{cls}"


def detect_dia_class(text: str) -> str:
    """Phân loại dấu tiếng Việt: full (có dấu đầy đủ), none (không dấu), partial (trộn)."""
    tokens = text.split()
    if not tokens:
        return "none"
    has_acc_tokens = [remove_diacritics(t) != t for t in tokens]
    if not any(has_acc_tokens):
        return "none"
    # Kiểm tra các từ có thể có dấu nhưng bị mất dấu xen kẽ với từ có dấu
    # Dùng heuristic: nếu câu chứa các từ thường gặp không dấu như 'mau', 'ao', 'quan', 'giay', 'tui', 'dong', 'ho', 'nam', 'nu'
    # trong khi vẫn có từ có dấu -> partial
    unaccented_markers = {"mau", "ao", "quan", "giay", "dep", "tui", "kinh", "dong", "ho", "nu", "trang", "den", "xanh", "nau", "xam", "hong", "vang", "tim", "bac", "cho", "di", "lam", "choi"}
    has_unacc_marker = any((t.lower() in unaccented_markers) for t in tokens)
    if any(has_acc_tokens) and has_unacc_marker:
        return "partial"
    return "full"


def inject_real_typo(text: str, rng: random.Random) -> tuple[str, str]:
    """Sinh lỗi gõ thật (có seed): swap_chars, drop_char, insert_char, wrong_tone."""
    typo_choices = ["swap_chars", "drop_char", "insert_char", "wrong_tone"]
    words = text.split()
    eligible = [idx for idx, w in enumerate(words) if len(w) >= 3]
    if not eligible:
        return text, "none"

    w_idx = rng.choice(eligible)
    w = words[w_idx]
    ttype = rng.choice(typo_choices)

    if ttype == "swap_chars" and len(w) >= 3:
        pos = rng.randint(0, len(w) - 2)
        w_new = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
        words[w_idx] = w_new
        return " ".join(words), "swap_chars"
    elif ttype == "drop_char" and len(w) >= 4:
        pos = rng.randint(1, len(w) - 2)
        w_new = w[:pos] + w[pos + 1:]
        words[w_idx] = w_new
        return " ".join(words), "drop_char"
    elif ttype == "insert_char":
        pos = rng.randint(1, len(w) - 1)
        dup = w[pos]
        w_new = w[:pos] + dup + w[pos:]
        words[w_idx] = w_new
        return " ".join(words), "insert_char"
    else:
        # wrong_tone: bỏ dấu ở 1 từ có dấu hoặc đổi thanh
        acc_indices = [idx for idx, tok in enumerate(words) if remove_diacritics(tok) != tok]
        if acc_indices:
            ti = rng.choice(acc_indices)
            words[ti] = remove_diacritics(words[ti])
            return " ".join(words), "wrong_tone"
        # fallback swap
        pos = rng.randint(0, len(w) - 2)
        words[w_idx] = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
        return " ".join(words), "swap_chars"


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng = random.Random(42)
    val_id_map = json.load(open(PROJECT_ROOT / "index/val_gallery/id_map.json", encoding="utf-8"))
    strict_cats = load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")["policies"]["hybrid"]["strict_categories"]

    # =========================================================================
    # 2b.2: Cập nhật Tập A (synthetic) & ghi catalog_vi_gaps.md
    # =========================================================================
    val_a = json.load(open(PROJECT_ROOT / "eval/val_queries_a_synthetic.json", encoding="utf-8"))
    for item in val_a:
        q = item["query"]
        # Sửa từ chưa dịch trong Tập A
        q = q.replace("sarees", "áo dài truyền thống sari")
        q = q.replace("màu steel", "màu thép kim loại").replace("mau steel", "mau thep kim loai")
        item["query"] = q
        item["spec_cluster_id"] = make_cluster_id(item["spec"])
        item["dia_class"] = "full" if item["has_diacritics"] else "none"

    p_a = PROJECT_ROOT / "eval/val_queries_a_synthetic.json"
    with open(p_a, "w", encoding="utf-8") as f:
        json.dump(val_a, f, indent=2, ensure_ascii=False)
    n_clusters_a = len(set(x["spec_cluster_id"] for x in val_a))
    print(f"2b.2 Tập A: {len(val_a)} câu, {n_clusters_a} spec_cluster_id độc nhất | SHA-256: {file_sha256(p_a)}")

    # Ghi catalog_vi_gaps.md
    df_cat = pd.read_parquet(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    at_gaps = []
    for at, vi in df_cat[["articleType", "articleType_vi"]].drop_duplicates().sort_values("articleType").values:
        if str(at).lower() == str(vi).lower() or str(vi).lower() in ("áo kurta", "áo kurti"):
            at_gaps.append((str(at), str(vi)))
    cl_gaps = []
    for cl, vi in df_cat[["baseColour", "baseColour_vi"]].drop_duplicates().sort_values("baseColour").values:
        if str(cl).lower() == str(vi).lower() or str(cl).lower() in ("navy blue",):
            cl_gaps.append((str(cl), str(vi)))

    gaps_md = PROJECT_ROOT / "reports/phase5b/catalog_vi_gaps.md"
    with open(gaps_md, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Các lỗ hổng Cột Tiếng Việt (`*_vi`) trong `catalog_rebuild.parquet`\n\n")
        f.write(f"- Tổng số `articleType` chưa dịch hoặc giữ nguyên tiếng Anh: **{len(at_gaps)}**\n")
        f.write(f"- Tổng số `baseColour` chưa dịch hoàn chỉnh: **{len(cl_gaps)}**\n\n")
        f.write("## 1. Danh sách `baseColour` chưa Việt hóa hoàn chỉnh\n\n| Giá trị gốc | Giá trị cột `baseColour_vi` |\n|---|---|\n")
        for cl, vi in cl_gaps:
            f.write(f"| `{cl}` | `{vi}` |\n")
        f.write("\n## 2. Danh sách `articleType` chưa Việt hóa hoàn chỉnh (top 30 mẫu)\n\n| Giá trị gốc | Giá trị cột `articleType_vi` |\n|---|---|\n")
        for at, vi in at_gaps[:30]:
            f.write(f"| `{at}` | `{vi}` |\n")
    print(f"Đã ghi {gaps_md}")

    # =========================================================================
    # 2b.3: Sửa & tách Tập B thành val_queries_b_synth.json + val_queries_b_human.json (trống)
    # =========================================================================
    # Bảng đồng nghĩa sạch: TUYỆT ĐỐI KHÔNG CHỨA GIỚI TÍNH trong cụm danh từ!
    CLEAN_SYNONYMS = {
        "Tshirts": ["áo phông", "áo thun lạnh", "áo phông cổ tròn", "áo cộc tay", "áo thun cotton"],
        "Shirts": ["áo sơ mi", "sơ mi công sở", "sơ mi dài tay", "áo sơ mi đi làm"],
        "Casual Shoes": ["giày vải", "giày đi chơi", "giày thể thao nhẹ", "giày sneaker thường"],
        "Sports Shoes": ["giày chạy bộ", "giày tập gym", "giày đá bóng", "giày thể dục"],
        "Formal Shoes": ["giày tây", "giày da công sở", "giày chú rể"],
        "Watches": ["đồng hồ đeo tay", "đồng hồ kim", "đồng hồ thời trang"],
        "Handbags": ["túi xách tay", "túi đeo vai", "túi công sở"],
        "Backpacks": ["ba lô", "cặp sách", "ba lô đi học", "ba lô laptop"],
        "Heels": ["guốc cao gót", "giày gót nhọn", "giày đi tiệc"],
        "Flats": ["giày bệt", "giày đế bằng", "giày mọi"],
        "Flip Flops": ["dép xỏ ngón", "dép kẹp", "dép đi biển", "dép lào"],
        "Sandals": ["dép quai hậu", "xăng đan", "dép quai ngang"],
        "Sunglasses": ["kính râm", "kính mát", "mắt kính đi nắng"],
        "Wallets": ["bóp tiền", "bóp da", "ví gập", "bóp cầm tay"],
        "Belts": ["dây nịt", "dây lưng da", "nịt quần"],
        "Jeans": ["quần bò", "quần jean dài", "quần bò ống đứng"],
        "Trousers": ["quần âu", "quần vải đi làm", "quần công sở"],
        "Shorts": ["quần đùi", "quần ngố", "quần lửng"],
        "Dresses": ["đầm dự tiệc", "váy liền thân", "đầm công sở", "đầm xòe"],
        "Skirts": ["váy ngắn", "váy xòe", "chân váy chữ a"],
        "Tops": ["áo kiểu", "áo cánh", "áo điệu"],
        "Jackets": ["áo khoác gió", "áo ấm", "áo khoác ngoài"],
        "Sweaters": ["áo len chui đầu", "áo len mùa đông"],
        "Sweatshirts": ["áo nỉ chui đầu", "áo sweater nỉ"],
        "Socks": ["vớ chân", "vớ cổ ngắn", "vớ thể thao"],
        "Caps": ["nón kết", "nón lưỡi trai", "mũ che nắng"],
        "Briefs": ["quần sịp", "quần lót"],
        "Earrings": ["hoa tai", "khuyên tai", "hoa tai bạc"],
        "Ties": ["cà vạt công sở", "caravat"],
    }

    CLEAN_GENDERS = {
        "Men": ["cho nam", "nam giới", "của nam", "nam"],
        "Women": ["cho nữ", "nữ giới", "của nữ", "nữ", "chị em"],
        "Unisex": ["unisex", "nam nữ đều mang được"],
        "Boys": ["bé trai", "trẻ em nam"],
        "Girls": ["bé gái", "trẻ em nữ"],
    }

    CLEAN_COLOURS = {
        "Black": ["màu đen", "đen tuyền", "đen"],
        "White": ["màu trắng", "trắng tinh", "trắng"],
        "Blue": ["màu xanh dương", "xanh biển", "xanh da trời"],
        "Navy Blue": ["xanh than", "xanh navy", "xanh tím than"],
        "Grey": ["màu xám", "xám ghi", "lông chuột"],
        "Red": ["màu đỏ", "đỏ tươi", "đỏ"],
        "Green": ["màu xanh lá", "xanh rêu", "xanh lục"],
        "Brown": ["màu nâu", "nâu đất", "nâu bò"],
        "Pink": ["màu hồng", "hồng phấn", "hồng"],
        "Silver": ["màu bạc", "bạc sáng", "ánh bạc"],
        "Beige": ["màu kem be", "màu be", "kem"],
        "Yellow": ["màu vàng", "vàng tươi", "vàng"],
        "Purple": ["màu tím", "tím"],
        "Maroon": ["đỏ đô", "đỏ mận"],
    }

    FILLER_PREF = ["", "", "shop ơi có ", "cần tìm ", "cho mình hỏi ", "tìm mua ", "tư vấn giúp "]
    FILLER_SUFF = ["", "", " mẫu mới nhất", " đi chơi", " đi làm", " mặc hàng ngày", " đẹp xịn", " giá tốt"]

    val_df = pd.DataFrame(val_id_map)
    valid_specs = []
    # Nhóm 3 slot
    for (at, gd, cl), grp in val_df.groupby(["articleType", "gender", "baseColour"]):
        if len(grp) >= 5 and at in CLEAN_SYNONYMS and gd in CLEAN_GENDERS and cl in CLEAN_COLOURS:
            spec = {"articleType": [at], "gender": gd, "baseColour": [cl]}
            n_rel = sum(1 for m in val_id_map if is_item_relevant(spec, m, "hybrid", strict_cats))
            if n_rel >= 5:
                valid_specs.append((spec, n_rel))
    # Nhóm 2 slot (thiếu màu)
    for (at, gd), grp in val_df.groupby(["articleType", "gender"]):
        if len(grp) >= 5 and at in CLEAN_SYNONYMS and gd in CLEAN_GENDERS:
            spec = {"articleType": [at], "gender": gd, "baseColour": []}
            n_rel = sum(1 for m in val_id_map if is_item_relevant(spec, m, "hybrid", strict_cats))
            if n_rel >= 5:
                valid_specs.append((spec, n_rel))
    # Nhóm 2 slot (thiếu giới tính)
    for (at, cl), grp in val_df.groupby(["articleType", "baseColour"]):
        if len(grp) >= 5 and at in CLEAN_SYNONYMS and cl in CLEAN_COLOURS:
            spec = {"articleType": [at], "gender": None, "baseColour": [cl]}
            n_rel = sum(1 for m in val_id_map if is_item_relevant(spec, m, "hybrid", strict_cats))
            if n_rel >= 5:
                valid_specs.append((spec, n_rel))

    rng.shuffle(valid_specs)

    b_synth = []
    seen_b = set()
    idx_ptr = 0
    # Tạo 165 câu sạch (chưa typo)
    while len(b_synth) < 165 and idx_ptr < 3000:
        idx_ptr += 1
        spec, n_rel = valid_specs[idx_ptr % len(valid_specs)]
        at = spec["articleType"][0]
        gd = spec["gender"]
        cl = spec["baseColour"][0] if spec["baseColour"] else None

        s_at = rng.choice(CLEAN_SYNONYMS[at])
        s_gd = rng.choice(CLEAN_GENDERS[gd]) if gd else ""
        s_cl = rng.choice(CLEAN_COLOURS[cl]) if cl else ""

        pref = rng.choice(FILLER_PREF)
        suff = rng.choice(FILLER_SUFF)
        q_raw = " ".join(f"{pref}{' '.join(p for p in [s_at, s_gd, s_cl] if p)}{suff}".split())

        # Bỏ dấu toàn bộ (~40%), trộn dấu (~15%), có dấu (~45%)
        r = rng.random()
        if r < 0.40:
            q_txt = remove_diacritics(q_raw)
        elif r < 0.55:
            # Trộn partial: bỏ dấu phần đầu hoặc phần cuối
            toks = q_raw.split()
            half = max(1, len(toks) // 2)
            q_txt = " ".join([remove_diacritics(t) if i < half else t for i, t in enumerate(toks)])
        else:
            q_txt = q_raw

        if q_txt in seen_b:
            continue
        seen_b.add(q_txt)

        q_norm = remove_diacritics(q_txt).lower()
        overlaps = any(pat in q_norm for pat in DEV13_VOCAB_PATTERNS)
        d_cls = detect_dia_class(q_txt)

        b_synth.append({
            "id": f"val_b_synth_{len(b_synth):03d}",
            "query": q_txt,
            "has_diacritics": (d_cls != "none"),
            "dia_class": d_cls,
            "typo_type": None,
            "n_relevant": n_rel,
            "spec": spec,
            "spec_cluster_id": make_cluster_id(spec),
            "overlaps_dev13_vocab": overlaps,
            "source": "template + bảng đồng nghĩa của agent",
        })

    # Thêm >= 45 câu có LỖI GÕ THẬT (sinh bằng hàm có seed=42 từ các spec hợp lệ)
    n_typo_added = 0
    t_ptr = 0
    while n_typo_added < 45 and t_ptr < 1000:
        t_ptr += 1
        base_item = b_synth[t_ptr % 120]
        q_typo, ttype = inject_real_typo(base_item["query"], rng)
        if q_typo == base_item["query"] or q_typo in seen_b:
            continue
        seen_b.add(q_typo)
        q_norm = remove_diacritics(q_typo).lower()
        overlaps = any(pat in q_norm for pat in DEV13_VOCAB_PATTERNS)
        d_cls = detect_dia_class(q_typo)
        b_synth.append({
            "id": f"val_b_synth_{len(b_synth):03d}",
            "query": q_typo,
            "has_diacritics": (d_cls != "none"),
            "dia_class": d_cls,
            "typo_type": ttype,
            "n_relevant": base_item["n_relevant"],
            "spec": copy.deepcopy(base_item["spec"]),
            "spec_cluster_id": base_item["spec_cluster_id"],
            "overlaps_dev13_vocab": overlaps,
            "source": "template + bảng đồng nghĩa của agent + real_typo",
        })
        n_typo_added += 1

    p_b_synth = PROJECT_ROOT / "eval/val_queries_b_synth.json"
    with open(p_b_synth, "w", encoding="utf-8") as f:
        json.dump(b_synth, f, indent=2, ensure_ascii=False)

    # Xóa bản draft cũ để tránh nhầm lẫn
    old_b_draft = PROJECT_ROOT / "eval/val_queries_b_natural_draft.json"
    if old_b_draft.exists():
        old_b_draft.unlink()

    # Thống kê dia_class & typo_type trên B-synth
    dia_dist = pd.Series([x["dia_class"] for x in b_synth]).value_counts().to_dict()
    typo_dist = pd.Series([x["typo_type"] or "none" for x in b_synth]).value_counts().to_dict()
    overlap_cnt = sum(1 for x in b_synth if x["overlaps_dev13_vocab"])
    print(f"2b.3 Tập B-synth: {len(b_synth)} câu | dia_class={dia_dist} | typos={typo_dist} | overlaps_dev13={overlap_cnt}")
    print(f"  SHA-256 val_queries_b_synth.json: {file_sha256(p_b_synth)}")

    # Tạo file trống eval/val_queries_b_human.json + tài liệu schema
    p_b_human = PROJECT_ROOT / "eval/val_queries_b_human.json"
    if not p_b_human.exists():
        with open(p_b_human, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

    p_b_schema = PROJECT_ROOT / "eval/val_queries_b_human_schema.md"
    with open(p_b_schema, "w", encoding="utf-8") as f:
        f.write("# Hướng dẫn Schema cho `eval/val_queries_b_human.json` (>= 40 câu do người dùng viết)\n\n")
        f.write("Mỗi phần tử trong mảng JSON có định dạng:\n```json\n")
        f.write(json.dumps({
            "id": "val_b_human_000",
            "query": "câu truy vấn tự nhiên của bạn",
            "has_diacritics": True,
            "dia_class": "full | none | partial",
            "typo_type": None,
            "spec": {
                "articleType": ["Tshirts"],
                "gender": "Men",
                "baseColour": ["Black"]
            },
            "source": "human_written"
        }, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
        f.write("- Lưu ý: Chỉ ghi các thuộc tính **nêu rõ** trong câu vào `spec` (`gender: null` nếu không nêu giới tính, `baseColour: []` nếu không nêu màu).\n")
        f.write("- Sau khi bạn điền xong, hệ thống sẽ kiểm tra `n_relevant >= 5` trên `index/val_gallery/` và gắn `spec_cluster_id` rồi khóa SHA-256.\n")

    # =========================================================================
    # 2b.4: Tập C làm lại (tiêm lỗi có seed=42, độc lập lexicon)
    # =========================================================================
    old_c = PROJECT_ROOT / "eval/val_queries_c_probe_draft.json"
    new_old_c = PROJECT_ROOT / "eval/val_queries_c_old_missing_probe.json"
    if old_c.exists():
        if new_old_c.exists():
            new_old_c.unlink()
        old_c.rename(new_old_c)

    # Lấy tập con 160 câu từ A (80 câu) + B-synth sạch (80 câu)
    clean_b = [x for x in b_synth if x["typo_type"] is None]
    sample_a = rng.sample(val_a, min(80, len(val_a)))
    sample_b = rng.sample(clean_b, min(80, len(clean_b)))
    c_items = []
    for src_item in (sample_a + sample_b):
        clean_p = {
            "articleType": list(src_item["spec"].get("articleType") or []),
            "gender": src_item["spec"].get("gender"),
            "baseColour": list(src_item["spec"].get("baseColour") or []),
        }
        c_items.append({
            "id": f"val_c_{len(c_items):03d}",
            "orig_id": src_item["id"],
            "query": src_item["query"],
            "has_diacritics": src_item["has_diacritics"],
            "dia_class": src_item["dia_class"],
            "n_relevant": src_item["n_relevant"],
            "spec": src_item["spec"],
            "spec_cluster_id": src_item["spec_cluster_id"],
            "clean_parsed": clean_p,
        })

    clean_parsed_list = [x["clean_parsed"] for x in c_items]
    injected_by_rate = {}
    for rate in [0.0, 0.1, 0.2, 0.3]:
        inj = inject_parse_errors(clean_parsed_list, error_rate=rate, seed=42)
        injected_by_rate[f"rate_{int(rate*100)}pct"] = inj

    val_c_payload = {
        "metadata": {
            "description": "Tập C mới (Parse-error injection độc lập lexicon, seed=42) để đo đường cong tác hại theo mức lỗi 0%, 10%, 20%, 30%.",
            "n_queries": len(c_items),
            "seed": 42,
            "error_rates": [0.0, 0.1, 0.2, 0.3],
            "injection_distribution": {
                "wrong_articleType_confusable": "30/70 (~42.9%)",
                "flip_gender": "10/70 (~14.3%)",
                "drop_slot": "20/70 (~28.6%)",
                "wrong_baseColour": "10/70 (~14.3%)",
            },
        },
        "queries": c_items,
        "injected_parsed": injected_by_rate,
    }
    p_c_new = PROJECT_ROOT / "eval/val_queries_c_error_injection.json"
    with open(p_c_new, "w", encoding="utf-8") as f:
        json.dump(val_c_payload, f, indent=2, ensure_ascii=False)
    print(f"2b.4 Tập C mới: {len(c_items)} câu x 4 mức lỗi | SHA-256: {file_sha256(p_c_new)}")

    # =========================================================================
    # 2b.5: Khóa Quy tắc chọn v2
    # =========================================================================
    selection_rule = {
        "version": "v2",
        "seed": 42,
        "objective": "Score = 0.5 * mAP(B-human) + 0.3 * mAP_cluster(B-synth) + 0.2 * mAP_cluster(A)",
        "objective_weights": {"B_human": 0.5, "B_synth_cluster": 0.3, "A_cluster": 0.2},
        "constraints": {
            "harm_curve_C_20pct": {
                "max_drop_vs_model_only": 0.03,
                "max_drop_vs_baseline_gamma03": 0.05,
                "description": "Trên Tập C tiêm lỗi 20%: mAP không thấp hơn model_only quá 0.03 (tuyệt đối) và không thấp hơn baseline gamma=0.3 quá 0.05.",
            },
            "catastrophic_drop_rate": {
                "max_rate": 0.05,
                "definition": "Tỉ lệ truy vấn có ảnh đúng trong top-10 ở baseline (pool toàn gallery, gamma=0.3) nhưng KHÔNG có ở cấu hình mới <= 5% trên cả B-human và B-synth.",
            },
        },
        "grid_and_cv": {
            "max_grid_size": 24,
            "cv_method": "5-fold Cross-Validation theo spec_cluster_id trên A + B-synth, xác nhận trên B-human + bootstrap 95% CI (1000 lần, resample theo cụm).",
        },
        "tie_break_simplicity": "Nếu chênh lệch Score nằm trong 95% paired bootstrap CI (chứa 0): chọn cấu hình đơn giản/nhanh hơn (gamma nhỏ, không mean-centering, không BM25).",
        "parallel_reporting": ["best_by_score", "safest_gamma_min_satisfying_constraints"],
    }
    p_rule_json = PROJECT_ROOT / "reports/phase5b/selection_rule_v2.json"
    with open(p_rule_json, "w", encoding="utf-8") as f:
        json.dump(selection_rule, f, indent=2, ensure_ascii=False)

    p_rule_md = PROJECT_ROOT / "reports/phase5b/selection_rule_v2.md"
    with open(p_rule_md, "w", encoding="utf-8") as f:
        f.write("# Quy tắc Chọn Cấu hình v2 (Khóa trước Bước 3)\n\n")
        f.write("1. **Mục tiêu tối đa hóa:**\n")
        f.write("   $$\\text{Score} = 0.5 \\cdot \\text{mAP}_{\\text{B-human}} + 0.3 \\cdot \\text{mAP}_{\\text{B-synth (theo cụm)}} + 0.2 \\cdot \\text{mAP}_{\\text{A (theo cụm)}}$$\n")
        f.write("2. **Ràng buộc tác hại trên Tập C mới (tiêm lỗi 20%):**\n")
        f.write("   - $\\text{mAP}_{C,20\\%} \\ge \\text{mAP}_{\\text{model\\_only}} - 0.03$\n")
        f.write("   - $\\text{mAP}_{C,20\\%} \\ge \\text{mAP}_{\\text{baseline } (\\gamma=0.3)} - 0.05$\n")
        f.write("3. **Ràng buộc trường hợp xấu nhất (Catastrophic Drop Rate):**\n")
        f.write("   - Tỉ lệ truy vấn có ảnh đúng trong Top-10 ở baseline nhưng **rớt khỏi Top-10** ở cấu hình mới $\\le 5\\%$ trên cả **B-human** và **B-synth**.\n")
        f.write("4. **Hạn chế winner's curse:**\n")
        f.write("   - Lưới $\\le 24$ cấu hình định nghĩa trước; chọn bằng **5-fold CV theo cụm (`spec_cluster_id`)** trên A + B-synth, xác nhận trên B-human kèm bootstrap CI.\n")
        f.write("5. **Ưu tiên đơn giản:** Nếu chênh lệch Score nằm trong 95% paired bootstrap CI (theo cụm), chọn cấu hình đơn giản/nhanh hơn ($\\gamma$ nhỏ, không mean-centering, không BM25).\n")
        f.write("6. **Báo cáo song song:** Báo cả cấu hình **Tốt nhất theo Score** và cấu hình **An toàn** (thỏa ràng buộc 2 & 3 với $\\gamma$ thấp nhất).\n")
    print(f"2b.5 Đã khóa quy tắc chọn tại {p_rule_json} | SHA-256: {file_sha256(p_rule_json)}")


if __name__ == "__main__":
    main()
