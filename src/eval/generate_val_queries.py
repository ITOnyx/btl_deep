"""
Sinh 3 tập truy vấn tuning độc lập cho Bước 2 (Pha 5b) trên index/val_gallery/ (1.884 sản phẩm val):
- Tập A (Synthetic): >= 200 truy vấn, >= 40% không dấu bằng remove_diacritics, mỗi truy vấn >= 5 ảnh đúng.
- Tập B (Natural-noisy draft): >= 150 truy vấn viết tự nhiên (từ đồng nghĩa/lóng ngoài lexicon, lỗi gõ,
  thiếu slot, từ thừa, trộn có/không dấu), không dùng lexicon của parser để sinh.
- Tập C (Parse-error probe draft): tập con/biến thể khiến parser hiện tại sai hoặc thiếu articleType,
  để đo đường cong tác hại của gamma lớn.
"""
import json
import random
import hashlib
import pathlib
import sys
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import load_gender_gt_policy, is_item_relevant
from src.serve.parser import VietnameseQueryParser, remove_diacritics


# Bảng từ đồng nghĩa / cách gọi tự nhiên ngoài lexicon (KHÔNG lấy từ parser.py) để tạo tập B (Natural-noisy)
NATURAL_SYNONYMS_MAP = {
    "Tshirts": ["áo phông", "áo thun lạnh", "áo phông cổ tròn", "áo cộc tay", "áo thun cotton", "ao phong"],
    "Shirts": ["áo sơ mi", "sơ mi công sở", "sơ mi dài tay", "áo sơ mi đi làm", "ao so mi"],
    "Casual Shoes": ["giày lười", "giày đi chơi", "giày bệt thể thao", "giày vải", "giay luoi", "giay di choi"],
    "Sports Shoes": ["giày chạy bộ", "giày tập gym", "giày đá bóng", "giày thể dục", "giay chay bo", "giay tap gym"],
    "Formal Shoes": ["giày tây", "giày da công sở", "giày chú rể", "giay tay", "giay da di lam"],
    "Watches": ["đồng hồ đeo tay", "đồng hồ kim", "đồng hồ thời trang", "dong ho deo tay"],
    "Handbags": ["túi xách tay", "túi đeo vai", "túi công sở", "tui xach tay", "tui deo vai"],
    "Backpacks": ["ba lô", "cặp sách", "ba lô đi học", "ba lô laptop", "ba lo", "cap sach"],
    "Heels": ["guốc cao gót", "giày gót nhọn", "giày đi tiệc", "guoc cao got"],
    "Flats": ["giày bệt", "giày mọi nữ", "giày đế bằng", "giay bet", "giay de bang"],
    "Flip Flops": ["dép xỏ ngón", "dép kẹp", "dép đi biển", "dép lào", "dep kep", "dep lao"],
    "Sandals": ["dép quai hậu", "xăng đan", "dép quai ngang", "xang dan", "dep quai hau"],
    "Sunglasses": ["kính râm", "kính mát", "mắt kính đi nắng", "kinh ram", "mat kinh"],
    "Wallets": ["bóp tiền", "bóp da", "ví gập", "bóp cầm tay", "bop tien", "bop da"],
    "Belts": ["dây nịt", "dây lưng da", "nịt quần", "day nit", "day lung"],
    "Jeans": ["quần bò", "quần jean dài", "quần bò ống đứng", "quan bo"],
    "Trousers": ["quần âu", "quần vải đi làm", "quần công sở", "quan au", "quan vai"],
    "Shorts": ["quần đùi", "quần ngố", "quần lửng", "quan dui", "quan ngo"],
    "Dresses": ["đầm dự tiệc", "váy liền thân", "đầm công sở", "đầm xòe", "dam du tiec", "vay lien"],
    "Skirts": ["váy ngắn", "váy xòe", "chân váy chữ a", "vay ngan", "vay xoe"],
    "Tops": ["áo kiểu nữ", "áo cánh", "áo điệu", "ao kieu nu"],
    "Jackets": ["áo khoác gió", "áo ấm", "áo khoác ngoài", "ao khoac gio"],
    "Sweaters": ["áo len chui đầu", "áo len mùa đông", "ao len mua dong"],
    "Sweatshirts": ["áo nỉ chui đầu", "áo sweater nỉ", "ao ni chui dau"],
    "Socks": ["vớ chân", "vớ cổ ngắn", "vớ thể thao", "vo chan"],
    "Caps": ["nón kết", "nón lưỡi trai", "mũ che nắng", "non ket"],
    "Kurtas": ["áo dài ấn độ kurta", "áo kurta truyền thống", "ao kurta"],
    "Briefs": ["quần sịp nam", "quần lót nam", "quan sip nam"],
    "Earrings": ["hoa tai", "khuyên tai", "hoa tai bạc", "khuyen tai"],
    "Ties": ["cà vạt công sở", "caravat", "ca vat", "caravat"],
}

NATURAL_GENDER_WORDS = {
    "Men": ["cho nam", "nam giới", "của nam", "phái mạnh", "nam", "cho bạn nam"],
    "Women": ["cho nữ", "nữ giới", "của nữ", "phái đẹp", "nữ", "cho bạn nữ", "chị em"],
    "Unisex": ["unisex", "nam nữ đều mang được", "cả nam và nữ", ""],
    "Boys": ["bé trai", "trẻ em nam"],
    "Girls": ["bé gái", "trẻ em nữ"],
}

NATURAL_COLOUR_WORDS = {
    "Black": ["màu đen", "đen tuyền", "đen", "màu tối đen", "mau den", "den"],
    "White": ["màu trắng", "trắng tinh", "trắng", "mau trang", "trang"],
    "Blue": ["màu xanh dương", "xanh biển", "xanh da trời", "xanh", "xanh duong"],
    "Navy Blue": ["xanh than", "xanh navy", "xanh tím than", "xanh đen", "xanh than"],
    "Grey": ["màu xám", "xám ghi", "lông chuột", "xám", "mau xam", "ghi"],
    "Red": ["màu đỏ", "đỏ tươi", "đỏ", "mau do"],
    "Green": ["màu xanh lá", "xanh rêu", "xanh lục", "xanh la"],
    "Brown": ["màu nâu", "nâu đất", "nâu bò", "nâu", "mau nau"],
    "Pink": ["màu hồng", "hồng phấn", "hồng", "mau hong"],
    "Silver": ["màu bạc", "bạc sáng", "ánh bạc", "bac"],
    "Beige": ["màu kem be", "màu be", "kem", "be"],
    "Yellow": ["màu vàng", "vàng tươi", "vàng"],
    "Purple": ["màu tím", "tím mộng mơ", "tím"],
    "Maroon": ["đỏ đô", "đỏ mận", "đỏ bordeaux"],
}

FILLER_PREFIXES = [
    "", "", "", "shop ơi có ", "cần tìm ", "cho mình hỏi ", "tìm mua ", "tư vấn giúp ", "muốn kiếm ",
]
FILLER_SUFFIXES = [
    "", "", "", " mẫu mới nhất", " đi chơi", " đi làm", " mặc hàng ngày", " đẹp xịn", " giá tốt", " trẻ trung", " lịch sự",
]


def count_rel(spec, id_map, strict_cats):
    return sum(1 for m in id_map if is_item_relevant(spec, m, "hybrid", strict_cats))


def main():
    rng = random.Random(42)

    with open(PROJECT_ROOT / "index/val_gallery/id_map.json", "r", encoding="utf-8") as f:
        val_id_map = json.load(f)

    policy_cfg = load_gender_gt_policy(PROJECT_ROOT / "eval/gender_gt_policy.json")
    strict_cats = policy_cfg["policies"]["hybrid"]["strict_categories"]

    # Lấy bảng tra từ Việt chuẩn từ catalog_rebuild cho Tập A
    df = pd.read_parquet(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    vi_type = df.dropna(subset=["articleType", "articleType_vi"]).drop_duplicates("articleType").set_index("articleType")["articleType_vi"].to_dict()
    vi_gender = df.dropna(subset=["gender", "gender_vi"]).drop_duplicates("gender").set_index("gender")["gender_vi"].to_dict()
    vi_colour = df.dropna(subset=["baseColour", "baseColour_vi"]).drop_duplicates("baseColour").set_index("baseColour")["baseColour_vi"].to_dict()

    # Tìm các nhóm (articleType, gender, baseColour) và (articleType, gender) có >= 5 ảnh trong val_gallery
    val_df = pd.DataFrame(val_id_map)
    groups_3 = val_df.groupby(["articleType", "gender", "baseColour"]).size().reset_index(name="cnt")
    groups_3 = groups_3[groups_3["cnt"] >= 5].sort_values("cnt", ascending=False)

    groups_2_g = val_df.groupby(["articleType", "gender"]).size().reset_index(name="cnt")
    groups_2_g = groups_2_g[groups_2_g["cnt"] >= 5].sort_values("cnt", ascending=False)

    groups_2_c = val_df.groupby(["articleType", "baseColour"]).size().reset_index(name="cnt")
    groups_2_c = groups_2_c[groups_2_c["cnt"] >= 5].sort_values("cnt", ascending=False)

    # ==========================================================
    # TẬP A: Synthetic (>= 200 queries, >= 40% không dấu, >= 5 rel)
    # ==========================================================
    val_a = []
    seen_q_a = set()

    templates_3 = [
        "{loai} {gioi} màu {mau}",
        "{loai} màu {mau} cho {gioi}",
        "tìm {loai} {gioi} màu {mau}",
    ]
    templates_2g = [
        "{loai} cho {gioi}",
        "{loai} {gioi}",
    ]

    # Sinh từ tất cả nhóm 3 thuộc tính có >= 5 ảnh
    for _, row in groups_3.iterrows():
        at, gd, cl = row["articleType"], row["gender"], row["baseColour"]
        loai_vi = vi_type.get(at, at).lower()
        gioi_vi = vi_gender.get(gd, gd).lower()
        mau_vi = vi_colour.get(cl, cl).lower()

        # Bỏ qua các tên chưa được Việt hóa rõ ràng nếu muốn câu tự nhiên, nhưng giữ đa số
        spec = {"articleType": [at], "gender": gd, "baseColour": [cl]}
        n_rel = count_rel(spec, val_id_map, strict_cats)
        if n_rel < 5:
            continue

        for tmpl in templates_3[:2]:
            q_vi = tmpl.format(loai=loai_vi, gioi=gioi_vi, mau=mau_vi).strip()
            if q_vi not in seen_q_a:
                seen_q_a.add(q_vi)
                val_a.append({
                    "id": f"val_a_{len(val_a):03d}",
                    "query": q_vi,
                    "has_diacritics": True,
                    "n_relevant": n_rel,
                    "spec": spec,
                })
            # Tạo biến thể không dấu
            q_nodiac = remove_diacritics(q_vi)
            if q_nodiac != q_vi and q_nodiac not in seen_q_a:
                seen_q_a.add(q_nodiac)
                val_a.append({
                    "id": f"val_a_{len(val_a):03d}",
                    "query": q_nodiac,
                    "has_diacritics": False,
                    "n_relevant": n_rel,
                    "spec": spec,
                })

    # Bổ sung từ nhóm 2 thuộc tính nếu cần thêm đa dạng
    for _, row in groups_2_g.head(40).iterrows():
        at, gd = row["articleType"], row["gender"]
        loai_vi = vi_type.get(at, at).lower()
        gioi_vi = vi_gender.get(gd, gd).lower()
        spec = {"articleType": [at], "gender": gd, "baseColour": []}
        n_rel = count_rel(spec, val_id_map, strict_cats)
        if n_rel < 5:
            continue
        q_vi = templates_2g[0].format(loai=loai_vi, gioi=gioi_vi).strip()
        if q_vi not in seen_q_a:
            seen_q_a.add(q_vi)
            val_a.append({
                "id": f"val_a_{len(val_a):03d}",
                "query": q_vi,
                "has_diacritics": True,
                "n_relevant": n_rel,
                "spec": spec,
            })
        q_nodiac = remove_diacritics(q_vi)
        if q_nodiac != q_vi and q_nodiac not in seen_q_a:
            seen_q_a.add(q_nodiac)
            val_a.append({
                "id": f"val_a_{len(val_a):03d}",
                "query": q_nodiac,
                "has_diacritics": False,
                "n_relevant": n_rel,
                "spec": spec,
            })

    rng.shuffle(val_a)
    val_a = val_a[:240]
    for idx, item in enumerate(val_a):
        item["id"] = f"val_a_{idx:03d}"

    n_nodiac_a = sum(1 for x in val_a if not x["has_diacritics"])
    print(f"Tập Val A (Synthetic): {len(val_a)} queries, {n_nodiac_a} không dấu ({n_nodiac_a/len(val_a)*100:.1f}%)")

    # ==========================================================
    # TẬP B: Natural-noisy draft (>= 150 queries, viết tự nhiên ngoài lexicon)
    # ==========================================================
    val_b = []
    seen_q_b = set()

    # Sinh từ các nhóm hợp lệ có từ đồng nghĩa tự nhiên
    candidate_specs = []
    for _, row in groups_3.iterrows():
        at, gd, cl = row["articleType"], row["gender"], row["baseColour"]
        if at in NATURAL_SYNONYMS_MAP and gd in NATURAL_GENDER_WORDS and cl in NATURAL_COLOUR_WORDS:
            spec = {"articleType": [at], "gender": gd, "baseColour": [cl]}
            n_rel = count_rel(spec, val_id_map, strict_cats)
            if n_rel >= 5:
                candidate_specs.append((spec, n_rel))

    # Thêm các nhóm thiếu màu hoặc thiếu giới tính (thiếu slot tự nhiên)
    for _, row in groups_2_g.iterrows():
        at, gd = row["articleType"], row["gender"]
        if at in NATURAL_SYNONYMS_MAP and gd in NATURAL_GENDER_WORDS:
            spec = {"articleType": [at], "gender": gd, "baseColour": []}
            n_rel = count_rel(spec, val_id_map, strict_cats)
            if n_rel >= 5:
                candidate_specs.append((spec, n_rel))

    for _, row in groups_2_c.iterrows():
        at, cl = row["articleType"], row["baseColour"]
        if at in NATURAL_SYNONYMS_MAP and cl in NATURAL_COLOUR_WORDS:
            spec = {"articleType": [at], "gender": None, "baseColour": [cl]}
            n_rel = count_rel(spec, val_id_map, strict_cats)
            if n_rel >= 5:
                candidate_specs.append((spec, n_rel))

    rng.shuffle(candidate_specs)

    attempts = 0
    while len(val_b) < 165 and attempts < 2000:
        attempts += 1
        spec, n_rel = candidate_specs[attempts % len(candidate_specs)]
        at = spec["articleType"][0]
        gd = spec["gender"]
        cl = spec["baseColour"][0] if spec["baseColour"] else None

        syn_at = rng.choice(NATURAL_SYNONYMS_MAP[at])
        syn_gd = rng.choice(NATURAL_GENDER_WORDS[gd]) if gd else ""
        syn_cl = rng.choice(NATURAL_COLOUR_WORDS[cl]) if cl else ""

        pref = rng.choice(FILLER_PREFIXES)
        suff = rng.choice(FILLER_SUFFIXES)

        parts = [p for p in [syn_at, syn_gd, syn_cl] if p]
        core = " ".join(parts)
        q_text = f"{pref}{core}{suff}".strip()
        q_text = " ".join(q_text.split())

        # Quyết định giữ nguyên dấu hay bỏ dấu toàn bộ / một phần
        r_mode = rng.random()
        if r_mode < 0.45:
            q_text = remove_diacritics(q_text)
            has_d = False
        else:
            # Kiểm tra xem bản thân syn_at có dấu không
            has_d = (remove_diacritics(q_text) != q_text)

        if q_text in seen_q_b:
            continue
        seen_q_b.add(q_text)
        val_b.append({
            "id": f"val_b_{len(val_b):03d}",
            "query": q_text,
            "has_diacritics": has_d,
            "n_relevant": n_rel,
            "spec": spec,
            "source": "agent_natural_noisy_draft",
        })

    n_nodiac_b = sum(1 for x in val_b if not x["has_diacritics"])
    print(f"Tập Val B (Natural-noisy draft): {len(val_b)} queries, {n_nodiac_b} không dấu ({n_nodiac_b/len(val_b)*100:.1f}%)")

    # ==========================================================
    # TẬP C: Parse-error probe (parser hiện tại SAI hoặc THIẾU articleType)
    # ==========================================================
    parser = VietnameseQueryParser()
    val_c = []
    for item in val_b:
        p = parser.parse(item["query"])
        p_types = set(p.get("articleType") or [])
        s_types = set(item["spec"].get("articleType") or [])
        # Chọn các câu mà parser thiếu hoặc sai articleType
        if not p_types or not (p_types & s_types):
            c_item = dict(item)
            c_item["id"] = f"val_c_{len(val_c):03d}"
            c_item["parser_baseline_articleType"] = list(p_types)
            c_item["probe_reason"] = "missing_articleType" if not p_types else "wrong_articleType"
            val_c.append(c_item)

    print(f"Tập Val C (Parse-error probe draft): {len(val_c)} queries (từ B có lỗi parser ở articleType)")

    # Lưu 3 tệp
    p_a = PROJECT_ROOT / "eval/val_queries_a_synthetic.json"
    p_b = PROJECT_ROOT / "eval/val_queries_b_natural_draft.json"
    p_c = PROJECT_ROOT / "eval/val_queries_c_probe_draft.json"

    for p, data in [(p_a, val_a), (p_b, val_b), (p_c, val_c)]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"Đã lưu {p.name} ({len(data)} câu) | SHA-256: {h}")


if __name__ == "__main__":
    main()
