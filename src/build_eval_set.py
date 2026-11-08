"""
Script to build:
1. eval/families.json (Color and category equivalence mappings)
2. eval/testset_v1.json (Fixed evaluation set: Q1 paired, Q2 attribute queries, Q4 image-to-image)
3. eval/handwritten_queries.json (100 diverse natural Vietnamese handwritten queries)
4. Update reports/run_manifest.json with testset sha256.
"""
import os
import sys
import json
import hashlib
import pathlib
import pandas as pd
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


def build_families():
    families = {
        "colour_families": {
            "Blue": ["Blue", "Navy Blue", "Teal", "Turquoise Blue"],
            "Black": ["Black", "Charcoal"],
            "White": ["White", "Off White", "Cream"],
            "Grey": ["Grey", "Grey Melange", "Steel"],
            "Brown": ["Brown", "Tan", "Bronze", "Copper", "Coffee Brown", "Mushroom Brown", "Taupe"],
            "Red": ["Red", "Maroon", "Burgundy", "Rust"],
            "Green": ["Green", "Olive", "Sea Green", "Lime Green", "Fluorescent Green"],
            "Pink": ["Pink", "Peach", "Rose", "Magenta"],
            "Purple": ["Purple", "Lavender", "Mauve"],
            "Yellow": ["Yellow", "Mustard", "Gold"],
            "Beige": ["Beige", "Khaki", "Nude", "Skin"],
            "Orange": ["Orange"],
            "Silver": ["Silver", "Metallic"],
            "Multi": ["Multi"]
        },
        "article_type_families": {
            "Tshirts": ["Tshirts", "Lounge Tshirts", "Tops"],
            "Shirts": ["Shirts"],
            "Tops": ["Tops", "Tshirts", "Tunics", "Shrug"],
            "Casual Shoes": ["Casual Shoes", "Sports Shoes"],
            "Sports Shoes": ["Sports Shoes", "Casual Shoes"],
            "Formal Shoes": ["Formal Shoes"],
            "Sandals": ["Sandals", "Flip Flops", "Sports Sandals"],
            "Heels": ["Heels", "Flats"],
            "Flats": ["Flats", "Heels"],
            "Jeans": ["Jeans", "Jeggings"],
            "Trousers": ["Trousers", "Track Pants", "Lounge Pants", "Churidar", "Salwar"],
            "Kurtas": ["Kurtas", "Kurtis", "Kurta Sets"],
            "Sarees": ["Sarees", "Lehenga Choli"],
            "Watches": ["Watches"],
            "Handbags": ["Handbags", "Clutches", "Messenger Bag"],
            "Wallets": ["Wallets", "Clutches"],
            "Backpacks": ["Backpacks", "Rucksacks", "Duffel Bag"],
            "Sunglasses": ["Sunglasses"]
        },
        "gender_compatibility": {
            "Men": ["Men", "Unisex"],
            "Women": ["Women", "Unisex"],
            "Unisex": ["Men", "Women", "Unisex"],
            "Boys": ["Boys", "Unisex"],
            "Girls": ["Girls", "Unisex"]
        },
        "ambiguous_groupings_for_review": [
            {
                "group": "Tshirts vs Tops",
                "reason": "Tops in Myntra is mostly women's fashion tops; Tshirts includes round-neck/polo. In family mode, Women's Tshirts match Tops."
            },
            {
                "group": "Casual Shoes vs Sports Shoes",
                "reason": "Sneakers / running shoes are often cross-tagged between Casual and Sports Shoes. In family mode, they match."
            },
            {
                "group": "Wallets vs Clutches",
                "reason": "Long women's wallets and small evening clutches overlap in design and usage. In family mode for Women, they match."
            },
            {
                "group": "Kurtas vs Kurtis",
                "reason": "Kurtis are short Kurtas; grouped under traditional Indian tunics."
            },
            {
                "group": "Teal / Turquoise Blue in Blue family",
                "reason": "Cyan/teal tones have blue and green components; assigned to Blue family."
            },
            {
                "group": "Khaki in Beige family vs Tan in Brown family",
                "reason": "Khaki is yellowish-beige; Tan is reddish-light brown."
            }
        ]
    }

    out_path = pathlib.Path("eval/families.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(families, f, indent=2, ensure_ascii=False)
    print(f"Saved families mapping to {out_path}")
    return families


def get_vietnamese_attribute_query(art_type: str, gender: str, colour: str) -> str:
    art_map = {
        "Tshirts": "áo thun",
        "Shirts": "áo sơ mi",
        "Watches": "đồng hồ",
        "Casual Shoes": "giày thường ngày",
        "Kurtas": "áo kurta",
        "Handbags": "túi xách tay",
        "Sports Shoes": "giày thể thao",
        "Backpacks": "balo",
        "Wallets": "ví tiền",
        "Tops": "áo kiểu",
        "Heels": "giày cao gót",
        "Sandals": "giày sandal",
        "Formal Shoes": "giày tây",
        "Jeans": "quần jean",
        "Sarees": "áo saree",
        "Sunglasses": "kính râm"
    }
    gender_map = {
        "Men": "nam",
        "Women": "nữ",
        "Boys": "bé trai",
        "Girls": "bé gái",
        "Unisex": "unisex"
    }
    colour_map = {
        "Black": "đen",
        "Red": "đỏ",
        "White": "trắng",
        "Blue": "xanh dương",
        "Grey": "xám",
        "Purple": "tím",
        "Green": "xanh lá",
        "Navy Blue": "xanh navy",
        "Brown": "nâu",
        "Silver": "bạc",
        "Pink": "hồng",
        "Yellow": "vàng",
        "Cream": "kem",
        "Maroon": "đỏ hạt dẻ",
        "Olive": "xanh ô liu"
    }

    vi_art = art_map.get(art_type, art_type.lower())
    vi_gender = gender_map.get(gender, gender.lower())
    vi_col = colour_map.get(colour, colour.lower())

    # Format: "{loại} {giới} màu {màu}"
    return f"{vi_art} {vi_gender} màu {vi_col}"


def build_testset():
    cat = pd.read_parquet("data/catalog/catalog.parquet")
    g_test = cat[(cat["split"] == "test") & (cat["in_scope"] == True)]
    g_full = cat[cat["in_scope"] == True]

    print(f"Loaded catalog: G_test={len(g_test):,}, G_full={len(g_full):,}")

    with open("data/checkpoints_siglip-vit/dense_captions_catalog.json", "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    # 1. Q1: Paired text -> image
    q1_queries = []
    caption_counts = Counter()
    for _, row in g_test.iterrows():
        item_id = str(row["item_id"])
        cap = captions_dict.get(item_id, "")
        caption_counts[cap] += 1
        q1_queries.append({
            "query_id": f"q1_{item_id}",
            "item_id": item_id,
            "query_text": cap,
            "target_id": item_id
        })

    collision_count = sum(1 for c in caption_counts.values() if c > 1)
    print(f"Q1 Paired Queries: {len(q1_queries)} queries. Caption collisions: {collision_count}")

    # 2. Q2: Attribute Queries
    # Combinations with >= 5 items in G_test
    groups = g_test.groupby(["articleType", "gender", "baseColour"]).size().reset_index(name="test_count")
    combo_ge5 = groups[groups["test_count"] >= 5].sort_values(by=["articleType", "test_count"], ascending=[True, False])

    q2_queries = []
    # Build fast lookup indices for target_ids
    # Pre-index G_test and G_full
    g_test_records = g_test[["item_id", "articleType", "gender", "baseColour"]].to_dict("records")
    g_full_records = g_full[["item_id", "articleType", "gender", "baseColour"]].to_dict("records")

    with open("eval/families.json", "r", encoding="utf-8") as f:
        families = json.load(f)

    colour_fam = families["colour_families"]
    art_fam = families["article_type_families"]
    gender_compat = families["gender_compatibility"]

    # Invert colour and art families for lookup
    col_to_fam = {}
    for fam_name, members in colour_fam.items():
        for m in members:
            col_to_fam[m] = fam_name

    art_to_fam = {}
    for fam_name, members in art_fam.items():
        for m in members:
            art_to_fam.setdefault(m, set()).update(members)

    for idx, row in combo_ge5.iterrows():
        art = row["articleType"]
        gen = row["gender"]
        col = row["baseColour"]
        query_text = get_vietnamese_attribute_query(art, gen, col)

        compat_genders = set(gender_compat.get(gen, [gen]))

        # Strict targets: exact articleType, compatible gender, exact baseColour
        strict_g_test = [
            str(r["item_id"]) for r in g_test_records
            if r["articleType"] == art and r["gender"] in compat_genders and r["baseColour"] == col
        ]
        strict_g_full = [
            str(r["item_id"]) for r in g_full_records
            if r["articleType"] == art and r["gender"] in compat_genders and r["baseColour"] == col
        ]

        # Family targets: family articleType, compatible gender, family baseColour
        fam_cols = set(colour_fam.get(col_to_fam.get(col, col), [col]))
        fam_arts = art_to_fam.get(art, {art})

        family_g_test = [
            str(r["item_id"]) for r in g_test_records
            if r["articleType"] in fam_arts and r["gender"] in compat_genders and r["baseColour"] in fam_cols
        ]
        family_g_full = [
            str(r["item_id"]) for r in g_full_records
            if r["articleType"] in fam_arts and r["gender"] in compat_genders and r["baseColour"] in fam_cols
        ]

        q2_queries.append({
            "query_id": f"q2_{idx+1:03d}",
            "query_text": query_text,
            "target_attributes": {
                "articleType": art,
                "gender": gen,
                "baseColour": col
            },
            "target_ids_strict": {
                "g_test": strict_g_test,
                "g_full": strict_g_full
            },
            "target_ids_family": {
                "g_test": family_g_test,
                "g_full": family_g_full
            }
        })

    print(f"Q2 Attribute Queries: {len(q2_queries)} queries generated across 16 articleTypes.")

    # 3. Q4: Image-to-Image Queries
    q4_queries = []
    for _, row in g_test.iterrows():
        item_id = str(row["item_id"])
        img_path = str(row["path"])
        q4_queries.append({
            "query_id": f"q4_{item_id}",
            "item_id": item_id,
            "image_path": img_path,
            "target_id": item_id,
            "evaluation_note": "Self-match target. Rank-1 is expected to be exact query image if unmasked."
        })

    testset = {
        "metadata": {
            "version": "v1",
            "seed": 42,
            "n_g_test": len(g_test),
            "n_g_full": len(g_full),
            "n_q1_paired": len(q1_queries),
            "n_q2_attribute": len(q2_queries),
            "n_q4_image": len(q4_queries),
            "q1_caption_collisions": collision_count
        },
        "g_test_ids": [str(x) for x in g_test["item_id"].tolist()],
        "g_full_ids": [str(x) for x in g_full["item_id"].tolist()],
        "q1_paired_queries": q1_queries,
        "q2_attribute_queries": q2_queries,
        "q4_image_queries": q4_queries
    }

    out_file = pathlib.Path("eval/testset_v1.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    testset_sha = compute_sha256(str(out_file))
    print(f"Saved testset to {out_file} (SHA256: {testset_sha})")

    # Update reports/run_manifest.json
    manifest_path = pathlib.Path("reports/run_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            run_manifest = json.load(f)
    else:
        run_manifest = {}

    run_manifest["eval_testset_v1_sha256"] = testset_sha
    run_manifest["eval_testset_v1_path"] = "eval/testset_v1.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2, ensure_ascii=False)
    print(f"Updated {manifest_path} with testset sha256.")

    return testset_sha


def build_handwritten_queries():
    # 100 queries:
    # Categories:
    # - Tops / Áo (25 queries): 19 accented, 6 unaccented
    # - Bottoms / Quần & Chân váy (20 queries): 15 accented, 5 unaccented
    # - Footwear / Giày dép (20 queries): 15 accented, 5 unaccented
    # - Bags / Túi ví & Balo (15 queries): 11 accented, 4 unaccented
    # - Watches / Đồng hồ (10 queries): 7 accented, 3 unaccented
    # - Accessories / Phụ kiện (10 queries): 8 accented, 2 unaccented
    # Total = 25 + 20 + 20 + 15 + 10 + 10 = 100 queries.
    # Total unaccented = 6 + 5 + 5 + 4 + 3 + 2 = 25 queries (exactly 25%)!

    queries = [
        # --- NHÓM 1: ÁO (TOPS / SHIRTS / TSHIRTS / JACKETS) - 25 CÂU ---
        {"id": "h001", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo thun nam màu trắng trơn cổ tròn", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["White"]}, "note": "Áo thun nam trắng cơ bản"},
        {"id": "h002", "category": "Áo", "has_diacritics": False, "style": "no_accent", "query": "ao thun nam trang", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["White"]}, "note": "Không dấu ngắn gọn"},
        {"id": "h003", "category": "Áo", "has_diacritics": True, "style": "context", "query": "áo sơ mi nam trắng công sở đi làm", "spec": {"articleType": ["Shirts"], "gender": "Men", "baseColour": ["White"]}, "note": "Có từ ngữ cảnh 'công sở đi làm'"},
        {"id": "h004", "category": "Áo", "has_diacritics": False, "style": "no_accent", "query": "ao so mi nam trang dai tay", "spec": {"articleType": ["Shirts"], "gender": "Men", "baseColour": ["White"]}, "note": "Không dấu có từ mô tả dài tay"},
        {"id": "h005", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo polo nam màu xanh navy thể thao", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Navy Blue", "Blue"]}, "note": "Áo polo thể thao"},
        {"id": "h006", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo phông nam màu đen in họa tiết", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}, "note": "Từ ngữ địa phương 'áo phông'"},
        {"id": "h007", "category": "Áo", "has_diacritics": False, "style": "no_accent", "query": "ao phong den nam", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Black"]}, "note": "Không dấu, đảo ngữ"},
        {"id": "h008", "category": "Áo", "has_diacritics": True, "style": "context", "query": "áo thun nữ mùa hè màu hồng dễ thương", "spec": {"articleType": ["Tshirts", "Tops"], "gender": "Women", "baseColour": ["Pink"]}, "note": "Ngữ cảnh mùa hè, tính từ cảm xúc"},
        {"id": "h009", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo sơ mi nữ kẻ sọc màu xanh dương", "spec": {"articleType": ["Shirts", "Tops"], "gender": "Women", "baseColour": ["Blue"]}, "note": "Họa tiết kẻ sọc"},
        {"id": "h010", "category": "Áo", "has_diacritics": False, "style": "no_accent", "query": "ao so mi nu cong so", "spec": {"articleType": ["Shirts", "Tops"], "gender": "Women", "baseColour": []}, "note": "Không dấu, thuộc tính công sở"},
        {"id": "h011", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo khoác jacket nam màu đen có khóa kéo", "spec": {"articleType": ["Jackets"], "gender": "Men", "baseColour": ["Black"]}, "note": "Áo khoác nam"},
        {"id": "h012", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo len cổ tròn nữ màu đỏ ấm áp", "spec": {"articleType": ["Sweaters"], "gender": "Women", "baseColour": ["Red"]}, "note": "Áo len nữ màu đỏ"},
        {"id": "h013", "category": "Áo", "has_diacritics": False, "style": "no_accent", "query": "ao len nu mau do", "spec": {"articleType": ["Sweaters"], "gender": "Women", "baseColour": ["Red"]}, "note": "Không dấu"},
        {"id": "h014", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo hoodie nỉ nam màu xám", "spec": {"articleType": ["Sweatshirts"], "gender": "Men", "baseColour": ["Grey"]}, "note": "Áo nỉ chui đầu"},
        {"id": "h015", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo kiểu nữ sát nách màu vàng", "spec": {"articleType": ["Tops"], "gender": "Women", "baseColour": ["Yellow"]}, "note": "Áo kiểu nữ"},
        {"id": "h016", "category": "Áo", "has_diacritics": True, "style": "context", "query": "áo blazer nam công sở lịch lãm màu đen", "spec": {"articleType": ["Blazers", "Jackets"], "gender": "Men", "baseColour": ["Black"]}, "note": "Áo vest blazer nam"},
        {"id": "h017", "category": "Áo", "has_diacritics": False, "style": "no_accent", "query": "ao blazer nam den", "spec": {"articleType": ["Blazers", "Jackets"], "gender": "Men", "baseColour": ["Black"]}, "note": "Không dấu ngắn gọn"},
        {"id": "h018", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo thun nam màu đỏ đô thể thao", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Red", "Maroon"]}, "note": "Màu đỏ đô / đỏ hạt dẻ"},
        {"id": "h019", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo thun tay dài nam màu xám tiêu", "spec": {"articleType": ["Tshirts"], "gender": "Men", "baseColour": ["Grey", "Grey Melange"]}, "note": "Chi tiết tay dài xám tiêu"},
        {"id": "h020", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo sơ mi nam cộc tay đi biển màu xanh lá", "spec": {"articleType": ["Shirts"], "gender": "Men", "baseColour": ["Green"]}, "note": "Ngữ cảnh đi biển, cộc tay"},
        {"id": "h021", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo thun nữ form rộng màu trắng", "spec": {"articleType": ["Tshirts", "Tops"], "gender": "Women", "baseColour": ["White"]}, "note": "Dáng form rộng"},
        {"id": "h022", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo sơ mi nam caro tím trẻ trung", "spec": {"articleType": ["Shirts"], "gender": "Men", "baseColour": ["Purple"]}, "note": "Họa tiết caro tím"},
        {"id": "h023", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo sát nách thể thao nam tập gym", "spec": {"articleType": ["Innerwear Vests", "Tshirts"], "gender": "Men", "baseColour": []}, "note": "Áo may ô tập gym"},
        {"id": "h024", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo khoác gió thể thao nam chống nước", "spec": {"articleType": ["Jackets", "Rain Jacket"], "gender": "Men", "baseColour": []}, "note": "Áo khoác gió"},
        {"id": "h025", "category": "Áo", "has_diacritics": True, "style": "natural", "query": "áo thun bé trai in hình siêu nhân", "spec": {"articleType": ["Tshirts"], "gender": "Boys", "baseColour": []}, "note": "Áo thun trẻ em bé trai"},

        # --- NHÓM 2: QUẦN & CHÂN VÁY & ĐẦM (BOTTOMS / DRESSES) - 20 CÂU ---
        {"id": "h026", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần jean nam ống đứng màu xanh dương", "spec": {"articleType": ["Jeans"], "gender": "Men", "baseColour": ["Blue"]}, "note": "Quần jean ống đứng"},
        {"id": "h027", "category": "Quần & Váy", "has_diacritics": False, "style": "no_accent", "query": "quan jean nam xanh", "spec": {"articleType": ["Jeans"], "gender": "Men", "baseColour": ["Blue"]}, "note": "Không dấu ngắn gọn"},
        {"id": "h028", "category": "Quần & Váy", "has_diacritics": True, "style": "colloquial", "query": "quần bò nam rách gối cá tính", "spec": {"articleType": ["Jeans"], "gender": "Men", "baseColour": ["Blue", "Black"]}, "note": "Từ địa phương 'quần bò'"},
        {"id": "h029", "category": "Quần & Váy", "has_diacritics": True, "style": "context", "query": "quần tây nam đi làm công sở màu đen", "spec": {"articleType": ["Trousers"], "gender": "Men", "baseColour": ["Black"]}, "note": "Quần âu tây công sở"},
        {"id": "h030", "category": "Quần & Váy", "has_diacritics": False, "style": "no_accent", "query": "quan tay nam cong so mau den", "spec": {"articleType": ["Trousers"], "gender": "Men", "baseColour": ["Black"]}, "note": "Không dấu"},
        {"id": "h031", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần short nam đi biển mùa hè thoáng mát", "spec": {"articleType": ["Shorts"], "gender": "Men", "baseColour": []}, "note": "Quần đùi short nam"},
        {"id": "h032", "category": "Quần & Váy", "has_diacritics": False, "style": "no_accent", "query": "quan short nam mau trang", "spec": {"articleType": ["Shorts"], "gender": "Men", "baseColour": ["White"]}, "note": "Không dấu"},
        {"id": "h033", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần jogger thể thao bo gấu màu xám", "spec": {"articleType": ["Track Pants"], "gender": "Men", "baseColour": ["Grey"]}, "note": "Quần nỉ thể thao jogger"},
        {"id": "h034", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần legging nữ ôm sát màu đen", "spec": {"articleType": ["Leggings", "Tights"], "gender": "Women", "baseColour": ["Black"]}, "note": "Quần legging ôm"},
        {"id": "h035", "category": "Quần & Váy", "has_diacritics": False, "style": "no_accent", "query": "quan legging nu den", "spec": {"articleType": ["Leggings", "Tights"], "gender": "Women", "baseColour": ["Black"]}, "note": "Không dấu"},
        {"id": "h036", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "chân váy chữ a dáng ngắn màu đen", "spec": {"articleType": ["Skirts"], "gender": "Women", "baseColour": ["Black"]}, "note": "Chân váy chữ A"},
        {"id": "h037", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "chân váy xòe hoa nhí dễ thương", "spec": {"articleType": ["Skirts"], "gender": "Women", "baseColour": []}, "note": "Chân váy xòe"},
        {"id": "h038", "category": "Quần & Váy", "has_diacritics": True, "style": "context", "query": "váy đầm dự tiệc sang trọng màu đỏ", "spec": {"articleType": ["Dresses"], "gender": "Women", "baseColour": ["Red"]}, "note": "Đầm dạ hội dự tiệc"},
        {"id": "h039", "category": "Quần & Váy", "has_diacritics": False, "style": "no_accent", "query": "dam du tiec mau do", "spec": {"articleType": ["Dresses"], "gender": "Women", "baseColour": ["Red"]}, "note": "Không dấu"},
        {"id": "h040", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "đầm suông dạo phố mùa hè màu trắng", "spec": {"articleType": ["Dresses"], "gender": "Women", "baseColour": ["White"]}, "note": "Đầm suông trắng"},
        {"id": "h041", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần kaki nam màu be dáng ôm", "spec": {"articleType": ["Trousers"], "gender": "Men", "baseColour": ["Beige", "Khaki"]}, "note": "Quần kaki be"},
        {"id": "h042", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần lót boxer nam co giãn tốt", "spec": {"articleType": ["Boxers", "Trunk"], "gender": "Men", "baseColour": []}, "note": "Đồ lót nam"},
        {"id": "h043", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "bộ đồ ngủ mặc nhà nữ dễ thương", "spec": {"articleType": ["Night suits", "Nightdress"], "gender": "Women", "baseColour": []}, "note": "Đồ ngủ mặc nhà"},
        {"id": "h044", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "quần jean nữ cạp cao ống loe", "spec": {"articleType": ["Jeans"], "gender": "Women", "baseColour": ["Blue"]}, "note": "Jean nữ cạp cao"},
        {"id": "h045", "category": "Quần & Váy", "has_diacritics": True, "style": "natural", "query": "áo liền quần jumpsuit nữ thanh lịch", "spec": {"articleType": ["Jumpsuit"], "gender": "Women", "baseColour": []}, "note": "Đồ bay jumpsuit"},

        # --- NHÓM 3: GIÀY DÉP (FOOTWEAR) - 20 CÂU ---
        {"id": "h046", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày thể thao nam chạy bộ màu trắng", "spec": {"articleType": ["Sports Shoes"], "gender": "Men", "baseColour": ["White"]}, "note": "Giày chạy bộ trắng"},
        {"id": "h047", "category": "Giày dép", "has_diacritics": False, "style": "no_accent", "query": "giay the thao nam trang", "spec": {"articleType": ["Sports Shoes"], "gender": "Men", "baseColour": ["White"]}, "note": "Không dấu ngắn gọn"},
        {"id": "h048", "category": "Giày dép", "has_diacritics": True, "style": "colloquial", "query": "giày sneaker nam cổ thấp màu đen", "spec": {"articleType": ["Casual Shoes", "Sports Shoes"], "gender": "Men", "baseColour": ["Black"]}, "note": "Từ mượn 'sneaker'"},
        {"id": "h049", "category": "Giày dép", "has_diacritics": True, "style": "context", "query": "giày tây nam da bóng buộc dây công sở", "spec": {"articleType": ["Formal Shoes"], "gender": "Men", "baseColour": ["Black", "Brown"]}, "note": "Giày tây công sở"},
        {"id": "h050", "category": "Giày dép", "has_diacritics": False, "style": "no_accent", "query": "giay tay nam da bo mau nau", "spec": {"articleType": ["Formal Shoes"], "gender": "Men", "baseColour": ["Brown"]}, "note": "Không dấu giày da nâu"},
        {"id": "h051", "category": "Giày dép", "has_diacritics": True, "style": "colloquial", "query": "giày lười nam da mềm màu nâu", "spec": {"articleType": ["Casual Shoes"], "gender": "Men", "baseColour": ["Brown"]}, "note": "Giày lười loafer"},
        {"id": "h052", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày cao gót nữ mũi nhọn màu đen 7cm", "spec": {"articleType": ["Heels"], "gender": "Women", "baseColour": ["Black"]}, "note": "Cao gót mũi nhọn"},
        {"id": "h053", "category": "Giày dép", "has_diacritics": False, "style": "no_accent", "query": "giay cao got nu mau den", "spec": {"articleType": ["Heels"], "gender": "Women", "baseColour": ["Black"]}, "note": "Không dấu cao gót"},
        {"id": "h054", "category": "Giày dép", "has_diacritics": True, "style": "context", "query": "giày cao gót nữ dự tiệc màu đỏ quyến rũ", "spec": {"articleType": ["Heels"], "gender": "Women", "baseColour": ["Red"]}, "note": "Cao gót dự tiệc đỏ"},
        {"id": "h055", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày búp bê đế bệt nữ màu kem", "spec": {"articleType": ["Flats"], "gender": "Women", "baseColour": ["Cream", "Beige", "White"]}, "note": "Giày búp bê bệt"},
        {"id": "h056", "category": "Giày dép", "has_diacritics": False, "style": "no_accent", "query": "giay bup be de bet nu", "spec": {"articleType": ["Flats"], "gender": "Women", "baseColour": []}, "note": "Không dấu"},
        {"id": "h057", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "dép quai hậu sandal nam đi học đi chơi", "spec": {"articleType": ["Sandals"], "gender": "Men", "baseColour": []}, "note": "Dép quai hậu sandal"},
        {"id": "h058", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "sandal nữ quai mảnh màu trắng thanh lịch", "spec": {"articleType": ["Sandals"], "gender": "Women", "baseColour": ["White"]}, "note": "Sandal nữ quai mảnh"},
        {"id": "h059", "category": "Giày dép", "has_diacritics": True, "style": "colloquial", "query": "dép kẹp xỏ ngón đi biển mùa hè", "spec": {"articleType": ["Flip Flops"], "gender": "Unisex", "baseColour": []}, "note": "Dép kẹp xỏ ngón"},
        {"id": "h060", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày thể thao nữ tập aerobic màu tím", "spec": {"articleType": ["Sports Shoes"], "gender": "Women", "baseColour": ["Purple"]}, "note": "Giày thể thao nữ tím"},
        {"id": "h061", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày vải nam cổ thấp màu xanh navy", "spec": {"articleType": ["Casual Shoes"], "gender": "Men", "baseColour": ["Navy Blue", "Blue"]}, "note": "Giày vải canvas"},
        {"id": "h062", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày bốt da nữ cổ thấp cá tính", "spec": {"articleType": ["Booties"], "gender": "Women", "baseColour": ["Black", "Brown"]}, "note": "Giày bốt cổ thấp"},
        {"id": "h063", "category": "Giày dép", "has_diacritics": False, "style": "no_accent", "query": "dep sandal nam quai dan", "spec": {"articleType": ["Sports Sandals", "Sandals"], "gender": "Men", "baseColour": []}, "note": "Không dấu sandal quai dán"},
        {"id": "h064", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "giày đá bóng sân cỏ nhân tạo nam", "spec": {"articleType": ["Sports Shoes"], "gender": "Men", "baseColour": []}, "note": "Giày thể thao đá bóng"},
        {"id": "h065", "category": "Giày dép", "has_diacritics": True, "style": "natural", "query": "tất vớ cổ ngắn thể thao nam màu trắng", "spec": {"articleType": ["Socks"], "gender": "Men", "baseColour": ["White"]}, "note": "Vớ tất thể thao"},

        # --- NHÓM 4: TÚI VÍ & BALO (BAGS & WALLETS) - 15 CÂU ---
        {"id": "h066", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "balo đựng laptop chống sốc đi học màu đen", "spec": {"articleType": ["Backpacks", "Laptop Bag"], "gender": "Unisex", "baseColour": ["Black"]}, "note": "Balo laptop đen"},
        {"id": "h067", "category": "Túi & Balo", "has_diacritics": False, "style": "no_accent", "query": "balo laptop chong nuoc mau den", "spec": {"articleType": ["Backpacks", "Laptop Bag"], "gender": "Unisex", "baseColour": ["Black"]}, "note": "Không dấu"},
        {"id": "h068", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "balo du lịch dã ngoại màu xanh dương", "spec": {"articleType": ["Backpacks", "Rucksacks"], "gender": "Unisex", "baseColour": ["Blue"]}, "note": "Balo dã ngoại xanh"},
        {"id": "h069", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "túi xách tay nữ da thật màu nâu sang trọng", "spec": {"articleType": ["Handbags"], "gender": "Women", "baseColour": ["Brown"]}, "note": "Túi xách nữ nâu"},
        {"id": "h070", "category": "Túi & Balo", "has_diacritics": False, "style": "no_accent", "query": "tui xach nu da mau nau", "spec": {"articleType": ["Handbags"], "gender": "Women", "baseColour": ["Brown"]}, "note": "Không dấu túi xách"},
        {"id": "h071", "category": "Túi & Balo", "has_diacritics": True, "style": "context", "query": "túi đeo chéo nữ nhỏ gọn đi chơi dạo phố", "spec": {"articleType": ["Handbags", "Messenger Bag"], "gender": "Women", "baseColour": []}, "note": "Túi đeo chéo đi chơi"},
        {"id": "h072", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "ví da nam bỏ túi nhỏ gọn màu đen", "spec": {"articleType": ["Wallets"], "gender": "Men", "baseColour": ["Black"]}, "note": "Ví da nam đen"},
        {"id": "h073", "category": "Túi & Balo", "has_diacritics": False, "style": "no_accent", "query": "vi da nam mau den", "spec": {"articleType": ["Wallets"], "gender": "Men", "baseColour": ["Black"]}, "note": "Không dấu"},
        {"id": "h074", "category": "Túi & Balo", "has_diacritics": True, "style": "colloquial", "query": "bóp tiền nam da bò màu nâu gập đôi", "spec": {"articleType": ["Wallets"], "gender": "Men", "baseColour": ["Brown"]}, "note": "Từ phương ngữ 'bóp tiền'"},
        {"id": "h075", "category": "Túi & Balo", "has_diacritics": True, "style": "context", "query": "ví cầm tay nữ dự tiệc sang chảnh clutch", "spec": {"articleType": ["Clutches", "Wallets"], "gender": "Women", "baseColour": []}, "note": "Ví clutch dự tiệc"},
        {"id": "h076", "category": "Túi & Balo", "has_diacritics": False, "style": "no_accent", "query": "vi cam tay nu du tiec", "spec": {"articleType": ["Clutches", "Wallets"], "gender": "Women", "baseColour": []}, "note": "Không dấu"},
        {"id": "h077", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "túi đeo chéo nam đựng ipad thể thao", "spec": {"articleType": ["Messenger Bag"], "gender": "Men", "baseColour": []}, "note": "Túi đeo chéo nam"},
        {"id": "h078", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "túi trống thể thao tập gym đựng đồ", "spec": {"articleType": ["Duffel Bag"], "gender": "Unisex", "baseColour": []}, "note": "Túi trống du lịch gym"},
        {"id": "h079", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "túi xách công sở nữ màu đen đựng tài liệu", "spec": {"articleType": ["Handbags"], "gender": "Women", "baseColour": ["Black"]}, "note": "Túi xách công sở"},
        {"id": "h080", "category": "Túi & Balo", "has_diacritics": True, "style": "natural", "query": "túi bao tử đeo hông cá tính", "spec": {"articleType": ["Waist Pouch"], "gender": "Unisex", "baseColour": []}, "note": "Túi bao tử đeo hông"},

        # --- NHÓM 5: ĐỒNG HỒ (WATCHES) - 10 CÂU ---
        {"id": "h081", "category": "Đồng hồ", "has_diacritics": True, "style": "natural", "query": "đồng hồ đeo tay nam mặt tròn dây da màu đen", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": ["Black"]}, "note": "Đồng hồ nam dây da đen"},
        {"id": "h082", "category": "Đồng hồ", "has_diacritics": False, "style": "no_accent", "query": "dong ho nam day da mau den", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": ["Black"]}, "note": "Không dấu"},
        {"id": "h083", "category": "Đồng hồ", "has_diacritics": True, "style": "natural", "query": "đồng hồ nam kim loại thép không gỉ màu bạc", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": ["Silver", "Steel"]}, "note": "Đồng hồ nam dây thép bạc"},
        {"id": "h084", "category": "Đồng hồ", "has_diacritics": False, "style": "no_accent", "query": "dong ho nam day kim loai", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": ["Silver", "Steel"]}, "note": "Không dấu"},
        {"id": "h085", "category": "Đồng hồ", "has_diacritics": True, "style": "natural", "query": "đồng hồ nữ mặt nhỏ dây kim loại màu vàng hồng", "spec": {"articleType": ["Watches"], "gender": "Women", "baseColour": ["Gold", "Silver", "Pink"]}, "note": "Đồng hồ nữ mặt nhỏ"},
        {"id": "h086", "category": "Đồng hồ", "has_diacritics": False, "style": "no_accent", "query": "dong ho nu mau trang thanh lich", "spec": {"articleType": ["Watches"], "gender": "Women", "baseColour": ["White"]}, "note": "Không dấu"},
        {"id": "h087", "category": "Đồng hồ", "has_diacritics": True, "style": "context", "query": "đồng hồ nam phong cách thể thao chống nước", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": ["Black"]}, "note": "Đồng hồ thể thao nam"},
        {"id": "h088", "category": "Đồng hồ", "has_diacritics": True, "style": "natural", "query": "đồng hồ thời trang nữ mặt số đính đá màu bạc", "spec": {"articleType": ["Watches"], "gender": "Women", "baseColour": ["Silver"]}, "note": "Đồng hồ nữ đính đá"},
        {"id": "h089", "category": "Đồng hồ", "has_diacritics": True, "style": "natural", "query": "đồng hồ nam mặt vuông dây cao su", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": ["Black"]}, "note": "Đồng hồ nam mặt vuông"},
        {"id": "h090", "category": "Đồng hồ", "has_diacritics": True, "style": "context", "query": "đồng hồ công sở nam thanh lịch sang trọng", "spec": {"articleType": ["Watches"], "gender": "Men", "baseColour": []}, "note": "Đồng hồ công sở"},

        # --- NHÓM 6: PHỤ KIỆN KHÁC (ACCESSORIES) - 10 CÂU ---
        {"id": "h091", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "kính râm nam phi công mắt đen thời trang", "spec": {"articleType": ["Sunglasses"], "gender": "Men", "baseColour": ["Black"]}, "note": "Kính râm phi công"},
        {"id": "h092", "category": "Phụ kiện", "has_diacritics": False, "style": "no_accent", "query": "kinh ram nam den", "spec": {"articleType": ["Sunglasses"], "gender": "Men", "baseColour": ["Black"]}, "note": "Không dấu"},
        {"id": "h093", "category": "Phụ kiện", "has_diacritics": True, "style": "colloquial", "query": "kính mát nữ mắt mèo sành điệu", "spec": {"articleType": ["Sunglasses"], "gender": "Women", "baseColour": []}, "note": "Từ đồng nghĩa 'kính mát'"},
        {"id": "h094", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "thắt lưng da nam khóa kim màu đen", "spec": {"articleType": ["Belts"], "gender": "Men", "baseColour": ["Black"]}, "note": "Thắt lưng da nam"},
        {"id": "h095", "category": "Phụ kiện", "has_diacritics": False, "style": "no_accent", "query": "day nit nam da bo", "spec": {"articleType": ["Belts"], "gender": "Men", "baseColour": ["Brown", "Black"]}, "note": "Phương ngữ 'dây nịt', không dấu"},
        {"id": "h096", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "cà vạt nam lụa tơ tằm màu xanh navy lịch lãm", "spec": {"articleType": ["Ties"], "gender": "Men", "baseColour": ["Navy Blue", "Blue"]}, "note": "Cà vạt lụa xanh navy"},
        {"id": "h097", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "mũ lưỡi trai thể thao nam màu đen", "spec": {"articleType": ["Caps"], "gender": "Men", "baseColour": ["Black"]}, "note": "Mũ lưỡi trai"},
        {"id": "h098", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "khăn quàng cổ len mùa đông ấm áp cho nữ", "spec": {"articleType": ["Scarves", "Mufflers"], "gender": "Women", "baseColour": []}, "note": "Khăn quàng cổ len"},
        {"id": "h099", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "bông tai bạc nữ hình tròn đính đá lấp lánh", "spec": {"articleType": ["Earrings"], "gender": "Women", "baseColour": ["Silver"]}, "note": "Hoa tai bông tai"},
        {"id": "h100", "category": "Phụ kiện", "has_diacritics": True, "style": "natural", "query": "dây chuyền bạc nữ mặt trái tim tinh tế", "spec": {"articleType": ["Necklace and Chains", "Pendant"], "gender": "Women", "baseColour": ["Silver"]}, "note": "Dây chuyền bạc nữ"}
    ]

    out_file = pathlib.Path("eval/handwritten_queries.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Saved 100 handwritten queries to {out_file}")

    # Verify counts
    n_total = len(queries)
    n_no_accent = sum(1 for q in queries if not q["has_diacritics"])
    print(f"Total handwritten queries: {n_total}, Unaccented queries: {n_no_accent} ({n_no_accent/n_total*100:.1f}%)")

    cat_counts = Counter(q["category"] for q in queries)
    print("Breakdown by category:", dict(cat_counts))


if __name__ == "__main__":
    build_families()
    sha = build_testset()
    build_handwritten_queries()
