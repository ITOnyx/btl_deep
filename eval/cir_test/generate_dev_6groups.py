"""
Generate Independent Dev Suite (60 cases, 6 challenge groups matching Held-out taxonomy + Category Mutation cases)
Zero overlap in ref_id and modifier_text with cir_heldout.csv.
Used strictly for parameter tuning and ablation design without touching held-out.
"""
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

def main():
    df_catalog = pd.read_parquet(ROOT_DIR / "data" / "catalog" / "catalog.parquet")
    ho_df = pd.read_csv(ROOT_DIR / "eval" / "cir_test" / "cir_heldout.csv")
    ho_ids = set(ho_df["ref_id"].astype(str))
    ho_mods = set(ho_df["modifier_text"].dropna().astype(str).str.strip().str.lower())

    sub = df_catalog[(df_catalog["in_scope"] == True) & (~df_catalog["item_id"].astype(str).isin(ho_ids))].copy()
    sub = sub.sort_values(by=["split", "item_id"], ascending=[False, True]) # prefer 'test' deterministically

    types = [
        "Shirts", "Tshirts", "Casual Shoes", "Sports Shoes", "Handbags",
        "Dresses", "Skirts", "Trousers", "Shorts", "Jackets", "Watches", "Backpacks", "Sweaters"
    ]
    img_dir = ROOT_DIR / "data" / "fashion-dataset" / "images"

    def pick_item(art_type: str, idx_offset: int = 0):
        cands = sub[(sub["articleType"] == art_type) & (sub["split"] == "test")]
        if len(cands) <= idx_offset:
            cands = sub[sub["articleType"] == art_type]
        valid = []
        for _, r in cands.iterrows():
            p = str(r["path"])
            if Path(p).exists() or (img_dir / f"{r['item_id']}.jpg").exists():
                valid.append(r)
                if len(valid) > idx_offset:
                    return valid[idx_offset]
        return cands.iloc[min(idx_offset, len(cands) - 1)]

    ref_cat = {t: pick_item(t, 0) for t in types}
    ref_upl = {t: pick_item(t, 1) for t in types}

    cases = []

    # 1. synonym_natural (10 cases)
    syn_specs = [
        ("Shirts", "màu xanh than công sở", "Shirts", "Navy Blue", None, None, "A"),
        ("Tshirts", "sắc đen tuyền cá tính", "Tshirts", "Black", None, None, "A"),
        ("Casual Shoes", "màu nâu hạt dẻ", "Casual Shoes", "Brown", None, None, "A"),
        ("Sports Shoes", "tông trắng tinh", "Sports Shoes", "White", None, None, "A"),
        ("Handbags", "gam hồng phấn nữ tính", "Handbags", "Pink", None, None, "A"),
        ("Dresses", "màu xanh ngọc dịu dàng", "Dresses", "Turquoise Blue", None, None, "A"),
        ("Trousers", "màu xám tro lịch sự", "Trousers", "Grey", None, None, "A"),
        ("Jackets", "tông xanh navy đậm", "Jackets", "Navy Blue", None, None, "A"), # test 'đậm' false positive on Dresses!
        ("Watches", "màu vàng đồng sang trọng", "Watches", "Gold", None, None, "A"),
        ("Backpacks", "gam màu đỏ nổi bật", "Backpacks", "Red", None, None, "A"),
    ]
    for i, (t, mod, tt, tc, tg, tb, st) in enumerate(syn_specs, 1):
        r = ref_cat[t]
        cases.append((f"DEV_SYN_{i:03d}", "synonym_natural", str(r["item_id"]), t, r["baseColour"], r["gender"], mod, "cir_catalog", tt, tc, tg, tb, f"Dev synonym ({st})"))

    # 2. unaccented (10 cases)
    unacc_specs = [
        ("Shirts", "mau xanh duong nhat", "Shirts", "Blue", None, None, "A"),
        ("Tshirts", "ao thun mau trang cho nam", "Tshirts", "White", "Men", None, "A"),
        ("Handbags", "mau den cho nu", "Handbags", "Black", "Women", None, "A"),
        ("Casual Shoes", "mau xam ghi", "Casual Shoes", "Grey", None, None, "A"),
        ("Sports Shoes", "mau do cho nam", "Sports Shoes", "Red", "Men", None, "A"),
        ("Shorts", "mau xanh than", "Shorts", "Navy Blue", None, None, "A"),
        ("Skirts", "mau den thanh lich", "Skirts", "Black", None, None, "A"),
        ("Watches", "mau vang gold", "Watches", "Gold", None, None, "A"),
        ("Dresses", "chuyen sang chan vay mau den", "Skirts", "Black", None, None, "B"), # Mutation unaccented
        ("Shirts", "doi sang ao khoac mau den", "Jackets", "Black", None, None, "B"),   # Mutation unaccented
    ]
    for i, (t, mod, tt, tc, tg, tb, st) in enumerate(unacc_specs, 1):
        r = ref_cat[t]
        cases.append((f"DEV_UNACC_{i:03d}", "unaccented", str(r["item_id"]), t, r["baseColour"], r["gender"], mod, "cir_catalog", tt, tc, tg, tb, f"Dev unaccented ({st})"))

    # 3. minor_typo (8 cases)
    typo_specs = [
        ("Shirts", "màu xanh navvy", "Shirts", "Navy Blue", None, None, "A"),
        ("Tshirts", "màu đenn tuyền", "Tshirts", "Black", None, None, "A"),
        ("Casual Shoes", "màu trắnng", "Casual Shoes", "White", None, None, "A"),
        ("Handbags", "màu nâau", "Handbags", "Brown", None, None, "A"),
        ("Shorts", "màu xanh dưong", "Shorts", "Blue", None, None, "A"),
        ("Jackets", "màu xanh lá cấy", "Jackets", "Green", None, None, "A"),
        ("Watches", "màu đenn nam", "Watches", "Black", "Men", None, "A"),
        ("Trousers", "màu xámm", "Trousers", "Grey", None, None, "A"),
    ]
    for i, (t, mod, tt, tc, tg, tb, st) in enumerate(typo_specs, 1):
        r = ref_cat[t]
        cases.append((f"DEV_TYPO_{i:03d}", "minor_typo", str(r["item_id"]), t, r["baseColour"], r["gender"], mod, "cir_catalog", tt, tc, tg, tb, f"Dev typo ({st})"))

    # 4. conversational_long (10 cases - includes explicit state A & B)
    conv_specs = [
        ("Shirts", "tôi muốn tìm chiếc áo giống vậy nhưng có màu trắng", "Shirts", "White", None, None, "A"),
        ("Tshirts", "có mẫu nào tương tự nhưng màu xanh lá cho nam không", "Tshirts", "Green", "Men", None, "A"),
        ("Handbags", "đổi kiểu túi này sang màu đỏ nổi bật giúp tôi", "Handbags", "Red", None, None, "A"),
        ("Casual Shoes", "cho tôi xem đôi giày dáng này màu đen cổ điển", "Casual Shoes", "Black", None, None, "A"),
        ("Dresses", "tìm đầm kiểu tương tự nhưng màu hồng nữ tính", "Dresses", "Pink", "Women", None, "A"),
        ("Watches", "muốn đổi đồng hồ này sang dây màu bạc lịch lãm", "Watches", "Silver", None, None, "A"),
        ("Shirts", "chuyển thành áo khoác màu đen cho nam", "Jackets", "Black", "Men", None, "B"),
        ("Dresses", "chuyển thành chân váy màu trắng thanh lịch", "Skirts", "White", None, None, "B"),
        ("Tshirts", "đổi sang áo sơ mi màu xanh navy công sở", "Shirts", "Navy Blue", None, None, "B"),
        ("Casual Shoes", "đổi sang giày thể thao màu đỏ năng động", "Sports Shoes", "Red", None, None, "B"),
    ]
    for i, (t, mod, tt, tc, tg, tb, st) in enumerate(conv_specs, 1):
        r = ref_cat[t]
        cases.append((f"DEV_CONV_{i:03d}", "conversational_long", str(r["item_id"]), t, r["baseColour"], r["gender"], mod, "cir_catalog", tt, tc, tg, tb, f"Dev conversational ({st})"))

    # 5. multi_condition (10 cases - includes 'có mũ', gender_shift, brand)
    multi_specs = [
        ("Jackets", "màu đen có mũ cho nam", "Jackets", "Black", "Men", None, "A"),
        ("Shirts", "màu trắng tay dài cho nữ", "Shirts", "White", "Women", None, "A"),
        ("Tshirts", "màu xanh dương cổ tròn của Puma", "Tshirts", "Blue", None, "Puma", "A"),
        ("Sports Shoes", "màu trắng chạy bộ của Nike cho nam", "Sports Shoes", "White", "Men", "Nike", "A"),
        ("Handbags", "màu đen đeo vai cho nữ", "Handbags", "Black", "Women", None, "A"),
        ("Trousers", "màu đen ống đứng công sở cho nam", "Trousers", "Black", "Men", None, "A"),
        ("Shorts", "màu xanh navy thể thao của Adidas", "Shorts", "Navy Blue", None, "Adidas", "A"),
        ("Sweaters", "màu xám dài tay cho nam", "Sweaters", "Grey", "Men", None, "A"),
        ("Watches", "màu đen dây da cho nam của Fossil", "Watches", "Black", "Men", "Fossil", "A"),
        ("Skirts", "màu đen xòe cho nữ", "Skirts", "Black", "Women", None, "A"),
    ]
    for i, (t, mod, tt, tc, tg, tb, st) in enumerate(multi_specs, 1):
        r = ref_cat[t]
        cases.append((f"DEV_MULTI_{i:03d}", "multi_condition", str(r["item_id"]), t, r["baseColour"], r["gender"], mod, "cir_catalog", tt, tc, tg, tb, f"Dev multi ({st})"))

    # 6. upload_dev (12 cases across diverse product types + blurred/rotated + non-fashion OOD!)
    upl_specs = [
        ("Shirts", ref_upl["Shirts"]["item_id"], "màu đỏ công sở", "Shirts", "Red", None, None),
        ("Tshirts", ref_upl["Tshirts"]["item_id"], "màu trắng cổ tròn", "Tshirts", "White", None, None),
        ("Handbags", ref_upl["Handbags"]["item_id"], "màu nâu da bò", "Handbags", "Brown", None, None),
        ("Casual Shoes", ref_upl["Casual Shoes"]["item_id"], "màu đen tối giản", "Casual Shoes", "Black", None, None),
        ("Dresses", ref_upl["Dresses"]["item_id"], "màu đen dự tiệc", "Dresses", "Black", None, None),
        ("Watches", ref_upl["Watches"]["item_id"], "màu đen mạnh mẽ", "Watches", "Black", None, None),
        ("Jackets", ref_upl["Jackets"]["item_id"], "màu xanh navy có mũ", "Jackets", "Navy Blue", None, None),
        ("Shorts", ref_upl["Shorts"]["item_id"], "màu đen thể thao", "Shorts", "Black", None, None),
        ("Trousers", ref_upl["Trousers"]["item_id"], "màu xanh navy lịch lãm", "Trousers", "Navy Blue", None, None),
        ("Sports Shoes", ref_upl["Sports Shoes"]["item_id"], "màu xanh dương năng động", "Sports Shoes", "Blue", None, None),
        ("Shirts", "img_blurred.jpg", "màu đen lịch sự", "Shirts", "Black", None, None),
        ("Shirts", "img_rotated.jpg", "màu trắng thanh lịch", "Shirts", "White", None, None),
    ]
    for i, (t, ref_id_val, mod, tt, tc, tg, tb) in enumerate(upl_specs, 1):
        cases.append((f"DEV_UPL_{i:03d}", "upload_dev", str(ref_id_val), t, "Unknown", "Unknown", mod, "cir_upload", tt, tc, tg, tb, f"Dev upload ({t})"))

    cols = ["case_id", "category", "ref_id", "ref_type", "ref_color", "ref_gender", "modifier_text", "flow", "target_type", "target_color", "target_gender", "target_brand", "description"]
    out_df = pd.DataFrame(cases, columns=cols)

    # Assert ZERO overlap with heldout
    dev_ids = set(out_df["ref_id"].astype(str))
    dev_mods = set(out_df["modifier_text"].astype(str).str.strip().str.lower())
    assert len(dev_ids & ho_ids) == 0, f"ID overlap detected: {dev_ids & ho_ids}"
    assert len(dev_mods & ho_mods) == 0, f"Modifier overlap detected: {dev_mods & ho_mods}"

    out_path = ROOT_DIR / "eval" / "cir_test" / "cir_dev_6groups.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Generated {len(out_df)} dev cases in {out_path}")
    print("Category breakdown:", out_df["category"].value_counts().to_dict())
    print("Zero overlap with heldout verified 100%!")

if __name__ == "__main__":
    main()
