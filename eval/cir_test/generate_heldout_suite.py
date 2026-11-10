"""
Generate Held-Out CIR Test Suite (>= 80 cases) BEFORE modifying any dictionary or logic.
Anti-overfitting validation benchmark stored in eval/cir_test/cir_heldout.csv.
"""
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

df_catalog = pd.read_parquet("data/catalog/catalog.parquet")

def build_heldout():
    # Pick diverse items across apparel, footwear, accessories
    types = ["Shirts", "Tshirts", "Casual Shoes", "Sports Shoes", "Handbags", "Dresses", "Skirts", "Trousers", "Shorts", "Jackets", "Watches", "Backpacks", "Sweaters"]
    ref_items = {}
    for t in types:
        sub = df_catalog[(df_catalog["articleType"] == t) & (df_catalog["in_scope"] == True)]
        if len(sub) > 1:
            ref_items[t] = sub.iloc[1] # Pick second item to be different from dev set
        elif len(sub) > 0:
            ref_items[t] = sub.iloc[0]

    heldout_cases = []

    # 1. Natural phrasing & synonyms (25 cases)
    synonym_cases = [
        ("Shirts", "tông màu xanh navy đậm", "Shirts", "Navy Blue", None, None),
        ("Shirts", "chuyển sang màu đen tuyền", "Shirts", "Black", None, None),
        ("Shirts", "gam màu trắng thanh lịch", "Shirts", "White", None, None),
        ("Shirts", "sắc đỏ rực rỡ", "Shirts", "Red", None, None),
        ("Shirts", "tông xám lông chuột", "Shirts", "Grey", None, None),
        ("Tshirts", "chuyển thành màu vàng chanh", "Tshirts", "Yellow", None, None),
        ("Tshirts", "gam xanh nước biển", "Tshirts", "Blue", None, None),
        ("Tshirts", "sắc hồng phấn nhẹ nhàng", "Tshirts", "Pink", None, None),
        ("Tshirts", "màu đen basic", "Tshirts", "Black", None, None),
        ("Casual Shoes", "đổi sang màu da bò", "Casual Shoes", "Tan", None, None),
        ("Casual Shoes", "tông màu trắng tinh khôi", "Casual Shoes", "White", None, None),
        ("Casual Shoes", "sắc xanh navy thể thao", "Casual Shoes", "Navy Blue", None, None),
        ("Casual Shoes", "gam màu đen cổ điển", "Casual Shoes", "Black", None, None),
        ("Sports Shoes", "chuyển thành màu đỏ đô", "Sports Shoes", "Red", None, None),
        ("Sports Shoes", "tông xanh da trời", "Sports Shoes", "Blue", None, None),
        ("Sports Shoes", "phối màu đen trắng", "Sports Shoes", "Black", None, None),
        ("Handbags", "gam màu be trang nhã", "Handbags", "Beige", None, None),
        ("Handbags", "tông màu nâu trầm", "Handbags", "Brown", None, None),
        ("Handbags", "sắc hồng cam pastel", "Handbags", "Peach", None, None),
        ("Dresses", "chuyển sang màu đỏ dạ hội", "Dresses", "Red", None, None),
        ("Dresses", "tông màu đen huyền bí", "Dresses", "Black", None, None),
        ("Skirts", "chân váy sắc trắng tinh", "Skirts", "White", None, None),
        ("Trousers", "quần âu tông màu xanh than", "Trousers", "Navy Blue", None, None),
        ("Jackets", "áo khoác sắc xanh rêu", "Jackets", "Olive", None, None),
        ("Sweaters", "áo len gam màu kem ấm áp", "Sweaters", "Cream", None, None)
    ]
    for idx, (t, mod, tgt_t, tgt_c, tgt_g, tgt_b) in enumerate(synonym_cases, 1):
        if t in ref_items:
            row = ref_items[t]
            heldout_cases.append({
                "case_id": f"HELDOUT_SYN_{idx:03d}",
                "category": "synonym_natural",
                "ref_id": str(row["item_id"]),
                "ref_type": t,
                "ref_color": row["baseColour"],
                "ref_gender": row["gender"],
                "modifier_text": mod,
                "flow": "cir_catalog",
                "target_type": tgt_t,
                "target_color": tgt_c,
                "target_gender": tgt_g,
                "target_brand": tgt_b,
                "description": f"Từ đồng nghĩa / sắc thái: {mod}"
            })

    # 2. Unaccented (không dấu) (15 cases)
    unaccented_cases = [
        ("Shirts", "mau xanh navy", "Shirts", "Navy Blue", None, None),
        ("Shirts", "mau trang", "Shirts", "White", None, None),
        ("Shirts", "mau den", "Shirts", "Black", None, None),
        ("Tshirts", "mau do", "Tshirts", "Red", None, None),
        ("Tshirts", "mau vang", "Tshirts", "Yellow", None, None),
        ("Tshirts", "mau xam", "Tshirts", "Grey", None, None),
        ("Casual Shoes", "mau den cho nam", "Casual Shoes", "Black", "Men", None),
        ("Casual Shoes", "mau trang nu", "Casual Shoes", "White", "Women", None),
        ("Sports Shoes", "mau xanh duong", "Sports Shoes", "Blue", None, None),
        ("Handbags", "mau hong cho nu", "Handbags", "Pink", "Women", None),
        ("Handbags", "mau nau da bo", "Handbags", "Brown", None, None),
        ("Dresses", "mau do da hoi", "Dresses", "Red", "Women", None),
        ("Shorts", "quan short mau vang", "Shorts", "Yellow", None, None),
        ("Jackets", "ao khoac mau den", "Jackets", "Black", None, None),
        ("Watches", "dong ho mau bac", "Watches", "Silver", None, None)
    ]
    for idx, (t, mod, tgt_t, tgt_c, tgt_g, tgt_b) in enumerate(unaccented_cases, 1):
        if t in ref_items:
            row = ref_items[t]
            heldout_cases.append({
                "case_id": f"HELDOUT_UNACC_{idx:03d}",
                "category": "unaccented",
                "ref_id": str(row["item_id"]),
                "ref_type": t,
                "ref_color": row["baseColour"],
                "ref_gender": row["gender"],
                "modifier_text": mod,
                "flow": "cir_catalog",
                "target_type": tgt_t,
                "target_color": tgt_c,
                "target_gender": tgt_g,
                "target_brand": tgt_b,
                "description": f"Truy vấn không dấu: {mod}"
            })

    # 3. Minor typos (sai chính tả nhẹ) (12 cases)
    typo_cases = [
        ("Shirts", "xang navy", "Shirts", "Navy Blue", None, None),
        ("Shirts", "màu trắg", "Shirts", "White", None, None),
        ("Shirts", "màu đn", "Shirts", "Black", None, None),
        ("Tshirts", "màu đó", "Tshirts", "Red", None, None),
        ("Tshirts", "màu xám ghi", "Tshirts", "Grey", None, None),
        ("Casual Shoes", "màu trắnh", "Casual Shoes", "White", None, None),
        ("Sports Shoes", "màu xah dương", "Sports Shoes", "Blue", None, None),
        ("Handbags", "màu hồg", "Handbags", "Pink", None, None),
        ("Handbags", "màu đenn", "Handbags", "Black", None, None),
        ("Dresses", "dáng dàii màu đỏ", "Dresses", "Red", None, None),
        ("Jackets", "áo khoác da đen", "Jackets", "Black", None, None),
        ("Shorts", "quần sooc màu vàg", "Shorts", "Yellow", None, None)
    ]
    for idx, (t, mod, tgt_t, tgt_c, tgt_g, tgt_b) in enumerate(typo_cases, 1):
        if t in ref_items:
            row = ref_items[t]
            heldout_cases.append({
                "case_id": f"HELDOUT_TYPO_{idx:03d}",
                "category": "minor_typo",
                "ref_id": str(row["item_id"]),
                "ref_type": t,
                "ref_color": row["baseColour"],
                "ref_gender": row["gender"],
                "modifier_text": mod,
                "flow": "cir_catalog",
                "target_type": tgt_t,
                "target_color": tgt_c,
                "target_gender": tgt_g,
                "target_brand": tgt_b,
                "description": f"Sai chính tả / gõ nhầm nhẹ: {mod}"
            })

    # 4. Long conversational sentences (15 cases)
    long_cases = [
        ("Shirts", "tôi muốn đổi sang chiếc áo màu xanh navy lịch lãm này", "Shirts", "Navy Blue", None, None),
        ("Shirts", "tìm cho tôi mẫu áo sơ mi tương tự nhưng màu trắng công sở", "Shirts", "White", None, None),
        ("Shirts", "có chiếc nào kiểu dáng này mà màu đen cho nam không", "Shirts", "Black", "Men", None),
        ("Tshirts", "hãy tìm áo thun dáng này nhưng có màu đỏ nổi bật", "Tshirts", "Red", None, None),
        ("Tshirts", "đổi kiểu này sang màu vàng tươi trẻ trung", "Tshirts", "Yellow", None, None),
        ("Casual Shoes", "tìm đôi giày phom này nhưng màu trắng phối đen", "Casual Shoes", "White", None, None),
        ("Casual Shoes", "tôi thích đôi giày này nhưng muốn màu nâu da bò", "Casual Shoes", "Brown", None, None),
        ("Sports Shoes", "tìm giày chạy bộ tương tự nhưng màu xanh dương", "Sports Shoes", "Blue", None, None),
        ("Sports Shoes", "đổi sang giày thể thao màu đen của Nike", "Sports Shoes", "Black", None, "Nike"),
        ("Handbags", "tìm chiếc túi xách cùng phom dáng nhưng màu hồng pastel", "Handbags", "Pink", "Women", None),
        ("Handbags", "túi kiểu này nhưng chất liệu da màu đen sang trọng", "Handbags", "Black", "Women", None),
        ("Dresses", "tôi cần chiếc đầm tương tự nhưng màu đỏ quyến rũ", "Dresses", "Red", "Women", None),
        ("Jackets", "áo khoác này mà có màu xanh rêu quân đội thì tốt", "Jackets", "Olive", None, None),
        ("Backpacks", "tìm mẫu balo tương tự màu đen để đi làm", "Backpacks", "Black", None, None),
        ("Trousers", "quần tây kiểu dáng này màu xanh navy có không", "Trousers", "Navy Blue", None, None)
    ]
    for idx, (t, mod, tgt_t, tgt_c, tgt_g, tgt_b) in enumerate(long_cases, 1):
        if t in ref_items:
            row = ref_items[t]
            heldout_cases.append({
                "case_id": f"HELDOUT_LONG_{idx:03d}",
                "category": "conversational_long",
                "ref_id": str(row["item_id"]),
                "ref_type": t,
                "ref_color": row["baseColour"],
                "ref_gender": row["gender"],
                "modifier_text": mod,
                "flow": "cir_catalog",
                "target_type": tgt_t,
                "target_color": tgt_c,
                "target_gender": tgt_g,
                "target_brand": tgt_b,
                "description": f"Câu hội thoại dài: {mod}"
            })

    # 5. Multi-condition combinations (15 cases)
    multi_cases = [
        ("Shirts", "áo sơ mi màu trắng dài tay cho nam", "Shirts", "White", "Men", None),
        ("Shirts", "màu xanh navy kẻ sọc cho nam", "Shirts", "Navy Blue", "Men", None),
        ("Shirts", "màu đỏ rượu vang cho nữ", "Shirts", "Red", "Women", None),
        ("Tshirts", "áo thun màu đen cổ bẻ polo", "Tshirts", "Black", None, None),
        ("Tshirts", "áo thun màu xanh dương cho bé trai", "Tshirts", "Blue", "Boys", None),
        ("Tshirts", "màu vàng tay ngắn cho nữ", "Tshirts", "Yellow", "Women", None),
        ("Casual Shoes", "giày lười màu nâu của Vans", "Casual Shoes", "Brown", None, "Vans"),
        ("Casual Shoes", "giày màu trắng đế thấp cho nữ", "Casual Shoes", "White", "Women", None),
        ("Sports Shoes", "giày thể thao màu đen của Adidas", "Sports Shoes", "Black", None, "Adidas"),
        ("Sports Shoes", "giày chạy bộ màu đỏ của Puma cho nam", "Sports Shoes", "Red", "Men", "Puma"),
        ("Handbags", "túi xách màu be đeo vai cho nữ", "Handbags", "Beige", "Women", None),
        ("Dresses", "váy đầm dạ hội dài màu đen cho nữ", "Dresses", "Black", "Women", None),
        ("Jackets", "áo khoác gió màu xanh có mũ cho nam", "Jackets", "Blue", "Men", None),
        ("Shorts", "quần đùi thể thao màu đen của Nike", "Shorts", "Black", None, "Nike"),
        ("Watches", "đồng hồ kim loại màu vàng cho nam", "Watches", "Gold", "Men", None)
    ]
    for idx, (t, mod, tgt_t, tgt_c, tgt_g, tgt_b) in enumerate(multi_cases, 1):
        if t in ref_items:
            row = ref_items[t]
            heldout_cases.append({
                "case_id": f"HELDOUT_MULTI_{idx:03d}",
                "category": "multi_condition",
                "ref_id": str(row["item_id"]),
                "ref_type": t,
                "ref_color": row["baseColour"],
                "ref_gender": row["gender"],
                "modifier_text": mod,
                "flow": "cir_catalog",
                "target_type": tgt_t,
                "target_color": tgt_c,
                "target_gender": tgt_g,
                "target_brand": tgt_b,
                "description": f"Đa điều kiện: {mod}"
            })

    # 6. Upload image flow cases (8 cases)
    upload_cases = [
        ("crop_uploaded_shirt.jpg", "tông màu xanh navy", "Shirts", "Navy Blue", None, None),
        ("crop_uploaded_shirt.jpg", "mau den cong so", "Shirts", "Black", None, None),
        ("crop_uploaded_shirt.jpg", "chuyển sang màu đỏ nổi bật", "Shirts", "Red", None, None),
        ("crop_uploaded_shirt.jpg", "áo sơ mi màu xám tro", "Shirts", "Grey", None, None),
        ("crop_uploaded_shirt.jpg", "tìm chiếc áo này màu xanh dương", "Shirts", "Blue", None, None),
        ("crop_uploaded_shirt.jpg", "màu đen cho nam của Arrow", "Shirts", "Black", "Men", "Arrow"),
        ("crop_uploaded_shirt.jpg", "mau trang co be", "Shirts", "White", None, None),
        ("crop_uploaded_shirt.jpg", "sắc đỏ mận sang trọng", "Shirts", "Red", None, None)
    ]
    for idx, (img, mod, tgt_t, tgt_c, tgt_g, tgt_b) in enumerate(upload_cases, 1):
        heldout_cases.append({
            "case_id": f"HELDOUT_UPL_{idx:03d}",
            "category": "upload_heldout",
            "ref_id": img,
            "ref_type": "Shirts",
            "ref_color": "White",
            "ref_gender": "Men",
            "modifier_text": mod,
            "flow": "cir_upload",
            "target_type": tgt_t,
            "target_color": tgt_c,
            "target_gender": tgt_g,
            "target_brand": tgt_b,
            "description": f"Upload held-out: {mod}"
        })

    df_out = pd.DataFrame(heldout_cases)
    out_path = ROOT_DIR / "eval" / "cir_test" / "cir_heldout.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã tạo tập kiểm tra giữ kín: {len(df_out)} ca tại {out_path}")
    print(df_out["category"].value_counts().to_dict())

if __name__ == "__main__":
    build_heldout()
