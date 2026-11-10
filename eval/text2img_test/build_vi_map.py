import sys
import json
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Base maps from vi_vocab
from src.vi_vocab import ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP

def build_vi_map():
    print("=" * 70)
    print("BUILDING vi_map.json AND vi_map_coverage.csv FROM catalog.parquet")
    print("=" * 70)

    df = pd.read_parquet("data/catalog/catalog.parquet")
    in_scope = df[df["in_scope"] == True]
    print(f"Tổng số sản phẩm in-scope: {len(in_scope):,}")

    # 1. GENDER MAP (lowercase Vietnamese values)
    genders_in_catalog = in_scope["gender"].dropna().unique().tolist()
    gender_counts = in_scope["gender"].value_counts().to_dict()
    
    vi_gender_map = {
        "Men": "nam",
        "Women": "nữ",
        "Boys": "bé trai",
        "Girls": "bé gái",
        "Unisex": "unisex"
    }

    # 2. COLOUR MAP (lowercase natural Vietnamese colour)
    colours_in_catalog = in_scope["baseColour"].dropna().unique().tolist()
    colour_counts = in_scope["baseColour"].value_counts().to_dict()

    vi_colour_map = {
        "Beige": "be",
        "Black": "đen",
        "Blue": "xanh dương",
        "Bronze": "đồng",
        "Brown": "nâu",
        "Burgundy": "đỏ burgundy",
        "Charcoal": "xám than",
        "Coffee Brown": "nâu cà phê",
        "Copper": "đồng đỏ",
        "Cream": "kem",
        "Fluorescent Green": "xanh dạ quang",
        "Gold": "vàng kim",
        "Green": "xanh lá",
        "Grey": "xám",
        "Grey Melange": "xám melange",
        "Khaki": "kaki",
        "Lavender": "tím lavender",
        "Lime Green": "xanh chanh",
        "Magenta": "đỏ cánh sen",
        "Maroon": "đỏ maroon",
        "Mauve": "tím mauve",
        "Metallic": "ánh kim",
        "Multi": "nhiều màu",
        "Mushroom Brown": "nâu nấm",
        "Mustard": "vàng mù tạt",
        "Navy Blue": "xanh navy",
        "Nude": "màu nude",
        "Off White": "trắng ngà",
        "Olive": "xanh ô liu",
        "Orange": "cam",
        "Peach": "đào",
        "Pink": "hồng",
        "Purple": "tím",
        "Red": "đỏ",
        "Rose": "hồng hoa",
        "Rust": "đỏ rỉ sét",
        "Sea Green": "xanh biển",
        "Silver": "bạc",
        "Skin": "màu da",
        "Steel": "xám thép",
        "Tan": "nâu tan",
        "Taupe": "nâu xám taupe",
        "Teal": "xanh teal",
        "Turquoise Blue": "xanh ngọc lam",
        "Unknown": "",
        "White": "trắng",
        "Yellow": "vàng"
    }

    # 3. ARTICLE TYPE MAP (Title case Vietnamese)
    types_in_catalog = in_scope["articleType"].dropna().unique().tolist()
    type_counts = in_scope["articleType"].value_counts().to_dict()

    # Refined natural eCommerce translations
    refined_type_map = {
        "Handbags": "Túi xách tay",
        "Tshirts": "Áo thun",
        "Shirts": "Áo sơ mi",
        "Casual Shoes": "Giày thường",
        "Sports Shoes": "Giày thể thao",
        "Heels": "Giày cao gót",
        "Flats": "Giày búp bê",
        "Tops": "Áo kiểu",
        "Kurtas": "Áo Kurta",
        "Kurtis": "Áo Kurti",
        "Dresses": "Đầm",
        "Jeans": "Quần jean",
        "Trousers": "Quần tây",
        "Shorts": "Quần short",
        "Track Pants": "Quần thể thao",
        "Backpacks": "Balo",
        "Watches": "Đồng hồ",
        "Sunglasses": "Kính mát",
        "Wallets": "Ví tiền",
        "Belts": "Thắt lưng",
        "Socks": "Tất vớ",
        "Flip Flops": "Dép tông",
        "Sandals": "Dép quai hậu",
        "Clutches": "Ví cầm tay",
        "Briefs": "Quần lót tam giác",
        "Boxers": "Quần lót boxer",
        "Trunk": "Quần lót trunk",
        "Innerwear Vests": "Áo lót ba lỗ",
        "Nightdress": "Váy ngủ",
        "Night suits": "Bộ đồ ngủ",
        "Sweaters": "Áo len",
        "Sweatshirts": "Áo nỉ",
        "Jackets": "Áo khoác",
        "Blazers": "Áo blazer",
        "Capris": "Quần lửng capri",
        "Skirts": "Chân váy",
        "Tunics": "Áo tunic",
        "Shrug": "Áo khoác lửng shrug",
        "Dupatta": "Khăn choàng dupatta",
        "Stoles": "Khăn quàng stole",
        "Scarves": "Khăn quàng cổ",
        "Deodorant": "Lăn xịt khử mùi",
        "Perfume and Body Mist": "Nước hoa & xịt thơm",
        "Lipstick": "Son môi",
        "Nail Polish": "Sơn móng tay",
        "Kajal and Eyeliner": "Chì kẻ mắt & eyeliner",
        "Foundation and Primer": "Kem nền & lót",
        "Compact": "Phấn phủ",
        "Lip Gloss": "Son bóng",
        "Eye Cream": "Kem mắt",
        "Mascara": "Chuốt mi mascara",
        "Sunscreen": "Kem chống nắng",
        "Face Wash and Cleanser": "Sữa rửa mặt",
        "Moisturiser": "Kem dưỡng ẩm",
        "Serum and Gel": "Serum dưỡng da",
        "Toner": "Nước hoa hồng toner",
        "Shampoo": "Dầu gội đầu",
        "Conditioner": "Dầu xả",
        "Hair Colour": "Thuốc nhuộm tóc",
        "Hair Accessory": "Phụ kiện tóc",
        "Jewellery Set": "Bộ trang sức",
        "Necklace and Chains": "Dây chuyền vòng cổ",
        "Earrings": "Bông tai",
        "Ring": "Nhẫn",
        "Bangle": "Vòng tay bản cứng",
        "Bracelet": "Vòng đeo tay",
        "Pendant": "Mặt dây chuyền",
        "Ties": "Cà vạt",
        "Cufflinks": "Khuy măng sét",
        "Pocket Squares": "Khăn cài túi áo vest",
        "Caps": "Mũ lưỡi trai",
        "Hat": "Nón rộng vành",
        "Accessory Gift Set": "Bộ quà tặng phụ kiện",
        "Key chain": "Móc chìa khóa",
        "Mufflers": "Khăn len quàng cổ",
        "Gloves": "Găng tay",
        "Swimwear": "Đồ bơi",
        "Bath Robe": "Áo choàng tắm",
        "Rompers": "Bộ áo liền quần rompers",
        "Jumpsuit": "Bộ đồ bay jumpsuit",
        "Salwar and Dupatta": "Bộ đồ Salwar & khăn",
        "Salwar": "Quần thụng Salwar",
        "Churidar": "Quần bó Churidar",
        "Leggings": "Quần legging",
        "Jeggings": "Quần jegging",
        "Stockings": "Quần tất dài",
        "Tights": "Quần tất bó",
        "Sports Sandal": "Sandal thể thao",
        "Formal Shoes": "Giày tây công sở",
        "Booties": "Giày bốt ngắn",
        "Travel Accessory": "Phụ kiện du lịch",
        "Luggage and Travel Bags": "Vali du lịch",
        "Messenger Bag": "Túi đeo chéo messenger",
        "Waist Pouch": "Túi bao tử đeo hông",
        "Duffel Bag": "Túi trống du lịch",
        "Laptop Bag": "Túi đựng laptop",
        "Tablet Sleeve": "Túi đựng máy tính bảng",
        "Baby Dolls": "Váy ngủ babydoll",
        "Bra": "Áo ngực bra",
        "Camisoles": "Áo hai dây camisole",
        "Bath and Body Gift Set": "Bộ quà tắm gội",
        "Body Lotion": "Sữa dưỡng thể",
        "Body Wash and Scrub": "Sữa tắm & tẩy da chết",
        "Face Scrub and Exfoliator": "Tẩy tế bào chết mặt",
        "Lip Care": "Son dưỡng môi",
        "Lip Plumper": "Son làm đầy môi",
        "Makeup Remover": "Nước tẩy trang",
        "Mens Grooming Kit": "Bộ chăm sóc nam giới",
        "Beard Oil": "Tinh dầu dưỡng râu",
        "Shaving Cream": "Kem cạo râu",
        "After Shave Lotion": "Nước dưỡng sau cạo râu"
    }

    # Ensure 100% of types in catalog are mapped
    vi_type_map = {}
    for t in types_in_catalog:
        if t in refined_type_map:
            vi_type_map[t] = refined_type_map[t]
        elif t in ARTICLE_TYPE_MAP:
            vi_type_map[t] = ARTICLE_TYPE_MAP[t]
        else:
            vi_type_map[t] = t

    # 4. BRAND CAPITALIZATION MAP (Clean up common brand casings)
    brand_casings = {
        "adidas": "Adidas",
        "ADIDAS": "Adidas",
        "Adidas": "Adidas",
        "nike": "Nike",
        "NIKE": "Nike",
        "Nike": "Nike",
        "puma": "Puma",
        "PUMA": "Puma",
        "Puma": "Puma",
        "reebok": "Reebok",
        "REEBOK": "Reebok",
        "Reebok": "Reebok",
        "fila": "Fila",
        "FILA": "Fila",
        "Fila": "Fila",
        "levis": "Levi's",
        "LEVIS": "Levi's",
        "Levi's": "Levi's",
        "united colors of benetton": "United Colors of Benetton",
        "wrangler": "Wrangler",
        "WRANGLER": "Wrangler",
        "Wrangler": "Wrangler",
        "roadster": "Roadster",
        "fabindia": "Fabindia",
        "murcia": "Murcia",
        "MURCIA": "Murcia",
        "Murcia": "Murcia",
        "lavie": "Lavie",
        "LAVIE": "Lavie",
        "Lavie": "Lavie",
        "catwalk": "Catwalk",
        "fastrack": "Fastrack",
        "casio": "Casio",
        "titan": "Titan",
        "fossil": "Fossil",
        "tommy hilfiger": "Tommy Hilfiger",
        "flying machine": "Flying Machine",
        "spykar": "Spykar",
        "woodland": "Woodland"
    }

    # Prepare coverage report
    coverage_rows = []

    # Coverage for Genders
    for g in sorted(genders_in_catalog):
        tr = vi_gender_map.get(g)
        cnt = gender_counts.get(g, 0)
        coverage_rows.append({
            "thuoc_tinh": "gender",
            "gia_tri_goc": g,
            "ban_dich_vi": tr,
            "so_san_pham": cnt,
            "da_phu": bool(tr),
            "can_duyet": False
        })

    # Coverage for Colours
    for c in sorted(colours_in_catalog):
        tr = vi_colour_map.get(c)
        cnt = colour_counts.get(c, 0)
        coverage_rows.append({
            "thuoc_tinh": "baseColour",
            "gia_tri_goc": c,
            "ban_dich_vi": tr if tr else "(bỏ qua)",
            "so_san_pham": cnt,
            "da_phu": tr is not None,
            "can_duyet": False
        })

    # Coverage for Article Types
    for t in sorted(types_in_catalog):
        tr = vi_type_map.get(t)
        cnt = type_counts.get(t, 0)
        # Mark can_duyet if it was newly translated or specialized
        can_duyet = t not in ARTICLE_TYPE_MAP
        coverage_rows.append({
            "thuoc_tinh": "articleType",
            "gia_tri_goc": t,
            "ban_dich_vi": tr,
            "so_san_pham": cnt,
            "da_phu": bool(tr),
            "can_duyet": can_duyet
        })

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_file = ROOT_DIR / "eval" / "text2img_test" / "vi_map_coverage.csv"
    coverage_df.to_csv(coverage_file, index=False, encoding="utf-8-sig")
    print(f"Đã xuất vi_map_coverage.csv: {len(coverage_df)} dòng.")

    # Check unmapped
    unmapped = coverage_df[coverage_df["da_phu"] == False]
    print(f"Số giá trị chưa phủ: {len(unmapped)}")
    assert len(unmapped) == 0, f"Có {len(unmapped)} giá trị chưa phủ!"

    # Export vi_map.json
    final_vi_map = {
        "gender": vi_gender_map,
        "baseColour": vi_colour_map,
        "articleType": vi_type_map,
        "brand": brand_casings
    }

    json_file = ROOT_DIR / "eval" / "text2img_test" / "vi_map.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_vi_map, f, ensure_ascii=False, indent=2)
    print(f"Đã xuất vi_map.json: {len(vi_gender_map)} genders, {len(vi_colour_map)} colours, {len(vi_type_map)} types.")

if __name__ == "__main__":
    build_vi_map()
