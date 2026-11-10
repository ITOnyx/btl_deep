import json
import csv
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Load catalog in_scope
catalog_path = ROOT_DIR / "data/catalog/catalog.parquet"
df_cat = pd.read_parquet(catalog_path)
df_in = df_cat[df_cat["in_scope"] == True].copy()

# Load brands
brands_cache = ROOT_DIR / "data/catalog/product_brands.json"
if brands_cache.exists():
    with open(brands_cache, "r", encoding="utf-8") as f:
        df_in["brand"] = json.load(f)
else:
    df_in["brand"] = ""

# Load vocab dictionaries
from src.vi_vocab import (
    ARTICLE_TYPE_MAP,
    COLOUR_MAP,
    GENDER_MAP,
    USAGE_MAP,
    remove_diacritics,
    generate_vietnamese_title
)

# 1. Build vi_map.json
vi_map = {
    "articleType": ARTICLE_TYPE_MAP,
    "baseColour": COLOUR_MAP,
    "gender": GENDER_MAP,
    "usage": USAGE_MAP,
    "season": {
        "Summer": "Mùa hè",
        "Winter": "Mùa đông",
        "Spring": "Mùa xuân",
        "Fall": "Mùa thu"
    }
}

vi_map_path = ROOT_DIR / "eval/text2img_test/vi_map.json"
with open(vi_map_path, "w", encoding="utf-8") as f:
    json.dump(vi_map, f, ensure_ascii=False, indent=2)
print(f"Saved vi_map.json ({len(ARTICLE_TYPE_MAP)} articleTypes, {len(COLOUR_MAP)} colours)")

# 2. Build queries list
queries = []
q_id_counter = 1

def add_query(group: str, text: str, expectation: dict, note: str):
    global q_id_counter
    qid = f"Q_{group}_{q_id_counter:04d}"
    q_id_counter += 1
    queries.append({
        "id": qid,
        "nhóm": group,
        "truy_vấn": text,
        "kỳ_vọng": json.dumps(expectation, ensure_ascii=False),
        "ghi_chú": note
    })

# --- Group A: Loại sản phẩm (mọi articleType >= 20 sản phẩm in-scope) ---
type_counts = df_in["articleType"].value_counts()
types_ge20 = type_counts[type_counts >= 20].index.tolist()

for at in types_ge20:
    vn_name = ARTICLE_TYPE_MAP.get(at, at)
    add_query("A", vn_name.lower(), {"articleType": at}, f"Loại sản phẩm: {at}")
    # Thêm cách gọi thứ hai nếu có trong từ điển
    if at == "Tshirts":
        add_query("A", "áo phông", {"articleType": "Tshirts"}, "Cách gọi khác của Tshirts")
    elif at == "Jeans":
        add_query("A", "quần bò", {"articleType": "Jeans"}, "Cách gọi khác của Jeans")
    elif at == "Handbags":
        add_query("A", "túi xách tay", {"articleType": "Handbags"}, "Cách gọi khác của Handbags")
    elif at == "Casual Shoes":
        add_query("A", "giày đi chơi", {"articleType": "Casual Shoes"}, "Cách gọi khác của Casual Shoes")
    elif at == "Dresses":
        add_query("A", "váy liền thân", {"articleType": "Dresses"}, "Cách gọi khác của Dresses")
    elif at == "Backpacks":
        add_query("A", "ba lô", {"articleType": "Backpacks"}, "Cách gọi khác của Backpacks")

# --- Group B: Loại + màu (<= 8 màu thực có mỗi loại) ---
selected_types_b = ["Tshirts", "Shirts", "Jeans", "Handbags", "Casual Shoes", "Dresses", "Heels", "Watches", "Shorts", "Kurtas", "Jackets", "Sweaters"]
for at in selected_types_b:
    vn_type = ARTICLE_TYPE_MAP.get(at, at).lower()
    sub_df = df_in[df_in["articleType"] == at]
    top_colours = sub_df["baseColour"].value_counts().head(8).index.tolist()
    for col in top_colours:
        vn_col = COLOUR_MAP.get(col, col).lower()
        if not vn_col.startswith("màu ") and not vn_col.startswith("đỏ ") and not vn_col.startswith("xanh ") and not vn_col.startswith("vàng ") and not vn_col.startswith("nâu ") and not vn_col.startswith("xám "):
            col_phrase = f"màu {vn_col}"
        else:
            col_phrase = vn_col
        text = f"{vn_type} {col_phrase}"
        add_query("B", text, {"articleType": at, "baseColour": col}, f"Loại {at} + màu {col}")

# --- Group C: Loại + giới tính ---
selected_types_c = ["Tshirts", "Shirts", "Jeans", "Casual Shoes", "Watches", "Shorts", "Trousers", "Jackets", "Sandals", "Flip Flops", "Sweaters", "Track Pants"]
genders_map = [
    ("nam", "Men"),
    ("nữ", "Women"),
    ("bé trai", "Boys"),
    ("bé gái", "Girls"),
    ("unisex", "Unisex")
]
for at in selected_types_c:
    vn_type = ARTICLE_TYPE_MAP.get(at, at).lower()
    for g_text, g_code in genders_map:
        # Check if exists in catalog
        count = len(df_in[(df_in["articleType"] == at) & (df_in["gender"] == g_code)])
        if count > 0:
            text = f"{vn_type} {g_text}"
            add_query("C", text, {"articleType": at, "gender": g_code}, f"Loại {at} + giới tính {g_code}")

# --- Group D: Loại + màu + giới tính (>= 150 ca) ---
top_types_d = ["Tshirts", "Shirts", "Jeans", "Casual Shoes", "Watches", "Shorts", "Trousers", "Dresses", "Jackets", "Sandals"]
d_count = 0
for at in top_types_d:
    vn_type = ARTICLE_TYPE_MAP.get(at, at).lower()
    for g_text, g_code in [("nam", "Men"), ("nữ", "Women"), ("bé trai", "Boys"), ("bé gái", "Girls")]:
        sub_d = df_in[(df_in["articleType"] == at) & (df_in["gender"] == g_code)]
        if len(sub_d) == 0:
            continue
        top_cols = sub_d["baseColour"].value_counts().head(6).index.tolist()
        for col in top_cols:
            vn_col = COLOUR_MAP.get(col, col).lower()
            if not vn_col.startswith("màu ") and not vn_col.startswith("đỏ ") and not vn_col.startswith("xanh ") and not vn_col.startswith("vàng ") and not vn_col.startswith("nâu ") and not vn_col.startswith("xám "):
                col_str = f"màu {vn_col}"
            else:
                col_str = vn_col
            text = f"{vn_type} {g_text} {col_str}"
            add_query("D", text, {"articleType": at, "baseColour": col, "gender": g_code}, f"Loại {at} + {col} + {g_code}")
            d_count += 1
            if d_count >= 180:
                break
        if d_count >= 180:
            break
    if d_count >= 180:
        break

# --- Group E: Thương hiệu (brand >= 30 sản phẩm; >= 100 ca) ---
brand_counts = df_in["brand"].value_counts()
brands_ge30 = [b for b in brand_counts[brand_counts >= 30].index if b and b != "Unknown" and b != "Black"][:50]

for b in brands_ge30:
    # 1. Chỉ tên brand
    add_query("E", b, {"brand": b}, f"Chỉ thương hiệu {b}")
    # 2. Brand + loại sản phẩm phổ biến nhất của brand đó
    b_types = df_in[df_in["brand"] == b]["articleType"].value_counts().head(2).index.tolist()
    for bt in b_types:
        vn_bt = ARTICLE_TYPE_MAP.get(bt, bt).lower()
        add_query("E", f"{vn_bt} {b}", {"brand": b, "articleType": bt}, f"{vn_bt} thương hiệu {b}")
    # 3. Brand + loại + giới tính nếu có
    b_items = df_in[df_in["brand"] == b]
    if len(b_items) > 0:
        top_row = b_items.iloc[0]
        at_e = top_row["articleType"]
        g_e = top_row["gender"]
        g_str = "nam" if g_e == "Men" else ("nữ" if g_e == "Women" else "")
        if g_str:
            add_query("E", f"{ARTICLE_TYPE_MAP.get(at_e, at_e).lower()} {b} {g_str}", {"brand": b, "articleType": at_e, "gender": g_e}, f"{at_e} {b} {g_str}")

# --- Group F: Thuộc tính khác (họa tiết, mùa, mục đích: thể thao, công sở, dự tiệc, đi biển, kiểu dáng) ---
other_attrs = [
    ("áo sơ mi công sở nam", {"articleType": "Shirts", "usage": "Formal", "gender": "Men"}, "Mục đích công sở Formal"),
    ("quần áo thể thao nam", {"usage": "Sports", "gender": "Men"}, "Mục đích thể thao Sports"),
    ("giày thể thao chạy bộ", {"articleType": "Sports Shoes", "usage": "Sports"}, "Giày chạy bộ thể thao"),
    ("váy dạ hội dự tiệc", {"articleType": "Dresses", "usage": "Party"}, "Váy dạ hội dự tiệc Party"),
    ("áo thun mùa hè", {"articleType": "Tshirts", "season": "Summer"}, "Mùa hè Summer"),
    ("áo khoác mùa đông", {"articleType": "Jackets", "season": "Winter"}, "Mùa đông Winter"),
    ("đồ bơi đi biển", {"usage": "Sports"}, "Đồ bơi đi biển"),
    ("quần jean ôm skinny", {"articleType": "Jeans"}, "Kiểu dáng quần jean ôm"),
    ("áo sơ mi kẻ caro", {"articleType": "Shirts"}, "Họa tiết kẻ caro"),
    ("áo thun kẻ sọc", {"articleType": "Tshirts"}, "Họa tiết kẻ sọc striped"),
    ("ví da nam công sở", {"articleType": "Wallets", "gender": "Men"}, "Ví da nam công sở"),
    ("đồng hồ dây kim loại", {"articleType": "Watches"}, "Đồng hồ dây kim loại metallic"),
    ("dép lê đi biển mùa hè", {"articleType": "Flip Flops", "season": "Summer"}, "Dép tông mùa hè"),
    ("áo khoác len ấm áp", {"articleType": "Sweaters"}, "Áo len sweaters"),
    ("kính mát chống nắng đi biển", {"articleType": "Sunglasses"}, "Kính râm đi biển"),
    ("váy hoa xòe mùa xuân", {"articleType": "Dresses", "season": "Spring"}, "Váy hoa mùa xuân")
]
for text, exp, note in other_attrs:
    add_query("F", text, exp, note)

# --- Group G: Đồng nghĩa ---
synonyms_pairs = [
    ("áo phông nam màu trắng", {"articleType": "Tshirts", "baseColour": "White", "gender": "Men"}, "Đồng nghĩa: áo phông"),
    ("áo thun nam màu trắng", {"articleType": "Tshirts", "baseColour": "White", "gender": "Men"}, "Đồng nghĩa: áo thun"),
    ("quần bò nam xanh", {"articleType": "Jeans", "baseColour": "Blue", "gender": "Men"}, "Đồng nghĩa: quần bò"),
    ("quần jean nam xanh", {"articleType": "Jeans", "baseColour": "Blue", "gender": "Men"}, "Đồng nghĩa: quần jean"),
    ("túi sách màu hồng", {"articleType": "Handbags", "baseColour": "Pink"}, "Đồng nghĩa / chính tả: túi sách"),
    ("túi xách màu hồng", {"articleType": "Handbags", "baseColour": "Pink"}, "Đồng nghĩa: túi xách"),
    ("ba lô thời trang", {"articleType": "Backpacks"}, "Đồng nghĩa: ba lô"),
    ("balo thời trang", {"articleType": "Backpacks"}, "Đồng nghĩa: balo"),
    ("dép xỏ ngón nam", {"articleType": "Flip Flops", "gender": "Men"}, "Đồng nghĩa: dép xỏ ngón"),
    ("dép tông nam", {"articleType": "Flip Flops", "gender": "Men"}, "Đồng nghĩa: dép tông"),
    ("giày bệt nữ màu đen", {"articleType": "Flats", "baseColour": "Black", "gender": "Women"}, "Đồng nghĩa: giày bệt"),
    ("giày búp bê nữ màu đen", {"articleType": "Flats", "baseColour": "Black", "gender": "Women"}, "Đồng nghĩa: giày búp bê")
]
for text, exp, note in synonyms_pairs:
    add_query("G", text, exp, note)

# --- Group H: Biến thể (không dấu, sai chính tả, viết hoa, thừa khoảng trắng, emoji, số) ---
variants = [
    # Không dấu
    ("ao thun puma nam", {"brand": "Puma", "articleType": "Tshirts", "gender": "Men"}, "Không dấu: ao thun puma nam"),
    ("quan jean nam mau xanh", {"articleType": "Jeans", "gender": "Men", "baseColour": "Blue"}, "Không dấu: quan jean nam xanh"),
    ("tui xach nu mau hong", {"articleType": "Handbags", "gender": "Women", "baseColour": "Pink"}, "Không dấu: tui xach nu mau hong"),
    ("giay the thao nam nike", {"brand": "Nike", "articleType": "Sports Shoes", "gender": "Men"}, "Không dấu: giay the thao nam nike"),
    ("dam da hoi mau do", {"articleType": "Dresses", "baseColour": "Red"}, "Không dấu: dam da hoi mau do"),
    ("dong ho nu day da", {"articleType": "Watches", "gender": "Women"}, "Không dấu: dong ho nu"),
    ("ao so mi trang nam", {"articleType": "Shirts", "gender": "Men", "baseColour": "White"}, "Không dấu: ao so mi trang nam"),
    ("quan short nu mau trang", {"articleType": "Shorts", "gender": "Women", "baseColour": "White"}, "Không dấu: quan short nu trang"),
    # Sai chính tả nhẹ / phương ngữ
    ("quần sọt nam", {"articleType": "Shorts", "gender": "Men"}, "Phương ngữ: quần sọt -> Shorts"),
    ("áo thung nam", {"articleType": "Tshirts", "gender": "Men"}, "Sai chính tả: thung -> thun"),
    ("quần jiên nam", {"articleType": "Jeans", "gender": "Men"}, "Sai chính tả: jiên -> jean"),
    ("giầy cao gót đen", {"articleType": "Heels", "baseColour": "Black"}, "Chính tả: giầy -> giày"),
    ("nón kết nam", {"articleType": "Caps", "gender": "Men"}, "Phương ngữ: nón kết -> Caps"),
    # HOA / thường
    ("ÁO THUN NAM PUMA", {"brand": "Puma", "articleType": "Tshirts", "gender": "Men"}, "VIẾT HOA TOÀN BỘ"),
    ("qUầN jEaN nAm XaNh", {"articleType": "Jeans", "gender": "Men", "baseColour": "Blue"}, "Chữ hoa thường lộn xộn"),
    # Thừa khoảng trắng & dấu câu
    ("   áo sơ mi    nam    trắng   ", {"articleType": "Shirts", "gender": "Men", "baseColour": "White"}, "Thừa nhiều khoảng trắng"),
    ("áo thun nam, màu đỏ, size L!", {"articleType": "Tshirts", "gender": "Men", "baseColour": "Red"}, "Chứa dấu phẩy và chấm than"),
    # Emoji & số
    ("áo thun Puma nam 👕", {"brand": "Puma", "articleType": "Tshirts", "gender": "Men"}, "Kèm emoji áo"),
    ("giày cao gót nữ 👠 màu đỏ", {"articleType": "Heels", "gender": "Women", "baseColour": "Red"}, "Kèm emoji giày cao gót"),
    ("túi xách nữ 👜 màu đen", {"articleType": "Handbags", "gender": "Women", "baseColour": "Black"}, "Kèm emoji túi xách"),
    ("áo thun nam 100% cotton", {"articleType": "Tshirts", "gender": "Men"}, "Chứa số 100 và ký hiệu %")
]
for text, exp, note in variants:
    add_query("H", text, exp, note)

# --- Group I: Câu dài / tự nhiên, câu > 128 token, câu 1 từ ---
long_queries = [
    # Câu 1 từ
    ("áo", {"articleType": "Tshirts"}, "Câu 1 từ: áo"),
    ("quần", {"articleType": "Jeans"}, "Câu 1 từ: quần"),
    ("giày", {"articleType": "Casual Shoes"}, "Câu 1 từ: giày"),
    ("túi", {"articleType": "Handbags"}, "Câu 1 từ: túi"),
    ("Puma", {"brand": "Puma"}, "Câu 1 từ: brand Puma"),
    ("Nike", {"brand": "Nike"}, "Câu 1 từ: brand Nike"),
    # Câu dài tự nhiên
    ("tôi cần tìm một chiếc áo sơ mi nam dài tay màu trắng thanh lịch để mặc đi làm công sở hàng ngày", {"articleType": "Shirts", "gender": "Men", "baseColour": "White"}, "Câu tự nhiên dài"),
    ("tìm giúp tôi một chiếc túi xách nữ màu hồng kiểu dáng trẻ trung thanh lịch dành cho phái nữ đi dạo phố", {"articleType": "Handbags", "gender": "Women", "baseColour": "Pink"}, "Câu tự nhiên dài"),
    ("muốn mua một đôi giày thể thao chạy bộ êm chân thương hiệu Nike cho nam màu xanh dương khỏe khoắn", {"brand": "Nike", "articleType": "Sports Shoes", "gender": "Men", "baseColour": "Blue"}, "Câu tự nhiên dài"),
    # Câu cực dài > 128 tokens để kiểm thử max_length=128 truncation
    ("áo thun nam Puma màu vàng tươi chất liệu cotton thoáng mát co giãn tốt thấm hút mồ hôi dành cho người chơi thể thao vận động nhiều ngoài trời " * 10, {"brand": "Puma", "articleType": "Tshirts", "gender": "Men", "baseColour": "Yellow"}, "Câu cực dài > 128 tokens để test truncation")
]
for text, exp, note in long_queries:
    add_query("I", text, exp, note)

# --- Group J: Tiếng Anh thuần và trộn Việt-Anh ---
lang_mix = [
    ("black leather handbag", {"articleType": "Handbags", "baseColour": "Black"}, "Tiếng Anh thuần"),
    ("blue denim jeans for men", {"articleType": "Jeans", "baseColour": "Blue", "gender": "Men"}, "Tiếng Anh thuần"),
    ("red evening party dress", {"articleType": "Dresses", "baseColour": "Red", "usage": "Party"}, "Tiếng Anh thuần"),
    ("men running shoes nike", {"brand": "Nike", "articleType": "Sports Shoes", "gender": "Men"}, "Tiếng Anh thuần"),
    ("áo tshirt nam màu đen", {"articleType": "Tshirts", "gender": "Men", "baseColour": "Black"}, "Trộn Việt-Anh: áo tshirt"),
    ("quần jean denim nam", {"articleType": "Jeans", "gender": "Men"}, "Trộn Việt-Anh: denim jeans"),
    ("túi tote bag nữ", {"articleType": "Handbags", "gender": "Women"}, "Trộn Việt-Anh: tote bag"),
    ("đầm maxi dress đi biển", {"articleType": "Dresses"}, "Trộn Việt-Anh: maxi dress"),
    ("áo jacket khoác nam", {"articleType": "Jackets", "gender": "Men"}, "Trộn Việt-Anh: áo jacket")
]
for text, exp, note in lang_mix:
    add_query("J", text, exp, note)

# --- Group K: Phủ định / ràng buộc / đa màu ---
constraints = [
    ("áo thun trơn không họa tiết", {"articleType": "Tshirts"}, "Phủ định: không họa tiết"),
    ("quần jean trơn không rách", {"articleType": "Jeans"}, "Phủ định: không rách"),
    ("áo sơ mi không cổ", {"articleType": "Shirts"}, "Phủ định: không cổ"),
    ("áo thun sọc đen trắng", {"articleType": "Tshirts"}, "Đa màu: đen trắng"),
    ("đầm liền phối hai màu xanh trắng", {"articleType": "Dresses"}, "Đa màu: xanh trắng"),
    ("giày thể thao trắng đen", {"articleType": "Sports Shoes"}, "Đa màu: trắng đen")
]
for text, exp, note in constraints:
    add_query("K", text, exp, note)

# --- Group L: Ngoài phạm vi (xe máy, iPhone, đồ ăn, mỹ phẩm...) ---
out_of_scope = [
    ("xe máy Honda Wave", {"out_of_scope": True}, "Ngoài phạm vi: xe máy"),
    ("điện thoại iPhone 15 Pro", {"out_of_scope": True}, "Ngoài phạm vi: điện thoại"),
    ("bánh mì kẹp thịt", {"out_of_scope": True}, "Ngoài phạm vi: đồ ăn"),
    ("son môi dưỡng ẩm MAC", {"out_of_scope": True}, "Ngoài phạm vi: mỹ phẩm trang điểm"),
    ("nước hoa Chanel cao cấp", {"out_of_scope": True}, "Ngoài phạm vi: nước hoa (bị loại khỏi catalog)"),
    ("bàn ghế phòng khách gỗ sồi", {"out_of_scope": True}, "Ngoài phạm vi: nội thất gia đình"),
    ("máy tính xách tay Dell XPS", {"out_of_scope": True}, "Ngoài phạm vi: laptop máy tính")
]
for text, exp, note in out_of_scope:
    add_query("L", text, exp, note)

# --- Group M: Mơ hồ / xung đột ---
ambiguous = [
    ("váy nam màu xanh", {"conflict": True, "note": "Váy thường là nữ nhưng truy vấn nam"}, "Xung đột: váy nam"),
    ("đồ ngủ nam màu hồng nữ tính", {"conflict": True}, "Xung đột: đồ ngủ nam màu hồng"),
    ("áo ngực nam", {"conflict": True}, "Xung đột: áo ngực nam"),
    ("đồ thời trang đẹp", {"ambiguous": True}, "Mơ hồ: quá chung chung"),
    ("món đồ gì đó màu đỏ", {"ambiguous": True, "baseColour": "Red"}, "Mơ hồ: chỉ có màu sắc")
]
for text, exp, note in ambiguous:
    add_query("M", text, exp, note)

# --- Group N: Biên / lỗi injection / định dạng lạ ---
edge_cases = [
    ("", {"empty": True}, "Biên: xâu rỗng"),
    ("    ", {"empty": True}, "Biên: chỉ toàn khoảng trắng"),
    ("a", {"edge": True}, "Biên: 1 ký tự"),
    ("X", {"edge": True}, "Biên: 1 ký tự hoa"),
    ("a" * 5000, {"edge": True}, "Biên: 5000 ký tự lặp"),
    ("<script>alert('xss')</script>", {"injection": True}, "Bảo mật: XSS Injection"),
    ("'; DROP TABLE catalog; --", {"injection": True}, "Bảo mật: SQL Injection"),
    ("áo thun\nnam\nmàu xanh", {"articleType": "Tshirts", "gender": "Men", "baseColour": "Blue"}, "Định dạng: chứa ký tự xuống dòng"),
    (" áo\tthun\r\nnam ", {"articleType": "Tshirts", "gender": "Men"}, "Định dạng: tab và crlf"),
    ("こんにちは T-shirt", {"articleType": "Tshirts"}, "Đa ngữ: Chữ Nhật Kanji/Hiragana"),
    ("你好 运动鞋", {"articleType": "Sports Shoes"}, "Đa ngữ: Chữ Hán giản thể"),
    ("Áo thun \u200B\u200C\u200D nam", {"articleType": "Tshirts", "gender": "Men"}, "Unicode zero-width characters")
]
for text, exp, note in edge_cases:
    add_query("N", text, exp, note)

# --- Group O: Lặp 5 lần cùng một câu để test tính tất định ---
for rep_i in range(1, 6):
    add_query("O", "áo thun Puma nam màu vàng", {"brand": "Puma", "articleType": "Tshirts", "gender": "Men", "baseColour": "Yellow"}, f"Lặp lần {rep_i}/5 test determinism")

# --- Group P: Đúng 7 câu gợi ý mẫu của App ---
app_samples = [
    ("túi sách màu hồng", {"articleType": "Handbags", "baseColour": "Pink"}, "Mẫu app 1: túi sách màu hồng"),
    ("áo thun Puma nam", {"brand": "Puma", "articleType": "Tshirts", "gender": "Men"}, "Mẫu app 2: áo thun Puma nam"),
    ("váy dài màu đỏ dạ hội", {"articleType": "Dresses", "baseColour": "Red", "usage": "Party"}, "Mẫu app 3: váy dài màu đỏ dạ hội"),
    ("giày cao gót nữ màu đen", {"articleType": "Heels", "gender": "Women", "baseColour": "Black"}, "Mẫu app 4: giày cao gót nữ màu đen"),
    ("ao do nam", {"baseColour": "Red", "gender": "Men"}, "Mẫu app 5: ao do nam (không dấu)"),
    ("đồ đi biển", {"usage": "Sports"}, "Mẫu app 6: đồ đi biển"),
    ("túi xách nữ Lavie", {"articleType": "Handbags", "gender": "Women", "brand": "Lavie", "out_of_catalog_brand": True}, "Mẫu app 7: túi xách nữ Lavie (thương hiệu ngoài catalog)")
]
for text, exp, note in app_samples:
    add_query("P", text, exp, note)

# 3. Save to queries.csv
queries_df = pd.DataFrame(queries)
csv_path = ROOT_DIR / "eval/text2img_test/queries.csv"
queries_df.to_csv(csv_path, index=False, encoding="utf-8")

# Print stats
print(f"\n=======================================================")
print(f"BỘ TRUY VẤN ĐÃ SINH: Tổng cộng {len(queries_df)} câu truy vấn")
print(f"Lưu tại: {csv_path}")
print(f"=======================================================")
stats = queries_df["nhóm"].value_counts().sort_index()
for grp, cnt in stats.items():
    grp_name = {
        "A": "Loại sản phẩm (articleType >= 20)",
        "B": "Loại + màu sắc",
        "C": "Loại + giới tính",
        "D": "Loại + màu sắc + giới tính",
        "E": "Thương hiệu (brand >= 30)",
        "F": "Thuộc tính khác (mùa, dịp, dáng)",
        "G": "Đồng nghĩa (áo phông/thun, balo...)",
        "H": "Biến thể (không dấu, lỗi gõ, HOA, emoji)",
        "I": "Độ dài câu (1 từ, tự nhiên, >128 tokens)",
        "J": "Tiếng Anh & trộn Việt - Anh",
        "K": "Phủ định, ràng buộc, đa màu",
        "L": "Ngoài phạm vi (xe máy, iPhone...)",
        "M": "Mơ hồ, xung đột (váy nam...)",
        "N": "Biên, Injection, XSS, Unicode lạ",
        "O": "Tính tất định (lặp lại 5 lần)",
        "P": "7 câu gợi ý mẫu của App"
    }.get(grp, "")
    print(f"Nhóm {grp} ({grp_name:40s}): {cnt:4d} câu")
print(f"=======================================================\n")
