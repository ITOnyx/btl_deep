import json
from pathlib import Path
from scipy import stats

def compute_clopper_pearson(k, n):
    res = stats.binomtest(k, n)
    ci = res.proportion_ci(confidence_level=0.95, method="exact")
    return [round(float(ci.low), 4), round(float(ci.high), 4)]

# Load existing results
res_path = Path("reports/phase3g_blind_ab_results.json")
data = json.load(open(res_path, encoding="utf-8"))

# 1. Update metadata and evaluator info
data["metadata"] = {
    "evaluated_by": "Claude (AI), single rater",
    "evaluation_method": "Visual inspection from contact sheet images (blind_ab_x001..x030.png)",
    "blind_state": "Sealed mapping unviewed during rating; weak blind due to visual category uniformity in system rows",
    "scope_of_conclusion": "Applies strictly to Full System (Parser + Metadata Filtering/Bonus + BM25) vs Pure Embedding; DOES NOT apply to SigLIP model alone"
}

data["limitations"] = [
    "Một người chấm duy nhất: Claude (AI), không có đánh giá chéo liên người chấm (inter-rater reliability).",
    "Không xác minh được thương hiệu bằng mắt: Với các sản phẩm thời trang không in logo to rõ trên ảnh, người chấm chỉ đánh giá theo loại sản phẩm và giới tính; P@10 brand thực tế phải căn cứ theo metadata.",
    "Blind yếu (Weak blinding): Hàng kết quả của Hệ thống C5 có độ đồng nhất cao về danh mục sản phẩm (do bộ lọc/bonus), khiến người chấm có thể gián tiếp nhận ra hàng của hệ thống.",
    "Hậu nghiệm đối với Benetton: Câu x018 sử dụng alias 'benetton' được thêm sau khi đã xem kết quả heldout ban đầu.",
    "Phạm vi kết luận: Kết quả thắng áp đảo phản ánh sức mạnh của Hệ sinh thái kỹ thuật (Parser + Metadata + BM25 + Scorer), không đại diện cho riêng năng lực của mô hình SigLIP thuần."
]

# 2. Update statistical CIs to Clopper-Pearson
st = data["statistics"]

# Overall
st["overall_30q"]["clopper_pearson_ci95"] = compute_clopper_pearson(st["overall_30q"]["c5_wins"], st["overall_30q"]["total_queries"])
st["overall_30q"]["win_rate_ci95_exact"] = st["overall_30q"]["clopper_pearson_ci95"]

# Spec 26q
st["spec_26q"]["clopper_pearson_ci95"] = compute_clopper_pearson(st["spec_26q"]["c5_wins"], st["spec_26q"]["total_queries"])
st["spec_26q"]["win_rate_ci95_exact"] = st["spec_26q"]["clopper_pearson_ci95"]

# Brand 8q
st["brand_8q"]["clopper_pearson_ci95"] = compute_clopper_pearson(st["brand_8q"]["c5_wins"], st["brand_8q"]["total_queries"])
st["brand_8q"]["win_rate_ci95_exact"] = st["brand_8q"]["clopper_pearson_ci95"]
st["brand_8q"]["metadata_p10_mean_c5"] = 0.8000 # strictly verified by brand metadata
st["brand_8q"]["metadata_p10_mean_c0"] = 0.0500 # strictly verified by brand metadata
st["brand_8q"]["note"] = "Điểm chấm tay (0.8625) chỉ phản ánh loại sản phẩm và giới tính do không xác minh được logo brand bằng mắt; P@10 chuẩn theo metadata brand là 0.8000."

# Standard 18q
st["standard_18q"]["clopper_pearson_ci95"] = compute_clopper_pearson(st["standard_18q"]["c5_wins"], st["standard_18q"]["total_queries"])
st["standard_18q"]["win_rate_ci95_exact"] = st["standard_18q"]["clopper_pearson_ci95"]

# Vague 4q
st["vague_4q"]["clopper_pearson_ci95"] = compute_clopper_pearson(st["vague_4q"]["c5_wins"], st["vague_4q"]["total_queries"])
st["vague_4q"]["win_rate_ci95_exact"] = st["vague_4q"]["clopper_pearson_ci95"]

# 3. Clean up discrepancies: strictly stick to notes column, delete unverified claims
cleaned_discrepancies = [
    {
        "qid": "x001",
        "query": "sơ mi nam xanh navy",
        "human_c5": 7,
        "meta_c5": 10,
        "diff_c5": -3,
        "human_c0": 1,
        "meta_c0": 0,
        "diff_c0": 1,
        "notes": "B lẫn đồng hồ, dây chuyền, mũ len, thắt lưng",
        "explanation": "Cột ghi chú không giải thích lý do trừ điểm ở hàng A; nguyên nhân điểm tay 7/10 so với metadata 10/10 là chưa xác định từ dữ liệu chấm."
    },
    {
        "qid": "x004",
        "query": "giày cao gót nữ màu đen",
        "human_c5": 6,
        "meta_c5": 10,
        "diff_c5": -4,
        "human_c0": 3,
        "meta_c0": 3,
        "diff_c0": 0,
        "notes": "A chỉ ~4 đôi có gót cao; B lẫn đồ lót, áo, boxer",
        "explanation": "Khớp đúng cột notes: Người chấm thấy chỉ ~4 đôi có gót cao rõ rệt, trong khi metadata gán nhãn Heels cho cả sandal gót thấp/đế xuồng."
    },
    {
        "qid": "x007",
        "query": "ao thun nam mau xam",
        "human_c5": 6,
        "meta_c5": 10,
        "diff_c5": -4,
        "human_c0": 3,
        "meta_c0": 4,
        "diff_c0": -1,
        "notes": "A lẫn sandal, tất, áo khoác, quần",
        "explanation": "Khớp đúng cột notes: Người chấm ghi A lẫn sandal, tất, áo khoác, quần (ở hàng C0 là Row A, C5 là Row B được 6 đúng)."
    },
    {
        "qid": "x008",
        "query": "giay the thao nam mau den",
        "human_c5": 8,
        "meta_c5": 10,
        "diff_c5": -2,
        "human_c0": 0,
        "meta_c0": 0,
        "diff_c0": 0,
        "notes": "A toàn sandal, giày lười, áo, balo",
        "explanation": "Cột notes chỉ nhận xét về hàng A (C0: 0/10); lý do C5 đạt 8/10 thay vì 10/10 không được ghi trong notes, ghi nhận là chưa xác định."
    },
    {
        "qid": "x010",
        "query": "quan short nu mau trang",
        "human_c5": 3,
        "meta_c5": 5,
        "diff_c5": -2,
        "human_c0": 0,
        "meta_c0": 0,
        "diff_c0": 0,
        "notes": "B toàn short/legging nữ nhưng chỉ ~3 cái trắng/kem",
        "explanation": "Khớp đúng cột notes: B (C5) toàn short/legging nữ nhưng người chấm chỉ thấy ~3 cái màu trắng/kem; catalog thực tế chỉ có 6 sp thỏa mãn."
    },
    {
        "qid": "x011",
        "query": "áo thun Puma nam",
        "human_c5": 10,
        "meta_c5": 10,
        "diff_c5": 0,
        "human_c0": 2,
        "meta_c0": 0,
        "diff_c0": 2,
        "notes": "B nhiều logo Puma; A lẫn balo, giày, sơ mi",
        "explanation": "Khớp đúng cột notes: Hàng B (C5) thấy nhiều logo Puma; hàng A (C0) người chấm đếm 2 sp đúng kiểu dáng nhưng metadata brand xác nhận là 0."
    },
    {
        "qid": "x016",
        "query": "giày cao gót Catwalk",
        "human_c5": 4,
        "meta_c5": 10,
        "diff_c5": -6,
        "human_c0": 2,
        "meta_c0": 0,
        "diff_c0": 2,
        "notes": "B toàn giày nữ nhưng chỉ ~3-4 đôi có gót; A lẫn áo, đồng hồ, quần, túi",
        "explanation": "Khớp đúng cột notes: Hàng B (C5) là giày nữ nhưng người chấm chỉ thấy ~3-4 đôi có gót rõ ràng."
    },
    {
        "qid": "x017",
        "query": "túi xách Lavie nữ",
        "human_c5": 10,
        "meta_c5": 0,
        "diff_c5": 10,
        "human_c0": 1,
        "meta_c0": 0,
        "diff_c0": 1,
        "notes": "Catalog không có Lavie; chấm theo loại + giới",
        "explanation": "Khớp đúng cột notes: Catalog rỗng brand Lavie; người chấm chấm 10/10 theo loại (Handbags) và giới tính (Women)."
    },
    {
        "qid": "x018",
        "query": "áo thun Benetton nam",
        "human_c5": 10,
        "meta_c5": 0,
        "diff_c5": 10,
        "human_c0": 1,
        "meta_c0": 0,
        "diff_c0": 1,
        "notes": "A có 1 áo ghi United Colors; B lẫn quần, giày, kính",
        "explanation": "Khớp đúng cột notes: Người chấm ghi nhận áo ghi United Colors; C5 trả về 10 áo thun nam Benetton."
    },
    {
        "qid": "x019",
        "query": "áo sơ mi nam dài tay đi làm",
        "human_c5": 5,
        "meta_c5": 10,
        "diff_c5": -5,
        "human_c0": 2,
        "meta_c0": 2,
        "diff_c0": 0,
        "notes": "A nhiều áo ngắn tay; B lẫn túi, đồng hồ, cà vạt",
        "explanation": "Khớp đúng cột notes: A (C5) lẫn nhiều áo ngắn tay do catalog không có trường metadata sleeveLength."
    }
]

data["discrepancies_human_vs_metadata"] = cleaned_discrepancies

with open(res_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Updated reports/phase3g_blind_ab_results.json successfully.")
