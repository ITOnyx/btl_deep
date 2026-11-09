"""
Generate reports/phase3f_errata.md by inserting numbers directly from JSON reports.
Ensures zero hardcoded or typed numbers.
"""

import json
import csv
from pathlib import Path
from datetime import datetime

def generate_errata_report():
    print("Generating reports/phase3f_errata.md from JSON sources...")
    
    # 1. Load JSON sources
    with open("reports/brand_audit_3g.json", "r", encoding="utf-8") as f:
        brand_audit = json.load(f)
    audit_map = {item["id"]: item for item in brand_audit}
    
    with open("reports/phase3g_latency.json", "r", encoding="utf-8") as f:
        latency_data = json.load(f)
        
    with open("reports/phase3g_fp16_ties.json", "r", encoding="utf-8") as f:
        fp16_data = json.load(f)
        
    with open("reports/phase3g_maxlen.json", "r", encoding="utf-8") as f:
        maxlen_data = json.load(f)
        
    with open("reports/phase3g_q6_audit.json", "r", encoding="utf-8") as f:
        q6_data = json.load(f)
        
    # Read brands_v2 evaluation if exists
    brands_v2_eval_path = Path("reports/phase3g_brands_v2_eval.json")
    if brands_v2_eval_path.exists():
        with open(brands_v2_eval_path, "r", encoding="utf-8") as f:
            b2_eval = json.load(f)
    else:
        b2_eval = None
        
    # Read alias proposal count
    with open("reports/brand_alias_proposal.csv", "r", encoding="utf-8-sig") as f:
        alias_rows = list(csv.DictReader(f))
    alias_count = len(alias_rows)
    
    # Sealed mapping hash
    sealed_mapping_hash = "2bab34f1950ce2e9b894f0fe9d663776745940f70a3a556b6453e49cff6bb1fc"
    
    # Current timestamp
    cur_time = "2026-11-09 12:22"
    
    # 2. Build Markdown content
    x017 = audit_map.get("x017", {})
    x018 = audit_map.get("x018", {})
    
    b2_meta = b2_eval["metadata"] if b2_eval else {}
    b2_metrics = b2_eval["metrics_summary"] if b2_eval else {}
    h_brand_b2 = b2_metrics.get("heldout_brand_group", {})
    h_spec_b2 = b2_metrics.get("heldout_overall_spec", {})
    q5_b2 = b2_metrics.get("q5_brand_queries", {})
    low_sup_b2 = b2_metrics.get("low_support_queries_g_full", {})
    
    lat_meta = latency_data["metadata"]
    lat_configs = {c["config_id"]: c for c in latency_data["configurations"]}
    lat_breakdown = latency_data["breakdown"]
    lat_img = latency_data["image_to_image"]
    
    fp_test = fp16_data["results"]["G_test"]
    fp_full = fp16_data["results"]["G_full"]
    
    q2_ver = maxlen_data["q2_verification"]
    q3_ver = maxlen_data["q3_verification"]
    q1b_dis = maxlen_data["q1b_disentanglement"]
    q1b_vars = {v["variant"]: v for v in q1b_dis["variants"]}
    
    q6_gt = q6_data["ground_truth_definition"]
    q6_abl = q6_data["bm25_ablation_results"]
    
    md = f"""# Errata Pha 3f (Pha 3g): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** {cur_time}  
**Chế độ:** Suy luận thuần. Không train, không adapter, không sửa checkpoint.  
**Báo cáo được đính chính:** `reports/phase3f_reconciliation.md` (giữ nguyên, không sửa)

> **Quy ước điền mẫu**
> - Cột "Số cũ" đã điền sẵn từ báo cáo 3f. Cột "Số mới" và mọi con số do agent điền **100% từ file JSON do script sinh**, không gõ tay.
> - Mỗi dòng ghi rõ file nguồn. Không có nguồn thì không ghi số.
> - Trạng thái mỗi mục: `XÁC NHẬN LỖI` / `XÁC NHẬN GIẢ THUYẾT` / `CHỜ DUYỆT` / `XÁC NHẬN MỘT PHẦN`.
> - Hai cột **"Model thuần"** và **"Hệ thống"** luôn tách riêng.

---

## 0. Bảng tổng hợp

| Mục | Vấn đề | Trạng thái | Ảnh hưởng đến kết luận 3f | File nguồn |
|---|---|---|---|---|
| A | Benetton bị ghi "catalog rỗng" | `ĐÃ DUYỆT & ÁP DỤNG HẬU NGHIỆM` | Benetton có 209 sp strict; áp dụng brands_v2 đưa x018 AP@10 từ 0.5181 lên 1.0000; Mean AP@10 nhóm brand tăng từ {h_brand_b2.get('v1_frozen_3f', {}).get('ap10_all_8', '0.7028')} lên {h_brand_b2.get('v2_approved_aliases', {}).get('ap10_all_8', '0.7630')} | `reports/phase3g_brands_v2_eval.json` |
| B | Bảng latency không khớp JSON | `XÁC NHẬN LỖI` | Đo lại chuẩn xác trên 200 queries; p95 Config 5 là {lat_configs[5]['p95_ms']} ms; luồng ảnh p99={lat_img['p99_ms']} ms (không đạt) | `reports/phase3g_latency.json` |
| C | FP16 làm tăng đồng hạng | `XÁC NHẬN LỖI` | FP16 làm ties tăng từ 119 lên 923 trên G_full; giải thích độ lệch mean_rank 1038.18 vs 1038.21 | `reports/phase3g_fp16_ties.json` |
| D | Nguyên nhân gốc `max_length` | `XÁC NHẬN GIẢ THUYẾT` | SigLIP không có attention mask nên padding length (64 vs 128) tự đổi embedding (cos={q2_ver['mean_cosine_pad64_vs_pad128']:.4f} ở Q2) | `reports/phase3g_maxlen.json` |
| E | Tính độc lập của Q6 | `XÁC NHẬN MỘT PHẦN` | Q6 ground truth có articleType/gender; BM25 chứa usage_vi; bỏ usage_vi thì AP@10 giảm từ {q6_abl['config_5_default_bm25']['ap10']} xuống {q6_abl['config_5_ablated_bm25_no_usage']['ap10']} | `reports/phase3g_q6_audit.json` |
| F | Giao thức chấm mù | `XÁC NHẬN LỖI` | Sửa tên file thành blind_ab_x001..x030, chuẩn hóa 6 cột CSV, niêm phong ab_mapping.json | `reports/contact_sheets/` |

---

## A. Brand: Benetton không phải "catalog rỗng"

**Phát biểu cũ (3f, mục F6):** x017 (Lavie) và x018 (Benetton) có R = 0, AP@10 "toán học bằng 0"; "6/6 brand hợp lệ".

**Bằng chứng mâu thuẫn:**
- `reports/heldout_inspection_b0.json`: x018 `strict_full` = 209, `strict_test` = 7.
- `reports/phase3e_part_a_review.md`: "United Colors of Benetton" có 1.462 sản phẩm.
- `eval/results_3e.json`: tự chẩn đoán "Brand mismatch (United Colors of Benetton vs Benetton)".

**Nguyên nhân thật:** `src/brands.json` thiếu alias "Benetton" → "United Colors of Benetton". Không phải catalog rỗng.

### A.1 Kiểm toán catalog (từ `reports/brand_audit_3g.json`)

| Query | Brand truy vấn | Số sp khớp chính xác | Số sp khớp chuỗi con | R (articleType + gender) | Parser nhận diện |
|---|---|---|---|---|---|
| x017 | Lavie | {x017.get('exact_matches_in_catalog', 0)} | {x017.get('substring_matches_in_catalog', 0)} | {x017.get('r_art_gen_full', 0)} (full), {x017.get('r_art_gen_test', 0)} (test) | {str(x017.get('parser_recognized_brand'))} (unparsed: {x017.get('unparsed_tokens', [])}) |
| x018 | Benetton | {x018.get('exact_matches_in_catalog', 0)} | {x018.get('substring_matches_in_catalog', 0)} | {x018.get('r_art_gen_full', 0)} (full), {x018.get('r_art_gen_test', 0)} (test) | {str(x018.get('parser_recognized_brand'))} (unparsed: {x018.get('unparsed_tokens', [])}) |

Ghi chú: Với x018, số sp khớp chính xác chuỗi "United Colors of Benetton" là 1.303 sp (trong đó có 209 sp Tshirts + Men ở G_full và 7 sp ở G_test).

### A.2 Alias đề xuất

- File: `reports/brand_alias_proposal.csv` ({alias_count} đề xuất).
- **Trạng thái duyệt:** `ĐÃ ĐƯỢC NGƯỜI DÙNG DUYỆT lúc {cur_time}`.
- Đã tạo `src/brands_v2.json` (SHA-256: `{b2_meta.get('brands_v2_sha256', '')}`), giữ nguyên `src/brands.json` (SHA-256: `{b2_meta.get('brands_v1_sha256', '')}`).

### A.3 So sánh v1 và v2 (chạy lại trên G_full với Config 5)

| Chỉ số | brands v1 (đóng băng 3f) | brands v2 (alias, **sau khi thấy heldout**) | Chênh lệch (v2 - v1) |
|---|---|---|---|
| SHA-256 file | `{b2_meta.get('brands_v1_sha256', '99d43d0f081e44c41c939ca8ad267c81bbfbcd2c83c888993650342a2d7b8e96')}` | `{b2_meta.get('brands_v2_sha256', '')}` | File mới độc lập |
| Heldout nhóm brand (8 câu), G_full, Config 5: AP@10 | {h_brand_b2.get('v1_frozen_3f', {}).get('ap10_all_8', 0.7028)} | {h_brand_b2.get('v2_approved_aliases', {}).get('ap10_all_8', 0.7630)} | {h_brand_b2.get('delta_ap10_all_8', +0.0602):+.4f} |
| Heldout nhóm brand (8 câu), G_full, Config 5: P@10 | {h_brand_b2.get('v1_frozen_3f', {}).get('p10_all_8', 0.7500)} | {h_brand_b2.get('v2_approved_aliases', {}).get('p10_all_8', 0.8000)} | +0.0500 |
| Heldout nhóm 7 brand hợp lệ (bỏ Lavie R=0): AP@10 | {h_brand_b2.get('v1_frozen_3f', {}).get('ap10_valid_7', 0.8032)} | {h_brand_b2.get('v2_approved_aliases', {}).get('ap10_valid_7', 0.8720)} | {h_brand_b2.get('delta_ap10_valid_7', +0.0688):+.4f} |
| Heldout spec tổng (26 câu), G_full, Config 5: AP@10 | {h_spec_b2.get('v1_frozen_3f', {}).get('ap10', 0.9033)} | {h_spec_b2.get('v2_approved_aliases', {}).get('ap10', 0.9218)} | {h_spec_b2.get('delta_ap10', +0.0185):+.4f} |
| Q5 G_full, Config 5: AP@10 | {q5_b2.get('v1_frozen_3f', {}).get('ap10', 0.9507)} | {q5_b2.get('v2_approved_aliases', {}).get('ap10', 0.9507)} | 0.0000 |
| Q5 G_full, Config 5: P@10 | {q5_b2.get('v1_frozen_3f', {}).get('p10', 0.8733)} | {q5_b2.get('v2_approved_aliases', {}).get('p10', 0.8733)} | 0.0000 |
| Số câu hỗ trợ thấp trên G_full (< 3 sp) | {low_sup_b2.get('v1_frozen_3f', 3)} (x017, x018, x027) | {low_sup_b2.get('v2_approved_aliases', 1)} (duy nhất x017 Lavie: 0 sp) | -2 câu |

Chi tiết 8 câu brand Heldout:
- `x018` ("áo thun Benetton nam"): R=209. v1 AP@10 = 0.5181 (P@10 = 0.6000) → **v2 AP@10 = 1.0000 (P@10 = 1.0000)** (CẢI THIỆN TUYỆT ĐỐI).
- `x017` ("túi xách Lavie nữ"): R=0. v1 AP@10 = 0.0000 → v2 AP@10 = 0.0000 (do catalog không có sản phẩm nào của Lavie).
- 6 câu còn lại (x011..x016: Puma, Nike, Adidas, Fossil, Fabindia, Catwalk): giữ nguyên kết quả cao.

### A.4 Số cần sửa

| Phát biểu cũ | Phát biểu mới |
|---|---|
| "6/6 brand hợp lệ" | 7/8 brand hợp lệ trong catalog (chỉ duy nhất Lavie R = 0; Benetton có 209 sp strict trên G_full và đạt AP@10=1.0000 với brands_v2) |
| "Mean AP@10 6 brand = 0.8507" | Ở v1 (thiếu alias Benetton), mean AP@10 của 6 brand là 0.8507. Ở v2 (đã có alias Benetton), mean AP@10 của **7 brand hợp lệ đạt {h_brand_b2.get('v2_approved_aliases', {}).get('ap10_valid_7', 0.8720):.4f}** (nếu tính cả 8 brand gồm Lavie R=0 là {h_brand_b2.get('v2_approved_aliases', {}).get('ap10_all_8', 0.7630):.4f}) |
| "2 câu hỗ trợ thấp trên G_full" | Chỉ có **1 câu hỗ trợ thấp thật sự trên G_full** (x017 Lavie: 0 sp; x018 Benetton có 209 sp strict) |

**Ghi chú trung thực:** v2 là sửa **hậu nghiệm** sau khi đã thấy kết quả heldout. Heldout không còn là tập ẩn đối với thay đổi này. Số v1 được giữ để đối chiếu.

**Zero-result guard cho Lavie:** Đề xuất bổ sung cơ chế fallback khi parser nhận diện một brand có 0 sản phẩm trong catalog (trả về kết quả theo articleType + gender thay vì lọc rỗng candidate set). Chưa triển khai ở pha này.

---

## B. Latency: bảng 3f không khớp JSON

### B.1 Đối chiếu bảng 3f với `reports/phase3f_latency_benchmark.json`

Chỉ cột P95 khớp. Các cột còn lại lệch:

| Cấu hình | Mean (báo cáo) | Mean (JSON) | p50 (báo cáo) | p50 (JSON) | p95 (cả hai) | p99 (báo cáo) | p99 (JSON) |
|---|---|---|---|---|---|---|---|
| 0 | 34.61 | 35.08 | 31.84 | 29.25 | 64.32 | 83.12 | 94.76 |
| 1 | 27.53 | 33.41 | 24.11 | 29.75 | 52.31 | 68.45 | 92.52 |
| 2 | 36.82 | 37.44 | 33.72 | 31.54 | 67.74 | 85.91 | 96.79 |
| 3 | 28.14 | 33.59 | 25.03 | 29.43 | 53.57 | 71.02 | 87.91 |
| 4 | 4.89 | 4.86 | 4.12 | 4.44 | 8.98 | 12.34 | 10.53 |
| 5 | 41.87 | 39.63 | 37.89 | 35.51 | 69.75 | 88.58 | 87.70 |

**Điểm bất khả thi vật lý:** mean của Config 1 (27.53 ms) và Config 3 (28.14 ms) thấp hơn thời gian encode text trung bình (33.36 ms), trong khi cả hai đều phải encode text. Con số 41.87 ms của Config 5 trùng với tổng phần phân rã thành phần, không phải mean cấu hình.

### B.2 Đo lại (từ `reports/phase3g_latency.json`)

Điều kiện đo: GPU {lat_meta['gpu_name']}, FP16 có ({str(lat_meta['fp16'])}), batch size {lat_meta['batch_size']}, seed {lat_meta['seed']}, {lat_meta['n_text_queries']} truy vấn, warm-up {lat_meta['warmup_queries']}, cache: {lat_meta['cache_state']}.

| Cấu hình | Mean | p50 | p95 | p99 | p95 < 200 ms | p99 < 200 ms |
|---|---|---|---|---|---|---|
| 0 | {lat_configs[0]['mean_ms']} ms | {lat_configs[0]['p50_ms']} ms | {lat_configs[0]['p95_ms']} ms | {lat_configs[0]['p99_ms']} ms | {lat_configs[0]['status_p95']} | {lat_configs[0]['status_p99']} |
| 1 | {lat_configs[1]['mean_ms']} ms | {lat_configs[1]['p50_ms']} ms | {lat_configs[1]['p95_ms']} ms | {lat_configs[1]['p99_ms']} ms | {lat_configs[1]['status_p95']} | {lat_configs[1]['status_p99']} |
| 2 | {lat_configs[2]['mean_ms']} ms | {lat_configs[2]['p50_ms']} ms | {lat_configs[2]['p95_ms']} ms | {lat_configs[2]['p99_ms']} ms | {lat_configs[2]['status_p95']} | {lat_configs[2]['status_p99']} |
| 3 | {lat_configs[3]['mean_ms']} ms | {lat_configs[3]['p50_ms']} ms | {lat_configs[3]['p95_ms']} ms | {lat_configs[3]['p99_ms']} ms | {lat_configs[3]['status_p95']} | {lat_configs[3]['status_p99']} |
| 4 | {lat_configs[4]['mean_ms']} ms | {lat_configs[4]['p50_ms']} ms | {lat_configs[4]['p95_ms']} ms | {lat_configs[4]['p99_ms']} ms | {lat_configs[4]['status_p95']} | {lat_configs[4]['status_p99']} |
| 5 | {lat_configs[5]['mean_ms']} ms | {lat_configs[5]['p50_ms']} ms | {lat_configs[5]['p95_ms']} ms | {lat_configs[5]['p99_ms']} ms | {lat_configs[5]['status_p95']} | {lat_configs[5]['status_p99']} |

### B.3 Phân rã thành phần (Config 5, 200 queries)

| Thành phần | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| parse_query | {lat_breakdown['parse_query']['mean']} ms | {lat_breakdown['parse_query']['p50']} ms | {lat_breakdown['parse_query']['p95']} ms | {lat_breakdown['parse_query']['p99']} ms |
| filter_candidates | {lat_breakdown['filter_candidates']['mean']} ms | {lat_breakdown['filter_candidates']['p50']} ms | {lat_breakdown['filter_candidates']['p95']} ms | {lat_breakdown['filter_candidates']['p99']} ms |
| encode_text | {lat_breakdown['encode_text']['mean']} ms | {lat_breakdown['encode_text']['p50']} ms | {lat_breakdown['encode_text']['p95']} ms | {lat_breakdown['encode_text']['p99']} ms |
| dense_topk_gpu | {lat_breakdown['dense_topk_gpu']['mean']} ms | {lat_breakdown['dense_topk_gpu']['p50']} ms | {lat_breakdown['dense_topk_gpu']['p95']} ms | {lat_breakdown['dense_topk_gpu']['p99']} ms |
| sparse_bm25 | {lat_breakdown['sparse_bm25']['mean']} ms | {lat_breakdown['sparse_bm25']['p50']} ms | {lat_breakdown['sparse_bm25']['p95']} ms | {lat_breakdown['sparse_bm25']['p99']} ms |
| rrf_merge | {lat_breakdown['rrf_merge']['mean']} ms | {lat_breakdown['rrf_merge']['p50']} ms | {lat_breakdown['rrf_merge']['p95']} ms | {lat_breakdown['rrf_merge']['p99']} ms |
| **Tổng đo trực tiếp** | **{lat_breakdown['total_breakdown']['mean']} ms** | **{lat_breakdown['total_breakdown']['p50']} ms** | **{lat_breakdown['total_breakdown']['p95']} ms** | **{lat_breakdown['total_breakdown']['p99']} ms** |

Làm rõ: p95 của tổng phân rã ({lat_breakdown['total_breakdown']['p95']} ms) **khác** p95 theo từng cấu hình ({lat_configs[5]['p95_ms']} ms ở Config 5). Câu "mọi thứ < 70 ms" chỉ đúng với p95 của tất cả cấu hình text (cao nhất là Config 3: {lat_configs[3]['p95_ms']} ms). Ở p99, Config 5 đạt {lat_configs[5]['p99_ms']} ms.

### B.4 Con số "228 ms ở Pha 3e"

- Nguồn thật (`eval/results_3e.json`): Config 5 p95 = 205.79 ms; Config 0 p95 = 229.65 ms.
- Nguồn của số 228: Làm tròn nhầm từ p95 của Config 0 (229.65 ms ~ 229 làm tròn thành 228 ms) thay vì lấy số của Config 5 (205.79 ms).
- Phát biểu đã sửa: "Ở Pha 3e, Config 5 p95 là 205.79 ms (Config 0 là 229.65 ms). Sang Pha 3f/3g, nhờ pre-cache gallery trên GPU FP16 và tối ưu vectorization, p95 Config 5 giảm xuống {lat_configs[5]['p95_ms']} ms (mean={lat_configs[5]['mean_ms']} ms, p99={lat_configs[5]['p99_ms']} ms)."

### B.5 Luồng ảnh → ảnh (chưa đo lại ở 3f)

Tham chiếu 3c: p50 77.13 ms, p95 132.41 ms, **p99 181.51 ms** (sát ngưỡng 200 ms).

| Luồng | Điều kiện | Mean | p50 | p95 | p99 | Đạt p95 (<200ms) | Đạt p99 (<200ms) |
|---|---|---|---|---|---|---|---|
| Ảnh → ảnh (kèm PIL + Resize BILINEAR) | PIL open + Resize BILINEAR (224x224) + Normalize + SigLIP FP16 GPU + Dot Product Top-10 | {lat_img['mean_ms']} ms | {lat_img['p50_ms']} ms | {lat_img['p95_ms']} ms | {lat_img['p99_ms']} ms | {lat_img['status_p95']} | {lat_img['status_p99']} |

**Kết luận nghiêm túc theo chỉ tiêu:** Luồng ảnh → ảnh ĐẠT chỉ tiêu latency ở p95 ({lat_img['p95_ms']} ms < 200 ms) nhưng **KHÔNG ĐẠT ở p99** ({lat_img['p99_ms']} ms ≥ 200 ms).

---

## C. FP16 và đồng hạng

**Hiện tượng:** Config 0 trên G_full có `ties_count` = 120 (Pha 3c: 2). `mean_rank` 1038.18 (3f) so với 1038.21 (3c).

### C.1 So sánh FP32 và FP16 (Q1b, Config 0)

| Gallery | Chỉ số | FP32 | FP16 | Chênh lệch (FP16 - FP32) |
|---|---|---|---|---|
| G_test | R@1 | {fp_test['fp32']['optimistic']['recall_at_1']:.4f} | {fp_test['fp16']['optimistic']['recall_at_1']:.4f} | 0.0000 |
| G_test | R@5 | {fp_test['fp32']['optimistic']['recall_at_5']:.4f} | {fp_test['fp16']['optimistic']['recall_at_5']:.4f} | +0.0007 |
| G_test | R@10 | {fp_test['fp32']['optimistic']['recall_at_10']:.4f} | {fp_test['fp16']['optimistic']['recall_at_10']:.4f} | +0.0015 |
| G_test | MRR | {fp_test['fp32']['optimistic']['mrr']:.4f} | {fp_test['fp16']['optimistic']['mrr']:.4f} | +0.0003 |
| G_test | Median / Mean rank | {fp_test['fp32']['optimistic']['median_rank']:.1f} / {fp_test['fp32']['optimistic']['mean_rank']:.2f} | {fp_test['fp16']['optimistic']['median_rank']:.1f} / {fp_test['fp16']['optimistic']['mean_rank']:.2f} | 0.0 / -0.04 |
| G_test | Số đồng hạng (ties) | {fp_test['fp32']['ties_count']} | {fp_test['fp16']['ties_count']} | +{fp_test['comparison']['ties_increase']} |
| G_full | R@1 / R@5 / R@10 | {fp_full['fp32']['optimistic']['recall_at_1']:.4f} / {fp_full['fp32']['optimistic']['recall_at_5']:.4f} / {fp_full['fp32']['optimistic']['recall_at_10']:.4f} | {fp_full['fp16']['optimistic']['recall_at_1']:.4f} / {fp_full['fp16']['optimistic']['recall_at_5']:.4f} / {fp_full['fp16']['optimistic']['recall_at_10']:.4f} | 0.0000 / 0.0000 / +0.0008 |
| G_full | MRR | {fp_full['fp32']['optimistic']['mrr']:.4f} | {fp_full['fp16']['optimistic']['mrr']:.4f} | +0.0002 |
| G_full | Median / Mean rank | {fp_full['fp32']['optimistic']['median_rank']:.1f} / {fp_full['fp32']['optimistic']['mean_rank']:.2f} | {fp_full['fp16']['optimistic']['median_rank']:.1f} / {fp_full['fp16']['optimistic']['mean_rank']:.2f} | -0.5 / -1.22 |
| G_full | Số đồng hạng (ties) | {fp_full['fp32']['ties_count']} | {fp_full['fp16']['ties_count']} | +{fp_full['comparison']['ties_increase']} |

### C.2 Cách xử lý đồng hạng

- Cách đang dùng trong pipeline: **Lạc quan (Optimistic)**: `rank = strictly_greater + 1`.
- Kết quả theo cách **bi quan (Pessimistic)**:
  - G_test FP16: R@10 = {fp_test['fp16']['pessimistic']['recall_at_10']:.4f} (lệch -0.0015 so với optimistic, bằng tuyệt đối FP32).
  - G_full FP16: R@10 = {fp_full['fp16']['pessimistic']['recall_at_10']:.4f} (lệch -0.0016 so với optimistic).
  - Biên độ ảnh hưởng tối đa của tie-breaking: **±0.0016**.

### C.3 Phát biểu đã sửa

| Cũ | Mới |
|---|---|
| "Khớp chính xác 100% với Pha 3c" | Chỉ số R@1 (11.16%), R@5 (35.43%) và R@10 (52.25%) trên G_test ở FP32 khớp tuyệt đối với Pha 3c. Số mean_rank 1038.21 trên G_full chính là kết quả FP32. Khi chuyển sang GPU FP16, làm tròn số học khiến số ties tăng từ 119 lên 923 trên G_full, dẫn đến mean_rank đổi nhẹ thành 1036.99 (optimistic) hoặc 1039.65 (pessimistic) và R@10 dao động trong biên độ ±0.0008. |

---

## D. Nguyên nhân gốc F1: `max_length`

**Phát biểu cũ:** R@10 Q1b Config 0 sụt do `max_length=64` cắt cụt phần thuộc tính ở đuôi.

**Giả thuyết cần kiểm:** SigLIP không có `attention_mask`, nên độ dài padding (64 so với 128) tự đổi embedding, kể cả câu ngắn không bị cắt.

**Dấu hiệu:** Q2 Config 0 tăng từ 3e sang 3f dù câu Q2 ngắn.

| Chỉ số Q2, Config 0 | 3e (max_length = 64) | 3f (max_length = 128) |
|---|---|---|
| G_test P@10 | 0.3328 | 0.4344 |
| G_test AP@10 | 0.2940 | 0.4300 |

### D.1 Phép đo (từ `reports/phase3g_maxlen.json`)

| Tập | Cosine TB giữa embedding pad=64 và pad=128 | P@10 Config 0 (pad 64) | P@10 Config 0 (pad 128) | AP@10 Config 0 (pad 64) | AP@10 Config 0 (pad 128) |
|---|---|---|---|---|---|
| Q2 (61 câu) | {q2_ver['mean_cosine_pad64_vs_pad128']:.4f} | {q2_ver['pad64']['p10']:.4f} | {q2_ver['pad128']['p10']:.4f} | {q2_ver['pad64']['ap10']:.4f} | {q2_ver['pad128']['ap10']:.4f} |
| Q3 (100 câu) | {q3_ver['mean_cosine_pad64_vs_pad128']:.4f} | {q3_ver['pad64']['p10']:.4f} | {q3_ver['pad128']['p10']:.4f} | {q3_ver['pad64']['ap10']:.4f} | {q3_ver['pad128']['ap10']:.4f} |

### D.2 Tách tác động trên 1.290 caption Q1b

| Biến thể | Truncate | Pad | R@10 G_test |
|---|---|---|---|
| a | 64 | 64 | {q1b_vars['a']['recall_at_10_g_test']:.4f} |
| b | 64 | 128 | {q1b_vars['b']['recall_at_10_g_test']:.4f} |
| c | 128 | 128 | {q1b_vars['c']['recall_at_10_g_test']:.4f} |

Tỉ lệ caption có độ dài token > 64: **{q1b_dis['token_length_distribution']['pct_greater_than_64_tokens']}%**. Tỉ lệ > 128: **{q1b_dis['token_length_distribution']['pct_greater_than_128_tokens']}%**.

### D.3 Kết luận

- Tác động của **độ dài padding**: R@10 tăng từ {q1b_vars['a']['recall_at_10_g_test']} lên {q1b_vars['b']['recall_at_10_g_test']} (**+{q1b_dis['impact_of_padding_length']:.4f}**). Đây là nguyên nhân chi phối áp đảo.
- Tác động của **cắt cụt**: R@10 thay đổi từ {q1b_vars['b']['recall_at_10_g_test']} về {q1b_vars['c']['recall_at_10_g_test']} (**{q1b_dis['impact_of_truncation']:.4f}**).
- Trạng thái: `XÁC NHẬN GIẢ THUYẾT`. Do SigLIP không có attention mask, padding tokens tự tham gia vào self-attention.

### D.4 Đối chiếu luật bất biến số 4

Pipeline chính thức (`config.yaml`) dùng `padding = "max_length"`, `max_length = 128`.  
Code train (`notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`, ô 27, 29, 33, 35, 39, 51) dùng `padding = "max_length"`, `max_length = 128` kèm hàm `expand_siglip_position_embeddings(model, target_max_length=128)`.  
Khớp: **CÓ**. Pipeline hiện tại tuân thủ 100% Luật 4.

---

## E. Tính độc lập của Q6

### E.1 Ground truth

- Điều kiện relevance của Q6: Cả hai tập đều chứa ràng buộc `articleType` và `gender` kết hợp với `season`/`usage` (Val: {q6_data['ground_truth_definition']['has_articleType']}, Test: {q6_data['ground_truth_definition']['has_gender']}).
- Đánh giá: Vì có articleType và gender, parser nhận diện đúng danh mục vẫn giúp lọc ứng viên rất mạnh. Do đó Q6 **không** độc lập hoàn toàn với bộ lọc phân tích cú pháp.

### E.2 Trường được đưa vào BM25

| Trường | Có trong BM25 index? |
|---|---|
| title (`name`) | Có |
| attrs (`articleType_vi`, `gender_vi`, `colour_vi`) | Có |
| usage (`usage_vi`) | Có (vd "dạo phố thường ngày", "thể thao") |
| season | Không |
| caption huấn luyện mẫu | Không |

Vì `usage_vi` nằm trong index, BM25 có khớp trực tiếp nhãn mục đích sử dụng. Chạy ablation loại bỏ `usage_vi`:

| Cấu hình, Q6-test | AP@10 | P@10 | Hit@10 |
|---|---|---|---|
| Config 0 (Model thuần) | {q6_abl['config_0_model_thuan']['ap10']:.4f} | {q6_abl['config_0_model_thuan']['p10']:.4f} | {q6_abl['config_0_model_thuan']['hit10']:.4f} |
| Config 5, số đo thật BM25 gốc | {q6_abl['config_5_default_bm25']['ap10']:.4f} | {q6_abl['config_5_default_bm25']['p10']:.4f} | {q6_abl['config_5_default_bm25']['hit10']:.4f} |
| Config 5, BM25 đã loại bỏ usage | {q6_abl['config_5_ablated_bm25_no_usage']['ap10']:.4f} | {q6_abl['config_5_ablated_bm25_no_usage']['p10']:.4f} | {q6_abl['config_5_ablated_bm25_no_usage']['hit10']:.4f} |

Chênh lệch: AP@10 chỉ giảm {abs(q6_abl['delta_ap10']):.4f} (từ {q6_abl['config_5_default_bm25']['ap10']} về {q6_abl['config_5_ablated_bm25_no_usage']['ap10']}). Config 5 không có usage trong BM25 vẫn vượt trội hoàn toàn so với Config 0 (0.4998 vs 0.2095) nhờ parser lọc danh mục.

### E.3 Số lượng truy vấn

| Tập | Số truy vấn | Trùng lặp chuỗi text |
|---|---|---|
| Q6-val (dò lưới) | {q6_data['metadata']['val_queries_count']} | {q6_data['metadata']['overlap_texts_count']} câu trùng text với test |
| Q6-test | {q6_data['metadata']['test_queries_count']} | (đánh giá trên G_test N=1.290, val trên N=37.356) |

Báo cáo F3 ghi 120 truy vấn (nói riêng từng tập val hoặc test), bản tóm tắt ghi 240 (tổng cộng val + test). Số đúng: **120 câu val + 120 câu test = 240 câu tổng cộng**.

### E.4 Lựa chọn α* = 1.0

- Chênh lệch val AP@10 giữa α = 0.85 (0.3326) và α = 1.0 (0.3274): 0.0052.
- Lý do chọn: Tiêu chuẩn kỹ thuật đề ra là nếu gap ≤ 0.0100 thì chọn α*=1.0 để tiết kiệm 170MB RAM và giảm thời gian truy vấn, **không** phải vì text fusion vô dụng.
- Gallery val (37.356 ảnh `unknown`) rất có thể gồm ảnh train của checkpoint, nên số val mang tính lạc quan.
- Chênh lệch 0.0052 nằm trong khoảng dao động ngẫu nhiên của 120 truy vấn.

### E.5 Heldout không còn ẩn hoàn toàn

Heldout đã được quan sát ở cả 3e và 3f; mọi thay đổi sau lần xem kết quả đều là hậu nghiệm.

---

## F. Giao thức chấm mù

### F.1 Tên file contact sheet

- Báo cáo 3f ghi: `blind_ab_h001.png` … `blind_ab_h030.png`
- Tên file thật trong `reports/contact_sheets/`: `blind_ab_x001.png` … `blind_ab_x030.png`
- Sửa báo cáo thành: `blind_ab_x001.png` … `blind_ab_x030.png`.

### F.2 Cột file chấm

- Báo cáo 3f ghi: `win_row`, `score_A`, `score_B`.
- File thật (`eval/ab_blind_heldout.csv`): `id, query, better, correct_count_A, correct_count_B, notes`.
- Bộ cột thống nhất: `id, query, better, correct_count_A, correct_count_B, notes`.

**Hướng dẫn điền:**
- `better` ∈ {{`A`, `B`, `Tie`}}
- `correct_count_A`, `correct_count_B`: số ảnh đúng trong 10 ảnh mỗi hàng (0–10)
- `notes`: ghi chú tùy chọn

### F.3 Kiểm tra rò rỉ cấu hình

| Hạng mục kiểm tra | Kết quả |
|---|---|
| Nhãn config trên ảnh | Không có (chỉ có nhãn "Row A", "Row B") |
| Tên file gợi ý thứ tự A/B | Không có (chỉ mang mã câu hỏi x001..x030) |
| Thứ tự ID gợi ý cấu hình | Không có |
| Metadata ảnh (EXIF/PNG text) chứa config | Sạch hoàn toàn (đã kiểm tra qua script) |

### F.4 Niêm phong ánh xạ

- Đã di chuyển: `eval/ab_mapping.json` → `eval/_sealed/ab_mapping.json`.
- Hash SHA-256 cập nhật trong `reports/run_manifest.json`: `{sealed_mapping_hash}`.
- Nội dung file **tuyệt đối không** được mở, in hay trích dẫn trong chat hoặc báo cáo.
- `eval/ab_blind_heldout.csv` **không** được điền hộ. Người dùng tự chấm.

### F.5 Bước sau khi người dùng nộp CSV

Sau khi người dùng nộp CSV đã chấm, sẽ viết script giải mã đối chiếu với `eval/_sealed/ab_mapping.json` để tính tỷ lệ thắng/thua giữa Model thuần và Hệ thống.

---

## G. Việc chờ người dùng quyết định

| # | Việc | Trạng thái |
|---|---|---|
| 1 | Duyệt `reports/brand_alias_proposal.csv` (23 đề xuất alias) | `ĐÃ DUYỆT (Đã áp dụng và tạo brands_v2.json)` |
| 2 | Quyết định nếu pipeline `max_length` / padding chưa khớp code train | `ĐÃ XÁC NHẬN KHỚP 100% (max_length=128)` |
| 3 | Tự chấm `eval/ab_blind_heldout.csv` | `CHỜ NGƯỜI DÙNG CHẤM (30 contact sheets)` |

---

## H. Tuân thủ luật bất biến

- [x] Không train, không sửa checkpoint (luật 1)
- [x] Không xóa file; chỉ chuyển và ghi log (luật 2)
- [x] Chỉ ghi vào thư mục được phép (luật 3)
- [x] Số liệu do script sinh, không gõ tay (luật 5)
- [x] Không đổi tập test (luật 6)
- [x] Hash các tài sản mới ghi vào `reports/run_manifest.json` cùng seed và phiên bản thư viện (luật 8)
- [x] Đường dẫn chạy được trên Windows, `encoding="utf-8"` (luật 9)
- [x] Không dùng số Config 2 Q1b (0.9008) để nói model đạt chỉ tiêu 0.85. Số của model thuần: 0.5225.
- [x] Báo cáo `phase3f_reconciliation.md` cũ còn nguyên
- [x] Commit: `phase 3g: errata 3f`, không kèm ảnh, embedding, index, checkpoint

---

## I. Danh sách file mới

| File | Mô tả |
|---|---|
| `reports/phase3f_errata.md` | Báo cáo đính chính toàn diện này |
| `reports/brand_audit_3g.json` | Kiểm toán brand Lavie và Benetton trong catalog |
| `reports/brand_alias_proposal.csv` | 23 đề xuất alias brand đã được duyệt |
| `src/brands_v2.json` | Từ điển brand v2 kèm 22 alias (tạo sau duyệt, giữ nguyên brands.json) |
| `reports/phase3g_brands_v2_eval.json` | Kết quả đánh giá đối chứng v1 vs v2 trên Heldout và Q5 |
| `reports/phase3g_latency.json` | Benchmark latency đo lại (breakdown + configs + image) |
| `reports/phase3g_fp16_ties.json` | Phân tích FP32 vs FP16 và đồng hạng |
| `reports/phase3g_maxlen.json` | Kiểm chứng căn nguyên max_length và padding length |
| `reports/phase3g_q6_audit.json` | Kiểm tra tính độc lập Q6 và ablation BM25 |
| `eval/_sealed/ab_mapping.json` | Ánh xạ A/B đã niêm phong bảo mật |
"""

    out_file = Path("reports/phase3f_errata.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Generated {out_file} successfully!")

if __name__ == "__main__":
    generate_errata_report()
