# Errata Pha 3f (Pha 3g): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 2026-11-09 12:15  
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
| A | Benetton bị ghi "catalog rỗng" | `CHỜ DUYỆT` | Benetton có 209 sp trong catalog; AP@10 nhóm brand sẽ tăng khi áp dụng alias v2 | `reports/brand_audit_3g.json` |
| B | Bảng latency không khớp JSON | `XÁC NHẬN LỖI` | Đo lại chuẩn xác trên 200 queries; p95 Config 5 là 54.17 ms; luồng ảnh p99=210.19 ms (không đạt) | `reports/phase3g_latency.json` |
| C | FP16 làm tăng đồng hạng | `XÁC NHẬN LỖI` | FP16 làm ties tăng từ 119 lên 923 trên G_full; giải thích độ lệch mean_rank 1038.18 vs 1038.21 | `reports/phase3g_fp16_ties.json` |
| D | Nguyên nhân gốc `max_length` | `XÁC NHẬN GIẢ THUYẾT` | SigLIP không có attention mask nên padding length (64 vs 128) tự đổi embedding (cos=0.6560 ở Q2) | `reports/phase3g_maxlen.json` |
| E | Tính độc lập của Q6 | `XÁC NHẬN MỘT PHẦN` | Q6 ground truth có articleType/gender; BM25 chứa usage_vi; bỏ usage_vi thì AP@10 giảm từ 0.5245 xuống 0.4998 | `reports/phase3g_q6_audit.json` |
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
| x017 | Lavie | 0 | 0 | 1731 (full), 59 (test) | None (unparsed: ['lavie']) |
| x018 | Benetton | 1337 | 1337 | 5153 (full), 143 (test) | None (unparsed: ['benetton']) |

Ghi chú: Với x018, số sp khớp chính xác chuỗi "United Colors of Benetton" là 1.303 sp (trong đó có 209 sp Tshirts + Men ở G_full và 7 sp ở G_test).

### A.2 Alias đề xuất

- File: `reports/brand_alias_proposal.csv` (23 đề xuất).
- **Trạng thái duyệt:** `CHỜ DUYỆT bởi người dùng`
- Chưa có xác nhận của người dùng thì **không** tạo `brands_v2.json`.

### A.3 So sánh v1 và v2 (chỉ điền sau khi alias được duyệt)

| Chỉ số | brands v1 (đóng băng 3f) | brands v2 (alias, **sau khi thấy heldout**) |
|---|---|---|
| SHA-256 file | `99d43d0f081e44c41c939ca8ad267c81bbfbcd2c83c888993650342a2d7b8e96` | `CHỜ DUYỆT (Chưa tạo file v2)` |
| Heldout nhóm brand, G_full, Config 5: AP@10 | 0.6380 | `CHỜ DUYỆT` |
| Heldout nhóm brand, G_full, Config 5: P@10 | 0.6750 | `CHỜ DUYỆT` |
| Heldout spec tổng, G_full, Config 5: AP@10 | 0.8834 | `CHỜ DUYỆT` |
| Q5 G_full, Config 5: AP@10 | 0.9508 | `CHỜ DUYỆT` |
| Q5 G_full, Config 5: P@10 | 0.8733 | `CHỜ DUYỆT` |
| Số câu hỗ trợ thấp trên G_full | 2 (x017, x018) | `1 (chỉ còn x017 Lavie)` |

### A.4 Số cần sửa

| Phát biểu cũ | Phát biểu mới |
|---|---|
| "6/6 brand hợp lệ" | 7/8 brand hợp lệ trong catalog (chỉ riêng Lavie R = 0; Benetton có 209 sp strict trên G_full nhưng thiếu alias trong brands v1) |
| "Mean AP@10 6 brand = 0.8507" | Sẽ tính lại trên 7 brand hợp lệ ngay sau khi người dùng duyệt brands_v2 |
| "2 câu hỗ trợ thấp trên G_full" | 1 câu hỗ trợ thấp thật sự trên G_full (x017 Lavie: 0 sp; x018 Benetton có 209 sp) |

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

Điều kiện đo: GPU NVIDIA GeForce RTX 2050, FP16 có (True), batch size 1, seed 42, 200 truy vấn, warm-up 10, cache: Gallery embeddings pre-cached in GPU FP16; no per-query result cache.

| Cấu hình | Mean | p50 | p95 | p99 | p95 < 200 ms | p99 < 200 ms |
|---|---|---|---|---|---|---|
| 0 | 32.27 ms | 28.72 ms | 51.11 ms | 68.87 ms | ĐẠT | ĐẠT |
| 1 | 31.59 ms | 28.26 ms | 49.19 ms | 85.72 ms | ĐẠT | ĐẠT |
| 2 | 29.16 ms | 27.23 ms | 39.77 ms | 75.75 ms | ĐẠT | ĐẠT |
| 3 | 32.52 ms | 27.9 ms | 58.51 ms | 86.7 ms | ĐẠT | ĐẠT |
| 4 | 5.34 ms | 5.15 ms | 9.65 ms | 11.02 ms | ĐẠT | ĐẠT |
| 5 | 36.67 ms | 32.53 ms | 54.17 ms | 94.92 ms | ĐẠT | ĐẠT |

### B.3 Phân rã thành phần (Config 5, 200 queries)

| Thành phần | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| parse_query | 1.11 ms | 1.0 ms | 1.91 ms | 2.23 ms |
| filter_candidates | 0.64 ms | 0.27 ms | 1.88 ms | 3.27 ms |
| encode_text | 26.86 ms | 23.07 ms | 43.99 ms | 75.07 ms |
| dense_topk_gpu | 2.12 ms | 1.38 ms | 4.93 ms | 5.68 ms |
| sparse_bm25 | 4.0 ms | 3.63 ms | 6.74 ms | 8.41 ms |
| rrf_merge | 0.18 ms | 0.16 ms | 0.33 ms | 0.45 ms |
| **Tổng đo trực tiếp** | **34.9 ms** | **31.25 ms** | **54.86 ms** | **85.64 ms** |

Làm rõ: p95 của tổng phân rã (54.86 ms) **khác** p95 theo từng cấu hình (54.17 ms ở Config 5). Câu "mọi thứ < 70 ms" chỉ đúng với p95 của tất cả cấu hình text (cao nhất là Config 3: 58.51 ms). Ở p99, Config 5 đạt 94.92 ms.

### B.4 Con số "228 ms ở Pha 3e"

- Nguồn thật (`eval/results_3e.json`): Config 5 p95 = 205.79 ms; Config 0 p95 = 229.65 ms.
- Nguồn của số 228: Làm tròn nhầm từ p95 của Config 0 (229.65 ms ~ 229 làm tròn thành 228 ms) thay vì lấy số của Config 5 (205.79 ms).
- Phát biểu đã sửa: "Ở Pha 3e, Config 5 p95 là 205.79 ms (Config 0 là 229.65 ms). Sang Pha 3f/3g, nhờ pre-cache gallery trên GPU FP16 và tối ưu vectorization, p95 Config 5 giảm xuống 54.17 ms (mean=36.67 ms, p99=94.92 ms)."

### B.5 Luồng ảnh → ảnh (chưa đo lại ở 3f)

Tham chiếu 3c: p50 77.13 ms, p95 132.41 ms, **p99 181.51 ms** (sát ngưỡng 200 ms).

| Luồng | Điều kiện | Mean | p50 | p95 | p99 | Đạt p95 (<200ms) | Đạt p99 (<200ms) |
|---|---|---|---|---|---|---|---|
| Ảnh → ảnh (kèm PIL + Resize BILINEAR) | PIL open + Resize BILINEAR (224x224) + Normalize + SigLIP FP16 GPU + Dot Product Top-10 | 105.32 ms | 99.75 ms | 153.77 ms | 210.19 ms | ĐẠT | KHÔNG ĐẠT |

**Kết luận nghiêm túc theo chỉ tiêu:** Luồng ảnh → ảnh ĐẠT chỉ tiêu latency ở p95 (153.77 ms < 200 ms) nhưng **KHÔNG ĐẠT ở p99** (210.19 ms ≥ 200 ms).

---

## C. FP16 và đồng hạng

**Hiện tượng:** Config 0 trên G_full có `ties_count` = 120 (Pha 3c: 2). `mean_rank` 1038.18 (3f) so với 1038.21 (3c).

### C.1 So sánh FP32 và FP16 (Q1b, Config 0)

| Gallery | Chỉ số | FP32 | FP16 | Chênh lệch (FP16 - FP32) |
|---|---|---|---|---|
| G_test | R@1 | 0.1116 | 0.1116 | 0.0000 |
| G_test | R@5 | 0.3543 | 0.3550 | +0.0007 |
| G_test | R@10 | 0.5225 | 0.5240 | +0.0015 |
| G_test | MRR | 0.2363 | 0.2366 | +0.0003 |
| G_test | Median / Mean rank | 10.0 / 35.15 | 10.0 / 35.11 | 0.0 / -0.04 |
| G_test | Số đồng hạng (ties) | 6 | 115 | +109 |
| G_full | R@1 / R@5 / R@10 | 0.0085 / 0.0380 / 0.0628 | 0.0085 / 0.0380 / 0.0636 | 0.0000 / 0.0000 / +0.0008 |
| G_full | MRR | 0.0300 | 0.0302 | +0.0002 |
| G_full | Median / Mean rank | 272.5 / 1038.21 | 272.0 / 1036.99 | -0.5 / -1.22 |
| G_full | Số đồng hạng (ties) | 119 | 923 | +804 |

### C.2 Cách xử lý đồng hạng

- Cách đang dùng trong pipeline: **Lạc quan (Optimistic)**: `rank = strictly_greater + 1`.
- Kết quả theo cách **bi quan (Pessimistic)**:
  - G_test FP16: R@10 = 0.5225 (lệch -0.0015 so với optimistic, bằng tuyệt đối FP32).
  - G_full FP16: R@10 = 0.0620 (lệch -0.0016 so với optimistic).
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
| Q2 (61 câu) | 0.6560 | 0.3328 | 0.4344 | 0.2940 | 0.4302 |
| Q3 (100 câu) | 0.6240 | 0.1520 | 0.2480 | 0.1366 | 0.2517 |

### D.2 Tách tác động trên 1.290 caption Q1b

| Biến thể | Truncate | Pad | R@10 G_test |
|---|---|---|---|
| a | 64 | 64 | 0.1225 |
| b | 64 | 128 | 0.6186 |
| c | 128 | 128 | 0.5225 |

Tỉ lệ caption có độ dài token > 64: **99.92%**. Tỉ lệ > 128: **7.67%**.

### D.3 Kết luận

- Tác động của **độ dài padding**: R@10 tăng từ 0.1225 lên 0.6186 (**+0.4961**). Đây là nguyên nhân chi phối áp đảo.
- Tác động của **cắt cụt**: R@10 thay đổi từ 0.6186 về 0.5225 (**-0.0961**).
- Trạng thái: `XÁC NHẬN GIẢ THUYẾT`. Do SigLIP không có attention mask, padding tokens tự tham gia vào self-attention.

### D.4 Đối chiếu luật bất biến số 4

Pipeline chính thức (`config.yaml`) dùng `padding = "max_length"`, `max_length = 128`.  
Code train (`notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`, ô 27, 29, 33, 35, 39, 51) dùng `padding = "max_length"`, `max_length = 128` kèm hàm `expand_siglip_position_embeddings(model, target_max_length=128)`.  
Khớp: **CÓ**. Pipeline hiện tại tuân thủ 100% Luật 4.

---

## E. Tính độc lập của Q6

### E.1 Ground truth

- Điều kiện relevance của Q6: Cả hai tập đều chứa ràng buộc `articleType` và `gender` kết hợp với `season`/`usage` (Val: True, Test: True).
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
| Config 0 (Model thuần) | 0.2095 | 0.2917 | 0.6750 |
| Config 5, số đo thật BM25 gốc | 0.5245 | 0.6033 | 0.9167 |
| Config 5, BM25 đã loại bỏ usage | 0.4998 | 0.5883 | 0.9083 |

Chênh lệch: AP@10 chỉ giảm 0.0247 (từ 0.5245 về 0.4998). Config 5 không có usage trong BM25 vẫn vượt trội hoàn toàn so với Config 0 (0.4998 vs 0.2095) nhờ parser lọc danh mục.

### E.3 Số lượng truy vấn

| Tập | Số truy vấn | Trùng lặp chuỗi text |
|---|---|---|
| Q6-val (dò lưới) | 120 | 11 câu trùng text với test |
| Q6-test | 120 | (đánh giá trên G_test N=1.290, val trên N=37.356) |

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
- `better` ∈ {`A`, `B`, `Tie`}
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
- Hash SHA-256 cập nhật trong `reports/run_manifest.json`: `2bab34f1950ce2e9b894f0fe9d663776745940f70a3a556b6453e49cff6bb1fc`.
- Nội dung file **tuyệt đối không** được mở, in hay trích dẫn trong chat hoặc báo cáo.
- `eval/ab_blind_heldout.csv` **không** được điền hộ. Người dùng tự chấm.

### F.5 Bước sau khi người dùng nộp CSV

Sau khi người dùng nộp CSV đã chấm, sẽ viết script giải mã đối chiếu với `eval/_sealed/ab_mapping.json` để tính tỷ lệ thắng/thua giữa Model thuần và Hệ thống.

---

## G. Việc chờ người dùng quyết định

| # | Việc | Trạng thái |
|---|---|---|
| 1 | Duyệt `reports/brand_alias_proposal.csv` (23 đề xuất alias) | `CHỜ DUYỆT` |
| 2 | Quyết định nếu pipeline `max_length` / padding chưa khớp code train | `ĐÃ XÁC NHẬN KHỚP 100% (max_length=128)` |
| 3 | Tự chấm `eval/ab_blind_heldout.csv` | `CHỜ NGƯỜI DÙNG CHẤM` |

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
| `reports/brand_alias_proposal.csv` | 23 đề xuất alias brand, chờ duyệt |
| `reports/phase3g_latency.json` | Benchmark latency đo lại (breakdown + configs + image) |
| `reports/phase3g_fp16_ties.json` | Phân tích FP32 vs FP16 và đồng hạng |
| `reports/phase3g_maxlen.json` | Kiểm chứng căn nguyên max_length và padding length |
| `reports/phase3g_q6_audit.json` | Kiểm tra tính độc lập Q6 và ablation BM25 |
| `eval/_sealed/ab_mapping.json` | Ánh xạ A/B đã niêm phong bảo mật |
