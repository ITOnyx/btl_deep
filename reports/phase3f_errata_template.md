# Errata Pha 3f (Pha 3g): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)
**Ngày thực hiện:** ⟨YYYY-MM-DD HH:MM⟩
**Chế độ:** Suy luận thuần. Không train, không adapter, không sửa checkpoint.
**Báo cáo được đính chính:** `reports/phase3f_reconciliation.md` (giữ nguyên, không sửa)

> **Quy ước điền mẫu**
> - Cột "Số cũ" đã điền sẵn từ báo cáo 3f. Cột "Số mới" và mọi ⟨…⟩ do agent điền **từ file JSON do script sinh**, không gõ tay.
> - Mỗi dòng ghi rõ file nguồn. Không có nguồn thì không ghi số.
> - Trạng thái mỗi mục: `XÁC NHẬN LỖI` / `BÁC BỎ GIẢ THUYẾT` / `CHỜ DUYỆT` / `CHƯA XÁC ĐỊNH`.
> - Hai cột **"Model thuần"** và **"Hệ thống"** luôn tách riêng.

---

## 0. Bảng tổng hợp

| Mục | Vấn đề | Trạng thái | Ảnh hưởng đến kết luận 3f | File nguồn |
|---|---|---|---|---|
| A | Benetton bị ghi "catalog rỗng" | ⟨…⟩ | ⟨…⟩ | `reports/brand_audit_3g.json` |
| B | Bảng latency không khớp JSON | ⟨…⟩ | ⟨…⟩ | `reports/phase3g_latency.json` |
| C | FP16 làm tăng đồng hạng | ⟨…⟩ | ⟨…⟩ | `reports/phase3g_fp16_ties.json` |
| D | Nguyên nhân gốc `max_length` | ⟨…⟩ | ⟨…⟩ | `reports/phase3g_maxlen.json` |
| E | Tính độc lập của Q6 | ⟨…⟩ | ⟨…⟩ | `reports/phase3g_q6_audit.json` |
| F | Giao thức chấm mù | ⟨…⟩ | ⟨…⟩ | `reports/contact_sheets/` |

---

## A. Brand: Benetton không phải "catalog rỗng"

**Phát biểu cũ (3f, mục F6):** x017 (Lavie) và x018 (Benetton) có R = 0, AP@10 "toán học bằng 0"; "6/6 brand hợp lệ".

**Bằng chứng mâu thuẫn:**
- `reports/heldout_inspection_b0.json`: x018 `strict_full` = 209.
- `reports/phase3e_part_a_review.md`: "United Colors of Benetton" có 1.462 sản phẩm.
- `eval/results_3e.json`: tự chẩn đoán "Brand mismatch (United Colors of Benetton vs Benetton)".

**Nguyên nhân thật:** `src/brands.json` thiếu alias "Benetton" → "United Colors of Benetton". Không phải catalog rỗng.

### A.1 Kiểm toán catalog (từ `reports/brand_audit_3g.json`)

| Query | Brand truy vấn | Số sp khớp chính xác | Số sp khớp chuỗi con | R (articleType + gender) | Parser nhận diện |
|---|---|---|---|---|---|
| x017 | Lavie | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| x018 | Benetton | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

### A.2 Alias đề xuất

- File: `reports/brand_alias_proposal.csv` (⟨N⟩ dòng).
- **Trạng thái duyệt:** ⟨CHỜ DUYỆT / ĐÃ DUYỆT bởi người dùng lúc …⟩
- Chưa có xác nhận của người dùng thì **không** tạo `brands_v2.json`.

### A.3 So sánh v1 và v2 (chỉ điền sau khi alias được duyệt)

| Chỉ số | brands v1 (đóng băng 3f) | brands v2 (alias, **sau khi thấy heldout**) |
|---|---|---|
| SHA-256 file | `99d43d0f081e44c41c939ca8ad267c81bbfbcd2c83c888993650342a2d7b8e96` | ⟨…⟩ |
| Heldout nhóm brand, G_full, Config 5: AP@10 | 0.6380 | ⟨…⟩ |
| Heldout nhóm brand, G_full, Config 5: P@10 | 0.6750 | ⟨…⟩ |
| Heldout spec tổng, G_full, Config 5: AP@10 | 0.8834 | ⟨…⟩ |
| Q5 G_full, Config 5: AP@10 | 0.9508 | ⟨…⟩ |
| Q5 G_full, Config 5: P@10 | 0.8733 | ⟨…⟩ |
| Số câu hỗ trợ thấp trên G_full | ⟨…⟩ | ⟨…⟩ |

### A.4 Số cần sửa

| Phát biểu cũ | Phát biểu mới |
|---|---|
| "6/6 brand hợp lệ" | ⟨7/8 (chỉ Lavie R = 0), kèm số v2⟩ |
| "Mean AP@10 6 brand = 0.8507" | ⟨tính lại trên tập brand đúng⟩ |
| "2 câu hỗ trợ thấp trên G_full" | ⟨…⟩ |

**Ghi chú trung thực:** v2 là sửa **hậu nghiệm** sau khi đã thấy kết quả heldout. Heldout không còn là tập ẩn đối với thay đổi này. Số v1 được giữ để đối chiếu.

**Zero-result guard cho Lavie:** chỉ đề xuất, chưa triển khai ở pha này.

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

Điều kiện đo: GPU ⟨…⟩, FP16 ⟨có/không⟩, batch size ⟨1⟩, seed ⟨42⟩, ⟨N⟩ truy vấn, warm-up ⟨10⟩, cache ⟨…⟩.

| Cấu hình | Mean | p50 | p95 | p99 | p95 < 200 ms | p99 < 200 ms |
|---|---|---|---|---|---|---|
| 0 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 1 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 2 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 3 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 4 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 5 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

### B.3 Phân rã thành phần

| Thành phần | Mean | p95 |
|---|---|---|
| parse_query | ⟨…⟩ | ⟨…⟩ |
| filter_candidates | ⟨…⟩ | ⟨…⟩ |
| encode_text | ⟨…⟩ | ⟨…⟩ |
| dense_topk_gpu | ⟨…⟩ | ⟨…⟩ |
| sparse_bm25 | ⟨…⟩ | ⟨…⟩ |
| rrf_merge | ⟨…⟩ | ⟨…⟩ |
| **Tổng** | ⟨…⟩ | ⟨…⟩ |

Làm rõ: p95 của tổng phân rã (bản cũ: 78.53 ms) **khác** p95 theo từng cấu hình (69.75 ms ở Config 5). Câu "mọi thứ < 70 ms" chỉ đúng nếu ⟨…⟩.

### B.4 Con số "228 ms ở Pha 3e"

- Nguồn thật (`eval/results_3e.json`): Config 5 p95 = 205.79 ms; Config 0 p95 = 229.65 ms.
- Nguồn của số 228: ⟨…⟩
- Phát biểu đã sửa: ⟨…⟩

### B.5 Luồng ảnh → ảnh (chưa đo lại ở 3f)

Tham chiếu 3c: p50 77.13 ms, p95 132.41 ms, **p99 181.51 ms** (sát ngưỡng 200 ms).

| Luồng | Điều kiện | p50 | p95 | p99 | Đạt p95 | Đạt p99 |
|---|---|---|---|---|---|---|
| Ảnh → ảnh (kèm PIL + Resize BILINEAR) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

Nếu p99 ≥ 200 ms: ghi thẳng **không đạt ở p99**.

---

## C. FP16 và đồng hạng

**Hiện tượng:** Config 0 trên G_full có `ties_count` = 120 (Pha 3c: 2). `mean_rank` 1038.18 (3f) so với 1038.21 (3c).

### C.1 So sánh FP32 và FP16 (Q1b, Config 0)

| Gallery | Chỉ số | FP32 | FP16 | Chênh lệch |
|---|---|---|---|---|
| G_test | R@1 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_test | R@5 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_test | R@10 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_test | MRR | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_test | Median / Mean rank | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_test | Số đồng hạng | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_full | R@1 / R@5 / R@10 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_full | MRR | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_full | Median / Mean rank | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| G_full | Số đồng hạng | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

### C.2 Cách xử lý đồng hạng

- Cách đang dùng trong pipeline: ⟨lạc quan / bi quan / trung bình⟩
- Kết quả theo cách **bi quan**: R@10 = ⟨…⟩ (G_test), ⟨…⟩ (G_full). Biên độ ảnh hưởng: ⟨…⟩

### C.3 Phát biểu đã sửa

| Cũ | Mới |
|---|---|
| "Khớp chính xác 100% với Pha 3c" | ⟨Chỉ số nào khớp tuyệt đối, chỉ số nào lệch bao nhiêu, vì sao⟩ |

---

## D. Nguyên nhân gốc F1: `max_length`

**Phát biểu cũ:** R@10 Q1b Config 0 sụt do `max_length=64` cắt cụt phần thuộc tính ở đuôi.

**Giả thuyết cần kiểm:** SigLIP không có `attention_mask`, nên độ dài padding (64 so với 128) tự đổi embedding, kể cả câu ngắn không bị cắt.

**Dấu hiệu:** Q2 Config 0 tăng từ 3e sang 3f dù câu Q2 ngắn.

| Chỉ số Q2, Config 0 | 3e (max_length = 64) | 3f (max_length = 128) |
|---|---|---|
| G_test P@10 | 0.3328 | 0.4344 |
| G_test AP@10 | 0.2940 | 0.4300 |
| G_full P@10 | 0.3721 | 0.6213 |
| G_full AP@10 | 0.2821 | 0.5373 |

### D.1 Phép đo (từ `reports/phase3g_maxlen.json`)

| Tập | Cosine TB giữa embedding pad=64 và pad=128 | P@10 Config 0 (pad 64) | P@10 Config 0 (pad 128) |
|---|---|---|---|
| Q2 (61 câu) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| Q3 (100 câu) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

### D.2 Tách tác động trên 1.290 caption Q1b

| Biến thể | Truncate | Pad | R@10 G_test |
|---|---|---|---|
| a | 64 | 64 | ⟨…⟩ |
| b | 64 | 128 | ⟨…⟩ |
| c | 128 | 128 | ⟨…⟩ |

Tỉ lệ caption có độ dài token > 64: ⟨…⟩%. Tỉ lệ > 128: ⟨…⟩%.

### D.3 Kết luận

- Tác động của **cắt cụt**: ⟨…⟩
- Tác động của **độ dài padding**: ⟨…⟩
- Trạng thái: ⟨XÁC NHẬN / BÁC BỎ / MỘT PHẦN⟩

### D.4 Đối chiếu luật bất biến số 4

Pipeline chính thức (`config.yaml`) dùng `padding` = ⟨…⟩, `max_length` = ⟨…⟩. Code train (notebook, ô ⟨…⟩) dùng `padding` = ⟨…⟩, `max_length` = ⟨…⟩. Khớp: ⟨CÓ / KHÔNG / CHƯA XÁC ĐỊNH⟩. Nếu chưa xác định thì dừng và hỏi.

---

## E. Tính độc lập của Q6

### E.1 Ground truth

- Điều kiện relevance của Q6: ⟨chỉ usage/season, hay có articleType/gender⟩ (nguồn: `eval/q6_queries.json`).
- Nếu có articleType/gender: lọc theo parser vẫn có lợi, nên Q6 **không** độc lập hoàn toàn. Ghi chú: ⟨…⟩

### E.2 Trường được đưa vào BM25

| Trường | Có trong BM25 index? |
|---|---|
| title | ⟨…⟩ |
| attrs (articleType, gender, colour) | ⟨…⟩ |
| usage / season | ⟨…⟩ |
| caption huấn luyện mẫu ("phù hợp dạo phố thường ngày (mùa hè…)") | ⟨…⟩ |

Nếu usage/season nằm trong index, BM25 đang khớp thẳng nhãn đánh giá. Chạy ablation:

| Cấu hình, Q6-test | AP@10 | P@10 | Hit@10 |
|---|---|---|---|
| Config 0 (Model thuần) | 0.2091 | 0.2908 | 0.6750 |
| Config 5, số cũ (3f) | 0.5172 | 0.5967 | 0.9167 |
| Config 5, BM25 đã loại usage/season/caption mẫu | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

### E.3 Số lượng truy vấn

| Tập | Số truy vấn | Trùng nhau? |
|---|---|---|
| Q6-val (dò lưới) | ⟨…⟩ | ⟨…⟩ |
| Q6-test | ⟨…⟩ | |

Báo cáo F3 ghi 120 truy vấn, bản tóm tắt gửi người dùng ghi 240. Số đúng: ⟨…⟩.

### E.4 Lựa chọn α* = 1.0

- Chênh lệch val AP@10 giữa α = 0.85 (0.3326) và α = 1.0 (0.3274): 0.0052.
- Lý do chọn đúng: quy tắc ≤ 0.01 và tiết kiệm RAM, **không** phải vì text fusion vô dụng.
- Gallery val (37.356 ảnh `unknown`) rất có thể gồm ảnh train của checkpoint, nên số val mang tính lạc quan.
- Chênh lệch 0.0052 nằm trong nhiễu của ⟨N⟩ truy vấn (CI ⟨…⟩).

### E.5 Heldout không còn ẩn hoàn toàn

Heldout đã được dùng ở 3e và 3f; mọi thay đổi sau lần xem kết quả đều là hậu nghiệm.

---

## F. Giao thức chấm mù

### F.1 Tên file contact sheet

- Báo cáo 3f ghi: `blind_ab_h001.png` … `blind_ab_h030.png`
- Tên file thật: ⟨liệt kê từ `reports/contact_sheets/`⟩
- Sửa báo cáo thành: ⟨…⟩

### F.2 Cột file chấm

- Báo cáo 3f ghi: `win_row`, `score_A`, `score_B`.
- File thật (`eval/ab_blind_heldout.csv`): `id, query, better, correct_count_A, correct_count_B, notes`.
- Bộ cột thống nhất: ⟨…⟩

**Hướng dẫn điền:**
- `better` ∈ {`A`, `B`, `Tie`}
- `correct_count_A`, `correct_count_B`: số ảnh đúng trong 10 ảnh mỗi hàng (0–10)
- `notes`: tùy chọn

### F.3 Kiểm tra rò rỉ cấu hình

| Hạng mục kiểm tra | Kết quả |
|---|---|
| Nhãn config trên ảnh | ⟨…⟩ |
| Tên file gợi ý thứ tự A/B | ⟨…⟩ |
| Thứ tự ID gợi ý cấu hình | ⟨…⟩ |
| Metadata ảnh (EXIF/PNG text) chứa config | ⟨…⟩ |

### F.4 Niêm phong ánh xạ

- `eval/ab_mapping.json` → `eval/_sealed/ab_mapping.json` (di chuyển, không xóa).
- Hash cập nhật trong `reports/run_manifest.json`: ⟨…⟩
- Nội dung file **không** được mở, in hay trích dẫn trong chat hoặc báo cáo.
- `eval/ab_blind_heldout.csv` **không** được điền hộ. Người dùng tự chấm.

### F.5 Bước sau khi người dùng nộp CSV

Viết script giải mã, tính thắng/thua A/B giữa Model thuần và Hệ thống. Chưa làm ở pha này.

---

## G. Việc chờ người dùng quyết định

| # | Việc | Trạng thái |
|---|---|---|
| 1 | Duyệt `reports/brand_alias_proposal.csv` | ⟨…⟩ |
| 2 | Quyết định nếu pipeline `max_length` / padding chưa khớp code train | ⟨…⟩ |
| 3 | Tự chấm `eval/ab_blind_heldout.csv` | ⟨…⟩ |

---

## H. Tuân thủ luật bất biến

- [ ] Không train, không sửa checkpoint (luật 1)
- [ ] Không xóa file; chỉ chuyển và ghi log (luật 2)
- [ ] Chỉ ghi vào thư mục được phép (luật 3)
- [ ] Số liệu do script sinh, không gõ tay (luật 5)
- [ ] Không đổi tập test (luật 6)
- [ ] Hash các tài sản mới ghi vào `reports/run_manifest.json` cùng seed và phiên bản thư viện (luật 8)
- [ ] Đường dẫn chạy được trên Windows, `encoding="utf-8"` (luật 9)
- [ ] Không dùng số Config 2 Q1b (0.9008) để nói model đạt chỉ tiêu 0.85. Số của model thuần: 0.5225.
- [ ] Báo cáo `phase3f_reconciliation.md` cũ còn nguyên
- [ ] Commit: `phase 3g: errata 3f`, không kèm ảnh, embedding, index, checkpoint

---

## I. Danh sách file mới

| File | Mô tả |
|---|---|
| `reports/phase3f_errata.md` | Báo cáo này |
| `reports/brand_audit_3g.json` | Kiểm toán brand trong catalog |
| `reports/brand_alias_proposal.csv` | Đề xuất alias, chờ duyệt |
| `src/brands_v2.json` | Chỉ tạo sau khi duyệt |
| `reports/phase3g_latency.json` | Latency đo lại |
| `reports/phase3g_fp16_ties.json` | So sánh FP32/FP16 và đồng hạng |
| `reports/phase3g_maxlen.json` | Kiểm chứng `max_length` / padding |
| `reports/phase3g_q6_audit.json` | Kiểm tra độc lập Q6 và ablation BM25 |
| `eval/_sealed/ab_mapping.json` | Ánh xạ A/B đã niêm phong |
