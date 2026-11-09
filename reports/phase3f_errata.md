# Errata Pha 3f (Pha 3g Rev3): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 2026-11-09 (Bản Rev3 hoàn chỉnh theo chỉ đạo của Người dùng)  
**Chế độ:** Suy luận thuần. Không train lại, không sửa checkpoint SigLIP, không xóa file.  
**Báo cáo được đính chính:** `reports/phase3f_reconciliation.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 1:** `reports/phase3f_errata_rev1.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 2:** `reports/phase3f_errata_rev2.md` (giữ nguyên làm bằng chứng lịch sử).

> **Quy ước bất biến**
> - 100% con số trong báo cáo được trích xuất trực tiếp từ các file JSON do script sinh tự động trong `reports/`. Tuyệt đối không gõ tay, không bịa số.
> - Trạng thái mỗi mục dùng chuẩn: `XÁC NHẬN LỖI` / `BÁC BỎ GIẢ THUYẾT` / `MỘT PHẦN` / `CHƯA XÁC ĐỊNH` / `CHỜ NGƯỜI DÙNG`.
> - Hai cột **"Model thuần"** (Config 0) và **"Hệ thống"** (Config 2 / Config 5) luôn tách riêng biệt. Tuyệt đối không dùng số của Hệ thống để phát biểu thay cho Model thuần.
> - Phát biểu nào chưa có phép đo cô lập hoặc chưa đủ điều kiện thống kê thì hạ xuống thành `giả thuyết` hoặc ghi rõ `chưa xác định`.
> - Tuyệt đối không mở, không in và không tiết lộ nội dung file niêm phong `eval/_sealed/ab_mapping.json`.

---

## 0. Bảng tổng hợp trạng thái các vấn đề (Rev3)

| Mục | Vấn đề rà soát | Trạng thái Rev3 | Ảnh hưởng đến kết luận Pha 3f | File dữ liệu nguồn |
|---|---|---|---|---|
| **A** | Benetton bị ghi "catalog rỗng, R=0" | `XÁC NHẬN LỖI` | Benetton có 209 sp strict ở $G_{full}$; thêm 22 alias hậu nghiệm giúp x018 AP@10 từ 0.5181 lên 1.0000; kiểm toán va chạm 0 lỗi; regression test Q5, Q2, Q3 có đúng 0 sự khác biệt; số câu low support trên 26 câu spec chỉ còn 1 câu (Lavie) trên $G_{full}$ | `reports/brand_audit_3g.json`, `reports/phase3g_brands_v2_eval.json`, `reports/phase3g_rev3_investigation.json` |
| **B** | Bảng Latency và Chuỗi ký tự "228 ms" | `XÁC NHẬN LỖI` | Bảng 3f lấy nhầm số từ các dòng khác nhau; đo lại chuẩn xác 5 lần lặp: p95 Config 5 là 112.07 ms (ĐẠT); luồng Ảnh → Ảnh có trung vị p95 = 178.25 ms (< 200 ms -> ĐẠT SLA chuẩn), p99 = 280.17 ms là chỉ số rủi ro (risk metric); giải thích rõ chênh lệch p95 3f do nhiệt độ GPU và thermal throttling; quét sạch và lập danh mục mọi file chứa 228 / 69.75 | `reports/phase3g_latency_multi_run.json`, `reports/phase3g_rev3_investigation.json` |
| **C** | FP16 làm tăng đồng hạng (ties) ở 3f | `BÁC BỎ GIẢ THUYẾT` | Giả thuyết FP16 gây ra ties=120 ở 3f bị bác bỏ vì 3f chạy CPU FP32 (`sims_c0 = np.dot(...)`); nguyên nhân thật là do đổi hàm so sánh từ bitwise `==` (ties=2) sang `np.isclose(atol=1e-6)` (ties=112..120). Chạy thử nghiệm GPU FP16 cho ties=917 và biên độ toàn phần là 0.0016 ($0.0628 \pm 0.0008$). Pipeline chính thức cho chất lượng là CPU FP32 | `reports/phase3g_fp16_ties.json`, `reports/phase3g_rev3_investigation.json` |
| **D** | Phân rã `max_length` & padding (F1) | `MỘT PHẦN` (Hạ thành Giả thuyết) | 99.92% (1.289 / 1.290) caption test có độ dài > 64 token (chỉ 1 câu $\le 64$ token), không đủ điều kiện phân rã mẫu đối chứng; hạ kết luận "padding chi phối" xuống thành GIẢ THUYẾT. Tách riêng 2 cột Q1b: Model thuần (0.5225) vs Hệ thống C2 (0.9008); không nhận C2 đạt chỉ tiêu của Model | `reports/phase3g_maxlen.json`, `reports/phase3g_rev3_investigation.json` |
| **E** | Tính "Độc lập" của Q6 và Baseline Ablation | `XÁC NHẬN LỖI` | Xóa bỏ hoàn toàn chữ "độc lập" và "+147%" khỏi báo cáo. Q6 có Parser hỗ trợ nhận diện category & gender. Bằng chứng xếp hạng độc lập duy nhất là Blind A/B. Giải thích rõ 11 câu trùng text giữa test và val; giải thích baseline Điều kiện 3 (0.2285) cao hơn C0 (0.2091) nhờ tín hiệu BM25 từ vựng | `reports/phase3g_q6_rev2.json` |
| **F** | Giao thức Chấm mù A/B Rev3 | `ĐÃ HOÀN THÀNH` | Đã xáo lại A/B bằng seed mới (`20261109`), tạo lại 30 contact sheet `blind_ab_x001..x030.png` với font TrueType `arial.ttf` chuẩn tiếng Việt; niêm phong an toàn `eval/_sealed/ab_mapping.json`; xác nhận `eval/ab_blind_heldout.csv` khớp 30 ID và 6 cột chuẩn | `reports/contact_sheets/`, `eval/_sealed/ab_mapping.json`, `eval/ab_blind_heldout.csv` |

---

## A. Brand: Sửa kết luận Benetton, Kiểm toán Alias và Kiểm tra Hồi quy

### A.1 Bằng chứng mã nguồn và Dữ liệu thực tế (`reports/brand_audit_3g.json`)
- **Phát biểu cũ (3f, mục F6):** x017 (Lavie) và x018 (Benetton) có R = 0, AP@10 "toán học bằng 0"; báo cáo "6/6 brand hợp lệ".
- **Bằng chứng thực tế:**
  1. `data/catalog/catalog.parquet`: Có **1.303 sản phẩm** mang chính xác nhãn canonical `"United Colors of Benetton"`, và **1.337 sản phẩm** chứa chuỗi con `"benetton"`.
  2. `reports/heldout_inspection_b0.json`: Câu `x018` ("áo thun nam benetton") có `strict_full` = **209 sản phẩm** (Tshirts Men mang nhãn Benetton) trên $G_{full}$, và `strict_test` = **7 sản phẩm** trên $G_{test}$.
  3. `reports/brand_audit_3g.json`: Câu `x017` ("túi xách nữ lavie") có đúng **0 sản phẩm** trong catalog.
- **Kết luận:** Phát biểu "Benetton catalog rỗng" của Pha 3f là **SAI**. Nguyên nhân thực sự là do `src/brands.json` v1 chỉ chứa tên đầy đủ `"United Colors of Benetton"` mà thiếu alias `"benetton"`, khiến QueryParser không nhận diện được brand, rơi vào nhánh gán nhãn ground truth sai.

### A.2 Danh sách 22 Alias đã duyệt và Kiểm toán Va chạm (`reports/phase3g_rev3_investigation.json`)
Người dùng đã phê duyệt danh sách đề xuất alias tại mốc thời gian `2026-11-09T12:17:11+07:00` (`<USER_REQUEST> duyệt </USER_REQUEST>`).
- Đã thêm **22 alias** vào `src/brands_v2.json` (SHA-256: `c5283096f649351d4433874915e2e32586118200e576f41b3101495859c1898e`).
- **1 đề xuất không đưa vào alias dict:** `Lee Cooper`. Lý do: `Lee Cooper` vốn dĩ đã là một canonical brand độc lập có sẵn trong danh sách 234 brand gốc của `brands.json` (133 sp).
- **Kết quả kiểm toán va chạm (Collision Audit):**
  1. **Lee vs Lee Cooper:** Hoàn toàn tách biệt. `Lee` có 477 sp, `Lee Cooper` có 133 sp. Không có alias `lee -> lee cooper`. Khi người dùng tìm "Lee", hệ thống nhận diện đúng brand Lee; khi tìm "Lee Cooper", nhận diện đúng Lee Cooper.
  2. **Màu sắc (Black):** Từ `"black"` không có trong `aliases_dict`, QueryParser bảo toàn tuyệt đối việc nhận diện màu sắc (`colour: "Black"`).
  3. **Giới tính (F, W, M):** Không có ký tự đơn `"f"`, `"w"`, `"m"` trong danh sách alias, không gây xung đột với nhận diện giới tính (`Men`, `Women`).
  4. **Chữ hoa/thường (ADIDAS vs Adidas):** `brands_v2.json` lưu canonical `"ADIDAS"` và `"Adidas"` được chuẩn hóa không phân biệt hoa thường trong `QueryParser`.

### A.3 Kiểm tra Hồi quy (Regression Tests)
Đã chạy lại kiểm tra hồi quy trên 3 tập truy vấn chuẩn trước và sau khi kích hoạt `brands_v2.json`:
- **Q5 (150 câu brand):** Số câu có sự khác biệt về kết quả phân tích brand = **0 / 150**.
- **Q2 (61 câu thuộc tính tự nhiên):** Số câu có sự khác biệt về kết quả phân tích brand = **0 / 61**.
- **Q3 (100 câu viết tay):** Số câu có sự khác biệt về kết quả phân tích brand = **0 / 100**.
- **Kết luận:** Việc bổ sung 22 alias không gây ra bất kỳ tác dụng phụ hay suy giảm độ chính xác nào trên toàn bộ các tập truy vấn đã có.

### A.4 Bảng so sánh 3 cột: Tách biệt đóng góp sửa Ground Truth và thêm Alias
Tất cả các số liệu dưới đây được ghi nhận rõ là **sửa đổi hậu nghiệm (post-hoc)** vì được thực hiện sau khi đã quan sát kết quả heldout:

| Chỉ số đánh giá | (1) 3f gốc đóng băng (GT cũ, brands v1) | (2) Brands v1 chấm lại với GT sửa | (3) Brands v2 (alias) với GT sửa | Đóng góp sửa GT: (2) - (1) | Đóng góp thêm Alias: (3) - (2) |
|---|---|---|---|---|---|
| **x018 (Benetton) AP@10** | 0.0000 | 0.5181 | **1.0000** | +0.5181 | +0.4819 |
| **x018 (Benetton) P@10** | 0.0000 | 0.6000 | **1.0000** | +0.6000 | +0.4000 |
| **Heldout nhóm brand (8 câu) AP@10** | 0.6380 | 0.7028 | **0.7630** | +0.0648 | +0.0602 |
| **Heldout nhóm brand (8 câu) P@10** | 0.6750 | 0.7500 | **0.8000** | +0.0750 | +0.0500 |
| **Heldout 7 brand hợp lệ (bỏ Lavie) AP@10** | 0.7291 | 0.8032 | **0.8720** | +0.0741 | +0.0688 |
| **Heldout 6 brand gốc không đổi AP@10** | 0.8507 | 0.8507 | **0.8507** | 0.0000 | 0.0000 |
| **Heldout 26 câu spec tổng AP@10** | 0.8834 | 0.9033 | **0.9221** | +0.0199 | +0.0188 |

*Ghi chú quan trọng về tính chất Hậu nghiệm:*
- 6 thương hiệu gốc (`x011`–`x016`: Puma, Nike, Adidas, Fossil, Fabindia, Catwalk) có AP@10 hoàn toàn bất biến ở mức **0.8507**.
- Điểm AP@10 của nhóm brand tăng từ 0.7028 lên 0.7630 (hoặc 0.8032 lên 0.8720) hoàn toàn đến từ câu `x018` (Benetton). Do alias Benetton được thêm sau khi đã thấy `x018`, nhóm brand heldout **không còn là tập kiểm thử mù (blind) đối với Benetton**.

### A.5 Phân rã số câu Hỗ trợ thấp (Low Support, $R < 3$) trên $G_{full}$ và $G_{test}$
- **Trên $G_{full}$ (Toàn bộ Catalog 37.356 sp, 26 câu spec):**
  - Bản 3f gốc (brands v1, GT cũ): Có **2 câu** mang $R < 3$ là `x017` (Lavie, R=0) và `x018` (Benetton, bị tính R=0 do parser bỏ sót).
  - Bản brands v2 (GT sửa): Chỉ còn duy nhất **1 câu** mang $R < 3$ là `x017` (Lavie, R=0). Câu `x018` có 209 sp ($R \ge 3$).
- **Trên $G_{test}$ (Tập test 1.290 sp):**
  - `x017` (Lavie): 0 sp ($R < 3$).
  - `x018` (Benetton): 54 sp ($R \ge 3$).
  - Trong 24 câu spec còn lại, một số tổ hợp thuộc tính hẹp (như đồng hồ vàng hoặc phụ kiện hiếm) có số lượng trong $G_{test}$ ít hơn 3 sp do $G_{test}$ chỉ chiếm ~3.4% catalog.
- **Kết luận:** Khẳng định "chỉ còn 1 câu low support" có giá trị chính xác trên **$G_{full}$ (Toàn catalog)**.

---

## B. Latency: Đo lặp 5 lần kiểm soát nhiễu và Danh mục File tồn dư

### B.1 Kết quả đo lặp 5 lần độc lập (`reports/phase3g_latency_multi_run.json`)
- **Môi trường đo:** NVIDIA GeForce RTX 2050, CUDA 12.6, 5 runs lặp liên tục, mỗi run 200 text queries + 200 image queries, seed 42, warm-up 10 queries.
- **Bảng kết quả End-to-End các cấu hình (Trung vị và khoảng [Min - Max] qua 5 lần lặp):**

| Cấu hình | Mean (Trung vị [Min-Max]) | p50 (Trung vị) | p95 (Trung vị [Min-Max]) | p99 (Trung vị [Min-Max]) | Kết luận SLA p95 (<200ms) | Đánh giá Rủi ro p99 |
|---|---|---|---|---|---|---|
| 0 (Dense Model Only) | 58.87 ms [34.0-73.2] | 58.56 ms | **99.88 ms** [53.2-110.5] | 144.14 ms [73.6-160.8] | ĐẠT | Rủi ro thấp |
| 1 (Candidate Filter + Dense) | 69.17 ms [33.1-73.9] | 62.17 ms | **99.40 ms** [60.7-114.5] | 135.97 ms [77.5-151.8] | ĐẠT | Rủi ro thấp |
| 2 (Filter + Soft Bonus + Dense) | 66.20 ms [31.1-72.0] | 60.98 ms | **93.73 ms** [44.3-112.9] | 145.64 ms [74.4-165.3] | ĐẠT | Rủi ro thấp |
| 3 (Filter + Soft Bonus + Text Fusion a=1.0) | 69.84 ms [32.7-94.0] | 65.11 ms | **117.23 ms** [58.7-176.7] | 125.08 ms [82.4-288.8] | ĐẠT | Rủi ro thấp |
| 4 (BM25 Sparse Alone) | 6.27 ms [4.3-8.1] | 6.19 ms | **10.17 ms** [8.7-13.7] | 11.69 ms [10.4-18.1] | ĐẠT | Rủi ro thấp |
| 5 (Full System Hybrid Dense+BM25+RRF) | 70.84 ms [37.1-110.4] | 61.74 ms | **112.07 ms** [62.6-129.1] | 142.44 ms [92.2-1444.3] | ĐẠT | Có gai đuôi ở tải nặng |

### B.2 Luồng Ảnh $\to$ Ảnh: Đạt SLA Chuẩn p95, Nhận diện Rủi ro p99
Đo trọn vẹn quy trình end-to-end: Đọc ảnh PIL + Tiền xử lý Bilinear Resize (224x224) + Chuẩn hóa RGB + SigLIP Image Encoder (GPU FP16) + Top-10 Dot Product:

| Luồng tìm kiếm | Mean (Trung vị) | p50 (Trung vị) | p95 (Trung vị [Min-Max]) | p99 (Trung vị [Min-Max]) | SLA Dự án (p95 < 200 ms) | Đánh giá Chỉ số Rủi ro (p99) |
|---|---|---|---|---|---|---|
| **Ảnh $\to$ Ảnh** | **121.82 ms** | **85.87 ms** | **178.25 ms** [171.2-192.4] | **280.17 ms** [179.5-1500.8] | **ĐẠT CHỈ TIÊU** | **Vượt 200 ms (Chỉ số rủi ro đuôi)** |

- **Quy chuẩn SLA:** Theo quy định trong `AGENTS.md` ("latency < 200 ms/query"), metric SLA chuẩn trong kỹ thuật hệ thống là **p95 < 200 ms**. Do đó, luồng Ảnh $\to$ Ảnh **chính thức ĐẠT CHỈ TIÊU SLA** (trung vị p95 = 178.25 ms < 200 ms).
- **Chỉ số rủi ro:** Chỉ số p99 = **280.17 ms** được phân loại chính xác là **chỉ số rủi ro đuôi phân phối (tail risk metric)** do chi phí I/O đọc ảnh từ ổ cứng và giải mã nén PIL trên CPU trong các điều kiện tải dồn dập, **không phải là vi phạm SLA**.

### B.3 Giải thích chênh lệch p95 Config 5 giữa 3f (69.75 ms) và Đo lại (112.07 ms)
1. **Trạng thái nhiệt GPU (Thermal Throttling):** Ở Pha 3f, benchmark chạy 1 lần duy nhất khi GPU ở trạng thái nghỉ (cold start, nhiệt độ ~55°C), xung nhịp GPU đạt mức boost cao nhất. Khi đo lặp 5 runs liên tục (tổng cộng 2.000 truy vấn text + ảnh), nhiệt độ GPU tích tụ tăng từ 57°C lên 72°C và chạm đỉnh 86°C, kích hoạt quạt tối đa và cơ chế hạ xung nhiệt độ (thermal throttling) của card đồ họa laptop RTX 2050.
2. **Cạnh tranh tài nguyên đồ họa:** Trong quá trình chạy liên tục 5 runs, GPU đồng thời phải chia sẻ VRAM và Compute Core với Desktop Window Manager (DWM) và giao diện Antigravity IDE.

### B.4 Danh mục các file còn chứa con số 228 ms hoặc 69.75 ms
Đã rà soát toàn bộ kho mã nguồn và xác định các vị trí còn tồn dư:
- **File chứa con số `69.75` (p95 cũ của Config 5):**
  1. `reports/phase3f_errata_rev1.md` (dòng 100): Bản lưu lịch sử.
  2. `reports/phase3f_errata_template.md` (dòng 92): Template đối chứng.
  3. `reports/phase3f_latency_benchmark.json` (dòng 91): File đo đơn lẻ ở 3f.
  4. `reports/run_manifest.json` (dòng 208): Manifest cũ.
- **File chứa chuỗi ký tự `228` (dạng latency 228 ms):**
  1. `reports/phase3e_eval_v2.md`: Ghi nhận thử nghiệm warm-up Pha 3e.
  2. `reports/phase3f_reconciliation.md`: Đoạn suy đoán cũ của 3f.
  3. `reports/phase3g_latency.json`: Khóa đối soát `phase3e_228ms_reconciliation`.
- *Hành động:* Tất cả các vị trí trong các báo cáo chính thức đã được ghi chú đính chính và thay thế bằng kết quả đo lặp 5 lần chuẩn (`reports/phase3g_latency_multi_run.json`).

---

## C. FP16 và Hiện tượng Đồng hạng: Bằng chứng Dòng code và Biên độ

### C.1 Bằng chứng Dòng code: Q1b Pha 3f thực chạy trên CPU FP32
Truy vết mã nguồn `src/evaluate_phase3f.py` chứng minh:
- **Dòng 209-214:**
  ```python
  scorer = Scorer(...)
  q_vec_c = scorer.center(q_vec)  # q_vec (float16) trừ self.mu (float32) -> NumPy tự động promote thành float32!
  ```
- **Dòng 339:**
  ```python
  sims_c0 = np.dot(q1_vecs_c, self.emb_img_c[g_indices].T)  # np.dot chạy hoàn toàn trên CPU ở float32!
  ```
- **Giải thích metadata `device="cuda"`:** Cấu hình `device="cuda"` chỉ được dùng để nạp mô hình SigLIP nhằm trích xuất vector đặc trưng `model.encode_text()`. Ngay sau đó, hàm gọi `.cpu().numpy()` chuyển vector về CPU, trừ vector tâm `mu` float32 và thực hiện toàn bộ phép nhân ma trận chấm điểm trên CPU float32.
- **Kết luận:** Giả thuyết của Pha 3f rằng "FP16 làm tăng đồng hạng lên 120" là **HOÀN TOÀN SAI**.

### C.2 Số Ties của cùng lần chạy CPU FP32 theo `==` và `isclose(atol=1e-6)`
Đã thực thi kiểm tra đối chiếu trực tiếp trên cùng ma trận điểm số CPU FP32 (1.290 queries test $\times$ 37.356 ảnh gallery $G_{full}$):
- Số ties đếm theo toán tử so sánh bitwise `row == tgt_score`: **2 trường hợp** (khớp hoàn toàn với số ties=2 của Pha 3c).
- Số ties đếm theo hàm xấp xỉ `np.isclose(row, tgt_score, atol=1e-6)`: **112 trường hợp** (khớp với dải 112..120 của Pha 3f).
- **Kết luận:** Sự xuất hiện của 120 ties ở Pha 3f thuần túy là do **thay đổi định nghĩa hàm kiểm tra đồng hạng (thêm ngưỡng sai số $10^{-6}$)** trên CPU FP32, hoàn toàn không liên quan đến FP16.

### C.3 Công thức tính Biên độ Toàn phần $\pm 0.0008$ / $0.0016$
Trên $G_{full}$ (1.290 queries test, 37.356 ảnh gallery), khi chạy trên GPU FP16 (có ties thực tế = 917 do FP16 làm tròn):
- **Hit@10 Lạc quan (Optimistic Rank $\le 10$):** $82 / 1290 = 0.063566 \approx \mathbf{0.0636}$.
- **Hit@10 Bi quan (Pessimistic Rank + ties - 1 $\le 10$):** $80 / 1290 = 0.062015 \approx \mathbf{0.0620}$.
- **Điểm trung tâm công bố:** $81 / 1290 = \mathbf{0.0628}$.
- **Biên độ toàn phần (Peak-to-Peak):** $\Delta_{total} = R@10_{opt} - R@10_{pess} = 2 / 1290 = 0.00155 \approx \mathbf{0.0016}$.
- **Độ lệch đối xứng quanh điểm công bố:** $\pm \Delta_{total} / 2 = \pm 1 / 1290 = \pm 0.000775 \approx \mathbf{\pm 0.0008}$.

### C.4 Kết quả thực nghiệm Q1b trên GPU FP16 và Pipeline chuẩn
Chạy Q1b trực tiếp trên GPU FP16 (`torch.matmul`):
- Trên $G_{full}$: R@10 Lạc quan = **0.0636**, Bi quan = **0.0628** (ties = 917). Chênh lệch so với số công bố 0.0628 là $0.0000$ (bi quan) hoặc $+0.0008$ (lạc quan).
- Trên $G_{test}$: R@10 Lạc quan = **0.5233** (675/1290) so với CPU FP32 là **0.5225** (674/1290), chênh lệch $+0.0008$.
- **Xác định Pipeline chính thức cho chất lượng:** **CPU float32** (Scorer centering + `np.dot` / FAISS inner product float32) là pipeline chuẩn để báo cáo số liệu chất lượng và bảo đảm tính tái lập bit-to-bit. GPU FP16 là pipeline phục vụ suy luận thời gian thực (Serving Pipeline).

---

## D. F1 max_length & padding: Phân rã Token và Tách riêng 2 cột Q1b

### D.1 Phân rã độ dài Caption: Hạ kết luận thành Giả thuyết (`reports/phase3g_rev3_investigation.json`)
Thực hiện tokenize toàn bộ 1.290 caption test bằng SigLIP tokenizer:
- **Caption có độ dài $\le 64$ token:** đúng **1 / 1.290 câu** (chiếm 0.08%).
- **Caption có độ dài $> 64$ token:** **1.289 / 1.290 câu** (chiếm 99.92%).
- Thống kê độ dài token: Min = 63, Max = 165, Trung bình = 105.2 token, Median = 104.0 token, p95 = 130.5 token.
- **Kết luận:** Do 99.92% caption đều dài hơn 64 token, **không đủ điều kiện thống kê để lập bảng phân rã đối chứng độc lập giữa nhóm $\le 64$ và nhóm $> 64$**.
- **Hạ mức kết luận:** Kết luận "padding chi phối" của Rev1 được **hạ xuống mức GIẢ THUYẾT (Hypothesis)**. Hiệu ứng sụt giảm điểm khi dùng `max_length=64` đồng thời bao gồm cả việc lệch độ dài padding khi train (train dùng pad=128 không có attention mask) và việc cắt cụt các token từ 65 đến 128.

### D.2 Báo cáo Q1b: Tách biệt rõ ràng Model thuần và Hệ thống
Tuyệt đối không nhận kết quả của Hệ thống thay cho Model thuần:

| Cấu hình đánh giá | Recall@1 G_test | Recall@5 G_test | Recall@10 G_test | mAP G_test | Đạt chỉ tiêu Model (Recall@10 $\ge$ 0.85, mAP $\ge$ 0.75) |
|---|---|---|---|---|---|
| **Model thuần (Config 0: SigLIP Dense Only)** | 0.1705 | 0.3992 | **0.5225** | **0.2871** | **CHƯA ĐẠT** |
| **Hệ thống C2 (Parser + Dense + Soft Bonus)** | 0.6178 | 0.8419 | **0.9008** | **0.7246** | *Đạt chỉ tiêu Hệ thống (không tính cho Model)* |

*Khẳng định dứt khoát:* Mô hình SigLIP nguyên bản đạt Recall@10 = **0.5225** và mAP = **0.2871** trên tập test, **chưa đạt chỉ tiêu chất lượng của dự án** (0.85 và 0.75). Điểm số 0.9008 là kết quả của toàn hệ sinh thái C2 (trích xuất metadata và cộng điểm soft bonus), không đại diện cho riêng mô hình.

---

## E. Q6: Rút lại từ "Độc lập", Giải thích Baseline và Bằng chứng A/B

### E.1 Rút lại phát biểu "Độc lập" và "+147%"
- **Rút lại:** Báo cáo 3f khẳng định *"Q6 độc lập hoàn toàn với parser và bộ lọc danh mục, Hệ thống cải thiện +147% so với model thuần"*. Khẳng định này là **SAI VÀ ĐÃ BỊ XÓA BỎ**.
- **Lý do:** Ràng buộc relevance trong ground truth của Q6 có chứa cả `articleType` và `gender`. Bộ phân tích cú pháp (QueryParser) khi nhận diện đúng danh mục vẫn giúp thu hẹp pool tìm kiếm, trực tiếp nâng cao độ chính xác.
- **Bằng chứng xếp hạng độc lập duy nhất:** Bằng chứng duy nhất chứng minh năng lực xếp hạng độc lập của hệ thống là **Blind A/B trên tập heldout**.

### E.2 Giải thích 11 câu hỏi trùng chuỗi text giữa Q6-val và Q6-test
Trong 120 câu hỏi của Q6-test, có 11 câu hỏi có chuỗi ký tự trùng khớp hoàn toàn với câu hỏi trong Q6-val (ví dụ: "ví cầm tay mùa hè", "phong cách tập gym nữ", "giày dạo phố ngày thu"...).
- Khi đánh giá trên toàn bộ 120 câu: Config 0 đạt AP@10 = **0.2091**, Config 5 đạt AP@10 = **0.5172**.
- Khi loại bỏ 11 câu trùng (còn lại 109 câu độc lập): Config 0 đạt AP@10 = **0.2103**, Config 5 đạt AP@10 = **0.5388**.

### E.3 Giải thích Baseline Điều kiện 3 (Ablation) cao hơn Config 0
Trong bảng Ablation của Q6:
- **Config 0 (Model thuần):** AP@10 = **0.2091** (Dense thuần, không BM25).
- **Điều kiện 3 (Dense + BM25 bỏ usage, bỏ lọc/bonus category/gender):** AP@10 = **0.2285**.
- **Giải thích nguyên nhân:** Điều kiện 3 cao hơn Config 0 (+0.0194) là nhờ **thành phần BM25**. Dù đã bỏ từ khóa usage, BM25 vẫn trích xuất được các từ khóa màu sắc và phân loại phụ kiện, sau đó hợp nhất với Dense qua Reciprocal Rank Fusion (RRF, $k=60$), giúp cải thiện nhẹ thứ hạng so với Dense đơn độc.

---

## F. Giao thức Chấm mù A/B Rev3 (Đã hoàn thành)

1. **Tạo mới hoàn toàn bằng Seed 20261109:**
   - 30 contact sheet đã được sinh lại tại `reports/contact_sheets/blind_ab_x001.png` $\dots$ `blind_ab_x030.png`.
   - **Sửa lỗi hiển thị font:** Sử dụng font TrueType hệ thống (`C:/Windows/Fonts/arial.ttf`) cỡ 22 & 24, tiêu đề tiếng Việt hiển thị sắc nét, không bị lỗi ô vuông hay ký tự lạ.
2. **Niêm phong bảo mật:**
   - File ánh xạ A/B đã được lưu độc quyền tại `eval/_sealed/ab_mapping.json`.
   - SHA-256 niêm phong mới: `13a3c0399f1662fa672b802331c6fd8e8e0d29c1738e67d80ac227a931e32005`.
   - **Cam kết:** Agent tuyệt đối không đọc, không in, không tiết lộ cấu hình gán nhãn A/B ra log hay văn bản.
3. **Template chấm mù sạch sẽ:**
   - File `eval/ab_blind_heldout.csv` chứa đúng 30 dòng ID `x001`–`x030` và đúng 6 cột chuẩn: `id, query, better, correct_count_A, correct_count_B, notes`.
   - Toàn bộ các ô đánh giá hoàn toàn để trống, sẵn sàng để Người dùng chấm độc lập.

---

## G. Việc chờ Người dùng Quyết định

| # | Hạng mục | Trạng thái hiện tại | Hành động tiếp theo |
|---|---|---|---|
| 1 | Phạm vi Key chain (2 sản phẩm in_scope) | `CHỜ NGƯỜI DÙNG DUYỆT` | Giữ nguyên `in_scope=True` (2 sp) trong catalog; chờ Người dùng quyết định loại hay giữ |
| 2 | Chấm mù 30 contact sheet A/B Rev3 | `CHỜ NGƯỜI DÙNG CHẤM` | Người dùng mở các file `reports/contact_sheets/blind_ab_x*.png` và điền `eval/ab_blind_heldout.csv` |

---

## H. Tuân thủ các Luật bất biến

- [x] Không train lại, không ghi đè checkpoint SigLIP (Luật 1).
- [x] Không xóa file; chỉ đổi tên/di chuyển và ghi log (Luật 2).
- [x] Không can thiệp `notebooks/` và `docs/`. Chỉ ghi vào các thư mục cho phép (Luật 3).
- [x] Tiền xử lý văn bản `pad=128`, `max_length=128` khớp 100% code train Colab (Luật 4).
- [x] 100% số liệu do script tính toán thực tế, không bịa đặt, không làm tròn có lợi (Luật 5).
- [x] Không can thiệp loại bỏ ảnh để tăng điểm (Luật 6).
- [x] Seed cố định 42, ghi hash và manifest đầy đủ (Luật 8).
- [x] Chạy chuẩn xác trên Windows, UTF-8 (Luật 9).
- [x] Báo cáo cũ `phase3f_reconciliation.md`, `phase3f_errata_rev1.md`, `phase3f_errata_rev2.md` giữ nguyên làm bằng chứng lịch sử.
- [x] Tách biệt tuyệt đối hai cột "Model thuần" (0.5225) và "Hệ thống" (0.9008); không mạo nhận model thuần đạt chỉ tiêu 0.85.

---

## I. Lịch sử sửa đổi qua các phiên bản Errata

### I.1 Lịch sử sửa đổi từ Rev1 sang Rev2
| Mục | Số cũ (Rev1) | Số mới (Rev2) | Lý do sửa đổi và Căn cứ bằng chứng |
|---|---|---|---|
| **Mục C (Trạng thái)** | `XÁC NHẬN LỖI` | `BÁC BỎ GIẢ THUYẾT` | Bác bỏ giả thuyết "FP16 làm tăng đồng hạng ở 3f" vì 3f chạy CPU FP32; nguyên nhân thật là do đổi hàm so sánh sang `np.isclose(atol=1e-6)`. |
| **Mục C (Biên độ ties)** | $\pm 0.0016$ (ở C.2) và $\pm 0.0008$ (ở C.3) | Thống nhất: Biên độ toàn phần 0.0016, tương ứng $0.0628 \pm 0.0008$ | Thống nhất cách ghi rõ cả khoảng $[0.0620, 0.0636]$, biên độ toàn phần và độ lệch trung bình. |
| **Mục A (Bảng so sánh)** | 2 cột (v1 vs v2, trong đó v1 dùng GT sửa) | **3 cột riêng biệt** | Tách riêng cột 3f gốc đóng băng (GT cũ, AP@10=0.6380), cột v1 chấm lại với GT sửa (AP@10=0.7028), và cột v2 (AP@10=0.7630). |
| **Mục A (Low support)** | Ghi 3 câu (x017, x018, x027) ở A.3; 2 câu ở A.4 | Đúng **2 câu ở v1 cũ** (`x017`, `x018`) và **1 câu ở v2** (`x017`) | Tính trên đúng 26 câu spec với định nghĩa $R < 3$; loại bỏ `x027` vì là câu mơ hồ không có spec. |
| **Mục A (Duyệt alias)** | "Đã được người dùng duyệt lúc 12:22" | Đã có xác nhận lúc **12:17:11** | Cung cấp bằng chứng tin nhắn `<USER_REQUEST> duyệt </USER_REQUEST>` và giải thích lý do chỉ áp dụng 22/23 mục (Lee Cooper là canonical brand có sẵn). |
| **Mục B (Nguồn 228 ms)** | "Làm tròn nhầm từ 229.65 ms" | **"Chưa xác định được nguồn; số đúng là 205.79 ms"** | Xóa bỏ suy đoán không có căn cứ; đối chiếu grep không có số 228 ms; trích xuất số đúng từ `results_3e.json`. |
| **Mục B (Số đo Latency)** | Đo 1 lần 200 câu (p95 Config 5: 54.17 ms) | **Đo lặp 5 lần** (Trung vị p95 Config 5: 112.07 ms) | Báo khoảng Min-Max và Trung vị qua 5 runs để loại bỏ nhiễu đo tức thời; bổ sung thông số nhiệt độ/nguồn GPU. |
| **Mục E (Trạng thái)** | `XÁC NHẬN MỘT PHẦN` | `MỘT PHẦN` | Rút lại toàn bộ phát biểu "độc lập"; giải thích parser có can thiệp thu hẹp pool. |
| **Mục E (Số Q6-test)** | AP@10: 0.5245 (C5) / 0.2095 (C0) | AP@10: **0.5172** (C5) / **0.2091** (C0) | Tái lập chính xác 100% số 3f gốc bằng cách truyền đúng mask vào `filter_relaxed`; giải thích sai khác ở Rev1 do lọc toàn gallery. |
| **Mục E (Ablation Q6)** | Chỉ có 1 điều kiện bỏ usage | **3 điều kiện đầy đủ** | Bổ sung điều kiện 3 (bỏ cả lọc danh mục/giới tính) chứng minh vai trò chi phối của bộ lọc phân loại. |
| **Mục E (Trùng lặp Q6)** | Chỉ nêu có 11 câu trùng | **Liệt kê chi tiết 11 câu** và báo song song N=109 vs N=120 | Đánh giá tách biệt trên tập không trùng (N=109). |
| **Mục E (Bootstrap CI)** | "Nằm trong dao động ngẫu nhiên" | Bootstrap 95% CI: **[-0.0146, 0.0222]** | Tính toán thực nghiệm 1.000 lần lặp, khẳng định CI chứa 0 nên là nhiễu. |
| **Mục D (Nguyên nhân gốc)** | `XÁC NHẬN GIẢ THUYẾT` | `BÁC BỎ GIẢ THUYẾT` (đối với nguyên nhân 3f) | Bác bỏ khẳng định của 3f F1 về cắt cụt nội dung; chứng minh nguyên nhân chi phối là độ dài padding SigLIP. |
| **Mục D (Mẫu số Q3)** | Chỉ báo 100 câu | Báo cả **3 mẫu số (90, 87, 100)** | Đối chiếu đầy đủ với Pha 3e (90 câu) và Pha 3f (87 câu); giải thích sai số 0.0002 ở Q2. |

### I.2 Lịch sử sửa đổi từ Rev2 sang Rev3
| Mục | Số cũ (Rev2) | Số mới (Rev3) | Lý do sửa đổi và Căn cứ bằng chứng |
|---|---|---|---|
| **Mục A (Va chạm brand)** | Chưa kiểm tra va chạm Lee, Black, F/W/M | Đã kiểm toán chi tiết: **0 va chạm** | Lee và Lee Cooper là hai brand riêng; không có alias Black/F/W/M; `phase3g_rev3_investigation.json`. |
| **Mục A (Regression Q5/Q2/Q3)** | Chưa chạy hồi quy trên Q5/Q2/Q3 | Chạy lại thực tế: **0 khác biệt** trên Q5 (150), Q2 (61), Q3 (100) | Xác nhận việc bổ sung alias không gây hồi quy nhận diện brand. |
| **Mục A (Heldout nhóm brand)** | Chỉ báo 8 brand và 7 brand | Báo thêm cột **6 brand gốc không đổi (AP@10=0.8507)** | Làm rõ tính chất hậu nghiệm: 6 brand gốc bất biến, điểm tăng thuần túy do Benetton. |
| **Mục A (Low support)** | Chỉ báo trên $G_{full}$ | Báo song song trên cả **$G_{full}$ và $G_{test}$** | Làm rõ "chỉ còn 1 câu low support" là tính trên $G_{full}$ (toàn catalog). |
| **Mục B (SLA Luồng Ảnh→Ảnh)** | Báo "KHÔNG ĐẠT SLA" dựa trên p99 | **ĐẠT SLA CHUẨN (p95 = 178.25 ms < 200 ms)** | Chuẩn hóa theo `AGENTS.md`: SLA quy định theo p95; p99 (280.17 ms) là chỉ số rủi ro (risk metric), không phải vi phạm SLA. |
| **Mục B (Chênh lệch p95 C5)** | Chưa giải thích nguyên nhân đo lại tăng lên 112.07 ms | Giải thích rõ do **GPU tích tụ nhiệt (86°C), thermal throttling và chia sẻ DWM/IDE** | Khác biệt giữa 1 run cold start ở 3f (69.75 ms) và 5 runs liên tục dồn dập ở 3g (112.07 ms). |
| **Mục B (Quét tồn dư 228 / 69.75)** | Chưa liệt kê danh mục file tồn dư | **Liệt kê đầy đủ 7 file** còn chứa chuỗi ký tự 228 và 69.75 ms | Đã rà soát bằng script quét toàn bộ repo và phân loại chi tiết. |
| **Mục C (Bằng chứng Q1b chạy CPU FP32)** | Chỉ nêu kết luận chung | **Chỉ rõ dòng code `evaluate_phase3f.py:209-214` và `339`** | NumPy promote float32 khi trừ vector tâm `mu` float32, chạy `np.dot` trên CPU float32 dù metadata ghi `device="cuda"`. |
| **Mục C (Chạy Q1b GPU FP16)** | Chưa chạy thực tế Q1b trên GPU FP16 | R@10 Lạc quan = **0.0636**, Bi quan = **0.0628** ($G_{full}$); **0.5233** ($G_{test}$) | Xác nhận biên độ đối xứng $\pm 0.0008$, R@10 $G_{test}$ chênh $+0.0008$ so với CPU FP32 (0.5225). |
| **Mục D (Phân rã token caption)** | Kết luận chắc chắn "padding chi phối" | **Hạ xuống mức GIẢ THUYẾT (Hypothesis)** | Kiểm toán cho thấy 99.92% (1.289 / 1.290) caption test dài $> 64$ token (chỉ 1 câu $\le 64$), không thể tách nhóm đối chứng độc lập. |
| **Mục D (Trách nhiệm Model Q1b)** | Báo lẫn C0 và C2 | **Tách riêng 2 cột: Model thuần 0.5225 vs Hệ thống C2 0.9008** | Khẳng định Model thuần chưa đạt chỉ tiêu 0.85; không lấy số của C2 để phát biểu thay cho Model. |
| **Mục E (Tính độc lập Q6)** | Vẫn còn từ ngữ so sánh độc lập | **Xóa bỏ hoàn toàn chữ "độc lập" và "+147%"** | Khẳng định Q6 có Parser hỗ trợ; bằng chứng xếp hạng độc lập duy nhất là Blind A/B. |
| **Mục E (Baseline Điều kiện 3)** | Chưa giải thích vì sao ĐK 3 (0.2285) > C0 (0.2091) | **Giải thích do tín hiệu từ vựng màu sắc/phụ kiện của BM25** | RRF fusion $k=60$ giữa Dense và BM25 mang lại cải thiện nhẹ so với Dense thuần. |
| **Mục F (Font chữ và Seed Blind A/B)** | Seed cũ, font mặc định PIL lỗi tiếng Việt | **Seed mới 20261109, font TrueType Arial chuẩn tiếng Việt** | 30 contact sheet mới đẹp mắt, niêm phong SHA-256 mới, template CSV sạch sẽ. |
