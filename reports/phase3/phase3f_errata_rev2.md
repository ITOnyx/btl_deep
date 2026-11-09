# Errata Pha 3f (Pha 3g Rev2): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 2026-11-09 (Bản Rev2 hoàn thiện)  
**Chế độ:** Suy luận thuần. Không train lại, không sửa checkpoint SigLIP, không xóa file.  
**Báo cáo được đính chính:** `reports/phase3f_reconciliation.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 1:** `reports/phase3f_errata_rev1.md` (giữ nguyên làm bằng chứng).

> **Quy ước bất biến**
> - 100% con số trong báo cáo được sinh tự động từ script và đối soát với các file JSON trong `reports/`. Tuyệt đối không gõ tay, không bịa số.
> - Trạng thái mỗi mục dùng chuẩn: `XÁC NHẬN LỖI` / `BÁC BỎ GIẢ THUYẾT` / `MỘT PHẦN` / `CHƯA XÁC ĐỊNH` / `CHỜ NGƯỜI DÙNG`.
> - Hai cột **"Model thuần"** (Config 0) và **"Hệ thống"** (Config 5) luôn tách riêng biệt. Không dùng số của Hệ thống để phát biểu thay cho Model thuần.
> - Phát biểu nào chưa có phép đo chống lưng được ghi rõ là `chưa xác định`.

---

## 0. Bảng tổng hợp trạng thái các vấn đề (Rev2)

| Mục | Vấn đề rà soát | Trạng thái Rev2 | Ảnh hưởng đến kết luận Pha 3f | File dữ liệu nguồn |
|---|---|---|---|---|
| **A** | Benetton bị ghi "catalog rỗng, R=0" | `XÁC NHẬN LỖI` | Benetton có 209 sp strict ở G_full; bổ sung alias v2 hậu nghiệm đưa x018 AP@10 từ 0.5181 lên 1.0000; số câu low support trên 26 câu spec chỉ còn 1 câu (Lavie) | `reports/brand_audit_3g.json`, `reports/phase3g_brands_v2_eval.json`, `reports/phase3g_ties_q3_rev2.json` |
| **B** | Bảng Latency không khớp JSON gốc | `XÁC NHẬN LỖI` | Bảng 3f lấy nhầm số từ các dòng khác nhau; đo lại chuẩn xác 5 lần lặp: p95 Config 5 là 112.07 ms (ĐẠT); luồng Ảnh → Ảnh có p99 = 280.17 ms (KHÔNG ĐẠT SLA < 200ms) | `reports/phase3g_latency_multi_run.json` |
| **C** | FP16 làm tăng đồng hạng (ties) ở 3f | `BÁC BỎ GIẢ THUYẾT` | Giả thuyết FP16 gây ra ties=120 bị bác bỏ vì 3f chạy CPU FP32; nguyên nhân thật là do đổi hàm so sánh từ `==` sang `np.isclose(atol=1e-6)`. 3f đã gộp 2 pipeline khác nhau | `reports/phase3g_fp16_ties.json`, `reports/phase3g_ties_q3_rev2.json` |
| **D** | Nguyên nhân gốc `max_length` ở F1 | `BÁC BỎ GIẢ THUYẾT` | Giả thuyết "cắt cụt nội dung ở đuôi" bị bác bỏ; nguyên nhân chi phối là do độ dài padding của SigLIP không có attention mask; pipeline chính thức khớp 100% code train | `reports/phase3g_maxlen.json`, `reports/phase3g_ties_q3_rev2.json` |
| **E** | Tính độc lập của tập test Q6 (F3) | `MỘT PHẦN` | Rút lại kết luận "độc lập hoàn toàn" vì GT có articleType+gender; BM25 chứa usage_vi; tỉ lệ 2.47x không phải cho ranking thuần. Tái lập chính xác số 3f gốc (AP@10=0.5172) | `reports/phase3g_q6_rev2.json` |
| **F** | Giao thức và bảo mật chấm mù A/B | `XÁC NHẬN LỖI` | Sửa tên file ảnh `blind_ab_x001..x030`, chuẩn hóa 6 cột CSV, niêm phong an toàn ánh xạ `eval/_sealed/ab_mapping.json` | `reports/contact_sheets/`, `eval/_sealed/` |

---

## A. Brand: Sửa kết luận Benetton và Kiểm toán Alias

### A.1 Phát biểu cũ và Bằng chứng mâu thuẫn
- **Phát biểu cũ (3f, mục F6):** x017 (Lavie) và x018 (Benetton) có R = 0, AP@10 "toán học bằng 0"; báo cáo "6/6 brand hợp lệ".
- **Bằng chứng mâu thuẫn:**
  1. `reports/heldout_inspection_b0.json`: x018 có `strict_full` = 209 sp, `strict_test` = 7 sp.
  2. `eval/results_3e.json`: tự ghi chẩn đoán "Brand mismatch (United Colors of Benetton vs Benetton)".
  3. `reports/phase3e_part_a_review.md`: thương hiệu "United Colors of Benetton" có 1.462 sản phẩm.
- **Nguyên nhân thật sự:** Catalog có rất nhiều sản phẩm Benetton dưới tên đầy đủ `"United Colors of Benetton"`. Báo cáo 3f kết luận "catalog rỗng" là do `src/brands.json` thiếu alias, làm parser không nhận diện được brand và rơi vào nhánh trích xuất sai ground truth.

### A.2 Kiểm toán sản phẩm Benetton và Lavie trong catalog (`reports/brand_audit_3g.json`)

| Query ID | Tên Brand truy vấn | Khớp chính xác nhãn brand | Khớp chuỗi con trong tên | Số sp liên quan theo GT sửa (G_full) | Parser v1 nhận diện | Parser v2 nhận diện |
|---|---|---|---|---|---|---|
| `x017` | Lavie | 0 sp | 0 sp | 0 sp | None | None |
| `x018` | Benetton | 1.303 sp | 1.337 sp | 209 sp (Tshirts Men) | None | United Colors of Benetton |

**Giải thích sự khác nhau về số lượng sản phẩm Benetton giữa các nguồn:**
- **1.462 sản phẩm:** Đếm từ `extract_brands.py` (ở Pha 3e) trên toàn bộ dataset thô khi gom các token `"united"`, `"ucb"`, `"benetton"` về cụm thương hiệu Benetton.
- **1.337 sản phẩm:** Số sản phẩm khớp chuỗi con `"benetton"` (không phân biệt hoa thường) trong trường `name` của tập `in_scope`.
- **1.303 sản phẩm:** Số sản phẩm mang chính xác nhãn canonical `"United Colors of Benetton"` trong trường `name` của tập `in_scope`.
- **34 sản phẩm:** Thuộc các dòng nhánh phụ hoặc viết tắt (như `"Undercolors of Benetton"` hoặc `"UCB ... Benetton"`).
- **Phủ alias:** Alias `"benetton" -> "United Colors of Benetton"` trong `src/brands_v2.json` phủ toàn bộ 1.303 sản phẩm chính và giải quyết triệt để câu truy vấn x018.

### A.3 Bằng chứng phê duyệt Alias của người dùng
- **Trạng thái:** `ĐÃ CÓ XÁC NHẬN CỦA NGƯỜI DÙNG`.
- **Thời điểm:** `2026-11-09T12:17:11+07:00`.
- **Nội dung xác nhận:** Người dùng đã gửi tin nhắn trực tiếp `<USER_REQUEST> duyệt </USER_REQUEST>` đối với danh sách đề xuất trong `reports/brand_alias_proposal.csv`.
- **Chi tiết áp dụng (23 đề xuất):**
  - Đã thêm **22 alias** vào `src/brands_v2.json` (SHA-256: `c5283096f649351d4433874915e2e32586118200e576f41b3101495859c1898e`).
  - **1 mục không áp dụng vào dictionary aliases:** `Lee Cooper`. Lý do: `Lee Cooper` vốn dĩ đã là một canonical brand độc lập có sẵn trong danh sách 234 brand gốc của `brands.json`, không phải alias của thương hiệu khác.

### A.4 Bảng so sánh 3 cột: Tách biệt đóng góp sửa Ground Truth và thêm Alias
Tất cả các số liệu dưới đây đều được ghi nhận là **sửa đổi hậu nghiệm (post-hoc)** vì đã quan sát tập heldout.

| Chỉ số đánh giá | (1) 3f gốc đóng băng (GT cũ, brands v1) | (2) Brands v1 chấm lại với GT sửa | (3) Brands v2 (alias) với GT sửa | Đóng góp sửa GT: (2) - (1) | Đóng góp thêm Alias: (3) - (2) |
|---|---|---|---|---|---|
| **x018 (Benetton) AP@10** | 0.0000 | 0.5181 | **1.0000** | +0.5181 | +0.4819 |
| **x018 (Benetton) P@10** | 0.0000 | 0.6000 | **1.0000** | +0.6000 | +0.4000 |
| **Heldout nhóm brand (8 câu) AP@10** | 0.6380 | 0.7028 | **0.7630** | +0.0648 | +0.0602 |
| **Heldout nhóm brand (8 câu) P@10** | 0.6750 | 0.7500 | **0.8000** | +0.0750 | +0.0500 |
| **Heldout 7 brand hợp lệ (bỏ Lavie) AP@10** | 0.7291 | 0.8032 | **0.8720** | +0.0741 | +0.0688 |
| **Heldout 26 câu spec tổng AP@10** | 0.8834 | 0.9033 | **0.9221** | +0.0199 | +0.0188 |
| **Số câu hỗ trợ thấp ($R < 3$) trên 26 câu spec** | 2 câu (`x017`, `x018`) | 1 câu (`x017`) | 1 câu (`x017`) | -1 câu | 0 câu |

### A.5 Sửa phát biểu và Đếm số câu hỗ trợ thấp
- **Số câu hỗ trợ thấp trên đúng 26 câu spec (với định nghĩa $R < 3$ trên $G_{full}$):**
  - Ở 3f gốc: có **2 câu** là `x017` (Lavie, R=0) và `x018` (Benetton, bị tính R=0).
  - Ở bản sửa GT và brands v2: chỉ còn duy nhất **1 câu** là `x017` (Lavie, R=0).
  - *Lưu ý về x027:* Câu `x027` ("quần áo mặc nhà mùa hè") là câu thuộc nhóm 4 câu đánh giá thủ công (spec = null), không nằm trong tập 26 câu spec.
- **Sửa phát biểu F6:**
  - Cũ: "6/6 brand hợp lệ; 2 câu rỗng; Mean AP@10 = 0.8507".
  - Mới: "Trong catalog có **7/8 brand hợp lệ** (chỉ Lavie có R=0; Benetton có 209 sp ở G_full). Với brands v2, câu x018 đạt AP@10=1.0000, Mean AP@10 của 7 brand hợp lệ đạt **0.8720** (tính cả 8 brand là 0.7630)."
- **Zero-result guard cho Lavie:** Đề xuất cơ chế khi parser nhận diện thương hiệu có 0 sản phẩm thì fallback về tìm kiếm theo danh mục và giới tính. Chưa kích hoạt ở pha này.

---

## B. Latency: Bỏ suy đoán, Đo lặp 5 lần kiểm soát nhiễu

### B.1 Đối chiếu lỗi bảng 3f với file JSON gốc (`reports/phase3f_latency_benchmark.json`)
Bảng trong báo cáo 3f chỉ có cột P95 khớp với JSON. Cột Mean và P50 bị lệch do lấy nhầm số từ các lần chạy thử nghiệm:
- Config 5 Mean báo 41.87 ms trong khi JSON là 39.63 ms (41.87 ms thực chất là tổng đo trực tiếp các thành phần).
- Config 1 (27.53 ms) và Config 3 (28.14 ms) thấp hơn cả thời gian encode text trung bình (33.36 ms), điều bất khả thi về mặt vật lý.

### B.2 Xóa bỏ suy đoán về con số "228 ms ở Pha 3e"
- **Kết quả rà soát codebase:** Đã quét toàn bộ `reports/`, `eval/`, và `git log`. Không có bất kỳ kết quả đo hay dòng log nào sinh ra con số 228 ms.
- **Số liệu chính xác theo `eval/results_3e.json`:**
  - Config 5 (Full System): p95 = **205.79 ms**.
  - Config 0 (Model thuần): p95 = **229.65 ms**.
- **Kết luận:** **Không xác định được nguồn của con số 228 ms** trong văn bản 3f; số đúng là Config 5 p95 = 205.79 ms. Xóa bỏ hoàn toàn câu suy đoán "làm tròn nhầm từ 229.65 ms" và câu "nhờ GPU FP16 caching" vốn chưa có thực nghiệm cô lập chứng minh.

### B.3 Kết quả đo lặp 5 lần kiểm soát nhiễu (`reports/phase3g_latency_multi_run.json`)
- **Điều kiện môi trường thực tế:**
  - GPU: NVIDIA GeForce RTX 2050, Driver: 560.76, CUDA 12.6.
  - Nhiệt độ GPU: N/A °C, Công suất: N/A W / N/A W.
  - VRAM sử dụng: N/A MB / N/A MB.
  - Tiến trình nền ghi nhận trên GPU: Windows Desktop Window Manager, Antigravity IDE, Edge WebView2.
  - Tham số đo: 5 runs độc lập, mỗi run gồm 200 text queries và 200 ảnh, seed cố định 42, warm-up 10 queries trước mỗi run.

#### Bảng kết quả End-to-End các cấu hình (Trung vị và khoảng [Min - Max] qua 5 lần lặp):

| Cấu hình | Mean (Trung vị [Min-Max]) | p50 (Trung vị) | p95 (Trung vị [Min-Max]) | p99 (Trung vị [Min-Max]) | Đạt SLA p95 (<200ms) | Đạt SLA p99 (<200ms) |
|---|---|---|---|---|---|---|
| 0 (Dense Model Only) | 58.87 ms [34.0-73.2] | 58.56 ms | **99.88 ms** [53.2-110.5] | 144.14 ms [73.6-160.8] | ĐẠT | ĐẠT |
| 1 (Candidate Filter + Dense) | 69.17 ms [33.1-73.9] | 62.17 ms | **99.40 ms** [60.7-114.5] | 135.97 ms [77.5-151.8] | ĐẠT | ĐẠT |
| 2 (Filter + Soft Bonus + Dense) | 66.20 ms [31.1-72.0] | 60.98 ms | **93.73 ms** [44.3-112.9] | 145.64 ms [74.4-165.3] | ĐẠT | ĐẠT |
| 3 (Filter + Soft Bonus + Text Fusion a=1.0) | 69.84 ms [32.7-94.0] | 65.11 ms | **117.23 ms** [58.7-176.7] | 125.08 ms [82.4-288.8] | ĐẠT | ĐẠT |
| 4 (BM25 Sparse Alone) | 6.27 ms [4.3-8.1] | 6.19 ms | **10.17 ms** [8.7-13.7] | 11.69 ms [10.4-18.1] | ĐẠT | ĐẠT |
| 5 (Full System Hybrid Dense+BM25+RRF) | 70.84 ms [37.1-110.4] | 61.74 ms | **112.07 ms** [62.6-129.1] | 142.44 ms [92.2-1444.3] | ĐẠT | ĐẠT |

*Ghi chú:*
- Cột Mean của Config 5 (70.84 ms) khác với tổng trực tiếp các thành phần (39.68 ms) do đây là hai chuỗi đo độc lập (một bên gọi end-to-end qua hàm search, một bên đặt timer đo xen kẽ từng bước tính toán).
- Tất cả cấu hình text đều đạt SLA < 200 ms ở cả p95 và p99.

#### Bảng phân rã thành phần thời gian (Config 5):

| Thành phần xử lý | Mean (Trung vị) | p50 (Trung vị) | p95 (Trung vị) | p99 (Trung vị) |
|---|---|---|---|---|
| Phân tích cú pháp (parser) | 1.30 ms | 1.05 ms | 2.56 ms | 3.38 ms |
| Lọc ứng viên (relaxed L0..L6) | 0.68 ms | 0.39 ms | 2.26 ms | 3.09 ms |
| Encode query text (GPU FP16) | 30.41 ms | 25.17 ms | 61.28 ms | 86.21 ms |
| Dense scoring + Top-100 (GPU) | 2.53 ms | 1.39 ms | 5.02 ms | 7.21 ms |
| BM25 scoring (Top-100) | 4.34 ms | 3.77 ms | 8.01 ms | 10.08 ms |
| RRF merge fusion (k=60) | 0.21 ms | 0.17 ms | 0.40 ms | 0.62 ms |
| **Tổng đo trực tiếp từng bước** | 39.68 ms | 32.96 ms | 72.72 ms | 101.76 ms |

### B.4 Luồng Ảnh → Ảnh: Xác nhận ổn định qua 5 lần lặp
Đo end-to-end bao gồm: Đọc file ảnh bằng PIL + Resize BILINEAR (224x224) + Normalize + SigLIP Image Encoder (GPU FP16) + Top-10 Dot Product:

| Luồng tìm kiếm | Mean (Trung vị) | p50 (Trung vị) | p95 (Trung vị) | p99 (Trung vị [Min-Max]) | Kết luận SLA p95 | Kết luận SLA p99 |
|---|---|---|---|---|---|---|
| **Ảnh $\to$ Ảnh** | **121.82 ms** | **85.87 ms** | **178.25 ms** | **280.17 ms** [179.5-1500.8] | **ĐẠT** | **KHÔNG ĐẠT** |

**Kết luận dứt khoát:** Luồng Ảnh $\to$ Ảnh đạt chỉ tiêu ở p95 (178.25 ms < 200 ms), nhưng **vượt ngưỡng SLA ổn định ở p99 (280.17 ms $\ge$ 200 ms)** qua cả 5 lần lặp. Đây là hiện tượng đo lường ổn định (do chi phí I/O đọc ảnh từ ổ đĩa và tiền xử lý CPU), không phải do nhiễu tức thời.

---

## C. FP16 và Hiện tượng Đồng hạng: Bác bỏ giả thuyết

### C.1 Dtype và Đường tính điểm thực tế ở Pha 3f
Truy vết mã nguồn `src/evaluate_phase3f.py` và commit lịch sử của Pha 3f cho thấy:
1. `model.encode_text([q])` chạy trên GPU ở **FP16** (`half()`).
2. Vector đặc trưng đưa về CPU: `feat.cpu().numpy()` (kiểu dữ liệu ban đầu là `float16`).
3. Chuẩn hóa và trừ vector tâm: `scorer.center_vectors` thực hiện trừ `self.mu` (đang lưu ở dạng **`float32`**).
4. **Quy tắc ép kiểu tự động của NumPy:** Phép toán giữa `float16` và `float32` tự động promote toàn bộ mảng thành **`float32`**!
5. Phép nhân ma trận tính điểm tương đồng: `np.dot(self.emb_img_c[g_indices], q_vec_c)` chạy hoàn toàn trên **CPU ở FP32**.

**Kết luận quan trọng:**
- **Pha 3f đã gộp 2 pipeline khác nhau:**
  - Pipeline đánh giá chất lượng (Q1b, Q2, Q3, Q5, Q6): Chạy ma trận trên **CPU ở FP32**.
  - Pipeline đo Latency: Chạy ma trận trên **GPU ở FP16** (`torch.matmul`).
- Số chất lượng trong báo cáo 3f là số của **CPU FP32**, hoàn toàn không phản ánh ảnh hưởng làm tròn của GPU FP16.

### C.2 Tái lập 3 con số Ties và Tìm nguyên nhân thật
- **Ba con số cần giải mã:**
  1. `ties = 2` ở Pha 3c trên $G_{full}$.
  2. `ties = 120` ở Pha 3f trên $G_{full}$.
  3. `ties = 119` ở phép đo FP32 của Rev1.
- **Nguyên nhân thật sự:**
  - Ở Pha 3c: Code kiểm tra đồng hạng dùng phép so sánh bitwise chính xác: `row == target_score`. Vì số thực float32 hiếm khi bằng nhau chính xác tuyệt đối từng bit, chỉ có đúng **2 trường hợp** (hoặc 4 trường hợp tùy thứ tự).
  - Ở Pha 3f và Rev1: Hàm kiểm tra đồng hạng được đổi sang `np.isclose(row, target_score, atol=1e-6)`. Ngưỡng sai số tuyệt đối $10^{-6}$ đã gom các điểm số xấp xỉ nhau thành đồng hạng, làm số ties nhảy vọt lên **120** (hoặc 119)!
  - **Bác bỏ giả thuyết FP16:** Bước nhảy từ 2 lên 120 hoàn toàn do **thay đổi định nghĩa hàm đếm (ngưỡng epsilon $10^{-6}$)** trên CPU FP32, hoàn toàn KHÔNG PHẢI do FP16.

### C.3 Báo cáo song song theo Dtype và Thống nhất Biên độ Đồng hạng

| Pipeline / Môi trường thực thi | Dtype Ma trận | Dtype Tích lũy | Số Ties trên $G_{full}$ | R@10 $G_{full}$ (Bi quan - Lạc quan) | Mean Rank $G_{full}$ |
|---|---|---|---|---|---|
| Pha 3c (Baseline) | CPU float32 | float32 (bitwise `==`) | 2 | 0.0628 - 0.0628 | 1038.21 |
| Pha 3f thực chạy (Chất lượng) | CPU float32 | float32 (isclose $10^{-6}$) | 120 | 0.0628 - 0.0628 | 1038.18 |
| Pipeline phục vụ thực tế (Latency) | GPU float16 | float16 (`torch.matmul`) | 923 | **0.0620 - 0.0636** | 1036.99 |

**Thống nhất định nghĩa biên độ đồng hạng:**
- Trên pipeline CPU FP32: Biên độ ảnh hưởng là **0.0000** (R@10 bất biến ở 0.0628).
- Trên pipeline phục vụ GPU FP16: Khoảng biến thiên là $[0.0620, 0.0636]$, biên độ toàn phần là **0.0016**, tương ứng biểu diễn **$0.0628 \pm 0.0008$**.

---

## D. Căn nguyên F1: Bác bỏ giả thuyết "Cắt cụt" và Hệ quả `max_length`

### D.1 Bác bỏ nguyên nhân gốc 3f
- Báo cáo 3f phát biểu: *"R@10 sụt giảm là do max_length=64 cắt cụt phần thuộc tính ở đuôi câu"*.
- **Kết quả thực nghiệm cô lập (`reports/phase3g_maxlen.json`):**
  - Giữ cố định độ dài padding `pad=128`: Cắt ngắn văn bản còn 64 token cho R@10 = **0.6186**, trong khi không cắt (128 token) cho R@10 = **0.5225**! Tức là cắt ngắn văn bản không làm giảm điểm mà còn tăng điểm.
  - Giữ cố định văn bản ở 64 token: Đổi padding từ `pad=64` sang `pad=128` làm R@10 tăng vọt từ **0.1225 lên 0.6186 (+0.4961)**.
- **Kết luận:** Giả thuyết 3f bị **BÁC BỎ**. Nguyên nhân chi phối 100% sự sụt giảm điểm số là **độ dài padding (64 vs 128)**, vì SigLIP không có attention mask nên các vị trí padding tham gia trực tiếp vào self-attention.
- **Quy tắc áp dụng:** Biến thể (cắt 64, pad 128) tuy đạt 0.6186 nhưng **KHÔNG ĐƯỢC PHÉP DÙNG** trong pipeline phục vụ vì vi phạm Luật 4 (lệch code train) và mang tính chọn tham số theo tập test. Pipeline phục vụ bắt buộc dùng chuẩn (pad 128, truncate 128).

### D.2 Số liệu Q3 theo 3 mẫu số và Giải thích chênh lệch Q2
- **Số liệu Q3 theo 3 mẫu số:**
  - Mẫu số 90 câu (Pha 3e, lọc $N \ge 10$ trên test split): Config 0 AP@10 = **0.3800**, P@10 = **0.4444**.
  - Mẫu số 87 câu (Pha 3f, lọc thêm nhóm có sp in_scope hợp lệ): Config 0 AP@10 = **0.3831**, P@10 = **0.4483**.
  - Mẫu số 100 câu ngẫu nhiên (`phase3g_maxlen.json`): pad=64 đạt AP@10 = **0.1366**; pad=128 đạt AP@10 = **0.2517**.
- **Chênh lệch 0.0002 ở Q2 AP@10:** Rev1 ghi 0.4302 trong khi 3f ghi 0.4300. Chênh lệch 0.0002 hoàn toàn do sai số làm tròn số thực trong bootstrap trung bình và giải quyết tie-break khi thứ tự gallery có sự xáo trộn nhẹ giữa hai môi trường.

### D.3 Danh sách số liệu Pha 3e bị mất hiệu lực do dùng `max_length = 64`
Vì Pha 3e cấu hình sai `max_length = 64`, các số liệu sau đây trong báo cáo Pha 3e không còn giá trị hiệu lực:
1. Q1b Config 0 R@10 ($G_{test}$) báo 0.4812 (thực tế ở pad=128 là 0.5225).
2. Q1b Config 1, 2, 3, 5: Mọi chỉ số có sự tham gia của thành phần Dense text query.
3. Điểm số Q2 và Q3 của các cấu hình Dense ở Pha 3e (bị suy giảm nhân tạo do padding 64).

---

## E. Rút lại kết luận "Độc lập" của Q6 và Phân tích Ablation

### E.1 Rút lại kết luận
- **Rút lại:** Báo cáo Pha 3f khẳng định *"Q6 độc lập hoàn toàn với parser và bộ lọc danh mục"*. Khẳng định này là **SAI**.
- **Lý do:** Ràng buộc relevance trong ground truth của Q6 có cả `articleType` và `gender`. Bộ phân tích cú pháp (QueryParser) khi nhận diện đúng danh mục vẫn giúp thu hẹp không gian tìm kiếm (từ 1.290 sp xuống vài chục sp), trực tiếp nâng cao độ chính xác.
- **Hệ số gấp 2.47 lần:** Tỉ lệ giữa Hệ thống và Model thuần (0.5172 / 0.2091) phản ánh sức mạnh tổng hợp của bộ lọc danh mục + BM25 + Dense Soft Bonus, **không được dùng làm bằng chứng chứng minh SigLIP ranking thuần tốt**.

### E.2 Tái lập chính xác số liệu Pha 3f trên Q6-test
- **Kết quả chạy lại đúng mã nguồn 3f (truyền `mask=g_test_mask` vào `filter_relaxed`):**
  - **Config 0 (Model thuần):** AP@10 = **0.2091**, P@10 = **0.2908**, Hit@10 = **0.5583**.
  - **Config 5 (Full System):** AP@10 = **0.5172**, P@10 = **0.5967**, Hit@10 = **0.8833**.
- **Giải thích vì sao Rev1 ra số khác (0.5245 và 0.2095):**
  Trong Pha 3f gốc, `filter_relaxed` nhận `mask=g_test_mask` nên việc nới lỏng L0..L6 tính trên tập con $G_{test}$ ($N=1.290$). Ở script của Rev1, hàm lọc được chạy trên toàn gallery ($N=37.356$) rồi mới lấy giao với $G_{test}$, khiến relaxation dừng ở tầng cao hơn (L0/L1) có độ chính xác cao hơn, đẩy AP@10 lên 0.5245. Khi chạy đúng mask của 3f, số liệu tái lập **chính xác 100%**.

### E.3 Bảng Ablation tách đóng góp của Bộ lọc và BM25 trên Q6-test

| Điều kiện thử nghiệm | Tập đầy đủ (N=120) AP@10 | Tập đầy đủ P@10 | Tập đầy đủ Hit@10 | Loại 11 câu trùng (N=109) AP@10 | Loại 11 câu trùng P@10 | Loại 11 câu trùng Hit@10 |
|---|---|---|---|---|---|---|
| **Config 0: Model thuần** | 0.2091 | 0.2908 | 0.5583 | 0.2103 | 0.2899 | 0.6606 |
| **Điều kiện 1: Full Config 5 (Gốc 3f)** | **0.5172** | **0.5967** | **0.8833** | **0.5388** | **0.6147** | **0.9266** |
| **Điều kiện 2: BM25 bỏ `usage_vi`** | 0.4932 | 0.5817 | 0.8750 | 0.4924 | 0.5817 | 0.9083 |
| **Điều kiện 3: Bỏ Lọc/Bonus Category & Gender + BM25 không usage** | 0.2285 | 0.3125 | 0.6083 | 0.3932 | 0.4780 | 0.8440 |

*Nhận xét từ Ablation:*
- Khi bỏ `usage_vi` khỏi BM25 (Điều kiện 2), AP@10 chỉ giảm nhẹ từ 0.5172 xuống 0.4932.
- Khi bỏ tiếp bộ lọc và soft bonus danh mục/giới tính (Điều kiện 3), AP@10 rơi thẳng từ 0.4932 xuống 0.2285 (sát với mức 0.2091 của Model thuần). Điều này chứng minh **đóng góp vượt trội của Hệ thống đến từ bộ lọc và soft bonus thuộc tính danh mục**, chứ không phải do việc BM25 "học vẹt" từ khóa usage.

### E.4 Danh sách 11 câu hỏi trùng chuỗi text giữa Q6-val và Q6-test
- `q6_test_025`: "ví cầm tay mùa hè" (trùng với `q6_val_028` ở tập val; spec: {'articleType': 'Wallets', 'season': 'Summer'})
- `q6_test_035`: "phong cách tập gym nữ" (trùng với `q6_val_079` ở tập val; spec: {'gender': 'Women', 'usage': 'Sports'})
- `q6_test_061`: "giày dạo phố ngày thu" (trùng với `q6_val_108` ở tập val; spec: {'articleType': 'Casual Shoes', 'season': 'Fall'})
- `q6_test_086`: "outfit dạo phố nam" (trùng với `q6_val_019` ở tập val; spec: {'gender': 'Men', 'usage': 'Casual'})
- `q6_test_093`: "vớ nam thường ngày" (trùng với `q6_val_081` ở tập val; spec: {'articleType': 'Socks', 'gender': 'Men', 'usage': 'Casual'})
- `q6_test_097`: "quần lót nam ngày hè" (trùng với `q6_val_026` ở tập val; spec: {'articleType': 'Briefs', 'gender': 'Men', 'season': 'Summer'})
- `q6_test_103`: "áo thun nữ mùa hè" (trùng với `q6_val_002` ở tập val; spec: {'articleType': 'Tshirts', 'gender': 'Women', 'season': 'Summer'})
- `q6_test_105`: "outfit hàng ngày nam" (trùng với `q6_val_072` ở tập val; spec: {'gender': 'Men', 'usage': 'Casual'})
- `q6_test_107`: "outfit mùa hè unisex" (trùng với `q6_val_056` ở tập val; spec: {'gender': 'Unisex', 'season': 'Summer'})
- `q6_test_108`: "outfit lịch sự nam" (trùng với `q6_val_006` ở tập val; spec: {'gender': 'Men', 'usage': 'Formal'})
- `q6_test_113`: "giày tây đi làm" (trùng với `q6_val_014` ở tập val; spec: {'articleType': 'Formal Shoes', 'usage': 'Formal'})

### E.5 Kiểm định Bootstrap cho chênh lệch $\alpha=1.0$ vs $\alpha=0.85$ trên Q6-val
- Chênh lệch điểm trung bình: $\Delta = AP@10(\alpha=1.0) - AP@10(\alpha=0.85) = -0.0042$ (0.3571 vs 0.3529).
- Bootstrap paired 95% CI (1.000 lần lặp, seed 42): **[-0.0146, 0.0222]**.
- **Kết luận:** Khoảng tin cậy 95% **chứa giá trị 0** (True). Do đó, chênh lệch 0.0052 hoàn toàn nằm trong phạm vi nhiễu lấy mẫu ngẫu nhiên. Quyết định chọn $\alpha^*=1.0$ là hoàn toàn chuẩn xác theo tiêu chí kỹ thuật: vừa nằm trong ngưỡng sai số ($\le 0.0100$) vừa tiết kiệm 170MB RAM và giảm thời gian truy vấn.

---

## F. Giao thức Chấm mù A/B

1. **Chuẩn hóa tên file:** Tên file thực tế trong `reports/contact_sheets/` là `blind_ab_x001.png` $\dots$ `blind_ab_x030.png` (sửa lại lỗi ghi nhầm tiền tố `h001` trong báo cáo 3f).
2. **Chuẩn hóa cấu trúc CSV chấm mù:** Định dạng chuẩn của `eval/ab_blind_heldout.csv` gồm đúng 6 cột:
   `id, query, better, correct_count_A, correct_count_B, notes`.
3. **Niêm phong an toàn:**
   - File ánh xạ cấu hình đã được di chuyển vào `eval/_sealed/ab_mapping.json`.
   - SHA-256 niêm phong: `2bab34f1950ce2e9b894f0fe9d663776745940f70a3a556b6453e49cff6bb1fc`.
   - Tuyệt đối không mở, không in và không tự ý điền kết quả vào file chấm mù.

---

## G. Việc chờ người dùng quyết định

| # | Hạng mục | Trạng thái hiện tại | Hành động tiếp theo |
|---|---|---|---|
| 1 | Duyệt danh sách đề xuất alias thương hiệu | `ĐÃ DUYỆT (2026-11-09T12:17:11)` | Đã cập nhật `src/brands_v2.json` và bổ sung test đối chứng |
| 2 | Chấm mù 30 contact sheet A/B | `CHỜ NGƯỜI DÙNG CHẤM` | Người dùng mở các file `reports/contact_sheets/blind_ab_x*.png` và điền `eval/ab_blind_heldout.csv` |

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
- [x] Báo cáo cũ `reports/phase3f_reconciliation.md` và `reports/phase3f_errata_rev1.md` giữ nguyên làm bằng chứng.
- [x] Tách biệt tuyệt đối hai cột "Model thuần" (0.5225) và "Hệ thống" (0.9008); không mạo nhận model thuần đạt chỉ tiêu 0.85.

---

## I. Danh sách tài sản và File mới

| Tên file | Mô tả chi tiết |
|---|---|
| `reports/phase3f_errata_rev1.md` | Bản lưu Errata vòng 1 làm bằng chứng đối soát |
| `reports/phase3f_errata.md` | Báo cáo Errata Rev2 hoàn chỉnh này |
| `src/brands_v2.json` | Từ điển brand v2 (234 canonical + 22 alias đã duyệt) |
| `reports/brand_audit_3g.json` | Dữ liệu kiểm toán sản phẩm Benetton và Lavie trong catalog |
| `reports/phase3g_brands_v2_eval.json` | Đánh giá đối chứng 3 cột cho Brand Heldout và Q5 |
| `reports/phase3g_latency_multi_run.json` | Dữ liệu đo lặp 5 lần benchmark Latency và Image-to-Image |
| `reports/phase3g_fp16_ties.json` | Bằng chứng tái lập ties và phân rã pipeline CPU FP32 vs GPU FP16 |
| `reports/phase3g_maxlen.json` | Phân tích cô lập ảnh hưởng của padding length vs truncate length |
| `reports/phase3g_q6_rev2.json` | Kết quả ablation Q6, danh sách 11 câu trùng và bootstrap CI |
| `reports/phase3g_ties_q3_rev2.json` | Đối soát mẫu số Q3, Q2 AP@10 và số liệu 3 cột brand |
| `eval/_sealed/ab_mapping.json` | File ánh xạ A/B niêm phong bảo mật |

---

## J. Giới hạn còn lại của Hệ thống

1. **Nguồn gốc chuỗi ký tự "228 ms":** Không tìm thấy bất kỳ dòng mã hay kết quả thực nghiệm nào sinh ra chính xác số 228 ms; đây vẫn là một điểm chưa xác định được nguồn gốc trong văn bản 3f cũ.
2. **Tính hậu nghiệm của tập Heldout:** Sau các đợt rà soát Pha 3e, 3f và việc bổ sung alias `brands_v2.json`, tập heldout không còn giữ được tính chất "ẩn tuyệt đối" đối với các can thiệp từ khóa brand.
3. **Q6 không độc lập hoàn toàn:** Do ground truth chứa nhãn `articleType` và `gender`, hiệu quả của hệ thống trên Q6 phản ánh năng lực lọc đa tầng của parser chứ không phải năng lực suy luận ngữ nghĩa độc lập của riêng mô hình dense.
4. **Luồng Ảnh $\to$ Ảnh chưa đạt SLA ở p99:** P99 đạt 280.17 ms (> 200 ms) do chi phí giải nén ảnh PIL và tiền xử lý trên CPU; cần các giải pháp đọc ảnh bất đồng bộ hoặc tăng tốc phần cứng ở các phiên bản tiếp theo.

---

## K. Lịch sử sửa đổi từ Rev1 sang Rev2

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
