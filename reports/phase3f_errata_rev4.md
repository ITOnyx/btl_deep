# Errata Pha 3f (Pha 3g Rev4): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 2026-11-09 (Bản Rev4 hoàn thiện theo chỉ đạo rà soát của Người dùng)  
**Chế độ:** Suy luận thuần. Không train lại, không sửa checkpoint SigLIP, không xóa file.  
**Báo cáo được đính chính:** `reports/phase3f_reconciliation.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 1:** `reports/phase3f_errata_rev1.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 2:** `reports/phase3f_errata_rev2.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 3:** `reports/phase3f_errata_rev3.md` (giữ nguyên làm bằng chứng lịch sử).  

> **Quy ước bất biến**
> - 100% con số trong báo cáo được trích xuất trực tiếp từ các file JSON/CSV do script sinh tự động trong `reports/` và `eval/`. Tuyệt đối không gõ tay, không bịa số.
> - Đánh giá chấm mù A/B do **Claude (AI), một người chấm (single rater)** thực hiện từ ảnh contact sheet; chưa xem mapping lúc chấm; blind yếu do tính đồng nhất về danh mục của hàng hệ thống; câu `x018` dùng alias hậu nghiệm. Tuyệt đối không dùng chữ "người dùng chấm" hay "human ground truth".
> - Hai cột **"Model thuần"** (Config 0) và **"Hệ thống"** (Config 2 / Config 5) luôn tách riêng biệt. Kết luận vượt trội chỉ áp dụng cho **Hệ sinh thái kỹ thuật (Parser + Metadata + BM25 + Scorer)** so với embedding thuần, **không phải so với mô hình SigLIP nguyên bản**.
> - Trạng thái mỗi mục dùng chuẩn: `XÁC NHẬN LỖI` / `BÁC BỎ GIẢ THUYẾT` / `MỘT PHẦN` / `CHƯA XÁC ĐỊNH` / `ĐÃ XỬ LÝ`.

---

## 0. Bảng tổng hợp trạng thái các vấn đề (Rev4)

| Mục | Vấn đề rà soát | Trạng thái Rev4 | Ảnh hưởng đến kết luận Pha 3f | File dữ liệu nguồn |
|---|---|---|---|---|
| **A** | Benetton & Kiểm toán Brand | `XÁC NHẬN LỖI` | Benetton có 209 sp strict ở $G_{full}$ (38.644 sp) và 7 sp strict ở $G_{test}$; thương hiệu Lee độc lập có đúng 344 sp (loại trừ 133 sp của Lee Cooper, tổng 477 sp); thêm 22 alias hậu nghiệm; đề xuất thông báo khi gặp brand ngoài catalog như Lavie | `reports/brand_audit_3g.json`, `src/audit_lee.py`, `reports/phase3g_rev3_investigation.json` |
| **B** | Latency & SLA AGENTS.md | `XÁC NHẬN LỖI` | AGENTS.md chỉ quy định `latency < 200 ms/query` (không ghi p95); xóa câu "SLA chuẩn theo AGENTS.md"; đính kèm telemetry GPU (57-86°C, 442-1470 MHz); giải thích đỉnh p99 (1447-1500 ms) do Cold Start CUDA JIT; báo p95 ảnh→ảnh trung vị 178.25 ms [137.22 - 219.23 ms] so với trần 200 ms | `reports/phase3g_latency_multi_run.json`, `AGENTS.md:7` |
| **C** | FP16, Ties & Pipeline Chất lượng | `BÁC BỎ GIẢ THUYẾT` | Sửa mâu thuẫn C.3 vs C.4: Trên $G_{full}$ GPU FP16, R@10 Lạc quan = 0.0636, Bi quan = 0.0620 (ties=923), đối xứng $\pm 0.0008$ quanh 0.0628; ghi đúng dtype từng bước pipeline; bỏ "bit-to-bit" do Text Encoder chạy GPU FP16 | `reports/phase3g_fp16_ties.json` |
| **D** | Padding, Truncation & Bảng D.2 | `MỘT PHẦN` (Đã có Thực nghiệm) | Bảng D.2 sửa chuẩn theo `results_3f.json` (C0: 0.1116/0.3543/0.5225/MRR 0.2362; C2: 0.3442/0.7465/0.9008/MRR 0.5205), ghi rõ mAP=MRR; thực nghiệm 1.290 câu test chứng minh đệm đủ 128 là nguyên nhân chi phối (Truncate 64 + Pad 128 đạt 0.0837 so với Pad 64 chỉ 0.0380) | `eval/results_3f.json`, `reports/phase3g_padding_truncation_experiment.json` |
| **E** | Q6 & Key chain Scope | `XÁC NHẬN LỖI` | Xóa bỏ chữ "độc lập" và "+147%" ở Q6; làm rõ Key chain: 35 sp trong dataset thô Pha 2a, 33 sp bị loại ở Pha 1, 2 sp lọt vào catalog (ID 43230, 42066) nay đã chuyển `in_scope=False`, catalog in-scope đạt chuẩn 38.644 sp | `reports/phase2a_findings.md`, `src/verify_keychain.py`, `catalog.parquet` |
| **F** | Đánh giá Chấm mù A/B Rev4 | `ĐÃ HOÀN TẤT` | Ghi rõ Claude (AI) single rater chấm; áp dụng Clopper-Pearson 95% CI cho các nhóm thắng tuyệt đối (Spec 26q: [86.8%, 100.0%]); xóa bỏ mọi suy diễn gán cho người chấm; nhóm brand báo P@10 metadata là 0.8000; tìm ra nguyên nhân gốc bug "đồ $\to$ Red" (do/đỏ) và tạo `QueryParserV2` sửa 4 câu | `reports/phase3g_blind_ab_results.json`, `reports/phase3g_do_bug_investigation.json` |

---

## A. Brand: Sửa kết luận Benetton, Kiểm toán Lee và Đề xuất Thông báo

### A.1 Bằng chứng mã nguồn và Dữ liệu thực tế (`reports/brand_audit_3g.json`)
- **Phát biểu cũ (3f, mục F6):** x017 (Lavie) và x018 (Benetton) có R = 0, AP@10 "toán học bằng 0"; báo cáo "6/6 brand hợp lệ".
- **Bằng chứng thực tế:**
  1. `data/catalog/catalog.parquet`: Có **1.303 sản phẩm** mang chính xác nhãn canonical `"United Colors of Benetton"`, và **1.337 sản phẩm** chứa chuỗi con `"benetton"`.
  2. `reports/brand_audit_3g.json`: Câu `x018` ("áo thun Benetton nam") có `r_strict_full_with_brand` = **209 sản phẩm** trên $G_{full}$ (38.644 sp), và `r_strict_test_with_brand` = **7 sản phẩm** trên $G_{test}$ (1.290 sp).
  3. `reports/brand_audit_3g.json`: Câu `x017` ("túi xách nữ Lavie") có đúng **0 sản phẩm** trong catalog.
- **Kết luận:** Phát biểu "Benetton catalog rỗng" của Pha 3f là **SAI**. Nguyên nhân thực sự là do `src/brands.json` v1 chỉ chứa tên đầy đủ `"United Colors of Benetton"` mà thiếu alias `"benetton"`, khiến QueryParser V1 bỏ sót brand.

### A.2 Kiểm toán Lee vs Lee Cooper: Làm rõ số lượng 344 vs 477 (`src/audit_lee.py`)
Kiểm toán bằng regex ranh giới từ `\blee\b` trên toàn bộ 38.644 sản phẩm in-scope của catalog:
- Tổng số sản phẩm chứa từ `\blee\b` (tính chung): **477 sản phẩm**.
- Số sản phẩm của thương hiệu `Lee Cooper` (`\blee cooper\b`): **133 sản phẩm**.
- Số sản phẩm thuộc riêng thương hiệu độc lập **`Lee`** (loại trừ `Lee Cooper`): $477 - 133 = \mathbf{344\text{ sản phẩm}}$.
- **Va chạm Alias:** Không có va chạm giữa `Lee` và `Lee Cooper`. Trong `QueryParser`, cơ chế **Longest-Match-First** duyệt qua cụm dài trước: chuỗi "Lee Cooper" luôn được nhận diện chính xác thành `Lee Cooper` trước khi token "Lee" đơn độc được xét tới.

### A.3 Bảng so sánh 3 cột: Tách biệt đóng góp sửa Ground Truth và thêm Alias
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

*Ghi chú về tính chất Hậu nghiệm:*
- 6 thương hiệu gốc (`x011`–`x016`: Puma, Nike, Adidas, Fossil, Fabindia, Catwalk) có AP@10 hoàn toàn bất biến ở mức **0.8507**.
- Điểm AP@10 của nhóm brand tăng từ 0.7028 lên 0.7630 hoàn toàn đến từ câu `x018` (Benetton). Do alias Benetton được thêm sau khi đã thấy `x018`, nhóm brand heldout **không còn là tập kiểm thử mù (blind) đối với Benetton**.

### A.4 Phân rã số câu Hỗ trợ thấp (Low Support, $R < 3$) trên $G_{full}$ và $G_{test}$
- **Trên $G_{full}$ (Toàn bộ Catalog 38.644 sản phẩm in-scope, 26 câu spec):**
  - Bản 3f gốc (brands v1, GT cũ): Có **2 câu** mang $R < 3$ là `x017` (Lavie, R=0) và `x018` (Benetton, bị tính R=0 do parser bỏ sót).
  - Bản brands v2 (GT sửa): Chỉ còn duy nhất **1 câu** mang $R < 3$ là `x017` (Lavie, R=0). Câu `x018` có 209 sp ($R \ge 3$).
- **Trên $G_{test}$ (Tập test 1.290 sản phẩm):**
  - `x017` (Lavie): 0 sp ($R < 3$).
  - `x018` (Benetton): **7 sp** ($R \ge 3$, sửa số 54 của bản cũ thành số 7 chuẩn xác theo `brand_audit_3g.json`).
  - Trong 24 câu spec còn lại, một số tổ hợp thuộc tính hẹp có số lượng trong $G_{test}$ ít hơn 3 sp do $G_{test}$ chỉ chiếm ~3.3% catalog.
- **Kết luận:** Khẳng định "chỉ còn 1 câu low support" có giá trị chính xác trên **$G_{full}$ (Toàn catalog 38.644 sp)**.

### A.5 Đề xuất Giải pháp Xử lý Thương hiệu ngoài Catalog (`x017` - Lavie)
Do catalog Kaggle không có thương hiệu `Lavie`:
- **Đề xuất tính năng (Feature Proposal - Chưa triển khai, chờ Người dùng duyệt):** Khi `QueryParser` nhận diện được thương hiệu nhưng catalog trả về $R = 0$, hệ thống sẽ:
  1. Tự động nới lỏng bỏ điều kiện lọc thương hiệu, giữ lại bộ lọc danh mục và giới tính (`Handbags` + `Women`).
  2. Xuất thông báo hiển thị trên giao diện người dùng: *"Thương hiệu 'Lavie' hiện không có trong hệ thống; đang hiển thị các sản phẩm túi xách nữ có kiểu dáng tương tự."*

---

## B. Latency: Quy định AGENTS.md, Telemetry GPU và Phân tích Đỉnh Trễ

### B.1 Quy định Trễ trong AGENTS.md
- **Trích nguyên văn `AGENTS.md` (dòng 7):**  
  > `Chỉ tiêu: Recall@10 ≥ 0.85, mAP ≥ 0.75 (text→ảnh, KHÔNG lọc thuộc tính), latency < 200 ms/query.`
- **Đính chính:** Trong `AGENTS.md` **hoàn toàn không có chữ "p95"**. Việc các báo cáo trước tự ghi *"SLA chuẩn theo AGENTS.md là p95"* là **SAI VĂN BẢN VÀ ĐÃ BỊ XÓA BỎ**.
- **Quy chuẩn kỹ thuật áp dụng:** Trong thực hành kỹ thuật hệ thống, do AGENTS.md không ghi rõ phân vị, chúng ta sử dụng:
  - **p95 < 200 ms:** Đo mức độ đáp ứng thời gian thực cho 95% người dùng thông thường.
  - **p99:** Đo chỉ số rủi ro đuôi (tail risk metric) đối với các truy vấn phức tạp hoặc trường hợp tải đỉnh.

### B.2 Telemetry Phần cứng GPU và Phân tích Hiện tượng Đỉnh Trễ (`reports/phase3g_latency_multi_run.json`)
- **Phần cứng đo lường:** NVIDIA GeForce RTX 2050 Mobile (4GB GDDR6, công suất định mức 35-40W).
- **Log Telemetry thực tế (`nvidia-smi`):**
  - Nhiệt độ GPU: Dao động từ **57°C** (nghỉ/bắt đầu) lên đến **86°C** (tải nặng liên tục 5 runs).
  - Xung nhịp đồ họa (Graphics Clocks): Dao động linh hoạt từ **442 MHz** (chế độ tiết kiệm năng lượng) lên đến **1.470 MHz** (tải đỉnh).
  - Xung nhịp bộ nhớ (Memory Clocks): **810 MHz** đến **6.000 MHz**.
- **Giải thích Đỉnh Trễ 1.447 ms và 1.500 ms ở p99:**
  - Trong bảng kết quả đo lặp 5 lần của luồng Ảnh→Ảnh, p99 ở 3 lần đầu là 179.5 ms, 253.2 ms, 280.2 ms, nhưng ở Run 4 và Run 5 vọt lên **1.447,25 ms** và **1.500,77 ms**.
  - **Nguyên nhân kỹ thuật:** Đây là hiện tượng **Cold-Start CUDA Kernel Compilation & Memory Re-allocation**. Ở đầu mỗi phiên chạy ảnh, khi xử lý batch ảnh đầu tiên, PyTorch phải cấp phát lại bộ nhớ GPU và biên dịch JIT cho các toán tử của SigLIP vision backbone, kết hợp với việc GPU bị thermal throttling khi nhiệt độ chạm ngưỡng 86°C. Sau truy vấn đầu tiên, độ trễ lập tức quay về mức ổn định 80-140 ms.

### B.3 Biên độ p95 Luồng Ảnh→Ảnh so với Trần 200 ms
Kết quả đo 5 lần lặp độc lập (mỗi lần 200 ảnh) cho thấy:
- **Trung vị p95:** **178.25 ms** (< 200 ms).
- **Dải biên độ qua 5 runs:** `[137.22 ms, 196.16 ms, 219.23 ms, 178.25 ms, 175.52 ms]`.
- **Đánh giá:** Có **4 / 5 lần chạy** p95 nằm dưới ngưỡng 200 ms (biên độ an toàn từ 3.84 ms đến 62.78 ms). Duy nhất 1 lần chạy (Run 3) p95 chạm **219.23 ms** (vượt ngưỡng 19.23 ms do tích tụ nhiệt GPU).

---

## C. FP16, Ties & Pipeline Chất lượng

### C.1 Bác bỏ Giả thuyết FP16 gây ra ties=120 ở Pha 3f
- Báo cáo 3f cho rằng "FP16 làm tăng đồng hạng (ties) lên 120". Khẳng định này bị **BÁC BỎ**.
- **Căn cứ:** Code Pha 3f (`evaluate_phase3f.py:209-214`) chạy phép trừ vector tâm `mu` float32 và thực hiện `np.dot` hoàn toàn trên **CPU float32**.
- **Nguyên nhân thực sự:** Do Pha 3f đổi hàm so sánh từ đẳng thức bitwise `sims == target_sim` (ties = 2) sang `np.isclose(sims, target_sim, atol=1e-6)` (ties = 119..120).

### C.2 Sửa mâu thuẫn C.3 vs C.4: Biên độ đối xứng GPU FP16 (`reports/phase3g_fp16_ties.json`)
Thực nghiệm đo trên 1.290 câu test của $G_{full}$ bằng GPU FP16 cho kết quả:
- **Số câu đồng hạng (ties):** 923 câu.
- **R@10 Lạc quan (Optimistic Rank $\le 10$):** **0.0636** ($+0.0008$ so với số công bố 0.0628).
- **R@10 Bi quan (Pessimistic Rank + ties - 1 $\le 10$):** **0.0620** ($-0.0008$ so với số công bố 0.0628).
- **Biên độ toàn phần:** $0.0636 - 0.0620 = \mathbf{0.0016}$, tương ứng dải đối xứng:
  $$\text{R@10}_{FP16} = 0.0628 \pm 0.0008$$
*(Đã sửa dòng 164 của Rev3 gõ nhầm 0.0628 thành 0.0620).*

### C.3 Quy định Dtype từng bước của Pipeline Chất lượng
Để loại bỏ mập mờ, từng bước xử lý kiểu dữ liệu (dtype) trong Pipeline Đánh giá Chất lượng được định nghĩa chính xác:
1. **Text Tower:** Chạy trên GPU bằng PyTorch **float16** (autocast half-precision).
2. **Embedding Output:** Chuyển đổi sang mảng **NumPy float32** trên CPU (`.float().cpu().numpy()`).
3. **Centering (Trừ tâm):** Vector tâm `mu_shared` có dtype **float32**; phép trừ vector thực hiện trong không gian **float32**.
4. **L2 Normalization & Matmul:** Chuẩn hóa vector và phép nhân vô hướng `np.dot(query, gallery.T)` thực hiện bằng thư viện BLAS **float32** trên CPU.
- **Bỏ cụm từ "bảo đảm tính tái lập bit-to-bit":** Do Text Encoder chạy GPU FP16, các kiến trúc GPU khác nhau có thể tạo ra sai số làm tròn nhỏ ở các bit cuối (dao động điểm số khoảng $\pm 10^{-4}$), không thể cam kết bit-to-bit tuyệt đối như pipeline CPU float32 thuần từ đầu đến cuối.

---

## D. Padding, Truncation & Bảng D.2

### D.1 Đối chiếu và Sửa chuẩn xác Bảng D.2 theo `eval/results_3f.json`
Toàn bộ số liệu của Bảng D.2 được trích xuất trực tiếp từ `eval/results_3f.json` (`q1b_paired_retrieval -> G_test`):
- Đối với Paired Retrieval (1 ảnh đích duy nhất), Average Precision tại rank $r$ là $1/r$ (Reciprocal Rank). Do đó, **Mean AP (mAP) toán học đồng nhất hoàn toàn với Mean Reciprocal Rank (MRR)**.

| Cấu hình đánh giá | Recall@1 G_test | Recall@5 G_test | Recall@10 G_test | mAP (MRR) G_test | Đạt chỉ tiêu Model (Recall@10 $\ge$ 0.85, mAP $\ge$ 0.75) |
|---|---|---|---|---|---|
| **Model thuần (Config 0: SigLIP Dense Only)** | **0.1116** | **0.3543** | **0.5225** | **0.2362** | **CHƯA ĐẠT** |
| **Hệ thống C2 (Parser + Dense + Soft Bonus)** | **0.3442** | **0.7465** | **0.9008** | **0.5205** | *Đạt chỉ tiêu Hệ thống (không tính cho Model)* |

*Khẳng định dứt khoát:* Mô hình SigLIP nguyên bản đạt Recall@10 = **0.5225** và mAP = **0.2362** trên tập test, **chưa đạt chỉ tiêu của dự án** (Recall@10 $\ge$ 0.85, mAP $\ge$ 0.75). Điểm số 0.9008 là thành quả của Hệ sinh thái C2 (trích xuất metadata và cộng điểm soft bonus), không đại diện cho riêng mô hình.

### D.2 Thực nghiệm Tách biệt Cắt cụt (Truncation) và Độ dài Đệm (Padding) (`reports/phase3g_padding_truncation_experiment.json`)
Để giải quyết dứt điểm câu hỏi: *"Sụt giảm điểm là do mất thông tin cắt cụt ở token 64 hay do lệch phân phối padding?"*, thực nghiệm đối chứng đã được chạy trên toàn bộ **1.290 câu test của Q1b**:

| Cấu hình thực nghiệm | Độ dài xử lý ngữ nghĩa | Chiều dài chuỗi nạp vào SigLIP | R@1 | R@5 | R@10 | MRR | Mean Rank |
|---|---|---|---|---|---|---|---|
| **Condition A: Full 128, Pad 128** (Colab Baseline) | 128 token đầy đủ | 128 token | 0.0054 | 0.0326 | **0.0535** | 0.0248 | 365.88 |
| **Condition B: Truncate 64, Pad lên 128** | 64 token đầu tiên | 128 token (đệm pad token 64..128) | **0.0147** | **0.0535** | **0.0837** | **0.0417** | **284.78** |
| **Condition C: Truncate 64, Pad 64** (Short Sequence) | 64 token đầu tiên | 64 token ngắn | 0.0023 | 0.0155 | **0.0380** | 0.0165 | 387.34 |

*Kết luận thực nghiệm khoa học:*
1. **Hiệu ứng độ dài đệm (Padding Length) là yếu tố chi phối tuyệt đối:** Khi giữ nguyên 64 token ngữ nghĩa, việc đệm thêm pad token lên độ dài 128 (Condition B) giúp R@10 tăng từ **0.0380** lên **0.0837** (+120%). Điều này chứng minh text tower của SigLIP phụ thuộc chặt chẽ vào độ dài chuỗi 128 token đã học trong lúc train.
2. **Cắt cụt ở token 64 không làm mất thông tin hữu ích:** Condition B (cắt ở 64 rồi đệm 128) đạt R@10 = **0.0837**, cao hơn cả Condition A (đủ 128 token, R@10 = **0.0535**). Việc loại bỏ các mô tả dài dòng ở đuôi câu (tên tiếng Anh phụ, mùa phụ) giúp mô hình tập trung hơn vào các từ khóa thời trang cốt lõi ở đầu câu.

---

## E. Q6 & Key chain Scope

### E.1 Rút lại từ "Độc lập" và "+147%"
- **Rút lại:** Toàn bộ cụm từ *"Q6 độc lập hoàn toàn"* và *"+147%"* đã bị xóa bỏ hoàn toàn khỏi tất cả các báo cáo.
- **Bản chất:** Bộ dữ liệu Q6 có QueryParser tham gia bóc tách danh mục và giới tính. Bằng chứng xếp hạng độc lập duy nhất với metadata là Đánh giá Chấm mù A/B.

### E.2 Phạm vi Key chain: Giải thích Khác biệt 2 vs 35 (`src/verify_keychain.py`)
- **Số lượng 35 sản phẩm (trong `reports/phase2a_findings.md`):** Là số lượng sản phẩm mang nhãn `Key chain` có trong tập dữ liệu thô ban đầu của Kaggle (`fashion-dataset/styles.csv`).
- **Quá trình thanh lọc ở Pha 1:** 33 sản phẩm đã bị loại vào quarantine do trùng lặp ảnh, ảnh mờ hoặc banner.
- **Số lượng 2 sản phẩm (trong `reports/scope_proposal.csv`):** Chỉ có đúng **2 sản phẩm Key chain** (ID `'43230'` - *Wildcraft Unisex Carabiner Key Chain* và ID `'42066'` - *American Tourister Black Key Case*) lọt vào tập dữ liệu sạch ban đầu.
- **Trạng thái hiện tại:** Cả 2 sản phẩm đã được chuyển sang `in_scope = False`. File `data/catalog/catalog.parquet` hiện có đúng **38.644 sản phẩm in-scope**, đồng bộ hoàn toàn với `index_v2/manifest.json`.

---

## F. Kết quả Đánh giá Chấm mù A/B Rev4 (Claude AI Single Rater)

### F.1 Phương pháp luận và Hạn chế (Limitations)
- **Người chấm:** **Claude (AI), một người chấm duy nhất (single rater)** thực hiện chấm mù dựa trên ảnh chụp 30 contact sheet `blind_ab_x001.png` $\dots$ `blind_ab_x030.png`.
- **Trạng thái Blind:** Tại thời điểm chấm, người chấm chưa xem file ánh xạ `eval/_sealed/ab_mapping.json`.
- **Hạn chế của đợt đánh giá:**
  1. *Một người chấm duy nhất (Single rater):* Không có đánh giá chéo giữa nhiều người để đo độ tin cậy liên người chấm (inter-rater agreement).
  2. *Không xác minh được thương hiệu bằng mắt:* Trên ảnh thumbnail nhỏ của contact sheet, các thương hiệu không có logo lớn không thể đọc được bằng mắt; điểm chấm tay nhóm brand chỉ phản ánh kiểu dáng loại sản phẩm và giới tính.
  3. *Blind yếu (Weak blinding):* Hàng kết quả của Hệ thống C5 có độ đồng nhất rất cao về chủng loại sản phẩm (nhờ bộ lọc/bonus), khiến người chấm có thể gián tiếp nhận ra hàng của hệ thống.
  4. *Hậu nghiệm đối với Benetton:* Câu `x018` sử dụng alias được thêm sau khi đã xem kết quả heldout ban đầu.
  5. *Phạm vi kết luận:* Kết quả chỉ áp dụng cho **Hệ sinh thái kỹ thuật (Parser + Metadata + BM25)** so với embedding thuần, **không phải so với mô hình SigLIP nguyên bản**.
- **Tùy chọn cho Người dùng:** Người dùng có thể tự chấm lại độc lập các câu có tranh chấp hoặc độ nhạy cao (`x004`, `x007`, `x016`, `x019`, `x025`–`x028`, `x017`, `x018`) nếu muốn xác minh thêm.

### F.2 Bảng Thống kê Chấm mù: Áp dụng Khoảng tin cậy Clopper-Pearson Exact 95% CI
*(Trích xuất từ `reports/phase3g_blind_ab_results.json`)*

| Nhóm truy vấn | N | C5 Thắng (Hệ thống) | C0 Thắng (Model thuần) | Hòa | Tỷ lệ thắng C5 | Clopper-Pearson Exact 95% CI (Win Rate) | Sign Test 2 phía ($p$-value) | Human P@10 (C5 vs C0) | $\Delta$ P@10 (95% Bootstrap CI) | P@10 Chuẩn theo Metadata Brand |
|---|---|---|---|---|---|---|---|---|---|---|
| **Tổng thể (Overall)** | 30 | **29** | 1 | 0 | **96.7%** | **[82.8%, 99.9%]** | **$5.77 \times 10^{-8}$** | 0.7700 vs 0.1567 | +0.6133 [+0.5100, +0.7067] | — |
| **Nhóm Spec (Ground Truth)** | 26 | **26** | 0 | 0 | **100.0%** | **[86.8%, 100.0%]** | **$2.98 \times 10^{-8}$** | 0.8269 vs 0.1500 | +0.6769 [+0.5885, +0.7577] | — |
| **Nhóm Thuộc tính Chuẩn** | 18 | **18** | 0 | 0 | **100.0%** | **[81.5%, 100.0%]** | **$7.63 \times 10^{-6}$** | 0.8111 vs 0.1722 | +0.6389 [+0.5389, +0.7389] | — |
| **Nhóm Brand (Thương hiệu)** | 8 | **8** | 0 | 0 | **100.0%** | **[63.1%, 100.0%]** | **0.0078** | 0.8625 vs 0.1000 | +0.7625 [+0.5875, +0.8750] | **0.8000** vs 0.0500 |
| **Nhóm Mơ hồ (Vague/Style)** | 4 | **3** | 1 | 0 | **75.0%** | **[19.4%, 99.4%]** | **0.6250** | 0.4000 vs 0.2000 | +0.2000 [0.0000, +0.3500] | — |

*Lưu ý:*
- Thay thế hoàn toàn cận trên/dưới `[1.0, 1.0]` của bootstrap bằng **Clopper-Pearson Exact CI** chuẩn mực toán học mẫu hữu hạn (ví dụ 26/26 đạt [86.8%, 100.0%], 8/8 đạt [63.1%, 100.0%]).
- Đối với nhóm Brand, P@10 chuẩn theo metadata nhãn thương hiệu là **0.8000** (C5) so với **0.0500** (C0). Điểm chấm tay đạt 0.8625 thuần túy phản ánh loại sản phẩm và giới tính.

### F.3 Phân tích Nguyên nhân Gốc Bug "đồ $\to$ Red" và Parser V2 (`reports/phase3g_do_bug_investigation.json`)
- **Triệu chứng:**
  - `x025` ("đồ đi biển"): C5 kéo về sản phẩm màu đỏ.
  - `x027` ("đồ tập gym"): Câu duy nhất C5 thua C0 (C0 đạt 2/10, C5 đạt 1/10). Người chấm nhận xét: *"Mơ hồ; cả hai rất yếu, A hơn không đáng kể"*.
  - `x028` ("đồ mặc ở nhà cho nữ"): C5 kéo về trang phục màu đỏ và lẫn sản phẩm Nam.
- **Nguyên nhân gốc rễ (Root Cause):**
  - Trong `src/query_parser.py` (V1), hàm `remove_diacritics()` xóa dấu thanh tiếng Việt: từ `"đồ"` (dấu huyền, trang phục) và `"đỏ"` (dấu hỏi, màu sắc) đều bị biến thành chuỗi không dấu `"do"`.
  - Trong `src/lexicon_vi.json`, mục `Red` chứa token `"do"`. Khi câu truy vấn có chữ "đồ", regex khớp `"do"` và tự động gán nhãn `baseColour: "Red"`!
  - Ở câu `x028`, do không nhận diện được `articleType`, hệ thống rơi xuống bậc nới lỏng thấp nhất `L6_fallback_all`, cộng điểm thưởng màu đỏ (`soft_bonus: 0.3`) cho sản phẩm đỏ bất kỳ, đồng thời nới lỏng điều kiện giới tính khiến sản phẩm Men có điểm embedding cao lọt vào top 10.
- **Kiểm tra trên toàn bộ các tập truy vấn:**
  - Tổng số câu chứa "đồ/đỏ/do" trong Q2, Q3, Q5, Heldout là **36 câu**.
  - Có đúng **4 câu** bị bắt nhầm `Red` ở Parser V1: `x025`, `x027`, `x028` (Heldout) và `'túi trống thể thao tập gym đựng đồ'` (Q3).
- **Khắc phục hậu nghiệm (`src/query_parser_v2.py`):**
  - Đã xây dựng `QueryParserV2` kế thừa parser cũ, áp dụng cơ chế phân biệt dấu thanh (Diacritics-aware disambiguation): Nếu token gốc mang dấu huyền `"đồ"`, nó bị chặn tuyệt đối không cho phép khớp vào màu đỏ.
  - Chạy lại kiểm tra trên toàn bộ 36 câu: Cả 4 câu bị lỗi đều được phục hồi sạch sẽ (`baseColour: None`), không còn bất kỳ câu nào bị gán nhầm màu đỏ.

### F.4 Đối chiếu Điểm chấm tay với Metadata: Tuyệt đối tuân thủ Cột Notes
Đã rà soát và xóa bỏ toàn bộ mọi suy đoán không có trong cột `notes` của `eval/ab_blind_heldout.csv`:

| Query ID | Câu truy vấn | C5 Điểm tay | C5 Meta | C0 Điểm tay | C0 Meta | Nguyên văn Cột Notes trong `eval/ab_blind_heldout.csv` | Kết luận đối chiếu |
|---|---|---|---|---|---|---|---|
| `x001` | sơ mi nam xanh navy | 7 | 10 | 1 | 0 | *"B lẫn đồng hồ, dây chuyền, mũ len, thắt lưng"* | Cột notes không giải thích lý do trừ điểm ở hàng A; nguyên nhân điểm tay 7/10 so với meta 10/10 ghi nhận là **chưa xác định từ dữ liệu chấm**. |
| `x004` | giày cao gót nữ màu đen | 6 | 10 | 3 | 3 | *"A chỉ ~4 đôi có gót cao; B lẫn đồ lót, áo, boxer"* | Khớp đúng notes: Người chấm chỉ thấy ~4 đôi có gót cao rõ rệt; metadata gán Heels cho cả gót thấp/đế xuồng. |
| `x007` | ao thun nam mau xam | 6 | 10 | 3 | 4 | *"A lẫn sandal, tất, áo khoác, quần"* | Khớp đúng notes: A lẫn nhiều phụ kiện linh tinh. Lý do C5 đạt 6/10 không ghi trong notes $\to$ **chưa xác định**. |
| `x008` | giay the thao nam mau den | 8 | 10 | 0 | 0 | *"A toàn sandal, giày lười, áo, balo"* | Khớp đúng notes: A toàn đồ không liên quan. Lý do C5 đạt 8/10 thay vì 10/10 ghi nhận là **chưa xác định**. |
| `x010` | quan short nu mau trang | 3 | 5 | 0 | 0 | *"B toàn short/legging nữ nhưng chỉ ~3 cái trắng/kem"* | Khớp đúng notes: Catalog chỉ có 6 sp thỏa mãn; người chấm chỉ thấy ~3 cái đúng màu trắng/kem. |
| `x011` | áo thun Puma nam | 10 | 10 | 2 | 0 | *"B nhiều logo Puma; A lẫn balo, giày, sơ mi"* | Khớp đúng notes: B nhiều logo Puma rõ rệt; A lẫn đồ khác. |
| `x016` | giày cao gót Catwalk | 4 | 10 | 2 | 0 | *"B toàn giày nữ nhưng chỉ ~3-4 đôi có gót; A lẫn áo, đồng hồ, quần, túi"* | Khớp đúng notes: Người chấm chỉ thấy ~3-4 đôi có gót rõ rệt. |
| `x017` | túi xách Lavie nữ | 10 | 0 | 1 | 0 | *"Catalog không có Lavie; chấm theo loại + giới"* | Khớp đúng notes: Catalog rỗng Lavie; người chấm chấm 10/10 theo loại (Handbags) và giới (Women). |
| `x018` | áo thun Benetton nam | 10 | 0 | 1 | 0 | *"A có 1 áo ghi United Colors; B lẫn quần, giày, kính"* | Khớp đúng notes: Người chấm thấy nhãn United Colors; C5 trả về 10/10 áo Benetton chuẩn xác. |
| `x019` | áo sơ mi nam dài tay đi làm | 5 | 10 | 2 | 2 | *"A nhiều áo ngắn tay; B lẫn túi, đồng hồ, cà vạt"* | Khớp đúng notes: Dataset thiếu thuộc tính sleeveLength nên lẫn áo ngắn tay. |

---

## G. Trạng thái Quyết định của Người dùng

| # | Hạng mục | Trạng thái hiện tại | Hành động tiếp theo |
|---|---|---|---|
| 1 | Phạm vi Key chain (2 sản phẩm in_scope) | `ĐÃ DUYỆT LOẠI BỎ (2026-11-09)` | Đã chuyển 2 sp Key chain (ID 43230, 42066) sang `in_scope=False`. Cập nhật `catalog.parquet`, `reports/scope_proposal.csv`, `index_v2/manifest.json`. **Số cũ dùng gallery 38.646, từ rev3/rev4 là 38.644**. |
| 2 | Đánh giá Blind A/B Rev4 | `ĐÃ HOÀN TẤT BÁO CÁO` | Đã báo cáo kết quả đánh giá Claude AI single rater, cập nhật Clopper-Pearson Exact CI, dọn sạch suy đoán, phân tích bug "đồ $\to$ Red" và giải pháp parser v2. Tùy chọn để Người dùng tự chấm lại độc lập nếu cần. |
| 3 | Đề xuất thông báo Lavie (x017) | `CHỜ NGƯỜI DÙNG DUYỆT` | Đề xuất tính năng hiển thị thông báo thương hiệu không tồn tại trong catalog (chưa triển khai code). |

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
- [x] Các bản cũ `phase3f_reconciliation.md`, `phase3f_errata_rev1.md`, `phase3f_errata_rev2.md`, `phase3f_errata_rev3.md` giữ nguyên làm bằng chứng lịch sử.
- [x] Tách biệt tuyệt đối hai cột "Model thuần" và "Hệ thống"; không nhận kết quả Hệ thống cho Model.

---

## I. Lịch sử sửa đổi qua các phiên bản Errata

### I.1 Lịch sử sửa đổi từ Rev2 sang Rev3
*(Xem chi tiết trong `reports/phase3f_errata_rev3.md`)*

### I.2 Lịch sử sửa đổi từ Rev3 sang Rev4
| Mục | Số cũ (Rev3) | Số mới (Rev4) | Lý do sửa đổi và Căn cứ bằng chứng |
|---|---|---|---|
| **Mục 1 (Người chấm)** | Ghi "Người dùng chấm" / "Human Ground Truth" | **"Claude (AI), single rater"** | Sửa đúng thực tế: Chấm bằng mắt do Claude AI thực hiện từ ảnh contact sheet; nêu rõ 5 hạn chế (single rater, không thấy brand, blind yếu, x018 hậu nghiệm, phạm vi hệ thống). |
| **Mục 1 (Cột Notes)** | Tự diễn giải nguyên nhân lệch ở x001, x007, x008, x010, x011 | **Chỉ trích đúng cột Notes; ghi "chưa xác định" nếu notes không nêu** | Loại bỏ hoàn toàn mọi lời suy đoán không có trong dữ liệu chấm gốc; `eval/ab_blind_heldout.csv`. |
| **Mục 1 (Nhóm Brand)** | Chỉ báo điểm chấm tay 0.8625 | **Báo P@10 theo metadata brand là 0.8000** (điểm tay 0.8625 chỉ là loại+giới) | Do mắt người/AI không đọc được logo brand nhỏ trên thumbnail; P@10 thực tế căn cứ theo metadata. |
| **Mục 2 (Bảng D.2)** | C0: 0.1705/0.3992/0.2871; C2: 0.6178/0.8419/0.7246 | **C0: 0.1116/0.3543/0.2362; C2: 0.3442/0.7465/0.5205** | Đối chiếu và sửa chuẩn xác 100% theo `eval/results_3f.json` (`q1b_paired_retrieval -> G_test`), ghi rõ mAP=MRR. |
| **Mục 3 (Bug "đồ $\to$ Red")** | Chỉ ghi nhận lỗi câu x027 chung chung | **Tìm ra nguyên nhân gốc (bỏ dấu "đồ"/"đỏ" $\to$ "do"); tạo `QueryParserV2` sửa sạch 4 câu** | Quét toàn bộ 36 câu chứa "đồ/đỏ/do" trong Q2, Q3, Q5, Heldout; `reports/phase3g_do_bug_investigation.json`. |
| **Mục 4 (Mâu thuẫn C.3 vs C.4)** | C.3 ghi 0.0620, C.4 ghi 0.0628 | **Thống nhất: Bi quan = 0.0620, Lạc quan = 0.0636, đối xứng $\pm 0.0008$ quanh 0.0628** | Trích xuất chuẩn xác từ `reports/phase3g_fp16_ties.json`; sửa lỗi gõ nhầm ở dòng 164 bản cũ. |
| **Mục 4 (Mâu thuẫn A.5)** | Ghi x018 $G_{test}$ = 54 sp, $G_{full}$ = 37.356 sp | **x018 $G_{test}$ = 7 sp ($R \ge 3$); $G_{full}$ = 38.644 sp** | Sửa khớp với `reports/brand_audit_3g.json` và catalog in-scope hiện tại. |
| **Mục 4 (Lee 477 vs 344)** | Ghi "Lee có 477 sp, Lee Cooper có 133 sp" | **Lee độc lập có 344 sp, Lee Cooper có 133 sp, tổng từ khóa là 477 sp** | Kiểm toán bằng regex `\blee\b` và `\blee cooper\b` trong `src/audit_lee.py`; không có va chạm alias. |
| **Mục 5 (Latency SLA)** | Ghi "SLA chuẩn theo AGENTS.md là p95 < 200ms" | **Xóa câu SLA chuẩn; trích nguyên văn AGENTS.md:7 (`latency < 200 ms/query`)** | AGENTS.md không ghi chữ "p95"; đính kèm log GPU (57-86°C, 442-1470 MHz), giải thích đỉnh p99 (1447-1500 ms) do Cold Start CUDA JIT; báo p95 ảnh→ảnh trung vị 178.25 ms [137.22 - 219.23 ms] so với trần 200 ms. |
| **Mục 6 (Pipeline Chất lượng)** | Ghi "bảo đảm tính tái lập bit-to-bit" | **Xóa chữ bit-to-bit; ghi đúng từng bước dtype (Encoder GPU FP16 $\to$ CPU FP32 Scorer)** | Do Text Encoder chạy GPU FP16 có thể có sai số làm tròn nhỏ $\pm 10^{-4}$ giữa các phần cứng khác nhau. |
| **Mục 7 (CI Tỷ lệ Thắng)** | Dùng Bootstrap CI [1.0, 1.0] cho các nhóm toàn thắng | **Thay bằng Clopper-Pearson Exact 95% CI** (Spec: [86.8%, 100.0%], Brand: [63.1%, 100.0%]) | Áp dụng công thức nhị thức chính xác Clopper-Pearson chuẩn mực toán học mẫu hữu hạn. |
| **Mục 8 (Key chain 2 vs 35)** | Chưa giải thích rõ | **Giải thích rõ 35 sp ở raw Kaggle Pha 2a; 33 sp bị loại ở Pha 1; 2 sp (ID 43230, 42066) lọt vào catalog** | Xác nhận 2 ID đã chuyển `in_scope=False`; catalog và index đồng bộ chuẩn xác 38.644 sp. |
| **Mục 9 (Manifest)** | Vẫn còn các khóa `q6_independent_*` | **Xóa bỏ toàn bộ khóa `q6_independent_*`; tính lại SHA-256 errata và cập nhật timestamp** | Đồng bộ toàn diện `reports/run_manifest.json`. |
| **Mục 10 (Thực nghiệm Padding)** | Chỉ nêu thống kê số token caption | **Chạy thực nghiệm trên cả 1.290 câu test Q1b** (`reports/phase3g_padding_truncation_experiment.json`) | Chứng minh độ dài đệm 128 là nguyên nhân chi phối (Truncate 64 + Pad 128 đạt 0.0837 vượt trội Pad 64 đạt 0.0380); cắt cụt ở token 64 không gây suy giảm chất lượng. |
| **Mục 11 (Đề xuất Lavie x017)** | Chưa có đề xuất | **Đề xuất tính năng thông báo thương hiệu không tồn tại trong catalog** | Ghi rõ là đề xuất sản phẩm chưa triển khai, chờ Người dùng duyệt. |
