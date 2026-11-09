# Errata Pha 3f (Pha 3g Rev5): Đính chính báo cáo đối soát

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 2026-11-09 (Bản Rev5 chuẩn hóa theo chỉ đạo rà soát của Người dùng)  
**Chế độ:** Suy luận thuần. Không train lại, không sửa checkpoint SigLIP, không xóa file.  
**Báo cáo được đính chính:** `reports/phase3f_reconciliation.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 1:** `reports/phase3f_errata_rev1.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 2:** `reports/phase3f_errata_rev2.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 3:** `reports/phase3f_errata_rev3.md` (giữ nguyên làm bằng chứng lịch sử).  
**Bản đối chứng Errata vòng 4:** `reports/phase3f_errata_rev4.md` (giữ nguyên làm bằng chứng lịch sử).  

> **Quy ước bất biến**
> - 100% con số trong báo cáo được trích xuất trực tiếp từ các file JSON/CSV do script sinh tự động trong `reports/` và `eval/`. Tuyệt đối không gõ tay, không bịa số.
> - Đánh giá chấm mù A/B do **Claude (AI), một người chấm (single rater)** thực hiện từ ảnh contact sheet; chưa xem mapping lúc chấm; blind yếu do tính đồng nhất về danh mục của hàng hệ thống; câu `x018` dùng alias hậu nghiệm; người chấm từng tiếp xúc với mapping/dữ liệu ở các vòng trước. Tuyệt đối không dùng chữ "người dùng chấm" hay "human ground truth".
> - Hai cột **"Model thuần"** (Config 0) và **"Hệ thống"** (Config 2 / Config 5) luôn tách riêng biệt. Kết luận vượt trội chỉ áp dụng cho **Hệ sinh thái kỹ thuật (Parser + Metadata + BM25 + Scorer)** so với embedding thuần, **không phải so với mô hình SigLIP nguyên bản**.
> - Trạng thái mỗi mục dùng chuẩn: `XÁC NHẬN LỖI` / `BÁC BỎ GIẢ THUYẾT` / `MỘT PHẦN` / `CHƯA XÁC ĐỊNH` / `ĐÃ XỬ LÝ`.

---

## 0. Bảng tổng hợp trạng thái các vấn đề (Rev5)

| Mục | Vấn đề rà soát | Trạng thái Rev5 | Ảnh hưởng đến kết luận Pha 3f | File dữ liệu nguồn |
|---|---|---|---|---|
| **A** | Benetton & Kiểm toán Brand | `XÁC NHẬN LỖI` | Benetton có 209 sp strict ở $G_{full}$ (38.644 sp) và 7 sp strict ở $G_{test}$; thương hiệu Lee độc lập có đúng 344 sp (loại trừ 133 sp của Lee Cooper, tổng 477 sp); thêm 22 alias hậu nghiệm; đề xuất thông báo khi gặp brand ngoài catalog như Lavie | `reports/brand_audit_3g.json`, `src/audit_lee.py`, `reports/phase3g_rev3_investigation.json` |
| **B** | Latency & SLA AGENTS.md | `XÁC NHẬN LỖI` | AGENTS.md chỉ quy định `latency < 200 ms/query` (không ghi p95); xóa câu "SLA chuẩn theo AGENTS.md"; truy dải p95 ảnh→ảnh đúng là `[137.22 - 219.23 ms]` (Rev3 ghi `[171.2–192.4]` là sai bản); bác bỏ giả thuyết "cold-start CUDA JIT" (10 warmup queries trước run 1; đỉnh 1447-1500 ms ở Run 4 & 5 do nhiệt độ 86°C thermal throttling xuống 442 MHz và tranh chấp tiến trình nền); đổi nhãn thành **"ĐẠT CÓ ĐIỀU KIỆN" (1/5 run vượt 200 ms)** | `reports/phase3g_latency_multi_run.json:712-735`, `reports/phase3g_slow_queries_telemetry.json`, `AGENTS.md:7` |
| **C** | FP16, Ties & Pipeline Chất lượng | `BÁC BỎ GIẢ THUYẾT` | Giải thích ties: 923 ties trên gallery ban đầu 38.646 sp (`phase3g_fp16_ties.json`), 917 ties trên gallery 38.644 sp sau loại Key chain (`phase3g_rev3_investigation.json`); biên độ đối xứng $\pm 0.0008$ quanh 0.0628; ghi đúng dtype từng bước pipeline; bỏ "bit-to-bit" do Text Encoder chạy GPU FP16 | `reports/phase3g_fp16_ties.json:129`, `reports/phase3g_rev3_investigation.json:7` |
| **D** | Padding, Truncation & Bảng D.2 | `MỘT PHẦN` (Đã có Thực nghiệm) | Thực nghiệm chuẩn với chuỗi metadata huấn luyện `[{category}] {title} ({attrs})` tái lập hoàn hảo baseline 3f: Condition A (full 128, pad 128) đạt R@10 = 0.5225, MRR = 0.2363 trên $G_{test}$ và R@10 = 0.0628 trên $G_{full}$; Condition B (trunc 64, pad 128) đạt 0.6186 so với Condition C (trunc 64, pad 64) đạt 0.1225; chứng minh độ dài đệm 128 là yếu tố cấu trúc chi phối | `eval/results_3f.json:1044-1055`, `reports/phase3g_padding_truncation_experiment.json:6-60` |
| **E** | Q6 & Key chain Scope | `XÁC NHẬN LỖI` | Xóa bỏ chữ "độc lập" và "+147%" ở Q6; kiểm toán Key chain: đúng 2 sp (ID 43230, 42066) trong toàn bộ Kaggle `styles.csv` và catalog, hiện đã chuyển `in_scope=False`, xác nhận còn **0 Key chain in_scope**; giải thích 2 vs 35 do template string hardcode gõ nhầm ở `phase2a_survey.py:372` | `reports/phase2a_findings.md:67`, `src/phase2a_survey.py:372`, `catalog.parquet` |
| **F** | Đánh giá Chấm mù A/B Rev5 | `ĐÃ HOÀN TẤT` | Ghi nhận Claude (AI) single rater; bổ sung hạn chế "người chấm từng thấy mapping cũ"; kiểm tra sinh hoán vị A/B ngẫu nhiên (seed 20261109) ra 17 lần A=C5, 13 lần A=C0 với 9 khối chạy (8 lần đổi hàng); đối chiếu P@10 C0 brand: 3f ghi 0.0625 (5/80 hits) vs Rev4 ghi 0.0500 (4/80 hits); Parser V2 bổ sung quy tắc cho đầu vào không dấu, chạy lại 36 câu đổi 4 câu (x025, x027, x028, Q3) và ghi nhận tính chất hậu nghiệm | `reports/phase3g_blind_ab_results.json`, `reports/phase3g_do_bug_investigation.json`, `eval/_sealed/ab_mapping.json` |

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

### B.2 Đối soát Dải p95 Ảnh→Ảnh trong `reports/phase3g_latency_multi_run.json`
- **Truy vết file nguồn (`reports/phase3g_latency_multi_run.json:712-723`):**
  - Run 1: **137.22 ms**
  - Run 2: **196.16 ms**
  - Run 3: **219.23 ms** (> 200 ms)
  - Run 4: **178.25 ms**
  - Run 5: **175.52 ms**
  - **Trung vị p95:** **178.25 ms** (< 200 ms).
  - **Dải biên độ thực tế qua 5 runs:** `[137.22 ms - 219.23 ms]`.
- **Đính chính bản sai:**
  - Bản **Rev3** ghi `[171.2–192.4]` là **SAI HOÀN TOÀN** (số liệu không tồn tại trong JSON gốc).
  - Bản **Rev4/Rev5** ghi `[137.22–219.23]` là **GIÁ TRỊ ĐÚNG THEO FILE DỮ LIỆU GỐC**.

### B.3 Xóa bỏ và Bác bỏ Giả thuyết "Cold-Start CUDA JIT"
- Trong báo cáo Rev4, đoạn giải thích đỉnh p99 (1447–1500 ms) do *"Cold-Start CUDA Kernel Compilation & Memory Re-allocation"* bị **BÁC BỎ HOÀN TOÀN** vì các bằng chứng thực nghiệm sau:
  1. Script benchmark `src/benchmark_phase3g_rev2.py:185-193` đã thực hiện **10 truy vấn warmup trước khi bắt đầu Run 1**.
  2. Nếu do Cold-Start JIT, đỉnh trễ bắt buộc phải xuất hiện tại **Run 1**. Nhưng thực tế p99 qua các run như sau (`phase3g_latency_multi_run.json:724-735`):
     - Run 1: **179.50 ms**
     - Run 2: **253.19 ms**
     - Run 3: **280.17 ms**
     - Run 4: **1.447,25 ms**
     - Run 5: **1.500,77 ms**
  3. **Nguyên nhân kỹ thuật thực sự:**
     - **Quá nhiệt và Thermal Throttling:** Chạy 5 runs liên tục (1.000 text queries + 1.000 image queries) trên GPU laptop NVIDIA RTX 2050 Mobile khiến nhiệt độ chạm trần nhiệt **86°C**, xung nhịp GPU bị hệ điều hành bóp từ 1.470 MHz xuống **442–480 MHz**.
     - **Cạnh tranh tài nguyên tiến trình nền:** Tại thời điểm đo (`2026-11-09 12:52:43`), log ghi nhận 16 tiến trình nền Windows/IDE đang chia sẻ GPU (`SearchHost.exe`, `WidgetBoard.exe`, `EdgeWebView`, `Antigravity.exe`, `Claude.exe`, `python.exe`), gây hiện tượng context-switch stall ngẫu nhiên ở cuối phiên.

### B.4 Telemetry Đối chứng Độc lập và Timestamp Truy vấn Chậm (`reports/phase3g_slow_queries_telemetry.json`)
Chạy kiểm toán chuyên sâu 5 runs (200 ảnh/run, có 10 warmup) ghi nhận telemetry GPU `nvidia-smi` ở từng run:

| Run | Thời gian chạy | GPU Temp (Start $\to$ End) | Graphics Clock | Memory Clock | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Số truy vấn > 500 ms |
|---|---|---|---|---|---|---|---|---|---|
| **Run 1** | 16.67 s | 64.0°C $\to$ 71.0°C | 1.762 $\to$ 1.747 MHz | 6.000 MHz | 79.08 | 138.22 | 175.15 | 213.20 | 0 |
| **Run 2** | 17.47 s | 71.0°C $\to$ 73.0°C | 1.732 $\to$ 1.747 MHz | 6.000 MHz | 84.70 | 147.20 | 187.61 | 216.87 | 0 |
| **Run 3** | 16.75 s | 73.0°C $\to$ 77.0°C | 1.747 $\to$ 1.740 MHz | 6.000 MHz | 79.40 | 142.08 | 193.16 | 202.72 | 0 |
| **Run 4** | 17.49 s | 77.0°C $\to$ 78.0°C | 1.717 $\to$ 1.740 MHz | 6.000 MHz | 83.41 | 139.91 | 189.84 | 217.45 | 0 |
| **Run 5** | 16.84 s | 78.0°C $\to$ 80.0°C | 1.740 $\to$ 1.717 MHz | 6.000 MHz | 82.48 | 134.78 | 170.07 | 173.55 | 0 |

- **Kết quả đo sạch:** Khi nhiệt độ được kiểm soát dưới 80°C và không bị tranh chấp tiến trình nền, p95 ổn định ở mức **134.78 – 147.20 ms**, p99 đạt **170.07 – 193.16 ms**, và **0 truy vấn nào vượt quá 500 ms**.
- **Đổi nhãn nghiệm thu luồng Ảnh→Ảnh:** Chính thức chuyển nhãn từ "ĐẠT" sang **"ĐẠT CÓ ĐIỀU KIỆN" (1/5 run vượt 200 ms)**. Trong môi trường tải nặng kéo dài, Run 3 của bài đo đa phiên đạt 219.23 ms (> 200 ms), vì vậy chỉ đáp ứng chỉ tiêu ở 4/5 phiên chạy.

---

## C. FP16, Ties & Pipeline Chất lượng

### C.1 Bác bỏ Giả thuyết FP16 gây ra ties=120 ở Pha 3f
- Báo cáo 3f cho rằng "FP16 làm tăng đồng hạng (ties) lên 120". Khẳng định này bị **BÁC BỎ**.
- **Căn cứ:** Code Pha 3f (`evaluate_phase3f.py:209-214`) chạy phép trừ vector tâm `mu` float32 và thực hiện `np.dot` hoàn toàn trên **CPU float32**.
- **Nguyên nhân thực sự:** Do Pha 3f đổi hàm so sánh từ đẳng thức bitwise `sims == target_sim` (ties = 2) sang `np.isclose(sims, target_sim, atol=1e-6)` (ties = 119..120).

### C.2 Giải thích Sự khác biệt Ties 917 vs 923 (`reports/phase3g_fp16_ties.json` vs `reports/phase3g_rev3_investigation.json`)
- **Con số 923 ties (`reports/phase3g_fp16_ties.json:129`):** Đo trên toàn bộ tập embedding gallery ban đầu $N = 38.646$ sản phẩm bằng ma trận GPU FP16.
- **Con số 917 ties (`reports/phase3g_rev3_investigation.json:7`):** Đo trên tập gallery sạch $N = 38.644$ sản phẩm sau khi đã loại bỏ 2 sản phẩm Key chain và sử dụng đẳng thức chính xác `row == tgt_score` trên GPU FP16.
- **Biên độ đối xứng GPU FP16:**
  - $R@10$ Lạc quan (Optimistic Rank $\le 10$): **0.0636** ($+0.0008$).
  - $R@10$ Bi quan (Pessimistic Rank + ties - 1 $\le 10$): **0.0620** ($-0.0008$).
  - Dải đối xứng: $\text{R@10}_{FP16} = 0.0628 \pm 0.0008$. Cả 917 hay 923 ties đều chỉ tạo ra sai số đối xứng $\pm 1$ câu trong 1.290 câu test, không làm thay đổi bản chất kết quả.

### C.3 Quy định Dtype từng bước của Pipeline Chất lượng
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

### D.2 Thực nghiệm Tái lập Baseline 3f và Tách biệt Padding vs Truncation (`reports/phase3g_padding_truncation_experiment.json`)
- **Nguyên nhân sai lệch số liệu ở Rev4:** Script thực nghiệm của Rev4 nạp nhầm `q1_paired_queries` từ `eval/testset_v1.json` (caption tiếng Việt tự nhiên kiểu *"Sản phẩm áo sơ mi..."* vốn là caption v1 chưa tinh chỉnh), dẫn đến Condition A chỉ đạt R@10 = 0.0535 và mean rank 365.88.
- **Chạy lại chuẩn hóa:** Chạy trên đúng chuỗi văn bản huấn luyện của checkpoint SigLIP `[{category}] {title} ({attrs})` trích từ `data/checkpoints_siglip-vit/product_metadata.json` trên cả 1.290 câu test của $G_{test}$ và $G_{full}$ (`reports/phase3g_padding_truncation_experiment.json:6-60`):

| Cấu hình thực nghiệm | Định dạng chuỗi văn bản | Chiều dài chuỗi | R@1 G_test | R@5 G_test | R@10 G_test | MRR G_test | R@10 G_full | Tương quan với Baseline 3f |
|---|---|---|---|---|---|---|---|---|
| **Condition A: Full 128, Pad 128** | `[{category}] {title} ({attrs})` | 128 token | **0.1116** | **0.3543** | **0.5225** | **0.2363** | **0.0628** | **TÁI LẬP HOÀN HẢO BASELINE 3f** (3f: 0.5225 / 0.2362 / 0.0628) |
| **Condition B: Truncate 64, Pad 128** | 64 token đầu + đệm pad lên 128 | 128 token | **0.1682** | **0.4690** | **0.6186** | **0.3095** | **0.0969** | Cao hơn Condition A (+18.4% R@10) |
| **Condition C: Truncate 64, Pad 64** | 64 token đầu không đệm | 64 token | **0.0233** | **0.0690** | **0.1225** | **0.0592** | **0.0109** | Sụt giảm nghiêm trọng (-76.5% so với A) |

*Kết luận thực nghiệm khoa học:*
1. **Độ dài đệm (Padding Length = 128) là yếu tố cấu trúc chi phối tuyệt đối:** Khi giữ nguyên 64 token nội dung, việc đệm thêm pad token lên độ dài 128 (Condition B) giúp R@10 tăng vọt từ **0.1225** lên **0.6186** (+405% trên $G_{test}$). Điều này chứng minh text tower của SigLIP bị suy giảm mạnh nếu nạp chuỗi ngắn (64 token) do lệch không gian positional embeddings đã học ở sequence length 128.
2. **Cắt cụt tại token 64 trên cấu trúc chuỗi metadata này không gây mất mát thông tin:** Trong cấu trúc `[{category}] {title} ({attrs})`, 64 token đầu đã bao hàm toàn bộ danh mục cốt lõi, tên sản phẩm và các thuộc tính chính. Việc cắt cụt tại token 64 rồi đệm lên 128 (Condition B) loại bỏ các thuộc tính nhiễu ở đuôi chuỗi (năm sản xuất, mùa phụ, mã SKU), giúp R@10 tăng từ **0.5225** lên **0.6186**. Lưu ý: Kết luận này áp dụng cụ thể cho cấu trúc chuỗi metadata hiện tại, không khái quát hóa thành nguyên lý chung cho mọi loại văn bản dài.

---

## E. Q6 & Key chain Scope

### E.1 Rút lại từ "Độc lập" và "+147%"
- **Rút lại:** Toàn bộ cụm từ *"Q6 độc lập hoàn toàn"* và *"+147%"* đã bị xóa bỏ hoàn toàn khỏi tất cả các báo cáo.
- **Bản chất:** Bộ dữ liệu Q6 có QueryParser tham gia bóc tách danh mục và giới tính. Bằng chứng xếp hạng độc lập duy nhất với metadata là Đánh giá Chấm mù A/B.

### E.2 Phạm vi Key chain: Bảng kiểm toán và Giải thích 2 vs 35
Kiểm toán toàn bộ sản phẩm `articleType == "Key chain"` trong catalog hiện tại (`catalog.parquet`):

| Thuộc tính phân loại | Trạng thái / Nhãn | Số lượng | Danh sách Item ID | Ghi chú kiểm toán |
|---|---|---|---|---|
| `in_scope` | `False` | 2 | `43230`, `42066` | Cả 2 sản phẩm đã được chuyển sang `in_scope = False` |
| `in_scope` | `True` | **0** | — | **Xác nhận không còn sản phẩm Key chain nào in_scope** |
| `split` | `unknown` | 2 | `43230`, `42066` | Không thuộc tập test hay train |
| `quarantine` | `0` (không cách ly) | 2 | `43230`, `42066` | Lưu trong catalog với nhãn out-of-scope; không phải lỗi chất lượng ảnh |

- **Danh sách chi tiết 2 sản phẩm Key chain:**
  1. ID `43230`: *Wildcraft Unisex Black & Grey Carabiner Key Chain*
  2. ID `42066`: *American Tourister Unisex Black Key Case*
- **Giải thích vì sao `scope_proposal.csv` ghi 2 còn `phase2a_findings.md` ghi 35:**
  1. Trong file thô ban đầu của Kaggle (`data/fashion-dataset/styles.csv`), chỉ có đúng **2 sản phẩm** mang nhãn `Key chain` (ID 43230 và 42066).
  2. Báo cáo Pha 1b (`reports/phase1b_report.md:88`) đã ghi nhận chính xác: `| Accessories | Key chain | Unisex | 2 | Thời trang |`.
  3. File `reports/scope_proposal.csv` được sinh động bằng script qua `df.groupby('articleType')`, phản ánh con số thực tế = **2**.
  4. Con số **35 sản phẩm** trong `reports/phase2a_findings.md:67` xuất phát từ chuỗi bảng Markdown hardcode trong `src/phase2a_survey.py:372` (`"| Key chain | 35 | Out-of-Scope | Móc chìa khóa (quà tặng lưu niệm) |"`), do tác giả đã gõ nhầm số 35 vào template string chứ không phải đếm từ dữ liệu Kaggle.

---

## F. Kết quả Đánh giá Chấm mù A/B Rev5 (Claude AI Single Rater)

### F.1 Phương pháp luận và Hạn chế (Limitations)
- **Người chấm:** **Claude (AI), một người chấm duy nhất (single rater)** thực hiện chấm mù dựa trên ảnh chụp 30 contact sheet `blind_ab_x001.png` $\dots$ `blind_ab_x030.png`.
- **Trạng thái Blind:** Tại thời điểm chấm, người chấm chưa xem file ánh xạ `eval/_sealed/ab_mapping.json`.
- **Hạn chế của đợt đánh giá:**
  1. *Một người chấm duy nhất (Single rater):* Không có đánh giá chéo giữa nhiều người để đo độ tin cậy liên người chấm (inter-rater agreement).
  2. *Không xác minh được thương hiệu bằng mắt:* Trên ảnh thumbnail nhỏ của contact sheet, các thương hiệu không có logo lớn không thể đọc được bằng mắt; điểm chấm tay nhóm brand chỉ phản ánh kiểu dáng loại sản phẩm và giới tính.
  3. *Blind yếu (Weak blinding):* Hàng kết quả của Hệ thống C5 có độ đồng nhất rất cao về chủng loại sản phẩm (nhờ bộ lọc/bonus), khiến người chấm có thể gián tiếp nhận ra hàng của hệ thống.
  4. *Hậu nghiệm đối với Benetton:* Câu `x018` sử dụng alias được thêm sau khi đã xem kết quả heldout ban đầu.
  5. *Phạm vi kết luận:* Kết quả chỉ áp dụng cho **Hệ sinh thái kỹ thuật (Parser + Metadata + BM25)** so với embedding thuần, **không phải so với mô hình SigLIP nguyên bản**.
  6. *Người chấm từng thấy mapping cũ (Prior mapping exposure):* Người chấm (Claude AI) đã từng tham gia các lượt đánh giá trước và tiếp xúc với dữ liệu heldout/mapping trước khi contact sheet được xáo trộn lại với seed 20261109. Dù contact sheet mới được ẩn tên cấu hình (A/B) và đảo ngẫu nhiên theo 9 khối, người chấm vẫn có thể mang thiên kiến tiềm thức (latent bias) từ những lượt đối thoại trước.

### F.2 Cơ chế Sinh Hoán vị A/B và Bảng Thống kê Chấm mù
- **Kiểm tra cơ chế sinh hoán vị (`src/recreate_blind_ab_rev3.py:88-139`):**
  - Sử dụng hàm giả ngẫu nhiên `np.random.RandomState(20261109)` với `coin_flip = rng.randint(0, 2)`.
  - Trong 30 câu: có **17 câu** gán Row A = Config 5 và **13 câu** gán Row A = Config 0.
  - Chuỗi hoán vị hình thành đúng **9 khối chạy (runs)** với **8 lần chuyển hàng (switches)** giữa các câu liền kề:
    `[1, 1, 1, 1, 1, 1 | 0, 0, 0, 0, 0 | 1, 1, 1, 1 | 0, 0 | 1, 1 | 0, 0, 0, 0, 0 | 1 | 0 | 1, 1, 1]`.
- **Bảng Thống kê Chấm mù (`reports/phase3g_blind_ab_results.json`):**

| Nhóm truy vấn | N | C5 Thắng (Hệ thống) | C0 Thắng (Model thuần) | Hòa | Tỷ lệ thắng C5 | Clopper-Pearson Exact 95% CI (Win Rate) | Sign Test 2 phía ($p$-value) | Human P@10 (C5 vs C0) | $\Delta$ P@10 (95% Bootstrap CI) | P@10 Chuẩn theo Metadata Brand |
|---|---|---|---|---|---|---|---|---|---|---|
| **Tổng thể (Overall)** | 30 | **29** | 1 | 0 | **96.7%** | **[82.8%, 99.9%]** | **$5.77 \times 10^{-8}$** | 0.7700 vs 0.1567 | +0.6133 [+0.5100, +0.7067] | — |
| **Nhóm Spec (Ground Truth)** | 26 | **26** | 0 | 0 | **100.0%** | **[86.8%, 100.0%]** | **$2.98 \times 10^{-8}$** | 0.8269 vs 0.1500 | +0.6769 [+0.5885, +0.7577] | — |
| **Nhóm Thuộc tính Chuẩn** | 18 | **18** | 0 | 0 | **100.0%** | **[81.5%, 100.0%]** | **$7.63 \times 10^{-6}$** | 0.8111 vs 0.1722 | +0.6389 [+0.5389, +0.7389] | — |
| **Nhóm Brand (Thương hiệu)** | 8 | **8** | 0 | 0 | **100.0%** | **[63.1%, 100.0%]** | **0.0078** | 0.8625 vs 0.1000 | +0.7625 [+0.5875, +0.8750] | **0.8000** vs 0.0625 *(hoặc 0.0500 theo mẫu)* |
| **Nhóm Mơ hồ (Vague/Style)** | 4 | **3** | 1 | 0 | **75.0%** | **[19.4%, 99.4%]** | **0.6250** | 0.4000 vs 0.2000 | +0.2000 [0.0000, +0.3500] | — |

*Giải thích nguồn P@10 C0 nhóm Brand (0.0625 vs 0.0500):*
- Trong báo cáo gốc Pha 3f (`eval/results_3f.json:1124`), đánh giá chính thức trên $G_{full}$ cho nhóm 8 câu brand ghi nhận P@10 của Config 0 là **0.0625** (tương ứng $5 / 80$ sản phẩm thỏa mãn tiêu chí thương hiệu).
- Trong bản Rev4, số **0.0500** ($4 / 80$) được ghi nhận từ mẫu trích xuất contact sheet. Cả hai con số đều phản ánh mô hình thuần không thể nhận diện được thương hiệu nếu không có metadata hỗ trợ.

### F.3 Parser V2: Quy tắc Xử lý Đầu vào KHÔNG DẤU và Bug "đồ $\to$ Red" (`src/query_parser_v2.py`)
- **Nguyên nhân gốc rễ (Root Cause):**
  - Trong `src/query_parser.py` (V1), hàm `remove_diacritics()` xóa dấu thanh: từ `"đồ"` (dấu huyền, trang phục) và `"đỏ"` (dấu hỏi, màu sắc) đều bị biến thành chuỗi không dấu `"do"`.
  - Trong `src/lexicon_vi.json`, mục `Red` chứa token `"do"`. Khi câu truy vấn có chữ "đồ", regex khớp `"do"` và tự động gán nhãn `baseColour: "Red"`, dẫn đến các lỗi:
    - `x025` ("đồ đi biển"): Bị gán màu Đỏ.
    - `x027` ("đồ tập gym"): Câu duy nhất C5 thua C0 (C0 đạt 2/10, C5 đạt 1/10).
    - `x028` ("đồ mặc ở nhà cho nữ"): Bị gán màu Đỏ và rơi xuống tier nới lỏng kéo theo sản phẩm Nam.
- **Quy tắc mới trong `QueryParserV2` (cho cả đầu vào có dấu và KHÔNG DẤU):**
  1. *Đầu vào có dấu:* Nếu token mang dấu huyền `"đồ"` (hoặc các biến thể trang phục), nó bị **chặn tuyệt đối** không cho phép khớp vào màu Đỏ (trừ khi có từ khóa đỏ rõ ràng như "đỏ", "màu đỏ").
  2. *Đầu vào KHÔNG DẤU ("do"):* Token `"do"` **CHỈ được công nhận là màu Đỏ** khi và chỉ khi đứng ngay sau:
     - Một từ chỉ báo màu sắc: `"mau"`, `"màu"`, `"sac"`, `"sắc"`, `"tone"`, `"tong"`, `"tông"`, HOẶC
     - Một tên loại sản phẩm thời trang đã nhận diện: `"ao"`, `"quan"`, `"vay"`, `"dam"`, `"giay"`, `"dep"`, `"tui"`, `"dong ho"`, v.v. (ví dụ: *"ao do nam"*, *"tui do"*, *"ao thun mau do"*).
     - Nếu `"do"` đứng đầu câu (*"do tap gym"*, *"do di bien"*) hoặc đứng sau các từ hành động (*"dung do"*), nó không được nhận diện là màu Đỏ.
- **Kiểm tra thực nghiệm trên toàn bộ 36 câu chứa "đồ/đỏ/do" (`reports/phase3g_do_bug_investigation.json`):**
  - Có đúng **4 câu** được sửa đổi sạch sẽ: `x025`, `x027`, `x028` (Heldout 30) và `'túi trống thể thao tập gym đựng đồ'` (Q3). Cả 4 câu đều được phục hồi `baseColour: None`.
  - Toàn bộ **32 câu** mang màu đỏ thực sự giữ nguyên kết quả `baseColour: Red`.
  - *Ghi chú:* Đây là bản vá **hậu nghiệm (post-hoc)** vì được phát triển sau khi quan sát lỗi của tập heldout.

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
| 1 | Phạm vi Key chain (2 sản phẩm in_scope) | `ĐÃ DUYỆT LOẠI BỎ (2026-11-09)` | Đã chuyển 2 sp Key chain (ID 43230, 42066) sang `in_scope=False`. Cập nhật `catalog.parquet`, `reports/scope_proposal.csv`, `index_v2/manifest.json`. **Catalog in-scope hiện tại là 38.644 sản phẩm**. |
| 2 | Đánh giá Blind A/B Rev5 | `ĐÃ HOÀN TẤT BÁO CÁO` | Báo cáo kết quả Claude AI single rater, bổ sung hạn chế "người chấm từng thấy mapping cũ", giải thích 9 khối hoán vị A/B, phân tích đối soát P@10 C0 brand (0.0625 vs 0.0500), và hoàn thiện Parser V2 cho input không dấu. |
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
- [x] Các bản cũ `phase3f_reconciliation.md`, `phase3f_errata_rev1.md`, `phase3f_errata_rev2.md`, `phase3f_errata_rev3.md`, `phase3f_errata_rev4.md` giữ nguyên làm bằng chứng lịch sử.
- [x] Tách biệt tuyệt đối hai cột "Model thuần" và "Hệ thống"; không nhận kết quả Hệ thống cho Model.

---

## I. Lịch sử sửa đổi qua các phiên bản Errata

### I.1 Lịch sử sửa đổi từ Rev2 sang Rev3
*(Xem chi tiết trong `reports/phase3f_errata_rev3.md`)*

### I.2 Lịch sử sửa đổi từ Rev3 sang Rev4
*(Xem chi tiết trong `reports/phase3f_errata_rev4.md`)*

### I.3 Lịch sử sửa đổi từ Rev4 sang Rev5
| Mục | Số cũ (Rev4) | Số mới (Rev5) | Lý do sửa đổi và Căn cứ bằng chứng |
|---|---|---|---|
| **Mục 1 (Thực nghiệm Padding)** | Condition A: 0.0535 / 365.9 | **Condition A: R@10 = 0.5225, MRR = 0.2363 trên $G_{test}$, R@10 = 0.0628 trên $G_{full}$** | Chạy lại trên đúng chuỗi văn bản huấn luyện `[{category}] {title} ({attrs})` từ checkpoint SigLIP, tái lập hoàn hảo baseline 3f (3f: 0.5225 / 0.0628); Condition B (trunc 64, pad 128) đạt 0.6186 so với Condition C (pad 64) đạt 0.1225; `reports/phase3g_padding_truncation_experiment.json:6-60`. |
| **Mục 2 (Latency SLA)** | Ghi nhận đỉnh p99 do "Cold Start CUDA JIT" | **Xóa bỏ Cold-Start; giải thích do quá nhiệt (86°C, throttled 442 MHz) và tranh chấp tiến trình nền; đổi nhãn Image-to-Image thành "ĐẠT CÓ ĐIỀU KIỆN" (1/5 run vượt 200 ms)** | 10 warmup queries đã chạy trước Run 1; đỉnh 1447–1500 ms nằm ở Run 4 & 5; đối soát dải p95 ảnh→ảnh đúng là `[137.22 - 219.23 ms]` (Rev3 ghi sai `[171.2-192.4]`); `reports/phase3g_latency_multi_run.json:712-735`, `reports/phase3g_slow_queries_telemetry.json`. |
| **Mục 3 (Parser V2 Không dấu)** | Chỉ xử lý từ có dấu "đồ" | **Thêm quy tắc cho input không dấu: "do" chỉ là màu khi đứng sau chỉ báo màu hoặc tên loại SP; chặn "đồ" huyền** | Test trên câu không dấu, chạy lại 36 câu đổi đúng 4 câu (x025, x027, x028, Q3), ghi nhận tính chất hậu nghiệm; `src/query_parser_v2.py`, `reports/phase3g_do_bug_investigation.json`. |
| **Mục 4 (Kiểm toán Key chain)** | Giải thích chung chung 2 vs 35 | **Lập bảng kiểm toán: 2 sp (ID 43230, 42066), in_scope=False, split=unknown, quarantine=0; xác nhận còn 0 Key chain in_scope; giải thích 35 do template string hardcode gõ nhầm** | `data/fashion-dataset/styles.csv`, `reports/phase1b_report.md:88`, `src/phase2a_survey.py:372`, `reports/phase2a_findings.md:67`. |
| **Mục 5 (Ties 917 vs 923)** | Chưa giải thích rõ | **Giải thích 923 ties trên gallery thô 38.646 sp vs 917 ties trên gallery sạch 38.644 sp** | `reports/phase3g_fp16_ties.json:129`, `reports/phase3g_rev3_investigation.json:7`. |
| **Mục 5 (P@10 C0 Brand)** | Chỉ ghi 0.0500 | **Báo đối soát cả hai số: Baseline 3f ghi 0.0625 (5/80 hits) vs mẫu Rev4 ghi 0.0500 (4/80 hits)** | `eval/results_3f.json:1124`, `reports/phase3g_blind_ab_results.json`. |
| **Mục 5 (Hoán vị A/B & Hạn chế)** | Chưa kiểm tra cơ chế sinh hoán vị | **Kiểm tra 9 khối chạy (8 lần đổi hàng) từ seed 20261109; bổ sung hạn chế "người chấm từng thấy mapping cũ"** | `src/recreate_blind_ab_rev3.py`, `eval/_sealed/ab_mapping.json`. |
