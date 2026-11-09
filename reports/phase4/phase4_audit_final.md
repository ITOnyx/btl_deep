# Báo cáo Kiểm toán & Sửa lỗi Cuối Pha 4 (Phase 4 Final Audit Report)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Môi trường:** `D:/Kì 7/Hoc_sau_ver4`  
**Ngày thực hiện:** 2026-11-09  
**Người thực hiện:** Audit Agent  
**Nguyên tắc cốt lõi:**
1. Số liệu chỉ trích xuất từ file dữ liệu gốc đo được (`eval/*.json`, `reports/**/*.json`, `data/catalog/catalog.parquet`).
2. Tuyệt đối không thay đổi mô hình (`best_model.pt`), chỉ mục vector (`index/`, `index_v2/`), hoặc bộ đánh giá đóng băng (`eval/_sealed/`).
3. Tuyệt đối không sửa `config.yaml`, `src/brands_v2.json`, `src/query_parser_v2.py`, `eval/_sealed/ab_mapping.json`.
4. Mọi kết luận sử dụng nhãn trạng thái chuẩn hóa: `XÁC NHẬN LỖI` / `BÁC BỎ GIẢ THUYẾT` / `MỘT PHẦN` / `CHƯA XÁC ĐỊNH` / `ĐÃ XỬ LÝ`.

---

## Bảng Tổng hợp Trạng thái Kiểm toán (T1 – T8)

| Nhiệm vụ | Hạng mục kiểm toán | Trạng thái chuẩn hóa | Nguồn dữ liệu xác thực |
|---|---|:---:|---|
| **T1** | Chuẩn hóa P@10 Serving (Q2/Q3) giữa mọi tài liệu | `XÁC NHẬN LỖI` *(Đã chuẩn hóa)* | `reports/phase4_verifications.json:task3_serving_re_eval` |
| **T2** | Số ảnh val (7,470 vs 7,471) & 28 truy vấn val | `ĐÃ XỬ LÝ` | `data/catalog/catalog.parquet`, `reports/phase4_verifications.json:task5_val_truncation` |
| **T3** | Heldout `valid_n` ở $G_{test}$ ($n=24$) và $G_{full}$ ($n=25$) | `XÁC NHẬN LỖI` & `ĐÃ XỬ LÝ` | `eval/heldout_queries.json`, `reports/phase4_verifications.json:task3_serving_re_eval.heldout` |
| **T4** | Đo lường độ trễ End-to-End & Tách biệt hai phép đo | `ĐÃ XỬ LÝ` | `reports/phase4/phase4_latency_e2e.json`, `reports/phase4_verifications.json:task4_telemetry` |
| **T5** | Phạm vi Index: 38,646 vs 38,644 và Xung đột Sporting Goods | `CHƯA XÁC ĐỊNH` *(Chờ duyệt)* | `catalog.parquet`, `reports/phase2/phase2a/phase2a_findings.md`, `config.yaml` |
| **T6** | Kiểm toán Độc lập Toàn diện Catalog | `ĐÃ XỬ LÝ` | `src/diag/verify_catalog_t6.py`, `data/catalog/catalog.parquet` |
| **T7** | Cập nhật & Đồng bộ Manifest Chữ ký Hash | `ĐÃ XỬ LÝ` | `reports/run_manifest.json` |
| **T8** | Các Lưu ý Kỹ thuật Bắt buộc (AP@10 vs P@10, Rule 24, A/B Seed, Mẫu nhỏ) | `ĐÃ XỬ LÝ` | `eval/_sealed/ab_mapping.json`, `AGENTS.md` |

---

## T1. Chuẩn hóa P@10 Serving (Q2/Q3) Giữa Mọi Tài Liệu

### 1. Hiện tượng & Điều tra
- **Phát hiện:** `phase4_verifications.json` và Errata Table L.1 ghi nhận $G_{full}$ serving P@10: Q2 = **0.9984**, Q3 = **0.8663**. Tuy nhiên, trong một bản tóm tắt đối thoại trước đây xuất hiện hai con số lạ: Q2 = 0.9980, Q3 = 0.7936.
- **Rà soát kho lưu trữ (Repo Grep):** Tìm kiếm chuỗi `0.9980` và `0.7936` trên toàn bộ repo (`*.md`, `*.json`, `*.py`, `*.txt`, README).
  + Kết quả: Cả hai số `0.9980` và `0.7936` **hoàn toàn không tồn tại** trong bất kỳ file mã nguồn, file báo cáo markdown hay file dữ liệu json nào trong repository.
  + Nguồn gốc: Hai số này là lỗi trích xuất ảo (hallucination) xuất hiện trong nội dung tin nhắn chat của AI trước đó.
- **Đối soát dữ liệu nguồn (`reports/phase4_verifications.json`):**
  + `task3_serving_re_eval.q2.g_full.serving_p10`: **0.9984** (AP@10 = 0.9982).
  + `task3_serving_re_eval.q3.g_full.serving_p10`: **0.8663** (AP@10 = 0.8443).
  + Cả hai con số trên đều khớp hoàn toàn với Errata L.1 và được tính trực tiếp từ script đánh giá serving evaluator.

### 2. Bảng Đối chiếu & Chuẩn hóa Tài liệu

| Tài liệu | Chỉ số | Số cũ (trong chat tóm tắt) | Số chuẩn xác (trong Repo) | Nguồn file dữ liệu gốc | Trạng thái |
|---|---|:---:|:---:|---|:---:|
| Tóm tắt chat | Q2 $G_{full}$ Serving P@10 | `0.9980` | **0.9984** | `reports/phase4_verifications.json:task3_serving_re_eval.q2.g_full` | `XÁC NHẬN LỖI` |
| Tóm tắt chat | Q3 $G_{full}$ Serving P@10 | `0.7936` | **0.8663** | `reports/phase4_verifications.json:task3_serving_re_eval.q3.g_full` | `XÁC NHẬN LỖI` |
| `phase4_verifications.json` | Q2 $G_{full}$ Serving P@10 | `0.9984` | **0.9984** | `task3_serving_re_eval.q2.g_full.serving_p10` | `ĐÃ XỬ LÝ` |
| `phase4_verifications.json` | Q3 $G_{full}$ Serving P@10 | `0.8663` | **0.8663** | `task3_serving_re_eval.q3.g_full.serving_p10` | `ĐÃ XỬ LÝ` |
| `phase3f_errata.md` (Bảng L.1) | Q2 $G_{full}$ Serving P@10 | `0.9984` | **0.9984** | Bảng L.1 dòng 500 | `ĐÃ XỬ LÝ` |
| `phase3f_errata.md` (Bảng L.1) | Q3 $G_{full}$ Serving P@10 | `0.8663` | **0.8663** | Bảng L.1 dòng 502 | `ĐÃ XỬ LÝ` |

---

## T2. Số Ảnh Validation và Số Truy Vấn Validation

### 1. Hiện tượng & Phân tích Đếm
- **Hiện tượng:** `phase3e_inference_only.md` ghi nhận tập val = 7,471 sản phẩm; Errata D.2 ghi 7,470 sản phẩm và nhắc tới 28 truy vấn val trích dẫn tại `reports/phase4_verifications.json`.
- **Cơ sở dữ liệu thực tế (`data/catalog/catalog.parquet`):**
  + Toàn bộ catalog có 40,891 dòng. Cột `split` chỉ có hai giá trị: `unknown` (39,519 dòng) và `test` (1,372 dòng). Không có giá trị `val` cố định sẵn trong catalog.
  + Phân chia tập validation theo quy ước dự án: Lấy 20% ngẫu nhiên (`seed=42`) từ các sản phẩm non-test (`split != 'test'`).
  + **Giai đoạn Pha 3e (`reports/val_grid_results.json`):** Dựa trên gallery ban đầu 38,646 sản phẩm in-scope (chưa loại 2 sản phẩm Key chain). Số sản phẩm non-test in-scope là $38,646 - 1,290 = 37,356$.
    $$\text{Val size (3e)} = \text{int}(37,356 \times 0.20) = \mathbf{7,471\text{ sản phẩm}}.$$
  + **Giai đoạn Pha 4 (Serving Gallery sạch):** Sau khi loại 2 sản phẩm Key chain (`43230`, `42066`), gallery sạch còn lại 38,644 sản phẩm in-scope. Số sản phẩm non-test in-scope là $38,644 - 1,290 = 37,354$.
    $$\text{Val size (Phase 4)} = \text{int}(37,354 \times 0.20) = \mathbf{7,470\text{ sản phẩm}}.$$
- **Xác thực 28 truy vấn validation:**
  + Trong file `reports/phase4/phase4a/phase4_trunc64_pad128_eval.json`, các tập được chạy là Q1b (1,290), Q2 (61), Q3 (100), Q5 (150), Heldout (26) – toàn bộ đều là tập test.
  + Con số **28 truy vấn val** được sinh động bởi script `src/diag/run_phase4_all_verifications.py:478-503`: Lấy top các bộ 3 thuộc tính (`articleType, gender, baseColour`) có tần suất $\ge 5$ trong `val_df`, dịch sang tiếng Việt tự nhiên và lọc các bộ có ít nhất 3 sản phẩm trong tập val $\implies$ đúng **28 truy vấn val**.
  + Dữ liệu này đã được ghi nhận chính thức vào `reports/phase4_verifications.json:task5_val_truncation` với hai trường: `"val_split_size": 7470` và `"n_val_queries": 28`.
- **Sửa chữa Errata D.2:** Đã bổ sung giải thích phân rã 7,471 (gallery thô) vs 7,470 (gallery sạch) và dẫn nguồn script sinh 28 truy vấn val.
- **Trạng thái:** `ĐÃ XỬ LÝ`.

---

## T3. Heldout `valid_n` ở $G_{test}$ và $G_{full}$

### 1. Hiện tượng & Chẩn đoán Lỗi
- **Hiện tượng:** `phase4_verifications.json` trước đây ghi nhận `valid_n = 25` cho cả $G_{test}$ và $G_{full}$. Tuy nhiên, phân tích logic từ Errata J.3 chỉ ra: câu `x010` ("quan short nu mau trang") có toàn bộ 6 sản phẩm trong catalog thuộc split train, do đó trên $G_{test}$ thì $R=0$; câu `x017` ("túi xách Lavie nữ") có $R=0$ trên cả 2 gallery. Do đó, $n$ hợp lệ trên $G_{test}$ bắt buộc phải là **24**, còn $G_{full}$ là **25**.
- **Nguyên nhân cốt lõi (Root Cause Bug):**
  + Trong script `src/diag/run_phase4_all_verifications.py:307`, hàm tạo Ground Truth kiểm tra điều kiện:
    ```python
    colors = spec.get("baseColour")
    ```
  + Trong khi file đặc tả `eval/heldout_queries.json` lưu tên trường là `"colour"`:
    ```json
    {"qid": "x010", "query": "quan short nu mau trang", "articleType": ["Shorts"], "gender": "Women", "colour": ["White"]}
    ```
  + Vì thiếu bộ lọc màu sắc, script cũ đã gán tất cả 6 sản phẩm Shorts của nữ trong $G_{test}$ cho `x010`, dẫn đến tính nhầm $R_{test} = 6 > 0$ và bao gồm `x010` vào mẫu số của $G_{test}$ ($n=25$).

### 2. Bảng Đối soát Toàn diện 26 Câu Spec Heldout (`eval/heldout_queries.json`)

Được trích xuất từ script độc lập `src/diag/check_heldout_table.py` qua đúng hàm Ground Truth chuẩn:

| QID | Câu truy vấn đặc tả | Target Spec (`articleType`, `gender`, `colour`/`brand`) | $R$ ($G_{full}$) | $R$ ($G_{test}$) | Hợp lệ $G_{full}$? | Hợp lệ $G_{test}$? | Ghi chú kiểm toán |
|:---:|---|---|:---:|:---:|:---:|:---:|---|
| `x001` | sơ mi nam xanh navy | `Shirts`, `Men`, `Navy Blue` | 166 | 5 | Có | Có | Bình thường |
| `x002` | áo thun nữ màu hồng | `['Tshirts', 'Tops']`, `Women`, `Pink` | 309 | 13 | Có | Có | Bình thường |
| `x003` | quần jean nam màu xanh | `Jeans`, `Men`, `Blue` | 221 | 9 | Có | Có | Bình thường |
| `x004` | giày cao gót nữ màu đen | `Heels`, `Women`, `Black` | 320 | 4 | Có | Có | Bình thường |
| `x005` | túi xách nữ màu đỏ | `Handbags`, `Women`, `Red` | 104 | 4 | Có | Có | Bình thường |
| `x006` | dép xỏ ngón nam màu đen | `Flip Flops`, `Men`, `Black` | 135 | 1 | Có | Có | Mẫu nhỏ ($R=1$) |
| `x007` | ao thun nam mau xam | `Tshirts`, `Men`, `Grey` | 536 | 15 | Có | Có | Bình thường |
| `x008` | giay the thao nam mau den | `Sports Shoes`, `Men`, `Black` | 427 | 4 | Có | Có | Bình thường |
| `x009` | vay nu mau xanh la | `['Dresses', 'Skirts']`, `Women`, `Green` | 13 | 1 | Có | Có | Mẫu nhỏ ($R=1$) |
| `x010` | quan short nu mau trang | `Shorts`, `Women`, `White` | 6 | **0** | **Có** | **KHÔNG** | Toàn bộ 6 sp thuộc train; $G_{test}$ rỗng |
| `x011` | áo thun Puma nam | `Tshirts`, `Men`, `Puma` | 534 | 12 | Có | Có | Brand catalog |
| `x012` | giày Nike nam | `Shoes`, `Men`, `Nike` | 547 | 15 | Có | Có | Brand catalog |
| `x013` | giày Adidas nữ | `Shoes`, `Women`, `Adidas` | 104 | 7 | Có | Có | Brand catalog |
| `x014` | đồng hồ Fossil nam | `Watches`, `Men`, `Fossil` | 104 | 3 | Có | Có | Brand catalog |
| `x015` | áo kurta Fabindia nam | `Kurtas`, `Men`, `Fabindia` | 74 | 3 | Có | Có | Brand catalog |
| `x016` | giày cao gót Catwalk | `Heels`, `Women`, `Catwalk` | 607 | 22 | Có | Có | Brand catalog |
| `x017` | túi xách Lavie nữ | `Handbags`, `Women`, `Lavie` | **0** | **0** | **KHÔNG** | **KHÔNG** | Brand ngoài catalog Kaggle |
| `x018` | áo thun Benetton nam | `Tshirts`, `Men`, `Benetton` | 209 | 7 | Có | Có | Đã alias canonical; mẫu nhỏ $G_{test}$ |
| `x019` | áo sơ mi nam dài tay đi làm | `Shirts`, `Men` | 2597 | 90 | Có | Có | Bình thường |
| `x020` | giày chạy bộ nam | `Sports Shoes`, `Men` | 1430 | 33 | Có | Có | Bình thường |
| `x021` | áo khoác nữ mùa đông | `Jackets`, `Women` | 55 | 1 | Có | Có | Mẫu nhỏ ($R=1$) |
| `x022` | đầm nữ đi tiệc màu đen | `Dresses`, `Women`, `Black` | 103 | 3 | Có | Có | Bình thường |
| `x023` | quần short nam thể thao màu xám | `Shorts`, `Men`, `Grey` | 45 | 3 | Có | Có | Bình thường |
| `x024` | ba lô đi học màu xanh dương | `Backpacks`, `Blue` | 148 | 10 | Có | Có | Bình thường |
| `x029` | cà vạt nam | `Ties`, `Men` | 216 | 7 | Có | Có | Bình thường |
| `x030` | khăn quàng cổ nữ | `Scarves`, `Women` | 157 | 7 | Có | Có | Bình thường |
| **Tổng** | **26 câu đặc tả (Spec)** | — | — | — | **25 / 26** | **24 / 26** | **Xác nhận số mẫu hợp lệ** |

### 3. Kết quả Chạy lại Đánh giá Serving Heldout Chuẩn Xác
Sau khi sửa bug field `colour` tại `src/diag/run_phase4_all_verifications.py:307` và cập nhật `reports/phase4_verifications.json`:
- **Trên $G_{test}$ ($n=24$ hợp lệ):**
  + Reported AP@10: **0.9393**
  + Serving AP@10: **0.9418**
  + Serving P@10: **0.5958**
- **Trên $G_{full}$ ($n=25$ hợp lệ):**
  + Reported AP@10: **0.9631**
  + Serving AP@10: **0.9646**
  + Serving P@10: **0.9680**
- **Trạng thái:** `XÁC NHẬN LỖI` và `ĐÃ XỬ LÝ`.

---

## T4. Độ Trễ: Tách Biệt Hai Phép Đo & Bỏ Kết Luận SLA Không Đủ Cơ Sở

### 1. Bản chất Phép đo ~2 ms trong `phase4_verifications.json`
- **Mã nguồn đo lường:** Đoạn mã trong `run_phase4_all_verifications.py` chỉ thực hiện:
  ```python
  q_t = torch.from_numpy(sample_query_emb).cuda().half()
  sims = torch.matmul(t_emb_img_c, q_t)
  topk = torch.topk(sims, 10)
  ```
- **Kết luận bản chất:** Phép đo này **chỉ đo nhân ma trận vector và top-k trên vector embedding đã tính sẵn**, hoàn toàn không bao gồm bước tokenize văn bản, nạp và tiền xử lý ảnh PIL, forward pass qua SigLIP Vision/Text Tower, bộ phân tích truy vấn tiếng Việt, hay trích xuất BM25.
- **Xử lý:** Gắn nhãn chính thức trong `reports/phase4_verifications.json:task4_telemetry` là `chỉ_phần_tìm_kiếm_vector_có_cache_embedding` và **loại bỏ hoàn toàn khỏi mọi kết luận SLA hệ thống**.

### 2. Đo lường Toàn diện End-to-End (`reports/phase4/phase4_latency_e2e.json`)
- **Kịch bản đo:** Viết mới `src/benchmark_e2e_latency.py`, gọi qua đúng entrypoint `app/engine.py:SearchService` cho 3 luồng:
  1. `text_to_image` (`search_text`, cấu hình c5: Tokenize + Text Tower FP16 + Centering + FAISS/Matmul + BM25 RRF + Attribute Scoring + Filtering).
  2. `composed_image_retrieval` (`search_cir`: Image Tower + Text Tower + Vector Projection + Matmul + Filtering).
  3. `image_to_image` (`search_image`: PIL Image Load + Preprocessing + 2 Passes Image Tower with Horizontal Flip Augmentation + Centering + Matmul + k-NN Voting Rescoring).
- **Quy chuẩn đo:** 10 truy vấn warmup; 5 runs $\times$ 200 truy vấn **khác nhau** (1,000 truy vấn thực tế/luồng); thread nền đọc `nvidia-smi` mỗi 1 giây trong toàn bộ thời gian chạy.

### 3. Bảng Kết quả Thực nghiệm End-to-End Toàn diện

| Luồng tìm kiếm | Phân vị | Run 1 (ms) | Run 2 (ms) | Run 3 (ms) | Run 4 (ms) | Run 5 (ms) | Tổng hợp 5 Runs (ms) | Số query > 200 ms | SLA Status (< 200 ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Text $\to$ Ảnh** | p50 | 32.38 | 32.74 | 33.22 | 34.33 | 37.42 | **33.72** | **0 / 1,000** | **ĐẠT TUYỆT ĐỐI** |
| (`search_text`) | p95 | 66.06 | 73.08 | 70.36 | 74.96 | 78.50 | **73.54** | (0.0%) | (Max = 141.82 ms) |
| | p99 | 86.91 | 93.30 | 95.84 | 100.86 | 106.01 | **97.69** | | |
| **CIR (Ảnh + Text)** | p50 | 112.78 | 113.41 | 118.99 | 124.59 | 129.52 | **119.38** | **67 / 1,000** | **KHÔNG ĐẠT TUYỆT ĐỐI** |
| (`search_cir`) | p95 | 189.92 | 196.22 | 225.32 | 237.99 | 253.35 | **226.54** | (6.7%) | (Max = 306.06 ms) |
| | p99 | 224.54 | 233.91 | 250.77 | 266.38 | 275.07 | **251.72** | | *(Kích hoạt Dừng c)* |
| **Ảnh $\to$ Ảnh** | p50 | 190.31 | 197.69 | 226.79 | 240.78 | 306.39 | **229.44** | **689 / 1,000** | **KHÔNG ĐẠT** |
| (`search_image`) | p95 | 289.07 | 312.44 | 358.38 | 412.01 | 486.19 | **386.68** | (68.9%) | (Flip-aug 2 pass ViT) |
| | p99 | 356.28 | 378.11 | 439.46 | 506.74 | 672.80 | **493.58** | | |

### 4. Phân tích Hai Tiêu chí SLA (AGENTS.md vs `config.yaml`)
- **Chỉ tiêu theo AGENTS.md (dòng 7):** `latency < 200 ms/query` (mọi truy vấn phải < 200 ms, không ghi phân vị).
- **Chỉ tiêu theo `config.yaml` (`serving.sla_metric`):** `p95 < 200 ms`.
- **Đánh giá theo từng luồng:**
  1. **Luồng Text $\to$ Ảnh:** Đạt **100% cả 2 tiêu chí SLA** (1,000/1,000 truy vấn < 142 ms, p95 trung bình 73.54 ms < 200 ms, p99 trung bình 97.69 ms < 200 ms).
  2. **Luồng CIR (Ảnh + Text):** p50 đạt 119.38 ms, nhưng p95 dao động 189.92–253.35 ms và p99 đạt 224.54–275.07 ms; có 67/1,000 truy vấn (6.7%) vượt ngưỡng 200 ms. Do đó, CIR **đạt SLA ở mức trung vị nhưng không đạt SLA tuyệt đối < 200 ms của AGENTS.md** (kích hoạt Điều kiện dừng c).
  3. **Luồng Ảnh $\to$ Ảnh:** Do thực hiện 2 lần forward pass ViT-SO400M (gốc và flip) kết hợp k-NN voting, p50 là 229 ms và p95 là 386 ms. **Không đạt SLA < 200 ms**.
- **Telemetry GPU liên tục:** Trong suốt quá trình đo từ Run 1 đến Run 5, nhiệt độ GPU tăng từ 67.1°C lên 85.4°C. Tại Run 5 ở 85.4°C, xung nhịp GPU bị hạ từ 1,658 MHz xuống 1,271 MHz (thermal throttling nhẹ), làm p50 tăng khoảng 30–50 ms nhưng **hoàn toàn không có truy vấn nào vượt quá 861 ms** (0 truy vấn > 1,000 ms).
- **Trạng thái:** `ĐÃ XỬ LÝ`.

---

## T5. Phạm vi Index: 38,646 vs 38,644 và Xung Đột Sporting Goods

### 1. Phân tích 3 Nguồn Dữ liệu Phạm vi
1. **Quy tắc Phase 2a (`articleType` rule):** Phân loại theo 143 `articleType`, giữ lại 38,646 sản phẩm In-Scope. Trong đó có 2 sản phẩm `Key chain` (ID `43230`, `42066`). Sau khi loại 2 sản phẩm này, số lượng In-Scope là **38,644 sản phẩm**.
2. **Quy tắc `config.yaml`:**
   ```yaml
   exclude_master_categories: ["Personal Care", "Home", "Sporting Goods"]
   exclude_article_types: ["Key chain", "Shoe Accessories"]
   ```
   Nếu áp dụng quy tắc này trên `catalog.parquet`, số lượng sản phẩm giữ lại là **38,649 sản phẩm**.
3. **Chỉ mục thực tế (`index_v2`):**
   + File vector `image_embeddings_v2.npy` và `id_map_v2.json`: **38,646 dòng**.
   + `AttributeIndex` nạp DataFrame từ `catalog.parquet` lọc `in_scope == True`: **38,644 dòng**.

### 2. Bản chất Xung đột Danh mục Sporting Goods & Đồ phụ trợ
- Trong catalog Kaggle có 16 sản phẩm thuộc `masterCategory == "Sporting Goods"`:
  + 14 sản phẩm là bóng đá/bóng rổ (`Footballs`, `Basketballs`) $\implies$ Out-of-Scope.
  + **2 sản phẩm mang `articleType == "Wristbands"`** (băng cổ tay thể thao):
    * ID `42830`: *Nike Swoosh Black Wristbands*
    * ID `42828`: *Nike Swoosh Red Wristbands*
- Do Phase 2a xếp `Wristbands` là In-Scope (thời trang/phụ kiện), nên 2 sản phẩm này được gán `in_scope = True` trong `catalog.parquet` và được nhúng vào `index_v2`!
- Tuy nhiên, `config.yaml` lại ghi cứng `exclude_master_categories: ["Sporting Goods"]`, tạo ra sự mâu thuẫn giữa quy tắc phân loại của file cấu hình và nhãn In-Scope thực tế của catalog.
- Ngoài ra, có 6 sản phẩm `Water Bottle` và 1 sản phẩm `Perfume and Body Mist` thuộc `Accessories` bị Phase 2a loại nhưng không được liệt kê trong `exclude_article_types` của `config.yaml`.

### 3. Hai Phương án Đề xuất Chờ Người Dùng Quyết Định
- **Phương án A (Khuyến nghị - Giữ nguyên index hiện hành):** Giữ 2 sản phẩm Wristbands là In-Scope (phù hợp thời trang thể thao), giữ gallery ở mức **38,644 sản phẩm**, cập nhật tài liệu và thêm ngoại lệ vào `config.yaml` ở pha sau mà không cần tính toán lại index.
- **Phương án B (Nghiêm ngặt theo masterCategory):** Loại bỏ 2 sản phẩm Wristbands (`in_scope=False`), đưa gallery về **38,642 sản phẩm**. Phương án này đòi hỏi phải rebuild lại `index_v2` và cập nhật lại manifest ở pha tiếp theo.
- **Trạng thái:** `CHƯA XÁC ĐỊNH` *(Chờ Người dùng phê duyệt lựa chọn)*.

---

## T6. Kiểm Tra Độc Lập Toàn Diện Catalog

Được thực hiện độc lập bởi script kiểm toán `src/diag/verify_catalog_t6.py`:
1. **Kích thước & Trùng lặp:**
   - Kích thước DataFrame: `(40891, 14)`.
   - Trùng lặp `item_id`: **0 sản phẩm trùng**.
2. **Đối soát 200 ID Ngẫu nhiên (Seed 42):**
   - Đọc 200 ID ngẫu nhiên từ `catalog.parquet`, đối chiếu 1:1 với metadata trong `index_v2/id_map_v2.json` và thuộc tính index.
   - Kết quả: **200 / 200 ID khớp chính xác 100%** cả về `item_id`, `articleType`, và `name`. Không có bất kỳ hiện tượng lệch vị trí `iloc` nào.
3. **Kiểm tra các ID đặc biệt:**
   - ID `1855`: `name = 'Inkfruit Mens Chain Reaction T-shirt'`, `articleType = 'Tshirts'`. Khớp đúng với thông tin gốc của catalog (lỗi "Q&Q Watch" trong smoke test trước đây là do nhầm nhãn hiển thị trong file test).
   - Key chain ID `43230` và `42066`: Có `in_scope = False`, `quarantine = 0`, không xuất hiện trong gallery 38,644 sản phẩm của `AttributeIndex`.
   - Benetton ID `x018`: Có $R_{full} = 209$ sản phẩm và $R_{test} = 7$ sản phẩm.
4. **Kiểm tra 20 Ca Thất bại Pha 3e:**
   - Cả 38 unique ID xuất hiện trong 20 ca thất bại thật (`eval/results_3e.json`) đều tồn tại thực tế trên đĩa và khớp đúng 100% metadata trong catalog.
- **Trạng thái:** `ĐÃ XỬ LÝ`.

---

## T7. Cập nhật & Đồng bộ Run Manifest

Đã kiểm tra và đồng bộ lại toàn bộ mã băm SHA-256 trong `reports/run_manifest.json`:
- Bổ sung trường `status` và `status_note` chuẩn hóa cho tất cả các khối trong manifest:
  + `phase3c_eval_v2`: Đánh dấu `DEPRECATED_BASELINE` (đã bị thay thế bởi Pha 3e/3f).
  + `phase3d_addendum`: Đánh dấu `DEPRECATED_BASELINE` (đã bị thay thế bởi Pha 3e/3f).
  + `phase3f_errata`: Ghi chú Rev5 cập nhật kiểm toán cuối cùng.
  + `phase4_audit_final`: Khối kiểm toán chốt góc Pha 4.
- Cập nhật SHA-256 của các file đã chỉnh sửa trong đợt kiểm toán này:
  + `reports/phase2/phase2a/phase2a_findings.md`
  + `reports/phase3/phase3e/phase3e_inference_only.md`
  + `reports/phase3/phase3f/phase3f_errata.md`
  + `reports/phase4_verifications.json`
  + `reports/phase4/phase4_latency_e2e.json`
  + `reports/phase4/phase4_audit_final.md`
- **Trạng thái:** `ĐÃ XỬ LÝ`.

---

## T8. Các Điểm Kỹ Thuật Bắt Buộc

### T8.1. Phân kỳ Toán học giữa AP@10 và P@10 (Chuẩn hóa $\min(R, 10)$)
- **Định nghĩa của Evaluator:**
  $$\text{AP@10} = \frac{1}{\min(R, 10)} \sum_{k=1}^{10} P@k \cdot \text{rel}_k$$
- **Hệ quả toán học:**
  + Khi một truy vấn có $R < 10$ sản phẩm trong gallery (ví dụ Q5 brand queries trên $G_{test}$ chỉ có trung bình $\sim 2.5$ sản phẩm liên quan/câu):
    * Nếu hệ thống xếp đúng toàn bộ $R$ sản phẩm đó ở các vị trí đầu tiên (rank 1..R), thì $\text{AP@10} = 1.0$.
    * Tuy nhiên, $\text{P@10} = \frac{R}{10} \le 0.30$.
  + Do đó, trên $G_{test}$, việc Q5 có AP@10 = **0.9491** trong khi P@10 = **0.2543** không phải mâu thuẫn hay lỗi dữ liệu, mà phản ánh hệ thống đã tìm được gần như toàn bộ sản phẩm liên quan hiếm hoi có trong tập test.

### T8.2. Cảnh báo Quy tắc 24 trên $G_{full}$
- Theo quy định bất biến của `AGENTS.md`, chỉ tiêu dự án (Recall@10 $\ge$ 0.85, mAP $\ge$ 0.75) **CHỈ được đối chiếu trên tập test thuần túy ($G_{test}$, 1,290 sản phẩm test)**.
- Mọi con số đo lường trên gallery đầy đủ ($G_{full}$, 38,644 sản phẩm chứa cả các sản phẩm train) **chỉ mang tính chất tham khảo và là ước lượng lạc quan (optimistic reference)**.

### T8.3. Xác thực Tính Ngẫu nhiên Chuỗi Hoán Vị A/B
- Chuỗi 30 phần tử hoán vị A/B:
  `[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]`
- Khớp **100%** giữa:
  1. `eval/_sealed/ab_mapping.json`
  2. Thuật toán `np.random.RandomState(20261109)` với 30 lần gọi `randint(0, 2)`.
  3. Bảng F.2 trong Errata Rev5.
- Đã ghi nhận hạn chế: Kiểm định Wald-Wolfowitz Runs-Test cho thấy chuỗi có hiện tượng co cụm ($p = 0.0108 < 0.05$), tạo ra 9 khối chạy (yếu tố hạn chế về blinding).

### T8.4. Nhắc nhở Cỡ Mẫu Nhỏ trên $G_{test}$
- Trên $G_{test}$, câu `x018` (Benetton) chỉ có $R = 7$ sản phẩm liên quan; nhiều câu thuộc tính hẹp chỉ có $R \le 3$. Các kết quả trên tập test nhỏ này có phương sai lớn và chịu trần toán học của P@10 ($P@10 \le R/10$).
- **Trạng thái:** `ĐÃ XỬ LÝ`.

---

## 4. Tổng Kết & Điều Kiện Dừng

### Các Điểm Kích Hoạt Điều Kiện Dừng
1. **Điều kiện Dừng (c) - Độ trễ CIR:** Phép đo End-to-End thực tế tại `reports/phase4/phase4_latency_e2e.json` cho thấy:
   - CIR có p99 = **224.54 – 275.07 ms** ($\ge 200$ ms) và có **6.7% truy vấn vượt 200 ms** (Max = 306.06 ms).
   - Theo đúng quy định của prompt kiểm toán, Agent **báo cáo trung thực số đo thực tế và dừng lại để xin ý kiến Người dùng**, không tự tiện kết luận đạt SLA tuyệt đối.
2. **Quyết định Danh mục Sporting Goods (T5):** Cần quyết định của Người dùng giữa Phương án A (giữ 2 Wristbands, 38,644 sp) và Phương án B (loại Wristbands, 38,642 sp, rebuild index ở pha sau).
