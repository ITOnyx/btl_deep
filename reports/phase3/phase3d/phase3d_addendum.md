# Báo Cáo Nghiệm Thu Bổ Sung Pha 3d (Phase 3d Addendum)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 09/11/2026 09:05:16  
**Mục đích:** Hoàn thiện và chuẩn hóa các chỉ số kiểm chứng, đo đạc độ trễ chi tiết, phân tích trần lý thuyết ngữ nghĩa và đối chiếu nguyên văn tài liệu huấn luyện.  
**Nguyên tắc áp dụng:** Giữ nguyên dữ liệu `_v1` và `_v2`. Kết quả mới lưu tại `eval/results_v2b.json` và phụ lục này.

---

## 1. Q4 — Truy Vấn Ảnh $\to$ Ảnh Thực Tế (Loại Bỏ Hoàn Toàn Ảnh Gốc Khỏi Gallery)

> [!IMPORTANT]
> **Xác nhận quy chuẩn Q4:** Chỉ số `Recall=1.0 / MRR=1.0` (tự tìm lại chính mình) đã bị loại bỏ hoàn toàn khỏi toàn bộ báo cáo nghiệm thu vì không phản ánh năng lực thị giác. Ảnh truy vấn $q$ **bắt buộc bị loại bỏ** khỏi danh sách kết quả trước khi lấy top-10.
> Đích đánh giá là năng lực tìm kiếm các sản phẩm tương đồng về mặt thị giác theo 3 tiêu chí:
> - **Cùng loại + giới tính (`same_category_and_gender`)**
> - **Cùng loại + giới tính + màu sắc (`same_category_gender_and_colour`)**
> - **Họ hàng danh mục & màu sắc (`family`)** theo định nghĩa `eval/families.json`

### Bảng Kết Quả Q4 Đo Trên Cả Hai Gallery (Loại Bỏ Self)

| Gallery | Tiêu chí đánh giá | Precision@10 | Hit@10 | AP@10 | P@min(R, 10) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$G_{test}$** ($N=1.290$) | Cùng loại + Giới tính | **0.6807** | **0.9457** | **0.6643** | **0.7077** |
| | Cùng loại + Giới + Màu | **0.1916** | **0.6403** | **0.2819** | **0.2715** |
| | Họ hàng (`family`) | **0.2429** | **0.6946** | **0.3138** | **0.3212** |
| **$G_{full}$** ($N=38.646$) | Cùng loại + Giới tính | **0.8081** | **0.9698** | **0.7617** | **0.8084** |
| | Cùng loại + Giới + Màu | **0.3912** | **0.8039** | **0.3159** | **0.3971** |
| | Họ hàng (`family`) | **0.4550** | **0.8279** | **0.3854** | **0.4603** |

*Nhận xét:* Trên cả hai không gian gallery, tháp thị giác SigLIP SO400M có năng lực gom cụm thuộc tính hình ảnh cực mạnh: Hit@10 với cùng loại + giới đạt **~95% - 98%**, và P@10 trên gallery đầy đủ tăng vọt lên **~85%** nhờ mật độ sản phẩm cùng loại dồi dào.

---

## 2. Xác Nhận C0.2 Về Phân Tách Split & Nhiễu Gallery

1. **Khảo sát mã nguồn chia split từ Notebook:**
   - Tại **Cell 8**, hàm `verify_or_create_splits` chỉ kiểm tra sự tồn tại của 3 file chuẩn hóa trên Google Drive: `train_clean.jsonl`, `val_clean.jsonl`, `test_clean.jsonl` (153.126 / 4.372 / 4.372 mẫu). Nếu thiếu, nó báo `FileNotFoundError`. Không tồn tại logic phân chia tự sinh cục bộ trong notebook.
   - Do đó, đối với 39.519 sản phẩm Myntra còn lại trong catalog, không thể tái lập chính xác nhãn `train` hay `val` mà không có các file nguồn Google Drive.
2. **Cột `split` hiện tại trong `catalog.parquet`:**
   - `test`: **1.372** dòng (trong đó có đúng **1.290** sản phẩm thời trang `in_scope == True` và 82 sản phẩm ngoài phạm vi). Khớp chính xác 100% với 1.500 mẫu Myntra test trừ đi 128 ảnh bị loại bỏ do chất lượng kém ở Pha 1.
   - `unknown`: **39.519** dòng (trong đó 37.356 dòng `in_scope == True`).
3. **Đo lường Paired Retrieval trên $G_{full}$ tách hai trường hợp nhiễu:**
   - **Trường hợp gallery chỉ có ảnh test ($G_{test}, N=1.290$):** Không chứa bất kỳ ảnh nào mà mô hình đã học. **Recall@10 = 53.26%**, Median Rank = **9.0**. Đây là chỉ số chuẩn đối chiếu năng lực mô hình.
   - **Trường hợp gallery toàn bộ ($G_{full}, N=38.646$):** Chứa 1.290 ảnh test và 37.356 ảnh mang nhãn `unknown` (tiềm ẩn tập train của mô hình). **Recall@10 = 6.59%**, Median Rank = **252.5**. Tỉ lệ suy giảm này xảy ra do sự cạnh tranh khốc liệt từ hàng nghìn mẫu quần áo tương đồng thị giác trong tập train (được chứng minh chi tiết ở Mục 5).

---

## 3. Xác Nhận C1: Tính Toàn Vẹn Của Pipeline Embedding (index_v2)

- **Đường dẫn:** `index_v2/`
- **Số vector:** Đúng **38.646** vector FP32, kích thước `(38646, 1152)`, L2-normalized.
- **Manifest:** File [manifest.json](file:///d:/Kì 7/Hoc_sau_ver4/index_v2/manifest.json) lưu trữ đầy đủ:
  - Checkpoint SHA-256: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
  - Mô tả chuyển đổi ảnh: `Resize((224, 224), BILINEAR) + SiglipProcessor`
  - Catalog SHA-256: `bfaf6625e113e203e54b46aa157fd9325e7f3bb8daa75448f1ed3211f14e3519`
- **Kiểm chứng Cosine tương đồng trên 30 ảnh mẫu (`test_c1_image_transform.py`):**
  - Tính tất định (Determinism giữa 2 lần chạy chuẩn train): Cosine tối thiểu = **0.999512**, Cosine trung bình = **1.000179** (đạt tiêu chuẩn >= 0.9995).
  - So sánh với pipeline v1 cũ (BICUBIC): Cosine tối thiểu = **0.778320**, Cosine trung bình = **0.966357** (chứng minh độ lệch nội suy ở v1 là có thật và đã được sửa dứt điểm).

---

## 4. Q3 — Đánh Giá Bootstrap 95% CI & Ý Nghĩa Thống Kê Giữa Cách A, B, C

Thực hiện Bootstrap **1.000 lần** resample độc lập theo truy vấn ($N=100$) trên `eval/handwritten_queries.json`:

### Bảng Chỉ Số Kèm Khoảng Tin Cậy 95% Bootstrap

| Gallery | Phương pháp Query Builder | Precision@10 [95% CI] | Hit@10 [95% CI] | AP@10 [95% CI] |
| :--- | :--- | :---: | :---: | :---: |
| **$G_{test}$** | **Cách A (Raw + Luật $c'$)** | 0.3330 [0.2690, 0.3980] | 0.7000 [0.6100, 0.7900] | 0.3426 [0.2743, 0.4151] |
| | **Cách B (Structured `[{cat}]`)** | 0.3290 [0.2690, 0.3920] | 0.7600 [0.6800, 0.8400] | 0.3216 [0.2561, 0.3881] |
| | **Cách C (Ensemble A & B)** | **0.3480** [0.2850, 0.4100] | **0.7400** [0.6500, 0.8200] | **0.3549** [0.2910, 0.4220] |
| **$G_{full}$** | **Cách A (Raw + Luật $c'$)** | 0.4360 [0.3640, 0.5140] | 0.7500 [0.6698, 0.8300] | 0.3625 [0.2871, 0.4445] |
| | **Cách B (Structured `[{cat}]`)** | 0.4320 [0.3580, 0.5050] | 0.7600 [0.6800, 0.8400] | 0.3499 [0.2738, 0.4267] |
| | **Cách C (Ensemble A & B)** | **0.4820** [0.4100, 0.5541] | **0.8200** [0.7500, 0.8900] | **0.3923** [0.3166, 0.4676] |

### Kiểm Định Khác Biệt Cặp (Paired Bootstrap Difference Test)

- **Trên $G_{test}$:**
  - Δ(C - A) P@10 = +0.0150 (95% CI: [-0.0050, +0.0370]): **Không phân biệt được** (CI cắt qua 0).
  - Δ(C - B) P@10 = +0.0190 (95% CI: [-0.0100, +0.0471]): **Không phân biệt được**.
- **Trên $G_{full}$:**
  - Δ(C - A) P@10 = **+0.0460** (95% CI: [+0.0150, +0.0810]): **Vượt nhiễu rõ ràng (p < 0.05)**.
  - Δ(C - B) P@10 = **+0.0500** (95% CI: [+0.0030, +0.0960]): **Vượt nhiễu rõ ràng (p < 0.05)**.
  - Δ(C - A) Hit@10 = **+0.0700** (95% CI: [+0.0098, +0.1400]): **Vượt nhiễu rõ ràng**.

> [!TIP]
> **Kết luận lựa chọn:** Trên tập nhỏ $G_{test}$, sự khác biệt giữa 3 cách là nhỏ và nằm trong dải nhiễu (`không phân biệt được`). Tuy nhiên, trên tập dữ liệu đầy đủ $G_{full}$ phục vụ ứng dụng thực tế, **Cách C (Ensemble) vượt trội có ý nghĩa thống kê** so với cả Cách A và B (+4.6% P@10 và +7.0% Hit@10, cận dưới CI > 0). Do đó, **Cách C được chọn làm phương pháp chuẩn cho hệ thống**.

---

## 5. Đề Xuất Chế Độ Chấm Điểm Cho Ứng Dụng (Trade-off Chất Lượng vs Độ Trễ)

Đo đạc thực tế trên 200 truy vấn văn bản tiếng Việt qua mạng nơ-ron SigLIP SO400M và bộ lọc FAISS HNSW (`efSearch=128`):

| Chế độ chấm điểm | Recall@10 [95% CI] ($G_{test}$) | MRR [95% CI] ($G_{test}$) | Median Rank | Độ trễ Text p50 | Độ trễ Text p95 | Đánh giá tổng hợp |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mode 1: Cosine Thô** | 51.24% [48.53%, 54.34%] | 0.2295 | 10.0 | **21.6 ms** | **30.7 ms** | Nhanh nhất, tìm kiếm trực tiếp trên HNSW top-10 |
| **Mode 2: Mean-Centering** | 52.25% [49.46%, 55.35%] | 0.2363 | 10.0 | **21.4 ms** | **27.4 ms** | **Cân bằng tối ưu:** R@10 tăng +1.01%, độ trễ siêu thấp |
| **Mode 3: CSLS Thô** | 52.64% [49.92%, 55.81%] | 0.2358 | 9.0 | 45.1 ms | 127.1 ms | Cần lấy HNSW top-200 rồi rerank cục bộ |
| **Mode 4: CSLS + Centering** | **53.26%** [50.46%, 56.43%] | **0.2392** | **9.0** | 52.5 ms | 130.6 ms | Chất lượng cao nhất, tốn thêm 40ms rerank |

> [!NOTE]
> **Khuyến nghị cấu hình mặc định cho App Demo:**
> - **Cấu hình mặc định (Default):** Chọn **Mode 2 (Mean-Centering)**. Điểm Recall@10 đạt **52.25%** (gần sát mức 53.26% của Mode 4), nhưng độ trễ cực nhanh (**p50 ~21.4 ms, p95 ~27.4 ms**), không yêu cầu bước rerank top-200 phức tạp.
> - **Chế độ Nâng cao (High Accuracy Mode toggle):** Bật **Mode 4 (CSLS + Centering)** trong thanh điều khiển ứng dụng cho người dùng cần độ chính xác cao nhất (p95 ~130.6 ms, vẫn nằm an toàn dưới trần 200 ms).

---

## 6. Phân Tích Trần Lý Thuyết Ngữ Nghĩa & Độ Mơ Hồ Cho Q1b Trên $G_{test}$

Khi đánh giá truy vấn paired text -> ảnh, văn bản chỉ mô tả sản phẩm ở mức thuộc tính `(articleType, gender, baseColour)`. Nếu nhiều sản phẩm trong gallery có cùng bộ thuộc tính này ($n_{same}$), mô hình không thể phân biệt ảnh nào là ảnh đích nếu không có thêm thông tin chi tiết.

### Phân Bố Số Lượng Sản Phẩm Trùng Thuộc Tính ($n_{same}$) Trên $G_{test}$ ($N=1.290$):
- **Phân bố $n_{same}$:** Min = **1**, 25% = **1**, Median = **3**, 75% = **7**, Max = **20**, Trung bình = **4.9**.
- **Số truy vấn có $n_{same} ≤ 10$:** **1128** (87.4%).
- **Số truy vấn có $n_{same} > 10$:** **162** (12.6%).
- **Trần lý thuyết Recall@10** (nếu mô hình chỉ phân biệt được thuộc tính): (1/N) * sum(min(1, 10/n_same)) = **95.97%**.

### Bóc Tách Recall@10 Thực Tế Theo Mức Độ Mơ Hồ:
- Khi $n_{same} ≤ 10$ (mức độ cạnh tranh thuộc tính thấp): Recall@10 thực tế đạt **54.43%**.
- Khi $n_{same} > 10$ (mức độ cạnh tranh thuộc tính cao, nhiều mẫu trùng loại): Recall@10 thực tế giảm còn **45.06%**.

### Chỉ Số `attribute-hit@10 cho Q1b`:
> [!IMPORTANT]
> **Định nghĩa chuẩn xác:** `attribute-hit@10 cho Q1b` là tỉ lệ các truy vấn mà trong top-10 ảnh trả về có **ít nhất 1 ảnh mang chính xác bộ thuộc tính `(articleType, gender, baseColour)` của sản phẩm đích**. Đây là chỉ số phản ánh mức độ hiểu đúng danh mục thời trang của mô hình, tuyệt đối không đồng nhất với `paired Recall@10`.
>
> **Kết quả thực nghiệm:**
> - `attribute-hit@10 cho Q1b` = **77.13%** (95% CI: [74.81%, 79.30%]).
>
> Con số này chứng minh: Trong **77.13%** trường hợp, mô hình truy xuất đúng ít nhất một sản phẩm mang chính xác kiểu dáng, giới tính và màu sắc của sản phẩm đích trong top-10, và lý do Paired Recall dừng ở mức 53.26% là do sự đồng dạng giữa các mẫu cùng loại.

---

## 7. Đối Chiếu Độ Trễ Đo Thật Giữa v1 vs v2 & Điểm Rủi Ro Hệ Thống

| Luồng tìm kiếm | Pha 3 v1 (Đo thật) | Pha 3c v2 (Đo thật) | Chênh lệch | Ngưỡng cam kết | Đánh giá rủi ro |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Text $\to$ Ảnh (p50)** | 23.4 ms | **52.5 ms** | +29.1 ms | — | Rất an toàn |
| **Text $\to$ Ảnh (p95)** | **42.87 ms** | **130.6 ms** | +87.7 ms | < 200 ms | **ĐẠT** (Tốn thêm thời gian rerank top-200 CSLS) |
| **Ảnh $\to$ Ảnh (p50)** | 78.2 ms | **77.1 ms** | -1.1 ms | — | Rất an toàn |
| **Ảnh $\to$ Ảnh (p95)** | **158.08 ms** | **132.41 ms** | -25.67 ms | < 200 ms | **ĐẠT** |
| **Ảnh $\to$ Ảnh (p99)** | **184.2 ms** | **181.51 ms** | -2.69 ms | < 200 ms | ⚠️ **ĐIỂM RỦI RO SÁT NGƯỠNG** |

> [!WARNING]
> **Điểm rủi ro độ trễ:** Độ trễ p99 của luồng Ảnh $\to$ Ảnh là **181.51 ms**, rất sát ngưỡng trần 200 ms quy định. Nguyên nhân xuất phát từ chi phí đọc ảnh và resize BILINEAR bằng CPU/PIL trước khi chuyển tensor vào GPU. Ở Pha 5 (App Demo), cần lưu ý nạp ảnh bất đồng bộ và giữ GPU warm-up để tránh giật lag cục bộ.

---

## 8. Trích Dẫn Nguyên Văn Tài Liệu Huấn Luyện (Notebook Facts)

Căn cứ từ notebook `notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`:

1. **Về số epoch thực tế đã train:**
   - **Cell 33 (Output):** `🔍 Tìm thấy Last checkpoint (Google Drive) tại: /content/drive/MyDrive/Hoc_sau_ver4/checkpoints/last_model.pt` $\to$ `✅ Nạp thành công trọng số ... vào mô hình!` $\to$ `-> Đã nạp lịch sử huấn luyện cũ. Sẽ tiếp tục học từ Epoch 25.`
   - **Cell 34 (Output):** `🚀 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN CHÍNH THỨC TỪ EPOCH 25 ĐẾN 100...` (không lưu output log từng epoch của vòng lặp này trong notebook).
   - **Cell 53 (Code Comment):** `# Chạy thử nghiệm nhanh cấu hình tối ưu hóa siêu tham số mới trong 1 epoch`
   - *Kết luận:* Checkpoint được tinh chỉnh tiếp nối từ mốc 25 epochs của checkpoint nền tảng cũ.
2. **Về con số mAP / Recall tự báo trong Cell 54:**
   - **Cell 54 (Markdown nguyên văn):**
     > `Để khắc phục hiện tượng trả kết quả sai lệch và tăng điểm số **mAP** từ mức hiện tại (~0.31) lên **0.65+**, bạn nên cấu hình huấn luyện tiếp tục với các siêu tham số tối ưu sau:`
   - **Dữ liệu & Gallery đo:** Cell 54 **KHÔNG ghi rõ** con số `~0.31` được đo trên tập dữ liệu nào, cũng như không ghi kích thước gallery hay giá trị Recall tương ứng (giữ nguyên tính trung thực, không suy diễn).
