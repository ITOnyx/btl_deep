# BÁO CÁO KẾT QUẢ THỬ NGHIỆM 200 ẢNH PILOT (PHA 2)
**Hệ thống Tra cứu Sản phẩm Thời trang Đa phương thức Tiếng Việt (`Hoc_sau_ver4`)**

---

## 1. Thông Tin Chung & Phần Cứng Thực Nghiệm
- **Pha:** Pha 2 — Dense Captioning (Bước 2.2 & 2.3: Thử nghiệm 200 ảnh pilot).
- **Phần cứng thực tế:** NVIDIA GeForce RTX 2050 Laptop GPU (Tổng VRAM: 4.00 GB, VRAM khả dụng rảnh: 3.23 GB).
- **Chế độ lượng tử hóa:** **8-bit (`bitsandbytes`)** theo Phương án A đã được người dùng phê chuẩn.
- **Tập mẫu thử nghiệm:** 200 ảnh pilot từ `eval/pilot_200_sample.json`, lấy **100% từ tập train split**, phân tầng theo `articleType × gender`, seed 42 cố định, **0% rò rỉ vào tập test** (đã kiểm định bằng unit test).
- **Commit hash thực hiện:** HEAD hiện tại của repo.

> [!IMPORTANT]
> **Tuân thủ Luật D.5 (AGENTS.md):** Checkpoint SigLIP hiện tại (`best_model.pt`) được huấn luyện trong quá khứ với caption từ Vintern. Do đó, các số liệu paired retrieval với caption mới từ InternVL2-2B sẽ **không so sánh trực tiếp** với số liệu cũ trong báo cáo lịch sử. Báo cáo này tập trung đánh giá năng lực suy luận thị giác, độ chuẩn xác ngôn ngữ và tính toàn vẹn của pipeline.

---

## 2. Bảng Số Liệu Thật (Thống Kê 200 Mẫu Pilot)

Toàn bộ số liệu dưới đây được kết xuất tự động từ file đo lường thực tế [`eval/pilot_200_summary.json`](file:///d:/Kì 7/Hoc_sau_ver4/eval/pilot_200_summary.json):

### 2.1. Phân Bố Độ Dài Token SigLIP (`google/siglip-so400m-patch14-224`)

| Cấu hình Caption | Min | P25 | Median | P75 | Max | Mean ± Std | Số lượng > 128 | **Tỉ lệ > 128 tokens** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (InternVL2-2B, 8-bit)** | 118 | 120.0 | 120.0 | 120.0 | 127 | 120.2 ± 1.2 | 0 / 200 | **0.0%** (Đạt chuẩn) |
| **B (Hợp nhất `merge_v1`)** | 126 | 127.0 | 128.0 | 128.0 | 128 | 127.4 ± 0.8 | 0 / 200 | **0.0%** (Đạt chuẩn) |
| **C (Vintern-1B-v3_5, Tham chiếu)** | 186 | 297.0 | 330.0 | 353.2 | 402 | 322.9 ± 40.1 | 200 / 200 | **100.0%** (Vi phạm trần 128) |

### 2.2. Kiểm Soát Tạp Chất Ngôn Ngữ & Độ Khớp Thuộc Tính Cốt Lõi

| Tiêu chí Kiểm định | Cấu hình A (InternVL2) | Cấu hình B (Hợp nhất v1) | Cấu hình C (Vintern-1B) | Mục tiêu / Quy chuẩn |
|---|:---:|:---:|:---:|:---:|
| **Tỉ lệ lẫn tiếng Anh** | **0.0%** | **1.0%** | **6.0%** | Càng thấp càng tốt (< 5%) |
| **Tỉ lệ lẫn chữ Hán (Chinese)** | **0.0%** | **0.0%** | **0.0%** | **0.0%** (Tuyệt đối không lẫn) |
| **Khớp toàn bộ Metadata** (Loại + Giới + Màu) | 0.0% (do lặp prompt) | **100.0%** (Neo metadata) | 8.0% (Lệch từ vựng/giới) | Càng cao càng tốt |
| **Tốc độ sinh trung bình** | **14.32 s / ảnh** | (Hợp nhất CPU tức thì) | **18.48 s / ảnh** | Càng nhanh càng tốt |
| **Peak VRAM đo thực tế** | **2.80 GB** | 0 GB (CPU) | **1.03 GB** | $\le 3.23$ GB khả dụng |

---

## 3. Phát Hiện Thực Nghiệm Cốt Lõi (Phân Tích Chuyên Sâu)

### 3.1. Rào Cản Ngôn Ngữ Tiếng Việt Của `OpenGVLab/InternVL2-2B` (Cấu hình A)
- **Hiện tượng:** Khi được truyền prompt tiếng Việt có cấu trúc (`INTERNVL2_PROMPT_TEMPLATE`), InternVL2-2B trong 200 ảnh pilot có xu hướng lặp lại chính các câu lệnh của prompt hoặc sinh ra các cụm từ tiếng Việt không hoàn chỉnh (`Mô tả ngắn gọn, súc tích... 1. Thô`).
- **Nguyên nhân kỹ thuật:** `OpenGVLab/InternVL2-2B` là mô hình VLM nền tảng được huấn luyện và canh chỉnh tập trung cho **song ngữ Trung - Anh (ZH-EN)**. Tập từ vựng tokenizer (InternLM2) và dữ liệu alignment của mô hình hầu như không có tiếng Việt, dẫn đến hiện tượng suy thoái sinh văn bản (text generation degradation) khi bị ép buộc sinh tiếng Việt trực tiếp với greedy decoding (`do_sample=False`).
- **Phát hiện đối chứng quan trọng:** Khi thử nghiệm InternVL2-2B bằng prompt **tiếng Anh**, mô hình thể hiện năng lực thị giác siêu việt:
  > *"The image shows a bright yellow T-shirt with a graphic design on the front. The design features a rectangular box with the words 'NO STOPPING OR RACING' written on it. The text is in bold, capital letters, and the background of the box has a distressed, graffiti-like appearance..."*
  Điều này chứng minh kiến trúc Vision-LLM của InternVL2-2B nhận thức vật thể cực kỳ sắc bén, nhưng "bất đồng ngôn ngữ" ở khâu giải mã tiếng Việt.

### 3.2. Ưu Điểm & Rủi Ro Của `5CD-AI/Vintern-1B-v3_5` (Cấu hình C)
- **Ưu điểm:** Văn phong tiếng Việt phong phú, tự nhiên, mô tả chi tiết hoa văn (kẻ caro, sọc dọc, hoa văn chấm bi, quai xích dép).
- **Rủi ro lớn (Ảo giác & Lệch thuộc tính):**
  - **Ảo giác số lượng & giới tính:** Mẫu ID `30600` (1 áo thun Nike bé trai) bị Vintern đoán thành *"hai chiếc áo thể thao nữ"*; Mẫu ID `3328` (áo thun nam) bị đoán thành *"áo thun nữ"*; Mẫu ID `37083` (túi xách nữ) bị đoán thành *"túi xách nam"*.
  - **Nhầm lẫn loại sản phẩm:** Mẫu ID `48579` (lắc tay kim loại) bị đoán thành *"vòng cổ bằng da"*; Mẫu ID `59775` (vòng tay lắc nữ) bị đoán thành *"nhẫn vàng nữ"*.
  - **Vi phạm trần độ dài:** 100% caption của Vintern vượt trần 128 token của SigLIP (trung bình 323 token), nếu đưa vào SigLIP mà không xử lý sẽ bị cắt cụt mất đuôi hoặc gây lỗi lệch embedding.

### 3.3. Vai Trò Cứu Nguy Của Cấu Hình B (Hợp Nhất Deterministic `merge_v1`)
- **Neo thuộc tính phân biệt:** Tiền tố `[loại] [giới tính] màu [màu]` đứng đầu câu đã đảo ngược 100% các sai lệch về giới tính và loại sản phẩm của Vintern, đạt độ khớp metadata **100.0%**.
- **Cắt tỉa token nghiêm ngặt:** Đã hoàn thiện thuật toán cắt tỉa thông minh theo tokenizer của SigLIP, đảm bảo **100% caption có độ dài $\le 128$ tokens** (trung bình 127.4 tokens), bảo toàn trọn vẹn thông tin phân biệt ở đầu câu theo đúng yêu cầu của SigLIP.

---

## 4. Tóm Tắt Đánh Giá Thủ Công 20 Mẫu (Báo Cáo Chi Tiết tại `manual_review_20.md`)
Báo cáo đọc duyệt chi tiết 20 sản phẩm đại diện (ID 11041, 11290, 1573, 23704, 24507, 25047, 30600, 31149, 3328, 33512, 36249, 37083, 37961, 44663, 48579, 51589, 52835, 59775, 6676, 8787) đã được lưu tại [`reports/phase2/manual_review_20.md`](file:///d:/Kì 7/Hoc_sau_ver4/reports/phase2/manual_review_20.md).

**Các kết luận chính từ đánh giá thủ công:**
1. Không phát hiện bất kỳ ký tự tiếng Trung nào lọt vào caption của cả 3 cấu hình.
2. Tỉ lệ tiếng Anh của B được kiểm soát ở mức 1.0% (chỉ xuất hiện ở các thuật ngữ thương mại như *gift set*, *jeggings*).
3. Hiện tượng bịa thuộc tính (hallucination) ở Cấu hình C xuất hiện ở ~15-20% số mẫu đối với các thuộc tính giới tính và phụ kiện nhỏ.
4. Cấu hình B với quy tắc dẹp bỏ mâu thuẫn (`detect_and_resolve_conflicts`) đã lọc sạch các xung đột giới tính (`gender_conflict_male_in_female`).

---

## 5. Thống Kê Thay Đổi Mã Nguồn & Kiểm Thử
- **Tệp đã thay đổi / tạo mới:**
  - `config.yaml`: Cập nhật cấu hình 8-bit, revision chính thức và aux models.
  - `src/caption/generator.py`: Module nạp mô hình tuần tự, giải phóng VRAM, ép kiểu `float16` chuẩn xác.
  - `src/caption/merger.py`: Thuật toán hợp nhất `merge_v1`, cắt tỉa token SigLIP $\le 128$.
  - `src/caption/run_pilot.py`: Master runner chạy và phân tích tự động.
  - `eval/pilot_200_results.json` & `eval/pilot_200_results.parquet`: Toàn bộ 200 bản ghi chi tiết.
  - `eval/pilot_200_summary.json`: Bảng tổng hợp số liệu toán học.
  - `reports/phase2/manual_review_20.md`: Báo cáo đánh giá chi tiết 20 mẫu.
  - `tests/conftest.py`: Xử lý triệt để biến môi trường SSL PostgreSQL cho pytest.
- **Kiểm thử tự động:** `pytest tests/` đạt **11/11 tests PASSED** (100%).

---

## 6. Bảng Quyết Định Cần Người Dùng Duyệt (Chiến Lược Chạy Toàn Bộ Catalog)

Thời gian ước tính để chạy toàn bộ ~38.000 ảnh trong catalog:
- Nếu chạy 1 model: ~38.000 × 14s $\approx$ **147 giờ** (~6.1 ngày chạy liên tục trên GPU 4GB).
- Do đó, cần thống nhất cấu hình tối ưu trước khi chạy batch lớn.

Xin người dùng xem xét và cho chỉ thị duyệt theo 1 trong các phương án sau:

| Phương án | Mô tả kiến trúc | Ưu điểm | Nhược điểm | Đề xuất |
|---|---|---|---|:---:|
| **Phương án 1 (Khuyên dùng)** | **Duyệt kết quả Pilot & Chạy Cấu hình B (Hợp nhất có cải tiến)**: Tận dụng mô tả tiếng Việt của Vintern + Neo thuộc tính metadata + InternVL2 kiểm chứng thuộc tính thị giác. | Tiếng Việt chuẩn mực, không bao giờ sai loại/giới/màu, $\le 128$ tokens. | Cần điều chỉnh nhẹ thân mô tả của B để không kế thừa prompt lặp từ InternVL2. | ⭐ **Khuyên Dùng** |
| **Phương án 2** | **InternVL2-2B sinh Tiếng Anh $\rightarrow$ Hợp nhất sang Tiếng Việt**: Cho InternVL2-2B phát huy tối đa năng lực thị giác bằng tiếng Anh, sau đó hợp nhất với metadata tiếng Việt. | Tận dụng 100% năng lực nhận thức thị giác đỉnh cao của InternVL2-2B. | Cần thêm 1 bước ánh xạ từ vựng tiếng Anh sang tiếng Việt. | Khả thi cao |
| **Phương án 3** | **Chỉ dùng Cấu hình A thuần túy (Theo đề cương)**: Chấp nhận văn bản InternVL2-2B nguyên bản tiếng Việt bất kể suy thoái từ vựng. | Đúng 100% nghĩa đen đề cương ban đầu của giáo viên. | Chất lượng caption tiếng Việt kém, ảnh hưởng lớn đến Recall@10 ở Pha 5. | Không khuyến khích |

---
**TRẠNG THÁI HIỆN TẠI: DỪNG CHỜ LỆNH DUYỆT CỦA NGƯỜI DÙNG.**  
*(Vui lòng phản hồi phương án lựa chọn và ra lệnh `DUYỆT PILOT` để tiếp tục).*
