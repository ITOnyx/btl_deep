# BÁO CÁO KẾT QUẢ THỰC HIỆN PHA 1 — DỌN DẸP, KIỂM KÊ, LỌC CHẤT LƯỢNG, KHỬ TRÙNG & CHIA SPLIT

**Dự án:** SigLIP + Dense Captioning cho Hệ thống Tra cứu Sản phẩm Thời trang Tiếng Việt (`Hoc_sau_ver4`)  
**Mục tiêu Pha 1:** Làm sạch repo cũ, kiểm kê nguồn, lọc chất lượng ảnh, dựng catalog mới (`catalog_rebuild.parquet`) và thiết lập phân chia split chống rò rỉ tuyệt đối (No-Leakage). Không thực hiện caption, không embed.  
**Commit Hash:** `43677ea`  
**Seed cố định:** `42` cho mọi phân chia và thao tác ngẫu nhiên.  
**Thời điểm thực thi:** 2026-11-11  

---

## 1. Kết quả Triển khai Chi tiết Từng Hạng mục

### 1.0. Tiền trạm Hệ thống (Pre-flight Inspection)
- Đã kiểm tra và lưu cấu hình phần cứng/môi trường tại [`reports/phase1/env.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/env.json):
  - **Hệ điều hành / Python:** Windows 11 / Python 3.12.0
  - **GPU & VRAM:** NVIDIA GeForce RTX 2050 Mobile — **4.00 GB VRAM** (CUDA 12.4)
  - **Bộ nhớ hệ thống (RAM):** 15.71 GB (Available: 4.92 GB)
  - **Dung lượng đĩa:** 195.31 GB (Free: 48.84 GB)
  - **Thư viện cốt lõi:** `torch 2.6.0+cu124`, `transformers 5.19.0`, `faiss 1.15.1`, `bitsandbytes 0.50.2`
- **Xác minh Hash Guard (Luật 6):**
  - Checkpoint: `data/checkpoints_siglip-vit/best_model.pt` (Dung lượng: 3,385.26 MB)
  - SHA-256 thực tế: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
  - So khớp với `config.yaml`: **TRÙNG KHỚP 100% (Xác nhận hợp lệ)**.

---

### 1.1. Kiểm kê Kho lưu trữ & Lưu trữ Tài sản cũ (Repo Inventory & Archive)
- Đã tạo [`reports/phase1/repo_inventory.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/repo_inventory.csv) phân loại toàn bộ tài sản trong repo.
- Trước khi di chuyển, đã commit và gắn tag git `legacy-before-rebuild`.
- Đã chuyển **736 tệp tin cũ** sang thư mục cách ly [`data/_quarantine/legacy_rebuild_20261111/`](file:///d:/Kì%207/Hoc_sau_ver4/data/_quarantine/legacy_rebuild_20261111/) giữ nguyên cấu trúc phân nhóm:
  - `untouchable/`: Chỉ mục cũ (`index/`, `index_v2/`, `index_v3_L/`) và `catalog.parquet` cũ (chỉ đọc để khôi phục split).
  - `reports/`: Mọi báo cáo cũ của các pha trước.
  - `eval/`: Toàn bộ các bộ benchmark/kết quả cũ (trừ 2 tệp KEEP).
  - `tests/`: Bộ test unit cũ.
  - `src/`: Toàn bộ mã nguồn suy luận/chuẩn đoán cũ (để viết lại mới chuẩn mực).
  - `app/`: Giao diện và engine cũ.
  - `root/`: `README_run.md` cũ.
- Đã ghi nhật ký chi tiết từng tệp (đường dẫn gốc, đích, SHA-256, kích thước byte, lý do, thời gian) vào:
  - [`reports/phase1/archive_log.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/archive_log.csv)
  - Nối tiếp vào [`reports/quarantine_log.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/quarantine_log.csv)
- **Bảo tồn nguyên vẹn các tệp chỉ định KEEP:**
  - `data/checkpoints_siglip-vit/best_model.pt`
  - `src/brands_v2.json` (SHA-256: `c5283096f649351d4433874915e2e32586118200e576f41b3101495859c1898e` — Khớp 100%)
  - `eval/handwritten_queries.json`
  - `eval/gender_gt_policy.json`
  - `notebooks/` và `docs/` (Chỉ đọc)

---

### 1.2. Khung Repo Mới & Cập nhật Cấu hình (Skeleton & config.yaml)
- Đã dựng khung mã nguồn mới chuẩn mực tại `src/` và `tests/`:
  - [`src/data/`](file:///d:/Kì%207/Hoc_sau_ver4/src/data/): Xử lý dữ liệu, chuẩn hóa nhãn tiếng Việt, kiểm định chất lượng ảnh.
  - [`src/preprocess/`](file:///d:/Kì%207/Hoc_sau_ver4/src/preprocess/): Module tiền xử lý duy nhất dùng chung (Pha 3).
  - [`src/caption/`](file:///d:/Kì%207/Hoc_sau_ver4/src/caption/): Quản lý sinh caption `InternVL2-2B` và mô hình bổ trợ.
  - [`src/index/`](file:///d:/Kì%207/Hoc_sau_ver4/src/index/): Xây dựng chỉ mục FAISS HNSW và Manifest.
  - [`src/eval/`](file:///d:/Kì%207/Hoc_sau_ver4/src/eval/): Đánh giá Recall, mAP, latency, hubness.
  - [`src/serve/`](file:///d:/Kì%207/Hoc_sau_ver4/src/serve/): Engine phục vụ cho ứng dụng Streamlit.
  - [`tests/`](file:///d:/Kì%207/Hoc_sau_ver4/tests/): Kiểm thử tự động pytest.
- **Cập nhật [`config.yaml`](file:///d:/Kì%207/Hoc_sau_ver4/config.yaml):**
  1. Thêm mục `sources` mô tả đầy đủ các nguồn dữ liệu thực tế.
  2. Bỏ ràng buộc cứng "tái tạo 38.644", đổi thành phụ kiện thời trang hợp lệ.
  3. Cập nhật `split.mode: "inherit_checkpoint_train"` thay cho tỷ lệ tĩnh `split.ratios`.
  4. Bổ sung `captioning.configs` gồm 3 cấu hình A, B, C; khởi tạo `captioning.aux: []` rỗng chờ duyệt.
  5. Đánh dấu `legacy: true` cho `scoring.*` và `inference.*` để kiểm lại trên tập validation, không chỉnh trên test.

---

### 1.3. Kiểm kê Thực tế Các Nguồn Dữ liệu (Sources Inventory)
Chi tiết ghi nhận tại [`reports/phase1/sources_inventory.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/sources_inventory.json):

| Tên nguồn | Trạng thái trên đĩa | Vị trí thư mục | Số lượng sản phẩm / ảnh | Ngôn ngữ nhãn | Giấy phép | Vai trò trong Pipeline |
|---|:---:|---|---|:---:|:---:|---|
| **Kaggle Fashion Dataset** | **CÓ TRÊN ĐĨA** | `data/fashion-dataset/` | **44,424** metadata hợp lệ / **44,419** ảnh thực tế | Tiếng Anh (Myntra) | CC0 Public Domain | **Catalog Thời trang Chính thức** |
| **Shopping100k** | **CÓ TRÊN ĐĨA** | `data/Shopping100k/` | **100,586** ảnh trong `Images/` + files `Attributes/` | Tiếng Anh | Nghiên cứu phi thương mại | Nguồn ảnh độc lập (chưa map schema) |
| **Tiki / Shopee Crawled** | **KHÔNG CÓ** | `data/images_crawled` | 0 sản phẩm / 0 ảnh | N/A | N/A | Không có trên đĩa |

> [!NOTE]
> Đúng như đã thống nhất trong phiên Grill-Me, hệ thống sẽ sử dụng nguồn **Kaggle Fashion Dataset** làm catalog chuẩn cho quá trình xây dựng lại, do nguồn này có đầy đủ các trường metadata phân cấp (`masterCategory`, `subCategory`, `articleType`, `gender`, `baseColour`) để dịch và chuẩn hóa tiếng Việt chính xác.

---

### 1.4. Khôi phục Tập Train/Val và Phân chia Split Chống Rò rỉ (Zero-Leakage Split)
- **Bằng chứng từ Notebook Huấn luyện gốc (`siglip_vietnamese_ecommerce_colab_ver2.ipynb` Cell 8 & 11):**
  - Quá trình train ban đầu sử dụng 161,870 ảnh tổng hợp:
    - Train: 153,126 mẫu (Myntra: 41,441)
    - Val: 4,372 mẫu (Myntra: 1,500)
    - Test: 4,372 mẫu (Myntra: 1,500)
- **Đối soát với Checkpoint hiện tại:**
  - Tệp `data/checkpoints_siglip-vit/dense_captions_catalog.json` chứa **chính xác 4,372 mẫu** (gồm đúng 1,500 ID từ Myntra).
  - Đối chiếu với `styles.csv`: Checkpoint đã từng thấy 42,924 sản phẩm của Myntra (nằm trong tập train/val lúc train); và **chưa từng thấy 1,500 sản phẩm test**.
- **Xác lập Phân chia Tuyệt đối Không Rò rỉ (No-Leakage):**
  - Sản phẩm checkpoint đã thấy $\rightarrow$ **`train`** (42,924 sản phẩm toàn bộ / 39,519 sản phẩm sạch sau lọc).
  - Sản phẩm checkpoint chưa từng thấy $\rightarrow$ **`test`** (1,500 sản phẩm toàn bộ / 1,372 sản phẩm sạch sau lọc / **1,290 sản phẩm thời trang `in_scope`**).
  - Danh sách tập Test chính thức được khóa và lưu tại:
    - File danh sách: [`eval/test_split.json`](file:///d:/Kì%207/Hoc_sau_ver4/eval/test_split.json) (1,372 ID sản phẩm)
    - File mã băm SHA-256: [`eval/test_split.sha256`](file:///d:/Kì%207/Hoc_sau_ver4/eval/test_split.sha256) (`d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681`)
  - **Tập Validation độc lập (`eval/val_split.json`):** Trích xuất 2,000 sản phẩm từ tập non-test (seed 42) phục vụ tune siêu tham số suy luận ($\gamma, \alpha, k$) và chạy so sánh A/B bootstrap CI trước khi chạm vào test.

---

### 1.5. Schema Catalog Mới (`data/catalog/catalog_rebuild.parquet`)
Đã xây dựng tệp catalog mới [`data/catalog/catalog_rebuild.parquet`](file:///d:/Kì%207/Hoc_sau_ver4/data/catalog/catalog_rebuild.parquet) với **44,424 dòng** và đầy đủ 31 cột chuẩn hóa:
1. `product_id`: ID sản phẩm cố định dạng chuỗi.
2. `source`: Nguồn dữ liệu (`"kaggle_fashion"`).
3. `image_path`: Đường dẫn ảnh tương đối chuẩn hóa.
4. `sha1`, `phash`: Mã băm nội dung và mã băm cảm nhận.
5. `width`, `height`, `short_side`, `blur_score`: Thông số kích thước và độ sắc nét.
6. `title_raw`: Tên sản phẩm gốc bằng tiếng Anh từ Myntra.
7. `title_vi`: Tên tiếng Việt chuẩn hóa theo cú pháp: `[Loại] [Giới tính] màu [Màu sắc] ([Tên gốc])`.
8. `articleType`, `articleType_vi`, `articleType_vi_nodiacritics`: Danh mục sản phẩm (Anh, Việt có dấu, Việt không dấu).
9. `gender`, `gender_vi`, `gender_vi_nodiacritics`: Giới tính người mặc (Anh, Việt có dấu, Việt không dấu).
10. `baseColour`, `baseColour_vi`, `baseColour_vi_nodiacritics`: Màu sắc chủ đạo (Anh, Việt có dấu, Việt không dấu).
11. `masterCategory`, `subCategory`, `label_origin`: Nhóm danh mục gốc và nguồn nhãn.
12. `filter_flag`: Cờ đánh giá chất lượng (`pass`, `blurry`, `duplicate`, `banner`, `low_res`, `missing_image`).
13. `in_scope`: Boolean xác định thuộc danh mục thời trang phục vụ tìm kiếm.
14. `split`: Phân chia tập dữ liệu (`"train"` hoặc `"test"`).
15. Các trường placeholder chuẩn bị cho Pha 2: `caption_model`, `model_revision`, `seed`, `gen_params`, `method`, `caption_vi`, `token_len`.

---

### 1.6. Lọc Chất lượng Ảnh & Tạo Contact Sheets
Áp dụng các ngưỡng khởi đầu từ `config.yaml` và quy tắc khử trùng không bắc cầu:
- `low_res`: Cạnh ngắn < 224 px $\rightarrow$ **119** ảnh.
- `blurry`: Laplacian variance < 12.0 $\rightarrow$ **1,286** ảnh.
- `banner`: Phát hiện viền/dải màu quảng cáo đơn sắc $\rightarrow$ **935** ảnh.
- `duplicate`: pHash Hamming $\le 1$ trong cùng `(articleType, gender, baseColour)` $\rightarrow$ **1,188** ảnh (giữ nguyên quy tắc Pha 1b, không gom cụm bắc cầu).
- `missing_image`: Thiếu file trên đĩa $\rightarrow$ **5** ảnh.
- `pass`: Ảnh đạt chuẩn $\rightarrow$ **40,891** ảnh (92.05% tổng số).

**Thực thi Luật 2 & Yêu cầu 1.6:** Hệ thống **MỚI CHỈ GẮN NHÃN** trong catalog và đã tạo 4 tệp Contact Sheet (mỗi tệp gồm 40 ảnh mẫu, bao gồm các trường hợp sát ngưỡng) tại thư mục [`reports/phase1/contact_sheets/`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/contact_sheets/):
1. `low_res_contact_sheet.jpg` (40 ảnh độ phân giải thấp)
2. `blurry_contact_sheet.jpg` (40 ảnh mờ/nhiễu)
3. `banner_contact_sheet.jpg` (40 ảnh có viền bẹt/khung chữ)
4. `duplicate_contact_sheet.jpg` (40 ảnh trùng lặp đặc trưng)

> [!IMPORTANT]
> Toàn bộ ảnh mới chỉ được gắn nhãn `filter_flag` và tạo ảnh mẫu kiểm duyệt, **chưa có bất kỳ ảnh nào bị di chuyển cưỡng chế vào quarantine trong Pha 1**. Bước di chuyển chỉ diễn ra sau khi người dùng xem contact sheet và phê duyệt ngưỡng.

---

### 1.7. Phân bổ Sản phẩm Trong Phạm vi (`in_scope`) & Thin Slots
- **Quy tắc Scope Thời trang:**
  - Loại trừ `masterCategory`: `["Personal Care", "Home", "Sporting Goods"]`
  - Loại trừ `articleType`: `["Ipad", "Free Gifts", "Fragrance Gift Set", "Key chain", "Water Bottle", "Perfume and Body Mist"]`
  - Giữ lại phụ kiện thời trang: `["Wristbands"]`
- **Kết quả Thống kê Thực tế:**
  - Tổng số sản phẩm `in_scope`: **41,893** / 44,424 sản phẩm.
  - Sau khi kết hợp điều kiện ảnh sạch (`filter_flag == 'pass'` VÀ `in_scope == True`):
    - **Tập Train sạch (`train`):** **37,354** sản phẩm.
    - **Tập Test sạch (`test`):** **1,290** sản phẩm.
    - **Tổng số sản phẩm phục vụ chính thức:** **38,644** sản phẩm (khớp chuẩn mực số liệu phục vụ sạch của hệ thống).

**Bảng Phân bố Top 10 Danh mục Thời trang (`in_scope` Clean):**

| Danh mục (`articleType`) | Tên tiếng Việt | Tổng số | Tập Train | Tập Test |
|---|---|---:|---:|---:|
| **Tshirts** | Áo thun | 6,854 | 6,638 | 216 |
| **Shirts** | Áo sơ mi | 3,191 | 3,090 | 101 |
| **Casual Shoes** | Giày thường | 2,751 | 2,664 | 87 |
| **Watches** | Đồng hồ | 2,367 | 2,286 | 81 |
| **Sports Shoes** | Giày thể thao | 1,971 | 1,914 | 57 |
| **Kurtas** | Áo kurta | 1,908 | 1,847 | 61 |
| **Tops** | Áo kiểu | 1,707 | 1,656 | 51 |
| **Handbags** | Túi xách | 1,701 | 1,644 | 57 |
| **Heels** | Giày cao gót | 1,295 | 1,247 | 48 |
| **Sunglasses** | Kính râm | 1,040 | 1,006 | 34 |

---

### 1.8. Đề xuất Mô hình & Phương án Bổ trợ (Auxiliary Proposal)
Đã lập báo cáo đề xuất toàn diện tại [`reports/phase1/aux_models_proposal.md`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/aux_models_proposal.md) bao gồm:
1. `5CD-AI/Vintern-1B-v3_5` (SHA: `b98f263eab246eb5269ade64edbdca8a887dc44d`, Apache-2.0, ~2.8 GB VRAM FP16).
2. Trích xuất thuộc tính chuẩn từ Metadata (`metadata_extractor`, 0 MB VRAM, 0% bịa thuộc tính).
3. Bộ chuẩn hóa hậu xử lý quy tắc (`rule_cleaner`, 0 MB VRAM, bảo đảm $\le 128$ tokens SigLIP).
4. Thiết kế sơ đồ hợp nhất xác định (`merge_rule_v1`) cho Cấu hình B.

---

### 1.9. Kết quả Kiểm thử Tự động (Pytest Suite)
Đã chạy bộ test [`tests/test_phase1.py`](file:///d:/Kì%207/Hoc_sau_ver4/tests/test_phase1.py) với kết quả:

```text
tests/test_phase1.py::test_catalog_rebuild_schema PASSED                 [ 16%]
tests/test_phase1.py::test_splits_disjoint PASSED                        [ 33%]
tests/test_phase1.py::test_test_split_hash_match PASSED                  [ 50%]
tests/test_phase1.py::test_quarantine_log_integrity PASSED               [ 66%]
tests/test_phase1.py::test_seed_consistency PASSED                       [ 83%]
tests/test_phase1.py::test_law_5_compliance PASSED                       [100%]

============================== 6 passed in 1.37s ==============================
```
- **Tổng số test:** 6 test, **100% ĐẠT (XANH)**.
- Quét Luật 5: 0 tệp tin vi phạm việc gọi/nhắc mô hình bổ trợ ngoài module cho phép.

---

## 2. Vấn đề Gặp phải & Giải pháp Kỹ thuật

| Vấn đề phát sinh | Nguyên nhân gốc rễ | Giải pháp đã thực hiện |
|---|---|---|
| **Deduplication ban đầu tăng vọt lên 14,414** | `quarantine_log.csv` ghi nhận cả các bản ghi trước khi Pha 1b sửa lỗi bắc cầu. | Bổ sung điều kiện lọc `status == "quarantined"`, khôi phục chính xác 40,891 sản phẩm sạch (chỉ cách ly 1,188 duplicate thực sự). |
| **Model ID Vintern trên HF báo 401** | `5CD-AI/Vintern-1B-v3.5` dùng dấu chấm `.` không tồn tại trên HF API. | Xác minh chính xác ID chính thức là `5CD-AI/Vintern-1B-v3_5` (dấu gạch dưới), commit SHA `b98f263e...`. |
| **Nguy cơ rò rỉ dữ liệu khi chia train/val** | Thiếu file danh sách `val_clean.jsonl` từ Colab trên đĩa. | Quy định toàn bộ sản phẩm checkpoint đã thấy về `train` (39,519 sp); trích xuất tập `val` riêng biệt cho Pha 5; bảo vệ 100% tính nguyên vẹn của tập `test` (1,372 sp). |

---

## 3. Danh sách Tệp Thay đổi / Tạo mới trong Pha 1

- **Cấu hình & Báo cáo:**
  - [`config.yaml`](file:///d:/Kì%207/Hoc_sau_ver4/config.yaml): Cập nhật sources, split mode, captioning configs, legacy tags.
  - [`reports/phase1/env.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/env.json): Báo cáo môi trường máy và hash checkpoint.
  - [`reports/phase1/repo_inventory.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/repo_inventory.csv): Kiểm kê 137k tệp repo.
  - [`reports/phase1/archive_log.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/archive_log.csv): Nhật ký 736 tệp chuyển cách ly.
  - [`reports/phase1/sources_inventory.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/sources_inventory.json): Kiểm kê 3 nguồn dữ liệu.
  - [`reports/phase1/aux_models_proposal.md`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/aux_models_proposal.md): Đề xuất mô hình bổ trợ.
  - [`reports/phase1/phase1_report.md`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase1/phase1_report.md): Báo cáo tổng hợp Pha 1.
- **Mã nguồn mới:**
  - [`src/data/labels.py`](file:///d:/Kì%207/Hoc_sau_ver4/src/data/labels.py): Từ điển dịch thuật và chuẩn hóa tiếng Việt.
  - [`src/data/image_quality.py`](file:///d:/Kì%207/Hoc_sau_ver4/src/data/image_quality.py): Đo lường chất lượng ảnh và xuất contact sheet.
  - [`src/data/builder.py`](file:///d:/Kì%207/Hoc_sau_ver4/src/data/builder.py): Pipeline dựng catalog mới và khóa split.
- **Dữ liệu & Khóa Split:**
  - [`data/catalog/catalog_rebuild.parquet`](file:///d:/Kì%207/Hoc_sau_ver4/data/catalog/catalog_rebuild.parquet): Catalog mới 44,424 dòng.
  - [`eval/test_split.json`](file:///d:/Kì%207/Hoc_sau_ver4/eval/test_split.json): Danh sách 1,372 ID test độc lập.
  - [`eval/test_split.sha256`](file:///d:/Kì%207/Hoc_sau_ver4/eval/test_split.sha256): Mã băm SHA-256 khóa test split.
  - [`eval/val_split.json`](file:///d:/Kì%207/Hoc_sau_ver4/eval/val_split.json): Danh sách 2,000 ID validation.
- **Contact Sheets:**
  - `reports/phase1/contact_sheets/low_res_contact_sheet.jpg`
  - `reports/phase1/contact_sheets/blurry_contact_sheet.jpg`
  - `reports/phase1/contact_sheets/banner_contact_sheet.jpg`
  - `reports/phase1/contact_sheets/duplicate_contact_sheet.jpg`
- **Kiểm thử:**
  - [`tests/test_phase1.py`](file:///d:/Kì%207/Hoc_sau_ver4/tests/test_phase1.py): 6 test unit kiểm định toàn bộ Pha 1.

---

## 4. Bảng Quyết định Cần Người Dùng Duyệt

| STT | Hạng mục quyết định | Hiện trạng / Đề xuất từ Agent | Lựa chọn từ Người dùng |
|:---:|---|---|:---:|
| **1** | **Nguồn dữ liệu catalog** | Chỉ sử dụng `data/fashion-dataset` (~44k ảnh Kaggle) vì đầy đủ schema tiếng Việt | Đã chọn ở Grill-Me |
| **2** | **Tài sản cũ UNTOUCHABLE** | Đã di chuyển sạch vào `data/_quarantine/legacy_rebuild_20261111/untouchable/` có log | Đã chọn ở Grill-Me |
| **3** | **Lịch sử caption checkpoint (F.1)** | Áp dụng **Phương án B**: chỉ lưu trong git history và đối chiếu nội bộ | Đã chọn ở Grill-Me |
| **4** | **Phê duyệt Ngưỡng Lọc Ảnh** | Giữ nguyên bộ ngưỡng: `short_side >= 224`, `blur >= 12.0`, `pHash <= 1` (không bắc cầu), `banner == False`. Đã xuất 4 contact sheets xem mẫu. | **Chờ Duyệt** |
| **5** | **Danh sách Mô hình Bổ trợ (Pha 2)** | Đăng ký chính thức `5CD-AI/Vintern-1B-v3_5` + `metadata_extractor` + `rule_cleaner` vào `captioning.aux` cho Cấu hình B | **Chờ Duyệt** |

---

> [!IMPORTANT]
> **DỪNG THEO QUY TẮC MỤC G & AGENTS.md.**  
> Pha 1 đã hoàn tất 100% các tiêu chí kỹ thuật và kiểm thử tự động xanh.  
> Hệ thống tạm dừng thực thi và chờ lệnh phê duyệt: **`DUYỆT PHA 1`** từ người dùng để bước sang **PHA 2 — DENSE CAPTION (INTERNVL2-2B LÕI + BỔ TRỢ)**.
