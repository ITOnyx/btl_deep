# Pha 7a — GĐ3: Báo cáo Chuẩn hóa, Lọc theo Ngưỡng Đã Duyệt & Chống trùng

- **Thời gian chạy:** 397.5s | **Seed:** `42`
- **Quy tắc đã duyệt áp dụng thống nhất 2 nguồn:** `min_short_side=224`, `detect_banner=False`, không lọc `blur` (gắn cờ `low_contrast_flag = blur < 12.0`), `pHash <= 1` không bắc cầu chỉ khi đủ `(articleType, gender, baseColour)` đều xác định (thiếu màu chỉ dedup `SHA-1` giống hệt).
- **Đầu ra:** `eval/phase7/clean_candidates_gd3.parquet` (**98,289** dòng), `eval/phase7/candidate_rejects.csv` (**40,850** dòng), `reports/phase7/contact_sheets/gd3_candidates_sample_60.png`.

---

## 1. Số ảnh trước / sau từng bước theo nguồn

| Bước xử lý | Tổng còn lại | `fashion` | `shopping100k` | Bị loại ở bước này |
|---|---:|---:|---:|---:|
| 0. Đầu vào từ GĐ2 (all.parquet) | **139,139** | 40,953 | 98,186 | 0 |
| 1a. Sau kiểm tra đọc được (unreadable) | **139,139** | 40,953 | 98,186 | 0 |
| 1b. Sau lọc low_res (short_side >= 224) | **139,124** | 40,947 | 98,177 | 15 |
| 2a. Sau dedup SHA-1 nội bộ | **139,113** | 40,939 | 98,174 | 11 |
| 2b. Sau dedup pHash<=1 (chỉ nhóm đủ màu) | **139,112** | 40,938 | 98,174 | 1 |
| 3a. Sau loại trùng catalog in_scope (ID/SHA-1/pHash) | **100,528** | 2,354 | 98,174 | 38,584 |
| 3b. Sau loại Keychain & ngoài scope (giữ 94 protected có gắn cờ) | **98,289** | 115 | 98,174 | 2,239 |
| 4. Sau kiểm tra bắt buộc RGB & đường dẫn tuyệt đối | **98,289** | 115 | 98,174 | 0 |

### Chi tiết số dòng bị loại theo `reason` × `source` (`candidate_rejects.csv`)

| Lý do (`reason`) | `fashion` | `shopping100k` | Tổng |
|---|---:|---:|---:|
| `duplicate_catalog_inscope_id_or_sha1` | 38,584 | 0 | **38,584** |
| `duplicate_internal_phash` | 1 | 0 | **1** |
| `duplicate_internal_sha1` | 8 | 3 | **11** |
| `excluded_keychain` | 2 | 0 | **2** |
| `low_res` | 6 | 9 | **15** |
| `out_of_scope` | 2,237 | 0 | **2,237** |

---

## 2. Kiểm toán 100 ảnh `fashion legacy_test` (`seen_in_train='no'`) ngoài catalog

- **Tổng ban đầu:** `100` ảnh (`seen_in_train='no'` và `not in_catalog_any`).
- **Bị loại:** **`6` / 100** ảnh, lý do duy nhất là **`low_res` (`short_side = 150 < 224`)**: `['fashion_15864', 'fashion_15884', 'fashion_32176', 'fashion_5375', 'fashion_5378', 'fashion_5397']` (`unreadable = 0`).
- **Giữ lại & gắn cờ:** **`94` / 100** ảnh (`legacy_test_protected = True`). Phân rã các cờ trên `94` ảnh giữ lại:
  - `cfg_in_scope == True` (hợp lệ thời trang): **88** | `cfg_in_scope == False` (Personal Care ngoài scope — gắn cờ, chưa bỏ): **6** (`['fashion_55581', 'fashion_55038', 'fashion_55021', 'fashion_55111', 'fashion_55506', 'fashion_44125']`)
  - `low_contrast_flag` (`blur < 12.0`): **47**
  - `banner_flag` (cờ Pha 1 cũ, không lọc theo ngưỡng đã duyệt): **39**
  - `catalog_phash_dup_flag` (`pHash<=1` cùng 3 thuộc tính vs `catalog_inscope`): **4** (`['fashion_31202', 'fashion_31193', 'fashion_35652', 'fashion_35650']`)
  - `internal_sha1_dup_flag`: **1** | `internal_phash_dup_flag`: **2**

---

## 3. Thống kê tập ứng viên sạch cuối GĐ3 (`clean_candidates_gd3.parquet`)

- **Tổng ứng viên sạch GĐ3:** **98,289** (`fashion`: **115**, `shopping100k`: **98,174**).
- **Phân bố `seen_in_train`:** `{('fashion', 'no'): 94, ('fashion', 'unknown'): 21, ('shopping100k', 'no'): 1468, ('shopping100k', 'unknown'): 96706}`.
- **Cờ `low_contrast_flag` (`blur < 12.0`):** `fashion`: 47, `shopping100k`: 770.
- **Ảnh thiếu màu (`color` thiếu, chỉ dedup SHA-1):** 17,885 ảnh (trong đó `shopping100k`: 17,885).
- **Kiểm tra chuyển RGB & đường dẫn tuyệt đối:** **98,289 / 98,289** đạt (`0` lỗi — gồm cả 2 ảnh mode `L` đã chuyển `RGB` thành công).
- **Contact sheet kiểm tra bằng mắt (60 ảnh ngẫu nhiên, seed=42):** [gd3_candidates_sample_60.png](./contact_sheets/gd3_candidates_sample_60.png).
