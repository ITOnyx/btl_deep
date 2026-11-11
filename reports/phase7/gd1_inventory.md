# Pha 7a — GĐ1: Báo cáo Kiểm kê & Đối chiếu `merged_dataset`

- **Thời gian chạy:** 11.9s | **Seed:** `42`
- **Nguồn đọc (chỉ đọc):** `D:\Kì 7\Hoc_sau_ver4\_build\Hoc_sau_ver4\data\merged_dataset`
- **Catalog hiện tại (chỉ đọc):** `D:\Kì 7\Hoc_sau_ver4\data\catalog\catalog.parquet` (40,891 dòng, 38,644 `in_scope`)
- **Dữ liệu xuất:** `eval/phase7/inventory.parquet` (139,139 dòng x 48 cột), `eval/phase7/candidate_rejects.csv` (0 dòng)

---

## 1. Số lượng theo nguồn, split và tỷ lệ thiếu metadata (`all.parquet`)

Lưu ý cấu trúc thực tế trên đĩa: thư mục `images/` trong `merged_dataset` đã được làm phẳng thành `images/<uid>.jpg` (`139.139` ảnh, không chia thư mục con `images/fashion/` hay `images/shopping100k/`).

### 1.1 Phân bố theo Nguồn × Split & `legacy_test`

| Nguồn (`source`) | `train` | `val` | `test` | Tổng | Trong đó `legacy_test` |
|---|---:|---:|---:|---:|---:|
| `fashion` | 32,763 | 4,095 | 4,095 | **40,953** | 1,500 |
| `shopping100k` | 78,548 | 9,819 | 9,819 | **98,186** | 1,471 |
| **Tổng** | 111,311 | 13,914 | 13,914 | **139,139** | **2,971** |

### 1.2 Cột metadata thực tế & Tỷ lệ thiếu theo nguồn

(Tính cả `NaN`/`None` và chuỗi rỗng/sentinel như `""`, `"NA"`, `"Unknown"`, `"{}"`; lưu ý schema `all.parquet` dùng tên cột `article_type`, `color`, `title` tương ứng với `articleType`, `baseColour`, `name` ở `catalog.parquet`.)

| Cột (`all.parquet`) | Vai trò | Thiếu `fashion` (40.953) | % | Thiếu `shopping100k` (98.186) | % |
|---|---|---:|---:|---:|---:|
| `title` | Tên tiếng Anh (gốc/tổng hợp) | 7 | 0.02% | 0 | 0.00% |
| `title_vi` | Tên tiếng Việt (tổng hợp/cũ) | 0 | 0.00% | 0 | 0.00% |
| `brand` | Thương hiệu | 0 | 0.00% | 98,186 | 100.00% |
| `category` | Danh mục cấp 1 (masterCategory) | 0 | 0.00% | 0 | 0.00% |
| `sub_category` | Danh mục cấp 2 (subCategory) | 0 | 0.00% | 0 | 0.00% |
| `article_type` | Loại sản phẩm (articleType) | 0 | 0.00% | 0 | 0.00% |
| `gender` | Giới tính | 0 | 0.00% | 0 | 0.00% |
| `color` | Màu sắc (baseColour) | 13 | 0.03% | 17,885 | 18.22% |
| `usage` | Mục đích sử dụng (usage) | 288 | 0.70% | 0 | 0.00% |
| `season` | Mùa (season) | 0 | 0.00% | 63,758 | 64.94% |
| `captions` | JSON caption đa góc nhìn | 0 | 0.00% | 0 | 0.00% |
| `attributes` | JSON thuộc tính chi tiết | 0 | 0.00% | 0 | 0.00% |
| `label_group` | Cụm biến thể | 0 | 0.00% | 0 | 0.00% |

Nhận xét thiếu metadata ở `shopping100k`: `brand` thiếu 100.00%, `season` thiếu 64.94% (phần có nhãn do `build_merged_dataset.py` suy diễn quy tắc cho nhóm mùa đông/hè), `color` thiếu 18.22% (`color.txt` có nhãn `-1`).

---

## 2. Kiểm tra đọc ảnh bằng PIL & thuộc tính pixel

- **Tổng file tồn tại trên đĩa:** 139,139 / 139,139
- **Đọc giải mã pixel PIL thành công (`im.load()`):** **139,139 / 139,139** | Lỗi `unreadable`: **0** (đã ghi `eval/phase7/candidate_rejects.csv`)
- **Định dạng ảnh (`pil_format`):** {'JPEG': 139139}
- **Mode ảnh theo nguồn:** {'fashion:RGB': 40953, 'shopping100k:L': 2, 'shopping100k:RGB': 98184}
- **EXIF Orientation khác 1 (`has_exif_rotation`):** {'fashion': 0, 'shopping100k': 0} (phân bố tag: {1: 139139})
- **Kênh Alpha (`has_alpha`):** {'fashion': 0, 'shopping100k': 0} | **CMYK:** {'fashion': 0, 'shopping100k': 0} | **Ảnh xám (`L/1/I/F/LA`):** {'fashion': 0, 'shopping100k': 2}
- **Sai lệch `(pil_width, pil_height)` so với `all.parquet`:** 0 ảnh

---

## 3. Đối chiếu trùng lặp với `catalog.parquet` (40.891 dòng / 38.644 `in_scope`)

| Nguồn (`merged_dataset`) | Tổng trong merged | Trùng ID catalog (40.891) | Trùng thêm chỉ qua SHA-1 | Tổng TRÙNG catalog (ID ∪ SHA-1) | Thực sự MỚI so với 40.891 | Trong đó trùng `in_scope` (38.644) |
|---|---:|---:|---:|---:|---:|---:|
| `fashion` | 40,953 | 40,803 | 28 | **40,831** | **122** | 38,593 |
| `shopping100k` | 98,186 | 0 | 0 | **0** | **98,186** | 0 |
| **Tổng** | **139,139** | 40,803 | 28 | **40,831** | **98,308** | 38,593 |

- **Đối chiếu theo đường dẫn gốc (`orig_rel_path` vs `catalog.path`):** 40,803 dòng khớp `data/fashion-dataset/images/<id>.jpg`.
- **Chiều ngược lại (`catalog.parquet` → `merged_dataset`):** 40,803 / 40,891 ID của catalog có mặt trong `merged_dataset` (và 40,884 khớp SHA-1); 88 ID catalog vắng mặt trong `merged_dataset` (do `build_merged_dataset.py` khử trùng MD5 / `pHash=0` hoặc `pHash` cụm đa nguồn).
- **Giải thích 150 ảnh `fashion` KHÔNG trùng ID trong `catalog.parquet`:** {'quarantine_blurry': 47, 'quarantine_duplicate+sha1_dup_in_catalog': 28, 'quarantine_banner': 38, 'quarantine_duplicate': 9, 'quarantine_low_res': 6, 'other_not_in_catalog': 22} (gồm các ảnh bị loại ở Pha 1 vào `_quarantine` nhưng được `build_merged_dataset.py` cứu do thuộc 1.500 `legacy_test`, các dòng lỗi dấu phẩy trong `styles.csv` được cứu bằng `split(',', 9)`, và các trường hợp trùng SHA-1 với ID khác trong catalog).

---

## 4. Kiểm toán bộ lọc của `scripts/build_merged_dataset.py` so với `config.yaml`

| Tiêu chí | `config.yaml` (Chuẩn dự án) | `scripts/build_merged_dataset.py` (Thực tế đã chạy) | Sai khác & Rủi ro còn lại |
|---|---|---|---|
| `min_short_side` | **224 px** | **`100 px`** (và bỏ qua nếu là `legacy_test`) | Shopping100k và `legacy_test` có thể chứa ảnh `100..223 px` (< 224 px). |
| Độ mờ (`blur`) | **`blur_laplacian_var_threshold = 12.0`** | **Không tính (`False`)**; `fashion` chỉ kế thừa `_quarantine/blurry` (trừ `legacy_test`) | **98.186 ảnh Shopping100k chưa từng được kiểm tra độ mờ Laplacian!** |
| Banner / viền chữ | **`detect_banner = True`** | **Không chạy (`False`)**; `fashion` chỉ kế thừa `_quarantine/banner` (trừ `legacy_test`) | **98.186 ảnh Shopping100k chưa từng chạy bộ phát hiện banner/chữ.** |
| Khử trùng lặp (`pHash`) | `pHash <= 1`, cùng `['articleType', 'gender', 'baseColour']`, `transitive=False`, ưu tiên `short_side -> blur -> id` | Chỉ xóa **MD5 giống hệt** và **`pHash == 0` cùng thuộc tính**; còn **`pHash == 1`** chỉ gom `label_group`, KHÔNG loại | Ảnh gần trùng `pHash == 1` vẫn còn trong `merged_dataset`; không dùng `blur_score` để chọn đại diện. |
| Phạm vi (`in_scope`) | `exclude_master_categories`, `exclude_article_types`, giữ `Wristbands` | Đọc từ `config.yaml`, nhưng giữ cả **`2,246` dòng `in_scope=False`** vào dataset (`mismatch=0`) | Phải lọc lại `in_scope == True` & 2 Keychain (`43230`, `42066`) ở GĐ3. |

Thống kê báo cáo gốc của `merged_dataset`: `rejected.csv` có **4,789** dòng (`{'legacy_test_external_tiki_no_image': 1372, 'quarantine_blurry': 1239, 'quarantine_duplicate': 1151, 'quarantine_banner': 897, 'quarantine_low_res': 113, 'legacy_test_external_shopee_no_image': 12, 'missing_image': 5}`), `duplicates.csv` có **4,876** dòng.

---

## 5. Gắn nhãn `seen_in_train` (`no` / `yes` / `unknown`)

### Căn cứ từng giá trị
- **`no` (chắc chắn KHÔNG nằm trong tập train của checkpoint hiện tại):** Các UID đối chiếu khớp được với tập Test gốc **4.372 sản phẩm** trong `data/checkpoints_siglip-vit/dense_captions_catalog.json` (tương ứng `product_metadata_old.json` gồm `1.500 myntra + 1.500 shop100k + 1.372 tiki`), khớp `splits.json['legacy_test_uids']` (`2,971` UID = `1.500 fashion + 1.471 shopping100k` duy nhất), và `catalog.parquet['split'] == 'test'` (`1.372 fashion`). Ghi chú: `dense_captions_catalog.json` ánh xạ được `1.500 fashion` + `1.471 shopping100k` duy nhất (`17` mã `shop100k` trỏ trùng ảnh vật lý, còn `1.372 tiki` + `12 shopee` là nguồn ngoài).
- **`yes` (chắc chắn nằm trong tập train của checkpoint hiện tại):** **`0` ảnh.** Lý do: Notebook huấn luyện (`notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`) huấn luyện trên **161.870 ảnh** (`Train = 153.126 | Val = 4.372 | Test = 4.372`), nhưng danh sách ID của 153.126 ảnh `train` và 4.372 ảnh `val` **không tồn tại trên đĩa** (ngay cả `catalog.parquet` cũng chỉ gắn được `test = 1.372`, còn lại `39.519` gắn `unknown`). Ngoài ra `splits.json['train'/'val'/'test']` (111.311 / 13.914 / 13.914) là phân chia mới do `build_merged_dataset.py` tạo *sau này* — không phải phân chia lúc train checkpoint đang chạy (`10.943` ảnh `test` mới trong `splits.json` KHÔNG thuộc 4.372 test gốc). Theo nguyên tắc không chắc thì không đoán, không gắn `yes` bừa.
- **`unknown`:** Toàn bộ các ảnh không nằm trong danh sách 4.372 test gốc (về mặt xác suất phần lớn nằm trong 153.126 ảnh train của checkpoint nhưng không có manifest ID trên đĩa để khẳng định từng ID là train vs val hay bị lọc trước Colab).

| Nguồn | `seen_in_train = no` | `seen_in_train = yes` | `seen_in_train = unknown` | Tổng |
|---|---:|---:|---:|---:|
| `fashion` | 1,500 | 0 | 39,453 | 40,953 |
| `shopping100k` | 1,471 | 0 | 96,715 | 98,186 |
| **Tổng** | **2,971** | **0** | **136,168** | **139,139** |

Chi tiết lý do `seen_in_train_reason`: `{'no_colab_161k_train_val_id_manifest_on_disk': 136168, 'confirmed_test:dense_captions+splits_legacy_test': 1599, 'confirmed_test:dense_captions+splits_legacy_test+catalog_test': 1372}`.

---

## 6. Kết luận GĐ1 & Định hướng GĐ2

1. **Phần lớn `fashion` (40,803 / 40,953 theo ID; 40,831 theo ID ∪ SHA-1) là ảnh ĐÃ CÓ trong `catalog.parquet`**; nguồn bổ sung thực sự mới chủ yếu là **`shopping100k` (98,186 / 98,186 ảnh)**. Số ảnh đọc/giải mã PIL thành công: **139,139 / 139,139** (`0` lỗi `unreadable`).
2. `shopping100k` **chưa từng đi qua bộ lọc chất lượng** (`min_short_side=224`, `Laplacian blur >= 12.0`, `banner_flag`, `pHash <= 1`). Bước GĐ2 sẽ đo toàn bộ các chỉ số chất lượng này trên toàn bộ ứng viên và xuất contact sheet để người dùng duyệt ngưỡng trước khi lọc chính thức ở GĐ3.
