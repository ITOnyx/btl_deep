# Bổ sung Kiểm định Pha 2 — Lượng hóa Ảnh hưởng Rò rỉ & Giải thích Chênh lệch (Mục A.1–A.5, B.6–B.9)

- **Thời điểm chạy (UTC):** `2026-11-19T15:15:23Z`
- **Commit đối chiếu:** `0c8e49e2afe14328bcd272fa2a50e184389702ea`
- **Bằng chứng:**
  - `reports/verification/phase2/run_phase2_addendum.py`
  - `reports/verification/phase2/phase2_addendum_results.json`
  - `reports/verification/phase2/phase2_addendum_stdout.txt`
  - `reports/verification/phase2/leak_pairs.csv`

---

## A. Lượng hóa Ảnh hưởng Rò rỉ (Mục 8 — Chỉ đếm, KHÔNG chấm điểm lại)

### A.1. Số sản phẩm `test` DISTINCT bị ảnh hưởng (`test_clean` 1,290 vs `train_all_clean` 37,354)
- **(a) Trùng tuyệt đối byte ảnh (`SHA-1`):**
  - Tổng số cặp `(test_id, train_id)`: **3 cặp**
  - Số sản phẩm `test` DISTINCT: **3 / 1,290** (`0.23%`) — IDs: `8721`, `28174`, `28352`
  - Số sản phẩm `train` DISTINCT: **3 / 37,354** — IDs: `9609`, `28182`, `28354`
- **(b) Trùng `pHash` (`Hamming = 0` và `Hamming ≤ 1`):**
  - *Ghi chú toán học:* Mã `imagehash.phash` 64-bit dùng ngưỡng trung vị (median) trên hệ số DCT nên luôn có số bit 1 cố định, dẫn đến khoảng cách Hamming giữa hai mã pHash bất kỳ luôn là **số chẵn** (`0, 2, 4, ...`). Do đó tập `Hamming ≤ 1` trùng khít 100% với tập `Hamming = 0` (`0` cặp có `Hamming = 1`).
  - Tổng số cặp `(test_id, train_id)` có `pHash Hamming = 0` (đã bao gồm 3 cặp trùng `SHA-1`): **78 cặp**
  - Số sản phẩm `test` DISTINCT bị ảnh hưởng: **45 / 1,290** (`3.49%`), ứng với **45 mã `pHash` phân biệt**
  - Số sản phẩm `train` DISTINCT liên quan: **78 / 37,354** (`0.21%`)
  - Danh sách đầy đủ 78 cặp đã xuất tại: [`reports/verification/phase2/leak_pairs.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/verification/phase2/leak_pairs.csv).

### A.2. Phân tích so sánh nhãn `(articleType, gender, baseColour)` trên 78 cặp `test–train`

| Nhóm cặp trùng | Tổng số cặp | Cùng cả 3 thuộc tính | Khác ít nhất 1 thuộc tính | Cùng `articleType` / Khác | Cùng `gender` / Khác | Cùng `baseColour` / Khác | Chỉ khác `baseColour` (cùng loại & giới) | Chỉ khác `articleType` (cùng giới & màu) | Chỉ khác `gender` (cùng loại & màu) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **(a) Trùng `SHA-1` byte ảnh** | **3** | **0** | **3 (100%)** | 3 / 0 | 3 / 0 | 0 / 3 | **3** (`White↔Purple`, `Bronze↔Copper`, `Bronze↔Brown`) | 0 | 0 |
| **(b) Trùng `pHash` (`Hamming=0`, gồm cả 3 cặp SHA-1)** | **78** | **0** | **78 (100%)** | 64 / 14 | 62 / 16 | 3 / 75 | **49** (`62.8%`) | **1** (`1.3%`) | **2** (`2.6%`) |

- **Nhận xét bản chất:** **0/78 cặp** có cùng đồng thời cả 3 nhãn `(articleType, gender, baseColour)`. Đa số (`49/78 = 62.8%`, trong đó có cả 3/3 cặp trùng `SHA-1`) là các biến thể màu sắc của cùng một mẫu sản phẩm (`same articleType`, `same gender`, khác `baseColour`) mà sàn Myntra dùng lại ảnh chụp studio gần giống hoặc dùng chung 1 file ảnh; phần còn lại (`29/78`) là khác `articleType` gần nghĩa (`Heels↔Flats`, `Sports Sandals↔Sandals`, `Formal Shoes↔Casual Shoes`, `Kurtas↔Kurtis`) hoặc khác `gender` (`Men↔Unisex`, `Women↔Unisex`).

### A.3. Số truy vấn trong 151 câu `held-out v2` có chứa sản phẩm bị ảnh hưởng trong tập GT Hybrid (chỉ đếm, không chấm điểm)
- Trong 151 truy vấn của `eval/test_queries_heldout_v2.json` (tập GT theo chính sách `hybrid`):
  - **Theo 3 sản phẩm `test` trùng `SHA-1` (`8721`, `28174`, `28352`):** chỉ có **1 / 151 truy vấn** (`0.66%`, mã truy vấn `h2_007`) chứa ít nhất 1 sản phẩm trong tập GT hybrid.
  - **Theo 45 sản phẩm `test` trùng `pHash` (`Hamming = 0` / `≤ 1`):** có **22 / 151 truy vấn** (`14.57%`) chứa ít nhất 1 sản phẩm thuộc nhóm 45 ID này trong tập GT hybrid.

### A.4. Kiểm tra tương tự cho `test–val`, `val–train (non-val)` và nội bộ `test_gallery`

| Cặp tập hợp đối chiếu | Quy mô (`N_A × N_B`) | Cặp trùng `SHA-1` (ID A / ID B distinct) | Cặp trùng `pHash Hamming ≤ 1` (ID A / ID B distinct) | Cùng cả 3 thuộc tính | Khác ít nhất 1 thuộc tính (Chỉ khác `baseColour`) |
|---|---|---:|---:|---:|---:|
| **`test_clean` vs `val_clean`** | `1,290 × 1,884` | **0** (`0 / 0`) | **3** (`3 / 3`) | **0** | **3** (`3` chỉ khác `baseColour`) |
| **`val_clean` vs `train \ val`** *(lưu ý `val ⊂ train`)* | `1,884 × 35,470` | **6** (`6 / 6`) | **101** (`57 / 97`) | **0** | **101** (`60` chỉ khác `baseColour`) |
| **Nội bộ `test_gallery` (`test` vs `test`, `i < j`)** | `1,290 × 1,289 / 2` | **0** (`0 / 0`) | **1** (`1 / 1`) | **0** | **1** (`1` chỉ khác `baseColour`) |

### A.5. Bằng chứng mã nguồn & tài liệu giải thích vì sao khử trùng Pha 1 không lọc các cặp này
- Khử trùng Pha 1b được thiết kế chủ đích **chỉ khử trùng trong cùng tổ hợp 3 thuộc tính `(articleType, gender, baseColour)`** để tránh loại nhầm các biến thể màu sắc/kiểu dáng khác nhau trên ảnh xám pHash:
  1. `config.yaml` dòng 71–74:
     ```yaml
     phash_hamming_threshold: 1
     dedup_hashes: ["sha1", "phash"]
     dedup_match_attributes: ["articleType", "gender", "baseColour"]
     dedup_transitive: false
     ```
  2. `data/_quarantine/legacy_rebuild_20261111/src/src/diag/phase1b_apply_fix.py` dòng 52 & 88:
     ```python
     print(f"Rule: same (articleType, gender, baseColour) AND phash Hamming <= {phash_thresh}")
     ...
     grouped = candidate_df.groupby(["articleType", "gender", "baseColour"])
     ```
  3. `reports/phase1/phase1b/phase1b_report.md` dòng 11:
     > *"Điều kiện trùng: Hai ảnh chỉ bị coi là trùng khi CÙNG `articleType` + `gender` + `baseColour` VÀ pHash Hamming ≤ 1."*

---

## B. Giải thích Các Chênh lệch (Mục B.6 – B.9)

### B.6. Truy vết chuỗi `1,500 → 1,372 → 1,290` của tập `test`
1. **`1,500` (tổng sản phẩm Myntra unseen thuộc `split == "test"`):**
   - Trong `src/data/builder.py` dòng 32–43 (`load_legacy_test_ids`) và dòng 127, 254–260: hệ thống đọc `data/checkpoints_siglip-vit/dense_captions_catalog.json` (chứa 4,372 ID unseen lúc train checkpoint trên Colab, trong đó có đúng **1,500** ID thuộc Myntra `styles.csv`) và gán `df["split"] = "test"` cho 1,500 dòng này trong `data/catalog/catalog_rebuild.parquet`.
2. **`1,372` (sản phẩm `test` đạt lọc chất lượng ảnh `filter_flag == "pass"`, ghi vào `eval/test_split.json`):**
   - Trong 1,500 dòng `split == "test"`, có **128** ảnh bị loại bởi bộ lọc chất lượng (`47 blurry + 38 banner + 37 duplicate + 6 low_res = 128`), còn lại `1,500 - 128 = 1,372` ảnh đạt `filter_flag == "pass"`.
   - Tại `src/data/builder.py` dòng 284 & 293–294:
     ```python
     test_ids = sorted(df[(df["split"] == "test") & (df["filter_flag"] == "pass")]["product_id"].tolist())
     with open(test_split_path, "w", encoding="utf-8") as f:
         json.dump(test_ids, f, indent=2)
     ```
     Do đó tệp `eval/test_split.json` lưu danh sách **1,372** ID đã qua lọc chất lượng ảnh (chưa lọc `in_scope`).
3. **`1,290` (sản phẩm `test` vừa đạt `pass` vừa thuộc phạm vi thời trang `in_scope == True`, đưa vào `index/test_gallery/`):**
   - Trong 1,372 ID của `eval/test_split.json`, có **82** sản phẩm ngoài phạm vi thời trang (`in_scope == False`: `Personal Care`, `Home`, `Sporting Goods`, `Free Gifts`, `Perfume and Body Mist`, v.v.), còn lại `1,372 - 82 = 1,290` sản phẩm.
   - Tại `src/index/build_index.py` dòng 68:
     ```python
     df_test_inscope = df[(df["product_id"].astype(str).isin(test_split_ids)) & (df["in_scope"])].reset_index(drop=True)
     ```
     Chính 1,290 sản phẩm này tạo nên `index/test_gallery/`.

### B.7. Chênh lệch 22 dòng giữa `styles.csv` (44,446) và `catalog_rebuild.parquet` (44,424) cùng 22 ảnh thừa trên đĩa
- **Xác nhận:** 22 dòng bị bỏ qua trong `styles.csv` và 22 tệp ảnh `.jpg` thừa trong `data/fashion-dataset/images/` là **cùng chính xác 22 mã `product_id` (khớp 22/22 = 100%)**:
  `['23100', '23101', '23103', '23107', '23108', '23109', '23110', '23111', '23112', '23114', '23115', '23116', '23117', '23118', '23124', '23127', '3413', '3418', '3428', '44065', '4854', '59768']`.
- **Nguyên nhân:** File `data/fashion-dataset/styles.csv` gốc của Myntra có header 10 cột, nhưng 22 dòng trên chứa dấu phẩy `,` không bọc ngoặc kép trong cột cuối `productDisplayName` (ví dụ dòng 6569: `3413,Men,Apparel,Topwear,Tshirts,White,Summer,2011,Casual,Myntra Men's Yes, its all about me White T-shirt` thành 11 trường). Khi `src/data/builder.py` dòng 77 gọi `pd.read_csv(styles_csv, on_bad_lines="skip")`, pandas tự động bỏ qua 22 dòng bad lines này (`44,446 - 22 = 44,424`), trong khi 22 file ảnh `.jpg` của chúng vẫn nằm nguyên trong `data/fashion-dataset/images/`.

### B.8. Cột `sha1` / `phash` rỗng trong `catalog_rebuild.parquet`
- **Xác nhận:** Khử trùng Pha 1b (`data/_quarantine/legacy_rebuild_20261111/src/src/diag/phase1b_apply_fix.py` dòng 57) đọc `phash` trực tiếp từ `reports/phase1/phase1a/quality_metrics.parquet` (nơi cả 44,419 ảnh đều có đủ `phash`, `width`, `height`, `short_side`, `blur_score`). Sau đó `src/data/builder.py` (dòng 161–219) khi dựng lại `catalog_rebuild.parquet` đã dùng nhánh fast-path kế thừa kết quả từ `reports/quarantine_log.csv` và gán hằng số mặc định `sha1 = ""`, `phash = ""`, `width = 300`, `height = 400`, `short_side = 300`, `blur_score = 45.0`.
- **Phân loại:**
  1. Đối với pipeline chính (Pha 1–7): đây là **sai lệch tài liệu** (`reports/data_report.md` mục 3 ghi `catalog_rebuild.parquet` chứa mã băm `sha1`/`phash`, nhưng thực tế dữ liệu nằm ở `quality_metrics.parquet`; riêng mục 7.2 của `data_report.md` đã ghi nhận việc 4 cột kích thước/độ nét mang hằng số mặc định).
  2. Đối với script WIP crawl ngoài miền (`scripts/crawl_external_test/precompute_seen_hashes.py` dòng 65–78 và `dedup_and_clean.py` dòng 72–78): gây **hệ quả chức năng nhẹ ở nhánh WIP** vì hai script này đọc cột `phash` từ `catalog_rebuild.parquet` nên thu được `0 pHash` của tập Kaggle (chỉ thu được `40,913 MD5` từ thư mục ảnh đĩa và `100,586 pHash + MD5` từ `Shopping100k`, kèm bước kiểm tra semantic cosine bằng SigLIP trên `reference_gallery`).

### B.9. Bốn dòng `quarantine_log.csv` WIP (`data/external_test/_rejected/out_of_scope/...`)
- **Truy vết nguồn gốc:** Bốn dòng này (`ext_000033.jpg`, `ext_000033_alt.jpg`, `ext_000041.jpg`, `ext_000041_alt.jpg` — sản phẩm mỹ phẩm `Juno Beauty`) được ghi bởi `scripts/crawl_external_test/finalize_external_test.py` (hàm `move_to_rejected` dòng 48–76 và dòng 364–369) lúc `2026-11-19T18:19:00` trong đợt crawl pilot (94 bản ghi).
- **Vị trí tệp thực tế trên đĩa:** Sau đợt pilot, thư mục `data/external_test/` của đợt pilot đã được đổi tên sang `data/crawl/` để chạy đợt crawl full mới (hiện có tiến trình nền `PID 9292`: `python.exe -m scripts.crawl_external_test.run_crawl --mode full` đang chạy từ `20:39:46`). Cả **4 tệp ảnh này vẫn tồn tại nguyên vẹn trên đĩa** tại:
  - `data/crawl/_rejected/out_of_scope/ext_000033.jpg`
  - `data/crawl/_rejected/out_of_scope/ext_000033_alt.jpg`
  - `data/crawl/_rejected/out_of_scope/ext_000041.jpg`
  - `data/crawl/_rejected/out_of_scope/ext_000041_alt.jpg`
- **Đánh giá theo Luật 2:** **Không có tệp nào bị xóa vĩnh viễn** (đúng tinh thần Luật 2). Tuy nhiên về mặt quy ước đường dẫn, `finalize_external_test.py` chuyển ảnh loại của tập crawl mới vào `data/external_test/_rejected/` thay vì `data/_quarantine/`, và khi đổi tên thư mục `data/external_test/` → `data/crawl/` thì 4 dòng WIP trong `reports/quarantine_log.csv` chưa được cập nhật lại đường dẫn.
