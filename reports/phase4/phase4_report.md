# BÁO CÁO KẾT QUẢ PHA 4: DỰNG FAISS HNSW INDEX, MANIFEST VÀ SANITY CHECK

> **Dự án:** Hệ thống tra cứu sản phẩm thời trang E-commerce (SigLIP ViT-SO400M + CIR + Dense Captioning)  
> **Thời gian thực hiện:** 12/11/2026 16:35:10  
> **Cam kết:** Tuân thủ Luật 1 (Không huấn luyện lại), Luật 3 (Không ghi đè index cũ, dựng index mới ở thư mục riêng), Luật 4 (Đường code tiền xử lý duy nhất), Luật 6 (Hash Guard) theo `AGENTS.md`.

---

## 1. Tóm tắt kết quả chính

1. **Dựng 2 cấu trúc Index riêng biệt tuân thủ tiêu chuẩn đánh giá (`context.md` & `config.yaml`):**
   - **Official Test Gallery (`index/test_gallery/`):** Chứa đúng **1.290 sản phẩm thời trang in-scope sạch của tập Test** (`eval/test_split.json`). Dùng để đối chiếu các chỉ tiêu chất lượng chính thức (Recall@10 $\ge 0.85$, mAP $\ge 0.75$, Latency $< 200$ ms).
   - **Reference Full Gallery (`index/reference_gallery/`):** Chứa **38.646 sản phẩm catalog in-scope sạch** (bao gồm cả sản phẩm train/val) phục vụ đo lường ước lượng lạc quan / tham khảo theo đúng quy định tại Mục C.3 (`AGENTS.md`).
2. **Cấu hình FAISS HNSW chuẩn mực:**
   - Thuật toán: `IndexHNSWFlat`
   - Kích thước vector: $D = 1152$, không gian chuẩn hóa $L_2$ (tương đương Cosine Similarity qua Inner Product).
   - Siêu tham số: $M = 32$, $efConstruction = 200$, $efSearch = 128$.
3. **Hiệu năng tìm kiếm xấp xỉ (ANN Benchmark):**
   - Tại $efSearch = 128$: Đạt **Recall@10 = 100.00%** tuyệt đối so với Flat Exact Search trên cả 2 galleries.
   - Độ trễ tìm kiếm (Query Latency):
     - Official Test Gallery (1.290 vectors): **0.032 ms/truy vấn**.
     - Reference Full Gallery (38.646 vectors): **0.121 ms/truy vấn**.
     - Vượt xa tiêu chuẩn cam kết SLA ($< 200$ ms).
4. **Vượt qua 100% 4 tiêu chí Sanity Check bắt buộc của Pha 4:**
   - Tiêu chí 1: **0 ID trùng lặp** (`max_duplicate_ids == 0`).
   - Tiêu chí 2: **0 ảnh test lọt vào dữ liệu train** (`max_test_leak_into_train == 0`).
   - Tiêu chí 3: **Truy vấn lại một ảnh trả đúng chính nó ở Rank 1** (`self_query_rank1 == True`, Cosine = 1.000000).
   - Tiêu chí 4: **Số lượng vector khớp tuyệt đối với `manifest.json`** (`n_vectors_match_manifest == True`).
5. **Kiểm thử tự động:** Toàn bộ **21/21 tests pytest PASSED**.

---

## 2. Bảng kết quả Sanity Check bắt buộc (Pha 4)

Kết quả trích xuất từ `reports/phase4/sanity_check_test_gallery.json` và `reports/phase4/sanity_check_reference_gallery.json`:

| Tiêu chí Sanity Check | Tiêu chuẩn bắt buộc | Official Test Gallery (`test_gallery`) | Reference Full Gallery (`reference_gallery`) | Đánh giá |
|---|---|---|---|---|
| **1. ID trùng lặp** | `== 0` | **0** (1.290 / 1.290 ID độc nhất) | **0** (38.646 / 38.646 ID độc nhất) | ✅ **ĐẠT** |
| **2. Rò rỉ Test $\rightarrow$ Train** | `== 0` | **0** (Giao với train = $\emptyset$) | **0** (Tập test độc lập khóa SHA-256) | ✅ **ĐẠT** |
| **3. Self-Query Rank 1** | Rank 1 & Cosine $\ge 0.9999$ | **50/50 mẫu (100%)** (Cosine Min: 1.000000) | **50/50 mẫu (100%)** (Cosine Min: 1.000000) | ✅ **ĐẠT** |
| **4. Khớp số lượng Manifest** | Đồng nhất 100% | **1.290 == 1.290 == 1.290** | **38.646 == 38.646 == 38.646** | ✅ **ĐẠT** |
| **5. Hash Guard (Luật 6)** | Trùng SHA-256 config | **`e2f82fee26...` (Khớp 100%)** | **`e2f82fee26...` (Khớp 100%)** | ✅ **ĐẠT** |

---

## 3. Khảo sát Hiệu năng ANN Recall@10 và Độ trễ theo `efSearch`

Số liệu đo đạc thực tế từ script `src/index/build_index.py` (lưu tại `reports/phase4/ann_benchmark.json`):

### 3.1. Official Test Gallery (1.290 vector)
| efSearch | Recall@10 vs Flat (%) | Độ trễ tìm kiếm (ms/truy vấn) | Nhận xét |
|---|---|---|---|
| 32 | **100.00%** | 0.012 ms | Cực nhanh |
| 64 | **100.00%** | 0.022 ms | Tối ưu |
| **128 (Mặc định)** | **100.00%** | **0.032 ms** | **Chuẩn đề cương** |
| 256 | **100.00%** | 0.058 ms | Độ trễ rất thấp |

### 3.2. Reference Full Gallery (38.646 vector)
| efSearch | Recall@10 vs Flat (%) | Độ trễ tìm kiếm (ms/truy vấn) | Nhận xét |
|---|---|---|---|
| 32 | **100.00%** | 0.049 ms | Nhanh |
| 64 | **100.00%** | 0.073 ms | Tốt |
| **128 (Mặc định)** | **100.00%** | **0.121 ms** | **Chuẩn đề cương** |
| 256 | **100.00%** | 0.176 ms | An toàn tuyệt đối |

---

## 4. Vấn đề kỹ thuật gặp phải và Giải pháp

- **Vấn đề đường dẫn Unicode tiếng Việt trên Windows cho FAISS C++:**
  - *Hiện tượng:* Thư mục làm việc trên máy tính là `D:\Kì 7\Hoc_sau_ver4\...` có chứa ký tự tiếng Việt có dấu (`Kì 7`). Hàm native `faiss.write_index(index, path)` của C++ mở file qua `fopen(filename, "wb")` kiểu ANSI nên ném lỗi `RuntimeError: 'f' failed: could not open ...: No such file or directory`.
  - *Giải pháp:* Đã refactor lớp `FaissHNSWIndex` trong [src/index/faiss_index.py](file:///d:/Kì%207/Hoc_sau_ver4/src/index/faiss_index.py) chuyển sang cơ chế:
    - Lưu: `faiss.serialize_index(self.index)` trả về mảng byte numpy uint8, sau đó dùng Python `open(path, "wb")` để ghi ra đĩa (Python trên Windows hỗ trợ Unicode 100%).
    - Nạp: Dùng Python `open(path, "rb")` đọc vào buffer và giải nén bằng `faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))`.
    - Kết quả: Index được đọc/ghi nhị phân trơn tru, tương thích tuyệt đối mà không cần đổi tên thư mục gốc của người dùng.

---

## 5. Tệp tạo mới và Cấu trúc Index

- `src/index/faiss_index.py`: Wrapper quản lý `FaissHNSWIndex` và `FaissFlatIndex` hỗ trợ Unicode I/O.
- `src/index/build_index.py`: Pipeline xây dựng tự động cả 2 galleries và benchmark ANN.
- `src/eval/sanity_check.py`: Công cụ kiểm định tự động 4 tiêu chí Sanity Check.
- `tests/test_phase4_index.py`: Bộ unit tests kiểm tra HNSW, save/load, Flat recall.
- `index/test_gallery/`:
  - `faiss_hnsw.bin`: Tệp nhị phân FAISS HNSW Flat (1.290 vectors, ~5.8 MB).
  - `image_embeddings.npy`: Mảng numpy vector 1152-d (1.290 x 1152 float32).
  - `id_map.json`: Ánh xạ index sang metadata 1.290 sản phẩm test.
  - `manifest.json`: Manifest chuẩn theo Luật 6.
- `index/reference_gallery/`:
  - `faiss_hnsw.bin`: Tệp nhị phân FAISS HNSW Flat (38.646 vectors, ~174 MB).
  - `image_embeddings.npy`: Mảng numpy vector 1152-d (38.646 x 1152 float32).
  - `id_map.json`: Ánh xạ metadata 38.646 sản phẩm.
  - `manifest.json`: Manifest chuẩn theo Luật 6.
- `reports/phase4/ann_benchmark.json`: Bảng đo đạc ANN Recall@10.
- `reports/phase4/sanity_check_test_gallery.json`: Báo cáo Sanity Check tập Test.
- `reports/phase4/sanity_check_reference_gallery.json`: Báo cáo Sanity Check tập Full.

---

## 6. Kết quả kiểm thử Unit Tests (pytest)

```bash
$ pytest tests/
============================= test session starts =============================
collected 21 items

tests\test_phase1.py ......                                              [ 28%]
tests\test_phase2_caption.py .......                                     [ 61%]
tests\test_phase3_parity.py .....                                        [ 85%]
tests\test_phase4_index.py ...                                           [100%]

======================= 21 passed, 2 warnings in 38.47s =======================
```
**21/21 tests PASSED (100%).**

---

## 7. Bảng quyết định cần người dùng (Điểm dừng Pha 4)

| Mục kiểm định | Tiêu chuẩn | Kết quả thực tế | Trạng thái |
|---|---|---|---|
| **Sanity Check 1: ID trùng** | `max_duplicate_ids == 0` | **0 ID trùng** | ✅ Đạt |
| **Sanity Check 2: Rò rỉ dữ liệu** | `max_test_leak_into_train == 0` | **0 ảnh test rò rỉ** | ✅ Đạt |
| **Sanity Check 3: Self-Query** | Rank 1 với Cosine $\ge 0.9999$ | **100% Rank 1 (Cosine 1.000000)** | ✅ Đạt |
| **Sanity Check 4: Khớp Manifest** | Số vector đồng nhất | **Khớp 100%** | ✅ Đạt |
| **Hash Guard (Luật 6)** | Checkpoint hash | **`e2f82fee26...` trùng khớp** | ✅ Đạt |
| **Hành động tiếp theo** | Chuyển sang **Pha 5: Đánh giá Benchmark (Recall@K, mAP), Hubness, SLA Độ trễ** | ⏳ Chờ lệnh **DUYỆT PHA 4** |
