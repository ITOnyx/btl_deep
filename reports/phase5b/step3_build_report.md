# Báo cáo Đóng băng `parser_v2` & Hạ tầng Bước 3 (PROMPT 6c)

- **Nhánh Git:** `phase7a` | **Tag đóng băng:** `phase5b-parser-v2-frozen` | **Seed:** `42`
- **Trạng thái:** Đã hoàn tất Bước 0 + Phần 2–3 (xây & đóng băng `parser_v2`, `query_rewriter`, `canonical_eval`). **KHÔNG** chạy lưới 22 cấu hình, **KHÔNG** chạy B-human trên cấu hình nào. **DỪNG** chờ duyệt trước Bước 3 (chạy lưới).
- **Bất biến giữ nguyên:** `parser_v1.py` (`111002f6...`), `selection_rule_v3.*` (`fa4e0df3...` / `d9e7b56d...`), Tập A/B-synth/C, và nội dung câu `val_queries_b_human.json` đều không bị sửa.

---

## 1. Bước 0 — Xác nhận Nguồn gốc `B-human`

- Phân bố `source` trong `eval/val_queries_b_human.json`: **`ai_written`: 131/131 (100%)**.
- Phân bố `source` trong `eval/val_queries_b_human.validated.json`: **`ai_written`: 65/65 (100%)** (khớp 100% danh sách 65 ID so với bản trước; `SHA-256: 19f81e8aa35a034f1528dc101e108ad0469201d50372e16afbc5286e78bd8deb`, hash giữ nguyên do tệp gốc đã mang `source="ai_written"`).
- **Mô tả cổng B-human:** **"kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối"** (do toàn bộ câu do AI soạn).
- **Thống kê bổ sung (trên 65 câu hợp lệ):**
  - Có màu: **`31`** câu | Không màu: **`34`** câu.
  - Số cụm `spec_cluster_id` độc nhất: **`61`** cụm, trong đó **`4`** cụm có $\ge 2$ câu (`Tshirts__Men__Black`: 2, `Tshirts__Men__Purple`: 2, `Shirts__Men__`: 2, `Watches__Women__Silver`: 2).

---

## 2. Phần 2 — Các Module Đã Xây & Đóng băng

| Module | Vai trò | SHA-256 |
|---|---|---|
| `src/serve/parser_v2.py` | **3b:** Khớp ưu tiên có dấu, bảo vệ cụm cố định từ bảng homograph `2c.1c`, chặn màu đơn âm bỏ dấu thiếu ngữ cảnh, mở rộng từ khóa truy nguyên `catalog_rebuild.parquet` | `552f3b1f4ec41c49f27c72b4fa984969ee0807fb9b318f539579e25825ebcffa` |
| `src/serve/query_rewriter.py` | **3c & 3e:** Khôi phục dấu tiếng Việt nhẹ (`restore_diacritics_for_query`), `dual_encode_texts`, `BM25GalleryScorer` | `8d4f68d4d9853ba3f8470bd9aac0989be8efd988bf036a386d1136db723d0c13` |
| `src/eval/canonical_eval.py` | **3e:** Hỗ trợ đầy đủ `cfg_01..cfg_22` qua 1 hàm chuẩn (`zscore_cosine`, `slot_weights`, `bm25_scores`, `tie_rule="conservative"`, `force_exact=True`) | `e3d1dafb2099514180400caddc61a4b84e049a60a15bd812a3b84a2c28bbb199` |

---

## 3. Độ phủ & Dương tính giả của `parser_v2` vs `parser_v1` trên `B-human validated` (65 câu)

*(Chỉ đo trên `B-human validated` ở mức parser để báo cáo trung thực; KHÔNG chấm điểm retrieval và KHÔNG vá lexicon theo câu.)*

| Nhóm | Parser | `articleType` Cov / FP(query) | `gender` Cov / FP(query) | `baseColour` Cov / FP(query) / FP(pred) |
|---|---|---:|---:|---:|
| **Toàn bộ (65 câu)** | `parser_v1` | `0.8000` / `0.0615` | `1.0000` / `0.0000` | `0.9355` / `0.1538` / `0.2778` |
| **Toàn bộ (65 câu)** | **`parser_v2`** | **`0.9385`** / **`0.0154`** | `1.0000` / `0.0000` | `0.9355` / **`0.0154`** / **`0.0333`** |
| **Có màu (31 câu)** | `parser_v1` | `0.9355` / `0.0323` | `1.0000` / `0.0000` | `0.9355` / `0.1290` / `0.1333` |
| **Có màu (31 câu)** | **`parser_v2`** | **`0.9677`** / **`0.0000`** | `1.0000` / `0.0000` | `0.9355` / **`0.0323`** / **`0.0333`** |
| **Không màu (34 câu)** | `parser_v1` | `0.6765` / `0.0882` | `1.0000` / `0.0000` | `—` / `0.1765` (`6/34`) / `1.0000` |
| **Không màu (34 câu)** | **`parser_v2`** | **`0.9118`** / **`0.0294`** | `1.0000` / `0.0000` | `—` / **`0.0000`** (`0/34`) / **`0.0000`** |

### Nhãn màu hiếm trên `B-human validated`
- **`Steel`** (`n=1`, `val_b_human_088`: `"đồng hồ màu thép"`): `v1=[]` → **`v2=['Steel']`** (khớp).
- **`Olive`** (`n=1`, `val_b_human_079`: `"giày sneaker nam màu xanh rêu"`): `v1=['Green']` → `v2=['Green']` (ghi nhận hổng trung thực: `"xanh rêu"` ánh xạ `Green` trong lexicon, còn `Olive` ánh xạ `"xanh ô liu"`/`"olive"`; không vá theo câu).
- **`Navy Blue`** (`n=1`, `val_b_human_091`: `"giày thường ngày màu xanh navy"`): `v1=['Navy Blue']` → **`v2=['Navy Blue']`** (khớp).

---

## 4. Kiểm thử (`pytest`)
- `pytest tests/ -v`: **39 passed, 0 failed** (gồm 36 test trước + 3 test mới trong `tests/test_phase5b_step3_build.py`).
