# Báo cáo Chuẩn bị Dữ liệu Pha 5c (Bước 11.1 & 11.2)

- **Seed cố định:** `42`
- **Luật tuân thủ:** Luật 1 (không train/sửa `best_model.pt`), Luật 2/3/6 (thư mục được phép, manifest + Hash Guard), Luật 4 (1 đường tiền xử lý `224x224 BILINEAR` + `SiglipProcessor max_length=128`), Luật 5 (tách `internvl2_only`, `internvl2_plus_auxiliary`, `metadata_aux`), Luật 7 & C.5/C.10 (không chạy model trên `test_gallery`).

---

## 1. Bước 11.2 — Worksheet Tập Xác Nhận Mới & CIR-eval (Đã hoàn thành, KHÔNG chạy model)

### 1.1. Nâng cấp công cụ kiểm tra & Quyền ghi `scripts/`
- Thêm `"scripts/"` vào `safety.allowed_write` trong `config.yaml` và gỡ `scripts/` khỏi `.gitignore` để theo dõi và commit các script chuẩn.
- Tạo `scripts/check_query_set.py` (nâng cấp từ `scripts/check_b_human.py` dùng chung cho `val_gallery` và `test_gallery`):
  - Kiểm tra `n_relevant` (chính sách `hybrid`), `dia_class` tự động + cảnh báo lệch khai báo, trùng lặp Jaccard `>= 0.8` với `DEV / A / B-synth / C / B-human cũ` và nội bộ, token ngoài **lexicon v1** và **lexicon v2.1**, trường `source`, và kiểm tra ngưỡng preset (`b_human_v4`, `heldout_v2`).
  - Giữ nguyên `scripts/check_b_human.py` để bảo toàn tương thích ngược với các test hiện hành.

### 1.2. Worksheet `B-human-v4` (`index/val_gallery/` — 1.884 sản phẩm)
- Script: `scripts/generate_phase5c_worksheets.py` (KHÔNG viết câu truy vấn).
- Đầu ra:
  - [`reports/phase5c/b_human_v4_worksheet.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/b_human_v4_worksheet.csv)
  - [`reports/phase5c/b_human_v4_worksheet.md`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/b_human_v4_worksheet.md)
- Số liệu thật:
  - Tổng cụm khả dụng (`n_relevant >= 5`): **225 cụm** (đã loại **84 cụm** từng dùng trong `B-human` cũ).
  - Cụm hiếm (`is_rare=True`): **157 cụm**.
  - Cụm chưa từng dùng ở `DEV / A / B-synth`: **60 cụm**.
- Yêu cầu khi điền: `>= 60` câu hợp lệ, `>= 30` cụm, `>= 20` câu ngoài lexicon v2.1, `>= 20` câu không dấu, đúng `10` câu lỗi gõ, `n_relevant >= 5`.

### 1.3. Worksheet `held-out v2` (`index/test_gallery/` — 1.290 sản phẩm, KHÔNG chạy model)
- Đầu ra:
  - [`reports/phase5c/heldout_v2_worksheet.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/heldout_v2_worksheet.csv)
  - [`reports/phase5c/heldout_v2_worksheet.md`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/heldout_v2_worksheet.md)
- Số liệu thật (ghép `usage`/`season` từ `data/fashion-dataset/styles.csv`, khớp 1.290/1.290 sản phẩm):
  - Tổng cụm khả dụng (`n_relevant >= 5`): **242 cụm**.
  - Cụm slot mỏng (`is_thin_slot=True`): **177 cụm**.
  - Nhóm **3-slot** có tín hiệu `usage`/`season` trong khoảng **5%..70%**: **65 cụm**.
- Yêu cầu khi điền: `>= 100` truy vấn, mỗi truy vấn `>= 5` ảnh đúng, `>= 40` không dấu, `>= 30%` ngoài lexicon v2.1, `>= 40` truy vấn có tín hiệu `usage`/`season`, phủ đủ slot mỏng.

### 1.4. Bộ đánh giá `CIR-eval` (sinh từ thuộc tính catalog, `seed=42`)
- Script: `scripts/build_cir_eval.py`
- Quy tắc: GT = cùng `articleType`, đổi đúng 1 thuộc tính (`baseColour` hoặc `gender`), giữ nguyên thuộc tính còn lại; sinh thuần túy từ nhãn catalog, không lấy từ câu của tập khóa.
- Đầu ra & số liệu thật:
  - `eval/cir_eval_val.json` (`.sha256`): **120 cặp** (80 đổi màu, 40 đổi giới, 39 `articleType`, `n_gt >= 3`, dùng để tuning).
  - `eval/cir_eval_test.json` (`.sha256`): **120 cặp** (80 đổi màu, 40 đổi giới, 33 `articleType`, `n_gt >= 3`, dùng báo cáo cuối, chạy 1 lần).

---

## 2. Bước 11.1 — Caption `val_gallery` bằng InternVL2-2B & `index/val_gallery_text_v5/`
- Script đã tạo:
  - [`scripts/caption_val_gallery.py`](file:///d:/Kì%207/Hoc_sau_ver4/scripts/caption_val_gallery.py): hỗ trợ `--mode pilot` (200 ảnh val, in 20 caption, kiểm tra 3 điều kiện dừng: `>128 token > 0%`, `chữ Hán > 0%`, `khớp metadata < 100%`) và `--mode full` (1.884 ảnh val theo shard 500 + resume + log lỗi `reports/caption_errors.csv`).
  - [`scripts/build_val_text_index_v5.py`](file:///d:/Kì%207/Hoc_sau_ver4/scripts/build_val_text_index_v5.py): dựng `index/val_gallery_text_v5/` tách riêng `internvl2_only_embeddings.npy`, `internvl2_plus_aux_embeddings.npy`, `metadata_aux_embeddings.npy`, kèm `manifest.json` + SHA-256 + ghi chú Luật D.5.
- **Kết quả Pilot 200 ảnh `val_gallery` (`eval/val_pilot_200_summary.json`, `seed=42`):**
  - **Config A (`InternVL2-2B` 8-bit):** `>128 token = 0.0%`, `chữ Hán = 0.0%`, `tiếng Anh = 0.0%`, `khớp metadata = 0.0%` (lặp lại tiền tố prompt tiếng Việt, giống hệt hiện tượng đã ghi nhận ở Pha 2). Tốc độ thực đo: `8.58 s/ảnh`, Peak VRAM `2.80 GB`.
  - **Config B (`merge_v1` = InternVL2-2B + Vintern-1B-v3_5 + Metadata Anchor):** `>128 token = 0.0%`, `chữ Hán = 0.0%`, `tiếng Anh = 1.5%`, **`khớp metadata = 100.0%`** (`stop_gate_config_B_triggered = false`). Peak VRAM Vintern `1.03 GB`.
  - **Thời gian ước lượng toàn bộ 1.884 ảnh val:** ~`18.1 s/ảnh` ≈ **~9,5 giờ** cho `--mode full`. Đã dừng sau pilot 200 ảnh để người dùng đọc 20 caption mẫu và duyệt trước khi chạy `--mode full` + `build_val_text_index_v5.py`.
- **Kết quả chạy toàn bộ 1.884 ảnh `val_gallery` & dựng `index/val_gallery_text_v5/` (Đã hoàn tất 100%):**
  - **Config A (`InternVL2-2B` 8-bit):** `1.884 / 1.884` ảnh (`shard_val_core_0000..0003.parquet`).
  - **Config B (`merge_v1`: InternVL2-2B + Vintern-1B-v3_5 + Metadata):** `1.884 / 1.884` ảnh (`shard_val_0000..0003.parquet`), `token_len max = 128` (`0` ảnh > 128 token), `0` lỗi (`reports/caption_errors.csv` không phát sinh).
  - **`index/val_gallery_text_v5/`:** Dựng thành công trong `104.50 s`, `n_vectors = 1.884`, `embedding_dim = 1152`, `seed = 42`, tách riêng `internvl2_only_embeddings.npy` (SHA-256 `a1dd96ab...`), `internvl2_plus_aux_embeddings.npy` (SHA-256 `ea0df85b...`), `metadata_aux_embeddings.npy` (SHA-256 `faf6880e...`), kèm `manifest.json` (Hash Guard + ghi chú Luật D.5).

---

## 3. Kiểm thử (`pytest -q`)
- Kết quả: **`56 passed, 2 warnings in 65.10s`** (gồm 51 test hiện có + 5 test mới trong [`tests/test_phase5c_data_prep.py`](file:///d:/Kì%207/Hoc_sau_ver4/tests/test_phase5c_data_prep.py)).
