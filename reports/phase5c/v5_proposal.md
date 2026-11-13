# Đề xuất Pha 5c — Lưới v5 (Tối ưu độ trễ I3 bằng Cache & Kết hợp `tier_soft`)

> **Trạng thái:** ĐỀ XUẤT — **CHƯA** chạy thực nghiệm, **CHƯA** khóa `selection_rule_v5.json/.md`. Chỉ khóa và thực thi sau khi người dùng duyệt.
> **Tuân thủ:** `AGENTS.md` (Luật 1, 3, 4, 5, 6, 7, C.2, C.5, C.8, C.9, C.10), `seed=42`, không sửa `best_model.pt`, `parser_v1.py`, `parser_v2.py` (`v2.1`), các tập khóa và `selection_rule_v3/v4`.

---

## 1. Lưới cấu hình v5 (tối đa 8 cấu hình) & Thứ tự độ đơn giản

Kết quả Bước 4 (`v4`) cho thấy:
- `cfg_v4_05` (`I1_tier_soft`, `gamma=1.5`) đạt `CV Score = 0.5296` (tốt nhất trong các cấu hình đạt `5(i)-(v)`), nhưng chưa kết hợp với `I3`.
- `cfg_v4_14` (`I3_multi_template_batch`, `k=2`) đạt `CV Score = 0.6037` (đặc biệt kéo `B-clean mAP` từ `0.2473` lên `0.5216`), nhưng bị **FAIL `5(iv)`** do tốn thêm ~1 forward text trực tuyến mỗi truy vấn.

Lưới **v5** gồm **8 cấu hình** nhằm kiểm định việc thay `I3` encode trực tiếp bằng **cache embedding mẫu chuẩn hóa (offline)** để triệt tiêu độ trễ tăng thêm của `I3`, đồng thời kết hợp `I3_cached` với `cfg_v4_05` (`tier_soft`):

| Config ID | Vai trò / Ablation | `bonus_mode` | `gamma` | `I3` (đa mẫu `k=2`) | `conf_gating` (I2) | `fuzzy_typo` (I7) | `text_gallery` (I5) |
|---|---|---|---|---|---|---|---|
| `cfg_v5_01` | Baseline v4 winner (`cfg_v4_05`) | `tier_soft` | 1.5 | `none` | `false` | `false` | `none` |
| `cfg_v5_02` | Đối chứng `I3` direct (`cfg_v4_14`) | `zscore` | 1.5 | `direct` (2 forward) | `false` | `false` | `none` |
| `cfg_v5_03` | Ablation `I3_cached` (tương đương `02`) | `zscore` | 1.5 | `cached` (1 forward) | `false` | `false` | `none` |
| `cfg_v5_04` | Kết hợp `cfg_v4_05` + `I3_cached` | `tier_soft` | 1.5 | `cached` (1 forward) | `false` | `false` | `none` |
| `cfg_v5_05` | Ablation `+I2` trên nền `04` | `tier_soft` | 1.5 | `cached` (1 forward) | `true` | `false` | `none` |
| `cfg_v5_06` | Ablation `+I7` trên nền `04` | `tier_soft` | 1.5 | `cached` (1 forward) | `false` | `true` | `none` |
| `cfg_v5_07` | Tổ hợp `04 + I2 + I7` | `tier_soft` | 1.5 | `cached` (1 forward) | `true` | `true` | `none` |
| `cfg_v5_08` | Tổ hợp `04 + I5` (`title_vi`, `alpha=0.85`) | `tier_soft` | 1.5 | `cached` (1 forward) | `false` | `false` | `title_vi` |

### Thứ tự độ đơn giản (`simplicity_order`, từ đơn giản đến phức tạp)
1. `text_gallery=none` < `text_gallery=title_vi`
2. `I3=none` < `I3=cached` < `I3=direct`
3. `fuzzy_typo=false` < `fuzzy_typo=true`
4. `conf_gating=false` < `conf_gating=true`
5. `bonus_mode=zscore` < `bonus_mode=tier_soft`
6. `smaller_gamma` < `larger_gamma`

### Phương án RIÊNG chờ người dùng duyệt: Mở rộng `γ` cho `tier_soft`
Trong lưới v4, `tier_soft` chỉ có 1 điểm `γ = 1.5` (vừa là điểm duy nhất vừa là biên lưới). Nếu người dùng duyệt mở rộng, có thể thêm nhánh quét `γ ∈ {1.0, 1.5, 2.0, 2.5}` cho `tier_soft` (trên nền `cfg_v5_01` hoặc `cfg_v5_04`), **tách riêng** hoặc thay thế 2–3 slot cuối để giữ tổng số cấu hình `<= 8`. Agent không tự mở rộng khi chưa được duyệt.

---

## 2. Thiết kế Cache Embedding Mẫu Chuẩn Hóa (`I3_cached`)

- **Nguồn sinh mẫu (chống rò rỉ tuyệt đối):** Toàn bộ mẫu chuẩn hóa được sinh **chỉ từ bảng nhãn hợp lệ trong `catalog_rebuild.parquet`** (các tổ hợp `(articleType, gender, baseColour)` và các tổ hợp 2-slot / 1-slot có trong catalog, dùng đúng `build_catalog_template(spec)` với `ARTICLE_TYPE_LEXICON_V2`, `GENDER_LEXICON_V2`, `COLOUR_LEXICON_V2`). **Tuyệt đối không** đọc hay sinh từ câu truy vấn của `DEV`, `Val A`, `B-synth`, `C`, `B-human` hay `test`.
- **Precompute offline & Lưu trữ:**
  - Dùng đúng `SiglipPreprocessor` (`max_length=128`, `padding="max_length"`, Luật 4) và `best_model.pt` để encode toàn bộ mẫu chuẩn hóa thành `index/template_cache_v5/template_embeddings.npy` (float32, L2-normalized) + `template_key_map.json`.
  - Lưu `index/template_cache_v5/manifest.json` gồm: `checkpoint_hash`, `catalog_hash`, `cache_hash` (SHA-256 của `template_embeddings.npy` + `template_key_map.json`), `n_templates`, `seed=42`, `commit_hash`.
- **Tra cứu trực tuyến (1 forward GPU duy nhất):**
  - Khi nhận truy vấn `q`: chạy `restore_diacritics_for_query(q)` + `parser_v2.1` (hoặc `parser_v2_conf`), encode câu đã khôi phục dấu **1 lần** (`v_q`).
  - Tạo khóa tra cache `(tuple(sorted(articleType)), gender or "", tuple(sorted(baseColour)))`. Nếu khớp khóa trong `template_key_map.json`, lấy `v_tpl` từ RAM và tính `v_final = normalize(0.5 * v_q + 0.5 * v_tpl)`. Không phát sinh forward text thứ 2 trên GPU.
- **Kiểm định bắt buộc (pytest):**
  - **Test tương đương:** Trên 200 tổ hợp thuộc tính lấy ngẫu nhiên từ catalog (`seed=42`), cosine giữa vector từ `I3_cached` và vector encode trực tiếp từ `multi_template_encode_texts(..., n_templates=2)` phải **`>= 0.999`**.
  - **Hash Guard:** Engine từ chối nạp `template_cache_v5` nếu `checkpoint_hash`, `catalog_hash` hoặc `cache_hash` bị lệch.

---

## 3. Kế hoạch tập xác nhận mới (`B-human-v4` & held-out v2)

- **Cổng `B-human-v4`:** Người dùng tự viết (hoặc kiểm duyệt độc lập) `>= 60` câu truy vấn hợp lệ (đạt chuẩn `check_b_human.py`, đủ `>= 30` cụm `spec_cluster_id`, không trùng Jaccard `>= 0.8` với `DEV / A / B-synth / C / B-human cũ`), khóa bằng SHA-256 trước khi chạy lưới v5.
- **Tập held-out v2 cho Bước 5:** Chỉ chạy đúng **1 lần** trên `index/test_gallery/` sau khi chọn xong cấu hình thắng từ v5 (Mục C.5).
- **Quy ước khi thiếu tập mới:** Nếu `B-human-v4` chưa được tạo trước khi chạy v5, mọi con số trên `B-human` cũ (65 câu) **chỉ được ghi nhãn THAM KHẢO**, không dùng làm cổng xác nhận độc lập.

---

## 4. Giao thức chọn & đánh giá v5

- **Ràng buộc giữ nguyên 100% từ v4:** `5(i)` (`C-20%` mAP không giảm quá ngưỡng), `5(ii)` (catastrophic drop rate trên `B-clean` `<= 0.05`), `5(iii)` (`R@10` trên `A` & `B-clean` không tụt quá `0.01`), `5(iv)` (`T2I p95 < 200 ms` và tăng `<= +20 ms` so với `cfg_v4_01`), `5(v)` (`FP` parser không vượt `parser_v1`).
- **Chọn cấu hình:** 5-fold cluster CV (`0.4 * mAP_A_cluster + 0.6 * mAP_B_clean_cluster`) trên `folds.json`, kết hợp paired cluster bootstrap (1.000 lần, `seed=42`) để tie-break theo `simplicity_order`.
- **Báo cáo song song:** Tách rõ **"Tốt nhất"**, **"An toàn"** (đạt `5(i)` + `5(ii)` đơn giản nhất), **"Model thuần"**, và **ablation từng ý tưởng** (`I1_tier_soft`, `I3_cached`, `I2`, `I7`, `I5`).

---

## 5. Giới hạn trung thực (Luật 1 & C.10)

- Chỉ tiêu **`Recall@10 >= 0.85`** và **`mAP >= 0.75`** chỉ được xét trên **`index/test_gallery/`** (Mục C.2). Mọi kết quả trên `val_gallery` (kể cả `Val A`) là ước lượng lạc quan/tham khảo vì ảnh val nằm trong tập checkpoint đã thấy.
- Điểm hệ thống chịu giới hạn đồng thời từ 3 phía: (1) **Model** (`best_model.pt` giữ nguyên không fine-tune), (2) **Parser** (độ phủ từ vựng trên câu tự nhiên/ngoài lexicon), và (3) **GT 3-slot** (`articleType`, `gender`, `baseColour`).
- Nếu sau Bước 5 trên `test_gallery`, hệ thống hoặc `model_only` vẫn chưa đạt ngưỡng `0.85` / `0.75`, báo cáo sẽ ghi nhận trung thực nguyên nhân và để lựa chọn **"cần train"** cho **NGƯỜI DÙNG** quyết định theo Luật 1; agent không tự ý đề xuất bắt đầu train.
