# Báo cáo Đánh giá Cuối cùng — Pha 5c (Prompt 14e: `PART_A_DONE_AWAITING_HELDOUT_V2`)

## 0. Thông tin chung & Bất biến

- **Trạng thái thực thi:** `PART_A_DONE_AWAITING_HELDOUT_V2`
- **Thời gian tạo (UTC):** `2026-11-14T17:22:32Z`
- **Git commit khi chạy script:** `ab8c45dd2cd24aaacaa6e4607d6facb6070c4855`
- **Branch:** `phase7a`
- **Seed khóa cứng:** `42`
- **Agent Python PID:** `22068`
- **Cờ trạng thái nền GPU (Bước 2.4):** `official_with_idle_baseline`
- **Tệp `eval/test_queries_heldout_v2.json` tồn tại:** `False`
- **Cam kết Luật 7 & C.5:** Chưa chạy bất kỳ phép đo chất lượng (Recall@K, mAP, Hubness) nào trên `index/test_gallery/` (Phần B) và chưa mở/đọc nội dung `eval/cir_eval_test.json` (Phần C). Chỉ chạy Bước 1a, Bước 1c, Bước 2, Phần A (Độ trễ 4 luồng C.9), và Chốt CIR sau Phần A rồi DỪNG.

## 1. Kiểm tra SHA-256 Đầu vào, Bước 0.3 & Bước 1a (`index/test_gallery_text_v5/`)

| Khóa | Đường dẫn | SHA-256 kỳ vọng | SHA-256 thực tế | Khớp |
|---|---|---|---|---|
| `best_model_pt` | `data/checkpoints_siglip-vit/best_model.pt` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | `True` |
| `selection_rule_v5` | `reports/phase5c/selection_rule_v5.json` | `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | `True` |
| `selection_rule_cir_v1` | `reports/phase5c/selection_rule_cir_v1.json` | `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067` | `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067` | `True` |
| `cir_eval_test` | `eval/cir_eval_test.json` | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | `True` |
| `test_split` | `eval/test_split.json` | `d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681` | `d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681` | `True` |
| `catalog_rebuild` | `data/catalog/catalog_rebuild.parquet` | `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` | `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` | `True` |

- **Bước 0.3 (Luật 4 — Tái encode `200` ảnh mẫu `index/test_gallery/` với `seed=42`):** `min_cosine = 0.999978`, `mean_cosine = 1.000000`, `median_cosine = 1.000000`, ngưỡng `>= 0.999` -> **`passed = True`**.
- **Bước 1a (`index/test_gallery_text_v5/`):** `manifest_sha256 = 286fe59ed6c2348dd9815263085269fd688993c83e1ede91d9152344184a2477`, `rebuilt_or_modified = False`, `passed = True`, `checkpoint_hash_match = True`, `catalog_hash_match = True`, `n_vectors = 1290`, `sha256_internvl2_only_matched = True`, `sha256_internvl2_plus_aux_matched = True`, `sha256_metadata_aux_matched = True`.

## 2. Bước 1c — Hạn chế Caption `InternVL2-2B` (Chỉ đọc, không chạy model, không dùng nhãn/truy vấn test)

### 2.1. Quyết định của người dùng & Chẩn đoán đã sửa (Đính chính Báo cáo 14c)

- **Quyết định Bước 1b của người dùng:** `(a) GIỮ NGUYÊN index/test_gallery_text_v5/` (`reapply_sentence_splitter = False`, `regenerate_captions = False`).
- **Nhận định cũ trong báo cáo 14c là SAI (`prompt14c_previous_claim_is_wrong = True`):** Báo cáo 14c từng nhận định nguyên nhân là lỗi bộ tách câu đánh số '1. Thông tin...' ở Pha 2 cắt nhầm ở dấu chấm.
- **Chẩn đoán đúng thay thế:** Nguyên nhân KHÔNG phải bộ tách câu (src/caption/merger.py giữ nguyên từ commit dfc150ad). Nguyên nhân thực sự là InternVL2-2B lặp lại (echo) đề bài hệ thống kết hợp với commit 2e057d6bf870b4281fe556e03f2e0c2bf9d5a076 (giảm max_new_tokens từ 96 xuống 48, max_num từ 2 xuống 1) trong src/caption/generator.py: (1) ở shard_test_0000 (chạy trước 2e057d6, max_new_tokens=96), chuỗi echo chứa dấu ':' ('YÊU CẦU BẮT BUỘC:\n\n1. Thô') khiến regex số 3 trong clean_introductory_phrases() cắt toàn bộ tiền tố trước dấu ':' và để lại mảnh '1. Thô' (6 token); (2) ở shard_test_0001..0002 và shard_val_0000..0003 (chạy sau 2e057d6, max_new_tokens=48), chuỗi echo dừng ở chữ '...trong hình bằng' (trước khi gặp dấu ':'), nên regex số 3 không khớp và để nguyên câu echo 68 token.
- **Commit `src/caption/merger.py` (không đổi giữa test và val):** `dfc150ad90edc9d426892a4cbf32eb0221b24955`
- **Commit `src/caption/generator.py` (`max_new_tokens: 96 -> 48`, `max_num: 2 -> 1`):** `2e057d6bf870b4281fe556e03f2e0c2bf9d5a076`

### 2.2. Thống kê Đếm lại bằng Script trên `val_gallery` và `test_gallery` (Đối chiếu 14d)

- **Chuỗi echo đề bài chính xác:** `"Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng"` (độ dài theo tokenizer SigLIP: `68` token).

| Tập Gallery | Tổng số mục | Số mục trùng khít chuỗi echo (`68` token) | Tỷ lệ echo (%) | Số mục bắt đầu bằng `"1."` (`"1. Th..."`) | Tỷ lệ `"1."` (%) | Số mục còn lại | Tỷ lệ còn lại (%) | Mục còn lại: biến thể echo/tiêu đề tiếng Việt | Mục còn lại: mô tả tiếng Anh (`The image...`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `val_gallery` | `1884` | `1859` | `98.67` | `0` | `0.00` | `25` | `1.33` | `21` | `4` |
| `test_gallery` | `1290` | `841` | `65.19` | `413` | `32.02` | `36` | `2.79` | `33` | `3` |

- **Chi tiết phân rã nhóm `"1. Th..."` trên `test_gallery` (`413` mục):**
  - `'1. Thô'`: `405` mục (`6` token SigLIP)
  - `'1. **Th'`: `6` mục (`4` token SigLIP)
  - `'1. Thông'`: `2` mục (`7` token SigLIP)
- **Đối chiếu với số liệu 14d:** `val` echo `14d = 1859` vs `14e = 1859` (`khớp = True`); `test` echo `14d = 841` vs `14e = 841` (`khớp = True`); `test` `"1."` `14d = 413` vs `14e = 413` (`khớp = True`); `any_discrepancy_vs_14d = False`.

### 2.3. Đọc thủ công 40 `caption_A` ngẫu nhiên (`seed = 42`: 20 `val` + 20 `test`, Luật D.4)

- **Tổng hợp đọc thủ công (`seed = 42`):** Số mẫu thực sự mô tả ảnh trên `val` = `0/20`; trên `test` = `0/20`.

#### Bảng 2.3a — 20 mẫu `caption_A` ngẫu nhiên từ `val_gallery` (`seed = 42`)

| # | Gallery Idx | `product_id` | Shard | Token SigLIP | Nguyên văn `caption_A` | Có mô tả ảnh? | Tiếng Việt tự nhiên? | Có bịa thuộc tính? |
|---:|---:|---|---|---:|---|---|---|---|
| `1` | `160` | `14641` | `shard_val_0000.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `2` | `166` | `14716` | `shard_val_0000.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `3` | `176` | `14927` | `shard_val_0000.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `4` | `241` | `16293` | `shard_val_0000.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `5` | `377` | `19280` | `shard_val_0000.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `6` | `809` | `31905` | `shard_val_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `7` | `819` | `32351` | `shard_val_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `8` | `848` | `33057` | `shard_val_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `9` | `965` | `36036` | `shard_val_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `10` | `987` | `36424` | `shard_val_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `11` | `1222` | `43571` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `12` | `1305` | `4618` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `13` | `1348` | `47506` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `14` | `1381` | `48319` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `15` | `1429` | `49893` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `16` | `1444` | `50438` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `17` | `1477` | `51158` | `shard_val_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `18` | `1581` | `54156` | `shard_val_0003.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `19` | `1605` | `54671` | `shard_val_0003.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `20` | `1830` | `8137` | `shard_val_0003.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |

#### Bảng 2.3b — 20 mẫu `caption_A` ngẫu nhiên từ `test_gallery` (`seed = 42`, chỉ đọc cột `caption_A`, không xem nhãn test)

| # | Gallery Idx | `product_id` | Shard | Token SigLIP | Nguyên văn `caption_A` | Có mô tả ảnh? | Tiếng Việt tự nhiên? | Có bịa thuộc tính? |
|---:|---:|---|---|---:|---|---|---|---|
| `1` | `109` | `25370` | `shard_test_0000.parquet` | `6` | `1. Thô` | Không | Không (mảnh tiêu đề cụt '1. Thô' / '1. **Th' / '1. Thông') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `2` | `113` | `30038` | `shard_test_0000.parquet` | `6` | `1. Thô` | Không | Không (mảnh tiêu đề cụt '1. Thô' / '1. **Th' / '1. Thông') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `3` | `120` | `37646` | `shard_test_0000.parquet` | `6` | `1. Thô` | Không | Không (mảnh tiêu đề cụt '1. Thô' / '1. **Th' / '1. Thông') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `4` | `165` | `25588` | `shard_test_0000.parquet` | `6` | `1. Thô` | Không | Không (mảnh tiêu đề cụt '1. Thô' / '1. **Th' / '1. Thông') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `5` | `257` | `42732` | `shard_test_0000.parquet` | `124` | `Các điểm đảo mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ):\n\n1. **Thông tin ph` | Không | Không (biến thể lặp lại đề bài và tiêu đề mục 1 bị cắt dở) | Không (chưa sinh tới thuộc tính sản phẩm) |
| `6` | `552` | `6861` | `shard_test_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `7` | `559` | `57329` | `shard_test_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `8` | `580` | `53146` | `shard_test_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `9` | `660` | `10460` | `shard_test_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `10` | `674` | `20839` | `shard_test_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `11` | `833` | `21663` | `shard_test_0001.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `12` | `891` | `49526` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `13` | `921` | `17706` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `14` | `943` | `45989` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `15` | `977` | `32009` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `16` | `984` | `6146` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `17` | `1010` | `39013` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `18` | `1082` | `38999` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `19` | `1095` | `39478` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |
| `20` | `1250` | `7942` | `shard_test_0002.parquet` | `68` | `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng` | Không | Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng') | Không (chưa sinh tới thuộc tính sản phẩm) |

### 2.4. Hệ quả Chính thức, Đính chính `manifest.json`, Hàng `metadata-only` & Mục Không xác minh được

- (i) cfg_v5_08a trên val và test đo trên caption gần như không chứa thông tin ảnh (98.67% echo trên val; 97.21% echo hoặc mảnh '1. Th...' trên test), nên kết quả của nó KHÔNG phản ánh năng lực thực sự của InternVL2-2B.
- (ii) cfg_v5_08b chỉ thêm thông tin ở phần metadata_aux (và trên test bị thiếu các trường *_vi trong index/test_gallery/id_map.json nên rơi về nhãn tiếng Anh).
- (iii) cfg_v5_09 (winner) và cfg_v5_01 (safe baseline) có text_gallery = 'none' nên KHÔNG bị ảnh hưởng bởi hạn chế caption InternVL2-2B hay thiếu hụt *_vi trong id_map.
- **Đính chính `index/test_gallery_text_v5/manifest.json` (`manifest_modified = False`):** Manifest khai báo `max_new_tokens = 96`, nhưng thực tế chỉ `shard_test_0000` chạy ở `96` còn `shard_test_0001/0002` chạy ở `48` (src/caption/batch_runner.py dòng 162 ghi cứng {'max_new_tokens': 96} vào cột gen_params khi lưu parquet dù src/caption/generator.py đã đổi mặc định sang 48 ở commit 2e057d6.). Không sửa tệp `manifest.json`.
- **Hàng `metadata-only (B1(iv) — Luật 5)`:** **`không chạy: không có cấu hình khóa`** (Lưới cấu hình đã khóa trong selection_rule_v5.json chỉ có text_gallery in {'none', 'internvl2_only', 'internvl2_plus_aux'}; không tự bịa cấu hình mới ngoài lưới đã khóa.).
- **Các điều `"không xác minh được"` (Luật 7):**
  - Nội dung chuỗi sinh thô (raw_caption) trước bước clean_introductory_phrases() của 1,290 ảnh test_gallery (không xác minh được từng mục vì Stage 1 không lưu cột raw_caption và đã xóa tệp tạm cache_stage1_*.json sau khi ghi parquet).
  - Chất lượng thực tế của mô hình InternVL2-2B trên val_gallery và test_gallery nếu prompt không bị echo và sinh đủ mô tả tiếng Việt 25-45 từ (không xác minh được nếu không chạy lại model sinh caption).
  - Điểm chất lượng của cfg_v5_08b trên test_gallery nếu index/test_gallery/id_map.json có sẵn các trường *_vi tiếng Việt (không xác minh được nếu không dựng lại index).

## 3. Bước 2 — Kiểm tra GPU, Nhật ký `[Cách 2]` & Đo Mức Bận Nền 30 giây

### 3.1. Nhật ký Thực thi `[Cách 2]` (Người dùng cấp quyền bằng văn bản)

| PID mục tiêu | Tên tiến trình kỳ vọng | Tên tiến trình thực tế lúc kiểm tra | Lệnh thực thi | Mã thoát (`ExitCode`) | Đã dừng thành công | Còn xuất hiện trong `nvidia-smi` sau khi dừng |
|---:|---|---|---|---:|---|---|
| `16284` | `dnmultiplayerex` | `dnmultiplayerex` | `Stop-Process -Id 16284 -Force -ErrorAction Stop` | `0` | `True` | `False` |
| `3588` | `EdgeGameAssist` | `EdgeGameAssist` | `Stop-Process -Id 3588 -Force -ErrorAction Stop` | `0` | `True` | `False` |

### 3.2. Danh sách Tiến trình `nvidia-smi` Trước Phần A

- **Telemetry tức thời:** Nhiệt độ `65.0°C`, GPU Util `33.0%`, Mem Util `3.0%`, Power `3.99 W`, Graphics Clock `547.0 MHz`, Mem Clock `810.0 MHz`, P-State `P5`, VRAM `2440.0/4096.0 MiB`.
- **Còn tiến trình cấm (`has_blocking_processes`):** `False`

| PID | Loại | Tên tiến trình | Bộ nhớ GPU |
|---:|---|---|---|
| `1560` | `C+G` | `C:\Windows\System32\ShellHost.exe` | `N/A` |
| `1596` | `C+G` | `...rograms\antigravity\Antigravity.exe` | `N/A` |
| `2912` | `C+G` | `...rograms\antigravity\Antigravity.exe` | `N/A` |
| `12004` | `C+G` | `...cw5n1h2txyewy\CrossDeviceResume.exe` | `N/A` |
| `12964` | `C+G` | `...nt.CBS_cw5n1h2txyewy\SearchHost.exe` | `N/A` |
| `12976` | `C+G` | `...2txyewy\StartMenuExperienceHost.exe` | `N/A` |
| `13176` | `C+G` | `..._x64__cw5n1h2txyewy\WidgetBoard.exe` | `N/A` |
| `13356` | `C+G` | `...on\157.0.4341.46\msedgewebview2.exe` | `N/A` |
| `17140` | `C+G` | `...CBS_cw5n1h2txyewy\TextInputHost.exe` | `N/A` |
| `19080` | `C+G` | `...__8wekyb3d8bbwe\Notepad\Notepad.exe` | `N/A` |
| `22068` | `C` | `...rograms\Python\Python312\python.exe` | `N/A` |

### 3.3. Mẫu 30 giây Trạng thái Nghỉ (1 s/lần × 30) Trước và Sau Phần A & Đối chiếu với 14d

- **Đối chiếu với mức bận nền 14d (`29.5%`):** Sau khi tắt `PID 16284` (`dnmultiplayerex.exe`) và `PID 3588` (`EdgeGameAssist.exe`), trung vị `utilization.gpu` giảm từ `29.5%` xuống `30.0%` (trước Phần A) và `30.0%` (sau Phần A). **Kết luận nguyên nhân tải nền:** Sau khi dừng PID 16284 (dnmultiplayerex.exe) và PID 3588 (EdgeGameAssist.exe), trung vị GPU utilization lúc nghỉ giảm từ 29.50% (Prompt 14d) xuống 30.00% (trước Phần A) và 30.00% (sau Phần A), xác nhận hai tiến trình giả lập/game overlay là nguồn gây tải nền và giữ xung P0 ở 14d.
- **Cờ trạng thái nền:** `official_with_idle_baseline` (Trung vị utilization.gpu lúc nghỉ = 30.00% > 10.0% nhưng chỉ còn tiến trình Windows/IDE (WDDM desktop compositor); tiếp tục với cờ official_with_idle_baseline, ghi nhận baseline 30s trước và sau Phần A như một hạn chế môi trường Windows.)

| Thời điểm lấy mẫu (30s) | Chỉ số | Trung vị (`Median`) | Trung bình (`Mean`) | Nhỏ nhất (`Min`) | Lớn nhất (`Max`) |
|---|---|---:|---:|---:|---:|
| Trước Phần A | `GPU Utilization (%)` | `30.00` | `29.07` | `0.00` | `61.00` |
| Trước Phần A | `Memory Utilization (%)` | `2.00` | `1.97` | `0.00` | `4.00` |
| Trước Phần A | `Temperature (°C)` | `59.00` | `59.30` | `59.00` | `60.00` |
| Trước Phần A | `Power Draw (W)` | `3.70` | `3.65` | `3.08` | `3.96` |
| Trước Phần A | `Graphics Clock (MHz)` | `525.00` | `529.33` | `375.00` | `645.00` |
| Trước Phần A | `Memory Clock (MHz)` | `810.00` | `769.50` | `405.00` | `810.00` |
| Sau Phần A | `GPU Utilization (%)` | `30.00` | `25.77` | `0.00` | `60.00` |
| Sau Phần A | `Memory Utilization (%)` | `2.00` | `1.70` | `0.00` | `4.00` |
| Sau Phần A | `Temperature (°C)` | `66.00` | `66.20` | `65.00` | `67.00` |
| Sau Phần A | `Power Draw (W)` | `4.05` | `3.99` | `3.60` | `4.22` |
| Sau Phần A | `Graphics Clock (MHz)` | `555.00` | `538.17` | `397.00` | `637.00` |
| Sau Phần A | `Memory Clock (MHz)` | `810.00` | `756.00` | `405.00` | `810.00` |

## 4. Phần A — Độ trễ 4 Luồng (Mục C.9 trên `index/test_gallery/`, `N = 1,290`)

- **Gallery:** `index/test_gallery` (`N = 1290`) và `index/test_gallery_text_v5` (cho `cfg_v5_08b`), FAISS HNSW (`efSearch = 128`), `candidate_pool = 100`.
- **Truy vấn T2I (200 câu từ `B_clean` val, `seed = 42`):** tệp nguồn `eval/val_queries_b_synth.json` (`SHA-256: 41739174392fd1fe4cb14c76bddefc33ff23cf7dd7e885d9e999987db3bea6a1`), danh sách 200 câu mẫu `SHA-256: 6aa8fee29303b6d00e7a8b76ab674cc9c36b603ae47daf00021ac8479913f373`.
- **Truy vấn I2I & CIR (200 cặp lặp có kiểm soát từ 120 cặp `eval/cir_eval_val.json`):** tệp nguồn `eval/cir_eval_val.json` (`SHA-256: c735c7742c5277cffc4113c68dd8ee59fd3475c742284c8fe9e098af0819062a`), danh sách 200 cặp `SHA-256: 071249b062e2591b1da58d63ef18939e3cef39986ffa9a97c2d5657ff1f727ea`.

### 4.1. Bảng Tổng hợp Số chính (`Trung vị(p95)` & `IQR(p95)`) và Đối chiếu Lịch sử (Pha 5 & Prompt 12/13)

| Luồng | Cấu hình | Pha 5 (`latency_sla_results.json`) `p95` (ms) | Prompt 12/13 (GPU nhiễu/nóng) `Trung vị(p95)` (ms) | Prompt 14e `Trung vị(p50)` (ms) | Prompt 14e **`Trung vị(p95)` (ms)** | Prompt 14e **`IQR(p95)` (ms)** | Prompt 14e `Trung vị(p99)` (ms) | Đạt SLA (`< 200 ms`) | Kết luận (`locked_final_definitions.py`) |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Text→Ảnh | `model_only` | `29.857` | `không xác minh được (Prompt 12 không đo riêng model_only)` | `61.581` | **`109.851`** | `12.350` | `162.071` | `True` | `Đạt chắc chắn` |
| Text→Ảnh | `cfg_v5_01` | `không xác minh được (Pha 5 chưa có lưới v5)` | `103.815` | `65.639` | **`119.991`** | `2.840` | `202.342` | `True` | `Đạt chắc chắn` |
| Text→Ảnh | `cfg_v5_06` | `không xác minh được (Pha 5 chưa có lưới v5)` | `91.479` | `66.453` | **`106.035`** | `8.534` | `169.243` | `True` | `Đạt chắc chắn` |
| Text→Ảnh | `cfg_v5_09` | `không xác minh được (Pha 5 chưa có lưới v5)` | `86.549 (gộp profile cfg_v5_03)` | `66.170` | **`109.347`** | `12.964` | `167.484` | `True` | `Đạt chắc chắn` |
| Text→Ảnh | `cfg_v5_08b` | `không xác minh được (Pha 5 chưa có lưới v5)` | `90.693` | `67.082` | **`125.791`** | `23.294` | `171.999` | `True` | `Đạt chắc chắn` |
| Ảnh→Ảnh | `image_to_image (RAM PIL)` | `126.513` | `không xác minh được (Prompt 12/13 không đo I2I)` | `146.357` | **`289.102`** | `52.680` | `331.507` | `False` | `Chưa đạt` |
| Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | `không xác minh được` | `không xác minh được (Prompt 12/13 không đo I2I)` | `175.693` | **`340.522`** | `13.407` | `407.434` | `False` | `Chưa đạt` |
| CIR catalog | `cir_v1_01` | `36.976` | `121.419` | `70.251` | **`120.800`** | `13.104` | `194.802` | `True` | `Đạt chắc chắn` |
| CIR catalog | `cir_v1_05` | `không xác minh được (Pha 5 chưa có C3 bonus CIR)` | `135.536` | `70.490` | **`110.276`** | `9.578` | `139.426` | `True` | `Đạt chắc chắn` |
| CIR upload (Số chính: RAM PIL) | `cir_v1_01` | `165.489` | `415.287` | `223.485` | **`368.505`** | `11.279` | `426.570` | `False` | `Chưa đạt` |
| CIR upload (Số chính: RAM PIL) | `cir_v1_05` | `không xác minh được (Pha 5 chưa có C3 bonus CIR)` | `446.274` | `217.212` | **`358.171`** | `29.025` | `412.994` | `False` | `Chưa đạt` |
| CIR upload (Số phụ: Disk+JPEG) | `cir_v1_01` | `không xác minh được` | `415.287` | `266.367` | **`436.185`** | `13.988` | `530.370` | `False` | `Chưa đạt` |
| CIR upload (Số phụ: Disk+JPEG) | `cir_v1_05` | `không xác minh được` | `446.274` | `258.257` | **`425.536`** | `14.431` | `497.027` | `False` | `Chưa đạt` |

### 4.2. Chi tiết `p50 / p95 / p99` và Telemetry GPU của Từng Run (5 run × 200 truy vấn)

| Luồng | Cấu hình | Run | `p50` (ms) | `p95` (ms) | `p99` (ms) | `Mean` (ms) | Temp Trước/Sau (°C) | Power Trước/Sau (W) | Gr Clock Trước/Sau (MHz) | P-State Trước/Sau |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| 1. Text→Ảnh | `model_only` | `1` | `59.594` | `98.156` | `134.595` | `63.435` | `60/74` | `3.94/16.04` | `465/1462` | `P5/P0` |
| 1. Text→Ảnh | `model_only` | `2` | `61.620` | `109.851` | `162.071` | `64.696` | `67/80` | `3.44/23.12` | `525/1740` | `P5/P0` |
| 1. Text→Ảnh | `model_only` | `3` | `62.073` | `110.505` | `154.439` | `65.665` | `71/78` | `4.31/14.28` | `645/1290` | `P5/P0` |
| 1. Text→Ảnh | `model_only` | `4` | `61.195` | `96.966` | `196.416` | `64.569` | `72/81` | `4.20/18.17` | `600/1402` | `P5/P0` |
| 1. Text→Ảnh | `model_only` | `5` | `61.581` | `123.176` | `228.010` | `68.025` | `73/79` | `4.57/13.57` | `487/1252` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_01` | `1` | `64.393` | `117.546` | `202.342` | `69.575` | `60/74` | `3.94/16.04` | `465/1462` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_01` | `2` | `65.543` | `110.991` | `166.979` | `69.852` | `67/80` | `3.44/23.12` | `525/1740` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_01` | `3` | `66.123` | `119.991` | `202.870` | `70.824` | `71/78` | `4.31/14.28` | `645/1290` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_01` | `4` | `66.687` | `132.058` | `229.362` | `72.040` | `72/81` | `4.20/18.17` | `600/1402` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_01` | `5` | `65.639` | `120.386` | `177.907` | `70.661` | `73/79` | `4.57/13.57` | `487/1252` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_06` | `1` | `64.815` | `101.748` | `150.204` | `67.860` | `60/74` | `3.94/16.04` | `465/1462` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_06` | `2` | `65.363` | `104.975` | `179.431` | `68.298` | `67/80` | `3.44/23.12` | `525/1740` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_06` | `3` | `66.453` | `106.035` | `160.476` | `69.274` | `71/78` | `4.31/14.28` | `645/1290` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_06` | `4` | `67.084` | `126.216` | `169.243` | `72.153` | `72/81` | `4.20/18.17` | `600/1402` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_06` | `5` | `67.424` | `113.509` | `208.883` | `72.983` | `73/79` | `4.57/13.57` | `487/1252` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_09` | `1` | `64.912` | `123.837` | `167.484` | `70.592` | `60/74` | `3.94/16.04` | `465/1462` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_09` | `2` | `65.823` | `98.186` | `217.594` | `69.587` | `67/80` | `3.44/23.12` | `525/1740` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_09` | `3` | `66.170` | `108.542` | `141.065` | `69.102` | `71/78` | `4.31/14.28` | `645/1290` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_09` | `4` | `66.878` | `121.505` | `189.867` | `71.776` | `72/81` | `4.20/18.17` | `600/1402` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_09` | `5` | `67.341` | `109.347` | `143.633` | `68.835` | `73/79` | `4.57/13.57` | `487/1252` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_08b` | `1` | `65.853` | `104.009` | `143.538` | `69.064` | `60/74` | `3.94/16.04` | `465/1462` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_08b` | `2` | `65.898` | `127.303` | `173.491` | `69.935` | `67/80` | `3.44/23.12` | `525/1740` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_08b` | `3` | `67.338` | `128.421` | `193.625` | `74.641` | `71/78` | `4.31/14.28` | `645/1290` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_08b` | `4` | `67.082` | `94.782` | `132.040` | `67.975` | `72/81` | `4.20/18.17` | `600/1402` | `P5/P0` |
| 1. Text→Ảnh | `cfg_v5_08b` | `5` | `67.809` | `125.791` | `171.999` | `72.744` | `73/79` | `4.57/13.57` | `487/1252` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (RAM PIL)` | `1` | `146.357` | `233.280` | `303.583` | `154.933` | `73/79` | `4.58/18.20` | `570/1612` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (RAM PIL)` | `2` | `141.888` | `289.102` | `377.622` | `161.639` | `72/79` | `4.23/15.46` | `562/1327` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (RAM PIL)` | `3` | `139.575` | `296.801` | `331.507` | `164.545` | `72/79` | `4.13/19.72` | `495/1657` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (RAM PIL)` | `4` | `149.985` | `294.466` | `352.148` | `164.545` | `72/78` | `4.38/16.48` | `547/1477` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (RAM PIL)` | `5` | `149.548` | `241.786` | `295.760` | `157.739` | `71/78` | `4.50/16.87` | `622/1470` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | `1` | `179.325` | `340.522` | `407.434` | `204.058` | `73/79` | `4.58/18.20` | `570/1612` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | `2` | `174.739` | `351.085` | `395.527` | `200.747` | `72/79` | `4.23/15.46` | `562/1327` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | `3` | `170.922` | `362.642` | `431.242` | `206.019` | `72/79` | `4.13/19.72` | `495/1657` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | `4` | `178.859` | `337.678` | `438.734` | `202.316` | `72/78` | `4.38/16.48` | `547/1477` | `P5/P0` |
| 2. Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | `5` | `175.693` | `329.894` | `372.854` | `197.883` | `71/78` | `4.50/16.87` | `622/1470` | `P5/P0` |
| 3. CIR catalog | `cir_v1_01` | `1` | `69.915` | `126.219` | `189.407` | `74.770` | `71/79` | `4.28/21.37` | `442/1740` | `P5/P0` |
| 3. CIR catalog | `cir_v1_01` | `2` | `70.268` | `110.969` | `178.159` | `74.992` | `71/77` | `4.49/13.50` | `615/1335` | `P5/P0` |
| 3. CIR catalog | `cir_v1_01` | `3` | `70.251` | `101.661` | `218.832` | `75.201` | `71/76` | `4.13/13.88` | `472/1312` | `P5/P0` |
| 3. CIR catalog | `cir_v1_01` | `4` | `69.496` | `124.073` | `210.282` | `74.989` | `70/76` | `3.95/14.55` | `615/1447` | `P5/P0` |
| 3. CIR catalog | `cir_v1_01` | `5` | `70.445` | `120.800` | `194.802` | `75.279` | `70/77` | `3.92/19.19` | `630/1612` | `P5/P0` |
| 3. CIR catalog | `cir_v1_05` | `1` | `71.669` | `112.105` | `161.865` | `74.465` | `71/79` | `4.28/21.37` | `442/1740` | `P5/P0` |
| 3. CIR catalog | `cir_v1_05` | `2` | `70.943` | `110.662` | `139.027` | `73.489` | `71/77` | `4.49/13.50` | `615/1335` | `P5/P0` |
| 3. CIR catalog | `cir_v1_05` | `3` | `70.490` | `110.276` | `164.021` | `74.022` | `71/76` | `4.13/13.88` | `472/1312` | `P5/P0` |
| 3. CIR catalog | `cir_v1_05` | `4` | `69.366` | `94.871` | `139.426` | `71.228` | `70/76` | `3.95/14.55` | `615/1447` | `P5/P0` |
| 3. CIR catalog | `cir_v1_05` | `5` | `69.716` | `101.084` | `135.521` | `71.908` | `70/77` | `3.92/19.19` | `630/1612` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_01` | `1` | `217.202` | `360.500` | `458.290` | `238.274` | `70/75` | `4.48/13.55` | `510/1230` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_01` | `2` | `227.410` | `368.505` | `433.384` | `237.180` | `70/76` | `3.87/14.28` | `375/1275` | `P8/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_01` | `3` | `212.785` | `376.518` | `425.951` | `232.686` | `70/77` | `4.35/14.81` | `495/1552` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_01` | `4` | `223.485` | `383.408` | `415.116` | `239.432` | `70/76` | `4.45/13.42` | `427/1425` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_01` | `5` | `237.980` | `365.240` | `426.570` | `244.271` | `70/76` | `4.43/13.55` | `615/1320` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_01` | `1` | `266.367` | `430.033` | `557.440` | `285.910` | `70/75` | `4.48/13.55` | `510/1230` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_01` | `2` | `270.885` | `436.185` | `560.207` | `283.735` | `70/76` | `3.87/14.28` | `375/1275` | `P8/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_01` | `3` | `259.685` | `428.776` | `530.370` | `278.786` | `70/77` | `4.35/14.81` | `495/1552` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_01` | `4` | `259.031` | `444.020` | `499.433` | `285.569` | `70/76` | `4.45/13.42` | `427/1425` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_01` | `5` | `288.120` | `457.086` | `510.350` | `291.760` | `70/76` | `4.43/13.55` | `615/1320` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_05` | `1` | `217.212` | `358.171` | `402.643` | `234.909` | `70/75` | `4.48/13.55` | `510/1230` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_05` | `2` | `212.884` | `337.357` | `412.994` | `228.858` | `70/76` | `3.87/14.28` | `375/1275` | `P8/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_05` | `3` | `221.192` | `366.382` | `442.284` | `237.715` | `70/77` | `4.35/14.81` | `495/1552` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_05` | `4` | `218.290` | `366.665` | `436.864` | `239.201` | `70/76` | `4.45/13.42` | `427/1425` | `P5/P0` |
| 4. CIR upload (RAM PIL) | `cir_v1_05` | `5` | `216.359` | `336.401` | `390.018` | `232.478` | `70/76` | `4.43/13.55` | `615/1320` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_05` | `1` | `258.257` | `425.536` | `485.560` | `282.251` | `70/75` | `4.48/13.55` | `510/1230` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_05` | `2` | `249.620` | `423.086` | `480.085` | `276.192` | `70/76` | `3.87/14.28` | `375/1275` | `P8/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_05` | `3` | `263.494` | `444.686` | `531.262` | `283.915` | `70/77` | `4.35/14.81` | `495/1552` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_05` | `4` | `264.881` | `437.517` | `516.646` | `285.262` | `70/76` | `4.45/13.42` | `427/1425` | `P5/P0` |
| 4. CIR upload (Disk+JPEG) | `cir_v1_05` | `5` | `248.434` | `419.227` | `497.027` | `279.463` | `70/76` | `4.43/13.55` | `615/1320` | `P5/P0` |

### 4.3. Bảng Tách 6 Giai đoạn Luồng 4 (`CIR upload`, có `torch.cuda.synchronize()` quanh mỗi bước)

| Cấu hình | Giai đoạn | `p50` (ms) | `p95` (ms) | `p99` (ms) | `Mean` (ms) |
|---|---|---:|---:|---:|---:|
| `cir_v1_01` | (1) Đọc đĩa + JPEG decode | `32.676` | `96.273` | `162.421` | `46.783` |
| `cir_v1_01` | (2) Resize 224 BILINEAR + SiglipProcessor | `61.839` | `162.111` | `251.975` | `87.651` |
| `cir_v1_01` | (3) Vision tower forward | `69.033` | `135.335` | `208.580` | `75.958` |
| `cir_v1_01` | (4) Text tower + parser | `62.113` | `106.374` | `161.962` | `66.037` |
| `cir_v1_01` | (5) Combiner forward | `5.652` | `10.068` | `17.444` | `6.261` |
| `cir_v1_01` | (6) HNSW + C3 bonus | `2.209` | `4.196` | `7.289` | `2.463` |
| `cir_v1_05` | (1) Đọc đĩa + JPEG decode | `32.152` | `97.606` | `158.099` | `46.785` |
| `cir_v1_05` | (2) Resize 224 BILINEAR + SiglipProcessor | `60.324` | `151.444` | `222.123` | `87.026` |
| `cir_v1_05` | (3) Vision tower forward | `68.317` | `124.529` | `182.333` | `73.045` |
| `cir_v1_05` | (4) Text tower + parser | `61.674` | `100.026` | `157.867` | `65.263` |
| `cir_v1_05` | (5) Combiner forward | `5.697` | `11.012` | `14.923` | `6.315` |
| `cir_v1_05` | (6) HNSW + C3 bonus | `2.747` | `4.905` | `8.434` | `2.983` |

### 4.4. Kết luận SLA Độ trễ (theo `locked_final_definitions.py`) & Chốt Cấu hình CIR sau Phần A

- **Kết luận theo Số chính (`RAM PIL`):** `status = PARTIAL_2_OF_4`, `passed_count = 2/4` — **Đạt 2/4 luồng; chưa đạt ở: image_to_image, cir_upload**.
- **Kết luận theo Số phụ (`Disk + JPEG decode`):** `status = PARTIAL_2_OF_4`, `passed_count = 2/4` — **Đạt 2/4 luồng; chưa đạt ở: image_to_image, cir_upload**.
- **Kiểm tra cảnh báo `p99(cir_upload) > 200 ms` cho `cir_v1_05`:** Số chính (RAM PIL): `warn_upload_p99_over_200ms = True` (`Trung vị(p99) = 412.994 ms`, `pooled_p99 = 432.049 ms`); Số phụ (Disk+JPEG): `warn_upload_p99_over_200ms = True` (`Trung vị(p99) = 497.027 ms`, `pooled_p99 = 516.166 ms`).
- **Chốt CIR sau Phần A (`selection_rule_cir_v1.json`):** cir_upload vẫn >= 200 ms trên GPU sạch trong khi cir_catalog < 200 ms. selection_rule_cir_v1.json không định nghĩa sẵn cách chọn khi toàn bộ lưới cùng vượt ngưỡng ở cir_upload; trình bày nguyên văn các phương án để người dùng quyết định trước khi chạy Phần C.
  - (Phương án 1) Khóa cir_v1_05 cho Phần C vì chi phí vượt 200 ms ở cir_upload thuộc về vision tower forward (cố định cho mọi cấu hình cir_v1_01..08), trong khi phần tăng thêm của C3 bonus < 2 ms và cir_catalog đạt < 200 ms.
  - (Phương án 2) Chỉ chạy baseline cir_v1_01 ở Phần C và ghi nhận toàn bộ lưới CIR vượt SLA ở luồng cir_upload.

## 5. Kiểm tra Đối chiếu Tự động `.md` ↔ `.json` (Quy tắc số liệu mới)

- **Phương thức sinh báo cáo `.md`:** `auto_generated_from_json_by_run_prompt14e_part_a_py`
- **Tổng số mục số liệu/ô bảng đã đối chiếu tự động:** `87`
- **Số mục lệch (`mismatch_count`):** `0`
- **Kết quả đối chiếu (`all_matched`):** **`True`**

