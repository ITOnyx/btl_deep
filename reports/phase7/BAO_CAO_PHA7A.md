# BÁO CÁO PHA 7a — TIỀN XỬ LÝ & TẠO TẬP ỨNG VIÊN `candidates_v3.parquet`

- **Ngày hoàn tất:** 2026-11-11 18:16:29 | **Seed:** `42` | **Nhánh git:** `phase7a`
- **Đầu ra chính:** [`data/catalog/candidates_v3.parquet`](../../data/catalog/candidates_v3.parquet) (**98,289** dòng × **38** cột), [`eval/phase7/s100k_mapping.csv`](../../eval/phase7/s100k_mapping.csv) (**98,174** dòng).
- **Tuân thủ bất biến:** KHÔNG ghi đè `catalog.parquet`, KHÔNG sửa `src/lexicon_vi.json`, KHÔNG embed/dựng index, KHÔNG đụng `_build/`, KHÔNG sửa/crop ảnh gốc.

---

## 1. Phễu số lượng ảnh qua từng giai đoạn (GĐ1 → GĐ5)

| Giai đoạn | Tổng ảnh | `fashion` | `shopping100k` | Ghi chú |
|---|---:|---:|---:|---|
| Đầu vào `merged_dataset/annotations/all.parquet` (GĐ1) | **139,139** | 40,953 | 98,186 | 100% ảnh đọc được (0 unreadable, 2 ảnh xám `L`, 0 EXIF xoay) |
| Loại ở GĐ3 (`candidate_rejects.csv`) | **-40,850** | -40,838 | -12 | Bảo vệ 94/100 ảnh `fashion legacy_test` (chỉ loại 6 ảnh `low_res=150`) |
| **Tập sạch GĐ3 → GĐ5 (`candidates_v3.parquet`)** | **98,289** | **115** | **98,174** | Tách cờ `serving_eligible`, `eval_ext_eligible`, `bm25_eligible`, `pending_lexicon` |
| ↳ `serving_eligible = True` | **98,279** | 105 | 98,174 | Loại 10 ảnh `fashion` khỏi phục vụ (6 ngoài scope + 4 trùng pHash catalog; giữ phục vụ đánh giá) |
| ↳ `eval_ext_eligible = True` (`seen_in_train='no'`) | **1,562** | 94 | 1,468 | Đánh giá mở rộng Pha 7b (chi tiết overlap dense captions ở mục 5c) |

### Chi tiết lý do loại ở GĐ3 (`candidate_rejects.csv`: 40.850 ảnh)

| Lý do loại | Số ảnh |
|---|---:|
| `duplicate_catalog_inscope_id_or_sha1` | 38,584 |
| `out_of_scope` | 2,237 |
| `low_res` | 15 |
| `duplicate_internal_sha1` | 11 |
| `excluded_keychain` | 2 |
| `duplicate_internal_phash` | 1 |

---

## 2. Phân bố `attr_confidence` cuối & các quy tắc đã kiểm chứng / hạ cấp

| Nguồn | `high` (`AttributeIndex+dense`) | `medium` (`dense_only`) | `low` (`dense_only`) | Tổng |
|---|---:|---:|---:|---:|
| `shopping100k` | **33,101** (33.72%) | **15,389** (15.68%) | **49,684** (50.61%) | **98,174** |
| `fashion` (`serving_eligible`) | **102** | **3** | **0** | **105** (+10 eval_only) |
| **Toàn tập phục vụ (`serving_eligible=True`)** | **33,203** | **15,392** | **49,684** | **98,279** |

### Bảng kiểm chứng tiếp xúc (contact sheet 40 ảnh) & quyết định cập nhật quy tắc

| Quy tắc / Nhóm | Tổng ảnh gốc | Kết quả kiểm chứng bằng mắt (40 mẫu/hết) | Tỷ lệ sai ước lượng | Hành động áp dụng ở GĐ4/GĐ5 | Sheet kiểm chứng |
|---|---:|---|---:|---|---|
| `F4_polo` | 324 | 40/40 đều là áo polo nữ chuẩn | **0.0%** (0/40) | **Giữ `high`** | `gd4_verify_F4_polo.png` |
| `M4_polo` | 2,048 | 40/40 đều là áo polo nam chuẩn | **0.0%** (0/40) | **Giữ `high`** | `gd4_verify_M4_polo.png` |
| `F5_polo` | 47 | 35/40 là áo cổ lọ / cổ cao / áo kiểu không tay | **87.5%** (35/40) | **Hạ `high → medium`** (`dense_only`) | `gd4_verify_F5_polo.png` |
| `M5_polo` | 33 | 17 cổ lọ/hoodie + 1 áo cổ tròn, chỉ 15 polo | **54.5%** (18/33) | **Hạ `high → medium`** (`dense_only`) | `gd4_verify_M5_polo.png` |
| `F5_classic_short_tee` | 1,545 | 6/40 là áo kiểu ren/bèo/cowl (+ áo hở vai trên `shirts_tops_40`) | **15.0%** (6/40) | **Hạ `high → medium`** (`dense_only`) | `gd4_verify_F5_classic_short_tee.png` |
| `Swimwear` (`long`/`belt`) | 73 | 40/40 đều là áo choàng tắm / kimono robe | **100.0%** (40/40) | **Tách sang `Bath Robe`** (164 tổng), `medium` + `pending_lexicon=True` | `gd4_verify_Swimwear_long_or_belt.png` |
| `J_sweater` (`fastening=Zip`) | 506 | 35/40 là áo khoác thể thao / áo khoác nỉ kéo khóa | **87.5%** (35/40) | **Tách `J_sweater_zip`, hạ `medium → low`** | `gd4_verify_J_sweater_zip.png` |
| `TracksuitBottoms_pending_lex` & nhóm thiếu từ điển | 10,370 | `Track Pants` (3.634) trộn jogger + legging yoga; `Swimwear` (5.119), `Jumpsuit` (1.240), `Tracksuits` (213), `Bath Robe` (164) chưa mở khóa `lexicon_vi.json` | — | **Gắn `pending_lexicon=True`**, giữ **`attr_confidence <= medium`**, `dense_only` | `gd4_sample_200_part*.png` |

- **Hiển thị & BM25:** `bm25_eligible=True` chỉ bật cho **`33,203`** ảnh (`33,101 shopping100k + 102 fashion`, đúng bằng tập `serving_eligible & attr_confidence == 'high'`). Cột `title_vi_display` chỉ chèn màu khi `color_conf == 'high'`; bỏ màu ở toàn bộ **43,198** ảnh phục vụ còn lại (`color_conf != 'high'`).

---

## 3. Phân tích kiểm chứng màu pixel (Mục 4a, 4b, 4c)

- **Bảo toàn nhãn gốc:** toàn bộ **17,885** ảnh thiếu màu (`color == -1`) vẫn giữ `baseColour = 'Unknown'` (`color_source = 'image_estimated'`), `color_conf = 'low'`, tuyệt đối KHÔNG tự điền vào `baseColour`.
- **(4c) Rủi ro cùng sai ở màu `White` trong tập `high`:** **`3,987 / 33,101`** (12.04%) ảnh `shopping100k high` mang nhãn `baseColour = 'White'` (+ 14 ảnh `fashion` = **4,001**). Vì ảnh sáng/màu pastel nhạt dễ được `color_pixel` ước lượng thành `White`, nếu nhãn gốc Shopping100k cũng ghi nhầm `White` thì cặp `(label=White, pixel=White)` sẽ cùng lọt `color_conf=high` (cần giới hạn cap và lấy mẫu kiểm lại ở Pha 7b).

### (4a) Phân bố `baseColour` (`shopping100k high` vs `shopping100k 98.174` vs `index_v2 38.644`)

| `baseColour` | `s100k high` (33.101) | % `high` | `s100k all` (98.174) | % `all` | `index_v2` (38.644) | % `v2` |
|---|---:|---:|---:|---:|---:|---:|
| `Black` | 10,777 | 32.56% | 21,315 | 21.71% | 8,710 | 22.54% |
| `Unknown` | 0 | 0.00% | 17,885 | 18.22% | 5 | 0.01% |
| `Blue` | 8,043 | 24.30% | 13,082 | 13.33% | 4,265 | 11.04% |
| `Navy Blue` | 5,157 | 15.58% | 12,521 | 12.75% | 1,674 | 4.33% |
| `White` | 3,987 | 12.04% | 9,690 | 9.87% | 4,769 | 12.34% |
| `Grey` | 1,122 | 3.39% | 7,305 | 7.44% | 2,566 | 6.64% |
| `Red` | 1,302 | 3.93% | 2,981 | 3.04% | 2,102 | 5.44% |
| `Beige` | 462 | 1.40% | 2,966 | 3.02% | 661 | 1.71% |
| `Green` | 506 | 1.53% | 2,343 | 2.39% | 1,858 | 4.81% |
| `Charcoal` | 652 | 1.97% | 2,030 | 2.07% | 207 | 0.54% |
| `Olive` | 298 | 0.90% | 1,552 | 1.58% | 388 | 1.00% |
| `Orange` | 114 | 0.34% | 940 | 0.96% | 441 | 1.14% |
| `Brown` | 42 | 0.13% | 857 | 0.87% | 2,921 | 7.56% |
| `Multi` | 45 | 0.14% | 830 | 0.85% | 380 | 0.98% |
| `Maroon` | 298 | 0.90% | 630 | 0.64% | 494 | 1.28% |
| `Yellow` | 207 | 0.63% | 486 | 0.50% | 671 | 1.74% |
| `Pink` | 49 | 0.15% | 396 | 0.40% | 1,443 | 3.73% |
| `Purple` | 40 | 0.12% | 365 | 0.37% | 1,418 | 3.67% |

### (4b) Độ đúng của `color_pixel` theo họ màu (toàn tập 80.289 có nhãn + chấm mắt 100 mẫu `seed=42`)

- **Trên 100 ảnh mẫu phân tầng theo họ màu (`gd5_color_visual_eval_100.csv`):**
  - Khớp tự động `color_label == color_pixel` (cùng họ): **56/100 (56.0%)**
  - Thêm **12/100** ca lệch mà mắt kiểm tra thấy `color_pixel` **ĐÚNG** do nhãn gốc Shopping100k sai (`#1, #8, #18, #30, #33, #52, #65, #80, #85, #92, #93, #100`) → **Tỷ lệ ước lượng `color_pixel` đúng thực tế = 68/100 (68.0%)**.
  - **32/100 lỗi của `color_pixel`:** màu nhạt/pastel ra `White` (**11/32**: #16, #17, #38, #41, #58, #59, #70, #73, #88, #94, #96); đồ đa sắc/họa tiết/xanh lá tối (**7/32**: #3, #25, #63, #66, #68, #76, #81); cam/nâu/be/xám ranh giới (**6/32**: #13, #21, #49, #74, #86, #97); đồ trắng dính bóng/chữ ra `Grey/Charcoal/Navy` (**5/32**: #15, #44, #60, #90, #99); hồng đậm/tím mauve ra `Red` (**2/32**: #45, #71); navy họa tiết đậm ra `Purple` (**1/32**: #64).

| Họ màu (`eval/families.json`) | Tổng ảnh có nhãn | Khớp nhãn-pixel toàn tập | Mẫu 100 (khớp tự động) | Mẫu 100 (`color_pixel` đúng thật) | Đánh giá độ tin cậy `color_pixel` |
|---|---:|---:|---:|---:|---|
| `Brown` | 857 | 52 (6.1%) | 0/4 | 2/4 (50%) | **KÉM** |
| `Multi` | 830 | 94 (11.3%) | 0/4 | 0/4 (0%) | **KÉM** |
| `Orange` | 940 | 190 (20.2%) | 0/4 | 2/4 (50%) | **KÉM** |
| `Pink` | 396 | 82 (20.7%) | 1/3 | 2/3 (67%) | **KÉM** |
| `Purple` | 365 | 76 (20.8%) | 1/3 | 2/3 (67%) | **KÉM** |
| `Beige` | 2,966 | 764 (25.8%) | 1/5 | 3/5 (60%) | **KÉM** |
| `Grey` | 7,305 | 2,067 (28.3%) | 2/9 | 3/9 (33%) | **KÉM** |
| `Green` | 3,895 | 1,266 (32.5%) | 2/6 | 2/6 (33%) | **KÉM** |
| `White` | 9,690 | 7,258 (74.9%) | 5/10 | 5/10 (50%) | Tốt |
| `Red` | 3,611 | 2,756 (76.3%) | 2/6 | 5/6 (83%) | Tốt |
| `Blue` | 25,603 | 19,735 (77.1%) | 19/22 | 19/22 (86%) | Tốt |
| `Yellow` | 486 | 376 (77.4%) | 2/3 | 2/3 (67%) | Tốt |
| `Black` | 23,345 | 20,260 (86.8%) | 21/21 | 21/21 (100%) | Tốt |

- **Các họ màu ước lượng kém cần lưu ý:** `Brown` (khớp 6,1% — áo nâu nhạt/be dễ ra White/Orange hoặc đen tối), `Multi` (11,3% — hay bị chọn màu nền chủ đạo), `Orange` (20,2% — dễ lệch `Yellow`/`Red`/`Beige`), `Pink` (20,7% — hồng nhạt ra `White`, hồng đậm ra `Red`), `Purple` (20,8% — ranh giới `Navy`/`Black`/`Red`), `Beige` (25,8% — nhạt ra `White`), `Grey` (28,3% — xám sáng ra `White`, xám đậm ra `Charcoal`), `Green` (32,5% — rêu/xanh đậm lệch `Grey`/`Black`/`Brown`).

---

## 4. Hạn chế còn lại & Đề xuất cho Pha 7b

1. **Lệch phân bố nặng (Bias nguồn & màu):** Riêng Shopping100k `high` (`33,101` ảnh) tập trung **84.48%** (`27,964 / 33,101`) vào `Black` (`10,777`), `Blue + Navy Blue` (`13,200` = `8,043 Blue` + `5,157 Navy Blue`), `White` (`3,987`), và dồn rất mạnh vào vài slot (`(Dresses, Women) +6,522`, `(Tshirts, Men) +5,027`). Nếu nhúng toàn bộ 33k ảnh vào index, không gian vector và prior truy vấn text→ảnh sẽ lệch mạnh sang ảnh studio châu Âu (ghost-mannequin) của Shopping100k.
2. **Độ tin cậy màu & nhãn gốc:** Shopping100k có ~12–15% nhãn màu gốc sai (ví dụ `Navy Blue` trên áo hồng đậm, `Beige` trên áo trắng) và `3,987` ảnh `high + White` mang rủi ro *double-agreement bias* (`label=White` + `pixel=White`).
3. **Đề xuất chọn lọc cho Pha 7b:** **KHÔNG đưa toàn bộ 33.203 ảnh `high` vào index.** Áp dụng **cap theo từng slot `(articleType × gender)`** (ví dụ: ưu tiên lấp các slot `index_v2 < 300` đến ngưỡng mục tiêu `300–400 sp/slot`; các slot đã dày như `(Tshirts, Men) = 5,089` hoặc `(Shirts, Men) = 2,597` chỉ lấy mẫu nhỏ `<= 200–300` đa dạng màu hoặc bằng `0` cho `AttributeIndex`, đồng thời rà nhanh slot `White`). Với 5 nhóm `pending_lexicon=True`, hoàn thiện `lexicon_vi.json` + test parser trước khi mở khóa vào `AttributeIndex`.

---

## 5. Thống kê phục vụ Pha 7b (Mục 7a, 7b, 7c)

### (7c) Tập đánh giá mở rộng `eval_ext_eligible` (`seen_in_train == 'no'`) & đối chiếu ID với `dense_captions_catalog.json`

- `fashion` (`eval_ext_eligible=True`): **94** ảnh — khớp khóa trong `dense_captions_catalog.json`: **94** (`0` thiếu).
- `shopping100k` (`eval_ext_eligible=True`): **1,468** ảnh — khớp khóa trong `dense_captions_catalog.json`: **1,468** (`0` thiếu; gồm 448 khóa đầy đủ `shop100k_<gender>_<cat>_<stem>` và 1,020 khóa ngắn `shop100k_<stem>`, trong đó 17 ảnh có cả 2 khóa đã ưu tiên khóa đầy đủ; 3/1,471 ảnh `shopping100k legacy_test` bị loại ở GĐ3 do `duplicate_internal_sha1`). Bảng đối chiếu chi tiết kèm cột `dense_caption_key` lưu tại [`eval/phase7/eval_ext_dense_captions.csv`](../../eval/phase7/eval_ext_dense_captions.csv).

### (7b) Các slot mỏng trong `index_v2` (`< 300` sản phẩm) có ứng viên Pha 7a bổ sung (gồm cả `pending_lexicon`)

| `(articleType, gender)` | `index_v2` | `+ high` | `+ medium` (`color=high`) | `+ low` | `pending_lexicon` | Ghi chú Pha 7b |
|---|---:|---:|---:|---:|---|---|
| `(Tracksuits, Women)` | 1 | +0 | +49 (49 màu ok) | +16 | True | Chờ lexicon_vi ở Pha 7b |
| `(Suspenders, Men)` | 2 | +0 | +3 (3 màu ok) | +0 | False | Chỉ có `medium`/`low` (`dense_only`) |
| `(Swimwear, Women)` | 12 | +0 | +1,828 (1,828 màu ok) | +1,961 | True | Chờ lexicon_vi ở Pha 7b |
| `(Jumpsuit, Women)` | 16 | +0 | +787 (787 màu ok) | +446 | True | Chờ lexicon_vi ở Pha 7b |
| `(Capris, Girls)` | 18 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Bath Robe, Women)` | 20 | +0 | +65 (65 màu ok) | +65 | True | Chờ lexicon_vi ở Pha 7b |
| `(Tracksuits, Men)` | 28 | +0 | +109 (109 màu ok) | +39 | True | Chờ lexicon_vi ở Pha 7b |
| `(Sandals, Girls)` | 29 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Sports Shoes, Unisex)` | 30 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Bracelet, Women)` | 46 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Jewellery Set, Women)` | 52 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Jackets, Women)` | 55 | +2,196 | +0 (0 màu ok) | +1,762 | False | Có ảnh `high` lấp trực tiếp |
| `(Shorts, Women)` | 56 | +1,276 | +0 (0 màu ok) | +912 | False | Có ảnh `high` lấp trực tiếp |
| `(Sweatshirts, Women)` | 58 | +438 | +230 (230 màu ok) | +746 | False | Có ảnh `high` lấp trực tiếp |
| `(Night suits, Men)` | 62 | +230 | +0 (0 màu ok) | +168 | False | Có ảnh `high` lấp trực tiếp |
| `(Tshirts, Unisex)` | 64 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Trousers, Women)` | 70 | +1,523 | +0 (0 màu ok) | +1,129 | False | Có ảnh `high` lấp trực tiếp |
| `(Sandals, Unisex)` | 70 | +3 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Night suits, Women)` | 77 | +292 | +0 (0 màu ok) | +373 | False | Có ảnh `high` lấp trực tiếp |
| `(Track Pants, Women)` | 79 | +0 | +1,615 (1,615 màu ok) | +899 | True | Chờ lexicon_vi ở Pha 7b |
| `(Sweaters, Women)` | 86 | +0 | +2,026 (2,026 màu ok) | +2,598 | False | Chỉ có `medium`/`low` (`dense_only`) |
| `(Ring, Women)` | 87 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Skirts, Women)` | 90 | +1,403 | +0 (0 màu ok) | +1,060 | False | Có ảnh `high` lấp trực tiếp |
| `(Accessory Gift Set, Men)` | 92 | +14 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Necklace and Chains, Women)` | 95 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Sunglasses, Unisex)` | 97 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Leggings, Women)` | 98 | +3 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Flip Flops, Unisex)` | 126 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Tops, Girls)` | 132 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Jeans, Women)` | 163 | +2,173 | +0 (0 màu ok) | +1,421 | False | Có ảnh `high` lấp trực tiếp |
| `(Sweaters, Men)` | 189 | +0 | +1,181 (1,181 màu ok) | +1,207 | False | Chỉ có `medium`/`low` (`dense_only`) |
| `(Jackets, Men)` | 191 | +2,114 | +0 (0 màu ok) | +1,272 | False | Có ảnh `high` lấp trực tiếp |
| `(Track Pants, Men)` | 198 | +3 | +774 (774 màu ok) | +346 | True | Chờ lexicon_vi ở Pha 7b |
| `(Casual Shoes, Unisex)` | 207 | +3 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Ties, Men)` | 216 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Flip Flops, Women)` | 217 | +2 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Sweatshirts, Men)` | 218 | +1,035 | +796 (796 màu ok) | +1,367 | False | Có ảnh `high` lấp trực tiếp |
| `(Innerwear Vests, Men)` | 227 | +1 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Sunglasses, Women)` | 232 | +3 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |
| `(Jeans, Men)` | 261 | +1,285 | +0 (0 màu ok) | +1,211 | False | Có ảnh `high` lấp trực tiếp |
| `(Shirts, Women)` | 282 | +818 | +70 (70 màu ok) | +570 | False | Có ảnh `high` lấp trực tiếp |
| `(Trousers, Men)` | 288 | +970 | +0 (0 màu ok) | +904 | False | Có ảnh `high` lấp trực tiếp |
| `(Casual Shoes, Women)` | 288 | +3 | +0 (0 màu ok) | +0 | False | Có ảnh `high` lấp trực tiếp |

### (7a) Bảng đầy đủ theo từng slot `(articleType × gender)` đối chiếu `index_v2` (tất cả `serving_eligible` có ứng viên)

| `(articleType, gender)` | `index_v2` | `high` | `medium` | `low` | Tổng ứng viên |
|---|---:|---:|---:|---:|---:|
| `(Dresses, Women)` | 372 | 6,522 | 0 | 5,187 | **11,709** |
| `(Tshirts, Men)` | 5,089 | 5,034 | 1,685 | 5,100 | **11,819** |
| `(Shirts, Men)` | 2,597 | 2,306 | 62 | 1,324 | **3,692** |
| `(Jackets, Women)` | 55 | 2,196 | 0 | 1,762 | **3,958** |
| `(Jeans, Women)` | 163 | 2,173 | 0 | 1,421 | **3,594** |
| `(Jackets, Men)` | 191 | 2,114 | 0 | 1,272 | **3,386** |
| `(Shorts, Men)` | 340 | 1,725 | 0 | 1,119 | **2,844** |
| `(Tops, Women)` | 1,492 | 1,587 | 1,607 | 2,571 | **5,765** |
| `(Trousers, Women)` | 70 | 1,523 | 0 | 1,129 | **2,652** |
| `(Skirts, Women)` | 90 | 1,403 | 0 | 1,060 | **2,463** |
| `(Jeans, Men)` | 261 | 1,285 | 0 | 1,211 | **2,496** |
| `(Shorts, Women)` | 56 | 1,276 | 0 | 912 | **2,188** |
| `(Sweatshirts, Men)` | 218 | 1,035 | 796 | 1,367 | **3,198** |
| `(Trousers, Men)` | 288 | 970 | 0 | 904 | **1,874** |
| `(Shirts, Women)` | 282 | 818 | 70 | 570 | **1,458** |
| `(Sweatshirts, Women)` | 58 | 438 | 230 | 746 | **1,414** |
| `(Night suits, Women)` | 77 | 292 | 0 | 373 | **665** |
| `(Night suits, Men)` | 62 | 230 | 0 | 168 | **398** |
| `(Tshirts, Women)` | 1,083 | 197 | 1,505 | 1,557 | **3,259** |
| `(Accessory Gift Set, Men)` | 92 | 14 | 0 | 0 | **14** |
| `(Sports Shoes, Men)` | 1,430 | 7 | 0 | 0 | **7** |
| `(Heels, Women)` | 1,268 | 4 | 0 | 0 | **4** |
| `(Track Pants, Men)` | 198 | 3 | 774 | 346 | **1,123** |
| `(Casual Shoes, Men)` | 2,017 | 3 | 0 | 0 | **3** |
| `(Casual Shoes, Unisex)` | 207 | 3 | 0 | 0 | **3** |
| `(Casual Shoes, Women)` | 288 | 3 | 0 | 0 | **3** |
| `(Earrings, Women)` | 376 | 3 | 0 | 0 | **3** |
| `(Leggings, Women)` | 98 | 3 | 0 | 0 | **3** |
| `(Sandals, Unisex)` | 70 | 3 | 0 | 0 | **3** |
| `(Sunglasses, Women)` | 232 | 3 | 0 | 0 | **3** |
| `(Flip Flops, Unisex)` | 126 | 2 | 0 | 0 | **2** |
| `(Flip Flops, Women)` | 217 | 2 | 0 | 0 | **2** |
| `(Necklace and Chains, Women)` | 95 | 2 | 0 | 0 | **2** |
| `(Sandals, Girls)` | 29 | 2 | 0 | 0 | **2** |
| `(Sunglasses, Men)` | 478 | 2 | 0 | 0 | **2** |
| `(Sunglasses, Unisex)` | 97 | 2 | 0 | 0 | **2** |
| `(Ties, Men)` | 216 | 2 | 0 | 0 | **2** |
| `(Tops, Girls)` | 132 | 2 | 0 | 0 | **2** |
| `(Belts, Men)` | 524 | 1 | 0 | 0 | **1** |
| `(Bracelet, Women)` | 46 | 1 | 0 | 0 | **1** |
| `(Capris, Girls)` | 18 | 1 | 0 | 0 | **1** |
| `(Flip Flops, Men)` | 439 | 1 | 0 | 0 | **1** |
| `(Innerwear Vests, Men)` | 227 | 1 | 0 | 0 | **1** |
| `(Jewellery Set, Women)` | 52 | 1 | 0 | 0 | **1** |
| `(Kurtas, Women)` | 1,754 | 1 | 0 | 0 | **1** |
| `(Ring, Women)` | 87 | 1 | 0 | 0 | **1** |
| `(Sandals, Men)` | 697 | 1 | 0 | 0 | **1** |
| `(Sports Shoes, Unisex)` | 30 | 1 | 0 | 0 | **1** |
| `(Sports Shoes, Women)` | 333 | 1 | 0 | 0 | **1** |
| `(Tshirts, Boys)` | 363 | 1 | 0 | 0 | **1** |
| `(Tshirts, Unisex)` | 64 | 1 | 0 | 0 | **1** |
| `(Wallets, Men)` | 410 | 1 | 0 | 0 | **1** |
| `(Tops/Shirts-ambiguous, Women)` | 0 | 0 | 0 | 9,931 | **9,931** |
| `(Sweaters, Women)` | 86 | 0 | 2,026 | 2,598 | **4,624** |
| `(Swimwear, Women)` | 12 | 0 | 1,828 | 1,961 | **3,789** |
| `(Track Pants, Women)` | 79 | 0 | 1,615 | 899 | **2,514** |
| `(Sweaters, Men)` | 189 | 0 | 1,181 | 1,207 | **2,388** |
| `(Swimwear, Men)` | 0 | 0 | 757 | 573 | **1,330** |
| `(Jumpsuit, Women)` | 16 | 0 | 787 | 446 | **1,233** |
| `(Coats, Women)` | 0 | 0 | 0 | 1,084 | **1,084** |
| `(Coats, Men)` | 0 | 0 | 0 | 445 | **445** |
| `(Suits, Men)` | 0 | 0 | 222 | 105 | **327** |
| `(Tops/Shirts-ambiguous, Men)` | 0 | 0 | 0 | 196 | **196** |
| `(Tracksuits, Men)` | 28 | 0 | 109 | 39 | **148** |
| `(Bath Robe, Women)` | 20 | 0 | 65 | 65 | **130** |
| `(Tracksuits, Women)` | 1 | 0 | 49 | 16 | **65** |
| `(Bath Robe, Men)` | 0 | 0 | 17 | 17 | **34** |
| `(Jumpsuit, Men)` | 0 | 0 | 4 | 3 | **7** |
| `(Suspenders, Men)` | 2 | 0 | 3 | 0 | **3** |
