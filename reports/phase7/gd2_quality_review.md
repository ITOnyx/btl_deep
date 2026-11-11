# Pha 7a — GĐ2: Kiểm tra Chất lượng Ảnh & Contact Sheets (Chờ duyệt ngưỡng)

- **Thời gian chạy:** 1428.0s | **Seed:** `42`
- **Ngưỡng `config.yaml` đang thử nghiệm (CHƯA áp dụng chốt):** `min_short_side=224`, `blur_laplacian_var_threshold=12.0`, `detect_banner=True`, `phash_hamming_threshold=1`, `dedup_match_attributes=['articleType', 'gender', 'baseColour']`, `dedup_transitive=False`.
- **File metrics:** `eval/phase7/quality_metrics.parquet` (139,139 dòng)

---

## 1. Bảng giữ/loại theo ngưỡng `config.yaml` hiện hành (mô phỏng tầng nối tiếp)

Thứ tự tầng lọc: `low_res (short_side < 224)` → `blurry (blur < 12.0)` → `banner (thuật toán Pha 1)` → `duplicate (cùng article_type+gender+color & pHash <= 1)`.

| Tập kiểm tra | Tổng ảnh | Loại `low_res` | Loại `blurry` | Loại `banner` | Loại `duplicate` | Tổng loại | Tỷ lệ loại | Giữ lại | Tỷ lệ giữ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fashion (tất cả) | 40,953 | 6 | 47 | 39 | 10 | **102** | **0.25%** | **40,851** | **99.75%** |
| fashion (chưa có trong catalog) | 122 | 6 | 47 | 39 | 2 | **94** | **77.05%** | **28** | **22.95%** |
| shopping100k (tất cả) | 98,186 | 9 | 770 | 32,549 | 3 | **33,331** | **33.95%** | **64,855** | **66.05%** |
| shopping100k (new & in_scope) | 98,186 | 9 | 770 | 32,549 | 3 | **33,331** | **33.95%** | **64,855** | **66.05%** |
| Tổng merged readable | 139,139 | 15 | 817 | 32,588 | 13 | **33,433** | **24.03%** | **105,706** | **75.97%** |

- Cặp trùng pHash <= 1 cùng thuộc tính trong pool sạch: tổng **13** cặp (trong đó xuyên nguồn `fashion` ↔ `shopping100k`: **0** cặp).

---

## 2. Phân bố chỉ số gốc & phân rã nguyên nhân theo nguồn (không chồng tầng)

| Nguồn | `short_side` min / p5 / p25 / p50 / p95 | `<180` | `180–223` | `224–255` | `>=256` | `blur` p1 / p5 / p10 / p50 |
|---|---|---:|---:|---:|---:|---|
| `fashion` (40,953) | 150 / 1080 / 1080 / 1080 / 1800 | 6 | 0 | 0 | 40,947 | 13.68 / 20.94 / 28.71 / 114.46 |
| `shopping100k` (98,186) | 148 / 762 / 762 / 762 / 762 | 9 | 0 | 0 | 98,177 | 13.71 / 42.50 / 85.09 / 357.82 |

### Phân rã 4 điều kiện cờ Banner (Pha 1)

| Nguồn | `is_top_solid` (`mean<230 & std<10`) | `is_bot_solid` (`mean<230 & std<10`) | `is_top_text` (`top_edge>0.025`) | `is_side_frame` (`side_edge>0.030`) | Tổng `banner_flag=True` |
|---|---:|---:|---:|---:|---:|
| `fashion` | 22 | 22 | 19 | 1 | **43** (0.10%) |
| `shopping100k` | 1 | 3 | 32,166 | 1,048 | **32,557** (33.16%) |

---

## 3. Phân tích Contact Sheets & Đề xuất ngưỡng theo nguồn (chưa tự chốt)

### Quan sát trực quan từ Contact Sheets
1. **`shopping100k_banner_rejected.png` (32.557 ảnh / 33,16%):** ~100% mẫu kiểm tra trực quan là **ảnh sản phẩm studio/ghost-mannequin hoàn toàn sạch, KHÔNG có banner hay chữ quảng cáo**. Nguyên nhân là ảnh Shopping100k được crop gọn sát trang phục (`762×...`), khiến cổ áo, cạp quần, vai ren chạm vùng mép trên 8% (`is_top_text`: 32.166 ảnh) và hai bên 5% (`is_side_frame`: 1.048 ảnh).
2. **`shopping100k_blurry_rejected.png` (770 ảnh / 0,78%):** Thống kê theo màu cho thấy `White` chiếm **539/770 (70,00%)**, `NaN` **117 (15,19%)**, `Beige` **54 (7,01%)**, `Yellow` **20** (`blur` thấp nhất toàn tập = `0.6661`, `< 0.5` = `0`, `< 1.0` = `4`). Trực quan xác nhận đây đều là **áo/váy/quần màu trắng hoặc sáng nét trên nền trắng tinh**, bị tụt Laplacian variance toàn cục chứ không mờ do out-of-focus.
3. **`shopping100k_low_res_rejected.png` (9 ảnh / 0,009%):** Tất cả 9 ảnh là `148×215`, trong khi **98.177 / 98.186 (99,991%)** đều có `short_side >= 762 px`.
4. **`shopping100k_duplicates_pairs.png` (3 cặp):** Do `17.885` ảnh Shopping100k thiếu màu (`color = NaN -> Unknown`), 2/3 cặp trùng thuộc tính + `pHash=0` lại là **hai biến thể khác màu** (*Jacket xanh rêu vs đen*, *Swimwear xanh ngọc vs đỏ*).

### Bảng các phương án cho `shopping100k` (trước dedup)

| Phương án | `min_short_side` | Ngưỡng `blur` | Luật banner | Ảnh `shopping100k` giữ | Bị loại | Tỷ lệ giữ | Nhận xét |
|---|---:|---:|---|---:|---:|---:|---|
| **Đề xuất A (Khuyến nghị)** | **224** | **`0.5`** *(hoặc tắt)* | **`off` (`detect_banner=False`)** | **98,177** | **9** | **99.99%** | Chỉ bỏ 9 ảnh `148×215`; bảo toàn 32.557 ảnh crop sát mép và 770 ảnh trắng sắc nét. |
| **Đề xuất B (Thận trọng vừa)** | 224 | 12.0 | `off` (`detect_banner=False`) | 97,407 | 779 | 99.21% | Không bỏ ảnh sát mép, nhưng vẫn mất 770 sản phẩm trắng/be trên nền trắng. |
| **Phương án C (Nguyên `config.yaml`)** | 224 | 12.0 | `pha1_all` (`detect_banner=True`) | 64,858 | 33,328 | 66.06% | Loại oan ~33,3% ảnh sạch do heuristic Pha 1 không phù hợp bố cục ghost-mannequin. |
| Tham khảo D (`blur>=12`, giữ `text_only`) | 180 | 12.0 | `text_only` (bỏ `Ts/Bs`) | 64,860 | 33,326 | 66.06% | Không giải quyết được `is_top_text` (32.166 ảnh). |
| Tham khảo E (`side>=150, blur>=10, off`) | 150 | 10.0 | `off` | 97,601 | 585 | 99.40% | Mất 585 ảnh trắng `< 10` và nới resolution xuống 150. |

*Với nguồn `fashion` (40.953 ảnh):* **40.831** đã trùng `catalog.parquet`; **122** ảnh ngoài catalog gồm `100` `legacy_test` (`seen_in_train='no'`) và `22` `unknown` (trong đó 94 ảnh từng nằm trong `_quarantine` của Pha 1 theo đúng ngưỡng `config.yaml`).

---

## 4. Danh sách Contact Sheets đã xuất (`reports/phase7/contact_sheets/`)

- `fashion_banner_borderline.png`
- `fashion_banner_rejected.png`
- `fashion_blurry_borderline.png`
- `fashion_blurry_rejected.png`
- `fashion_duplicates_pairs.png`
- `fashion_low_res_rejected.png`
- `fashion_pass_clear.png`
- `shopping100k_banner_borderline.png`
- `shopping100k_banner_rejected.png`
- `shopping100k_blurry_borderline.png`
- `shopping100k_blurry_rejected.png`
- `shopping100k_duplicates_pairs.png`
- `shopping100k_low_res_rejected.png`
- `shopping100k_pass_clear.png`

---

## 5. Điểm dừng [DỪNG] chờ duyệt ngưỡng

Theo ngưỡng `config.yaml` hiện hành, `shopping100k` bị loại **33.95%** (giữ **64,855 / 98,186**), chủ yếu do false positive từ `is_top_text` (32,166 ảnh) và `blur < 12` trên đồ trắng (770 ảnh). Xin bạn xác nhận 3 điểm trước khi tôi chạy GĐ3:
1. **Ngưỡng `shopping100k`:** Chọn **Đề xuất A** (`min_short_side=224`, `detect_banner=False`, `blur>=0.5` — giữ ~98.177), **Đề xuất B** (`detect_banner=False`, `blur>=12.0` — giữ 97.407), hay **Phương án C** (nguyên `config.yaml` — giữ 64.855)?
2. **Ngưỡng `fashion`:** Giữ nguyên ngưỡng `config.yaml` (`side>=224, blur>=12.0, detect_banner=True`) như Pha 1? (Lưu ý 40.831/40.953 ảnh `fashion` đã trùng `catalog` và sẽ bị lọc trùng với catalog ở GĐ3.3).
3. **Dedup khi thiếu màu (`color=Unknown`, 17.885 ảnh Shopping100k):** Đồng ý chỉ dedup `pHash <= 1` khi CẢ 3 thuộc tính `(articleType, gender, baseColour)` đều xác định, còn khi thiếu màu chỉ dedup exact `SHA-1` để không xóa nhầm biến thể khác màu (như cặp bikini xanh vs đỏ trong `shopping100k_duplicates_pairs.png`)?
