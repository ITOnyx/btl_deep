# Báo cáo Pha 1 — Làm sạch & Lập Catalog

**Thời gian thực hiện:** 2026-11-08 20:20:35

## 1. Ngưỡng lọc chất lượng cuối cùng
- `min_short_side`: 224 px (loại bỏ ảnh độ phân giải thấp)
- `blur_laplacian_var_threshold`: 12.0 (loại bỏ ảnh mờ)
- `phash_hamming_threshold`: 4 (phát hiện cụm trùng)
- `detect_banner`: True (loại bỏ ảnh có viền màu bẹt hoặc chữ khuyến mãi dày đặc)

## 2. Thống kê số lượng qua từng bước lọc
- Tổng dòng ban đầu trong `styles.csv`: 44.446 dòng
- Bỏ qua do lỗi định dạng (bad lines): 22 dòng
- Dòng hợp lệ đọc được: 44.424 dòng
- Số dòng có file ảnh tồn tại trên đĩa: 44.419 dòng (5 ID thiếu file ảnh)
- Số file hỏng/0-byte (`corrupt`): 0 ảnh
- Loại do độ phân giải thấp (`low_res`, cạnh ngắn < 224): 119 ảnh
- Loại do mờ nhòe (`blurry`, Laplacian var < 12.0): 1286 ảnh
- Loại do banner/khung viền (`banner`): 935 ảnh
- Loại do trùng lặp (`duplicate` - chỉ giữ 1 ảnh nét nhất/cụm): 14414 ảnh
- **Tổng số ảnh chuyển vào quarantine:** 16754 ảnh (37.72%)
- **Số sản phẩm sạch được giữ lại trong `catalog.parquet`:** **27665 sản phẩm** (62.28%)

## 3. Tỉ lệ nhóm rủi ro nhiễu (Accessories / Cận người)
- Tổng số sản phẩm thuộc nhóm Accessories / Headwear / Eyewear / Watches / Bags: **7766 sản phẩm** (28.07% danh mục)
- Nhóm này được giữ lại đầy đủ theo đúng thiết kế của tập Fashion, tuy nhiên người mẫu đeo phụ kiện cận mặt có thể tạo nhiễu khi tìm kiếm (đã xuất contact sheet `risk_categories.png`).

## 4. Bổ sung kho phụ (`old_images`, `crawled_images`)
- Kiểm tra: Thư mục `data/images` và `data/images_crawled` không tồn tại trong workspace.
- Toàn bộ sản phẩm được trích xuất an toàn từ tập gốc Kaggle (`source = 'kaggle'`).

## 5. Thống kê nhóm danh mục có ít hơn 30 ảnh (`articleType x gender < 30`)
- Tổng cộng: 165 nhóm có < 30 ảnh.

| articleType | gender | Số lượng |
|---|---|---|
| Body Wash and Scrub | Men | 1 |
| Blazers | Boys | 1 |
| Cushion Covers | Unisex | 1 |
| Caps | Boys | 1 |
| Gloves | Unisex | 1 |
| Innerwear Vests | Women | 1 |
| Fragrance Gift Set | Unisex | 1 |
| Face Serum and Gel | Women | 1 |
| Jackets | Unisex | 1 |
| Sandals | Women | 1 |
| Shoe Laces | Unisex | 1 |
| Shirts | Girls | 1 |
| Socks | Boys | 1 |
| Lip Plumper | Women | 1 |
| Mufflers | Unisex | 1 |
| Mens Grooming Kit | Men | 1 |
| Rain Trousers | Men | 1 |
| Sports Sandals | Unisex | 1 |
| Tracksuits | Women | 1 |
| Toner | Women | 1 |
| Sweatshirts | Unisex | 1 |
| Tablet Sleeve | Women | 1 |
| Waist Pouch | Men | 1 |
| Trousers | Girls | 1 |
| Tops | Men | 1 |
| Track Pants | Unisex | 1 |
| Sunscreen | Unisex | 1 |
| Sunscreen | Men | 1 |
| Jeans | Unisex | 2 |
| Lehenga Choli | Women | 2 |
| ... (135 nhóm khác) | ... | ... |

## 6. Kiểm tra nghiệm thu
- [x] Không có file nào bị xóa vĩnh viễn: mọi ảnh bị lọc đều chuyển vào `data/_quarantine/<lý_do>/` và có bản ghi chi tiết tại `reports/quarantine_log.csv`.
- [x] Toàn bộ 100% dòng trong `catalog.parquet` đều có đầy đủ `gender`, `articleType` và đường dẫn ảnh tồn tại.
- [x] Không còn ảnh dưới ngưỡng 224px, không còn ảnh trùng lặp trong catalog.
- [x] `notebooks/` và `docs/` được bảo toàn nguyên vẹn, không bị chỉnh sửa.