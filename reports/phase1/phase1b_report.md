# Báo cáo Pha 1b — Sửa lỗi Deduplication & Khôi phục Ảnh

**Thời gian thực hiện:** 2026-11-08 20:29:14

## 1. Tóm tắt sự cố Deduplication Pha 1
- **Nguyên nhân cốt lõi:** Thuật toán gom cụm bắc cầu (connected components) với ngưỡng pHash $\le 4$ đã liên kết dây chuyền các sản phẩm qua trung gian, tạo ra siêu cụm lên tới 8.425 ảnh. pHash trên ảnh xám bỏ qua màu sắc, dẫn đến việc các biến thể màu sắc khác nhau của cùng một kiểu dáng bị gom chung và chỉ giữ lại đúng 1 ảnh.
- **Tỉ lệ cụm trùng cũ chứa ảnh khác thuộc tính:** **64,5% số cụm** (1.393 / 2.159 cụm) chứa ảnh khác màu, khác loại hoặc khác giới tính.
- **Tỉ lệ ảnh bị loại oan:** Trong 14.414 ảnh bị chuyển vào quarantine vì trùng lặp, có tới **12.691 ảnh (88,0%)** khác màu, loại hoặc giới tính so với ảnh đại diện.

## 2. Quy tắc Deduplication An toàn Mới
- **Điều kiện trùng:** Hai ảnh chỉ bị coi là trùng khi **CÙNG `articleType` + `gender` + `baseColour`** VÀ **pHash Hamming $\le 1$**.
- **Không gom cụm bắc cầu:** So sánh trực tiếp từng ảnh ứng viên với ảnh đại diện (representative) của cụm.
- **Tiêu chí chọn đại diện:** Ưu tiên ảnh có độ phân giải cao nhất (`short_side`), hòa thì chọn ảnh nét nhất (`blur_score`), hòa thì chọn `id` nhỏ nhất.

## 3. Kết quả Khôi phục & Thống kê Catalog Mới
- **Số ảnh khôi phục:** **13,226 ảnh** (chuyển ngược từ `data/_quarantine/duplicate/` về `data/fashion-dataset/images/`).
- **Số ảnh còn bị loại vì trùng:** **1,188 ảnh** (giữ nguyên trong quarantine).
- **Tổng số sản phẩm trong Catalog mới:** **40,891 sản phẩm** (tăng từ 27.665 lên 40.891 sản phẩm, đạt **92,1%** toàn bộ dataset).

## 4. Phân bố Catalog mới theo MasterCategory

| masterCategory | Số lượng | Tỉ lệ (%) |
|---|---|---|
| Apparel | 19,818 | 48.47% |
| Accessories | 10,572 | 25.85% |
| Footwear | 8,251 | 20.18% |
| Personal Care *(Ngoài thời trang)* | 2,136 | 5.22% |
| Free Items *(Ngoài thời trang)* | 97 | 0.24% |
| Sporting Goods *(Ngoài thời trang)* | 16 | 0.04% |
| Home *(Ngoài thời trang)* | 1 | 0.00% |

## 5. Thống kê nhóm danh mục hiếm (`articleType × gender < 30`)
- **Trước khi sửa:** 165 nhóm có < 30 ảnh.
- **Sau khi sửa & khôi phục:** **158 nhóm** có < 30 ảnh (7 nhóm đã được phục hồi vượt ngưỡng >= 30 ảnh).

| masterCategory | articleType | gender | Số lượng | Ghi chú |
|---|---|---|---|---|
| Free Items | Backpacks | Women | 1 | ⚠️ Ngoài thời trang |
| Free Items | Backpacks | Unisex | 1 | ⚠️ Ngoài thời trang |
| Personal Care | Body Wash and Scrub | Men | 1 | ⚠️ Ngoài thời trang |
| Apparel | Blazers | Boys | 1 | Thời trang |
| Home | Cushion Covers | Unisex | 1 | ⚠️ Ngoài thời trang |
| Free Items | Clutches | Women | 1 | ⚠️ Ngoài thời trang |
| Accessories | Caps | Boys | 1 | Thời trang |
| Accessories | Caps | Girls | 1 | Thời trang |
| Free Items | Ipad | Women | 1 | ⚠️ Ngoài thời trang |
| Accessories | Gloves | Unisex | 1 | Thời trang |
| Apparel | Innerwear Vests | Women | 1 | Thời trang |
| Footwear | Heels | Men | 1 | Thời trang |
| Personal Care | Fragrance Gift Set | Unisex | 1 | ⚠️ Ngoài thời trang |
| Apparel | Jackets | Unisex | 1 | Thời trang |
| Accessories | Earrings | Girls | 1 | Thời trang |
| Free Items | Laptop Bag | Women | 1 | ⚠️ Ngoài thời trang |
| Accessories | Perfume and Body Mist | Men | 1 | Thời trang |
| Accessories | Shoe Laces | Unisex | 1 | Thời trang |
| Apparel | Shirts | Girls | 1 | Thời trang |
| Free Items | Scarves | Men | 1 | ⚠️ Ngoài thời trang |
| Accessories | Socks | Boys | 1 | Thời trang |
| Accessories | Mufflers | Unisex | 1 | Thời trang |
| Personal Care | Mens Grooming Kit | Men | 1 | ⚠️ Ngoài thời trang |
| Apparel | Trousers | Girls | 1 | Thời trang |
| Apparel | Tops | Men | 1 | Thời trang |
| Apparel | Track Pants | Unisex | 1 | Thời trang |
| Accessories | Tablet Sleeve | Women | 1 | Thời trang |
| Apparel | Tights | Men | 1 | Thời trang |
| Apparel | Sweatshirts | Unisex | 1 | Thời trang |
| Personal Care | Sunscreen | Unisex | 1 | ⚠️ Ngoài thời trang |
| Personal Care | Sunscreen | Men | 1 | ⚠️ Ngoài thời trang |
| Accessories | Sunglasses | Boys | 1 | Thời trang |
| Accessories | Waist Pouch | Men | 1 | Thời trang |
| Apparel | Tunics | Men | 1 | Thời trang |
| Free Items | Wallets | Men | 1 | ⚠️ Ngoài thời trang |
| Apparel | Tracksuits | Women | 1 | Thời trang |
| Accessories | Tshirts | Men | 1 | Thời trang |
| Apparel | Lehenga Choli | Girls | 2 | Thời trang |
| Apparel | Kurtis | Men | 2 | Thời trang |
| Footwear | Sports Shoes | Boys | 2 | Thời trang |
| Sporting Goods | Footballs | Unisex | 2 | ⚠️ Ngoài thời trang |
| Apparel | Blazers | Women | 2 | Thời trang |
| Sporting Goods | Wristbands | Unisex | 2 | ⚠️ Ngoài thời trang |
| Accessories | Wristbands | Unisex | 2 | Thời trang |
| Apparel | Suspenders | Men | 2 | Thời trang |
| Accessories | Water Bottle | Unisex | 2 | Thời trang |
| Free Items | Ties | Men | 2 | ⚠️ Ngoài thời trang |
| Accessories | Tablet Sleeve | Unisex | 2 | Thời trang |
| Accessories | Ties and Cufflinks | Women | 2 | Thời trang |
| Accessories | Sunglasses | Girls | 2 | Thời trang |
| Accessories | Key chain | Unisex | 2 | Thời trang |
| Apparel | Lehenga Choli | Women | 2 | Thời trang |
| Apparel | Rain Trousers | Men | 2 | Thời trang |
| Personal Care | Hair Colour | Unisex | 2 | ⚠️ Ngoài thời trang |
| Accessories | Shoe Accessories | Men | 2 | Thời trang |
| Accessories | Backpacks | Girls | 2 | Thời trang |
| Personal Care | Face Serum and Gel | Women | 2 | ⚠️ Ngoài thời trang |
| Free Items | Handbags | Women | 2 | ⚠️ Ngoài thời trang |
| Accessories | Hat | Unisex | 2 | Thời trang |
| Apparel | Clothing Set | Girls | 3 | Thời trang |
| Apparel | Belts | Men | 3 | Thời trang |
| Apparel | Leggings | Boys | 3 | Thời trang |
| Apparel | Churidar | Men | 3 | Thời trang |
| Personal Care | Deodorant | Unisex | 3 | ⚠️ Ngoài thời trang |
| Apparel | Swimwear | Unisex | 3 | Thời trang |
| Apparel | Waistcoat | Boys | 3 | Thời trang |
| Footwear | Sandals | Women | 3 | Thời trang |
| Apparel | Jeans | Unisex | 3 | Thời trang |
| Accessories | Stoles | Men | 3 | Thời trang |
| Apparel | Sweatshirts | Boys | 3 | Thời trang |
| Accessories | Trolley Bag | Unisex | 3 | Thời trang |
| Accessories | Travel Accessory | Women | 4 | Thời trang |
| Apparel | Kurta Sets | Boys | 4 | Thời trang |
| Apparel | Leggings | Girls | 4 | Thời trang |
| Accessories | Duffel Bag | Men | 4 | Thời trang |
| Apparel | Innerwear Vests | Boys | 4 | Thời trang |
| Accessories | Water Bottle | Men | 4 | Thời trang |
| Personal Care | Makeup Remover | Women | 4 | ⚠️ Ngoài thời trang |
| Apparel | Robe | Women | 4 | Thời trang |
| Personal Care | Lip Plumper | Women | 4 | ⚠️ Ngoài thời trang |
| Apparel | Jackets | Boys | 4 | Thời trang |
| Personal Care | Beauty Accessory | Women | 4 | ⚠️ Ngoài thời trang |
| Personal Care | Toner | Women | 5 | ⚠️ Ngoài thời trang |
| Apparel | Trousers | Boys | 5 | Thời trang |
| Apparel | Salwar | Girls | 5 | Thời trang |
| Apparel | Waistcoat | Women | 5 | Thời trang |
| Apparel | Kurtas | Boys | 5 | Thời trang |
| Accessories | Scarves | Unisex | 5 | Thời trang |
| Apparel | Innerwear Vests | Girls | 5 | Thời trang |
| Personal Care | Face Scrub and Exfoliator | Women | 5 | ⚠️ Ngoài thời trang |
| Accessories | Headband | Unisex | 5 | Thời trang |
| Accessories | Handbags | Men | 5 | Thời trang |
| Apparel | Clothing Set | Boys | 5 | Thời trang |
| Apparel | Nehru Jackets | Men | 5 | Thời trang |
| Apparel | Blazers | Men | 5 | Thời trang |
| Apparel | Capris | Boys | 5 | Thời trang |
| Accessories | Umbrellas | Women | 6 | Thời trang |
| Footwear | Sports Sandals | Unisex | 6 | Thời trang |
| Apparel | Rain Jacket | Men | 6 | Thời trang |
| Apparel | Shrug | Women | 6 | Thời trang |
| Accessories | Gloves | Boys | 6 | Thời trang |
| Apparel | Booties | Girls | 6 | Thời trang |
| Personal Care | Body Lotion | Women | 6 | ⚠️ Ngoài thời trang |
| Apparel | Booties | Boys | 6 | Thời trang |
| Footwear | Flip Flops | Girls | 6 | Thời trang |
| Accessories | Gloves | Girls | 6 | Thời trang |
| Personal Care | Eye Cream | Women | 6 | ⚠️ Ngoài thời trang |
| Personal Care | Nail Essentials | Women | 6 | ⚠️ Ngoài thời trang |
| Accessories | Mobile Pouch | Unisex | 6 | Thời trang |
| Apparel | Rompers | Boys | 6 | Thời trang |
| Apparel | Rompers | Girls | 6 | Thời trang |
| Apparel | Salwar and Dupatta | Women | 7 | Thời trang |
| Apparel | Jeans | Girls | 7 | Thời trang |
| Accessories | Duffel Bag | Women | 7 | Thời trang |
| Accessories | Gloves | Men | 7 | Thời trang |
| Apparel | Waistcoat | Men | 7 | Thời trang |
| Apparel | Tights | Women | 7 | Thời trang |
| Accessories | Mufflers | Women | 8 | Thời trang |
| Accessories | Belts | Unisex | 8 | Thời trang |
| Free Items | Free Gifts | Unisex | 8 | ⚠️ Ngoài thời trang |
| Accessories | Pendant | Men | 8 | Thời trang |
| Accessories | Watches | Boys | 8 | Thời trang |
| Apparel | Lounge Shorts | Women | 8 | Thời trang |
| Personal Care | Perfume and Body Mist | Unisex | 8 | ⚠️ Ngoài thời trang |
| Accessories | Laptop Bag | Women | 9 | Thời trang |
| Apparel | Shapewear | Women | 9 | Thời trang |
| Personal Care | Concealer | Women | 10 | ⚠️ Ngoài thời trang |
| Accessories | Rucksacks | Unisex | 11 | Thời trang |
| Footwear | Sandals | Boys | 11 | Thời trang |
| Accessories | Messenger Bag | Men | 11 | Thời trang |
| Sporting Goods | Basketballs | Unisex | 12 | ⚠️ Ngoài thời trang |
| Apparel | Swimwear | Women | 12 | Thời trang |
| Accessories | Travel Accessory | Unisex | 12 | Thời trang |
| Personal Care | Mask and Peel | Women | 12 | ⚠️ Ngoài thời trang |
| Apparel | Rain Jacket | Unisex | 12 | Thời trang |
| Apparel | Skirts | Girls | 13 | Thời trang |
| Personal Care | Mascara | Women | 13 | ⚠️ Ngoài thời trang |
| Footwear | Casual Shoes | Girls | 13 | Thời trang |
| Accessories | Bangle | Men | 14 | Thời trang |
| Personal Care | Lip Care | Women | 14 | ⚠️ Ngoài thời trang |
| Accessories | Watches | Girls | 15 | Thời trang |
| Apparel | Jumpsuit | Women | 16 | Thời trang |
| Accessories | Waist Pouch | Unisex | 16 | Thời trang |
| Footwear | Flip Flops | Boys | 16 | Thời trang |
| Personal Care | Hair Colour | Women | 16 | ⚠️ Ngoài thời trang |
| Accessories | Bracelet | Men | 16 | Thời trang |
| Apparel | Jeans | Boys | 16 | Thời trang |
| Accessories | Laptop Bag | Men | 16 | Thời trang |
| Apparel | Baby Dolls | Women | 16 | Thời trang |
| Apparel | Capris | Girls | 18 | Thời trang |
| Apparel | Bath Robe | Women | 20 | Thời trang |
| Accessories | Shoe Accessories | Unisex | 20 | Thời trang |
| Apparel | Shorts | Girls | 20 | Thời trang |
| Accessories | Wallets | Unisex | 21 | Thời trang |
| Apparel | Churidar | Women | 22 | Thời trang |
| Accessories | Caps | Women | 22 | Thời trang |
| Footwear | Casual Shoes | Boys | 22 | Thời trang |
| Personal Care | Sunscreen | Women | 23 | ⚠️ Ngoài thời trang |
| Apparel | Lounge Pants | Women | 23 | Thời trang |
| Apparel | Lounge Shorts | Men | 24 | Thời trang |
| Personal Care | Fragrance Gift Set | Women | 25 | ⚠️ Ngoài thời trang |
| Footwear | Sports Sandals | Women | 26 | Thời trang |
| Apparel | Jeggings | Women | 26 | Thời trang |
| Apparel | Salwar | Women | 26 | Thời trang |
| Accessories | Ring | Men | 26 | Thời trang |
| Apparel | Tracksuits | Men | 28 | Thời trang |
| Apparel | Stockings | Women | 28 | Thời trang |
| Personal Care | Face Wash and Cleanser | Women | 28 | ⚠️ Ngoài thời trang |
| Footwear | Sandals | Girls | 29 | Thời trang |
| Accessories | Mufflers | Men | 29 | Thời trang |

## 6. Nghiệm thu Pha 1b
- [x] Không có file nào bị xóa: mọi thao tác khôi phục và cách ly đều được ghi nhận trong `reports/quarantine_log.csv` (với cột `status` và `restored_at`).
- [x] Toàn bộ 40.891 ảnh trong `catalog.parquet` đều tồn tại trên đĩa, không trùng `item_id`.
- [x] Không còn 2 ảnh nào cùng (`articleType`, `gender`, `baseColour`) với pHash Hamming $\le 1$ trong catalog (0 vi phạm).
- [x] Báo cáo nêu rõ tỉ lệ cụm trùng cũ chứa ảnh khác thuộc tính (64,5%), số ảnh bị loại oan (88,0%), và số ảnh khôi phục (13.226 ảnh).
- [x] `notebooks/` và `docs/` được bảo toàn nguyên vẹn, không bị sửa.