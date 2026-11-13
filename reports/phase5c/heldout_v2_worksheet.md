# Worksheet held-out v2 (`index/test_gallery/` — 1.290 sản phẩm, KHÔNG chạy model)

> **Tuân thủ Luật C.5 & C.10:** Chỉ thống kê thuộc tính catalog (`id_map.json` + `styles.csv`). KHÔNG nạp/chạy model trên `test_gallery`. KHÔNG viết sẵn câu truy vấn.

## 1. Yêu cầu tập `held-out v2`
- Số câu truy vấn: **`>= 100` câu**
- Số lượng ảnh đúng mỗi câu (`n_relevant`): **`>= 5`**
- Câu không dấu (`dia_class == "none"`): **`>= 40` câu**
- Câu ngoài lexicon v2.1: **`>= 30%`** tổng số câu
- Câu có tín hiệu `usage`/`season` (nhóm khớp 3 slot có **5%..70%** ảnh khớp ràng buộc phụ): **`>= 40` câu**
- Phủ đủ các **slot mỏng** (`is_thin_slot=True`).

## 2. Thống kê cụm khả dụng trên `test_gallery`
- Tổng số cụm khả dụng (`n_relevant >= 5`): **242** (1-slot: 41, 2-slot gender: 68, 2-slot colour: 63, 3-slot: 70)
- Số cụm slot mỏng (`is_thin_slot=True`): **177**
- Số cụm **3-slot** có tín hiệu `usage`/`season` trong khoảng **5%..70%**: **65**
- File CSV đầy đủ: `reports/phase5c/heldout_v2_worksheet.csv`

## 3. Top 50 Nhóm 3-Slot Có Tín Hiệu `usage`/`season` (5%..70%)

| `articleType` | `gender` | `baseColour` | `n_rel` | `usage` (5..70%) | `season` (5..70%) |
|---|---|---|---:|---|---|
| `Tshirts` | `Men` | `Black` | 20 | Sports:4/20(20%) | Fall:11/20(55%); Summer:9/20(45%) |
| `Tshirts` | `Men` | `Red` | 20 | Sports:4/20(20%) | Summer:11/20(55%); Fall:9/20(45%) |
| `Shirts` | `Men` | `Blue` | 16 | Formal:3/16(19%) | Summer:8/16(50%); Fall:8/16(50%) |
| `Shirts` | `Men` | `White` | 16 | Casual:11/16(69%); Formal:5/16(31%) | Winter:1/16(6%); Summer:1/16(6%) |
| `Tshirts` | `Men` | `Grey` | 15 | Sports:3/15(20%) | Summer:9/15(60%); Fall:5/15(33%); Winter:1/15(7%) |
| `Tshirts` | `Men` | `Blue` | 14 | Casual:8/14(57%); Sports:6/14(43%) | Summer:8/14(57%); Fall:6/14(43%) |
| `Tshirts` | `Men` | `White` | 14 | Sports:3/14(21%) | Summer:8/14(57%); Fall:6/14(43%) |
| `Watches` | `Men` | `Black` | 14 | Sports:1/14(7%) | - |
| `Shirts` | `Men` | `Purple` | 13 | Casual:7/13(54%); Formal:6/13(46%) | Summer:2/13(15%); Winter:1/13(8%) |
| `Watches` | `Men` | `White` | 12 | Sports:1/12(8%) | - |
| `Formal Shoes` | `Men` | `Brown` | 11 | Smart Casual:1/11(9%) | Fall:5/11(45%); Summer:4/11(36%); Spring:1/11(9%); Winter:1/11(9%) |
| `Casual Shoes` | `Men` | `Red` | 10 | - | Summer:7/10(70%); Fall:2/10(20%); Spring:1/10(10%) |
| `Tshirts` | `Women` | `Black` | 10 | Casual:7/10(70%); Sports:3/10(30%) | Summer:7/10(70%); Fall:3/10(30%) |
| `Backpacks` | `Unisex` | `Blue` | 9 | Sports:2/9(22%) | Summer:6/9(67%); Winter:1/9(11%); Fall:1/9(11%); Spring:1/9(11%) |
| `Casual Shoes` | `Men` | `Black` | 9 | - | Fall:5/9(56%); Summer:2/9(22%); Spring:1/9(11%); Winter:1/9(11%) |
| `Jeans` | `Men` | `Blue` | 9 | - | Fall:2/9(22%) |
| `Shirts` | `Men` | `Green` | 9 | Casual:6/9(67%); Formal:3/9(33%) | Summer:5/9(56%); Fall:4/9(44%) |
| `Sports Shoes` | `Men` | `White` | 9 | - | Fall:2/9(22%) |
| `Tops` | `Women` | `Black` | 9 | Ethnic:1/9(11%) | Summer:5/9(56%); Fall:4/9(44%) |
| `Tops` | `Women` | `Pink` | 9 | Ethnic:1/9(11%) | Fall:1/9(11%) |
| `Tshirts` | `Men` | `Green` | 9 | - | Fall:5/9(56%); Summer:4/9(44%) |
| `Tshirts` | `Men` | `Purple` | 9 | Sports:1/9(11%) | Fall:2/9(22%) |
| `Watches` | `Women` | `Black` | 9 | Sports:1/9(11%) | - |
| `Casual Shoes` | `Men` | `White` | 8 | - | Fall:4/8(50%); Summer:3/8(38%); Spring:1/8(12%) |
| `Handbags` | `Women` | `Brown` | 8 | - | Winter:2/8(25%) |
| `Sunglasses` | `Men` | `Black` | 8 | - | Summer:1/8(12%) |
| `Tshirts` | `Men` | `Navy Blue` | 8 | Sports:1/8(12%) | Fall:2/8(25%) |
| `Casual Shoes` | `Girls` | `Red` | 7 | - | Summer:4/7(57%); Fall:1/7(14%); Spring:1/7(14%); Winter:1/7(14%) |
| `Casual Shoes` | `Women` | `Brown` | 7 | - | Summer:4/7(57%); Winter:2/7(29%); Spring:1/7(14%) |
| `Casual Shoes` | `Women` | `Red` | 7 | - | Summer:4/7(57%); Fall:2/7(29%); Spring:1/7(14%) |
| `Heels` | `Women` | `Blue` | 7 | - | Summer:1/7(14%) |
| `Kurtas` | `Women` | `Green` | 7 | - | Fall:1/7(14%) |
| `Kurtas` | `Women` | `White` | 7 | - | Fall:2/7(29%) |
| `Shirts` | `Men` | `Black` | 7 | Formal:1/7(14%) | Fall:4/7(57%); Summer:3/7(43%) |
| `Shirts` | `Men` | `Red` | 7 | Formal:1/7(14%) | Fall:4/7(57%); Summer:2/7(29%); Spring:1/7(14%) |
| `Tshirts` | `Men` | `Brown` | 7 | - | Summer:4/7(57%); Fall:3/7(43%) |
| `Tshirts` | `Women` | `Green` | 7 | Sports:2/7(29%) | Summer:4/7(57%); Fall:3/7(43%) |
| `Watches` | `Women` | `White` | 7 | Sports:1/7(14%) | - |
| `Backpacks` | `Unisex` | `Black` | 6 | - | Summer:3/6(50%); Fall:3/6(50%) |
| `Casual Shoes` | `Men` | `Brown` | 6 | - | Summer:4/6(67%); Fall:2/6(33%) |
| `Casual Shoes` | `Men` | `Navy Blue` | 6 | - | Spring:3/6(50%); Summer:2/6(33%); Fall:1/6(17%) |
| `Casual Shoes` | `Unisex` | `Red` | 6 | - | Summer:4/6(67%); Fall:1/6(17%); Spring:1/6(17%) |
| `Casual Shoes` | `Women` | `Black` | 6 | - | Winter:1/6(17%) |
| `Handbags` | `Women` | `Black` | 6 | - | Summer:3/6(50%); Winter:2/6(33%); Fall:1/6(17%) |
| `Kurtas` | `Women` | `Cream` | 6 | - | Fall:3/6(50%); Summer:3/6(50%) |
| `Sandals` | `Men` | `Brown` | 6 | - | Summer:4/6(67%); Winter:1/6(17%); Fall:1/6(17%) |
| `Sports Shoes` | `Men` | `Black` | 6 | - | Summer:3/6(50%); Fall:3/6(50%) |
| `Sports Shoes` | `Women` | `Purple` | 6 | - | Summer:3/6(50%); Fall:2/6(33%); Winter:1/6(17%) |
| `Tshirts` | `Women` | `Yellow` | 6 | - | Fall:3/6(50%); Summer:3/6(50%) |
| `Wallets` | `Men` | `Black` | 6 | - | Summer:4/6(67%); Winter:2/6(33%) |

## 4. Top 40 Cụm Slot Mỏng (`is_thin_slot=True`, `n_relevant >= 5`)

| Loại slot | `articleType` | `gender` | `baseColour` | `n_rel` | `usage/season` (5..70%) |
|---|---|---|---|---:|---|
| 3_slot (articleType+gender+baseColour) | `Belts` | `Men` | Black | 5 | Summer:3/5(60%); Winter:2/5(40%) |
| 2_slot (articleType+baseColour) | `Belts` | `null` | Black | 5 | Summer:3/5(60%); Winter:2/5(40%) |
| 2_slot (articleType+baseColour) | `Belts` | `null` | Blue | 5 | Winter:2/5(40%); Summer:2/5(40%); Fall:1/5(20%) |
| 1_slot (articleType) | `Capris` | `null` | `[]` | 5 | Sports:1/5(20%); Summer:1/5(20%) |
| 2_slot (articleType+gender) | `Capris` | `Women` | `[]` | 5 | Sports:1/5(20%); Summer:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Casual Shoes` | `Men` | Blue | 5 | - |
| 2_slot (articleType+baseColour) | `Casual Shoes` | `null` | Beige | 5 | Summer:2/5(40%); Spring:2/5(40%); Fall:1/5(20%) |
| 2_slot (articleType+baseColour) | `Casual Shoes` | `null` | Blue | 5 | - |
| 2_slot (articleType+baseColour) | `Handbags` | `null` | Olive | 5 | Fall:1/5(20%) |
| 2_slot (articleType+baseColour) | `Handbags` | `null` | Purple | 5 | Winter:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Handbags` | `Women` | Olive | 5 | Fall:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Handbags` | `Women` | Purple | 5 | Winter:1/5(20%) |
| 2_slot (articleType+baseColour) | `Heels` | `null` | Red | 5 | Summer:3/5(60%); Winter:2/5(40%) |
| 3_slot (articleType+gender+baseColour) | `Heels` | `Women` | Red | 5 | Summer:3/5(60%); Winter:2/5(40%) |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Brown | 5 | Fall:3/5(60%); Summer:2/5(40%) |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Maroon | 5 | Summer:3/5(60%); Fall:2/5(40%) |
| 3_slot (articleType+gender+baseColour) | `Kurtas` | `Women` | Brown | 5 | Fall:3/5(60%); Summer:2/5(40%) |
| 3_slot (articleType+gender+baseColour) | `Kurtas` | `Women` | Maroon | 5 | Summer:3/5(60%); Fall:2/5(40%) |
| 1_slot (articleType) | `Night suits` | `null` | `[]` | 5 | - |
| 2_slot (articleType+gender) | `Pendant` | `Women` | `[]` | 5 | Summer:3/5(60%); Winter:1/5(20%); Fall:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Sandals` | `Men` | Red | 5 | Summer:1/5(20%) |
| 2_slot (articleType+baseColour) | `Sandals` | `null` | Red | 5 | Summer:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Shirts` | `Men` | Navy Blue | 5 | Formal:1/5(20%); Summer:1/5(20%) |
| 2_slot (articleType+baseColour) | `Shirts` | `null` | Navy Blue | 5 | Formal:1/5(20%); Summer:1/5(20%) |
| 2_slot (articleType+baseColour) | `Shorts` | `null` | Black | 5 | Casual:3/5(60%); Sports:2/5(40%); Summer:2/5(40%); Winter:2/5(40%); Fall:1/5(20%) |
| 2_slot (articleType+baseColour) | `Socks` | `null` | Black | 5 | Fall:1/5(20%) |
| 2_slot (articleType+gender) | `Socks` | `Women` | `[]` | 5 | Sports:1/5(20%); Summer:3/5(60%); Fall:2/5(40%) |
| 3_slot (articleType+gender+baseColour) | `Sports Shoes` | `Men` | Silver | 5 | Fall:1/5(20%) |
| 2_slot (articleType+baseColour) | `Sports Shoes` | `null` | Grey | 5 | Fall:1/5(20%) |
| 2_slot (articleType+gender) | `Sweaters` | `Women` | `[]` | 5 | - |
| 2_slot (articleType+gender) | `Track Pants` | `Men` | `[]` | 5 | Casual:1/5(20%); Summer:3/5(60%); Fall:2/5(40%) |
| 2_slot (articleType+gender) | `Trunk` | `Men` | `[]` | 5 | Winter:1/5(20%) |
| 1_slot (articleType) | `Trunk` | `null` | `[]` | 5 | Winter:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Tshirts` | `Boys` | Black | 5 | Fall:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Tshirts` | `Men` | Yellow | 5 | Fall:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Tshirts` | `Women` | Blue | 5 | Summer:3/5(60%); Fall:2/5(40%) |
| 3_slot (articleType+gender+baseColour) | `Tshirts` | `Women` | White | 5 | Sports:1/5(20%); Fall:1/5(20%) |
| 3_slot (articleType+gender+baseColour) | `Wallets` | `Women` | Black | 5 | Summer:3/5(60%); Winter:1/5(20%); Fall:1/5(20%) |
| 2_slot (articleType+gender) | `Watches` | `Boys` | `[]` | 5 | Casual:3/5(60%); Sports:2/5(40%) |
| 2_slot (articleType+gender) | `Watches` | `Girls` | `[]` | 5 | Casual:3/5(60%); Sports:2/5(40%) |