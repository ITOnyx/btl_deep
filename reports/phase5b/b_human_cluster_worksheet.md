# Bảng Cụm Khả dụng trên `index/val_gallery/` (Hỗ trợ Bổ sung B-human)

- **Nguồn:** `index/val_gallery/id_map.json` (1.884 sản phẩm val sạch), chính sách GT `hybrid` (`eval/gender_gt_policy.json`).
- **Điều kiện lọc:** Mọi tổ hợp 1-slot (`articleType`), 2-slot (`articleType+gender`, `articleType+baseColour`), 3-slot (`articleType+gender+baseColour`) có `n_relevant >= 5`.
- **Lưu ý slot không nêu:** `gender=null` (không lọc giới tính), `baseColour` rỗng (không lọc màu).
- **Tổng số cụm khả dụng (`n_relevant >= 5`):** **309** cụm (1-slot: 50, 2-slot gender: 75, 2-slot colour: 87, 3-slot: 97).
- **Cụm chưa ai dùng (`total_used == 0`):** **113** cụm.
- **Cụm đã dùng >= 3 lần (`total_used >= 3`):** **92** cụm (cần tránh để hạn chế trùng lặp).
- **File CSV đầy đủ (309 dòng):** `reports/phase5b/b_human_cluster_worksheet.csv`.

## 1. Nhãn chuẩn trong `index/val_gallery/` (chép nguyên văn vào `spec`)

- **`articleType` (80):** `Accessory Gift Set`, `Baby Dolls`, `Backpacks`, `Bangle`, `Bath Robe`, `Belts`, `Boxers`, `Bra`, `Bracelet`, `Briefs`, `Capris`, `Caps`, `Casual Shoes`, `Clothing Set`, `Clutches`, `Cufflinks`, `Dresses`, `Duffel Bag`, `Dupatta`, `Earrings`, `Flats`, `Flip Flops`, `Formal Shoes`, `Gloves`, `Handbags`, `Headband`, `Heels`, `Innerwear Vests`, `Jackets`, `Jeans`, `Jewellery Set`, `Kurta Sets`, `Kurtas`, `Kurtis`, `Laptop Bag`, `Leggings`, `Lehenga Choli`, `Lounge Pants`, `Messenger Bag`, `Mobile Pouch`, `Mufflers`, `Necklace and Chains`, `Night suits`, `Nightdress`, `Patiala`, `Pendant`, `Rain Jacket`, `Ring`, `Salwar`, `Salwar and Dupatta`, `Sandals`, `Sarees`, `Scarves`, `Shirts`, `Shoe Laces`, `Shorts`, `Shrug`, `Skirts`, `Socks`, `Sports Sandals`, `Sports Shoes`, `Stockings`, `Stoles`, `Sunglasses`, `Sweaters`, `Sweatshirts`, `Swimwear`, `Ties`, `Tops`, `Track Pants`, `Tracksuits`, `Travel Accessory`, `Trousers`, `Trunk`, `Tshirts`, `Tunics`, `Waist Pouch`, `Waistcoat`, `Wallets`, `Watches`
- **`gender` (5 + `null`):** `Boys`, `Girls`, `Men`, `Unisex`, `Women`, `null`
- **`baseColour` (41 + `[]`):** `Beige`, `Black`, `Blue`, `Bronze`, `Brown`, `Burgundy`, `Charcoal`, `Coffee Brown`, `Copper`, `Cream`, `Gold`, `Green`, `Grey`, `Grey Melange`, `Khaki`, `Lavender`, `Lime Green`, `Magenta`, `Maroon`, `Metallic`, `Multi`, `Mushroom Brown`, `Mustard`, `Navy Blue`, `Off White`, `Olive`, `Orange`, `Peach`, `Pink`, `Purple`, `Red`, `Rust`, `Silver`, `Skin`, `Steel`, `Tan`, `Taupe`, `Teal`, `Turquoise Blue`, `White`, `Yellow`, `[]`

## 2. Top 60 Cụm Chưa Ai Dùng (`total_used == 0`, sắp theo `n_relevant` giảm dần)

| Loại slot | `articleType` | `gender` | `baseColour` | `n_relevant` | `spec_cluster_id` |
|---|---|---|---|---:|---|
| 1_slot (articleType) | `Tshirts` | `null` | `[]` | 314 | `Tshirts__None__` |
| 1_slot (articleType) | `Shirts` | `null` | `[]` | 147 | `Shirts__None__` |
| 1_slot (articleType) | `Casual Shoes` | `null` | `[]` | 125 | `Casual Shoes__None__` |
| 1_slot (articleType) | `Watches` | `null` | `[]` | 118 | `Watches__None__` |
| 1_slot (articleType) | `Kurtas` | `null` | `[]` | 95 | `Kurtas__None__` |
| 1_slot (articleType) | `Sports Shoes` | `null` | `[]` | 95 | `Sports Shoes__None__` |
| 1_slot (articleType) | `Handbags` | `null` | `[]` | 86 | `Handbags__None__` |
| 1_slot (articleType) | `Tops` | `null` | `[]` | 84 | `Tops__None__` |
| 1_slot (articleType) | `Heels` | `null` | `[]` | 63 | `Heels__None__` |
| 1_slot (articleType) | `Wallets` | `null` | `[]` | 45 | `Wallets__None__` |
| 1_slot (articleType) | `Belts` | `null` | `[]` | 37 | `Belts__None__` |
| 1_slot (articleType) | `Sandals` | `null` | `[]` | 34 | `Sandals__None__` |
| 2_slot (articleType+baseColour) | `Watches` | `null` | Black | 34 | `Watches__None__Black` |
| 1_slot (articleType) | `Backpacks` | `null` | `[]` | 33 | `Backpacks__None__` |
| 1_slot (articleType) | `Sunglasses` | `null` | `[]` | 32 | `Sunglasses__None__` |
| 1_slot (articleType) | `Briefs` | `null` | `[]` | 31 | `Briefs__None__` |
| 2_slot (articleType+gender) | `Backpacks` | `Men` | `[]` | 30 | `Backpacks__Men__` |
| 2_slot (articleType+gender) | `Backpacks` | `Women` | `[]` | 30 | `Backpacks__Women__` |
| 1_slot (articleType) | `Flip Flops` | `null` | `[]` | 30 | `Flip Flops__None__` |
| 1_slot (articleType) | `Socks` | `null` | `[]` | 29 | `Socks__None__` |
| 2_slot (articleType+baseColour) | `Sports Shoes` | `null` | Black | 27 | `Sports Shoes__None__Black` |
| 1_slot (articleType) | `Bra` | `null` | `[]` | 24 | `Bra__None__` |
| 1_slot (articleType) | `Earrings` | `null` | `[]` | 24 | `Earrings__None__` |
| 1_slot (articleType) | `Formal Shoes` | `null` | `[]` | 24 | `Formal Shoes__None__` |
| 1_slot (articleType) | `Track Pants` | `null` | `[]` | 22 | `Track Pants__None__` |
| 1_slot (articleType) | `Flats` | `null` | `[]` | 21 | `Flats__None__` |
| 1_slot (articleType) | `Jeans` | `null` | `[]` | 21 | `Jeans__None__` |
| 1_slot (articleType) | `Caps` | `null` | `[]` | 20 | `Caps__None__` |
| 1_slot (articleType) | `Sarees` | `null` | `[]` | 19 | `Sarees__None__` |
| 1_slot (articleType) | `Trousers` | `null` | `[]` | 19 | `Trousers__None__` |
| 1_slot (articleType) | `Shorts` | `null` | `[]` | 18 | `Shorts__None__` |
| 2_slot (articleType+gender) | `Casual Shoes` | `Boys` | `[]` | 17 | `Casual Shoes__Boys__` |
| 2_slot (articleType+gender) | `Casual Shoes` | `Girls` | `[]` | 16 | `Casual Shoes__Girls__` |
| 2_slot (articleType+gender) | `Innerwear Vests` | `Men` | `[]` | 16 | `Innerwear Vests__Men__` |
| 1_slot (articleType) | `Innerwear Vests` | `null` | `[]` | 16 | `Innerwear Vests__None__` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Blue | 16 | `Kurtas__None__Blue` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Green | 16 | `Kurtas__None__Green` |
| 1_slot (articleType) | `Ties` | `null` | `[]` | 16 | `Ties__None__` |
| 1_slot (articleType) | `Dresses` | `null` | `[]` | 15 | `Dresses__None__` |
| 1_slot (articleType) | `Jackets` | `null` | `[]` | 15 | `Jackets__None__` |
| 3_slot (articleType+gender+baseColour) | `Heels` | `Women` | Brown | 14 | `Heels__Women__Brown` |
| 1_slot (articleType) | `Clutches` | `null` | `[]` | 13 | `Clutches__None__` |
| 2_slot (articleType+gender) | `Caps` | `Women` | `[]` | 12 | `Caps__Women__` |
| 2_slot (articleType+baseColour) | `Casual Shoes` | `null` | Brown | 12 | `Casual Shoes__None__Brown` |
| 2_slot (articleType+baseColour) | `Shirts` | `null` | Black | 12 | `Shirts__None__Black` |
| 2_slot (articleType+baseColour) | `Shirts` | `null` | Green | 12 | `Shirts__None__Green` |
| 2_slot (articleType+baseColour) | `Tops` | `null` | White | 12 | `Tops__None__White` |
| 3_slot (articleType+gender+baseColour) | `Backpacks` | `Men` | Black | 11 | `Backpacks__Men__Black` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Black | 11 | `Kurtas__None__Black` |
| 1_slot (articleType) | `Kurtis` | `null` | `[]` | 11 | `Kurtis__None__` |
| 2_slot (articleType+gender) | `Kurtis` | `Women` | `[]` | 11 | `Kurtis__Women__` |
| 1_slot (articleType) | `Night suits` | `null` | `[]` | 11 | `Night suits__None__` |
| 2_slot (articleType+baseColour) | `Track Pants` | `null` | Black | 11 | `Track Pants__None__Black` |
| 2_slot (articleType+baseColour) | `Watches` | `null` | Steel | 11 | `Watches__None__Steel` |
| 2_slot (articleType+gender) | `Accessory Gift Set` | `Men` | `[]` | 10 | `Accessory Gift Set__Men__` |
| 1_slot (articleType) | `Accessory Gift Set` | `null` | `[]` | 10 | `Accessory Gift Set__None__` |
| 2_slot (articleType+baseColour) | `Heels` | `null` | Gold | 10 | `Heels__None__Gold` |
| 1_slot (articleType) | `Necklace and Chains` | `null` | `[]` | 10 | `Necklace and Chains__None__` |
| 2_slot (articleType+gender) | `Necklace and Chains` | `Women` | `[]` | 10 | `Necklace and Chains__Women__` |
| 1_slot (articleType) | `Nightdress` | `null` | `[]` | 10 | `Nightdress__None__` |

## 3. Các Cụm Đã Dùng >= 3 Lần (Nên tránh)

| `spec_cluster_id` | `n_relevant` | DEV | A | B-synth | B-human (hợp lệ) | Tổng dùng |
|---|---:|---:|---:|---:|---:|---:|
| `Jeans__Men__Blue` | 14 | 2 | 4 | 1 | 1 | 8 |
| `Shirts__Men__White` | 23 | 2 | 3 | 2 | 1 | 8 |
| `Handbags__Women__Brown` | 15 | 2 | 3 | 2 | 0 | 7 |
| `Tshirts__Men__White` | 29 | 2 | 3 | 2 | 0 | 7 |
| `Watches__Men__Black` | 26 | 4 | 2 | 1 | 0 | 7 |
| `Watches__Women__Silver` | 6 | 1 | 3 | 1 | 2 | 7 |
| `Belts__Men__Brown` | 10 | 0 | 3 | 3 | 0 | 6 |
| `Casual Shoes__Women__White` | 10 | 0 | 4 | 2 | 0 | 6 |
| `Earrings__Women__Silver` | 12 | 1 | 3 | 2 | 0 | 6 |
| `Sports Shoes__Men__White` | 38 | 2 | 3 | 1 | 0 | 6 |
| `Sunglasses__Men__Black` | 12 | 2 | 2 | 1 | 1 | 6 |
| `Tshirts__Women__Blue` | 7 | 0 | 4 | 2 | 0 | 6 |
| `Watches__Men__Silver` | 9 | 0 | 4 | 2 | 0 | 6 |
| `Watches__Men__White` | 19 | 0 | 4 | 2 | 0 | 6 |
| `Casual Shoes__Men__Blue` | 7 | 0 | 4 | 1 | 0 | 5 |
| `Handbags__Women__Black` | 19 | 1 | 2 | 1 | 1 | 5 |
| `Shirts__Men__Green` | 12 | 1 | 3 | 1 | 0 | 5 |
| `Shirts__Men__Red` | 8 | 0 | 3 | 2 | 0 | 5 |
| `Sports Shoes__Women__White` | 6 | 0 | 2 | 3 | 0 | 5 |
| `Tshirts__Men__Black` | 33 | 2 | 0 | 1 | 2 | 5 |
| `Tshirts__Women__Green` | 8 | 0 | 3 | 2 | 0 | 5 |
| `Tshirts__Women__White` | 14 | 0 | 3 | 2 | 0 | 5 |
| `Wallets__Women__Brown` | 5 | 0 | 2 | 3 | 0 | 5 |
| `Watches__Women__White` | 21 | 1 | 2 | 2 | 0 | 5 |
| `Backpacks__Unisex__Black` | 10 | 0 | 3 | 1 | 0 | 4 |
| `Backpacks__Unisex__Red` | 8 | 0 | 3 | 1 | 0 | 4 |
| `Belts__Men__Black` | 11 | 1 | 3 | 0 | 0 | 4 |
| `Casual Shoes__Men__` | 107 | 0 | 1 | 3 | 0 | 4 |
| `Casual Shoes__Men__White` | 15 | 0 | 3 | 1 | 0 | 4 |
| `Casual Shoes__Unisex__White` | 5 | 0 | 3 | 1 | 0 | 4 |
| `Earrings__Women__` | 24 | 0 | 2 | 2 | 0 | 4 |
| `Formal Shoes__Men__Black` | 14 | 0 | 2 | 2 | 0 | 4 |
| `Formal Shoes__Men__Brown` | 9 | 1 | 2 | 1 | 0 | 4 |
| `Handbags__Women__Blue` | 8 | 0 | 2 | 2 | 0 | 4 |
| `Handbags__Women__Green` | 5 | 0 | 3 | 1 | 0 | 4 |
| `Sandals__Men__` | 32 | 1 | 1 | 2 | 0 | 4 |
| `Sandals__Men__Brown` | 11 | 0 | 3 | 1 | 0 | 4 |
| `Shirts__Men__` | 132 | 0 | 1 | 1 | 2 | 4 |
| `Shirts__Men__Blue` | 33 | 0 | 2 | 2 | 0 | 4 |
| `Shirts__Men__Brown` | 6 | 0 | 2 | 2 | 0 | 4 |
| `Shirts__Men__Grey` | 11 | 0 | 1 | 3 | 0 | 4 |
| `Shirts__Men__Pink` | 5 | 0 | 3 | 1 | 0 | 4 |
| `Socks__Men__White` | 6 | 1 | 1 | 2 | 0 | 4 |
| `Sports Shoes__Women__` | 21 | 0 | 2 | 2 | 0 | 4 |
| `Sunglasses__Women__` | 18 | 1 | 0 | 2 | 1 | 4 |
| `Sunglasses__Women__Brown` | 8 | 0 | 2 | 2 | 0 | 4 |
| `Tops__Women__Pink` | 11 | 0 | 3 | 1 | 0 | 4 |
| `Tshirts__Boys__` | 16 | 1 | 1 | 2 | 0 | 4 |
| `Tshirts__Men__Grey` | 39 | 0 | 3 | 1 | 0 | 4 |
| `Tshirts__Men__Purple` | 11 | 0 | 1 | 1 | 2 | 4 |
| `Tshirts__Women__Pink` | 7 | 0 | 4 | 0 | 0 | 4 |
| `Wallets__Men__Black` | 13 | 2 | 1 | 1 | 0 | 4 |
| `Backpacks__None__Black` | 11 | 0 | 0 | 2 | 1 | 3 |
| `Backpacks__None__Blue` | 5 | 0 | 0 | 3 | 0 | 3 |
| `Belts__Men__` | 24 | 0 | 2 | 1 | 0 | 3 |
| `Bra__Women__Black` | 7 | 0 | 3 | 0 | 0 | 3 |
| `Briefs__Men__` | 20 | 0 | 1 | 2 | 0 | 3 |
| `Casual Shoes__Men__Navy Blue` | 9 | 0 | 3 | 0 | 0 | 3 |
| `Earrings__Women__Gold` | 6 | 0 | 3 | 0 | 0 | 3 |
| `Handbags__None__Black` | 19 | 0 | 0 | 3 | 0 | 3 |
| `Handbags__Women__` | 86 | 0 | 1 | 2 | 0 | 3 |
| `Handbags__Women__Purple` | 5 | 0 | 2 | 1 | 0 | 3 |
| `Heels__Women__Black` | 17 | 2 | 1 | 0 | 0 | 3 |
| `Heels__Women__Gold` | 10 | 0 | 3 | 0 | 0 | 3 |
| `Jackets__Men__Black` | 5 | 1 | 2 | 0 | 0 | 3 |
| `Jeans__Women__` | 5 | 0 | 0 | 2 | 1 | 3 |
| `Kurtas__Women__Blue` | 14 | 0 | 3 | 0 | 0 | 3 |
| `Pendant__Women__Silver` | 8 | 0 | 3 | 0 | 0 | 3 |
| `Shirts__Men__Black` | 9 | 0 | 3 | 0 | 0 | 3 |
| `Shirts__Men__Navy Blue` | 9 | 0 | 2 | 1 | 0 | 3 |
| `Shirts__Men__Purple` | 7 | 1 | 2 | 0 | 0 | 3 |
| `Shirts__Women__` | 14 | 0 | 2 | 1 | 0 | 3 |
| `Shorts__Men__` | 16 | 1 | 1 | 1 | 0 | 3 |
| `Sports Shoes__Men__` | 77 | 1 | 1 | 1 | 0 | 3 |
| `Sports Shoes__Men__Black` | 23 | 0 | 1 | 2 | 0 | 3 |
| `Tops__Women__` | 76 | 0 | 2 | 1 | 0 | 3 |
| `Tops__Women__Blue` | 11 | 0 | 2 | 1 | 0 | 3 |
| `Tops__Women__Navy Blue` | 6 | 0 | 2 | 1 | 0 | 3 |
| `Tops__Women__White` | 10 | 0 | 2 | 1 | 0 | 3 |
| `Track Pants__Men__Grey` | 5 | 1 | 2 | 0 | 0 | 3 |
| `Tshirts__Men__` | 244 | 0 | 1 | 2 | 0 | 3 |
| `Tshirts__Men__Blue` | 45 | 0 | 1 | 2 | 0 | 3 |
| `Tshirts__Men__Green` | 19 | 0 | 1 | 2 | 0 | 3 |
| `Tshirts__Men__Red` | 20 | 0 | 3 | 0 | 0 | 3 |
| `Tshirts__Men__Yellow` | 7 | 0 | 3 | 0 | 0 | 3 |
| `Tshirts__Women__` | 50 | 0 | 0 | 3 | 0 | 3 |
| `Wallets__Men__Brown` | 7 | 1 | 1 | 1 | 0 | 3 |
| `Wallets__None__Black` | 14 | 0 | 0 | 3 | 0 | 3 |
| `Wallets__Women__` | 26 | 0 | 2 | 1 | 0 | 3 |
| `Watches__Men__` | 71 | 1 | 1 | 1 | 0 | 3 |
| `Watches__Women__` | 54 | 0 | 1 | 2 | 0 | 3 |
| `Watches__Women__Steel` | 7 | 0 | 3 | 0 | 0 | 3 |