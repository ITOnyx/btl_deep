# Worksheet B-human-v4 (`index/val_gallery/` — 1.884 sản phẩm)

> **Chỉ dẫn:** Worksheet chỉ liệt kê cụm khả dụng (`n_relevant >= 5`) và **đã loại bỏ** toàn bộ cụm từng dùng trong `B-human` cũ (`eval/val_queries_b_human.json`). Không viết sẵn câu truy vấn.

## 1. Yêu cầu tập `B-human-v4`
- Số câu hợp lệ: **`>= 60` câu**
- Số cụm `spec_cluster_id` khác nhau: **`>= 30` cụm**
- Số lượng ảnh đúng mỗi câu (`n_relevant`): **`>= 5`** (theo chính sách `hybrid`)
- Câu ngoài lexicon v2.1: **`>= 20` câu**
- Câu không dấu (`dia_class == "none"`): **`>= 20` câu**
- Câu lỗi gõ (`typo_type` khác null): **đúng `10` câu**
- Không trùng lặp Jaccard `>= 0.8` với `DEV / A / B-synth / C / B-human cũ` và nội bộ.

## 2. Thống kê cụm khả dụng
- Số cụm đã loại vì trùng `B-human` cũ: **84**
- Tổng cụm khả dụng còn lại (`n_relevant >= 5`): **225** (1-slot: 35, 2-slot gender: 57, 2-slot colour: 69, 3-slot: 64)
- Số cụm hiếm (`is_rare=True`, `n_relevant` 5..15 hoặc `articleType`/`baseColour` < 20 mẫu trong val): **157**
- Số cụm chưa từng xuất hiện ở DEV/A/B-synth: **60**
- File CSV đầy đủ: `reports/phase5c/b_human_v4_worksheet.csv`

## 3. Mẫu Top 40 Cụm Hiếm Chưa Từng Dùng (`is_rare=True`, `completely_unused=True`)

| Loại slot | `articleType` | `gender` | `baseColour` | `n_relevant` | `spec_cluster_id` |
|---|---|---|---|---:|---|
| 1_slot (articleType) | `Sarees` | `null` | `[]` | 19 | `Sarees__None__` |
| 1_slot (articleType) | `Trousers` | `null` | `[]` | 19 | `Trousers__None__` |
| 1_slot (articleType) | `Shorts` | `null` | `[]` | 18 | `Shorts__None__` |
| 1_slot (articleType) | `Innerwear Vests` | `null` | `[]` | 16 | `Innerwear Vests__None__` |
| 1_slot (articleType) | `Dresses` | `null` | `[]` | 15 | `Dresses__None__` |
| 1_slot (articleType) | `Jackets` | `null` | `[]` | 15 | `Jackets__None__` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Black | 11 | `Kurtas__None__Black` |
| 1_slot (articleType) | `Kurtis` | `null` | `[]` | 11 | `Kurtis__None__` |
| 1_slot (articleType) | `Night suits` | `null` | `[]` | 11 | `Night suits__None__` |
| 2_slot (articleType+baseColour) | `Track Pants` | `null` | Black | 11 | `Track Pants__None__Black` |
| 2_slot (articleType+gender) | `Accessory Gift Set` | `Men` | `[]` | 10 | `Accessory Gift Set__Men__` |
| 1_slot (articleType) | `Accessory Gift Set` | `null` | `[]` | 10 | `Accessory Gift Set__None__` |
| 1_slot (articleType) | `Necklace and Chains` | `null` | `[]` | 10 | `Necklace and Chains__None__` |
| 1_slot (articleType) | `Nightdress` | `null` | `[]` | 10 | `Nightdress__None__` |
| 1_slot (articleType) | `Sweatshirts` | `null` | `[]` | 10 | `Sweatshirts__None__` |
| 1_slot (articleType) | `Tunics` | `null` | `[]` | 9 | `Tunics__None__` |
| 1_slot (articleType) | `Dupatta` | `null` | `[]` | 8 | `Dupatta__None__` |
| 2_slot (articleType+baseColour) | `Flats` | `null` | Black | 8 | `Flats__None__Black` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | White | 8 | `Kurtas__None__White` |
| 1_slot (articleType) | `Pendant` | `null` | `[]` | 8 | `Pendant__None__` |
| 2_slot (articleType+baseColour) | `Pendant` | `null` | Silver | 8 | `Pendant__None__Silver` |
| 2_slot (articleType+gender) | `Pendant` | `Women` | `[]` | 8 | `Pendant__Women__` |
| 2_slot (articleType+gender) | `Tops` | `Girls` | `[]` | 8 | `Tops__Girls__` |
| 2_slot (articleType+baseColour) | `Bra` | `null` | Black | 7 | `Bra__None__Black` |
| 2_slot (articleType+baseColour) | `Bra` | `null` | White | 7 | `Bra__None__White` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Purple | 7 | `Kurtas__None__Purple` |
| 1_slot (articleType) | `Leggings` | `null` | `[]` | 7 | `Leggings__None__` |
| 1_slot (articleType) | `Kurta Sets` | `null` | `[]` | 6 | `Kurta Sets__None__` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Brown | 6 | `Kurtas__None__Brown` |
| 2_slot (articleType+gender) | `Leggings` | `Women` | `[]` | 6 | `Leggings__Women__` |
| 2_slot (articleType+gender) | `Night suits` | `Women` | `[]` | 6 | `Night suits__Women__` |
| 2_slot (articleType+gender) | `Socks` | `Women` | `[]` | 6 | `Socks__Women__` |
| 1_slot (articleType) | `Bracelet` | `null` | `[]` | 5 | `Bracelet__None__` |
| 1_slot (articleType) | `Capris` | `null` | `[]` | 5 | `Capris__None__` |
| 2_slot (articleType+baseColour) | `Casual Shoes` | `null` | Olive | 5 | `Casual Shoes__None__Olive` |
| 3_slot (articleType+gender+baseColour) | `Casual Shoes` | `Women` | Black | 5 | `Casual Shoes__Women__Black` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Grey | 5 | `Kurtas__None__Grey` |
| 1_slot (articleType) | `Lounge Pants` | `null` | `[]` | 5 | `Lounge Pants__None__` |
| 2_slot (articleType+gender) | `Night suits` | `Men` | `[]` | 5 | `Night suits__Men__` |
| 2_slot (articleType+baseColour) | `Sarees` | `null` | Green | 5 | `Sarees__None__Green` |

## 4. Mẫu Top 40 Cụm Phổ Biến Chưa Từng Dùng (`is_rare=False`, `completely_unused=True`)

| Loại slot | `articleType` | `gender` | `baseColour` | `n_relevant` | `spec_cluster_id` |
|---|---|---|---|---:|---|
| 1_slot (articleType) | `Shirts` | `null` | `[]` | 147 | `Shirts__None__` |
| 1_slot (articleType) | `Casual Shoes` | `null` | `[]` | 125 | `Casual Shoes__None__` |
| 1_slot (articleType) | `Kurtas` | `null` | `[]` | 95 | `Kurtas__None__` |
| 1_slot (articleType) | `Sports Shoes` | `null` | `[]` | 95 | `Sports Shoes__None__` |
| 1_slot (articleType) | `Handbags` | `null` | `[]` | 86 | `Handbags__None__` |
| 1_slot (articleType) | `Heels` | `null` | `[]` | 63 | `Heels__None__` |
| 1_slot (articleType) | `Backpacks` | `null` | `[]` | 33 | `Backpacks__None__` |
| 1_slot (articleType) | `Sunglasses` | `null` | `[]` | 32 | `Sunglasses__None__` |
| 2_slot (articleType+gender) | `Backpacks` | `Men` | `[]` | 30 | `Backpacks__Men__` |
| 2_slot (articleType+gender) | `Backpacks` | `Women` | `[]` | 30 | `Backpacks__Women__` |
| 1_slot (articleType) | `Flip Flops` | `null` | `[]` | 30 | `Flip Flops__None__` |
| 1_slot (articleType) | `Bra` | `null` | `[]` | 24 | `Bra__None__` |
| 1_slot (articleType) | `Track Pants` | `null` | `[]` | 22 | `Track Pants__None__` |
| 1_slot (articleType) | `Flats` | `null` | `[]` | 21 | `Flats__None__` |
| 1_slot (articleType) | `Caps` | `null` | `[]` | 20 | `Caps__None__` |
| 2_slot (articleType+baseColour) | `Kurtas` | `null` | Green | 16 | `Kurtas__None__Green` |