# Báo Cáo Nghiệm Thu Pha 3: Đánh Giá Cố Định (Fixed Evaluation)
**Thời gian thực hiện**: 2026-11-08 23:06:09
**Checkpoint SigLIP**: `data/checkpoints_siglip-vit/best_model.pt`
**Index FAISS**: `index/faiss_hnsw_index.bin` ($M=32, efConstruction=200$)

---

## 1. Bảng Đối Chiếu Chỉ Tiêu Dự Án (Nghiệm Thu)

| Chỉ tiêu | Mục tiêu | Kết quả thực tế ($G_{test}$) | Kết quả $G_{full}$ (Lạc quan) | Trạng thái |
| :--- | :--- | :--- | :--- | :---: |
| **Recall@10** (Text $\to$ Ảnh paired) | $\ge 0.85$ | **0.0597** (95% CI: [0.0465, 0.0736]) | 0.0016 (CI: [0.0000, 0.0039]) | ❌ KHÔNG ĐẠT |
| **mAP / MRR** (Text $\to$ Ảnh paired) | $\ge 0.75$ | **0.0228** (95% CI: [0.0180, 0.0274]) | 0.0008 (CI: [0.0004, 0.0012]) | ❌ KHÔNG ĐẠT |
| **Độ trễ Text $\to$ Ảnh** (p95) | $< 200$ ms | **42.87 ms** (p50: 23.79 ms) | — | ✅ ĐẠT |
| **Độ trễ Ảnh $\to$ Ảnh** (p95) | $< 200$ ms | **158.08 ms** (p50: 97.25 ms) | — | ✅ ĐẠT |

> **Ghi chú về Ground Rule Đánh giá**:
> - Checkpoint đã train trên 161k mẫu bao gồm toàn bộ phần lớn ảnh Myntra Kaggle.
> - Theo quy tắc chống rò rỉ tại `AGENTS.md`, số liệu nghiệm thu CHỈ tính trên tập con test độc lập $G_{test}$ ($n=1.290$).
> - Số liệu trên $G_{full}$ ($n=38.646$) chứa các ảnh trong tập train nên được ghi nhận là lạc quan (để tham khảo năng lực phục vụ thực tế của index).

---

## 2. Chi Tiết Kết Quả Paired Retrieval (Q1)

| Bộ Gallery | efSearch | Recall@1 | Recall@5 | Recall@10 | MRR (= mAP) | Median Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$G_{test}$ ($n=1.290$)** | Exact | 0.0039 | 0.0349 | 0.0597 | 0.0228 | 999999.0 |
| **$G_{full}$ ($n=38.646$)** | 128 | 0.0000 | 0.0000 | 0.0016 | 0.0008 | 999999.0 |
| **$G_{full}$ ($n=38.646$)** | 64 | 0.0000 | 0.0000 | 0.0016 | 0.0007 | 999999.0 |

---

## 3. Đánh Giá Truy Vấn Thuộc Tính Tự Nhiên (Q2) & Ba Chế Độ Văn Bản

| Chế độ văn bản | Mức độ | Precision@10 ($G_{test}$) | Hit@10 ($G_{test}$) | AP@10 ($G_{test}$) | P@10 ($G_{full}$ - Lạc quan) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a) Có dấu** | Strict | 0.3967 | 0.9508 | 0.3810 | 0.6049 |
| | Family | 0.5311 | 0.9672 | 0.4882 | 0.7279 |
| **(b) Không dấu thô** | Strict | 0.0820 | 0.2459 | 0.0754 | 0.1098 |
| | Family | 0.1148 | 0.2951 | 0.0938 | 0.1492 |
| **(c) Khôi phục dấu** | Strict | 0.3902 | 0.9180 | 0.3736 | 0.5770 |
| | Family | 0.5164 | 0.9344 | 0.4735 | 0.7000 |

---

## 4. Đánh Giá Bộ 100 Truy Vấn Viết Tay (Q3)

| Chế độ văn bản | Mức độ | Precision@10 ($G_{test}$) | Hit@10 ($G_{test}$) | AP@10 ($G_{test}$) | P@10 ($G_{full}$ - Lạc quan) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a) Gốc (25% ko dấu)** | Strict | 0.2460 | 0.6100 | 0.2412 | 0.3520 |
| | Family | 0.2910 | 0.6400 | 0.2807 | 0.3950 |
| **(b) Không dấu toàn bộ** | Strict | 0.0700 | 0.2200 | 0.0497 | 0.0850 |
| | Family | 0.0940 | 0.2600 | 0.0725 | 0.1100 |
| **(c) Khôi phục dấu** | Strict | 0.3440 | 0.7600 | 0.3421 | 0.4700 |
| | Family | 0.3900 | 0.7900 | 0.3838 | 0.5310 |

---

## 5. Đánh Giá Truy Vấn Ảnh $\to$ Ảnh (Q4 - Loại bỏ chính nó)

| Tiêu chí tương đồng | Precision@10 | Hit@10 | AP@10 |
| :--- | :---: | :---: | :---: |
| Cùng articleType + gender | 0.6983 | 0.9442 | 0.6870 |
| Cùng articleType + gender + baseColour | 0.1570 | 0.6124 | 0.2299 |

---

## 6. So Sánh Truy Vấn Có Lọc Thuộc Tính vs Không Lọc (B3)

| Tập truy vấn | Precision@10 Không lọc | Precision@10 Có lọc | Tỉ lệ Relaxation |
| :--- | :---: | :---: | :---: |
| Q2 (Template 61 câu) | 0.6049 | **0.8525** | 9.8% |
| Q3 (Viết tay 100 câu) | 0.3520 | **0.5810** | 49.0% |

---

## 7. Phân Tích Lỗi & Chẩn Đoán (B4)

- **Nhóm phụ kiện cận cảnh (Headwear / Eyewear / Watches / Bags)**: Tỉ lệ lỗi xếp hạng: **99.6%** ($n=254$).
- **Top 10 cặp nhầm lẫn ngành hàng (True $\to$ Predicted)**:
  - `Tshirts` bị nhầm sang `Handbags`: 69 lần
  - `Tshirts` bị nhầm sang `Backpacks`: 61 lần
  - `Handbags` bị nhầm sang `Backpacks`: 57 lần
  - `Kurtas` bị nhầm sang `Backpacks`: 48 lần
  - `Heels` bị nhầm sang `Backpacks`: 40 lần
  - `Watches` bị nhầm sang `Accessory Gift Set`: 38 lần
  - `Sports Shoes` bị nhầm sang `Duffel Bag`: 37 lần
  - `Casual Shoes` bị nhầm sang `Duffel Bag`: 33 lần
  - `Tops` bị nhầm sang `Backpacks`: 32 lần
  - `Casual Shoes` bị nhầm sang `Handbags`: 29 lần

- **Top 5 cặp nhầm lẫn màu sắc (True $\to$ Predicted)**:
  - Màu `Black` bị nhầm sang `Purple`: 82 lần
  - Màu `Blue` bị nhầm sang `Purple`: 57 lần
  - Màu `White` bị nhầm sang `Purple`: 54 lần
  - Màu `Blue` bị nhầm sang `Black`: 46 lần
  - Màu `Pink` bị nhầm sang `Purple`: 45 lần

- **Contact sheet mẫu 30 câu viết tay**: Đã xuất 30 ảnh vào `reports/contact_sheets/hw_*.png` và bảng đánh giá tại `eval/manual_ratings_template.csv`.

---

## 8. Phân Rã Theo Ngành Hàng (Subgroup Breakdown - Q1 trên $G_{test}$)

| Ngành hàng (articleType) | Số lượng ($n$) | Recall@10 | MRR | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| `Tshirts` | 216 | 0.0046 | 0.0028 | Đáng tin cậy |
| `Shirts` | 101 | 0.0000 | 0.0000 | Đáng tin cậy |
| `Casual Shoes` | 87 | 0.0000 | 0.0033 | Đáng tin cậy |
| `Watches` | 81 | 0.0617 | 0.0311 | Đáng tin cậy |
| `Kurtas` | 61 | 0.0000 | 0.0046 | Đáng tin cậy |
| `Sports Shoes` | 59 | 0.1186 | 0.0392 | Đáng tin cậy |
| `Handbags` | 59 | 0.0678 | 0.0233 | Đáng tin cậy |
| `Tops` | 55 | 0.0000 | 0.0031 | Đáng tin cậy |
| `Heels` | 41 | 0.0000 | 0.0025 | Đáng tin cậy |
| `Wallets` | 31 | 0.0323 | 0.0140 | Đáng tin cậy |
| `Sunglasses` | 29 | 0.1379 | 0.0593 | Đáng tin cậy |
| `Briefs` | 29 | 0.0000 | 0.0000 | Đáng tin cậy |
| `Flip Flops` | 27 | 0.0000 | 0.0030 | Đáng tin cậy |
| `Belts` | 26 | 0.0769 | 0.0315 | Đáng tin cậy |
| `Backpacks` | 24 | 0.0417 | 0.0143 | Đáng tin cậy |
| `Sandals` | 23 | 0.0000 | 0.0027 | Đáng tin cậy |
| `Socks` | 23 | 0.1739 | 0.0440 | Đáng tin cậy |
| `Jeans` | 18 | 0.0000 | 0.0017 | Tham khảo ($n < 20$) |
| `Flats` | 17 | 0.0000 | 0.0012 | Tham khảo ($n < 20$) |
| `Bra` | 16 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Shorts` | 16 | 0.3750 | 0.2005 | Tham khảo ($n < 20$) |
| `Dresses` | 15 | 0.0000 | 0.0087 | Tham khảo ($n < 20$) |
| `Formal Shoes` | 14 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Sarees` | 14 | 0.5714 | 0.1492 | Tham khảo ($n < 20$) |
| `Trousers` | 12 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Earrings` | 12 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Caps` | 10 | 0.7000 | 0.2751 | Tham khảo ($n < 20$) |
| `Clutches` | 10 | 0.1000 | 0.0344 | Tham khảo ($n < 20$) |
| `Sweatshirts` | 10 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Sweaters` | 9 | 0.5556 | 0.2236 | Tham khảo ($n < 20$) |
| `Kurtis` | 8 | 0.0000 | 0.0058 | Tham khảo ($n < 20$) |
| `Tunics` | 8 | 0.0000 | 0.0287 | Tham khảo ($n < 20$) |
| `Innerwear Vests` | 8 | 0.0000 | 0.0028 | Tham khảo ($n < 20$) |
| `Jackets` | 8 | 0.0000 | 0.0063 | Tham khảo ($n < 20$) |
| `Track Pants` | 7 | 0.0000 | 0.0116 | Tham khảo ($n < 20$) |
| `Ties` | 7 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Nightdress` | 6 | 0.6667 | 0.1547 | Tham khảo ($n < 20$) |
| `Pendant` | 6 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Trunk` | 5 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Night suits` | 5 | 0.6000 | 0.1924 | Tham khảo ($n < 20$) |
| `Capris` | 5 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Cufflinks` | 4 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Scarves` | 4 | 0.0000 | 0.0096 | Tham khảo ($n < 20$) |
| `Dupatta` | 4 | 0.5000 | 0.0718 | Tham khảo ($n < 20$) |
| `Skirts` | 3 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Ring` | 3 | 0.0000 | 0.0076 | Tham khảo ($n < 20$) |
| `Bangle` | 3 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Accessory Gift Set` | 3 | 0.6667 | 0.1810 | Tham khảo ($n < 20$) |
| `Duffel Bag` | 3 | 0.6667 | 0.1889 | Tham khảo ($n < 20$) |
| `Kurta Sets` | 3 | 0.6667 | 0.1259 | Tham khảo ($n < 20$) |
| `Laptop Bag` | 3 | 0.0000 | 0.0620 | Tham khảo ($n < 20$) |
| `Necklace and Chains` | 3 | 0.0000 | 0.0085 | Tham khảo ($n < 20$) |
| `Leggings` | 3 | 0.0000 | 0.0035 | Tham khảo ($n < 20$) |
| `Stoles` | 3 | 0.6667 | 0.1398 | Tham khảo ($n < 20$) |
| `Mobile Pouch` | 2 | 0.0000 | 0.0200 | Tham khảo ($n < 20$) |
| `Boxers` | 2 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Sports Sandals` | 2 | 0.0000 | 0.0333 | Tham khảo ($n < 20$) |
| `Lounge Pants` | 2 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Churidar` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Bath Robe` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Baby Dolls` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Camisoles` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Bracelet` | 1 | 0.0000 | 0.0526 | Tham khảo ($n < 20$) |
| `Jewellery Set` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Patiala` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Mufflers` | 1 | 0.0000 | 0.0200 | Tham khảo ($n < 20$) |
| `Messenger Bag` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Lounge Shorts` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Jumpsuit` | 1 | 1.0000 | 0.2000 | Tham khảo ($n < 20$) |
| `Gloves` | 1 | 1.0000 | 0.1000 | Tham khảo ($n < 20$) |
| `Jeggings` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Shoe Accessories` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Stockings` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Rain Jacket` | 1 | 0.0000 | 0.0556 | Tham khảo ($n < 20$) |
| `Salwar` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Swimwear` | 1 | 1.0000 | 0.2500 | Tham khảo ($n < 20$) |
| `Travel Accessory` | 1 | 0.0000 | 0.0435 | Tham khảo ($n < 20$) |
| `Tracksuits` | 1 | 1.0000 | 1.0000 | Tham khảo ($n < 20$) |
| `Waistcoat` | 1 | 0.0000 | 0.0000 | Tham khảo ($n < 20$) |
| `Waist Pouch` | 1 | 0.0000 | 0.0238 | Tham khảo ($n < 20$) |

---

## 9. Đo Độ Trễ End-to-End (B5)

| Luồng truy vấn | p50 (ms) | p95 (ms) | p99 (ms) | Trung bình (ms) | Ngưỡng chỉ tiêu | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Text $	o$ Ảnh** | 23.79 | **42.87** | 56.54 | 27.25 | $< 200$ ms | ✅ ĐẠT |
| **Ảnh $	o$ Ảnh** | 97.25 | **158.08** | 169.80 | 103.84 | $< 200$ ms | ✅ ĐẠT |