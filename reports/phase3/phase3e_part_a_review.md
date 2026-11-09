# Pha 3e — Báo Cáo Nghiệm Thu Phần A (Xây Dựng Thành Phần Suy Luận)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 09/11/2026  
**Trạng thái:** Hoàn tất A1 đến A6, DỪNG chờ người dùng duyệt từ điển & nộp `heldout_queries.json`.

---

## 1. Top 100 Ứng Viên Thương Hiệu (`src/brands.json`) Cần Duyệt

Được trích xuất từ 1–2 từ đầu của `productDisplayName` trên 38.646 sản phẩm `in_scope`, chỉ lấy các thương hiệu có $\ge 20$ sản phẩm (tổng cộng 234 thương hiệu đạt chuẩn):

| STT | Tên thương hiệu | Số sản phẩm | STT | Tên thương hiệu | Số sản phẩm |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 1 | **Nike** | 2.003 | 51 | Highlander | 204 |
| 2 | **ADIDAS** | 1.969 | 52 | Aneri | 199 |
| 3 | **Puma** | 1.898 | 53 | Tantra | 198 |
| 4 | **United Colors of Benetton** | 1.462 | 54 | Doodle | 196 |
| 5 | **Fabindia** | 737 | 55 | Kiara | 195 |
| 6 | **Catwalk** | 710 | 56 | Numero Uno | 192 |
| 7 | **Wrangler** | 668 | 57 | Titan | 189 |
| 8 | **Arrow** | 636 | 58 | Playboy | 188 |
| 9 | **Lino Perros** | 630 | 59 | Vans | 184 |
| 10 | **Baggit** | 622 | 60 | FNF | 182 |
| 11 | **Fastrack** | 621 | 61 | Carlton London | 178 |
| 12 | **Myntra** | 460 | 62 | Red Rose | 175 |
| 13 | **Fila** | 434 | 63 | John Miller | 166 |
| 14 | **French Connection** | 413 | 64 | Enamor | 163 |
| 15 | **Jealous 21** | 396 | 65 | Elle | 161 |
| 16 | **Jockey** | 382 | 66 | John Players | 159 |
| 17 | **Scullers** | 373 | 67 | Classic Polo | 158 |
| 18 | **W** | 352 | 68 | Converse | 157 |
| 19 | **Murcia** | 350 | 69 | H&M | 155 |
| 20 | **Lee** | 344 | 70 | Skechers | 151 |
| 21 | **Proline** | 342 | 71 | Timex | 149 |
| 22 | **Gini and Jony** | 341 | 72 | Franco Leone | 144 |
| 23 | **Wildcraft** | 338 | 73 | Clarks | 140 |
| 24 | **Flying Machine** | 335 | 74 | SDL by Sweet Dreams | 140 |
| 25 | **Lotto** | 334 | 75 | Facit | 139 |
| 26 | **Rocia** | 315 | 76 | Disney | 139 |
| 27 | **Fossil** | 312 | 77 | Park Avenue | 137 |
| 28 | **Mark Taylor** | 312 | 78 | Amante | 135 |
| 29 | **Casio** | 311 | 79 | Shree | 134 |
| 30 | **Basics** | 311 | 80 | Lee Cooper | 133 |
| 31 | **U.S. Polo Assn.** | 304 | 81 | Enroute | 132 |
| 32 | **Lucera** | 300 | 82 | Inkfruit | 126 |
| 33 | **Locomotive** | 296 | 83 | s.Oliver | 125 |
| 34 | **Peter England** | 294 | 84 | Diva | 124 |
| 35 | **Femella** | 287 | 85 | Tonga | 124 |
| 36 | **Mother Earth** | 262 | 86 | Turtle | 122 |
| 37 | **Reid & Taylor** | 244 | 87 | Status Quo | 122 |
| 38 | **Q&Q** | 244 | 88 | Allen Solly | 120 |
| 39 | **Maxima** | 242 | 89 | BIBA | 117 |
| 40 | **Chromozome** | 240 | 90 | Cobblerz | 116 |
| 41 | **Reebok** | 237 | 91 | Woodland | 113 |
| 42 | **Levi's** | 236 | 92 | Royal Diadem | 109 |
| 43 | **Vishudh** | 230 | 93 | Sepia | 107 |
| 44 | **Spykar** | 228 | 94 | Quiksilver | 107 |
| 45 | **Aurelia** | 220 | 95 | Avirate | 106 |
| 46 | **Indigo Nation** | 213 | 96 | Mr.Men | 102 |
| 47 | **Timberland** | 211 | 97 | Urban Yoga | 100 |
| 48 | **Hanes** | 211 | 98 | Bulchee | 100 |
| 49 | **Red Tape** | 211 | 99 | Portia | 99 |
| 50 | **Probase** | 209 | 100 | Genesis | 99 |

---

## 2. Thống Kê Từ Điển Ngữ Nghĩa (`src/lexicon_vi.json`)
- **60 ArticleTypes phổ biến nhất:** Mỗi loại có ít nhất 4–10 từ đồng nghĩa / cách diễn đạt tiếng Việt chuẩn hóa, độc lập hoàn toàn với giới tính (loại bỏ tiền tố giới để không gây xung đột khớp).
- **47 BaseColours:** Mỗi màu có từ 4–9 từ đồng nghĩa (kèm ánh sắc, cách gọi phổ thông và tiếng Anh).
- **5 Genders:** Men, Women, Boys, Girls, Unisex (kèm tiếng lóng và biến thể).
- **Usages & Seasons:** Phân loại rõ ràng cho Casual, Sports, Formal, Ethnic, Party, Travel, v.v.

---

## 3. Độ Phủ & Độ Chính Xác Parser Trên Q2 & Q3 (A6)

### Bộ Truy Vấn Thuộc Tính Q2 ($N=61$):
- **ArticleType:** **100.0%** (61/61)
- **Gender:** **100.0%** (61/61)
- **BaseColour:** **100.0%** (61/61)
- **Khớp hoàn hảo cả 3 thuộc tính:** **100.0%** (61/61)
- **Không nhận diện được gì:** **0 câu** (0.0%)
- **Từ chưa nhận dạng thường gặp:** `mau` (9 lần - hư từ), `ao` (1 lần - từ dư).

### Bộ Truy Vấn Viết Tay Q3 ($N=100$):
- **ArticleType:** **92.0%** (92/100)
- **Gender:** **86.0%** (86/100)
- **BaseColour:** **86.0%** (86/100)
- **Không nhận diện được gì:** **0 câu** (0.0%)
- **Từ chưa nhận dạng thường gặp (Q3):**
  - Chất liệu / Kiểu dáng phụ: `da` (11 - đồ da), `day` (6 - quai/dây đeo), `de` (5 - đế giày), `tron` (5 - trơn), `dung` (5 - dáng đứng/đựng), `bo` (5 - bò/bó).
  - Hành động / Ngữ cảnh: `co` (8 - có/cổ áo), `di` (8 - đi chơi/đi làm), `mat` (8 - mặt/mát), `lam` (4 - làm việc), `chong` (4 - chống nước).
  - Tính từ phong cách: `tinh` (4 - cá tính), `sang` (4 - sang trọng).

---

## 4. Bộ Truy Vấn Thương Hiệu Q5 (`eval/q5_brand.json`)
- **Tổng số truy vấn sinh ra:** Đúng **150 truy vấn** tiếng Việt kết hợp thương hiệu + chủng loại (+ giới tính tương thích).
- **Seed cố định:** 42.
- **Tiêu chí Relevance:** Cùng thương hiệu + cùng articleType (+ giới tính tương thích nếu truy vấn có giới tính).
- **SHA-256:** `ad9fe474b06ee2708e5ec7687f19bab68c740fac8f3d28ca0dce1878ecb89fb8`.
