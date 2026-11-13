# Báo cáo Kiểm tra Tập B-human (`eval/val_queries_b_human.json`) — Bước 2c.7

- **Trạng thái:** **KHÔNG ĐẠT NGƯỠNG TỐI THIỂU (`18 / 60` câu hợp lệ) — DỪNG CHỜ BỔ SUNG**
- **Tệp gốc:** `eval/val_queries_b_human.json` (giữ nguyên không sửa; `SHA-256: 964c46a13de86da308deacf188d73080970838a5abe03932757f73d7b7b47416`)
- **Cờ nguồn gốc (Mục 5):** **`provenance_unconfirmed`** — Trường `source` hiện giữ nguyên `"human_written"` (73/73 câu), người dùng chưa xác nhận đã đọc/sửa bản nháp. Cổng B-human khi đủ câu chỉ được mô tả là **"kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối"**.
- **Pytest:** `35 passed, 0 failed` (`pytest tests/ -v`)
- **Ràng buộc:** Không nạp model, không sửa `parser_v1.py`, `selection_rule_v3.*`, Tập A/B-synth/C, hay `val_queries_b_human.json`.

---

## 1. Tổng hợp Hợp lệ / Bị loại

| Hạng mục | Số lượng | Ghi chú |
|---|---:|---|
| Tổng số câu trong bản nháp | **73** | Yêu cầu tối thiểu $\ge 60$ câu hợp lệ |
| **Câu hợp lệ** | **18** | `n_relevant >= 5` trên `val_gallery`, nhãn hợp lệ, `Jaccard < 0.8` |
| **Câu bị loại** | **55** | **37** câu `n_relevant < 5` (trong đó **15** câu `n_relevant = 0`); **18** câu trùng/gần trùng `Jaccard >= 0.8` với DEV hoặc Tập A |
| **Số câu hợp lệ còn thiếu** | **42** | Cần tối thiểu $60 - 18 = 42$ câu hợp lệ bổ sung |
| `spec_cluster_id` độc nhất (toàn bộ 73 câu) | **57** | — |
| `spec_cluster_id` độc nhất (**18 câu hợp lệ**) | **14** | **Cảnh báo:** $< 30$ cụm — chưa đủ độ tin cậy cho bootstrap theo cụm ở cổng B-human |

> Vì số câu hợp lệ là **18 < 60**, theo mục 3–4 của Bước 2c.7: **KHÔNG** tạo `eval/val_queries_b_human.validated.json` hay `.sha256`, và **DỪNG** không chạy Bước 3.

### Lưu ý về `dia_class` (13 ca lệch giữa file và heuristic `detect_dia_class_auto`)
- **12 câu** ghi `"full"` trong file nhưng `detect_dia_class_auto` báo `"partial"` (`val_b_human_000, 001, 003, 004, 008, 012, 026, 035, 043, 045, 050, 053`) là do các từ tiếng Việt chuẩn vốn **không có dấu thanh** (`mua`, `can`, `cho`, `di`, `do`) nằm trong danh sách `unaccented_markers` của `check_b_human.py` — thực tế các câu này có dấu đầy đủ.
- **1 câu** (`val_b_human_070`: `"giày the thao nữ màu trắng"`) ghi `"partial"` là đúng thực tế (`"the thao"` thiếu dấu), trong khi `detect_dia_class_auto` nhận nhầm là `"full"` do `"the"`, `"thao"` chưa nằm trong `unaccented_markers`.

---

## 2. Token ngoài Lexicon `parser_v1` (CHỈ BÁO CÁO)

Tổng hợp trên 73 câu:
- `mua` (8), `di` (5), `tron` (3), `minh` (2), `muon` (2), `can` (2), `cho` (2), `rong` (1), `dang` (1), `diu` (1), `hoc` (1), `gong` (1), `hang` (1), `ngay` (1).

---

## 3. Bảng Đếm Phân bố Nhãn (73 câu trong bản nháp)

### 3.1. Theo `dia_class` (ghi trong file)
| `dia_class` | Số câu |
|---|---:|
| `full` | 56 |
| `none` | 13 |
| `partial` | 4 |

### 3.2. Theo `gender`
| `gender` | Số câu |
|---|---:|
| `Men` | 33 |
| `Women` | 29 |
| `None` (`null`) | 6 |
| `Boys` | 2 |
| `Girls` | 2 |
| `Unisex` | 1 |

### 3.3. Theo `baseColour`
`Black` (17), `White` (10), `Red` (7), `Pink` (5), `Brown` (5), `(empty)` (4), `Blue` (4), `Orange` (4), `Purple` (3), `Beige` (3), `Grey` (3), `Navy Blue` (3), `Silver` (2), `Off White` (1), `Yellow` (1), `Green` (1).

### 3.4. Theo `articleType`
`Tshirts` (13), `Dresses` (7), `Sports Shoes` (5), `Shirts` (4), `Wallets` (4), `Shorts` (4), `Handbags` (3), `Watches` (3), `Jeans` (2), `Backpacks` (2), `Casual Shoes` (2), `Flip Flops` (2), `Sunglasses` (2), `Caps` (2), `Socks` (2), `Belts` (2), `Jackets` (2), `Skirts` (1), `Heels` (1), `Flats` (1), `Sandals` (1), `Sweaters` (1), `Sweatshirts` (1), `Trousers` (1), `Track Pants` (1), `Tops` (1), `Scarves` (1), `Ties` (1), `Kurtas` (1).

*(Mọi nhãn đơn lẻ trong bản nháp đều tồn tại trong `catalog_rebuild.parquet`, nhưng nhiều tổ hợp 3 slot không đủ $\ge 5$ sản phẩm trong `index/val_gallery/`.)*

---

## 4. Danh sách 55 Câu Bị loại & Lý do Chi tiết

| ID | Truy vấn | Lý do bị loại |
|---|---|---|
| `val_b_human_006` | `đầm nữ màu đỏ đi tiệc` | `n_relevant=0 < 5` |
| `val_b_human_007` | `váy liền nữ màu hồng dịu dàng` | `n_relevant=3 < 5` |
| `val_b_human_008` | `tìm mua đầm công sở màu đen cho nữ` | `n_relevant=4 < 5` |
| `val_b_human_009` | `chân váy nữ màu be` | `n_relevant=0 < 5` |
| `val_b_human_010` | `túi xách nữ màu nâu` | `Jaccard=0.8333 >= 0.8 (DEV:h070)` |
| `val_b_human_012` | `ba lô đi học màu xanh dương` | `n_relevant=4 < 5` |
| `val_b_human_014` | `giày thể thao nam màu trắng` | `Jaccard=0.8333 >= 0.8 (DEV:h047)` |
| `val_b_human_015` | `giày thể thao nữ màu cam` | `n_relevant=0 < 5` |
| `val_b_human_017` | `giày cao gót nữ màu đỏ` | `n_relevant=3 < 5` |
| `val_b_human_018` | `giày búp bê nữ màu be` | `n_relevant=2 < 5` |
| `val_b_human_019` | `dép xỏ ngón nam màu đen` | `Jaccard=1.0 >= 0.8 (A:val_a_056)` |
| `val_b_human_020` | `dép sandal nữ màu nâu` | `n_relevant=0 < 5` |
| `val_b_human_021` | `đồng hồ nam dây da màu nâu` | `n_relevant=3 < 5` |
| `val_b_human_025` | `nón lưỡi trai màu đen` | `n_relevant=2 < 5` |
| `val_b_human_026` | `mũ lưỡi trai nam màu xanh navy` | `n_relevant=3 < 5` |
| `val_b_human_027` | `ví nam màu nâu` | `Jaccard=0.8 >= 0.8 (A:val_a_091)` |
| `val_b_human_028` | `ví cầm tay nữ màu đỏ` | `n_relevant=1 < 5` |
| `val_b_human_029` | `tất nam màu trắng` | `Jaccard=1.0 >= 0.8 (A:val_a_073)` |
| `val_b_human_030` | `vớ nữ màu hồng` | `n_relevant=1 < 5` |
| `val_b_human_031` | `thắt lưng nam màu đen` | `Jaccard=0.8333 >= 0.8 (A:val_a_048)` |
| `val_b_human_032` | `áo khoác nam màu xám` | `n_relevant=4 < 5` |
| `val_b_human_033` | `áo khoác nữ màu be` | `n_relevant=0 < 5` |
| `val_b_human_034` | `áo len nữ màu kem` | `n_relevant=0 < 5` |
| `val_b_human_035` | `áo nỉ nam màu xanh navy` | `n_relevant=0 < 5` |
| `val_b_human_036` | `quần short nam màu đen` | `n_relevant=3 < 5` |
| `val_b_human_037` | `quần short nữ màu trắng` | `n_relevant=0 < 5` |
| `val_b_human_038` | `quần tây nam màu xám` | `n_relevant=2 < 5` |
| `val_b_human_039` | `quần thể thao nam màu đen` | `Jaccard=0.8571 >= 0.8 (A:val_a_100)` |
| `val_b_human_040` | `áo kiểu nữ màu trắng` | `Jaccard=0.8333 >= 0.8 (A:val_a_126)` |
| `val_b_human_041` | `áo thun nữ màu hồng` | `Jaccard=0.8333 >= 0.8 (A:val_a_058)` |
| `val_b_human_042` | `áo thun nữ màu vàng` | `n_relevant=1 < 5` |
| `val_b_human_043` | `áo thun nam màu xanh lá` | `Jaccard=1.0 >= 0.8 (A:val_a_052)` |
| `val_b_human_044` | `áo thun nam màu cam` | `n_relevant=1 < 5` |
| `val_b_human_045` | `áo thun bé trai màu xanh dương` | `n_relevant=2 < 5` |
| `val_b_human_046` | `áo thun bé gái màu hồng` | `n_relevant=0 < 5` |
| `val_b_human_047` | `đầm bé gái màu trắng` | `n_relevant=0 < 5` |
| `val_b_human_048` | `quần short bé trai màu đen` | `n_relevant=0 < 5` |
| `val_b_human_049` | `áo thun màu đỏ` | `Jaccard=0.8 >= 0.8 (A:val_a_049)` |
| `val_b_human_052` | `khăn quàng cổ màu đỏ` | `n_relevant=0 < 5` |
| `val_b_human_053` | `cà vạt nam màu xanh navy` | `n_relevant=0 < 5` |
| `val_b_human_054` | `áo kurta nam màu trắng` | `n_relevant=0 < 5` |
| `val_b_human_055` | `ví nữ màu hồng` | `n_relevant=3 < 5` |
| `val_b_human_059` | `dam nu mau do di tiec` | `n_relevant=0 < 5` |
| `val_b_human_060` | `tim mua dam cong so mau den cho nu` | `n_relevant=4 < 5` |
| `val_b_human_061` | `giay the thao nam mau trang` | `Jaccard=0.8333 >= 0.8 (DEV:h047)` |
| `val_b_human_062` | `giay the thao nu mau cam` | `n_relevant=0 < 5` |
| `val_b_human_063` | `dep xo ngon nam mau den` | `Jaccard=1.0 >= 0.8 (A:val_a_056)` |
| `val_b_human_065` | `vi nam mau nau` | `Jaccard=0.8 >= 0.8 (A:val_a_091)` |
| `val_b_human_066` | `that lung nam mau den` | `Jaccard=0.8333 >= 0.8 (A:val_a_048)` |
| `val_b_human_067` | `quan short nam mau den` | `n_relevant=3 < 5` |
| `val_b_human_068` | `ao thun nam mau cam` | `n_relevant=1 < 5` |
| `val_b_human_069` | `ao thun nam mau tím` | `Jaccard=0.8333 >= 0.8 (A:val_a_007)` |
| `val_b_human_070` | `giày the thao nữ màu trắng` | `Jaccard=0.8571 >= 0.8 (A:val_a_074)` |
| `val_b_human_071` | `tui xách nu mau đen` | `Jaccard=0.8333 >= 0.8 (A:val_a_024)` |
| `val_b_human_072` | `dam nu màu do` | `n_relevant=0 < 5` |

---

## 5. Nhãn còn thiếu & Gợi ý Bổ sung (Đạt $\ge 60$ câu hợp lệ & $\ge 30$ cụm)

- Hiện chỉ có **18** câu hợp lệ (**14** cụm `spec_cluster_id`), chủ yếu rơi vào `Tshirts`, `Shirts`, `Jeans`, `Handbags`, `Backpacks`, `Casual Shoes`, `Watches`, `Sunglasses`.
- **Các nhóm có nhiều sản phẩm trong `index/val_gallery/` ($\ge 15$ sp) nhưng hiện có `0` câu hợp lệ trong B-human:**
  - `Kurtas` (95 sp — lưu ý chỉ có nhóm `Women`, không phải `Men+White`), `Sports Shoes` (95 sp), `Tops` (84 sp), `Heels` (63 sp), `Wallets` (45 sp), `Belts` (37 sp), `Sandals` (34 sp), `Briefs` (31 sp), `Flip Flops` (30 sp), `Socks` (29 sp), `Formal Shoes` (24 sp), `Bra` (24 sp), `Earrings` (24 sp), `Track Pants` (22 sp), `Flats` (21 sp), `Caps` (20 sp), `Trousers` (19 sp), `Sarees` (19 sp), `Shorts` (18 sp), `Ties` (16 sp), `Innerwear Vests` (16 sp), `Jackets` (15 sp), `Dresses` (15 sp).
- **Cách bổ sung hiệu quả để tránh `n_relevant < 5` và tránh `Jaccard >= 0.8`:**
  - Dùng mô tả **2 slot** (`articleType + gender` hoặc `articleType + baseColour`) cho các nhóm cỡ trung bình (`Tops`, `Heels`, `Formal Shoes`, `Sandals`, `Flip Flops`, `Flats`, `Track Pants`, `Trousers`, `Shorts`, `Jackets`, `Dresses`) thay vì ép cả 3 slot hiếm.
  - Diễn đạt tự nhiên, đa dạng ngữ cảnh (`cần tìm...`, `gợi ý mẫu...`, `mặc đi làm/đi chơi...`) để không trùng `Jaccard >= 0.8` với câu ngắn mẫu mực của Tập A/DEV.
