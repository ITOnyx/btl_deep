# Báo cáo Kiểm tra Tập B-human (`eval/val_queries_b_human.json`) — Bước 2c.7

- **Trạng thái:** **ĐẠT NGƯỠNG TỐI THIỂU (`65 / 60` câu hợp lệ, `61` cụm `spec_cluster_id` độc nhất) — ĐÃ CẬP NHẬT NGUỒN `ai_written`**
- **Tệp gốc:** `eval/val_queries_b_human.json` (giữ nguyên không sửa; `SHA-256: 21b1b85f40c4a45e6ff8a53e635f4cc4d9b20618253e412e6d9105e40468d4b8`)
- **Tệp đã xác thực:** `eval/val_queries_b_human.validated.json` (`65` câu hợp lệ, có `spec_cluster_id` và `n_relevant`; `SHA-256: 19f81e8aa35a034f1528dc101e108ad0469201d50372e16afbc5286e78bd8deb`, lưu tại `eval/val_queries_b_human.sha256`; khớp 100% danh sách 65 ID so với bản trước; hash giữ nguyên do tệp gốc đã mang `source="ai_written"`).
- **Nguồn gốc (`source`):** **`ai_written`: 131/131 (100%)** trên tệp gốc và **65/65 (100%)** trên `validated.json`. Do toàn bộ câu là AI soạn, cổng B-human được mô tả chính xác là **"kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối"**.
- **Thống kê bổ sung (trên 65 câu hợp lệ):** **`31` câu có màu / `34` câu không màu**; `61` cụm độc nhất, trong đó **`4` cụm có $\ge 2$ câu** (`Tshirts__Men__Black`: 2, `Tshirts__Men__Purple`: 2, `Shirts__Men__`: 2, `Watches__Women__Silver`: 2).
- **Ràng buộc:** Không nạp model, không sửa `parser_v1.py`, `selection_rule_v3.*`, Tập A/B-synth/C, hay `val_queries_b_human.json`.

---

## 1. Tổng hợp Hợp lệ / Bị loại

| Hạng mục | Số lượng | Ghi chú |
|---|---:|---|
| Tổng số câu trong tệp gốc | **131** | Yêu cầu tối thiểu $\ge 60$ câu hợp lệ |
| **Câu hợp lệ (`validated`)** | **65** | `n_relevant >= 5` trên `val_gallery`, nhãn hợp lệ trong catalog, `Jaccard < 0.8` |
| **Câu bị loại** | **66** | **37** câu `n_relevant < 5` (trong đó **15** câu `n_relevant = 0`); **29** câu trùng/gần trùng `Jaccard >= 0.8` với DEV hoặc Tập A |
| **`spec_cluster_id` độc nhất (65 câu hợp lệ)** | **61** | $\ge 30$ cụm — đủ độ tin cậy cho paired cluster bootstrap ở cổng B-human |
| Cảnh báo lệch `dia_class` (chỉ cảnh báo, không loại) | **4** | `val_b_human_077` (`thanh`), `107` (`tunic`), `111` (`trunk`), `125` (`du`) |

---

## 2. Token ngoài Lexicon `parser_v1` (CHỈ BÁO CÁO)

- Top token ngoài lexicon `parser_v1` trên `131` câu:
  `di` (10), `mua` (9), `cho` (8), `ngay` (6), `tron` (5), `mac` (5), `minh` (3), `hang` (3), `nho` (3), `muon` (2), `can` (2), `dang` (2), `hoc` (2), `gong` (2), `gon` (2), `lich` (2), `to` (2), `bong` (2), `tai` (2), `trong` (2), `ban` (2), `cotton` (2), cùng các token đơn lẻ (`rong`, `diu`, `hoi`, `bat`, `noi`, `thanh`, `dip`, `vuong`, ...).

---

## 3. Phân bố Nhãn trên 65 Câu Hợp lệ (`val_queries_b_human.validated.json`)

- **Theo `dia_class`:** `full=44`, `none=17`, `partial=4`.
- **Theo `gender`:** `None=25`, `Women=20`, `Men=17`, `Girls=2`, `Boys=1`.
- **Theo `baseColour`:** `(empty)=34`, `Black=9`, `Blue=4`, `Purple=3`, `White=2`, `Grey=2`, `Silver=2`, `Brown=2`, `Red=2`, `Gold=2`, `Olive=1`, `Steel=1`, `Navy Blue=1`.
- **Theo `articleType` (35 loại):** `Casual Shoes` (9), `Tshirts` (5), `Watches` (5), `Shirts` (4), `Backpacks` (4), `Sunglasses` (4), `Jeans` (3), `Handbags` (2), `Heels` (2), `Earrings` (2), và 25 loại mỗi loại 1 câu (`Kurtas`, `Sports Shoes`, `Innerwear Vests`, `Kurtis`, `Caps`, `Nightdress`, `Tunics`, `Dupatta`, `Necklace and Chains`, `Trunk`, `Capris`, `Sweatshirts`, `Kurta Sets`, `Wallets`, `Belts`, `Briefs`, `Formal Shoes`, `Bracelet`, `Clutches`, `Duffel Bag`, `Sweaters`, `Tops`, `Sandals`, `Socks`, `Ties`).

---

## 4. Tóm tắt 66 Câu Bị loại

- **37 câu `n_relevant < 5` (nằm trong nhóm `val_b_human_006..072`):** `006, 007, 008, 009, 012, 015, 017, 018, 020, 021, 025, 026, 028, 030, 032, 033, 034, 035, 036, 037, 038, 042, 044, 045, 046, 047, 048, 052, 053, 054, 055, 059, 060, 062, 067, 068, 072`.
- **18 câu `Jaccard >= 0.8` trong nhóm `000..072`:** `010, 014, 019, 027, 029, 031, 039, 040, 041, 043, 049, 061, 063, 065, 066, 069, 070, 071`.
- **11 câu `Jaccard >= 0.8` trong nhóm bổ sung `073..130`:**
  - `val_b_human_082` (`áo ngực nữ màu trắng`, Jaccard `1.0`, `A:val_a_045`)
  - `val_b_human_083` (`áo kurta nữ màu nâu`, Jaccard `1.0`, `A:val_a_065`)
  - `val_b_human_085` (`áo thun bé trai màu do`, Jaccard `0.8571`, `A:val_a_202`)
  - `val_b_human_086` (`áo kurta nữ màu xanh lá`, Jaccard `0.8571`, `A:val_a_182`)
  - `val_b_human_092` (`áo kurta màu xanh dương`, Jaccard `0.8333`, `A:val_a_040`)
  - `val_b_human_093` (`áo sơ mi màu xanh lá`, Jaccard `0.8571`, `A:val_a_086`)
  - `val_b_human_094` (`ao so mi mau den`, Jaccard `0.8333`, `A:val_a_125`)
  - `val_b_human_095` (`áo kiểu màu trắng`, Jaccard `0.8`, `A:val_a_127`)
  - `val_b_human_097` (`túi xach màu xanh dương`, Jaccard `0.8333`, `A:val_a_060`)
  - `val_b_human_098` (`túi xách màu tím`, Jaccard `0.8`, `A:val_a_062`)
  - `val_b_human_100` (`áo khoác màu đen`, Jaccard `0.8`, `A:val_a_055`)
