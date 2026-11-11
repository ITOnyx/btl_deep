# Pha 7a — GĐ4: Ánh xạ Metadata Shopping100k, Tách Lớp & Kiểm Chứng Màu (Chờ duyệt)

- **Thời gian chạy:** 54.7s | **Seed:** `42` | **Đầu vào:** `clean_candidates_gd3.parquet` (**98,289** ảnh = `98,174 shopping100k` + `115 fashion`)
- **Output chính:** `eval/phase7/candidates_mapped_gd4.parquet` (98,289 dòng), `eval/phase7/s100k_mapping.csv` (98,174 dòng), `eval/phase7/s100k_raw_attribute_stats.csv` (12 thuộc tính đầy đủ).
- **Kiểm tra `gender` (`Female/Male` folder vs `gender.txt`):** lệch = **0**.

---

## 1. Thống kê giá trị thô Shopping100k (98.174 ảnh sạch — tóm tắt trường chính, chi tiết 12 thuộc tính trong `s100k_raw_attribute_stats.csv`)

| Thuộc tính | Giá trị khác `-1` | Số ảnh `-1` (thiếu) | Phân bố chi tiết (`nhãn: số_ảnh`) |
|---|---:|---:|---|
| `category` | 98,174 | 0 | `T-shirt`:21,631, `Shirt`:14,546, `Dress`:11,709, `Jumper`:11,556, `Jacket`:7,343, `Jean`:6,087, `Swimming`:5,283, `Shorts`:5,030, ... (+8 giá trị) |
| `gender` | 98,174 | 0 | `Female`:62,489, `Male`:35,685 |
| `color` | 80,289 | 17,885 | `Black`:21,315, `Navy`:12,521, `Blue`:10,818, `White`:9,690, `Gray`:7,305, `Red`:2,981, `Beige`:2,966, `Green`:2,343, ... (+11 giá trị) |
| `collar` | 24,191 | 73,983 | `Mandarin`:6,100, `Hood`:5,053, `Polo`:2,728, `Lapel`:2,170, `Turndown`:1,906, `Kent`:1,783, `Shirt`:1,268, `Buttondown`:1,211, ... (+9 giá trị) |
| `fabric` | 51,267 | 46,907 | `Jersey`:29,202, `Sweat`:8,887, `Denim`:7,446, `Rib`:1,167, `Lace`:1,101, `Mesh Jersey`:898, `Mesh`:662, `Hardshell`:444, ... (+6 giá trị) |
| `fastening` | 39,421 | 58,753 | `Zip`:12,462, `Covered`:12,085, `Button`:11,851, `Back closure`:1,606, `No closure`:448, `Hook`:260, `Laces`:227, `Belt`:221, ... (+2 giá trị) |
| `neckline` | 45,023 | 53,151 | `Square`:28,861, `Low round`:5,144, `Off-the-shoulder`:2,675, `Backless`:1,958, `Envelope`:1,637, `Round`:1,480, `Cache-coeur`:1,196, `Low v-neck`:964, ... (+3 giá trị) |
| `sleevelen` | 69,577 | 28,597 | `Long`:27,010, `Short`:18,715, `Sleeveless`:8,237, `3/4`:4,153, `Extra Short`:3,527, `Extra Long`:3,147, `Elbow`:2,463, `Spaghetti`:2,061, ... (+1 giá trị) |
| `fit` | 89,040 | 9,134 | `Regular`:51,080, `Tailered`:12,512, `Large`:6,496, `Slim`:5,777, `Small`:4,026, `Straight`:2,107, `Skinny`:2,051, `Tapered`:1,765, ... (+7 giá trị) |
| `pattern` | 81,375 | 16,799 | `Plain`:39,996, `Print`:16,500, `Marl`:9,492, `Striped`:5,749, `Floral`:4,798, `Checked`:1,278, `Polka dot`:1,070, `Photo`:562, ... (+8 giá trị) |
| `pocket` | 22,857 | 75,317 | `Side`:15,602, `Zip`:3,644, `Flap`:2,144, `Sleeve`:712, `Cargo`:660, `Napoleon`:58, `Belt`:37 |
| `sport` | 12,298 | 85,876 | `Training`:4,410, `Running`:1,982, `Hiking`:1,806, `Football`:1,048, `Tennis`:709, `Climbing`:515, `Golf`:412, `Yoga`:377, ... (+7 giá trị) |

---

## 2. Quy tắc tách nhóm `shirts_tops`, `6_Jumpers`, `10_Swimming` & kiểm chứng màu pixel

### 2a. Phân bố `attr_confidence` trên 98.174 Shopping100k & toàn tập 98.289 ảnh

| Tập | `high` (`AttributeIndex+dense`) | `medium` (`dense_only`) | `low` (`dense_only`) | Tổng |
|---|---:|---:|---:|---:|
| `shopping100k` | **33,101** (33.72%) | **15,389** (15.68%) | **49,684** (50.61%) | **98,174** |
| `fashion` (serving_eligible) | **102** | **3** | **0** | **105** (+10 eval_only) |

### 2b. Kết quả kiểm chứng chéo màu (`color_label` vs `color_pixel` qua `eval/families.json`)

- Có nhãn màu gốc (`color >= 0`, `color_source='attributes'`): **80,289** ảnh:
  - Khớp họ màu (`color_conf='high'`): **54,976** (68.47%)
  - Lệch họ màu (`color_conf='low'`): **25,313** (31.53%) — giữ nguyên `baseColour=color_label`, ghi `color_pixel` để kiểm chứng
- Thiếu nhãn gốc (`color == -1`, `color_source='image_estimated'`): **17,885** ảnh — giữ `baseColour='Unknown'`, `color_conf='low'`, ghi `color_pixel`, KHÔNG đưa vào `AttributeIndex`.

### 2c. Bảng phân rã theo `articleType` (`shopping100k`, 98.174 ảnh)

| `articleType` | `high` | `medium` | `low` | Tổng | `color_conf=high` | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| `Bath Robe` | 0 | 82 | 82 | **164** | 82 | Chờ từ khóa lexicon_vi |
| `Coats` | 0 | 0 | 1,529 | **1,529** | 806 | Đề xuất dense-only (ngoài schema / mơ hồ) |
| `Dresses` | 6,522 | 0 | 5,187 | **11,709** | 6,522 | Đã kiểm định 4 bộ từ điển |
| `Jackets` | 4,309 | 0 | 3,034 | **7,343** | 4,309 | Đã kiểm định 4 bộ từ điển |
| `Jeans` | 3,455 | 0 | 2,632 | **6,087** | 3,455 | Đã kiểm định 4 bộ từ điển |
| `Jumpsuit` | 0 | 791 | 449 | **1,240** | 791 | Chờ từ khóa lexicon_vi |
| `Night suits` | 522 | 0 | 541 | **1,063** | 522 | Đã kiểm định 4 bộ từ điển |
| `Shirts` | 3,122 | 132 | 1,894 | **5,148** | 3,254 | Đã kiểm định 4 bộ từ điển |
| `Shorts` | 2,999 | 0 | 2,031 | **5,030** | 2,999 | Đã kiểm định 4 bộ từ điển |
| `Skirts` | 1,403 | 0 | 1,060 | **2,463** | 1,403 | Đã kiểm định 4 bộ từ điển |
| `Suits` | 0 | 222 | 105 | **327** | 222 | Chờ từ khóa lexicon_vi |
| `Sweaters` | 0 | 3,207 | 3,805 | **7,012** | 3,517 | Đã kiểm định 4 bộ từ điển |
| `Sweatshirts` | 1,473 | 1,026 | 2,113 | **4,612** | 2,499 | Đã kiểm định 4 bộ từ điển |
| `Swimwear` | 0 | 2,585 | 2,534 | **5,119** | 2,585 | Chờ từ khóa lexicon_vi |
| `Tops` | 1,587 | 1,607 | 2,571 | **5,765** | 3,194 | Đã kiểm định 4 bộ từ điển |
| `Tops/Shirts-ambiguous` | 0 | 0 | 10,127 | **10,127** | 5,370 | Đề xuất dense-only (ngoài schema / mơ hồ) |
| `Track Pants` | 0 | 2,389 | 1,245 | **3,634** | 2,389 | Đã kiểm định 4 bộ từ điển |
| `Tracksuits` | 0 | 158 | 55 | **213** | 158 | Chờ từ khóa lexicon_vi |
| `Trousers` | 2,487 | 0 | 2,033 | **4,520** | 2,487 | Đã kiểm định 4 bộ từ điển |
| `Tshirts` | 5,222 | 3,190 | 6,657 | **15,069** | 8,412 | Đã kiểm định 4 bộ từ điển |

---

## 3. Tách quy tắc chi tiết `shirts_tops`, `6_Jumpers`, `10_Swimming` (`s100k_mapping_rules.csv`)

| `rule_id` | Nhóm ánh xạ | `cat_conf` | Số ảnh | Cơ sở thuộc tính |
|---|---|---|---:|---|
| `Coat_out_of_schema` | `Coats` | **low** | 1,529 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Dress_1to1` | `Dresses` | **high** | 11,709 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F4_ambig` | `Tops/Shirts-ambiguous` | **low** | 4,809 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F4_feminine_top_cue` | `Tops` | **medium** | 1,280 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F4_polo` | `Tshirts` | **high** | 324 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F4_shirt_btn` | `Shirts` | **high** | 1,337 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F4_shirt_nobtn` | `Shirts` | **medium** | 119 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_ambig` | `Tops/Shirts-ambiguous` | **low** | 5,122 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_classic_short_tee` | `Tshirts` | **medium** | 1,545 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_hoodie` | `Sweatshirts` | **medium** | 51 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_other_jersey_tee` | `Tshirts` | **medium** | 896 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_polo` | `Tshirts` | **medium** | 47 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_sleeveless_top` | `Tops` | **medium** | 1,644 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_strong_top` | `Tops` | **high** | 2,841 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `F5_sweat_short_tee` | `Tshirts` | **medium** | 445 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `J_hoodie` | `Sweatshirts` | **high** | 2,636 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `J_sweat` | `Sweatshirts` | **medium** | 1,908 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `J_sweater` | `Sweaters` | **medium** | 6,506 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `J_sweater_zip` | `Sweaters` | **low** | 506 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Jacket_1to1` | `Jackets` | **high** | 7,343 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Jean_1to1` | `Jeans` | **high** | 6,087 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Jumpsuit_no_lex` | `Jumpsuit` | **medium** | 1,240 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M4_ambig` | `Tops/Shirts-ambiguous` | **low** | 196 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M4_jersey_tee` | `Tshirts` | **medium** | 741 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M4_mandarin_nobtn` | `Shirts` | **medium** | 95 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M4_polo` | `Tshirts` | **high** | 2,048 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M4_shirt_or_btn` | `Shirts` | **high** | 3,597 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M5_classic_tee` | `Tshirts` | **high** | 6,993 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M5_hoodie` | `Sweatshirts` | **medium** | 17 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M5_other_tee` | `Tshirts` | **medium** | 1,997 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `M5_polo` | `Tshirts` | **medium** | 33 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Pyjamas_1to1` | `Night suits` | **high** | 1,063 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Shorts_1to1` | `Shorts` | **high** | 5,030 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Skirt_1to1` | `Skirts` | **high** | 2,463 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Suit_partial_schema` | `Suits` | **medium** | 327 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Swim_bathrobe` | `Bath Robe` | **medium** | 164 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Swim_swimwear` | `Swimwear` | **medium** | 5,119 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `TracksuitBottoms_pending_lex` | `Track Pants` | **medium** | 3,634 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Tracksuit_no_lex` | `Tracksuits` | **medium** | 213 | Xem chi tiết trong `s100k_mapping_rules.csv` |
| `Trouser_1to1` | `Trousers` | **high** | 4,520 | Xem chi tiết trong `s100k_mapping_rules.csv` |

---

## 4. Cờ phạm vi, Đánh giá & BM25 (tất cả 98.289 ảnh)

- `serving_eligible = True`: **98,279** (`98,174 shopping100k` + `105 fashion`); `False`: **10** (`6` fashion Personal Care ngoài scope + `4` fashion trùng catalog pHash<=1).
- `eval_ext_eligible = True` (`seen_in_train == 'no'`): **1,562** (`fashion=94`, `shopping100k=1,468`).
- `bm25_eligible = True` (chỉ sinh BM25 từ `title_vi` khi đủ độ tin cậy): `shopping100k=33,101`, `fashion=102`.

---

## 5. Đối chiếu `(articleType × gender)` phần `high` với `index_v2` (38,644 ảnh, trừ 2 Keychain; 125 slot <30)

- Số slot hiếm `<30` trong `index_v2` được phần `attr_confidence='high'` lấp thêm: **Shopping100k = 0**, **Tổng (gồm Fashion) = 2**.

| Slot `(articleType, gender)` | `index_v2` | Thêm (`high`) | Tổng sau bổ sung |
|---|---:|---:|---:|
| `(Capris, Girls)` | 18 | +1 | **19** |
| `(Sandals, Girls)` | 29 | +2 | **31** |

- **Slot hiếm `<30` tiềm năng nếu bổ sung từ khóa vào `lexicon_vi.json` (`attr_confidence` hiện giữ `medium` do thiếu từ khóa parser, nhưng `color_conf='high'`):**

| Slot `(articleType, gender)` | `index_v2` | Ảnh Shopping100k (`medium` + màu `high`) | Tổng nếu mở khóa `lexicon_vi` |
|---|---:|---:|---:|
| `(Tracksuits, Women)` | 1 | +49 | **50** |
| `(Swimwear, Women)` | 12 | +1,828 | **1,840** |
| `(Jumpsuit, Women)` | 16 | +787 | **803** |
| `(Bath Robe, Women)` | 20 | +65 | **85** |
| `(Tracksuits, Men)` | 28 | +109 | **137** |

### Bảng `(articleType × gender)` phần `high` Shopping100k đối chiếu `index_v2`

| `(articleType, gender)` | `index_v2` | `shopping100k (high)` | `fashion (high)` | Tổng sau gộp nếu chọn |
|---|---:|---:|---:|---:|
| `(Dresses, Women)` | 372 | +6,522 | +0 | **6,894** |
| `(Tshirts, Men)` | 5,089 | +5,027 | +7 | **10,123** |
| `(Shirts, Men)` | 2,597 | +2,306 | +0 | **4,903** |
| `(Jackets, Women)` | 55 | +2,196 | +0 | **2,251** |
| `(Jeans, Women)` | 163 | +2,172 | +1 | **2,336** |
| `(Jackets, Men)` | 191 | +2,113 | +1 | **2,305** |
| `(Shorts, Men)` | 340 | +1,723 | +2 | **2,065** |
| `(Tops, Women)` | 1,492 | +1,587 | +0 | **3,079** |
| `(Trousers, Women)` | 70 | +1,522 | +1 | **1,593** |
| `(Skirts, Women)` | 90 | +1,403 | +0 | **1,493** |
| `(Jeans, Men)` | 261 | +1,283 | +2 | **1,546** |
| `(Shorts, Women)` | 56 | +1,276 | +0 | **1,332** |
| `(Sweatshirts, Men)` | 218 | +1,035 | +0 | **1,253** |
| `(Trousers, Men)` | 288 | +965 | +5 | **1,258** |
| `(Shirts, Women)` | 282 | +816 | +2 | **1,100** |
| `(Sweatshirts, Women)` | 58 | +438 | +0 | **496** |
| `(Night suits, Women)` | 77 | +292 | +0 | **369** |
| `(Night suits, Men)` | 62 | +230 | +0 | **292** |
| `(Tshirts, Women)` | 1,083 | +195 | +2 | **1,280** |
| `(Accessory Gift Set, Men)` | 92 | +0 | +14 | **106** |
| `(Sports Shoes, Men)` | 1,430 | +0 | +7 | **1,437** |
| `(Heels, Women)` | 1,268 | +0 | +4 | **1,272** |
| `(Casual Shoes, Unisex)` | 207 | +0 | +3 | **210** |
| `(Track Pants, Men)` | 198 | +0 | +3 | **201** |
| `(Casual Shoes, Men)` | 2,017 | +0 | +3 | **2,020** |
| `(Sunglasses, Women)` | 232 | +0 | +3 | **235** |
| `(Casual Shoes, Women)` | 288 | +0 | +3 | **291** |
| `(Sandals, Unisex)` | 70 | +0 | +3 | **73** |
| `(Leggings, Women)` | 98 | +0 | +3 | **101** |
| `(Earrings, Women)` | 376 | +0 | +3 | **379** |
| `(Necklace and Chains, Women)` | 95 | +0 | +2 | **97** |
| `(Flip Flops, Women)` | 217 | +0 | +2 | **219** |
| `(Sandals, Girls)` | 29 | +0 | +2 | **31** |
| `(Sunglasses, Unisex)` | 97 | +0 | +2 | **99** |
| `(Sunglasses, Men)` | 478 | +0 | +2 | **480** |
| `(Tops, Girls)` | 132 | +0 | +2 | **134** |
| `(Ties, Men)` | 216 | +0 | +2 | **218** |
| `(Flip Flops, Unisex)` | 126 | +0 | +2 | **128** |
| `(Innerwear Vests, Men)` | 227 | +0 | +1 | **228** |
| `(Sandals, Men)` | 697 | +0 | +1 | **698** |
| `(Tshirts, Unisex)` | 64 | +0 | +1 | **65** |
| `(Tshirts, Boys)` | 363 | +0 | +1 | **364** |
| `(Wallets, Men)` | 410 | +0 | +1 | **411** |
| `(Kurtas, Women)` | 1,754 | +0 | +1 | **1,755** |
| `(Sports Shoes, Women)` | 333 | +0 | +1 | **334** |
| `(Belts, Men)` | 524 | +0 | +1 | **525** |
| `(Capris, Girls)` | 18 | +0 | +1 | **19** |
| `(Flip Flops, Men)` | 439 | +0 | +1 | **440** |
| `(Ring, Women)` | 87 | +0 | +1 | **88** |
| `(Jewellery Set, Women)` | 52 | +0 | +1 | **53** |
| `(Sports Shoes, Unisex)` | 30 | +0 | +1 | **31** |
| `(Bracelet, Women)` | 46 | +0 | +1 | **47** |

---

## 6. Contact Sheets kiểm chứng bằng mắt (40 ảnh/sheet, `seed=42`) — [DỪNG] chờ duyệt

Tại `reports/phase7/contact_sheets/`:
- `gd4_sample_200_part1.png`
- `gd4_sample_200_part2.png`
- `gd4_sample_200_part3.png`
- `gd4_sample_200_part4.png`
- `gd4_sample_200_part5.png`
- `gd4_shirts_tops_40.png`
- `gd4_color_mismatch_40.png`
- `gd4_color_missing_40.png`
