# Báo cáo Đánh giá & Chọn Cấu hình CIR — Pha 5c (Prompt 13)

- **Rule SHA-256 (`selection_rule_cir_v1.json`):** `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067`
- **Tập tuning (`eval/cir_eval_val.json`):** `120` cặp (`80` đổi màu, `40` đổi giới; `SHA-256: c735c7742c5277cffc4113c68dd8ee59fd3475c742284c8fe9e098af0819062a`)
- **Kết quả JSON:** `reports/phase5c/step5c_cir_results.json` (`SHA-256: a4448834f69f65687e8e6c58b233594c67d9c2d465bdb24d2f91d1647b7136ab`)
- **Cấu hình CIR cuối (`winner_config_id`):** **`cir_v1_05`** (`mAP = 0.2902`, `ΔmAP vs cir_v1_01 = +0.2114`, 95% CI: `[+0.1615, +0.2684]`)
- **Nhóm ứng viên hòa trong 95% cluster bootstrap CI:** `['cir_v1_05', 'cir_v1_06', 'cir_v1_08']` (chọn `cir_v1_05` theo `simplicity_order`).

## 1. Bảng Kết quả Lưới 8 Cấu hình trên `CIR-eval val` (120 cặp: 60 có dấu + 60 không dấu)

| Config ID | Vai trò | R@1 | R@5 | R@10 | mAP | 95% CI (mAP) | ΔmAP vs `cir_v1_01` (95% CI) | mAP Màu (`n=80`) | mAP Giới (`n=40`) | mAP Có dấu (`n=60`) | mAP Không dấu (`n=60`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cir_v1_01` | baseline_combiner_vi | 0.0750 | 0.1917 | 0.3083 | **0.0910** | `[0.0667, 0.1222]` | `+0.0000 [+0.0000, +0.0000]` | 0.0902 | 0.0927 | 0.0803 | 0.1017 |
| `cir_v1_02` | c1_vi_to_en_only | 0.0333 | 0.0917 | 0.2167 | **0.0582** | `[0.0426, 0.0773]` | `-0.0335 [-0.0541, -0.0169]` | 0.0570 | 0.0608 | 0.0433 | 0.0731 |
| `cir_v1_03` | c2_text_fusion_b085 | 0.0417 | 0.1417 | 0.1833 | **0.0654** | `[0.0405, 0.0910]` | `-0.0277 [-0.0462, -0.0123]` | 0.0619 | 0.0723 | 0.0583 | 0.0724 |
| `cir_v1_04` | c2_text_fusion_b070 | 0.0000 | 0.0417 | 0.0833 | **0.0253** | `[0.0174, 0.0321]` | `-0.0679 [-0.0960, -0.0454]` | 0.0218 | 0.0324 | 0.0254 | 0.0253 |
| `cir_v1_05` | c3_mod_bonus_only | 0.2583 | 0.4833 | 0.6250 | **0.2902** | `[0.2441, 0.3663]` | `+0.2114 [+0.1615, +0.2684]` | 0.3634 | 0.1437 | 0.2747 | 0.3057 |
| `cir_v1_06` | c1_plus_c3 | 0.2250 | 0.4833 | 0.5750 | **0.2788** | `[0.2392, 0.3702]` | `+0.2095 [+0.1517, +0.2713]` | 0.3462 | 0.1440 | 0.2675 | 0.2901 |
| `cir_v1_07` | c1_c2_b085_c3 | 0.1167 | 0.3000 | 0.4000 | **0.1768** | `[0.1372, 0.2348]` | `+0.0896 [+0.0503, +0.1344]` | 0.2216 | 0.0873 | 0.1836 | 0.1701 |
| `cir_v1_08` | c1_c2_b085_c3_c4_rocchio | 0.2250 | 0.3917 | 0.4750 | **0.2664** | `[0.2224, 0.3604]` | `+0.1953 [+0.1358, +0.2612]` | 0.3367 | 0.1260 | 0.2732 | 0.2597 |

## 2. Chẩn đoán Toàn bộ 120 cặp Có dấu (`all_full`) vs 120 cặp Không dấu (`all_none`)

| Config ID | `all_full (n=120)` R@1 | `all_full` R@10 | `all_full` mAP | `all_none (n=120)` R@1 | `all_none` R@10 | `all_none` mAP |
|---|---:|---:|---:|---:|---:|---:|
| `cir_v1_01` | 0.0583 | 0.3083 | 0.0989 | 0.0583 | 0.3667 | 0.0961 |
| `cir_v1_02` | 0.0167 | 0.2083 | 0.0548 | 0.0333 | 0.1750 | 0.0541 |
| `cir_v1_03` | 0.0333 | 0.2000 | 0.0673 | 0.0417 | 0.1833 | 0.0642 |
| `cir_v1_04` | 0.0000 | 0.0833 | 0.0268 | 0.0000 | 0.0667 | 0.0246 |
| `cir_v1_05` | 0.2333 | 0.6083 | 0.3019 | 0.2667 | 0.6167 | 0.3003 |
| `cir_v1_06` | 0.2167 | 0.5667 | 0.2710 | 0.1917 | 0.5833 | 0.2657 |
| `cir_v1_07` | 0.1167 | 0.3833 | 0.1766 | 0.1083 | 0.3917 | 0.1716 |
| `cir_v1_08` | 0.2250 | 0.5083 | 0.2707 | 0.1833 | 0.4333 | 0.2463 |

## 3. Phân tích Thực nghiệm C1, C2, C3, C4 & Lý do Chọn `cir_v1_05`
1. **C1 (`cir_v1_02` — Dịch modifier tiếng Việt sang tiếng Anh bằng bảng thuộc tính catalog):**
   - `mAP` giảm từ `0.0910` (`cir_v1_01`) xuống `0.0582` (`ΔmAP = -0.0335`, 95% CI `[-0.0541, -0.0169]`).
   - **Nguyên nhân thực chứng:** Kiểm tra mã nguồn huấn luyện trong `notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb` (dòng 970–985, hàm `_synthesize_cir_modifier`) cho thấy lúc train, mẫu câu điều chỉnh có khung tiếng Việt (`"đổi sang {v}"`, `"thay bằng {v}"`, `"chuyển thành {v}"`) và nhánh Text Tower đã được fine-tune sâu trên tiếng Việt. Việc dịch toàn bộ cụm sang tiếng Anh (`"change to color black"`, `"style for women"`) làm vector văn bản lệch khỏi phân phối của Text Tower đã fine-tune. Кроме đó, với câu không dấu `"mau danh cho nam nu"`, từ `"mau"` (mẫu) bị khôi phục dấu thành `"màu"` (`color`) trước khi dịch sang tiếng Anh.
2. **C2 (`cir_v1_03` với `β=0.85`, `cir_v1_04` với `β=0.70` — Hợp nhất `cos(combiner)` và `cos(text_query_attr, img)`):**
   - `mAP` giảm xuống `0.0654` (`β=0.85`) và `0.0253` (`β=0.70`).
   - **Nguyên nhân:** Phần modifier chỉ nêu thuộc tính thay đổi (1 slot: `baseColour` hoặc `gender`, không chứa `articleType`). Do đó vector `text_query_attr` kéo mọi sản phẩm cùng màu/giới thuộc các ngành hàng khác (`Shoes`, `Watches`, `Bags`...) lên cao, làm loãng tín hiệu bảo toàn `articleType` từ ảnh tham chiếu của `combiner`.
3. **C3 (`cir_v1_05` — Bonus thuộc tính `tier_soft γ=1.0` từ `parser_v2.1` trên phần modifier đổi màu/giới, không gating):**
   - **Cải thiện vượt bậc:** `mAP` tăng từ **`0.0910`** lên **`0.2902`** (`ΔmAP = +0.2114`, 95% CI `[+0.1615, +0.2684]`, gấp **3,19×** baseline); `Recall@1` tăng từ **`0.0750`** lên **`0.2583`** (gấp **3,44×**); `Recall@5` từ **`0.1917`** lên **`0.4833`**; `Recall@10` từ **`0.3083`** lên **`0.6250`** (gấp **2,03×**).
   - **Cơ chế:** `C3` phân tầng cứng-mềm các sản phẩm khớp đúng thuộc tính mục tiêu (`baseColour` hoặc `gender` trích từ modifier), trong khi bên trong bậc khớp (`mf = 1.0`), điểm `cos(combiner, img)` giữ vai trò sắp xếp và bảo toàn chính xác `articleType` + kiểu dáng thị giác từ ảnh tham chiếu.
   - **Theo loại modifier:** Đổi màu (`n=80`) tăng `mAP` từ `0.0902` lên **`0.3634`**; Đổi giới (`n=40`) tăng từ `0.0927` lên **`0.1437`**.
   - **Theo có/không dấu:** Có dấu (`n=60`) đạt `mAP = 0.2747` (vs `0.0803`); Không dấu (`n=60`) đạt `mAP = 0.3057` (vs `0.1017`). Trên chẩn đoán toàn bộ 120 cặp (`all_full` vs `all_none`), `cir_v1_05` đạt `0.3019` (có dấu) và `0.3003` (không dấu), chứng tỏ bước khôi phục dấu xử lý tốt truy vấn không dấu cho CIR.
4. **C4 (`cir_v1_08` vs `cir_v1_07` — PRF Rocchio nhẹ `k=3, β_prf=0.15`):**
   - So với `cir_v1_07`, `C4` đạt `ΔmAP = +0.1056` (95% CI `[+0.0683, +0.1462]`, loại trừ `0`), nhưng tổng thể `cir_v1_08` (`mAP = 0.2664`) vẫn thấp hơn `cir_v1_05` (`mAP = 0.2902`) và nằm trong nhóm hòa 95% CI `['cir_v1_05', 'cir_v1_06', 'cir_v1_08']`. Theo `simplicity_order`, **`cir_v1_05`** (chỉ bật `C3`, đạt `mAP` cao nhất và đơn giản nhất) được chọn làm **cấu hình CIR cuối cùng**.

## 4. Độ trễ C.9 cho Luồng `CIR catalog` và `CIR upload` (5 run × 200 truy vấn, warmup 20)

| Luồng | Config ID | p50 (ms) | p95 median ± IQR (ms) | p99 (ms) | Đạt `p95 < 200 ms` | Cảnh báo `p99 upload > 200 ms` |
|---|---|---:|---:|---:|:---:|:---:|
| `cir_catalog` | `cir_v1_01` | 72.86 | **121.42 ± 9.34** | 215.70 | ✅ | Không |
| `cir_catalog` | `cir_v1_05` | 77.56 | **135.54 ± 16.17** | 215.14 | ✅ | Không |
| `cir_upload` | `cir_v1_01` | 267.19 | **415.29 ± 22.77** | 505.96 | ❌ | ⚠️ CÓ (`> 200 ms`) |
| `cir_upload` | `cir_v1_05` | 270.38 | **446.27 ± 21.82** | 535.76 | ❌ | ⚠️ CÓ (`> 200 ms`) |

- **Nhận xét trung thực về độ trễ (Luật C.10):**
  - Luồng **`cir_catalog`** của cả baseline (`121.42 ms`) và `cir_v1_05` (`135.54 ms`) đều **ĐẠT SLA `p95 < 200 ms`** (phần cộng điểm `C3` chỉ tăng `~4.7 ms` ở `p50` và `~14.1 ms` ở `p95`).
  - Luồng **`cir_upload`** vượt ngưỡng `200 ms` ở cả baseline (`p95 = 415.29 ms`, `p99 = 505.96 ms`) và `cir_v1_05` (`p95 = 446.27 ms`, `p99 = 535.76 ms`). Nguyên nhân: (1) `cir_upload` phải đọc ảnh từ đĩa + `Resize(224,224)` + chạy cả Vision Tower (`encode_image`) lẫn Text Tower (`encode_text`) + `cir_combiner`; (2) Trong suốt lúc đo, GPU RTX 2050 bị các tiến trình đồ họa nền ngoài dự án (`RobloxPlayerBeta.exe`, `dnmultiplayerex.exe`) chiếm ~27–32% tải ở nhiệt độ `85–86°C` (ghìm xung nhịp xuống `900–1425 MHz`). Khi đo lại trên GPU nguội hoàn toàn ở Prompt 14 (không có tiến trình game nền), độ trễ `encode_image + encode_text` sẽ giảm mạnh.