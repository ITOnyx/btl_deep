# Báo cáo Chọn Cấu hình Bước 3 — Pha 5b (`selection_rule_v3`)

- **Nhánh:** `phase7a` | **Seed:** `42` | **Commit gốc:** `b577f56b95f5dc9b6f91c747eaa7480f5cc784e7`
- **Parser:** `parser_v2.1` (`SHA-256`: `1262383c151ba020dfcfcadc6292a974bc8eb1b2ec0de3fb1698b3430cb4cc5a`)
- **Folds CV (5-fold theo `spec_cluster_id`):** [`reports/phase5b/folds.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/folds.json) (`SHA-256`: `dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25`)
- **Quy tắc:** [`reports/phase5b/selection_rule_v3.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/selection_rule_v3.json) (`SHA-256`: `fa4e0df3469b0f83a458c38a466efaec3ed3b4d951ee52b80c36dc4c783de07e`)
- **Hash kết quả khóa TRƯỚC khi chạy B-human:** `19948b5abf11ffb711704d23d62a76e7214292d9ea41a30f12f4a4bfe8456939`
- **Phạm vi:** `index/val_gallery/` (`1.884` sản phẩm), `exact search`, `tie_rule="conservative"` ($\epsilon = 10^{-5}$), `zscore_cosine=true`. **KHÔNG** chạy `test_gallery`, **KHÔNG** tune trên DEV.

---

## 1. Tóm tắt Quyết định Chọn Cấu hình

| Vai trò | Config | Mô tả | CV Score | A mAP(cl) | Bc mAP(cl) | Bt mAP(cl) | C20 mAP(cl) | p95 (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Thắng cuộc (sau phá hòa CI)** | **`cfg_13`** | `parser_v2.1`, $\gamma=1.5$ đều, `query_rewrite=true` | **`0.5053`** | `0.7546` | `0.3391` | `0.2522` | `0.4966` | `54.0` |
| **Tốt nhất theo Score thô** | `cfg_19` | `parser_v2.1`, $\gamma=1.5$ đều, `bm25=true` | `0.5391` | `0.7162` | `0.4209` | `0.3213` | `0.5362` | `55.7` |
| **An toàn ($\gamma$ thấp nhất đạt 5(i)(ii))** | `cfg_01` | `parser_v1`, $\gamma=0.3$ đều (baseline v3) | `0.2286` | `0.3660` | `0.1372` | `0.1145` | `0.2454` | `35.9` |
| **Ablation bắt buộc** | **`cfg_02`** | `parser_v2.1`, $\gamma=0.3$ đều | `0.2297` | `0.3677` | `0.1378` | `0.1149` | `0.2454` | `41.5` |

- **Phá hòa theo 95% paired cluster bootstrap CI (1.000 lần):** chênh lệch Score giữa `cfg_19` (`0.5391`) và `cfg_13` (`0.5053`) là `+0.0337` với CI 95% **`[-0.0003, +0.0638]`** (chứa `0.0`). Theo `simplicity_order` (`bm25=false < bm25=true`), chọn **`cfg_13`**.
- **Cảnh báo biên lưới (PROMPT 8 E.3):** Cả `cfg_19` và `cfg_13` đều nằm tại biên trên của lưới (**$\gamma = 1.5$**) và CV Score còn tăng đều theo $\gamma$ ($0.3 \to 0.5 \to 0.8 \to 1.0 \to 1.5$). Do đó **"lưới có thể chưa phủ vùng tối ưu"**; đề xuất khóa `selection_rule_v4` nếu muốn khảo sát $\gamma > 1.5$ (không tự mở rộng lưới ngoài quy tắc).

---

## 2. Tham chiếu `model_only` và `baseline Pha 5 (KHÔNG z-score)` & Hiệu ứng Z-score

| Cấu hình tham chiếu | Val A mAP(cl) / R@10 | B-synth clean mAP(cl) / R@10 | B-human mAP(cl) / R@10 |
|---|---:|---:|---:|
| **`model_only`** | `0.3058` (query) / `0.6042` | `0.1071` (query) / `0.2485` | `0.2680` / — |
| **`baseline Pha 5 (KHÔNG z-score, v1, γ=0.3)`** | **`0.5896`** / `0.8167` | **`0.3171`** / `0.6364` | **`0.5335`** / `0.8154` |
| **`cfg_01` (giao thức v3, `zscore=true`, `v1, γ=0.3`)** | `0.3660` / `0.6750` | `0.1372` / `0.3091` | `0.3249` / — |

- **Giải thích hiệu ứng z-score (PROMPT 8 E.2):** Độ lệch chuẩn cosine trung bình theo truy vấn trên Val A là **`0.1622`**. Khi KHÔNG z-score, $\gamma = 0.3$ tương đương $0.3 / 0.1622 \approx 1.85\,\sigma$. Khi bật `zscore_cosine=true`, cosine được chia cho $\sigma \approx 0.1622$, nên $\gamma = 0.3$ chỉ còn tương đương $0.3\,\sigma$ (yếu hơn ~`6.2x`), làm `cfg_01` tụt mạnh trên DEV (`0.3749` vs `0.5852`) và Val A (`0.3660` vs `0.5896`). Ngược lại, ở $\gamma = 1.5$ trên thang z-score ($1.5\,\sigma \approx 0.243$ theo thang cosine gốc), cường độ bonus mới tiệm cận $\gamma = 0.25$–$0.30$ trước đây.

---

## 3. Bảng Đầy đủ 22 Cấu hình (`cfg_01..cfg_22`)

*(Đo độ trễ T2I p95 theo AGENTS.md C.9: 5 run × 200 truy vấn, warmup 20, `torch.cuda.synchronize()`, GPU RTX 2050 nhiệt độ 68–86°C, xung nhịp 1477–1732 MHz. Đã đo cho `21/22` cấu hình — chỉ `cfg_20` không đo do fail ràng buộc `5(iii)` `ΔR@10 = -0.0144 < -0.01`.)*

| Config | Parser | γ | Slot W | Flags | CV Score | A mAP(cl) | Bc mAP(cl) | Bt mAP(cl) | C20 mAP(cl) | ΔR@10(AB) | CatDrop(B) | p95(ms) | Đạt 5(i)-(v) |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cfg_01 | parser_v1 | 0.3 | unif | — | 0.2286 | 0.3660 | 0.1372 | 0.1145 | 0.2454 | +0.0000 | 0.0000 | 35.9 | PASS |
| cfg_02 | parser_v2 | 0.3 | unif | — | 0.2297 | 0.3677 | 0.1378 | 0.1149 | 0.2454 | +0.0021 | 0.0000 | 41.5 | PASS |
| cfg_03 | parser_v2 | 0.5 | unif | — | 0.2552 | 0.4009 | 0.1582 | 0.1271 | 0.2703 | +0.0474 | 0.0000 | 38.4 | PASS |
| cfg_04 | parser_v2 | 0.8 | unif | — | 0.2960 | 0.4523 | 0.1919 | 0.1480 | 0.3086 | +0.0907 | 0.0000 | 45.2 | PASS |
| cfg_05 | parser_v2 | 1.0 | unif | — | 0.3229 | 0.4816 | 0.2172 | 0.1640 | 0.3325 | +0.1299 | 0.0000 | 38.2 | PASS |
| cfg_06 | parser_v2 | 1.5 | unif | — | 0.3926 | 0.5502 | 0.2876 | 0.2045 | 0.3945 | +0.1959 | 0.0000 | 43.7 | PASS |
| cfg_07 | parser_v2 | 0.5 | 0.6/0.2/0.2 | — | 0.2567 | 0.4025 | 0.1597 | 0.1293 | 0.2710 | +0.0536 | 0.0000 | 39.3 | PASS |
| cfg_08 | parser_v2 | 0.8 | 0.6/0.2/0.2 | — | 0.2984 | 0.4498 | 0.1975 | 0.1491 | 0.3071 | +0.1031 | 0.0041 | 48.4 | PASS |
| cfg_09 | parser_v2 | 1.0 | 0.6/0.2/0.2 | — | 0.3275 | 0.4794 | 0.2262 | 0.1638 | 0.3300 | +0.1505 | 0.0041 | 41.5 | PASS |
| cfg_10 | parser_v2 | 1.5 | 0.6/0.2/0.2 | — | 0.4054 | 0.5593 | 0.3026 | 0.2047 | 0.3958 | +0.2392 | 0.0000 | 46.5 | PASS |
| cfg_11 | parser_v2 | 0.5 | unif | RW | 0.3660 | 0.6147 | 0.2000 | 0.1708 | 0.3633 | +0.1959 | 0.0082 | 42.4 | PASS |
| cfg_12 | parser_v2 | 1.0 | unif | RW | 0.4378 | 0.6962 | 0.2654 | 0.2099 | 0.4317 | +0.2639 | 0.0082 | 48.8 | PASS |
| cfg_13 | parser_v2 | 1.5 | unif | RW | 0.5053 | 0.7546 | 0.3391 | 0.2522 | 0.4966 | +0.3134 | 0.0041 | 54.0 | PASS |
| cfg_14 | parser_v2 | 0.5 | unif | DE | 0.3431 | 0.5841 | 0.1824 | 0.1436 | 0.3410 | +0.1815 | 0.0041 | 100.6 | FAIL(5iv) |
| cfg_15 | parser_v2 | 1.0 | unif | DE | 0.4226 | 0.6862 | 0.2467 | 0.1835 | 0.4141 | +0.2454 | 0.0041 | 94.4 | FAIL(5iv) |
| cfg_16 | parser_v2 | 1.5 | unif | DE | 0.4949 | 0.7587 | 0.3190 | 0.2256 | 0.4831 | +0.3072 | 0.0041 | 109.0 | FAIL(5iv) |
| cfg_17 | parser_v2 | 0.5 | unif | BM25 | 0.3997 | 0.5968 | 0.2683 | 0.2093 | 0.4026 | +0.2289 | 0.0000 | 50.8 | PASS |
| cfg_18 | parser_v2 | 1.0 | unif | BM25 | 0.4701 | 0.6605 | 0.3431 | 0.2649 | 0.4724 | +0.2887 | 0.0000 | 57.3 | FAIL(5iv) |
| cfg_19 | parser_v2 | 1.5 | unif | BM25 | 0.5391 | 0.7162 | 0.4209 | 0.3213 | 0.5362 | +0.3361 | 0.0000 | 55.7 | PASS |
| cfg_20 | parser_v2 | 0.5 | unif | MC | 0.1896 | 0.2742 | 0.1331 | 0.1159 | 0.2071 | -0.0144 | 0.0327 | — | FAIL |
| cfg_21 | parser_v2 | 1.0 | unif | MC | 0.2495 | 0.3412 | 0.1882 | 0.1452 | 0.2651 | +0.0743 | 0.0122 | 49.8 | PASS |
| cfg_22 | parser_v2 | 1.5 | unif | MC | 0.3148 | 0.4077 | 0.2526 | 0.1792 | 0.3304 | +0.1464 | 0.0082 | 48.7 | PASS |

- **Ablation `cfg_02` vs `cfg_01`:** cùng `γ=0.3` và `zscore=true`, đổi `parser_v1` → `parser_v2.1` giúp CV Score nhích `+0.0011` (`0.2286` → `0.2297`), cho thấy phần lớn cải thiện của `cfg_13` đến từ `query_rewrite=true` + tăng $\gamma$ lên `1.5` (khớp thang z-score), kết hợp giảm FP của `parser_v2.1`.

---

## 4. Cổng Xác nhận `B-human validated` (`n=65`, `61` cụm — chạy SAU khi khóa `cfg_13`)

- **`cfg_13`:** cluster mAP = **`0.6285`** (`R@1=0.7538`, `R@5=0.8308`, `R@10=0.8615`, query mAP=`0.6314`).
- **`cfg_01` (baseline v3):** cluster mAP = **`0.3249`**.
- **`ΔmAP (cfg_13 - cfg_01)`:** **`+0.3036`**, 95% cluster bootstrap CI **`[+0.2390, +0.3744]`** (cận dưới $\ge -0.02$).
- **Tỉ lệ tụt thảm họa trên B-human:** **`0.0308`** (`3.08% <= 5%`) → **Đạt cổng B-human (`gate_pass=true`)**.
- **Tách nhóm:**
  - Có màu (`31` câu): `cfg_13` = **`0.6843`** vs `cfg_01` = `0.3789`.
  - Không màu (`34` câu): `cfg_13` = **`0.5810`** vs `cfg_01` = `0.2790`.
- **Nhãn màu hiếm:**
  - `Steel (val_b_human_088)`: AP `0.8379` (vs `cfg_01`: `0.1860`).
  - `Navy Blue (val_b_human_091)`: AP `0.8392` (vs `cfg_01`: `0.6653`).
  - `Olive (val_b_human_079)`: AP `0.3166` (vs `cfg_01`: `0.4427` — hổng trung thực do `"xanh rêu"` ánh xạ `Green`).

---

## 5. Hạn chế & Minh bạch

1. **`B-human` toàn bộ do AI soạn (`65/65` `ai_written`)** và **`61/65` cụm là đơn lẻ** (chỉ `4/61` cụm có $\ge 2$ câu), nên bootstrap CI theo cụm gần bằng theo câu và độ rộng CI phản ánh cỡ mẫu nhỏ.
2. **Tập `val_gallery/` đã nằm trong tập dữ liệu mà checkpoint SigLIP từng thấy khi train**, nên mọi số liệu trên `val_gallery/` chỉ dùng để **so sánh tương đối** giữa các cấu hình, không suy diễn thành hiệu năng tổng quát trên `test_gallery/`.
3. **`parser_v2.1` đã được chỉnh sửa sau khi audit trên Tập A** (sửa ưu tiên cấu trúc `Unisex` và gỡ 18 mục `ec87034`), do đó **Tập A không còn ở trạng thái "chưa thấy"**.
4. **Biên lưới $\gamma = 1.5$:** Điểm CV còn tăng theo $\gamma$, lưới hiện tại có thể chưa phủ vùng tối ưu.
