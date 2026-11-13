# Phụ lục Bước 4 (v4): Chẩn đoán, Sửa báo cáo & Đo lại độ trễ C.9

> **Commit nền:** `67c3f4acf81e6ea476249e444abff19082685bb0` (tag `phase5c-v4`) | **Seed:** `42`
> **Bất biến:** Không sửa `parser_v1.py`, `parser_v2.py` (`v2.1`), `selection_rule_v3/v4.*`, các tập khóa, và **không** chạy `index/test_gallery/`. Không đổi kết luận chọn cấu hình của `v4` (`cfg_v4_05`).

---

## Phần 1 — Tóm tắt các điểm đã sửa trong `step4_v4_report.md`

1. **Nhãn lạc quan trên `Val A`:** Mọi kết luận `"Đạt chắc chắn"` trên `Val A` đã được gắn nhãn **`"THAM KHẢO, ước lượng lạc quan (ảnh val đã được checkpoint thấy, parser gần 100% trên A, GT trùng thuộc tính hệ thống dùng)"`** và ghi rõ không dùng làm bằng chứng đạt chỉ tiêu `0.85` / `0.75`.
2. **Mục I8:** Sửa thành **`KHÔNG THỰC HIỆN (n_evaluated=0)`**, bỏ toàn bộ giá trị `0.0000`, nêu rõ `usage`/`season` đã đề xuất nhưng chưa có trong `catalog_rebuild.parquet` và chưa nối ánh xạ GT.
3. **Mục I5 (Luật 5):** Nêu rõ `val_gallery` không có caption `InternVL2-2B` trong `data/catalog/captions/`; `index/val_gallery_text_v4/` chỉ dùng `title_vi` (`cfg_v4_11/12/18`) và `metadata_aux` (`title_vi + articleType + gender + baseColour`, chứa trực tiếp thuộc tính GT, `cfg_v4_13`).
4. **Cấu hình tương đương & Số cấu hình hiệu dụng:**
   - Số cấu hình hiệu dụng khác nhau thật sự: **16** trên `(Val A, B-clean)` (và **18** trên cả 4 tập).
   - `cfg_v4_01` vs `cfg_v4_03`: trùng tuyệt đối trên `Val A` (`0.715725`) và `B-clean` (`0.247326`) vì `(cos - μ)/σ + γ·mf` là phép biến đổi tuyến tính đơn điệu của `cos + γ·σ·mf` khi không có thành phần điểm thứ 3.
   - `cfg_v4_01` vs `cfg_v4_08` (`fuzzy_typo=true`): trùng tuyệt đối trên `Val A` (`0.715725`) và `B-clean` (`0.247326`) vì không có lỗi chính tả độ dài `>= 4` làm đổi kết quả parse; chỉ khác trên `B-typo` (`0.172845 -> 0.187025`).
   - Đã gộp hàng `Hệ thống soft` và `Hệ thống gated` vì cùng là `cfg_v4_05` (không dùng `conf_gating`).

---

## Phần 2 — Chẩn đoán (chỉ báo cáo, không chọn lại cấu hình `v4`)

### 2.1. Oracle Parser (`spec` thay cho kết quả parse) — Tách phần mất điểm do Parser vs Model

Độ phủ `articleType` của `parser_v2.1`:
- `Val A`: `240/240` (`100.0%`)
- `B-clean`: **`130/165` (`78.79%`)**
- `B-typo`: `48/80` (`60.00%`)
- `B_human_ref` (THAM KHẢO): `62/65` (`95.38%`)

| Tập dữ liệu | Cấu hình | System mAP | Oracle mAP | Mất điểm do Parser (`Oracle - Sys`) | Khoảng trống Model (`1.0 - Oracle`) | System R@10 | Oracle R@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| `Val A` | `cfg_v4_01` | 0.7157 | 0.7183 | 0.0026 | 0.2817 | 0.9792 | 0.9792 |
| `Val A` | `cfg_v4_05` | 0.8650 | 0.8721 | 0.0071 | 0.1279 | 0.9833 | 0.9875 |
| `Val A` | `cfg_v4_14` | 0.7271 | 0.7306 | 0.0035 | 0.2694 | 0.9750 | 0.9792 |
| `B-clean` | `cfg_v4_01` | 0.2473 | 0.2606 | 0.0133 | 0.7394 | 0.5818 | 0.6364 |
| `B-clean` | `cfg_v4_05` | 0.3061 | 0.3314 | 0.0252 | 0.6686 | 0.6000 | 0.6667 |
| `B-clean` | `cfg_v4_14` | 0.5216 | 0.6456 | **0.1240** | **0.3544** | 0.7758 | **0.9455** |
| `B-typo` | `cfg_v4_01` | 0.1728 | 0.2189 | 0.0460 | 0.7811 | 0.4750 | 0.5875 |
| `B-typo` | `cfg_v4_05` | 0.1895 | 0.2616 | 0.0721 | 0.7384 | 0.5125 | 0.6250 |
| `B-typo` | `cfg_v4_14` | 0.3155 | 0.6061 | **0.2906** | **0.3939** | 0.6875 | **0.9125** |
| `B_human_ref` | `cfg_v4_01` | 0.5263 | 0.5489 | 0.0226 | 0.4511 | 0.8462 | 0.8923 |
| `B_human_ref` | `cfg_v4_05` | 0.6123 | 0.6511 | 0.0387 | 0.3489 | 0.8462 | 0.8923 |
| `B_human_ref` | `cfg_v4_14` | 0.6086 | 0.6402 | 0.0316 | 0.3598 | 0.8769 | 0.9077 |

**Nhận xét:** Trên `B-clean` và `B-typo`, với `cfg_v4_01`/`cfg_v4_05`, phần mất điểm chủ yếu nằm ở **Model/text embedding** khi gặp câu tự nhiên nhiều nhiễu (`1.0 - Oracle` tới `0.67–0.74`). Khi bật `I3` (`cfg_v4_14`), điểm Oracle nhảy vọt lên `0.6456` mAP / `0.9455` R@10 trên `B-clean` (và `0.6061` / `0.9125` trên `B-typo`), lúc đó nút thắt lộ rõ ở **Parser** (`0.1240` trên `B-clean` và `0.2906` trên `B-typo` do thiếu 35/165 câu `articleType` ngoài lexicon).

### 2.2. Phân rã `I3` (`cfg_v4_14` vs `cfg_v4_01`) theo `dia_class` & Trong/Ngoài Lexicon (Paired Cluster Bootstrap 1.000 lần)

| Tập | Nhóm phân rã | Số câu / cụm | `cfg_v4_14` mAP | `cfg_v4_01` mAP | `ΔmAP` (95% Cluster Bootstrap CI) |
|---|---|---:|---:|---:|---|
| `B-clean` | `dia_class = full` | 57 / 57 | 0.5563 | 0.2804 | **+0.2759** `[+0.1854, +0.3704]` |
| `B-clean` | `dia_class = partial` | 52 / 52 | 0.5113 | 0.2042 | **+0.3071** `[+0.2165, +0.4034]` |
| `B-clean` | `dia_class = none` | 56 / 56 | 0.4958 | 0.2537 | **+0.2421** `[+0.1641, +0.3266]` |
| `B-clean` | `articleType` **trong lexicon** | 130 / 130 | 0.6514 | 0.2975 | **+0.3539** `[+0.2954, +0.4097]` |
| `B-clean` | `articleType` **ngoài lexicon** | 35 / 35 | 0.0392 | 0.0611 | **-0.0218** `[-0.0397, -0.0076]` |
| `B-typo` | `dia_class = full` | 23 / 20 | 0.3794 | 0.2697 | **+0.1097** `[-0.0383, +0.2715]` |
| `B-typo` | `dia_class = partial` | 34 / 31 | 0.2718 | 0.0964 | **+0.1754** `[+0.0809, +0.2813]` |
| `B-typo` | `dia_class = none` | 23 / 23 | 0.3237 | 0.1812 | **+0.1426** `[+0.0353, +0.2816]` |
| `B-typo` | `articleType` **trong lexicon** | 48 / 42 | 0.4810 | 0.2112 | **+0.2698** `[+0.1667, +0.3705]` |
| `B-typo` | `articleType` **ngoài lexicon** | 32 / 30 | 0.0819 | 0.1133 | **-0.0314** `[-0.0696, +0.0005]` |
| `B_human_ref` | `dia_class = full` | 44 / 44 | 0.6736 | 0.5754 | **+0.0982** `[+0.0273, +0.1789]` |
| `B_human_ref` | `dia_class = partial` | 4 / 4 | 0.2111 | 0.1822 | **+0.0289** `[-0.1040, +0.1907]` |
| `B_human_ref` | `dia_class = none` | 17 / 17 | 0.5636 | 0.4726 | **+0.0910** `[+0.0061, +0.2024]` |
| `B_human_ref` | `articleType` **trong lexicon** | 62 / 58 | 0.6397 | 0.5516 | **+0.0881** `[+0.0280, +0.1575]` |
| `B_human_ref` | `articleType` **ngoài lexicon** | 3 / 3 | 0.0072 | 0.0377 | **-0.0305** `[-0.0914, +0.0000]` |

**Nhận xét:** Mức tăng của `I3` trên `B_human_ref` (`+0.0881` cho nhóm trong lexicon, CI dương `[+0.0280, +0.1575]`; `+0.0982` ở `full` và `+0.0910` ở `none`) chứng minh `I3` có ích thật trên câu tự nhiên, không chỉ do khớp template `B-synth`, dù biên độ trên `B-synth` (`+0.3539`) lớn hơn nhiều vì `B-synth` chứa nhiều câu dài/đồng nghĩa mà text encoder đơn khó khớp trực tiếp bằng mẫu chuẩn hóa. Ngược lại, với các câu **ngoài lexicon** `articleType`, `I3` làm giảm nhẹ (`-0.02` đến `-0.03`).

### 2.3. Hành vi theo `γ` của nhóm `I1` và `cfg_v4_05`
- **`I1_sigma_scaled`:** `γ=1.0` (`cfg_v4_02`) → `CV=0.3920` (FAIL); `γ=1.5` (`cfg_v4_03`) → `CV=0.4347` (PASS); `γ=2.0` (`cfg_v4_04`) → `CV=0.4685` (PASS). Điểm CV tăng đơn điệu theo `γ`.
- **`I1_tier_soft` (`cfg_v4_05`):** Chỉ có **1 điểm duy nhất `γ=1.5`** trong lưới `v4` (`CV=0.5296`) — đây là **biên lưới** của `tier_soft`.
- **Kích thước bonus so với độ lệch chuẩn cosine (`σ`) từng truy vấn:**
  - `σ` trung bình (mean / median [min, max]): `Val A` = `0.16005 / 0.16274 [0.11541, 0.21202]`; `B-clean` = `0.15814 / 0.15683 [0.10195, 0.21159]`; `B-typo` = `0.15605`; `B_human_ref` = `0.15606`.
  - Với `zscore` (`γ=1.5`), bonus tối đa (`mf=1.0`) tương đương **`1.5 σ`** (`~0.24` theo thang cosine gốc). Với `tier_soft` (`γ=1.5`), `s_val = cos + 1.5 * (mf + 0.5)` (khi `mf=1.0`, bonus = `2.25`, tương đương **`~14 σ`** theo thang cosine gốc — tức `tier_soft` thực chất đang hoạt động gần như soft-tier mạnh ưu tiên các mẫu khớp đủ 100% slot).

---

## Phần 3 — Đo lại độ trễ đúng protocol C.9 (so sánh cả 2 lần đo)

- **Protocol lần đo lại (`step4_latency_c9.json`):** Chạy xen kẽ `cfg_v4_01 -> cfg_v4_05 -> cfg_v4_14 -> cfg_v4_15` qua **5 vòng × 200 truy vấn**, warmup 20, `torch.cuda.synchronize()` quanh forward.
- **Nhiệt độ GPU & Vi phạm làm nguội:** Mục tiêu `< 60°C` trước mỗi cấu hình **KHÔNG đạt được** dù đã chờ 30s mỗi lượt — nhiệt độ thực tế trước khi chạy dao động **`65°C – 69°C`**, sau khi chạy lên **`75°C – 78°C`** (xung lõi `495–600 -> 1635+ MHz`, bộ nhớ `810 -> 6000 MHz`). Trong khi đó lần đo gốc chạy tuần tự ở trạng thái nóng hơn (`82°C – 85°C`).

| Config | Lần đo gốc (tuần tự, 82–85°C) p95 | Lần đo lại (xen kẽ, 65–78°C) p50 / p95 / p99 (ms) | Trung vị p95 qua 5 vòng (IQR) [Q1, Q3] | Số forward text TB / query | Chi phí encode text p50 / p95 (ms) | Δp50 / Δp95 tổng vs `cfg_v4_01` (lần đo lại) |
|---|---:|---|---|---:|---:|---:|
| `cfg_v4_01` | 45.711 ms | 23.872 / **48.610** / 79.770 | 53.056 (IQR 4.924) [48.150, 53.074] | 1.000 | 21.866 / 45.101 | 0.000 / 0.000 |
| `cfg_v4_05` | 51.431 ms | 23.903 / **48.444** / 76.460 | 48.482 (IQR 3.244) [47.800, 51.044] | 1.000 | 21.949 / 44.969 | +0.030 / -0.166 |
| `cfg_v4_14` (`I3 k=2`) | 69.840 ms (+24.13 ms, FAIL) | 37.454 / **55.328** / 82.322 | 55.019 (IQR 4.826) [50.847, 55.673] | **1.880** | 35.344 / 50.286 | **+13.581 / +6.718** |
| `cfg_v4_15` (`I3 k=3`) | 77.060 ms (+31.35 ms, FAIL) | 45.686 / **56.611** / 80.245 | 56.069 (IQR 4.463) [53.503, 57.966] | **2.405** | 43.395 / 53.282 | **+21.813 / +8.001** |

- **Phân tích chi phí I3:** Ở `cfg_v4_14`, số forward text trung bình là `1.880`/truy vấn; phần encode text tăng `+13.478 ms` ở p50 và `+5.185 ms` ở p95, chiếm gần như toàn bộ phần tăng tổng (`+13.581 ms` p50, `+6.718 ms` p95).
- **Kết luận 5(iv):** Theo lần đo gốc trong lưới `v4`, `cfg_v4_14` (+24.13 ms) và `cfg_v4_15` (+31.35 ms) vượt ngưỡng `+20 ms`. Theo lần đo xen kẽ (ít lệch nhiệt giữa các cấu hình hơn, nhưng vẫn vi phạm mục tiêu `< 60°C`), mức tăng p95 chỉ là `+6.718 ms` và `+8.001 ms`. **Không tự thay đổi kết luận `v4` (`cfg_v4_05` giữ nguyên là winner `v4`)**; việc công nhận lần đo nào làm căn cứ `5(iv)` chính thức do **người dùng quyết định**, và ở `v5` thiết kế cache `I3_cached` sẽ loại bỏ hoàn toàn forward text thứ 2.
