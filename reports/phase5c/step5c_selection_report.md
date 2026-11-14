# Báo cáo Chọn Cấu hình v5 — Pha 5c (`step5c_selection_report`)

- **Rule SHA-256:** `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | **Folds SHA-256:** `dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25` | **Seed:** `42`
- **Cổng B-human-v4:** `B-human-v4 (kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối)` (89 câu)
- **Tốt nhất (`winner_config_id` — ĐÃ DUYỆT CHÍNH THỨC CHO T2I):** **`cfg_v5_09`** (`γ=1.0, tier_soft, I3 cached, text_gallery=none`, CV Score `0.7031`, tie-break bootstrap 1.000 lần trong nhóm: `['cfg_v5_09', 'cfg_v5_04', 'cfg_v5_10', 'cfg_v5_11', 'cfg_v5_06', 'cfg_v5_08b']`, giữ nguyên `selection_rule_v5`, KHÔNG chọn lại dựa trên `B-human-v4`).
- **An toàn (`safe_config_id`):** **`cfg_v5_01`** (đạt `5(i)` & `5(ii)` đơn giản nhất).
- **Hàng tham khảo bổ sung đã khai báo trước cho Prompt 14 (báo cùng 1 lần chạy test, KHÔNG dùng để chọn lại):** **`cfg_v5_06`** (`+I7 fuzzy`) và **`cfg_v5_08b`** (`I5 Config B: internvl2_plus_aux`).

## 1. Bảng So sánh Model thuần vs Hệ thống Soft vs Hệ thống Gated (12 Cấu hình v5)

| Nhóm / Config | Tranh hạng | CV Score | A mAP(cl) | Bc mAP(cl) | Bt mAP(cl) | C20 mAP | Tụt thảm họa | Parser FP | p95 C.9 (med ± IQR ms) | 5(i)-(v) |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `model_only` | — | — | 0.4970 | 0.1177 | — | 0.2972 | 0.0000 | — | — | — |
| `cfg_v5_01` (system_soft) | ✅ | 0.5296 | 0.8650 | 0.3061 | 0.1895 | 0.5693 | 0.0000 | 0.0101 | 103.81 ± 16.31 | ✅ |
| `cfg_v5_02` (eq_ref) | ❌ | 0.6037 | 0.7271 | 0.5216 | 0.3155 | 0.6217 | 0.0061 | 0.0101 | 99.31 ± 13.41 | ✅ |
| `cfg_v5_03` (eq_ref) | ❌ | 0.6040 | 0.7271 | 0.5219 | 0.3154 | 0.6211 | 0.0061 | 0.0101 | 86.55 ± 9.86 | ✅ |
| `cfg_v5_04` (system_soft) | ✅ | 0.7031 | 0.8638 | 0.5961 | 0.3385 | 0.7193 | 0.0061 | 0.0101 | 86.55 ± 9.86 | ✅ |
| `cfg_v5_05` (system_gated) | ✅ | 0.7011 | 0.8638 | 0.5927 | 0.3385 | 0.7193 | 0.0061 | 0.0101 | 86.55 ± 9.86 | ✅ |
| `cfg_v5_06` (system_soft) | ✅ | 0.7056 | 0.8638 | 0.6003 | 0.4372 | 0.7224 | 0.0061 | 0.0101 | 91.48 ± 3.81 | ✅ |
| `cfg_v5_07` (system_gated) | ✅ | 0.7036 | 0.8638 | 0.5969 | 0.4372 | 0.7224 | 0.0061 | 0.0101 | 91.48 ± 3.81 | ✅ |
| `cfg_v5_08a` (system_soft) | ✅ | 0.6904 | 0.8421 | 0.5894 | 0.3355 | 0.7016 | 0.0061 | 0.0101 | 87.36 ± 4.88 | ✅ |
| `cfg_v5_08b` (system_soft) | ✅ | 0.7039 | 0.8640 | 0.5973 | 0.3461 | 0.7161 | 0.0061 | 0.0101 | 90.69 ± 3.08 | ✅ |
| `cfg_v5_09` (system_soft) | ✅ | 0.7031 | 0.8638 | 0.5961 | 0.3385 | 0.7195 | 0.0061 | 0.0101 | 86.55 ± 9.86 | ✅ |
| `cfg_v5_10` (system_soft) | ✅ | 0.7031 | 0.8638 | 0.5961 | 0.3385 | 0.7195 | 0.0061 | 0.0101 | 86.55 ± 9.86 | ✅ |
| `cfg_v5_11` (system_soft) | ✅ | 0.7031 | 0.8638 | 0.5961 | 0.3385 | 0.7197 | 0.0061 | 0.0101 | 86.55 ± 9.86 | ✅ |

## 2. Cổng Xác nhận `B-human-v4 (kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối)` (chạy 1 lần sau khi chọn, KHÔNG chọn lại)

| Cấu hình | Vai trò | mAP | R@1 | R@10 | ΔmAP vs `cfg_v5_01` (95% CI) | Tụt thảm họa | Có màu (`n=48`) | Không màu (`n=41`) | `full` (`n=62`) | `none` (`n=21`) | `partial` (`n=6`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_only` | Model thuần | 0.2168 | 0.3034 | 0.5843 | — | 0.0000 | — | — | — | — | — |
| `cfg_v5_01` | **An toàn (`safe`)** / Baseline v4 | 0.5237 | 0.6742 | 0.9101 | `+0.0000 [+0.0000, +0.0000]` | 0.0000 | 0.5095 | 0.5404 | 0.5255 | 0.5518 | 0.4071 |
| `cfg_v5_09` | **Tốt nhất (`winner`)** (`γ=1.0, I3 cached`) | **0.6070** | **0.8315** | **0.8989** | **`+0.0832 [+0.0234, +0.1459]`** | **0.0225** | **0.6723** | **0.5305** | **0.6074** | **0.6294** | **0.5241** |
| `cfg_v5_06` | Top CV Score (`0.7056`, `+I7 fuzzy`) | 0.6128 | 0.8427 | 0.9101 | `+0.0891 [+0.0296, +0.1518]` | 0.0225 | 0.6830 | 0.5305 | 0.6157 | 0.6294 | 0.5241 |
| `cfg_v5_08a` | Luật 5 Config A (`internvl2_only`) | 0.5972 | 0.8202 | 0.8989 | `+0.0734 [+0.0120, +0.1368]` | 0.0225 | 0.6563 | 0.5280 | 0.5982 | 0.6294 | 0.4739 |
| `cfg_v5_08b` | Luật 5 Config B (`internvl2_plus_aux`) | 0.6120 | 0.8315 | 0.9101 | `+0.0883 [+0.0283, +0.1504]` | 0.0225 | 0.6821 | 0.5300 | 0.6149 | 0.6286 | 0.5245 |

- **`cfg_v5_09` (Tốt nhất):** `mAP=0.607`, `ΔmAP vs cfg_v5_01 = +0.0832` (95% CI: `[+0.0234, +0.1459]`), tụt thảm họa `0.0225` (`<= 0.05` đạt cổng).
- **Ghi nhận đánh đổi `Recall@10` trên `B-human-v4`:** `Recall@10` của winner `cfg_v5_09` (`0.8989`, `80/89` câu) thấp hơn baseline `cfg_v5_01` (`0.9101`, `81/89` câu) một mức `-0.0112` (tương ứng `1/89` câu), trong khi `mAP` tăng `+0.0832` (`0.5237 -> 0.6070`) và `Recall@1` tăng `+0.1573` (`0.6742 -> 0.8315`).
- **Tách có màu / không màu (`cfg_v5_09`):** Có màu (`n=48`) `mAP=0.6723` | Không màu (`n=41`) `mAP=0.5305`.
- **Tách theo `dia_class` (`cfg_v5_09`):** `{'full': {'n': 62, 'mAP': 0.6074}, 'none': {'n': 21, 'mAP': 0.6294}, 'partial': {'n': 6, 'mAP': 0.5241}}`.
- **Nhận xét Luật 5 (`I5` dùng caption thật `val_gallery_text_v5`):**
  1. Caption `InternVL2-2B` độc lập (`cfg_v5_08a`, Config A `internvl2_only`) đạt `CV Score = 0.6904`, `B-human-v4 mAP = 0.5972`, **không đem lại cải thiện đo được** (thậm chí giảm nhẹ so với `cfg_v5_09` không dùng text gallery: `0.7031` / `0.6070`).
  2. Config B (`cfg_v5_08b`, `internvl2_plus_aux` qua `merge_v1`) đạt `CV Score = 0.7039`, `B-human-v4 mAP = 0.6120`, nhưng **không tách được đóng góp của `InternVL2-2B` khỏi `Vintern-1B-v3.5 + metadata`** (do ở pilot 200 ảnh, `InternVL2-2B` thuần khớp `0.0%` metadata và toàn bộ phần thuộc tính phân biệt trong `merge_v1` đến từ `Vintern + metadata anchor`).

## 3. Phân tích Chẩn đoán (Oracle Parser, Hubness C.8, Bonus vs Cosine Std, Fallback & Trùng lặp)
- **Oracle Parser (`cfg_v5_09`):** `B-clean mAP(cl)=0.6209` (vs parser `0.5961`), `B-human-v4 mAP=0.6530` (vs parser `0.6070`).
- **Hubness C.8 (`B-clean`):** Model thuần `{'n_queries': 165, 'gallery_size': 1884, 'top_k': 10, 'unique_items_retrieved': 638, 'coverage_ratio': 0.3386411889596603, 'max_appearances': 36, 'max_query_pct': 0.21818181818181817, 'top1_hub_slot_share': 0.02181818181818182, 'top5_hub_slot_share': 0.09818181818181818, 'top10_hub_slot_share': 0.15454545454545454, 'mean_appearances': 0.8757961783439491, 'std_appearances': 2.47824250976883, 'skewness': 7.594302400581234, 'hub_threshold': 5, 'n_hubs_above_threshold': 79, 'top5_counts': [36, 35, 34, 29, 28]}` → Winner `{'n_queries': 165, 'gallery_size': 1884, 'top_k': 10, 'unique_items_retrieved': 905, 'coverage_ratio': 0.48036093418259024, 'max_appearances': 11, 'max_query_pct': 0.06666666666666667, 'top1_hub_slot_share': 0.006666666666666667, 'top5_hub_slot_share': 0.030303030303030304, 'top10_hub_slot_share': 0.051515151515151514, 'mean_appearances': 0.8757961783439491, 'std_appearances': 1.2616624327996124, 'skewness': 2.4370804346582893, 'hub_threshold': 5, 'n_hubs_above_threshold': 34, 'top5_counts': [11, 11, 10, 10, 8]}`.
- **Bằng chứng `cfg_v5_04/09/10/11` trùng số liệu do `tier_soft` bão hòa theo `γ`:**
  - Trong `src/eval/canonical_eval.py`, khi `bonus_mode == "tier_soft"`, điểm xếp hạng là `s_val = base_s + γ * (mf + tier_b)` với `base_s` là raw cosine (không z-score), `mf ∈ {0, 1/3, 1/2, 2/3, 1.0}` và `tier_b = 0.5` khi `mf == 1.0` (`mf + tier_b = 1.5`).
  - Độ lệch chuẩn cosine trung bình trên `Val A` là **`sd = 0.1601`** (toàn bộ dải chênh lệch `max(cos) - min(cos)` trong cùng một truy vấn thường chỉ `~0.40 - 0.60`).
  - Trong khi đó, ngay tại `γ = 1.0` (`cfg_v5_09`), bước nhảy từ bậc khớp `2/3` (`mf = 0.6667`) lên `3/3` (`mf + tier_b = 1.5000`) là `Δ = +0.8333` (`= 5.20 × sd`), và giữa hai bậc kề nhau `1/3 -> 2/3` là `Δ = +0.3333` (`= 2.08 × sd`).
  - Vì bước nhảy bậc ở `γ = 1.0` đã vượt xa biên độ biến thiên của `base_s`, các bậc thuộc tính đã tách rời hoàn toàn (tương đương sắp xếp từ điển theo `(tier, base_s)`). Khi tăng `γ` lên `1.5` (`cfg_v5_04`), `2.0` (`cfg_v5_10`), hay `3.0` (`cfg_v5_11`), khoảng cách giữa các bậc chỉ giãn rộng thêm mà không làm đổi thứ tự `base_s` bên trong từng bậc.
  - Hệ quả: `cfg_v5_04`, `cfg_v5_09`, `cfg_v5_10`, `cfg_v5_11` trùng khít 100% trên `A` (`0.8638`), `B_clean` (`0.5961`), `B_typo` (`0.3385`), `B-human-v4` (`0.6070`); chỉ lệch `0.0002` trên `C_20` (`0.7193 / 0.7195 / 0.7197`) ở vài truy vấn bị tiêm lỗi parser.
- **I3 Cache Fallback:** `{'A': {'std': {'n_total': 240, 'n_cache_hit': 238, 'n_fallback': 2, 'fallback_rate': 0.0083}, 'fuzzy': {'n_total': 240, 'n_cache_hit': 238, 'n_fallback': 2, 'fallback_rate': 0.0083}}, 'B_clean': {'std': {'n_total': 165, 'n_cache_hit': 165, 'n_fallback': 0, 'fallback_rate': 0.0}, 'fuzzy': {'n_total': 165, 'n_cache_hit': 165, 'n_fallback': 0, 'fallback_rate': 0.0}}, 'B_typo': {'std': {'n_total': 80, 'n_cache_hit': 77, 'n_fallback': 3, 'fallback_rate': 0.0375}, 'fuzzy': {'n_total': 80, 'n_cache_hit': 77, 'n_fallback': 3, 'fallback_rate': 0.0375}}, 'C_20': {'std': {'n_total': 160, 'n_cache_hit': 160, 'n_fallback': 0, 'fallback_rate': 0.0}, 'fuzzy': {'n_total': 160, 'n_cache_hit': 160, 'n_fallback': 0, 'fallback_rate': 0.0}}, 'B_human': {'std': {'n_total': 89, 'n_cache_hit': 88, 'n_fallback': 1, 'fallback_rate': 0.0112}, 'fuzzy': {'n_total': 89, 'n_cache_hit': 88, 'n_fallback': 1, 'fallback_rate': 0.0112}}}`.
- **Các cấu hình trùng số liệu:** `{'exact_all_datasets_4dec': [['cfg_v5_09', 'cfg_v5_10']], 'identical_on_A_Bc_Bt_Bhuman_4dec': [['cfg_v5_04', 'cfg_v5_09', 'cfg_v5_10', 'cfg_v5_11']]}`.

## 4. Giải trình Độ trễ C.9 trên Val & Quy định Đo Riêng ở Prompt 14
1. **Vì sao nhiều cấu hình có cùng giá trị `86.55 ± 9.86 ms`?**
   - Trong `scripts/run_step5c_v5_grid.py`, hàm `measure_c9_interleaved_latency` đã **gom nhóm (`profile_rep_config_id`)** các cấu hình có cùng khóa thực thi `(i3_mode, fuzzy_typo, text_gallery)` thành 6 profile đại diện (`cfg_v5_01`, `cfg_v5_02`, `cfg_v5_03`, `cfg_v5_06`, `cfg_v5_08a`, `cfg_v5_08b`) và chưa đo vòng lặp cộng điểm metadata Python riêng cho từng cấu hình. Do đó `cfg_v5_03, 04, 05, 09, 10, 11` cùng nhận số đo từ profile `cfg_v5_03` (`86.55 ± 9.86 ms`), và `cfg_v5_06, 07` cùng nhận số đo từ profile `cfg_v5_06` (`91.48 ± 3.81 ms`).
2. **Vì sao `cfg_v5_01` (`p95 = 103.81 ms`) lại có `p95` cao hơn `cfg_v5_09` (`p95 = 86.55 ms`)?**
   - Xét tại trung vị **`p50`**, `cfg_v5_01` đạt **`53.924 ms`**, thực chất **nhanh hơn nhẹ** so với `cfg_v5_09` (`p50 = 54.113 ms`, do `cfg_v5_09` thêm bước tra bảng RAM `template_cache_v5`).
   - Sự đảo chiều ở đuôi **`p95`** (`103.815 ms` vs `86.549 ms`) đến từ hai nguyên nhân thực nghiệm:
     - *(i) Hiệu ứng vị trí biên trong vòng lặp đan xen:* `cfg_v5_01` nằm ở vị trí `0` của mảng `rep_configs = [cfg_v5_01, cfg_v5_02, cfg_v5_03, cfg_v5_06, cfg_v5_08a, cfg_v5_08b]`, sát cạnh `cfg_v5_02` (chạy 3 forward GPU liên tiếp) và là phần tử đầu tiên/cuối cùng khi đảo chiều chẵn-lẻ, dễ hứng xung trễ chuyển tiếp lập lịch CUDA/CPU hơn `cfg_v5_03` nằm giữa các profile 1-forward.
     - *(ii) Xung nhiệt & tiến trình nền GPU:* Khi đo, máy có tiến trình đồ họa nền (`RobloxPlayerBeta.exe`, `dnmultiplayerex.exe`) và GPU đang ở `84–87°C` (xung nhịp bị ghìm về `832 MHz`), tạo ra ~5% gai trễ đuôi (`>100 ms`) rơi nhiều hơn vào vị trí biên `cfg_v5_01`.
3. **Cam kết bắt buộc cho Prompt 14 (`test_gallery`):**
   - Đo **RIÊNG từng cấu hình** (`cfg_v5_01`, `cfg_v5_06`, `cfg_v5_09`, `cfg_v5_08b`) bao gồm đầy đủ toàn bộ bước rerank cuối-đến-cuối, hoàn toàn **không gom nhóm**.
   - Đan xen ngẫu nhiên/Latin-square để triệt tiêu hiệu ứng vị trí biên, chạy trên **GPU nguội thật sự** (ghi rõ nhiệt độ + xung nhịp trước/sau mỗi run), `>= 5 run × 200 truy vấn` (warmup 20), báo cáo đầy đủ `p50 / p95 / p99` cùng trung vị và IQR của `p95`.