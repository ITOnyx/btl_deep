# Báo cáo Chọn Cấu hình v5 — Pha 5c (`step5c_selection_report`)

- **Rule SHA-256:** `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | **Folds SHA-256:** `dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25` | **Seed:** `42`
- **Cổng B-human-v4:** `B-human-v4 (kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối)` (89 câu)
- **Tốt nhất (`winner_config_id`):** **`cfg_v5_09`** (CV Score `0.7031`, tie-break bootstrap 1.000 lần trong nhóm: `['cfg_v5_09', 'cfg_v5_04', 'cfg_v5_10', 'cfg_v5_11', 'cfg_v5_06', 'cfg_v5_08b']`)
- **An toàn (`safe_config_id`):** **`cfg_v5_01`** (đạt `5(i)` & `5(ii)` đơn giản nhất)

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
- **Tách có màu / không màu (`cfg_v5_09`):** Có màu (`n=48`) `mAP=0.6723` | Không màu (`n=41`) `mAP=0.5305`.
- **Tách theo `dia_class` (`cfg_v5_09`):** `{'full': {'n': 62, 'mAP': 0.6074}, 'none': {'n': 21, 'mAP': 0.6294}, 'partial': {'n': 6, 'mAP': 0.5241}}`.
- **Đối chiếu Luật 5 (`I5` dùng caption thật `val_gallery_text_v5`):** Config B (`cfg_v5_08b`, `internvl2_plus_aux`) đạt `CV Score = 0.7039`, `B-human-v4 mAP = 0.6120` cao hơn Config A (`cfg_v5_08a`, `internvl2_only`, `CV Score = 0.6904`, `B-human-v4 mAP = 0.5972`), nhưng chênh lệch so với `cfg_v5_09` nằm trong 95% bootstrap CI nên quy tắc `simplicity_order` chọn `cfg_v5_09` (`text_gallery=none`).

## 3. Phân tích Chẩn đoán (Oracle Parser, Hubness C.8, Bonus vs Cosine Std, Fallback & Trùng lặp)
- **Oracle Parser (`cfg_v5_09`):** `B-clean mAP(cl)=0.6209` (vs parser `0.5961`), `B-human-v4 mAP=0.6530` (vs parser `0.6070`).
- **Hubness C.8 (`B-clean`):** Model thuần `{'n_queries': 165, 'gallery_size': 1884, 'top_k': 10, 'unique_items_retrieved': 638, 'coverage_ratio': 0.3386411889596603, 'max_appearances': 36, 'max_query_pct': 0.21818181818181817, 'top1_hub_slot_share': 0.02181818181818182, 'top5_hub_slot_share': 0.09818181818181818, 'top10_hub_slot_share': 0.15454545454545454, 'mean_appearances': 0.8757961783439491, 'std_appearances': 2.47824250976883, 'skewness': 7.594302400581234, 'hub_threshold': 5, 'n_hubs_above_threshold': 79, 'top5_counts': [36, 35, 34, 29, 28]}` → Winner `{'n_queries': 165, 'gallery_size': 1884, 'top_k': 10, 'unique_items_retrieved': 905, 'coverage_ratio': 0.48036093418259024, 'max_appearances': 11, 'max_query_pct': 0.06666666666666667, 'top1_hub_slot_share': 0.006666666666666667, 'top5_hub_slot_share': 0.030303030303030304, 'top10_hub_slot_share': 0.051515151515151514, 'mean_appearances': 0.8757961783439491, 'std_appearances': 1.2616624327996124, 'skewness': 2.4370804346582893, 'hub_threshold': 5, 'n_hubs_above_threshold': 34, 'top5_counts': [11, 11, 10, 10, 8]}`.
- **Cosine std trung bình trên Val A:** `0.1601` (so sánh với biên độ bonus `tier_soft` / `gamma`).
- **I3 Cache Fallback:** `{'A': {'std': {'n_total': 240, 'n_cache_hit': 238, 'n_fallback': 2, 'fallback_rate': 0.0083}, 'fuzzy': {'n_total': 240, 'n_cache_hit': 238, 'n_fallback': 2, 'fallback_rate': 0.0083}}, 'B_clean': {'std': {'n_total': 165, 'n_cache_hit': 165, 'n_fallback': 0, 'fallback_rate': 0.0}, 'fuzzy': {'n_total': 165, 'n_cache_hit': 165, 'n_fallback': 0, 'fallback_rate': 0.0}}, 'B_typo': {'std': {'n_total': 80, 'n_cache_hit': 77, 'n_fallback': 3, 'fallback_rate': 0.0375}, 'fuzzy': {'n_total': 80, 'n_cache_hit': 77, 'n_fallback': 3, 'fallback_rate': 0.0375}}, 'C_20': {'std': {'n_total': 160, 'n_cache_hit': 160, 'n_fallback': 0, 'fallback_rate': 0.0}, 'fuzzy': {'n_total': 160, 'n_cache_hit': 160, 'n_fallback': 0, 'fallback_rate': 0.0}}, 'B_human': {'std': {'n_total': 89, 'n_cache_hit': 88, 'n_fallback': 1, 'fallback_rate': 0.0112}, 'fuzzy': {'n_total': 89, 'n_cache_hit': 88, 'n_fallback': 1, 'fallback_rate': 0.0112}}}`.
- **Các cấu hình trùng số liệu:** `{'exact_all_datasets_4dec': [['cfg_v5_09', 'cfg_v5_10']], 'identical_on_A_Bc_Bt_Bhuman_4dec': [['cfg_v5_04', 'cfg_v5_09', 'cfg_v5_10', 'cfg_v5_11']]}`.