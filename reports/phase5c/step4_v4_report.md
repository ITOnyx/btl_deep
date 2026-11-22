# Báo cáo Pha 5c — Bước 4: Đánh giá lưới 18 cấu hình (`selection_rule_v4`)

## 1. Thông tin tái lập & Bất biến
- **Git commit:** `924afcb8ef2556839606c8f1e5d25c13ab217681`
- **Seed:** `42`
- **`selection_rule_v4.json` SHA-256:** `bbf28f465c544b0babfdcedb315c8caee5b2a669b79e852957a7ecc4f2822fc6`
- **`folds.json` SHA-256:** `dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25`
- **Gallery:** `index/val_gallery/` (1.884 sản phẩm, **KHÔNG** đụng `index/test_gallery/`).
- **Luật 5 — Giới hạn I5 trên `val_gallery`:** Trong `data/catalog/captions/`, chỉ `test_gallery` có caption `InternVL2-2B`; `val_gallery` hiện có `caption_model=None`. Do đó `index/val_gallery_text_v4/` chỉ dùng `title_vi` (`cfg_v4_11/12/18`) và `metadata_aux` (`title_vi + articleType + gender + baseColour`, `cfg_v4_13` — chứa trực tiếp thuộc tính GT). Kết quả I5 trên `val_gallery` chỉ phản ánh text-fusion từ tiêu đề/metadata, **không** phản ánh đóng góp của `InternVL2-2B`.
- **Cảnh báo B-human:** Số liệu `B_human_ref` (65 câu, 61 cụm) chỉ mang tính **THAM KHẢO (REFERENCE ONLY)** vì tập này đã được quan sát ở v3 và chưa có tập `B-human-v4` mới.

## 2. Kết quả lưới 18 cấu hình (5-fold Cluster CV trên `Val A` + `B-synth clean`)

| Config | Ý tưởng | CV Score | Val A mAP (cụm) | B-clean mAP (cụm) | B-typo mAP | C-20% mAP | B-human_ref (THAM KHẢO) | Catastrophic Drop (B-clean) | T2I p95 (ms) | Đạt 5(i)-(v) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `model_only` | Baseline SigLIP thuần | 0.2694 | 0.4970 [0.4424, 0.5439] | 0.1177 [0.0853, 0.1556] | 0.1081 | 0.2700 | 0.2866 | 0.000 | — | — |
| `cfg_v4_01` | baseline_cfg13 | 0.4347 | 0.7157 [0.6669, 0.7638] | 0.2473 [0.2007, 0.2984] | 0.1728 | 0.4342 | 0.5263 | 0.000 | 45.71 | PASS |
| `cfg_v4_02` | I1_sigma_scaled | 0.3920 | 0.6650 [0.6118, 0.7152] | 0.2099 [0.1671, 0.2599] | 0.1572 | 0.3908 | 0.4745 | 0.000 | 52.55 | FAIL |
| `cfg_v4_03` | I1_sigma_scaled | 0.4347 | 0.7157 [0.6669, 0.7638] | 0.2473 [0.2007, 0.2984] | 0.1728 | 0.4342 | 0.5263 | 0.000 | 51.79 | PASS |
| `cfg_v4_04` | I1_sigma_scaled | 0.4685 | 0.7611 [0.7151, 0.8065] | 0.2736 [0.2242, 0.3255] | 0.1823 | 0.4629 | 0.5551 | 0.000 | 49.56 | PASS |
| `cfg_v4_05` | I1_tier_soft | 0.5296 | 0.8650 [0.8245, 0.9040] | 0.3061 [0.2525, 0.3643] | 0.1895 | 0.5290 | 0.6123 | 0.000 | 51.43 | PASS |
| `cfg_v4_06` | I2_conf_gating | 0.4416 | 0.7302 [0.6819, 0.7766] | 0.2493 [0.2014, 0.3007] | 0.1731 | 0.4342 | 0.5337 | 0.000 | 62.40 | PASS |
| `cfg_v4_07` | I2_conf_gating | 0.4751 | 0.7740 [0.7278, 0.8175] | 0.2759 [0.2266, 0.3285] | 0.1825 | 0.4629 | 0.5626 | 0.000 | 48.19 | PASS |
| `cfg_v4_08` | I7_fuzzy_typo | 0.4347 | 0.7157 [0.6669, 0.7638] | 0.2473 [0.2007, 0.2984] | 0.1870 | 0.4342 | 0.5263 | 0.000 | 50.13 | PASS |
| `cfg_v4_09` | I6_CSLS | 0.4044 | 0.6787 [0.6255, 0.7258] | 0.2215 [0.1784, 0.2689] | 0.1595 | 0.4065 | 0.5054 | 0.000 | 51.16 | FAIL |
| `cfg_v4_10` | I6_CSLS | 0.4000 | 0.6732 [0.6199, 0.7208] | 0.2177 [0.1744, 0.2653] | 0.1563 | 0.4021 | 0.4993 | 0.000 | 47.83 | FAIL |
| `cfg_v4_11` | I5_text_gallery_internvl2 | 0.3978 | 0.6650 [0.6187, 0.7126] | 0.2199 [0.1735, 0.2687] | 0.1660 | 0.3925 | 0.4672 | 0.018 | 54.93 | FAIL |
| `cfg_v4_12` | I5_text_gallery_internvl2 | 0.4305 | 0.7086 [0.6621, 0.7564] | 0.2452 [0.1992, 0.2964] | 0.1740 | 0.4294 | 0.5042 | 0.000 | 47.50 | PASS |
| `cfg_v4_13` | I5_text_gallery_aux_law5 | 0.4487 | 0.7507 [0.7011, 0.7976] | 0.2474 [0.1990, 0.2988] | 0.1771 | 0.4499 | 0.5228 | 0.000 | 47.34 | PASS |
| `cfg_v4_14` | I3_multi_template_batch | 0.6037 | 0.7271 [0.6777, 0.7737] | 0.5216 [0.4674, 0.5774] | 0.3155 | 0.5940 | 0.6086 | 0.006 | 69.84 | FAIL |
| `cfg_v4_15` | I3_multi_template_batch | 0.5926 | 0.7285 [0.6817, 0.7724] | 0.5021 [0.4457, 0.5585] | 0.2920 | 0.5796 | 0.6023 | 0.006 | 77.06 | FAIL |
| `cfg_v4_16` | I4_rocchio_prf | 0.4262 | 0.7139 [0.6636, 0.7627] | 0.2344 [0.1878, 0.2853] | 0.1656 | 0.4273 | 0.5292 | 0.000 | 60.12 | FAIL |
| `cfg_v4_17` | I1_I2_I7 | 0.4418 | 0.7302 [0.6819, 0.7766] | 0.2495 [0.2017, 0.3009] | 0.1874 | 0.4342 | 0.5337 | 0.000 | 48.68 | PASS |
| `cfg_v4_18` | I1_I2_I7_I5 | 0.3260 | 0.5618 [0.5081, 0.6186] | 0.1693 [0.1293, 0.2144] | 0.1312 | 0.3142 | 0.3737 | 0.061 | 45.06 | FAIL |

## 3. Chọn cấu hình & Tie-breaking (Paired Cluster Bootstrap 1.000 lần)
- **Best by CV Score (đạt mọi ràng buộc):** `cfg_v4_05` (CV = 0.5296)
- **Các cấu hình hòa trong 95% CI với best:** `cfg_v4_05`
- **Winner sau tie-break theo `simplicity_order`:** `cfg_v4_05`

## 4. Báo cáo song song 4 hàng (Val A & B-synth clean)

| Vai trò | Config | Val A R@10 | Val A mAP (cụm, 95% CI) | Kết luận mAP >= 0.75 | B-clean R@10 | B-clean mAP (cụm, 95% CI) | Kết luận mAP >= 0.75 |
|---|---|---:|---|---|---:|---|---|
| Model thuần | `model_only` | 0.9083 | 0.4970 [0.4424, 0.5439] | Chưa đạt | 0.2970 | 0.1177 [0.0853, 0.1556] | Chưa đạt |
| Hệ thống soft (v3 baseline) | `cfg_v4_01` | 0.9792 | 0.7157 [0.6669, 0.7638] | Chưa đạt | 0.5818 | 0.2473 [0.2007, 0.2984] | Chưa đạt |
| Hệ thống soft / gated (v4 winner, không dùng conf_gating) | `cfg_v4_05` | 0.9833 | 0.8650 [0.8245, 0.9040] | THAM KHẢO, ước lượng lạc quan (ảnh val đã được checkpoint thấy, parser gần 100% trên A, GT trùng thuộc tính hệ thống dùng) | 0.6000 | 0.3061 [0.2525, 0.3643] | Chưa đạt |
| Safe config (5i+5ii) | `cfg_v4_01` | 0.9792 | 0.7157 [0.6669, 0.7638] | Chưa đạt | 0.5818 | 0.2473 [0.2007, 0.2984] | Chưa đạt |

> **Lưu ý:** Không được dùng kết luận trên `Val A` làm bằng chứng đạt chỉ tiêu `0.85` / `0.75`. Chỉ số chính thức chỉ xét trên `test_gallery`.

## 4b. Cấu hình tương đương & Số cấu hình hiệu dụng
- **Số cấu hình hiệu dụng khác nhau thật sự:** **16** cấu hình trên `(Val A, B-clean)` (và **18** trên cả 4 tập `A, B-clean, B-typo, C-20%`).

| Cặp đối chiếu | Val A mAP | B-clean mAP | B-typo mAP | Giải thích vì sao trùng
|---|---:|---:|---:|---|
| `cfg_v4_01` (`zscore`, γ=1.5) vs `cfg_v4_03` (`sigma_scaled`, γ=1.5) | 0.715725 vs 0.715725 | 0.247326 vs 0.247326 | 0.172845 vs 0.172843 | `(cos - μ)/σ + γ·mf` là phép biến đổi tuyến tính đơn điệu của `cos + γ·σ·mf` khi không có thành phần điểm thứ ba (`extra_score_matrix`/`bm25`). Sai khác ở chữ số thứ 6 trên `B-typo` chỉ do làm tròn tie-breaking dấu phẩy động. |
| `cfg_v4_01` (`fuzzy_typo=false`) vs `cfg_v4_08` (`fuzzy_typo=true`) | 0.715725 vs 0.715725 | 0.247326 vs 0.247326 | 0.172845 vs 0.187025 | Trên `A` và `B-clean` không có lỗi chính tả độ dài ≥ 4 ký tự làm thay đổi kết quả parse, nên `fuzzy_typo` chỉ tác động trên `B-typo` (+0.0142 mAP). |

## 5. Hubness (Mục C.8 trên Val A)
- **`model_only` trước mean-centering:** max_appearances=14, coverage_ratio=0.4045, skewness=2.1166, n_hubs(>5)=152
- **`model_only` sau mean-centering:** max_appearances=35, coverage_ratio=0.2659, skewness=4.1026, n_hubs(>5)=177
- **Winner (`cfg_v4_05`):** max_appearances=13, coverage_ratio=0.4995, skewness=1.5740, n_hubs(>5)=108

## 6. Độ trễ T2I (Mục C.9: 5 run × 200 truy vấn, warmup 20, `torch.cuda.synchronize()`)
- **`cfg_v4_01` (v3 baseline):** p50=28.475 ms, p95=45.711 ms, p99=63.952 ms | GPU before: {'gpu': 'NVIDIA GeForce RTX 2050', 'temp_c': 82, 'graphics_clock_mhz': 367, 'mem_clock_mhz': 5500} | GPU after: {'gpu': 'NVIDIA GeForce RTX 2050', 'temp_c': 85, 'graphics_clock_mhz': 1687, 'mem_clock_mhz': 6000}
- **Winner (`cfg_v4_05`):** p50=35.129 ms, p95=51.431 ms, p99=66.914 ms (chênh lệch p95 so với `cfg_v4_01`: +5.720 ms) | GPU before: {'gpu': 'NVIDIA GeForce RTX 2050', 'temp_c': 83, 'graphics_clock_mhz': 495, 'mem_clock_mhz': 5500} | GPU after: {'gpu': 'NVIDIA GeForce RTX 2050', 'temp_c': 84, 'graphics_clock_mhz': 1635, 'mem_clock_mhz': 6000}

## 7. Kiểm tra thuộc tính phi vòng tròn (I8)
- **Trạng thái:** **KHÔNG THỰC HIỆN (`n_evaluated=0`)**.
- **Lý do:** Đã đề xuất kiểm tra trên các thuộc tính ngoài 3-slot (`usage`/`season` hoặc `masterCategory`/`subCategory`), nhưng `usage`/`season` chưa có trong `catalog_rebuild.parquet` và chưa nối ánh xạ GT cho phép đo này ở Bước 4.

## 8. Giới hạn trung thực (Luật 1 & C.10)
- Checkpoint `best_model.pt` giữ nguyên không fine-tune (Luật 1). Toàn bộ cải thiện nằm ở tầng suy luận.
- Chưa chạy `index/test_gallery/` (Bước 5 chỉ chạy 1 lần sau khi người dùng duyệt kết quả Bước 4 và/hoặc bổ sung cổng `B-human-v4` mới).
- Nếu sau Bước 5 trên `test_gallery`, điểm `model_only` hoặc `system` vẫn chưa đạt ngưỡng 0.85 R@10 / 0.75 mAP, báo cáo sẽ dừng ở mức hiện trạng và ghi nhận phương án **"cần train"** để người dùng quyết định.
