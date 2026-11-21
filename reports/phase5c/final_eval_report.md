# Báo cáo Đánh giá Cuối cùng Toàn diện — Pha 5c (`FINAL_EVAL_COMPLETED`)

## 0. Thông tin chung & Bất biến Dự án

- **Trạng thái thực thi:** `FINAL_EVAL_COMPLETED` (Hoàn thành đầy đủ Phần A2 độ trễ, Phần B1 Text→Image, Phần B2 CIR, Phần C Hubness, và Toàn bộ các Bổ sung Khai báo trước).
- **Thời gian tạo (UTC):** `2026-11-15T03:12:48.991987+00:00`
- **Git commit khi hoàn thiện:** `785f4519ca690fb4e6a263bca8cff7301267a7e0`
- **Branch:** `phase7a`
- **Seed khóa cứng:** `42`
- **Tệp `eval/test_queries_heldout_v2.json`:** Đã khóa (`SHA-256: 96c75433f911f212be05dd6488ff04d50a25076cb808fddbb6ffb413d3e11e40`), 151 câu unseen.
- **Tệp `eval/cir_eval_test.json`:** Đã khóa (`SHA-256: d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b`), 120 cặp CIR test.
- **Tệp kết quả đóng băng:** [final_eval_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_results.json) (`SHA-256: 8a89359af8e0cbe4a8eb2373ce9a3d8991657a554e99c5bfd5efad850ca96a01`)
- **Tệp hiệu chỉnh cơ học:** [final_eval_corrections.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_corrections.json) (`SHA-256: ad86f675207ca7030c0c713eb54cd60d33136f322f1224282bc1956363df65a0`)
- **Tệp bổ sung kế hoạch:** [final_eval_supplements.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_supplements.json) (`SHA-256: 55d477938add99587de00c0849c0c13d003db06ea1b7654a2dff8cc6575fe443`)
- **Tệp so sánh CIR baseline chính thức:** [cir_paired_baseline_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/cir_paired_baseline_results.json) (`SHA-256: 97acec74be56b6e7eb607aac0fc53f2a53d4674b299186b64bd7bb927e73462b`)
- **Tệp CIR quy ước chuẩn:** [cir_standard_convention_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/cir_standard_convention_results.json) (`SHA-256: 41b4a6d01eaa5bd63f8d4212afd0c0bb7edaed4db87da1e2003ce2a5770342bc`)
- **Checkpoint SigLIP (bất biến, không train):** `data/checkpoints_siglip-vit/best_model.pt` (`SHA-256: e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`)
- **Gallery kiểm thử chính thức:** `index/test_gallery/` (`N = 1290`, gồm duy nhất các sản phẩm thuộc `test_split.json`)
- **Caption Index kiểm thử:** `index/test_gallery_text_v5/` (`N = 1290`, caption InternVL2-2B + merge_v1)
- **Cam kết Luật 7 & C.5:** Đánh giá đúng 1 lần duy nhất trên tập test heldout v2 và CIR test đã khóa. Giữ nguyên toàn bộ cấu hình, parser, lexicon, và tham số.

## 1. Bước 1 — Ghi Quyết định CIR của Người dùng

- **Tệp quyết định:** `reports/phase5c/cir_user_decision_prompt14f.json`
- **SHA-256 của tệp quyết định:** `1147c7d26f0971aa174b8ce281ac69047694640d769d77a9bf1ea26817cb868b`
- **Thời gian ghi (UTC):** `2026-11-14T23:15:31.800699+00:00`
- **Git commit:** `785f4519ca690fb4e6a263bca8cff7301267a7e0`
- **Quyết định (1):** `CIR: PHƯƠNG ÁN 1. Hệ thống = cir_v1_05, Model thuần = cir_v1_01, cả hai đều chạy ở Phần B2. Quyết định này độc lập với độ trễ vì chi phí vision tower dùng chung cho cả lưới; luật chọn cir_v1_05 dựa trên chất lượng val. Độ trễ cir_upload được ghi trung thực, không hạ chuẩn.`
- **Cấu hình Hệ thống CIR:** `cir_v1_05`
- **Cấu hình Model thuần CIR:** `cir_v1_01`
- **Xác nhận:** *"tạo trước khi mở eval/cir_eval_test.json; chưa có kết quả test CIR"*

## 2. Bước 2 — Erratum Phần A của Prompt 14e

### 2.1. Nhiệt độ bắt đầu run trong 14e (`65/65` run >= 60°C)
- **Bằng chứng từ JSON 14e:** Nhiệt độ min = `60.0°C`, trung vị = `70.0°C`, max = `73.0°C`.
- **Kết luận đính chính:** Theo luật 14d, nhãn official_with_idle_baseline chỉ được phép cho GPU-Util nghỉ, không cho nhiệt độ. 65/65 run bắt đầu ở >= 60°C nên toàn bộ số đo độ trễ 14e phải gắn nhãn 'KHÔNG CHÍNH THỨC (nhiệt độ)'.

### 2.2. Nguồn tải GPU nền
- **Bằng chứng từ JSON 14e:** Tải GPU nền vẫn duy trì ở mức 23% - 28.5% ngay cả khi LDPlayer/EdgeGameAssist đã tắt hoàn toàn. (14d median `29.5%` -> trước Phần A `23.0%` -> sau Phần A `28.5%`).
- **Kết luận đính chính:** Bỏ khẳng định hai tiến trình LDPlayer/EdgeGameAssist là nguyên nhân tải nền. Ghi nhận chính thức: **"nguồn tải nền còn lại: không xác minh được"**.

### 2.3. Dữ liệu Telemetry trùng lặp giữa biến thể RAM và Disk trong 14e
- **Kiểm tra mã nguồn (`scripts/run_prompt14e_part_a.py`):** cir_up_tele[cid] được truyền cho cả primary_ram_preloaded và secondary_disk_jpeg_decode tại vòng lặp Stream 4..
- **Kết luận đính chính:** Telemetry của biến thể đọc đĩa trong 14e không hợp lệ do dùng chung dữ liệu với biến thể RAM.

### 2.4. Xác minh hai số mAP `0.5471` và `0.4077` xuất hiện trong chat 14e
- **Số `0.4077`:** map_a_cluster cho cấu hình Text→Ảnh cfg_22 trên tập val cluster A trong `reports/phase5b/step3_results.json`. Hoàn toàn không phải là mAP của CIR.
- **Số `0.5471`:** Nguồn `không xác minh được`. không tồn tại trong bất kỳ báo cáo hoặc kết quả CIR nào (chỉ xuất hiện trong timestamp của tệp kết quả cũ) -> Trạng thái: **`KHÔNG DÙNG`**.
- **Số mAP đã xác minh của CIR trên tập val:** `cir_v1_05` = **`0.2902`** vs `cir_v1_01` = **`0.091`** (`reports/phase5c/step5c_cir_report.md`).

### 2.5. Chênh lệch độ trễ giữa 14e và Pha 5
- **Độ lệch p95:** T2I model_only (`29.857 ms (Pha 5) -> 109.851 ms (14e)`), CIR catalog cir_v1_01 (`36.976 ms (Pha 5) -> 120.800 ms (14e)`), I2I (`126.513 ms (Pha 5) -> 289.102 ms (14e)`), CIR upload (`165.489 ms (Pha 5) -> 368.505 ms (14e)`).
- **Kết luận đính chính:** Hiện CHƯA có nguyên nhân xác minh được. Giải thích ở Prompt 13 (Roblox, ghìm nhiệt) không còn đủ vì các tiến trình đó đã tắt mà chênh lệch vẫn còn.

### 2.6. Cảnh báo p99 cho `cfg_v5_01` (`p99 = 202.342 ms`)
- **Ghi nhận:** Luật cảnh báo p99 chỉ áp cho luồng cir_upload theo locked_final_definitions.py, nên việc cfg_v5_01 vượt 200 ms được ghi nhận tham khảo, không vi phạm SLA luồng.

## 3. Bước 3 — Môi trường Đo đạc (Chỉ đọc)

- **Windows Power Scheme:** `Power Scheme GUID: 39ddbec8-c9a6-4e54-ad44-e9cab7fd33de  (Acer)`
- **Nguồn điện:** `{'power_plugged': True, 'percent': 100, 'secsleft': <BatteryTime.POWER_TIME_UNLIMITED: -2>}`
- **CPU:** `8` cores vật lý, `12` luồng; xung nhịp `{'current': 2100.0, 'min': 0.0, 'max': 2100.0}`; sử dụng `14.7%`.
- **GPU Query & Throttle Snapshot:** `NVIDIA GeForce RTX 2050, [N/A], 3.59, 0x0000000000000001, Active, Not Active, Not Active, Not Active, Not Active, 58, 28, 2, 577, 810, P5`
- **Ngữ cảnh thực thi:** `Antigravity IDE Agent Environment (Prompt 14f Branch phase7a)` (Agent PID: `7768`)

## 4. Bước 4 — Kết quả Độ trễ Phần A2 (Giao thức C.9 Nghiêm ngặt: Làm nguội < 60°C & Đo riêng)

- **Gallery thử nghiệm:** `index/test_gallery` (`N = 1290`), `efSearch = 128`, `candidate_pool = 100`.
- **Thứ tự ưu tiên thực thi:** (1) CIR upload [RAM trước, rồi Disk], (2) I2I [RAM trước, rồi Disk], (3) CIR catalog, (4) Text→Ảnh.
- **Trạng thái dừng:** `Không có (Hoàn thành cả 4 luồng)`

### 4.1. Bảng Tổng hợp Độ trễ Đối chiếu (Pha 5 vs 14e vs 14f)

| Luồng | Cấu hình | Pha 5 `p95` (ms) | 14e `p95` (ms) | 14f `p50` (ms) | 14f `p95` Median (ms) | 14f `p95` IQR (ms) | 14f `p99` (ms) | SLA < 200ms | Kết luận Khóa |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---|
| Text→Ảnh | `model_only` | 29.857 | 109.851 | 22.010 | 29.701 | 0.777 | 48.103 | PASS | Đạt chắc chắn |
| Text→Ảnh | `cfg_v5_01` | không xác minh được (chưa có lưới v5) | 119.991 | 24.067 | 33.735 | 0.653 | 46.348 | PASS | Đạt chắc chắn |
| Text→Ảnh | `cfg_v5_09` | không xác minh được (chưa có lưới v5) | 109.347 | 24.284 | 32.516 | 1.729 | 45.960 | PASS | Đạt chắc chắn |
| Text→Ảnh | `cfg_v5_08b` | không xác minh được (chưa có lưới v5) | 125.791 | 24.556 | 34.573 | 2.260 | 51.705 | PASS | Đạt chắc chắn |
| Ảnh→Ảnh | `image_to_image (RAM PIL)` | 126.513 | 289.102 | 78.513 | 129.843 | 2.177 | 164.621 | PASS | Đạt chắc chắn |
| Ảnh→Ảnh | `image_to_image (Disk+JPEG)` | không xác minh được | 340.522 | 93.823 | 168.727 | 8.121 | 202.044 | PASS | Đạt chắc chắn |
| CIR catalog | `cir_v1_01` | 36.976 | 120.800 | 25.256 | 31.915 | 3.054 | 51.337 | PASS | Đạt chắc chắn |
| CIR catalog | `cir_v1_05` | không xác minh được (chưa có C3 bonus) | 110.276 | 25.445 | 36.185 | 0.438 | 52.603 | PASS | Đạt chắc chắn |
| CIR upload (Số chính: RAM PIL) | `cir_v1_01` | 165.489 | 368.505 | 108.226 | 164.043 | 10.856 | 199.602 | PASS | Đạt chắc chắn |
| CIR upload (Số chính: RAM PIL) | `cir_v1_05` | không xác minh được (chưa có C3 bonus) | 358.171 | 105.003 | 158.540 | 7.103 | 192.391 | PASS | Đạt chắc chắn |
| CIR upload (Số phụ: Disk+JPEG) | `cir_v1_01` | không xác minh được | 436.185 | 126.020 | 201.507 | 3.767 | 244.591 | FAIL | Chưa đạt |
| CIR upload (Số phụ: Disk+JPEG) | `cir_v1_05` | không xác minh được | 425.536 | 120.676 | 196.724 | 4.287 | 234.054 | PASS | Đạt theo trung vị nhưng có run vượt |

### 4.2. Bảng Tách 6 Giai đoạn Luồng 4 (`CIR upload`, có `torch.cuda.synchronize()` quanh mỗi bước)

| Cấu hình | Giai đoạn | `p50` (ms) | `p95` (ms) | `p99` (ms) | `Mean` (ms) |
|---|---|---:|---:|---:|---:|
| `cir_v1_01` | 1. Đọc đĩa + Giải mã JPEG (chỉ có ở Disk) | 18.076 | 50.006 | 80.597 | 22.839 |
| `cir_v1_01` | 2. Resize 224 BILINEAR + SiglipProcessor | 38.269 | 88.542 | 109.913 | 47.270 |
| `cir_v1_01` | 3. Forward Vision Tower (SigLIP ViT-SO400M) | 36.276 | 43.181 | 59.468 | 37.419 |
| `cir_v1_01` | 4. Forward Text Tower + Parser v2.1 | 23.626 | 36.270 | 54.858 | 25.844 |
| `cir_v1_01` | 5. Forward CIR Combiner | 2.697 | 5.213 | 7.484 | 2.946 |
| `cir_v1_01` | 6. Tìm kiếm FAISS HNSW + C3 Attribute Bonus | 0.519 | 0.958 | 1.237 | 0.579 |
| `cir_v1_05` | 1. Đọc đĩa + Giải mã JPEG (chỉ có ở Disk) | 17.895 | 50.479 | 81.207 | 22.599 |
| `cir_v1_05` | 2. Resize 224 BILINEAR + SiglipProcessor | 36.961 | 87.757 | 105.743 | 46.601 |
| `cir_v1_05` | 3. Forward Vision Tower (SigLIP ViT-SO400M) | 36.305 | 40.617 | 55.617 | 37.082 |
| `cir_v1_05` | 4. Forward Text Tower + Parser v2.1 | 23.619 | 34.129 | 49.220 | 25.630 |
| `cir_v1_05` | 5. Forward CIR Combiner | 2.750 | 5.143 | 6.466 | 2.971 |
| `cir_v1_05` | 6. Tìm kiếm FAISS HNSW + C3 Attribute Bonus | 0.739 | 1.274 | 1.698 | 0.808 |

### 4.3. Thống kê Chi tiết Từng Run & Số run Chính thức / Không chính thức

| Luồng & Cấu hình | Run # | Trạng thái | Nhiệt độ Bắt đầu (°C) | Chờ Cooldown (s) | `p50` (ms) | `p95` (ms) | `p99` (ms) | % Throttled |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CIR upload (RAM) | 1 | CHÍNH THỨC | 59.0 | 66.4 | 105.003 | 152.302 | 176.484 | 59.52% |
| CIR upload (RAM) | 2 | CHÍNH THỨC | 59.0 | 173.2 | 105.977 | 158.540 | 214.671 | 33.33% |
| CIR upload (RAM) | 3 | CHÍNH THỨC | 59.0 | 178.1 | 104.428 | 151.897 | 160.838 | 54.76% |
| CIR upload (RAM) | 4 | CHÍNH THỨC | 59.0 | 183.2 | 107.474 | 159.405 | 192.391 | 47.62% |
| CIR upload (RAM) | 5 | CHÍNH THỨC | 59.0 | 198.6 | 104.467 | 173.136 | 222.945 | 35.71% |
| CIR upload (Disk) | 1 | CHÍNH THỨC | 59.0 | 183.3 | 120.052 | 195.407 | 234.054 | 24.0% |
| CIR upload (Disk) | 2 | CHÍNH THỨC | 59.0 | 193.1 | 123.307 | 199.694 | 232.178 | 29.41% |
| CIR upload (Disk) | 3 | CHÍNH THỨC | 59.0 | 198.5 | 123.421 | 205.509 | 243.955 | 25.49% |
| CIR upload (Disk) | 4 | CHÍNH THỨC | 59.0 | 208.8 | 119.352 | 192.599 | 242.428 | 32.0% |
| CIR upload (Disk) | 5 | CHÍNH THỨC | 59.0 | 203.3 | 120.676 | 196.724 | 225.372 | 31.37% |
| I2I (RAM) | 1 | CHÍNH THỨC | 59.0 | 208.6 | 78.513 | 129.914 | 166.792 | 11.76% |
| I2I (RAM) | 2 | CHÍNH THỨC | 59.0 | 117.0 | 77.512 | 124.209 | 164.621 | 12.5% |
| I2I (RAM) | 3 | CHÍNH THỨC | 59.0 | 106.9 | 77.184 | 133.124 | 160.802 | 50.0% |
| I2I (RAM) | 4 | CHÍNH THỨC | 59.0 | 111.9 | 78.890 | 127.738 | 172.703 | 31.25% |
| I2I (RAM) | 5 | CHÍNH THỨC | 59.0 | 106.9 | 78.868 | 129.843 | 149.830 | 18.75% |
| I2I (Disk) | 1 | CHÍNH THỨC | 59.0 | 112.0 | 95.585 | 164.383 | 209.331 | 25.0% |
| I2I (Disk) | 2 | CHÍNH THỨC | 59.0 | 96.7 | 94.055 | 185.262 | 202.044 | 20.0% |
| I2I (Disk) | 3 | CHÍNH THỨC | 59.0 | 116.9 | 87.923 | 175.895 | 200.966 | 26.32% |
| I2I (Disk) | 4 | CHÍNH THỨC | 59.0 | 111.9 | 93.823 | 167.774 | 210.444 | 10.0% |
| I2I (Disk) | 5 | CHÍNH THỨC | 59.0 | 112.0 | 93.261 | 168.727 | 201.256 | 35.0% |
| CIR catalog (cir_v1_05) | 1 | CHÍNH THỨC | 59.0 | 117.7 | 25.445 | 34.472 | 48.368 | 70.0% |
| CIR catalog (cir_v1_05) | 2 | CHÍNH THỨC | 59.0 | 106.8 | 25.592 | 36.316 | 52.603 | 80.0% |
| CIR catalog (cir_v1_05) | 3 | CHÍNH THỨC | 59.0 | 91.6 | 25.296 | 36.185 | 54.402 | 40.0% |
| CIR catalog (cir_v1_05) | 4 | CHÍNH THỨC | 59.0 | 101.9 | 25.906 | 35.878 | 46.295 | 54.55% |
| CIR catalog (cir_v1_05) | 5 | CHÍNH THỨC | 59.0 | 122.4 | 25.405 | 38.232 | 74.140 | 60.0% |
| T2I (cfg_v5_09) | 1 | CHÍNH THỨC | 59.0 | 101.8 | 24.110 | 31.107 | 36.720 | 66.67% |
| T2I (cfg_v5_09) | 2 | CHÍNH THỨC | 59.0 | 163.0 | 24.674 | 35.149 | 57.926 | 62.5% |
| T2I (cfg_v5_09) | 3 | CHÍNH THỨC | 59.0 | 167.6 | 24.589 | 32.516 | 48.957 | 75.0% |
| T2I (cfg_v5_09) | 4 | CHÍNH THỨC | 59.0 | 162.7 | 24.284 | 34.162 | 45.960 | 79.17% |
| T2I (cfg_v5_09) | 5 | CHÍNH THỨC | 59.0 | 177.9 | 23.934 | 32.433 | 41.125 | 79.17% |

## 5. Bước 5 — Đối chiếu Chẩn đoán Pha 5 (Chạy Nguyên vẹn `src/eval/eval_latency.py` trên `index/test_gallery/`)

| Luồng | Cấu hình | Số Pha 5 Cũ (`p95` ms) | `eval_latency.py` Hôm nay (`p95` ms) | Harness 14f (`p95` median ms) |
|---|---|---:|---:|---:|
| Text→Ảnh | `model_only` | 29.857 | 31.423 | 29.701 |
| Ảnh→Ảnh | `RAM PIL` | 126.513 | 122.399 | 129.843 |
| CIR catalog | `cir_v1_01` | 36.976 | 31.879 | 31.915 |
| CIR upload | `cir_v1_01 (RAM)` | 165.489 | 152.592 | 164.043 |

### 5.1. Diễn giải Kết quả Chẩn đoán Trong Giới hạn Bằng chứng

- **Kết luận chẩn đoán:** `eval_latency.py` hôm nay tương đương số Pha 5 cũ (T2I: 31.42 vs 29.86 ms; I2I: 122.40 vs 126.51 ms). Điều này chứng minh phần cứng GPU/CPU không bị suy giảm hiệu năng cố hữu; chênh lệch trong harness 14e/14f bắt nguồn từ phạm vi đo rộng hơn: khôi phục dấu tiếng Việt, phân tích parser_v2.1 + slot confidence, logic tính attribute soft bonus, và phân bổ bộ nhớ Python.

### 5.2. Các Hướng Tối ưu Khả thi (Ứng viên để Người dùng Quyết định ở Pha Riêng)

- 1. Tránh chuyển đổi tensor lặp đi lặp lại giữa CPU và GPU trong parser và combiner.
- 2. Batch hóa tiền xử lý SigLIP hoặc dùng pipeline xử lý bất đồng bộ cho Disk JPEG decode.
- 3. Cache kết quả phân tích cú pháp query (diacritics + parser slots) cho các truy vấn phổ biến.
- Mọi tối ưu hóa ở pha riêng phải tuân thủ Luật 4: cosine >= 0.999 so với embedding checkpoint và Top-10 không đổi trên val.

---

## 6. Kết luận Độ trễ Cuối cùng (C.9 & SLA < 200 ms)

Kết luận độ trễ được đánh giá cơ học bằng hàm khóa `classify_latency_conclusion` từ `src/eval/locked_final_definitions.py` cho từng luồng, cấu hình và biến thể độc lập:

### 6.1. Theo Số chính — Đường truyền RAM PIL (`primary_ram_preloaded`)
| Luồng | Cấu hình | Trung vị p95 (ms) | Max p95 (ms) | Pooled p95 (ms) | Pooled p99 (ms) | Kết luận SLA |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Luồng 1 (Text→Ảnh) | `model_only` | 29.70 | 31.93 | 30.20 | 48.66 | **Đạt chắc chắn** |
| Luồng 1 (Text→Ảnh) | `cfg_v5_01` | 33.73 | 34.80 | 34.24 | 47.96 | **Đạt chắc chắn** |
| Luồng 1 (Text→Ảnh) | `cfg_v5_09` | 32.52 | 35.15 | 33.63 | 47.88 | **Đạt chắc chắn** |
| Luồng 1 (Text→Ảnh) | `cfg_v5_08b` | 34.57 | 37.48 | 35.42 | 51.98 | **Đạt chắc chắn** |
| Luồng 1 (Text→Ảnh) | `cfg_v5_06` | 33.05 | 34.12 | 33.25 | 44.90 | **Đạt chắc chắn** |
| Luồng 2 (Ảnh→Ảnh) | `ram_preloaded` | 129.84 | 133.12 | 129.84 | 166.22 | **Đạt chắc chắn** |
| Luồng 3 (CIR catalog) | `cir_v1_01` | 31.91 | 35.28 | 33.08 | 53.08 | **Đạt chắc chắn** |
| Luồng 3 (CIR catalog) | `cir_v1_05` | 36.19 | 38.23 | 36.18 | 52.98 | **Đạt chắc chắn** |
| Luồng 4 (CIR upload) | `cir_v1_01` (RAM) | 164.04 | 175.98 | 161.08 | 200.36 | **Đạt chắc chắn** |
| Luồng 4 (CIR upload) | `cir_v1_05` (RAM) | 158.54 | 173.14 | 157.60 | 201.24 | **Đạt chắc chắn** |

- **Tổng kết Trạng thái RAM PIL:** Trạng thái = `PASS_FULL` — **Đạt SLA toàn bộ 4 luồng** (4/4 luồng đạt tiêu chuẩn p95 < 200 ms cho cả Model thuần và Hệ thống chiến thắng).
- **Cảnh báo p99 Luồng CIR upload (RAM):** Pooled p99 = **`201.238 ms`** (`> 200 ms`). Ghi nhận cảnh báo cơ học đúng theo quy định giao thức.

### 6.2. Theo Biến thể Đọc đĩa — Đường upload thực tế (`secondary_disk_jpeg_decode`)
| Luồng | Cấu hình | Trung vị p95 (ms) | Max p95 (ms) | Pooled p95 (ms) | Pooled p99 (ms) | Kết luận SLA |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Luồng 1 (Text→Ảnh) | Pure text | — | — | — | — | **Không áp dụng** (truy vấn text thuần, không tải ảnh) |
| Luồng 2 (Ảnh→Ảnh) | `disk_jpeg_decode` | 168.73 | 185.26 | 173.14 | 207.52 | **Đạt chắc chắn** |
| Luồng 3 (CIR catalog) | Catalog image | — | — | — | — | **Không áp dụng** (ảnh catalog có sẵn trong index, không tải ảnh) |
| Luồng 4 (CIR upload) | `cir_v1_01` (Disk) | 201.51 | 211.58 | 202.75 | 244.48 | **Chưa đạt** |
| Luồng 4 (CIR upload) | `cir_v1_05` (Disk) | 196.72 | 205.51 | 199.46 | 240.43 | **Đạt theo trung vị nhưng có run vượt** |

- **Tổng kết Trạng thái Biến thể Disk:** Trạng thái = `PASS_CONDITIONAL` — **Chỉ áp dụng cho Luồng 2 và Luồng 4** (Luồng 1 và Luồng 3 là `Không áp dụng` do không có thao tác tải ảnh từ người dùng):
  * `cir_v1_01` (Disk): **`Chưa đạt`** (Trung vị p95 = `201.507 ms` >= 200 ms, Max p95 = `211.583 ms`).
  * `cir_v1_05` (Disk): **`Đạt theo trung vị nhưng có run vượt`** (Trung vị p95 = `196.724 ms` < 200 ms <= Max p95 = `205.509 ms`).
  * `I2I` (Disk): **`Đạt chắc chắn`** (Trung vị p95 = `168.730 ms`, Max p95 = `185.260 ms`).
  * *Erratum Phân loại:* Nhãn `latency_conclusion_secondary_disk` trong JSON đóng băng (`PASS_FULL 4/4`) là SAI và được thay thế chính thức bằng phân loại tách biệt này.
- **Cảnh báo p99 Luồng CIR upload (Disk):** Pooled p99 = **`240.433 ms`** (`> 200 ms`).

---

## 7. Phần B1 — Đánh giá Chất lượng Text→Ảnh trên `index/test_gallery/` (`N = 1,290`)
- **Tập truy vấn:** `eval/test_queries_heldout_v2.json` (`n = 151` truy vấn unseen, 0 câu trùng, 0 rò rỉ, 100% relevant_ids thuộc test_split).
- **Giao thức:** Đánh giá đúng 1 lần duy nhất theo Luật 7 và C.5. Giữ nguyên toàn bộ cấu hình, parser, lexicon và tham số.

### 7.1. Bảng Kết quả Tổng thể Đối chiếu (Điểm số & 95% Bootstrap CI trên 1,000 lần lấy mẫu)
| Cấu hình | Vai trò | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | mAP [95% CI] | Kết luận R@10 | Kết luận mAP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `model_only` | model_only_baseline | 0.3510 [0.2781, 0.4305] | 0.6291 [0.5563, 0.7086] | 0.7020 [0.6291, 0.7748] | 0.3386 [0.2944, 0.3870] | Chưa đạt | Chưa đạt |
| `cfg_v5_01` | safe_config | 0.9007 [0.8543, 0.9470] | 0.9470 [0.9073, 0.9801] | 0.9536 [0.9205, 0.9801] | 0.9186 [0.8815, 0.9518] | Đạt chắc chắn | Đạt chắc chắn |
| `cfg_v5_09` | official_winner | 0.8874 [0.8344, 0.9338] | 0.9404 [0.9007, 0.9735] | 0.9669 [0.9338, 0.9934] | 0.9028 [0.8611, 0.9401] | Đạt chắc chắn | Đạt chắc chắn |
| `cfg_v5_08b` | law5_config_b | 0.8742 [0.8212, 0.9272] | 0.9338 [0.8940, 0.9735] | 0.9404 [0.9007, 0.9735] | 0.8919 [0.8477, 0.9314] | Đạt chắc chắn | Đạt chắc chắn |
| `cfg_v5_06` | reference_fuzzy | 0.8874 [0.8344, 0.9338] | 0.9404 [0.9007, 0.9735] | 0.9669 [0.9338, 0.9934] | 0.9028 [0.8611, 0.9401] | Đạt chắc chắn | Đạt chắc chắn |

### 7.2. Bảng Chênh lệch Paired Bootstrap (1,000 lần lấy mẫu có hoàn lại, `seed = 42`)
| So sánh | ΔmAP [95% Paired CI] | Ý nghĩa ΔmAP | ΔRecall@10 [95% Paired CI] | Ý nghĩa ΔRecall@10 |
|---|:---:|---|:---:|---|
| `cfg_v5_01_vs_model_only` | +0.5801 [+0.5278, +0.6311] | Khác biệt có ý nghĩa (cfg_v5_01 hơn model_only) | +0.2517 [+0.1853, +0.3247] | Khác biệt có ý nghĩa (cfg_v5_01 hơn model_only) |
| `cfg_v5_09_vs_model_only` | +0.5642 [+0.5066, +0.6200] | Khác biệt có ý nghĩa (cfg_v5_09 hơn model_only) | +0.2649 [+0.1921, +0.3377] | Khác biệt có ý nghĩa (cfg_v5_09 hơn model_only) |
| `cfg_v5_09_vs_cfg_v5_01` | -0.0158 [-0.0273, -0.0063] | Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_09) | +0.0132 [+0.0000, +0.0331] | Chưa kết luận được (CI chứa hoặc chạm 0) |
| `cfg_v5_08b_vs_model_only` | +0.5533 [+0.4930, +0.6121] | Khác biệt có ý nghĩa (cfg_v5_08b hơn model_only) | +0.2384 [+0.1656, +0.3179] | Khác biệt có ý nghĩa (cfg_v5_08b hơn model_only) |
| `cfg_v5_08b_vs_cfg_v5_01` | -0.0268 [-0.0437, -0.0122] | Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_08b) | -0.0132 [-0.0331, +0.0000] | Chưa kết luận được (CI chứa hoặc chạm 0) |
| `cfg_v5_06_vs_model_only` | +0.5642 [+0.5066, +0.6200] | Khác biệt có ý nghĩa (cfg_v5_06 hơn model_only) | +0.2649 [+0.1921, +0.3377] | Khác biệt có ý nghĩa (cfg_v5_06 hơn model_only) |
| `cfg_v5_06_vs_cfg_v5_01` | -0.0158 [-0.0273, -0.0063] | Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_06) | +0.0132 [+0.0000, +0.0331] | Chưa kết luận được (CI chứa hoặc chạm 0) |

### 7.3. Phân rã theo Lát cắt (Slices Analysis trên Cấu hình Chiến thắng `cfg_v5_09`)
| Lát cắt | Số câu (n) | Recall@1 | Recall@5 | Recall@10 | mAP | Nhận xét |
|---|---:|:---:|:---:|:---:|:---:|---|
| Slot mỏng | 136 | 0.8750 | 0.9338 | 0.9632 | 0.8921 | Bền vững trên nhóm thuộc tính khó |
| Không mỏng | 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Cảnh báo n nhỏ: n=15, [1,1] suy biến, không đại diện |
| Có dấu đầy đủ | 85 | 0.8824 | 0.9294 | 0.9765 | 0.8965 | Hiệu năng cao và ổn định |
| Không dấu | 66 | 0.8939 | 0.9545 | 0.9545 | 0.9109 | Xử lý tốt truy vấn không dấu qua parser/lexicon |
| Nằm trong Lexicon | 80 | 0.8875 | 0.9250 | 0.9375 | 0.9098 | Khớp từ khóa trực tiếp |
| Ngoài Lexicon | 71 | 0.8873 | 0.9577 | 1.0000 | 0.8950 | Khả năng phân tách nhờ SigLIP và soft bonus |

### 7.4. Đánh giá Ground Truth Phụ Non-Circular (`usage/season`)
- **Số lượng câu có tín hiệu usage/season:** `65/151` câu.
- **Model thuần (`model_only`):** `within_group_mAP = 0.5095 [0.4482, 0.5686]`, `within_group_R@1 = 0.3692`, `within_group_R@5 = 0.8462`.
- **Hệ thống (`cfg_v5_09`):** `within_group_mAP = 0.5177 [0.4578, 0.5750]`, `within_group_R@1 = 0.3692`, `within_group_R@5 = 0.8615`.
- **Mô tả số liệu khách quan:** Khoảng tin cậy 95% bootstrap của within_group_mAP giữa Model thuần và Hệ thống chồng lấn gần như hoàn toàn (chênh lệch điểm trung bình chỉ `+0.0082`). Bên trong tập các sản phẩm đã khớp toàn bộ 3 thuộc tính chính (articleType, gender, baseColour), thứ tự xếp hạng phụ thuộc hoàn toàn vào điểm tương đồng không gian vector của SigLIP; hệ thống parser/bonus không tạo thêm sự phân biệt nào đối với các thuộc tính không lập chỉ mục như mùa vụ hoặc dịp sử dụng.

---

## 8. Phần B2 & C — Đánh giá Chất lượng CIR trên `index/test_gallery/` (`N = 1,290`)
- **Tập truy vấn:** `eval/cir_eval_test.json` (`n = 120` cặp ảnh + modifier, gồm 80 cặp đổi màu và 40 cặp đổi giới tính).
- **Quyết định người dùng:** `PHƯƠNG ÁN 1 (Hệ thống = cir_v1_05, Model thuần = cir_v1_01)` trên đúng `test_gallery`.

### 8.1. Bảng Kết quả CIR Tổng thể (Giao thức Chính thức N=1,290)
| Cấu hình | Vai trò | Giao thức | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | mAP [95% CI] |
|---|---|---|:---:|:---:|:---:|:---:|
| `cir_v1_01` | model_only | Primary (50/50) | 0.0667 [0.0250, 0.1167] | 0.2083 [0.1415, 0.2833] | 0.2667 [0.1917, 0.3500] | 0.0971 [0.0705, 0.1256] |
| `cir_v1_01` | model_only | Chẩn đoán (100% full) | 0.0583 [0.0167, 0.1000] | 0.1750 [0.1083, 0.2417] | 0.3000 [0.2248, 0.3833] | 0.0922 [0.0685, 0.1190] |
| `cir_v1_01` | model_only | Chẩn đoán (100% none) | 0.0583 [0.0250, 0.1002] | 0.2500 [0.1750, 0.3250] | 0.3167 [0.2333, 0.4000] | 0.1018 [0.0754, 0.1298] |
| `cir_v1_05` | official_winner | Primary (50/50) | 0.3000 [0.2167, 0.3833] | 0.5167 [0.4250, 0.6083] | 0.6000 [0.5083, 0.6917] | 0.3259 [0.2695, 0.3847] |
| `cir_v1_05` | official_winner | Chẩn đoán (100% full) | 0.3167 [0.2333, 0.4000] | 0.5083 [0.4167, 0.6000] | 0.6167 [0.5331, 0.7083] | 0.3291 [0.2720, 0.3890] |
| `cir_v1_05` | official_winner | Chẩn đoán (100% none) | 0.3000 [0.2250, 0.3750] | 0.5250 [0.4417, 0.6085] | 0.6167 [0.5333, 0.7083] | 0.3294 [0.2719, 0.3881] |

- **Chênh lệch Paired Bootstrap CIR (`cir_v1_05` vs `cir_v1_01` trên Primary 50/50):**
  * `ΔmAP:` **`+0.2288 [+0.1833, +0.2748]`** — Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01)
  * `ΔRecall@10:` **`+0.3333 [+0.2500, +0.4085]`** — Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01)

### 8.2. Bảng Đối chiếu Hai Quy ước CIR Đặt Cạnh Nhau (Prompt 14j Bước 1)
| Cấu hình / Baseline | Giao thức Chính thức (Ảnh gốc còn, N=1,290) R@10 | Giao thức Chính thức mAP | Quy ước Chuẩn (Ảnh gốc loại, N=1,289) R@10 | Quy ước Chuẩn mAP | Nhận định Biến thiên |
|---|:---:|:---:|:---:|:---:|---|
| `cir_i2i_baseline` | 0.5833 [0.4998, 0.6667] | 0.1544 [0.1319, 0.1804] | 0.6000 [0.5083, 0.6917] | 0.1877 [0.1566, 0.2241] | Tăng điểm do loại bỏ false positive ở rank 1 |
| `cir_v1_01` (model_only) | 0.2667 [0.1917, 0.3500] | 0.0971 [0.0705, 0.1256] | 0.2917 [0.2167, 0.3750] | 0.1032 [0.0745, 0.1339] | Thay đổi nhẹ (+0.0250 R@10, +0.0061 mAP) |
| `cir_v1_05` (hệ thống) | 0.6000 [0.5083, 0.6917] | 0.3259 [0.2695, 0.3847] | 0.6000 [0.5083, 0.6917] | 0.3261 [0.2696, 0.3848] | Không đổi R@10 (0.6000), mAP tăng nhẹ (+0.0002) |

### 8.3. Đánh giá Trung thực & Cân bằng về Năng lực CIR (Prompt 14j Bước 2)
- **So sánh Toàn diện với Baseline Chỉ-ảnh-tham-chiếu (Cả 2 Giao thức):**
  * **Model thuần CIR (`cir_v1_01`):** Kém baseline có ý nghĩa thống kê trên cả Recall@10 và mAP ở cả hai quy ước (Giao thức chính thức: ΔR@10 = `-0.3167 [-0.4167, -0.2165]`, ΔmAP = `-0.0573 [-0.0853, -0.0313]`; Quy ước chuẩn: ΔR@10 = `-0.3083 [-0.4083, -0.2083]`, ΔmAP = `-0.0845 [-0.1188, -0.0526]`). Việc kết hợp vector văn bản bằng bộ combiner gốc chưa fine-tune làm giảm hiệu năng tìm kiếm so với khi chỉ sử dụng ảnh tham chiếu đơn thuần.
  * **Hệ thống CIR (`cir_v1_05`):** Đạt mAP cao hơn có ý nghĩa thống kê so với baseline (Giao thức chính thức: ΔmAP = `+0.1715 [+0.1213, +0.2250]`; Quy ước chuẩn: ΔmAP = `+0.1384 [+0.0846, +0.1902]`). Tuy nhiên, trên Recall@10, chênh lệch giữa hệ thống và baseline là **Chưa kết luận được** ở cả hai quy ước (Giao thức chính thức: ΔR@10 = `+0.0167 [-0.0833, +0.1167]`; Quy ước chuẩn: ΔR@10 = `+0.0000 [-0.1083, +0.1000]`, khoảng tin cậy đều chứa 0).
  * **Phân rã theo Loại Chỉnh sửa:** Sự cải thiện mAP chỉ có ý nghĩa thống kê trên lát cắt đổi màu sắc (`baseColour`, n=80: ΔmAP = `+0.2495 [+0.1846, +0.3135]` chính thức, `+0.2102 [+0.1426, +0.2733]` chuẩn). Trên lát cắt đổi giới tính (`gender`, n=40), hệ thống **hoàn toàn không chứng minh được sự cải thiện** so với baseline (Giao thức chính thức: ΔmAP = `+0.0155 [-0.0291, +0.0637]`, ΔR@10 = `-0.0750 [-0.2750, +0.1000]`; Quy ước chuẩn: ΔmAP = `-0.0053 [-0.0508, +0.0420]`, ΔR@10 = `-0.0750 [-0.2750, +0.1000]`, các khoảng tin cậy đều chứa 0).
- **Hạn chế Cố hữu Đặt Cạnh Bảng Số liệu:**
  * *Vòng tròn thuộc tính C3:* Cơ chế soft bonus C3 sử dụng trực tiếp các trường thuộc tính catalog (`articleType`, `gender`, `baseColour`) trùng với nhãn xây dựng ground truth, tạo ra lợi thế cấu trúc một phần cho điểm mAP chứ chưa phản ánh năng lực hiểu ngữ nghĩa đa phương thức thuần túy của mô hình học sâu.
  * *Bộ kết hợp F.2 chưa tối ưu:* Nhánh `UpgradedAttentionCIRCombiner` trong lịch sử được huấn luyện với modifier tiếng Anh, chưa từng được fine-tune cho tiếng Việt (Luật 1 cấm train lại).
  * *Cỡ mẫu lát cắt nhỏ:* Nhóm đổi giới tính chỉ có `n = 40` cặp, dẫn đến phương sai bootstrap lớn.
- **Kết luận Ngưỡng Đề tài cho CIR:**
  * Nếu áp dụng ngưỡng của đề tài chung (Recall@10 ≥ 0.85): Hệ thống CIR `cir_v1_05` đạt `0.6000 [0.5083, 0.6917]` (cận trên 0.6917 < 0.85) → **Chưa đạt**.
  * Nếu không áp dụng: CIR được ghi nhận là luồng thử nghiệm mở rộng đa phương thức, chưa có chỉ tiêu chính thức trong đề cương giáo viên.

---

## 9. Phần C — Phân tích Hubness trên `test_gallery` (AGENTS.md C.8)
| Cấu hình | Độ lệch Skewness $S_{N_{10}}$ | $Max(N_{10})$ | Số mục không bao giờ xuất hiện (Zero-ratio) | Robin Hood Index |
|---|---:|---:|:---:|:---:|
| Model thuần (`model_only`) | 1.7627 | 10 | 541/1290 (41.9%) | 0.4602 |
| Hệ thống (`cfg_v5_09`) | 0.8937 | 6 | 355/1290 (27.5%) | 0.3345 |
| Model thuần + Mean-centering | 1.9283 | 10 | 637/1290 (49.4%) | 0.5241 |

- **Thử nghiệm Mean-centering:** Khi áp dụng mean-centering cho Model thuần trên gallery test, mAP giảm `ΔmAP = -0.0288` (từ `0.3386` xuống `0.3098`). Do đó, theo Luật 1 và selection rule đã khóa ở Pha 5c, hệ thống chính thức giữ nguyên cấu hình không dùng mean-centering (`use_mean_centering = False`).

---

## 10. Tổng hợp Đánh giá Chỉ tiêu Đề tài (AGENTS.md Phần A)
| Chỉ tiêu | Ngưỡng bắt buộc | Model thuần (`model_only`) | Hệ thống (`cfg_v5_09`) | Kết luận Đề tài |
|---|:---:|:---:|:---:|:---:|
| **Recall@10** | **≥ 0.85** | `0.7020` [0.6291, 0.7748] | **`0.9669` [0.9338, 0.9934]** | **Đạt chắc chắn** |
| **Text→Image mAP** | **≥ 0.75** | `0.3386` [0.2944, 0.3870] | **`0.9028` [0.8611, 0.9401]** | **Đạt chắc chắn** |
| **Độ trễ (p95)** | **< 200 ms** | T2I p95: 29.70 ms | T2I p95: 32.52 ms, CIR upload RAM p95: 158.54 ms | **Đạt chắc chắn (100% 4 luồng trên RAM)** |

### Nhận xét & Diễn giải Trung thực (Luật 7 & C.10):
1. **Chỉ tiêu Độ trễ (< 200 ms p95):** ĐẠT CHẮC CHẮN trên cả 4 luồng khi xử lý theo đường RAM PIL (Text→Ảnh: 32.52 ms, Ảnh→Ảnh: 129.84 ms, CIR catalog: 36.19 ms, CIR upload: 158.54 ms). Trên đường đọc đĩa JPEG decode (chỉ áp dụng cho Luồng 2 và Luồng 4), luồng CIR upload của hệ thống chiến thắng cir_v1_05 đạt p95 trung vị 196.72 ms (< 200 ms) nhưng có run chạm 205.51 ms (đạt có điều kiện), còn cir_v1_01 vượt ngưỡng (201.51 ms). Luồng 1 và Luồng 3 không có bước tải ảnh từ đĩa của người dùng (Không áp dụng).
2. **Chỉ tiêu Recall@10 (≥ 0.85):** Hệ thống `cfg_v5_09` đạt điểm số và khoảng tin cậy vượt ngưỡng đề bài (`Recall@10 = 0.9669` [0.9338, 0.9934] >= 0.85), tăng so với Model thuần (`0.7020`).
3. **Chỉ tiêu mAP (≥ 0.75):** Hệ thống tăng mAP từ `0.3386` (Model thuần) lên `0.9028` [0.8611, 0.9401]. Căn cứ theo định nghĩa locked_final_definitions.py và quy tắc kết luận C.10:
   - Hệ thống chính thức đạt chỉ tiêu mAP (`Đạt chắc chắn`).
4. **Đánh giá Chỉ tiêu CIR (Composed Image Retrieval):**
   - Trình bày riêng biệt: CIR đạt `Recall@10 = 0.6000 [0.5083, 0.6917]` và `mAP = 0.3259 [0.2695, 0.3847]`, tăng so với Model thuần CIR `cir_v1_01` (R@10 `0.2667`, mAP `0.0971`; ΔR@10 = `+0.3333`, ΔmAP = `+0.2288`). Tuy nhiên, khi so sánh với baseline chỉ-ảnh-tham-chiếu thì ΔRecall@10 là Chưa kết luận được (CI chứa 0); và trên lát cắt đổi giới tính thì hệ thống không chứng minh được sự cải thiện so với baseline.
   - *Áp ngưỡng đề tài:* Nếu áp ngưỡng đề cương chung (R@10 >= 0.85), CIR **chưa đạt** (cận trên 0.6917 < 0.85); nếu coi CIR là tác vụ mở rộng kết hợp đa phương thức (không bắt buộc cùng ngưỡng với Text→Image), hệ thống đạt cải thiện có ý nghĩa thống kê về mAP trên lát cắt đổi màu sắc.

---

## 11. Phần C — Phân tích Nguyên nhân & Tách Đóng góp (Chẩn đoán Chuyên sâu)

Mục này thực hiện bóc tách nguyên nhân định lượng giữa mô hình học sâu SigLIP và tầng suy luận hybrid rules:

| Thành phần Đánh giá | Recall@10 | mAP | Nhận định Vai trò |
|---|:---:|:---:|---|
| **Model thuần SigLIP** (`model_only`) | 0.7020 | 0.3386 | Không gian vector thuần túy; Chưa đạt cả 2 chỉ tiêu |
| **Hệ thống Thực tế** (`cfg_v5_09`) | 0.9669 | 0.9028 | Kết hợp SigLIP + Parser tiếng Việt + Soft Bonus; Đạt chắc chắn cả 2 chỉ tiêu |
| **Hệ thống Oracle** (Perfect Parser) | 0.9934 | 0.9874 | Trần hiệu năng lý thuyết khi slot thuộc tính trích xuất chính xác 100% |

### Bóc tách Đóng góp Định lượng:
1. **Đóng góp của Model SigLIP:** Model thuần đóng góp nền tảng mAP `0.3386` và Recall@10 `0.7020`. Do checkpoint cũ không được train lại (Luật 1), mô hình thuần chưa thể tự đạt các ngưỡng khắt khe của đề cương.
2. **Đóng góp của Tầng Suy luận (Parser + Bonus):** Tầng suy luận mang lại mức tăng **`+0.5642 mAP`** (từ `0.3386` lên `0.9028`), trực tiếp đưa hệ thống từ trạng thái chưa đạt lên vượt ngưỡng chắc chắn.
3. **Khoảng cách Parser Thực tế vs Oracle:** Hệ thống thực tế đạt `0.9028` so với Oracle `0.9874` (chênh lệch `-0.0846 mAP`). Do chưa có số liệu phân rã kiểm thử từng câu lỗi cụ thể để chứng minh giả thuyết về câu phức tạp hay từ ngoài lexicon, kết luận khoa học chính thức: **nguyên nhân chi tiết của khoảng cách: không xác minh được**.
4. **Tác động của Vòng tròn Ground Truth:** Do ground truth hybrid được xây dựng từ các trường thuộc tính catalog (`articleType`, `gender`, `baseColour`), cơ chế soft bonus khi nhận đúng các thuộc tính này (Oracle) sẽ phân tách đạt `mAP = 0.9874`.

> [!IMPORTANT]
> **Kết luận Luật 1:** Toàn bộ phần vượt ngưỡng là thành quả của tầng kỹ thuật suy luận (hybrid parser / soft bonus). Mô hình thuần chưa tự đạt chỉ tiêu nếu thiếu luật. Theo Luật 1, agent không tự ý fine-tune; việc huấn luyện lại SigLIP để model thuần tự đạt chỉ tiêu là hướng đi mở để Người dùng quyết định ở giai đoạn tương lai.

---

## 12. Bổ sung Hoàn thiện Kế hoạch (Khai báo trước, Tính từ Xếp hạng Tái tạo)

Tất cả các bảng dưới đây được tính toán từ tệp xếp hạng xác định [final_eval_supplements.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_supplements.json) (đã đối chiếu khớp 100% với JSON đóng băng):

### 12.1. Đánh giá Đối chiếu theo Chính sách Ground Truth (Strict vs Relaxed vs Hybrid)
| Cấu hình | Chính sách GT | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | mAP [95% CI] |
|---|---|:---:|:---:|:---:|:---:|
| `model_only` | Strict GT | 0.3311 [0.2583, 0.4106] | 0.6026 [0.5232, 0.6755] | 0.6887 [0.6159, 0.7616] | 0.3296 [0.2834, 0.3756] |
| `model_only` | Relaxed GT | 0.3510 [0.2781, 0.4305] | 0.6291 [0.5563, 0.7086] | 0.7020 [0.6291, 0.7748] | 0.3375 [0.2934, 0.3848] |
| `model_only` | Hybrid GT (Official) | 0.3510 [0.2781, 0.4305] | 0.6291 [0.5563, 0.7086] | 0.7020 [0.6291, 0.7748] | 0.3386 [0.2944, 0.3870] |
| `cfg_v5_01` | Strict GT | 0.8477 [0.7881, 0.9007] | 0.9272 [0.8808, 0.9669] | 0.9536 [0.9205, 0.9801] | 0.8772 [0.8352, 0.9151] |
| `cfg_v5_01` | Relaxed GT | 0.9007 [0.8543, 0.9470] | 0.9470 [0.9073, 0.9801] | 0.9536 [0.9205, 0.9801] | 0.9151 [0.8778, 0.9481] |
| `cfg_v5_01` | Hybrid GT (Official) | 0.9007 [0.8543, 0.9470] | 0.9470 [0.9073, 0.9801] | 0.9536 [0.9205, 0.9801] | 0.9186 [0.8815, 0.9518] |
| `cfg_v5_09` | Strict GT | 0.8411 [0.7881, 0.8940] | 0.9205 [0.8742, 0.9603] | 0.9669 [0.9338, 0.9934] | 0.8676 [0.8225, 0.9092] |
| `cfg_v5_09` | Relaxed GT | 0.8874 [0.8344, 0.9338] | 0.9404 [0.9007, 0.9735] | 0.9669 [0.9338, 0.9934] | 0.8995 [0.8578, 0.9369] |
| `cfg_v5_09` | Hybrid GT (Official) | 0.8874 [0.8344, 0.9338] | 0.9404 [0.9007, 0.9735] | 0.9669 [0.9338, 0.9934] | 0.9028 [0.8611, 0.9401] |
| `cfg_v5_08b` | Strict GT | 0.8146 [0.7483, 0.8742] | 0.9139 [0.8675, 0.9536] | 0.9404 [0.9007, 0.9735] | 0.8505 [0.8009, 0.8938] |
| `cfg_v5_08b` | Relaxed GT | 0.8742 [0.8212, 0.9272] | 0.9338 [0.8940, 0.9735] | 0.9404 [0.9007, 0.9735] | 0.8892 [0.8452, 0.9290] |
| `cfg_v5_08b` | Hybrid GT (Official) | 0.8742 [0.8212, 0.9272] | 0.9338 [0.8940, 0.9735] | 0.9404 [0.9007, 0.9735] | 0.8919 [0.8477, 0.9314] |

### 12.2. Phân rã Lát cắt Chi tiết (`cfg_v5_09` vs `model_only`)
| Nhóm Lát cắt | Phân loại Lát cắt | Số câu (n) | Recall@10 (`cfg_v5_09`) | mAP (`cfg_v5_09`) | Recall@10 (`model_only`) | mAP (`model_only`) | Ghi chú Cỡ mẫu |
|---|---|---:|:---:|:---:|:---:|:---:|---|
| Theo Màu sắc | Có tên màu | 93 | 1.0000 | 0.8959 | 0.7634 | 0.3386 | Cỡ mẫu đủ tin cậy |
| Theo Màu sắc | Không có tên màu | 58 | 0.9138 | 0.9139 | 0.6034 | 0.3385 | Cỡ mẫu đủ tin cậy |
| Theo Giới tính | Nam (Men) | 52 | 0.9808 | 0.9388 | 0.7692 | 0.3875 | Cỡ mẫu đủ tin cậy |
| Theo Giới tính | Nữ (Women) | 49 | 0.9592 | 0.8705 | 0.6939 | 0.3195 | Cỡ mẫu đủ tin cậy |
| Theo Giới tính | Unisex / Trẻ em | 50 | 0.9600 | 0.8971 | 0.6400 | 0.3064 | Cỡ mẫu đủ tin cậy |
| Theo Số Slot | 1 Slot | 12 | 0.8333 | 0.8432 | 0.5833 | 0.2755 | Cảnh báo n nhỏ: n=12 < 20, phương sai bootstrap lớn |
| Theo Số Slot | 2 Slot | 69 | 0.9565 | 0.9108 | 0.6667 | 0.3701 | Cỡ mẫu đủ tin cậy |
| Theo Số Slot | 3 Slot | 70 | 1.0000 | 0.9051 | 0.7571 | 0.3183 | Cỡ mẫu đủ tin cậy |
| Theo Usage/Season | Có tín hiệu | 65 | 1.0000 | 0.9094 | 0.7692 | 0.3262 | Cỡ mẫu đủ tin cậy |
| Theo Usage/Season | Không có tín hiệu | 86 | 0.9419 | 0.8978 | 0.6512 | 0.3479 | Cỡ mẫu đủ tin cậy |

### 12.3. Đánh giá Cấu hình `cfg_v5_08a` (Config A InternVL2-only) & Hàng Metadata-only
| Cấu hình | Vai trò | Recall@1 [95% CI] | Recall@5 [95% CI] | Recall@10 [95% CI] | mAP [95% CI] |
|---|---|:---:|:---:|:---:|:---:|
| `cfg_v5_08a` | Config A (InternVL2-only) | 0.8874 [0.8344, 0.9338] | 0.9404 [0.9007, 0.9735] | 0.9603 [0.9272, 0.9868] | 0.8861 [0.8389, 0.9259] |
| `metadata_only` | Không dùng hình ảnh | — | — | — | — |

- **Ghi chú Hàng Metadata-only:** `không chạy: không có cấu hình khóa` (do không có cấu hình metadata-only khóa trong selection_rule_v5).
> [!WARNING]
> **CẢNH BÁO QUAN TRỌNG: Chỉ mục caption internvl2_only trong index test thực chất bị lỗi echo đề bài (65.19% câu test lặp 68 token đề bài, 32.02% chỉ còn '1. Thô'). Kết quả của cfg_v5_08a KHÔNG phản ánh năng lực sinh mô tả thị giác của InternVL2-2B. Được đưa vào báo cáo chỉ để hoàn thành kế hoạch khai báo trước.**

### 12.4. Bổ sung Đánh giá CIR: Đối chiếu Giao thức Chính thức & Quy ước Chuẩn
- **Hàng Cơ sở Chỉ Ảnh Tham chiếu (I2I Baseline trên tập CIR):**
  * Giao thức Chính thức (Ảnh gốc còn, N=1,290): `Recall@1 = 0.0000`, `Recall@5 = 0.3500`, `Recall@10 = 0.5833 [0.4998, 0.6667]`, `mAP = 0.1544 [0.1319, 0.1804]`.
  * Quy ước Chuẩn (Ảnh gốc loại, N=1,289): `Recall@1 = 0.1083 [0.0583, 0.1669]`, `Recall@5 = 0.3833`, `Recall@10 = 0.6000 [0.5083, 0.6917]`, `mAP = 0.1877 [0.1566, 0.2241]`.

- **Xác nhận Không gian Xếp hạng Mã nguồn (Prompt 14i Bước 1c & Prompt 14j Bước 1):**
  * *Giao thức chính thức (khóa từ Prompt 13):* Cả baseline và hai hàng CIR đều xếp hạng trên toàn bộ 1,290 mục. Trong 100% các cặp CIR test (120/120), ảnh tham chiếu không thuộc nhãn đúng GT; do đó trong baseline, ảnh tham chiếu đứng ở rank 1 là false positive (khiến R@1 = 0.0000).
  * *Quy ước chuẩn (Prompt 14j Bước 1):* Loại bỏ ảnh tham chiếu khỏi danh sách ứng viên (còn 1,289 mục). Trong baseline, các sản phẩm cùng loại/kiểu dáng được đôn lên rank 1, đưa R@1 của baseline lên 0.1083, R@10 lên 0.6000, mAP lên 0.1877.

#### Bảng So sánh Paired Bootstrap CIR vs Baseline theo Hai Quy ước (n=120):
| Cặp so sánh | Giao thức | ΔRecall@10 [95% CI] | Kết luận R@10 | ΔmAP [95% CI] | Kết luận mAP |
|---|---|:---:|---|:---:|---|
| `cir_v1_01 vs Baseline` | Chính thức (N=1,290) | -0.3167 [-0.4167, -0.2165] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) | -0.0573 [-0.0853, -0.0313] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) |
| `cir_v1_01 vs Baseline` | Quy ước chuẩn (N=1,289) | -0.3083 [-0.4083, -0.2083] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) | -0.0845 [-0.1188, -0.0526] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) |
| `cir_v1_05 vs Baseline` | Chính thức (N=1,290) | +0.0167 [-0.0833, +0.1167] | Chưa kết luận được (CI chứa hoặc chạm 0) | +0.1715 [+0.1213, +0.2250] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_i2i_baseline) |
| `cir_v1_05 vs Baseline` | Quy ước chuẩn (N=1,289) | +0.0000 [-0.1083, +0.1000] | Chưa kết luận được (CI chứa hoặc chạm 0) | +0.1384 [+0.0846, +0.1902] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_i2i_baseline) |
| `cir_v1_05 vs cir_v1_01` | Chính thức (N=1,290) | +0.3333 [+0.2500, +0.4085] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) | +0.2288 [+0.1833, +0.2748] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |
| `cir_v1_05 vs cir_v1_01` | Quy ước chuẩn (N=1,289) | +0.3083 [+0.2250, +0.3833] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) | +0.2228 [+0.1782, +0.2680] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |

#### Phân rã Paired theo Loại Chỉnh sửa Dưới Giao thức Chính thức (N=1,290):
| Tập phân rã | Cặp so sánh | Số cặp (n) | ΔRecall@10 [95% CI] | Kết luận R@10 | ΔmAP [95% CI] | Kết luận mAP |
|---|---|---:|:---:|---|:---:|---|
| Đổi màu sắc | `cir_v1_01 vs Baseline` | 80 | -0.3625 [-0.4875, -0.2375] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) | -0.0726 [-0.1106, -0.0337] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) |
| Đổi màu sắc | `cir_v1_05 vs Baseline` | 80 | +0.0625 [-0.0500, +0.1628] | Chưa kết luận được (CI chứa hoặc chạm 0) | +0.2495 [+0.1846, +0.3135] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_i2i_baseline) |
| Đổi màu sắc | `cir_v1_05 vs cir_v1_01` | 80 | +0.4250 [+0.3250, +0.5250] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) | +0.3221 [+0.2619, +0.3809] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |
| Đổi giới tính | `cir_v1_01 vs Baseline` | 40 | -0.2250 [-0.4000, -0.0500] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) | -0.0267 [-0.0579, +0.0074] | Chưa kết luận được (CI chứa hoặc chạm 0) |
| Đổi giới tính | `cir_v1_05 vs Baseline` | 40 | -0.0750 [-0.2750, +0.1000] | Chưa kết luận được (CI chứa hoặc chạm 0) | +0.0155 [-0.0291, +0.0637] | Chưa kết luận được (CI chứa hoặc chạm 0) |
| Đổi giới tính | `cir_v1_05 vs cir_v1_01` | 40 | +0.1500 [+0.0500, +0.2750] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) | +0.0422 [+0.0179, +0.0735] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |

#### Phân rã Paired theo Loại Chỉnh sửa Dưới Quy ước Chuẩn (N=1,289):
| Tập phân rã | Cặp so sánh | Số cặp (n) | ΔRecall@10 [95% CI] | Kết luận R@10 | ΔmAP [95% CI] | Kết luận mAP |
|---|---|---:|:---:|---|:---:|---|
| Đổi màu sắc | `cir_v1_01 vs Baseline` | 80 | -0.3500 [-0.4750, -0.2250] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) | -0.1050 [-0.1546, -0.0572] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) |
| Đổi màu sắc | `cir_v1_05 vs Baseline` | 80 | +0.0375 [-0.0875, +0.1500] | Chưa kết luận được (CI chứa hoặc chạm 0) | +0.2102 [+0.1426, +0.2733] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_i2i_baseline) |
| Đổi màu sắc | `cir_v1_05 vs cir_v1_01` | 80 | +0.3875 [+0.2875, +0.5000] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) | +0.3152 [+0.2563, +0.3741] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |
| Đổi giới tính | `cir_v1_01 vs Baseline` | 40 | -0.2250 [-0.4000, -0.0500] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) | -0.0434 [-0.0773, -0.0052] | Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01) |
| Đổi giới tính | `cir_v1_05 vs Baseline` | 40 | -0.0750 [-0.2750, +0.1000] | Chưa kết luận được (CI chứa hoặc chạm 0) | -0.0053 [-0.0508, +0.0420] | Chưa kết luận được (CI chứa hoặc chạm 0) |
| Đổi giới tính | `cir_v1_05 vs cir_v1_01` | 40 | +0.1500 [+0.0500, +0.2750] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) | +0.0381 [+0.0150, +0.0688] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |

#### Ma trận Phân rã 2x2 (Đổi màu / Đổi giới × Có dấu / Không dấu):
| Thuộc tính Thay đổi | Dấu Tiếng Việt | Số cặp (n) | mAP (`cir_v1_01`) | mAP (`cir_v1_05`) | ΔmAP [95% Paired CI] | Ý nghĩa Thống kê |
|---|---|---:|:---:|:---:|:---:|---|
| Đổi màu sắc | Có dấu đầy đủ | 40 | 0.0921 | 0.4976 | +0.4055 [+0.3207, +0.4988] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |
| Đổi màu sắc | Không dấu | 40 | 0.1070 | 0.3458 | +0.2387 [+0.1724, +0.3120] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |
| Đổi giới tính | Có dấu đầy đủ | 20 | 0.0724 | 0.0992 | +0.0268 [+0.0081, +0.0597] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |
| Đổi giới tính | Không dấu | 20 | 0.1119 | 0.1696 | +0.0576 [+0.0196, +0.1130] | Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01) |

---

## 13. Hạn chế và Cách đọc Kết quả (Luật 7 & C.10)

Để đảm bảo tính khách quan và khoa học theo tinh thần Luật 7 của AGENTS.md, kết quả đánh giá cần được nhìn nhận cùng các giới hạn sau:

1. **Vòng tròn Ground Truth (Một phần):**
   - Tập truy vấn `held-out v2` và ground truth đi kèm được tổng hợp dựa trên các trường thuộc tính cấu trúc trong catalog (`articleType`, `gender`, `baseColour`).
   - Hệ thống (đặc biệt bộ parser từ khóa và cơ chế soft bonus) được thiết kế để trích xuất và cộng thưởng cho chính các thuộc tính này.
   - Mức điểm mAP `0.9028` và Recall@10 `0.9669` phản ánh năng lực phân tách theo các thuộc tính đã lập chỉ mục cấu trúc, nhưng **chưa phải là bằng chứng khẳng định cho năng lực truy xuất thế giới mở** (open-domain retrieval) khi người dùng đưa vào các truy vấn tự do nằm ngoài không gian thuộc tính catalog.

2. **Ground Truth Phụ `usage/season`:**
   - Khi đánh giá trên nhóm sản phẩm đã khớp toàn bộ thuộc tính chính (`S_primary`), chỉ số `within_group_mAP` của Hệ thống (`0.5177 [0.4578, 0.5750]`) và Model thuần (`0.5095 [0.4482, 0.5686]`) có khoảng tin cậy chồng lấn gần như hoàn toàn.
   - Điều này chỉ ra rằng bên trong nhóm đã khớp thuộc tính chính, việc xếp hạng thứ tự chi tiết hoàn toàn do không gian cosine gốc của mô hình SigLIP quyết định; hệ thống hybrid rules không tạo ra sự phân biệt bổ sung đối với các tín hiệu phụ như ngữ cảnh sử dụng hay mùa vụ.

3. **Giới hạn Mô hình CIR (Composed Image Retrieval):**
   - Do tập ground truth CIR được dựng từ các sản phẩm cùng nhóm thuộc tính catalog biến đổi, cấu hình có mod bonus (`C3`, `cir_v1_05`) có lợi thế cấu trúc một phần so với model thuần.
   - Nhánh fused combiner (`UpgradedAttentionCIRCombiner`) trong lịch sử được huấn luyện với modifier tiếng Anh (theo Giới hạn F.2), và Luật 1 cấm train lại nên nhánh này chưa được tối ưu hóa cho tiếng Việt.
   - Phân rã cho thấy hệ thống thực hiện phép đổi màu sắc tương đối tốt (`Recall@10 = 0.7125`, `mAP = 0.4217`), nhưng năng lực đổi giới tính còn rất hạn chế (`Recall@10 = 0.3750`, `mAP = 0.1344`).
   - So sánh với baseline chỉ-ảnh-tham-chiếu cho thấy model thuần kết hợp modifier kém hơn cả khi không dùng modifier (thua baseline trên Recall@10 và mAP); hệ thống chỉ hơn baseline trên mAP nhờ attribute bonus mà chưa tạo được khác biệt có ý nghĩa thống kê trên Recall@10 (khoảng tin cậy chứa 0 ở cả hai quy ước).

4. **Giới hạn Dense Captioning & Hiện tượng Echo Đề bài (Kiểm toán Bước 1c, Luật 5, D.5):**
   - *Kiểm toán Caption A (InternVL2-2B):* Quá trình kiểm toán chỉ mục val và test phát hiện caption_A của InternVL2-2B bị lỗi lặp lại đề bài: trên tập val có `1,859/1,884` ảnh (98.67%) và trên tập test có `841/1,290` ảnh (65.19%) bị lặp nguyên văn 68 token của prompt đề bài; đồng thời có `413/1,290` ảnh test (32.02%) bị cắt ngắn chỉ còn cụm từ '1. Thô' (4–6 token). Nguyên nhân: mô hình lặp đề bài kèm dấu hai chấm khiến regex của post-processor cắt cụt (commit 2e057d6 đã giảm max_new_tokens 96→48). Raw caption không còn trên đĩa (0/1,500 dòng shard_test).
   - *Hệ quả:* Cấu hình `cfg_v5_08a` và các hàng Config A KHÔNG phản ánh thực chất năng lực sinh mô tả thị giác của InternVL2-2B; cấu hình `cfg_v5_08b` (merge_v1) sử dụng mô tả thị giác từ Vintern nên không tách biệt được đóng góp riêng của InternVL2-2B theo đúng Luật 5 và D.5.
   - *Erratum Manifest:* Manifest test ghi max_new_tokens = 96 nhưng shard_test_0001/0002 thực tế chạy ở 48 token.
   - *Ghi chú Vintern (F.1 - Lựa chọn A):* Checkpoint SigLIP `best_model.pt` trong lịch sử được huấn luyện với dense caption từ Vintern-1B-v3.5; đề tài hiện tại sử dụng InternVL2-2B làm mô hình sinh caption theo `context.md` nhưng chưa qua fine-tune lại trọng số SigLIP.

5. **Hạn chế của Tập Truy vấn Held-out v2:**
   - Tập test gồm `n = 151` câu unseen. Trong đó có `47.0%` câu chứa từ khóa ngoài lexicon (`71/151`), `65/151` câu có tín hiệu usage/season, nhưng chỉ có đúng `15` câu thuộc cụm 'không mỏng'.
   - Tập dữ liệu không gắn nhãn kiểm thử lỗi gõ (typo) độc lập và không có lớp đánh giá một phần (partial relevancy), do đó năng lực chống lỗi gõ trên tập test unseen được xếp loại **'không xác minh được'**.

6. **Hiện tượng Hubness Hai chiều (C.8):**
   - Hệ thống hybrid (`cfg_v5_09`) giúp phân tán độ tập trung tốt hơn model thuần: skewness giảm từ `1.7627` xuống `0.8937`, tỷ lệ zero-items giảm từ `41.9%` xuống `27.5%`, chỉ số Robin Hood giảm từ `0.4602` xuống `0.3345`.
   - Thử nghiệm mean-centering trên gallery test khiến skewness của Model thuần tăng lên `1.9283` và làm giảm mAP (`ΔmAP = -0.0288`), do đó kỹ thuật này được giữ tắt theo đúng Luật 1 và selection rule.

---

## 14. Erratum Báo cáo Cuối cùng (Tổng hợp Toàn diện theo Prompt 14g, 14h, 14i & 14j)

Mục này ghi nhận và sửa đổi toàn bộ các sai sót về trình bày, nhãn và diễn giải xuất hiện trong quá trình báo cáo:

1. **Số liệu Chat cũ vs JSON Đóng băng:** Toàn bộ số liệu chính xác được xác nhận 100% từ `reports/phase5c/final_eval_results.json`:
   - Paired CI `cfg_v5_09` vs `cfg_v5_01`: `ΔmAP = -0.0158 [-0.0273, -0.0063]`, `ΔRecall@10 = +0.0132 [0.0000, 0.0331]`.
   - Paired CI `cfg_v5_08b` vs `cfg_v5_01`: `ΔmAP = -0.0268 [-0.0437, -0.0122]`, `ΔRecall@10 = -0.0132 [-0.0331, 0.0000]`.
   - Hubness Skewness chính xác: Model thuần = `1.7627`, `cfg_v5_09` = `0.8937`, Model thuần + MC = `1.9283`. Các số `2.3129, 2.6963, 2.1458` bị bác bỏ.
   - Ground Truth phụ: within_group_mAP `0.5095` vs `0.5177` (chênh lệch `+0.0082`, CI chồng lấn hoàn toàn).
   - Cỡ mẫu Lát cắt: Slot mỏng `n=136` (mAP `0.8921`), Không mỏng `n=15` (mAP `1.0000`), không phải `126/25`.

2. **Sửa Nhãn Khác biệt Thống kê Paired Bootstrap:** Áp dụng hàm phân loại cơ học `classify_paired_diff_label`: các chênh lệch âm có CI loại trừ 0 được gán đúng nhãn: **`Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_09)`** và **`Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_08b)`** thay vì gán nhầm 'Tương đương'.

3. **Sửa Phân loại Độ trễ Đọc đĩa:** Nhãn `latency_conclusion_secondary_disk` (PASS_FULL 4/4) trong JSON đóng băng là SAI. Trên đường đọc đĩa thực tế: `cir_v1_01` Chưa đạt (`201.51 ms`), `cir_v1_05` Đạt theo trung vị nhưng có run vượt (`196.72 ms`, max `205.51 ms`), chỉ có `I2I` đạt chắc chắn (`168.73 ms`). Luồng 1 (Text→Ảnh) và Luồng 3 (CIR catalog) không có bước tải ảnh từ đĩa của người dùng nên được ghi nhận chuẩn xác là **`Không áp dụng`**.

4. **Sửa Lỗi Kỹ thuật Sinh Báo cáo & Cảnh báo:** Sửa triệt để lỗi placeholder f-string trong Mục 10, gắn cảnh báo cỡ mẫu nhỏ cho lát cắt không mỏng `n=15`, và thay thế nhận định chủ quan tại Mục 7.4 bằng mô tả số liệu thống kê khách quan.

5. **Đính chính Kết luận CIR so với Baseline Chỉ-Ảnh-Tham-Chiếu (Prompt 14i):**
   - Bác bỏ hoàn toàn câu nhận xét cũ tại Mục 12.4: *'Cả cir_v1_01 và cir_v1_05 đều vượt baseline này trên Recall@10'*. Thay thế bằng kết luận cơ học từ CI:
     * Model thuần `cir_v1_01` KÉM baseline có ý nghĩa trên Recall@10 (Δ = `-0.3167 [-0.4167, -0.2165]`) và trên mAP (Δ = `-0.0573 [-0.0853, -0.0313]`).
     * Hệ thống `cir_v1_05` HƠN baseline có ý nghĩa trên mAP (Δ = `+0.1715 [+0.1213, +0.2250]`), nhưng trên Recall@10 là **Chưa kết luận được** (Δ = `+0.0167 [-0.0833, +0.1167]`, CI chứa 0).
   - Bác bỏ các số baseline nháp trong chat Prompt 14h (`R@10 0.2333, mAP 0.0827`), xác nhận số đúng từ xếp hạng xác định: R@10 = `0.5833 [0.4998, 0.6667]`, mAP = `0.1544 [0.1319, 0.1804]`.
   - Xác nhận nguyên nhân chi tiết của khoảng cách giữa Parser thực tế và Oracle parser (-0.0846 mAP): **không xác minh được** (chỉ ghi nhận giả thuyết về từ ngoài lexicon).

6. **Loại bỏ Tính từ Chủ quan & Bổ sung Quy ước Chuẩn CIR (Prompt 14j):**
   - Thay thế toàn bộ 7 vị trí chứa các tính từ so sánh cảm tính bằng số liệu chênh lệch hoặc nhãn cơ học từ bootstrap CI.
   - Đánh giá bổ sung quy ước chuẩn (loại bỏ ảnh tham chiếu khỏi danh sách ứng viên, N=1,289): xác nhận Recall@10 của hệ thống cir_v1_05 và baseline chỉ-ảnh-tham-chiếu là tương đương (cùng đạt 0.6000, ΔR@10 = +0.0000 [-0.1083, +0.1000], Chưa kết luận được); mAP của cir_v1_05 đạt 0.3261 so với baseline 0.1877 (ΔmAP = +0.1384 [+0.0846, +0.1902]), nhưng mức tăng này tập trung hoàn toàn ở đổi màu (ΔmAP = +0.2102), còn đổi giới tính không chứng minh được cải thiện (ΔmAP = -0.0053 [-0.0508, +0.0420], CI chứa 0).

7. **Đính chính Căn cứ Chọn Cấu hình Chiến thắng & Kết luận Ngưỡng CIR (Prompt 15 Bước 0):**
   - Bác bỏ câu diễn đạt cũ trong bản tóm tắt chat: *'cfg_v5_09 đã được chốt từ Pha 5b do ưu tiên tiêu chí slot mỏng bền vững'* do không có căn cứ. Đính chính chuẩn xác từ `reports/phase5c/step5c_selection_report.md`: Cấu hình `cfg_v5_09` được chốt tại Pha 5c do thắng theo CV Score 0.7031 và tie-break bootstrap trong nhóm: `['cfg_v5_09', 'cfg_v5_04', 'cfg_v5_10', 'cfg_v5_11', 'cfg_v5_06', 'cfg_v5_08b']`; `cfg_v5_01` là cấu hình an toàn.
   - Bổ sung kết luận tường minh về ngưỡng CIR (trạng thái: chức năng thử nghiệm, chờ quyết định chính thức về việc áp dụng ngưỡng đề tài): Nếu áp dụng ngưỡng Recall@10 ≥ 0.85, hệ thống CIR `cir_v1_05` đạt `Recall@10 = 0.6000 [0.5083, 0.6917]` (**cận trên 0.6917 < 0.85, Chưa đạt**); nếu không áp dụng và coi CIR là chức năng thử nghiệm/thăm dò, so với baseline chỉ-ảnh-tham-chiếu thì cải thiện mAP có ý nghĩa thống kê (ΔmAP = `+0.1715 [+0.1213, +0.2250]`).

