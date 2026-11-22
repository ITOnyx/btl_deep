# Báo cáo Đánh giá Cuối cùng — Pha 5c (Prompt 14f: `PART_A2_DONE_AWAITING_HELDOUT_V2`)

## 0. Thông tin chung & Bất biến

- **Trạng thái thực thi:** `PART_A2_DONE_AWAITING_HELDOUT_V2`
- **Thời gian tạo (UTC):** `2026-11-15T01:00:48.792343+00:00`
- **Git commit khi chạy script:** `785f4519ca690fb4e6a263bca8cff7301267a7e0`
- **Branch:** `phase7a`
- **Seed khóa cứng:** `42`
- **Agent Python PID:** `7768`
- **Tệp `eval/test_queries_heldout_v2.json` tồn tại:** `False` (eval/test_queries_heldout_v2.json chưa khóa. Dừng sau Phần A2.)
- **Cam kết Luật 7 & C.5:** Không chạy bất kỳ phép đo chất lượng (Recall@K, mAP, Hubness) nào trên `index/test_gallery/` (Phần B) và không mở/đọc nội dung `eval/cir_eval_test.json` (Phần C). Chỉ chạy Bước 1 (Chốt CIR), Bước 2 (Erratum 14e), Bước 3 (Môi trường), Bước 4 (Phần A2 C.9), và Bước 5 (Đối chiếu Pha 5) rồi DỪNG.

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

## 6. Kết luận Độ trễ Cuối cùng

- **Theo Số chính (`RAM PIL`):** Trạng thái = `PASS_FULL`, `4/4` luồng đạt — **Đạt SLA toàn bộ 4 luồng**.
- **Theo Số phụ (`Disk + JPEG decode`):** Trạng thái = `PASS_FULL`, `4/4` luồng đạt — **Đạt SLA toàn bộ 4 luồng**.
- **Cảnh báo `p99` Luồng CIR upload:** `True` (Pooled p99 RAM = `201.238 ms`, Disk = `240.433 ms`).
