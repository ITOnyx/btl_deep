### BÁO CÁO TOÀN DIỆN HOÀN THIỆN ĐÁNH GIÁ CUỐI CÙNG (PROMPT 14i)

- **Khung Pháp lý:** Tuân thủ tuyệt đối AGENTS.md (Luật 1, 3, 5, 7) và Prompt 14i. **KHÔNG TRAIN LẠI**, không can thiệp checkpoint `best_model.pt` (`e2f82fee...`) hay mã suy luận. Toàn bộ số liệu chính thức trong `final_eval_results.json` (`8a89359af8e0cbe4...`) là **KẾT QUẢ ĐÓNG BĂNG** bất biến.
- **Tái tạo Xếp hạng Xác định & Baseline:** Toàn bộ thứ tự xếp hạng 1,290 sản phẩm của 151 câu Text và 120 cặp CIR đã được đối chiếu khớp 100% với JSON đóng băng. Cả 2 luồng CIR và baseline chỉ-ảnh-tham-chiếu đều chấm điểm và xếp hạng trên toàn bộ 1,290 sản phẩm của `test_gallery` (ảnh tham chiếu không nằm trong nhãn đúng của bất kỳ cặp nào).

#### 1. Bảng Tổng hợp Chỉ tiêu Đề tài (AGENTS.md Phần A)
| Chỉ tiêu | Ngưỡng bắt buộc | Model thuần (`model_only`) | Hệ thống (`cfg_v5_09`) | Kết luận Đề tài | Hạn chế đi kèm |
|---|:---:|:---:|:---:|:---:|---|
| **Recall@10** | **≥ 0.85** | `0.7020` [0.6291, 0.7748] | **`0.9669` [0.9338, 0.9934]** | **Đạt chắc chắn** | Nguồn `held-out v2`, GT hybrid; có yếu tố vòng tròn GT thuộc tính |
| **Text→Image mAP** | **≥ 0.75** | `0.3386` [0.2944, 0.3870] | **`0.9028` [0.8611, 0.9401]** | **Đạt chắc chắn** | Nguồn `held-out v2`, GT hybrid; thứ tự phụ do cosine SigLIP quyết định |
| **Độ trễ (p95 RAM)** | **< 200 ms** | T2I p95: 29.70 ms | T2I p95: 32.52 ms, CIR upload RAM: 158.54 ms | **Đạt chắc chắn (100% 4 luồng)** | Cảnh báo p99 CIR upload RAM = 201.24 ms (> 200 ms) |
| **Độ trễ (p95 Disk)** | **< 200 ms** | I2I p95: 168.73 ms | cir_v1_05: 196.72 ms (max 205.51 ms) | **Đạt có điều kiện** | Chỉ áp dụng Luồng 2 & 4; cir_v1_01 Disk Chưa đạt (201.51 ms); p99 CIR Disk = 240.43 ms |

#### 2. Kết quả CIR (Composed Image Retrieval) & So sánh Baseline Chỉ-Ảnh-Tham-Chiếu
- **Baseline Chỉ-ảnh-tham-chiếu (`cir_i2i_baseline`, N=1,290):** `Recall@1 = 0.0000`, `Recall@5 = 0.3500 [0.2583, 0.4333]`, `Recall@10 = 0.5833 [0.4998, 0.6667]`, `mAP = 0.1544 [0.1319, 0.1804]`.
- **Model thuần CIR (`cir_v1_01`):** `Recall@10 = 0.2667 [0.1917, 0.3500]`, `mAP = 0.0971 [0.0705, 0.1256]`.
  * *So với Baseline:* ΔRecall@10 = `-0.3167 [-0.4167, -0.2165]` — **Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01)**; ΔmAP = `-0.0573 [-0.0853, -0.0313]` — **Khác biệt có ý nghĩa (cir_i2i_baseline hơn cir_v1_01)**.
  * *Nhận định:* Bộ kết hợp vector chưa fine-tune làm giảm hiệu năng tìm kiếm so với khi chỉ dùng ảnh tham chiếu đơn thuần.
- **Hệ thống CIR (`cir_v1_05`):** `Recall@10 = 0.6000 [0.5083, 0.6917]`, `mAP = 0.3259 [0.2695, 0.3847]`.
  * *So với Baseline:* ΔmAP = `+0.1715 [+0.1213, +0.2250]` — **Khác biệt có ý nghĩa (cir_v1_05 hơn cir_i2i_baseline)**; ΔRecall@10 = `+0.0167 [-0.0833, +0.1167]` — **Chưa kết luận được (CI chứa hoặc chạm 0)** (khoảng tin cậy chứa 0).
  * *So với cir_v1_01:* ΔmAP = `+0.2288 [+0.1833, +0.2748]` — **Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01)**; ΔRecall@10 = `+0.3333 [+0.2500, +0.4085]` — **Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01)**.
- **Phân rã Đổi màu vs Đổi giới tính (`cir_v1_05`):**
  * Đổi màu sắc (n=80): R@10 = `0.7125`, mAP = `0.4217`. So với baseline: ΔmAP = `+0.2495 [+0.1846, +0.3135]` (hơn có ý nghĩa); ΔR@10 = `+0.0625 [-0.0500, +0.1628]` (Chưa kết luận được).
  * Đổi giới tính (n=40): R@10 = `0.3750`, mAP = `0.1344`. So với baseline: ΔmAP = `+0.0155 [-0.0291, +0.0637]` (Chưa kết luận được); ΔR@10 = `-0.0750 [-0.2750, +0.1000]` (Chưa kết luận được).
- **Kết luận Ngưỡng Đề tài cho CIR:** Nếu áp ngưỡng chung (R@10 >= 0.85), CIR **chưa đạt** (cận trên 0.6917 < 0.85); nếu coi CIR là tác vụ mở rộng kết hợp đa phương thức, hệ thống đạt cải thiện có ý nghĩa thống kê vượt trội về mAP so với model thuần.

#### 3. Bóc tách Nguyên nhân Định lượng (Phần C Chẩn đoán)
- **Model thuần SigLIP:** Đóng góp mAP `0.3386` (chưa đạt ngưỡng đề bài).
- **Tầng Suy luận (Parser + Soft Bonus):** Đóng góp mức tăng khổng lồ **`+0.5642 mAP`** đưa hệ thống lên `0.9028`.
- **Hệ thống Oracle (Perfect Parser):** Đạt mAP `0.9874` và Recall@10 `0.9934`. Khoảng cách giữa Parser thực tế và Oracle là `-0.0846 mAP`. Do chưa có phân rã từng câu lỗi, nguyên nhân chi tiết của khoảng cách: **không xác minh được**.
- **Vòng tròn GT:** Do nhãn GT hybrid xây dựng từ thuộc tính catalog (`articleType`, `gender`, `baseColour`), Oracle parser khi khớp đúng 3 trường này sẽ đẩy mAP lên gần tuyệt đối (`0.9874`).

#### 4. Bổ sung Đánh giá theo Kế hoạch Khai báo trước
- **Chính sách Ground Truth trên `cfg_v5_09`:**
  * Strict GT: `mAP = 0.8676 [0.8225, 0.9092]`, `Recall@10 = 0.9669`
  * Relaxed GT: `mAP = 0.8995 [0.8578, 0.9369]`, `Recall@10 = 0.9669`
  * Hybrid GT (Official): `mAP = 0.9028 [0.8611, 0.9401]`, `Recall@10 = 0.9669`
- **Cấu hình `cfg_v5_08a` (Config A InternVL2-only):** Đạt mAP `0.8861`, R@10 `0.9603`. **Cảnh báo:** Kiểm toán Bước 1c phát hiện 65.19% caption bị echo 68 token đề bài và 32.02% bị cắt cụt còn '1. Thô', do đó số liệu này KHÔNG phản ánh thực chất năng lực sinh mô tả thị giác của InternVL2-2B.
- **Hàng Metadata-only:** `không chạy: không có cấu hình khóa`.

#### 5. Erratum Hiệu chỉnh So với Các Báo cáo Trước (Prompt 14g, 14h & 14i)
- **Sửa Kết luận CIR vs Baseline:** Bác bỏ câu nhận xét cũ cho rằng cả 2 cấu hình đều vượt baseline trên R@10; xác nhận cir_v1_01 KÉM baseline có ý nghĩa trên R@10 và mAP; cir_v1_05 HƠN baseline có ý nghĩa trên mAP nhưng Chưa kết luận được trên R@10.
- **Sửa Số liệu Chat 14h:** Bác bỏ các số nháp `R@10 0.2333, mAP 0.0827` trong chat cũ, xác nhận số đúng là R@10 = `0.5833 [0.4998, 0.6667]`, mAP = `0.1544 [0.1319, 0.1804]`.
- **Sửa Kết luận Độ trễ Disk:** Luồng 1 (Text→Ảnh) và Luồng 3 (CIR catalog) ghi nhận chuẩn xác là **`Không áp dụng`** (không có bước tải ảnh từ đĩa của người dùng). Trên các luồng áp dụng: Luồng 2 đạt chắc chắn (168.73 ms); cir_v1_01 Chưa đạt (201.51 ms); cir_v1_05 Đạt theo trung vị nhưng có run vượt (196.72 ms).
- **Sửa Nhãn Paired Cơ học:** `cfg_v5_09_vs_cfg_v5_01` (ΔmAP = -0.0158) được phân loại chuẩn xác: **`Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_09)`**; ΔRecall@10 chạm 0 ([0.0000, 0.0331]) là **`Chưa kết luận được`**.
- **Sửa Nhãn Lát cắt:** Khẳng định chính xác Slot mỏng là `n=136` (mAP `0.8921`), Không mỏng là `n=15` (mAP `1.0000`, kèm cảnh báo cỡ mẫu nhỏ suy biến).
- **Secondary GT:** Mô tả số liệu khách quan chênh lệch chỉ `+0.0082` mAP, khoảng tin cậy chồng lấn hoàn toàn, thứ tự phụ do cosine SigLIP quyết định.

#### 6. Tổng kết Trạng thái Tệp & Kiểm thử
- [final_eval_report.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report.md) (`SHA-256: 165f7043884aad8630ce52f0719be9e0673918e95e1a2442fb48ee64c5167fc4`)
- [final_eval_corrections.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_corrections.json) (`SHA-256: 90a33f024a4233d2da19ad90756ab87517d5901c4ed49cbab21bf98324af13db`)
- [final_eval_supplements.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_supplements.json) (`SHA-256: 4e087e4bf41a59e8b9cd082cd7a3f08cb3cd4332e8b34a56dd75a84ab11f96bc`)
- [cir_paired_baseline_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/cir_paired_baseline_results.json) (`SHA-256: 97acec74be56b6e7eb607aac0fc53f2a53d4674b299186b64bd7bb927e73462b`)
- [final_eval_report_prompt14h.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report_prompt14h.md) (Bản lưu Prompt 14h đã đóng gói an toàn)
