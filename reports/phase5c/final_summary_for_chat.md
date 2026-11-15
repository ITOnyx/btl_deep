### BÁO CÁO TOÀN DIỆN HOÀN THIỆN ĐÁNH GIÁ CUỐI CÙNG (PROMPT 14h)

- **Khung Pháp lý:** Tuân thủ tuyệt đối AGENTS.md (Luật 1, 3, 5, 7) và Prompt 14h. **KHÔNG TRAIN LẠI**, không can thiệp checkpoint `best_model.pt` (`e2f82fee...`) hay mã suy luận. Toàn bộ số liệu chính thức trong `final_eval_results.json` (`8a89359af8e0cbe4...`) là **KẾT QUẢ ĐÓNG BĂNG** bất biến.
- **Tái tạo Xếp hạng Xác định (Bước 1):** Script `scripts/replay_rankings_final.py` đã tái tạo thành công thứ tự xếp hạng 1,290 sản phẩm cho 151 câu Text và 120 cặp CIR. Kết quả đối chiếu với `final_eval_results.json` đạt **EXACT MATCH 100% (0 lệch, dung sai 1e-9)**, đủ điều kiện đưa vào phân tích bổ sung.

#### 1. Bảng Tổng hợp Chỉ tiêu Đề tài (AGENTS.md Phần A)
| Chỉ tiêu | Ngưỡng bắt buộc | Model thuần (`model_only`) | Hệ thống (`cfg_v5_09`) | Kết luận Đề tài | Hạn chế đi kèm |
|---|:---:|:---:|:---:|:---:|---|
| **Recall@10** | **≥ 0.85** | `0.7020` [0.6291, 0.7748] | **`0.9669` [0.9338, 0.9934]** | **Đạt chắc chắn** | Nguồn `held-out v2`, GT hybrid; có yếu tố vòng tròn GT thuộc tính |
| **Text→Image mAP** | **≥ 0.75** | `0.3386` [0.2944, 0.3870] | **`0.9028` [0.8611, 0.9401]** | **Đạt chắc chắn** | Nguồn `held-out v2`, GT hybrid; thứ tự phụ do cosine SigLIP quyết định |
| **Độ trễ (p95 RAM)** | **< 200 ms** | T2I p95: 29.70 ms | T2I p95: 32.52 ms, CIR upload RAM: 158.54 ms | **Đạt chắc chắn (100% 4 luồng)** | Cảnh báo p99 CIR upload RAM = 201.24 ms (> 200 ms) |
| **Độ trễ (p95 Disk)** | **< 200 ms** | I2I p95: 168.73 ms | cir_v1_05: 196.72 ms (max 205.51 ms) | **Đạt có điều kiện** | cir_v1_01 Disk Chưa đạt (201.51 ms); p99 CIR Disk = 240.43 ms |

#### 2. Kết quả CIR (Composed Image Retrieval) — Đánh giá Riêng biệt
- **Model thuần CIR (`cir_v1_01`):** `Recall@10 = 0.2667 [0.1917, 0.3500]`, `mAP = 0.0971 [0.0705, 0.1256]`.
- **Hệ thống CIR (`cir_v1_05`):** `Recall@10 = 0.6000 [0.5083, 0.6917]`, `mAP = 0.3259 [0.2695, 0.3847]`.
- **Chênh lệch Paired Bootstrap:** `ΔmAP = +0.2288 [+0.1833, +0.2748]`, `ΔRecall@10 = +0.3333 [+0.2500, +0.4085]` — **Khác biệt có ý nghĩa (cir_v1_05 hơn cir_v1_01)**.
- **So sánh Cơ sở Chỉ Ảnh Tham chiếu (I2I Baseline):** R@10 = `0.2333 [0.1667, 0.3083]`, mAP = `0.0827 [0.0592, 0.1086]`. Cả 2 cấu hình CIR đều vượt baseline ảnh tham chiếu.
- **Đổi màu vs Đổi giới tính:** Đổi màu sắc đạt R@10 = `0.7125`, mAP = `0.4217`; Đổi giới tính chỉ đạt R@10 = `0.3750`, mAP = `0.1344` (nhánh fused combiner F.2 chưa được train cho tiếng Việt).
- **Kết luận Ngưỡng CIR:** Nếu áp ngưỡng chung của đề cương (R@10 >= 0.85) thì CIR **chưa đạt**; nếu coi CIR là tác vụ mở rộng kết hợp đa phương thức, hệ thống đã tạo cải tiến vượt bậc có ý nghĩa thống kê.

#### 3. Bóc tách Nguyên nhân Định lượng (Phần C Chẩn đoán)
- **Model thuần SigLIP:** Đóng góp mAP `0.3386` (chưa đạt ngưỡng đề bài).
- **Tầng Suy luận (Parser + Soft Bonus):** Đóng góp mức tăng khổng lồ **`+0.5642 mAP`** đưa hệ thống lên `0.9028`.
- **Hệ thống Oracle (Perfect Parser):** Đạt mAP `0.9874` và Recall@10 `0.9934`. Khoảng cách giữa Parser thực tế và Oracle là `-0.0846 mAP` do các câu phức tạp hoặc từ ngoài lexicon.
- **Vòng tròn GT:** Do nhãn GT hybrid xây dựng từ thuộc tính catalog (`articleType`, `gender`, `baseColour`), Oracle parser khi khớp đúng 3 trường này sẽ đẩy mAP lên gần tuyệt đối (`0.9874`).

#### 4. Bổ sung Đánh giá theo Kế hoạch Khai báo trước
- **Chính sách Ground Truth trên `cfg_v5_09`:**
  * Strict GT: `mAP = 0.8676 [0.8225, 0.9092]`, `Recall@10 = 0.9669`
  * Relaxed GT: `mAP = 0.8995 [0.8578, 0.9369]`, `Recall@10 = 0.9669`
  * Hybrid GT (Official): `mAP = 0.9028 [0.8611, 0.9401]`, `Recall@10 = 0.9669`
- **Cấu hình `cfg_v5_08a` (Config A InternVL2-only):** Đạt mAP `0.8861`, R@10 `0.9603`. **Cảnh báo:** Kiểm toán Bước 1c phát hiện 65.19% caption bị echo 68 token đề bài và 32.02% bị cắt cụt còn '1. Thô', do đó số liệu này KHÔNG phản ánh thực chất năng lực sinh mô tả thị giác của InternVL2-2B.
- **Hàng Metadata-only:** `không chạy: không có cấu hình khóa`.

#### 5. Erratum Hiệu chỉnh So với Các Báo cáo Trước
- **Sửa Nhãn Paired Cơ học:** `cfg_v5_09_vs_cfg_v5_01` (ΔmAP = -0.0158) được phân loại chuẩn xác: **`Khác biệt có ý nghĩa (cfg_v5_01 hơn cfg_v5_09)`**; ΔRecall@10 chạm 0 ([0.0000, 0.0331]) là **`Chưa kết luận được`**.
- **Sửa Kết luận Độ trễ Disk:** Thay thế nhãn sai 'PASS_FULL 4/4' trong JSON đóng băng bằng phân loại chuẩn xác theo `classify_latency_conclusion`: `cir_v1_01` Chưa đạt, `cir_v1_05` Đạt theo trung vị nhưng có run vượt.
- **Sửa Nhãn Lát cắt:** Khẳng định chính xác Slot mỏng là `n=136` (mAP `0.8921`), Không mỏng là `n=15` (mAP `1.0000`, kèm cảnh báo cỡ mẫu nhỏ suy biến).
- **Secondary GT:** Mô tả số liệu khách quan chênh lệch chỉ `+0.0082` mAP, khoảng tin cậy chồng lấn hoàn toàn, thứ tự phụ do cosine SigLIP quyết định.

#### 6. Tổng kết Trạng thái Tệp & Kiểm thử
- [final_eval_report.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report.md) (`SHA-256: 65f9f8b5abdad8df49d3ac74d2b96e48100133118ae7a5ea34345d33ed369cf4`)
- [final_eval_corrections.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_corrections.json) (`SHA-256: 56bfd1cb0623a4f544b22fb6d405b1e823c33ce8b7310acb30e84fd7c32a7662`)
- [final_eval_supplements.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_supplements.json) (`SHA-256: 01768f836f245c7693381226bfb4fc061e94d42f61582eff6018cad667337f71`)
- [final_eval_report_prompt14g.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report_prompt14g.md) (Bản lưu Prompt 14g đã đóng gói an toàn)
