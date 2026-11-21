### BÁO CÁO TOÀN DIỆN HOÀN THIỆN ĐÁNH GIÁ CUỐI CÙNG (PROMPT 14j)

- **Khung Pháp lý:** Tuân thủ tuyệt đối AGENTS.md (Luật 1, 3, 5, 7) và Prompt 14j. **KHÔNG TRAIN LẠI**, không can thiệp checkpoint `best_model.pt` (`e2f82fee...`) hay mã suy luận. Toàn bộ số liệu chính thức trong `final_eval_results.json` (`8a89359af8e0cbe4...`) là **KẾT QUẢ ĐÓNG BĂNG** bất biến.
- **Tái tạo Xếp hạng Xác định & Baseline:** Toàn bộ thứ tự xếp hạng 1,290 sản phẩm của 151 câu Text và 120 cặp CIR đã đối chiếu khớp 100% với JSON đóng băng (sai số 0.00e+00, dung sai 1e-9). Đã đánh giá đối chiếu cả 2 quy ước CIR (Giao thức chính thức N=1,290 và Quy ước chuẩn N=1,289 loại ảnh tham chiếu).

#### 1. Bảng Tổng hợp Chỉ tiêu Đề tài (AGENTS.md Phần A)
| Chỉ tiêu | Ngưỡng bắt buộc | Model thuần (`model_only`) | Hệ thống (`cfg_v5_09`) | Kết luận Đề tài | Hạn chế đi kèm |
|---|:---:|:---:|:---:|:---:|---|
| **Recall@10** | **≥ 0.85** | `0.7020` [0.6291, 0.7748] | **`0.9669` [0.9338, 0.9934]** | **Đạt chắc chắn** | Nguồn `held-out v2`, GT hybrid; có yếu tố vòng tròn GT thuộc tính |
| **Text→Image mAP** | **≥ 0.75** | `0.3386` [0.2944, 0.3870] | **`0.9028` [0.8611, 0.9401]** | **Đạt chắc chắn** | Nguồn `held-out v2`, GT hybrid; thứ tự phụ do cosine SigLIP quyết định |
| **Độ trễ (p95 RAM)** | **< 200 ms** | T2I p95: 29.70 ms | T2I p95: 32.52 ms, CIR upload RAM: 158.54 ms | **Đạt chắc chắn (100% 4 luồng)** | Cảnh báo p99 CIR upload RAM = 201.24 ms (> 200 ms) |
| **Độ trễ (p95 Disk)** | **< 200 ms** | I2I p95: 168.73 ms | cir_v1_05: 196.72 ms (max 205.51 ms) | **Đạt có điều kiện** | Chỉ áp dụng Luồng 2 & 4; cir_v1_01 Disk Chưa đạt (201.51 ms); p99 CIR Disk = 240.43 ms |

#### 2. Kết quả CIR Đối chiếu Giữa Hai Giao thức & Baseline Chỉ-Ảnh-Tham-Chiếu
| Cấu hình / Baseline | Giao thức Chính thức (Ảnh gốc còn, N=1,290) R@10 | Giao thức Chính thức mAP | Quy ước Chuẩn (Ảnh gốc loại, N=1,289) R@10 | Quy ước Chuẩn mAP |
|---|:---:|:---:|:---:|:---:|
| Baseline chỉ-ảnh-tham chiếu | 0.5833 [0.4998, 0.6667] | 0.1544 [0.1319, 0.1804] | 0.6000 [0.5083, 0.6917] | 0.1877 [0.1566, 0.2241] |
| `cir_v1_01` (model_only) | 0.2667 [0.1917, 0.3500] | 0.0971 [0.0705, 0.1256] | 0.2917 [0.2167, 0.3750] | 0.1032 [0.0745, 0.1339] |
| `cir_v1_05` (hệ thống) | 0.6000 [0.5083, 0.6917] | 0.3259 [0.2695, 0.3847] | 0.6000 [0.5083, 0.6917] | 0.3261 [0.2696, 0.3848] |

- **So sánh Paired Cơ học:**
  * `cir_v1_01 vs Baseline:` KÉM baseline có ý nghĩa thống kê trên cả R@10 (Δ = `-0.3167` chính thức, `-0.3083` chuẩn) và mAP (Δ = `-0.0573` chính thức, `-0.0845` chuẩn).
  * `cir_v1_05 vs Baseline:` HƠN baseline có ý nghĩa thống kê trên mAP (Δ = `+0.1715` chính thức, `+0.1384` chuẩn) nhờ C3 bonus; nhưng trên Recall@10 là **Chưa kết luận được** (Δ = `+0.0167` chính thức, `+0.0000` chuẩn, CI chứa 0).
  * `Phân rã đổi màu vs đổi giới:` Cải thiện mAP chỉ có ý nghĩa ở đổi màu sắc (ΔmAP = `+0.2495` chính thức, `+0.2102` chuẩn). Đổi giới tính (n=40) **không chứng minh được cải thiện** so với baseline (ΔmAP = `+0.0155 [-0.0291, +0.0637]` chính thức, `-0.0053 [-0.0508, +0.0420]` chuẩn, CI chứa 0).
  * `cir_v1_05 vs cir_v1_01:` Hệ thống hơn model thuần (ΔmAP = `+0.2288`, ΔR@10 = `+0.3333` chính thức; `+0.2228` và `+0.3083` chuẩn).
- **Kết luận về Ngưỡng CIR (chờ quyết định áp dụng hay không áp dụng ngưỡng đề tài Recall@10 ≥ 0.85):** Nếu áp dụng ngưỡng Recall@10 ≥ 0.85, hệ thống CIR `cir_v1_05` đạt `Recall@10 = 0.6000 [0.5083, 0.6917]` (**cận trên 0.6917 < 0.85, Chưa đạt**); nếu không áp dụng và coi CIR là chức năng thử nghiệm/thăm dò, so với baseline chỉ-ảnh-tham-chiếu thì cải thiện mAP có ý nghĩa thống kê (ΔmAP = `+0.1715 [+0.1213, +0.2250]`).

#### 3. Bóc tách Nguyên nhân Định lượng (Phần C Chẩn đoán)
- **Model thuần SigLIP:** Đóng góp mAP `0.3386` và Recall@10 `0.7020` (chưa đạt ngưỡng đề bài).
- **Tầng Suy luận (Parser + Soft Bonus):** Đóng góp mức tăng **`+0.5642 mAP`** đưa hệ thống lên `0.9028`.
- **Hệ thống Oracle (Perfect Parser):** Đạt mAP `0.9874` và Recall@10 `0.9934`. Khoảng cách giữa Parser thực tế và Oracle là `-0.0846 mAP`. Do chưa có phân rã từng câu lỗi, nguyên nhân chi tiết của khoảng cách: **không xác minh được**.

#### 4. Bảy Hạn chế Chính của Dự án (Kèm Con số Định lượng - Prompt 14j Bước 3)
1. **Vòng tròn Ground Truth:** Nhãn GT hybrid được xây dựng từ 3 thuộc tính catalog (`articleType`, `gender`, `baseColour`). Hệ thống parser/bonus cộng điểm đúng 3 trường này (Oracle mAP = `0.9874`), phản ánh năng lực lọc theo thuộc tính cấu trúc hơn là truy xuất thế giới mở; ở luồng CIR, C3 bonus cũng dùng chính các thuộc tính catalog này.
2. **Bộ kết hợp F.2 chưa tối ưu:** Nhánh fused combiner `UpgradedAttentionCIRCombiner` trong lịch sử được huấn luyện với modifier tiếng Anh; Luật 1 cấm train lại nên nhánh này chưa từng được tối ưu hóa cho tiếng Việt, khiến cir_v1_01 thua baseline chỉ-ảnh.
3. **Hiện tượng Echo Caption Đề bài (InternVL2-2B):** Kiểm toán phát hiện `841/1,290` ảnh test (65.19%) bị lặp 68 token đề bài và `413/1,290` ảnh (32.02%) bị cắt cụt còn '1. Thô'. Cấu hình `cfg_v5_08a` và `cfg_v5_08b` không phản ánh thực chất năng lực sinh mô tả độc lập của InternVL2-2B.
4. **Tập Kiểm thử Held-out v2:** Quy mô $n=151$ câu unseen, trong đó chỉ có đúng 15 câu thuộc cụm 'không mỏng' (phương sai bootstrap lớn), 47.0% câu chứa từ khóa ngoài lexicon (71/151 câu), và không có nhãn kiểm thử lỗi gõ (typo) độc lập hay đánh giá một phần.
5. **Độ trễ Đường Đọc Đĩa Thực tế:** Số chính đo trên RAM PIL đạt < 200 ms (T2I 32.52 ms, CIR 158.54 ms). Trên đường đọc đĩa JPEG decode thực tế (chỉ áp dụng cho Luồng 2 và 4): `cir_v1_01` Chưa đạt (p95 = `201.51 ms`), `cir_v1_05` Đạt theo trung vị nhưng có run vượt (`196.72 ms`, max `205.51 ms`), và pooled p99 của CIR upload đều vượt 200 ms (`201.24 ms` RAM, `240.43 ms` Disk).
6. **Đánh đổi Giữa các Cấu hình Đã Chốt:** Cấu hình `cfg_v5_01` đạt mAP cao hơn cấu hình chiến thắng `cfg_v5_09` có ý nghĩa thống kê (ΔmAP = `-0.0158 [-0.0273, -0.0063]`); cấu hình `cfg_v5_09` được chốt ở Pha 5c do thắng theo CV Score 0.7031 và tie-break bootstrap trong nhóm: ['cfg_v5_09', 'cfg_v5_04', 'cfg_v5_10', 'cfg_v5_11', 'cfg_v5_06', 'cfg_v5_08b'], còn `cfg_v5_01` là cấu hình an toàn (căn cứ [step5c_selection_report.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/step5c_selection_report.md)).
7. **Quy ước Ảnh Tham chiếu trong CIR:** Cả ở giao thức chính thức ($N=1,290$) và quy ước chuẩn loại ảnh tham chiếu ($N=1,289$), hệ thống `cir_v1_05` không khác biệt có ý nghĩa thống kê so với baseline chỉ-ảnh-tham-chiếu trên Recall@10 (ΔR@10 = `+0.0167` và `+0.0000`, CI chứa 0); trên mAP hệ thống hơn baseline nhưng cải thiện chỉ tập trung ở đổi màu (ΔmAP = `+0.2102` chuẩn), còn đổi giới tính ($n=40$) không chứng minh được sự cải thiện (ΔmAP = `-0.0053 [-0.0508, +0.0420]`).

#### 5. Tổng kết Trạng thái Tệp & Kiểm thử
- [final_eval_report.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report.md) (`SHA-256: ea8a571a520d5728b89e7bf7f05912e8a730fe06f6366b5250e393a8dfc9e4ad`)
- [final_eval_corrections.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_corrections.json) (`SHA-256: ad86f675207ca7030c0c713eb54cd60d33136f322f1224282bc1956363df65a0`)
- [final_eval_supplements.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_supplements.json) (`SHA-256: 55d477938add99587de00c0849c0c13d003db06ea1b7654a2dff8cc6575fe443`)
- [cir_standard_convention_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/cir_standard_convention_results.json) (`SHA-256: 41b4a6d01eaa5bd63f8d4212afd0c0bb7edaed4db87da1e2003ce2a5770342bc`)
- [final_eval_report_prompt14j.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report_prompt14j.md) (Bản lưu Prompt 14j)
- [final_summary_for_chat_prompt14j.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_summary_for_chat_prompt14j.md) (Bản lưu Prompt 14j)
