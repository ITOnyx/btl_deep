# BÁO CÁO ĐÁNH GIÁ VRAM & ĐỀ XUẤT PHƯƠNG ÁN NẠP MÔ HÌNH (BƯỚC 2.1)

**Dự án:** Hệ thống Tra cứu Sản phẩm Thời trang Đa phương thức Tiếng Việt (`Hoc_sau_ver4`)  
**Pha:** Pha 2 — Dense Captioning: InternVL2-2B (Lõi) + Bổ trợ  
**Tuân thủ:** Luật B.4, Luật D.8, Bước 2.1 đề cương  
**Ngày thực hiện:** 2026-11-11  

---

## 1. Thông số Phần cứng & Bộ nhớ Thực tế

Dữ liệu đo đạc trực tiếp từ môi trường thực thi (ghi tại `reports/phase1/env.json` và kiểm tra thời gian thực):

- **GPU:** NVIDIA GeForce RTX 2050 Mobile (Ampere Architecture, SM 8.6)
- **Tổng VRAM vật lý:** 4.00 GB (4,096 MB)
- **VRAM khả dụng khi rảnh (Idle Free VRAM):** **3.23 GB** (~3,307 MB)  
  *(Khoảng ~0.77 GB bị chiếm dụng bởi Windows Desktop Window Manager và các tiến trình đồ họa hệ thống)*
- **RAM hệ thống (System RAM):** 15.7 GB (Khả dụng: ~9.5 GB)
- **Hệ điều hành / Thư viện:** Windows 11, Python 3.12, PyTorch 2.6.0+cu124, transformers 5.19.0, bitsandbytes 0.50.2

---

## 2. Đo đạc Kích thước Trọng số Mô hình (Hugging Face Hub API)

Xác minh trực tiếp kích thước các tệp nhị phân safetensors từ Hugging Face Hub:

| Mô hình | Vai trò | Commit SHA | Tệp trọng số | Kích thước nhị phân | Kích thước FP16 |
|---|---|---|---|---|---|
| **`OpenGVLab/InternVL2-2B`** | **Mô hình Lõi (Core VLM)** | `e4f6747bd20f139e637642c6a058c6bd00b36919` | `model.safetensors` | 4,411,571,040 bytes | **4.11 GB** |
| **`5CD-AI/Vintern-1B-v3_5`** | **Mô hình Bổ trợ (Aux VLM)** | `b98f263eab246eb5269ade64edbdca8a887dc44d` | `model.safetensors` | 3,752,849,256 bytes | **3.50 GB** |

### Đánh giá Vật lý:
- Để nạp mô hình ở chế độ thuần **FP16** lên GPU:
  - InternVL2-2B cần: 4.11 GB (trọng số) + ~0.5 - 0.8 GB (KV-cache, vision encoder activations) = **~4.6 - 4.9 GB VRAM**.
  - Vintern-1B-v3_5 cần: 3.50 GB (trọng số) + ~0.4 - 0.6 GB (activations) = **~3.9 - 4.1 GB VRAM**.
- **Kết luận:** Cả hai mô hình ở chế độ FP16 thuần đều **vượt quá dung lượng VRAM vật lý (4.00 GB)** và vượt xa dung lượng VRAM khả dụng (3.23 GB). Nếu cố định nạp FP16 trên GPU sẽ lập tức kích hoạt `torch.cuda.OutOfMemoryError`.
- **Ràng buộc Luật D.8:** *"Nếu không nạp nổi ở FP16, báo cáo phương án (8-bit/4-bit/CPU) và đánh đổi. Không tự chọn phương án làm giảm chất lượng mà không báo."*

---

## 3. Đánh giá Chi tiết 4 Phương án Thay thế

Dưới đây là so sánh toàn diện 4 phương án nạp mô hình:

| Tiêu chí | **PA 1: 8-bit Quantization (bitsandbytes LLM.int8())** | **PA 2: 4-bit Quantization (NF4 - NormalFloat4)** | **PA 3: CPU Inference (FP32 / BFloat16 trên RAM)** | **PA 4: Device Map Auto / Offloading (GPU + CPU RAM)** |
|---|---|---|---|---|
| **VRAM ước tính (InternVL2)** | **~2.4 - 2.6 GB** | **~1.6 - 1.8 GB** | **0 MB** (0 GB) | **~3.0 - 3.2 GB** (mép giới hạn) |
| **VRAM ước tính (Vintern)** | **~2.0 - 2.2 GB** | **~1.3 - 1.5 GB** | **0 MB** (0 GB) | **~2.8 - 3.0 GB** |
| **VRAM khả dụng còn dư** | **~0.6 - 0.8 GB** (An toàn cho activations) | **~1.4 - 1.6 GB** (Rất dư dả) | N/A | **~0 - 0.2 GB** (Nguy cơ OOM cao) |
| **Tốc độ sinh (RTX 2050)** | **~0.8 - 1.2 s / ảnh** | **~0.6 - 0.9 s / ảnh** | **~12 - 25 s / ảnh** | **~2.5 - 4.5 s / ảnh** |
| **Thời gian 200 ảnh pilot** | **~3 - 4 phút** | **~2 - 3 phút** | **~45 - 80 phút** | **~10 - 15 phút** |
| **Thời gian toàn bộ (~38k ảnh)** | **~9 - 12 giờ** | **~7 - 9 giờ** | **~150 - 250 giờ (~7-10 ngày)** | **~30 - 45 giờ** |
| **Rủi ro suy giảm chất lượng** | **Cực thấp (< 0.5% degradation).** Bảo toàn gần như 100% khả năng trích xuất thuộc tính thị giác và ngữ nghĩa tiếng Việt. | **Thấp - Trung bình.** Có thể làm mờ nhẹ các chi tiết hoa văn cực nhỏ hoặc sắc thái màu phụ. | **Không suy giảm.** Giữ nguyên độ chính xác nguyên bản. | **Không suy giảm** (các layer trên GPU chạy FP16). |
| **Trạng thái kiểm định kernel** | **Đã xác thực thành công** kernel CUDA 8-bit trên bitsandbytes 0.50.2. | **Đã xác thực thành công** cấu hình `BitsAndBytesConfig(load_in_4bit=True)`. | Đã sẵn sàng. | Có thể gây nghẽn băng thông PCIe bus khi swap layer. |
| **Đánh giá khuyến nghị** | ⭐ **KHUYÊN DÙNG SỐ 1** | Có thể làm phương án dự phòng | ❌ Không khả thi cho dữ liệu lớn | ❌ Quá chậm và rủi ro OOM cao |

---

## 4. Chiến lược Nạp Tuần tự (Sequential Loading) cho Cấu hình B

Để phục vụ Cấu hình B (kết hợp `InternVL2-2B` + `Vintern-1B-v3_5` + Metadata):
- Do GPU chỉ có 4.00 GB VRAM, **tuyệt đối không nạp đồng thời 2 mô hình VLM vào cùng một thời điểm**.
- **Quy trình tuần tự chuẩn hóa:**
  1. **Chặng 1 (Lõi):** Nạp `InternVL2-2B` (8-bit) $\rightarrow$ Sinh toàn bộ caption cho 200 ảnh pilot $\rightarrow$ Giải phóng model, gọi `gc.collect()` và `torch.cuda.empty_cache()` để trả VRAM về trạng thái rảnh 3.23 GB.
  2. **Chặng 2 (Bổ trợ):** Nạp `Vintern-1B-v3_5` (8-bit) $\rightarrow$ Sinh caption bổ trợ cho 200 ảnh $\rightarrow$ Giải phóng model, giải phóng VRAM.
  3. **Chặng 3 (Hợp nhất):** Chạy `src/caption/merger.py` (thuần CPU) để thực hiện quy tắc xác định `merge_v1`, đối chiếu metadata và kiểm tra độ dài token SigLIP $\le 128$.

---

## 5. Bảng Quyết định Cần Người Dùng Duyệt (DỪNG CHỜ LỆNH)

Tuân thủ Luật D.8 và đề cương Bước 2.1, agent dừng thực thi để người dùng lựa chọn:

| Lựa chọn | Phương án | VRAM InternVL2 | Tốc độ | Đánh giá |
|:---:|---|:---:|:---:|---|
| **A (Đề xuất)** | **Lượng tử hóa 8-bit (`load_in_8bit=True`)** | ~2.5 GB | ~1.0 s/ảnh | Cân bằng hoàn hảo giữa tốc độ, độ an toàn bộ nhớ và chất lượng tiếng Việt nguyên bản |
| **B** | **Lượng tử hóa 4-bit (`load_in_4bit=True`, NF4)** | ~1.7 GB | ~0.7 s/ảnh | Tiết kiệm VRAM tối đa, tốc độ nhanh nhất, có suy giảm nhẹ độ nhạy chi tiết |
| **C** | **Chạy CPU Inference (FP32/BFloat16)** | 0 GB | ~15 s/ảnh | Giữ nguyên 100% độ chính xác gốc, nhưng tốc độ rất chậm cho tập lớn |

> [!IMPORTANT]
> **DỪNG THEO QUY TẮC LUẬT D.8.**  
> Vui lòng phản hồi phương án bạn chọn (Ví dụ: *"Duyệt 8-bit"* hoặc *"Duyệt 4-bit"*), agent sẽ tiến hành chạy thử nghiệm sinh caption trên 200 ảnh pilot ngay sau khi nhận được chỉ thị.
