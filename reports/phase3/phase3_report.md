# BÁO CÁO KẾT QUẢ PHA 3: NẠP CHECKPOINT, HASH GUARD VÀ KIỂM ĐỊNH TIỀN XỬ LÝ (PARITY CHECK)

> **Dự án:** Hệ thống tra cứu sản phẩm thời trang E-commerce (SigLIP ViT-SO400M + CIR + Dense Captioning)  
> **Thời gian thực hiện:** 12/11/2026 16:17:44  
> **Cam kết:** Tuân thủ Luật 1 (Không huấn luyện, không sửa checkpoint), Luật 4 (Kiểm định tiền xử lý Cosine $\ge 0.999$), Luật 6 (Hash Guard) theo `AGENTS.md`.

---

## 1. Tóm tắt kết quả chính

1. **Xác minh Hash Guard (Luật 6):**
   - Tệp checkpoint: `data/checkpoints_siglip-vit/best_model.pt`
   - SHA-256 thực tế: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
   - So với cấu hình `config.yaml` và `manifest.json`: **Trùng khớp 100% tuyệt đối**.
2. **Nạp kiến trúc mô hình hoàn chỉnh:**
   - Đã dựng module kiến trúc `HighlyEffectiveSigLIP` (kèm `UpgradedAttentionCIRCombiner` cho CIR) tại `src/model/siglip_model.py`.
   - Cơ chế mở rộng ma trận vị trí `expand_siglip_position_embeddings` từ 16/64 lên 128 tokens hoạt động trơn tru, khớp toàn bộ 908/908 tensor keys (0 missing, 0 unexpected).
3. **Đường code tiền xử lý thống nhất (Luật 4):**
   - Đã dựng lớp `SigLIPPreprocessor` tại `src/preprocess/pipeline.py` dùng chung cho toàn bộ pipeline (tạo index, truy vấn và đánh giá).
4. **Kiểm định Parity Check trên 200 ảnh mẫu (Luật 4):**
   - Tái mã hóa 200 ảnh mẫu đối chứng từ `data/_quarantine/legacy_rebuild_20261111/untouchable/index/id_map.json` và so sánh Cosine Similarity với vector gốc trong `image_embeddings.npy`.
   - **Kết quả:** **200/200 mẫu (100.00%) đạt $\ge 0.999$**; Cosine nhỏ nhất đạt **0.99966007** (vượt chuẩn yêu cầu tối thiểu $\ge 0.9990$).
5. **Kiểm thử tự động:**
   - Toàn bộ **18/18 tests pytest PASSED**.

---

## 2. Bảng số liệu thực nghiệm chi tiết (Parity Metrics)

Số liệu trích xuất từ script chạy thực tế `src/eval/parity_check.py` (lưu tại `reports/phase3/parity_report.json`):

| Chỉ số kiểm định | Tiêu chuẩn bắt buộc | Kết quả thực tế | Đánh giá |
|---|---|---|---|
| **Số lượng mẫu kiểm định** | 200 ảnh ngẫu nhiên đối chứng | **200** | Đầy đủ 100% |
| **Cosine nhỏ nhất (Min)** | $\ge 0.99900000$ | **0.99966007** | **ĐẠT (Vượt chuẩn)** |
| **Cosine lớn nhất (Max)** | $\le 1.00000000$ | **0.99999779** | **ĐẠT** |
| **Cosine trung bình (Mean)** | - | **0.99998093** | Gần như tuyệt đối |
| **Cosine trung vị (Median)** | - | **0.99999106** | Cực kỳ ổn định |
| **Độ lệch chuẩn (Std)** | - | **$3.4019 \times 10^{-5}$** | Phân tán cực nhỏ |
| **Tỷ lệ mẫu đạt $\ge 0.999$** | $100\%$ | **200/200 (100.00%)** | **ĐẠT CHUẨN XUẤT SẮC** |

### Top 5 mẫu có Cosine thấp nhất (vẫn vượt xa 0.999)
1. Item `18237` (Tshirts): `0.99966007`
2. Item `41268` (Wallets): `0.99979645`
3. Item `42419` (Tops): `0.99983728`
4. Item `42841` (Handbags): `0.99984753`
5. Item `19123` (Flip Flops): `0.99988705`

### Top 5 mẫu có Cosine cao nhất
1. Item `22198` (Casual Shoes): `0.99999779`
2. Item `2903` (Watches): `0.99999762`
3. Item `26358` (Flip Flops): `0.99999750`
4. Item `28676` (Briefs): `0.99999738`
5. Item `10464` (Flip Flops): `0.99999708`

---

## 3. Tài nguyên hệ thống và Thời gian thực thi

- **Thiết bị:** NVIDIA GeForce RTX 2050 Laptop GPU (Compute Capability 8.6, 4 GB VRAM)
- **VRAM chiếm dụng:** **1,727.5 MB (~1.73 GB)** (Nằm an toàn dưới ngưỡng 4 GB của máy).
- **Thời gian suy luận 200 ảnh (Batch size = 32):** **41.73s** (~4.8 ảnh/s, đã bao gồm thời gian nạp mô hình và nạp ảnh từ ổ đĩa).

---

## 4. Vấn đề kỹ thuật gặp phải và Giải pháp

1. **Lỗi biến môi trường TLS trên Windows:**
   - *Vấn đề:* Bản cài đặt PostgreSQL 17 trên máy tạo biến môi trường `CURL_CA_BUNDLE` trỏ vào đường dẫn không tồn tại, khiến HuggingFace hub ném lỗi `OSError: Could not find a suitable TLS CA certificate bundle`.
   - *Giải pháp:* Đã tự động gọi `os.environ.pop("CURL_CA_BUNDLE", None)` và `os.environ.pop("REQUESTS_CA_BUNDLE", None)` trong toàn bộ các module và test suite.
2. **Khớp tiền xử lý với checkpoint huấn luyện gốc:**
   - *Vấn đề:* Ban đầu nếu dùng `transforms.Resize((224, 224))` của torchvision trước khi đưa vào `SiglipProcessor`, ảnh bị áp dụng 2 lần nội suy bilinear với grid subpixel khác nhau, khiến Cosine Similarity bị lệch xuống ~0.90 - 0.97.
   - *Giải pháp:* Đưa trực tiếp ảnh PIL RGB vào `SiglipProcessor` (vốn đã tích hợp sẵn chuẩn hóa bilinear resize và normalize chuẩn mực của SigLIP như lúc chạy `embed_catalog.py`), giúp Cosine Similarity lập tức nhảy vọt lên $\ge 0.99966$ trên 100% mẫu.
3. **Đặc thù Tokenizer của SigLIP:**
   - *Vấn đề:* `SiglipProcessor` khi tokenize văn bản chỉ trả về `input_ids`, không tạo `attention_mask` (do SigLIP sử dụng bidirectional attention).
   - *Giải pháp:* Thiết kế `SigLIPPreprocessor.preprocess_texts()` hỗ trợ linh hoạt, chỉ trả về `attention_mask` khi có trong output của processor, đảm bảo hoàn toàn tương thích với cả `HighlyEffectiveSigLIP.encode_text()`.

---

## 5. Tệp tạo mới và Cập nhật

- `src/model/siglip_model.py`: Module mô hình chính, chứa `HighlyEffectiveSigLIP`, `UpgradedAttentionCIRCombiner`, `expand_siglip_position_embeddings`, `verify_checkpoint_hash`, `load_siglip_model`.
- `src/model/__init__.py`: Package export cho `src.model`.
- `src/preprocess/pipeline.py`: Đường code tiền xử lý duy nhất `SigLIPPreprocessor` cho cả Image và Text.
- `src/preprocess/__init__.py`: Package export cho `src.preprocess`.
- `src/eval/parity_check.py`: Script chạy tự động kiểm định Parity Check trên 200 mẫu kèm Hash Guard.
- `reports/phase3/parity_report.json`: Dữ liệu số liệu thực nghiệm chi tiết của 200 mẫu.
- `tests/test_phase3_parity.py`: Bộ 5 unit test kiểm tra Hash Guard, Embedding Expansion, Preprocessor, Parity Cosine.
- `reports/phase3/phase3_report.md`: Báo cáo chi tiết Pha 3 này.

---

## 6. Kết quả kiểm thử Unit Tests (pytest)

```bash
$ pytest tests/
============================= test session starts =============================
collected 18 items

tests\test_phase1.py ......                                              [ 33%]
tests\test_phase2_caption.py .......                                     [ 72%]
tests\test_phase3_parity.py .....                                        [100%]

======================= 18 passed, 2 warnings in 39.43s =======================
```
**18/18 tests PASSED (100%).**

---

## 7. Bảng quyết định cần người dùng (Điểm dừng Pha 3)

| Mục | Nội dung | Trạng thái |
|---|---|---|
| **Hash Guard (Luật 6)** | SHA-256 `e2f82fee26...` khớp tuyệt đối với checkpoint và config | ✅ Đạt |
| **Tiền xử lý (Luật 4)** | Cosine nhỏ nhất trên 200 mẫu: **0.99966007 $\ge 0.999$** | ✅ Đạt |
| **VRAM an toàn** | 1.73 GB / 4 GB | ✅ An toàn |
| **Commit Hash** | `6ef2eec` | ✅ Đã lưu |
| **Hành động tiếp theo** | Chuyển sang **Pha 4: Dựng FAISS HNSW Index** và tạo gallery test | ⏳ Chờ lệnh **DUYỆT PHA 3** |
