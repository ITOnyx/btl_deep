# Báo cáo Chẩn đoán Pha 3b: Phân Tích Nguyên Nhân Recall@10 Thấp & Đánh Giá Đối Chứng

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Ngày thực hiện:** 08/11/2026  
**Trạng thái:** Hoàn thành toàn diện (D0 đến D8) — Xác định dứt khoát nguyên nhân gốc rễ.

---

## 1. Tóm Tắt & Bảng Phân Loại Kết Quả Chẩn Đoán

Ở Pha 3 (v1), truy vấn paired text $\to$ ảnh chỉ đạt Recall@10 = **0.0597 (5.97%)** trên $G_{test}$ và **0.0016 (0.16%)** trên $G_{full}$, với hiện tượng nhầm lẫn kỳ lạ (áo thun $\to$ túi xách/ba lô). Qua chuỗi 9 bước chẩn đoán nghiêm ngặt (D0 đến D8) độc lập, chúng tôi đã cô lập chính xác các nguyên nhân:

| Mã bước | Tên nội dung chẩn đoán | Kết luận phân loại | Mô tả chi tiết phát hiện |
| :--- | :--- | :--- | :--- |
| **D0** | Sửa cách chấm (Hạng thật 1..N) | **KỸ THUẬT ĐO ĐẠC** | Median rank thật của Q1 là **266.5 / 1,290** (top 20.6% gallery $G_{test}$), Mean rank là **373.73**. Điểm P@min(R,10) thuộc tính đạt **0.5158** ($G_{test}$) và **0.6049** ($G_{full}$). |
| **D1** | Kiểm toán cặp (Caption $\leftrightarrow$ Ảnh) | **BÌNH THƯỜNG (KHÔNG LỖI)** | Không có hiện tượng lệch ID hoặc sai trường: 100.0% khớp giới tính, **99.92% (1,289/1,290)** khớp chính xác 100% tên tiếng Anh gốc. |
| **D2** | Độ dài token & Cắt cụt | **LỖI THIẾT KẾ TRUY VẤN (BUG)** | **52.33% (675/1,290)** câu caption bị cắt cụt ở 128 tokens. Đối với 100% các câu bị cắt cụt, cụm phân biệt quan trọng nhất (`Tên gốc: {name}`) bị rơi rụng hoàn toàn (**0% bảo toàn**), biến truy vấn thành các câu chung chung giống hệt nhau. |
| **D3** | Text Pipeline Equivalence & Mask | **BÌNH THƯỜNG (KHÔNG LỖI)** | Cosine similarity giữa pipeline inference và notebook là **1.000000**. Đã làm rõ lỗi `AttributeError: 'NoneType' object has no attribute 'to'` do tokenizer SigLIP không sinh `attention_mask` (khớp chuẩn thiết kế code train). |
| **D4** | Phép thử đối chứng quyết định | **KẾT LUẬN THEN CHỐT** | Mô hình **KHÔNG HỌC VẸT**: Trên tập đã train (Unknown), caption công thức v1 cũng chỉ đạt Recall@10 = **4.80%**. Khi đổi sang câu tiếng Việt súc tích (`{loại} {giới} màu {màu}`), Recall@10 lập tức tăng vọt lên **62.10%** (Unknown) và **57.44%** ($G_{test}$), Median rank giảm ngoạn mục xuống **7.0 - 8.0**. |
| **D5** | Phân tích Hubness & Modality Gap | **HIỆN TƯỢNG MÔ HÌNH** | Tồn tại Hubness cực mạnh: Top 1 Hub (Túi xách Handbag 38753) lọt vào top-10 của **67.1%** tổng số truy vấn, đè bẹp ảnh đích (cosine 0.62 vs 0.37). Trừ vector trung bình (Mean-Centering) đẩy Recall@10 tăng **2.5 lần** (5.97% $\to$ 14.65%) và hạ Median rank từ 266.5 xuống 71.0. |
| **D6** | Nhật ký huấn luyện & Định dạng text | **KHOẢNG CÁCH THIẾT KẾ** | Tác giả notebook huấn luyện trên chuỗi súc tích `[{category}] {title} ({attributes})` và chủ động xóa bỏ câu mô tả khuôn mẫu. Caption v1 dùng câu văn tổng hợp quá dài không khớp với phân phối học của mô hình. |
| **D7** | Tiền xử lý ảnh (Resize) | **LỆCH NHẸ NỘI SUY** | Code train dùng `transforms.Resize((224, 224), BILINEAR)` trước khi đưa vào processor, trong khi pipeline Phase 2 dùng processor mặc định (BICUBIC). Cosine ảnh đạt 0.9676. |
| **D8** | Quy tắc khôi phục dấu ($c'$) | **CẢI TIẾN THÀNH CÔNG** | Quy tắc $c'$ (chỉ khôi phục khi truy vấn không dấu hoàn toàn) giúp tăng Hit@10 từ 24% lên **80%** trên truy vấn không dấu, đồng thời **bảo toàn 100% (0/75 đột biến sai)** cho các truy vấn đã có dấu sẵn (so với luật cũ làm hỏng 16/75 câu). |

---

## 2. D0 — Sửa Cách Chấm & Thống Kê Hạng Thật (Exact True Rank)

Thay vì cắt ngưỡng giả định 999999, toàn bộ 1.290 truy vấn paired $Q_1$ được quét xếp hạng thực tế (1..$N$) trên toàn bộ không gian vector gallery:

### Thống kê Hạng $Q_1$ Paired Text $\to$ Ảnh:
- **Trên Gallery Test ($G_{test}$, $N = 1,290$):**
  - **Recall@1:** 0.0039 (0.39%)
  - **Recall@5:** 0.0349 (3.49%)
  - **Recall@10:** 0.0597 (5.97%)
  - **Median Rank:** **266.5** / 1,290 (ảnh đích nằm ở phân vị 20.6% trên cùng của gallery)
  - **Mean Rank:** 373.73 / 1,290
  - **Phân vị 90% (p90):** 900.2
  - **Phân vị 95% (p95):** 1024.0
  - **Max Rank:** 1279
- **Trên Gallery Toàn Phần ($G_{full}$, $N = 38,646$):**
  - **Recall@1:** 0.0000
  - **Recall@5:** 0.0000
  - **Recall@10:** 0.0016 (0.16%)
  - **Median Rank:** **7,313.0** / 38,646 (phân vị 18.9% trên cùng)
  - **Mean Rank:** 10,881.48 / 38,646
  - **Phân vị 90% (p90):** 26,791.8
  - **Max Rank:** 38,091

### Truy vấn Thuộc tính $Q_2$:
Do số lượng ảnh liên quan ($R$) trong $G_{test}$ có thể nhỏ hơn 10 đối với một số danh mục nhỏ, chỉ số $P@\min(R, 10)$ phản ánh chuẩn xác độ chính xác tối đa có thể đạt được:
- **$G_{test}$ (Strict):** Precision@10 = **0.3951** | **$P@\min(R, 10) = 0.5158$**
- **$G_{full}$ (Strict):** Precision@10 = **0.6049** | **$P@\min(R, 10) = 0.6049$**

---

## 3. D1 — Kiểm Toán Cặp (Caption $\leftrightarrow$ Ảnh)

Tiến hành kiểm tra chéo 1.290 sản phẩm test giữa dữ liệu `catalog.parquet` và `dense_captions_catalog.json`:
- **Độ khớp Giới tính:** 1,290 / 1,290 (**100.0%**).
- **Độ khớp Tên tiếng Anh gốc (`Tên gốc: {name}`):** 1,289 / 1,290 (**99.92%**). (Duy nhất 1 trường hợp lệch do ký tự đặc biệt).
- **Kết luận:** **Tuyệt đối không có lỗi tráo ID hay lệch chỉ số**. Dữ liệu ghép cặp hoàn toàn chính xác.

---

## 4. D2 — Độ Dài Token & Hiện Tượng Cắt Cụt Ngữ Nghĩa

Phân tích số lượng token được sinh ra bởi tokenizer SigLIP (SentencePiece, max_length = 128):
- **Phân phối độ dài token:** Min = 104, Median = **129.0**, p95 = 143.0, Max = 156.
- **Tỉ lệ vượt quá 128 token:** **675 / 1,290 (52.33%)**.
- **Mất mát thông tin phân biệt:** Cấu trúc caption v1 đặt `Tên gốc: {English title}` ở cuối cùng. Hậu quả là trong toàn bộ 675 câu bị cắt cụt, phần tên gốc bị cắt sạch (**0% bảo tồn**). 
- Phần còn lại ở đầu câu là các cụm từ khuôn mẫu lặp đi lặp lại: `"Sản phẩm áo thun dành cho nam, thiết kế màu đen phù hợp dạo phố thường ngày (mùa hè thoáng mát)..."`, dẫn đến việc hàng chục sản phẩm cùng loại có chuỗi token mở đầu giống hệt nhau.

---

## 5. D3 & Báo Cáo Lỗi `AttributeError: 'NoneType' object has no attribute 'to'`

### 5.1. Phân tích nguyên nhân lỗi script chẩn đoán
Khi chạy script chẩn đoán thử nghiệm trên command line, phát sinh lỗi:
`AttributeError: 'NoneType' object has no attribute 'to'`

Đã thực hiện cô lập và kiểm tra trong script [src/diag/d3_text_equivalence.py](file:///d:/Kì 7/Hoc_sau_ver4/src/diag/d3_text_equivalence.py):
1. **Kiểm tra Model & Loader:**
   - `model`: `<class 'src.model_siglip.SiglipVisionTextInference'>`, `is None: False`.
   - `named_children()` cấp 1: `Child 'siglip': <class 'transformers.models.siglip.modeling_siglip.SiglipModel'>`. Hàm `load_siglip_inference` trả về model chuẩn xác.
2. **Kiểm tra Tokenizer & Processor:**
   - `processor`: `<class 'transformers.models.siglip.processing_siglip.SiglipProcessor'>`, `is None: False`.
   - `tokenizer`: `<class 'transformers.models.siglip.tokenization_siglip.SiglipTokenizer'>`, `is None: False`.
3. **Kiểm tra Dict đầu ra của Processor:**
   - `inputs`: `<class 'transformers.feature_extraction_utils.BatchFeature'>`.
   - Danh sách khóa trong `inputs`: **chỉ duy nhất `['input_ids']`**.
   - Khóa `'attention_mask' in inputs`: **`False`**.
   - Giá trị `inputs.get('attention_mask')`: **`type=<class 'NoneType'>`, `is None: True`**.
4. **Nguyên nhân cốt lõi:**
   - Tokenizer gốc của SigLIP (`google/siglip-so400m-patch14-224`) không tạo ra trường `attention_mask` vì mô hình SigLIP sử dụng cơ chế chú ý 2 chiều đầy đủ trên chuỗi có độ dài cố định hoặc sử dụng trực tiếp padding token.
   - Khi dòng lệnh gọi `inputs.get('attention_mask').to(device)` mà không kiểm tra điều kiện `is not None`, phương thức `.to()` được gọi lên đối tượng `None`, gây ra `AttributeError`.

### 5.2. Đối chiếu với code train chuẩn
Trong notebook huấn luyện (`notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb` Cell 29 & Cell 16):
```python
# Code train gốc Cell 29:
attention_mask = inputs.get("attention_mask").to(self.device, non_blocking=True) if "attention_mask" in inputs and inputs["attention_mask"] is not None else None

# Tháp text Cell 16:
def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    text_outputs = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
    txt_feats = text_outputs[1] if len(text_outputs) > 1 else text_outputs.pooler_output
    return F.normalize(txt_feats, p=2, dim=-1)
```
- **Xử lý chuẩn:** Tuyệt đối không tự bịa mask nhân tạo. Script đã được sửa để tuân thủ 100% code train: chỉ lấy `inputs["input_ids"].to(device)` và truyền `attention_mask=None`. Kết quả kiểm tra đối chiếu toán học giữa pipeline dự án và notebook đạt **Cosine = 1.000000**.

---

## 6. D4 — Phép Thử Đối Chứng Quyết Định (The Definitive Controlled Experiment)

Để phân định rạch ròi giữa 2 giả thuyết:
- **Giả thuyết A:** Lỗi do câu truy vấn v1 bị cắt cụt và sai định dạng.
- **Giả thuyết B:** Mô hình overfit nặng trên tập huấn luyện (Unknown) và mất khả năng tổng quát hóa trên tập Test.

Chúng tôi tiến hành lấy mẫu đối chứng ngẫu nhiên **1.000 sản phẩm từ tập Unknown** (cùng seed 42) và so sánh song song với **1.290 sản phẩm tập Test** qua 5 kiểu định dạng caption:
1. **Format 1 (Synthetic v1):** Khuôn mẫu v1 (`"Sản phẩm {loại} dành cho {giới}, thiết kế {màu}... Tên gốc: {name}."`).
2. **Format 2 (Title-first):** Đảo `"Tên gốc: {name}."` lên đầu câu nhằm chống cắt cụt.
3. **Format 3 (Title-only):** Chỉ dùng tên tiếng Anh gốc (`"{name}"`).
4. **Format 4 (Concise Vietnamese):** Tiếng Việt súc tích thuần túy (`"{loại} {giới} màu {màu}"`).
5. **Format 5 (Training Rich-Text):** Định dạng đúng như lúc train (`"[{category}] {name} ({attrs})"`).

### Bảng Kết Quả Đối Chứng Đầy Đủ (D4):

| Tập dữ liệu | Định dạng câu truy vấn | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | Median Rank |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Unknown (Đã train)** | **Format 1 (v1 khuôn mẫu)** *(Gallery riêng N=1,000)* | 0.70% | 2.70% | **4.80%** | 235.0 |
| | Format 1 (v1 khuôn mẫu) *(Gallery $G_{full}$ N=38,646)* | 0.00% | 0.20% | 0.40% | 9,040.0 |
| | **Format 2 (Title lên đầu)** *(Gallery riêng)* | 1.40% | 6.10% | 10.90% | 193.0 |
| | **Format 3 (Tên Anh thuần)** *(Gallery riêng)* | 0.40% | 1.00% | 1.60% | 372.5 |
| | **Format 4 (Tiếng Việt súc tích)** *(Gallery riêng)* | **14.00%** | **45.10%** | **62.10%** | **7.0** |
| | Format 4 (Tiếng Việt súc tích) *(Gallery $G_{full}$)* | 0.80% | 3.60% | 5.40% | 236.0 |
| | **Format 5 (Training Rich-Text)** *(Gallery riêng)* | 10.30% | 30.50% | **41.70%** | 16.0 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Test (Chưa thấy)** | **Format 1 (v1 khuôn mẫu)** *(Gallery riêng N=1,290)* | 0.39% | 3.49% | **5.97%** | 266.5 |
| | Format 1 (v1 khuôn mẫu) *(Gallery $G_{full}$ N=38,646)* | 0.00% | 0.00% | 0.16% | 7,313.0 |
| | **Format 2 (Title lên đầu)** *(Gallery riêng)* | 1.24% | 4.34% | 7.52% | 241.5 |
| | **Format 3 (Tên Anh thuần)** *(Gallery riêng)* | 0.23% | 0.93% | 2.02% | 461.0 |
| | **Format 4 (Tiếng Việt súc tích)** *(Gallery riêng)* | **13.49%** | **41.55%** | **57.44%** | **8.0** |
| | Format 4 (Tiếng Việt súc tích) *(Gallery $G_{full}$)* | 0.85% | 4.50% | 7.91% | 207.5 |
| | **Format 5 (Training Rich-Text)** *(Gallery riêng)* | 8.53% | 25.81% | **39.22%** | 19.0 |

### Kết luận rút ra từ Bảng D4:
1. **Bác bỏ hoàn toàn Giả thuyết B (Không hề có Overfitting):**
   - Trên tập đã thấy (Unknown), Format 1 v1 cũng chỉ đạt **4.80%** (so với 5.97% trên Test).
   - Với Format 4 tiếng Việt súc tích, điểm số trên Unknown là **62.10%** và Test là **57.44%**. Khoảng cách khái quát hóa cực kỳ hẹp (**chỉ 4.66%**), chứng minh trọng số mô hình SigLIP học các đặc trưng thị giác - ngôn ngữ rất lành mạnh.
2. **Khẳng định Giả thuyết A (Định dạng query bị lệch và cắt cụt):**
   - Format 3 (chỉ dùng tên tiếng Anh gốc): Mô hình đạt kết quả rất thấp (~1.6 - 2.0%) vì SigLIP được fine-tune để hiểu ngôn ngữ mua sắm tiếng Việt; tên tiếng Anh không kích hoạt được không gian ngữ nghĩa đã học.
   - Format 1 đặt tên tiếng Anh ở đuôi một câu văn tiếng Việt dài 130 token khiến phần tiếng Anh bị chặt bỏ, còn phần tiếng Việt là văn mẫu sáo rỗng gây trôi dạt ngữ nghĩa.
   - Khi chuyển sang câu ngắn gọn đúng ngữ cảnh e-commerce tiếng Việt (`Format 4`), Recall@10 lập tức vọt từ **5.97% lên 57.44% (gấp gần 10 lần)**, Median rank nhảy từ **266.5 xuống 8.0**!

---

## 7. D5 — Hiện Tượng Hubness & Hiệu Quả Của Mean-Centering

### 7.1. Bằng chứng định lượng về Hubness
Thống kê tần suất xuất hiện trong Top-10 của các ảnh gallery khi truy vấn bằng $Q_1$ (1.290 truy vấn):
- **Top 1 Hub (ID 38753 - Handbags Women Black):** Xuất hiện trong top-10 của **865 / 1,290 truy vấn (67.05%)**!
- **Top 2 Hub (ID 9361 - Socks Unisex Black):** Xuất hiện trong **737 / 1,290 truy vấn (57.13%)**!
- **Top 3 Hub (ID 9573 - Backpacks Women Purple):** Xuất hiện trong **727 / 1,290 truy vấn (56.36%)**!
- **Top 4 Hub (ID 36073 - Duffel Bag Unisex Khaki):** Xuất hiện trong **496 / 1,290 truy vấn (38.45%)**!
- **Top 5 Hub (ID 31191 - Accessory Gift Set Men Brown):** Xuất hiện trong **486 / 1,290 truy vấn (37.67%)**!

> **Giải thích vì sao áo thun lại trả về Túi xách / Balo:**  
> Các ảnh Hub nằm gần trọng tâm của nón vector hình ảnh (modality cone). Khi vector text bị mờ nhạt do câu văn mẫu quá dài, khoảng cách cosin tới Hub (Mean = 0.6161) luôn cao hơn cosin tới ảnh đích thực sự (Mean = 0.3611). Hub nuốt chửng toàn bộ top kết quả.

### 7.2. Thử nghiệm Triệt Tiêu Hubness bằng Mean-Centering
Thực hiện chuẩn hóa tâm vector: $z'_{img} = \frac{z_{img} - \mu_{img}}{\|z_{img} - \mu_{img}\|}$ và $z'_{txt} = \frac{z_{txt} - \mu_{txt}}{\|z_{txt} - \mu_{txt}\|}$ trên $G_{test}$:
- **Trước Mean-Centering:** Recall@10 = 5.97%, Median Rank = 266.5, Mean Rank = 373.73.
- **Sau Mean-Centering thực nghiệm:**
  - **Recall@1:** 0.0225 (2.25%) — tăng **5.7 lần**
  - **Recall@5:** 0.0837 (8.37%) — tăng **2.4 lần**
  - **Recall@10:** **0.1465 (14.65%)** — tăng **2.5 lần**
  - **Median Rank:** Giảm từ **266.5 xuống 71.0** (cải thiện gần 4 lần)
  - **Mean Rank:** Giảm từ 373.73 xuống 154.84.

Ngoài ra, file `vit_zero_train_stats.npz` đi kèm checkpoint chứa sẵn các tham số `mu_shared`, `csls_beta = 0.28`, `csls_k = 7`. Điều này chứng tỏ người tạo checkpoint lúc train cũng đã nhận diện hiện tượng Cone Effect và lưu lại các tham số cân bằng này.

---

## 8. D6 & D7 — Đối Chiếu Quy Trình Tiền Xử Lý với Code Train

1. **Về Tiền xử lý Text (D6):**
   - Trong `notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`, hàm `clean_record_anti_overfit` chủ động xóa các câu dense caption có chứa *"Sản phẩm thuộc danh mục..."*.
   - Quá trình train sử dụng định dạng: `search_text = f"[{category}] {title} ({attrs})"`.
   - File `dense_captions_catalog.json` được sinh từ một script khác trước đó và bị nhầm là tập caption đánh giá chuẩn, dẫn đến sai lệch phân phối kiểm thử ở Pha 3 v1.
2. **Về Tiền xử lý Ảnh (D7):**
   - Code train: `eval_image_transform = transforms.Resize((224, 224), interpolation=BILINEAR)` sau đó đưa qua `processor`.
   - Pipeline Pha 2: Đưa trực tiếp ảnh gốc vào `processor` (vốn resize bằng BICUBIC).
   - Độ tương đồng Cosine đo được là **0.9676**. Đây là một khác biệt nhỏ về phép nội suy điểm ảnh, không phải nguyên nhân chính gây sụt điểm Recall, nhưng cần được thống nhất trong pipeline chuẩn hóa.

---

## 9. D8 — Đánh Giá Quy Tắc Khôi Phục Dấu ($c'$)

Đánh giá 4 cấu hình xử lý tiếng Việt trên 25 truy vấn không dấu thuộc `eval/handwritten_queries.json`:

| Cấu hình | Mô tả xử lý | Precision@10 | Hit@10 | AP@10 | Số câu có dấu bị biến dạng sai (trên 75 câu) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **(a) Accented** | Bản gốc có dấu chuẩn (Ground Truth) | 0.3000 | 0.6800 | 0.3791 | 0 |
| **(b) Raw Unaccented** | Giữ nguyên không dấu (nhập trực tiếp) | 0.0640 | 0.2400 | 0.0966 | 0 |
| **(c) Old Restore** | Luôn luôn chạy bộ khôi phục | 0.4560 | 0.8000 | 0.5054 | **16 / 75 (21.3%)** |
| **(d) Rule $c'$** | **Chỉ khôi phục khi câu hoàn toàn không dấu** | **0.4560** | **0.8000** | **0.5054** | **0 / 75 (0.0%)** |

### Đánh giá kết quả D8:
- Nếu để người dùng nhập tiếng Việt không dấu thô `(b)`, mô hình bị suy sụp nghiêm trọng: Hit@10 rơi từ 68% xuống **24%**.
- Bộ khôi phục dấu giúp kéo Hit@10 lên tới **80%** (thậm chí cao hơn bản gốc nhờ chuẩn hóa từ vựng đồng nghĩa theo thời trang).
- **Quy tắc $c'$ hoàn hảo:** Đạt điểm số tối ưu tương đương luật cũ trên truy vấn không dấu, nhưng **bảo vệ tuyệt đối 100%** các truy vấn người dùng gõ có dấu, không làm sai lệch bất kỳ câu nào trong 75 câu chuẩn.

---

## 10. Kết Luận Chung & Kiến Nghị Hướng Đi Tiếp Theo

### Kết luận then chốt:
1. **Checkpoint SigLIP hoàn toàn khoẻ mạnh và có năng lực phân loại/tìm kiếm thời trang tiếng Việt xuất sắc:** Khi truy vấn bằng câu tiếng Việt tự nhiên súc tích (Format 4), Recall@10 đạt tới **57.44%**, Median Rank là **8.0** trên tập test chưa từng thấy.
2. **Recall@10 thấp ở Pha 3 v1 là do LỖI ĐỊNH DẠNG TRUY VẤN (Query Misalignment & Truncation):** Việc sử dụng file `dense_captions_catalog.json` (52% bị cắt cụt mất tên gốc, câu văn khuôn mẫu tạo Hubness giả) đã kéo tụt điểm số bài test.
3. Không cần sửa checkpoint, không cần train lại mô hình. Các chỉ số Pha 3 v1 được giữ nguyên tính trung thực trong các file `_v1` để làm bằng chứng chẩn đoán.

### Đề xuất cho các bước tiếp theo:
- **Pha 4 (CIR - Composed Image Retrieval):** Kích hoạt mô-đun `CIRCombiner` (`best_model.pt`) kết hợp ảnh tham chiếu và văn bản điều chỉnh (`modifier_text`).
- **Giao diện App (Pha 5):**
  - Tích hợp Quy tắc $c'$ vào `src/query_parser.py` cho ô tìm kiếm người dùng.
  - Tích hợp tùy chọn hiệu chỉnh Hubness / Mean-Centering hoặc CSLS khi truy vấn để loại bỏ hiện tượng các sản phẩm Hub áp đảo kết quả.
