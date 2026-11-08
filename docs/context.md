# ĐỀ TÀI: SigLIP + Dense Captioning cho hệ thống tra cứu sản phẩm thương mại điện tử (E-commerce Visual Search)

---

## 1. Mô tả đề tài

### Mục đích
- Xây dựng hệ thống tìm kiếm sản phẩm đa phương thức (multimodal search) cho sàn thương mại điện tử.
- Nhận đầu vào là mô tả bằng ngôn ngữ tự nhiên (tiếng Việt) hoặc ảnh sản phẩm, sau đó truy xuất các sản phẩm tương đồng.
- Hỗ trợ **Composed Image Retrieval (CIR)**: kết hợp ảnh mẫu kèm mô tả chỉnh sửa (ví dụ: ảnh một chiếc áo + *"màu xanh, cổ tròn"*).

### Công nghệ chính
- **SigLIP (ViT-SO400M):** Fine-tuned với contrastive loss trên danh mục (catalog) sản phẩm.
- **InternVL2 (InternVL2-2B):** Dense captioning tự động cho sản phẩm.
- **FAISS (HNSW Index):** Vector database tối ưu hóa cho tra cứu tương đồng tốc độ cao.

### Yêu cầu kỹ thuật
- **Recall@10:** $\ge 0.85$
- **Text-to-Image Retrieval mAP:** $\ge 0.75$
- **Độ trễ (Latency):** $< 200\text{ ms / query}$

---

## 2. Repositories GitHub tham khảo

1. **[google-research/big_vision](https://github.com/google-research/big_vision)**  
   - Big Vision repository chứa SigLIP implementation và pretrained weights.
2. **[ABaldrati/CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir)**  
   - Composed Image Retrieval với CLIP: kết hợp ảnh + text query.
3. **[salesforce/BLIP](https://github.com/salesforce/BLIP)**  
   - BLIP/BLIP-2 cho dense captioning tự động sản phẩm.

---

## 3. Bài báo tham khảo

1. **"Sigmoid Loss for Language Image Pre-Training"** — *Zhai et al., 2023*  
   - Đề xuất SigLIP thay thế hàm softmax bằng sigmoid loss, giúp cải thiện hiệu năng học của CLIP trên các batch size nhỏ và mở rộng quy mô huấn luyện hiệu quả hơn.
2. **"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"** — *Li et al., 2023*  
   - Giới thiệu kiến trúc Q-Former giúp kết nối visual feature từ frozen vision encoder với LLM một cách hiệu quả, tiết kiệm chi phí tính toán.
3. **"Composed Image Retrieval using Contrastive Learning and Task-Oriented CLIP-Based Features"** — *Baldrati et al., 2022*  
   - Kỹ thuật Composed Image Retrieval (CIR) kết hợp ảnh tham chiếu (reference image) cùng văn bản điều chỉnh (text modifier) thông qua contrastive learning.

---

## 4. Pipeline triển khai

### Bước 1: Thu thập & Tiền xử lý dữ liệu (Dataset)
- **Shopping100k:** ~101K sản phẩm thời trang.
- **Shopee Product Dataset (Kaggle):** ~42K sản phẩm.
- **Dữ liệu tự crawl:** ~10K sản phẩm từ Tiki/Shopee (ảnh + tên sản phẩm).

### Bước 2: Thiết kế kiến trúc (Architecture)
- **Vision-Language Backbone:** SigLIP (ViT-SO400M), tiến hành fine-tune lớp chiếu/lớp cuối với contrastive loss.
- **Dense Captioning:** Sử dụng mô hình InternVL2-2B để tạo caption/mô tả tự động và phong phú cho từng sản phẩm.
- **Vector Database:** Xây dựng index FAISS (HNSW) phục vụ tìm kiếm embedding thời gian thực.

### Bước 3: Huấn luyện mô hình (Training)
- **Loss function:** NT-Xent loss (hoặc Sigmoid contrastive loss).
- **Batch size:** 256.
- **Optimizer:** AdamW, learning rate = `1e-4`, linear warmup 1,000 steps.
- **Epochs:** 10 epochs.

### Bước 4: Đánh giá mô hình (Evaluation)
- **Chỉ số đánh giá:** Recall@1, Recall@5, Recall@10, mAP.
- **Khả năng xử lý tiếng Việt:** Kiểm thử độ chính xác trên tập truy vấn ngôn ngữ tiếng Việt (Vietnamese query accuracy).

### Bước 5: Ứng dụng & Demo (Deployment)
- Xây dựng giao diện tương tác bằng **Streamlit**.
- Hỗ trợ người dùng:
  - Tải ảnh sản phẩm lên hoặc nhập truy vấn văn bản tiếng Việt (hoặc kết hợp cả hai cho CIR).
  - Hiển thị danh sách Top-10 sản phẩm tương đồng nhất cùng độ tương đồng tương ứng.

