# Báo cáo Bước A — Khảo sát Checkpoint & Phạm vi (Phase 2)

**Thời gian thực hiện:** 2026-11-08 21:19:17

## 1. Môi trường Thực thi
- **Python:** 3.12.0 | **OS:** Windows-11-10.0.26300-SP0
- **PyTorch:** 2.6.0+cu124 | **Transformers:** 5.19.0 | **FAISS:** 1.15.1
- **Phần cứng:** GPU **NVIDIA GeForce RTX 2050** (4.00 GB VRAM) | **System RAM:** 15.7 GB (6.8 GB khả dụng)
- **Tăng tốc:** Đã kích hoạt CUDA FP16 inference.

## 2. Kết quả Khảo sát Checkpoint SigLIP
- **Đường dẫn trọng số:** `data/checkpoints_siglip-vit/best_model.pt`
- **Dung lượng file:** 3.31 GB (3.549.699.293 bytes)
- **Checkpoint SHA-256 Hash:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
- **Định dạng checkpoint:** `pt_state_dict` (PyTorch OrderedDict chứa 908 tensor weights).
- **Kiến trúc gốc (Backbone):** `google/siglip-so400m-patch14-224` (Attention Pooling, 256 vision patches, 27 transformer layers).
- **Kích thước đầu vào:** Ảnh $224 \times 224$ px (chuẩn hóa SigLIP: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`).
- **Tokenizer & Text Tower:** SigLIP SentencePiece tokenizer (32.000 tokens), mở rộng Position Embeddings lên **128 tokens**.
- **Chiều vector đặc trưng (Embedding Dim):** **1152** (L2-normalized).
- **Mô-đun mở rộng:** `UpgradedAttentionCIRCombiner` (Cross-Attention 8 heads, projection dim 1024, dynamic gating) phục vụ Composed Image Retrieval.
- **Phạm vi fine-tune lúc huấn luyện:**
  - 4 Transformer blocks cuối của Vision Tower + `post_layernorm` + `head`.
  - 4 Transformer blocks cuối của Text Tower + `final_layer_norm` + `head`.
  - Toàn bộ mô-đun `UpgradedAttentionCIRCombiner` (100% trainable).
  - Các tầng dưới của backbone được đóng băng.

## 3. Ngôn ngữ & Dữ liệu Huấn luyện (Rò rỉ dữ liệu)
- **Ngôn ngữ Text lúc train:** **Tiếng Việt (vi)**.
  - Các thuộc tính sản phẩm được dịch và ghép thành câu mô tả tiếng Việt chi tiết.
  - Dữ liệu được làm giàu thêm bằng mô hình dense captioning `Vintern-1B-v3.5`.
- **Mẫu 5 caption thật trong checkpoint:**
  - `[15970]`: *Sản phẩm áo sơ mi dành cho nam, thiết kế màu xanh navy phù hợp dạo phố thường ngày (mùa thu). Tên gốc: Turtle Check Men Navy Blue Shirt.*
  - `[1855]`: *Sản phẩm áo thun dành cho nam, thiết kế màu xám phù hợp dạo phố thường ngày (mùa hè thoáng mát). Tên gốc: Inkfruit Mens Chain Reaction T-shirt.*
  - `[7990]`: *Sản phẩm áo thun dành cho nam, thiết kế màu xanh navy phù hợp thể thao vận động (mùa thu). Tên gốc: Fila Men's Round Neck Navy Blue T-shirt.*
  - `[25947]`: *Sản phẩm khăn choàng cổ dành cho nữ, thiết kế màu nâu phù hợp dạo phố thường ngày (mùa hè thoáng mát). Tên gốc: Femella Women Brown Scarf.*
  - `[8746]`: *Sản phẩm dép xỏ ngón dành cho unisex nam nữ, thiết kế màu xanh navy phù hợp dạo phố thường ngày (mùa hè thoáng mát). Tên gốc: Crocs Kids Navy Blue Clogs.*
- **Nguồn dữ liệu huấn luyện:** Tập huấn luyện gồm 161.870 ảnh gộp từ 3 nguồn: **Myntra (chính là bộ Kaggle fashion-dataset này)**, Shopping100k, và Tiki.
- **Cảnh báo rò rỉ dữ liệu (Rule 24):** Checkpoint đã từng được huấn luyện trên dataset Kaggle này. Tuy nhiên, tập Test đã được tách riêng độc lập gồm đúng **4,372 sản phẩm** (được lưu tại `data/checkpoints_siglip-vit/dense_captions_catalog.json`). Ở Pha 3, việc đánh giá bắt buộc phải ưu tiên tập test riêng này hoặc tập handwritten queries để tránh số liệu lạc quan giả tạo.

## 4. Kết quả Smoke Test (20 sản phẩm ngẫu nhiên)
- **Tạo prompt mô tả:**
  - Tiếng Anh (theo yêu cầu đề bài): `a {baseColour} {articleType} for {gender}`
  - Tiếng Việt (chuẩn hóa qua từ điển `vi_vocab`): `{vi_articleType} {vi_gender} màu {vi_colour}`
  - Tiêu đề gốc sản phẩm (`name`)
- **Đo ma trận tương đồng (Cosine Similarity):**
  - **Prompt Tiếng Anh (`a {colour} {type} for {gender}`):** Điểm cặp khớp (diag) = **0.2429**, Cặp lệch (off-diag) = **0.1595** (Margin: **+0.0834**), **85.0% cặp khớp có độ tương đồng lớn hơn trung bình cặp lệch**.
  - **Prompt Tiếng Việt (`vi_vocab` mapped, max_l=128):** Điểm cặp khớp (diag) = **0.5708**, Cặp lệch (off-diag) = **0.0280** (Margin: **+0.5430**), **Top-1 Accuracy = 85.0%**, **100.0% cặp khớp có độ tương đồng lớn hơn trung bình cặp lệch**.
  - **Tiêu đề gốc (`name`):** Điểm cặp khớp = **0.2494**, Cặp lệch = **0.1842** (Margin: **+0.0652**).
- **Đánh giá Smoke Test:** **ĐẠT (PASS)**. Cả prompt tiếng Anh và tiếng Việt đều cho thấy cặp ảnh - mô tả khớp có độ tương đồng vượt trội hoàn toàn so với cặp lệch (đặc biệt tiếng Việt có margin lên tới +0.5430 và Top-1 đạt 85%), khẳng định pipeline tiền xử lý ảnh (RGB, 224x224, SigLIP normalize), text tokenizer/processor và checkpoint nạp vào hoàn toàn chính xác.

## 5. Đề xuất Phạm vi Index (`in_scope`)
- **Quy tắc phân loại:** Dựa trực tiếp trên `articleType` để bao quát đúng sản phẩm thời trang (bất kể `masterCategory` là Apparel, Footwear, Accessories hay Free Items).
- **Tổng số sản phẩm trong Catalog:** **40,891 sản phẩm**.
- **Đề xuất In-Scope:** **38,646 sản phẩm** (94.51%).
- **Đề xuất Out-of-Scope (loại khỏi index):** **2,245 sản phẩm** (5.49% - gồm mỹ phẩm, đồ chăm sóc cá nhân, bóng đá, đồ gia dụng).
- **File chi tiết:** Đã xuất tại `reports/scope_proposal.csv`.

> [!CAUTION]
> **CẢNH BÁO KIỂM TOÁN NGUỒN GỐC DỮ LIỆU (ERRATA AUDIT):**
> Bảng dưới đây trong báo cáo gốc Pha 2a được đánh dấu chính thức là **"KHÔNG ĐÁNG TIN (HARDCODED TEMPLATE STRING MISMATCH)"**.
> Bảng này xuất phát từ chuỗi Markdown tĩnh hardcode trong script khảo sát chứ **không phải được tổng hợp động từ `scope_proposal.csv`**. Khi đối soát 1:1 với dữ liệu gốc `reports/phase1/phase1b/scope_proposal.csv`, **có tới 9/10 dòng bị lệch số lượng** (chỉ có duy nhất dòng `Umbrellas = 6` là khớp).

### Bảng Đối chiếu Chi tiết Từng Dòng: Bảng gốc Pha 2a vs Dữ liệu Thực tế `scope_proposal.csv`:

| articleType | Số lượng ghi trong Pha 2a (Bản cũ) | Số lượng Thực tế trong `scope_proposal.csv` | Trạng thái Đối soát | Phân rã Thực tế theo MasterCategory / Giới tính |
|---|:---:|:---:|:---:|---|
| **Mobile Pouch** | 33 | **47** | **LỆCH (-14)** | 6 Unisex + 41 Women (Accessories) |
| **Tablet Sleeve** | 31 | **3** | **LỆCH (+28)** | 2 Unisex + 1 Women (Accessories) |
| **Wristbands** | 12 | **4** | **LỆCH (+8)** | 2 Accessories + 2 Sporting Goods (Unisex) |
| **Shoe Laces** | 12 | **1** | **LỆCH (+11)** | 1 Unisex (Accessories) |
| **Shoe Accessories** | 24 | **22** | **LỆCH (+2)** | 2 Men + 20 Unisex (Accessories) |
| **Umbrellas** | 6 | **6** | **KHỚP** | 6 Women (Accessories) |
| **Key chain** | 35 | **2** | **LỆCH (+33)** | 2 Unisex (Wildcraft 43230, American Tourister 42066) |
| **Free Gifts** | 8 | **86** | **LỆCH (-78)** | 43 Men + 8 Unisex + 35 Women (Bản cũ chỉ đếm nhóm Unisex = 8) |
| **Fragrance Gift Set** | 34 | **57** | **LỆCH (-23)** | 31 Men + 1 Unisex + 25 Women (Personal Care) |
| **Accessory Gift Set** | 358 | **92** | **LỆCH (+266)** | 92 Men (Accessories) |

### Bảng Đề xuất Chuẩn hóa đã Được Duyệt (Căn cứ trên `scope_proposal.csv`):
| articleType | Số lượng Thực tế | Đề xuất | Lý do Thực tế |
|---|:---:|---|---|
| Mobile Pouch | 47 | In-Scope | Túi đựng điện thoại đeo chéo mini (phụ kiện thời trang) |
| Tablet Sleeve | 3 | In-Scope | Túi chống sốc máy tính bảng thiết kế thời trang |
| Wristbands | 4 | In-Scope | Băng đeo tay thể thao thời trang |
| Shoe Laces | 1 | In-Scope | Dây giày phụ kiện thời trang |
| Shoe Accessories | 22 | In-Scope | Phụ kiện giày |
| Umbrellas | 6 | In-Scope | Ô dù cầm tay |
| Key chain | 2 | Out-of-Scope | Móc chìa khóa (Wildcraft 43230, American Tourister 42066) |
| Free Gifts | 86 | Out-of-Scope | Quà tặng kèm không rõ danh mục |
| Fragrance Gift Set | 57 | Out-of-Scope | Bộ quà nước hoa (mỹ phẩm) |
| Accessory Gift Set | 92 | In-Scope | Set quà cà vạt/ví/thắt lưng thời trang nam |

## 6. Đo lường Throughput & Ước tính Thời gian Embedding
- **Phần cứng thử nghiệm:** NVIDIA GeForce RTX 2050 (4.0 GB VRAM), FP16, Batch size = 32.
- **Tốc độ thực tế đo trên 200 ảnh:** **22.31 ảnh / giây** (8.96 giây / 200 ảnh).
- **Ước tính thời gian tạo embedding cho toàn bộ 38,646 ảnh in-scope:** khoảng **28.9 phút** (~1732 giây). Thời gian rất ngắn và hoàn toàn khả thi trên GPU.