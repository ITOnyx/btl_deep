# Báo cáo Chẩn đoán Baseline & Ceiling Phân tích (Pha 5b)

- **Tập dữ liệu chẩn đoán:** `eval/handwritten_queries.json` (**DEV SET**, N=100)
- **Official Test Gallery:** `index/test_gallery/` (N=1.290 sản phẩm test)
- **Chính sách Ground Truth:** `hybrid` (86 truy vấn có $\ge 1$ ảnh đúng, 14 truy vấn không có GT)
- **Commit Baseline:** `bcc5284`
- **Seed:** 42

---

## 1. Phân tích Giới hạn Trần (Ceiling Analysis): Vai trò của Candidate Pool

Khi `candidate_pool = 100`, chỉ có 100 ứng viên gần nhất từ ANN HNSW được chuyển sang bộ Re-ranking (Soft Attribute Bonus). Nếu tăng `candidate_pool = 1.290` (toàn bộ Test Gallery), tất cả ảnh trong gallery đều được bộ lọc quy tắc thuộc tính chấm điểm.

| Cấu hình | candidate_pool | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| **Model thuần** | 100 | 0.2907 | 0.5698 | 0.7093 | 0.3279 |
| **Model thuần (Toàn gallery)** | 1.290 | 0.2907 | 0.5698 | 0.7093 | 0.3279 |
| **Hệ thống (Baseline Pha 5)** | 100 | **0.6047** | **0.8256** | **0.8488** | **0.5537** |
| **Hệ thống (Ceiling Pool toàn phần)** | 1.290 | **0.6047** | **0.8488** | **0.8721** | **0.5839** |
| **Mức chênh lệch ($\\Delta$)** | +1.190 | `+0.0000` | `+0.0233` | `+0.0233` | `+0.0302` |

### Trả lời câu hỏi định lượng:
- Khoảng cách từ baseline mAP (0.5537) tới ngưỡng chỉ tiêu 0.75 là **`0.1963`**.
- Mức tăng mAP khi mở rộng candidate_pool từ 100 lên 1.290 là **`+0.0302`** (tăng từ `0.5537` lên `0.5839`).
- Tỷ lệ khoảng cách mAP do candidate pool gây ra: **`15.38%`**.
- Đồng thời, Recall@10 tăng từ `0.8488` lên **`0.8721`** (thu hồi thêm 2 truy vấn vào Top-10).

---

## 2. Phân loại Nguyên nhân Lỗi (Error Taxonomy)

Trong số 86 truy vấn có GT trên test gallery, có **13 truy vấn** chưa đạt Top-10 ở baseline (sys_rank_first_rel > 10 hoặc > 100):

| Nhóm lỗi | Số lượng | Tỷ lệ (%) | Đặc điểm nhận diện |
|---|---:|---:|---|
| **(A) Không có ảnh đúng trong pool 100** | 5 | 38.5% | Xem bảng chi tiết bên dưới |
| **(C) Parser/lexicon hiểu sai thuộc tính** | 5 | 38.5% | Xem bảng chi tiết bên dưới |
| **(A) Không có ảnh đúng trong pool 100 + (C) Parser thiếu thuộc tính** | 3 | 23.1% | Xem bảng chi tiết bên dưới |

### Chi tiết từng truy vấn thất bại:

| ID | Truy vấn | Có dấu | n_rel | Rank thuần | Rank hệ thống | Rank ceiling (pool 1290) | Trong pool 100? | Nhóm lỗi & Chi tiết |
|---|---|:---:|---:|---:|---:|---:|:---:|---|
| 26 | `quan jean nam xanh` | Không | 9 | >100 | >100 | 51 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 (None) |
| 29 | `quan tay nam cong so mau den` | Không | 1 | >100 | >100 | 3 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 (None) |
| 41 | `quần lót boxer nam co giãn tốt` | Có | 7 | 14 | 14 | 14 | Có | (C) Parser/lexicon hiểu sai thuộc tính (Parser thiếu/sai articleType: spec=['Trunk', 'Boxers'], parsed=[]) |
| 44 | `áo liền quần jumpsuit nữ thanh lịch` | Có | 1 | >100 | >100 | >100 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 + (C) Parser thiếu thuộc tính (Parser thiếu/sai articleType: spec=['Jumpsuit'], parsed=[]) |
| 63 | `giày đá bóng sân cỏ nhân tạo nam` | Có | 39 | 62 | 48 | 48 | Có | (C) Parser/lexicon hiểu sai thuộc tính (Parser thiếu/sai articleType: spec=['Sports Shoes'], parsed=[]) |
| 72 | `vi da nam mau den` | Không | 6 | >100 | >100 | 5 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 (None) |
| 76 | `túi đeo chéo nam đựng ipad thể thao` | Có | 1 | 44 | 40 | 40 | Có | (C) Parser/lexicon hiểu sai thuộc tính (Parser thiếu/sai articleType: spec=['Messenger Bag'], parsed=['Handbags']) |
| 77 | `túi trống thể thao tập gym đựng đồ` | Có | 3 | 5 | 12 | 18 | Có | (C) Parser/lexicon hiểu sai thuộc tính (Parser thiếu/sai articleType: spec=['Duffel Bag'], parsed=[]; Parser thiếu/sai gender: spec=Unisex, parsed=None) |
| 81 | `dong ho nam day da mau den` | Không | 14 | >100 | >100 | >100 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 (None) |
| 83 | `dong ho nam day kim loai` | Không | 8 | >100 | >100 | >100 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 + (C) Parser thiếu thuộc tính (Parser thiếu/sai baseColour: spec=['Steel', 'Silver'], parsed=[]) |
| 85 | `dong ho nu mau trang thanh lich` | Không | 7 | >100 | >100 | >100 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 (None) |
| 88 | `đồng hồ nam mặt vuông dây cao su` | Có | 14 | 16 | 16 | 16 | Có | (C) Parser/lexicon hiểu sai thuộc tính (Parser thiếu/sai baseColour: spec=['Black'], parsed=[]) |
| 94 | `day nit nam da bo` | Không | 5 | >100 | >100 | 16 | **KHÔNG** | (A) Không có ảnh đúng trong pool 100 + (C) Parser thiếu thuộc tính (Parser thiếu/sai baseColour: spec=['Brown', 'Black'], parsed=[]) |

---

## 3. Phân tích Chênh lệch Có dấu vs Không dấu

| Nhóm truy vấn | Số lượng (có GT) | R@10 (Baseline Pool=100) | R@10 (Ceiling Pool=1290) | mAP (Baseline) | mAP (Ceiling) | Số query ảnh đúng rớt ngoài Pool 100 |
|---|---:|---:|---:|---:|---:|---:|
| **Không dấu (`has_diacritics=False`)** | 21 | 0.6667 | 0.7619 | 0.2240 | 0.2938 | 7 |
| **Có dấu (`has_diacritics=True`)** | 65 | 0.9077 | 0.9077 | 0.6602 | 0.6776 | 1 |

### Nhận định then chốt từ dữ liệu chẩn đoán:
1. **Khoảng cách mAP chủ yếu đến từ đâu?**
   - Mở rộng `candidate_pool` lên toàn bộ gallery test chỉ đóng góp **15.38%** mức tăng mAP (0.5537 -> 0.5839).
   - Phần còn lại (84.62%) là do **thứ tự xếp hạng sâu (ranking quality)**: ngay cả khi ở pool=1290, mAP cũng chỉ đạt 0.5839 (vẫn cách ngưỡng 0.75 một khoảng 0.1661). Lý do: tử số mAP tính precision tại mọi vị trí rank của các ảnh đúng, nhưng SigLIP text encoder chỉ đưa 1-2 ảnh đúng lên top 1-5, còn các ảnh đúng thứ 3..N bị phân tán ở các rank 20..100 dẫn đến Precision@K suy giảm nhanh.
2. **Vấn đề nhóm không dấu:**
   - Nhóm không dấu có mAP baseline chỉ đạt `0.2240` (so với `0.6602` ở nhóm có dấu).
   - Ở nhóm không dấu, có 7 truy vấn mà SigLIP ANN không tìm thấy bất kỳ ảnh đúng nào trong Top-100 candidates (Embedding của câu không dấu bị trôi xa trong không gian SigLIP).
   - Điều này khẳng định giả thuyết: Cần giải pháp **khôi phục dấu tiếng Việt (lexicon-based)** và **Dual-encode** ở Bước 3 để kéo vector câu không dấu về gần vùng biểu diễn của câu có dấu!
