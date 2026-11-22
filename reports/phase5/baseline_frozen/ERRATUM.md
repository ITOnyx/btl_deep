# ERRATUM — Đính chính về Số liệu Baseline Pha 5 Đóng băng

- **Các tệp đóng băng (giữ nguyên không sửa):** `retrieval_eval_results.json`, `caption_eval_ab.json`, `latency_sla_results.json`.
- **Nguồn gốc commit:** Số liệu Pha 5 được sinh từ working tree tại thời điểm HEAD `2e057d6` (trước khi commit các module `src/eval/*.py` và `src/serve/*.py`), và toàn bộ mã nguồn cùng kết quả đã được commit đầy đủ tại commit `bcc5284`.
- **Số liệu chính thức Baseline Pha 5 (Official Test Gallery `N=1.290`, `candidate_pool=100`, `eval_top_k=100`, chính sách `hybrid`, 86/100 truy vấn có $\ge 1$ ảnh đúng):**
  - **Model thuần (`model_only`):** Recall@1 = `0.2907`, Recall@5 = `0.5698`, Recall@10 = `0.7093`, mAP = `0.3279`.
  - **Hệ thống (`system`, `gamma=0.3`):** Recall@1 = `0.6047`, Recall@5 = `0.8256`, Recall@10 = `0.8488`, mAP = `0.5537`.
- **Giải thích sai lệch nhỏ khi dùng Exact Matmul ở Bước 1b:**
  1. `model_only` R@1 `0.2907` (FAISS HNSW trong `retrieval_eval_results.json`) vs `0.3023` (Exact numpy matmul trong 1b): khác biệt do **1 truy vấn duy nhất (Query 56: `dép quai hậu sandal nam đi học đi chơi`)** có 2 ảnh đầu bảng chênh nhau `6e-6` (`0.686506` vs `0.686500`) giữa tính toán dấu phẩy động của FAISS HNSW C++ (`idx=174`) và phép nhân ma trận numpy (`idx=274`).
  2. Hệ thống `pool=1290` mAP `0.5839` (HNSW `top_k=1290` trong 1) vs `0.5852` (Exact matmul trong 1b): do FAISS HNSW (`efSearch=128`) chỉ tìm kiếm xấp xỉ trên đồ thị khi yêu cầu `top_k = N` (toàn gallery), trong khi phép nhân ma trận `q @ E.T` là chính xác tuyệt đối 100%.
- **Quy ước chuẩn Canonical (`src/eval/canonical_eval.py`):**
  - Khi tái hiện baseline `pool=100`: sử dụng `engine.hnsw.search` khớp 100% với số liệu chính thức Pha 5 (`0.2907` / `0.5537`).
  - Khi đánh giá `pool = gallery_size` (duyệt toàn bộ gallery): sử dụng Exact inner product (`q @ E.T`) để loại bỏ sai số xấp xỉ đồ thị ở đuôi danh sách.
