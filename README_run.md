# SigLIP E-commerce Visual Search - Run Guide (Phase 3e)

Hệ thống tìm kiếm thời trang đa phương thức tiếng Việt trên tập dữ liệu thời trang (38.646 sản phẩm) sử dụng SigLIP So400M và Hybrid Retrieval (Parser + Attribute Index + Text Fusion + BM25 RRF).

---

## 1. Cấu trúc thư mục chính
- `data/catalog/catalog.parquet`: Danh mục sản phẩm đã làm sạch (38.646 sản phẩm in-scope).
- `index_v2/`:
  - `image_embeddings_v2.npy`: 38.646 vector ảnh chuẩn hóa (1152-d).
  - `text_embeddings_v2.npy`: 38.646 vector text sản phẩm chuẩn hóa (1152-d).
  - `id_map_v2.json`: Bản đồ index sang item_id.
- `src/`:
  - `query_parser.py`: Bộ phân tích truy vấn tiếng Việt trích xuất loại sản phẩm, giới tính, màu sắc, thương hiệu.
  - `lexicon_vi.json`: Từ điển 60 articleTypes, 47 baseColours, genders, usages.
  - `brands.json`: 234 thương hiệu nhận diện từ catalog.
  - `attr_index.py`: Chỉ mục thuộc tính mảng số NumPy, lọc chính xác và nới lỏng nhiều tầng (L0..L6).
  - `lexical.py`: BM25 CSR thưa tốc độ cao (~4 ms) với mở rộng song ngữ.
  - `search_pipeline.py`: Pipeline tìm kiếm hợp nhất (`VisualSearchEngine`).
- `eval/`: Bộ dữ liệu kiểm thử (Q1b, Q2, Q3, Q5, Heldout).
- `reports/`:
  - `val_grid_results.json`: Kết quả lưới 16 tổ hợp trên tập val ($\gamma^*=0.3, \alpha^*=0.5$).
  - `phase3e_inference_only.md`: Báo cáo chi tiết đánh giá Pha 3e.

---

## 2. Cách chạy Pipeline tìm kiếm (`VisualSearchEngine`)

```python
from src.search_pipeline import VisualSearchEngine

# Khởi tạo engine tìm kiếm (mặc định tham số tối ưu từ Validation)
engine = VisualSearchEngine(
    gamma=0.3,               # Thưởng thuộc tính khớp mềm
    alpha=0.5,               # Trọng số ghép ảnh và text sản phẩm
    rrf_k=60,                # Tham số hợp nhất RRF
    filter_mode="relaxed",   # Nới lỏng nhiều tầng L0..L6
    enable_bm25=True,        # Bật tìm kiếm từ vựng BM25
    enable_text_fusion=True  # Bật ghép text sản phẩm
)

# 1. Tìm kiếm theo văn bản tiếng Việt
results = engine.search_text("áo sơ mi nam trắng công sở", top_k=10)
for r in results:
    print(f"[{r['rank']}] {r['item_id']} - {r['name']} ({r['articleType']}, {r['baseColour']}) | Điểm: {r['score']:.4f} ({r['latency_ms']} ms)")

# 2. Tìm kiếm theo ảnh (Visual Search)
img_results = engine.search_image("data/images/15970.jpg", top_k=10, use_flip_aug=True, use_knn_voting=True)
for r in img_results:
    print(f"[{r['rank']}] {r['item_id']} - {r['name']} | Điểm: {r['score']:.4f}")
```

---

## 3. Chạy đánh giá toàn diện Pha 3e

```bash
# Đánh giá toàn bộ các bộ test (Q1b, Q2, Q3, Q5, Heldout, Q4)
python src/evaluate_phase3e_test.py
```
Kết quả lưu tại:
- `eval/results_3e.json`
- `reports/contact_sheets/heldout_*.png`
- `eval/manual_ratings_heldout.csv`

---

## 4. Chạy ứng dụng Web Demo (Streamlit - Pha 4)

Ứng dụng web phục vụ tìm kiếm thời trang đa phương thức được triển khai bằng Streamlit theo đúng cấu hình tại `config.yaml` (`port: 8501`).

### Khởi chạy ứng dụng:
```bash
streamlit run app/main.py --server.port 8501
```

### Các tính năng và Tab chính:
1. **Tuân thủ Luật bất biến 7 (Rule 7 Checkpoint Hash Verification):**
   - Ứng dụng tự động kiểm tra mã băm SHA-256 của checkpoint SigLIP (`best_model.pt`) đối chiếu với `config.yaml` và `index_v2/manifest.json`.
   - Nếu phát hiện sai lệch (mismatched hash), ứng dụng lập tức từ chối tải và hiển thị cảnh báo đỏ nhằm ngăn ngừa trôi dạt phân phối vector.
2. **Tab 1: Văn bản tiếng Việt $\to$ Ảnh:**
   - Trích xuất thực thể bằng `QueryParserV2` (loại sản phẩm, giới tính, màu sắc, thương hiệu, hỗ trợ cả câu có dấu và không dấu).
   - Nới lỏng đa tầng `L0..L6`, lọc BM25 RRF ($k=60$) kết hợp Soft Bonus $\gamma=0.3$ (Cấu hình C5 tối ưu).
   - Hỗ trợ chuyển đổi so sánh tức thì với Cấu hình C0 (Dense thuần SigLIP).
   - Hiển thị chi tiết độ trễ từng giai đoạn (Parser, Dense, BM25, RRF, Filter).
3. **Tab 2: Ảnh $\to$ Ảnh (Visual Search):**
   - Tải ảnh tải lên hoặc chọn ảnh mẫu từ catalog.
   - Hỗ trợ lật ảnh tăng cường (Flip Augmentation TTA) và bầu chọn thuộc tính k-NN ($k=5$).
4. **Tab 3: Ảnh + Văn bản bổ trợ (Composed Image Retrieval - CIR):**
   - Ghép đặc trưng ảnh tham chiếu và câu bổ trợ tiếng Việt (ví dụ: *"đổi sang màu đỏ"*, *"chuyển sang áo dài tay"*) qua bộ kết hợp `MultimodalSigLIPWithCIR`.
   - Hiển thị trực quan ảnh gốc bên cạnh danh sách sản phẩm gợi ý sau khi thay đổi thuộc tính.
5. **Tab 4: Đối soát & Báo cáo Hệ thống:**
   - Bảng tổng hợp đối chiếu chỉ tiêu Q1b giữa C5 vs C0.
   - Thống kê độ trễ qua 5 lần chạy (latency telemetry) và chứng nhận đạt SLA có điều kiện.
   - Bảng theo dõi và ghi chú Errata Rev5.
