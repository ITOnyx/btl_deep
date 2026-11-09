# SigLIP E-commerce Visual Search - Run Guide (Pha 4)

Hệ thống tìm kiếm thời trang đa phương thức tiếng Việt trên tập dữ liệu thời trang (38.644 sản phẩm in-scope) sử dụng SigLIP So400M và Hybrid Retrieval (Parser V2 + Attribute Index + BM25 RRF + CIR Combiner).

---

## 1. Cấu trúc thư mục chính
- `data/catalog/catalog.parquet`: Danh mục sản phẩm đã làm sạch (38.644 sản phẩm in-scope).
- `index_v2/`:
  - `image_embeddings_v2.npy`: 38.646 vector ảnh chuẩn hóa (1152-d, map 38.644 in-scope qua `id_map_v2.json`).
  - `text_embeddings_v2.npy`: 38.646 vector text sản phẩm chuẩn hóa (1152-d).
  - `id_map_v2.json`: Bản đồ index sang item_id.
- `src/`:
  - `model_siglip.py`, `model_siglip_cir.py`: Wrapper suy luận đa phương thức SigLIP và CIR Combiner.
  - `query_parser_v2.py`: Bộ phân tích truy vấn tiếng Việt (hỗ trợ có dấu và không dấu).
  - `lexicon_vi.json`: Từ điển 60 articleTypes, 47 baseColours, genders, usages.
  - `brands_v2.json`: 234 thương hiệu nhận diện từ catalog kèm alias đã duyệt.
  - `attr_index.py`: Chỉ mục thuộc tính mảng số NumPy, lọc chính xác và nới lỏng nhiều tầng (L0..L6).
  - `lexical.py`: BM25 CSR thưa tốc độ cao (~4 ms) với mở rộng song ngữ.
  - `scoring.py`: Chuẩn hóa vector và Scorer bù tâm đặc trưng huấn luyện.
  - `search_pipeline.py`: Pipeline tìm kiếm hợp nhất (`VisualSearchEngine`).
  - `diag/`: Các kịch bản chẩn đoán, audit, kiểm toán, trắc nghiệm độ trễ lịch sử.
  - `experiments/`: Các kịch bản thực nghiệm, tuning lưới tham số, huấn luyện offline.
- `tests/`: Bộ kiểm thử tự động hệ thống (`test_app_service.py`, `test_search_pipeline.py`, `test_cir_flow.py`, `test_manifest_verification.py`).
- `eval/`: Bộ dữ liệu kiểm thử (Q1b, Q2, Q3, Q5, Heldout).
- `reports/`:
  - `phase1/`: Báo cáo làm sạch dữ liệu, đề xuất phạm vi (scope).
  - `phase2/`: Báo cáo khảo sát và xây dựng chỉ mục vector.
  - `phase3/`: Báo cáo đối soát Pha 3f/3g (`phase3f_errata.md`), lưới tham số, dữ liệu thực nghiệm A/B.
  - `phase4/`: Báo cáo kiểm thử thực tế Pha 4 (`phase4_smoke.json`, `phase4_cir_20queries_eval.json`, `phase4_trunc64_pad128_eval.json`).
  - `quarantine_log.csv`: Nhật ký cách ly ảnh theo Luật bất biến 2.
  - `run_manifest.json`: Siêu dữ liệu chạy, hash môi trường và chỉ số tái lập theo Luật bất biến 8.

---

## 2. Cách chạy Pipeline tìm kiếm (`VisualSearchEngine`)

Đối chiếu chính xác tham số tối ưu từ `config.yaml` (`gamma=0.3, alpha=1.0, enable_text_fusion=False, rrf_k=60, filter_mode="relaxed"`):

```python
from src.search_pipeline import VisualSearchEngine

# Khởi tạo engine tìm kiếm (khớp chuẩn tham số từ config.yaml)
engine = VisualSearchEngine(
    gamma=0.3,                # Thưởng thuộc tính khớp mềm (scoring.mu_gamma)
    alpha=1.0,                # Trọng số ảnh trong text fusion (alpha*=1.0 tiết kiệm RAM)
    rrf_k=60,                 # Tham số hợp nhất RRF (inference.rrf_k)
    filter_mode="relaxed",    # Nới lỏng nhiều tầng L0..L6
    enable_bm25=True,         # Bật tìm kiếm từ vựng BM25 (inference.enable_bm25)
    enable_text_fusion=False  # Đóng băng text fusion theo quy tắc F3 (inference.enable_text_fusion)
)

# 1. Tìm kiếm theo văn bản tiếng Việt (Hybrid Config 5 hoặc Dense Config 0)
results = engine.search_text("áo sơ mi nam trắng công sở", top_k=10, use_soft_bonus=True)
for r in results:
    print(f"[{r['rank']}] {r['item_id']} - {r['name']} ({r['articleType']}, {r['baseColour']}) | Điểm: {r['score']:.4f} ({r['latency_ms']:.2f} ms)")

# 2. Tìm kiếm theo ảnh (Visual Search)
img_results = engine.search_image("data/fashion-dataset/images/15970.jpg", top_k=10, use_flip_aug=True, use_knn_voting=True)
for r in img_results:
    print(f"[{r['rank']}] {r['item_id']} - {r['name']} | Điểm: {r['score']:.4f}")

# 3. Tìm kiếm ảnh + văn bản bổ trợ (Composed Image Retrieval - CIR)
cir_results = engine.search_cir("data/fashion-dataset/images/15970.jpg", modifier_text="màu xanh navy", top_k=10)
for r in cir_results:
    print(f"[{r['rank']}] {r['item_id']} - {r['name']} ({r['baseColour']}) | Điểm: {r['score']:.4f}")
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
