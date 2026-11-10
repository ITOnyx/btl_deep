# SigLIP E-commerce Visual Search - Run Guide (Pha 5)

Hệ thống tìm kiếm thời trang đa phương thức tiếng Việt trên tập dữ liệu thời trang (38.644 sản phẩm in-scope) sử dụng SigLIP So400M và Hybrid Retrieval (Parser V2 + Attribute Index + BM25 RRF + CIR Combiner).

---

## 1. Cấu trúc thư mục chính
- `data/catalog/catalog.parquet`: Danh mục sản phẩm đã làm sạch (38.644 sản phẩm in-scope, `split == "test"` có 1.290 sản phẩm).
- `index_v2/`:
  - `image_embeddings_v2.npy`: 38.646 vector ảnh chuẩn hóa (1152-d, căn hàng 38.644 in-scope qua `id_map_v2.json`).
  - `text_embeddings_v2.npy`: 38.646 vector text sản phẩm chuẩn hóa (1152-d).
  - `id_map_v2.json`: Bản đồ index sang `item_id`.
  - `manifest.json`: Siêu dữ liệu index kèm mã băm `checkpoint_hash` theo Luật bất biến 7.
- `src/`:
  - `model_siglip.py`, `model_siglip_cir.py`: Wrapper suy luận đa phương thức SigLIP và CIR Combiner.
  - `query_parser_v2.py`: Bộ phân tích truy vấn tiếng Việt (hỗ trợ có dấu và không dấu).
  - `lexicon_vi.json`: Từ điển 60 `articleType`, 47 `baseColour`, `gender`, `usage`.
  - `brands_v2.json`: 234 thương hiệu nhận diện từ catalog kèm alias đã duyệt.
  - `attr_index.py`: Chỉ mục thuộc tính mảng số NumPy, lọc chính xác và nới lỏng nhiều tầng (L0..L6).
  - `lexical.py`: BM25 CSR thưa tốc độ cao (~4 ms) với mở rộng song ngữ.
  - `scoring.py`: Chuẩn hóa vector và Scorer bù tâm đặc trưng huấn luyện.
  - `search_pipeline.py`: Pipeline tìm kiếm hợp nhất (`VisualSearchEngine`).
  - `benchmark_e2e_latency_v3.py`: Đo kiểm độ trễ E2E có kiểm soát cho 4 luồng SLA (`text_to_image`, `image_to_image_fast`, `image_to_image_accurate`, `cir_catalog`, `cir_upload`).
  - `phase5a_i2i_quality.py`: Đối chiếu chất lượng Ảnh → Ảnh giữa chế độ `fast` (1 ảnh) và `accurate` (Batch-2 TTA) trên tập test $G_{test}$ ($n = 1,290$).
- `app/`:
  - `engine.py`: Lớp dịch vụ `SearchService` thực thi kiểm toán Luật 7, quản lý SLA theo 4 luồng và trả về phân rã thời gian từng bước (`breakdown`).
  - `main.py`: Giao diện ứng dụng Web Streamlit (4 Tab tương tác và báo cáo đối soát).
- `tests/`: Bộ kiểm thử tự động hệ thống (`test_app_service.py`, `test_search_pipeline.py`, `test_cir_flow.py`, `test_manifest_verification.py`, `test_keychain_masking_and_alignment.py`).
- `eval/`: Bộ dữ liệu kiểm thử (`testset_v1.json`, `handwritten_queries.json`, `heldout_queries.json`, `q5_brand.json`, `unseen_testset_v2.json`, `gender_gt_policy.json`).
- `reports/`: Báo cáo kiểm toán theo từng pha (`phase1/`, `phase2/`, `phase3/`, `phase4/`), `quarantine_log.csv`, và `run_manifest.json`.

---

## 2. Cách chạy Dịch vụ Tìm kiếm (`SearchService` & `VisualSearchEngine`)

Mọi tham số và đường dẫn được đọc từ `config.yaml`:

```python
from app.engine import SearchService

# Khởi tạo dịch vụ (tự động kiểm tra Luật 7: checkpoint_hash khớp index_v2/manifest.json)
service = SearchService(config_path="config.yaml")

# 1. Luồng text_to_image (C5 Hybrid hoặc C0 Model thuần)
results, meta = service.search_text("áo sơ mi nam trắng công sở", config_mode="c5", top_k=10, gamma=0.3)
print(f"Flow: {meta['flow']} | Latency: {meta['breakdown']['total_ms']:.2f} ms | SLA Pass: {meta['sla_pass']}")

# 2. Luồng image_to_image (mode='fast' 1 ảnh mặc định hoặc mode='accurate' Batch-2 TTA)
img_results, img_meta = service.search_image(
    service.kaggle_images_dir / "15970.jpg",
    top_k=10,
    exclude_self_id="15970",
    mode="fast"
)
print(f"Flow: {img_meta['flow']} ({img_meta['i2i_mode']}) | Latency: {img_meta['latency_ms']:.2f} ms")

# 3a. Luồng cir_catalog (Ảnh tham chiếu có trong catalog -> dùng GPU Embedding Cache)
cir_cat_res, cir_cat_meta = service.search_cir(
    service.kaggle_images_dir / "15970.jpg",
    modifier_text="màu xanh navy",
    top_k=10,
    ref_id="15970",
    force_upload=False
)
print(f"Flow: {cir_cat_meta['flow']} | Latency: {cir_cat_meta['latency_ms']:.2f} ms")

# 3b. Luồng cir_upload (Ảnh tải mới từ thiết bị -> chạy qua ViT Image Tower)
cir_up_res, cir_up_meta = service.search_cir(
    service.kaggle_images_dir / "15970.jpg",
    modifier_text="màu xanh navy",
    top_k=10,
    force_upload=True
)
print(f"Flow: {cir_up_meta['flow']} | Latency: {cir_up_meta['latency_ms']:.2f} ms")
```

---

## 3. Chạy Kiểm thử & Benchmark

```bash
# Chạy kiểm thử tích hợp dịch vụ ứng dụng (4 luồng SLA, Rule 7, Lavie, CIR Cache/Upload)
python tests/test_app_service.py

# Chạy kiểm thử toàn vẹn mask Key chain và căn hàng embedding
pytest tests/test_keychain_masking_and_alignment.py

# Đánh giá chất lượng Ảnh -> Ảnh (fast vs accurate trên 1.290 ảnh test)
python src/phase5a_i2i_quality.py

# Đo kiểm độ trễ E2E có kiểm soát (5 runs x 200 queries, warmup 20, Latin-square)
python src/benchmark_e2e_latency_v3.py --tag 5a_baseline
```

---

## 4. Chạy ứng dụng Web Demo (Streamlit - Pha 5)

Khởi chạy ứng dụng theo cổng cấu hình trong `config.yaml` (`app.port: 8501`):

```bash
streamlit run app/main.py --server.port 8501
```

### Các tính năng chính trên UI (`app/main.py`):
1. **Kiểm toán Luật bất biến 7 & Chính sách GT Giới tính Đóng băng:**
   - Tự động đối chiếu mã băm SHA-256 của checkpoint SigLIP (`best_model.pt`) giữa `config.yaml` và `index_v2/manifest.json`.
   - Hiển thị trạng thái đóng băng chính sách GT giới tính (`eval/gender_gt_policy.json`, chính thức: `HYBRID`).
2. **Tab 1 — Văn bản tiếng Việt → Ảnh (`flow: text_to_image`):**
   - Chuyển đổi trực tiếp giữa **Hệ thống (C5 Hybrid)** và **Model thuần (C0 SigLIP Dense)** (cùng C2/C4 ở thanh bên).
   - Hiển thị phân rã độ trễ 4 bước (`parse_ms`, `filter_ms`, `encode_ms`, `search_ms`) và đối chiếu nghiêm ngặt SLA (< 200 ms).
   - Tự động cảnh báo khi truy vấn thương hiệu ngoài catalog (như `Lavie`).
3. **Tab 2 — Ảnh → Ảnh (`flow: image_to_image`):**
   - Hỗ trợ chuyển đổi chế độ `serving.i2i_mode`: **`fast`** (1 ảnh gốc, mặc định theo `config.yaml`, đạt SLA p95 ~143 ms < 200 ms) và **`accurate`** (Batch-2 TTA Flip, p95 ~211–225 ms).
   - Tự động loại trừ chính sản phẩm truy vấn (`exclude_self_id`) khi chọn ảnh mẫu từ catalog.
   - Hiển thị phân rã độ trễ 5 bước (`decode_ms`, `preprocess_ms`, `vit_forward_ms`, `gallery_search_ms`, `rerank_ms`).
4. **Tab 3 — Ảnh + Text bổ trợ (`flow: cir_catalog` & `cir_upload`):**
   - Tách bạch rõ ràng luồng **`cir_catalog`** (sử dụng GPU FP16 Embedding Cache khi chọn ảnh catalog, p95 ~66.5 ms) và luồng **`cir_upload`** (chạy qua ViT Image Tower khi tải ảnh mới hoặc bật `force_upload=True`).
5. **Tab 4 — Đối soát SLA 4 Luồng, GT Giới tính & Báo cáo Hệ thống:**
   - Đọc trực tiếp số liệu thực tế từ `reports/phase4/phase4_latency_e2e_v3.json`, `reports/phase4/phase5a_i2i_quality.json`, `reports/phase4/unseen_testset_v2_eval.json` và `eval/gender_gt_policy.json`.
   - Báo cáo tách biệt từng luồng SLA, từng run (Run 1..5 kèm nhiệt độ và xung nhịp GPU), 3 chính sách GT giới tính (`Hybrid = 0.9180`, `Strict = 0.9010`, `Relaxed = 0.9180`), và bảng đối chiếu chất lượng `fast` vs `accurate`.
