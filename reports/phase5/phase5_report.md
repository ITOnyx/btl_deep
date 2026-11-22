# Báo cáo Pha 5 — Đánh giá Truy vấn, Hubness & Mean-Centering, và Độ trễ SLA

- **Trạng thái:** Hoàn thành đánh giá Pha 5 — **DỪNG CHỜ DUYỆT (`DUYỆT PHA 5`)**
- **Seed cố định:** `42`
- **Checkpoint SHA-256 (Hash Guard):** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` (không huấn luyện lại, không sửa trọng số — Luật 1)
- **Pytest:** `28/28 PASSED` (`pytest tests/ -v`)
- **Commit trước khi đóng gói báo cáo:** `2e057d6` (sẽ cập nhật commit mới nhất của Pha 5 sau khi commit)

---

## 1. Tổng hợp Đạt / Chưa đạt Chỉ tiêu (`config.yaml` & `AGENTS.md`)

Đánh giá chính thức trên **Official Test Gallery (`index/test_gallery/`, 1.290 sản phẩm)**, sử dụng bộ **100 truy vấn unseen (`eval/handwritten_queries.json`)** chạy **đúng 1 lần** (Luật C.5), chính sách giới tính chính thức **`hybrid`** (86/100 truy vấn có ≥ 1 sản phẩm Ground Truth trong `test_gallery`):

| Chỉ tiêu | Ngưỡng yêu cầu | Model thuần (cosine) | Hệ thống (`gamma=0.3`) | Kết quả |
|---|---:|---:|---:|---|
| **Text→Image Recall@10** | ≥ 0.85 | `0.7093` | **`0.8488`** (73/86) | **CHƯA ĐẠT** (thiếu 0.0012 ≈ 1 query) |
| **Text→Image mAP** | ≥ 0.75 | `0.3279` | **`0.5537`** | **CHƯA ĐẠT** |
| **Latency p95 (Text→Image)** | < 200 ms | — | **`29.86 ms`** | **ĐẠT** |
| **Latency p95 (Image→Image)** | < 200 ms | — | **`126.51 ms`** | **ĐẠT** |
| **Latency p95 (CIR Catalog)** | < 200 ms | — | **`36.98 ms`** | **ĐẠT** |
| **Latency p95 (CIR Upload)** | < 200 ms | — | **`165.49 ms`** | **ĐẠT** |

> **Báo cáo trung thực (Luật 7 & Mục C.10):** Không chỉnh ngưỡng, không chỉnh parser/lexicon sau khi xem kết quả tập test. Phân tích nguyên nhân gốc và đề xuất nằm ở Mục 6.

---

## 2. Kết quả Truy vấn Chi tiết (Tách Model thuần vs Hệ thống)

### 2.1. Official Test Gallery (`index/test_gallery/`, N = 1.290) — Trước Mean-Centering

| Cấu hình | Chính sách GT | n_queries (có GT) | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---|---:|---:|---:|---:|---:|
| **Model thuần** (`model_only`) | `hybrid` (chính thức) | 86 | `0.2907` | `0.5698` | `0.7093` | `0.3279` |
| **Model thuần** (`model_only`) | `strict` | 85 | `0.2706` | `0.5529` | `0.7059` | `0.3135` |
| **Model thuần** (`model_only`) | `relaxed` | 86 | `0.2907` | `0.5698` | `0.7093` | `0.3290` |
| **Hệ thống** (`system`, `gamma=0.3`) | `hybrid` (chính thức) | 86 | **`0.6047`** | **`0.8256`** | **`0.8488`** | **`0.5537`** |
| **Hệ thống** (`system`, `gamma=0.3`) | `strict` | 85 | `0.5765` | `0.8235` | `0.8471` | `0.5247` |
| **Hệ thống** (`system`, `gamma=0.3`) | `relaxed` | 86 | `0.6047` | `0.8256` | `0.8488` | `0.5543` |

- **Bootstrap 95% CI (1.000 lần lấy mẫu, `seed=42`) — Hệ thống vs Model thuần (`hybrid`):**
  - Δ Recall@1: **`+0.3140`** `[+0.2209, +0.4186]`
  - Δ Recall@5: **`+0.2558`** `[+0.1628, +0.3488]`
  - Δ Recall@10: **`+0.1395`** `[+0.0698, +0.2212]`
  - Δ mAP: **`+0.2257`** `[+0.1833, +0.2771]`
- **Image→Image (`test_gallery`, 200 ảnh lấy mẫu `seed=42`, 156 ảnh có ≥ 1 ảnh cùng `articleType + gender + baseColour`):**
  - Trước Mean-Centering: Recall@1 = `0.3974`, Recall@5 = `0.6410`, Recall@10 = `0.7436`, mAP = `0.3406`.

### 2.2. Reference Full Gallery (`index/reference_gallery/`, N = 38.646) — **Ước lượng lạc quan (tham khảo)**

> **Lưu ý (Quy tắc C.3):** Gallery này chứa cả ảnh tập train nên số liệu dưới đây **chỉ là ước lượng lạc quan (tham khảo)**, không dùng để đối chiếu chỉ tiêu chính thức.

| Cấu hình | Chính sách GT | n_queries (có GT) | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---|---:|---:|---:|---:|---:|
| **Model thuần** (`model_only`) | `hybrid` | 99 | `0.3131` | `0.5657` | `0.5859` | `0.1027` |
| **Hệ thống** (`system`, `gamma=0.3`) | `hybrid` | 99 | `0.6768` | `0.7071` | `0.7172` | `0.1514` |

*(mAP giảm mạnh trên 38.646 ảnh vì số lượng ảnh GT tăng lên rất nhiều trong khi `candidate_pool=100` chỉ xét top 100 ứng viên).*

---

## 3. Phân tích Chi tiết theo Slot mỏng, Có/Không dấu, Giới tính và Màu sắc (Official Test Gallery, `hybrid`)

### 3.1. Theo Có dấu vs Không dấu (`has_diacritics`)

| Nhóm | n | Model thuần R@1 / R@10 / mAP | Hệ thống R@1 / R@10 / mAP |
|---|---:|---|---|
| **Có dấu (`True`)** | 65 | `0.3692` / `0.8308` / `0.4166` | **`0.7077`** / **`0.9077`** / **`0.6602`** |
| **Không dấu (`False`)** | 21 | `0.0476` / `0.3333` / `0.0534` | `0.2857` / `0.6667` / `0.2240` |

### 3.2. Theo Slot mỏng (`by_thin_slot`)

| Slot mỏng | n | Model thuần R@1 / R@10 / mAP | Hệ thống R@1 / R@10 / mAP |
|---|---:|---|---|
| `Áo` | 19 | `0.4737` / `0.8947` / `0.3465` | `0.7368` / **`1.0000`** / `0.5523` |
| `Giày dép` | 17 | `0.2941` / `0.7059` / `0.3408` | `0.6471` / `0.9412` / `0.6221` |
| `Phụ kiện` | 10 | `0.3000` / `0.6000` / `0.3445` | `0.7000` / `0.9000` / `0.6652` |
| `Túi & Balo` | 14 | `0.2143` / `0.7143` / `0.2572` | `0.6429` / `0.7857` / `0.5186` |
| `Quần & Váy` | 16 | `0.3125` / `0.6250` / `0.3872` | `0.4375` / `0.7500` / `0.5354` |
| `Đồng hồ` | 10 | `0.0000` / `0.6000` / `0.2585` | `0.4000` / `0.6000` / `0.4065` |

### 3.3. Theo Giới tính (`by_gender`) — Hệ thống (`hybrid`)

| Giới tính | n | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| `Men` | 52 | `0.5962` | `0.8077` | `0.8077` | `0.5509` |
| `Women` | 28 | `0.6071` | `0.8571` | `0.9286` | `0.5378` |
| `Unisex` | 5 | `0.6000` | `0.8000` | `0.8000` | `0.6266` |
| `Boys` | 1 | `1.0000` | `1.0000` | `1.0000` | `0.7763` |

### 3.4. Theo Màu sắc (`by_colour` — các màu chính, n ≥ 3) — Hệ thống (`hybrid`)

| Màu sắc (`baseColour`) | n | Recall@1 | Recall@10 | mAP |
|---|---:|---:|---:|---:|
| `Black` | 24 | `0.6250` | `0.7917` | `0.5823` |
| `Unspecified` (không nêu màu) | 22 | `0.5000` | `0.7727` | `0.4894` |
| `White` | 11 | `0.3636` | `0.9091` | `0.4346` |
| `Blue` | 9 | `0.8889` | `0.8889` | `0.7385` |
| `Brown` | 7 | `0.5714` | `0.8571` | `0.4486` |
| `Silver` | 6 | `0.5000` | `0.8333` | `0.4371` |
| `Navy Blue` | 3 | `1.0000` | `1.0000` | `0.7313` |
| `Grey` | 3 | `0.6667` | `1.0000` | `0.6885` |

---

## 4. Phân tích Hubness & Thử nghiệm Mean-Centering (Quy tắc C.8)

So sánh trước và sau **Mean-Centering** trên `index/test_gallery/` (`1.290` ảnh, 100 truy vấn T2I và 156 truy vấn I2I):

| Cấu hình | Trạng thái | R@1 | R@10 | mAP | Max xuất hiện / 100 q | Skewness | Số hub (≥ 5 lần) |
|---|---|---:|---:|---:|---:|---:|---:|
| **T2I Model thuần** | Trước Mean-Centering | `0.2907` | **`0.7093`** | **`0.3279`** | 7 (7%) | `2.091` | 14 |
| **T2I Model thuần** | Sau Mean-Centering | **`0.3256`** | `0.6860` | `0.3229` | 7 (7%) | `1.854` | **11** |
| **T2I Hệ thống** | Trước Mean-Centering | **`0.6047`** | **`0.8488`** | **`0.5537`** | **6 (6%)** | `1.571` | 10 |
| **T2I Hệ thống** | Sau Mean-Centering | `0.5930` | `0.8372` | `0.5426` | 7 (7%) | **`1.518`** | **8** |
| **I2I (156 queries)** | Trước Mean-Centering | **`0.3974`** | `0.7436` | **`0.3406`** | 8 (5.1%) | `1.594` | 47 |
| **I2I (156 queries)** | Sau Mean-Centering | `0.3782` | **`0.7628`** | `0.3377` | 9 (5.8%) | **`1.568`** | **45** |

**Nhận xét Hubness:** Mức độ hubness ban đầu trên `test_gallery` khá nhẹ (ảnh xuất hiện nhiều nhất chỉ lặp 6–7 lần trên 100 truy vấn, chiếm 0.6–0.7% tổng slot Top-10). Mean-Centering giúp giảm nhẹ `skewness` (`2.091 → 1.854` với model thuần, `1.571 → 1.518` với hệ thống) và số lượng hub (`14 → 11`, `10 → 8`), tăng nhẹ Recall@1 cho model thuần (`0.2907 → 0.3256`) và Recall@10 cho I2I (`0.7436 → 0.7628`), nhưng làm giảm nhẹ Recall@10/mAP của hệ thống (`0.8488 → 0.8372`). Vì vậy cấu hình mặc định phục vụ giữ **không bật Mean-Centering** (`use_mean_centering=False`).

---

## 5. Đánh giá Caption Config A vs B vs C (Luật 5 & Quy tắc D.5)

Thực hiện Paired Text-to-Image Retrieval trên `1.290` sản phẩm test (`data/catalog/captions/shard_test_*.parquet`):
- **Config A (`caption_A`):** Chỉ `InternVL2-2B`.
- **Config B (`caption_B`):** `InternVL2-2B` + bổ trợ + Metadata tiếng Việt (`merge_v1`).
- **Config C (`caption_C`):** Tham chiếu mô hình phụ trợ độc lập.

| Cấu hình | n_items | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|---:|
| **Config A (`InternVL2-2B` only)** | 1.290 | `0.0008` | `0.0039` | `0.0101` | `0.0061` |
| **Config B (`InternVL2-2B` + Bổ trợ)** | 1.290 | **`0.0124`** | **`0.0504`** | **`0.0907`** | **`0.0421`** |
| **Config C (Tham chiếu phụ trợ)** | 1.290 | `0.0023` | `0.0202` | `0.0364` | `0.0170` |

- **Bootstrap 95% CI (1.000 lần, `seed=42`) — Config B vs Config A:**
  - Δ Recall@1: **`+0.0116`** `[+0.0054, +0.0178]`
  - Δ Recall@5: **`+0.0465`** `[+0.0349, +0.0581]`
  - Δ Recall@10: **`+0.0806`** `[+0.0659, +0.0969]`
  - Δ mAP: **`+0.0360`** `[+0.0291, +0.0433]`

> **Ghi chú bắt buộc (Luật D.5):** Checkpoint SigLIP (`best_model.pt`) **chưa từng được huấn luyện với caption của `InternVL2-2B`**, nên điểm paired retrieval dùng caption mới **không so trực tiếp** với số liệu lịch sử lúc train. Đồng thời, trong bài toán 1-1 instance retrieval trên 1.290 ảnh thời trang tương đồng cao, mô tả ngắn/bị cắt ở `caption_A` khó phân biệt từng sản phẩm riêng lẻ; việc hợp nhất metadata tiếng Việt ở đầu câu (`Config B`) giúp tăng Recall@10 gấp ~9 lần (`1.01% → 9.07%`) với ý nghĩa thống kê rõ rệt.

---

## 6. Benchmark Độ trễ SLA (Quy tắc C.9)

Đo riêng 4 luồng trên **NVIDIA GeForce RTX 2050** (`cuda`, `torch.float16`, `warmup=20`, **5 runs × 200 truy vấn = 1.000 truy vấn/luồng**, `torch.cuda.synchronize()` quanh forward/search). Nhiệt độ GPU: **`75°C → 86°C`**, VRAM sử dụng: `1924 MB → 2009 MB / 4096 MB`.

| Luồng | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Ngưỡng SLA (p95) | Kết quả |
|---|---:|---:|---:|---:|---:|---|
| `text_to_image` | `23.20` | **`29.86`** | `37.89` | `24.17` | < 200 ms | **ĐẠT** |
| `image_to_image` | `79.95` | **`126.51`** | `132.95` | `87.05` | < 200 ms | **ĐẠT** |
| `cir_catalog` | `30.25` | **`36.98`** | `39.84` | `30.65` | < 200 ms | **ĐẠT** |
| `cir_upload` | `110.90` | **`165.49`** | `199.90` | `115.57` | < 200 ms | **ĐẠT** |

---

## 7. Phân tích Nguyên nhân Gốc (Khi chưa đạt Recall@10 ≥ 0.85 & mAP ≥ 0.75) & Đề xuất

1. **Truy vấn không dấu (`has_diacritics=False`, 21/86 queries):** Model thuần chỉ đạt `Recall@10 = 0.3333` (`mAP = 0.0534`), kéo toàn bộ trung bình xuống dưới `0.85` (trong khi 65 truy vấn có dấu đạt `Recall@10 = 0.9077`, vượt xa `0.85`). Do checkpoint SigLIP không được train/augment với tiếng Việt không dấu, vector cosine thô cho câu không dấu bị lệch, khiến sản phẩm đúng không lọt vào `candidate_pool=100` để nhận soft-bonus.
2. **Giới hạn `candidate_pool=100` lên chỉ số mAP:** Với các truy vấn phổ biến có 20–50 sản phẩm đúng trong gallery, những sản phẩm đúng nằm ngoài top-100 cosine thô không được cộng điểm thuộc tính (`gamma=0.3`), làm giảm mạnh `mAP` (`0.5537` vs `0.75`).
3. **Độ trễ & Luật 1:** Việc giữ nguyên checkpoint cũ (Luật 1) giới hạn trần phân biệt chi tiết (đặc biệt nhóm `Đồng hồ` `R@10=0.60`, `Quần & Váy` `R@10=0.75`).
4. **Đề xuất (giữ nguyên Luật 1):**
   - Trong Pha 6 (Streamlit) hoặc tầng suy luận tiếp theo (nếu được phép trước khi đánh giá bộ mới), có thể bổ sung bộ khôi phục dấu tiếng Việt nhẹ cho câu không dấu và tăng `candidate_pool` từ `100` lên `300–500` trên gallery test.
   - Để vượt `mAP ≥ 0.75` ổn định trên mọi slot, về dài hạn cần fine-tune thêm với cặp ảnh–caption tiếng Việt không dấu/có dấu (hiện bị khóa theo Luật 1).

---

## 8. Tệp đã tạo / thay đổi & Kiểm thử

- `src/serve/__init__.py` — export `VietnameseQueryParser`, `ECommerceSearchEngine`, `remove_diacritics`, `compute_soft_attribute_match`.
- `src/eval/metrics.py` — hàm tính `Recall@K`, `AP/mAP`, `bootstrap_ci_diff`, `compute_hubness_stats`, `is_item_relevant` (`hybrid`/`strict`/`relaxed`).
- `src/eval/eval_captions.py` — benchmark Config A vs B vs C + Bootstrap 95% CI → `reports/phase5/caption_eval_ab.json`.
- `src/eval/eval_retrieval.py` — benchmark Official Test Gallery, Reference Full Gallery, Mean-Centering, Hubness, I2I → `reports/phase5/retrieval_eval_results.json`.
- `src/eval/eval_latency.py` — benchmark độ trễ 4 luồng + telemetry GPU → `reports/phase5/latency_sla_results.json`.
- `tests/test_phase5_eval.py` — 7 unit tests mới cho parser, policy, metrics, hubness, Hash Guard & mean-centering.
- **Kết quả kiểm thử:** `pytest tests/ -v` → **`28 passed, 0 failed`**.

---

## 9. Bảng Quyết định của Người dùng

| Lựa chọn | Hành động tiếp theo |
|---|---|
| **`DUYỆT PHA 5`** | Chấp nhận số liệu đánh giá Pha 5 (ghi nhận trung thực Recall@10 = `0.8488`, mAP = `0.5537`, SLA Latency đạt 4/4 luồng) và chuyển sang **Pha 6 (Ứng dụng Streamlit 4 luồng)**. |
| **Yêu cầu điều chỉnh tầng suy luận (chuyển bộ test thành DEV SET theo Quy tắc C.5)** | Nếu muốn thử mở rộng `candidate_pool` hoặc thêm bộ khôi phục dấu, bộ `handwritten_queries.json` sẽ bị gắn nhãn **DEV SET**. |
