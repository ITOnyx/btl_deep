# BÁO CÁO TOÀN DIỆN TẤT CẢ NHỮNG GÌ ĐANG CÓ TRONG DỰ ÁN (`Hoc_sau_ver4`)
**Thư mục gốc:** `D:/Kì 7/Hoc_sau_ver4`  
**Nhánh Git hiện tại:** `cir-cleanup` (commit mới nhất: `ce8a4e9`)  
**Ngày kiểm kê:** 2026-11-11  
**Tổng quy mô lưu trữ trên đĩa (trừ `.git/`):** **~87.20 GiB** (`89,295 MiB`) trên **424,033 tập tin** *(đã bao gồm 2 báo cáo tổng hợp và các file cache `.pyc`; đơn vị dung lượng thống nhất dùng hệ nhị phân `KiB`/`MiB`/`GiB`, 1 GiB = 1,024 MiB)*

---

## MỤC LỤC
1. [Bản đồ Thư mục Cấp 1 & Phân loại Trạng thái Sử dụng](#1-bản-đồ-thư-mục-cấp-1--phân-loại-trạng-thái-sử-dụng)
2. [Cấu hình, Quy tắc Dự án & Tài liệu Hướng dẫn](#2-cấu-hình-quy-tắc-dự-án--tài-liệu-hướng-dẫn)
3. [Kho Dữ liệu Gốc, Catalog, Quarantine & Gói Build Hợp nhất (`data/`, `_build/`, `scripts/`)](#3-kho-dữ-liệu-gốc-catalog-quarantine--gói-build-hợp-nhất-data-_build-scripts)
4. [Mô hình, Checkpoints & Notebooks Huấn luyện (`data/checkpoints_*`, `notebooks/`)](#4-mô-hình-checkpoints--notebooks-huấn-luyện-datacheckpoints_-notebooks)
5. [Chỉ mục Vector & Embeddings (`index/`, `index_v2/`)](#5-chỉ-mục-vector--embeddings-index-index_v2)
6. [Mã nguồn Lõi, Chẩn đoán & Thử nghiệm (`src/`)](#6-mã-nguồn-lõi-chẩn-đoán--thử-nghiệm-src)
7. [Ứng dụng Web Streamlit & Dịch vụ Phục vụ (`app/`, `_backup_ui/`, `.streamlit/`)](#7-ứng-dụng-web-streamlit--dịch-vụ-phục-vụ-app-_backup_ui-streamlit)
8. [Bộ Dữ liệu Kiểm thử & Hệ thống Test Tự động (`eval/`, `tests/`)](#8-bộ-dữ-liệu-kiểm-thử--hệ-thống-test-tự-động-eval-tests)
9. [Hệ thống Báo cáo, Contact Sheets & Manifest (`reports/`, `docs/`)](#9-hệ-thống-báo-cáo-contact-sheets--manifest-reports-docs)
10. [Tổng hợp Tình trạng Hoạt động: Cái gì đang chạy vs Cái gì là Lịch sử / Dự phòng](#10-tổng-hợp-tình-trạng-hoạt-động-cái-gì-đang-chạy-vs-cái-gì-là-lịch-sử--dự-phòng)

---

## 1. Bản đồ Thư mục Cấp 1 & Phân loại Trạng thái Sử dụng

| Thành phần (`Hoc_sau_ver4/`) | Số file *(đo thực tế)* | Dung lượng *(MiB / KiB)* | Vai trò trong dự án | Trạng thái |
|---|---:|---:|---|---|
| **`app/`** | 4 files *(2 `.py` + 2 `.pyc`)* | 0.19 MiB | Ứng dụng Web Streamlit (`main.py`) & Dịch vụ (`engine.py`) | 🟢 **Đang chạy (Serving)** |
| **`src/`** | 146 files *(+24 `.pyc` = 170)* | 1.73 MiB | Lõi suy luận, parser, từ điển, BM25, benchmark & script phân tích | 🟢 **Đang chạy / Hỗ trợ** |
| **`index_v2/`** | 47 files *(gồm 40 file shard)* | 702.26 MiB | Embedding chuẩn BILINEAR (38,646×1,152), FAISS HNSW v2, manifest | 🟢 **Đang chạy (Serving)** |
| **`data/catalog/`** | 2 files | 1.81 MiB | Bảng danh mục sạch `catalog.parquet` (40,891 sp) & `product_brands.json` | 🟢 **Đang chạy (Serving)** |
| **`data/checkpoints_siglip-vit/`** | 21 files | 7,021.40 MiB | Checkpoint chính `best_model.pt` (3.31 GiB), stats `mu_shared`, test gốc | 🟢 **Đang chạy (Chỉ đọc)** |
| **`data/fashion-dataset/`** | 174,262 files | 28,985.00 MiB | Ảnh sản phẩm phục vụ hiển thị UI & nguồn `styles.csv` Kaggle | 🟢 **Đang phục vụ ảnh** |
| **`data/_quarantine/`** | 3,528 ảnh | 1,022.87 MiB | Ảnh bị cách ly (mờ, trùng, banner, phân giải thấp) — bảo toàn theo Luật 2 | 🟡 **Lưu trữ bảo toàn** |
| **`data/Shopping100k/`** | 100,599 files | 6,789.57 MiB | Bộ ảnh thời trang Shopping100k (phục vụ train / ghép `_build`) | ⚪ **Nguồn phụ / Train** |
| **`data/checkpoints_siglip-base/`** | 7 files | 3,022.83 MiB | Checkpoint SigLIP-Base cũ (29 epochs, 1.41 GiB/file) | 🔴 **Cũ (Không dùng)** |
| **`index/`** | 45 files *(gồm 40 file shard)* | 533.71 MiB | Index v1 (bị lệch nội suy BICUBIC so với train) | 🔴 **Deprecated (Lịch sử)** |
| **`eval/`** | 113 files *(+2 `.pyc` = 115)* | 12.31 MiB | Toàn bộ tập truy vấn chuẩn, testset niêm phong, text2img & CIR suites | 🟢 **Kiểm thử & Benchmark** |
| **`tests/`** | 8 `.py` *(+14 `.pyc` = 22)* | 0.11 MiB | Bộ kiểm thử tự động tích hợp, Luật 7, Keychain mask, Text2Img | 🟢 **Kiểm thử tự động** |
| **`reports/`** | 146 files *(gồm 2 báo cáo mới)* | 43.52 MiB | Toàn bộ báo cáo Pha 1–5, 79 contact sheets, `run_manifest.json`, log | 🟢 **Hồ sơ kiểm toán** |
| **`notebooks/`** | 2 `.ipynb` | 0.29 MiB | Notebook huấn luyện Colab & tham khảo ban đầu *(Chỉ đọc theo Luật 3)* | 🟡 **Gốc tham khảo (Khóa)** |
| **`docs/`** | 3 `.md` | 0.01 MiB | Tài liệu mô tả đề tài ban đầu (`context.md`, `PROJECT.md`, `TEST_INFRA.md`) | 🟡 **Tài liệu gốc (Khóa)** |
| **`scripts/`** | 1 `.py` *(+1 `.pyc` = 2)* | 0.21 MiB | Script `build_merged_dataset.py` tạo gói dataset hợp nhất | ⚪ **Công cụ build dataset** |
| **`_build/`** | 145,046 files | 41,157.26 MiB | Đầu ra build dataset hợp nhất 139,139 ảnh + file `.zip` 19.41 GiB | ⚪ **Artifact đóng gói train** |
| **`_backup_ui/`** | 3 files | 0.10 MiB | Sao lưu `main.py` & `engine.py` trước các đợt nâng cấp UI/CIR | ⚪ **Sao lưu UI** |
| **`.agents/`, `.streamlit/`, gốc** | 5 files *(1 + 1 + 3)* | 17.72 KiB | `AGENTS.md`, `.streamlit/config.toml`, `.gitignore`, `config.yaml`, `README_run.md` | 🟢 **Cấu hình cốt lõi** |
| **`.pytest_cache/`** | 4 files | 1.13 KiB | Bộ nhớ đệm chạy kiểm thử `pytest` | ⚪ **Cache kiểm thử** |

---

## 2. Cấu hình, Quy tắc Dự án & Tài liệu Hướng dẫn

- **[`.agents/AGENTS.md`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/.agents/AGENTS.md)** *(4.38 KiB)*: Bộ luật bất biến 9 điều (cấm train lại/sửa checkpoint, cấm xóa file vĩnh viễn, khóa `notebooks/` & `docs/`, ép khớp tiền xử lý train, cấm bịa số liệu, Luật 7 hash guard, hỗ trợ Windows UTF-8), quy tắc tách biệt $G_{test}$ vs $G_{full}$, tách Model thuần vs Hệ thống, và quy định SLA 4 luồng + chính sách GT giới tính.
- **[`config.yaml`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/config.yaml)** *(5.00 KiB)*: Nguồn chân lý tham số duy nhất:
  - `paths`: Trỏ tới `data/fashion-dataset`, `data/checkpoints_siglip-vit`, `data/catalog/catalog.parquet`, `src/brands_v2.json` (kèm SHA-256 `c5283096...`).
  - `checkpoint`: `pt_state_dict`, `google/siglip-so400m-patch14-224`, hash `e2f82fee...`, `embedding_dim: 1152`, `max_text_length: 128`, `text_padding: "max_length"`.
  - `image_filter`: `min_short_side: 224`, `blur_laplacian_var_threshold: 12.0`, `phash_hamming_threshold: 1`, `dedup_match_attributes: ["articleType", "gender", "baseColour"]`, `dedup_transitive: false`.
  - `faiss`: `IndexHNSWFlat` (`M=32`, `ef_construction=200`, `ef_search=128`, `inner_product`).
  - `eval`: `seed: 42`, `quality_device: "cpu"` (`float32`), `latency_device: "cuda"` (`float16`), `gender_gt.official: "hybrid"` (`eval/gender_gt_policy.json`).
  - `latency`: `sla_metric: "p95"`, `min_runs: 5`, `warmup_queries: 20`, ngưỡng `200 ms` tách riêng cho `text_to_image`, `image_to_image`, `cir_catalog`, `cir_upload`.
  - `serving`: `i2i_mode: "fast"`, `use_flip_aug: false`, `gpu_preprocess: false` (`min_cosine: 0.999`).
  - `scope`: `index_scope: "fashion_only"`, loại trừ `["Personal Care", "Home", "Sporting Goods"]` và `["Ipad", "Free Gifts", "Fragrance Gift Set", "Key chain", "Water Bottle", "Perfume and Body Mist"]`, ngoại lệ giữ `["Wristbands"]`.
  - `inference`: `phase: "3g"`, `gamma: 0.3`, `alpha: 1.0`, `rrf_k: 60`, `min_candidates: 20`, `enable_parser: true`, `enable_bm25: true`, `enable_text_fusion: false`, `use_usage: false`, `use_season: false`, `filter_mode: "relaxed"`, `scoring_mode: "mean_centered"`.
- **[`README_run.md`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/README_run.md)** *(7.55 KiB)*: Hướng dẫn cấu trúc thư mục, ví dụ code gọi `SearchService` cho cả 4 luồng, lệnh chạy test/benchmark và lệnh khởi chạy Streamlit (`streamlit run app/main.py --server.port 8501`).
- **[`.streamlit/config.toml`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/.streamlit/config.toml)** *(224 B)*: Cấu hình Dark Theme (`#0F172A` nền chính, `#1E293B` thẻ, `#3B82F6` primary, `toolbarMode = "minimal"`, `showErrorDetails = false`).
- **[`.gitignore`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/.gitignore)** *(584 B)*: Chặn commit dữ liệu nặng (`_build/`, `scripts/`, `data/Shopping100k/`, `data/fashion-dataset/`, `data/_quarantine*`, `data/checkpoints*`, `*.npy`, `*.npz`, `*.bin`, `*.parquet`, `reports/contact_sheets/*.png`, `eval/**/*.png`, `_backup_ui/`, `index/*`, `index_v2/*`), chỉ giữ lại `index_v2/manifest.json` và mã nguồn/báo cáo.

---

## 3. Kho Dữ liệu Gốc, Catalog, Quarantine & Gói Build Hợp nhất (`data/`, `_build/`, `scripts/`)

### 3.1. Bộ Dữ liệu Thô & Phục vụ Ảnh (`data/`)
- **`data/fashion-dataset/`** *(174,262 files, 28,985.00 MiB / ~28.31 GiB)*:
  - `styles.csv` (4.13 MiB, 44,446 dòng gốc), `images.csv` (4.99 MiB).
  - `images/`: **40,913** ảnh `.jpg` còn lại trên đĩa sau khi chuyển ảnh lỗi sang quarantine.
  - `styles/`: **44,446** file JSON chi tiết từng sản phẩm (`10000.json`, ...).
  - `Attributes/`: 12 file `.txt` thuộc tính (`category.txt`, `collar.txt`, `color.txt`, `fabric.txt`, `fastening.txt`, `fit.txt`, ...).
  - `fashion-dataset/`: Thư mục lồng chứa bản giải nén đầy đủ ban đầu (`images`, `images.csv`, `styles`, `styles.csv`), chiếm phần lớn dung lượng thư mục này.
- **`data/Shopping100k/`** *(100,599 files, 6,789.57 MiB / ~6.63 GiB)*:
  - `Images/Female/` (**64,651** ảnh) + `Images/Male/` (**35,935** ảnh) = **100,586** ảnh thời trang (16 danh mục × 12 thuộc tính), kèm `Attributes/` và `Readme`. Không dùng cho app serving hiện tại (chỉ phục vụ build dataset huấn luyện `_build`).
- **`data/_quarantine/`** *(3,528 ảnh, 1,022.87 MiB / ~1.00 GiB)*:
  - `blurry/` (**1,286** ảnh mờ), `duplicate/` (**1,188** ảnh trùng sau khi đã khôi phục 13,226 ảnh ở Pha 1b), `banner/` (**935** ảnh khung viền/quảng cáo), `low_res/` (**119** ảnh cạnh ngắn < 224 px).

### 3.2. Danh mục Chuẩn hóa (`data/catalog/`)
- **`data/catalog/catalog.parquet`** *(1.38 MiB, 40,891 dòng × 14 cột)*:
  - **`in_scope == True`: 38,644 sản phẩm** — chia thành **`split == "test"` (1,290 sp)** và **`split == "unknown"` (37,354 sp)**.
  - **`in_scope == False`: 2,247 sản phẩm** (`Personal Care`: 2,136, `Free Items`: 87, `Sporting Goods`: 14, `Accessories`: 9, `Home`: 1 — gồm Ipad, Free Gifts, Fragrance Gift Set, Water Bottle, Perfume and Body Mist, và 2 Key chain `43230`, `42066`).
- **`data/catalog/product_brands.json`** *(0.43 MiB)*: Bảng ánh xạ thương hiệu tiền trích xuất theo catalog.

### 3.3. Bộ Xây dựng Dataset Huấn luyện Hợp nhất (`scripts/` & `_build/`)
- **[`scripts/build_merged_dataset.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/scripts/build_merged_dataset.py)** *(97.44 KiB, 2,250 dòng)*: Pipeline 5 pha gộp `fashion-dataset` + `Shopping100k`, lọc chất lượng, gom cụm biến thể `label_group`, chia 80/10/10 chống rò rỉ.
- **`_build/`** *(145,046 files, 41,157.26 MiB / ~40.19 GiB)*:
  - `Hoc_sau_ver4/data/merged_dataset/`: **139,139 ảnh** hợp lệ (`fashion` 40,953 + `shopping100k` 98,186), chia `train` (111,311), `val` (13,914), `test` (13,914, gồm 2,971 `legacy_test`), kèm `annotations/` (`all/train/val/test.parquet`, `*_clean.jsonl`, `splits.json`, `label_maps.json`) và `reports/` (`stats.json`, `rejected.csv`, `duplicates.csv`).
  - `_rejected/` (**5,888** ảnh bị loại từ 2 nguồn), `_cache/` (4 files), `Hoc_sau_ver4_merged_dataset.zip` (**19.41 GiB** / `20,845,341,089 bytes`, 139,152 files, SHA-256 `dc92b311...`), `zip_manifest.json`.

---

## 4. Mô hình, Checkpoints & Notebooks Huấn luyện (`data/checkpoints_*`, `notebooks/`)

### 4.1. Checkpoint Đang Phục vụ (`data/checkpoints_siglip-vit/` — 21 files, 7,021.40 MiB / ~6.86 GiB)
- **`best_model.pt`** *(3,549,699,293 bytes / 3,385.26 MiB ≈ 3.31 GiB, SHA-256 `e2f82fee...`)* & **`last_model.pt`** *(3,385.26 MiB ≈ 3.31 GiB)*:
  - Trọng số PyTorch `state_dict` (908 tensors) cho kiến trúc `google/siglip-so400m-patch14-224` (1152-d, 27 transformer layers, `max_length=128`) + `UpgradedAttentionCIRCombiner` (8-head Cross-Attention, `projection_dim=1024`, dynamic gate).
- **`vit_zero_train_stats.npz`** *(8.19 MiB)*: Chứa các thống kê huấn luyện dùng cho Mean-Centering và CSLS (`mu_shared`, `mu_img`, `mu_txt`, `cone_restore_gamma=0.75`, `csls_beta=0.28`, `csls_k=7`, `val_query_bank` 2,000 mẫu, `gallery_hub_scores`).
- **`dense_captions_catalog.json`** *(0.93 MiB, **4,372** sản phẩm)*: Tập test gốc kèm dense caption từ `Vintern-1B-v3.5` (trong đó 1,372 sp thuộc catalog Myntra Kaggle và 1,290 sp `in_scope`).
- Các file trung gian lúc chạy notebook trên Colab (không dùng ở serving): `dense_caption_embeddings_vit.npy`, `embeddings_{merged,new,pass_only}.npy`, `faiss_hnsw_index*.bin`, `product_metadata*.json`, `prototypes_so400m.pt`.

### 4.2. Checkpoint Cũ / Tham khảo (`data/checkpoints_siglip-base/` — 7 files, 3,022.83 MiB / ~2.95 GiB)
- Chứa `best_model.pt` & `last_model.pt` (`1,447.23 MiB` / `1.41 GiB` mỗi file — mô hình SigLIP-Base cũ), `faiss_hnsw_index.bin` (`104.77 MiB`), `product_metadata.json` (`23.22 MiB`), và `train_history.json` (`5,103 B`, 29 epochs: `val_r1` tới 38.11%, `val_r10` tới 85.96%, `val_map` tới 0.5207). **Không được nạp trong hệ thống hiện hành.**

### 4.3. Notebooks (`notebooks/` — 2 files, Chỉ đọc theo Luật 3)
- **[`siglip_vietnamese_ecommerce_colab_ver2.ipynb`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb)** *(57 cells, 252.10 KiB)*: Notebook huấn luyện chính thức trên Google Colab A100 (nạp checkpoint Epoch 25, train tới Epoch 100 + thử nghiệm 1 epoch, định dạng chuỗi `[{category}] {title} ({attrs})`, `Resize((224,224), BILINEAR)`).
- **[`shopee-v03.ipynb`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/notebooks/shopee-v03.ipynb)** *(28 cells, 40.20 KiB)*: Notebook tham khảo Kaggle Shopee (`DistilBertModel`, `timm`, `albumentations`).

---

## 5. Chỉ mục Vector & Embeddings (`index/`, `index_v2/`)

| Thuộc tính | `index/` (v1 — Deprecated) | `index_v2/` (v2 — **Đang phục vụ**) |
|---|---|---|
| **Quy trình tiền xử lý ảnh** | `SiglipProcessor` mặc định (BICUBIC) — bị lệch | `Resize((224, 224), BILINEAR)` + `SiglipProcessor` (khớp train) |
| **Số vector lưu trong `.npy`** | 38,646 × 1,152 (`float32`, L2-norm) | 38,646 × 1,152 (`float32`, L2-norm) *(Engine lọc mask còn 38,644 `in_scope`)* |
| **Embeddings Ảnh** | `image_embeddings.npy` (169.83 MiB) | `image_embeddings_v2.npy` (169.83 MiB, SHA-256 `ec02f2a6...`) |
| **Embeddings Text Catalog** | *Không có* | `text_embeddings_v2.npy` (169.83 MiB — tắt lúc serving vì $\alpha^*=1.0$) |
| **FAISS HNSW Index** | `faiss_hnsw_index.bin` (179.86 MiB) | `faiss_hnsw_index_v2.bin` (179.86 MiB, M=32, efC=200, efS=128) |
| **Bản đồ ID & Split** | `id_map.json`, `splits.json` | `id_map_v2.json`, `split_v2.json` (`test: 1290, unknown: 37356`) |
| **Shards** | `shards/shard_000..019.{npy,json}` (40 files) | `shards/shard_000..019.{npy,json}` (40 files) |
| **Manifest Luật 7** | `manifest.json` (catalog hash cũ `bfaf6625...`) | `manifest.json` (đồng bộ checkpoint hash `e2f82fee...`, note loại 2 Keychain) |

*Lưu ý phục vụ:* Trong đường chạy GPU (`app/engine.py` / `VisualSearchEngine`), hệ thống tải ma trận ảnh đã căn tâm lên GPU FP16 (`t_emb_img_c`, 38,644×1,152) và thực hiện nhân ma trận trực tiếp (`torch.matmul`, ~1–6 ms) thay vì gọi FAISS CPU để hỗ trợ mask thuộc tính chính xác từng dòng và áp dụng Soft Bonus.

---

## 6. Mã nguồn Lõi, Chẩn đoán & Thử nghiệm (`src/`)

Thư mục `src/` gồm **146 file** mã nguồn (cộng 24 `.pyc` trong `__pycache__` = 170 files), phân làm 3 tầng rõ rệt:

### 6.1. Tầng Lõi Suy luận, Từ điển & Benchmark (`src/` — 26 files)

| Nhóm | Tên file | Thành phần chính & Chức năng |
|---|---|---|
| **Model Wrapper** | [`model_siglip.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/model_siglip.py) | `SiglipVisionTextInference`, `load_siglip_inference` (nạp tháp Vision + Text, pos_emb 128) |
| | [`model_siglip_cir.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/model_siglip_cir.py) | `UpgradedAttentionCIRCombiner`, `MultimodalSigLIPWithCIR`, `load_trained_siglip_cir` |
| **Scoring** | [`scoring.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/scoring.py) | `Scorer` — Raw Cosine, Mean-Centering ($\mu_{shared}$), CSLS ($\beta=0.28, k=7$) |
| **NLP & Từ điển** | [`vi_vocab.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/vi_vocab.py) | `GENDER_MAP`, `COLOUR_MAP`, `ARTICLE_TYPE_MAP` (143 loại), `USAGE_MAP`, `remove_diacritics`, `generate_vietnamese_title` |
| | [`vi_restore.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/vi_restore.py) | `DiacriticRestorer`, khôi phục dấu tiếng Việt theo quy tắc $c'$ |
| | [`query_parser.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/query_parser.py) | `QueryParser` (V1, longest-match-first) |
| | [`query_parser_v2.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/query_parser_v2.py) | `QueryParserV2` (sửa bug `"đồ"→Red`, dùng `brands_v2.json`) |
| | [`query_builder.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/query_builder.py) | `ARTICLE_TO_CATEGORY`, Ensemble Cách A/B/C |
| | [`cir_normalizer.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/cir_normalizer.py) | `sanitize_parsed_modifier`, `classify_cir_intent` (State A/B), `normalize_cir_modifier` (mẫu P2), `recover_color_from_typos` |
| | `lexicon_vi.json`, `brands.json`, `brands_v2.json`, `vi_to_en_terms.json` | 60 loại / 47 màu / 5 giới / 7 usage / 4 season; 234 canonical brands (+22 alias trong V2); 29 mục song ngữ Việt–Anh |
| **Index & Search** | [`attr_index.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/attr_index.py) | `AttributeIndex` (mảng NumPy cho 38,644 sp, lọc cứng, nới lỏng L0..L6, Soft Bonus) |
| | [`lexical.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/lexical.py) | `BM25Index` (ma trận thưa Scipy CSR, mở rộng song ngữ), `rrf_merge` ($k=60$) |
| | [`search_pipeline.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/search_pipeline.py) | `VisualSearchEngine` — Pipeline thống nhất phục vụ Text, Visual, CIR |
| | [`search.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/src/search.py) | Wrapper FAISS HNSW Pha 2 (`verify_manifest`, `load_index`) — dùng trong `test_manifest_verification.py` |
| **Eval & Benchmark** | `benchmark_e2e_latency{,_v2,_v3}.py`, `benchmark_latency_optimizations.py`, `phase5a_i2i_quality.py`, `evaluate_phase3e_test.py`, `evaluate_cir_20queries.py`, `experiment_trunc64_pad128_eval.py`, `run_phase4_smoke_test.py` | Các script đo kiểm SLA có kiểm soát (`nvidia-smi`), đối chiếu chất lượng I2I `fast`/`accurate`, thực nghiệm padding & smoke test |

### 6.2. Tầng Chẩn đoán & Thử nghiệm (`src/diag/` — 94 scripts, `src/experiments/` — 26 scripts)
- **`src/diag/` (94 scripts):** Các script đơn mục đích phục vụ Pha 1/1b (`phase1_quality.py`, `phase1_apply_filters.py`, `phase1b_diagnose.py`, `phase1b_apply_fix.py`), khảo sát Pha 2a (`phase2a_survey.py`), chuỗi chẩn đoán D0–D8 (`diagnose_d0_d5.py`, `d3_text_equivalence.py`, `diagnose_d4.py`, `diagnose_d8.py`), kiểm toán Pha 3g/4 (`audit_lee.py`, `brand_audit_and_alias.py`, `compare_colour_gt.py`, `run_phase4_all_verifications.py`, `verify_catalog_t6.py`, `generate_errata_report.py`, `update_manifest_final.py`, v.v.).
- **`src/experiments/` (26 scripts):** Các script xây index (`embed_catalog.py`, `embed_catalog_v2.py`), dựng tập eval & từ điển (`build_eval_set.py`, `build_lexicon.py`, `build_q6_dataset.py`, `extract_brands.py`, `create_brands_v2.py`), dò tham số validation (`tune_phase3e_val.py`, `tune_phase3f_q6.py`), chạy các đợt eval lịch sử (`evaluate_phase3.py`, `evaluate_phase3c_v2.py`, `evaluate_phase3d_addendum.py`, `evaluate_phase3f.py`, `evaluate_phase3g_brands_v2.py`, `experiment_padding_truncation.py`), và tạo/giải mã Blind A/B (`fix_contact_sheets_rev3.py`, `recreate_blind_ab_rev3.py`, `analyze_blind_ab_results.py`, `update_blind_results_rev4.py`).

---

## 7. Ứng dụng Web Streamlit & Dịch vụ Phục vụ (`app/`, `_backup_ui/`, `.streamlit/`)

- **[`app/engine.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/app/engine.py)** *(39.01 KiB, 832 dòng)*:
  - `verify_rule_7_checkpoint_hash()`: Kiểm tra khớp SHA-256 checkpoint trước khi khởi tạo.
  - `SearchService`: Singleton wrapper tự warmup khi khởi động (`_warmup`), quản lý ngưỡng SLA riêng 4 luồng (`text_to_image`, `image_to_image`, `cir_catalog`, `cir_upload`), cung cấp:
    - `search_text(query, config_mode='c5'|'c2'|'c4'|'c0', top_k, gamma)` + breakdown 4 khâu (`parse_ms`, `filter_ms`, `encode_ms`, `search_ms`).
    - `search_image(image, top_k, exclude_self_id, mode='fast'|'accurate', use_knn_voting)` + breakdown 5 khâu (`decode_ms`, `preprocess_ms`, `vit_forward_ms`, `gallery_search_ms`, `rerank_ms`).
    - `search_cir(...)`: Hỗ trợ `cir_catalog` (GPU cache) và `cir_upload` (ViT forward), Dynamic Category Anchor 2 trạng thái (State A giữ loại / State B chuyển loại), Upload k-NN inference fallback (`_infer_upload_article_type`, `conf >= 0.62`, `top1_sim >= 0.65`), family supplement từ `families.json`, và nội suy vector `interpolate_cir_vector` theo $\lambda \in [0, 1]$.
- **[`app/main.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/app/main.py)** *(47.61 KiB, 1,077 dòng)*:
  - Giao diện Dark Mode WCAG AA với Sidebar (toggle *Chế độ kỹ thuật*, kiểm toán Luật 7, định mức SLA 4 luồng) và **4 Tab tương tác**:
    1. *Text tiếng Việt → Ảnh*: Chọn Hệ thống đầy đủ / Model thuần, expander cấu hình (C5/C2/C4, $\gamma$), 7 nút gợi ý mẫu, cảnh báo Lavie ngoài catalog, thẻ sản phẩm 100% tiếng Việt (`format_vietnamese_title`, `render_card_html`), expander hiệu năng.
    2. *Ảnh → Ảnh (Visual Search)*: Upload ảnh hoặc chọn ảnh mẫu (tự `exclude_self_id`), chọn `fast` (mặc định) vs `accurate`, checkbox k-NN voting.
    3. *Ảnh + Text bổ trợ (CIR)*: Upload ảnh ưu tiên (ẩn selectbox catalog, cảnh báo ảnh nhỏ/hỏng) hoặc chọn ảnh catalog (có toggle bỏ qua cache `force_upload`), thanh trượt $\lambda$ (`Mức độ biến đổi theo yêu cầu chữ`), checkbox `Khóa giữ nguyên loại trang phục gốc`, hiển thị trạng thái neo bằng tiếng Việt.
    4. *Đối soát SLA, GT & Báo cáo*: Đọc động dữ liệu thực từ `phase4_latency_e2e_v3.json` và `phase5a_i2i_quality.json`, hiển thị Q1b, GT Giới tính và bảng SLA 4 luồng.
- **`_backup_ui/`** *(3 files, 0.10 MiB)*: Lưu 2 snapshot cũ (`20261110_194135/main.py`, `20261110_222602/engine.py`, `20261110_222602/main.py`).

---

## 8. Bộ Dữ liệu Kiểm thử & Hệ thống Test Tự động (`eval/`, `tests/`)

### 8.1. Bộ Kiểm thử Gốc & Đóng băng (`eval/` — 16 files gốc + `eval/_sealed/` 1 file)
- **`eval/testset_v1.json`** *(1.98 MiB)*: Bộ Q1 paired (1,290 sp test), Q2 thuộc tính (61 câu), Q4 ảnh mẫu (1,290 ảnh).
- **`eval/handwritten_queries.json`** *(35.48 KiB)*: 100 câu truy vấn viết tay tiếng Việt (75 có dấu, 25 không dấu — Q3).
- **`eval/heldout_queries.json`** *(9.20 KiB)*: 30 câu truy vấn (26 Spec/Rare + 4 Vague `x025–x028` — đã gắn nhãn DEV SET).
- **`eval/q5_brand.json`** *(156.02 KiB)*: 150 câu truy vấn thương hiệu (Q5).
- **`eval/q6_queries.json`** *(2.10 MiB)*: 120 câu val + 120 câu test kiểm soát theo `usage`/`season` (Q6).
- **`eval/unseen_testset_v2.json`** *(7.19 KiB, SHA-256 `90e09202...`)*: 32 câu truy vấn độc lập mới niêm phong.
- **`eval/families.json`** *(3.77 KiB)*: Nhóm họ hàng `colour_families` (14 họ màu) và `article_type_families` (18 họ loại).
- **`eval/gender_gt_policy.json`** *(1.94 KiB, SHA-256 `9c0d2cb5...`)*: Chính sách GT giới tính đóng băng (`strict`, `relaxed`, `hybrid` chính thức).
- **`eval/ab_blind_heldout.csv`**, `manual_ratings_*.csv` & **`eval/_sealed/ab_mapping.json`**: Hồ sơ chấm mù 30 câu A/B (seed `20261109`).
- **Các file kết quả lịch sử:** `results.json` (Pha 3a), `results_v2.json` (Pha 3c), `results_v2b.json` (Pha 3d), `results_3e.json` (Pha 3e), `results_3f.json` (Pha 3f).

### 8.2. Suite Kiểm thử Text→Ảnh & Giao diện (`eval/text2img_test/` — 63 files, cộng 1 `.pyc`)
- `queries.csv` (**708** truy vấn chia 16 nhóm A–P), `results_raw.parquet` (**14,160** dòng C5 vs C0), `vi_map.json`, `vi_map_coverage.csv`, `failures_critical.csv`, `failures_img/` (20 ảnh ghép top-5 ca lỗi), `ui_before_after/` (20 ảnh chụp Playwright trước/sau các tab & truy vấn), `latency_benchmark_tui_sach_hong.json`.
- Bộ script test & kiểm toán (`run_tests.py`, `generate_queries_and_viamap.py`, `build_vi_map.py`, `test_card_render.py`, `test_card_edge_cases.py`, `test_title_formatting.py`, `test_vi_title_unit.py`, `test_clean_model_catalog.py`, `test_model_clean.py`, `benchmark_card_latency.py`, `verify_playwright_fix.py`, `verify_vi_titles_playwright.py`, `verify_post_cleanup.py`, `take_screenshots.py`, `take_after_screenshots.py`) + 2 báo cáo `BAO_CAO_TEXT2IMG.md`, `UI_CHANGELOG.md`.

### 8.3. Suite Kiểm thử Đa phương thức CIR (`eval/cir_test/` — 33 files, cộng 1 `.pyc`)
- **3 bộ CSV chuẩn:** `cir_queries.csv` (**175** ca 7 nhóm Phase A/B), `cir_dev_6groups.csv` (**60** ca tuning Phase C), `cir_heldout.csv` (**90** ca giữ kín 6 nhóm).
- **7 ảnh mẫu kiểm thử upload:** `crop_uploaded_shirt.jpg`, `img_blurred.jpg`, `img_corrupt.jpg` (0 byte), `img_large.jpg`, `img_non_fashion.jpg`, `img_rotated.jpg`, `img_tiny.jpg` + 3 ảnh chụp Playwright (`cir_tab3_before_upload.png`, `after_phase_b_cir_upload_1920x1080.png`, `after_step3_cir_upload_1920x1080.png`).
- **Kết quả JSON & Scripts:** `cir_benchmark_results.json`, `phase_b_{dev,heldout}_results.json`, `template_comparison.csv`, `results/{baseline_phase_b,baseline_phase_b_records,dev_tuning_results,step3_heldout_ablation}.json`, scripts `diagnose_cir_full.py`, `compare_cir_templates.py`, `generate_and_eval_cir_benchmark.py`, `generate_dev_6groups.py`, `generate_heldout_suite.py`, `evaluate_phase_b_full.py`, `tune_dev_params.py`, `evaluate_step3_heldout.py`, `test_cir_normalizer.py`, `test_cir_dynamic_anchor.py`, `test_cir_ui_playwright.py`, và `BAO_CAO_CIR.md`.

### 8.4. Bộ Kiểm thử Tự động Gốc (`tests/` — 8 files `.py`, cộng 14 `.pyc`)
- [`test_app_service.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_app_service.py): Kiểm tra Luật 7, 4 luồng SLA, `c0`/`c4`/`c5`, notice Lavie, I2I `fast`/`accurate` loại self, CIR cache vs upload.
- [`test_keychain_masking_and_alignment.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_keychain_masking_and_alignment.py): Kiểm tra mask `43230` & `42066`, căn hàng 38,644/38,644, re-encode cosine = 1.0000, 0 rò rỉ Keychain.
- [`test_manifest_verification.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_manifest_verification.py): Unit test `unittest` cho `verify_manifest` và chống giả mạo hash.
- [`test_all_text_to_image_cases.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_all_text_to_image_cases.py): Kiểm tra đồng nghĩa, có/không dấu, thương hiệu, Lavie, `name_vi`.
- [`test_q6_independent_filter.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_q6_independent_filter.py): Xác minh `AttributeIndex` không bị ảnh hưởng bởi `usage`/`season`.
- [`test_search_pipeline.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_search_pipeline.py), [`test_cir_flow.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_cir_flow.py), [`test_restore.py`](file:///d:/K%C3%AC%207/Hoc_sau_ver4/tests/test_restore.py): Smoke test cho `VisualSearchEngine`, `load_trained_siglip_cir` / luồng CIR, và `restore_diacritics`.

---

## 9. Hệ thống Báo cáo, Contact Sheets & Manifest (`reports/`, `docs/`)

- **`docs/` (3 files gốc — Khóa chỉ đọc):** `context.md` (mô tả bài toán & KPI ban đầu), `PROJECT.md` (thiết kế kiến trúc M1–M6 ban đầu), `TEST_INFRA.md` (khung kiểm thử).
- **`reports/` (146 files, 43.52 MiB):**
  - **Hồ sơ trung tâm (6 files ở gốc `reports/`):** `run_manifest.json` (SHA-256 & chỉ số đối soát tất cả các pha), `quarantine_log.csv`, `phase4_verifications.json`, `DE_XUAT_TOI_UU_CIR.md`, `BAO_CAO_TONG_QUAN_DU_AN.md`, và báo cáo hiện trạng này (`BAO_CAO_HIEN_TRANG_DU_AN.md`).
  - **`reports/phase1/` (4 files):** `phase1a/` (`phase1_report.md`, `quality_metrics.parquet`) & `phase1b/` (`phase1b_report.md`, `scope_proposal.csv`).
  - **`reports/phase2/` (2 files):** `phase2a/phase2a_findings.md` & `phase2b/phase2_report.md`.
  - **`reports/phase3/` (39 files):** `phase3a/` (`phase3_eval.md`), `phase3b/` (`phase3b_diagnosis.md`), `phase3c/` (`phase3c_eval_v2.md`), `phase3d/` (6 files: `phase3d_addendum.md`, `diagnose_*.json`, `heldout_inspection_b0.json`, `parser_coverage_a6.json`), `phase3e/` (3 files: `phase3e_inference_only.md`, `phase3e_part_a_review.md`, `val_grid_results.json`), `phase3f/` (9 files: `phase3f_reconciliation.md`, `phase3f_errata.md` + các bản lịch sử `rev1`–`rev4` & `template`, `phase3f_latency_benchmark.json`, `q6_val_grid_results.json`), `phase3g/` (16 files JSON/CSV/MD kiểm toán brand, ties, latency, blind A/B, do-bug, padding, q6, slow queries), cùng 2 file ở gốc `reports/phase3/` là `phase3g_latency_multi_run.json` và `phase3g_maxlen.json`.
  - **`reports/phase4/` (13 files):** 10 files ở gốc (`phase4_audit_final.md`, `phase4_fix_report.md`, `phase4_fix_report_v2.md`, `phase4_latency_e2e{,_v2,_v3}.json`, `phase4_latency_optimizations.json`, `phase5a_i2i_quality.json`, `unseen_testset_v2_eval.json`, `_cache_i2i_feats.npz`), `phase4a/` (2 files: `phase4_cir_20queries_eval.json`, `phase4_trunc64_pad128_eval.json`), `phase4b/` (1 file: `phase4_smoke.json`).
  - **`reports/contact_sheets/` (79 ảnh PNG, 25.59 MiB):** Contact sheets kiểm tra chất lượng ảnh Pha 1 (`banner_flagged/not_flagged`, `risk_categories`, ...), 30 contact sheet câu viết tay (`hw_*.png`), 30 contact sheet Blind A/B (`blind_ab_x001..x030.png`), 4 contact sheet câu mơ hồ (`heldout_x025..x028.png`).
  - **`reports/_archive/` (3 files, 0.10 MiB):** Bản sao lưu timestamp ngày `20261109_204420` của `phase2a_findings.md`, `phase3e_inference_only.md`, `phase3f_errata.md`.

---

## 10. Tổng hợp Tình trạng Hoạt động: Cái gì đang chạy vs Cái gì là Lịch sử / Dự phòng

```text
[ĐANG PHỤC VỤ THỰC TẾ (ACTIVE RUNTIME)]
  config.yaml ──► app/main.py (Streamlit 4 Tabs)
                     ├──► eval/text2img_test/vi_map.json
                     ├──► reports/phase4/{phase4_latency_e2e_v3.json, phase5a_i2i_quality.json}
                     └──► app/engine.py (SearchService + Rule 7)
                             └──► src/search_pipeline.py (VisualSearchEngine)
                                    ├── data/checkpoints_siglip-vit/best_model.pt (3.31 GiB)
                                    ├── data/checkpoints_siglip-vit/vit_zero_train_stats.npz
                                    ├── index_v2/{image_embeddings_v2.npy, id_map_v2.json, manifest.json}
                                    ├── data/catalog/catalog.parquet (38,644 in_scope)
                                    ├── data/fashion-dataset/images/ (ảnh sản phẩm)
                                    └── src/{query_parser_v2,cir_normalizer,attr_index,lexical,scoring,vi_vocab,brands_v2}
```

- **Nhóm đang vận hành:** `app/`, `src/` (nhóm serving + từ điển), `index_v2/`, `data/catalog/`, `data/checkpoints_siglip-vit/{best_model.pt, vit_zero_train_stats.npz}`, `data/fashion-dataset/images/`, `eval/text2img_test/vi_map.json`, `eval/families.json`, `config.yaml`.
- **Nhóm chỉ giữ để đối chứng / bảo toàn luật bất biến (Không phục vụ runtime):**
  - *Bị thay thế (`Deprecated`):* `index/` (v1 BICUBIC), `data/checkpoints_siglip-base/` (checkpoint base 29 epochs), `src/brands.json` & `src/query_parser.py` (được kế thừa bởi `brands_v2.json` và `QueryParserV2`), `text_embeddings_v2.npy` (nằm trong `index_v2/` nhưng bị bỏ qua nhờ $\alpha^*=1.0$).
  - *Lưu trữ cách ly / Đào tạo / Build:* `data/_quarantine/` (3,528 ảnh), `data/Shopping100k/`, `_build/` (~40.19 GiB / 41,157.26 MiB dataset hợp nhất + zip), `data/fashion-dataset/fashion-dataset/` (bản giải nén lồng nhau), `_backup_ui/` (3 file backup UI), `reports/_archive/`.
