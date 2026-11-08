# Project: Multimodal Visual Search for Vietnamese E-Commerce

## Architecture
- **Model Layer**: `MultimodalSigLIPWithCIR` loaded from `data/checkpoints_siglip-base/best_model.pt` (with multi-checkpoint support via `src/config.py` for `data/checkpoints_siglip-vit/`) on GPU (RTX 2050, FP16, VRAM ~740MB). Backbone `google/siglip-base-patch16-256-multilingual` with 128-token position embeddings and 1024-dim `UpgradedAttentionCIRCombiner`.
- **Indexing Layer**: FAISS HNSW index (`IndexHNSWFlat`, metric `METRIC_INNER_PRODUCT`, 768-dim) indexing L2-normalized image embeddings for catalog items, coupled with comprehensive product metadata (`product_id`, `title`, `category`, `image_path`).
- **Retrieval Engine**:
  - Vietnamese fashion query normalizer: Unicode NFC, colloquial synonym swaps (`áo phông` -> `áo thun`, `quần bò` -> `quần jean`, etc.), intent stripping.
  - Dense Vector Retrieval: SigLIP query embedding + FAISS HNSW search.
  - Lexical Search: Pure Python BM25 inverted index on product title and category.
  - Fusion & Reranking: Reciprocal Rank Fusion (RRF, $k=60$) + Attribute Matching Reranker to maximize Recall@10 and push mAP >= 0.75.
- **Application Layer**: Streamlit UI (`app/app_streamlit.py`) supporting Text-to-Image, Image-to-Image, and CIR Composed Image Retrieval with dynamic filters (category, score, top-k) and real image display from `data/images/`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Checkpoint Architecture & Offline Loading | Load `best_model.pt` with `max_length=128`, `projection_dim=1024`, offline HuggingFace config, FP16 GPU inference | M1 | Survey E1, E2 |
| 2 | FAISS HNSW Real Index & Metadata Sync | Real `IndexHNSWFlat` (768-d, Inner Product) with full metadata and image path mapping, zero mock vectors | M2 | Survey E2 |
| 3 | Vietnamese Query Preprocessing | Unicode NFC normalization, colloquial synonym expansion, fashion dictionary normalization | M3 | Survey E3 |
| 4 | Lexical BM25 & Hybrid RRF Reranker | Pure Python BM25 index + Reciprocal Rank Fusion ($k=60$) + Attribute Matching Reranker | M3 | Survey E2, E3 |
| 5 | Automated Metrics Verification | Script measuring Recall@10 (>=0.85), mAP (>=0.75), Latency (<200ms) on `golden_queries_vi.jsonl` | M4 | Survey E3, ORIGINAL_REQUEST |
| 6 | Streamlit UI (Text, Image, CIR) | Interactive UI with 3 tabs (Text-to-Image, Image-to-Image, CIR), dynamic category/score filters, real images | M5 | Survey E1, E3, ORIGINAL_REQUEST |
| 7 | End-to-End Handover Verification | Automated test runner, benchmark report generation, zero mock validation | M6 | ORIGINAL_REQUEST |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Model Loading & Offline Architecture Fix | Fix `scripts/model_siglip_cir.py` (`projection_dim=1024`, `max_length=128`), safe offline loader | none | PLANNED |
| M2 | Real FAISS Index & Metadata Synchronization | Verify & format `faiss_hnsw_index.bin` + `product_metadata.json` with resolved image paths | M1 | PLANNED |
| M3 | High-Precision Retrieval Engine (BM25 + RRF) | Implement query preprocessor, BM25 index, RRF combiner, and attribute reranker in `scripts/retrieval_engine.py` | M1, M2 | PLANNED |
| M4 | Automated Evaluation Harness & Metrics Target | Implement `scripts/evaluate_retrieval.py` and verify Recall@10 >= 0.85, mAP >= 0.75, Latency < 200ms | M3 | PLANNED |
| M5 | Streamlit Multimodal Visual Search Application | Refactor `scripts/app_streamlit.py` with 3 tabs, dynamic filters, real FAISS, real images, latency badge | M3, M4 | PLANNED |
| M6 | Acceptance Verification & Handover | Complete E2E verification test suite, audit clean check, final benchmark report | M4, M5 | PLANNED |

## Interface Contracts
### Model Layer ↔ Retrieval Engine
- `model.encode_text(input_ids, attention_mask=None) -> np.ndarray (N, 768)` (L2 normalized)
- `model.encode_image(pixel_values) -> np.ndarray (N, 768)` (L2 normalized)
- `model.compose_query(ref_pixels, mod_ids, mod_mask=None) -> np.ndarray (N, 768)` (L2 normalized)

### FAISS Index & Metadata ↔ Retrieval Engine
- Index: `faiss.read_index(...)` returning top-K `(distances, indices)`
- Metadata: `List[Dict[str, Any]]` where index `i` corresponds to vector `i` in FAISS, containing `product_id`, `title`, `category`, `image_path`.

### Retrieval Engine ↔ Streamlit UI
- `search_by_text(query: str, top_k: int = 10, category_filter: Optional[str] = None, min_score: float = 0.0) -> List[SearchResult]`
- `search_by_image(image: PIL.Image, top_k: int = 10, category_filter: Optional[str] = None, min_score: float = 0.0) -> List[SearchResult]`
- `search_by_cir(ref_image: PIL.Image, modifier_text: str, top_k: int = 10, category_filter: Optional[str] = None, min_score: float = 0.0) -> List[SearchResult]`
- `SearchResult`: dataclass/dict with `product_id`, `title`, `category`, `image_path`, `score` (0.0 to 1.0 or %), `rank`.

## Code Layout
- `src/config.py`: Centralized paths & multi-checkpoint configuration (`checkpoints_siglip-base`, `checkpoints_siglip-vit`)
- `src/model_siglip_cir.py`: Core model architecture (`MultimodalSigLIPWithCIR`, `UpgradedAttentionCIRCombiner`)
- `src/retrieval_engine.py`: Preprocessing, BM25, Dense search, RRF fusion, and Attribute Reranker
- `src/loss_and_metrics.py`: Contrastive SigLIP loss, triplet loss & evaluation metrics
- `src/data_augmentation.py`: Multimodal Vietnamese e-commerce augmentation
- `app/app_streamlit.py`: Streamlit application entry point
- `scripts/training/build_faiss_index.py`: Index builder & metadata synchronizer
- `scripts/training/train_pipeline.py`: End-to-end anti-overfit training pipeline
- `scripts/evaluation/evaluate_retrieval.py`: Benchmark script on `golden_queries_vi.jsonl` checking Recall@10, mAP, Latency
- `scripts/evaluation/evaluate_test_and_plot.py`: Test loss/accuracy evaluation & chart dashboard generator
- `outputs/reports/` & `outputs/figures/`: Evaluation JSON metrics and visualization charts
