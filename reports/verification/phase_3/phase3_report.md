# Báo cáo Kiểm định Độc lập — Pha 3: Checkpoint, Tiền xử lý và Chỉ mục FAISS HNSW

- **Thời điểm chạy (UTC):** `2026-11-19T17:09:41Z`
- **Môi trường thực thi:** Worktree sạch `reports/verification/_clean_head` tại đúng commit `0c8e49e2afe14328bcd272fa2a50e184389702ea` (`core.autocrlf=false`, nối junction chỉ-đọc `data/` và `index/`, đã gỡ sạch bằng `git worktree remove` sau khi chạy). Toàn bộ module `src.model.siglip_model`, `src.preprocess.pipeline`, `src.index.faiss_index` được xác minh nạp 100% từ `_clean_head`.
- **Seed:** `42`
- **Bằng chứng:**
  - `reports/verification/phase_3/run_phase3.py`
  - `reports/verification/phase_3/phase3_results.json` (và `reports/verification/phase3/phase3_results.json`)
  - `reports/verification/phase_3/phase3_stdout.txt` (và `reports/verification/phase3/phase3_stdout.txt`)

---

## Bảng kết quả Pha 3 (Mục 12 – 16)

| Mục | Tuyên bố báo cáo (`final_project_report.md`, `phase3_report.md`, `phase4_report.md`) | Giá trị đo lại thực tế (trên worktree sạch `0c8e49e`) | Chênh lệch | Bằng chứng | Kết luận |
|---|---|---|---|---|---|
| **12. Nạp checkpoint & kiến trúc** | 908/908 tensor key khớp (`strict=True`), `position_embeddings` 128 tokens, embedding 1152 chiều, norm L2 ≈ 1 | `checkpoint_tensor_key_count = 908`, `model_tensor_key_count = 908`, `missing = 0`, `unexpected = 0` (`908/908` `strict=True`). `position_embedding.weight.shape = [128, 1152]`, `max_position_embeddings = 128`, `embed_dim = 1152`. Norm L2: text `[0.999894 .. 1.000090]`, image `mean = 1.000003` (`min = 0.999651`, `max = 1.000381` ở FP16). | 0 | `phase3_results.json` (`12_checkpoint_and_architecture`) | **PASS** |
| **13. Parity tiền xử lý (200 ảnh mẫu đối chứng, Luật 4 & K4)** | `200/200 >= 0.999`; `min cosine = 0.99966007`, `mean = 0.99998093`, `std = 3.40e-5` | Qua `SigLIPPreprocessor` (`src/preprocess/pipeline.py`): **`200/200 >= 0.999 (100%)`**; `min cosine = 0.9996600747` (`|Δ| = 4.71e-9`), `mean = 0.9999809265` (`|Δ| = 3.49e-9`), `median = 0.99999106`, `max = 0.99999779`, `std = 3.4019e-5`. | `|Δ| < 5e-9` (khớp tuyệt đối 8 chữ số thập phân) | `phase3_results.json` (`13_parity_check_200`) | **PASS** |
| **14. Cấu hình & quy mô FAISS HNSW** | `IndexHNSWFlat`, `D=1152`, `M=32`, `efConstruction=200`, `efSearch=128`, Inner Product; `ntotal = 1,290` (`test_gallery`) và `38,646` (`reference_gallery`) | Cả 2 chỉ mục đều là `IndexHNSWFlat`, `d = 1152`, `M = 32` (`nb_neighbors(0) = 64`, `nb_neighbors(1) = 32`), `efConstruction = 200`, `efSearch = 128`, `metric_type = 0` (`METRIC_INNER_PRODUCT`). `ntotal`: `test_gallery = 1,290`, `reference_gallery = 38,646`. | 0 | `phase3_results.json` (`14_faiss_hnsw_params`) | **PASS** |
| **15. ANN Recall@10 vs `IndexFlatIP`, Self-Query & thứ tự ánh xạ ID** | ANN Recall@10 = `100.00%`; Self-query Rank 1 với `cosine ≈ 1.0000`; thứ tự `product_id` khớp thứ tự vector | - **ANN Recall@10 vs `IndexFlatIP`:** đạt **`1.0000 (100.00%)`** tại mọi `efSearch ∈ {32, 64, 128, 256}` trên 100 truy vấn mẫu (`seed=42`), đạt **`100.00%` trên toàn bộ `1,290/1,290` vector** của `test_gallery` (`efSearch=128`), và **`100.00%` trên `1,000/1,000` vector** của `reference_gallery` (`efSearch=128`).<br>- **Self-query:** `50/50` mẫu ngẫu nhiên (`seed=42`) đạt Rank 1 (`min cosine = 0.99999988`, `mean = 1.00000004`) trên cả 2 gallery; kiểm tra vét cạn **`1,290/1,290` vector** của `test_gallery` đạt **100% chính nó ở Rank 1** (`min cosine = 0.99999982`).<br>- **Khớp thứ tự vector & `id_map.json`:** `raw_index.reconstruct_n(0, ntotal)` khớp `image_embeddings.npy` (`max_abs_diff = 4.50e-5`); tái encode trực tiếp 5 ảnh ngẫu nhiên từ `id_map[i]` khớp vector thứ `i` với `cosine >= 0.9994`. | 0 | `phase3_results.json` (`15_ann_recall_self_query_and_mapping`) | **PASS** |
| **16. Bốn Sanity Check Pha 4** | (1) 0 ID trùng; (2) 0 rò rỉ `test → train`; (3) Self-query Rank 1 (`cosine >= 0.9999`); (4) Số vector đồng nhất với `manifest.json` + Hash Guard | Cả `test_gallery` (`1290 == 1290 == 1290 == 1290`) và `reference_gallery` (`38646 == 38646 == 38646 == 38646`) đều **ĐẠT 4/4 tiêu chí** của `src/eval/sanity_check.py` + Hash Guard (`e2f82fee...`). *(Lưu ý: Tiêu chí 2 trong `sanity_check.py` chỉ kiểm tra giao tập `product_id` = 0; phát hiện 3 ảnh trùng SHA-1 / 78 cặp trùng pHash khác thuộc tính đã được báo cáo ở Pha 2 Mục 8).* | 0 | `phase3_results.json` (`16_sanity_checks_phase4`) | **PASS** |

---

## Danh sách 10 ảnh có Cosine thấp nhất trong 200 ảnh đối chứng Parity Check (Mục 13)

| Hạng thấp nhất | Chỉ số (`0..199`) | `item_id` | `articleType` | Đường dẫn ảnh | Cosine Similarity | Đạt ngưỡng `≥ 0.999`? |
|---:|---:|---:|---|---|---:|---|
| 1 (Min) | 63 | `18237` | `Tshirts` | `data/fashion-dataset/images/18237.jpg` | **`0.99966007`** | ✅ Đạt |
| 2 | 94 | `41268` | `Wallets` | `data/fashion-dataset/images/41268.jpg` | `0.99979645` | ✅ Đạt |
| 3 | 15 | `42419` | `Tops` | `data/fashion-dataset/images/42419.jpg` | `0.99983728` | ✅ Đạt |
| 4 | 53 | `42841` | `Handbags` | `data/fashion-dataset/images/42841.jpg` | `0.99984753` | ✅ Đạt |
| 5 | 67 | `19123` | `Flip Flops` | `data/fashion-dataset/images/19123.jpg` | `0.99988705` | ✅ Đạt |
| 6 | 159 | `12500` | `Shirts` | `data/fashion-dataset/images/12500.jpg` | `0.99990964` | ✅ Đạt |
| 7 | 3 | `21379` | `Track Pants` | `data/fashion-dataset/images/21379.jpg` | `0.99992800` | ✅ Đạt |
| 8 | 91 | `23882` | `Sports Shoes` | `data/fashion-dataset/images/23882.jpg` | `0.99993336` | ✅ Đạt |
| 9 | 149 | `21510` | `Handbags` | `data/fashion-dataset/images/21510.jpg` | `0.99993366` | ✅ Đạt |
| 10 | 49 | `10866` | `Tshirts` | `data/fashion-dataset/images/10866.jpg` | `0.99994075` | ✅ Đạt |
