# Tóm tắt Toàn bộ Quá trình — Từ Pha 1 đến Pha 5b (Bước 2c)

- **Dự án:** SigLIP + Dense Captioning cho hệ thống tra cứu sản phẩm thương mại điện tử (`d:\Kì 7\Hoc_sau_ver4`)
- **Nhánh Git:** `phase7a` | **Seed:** `42` | **Checkpoint:** `best_model.pt` (`SHA-256: e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`, không train/sửa).

---

## Phần A — Tổng quan Pha 1 → Pha 5

1. **Pha 1 (`43677ea`, `10586c7`) — Làm sạch, Kiểm kê & Chia Split:**
   - Cách ly **736** tệp cũ vào `data/_quarantine/legacy_rebuild_20261111/` (tag `legacy-before-rebuild`), bảo tồn `best_model.pt`, `src/brands_v2.json`, `eval/handwritten_queries.json`, `eval/gender_gt_policy.json`.
   - Dựng `catalog_rebuild.parquet` và khôi phục split `train / val / test` chống rò rỉ.
   - Báo cáo: `reports/phase1/phase1_report.md`.
2. **Pha 2 (`f5d41c0`, `dfc150a`, `2e057d6`) — Dense Captioning (`InternVL2-2B` 8-bit):**
   - Chạy thử 200 ảnh pilot từ train split (peak VRAM 2.80 GB / 3.23 GB rảnh); `merge_v1` đạt 0.0% vượt trần 128 token SigLIP (`mean=127.4`), 100% khớp metadata, 0% chữ Hán.
   - Tối ưu tốc độ sinh (`max_num=1`, `max_new_tokens=48`, tăng tốc ~6x).
   - Báo cáo: `reports/phase2/phase2_pilot_report.md`.
3. **Pha 3 (`6ef2eec`, `5c2cd53`) — Nạp Checkpoint, Hash Guard & Parity Check:**
   - Dựng `HighlyEffectiveSigLIP` + `SigLIPPreprocessor` thống nhất; parity check 200 ảnh mẫu đạt **200/200 $\ge 0.999$** (min cosine **`0.99966007`**). Pytest: **18/18 PASSED**.
   - Báo cáo: `reports/phase3/phase3_report.md`.
4. **Pha 4 (`631d8e6`, `408d158`) — FAISS HNSW Index & Sanity Check:**
   - Dựng `IndexHNSWFlat` ($M=32$, `efConstruction=200`, `efSearch=128`) đạt **100% Recall@10** vs Exact: `index/test_gallery/` (**1.290** sp, `0.032 ms`/query) và `index/reference_gallery/` (**38.646** vector, `0.121 ms`/query). Đạt 4/4 sanity check. Pytest: **21/21 PASSED**.
   - Báo cáo: `reports/phase4/phase4_report.md`.
5. **Pha 5 (`bcc5284`) — Đánh giá Baseline Trung thực:**
   - Trên Official Test Gallery (`N=1.290`, 86/100 câu có GT theo `hybrid`):
     - **Model thuần:** `R@1=0.2907`, `R@5=0.5698`, `R@10=0.7093`, `mAP=0.3279`.
     - **Hệ thống (`gamma=0.3`, `pool=100`):** `R@1=0.6047`, `R@5=0.8256`, **`R@10=0.8488`** (73/86), **`mAP=0.5537`**.
     - **Latency p95:** Cả 4 luồng `< 200 ms` (`T2I=29.86 ms`, `I2I=126.51 ms`, `CIR catalog=36.98 ms`, `CIR upload=165.49 ms`).
   - Pytest: **28/28 PASSED**. Báo cáo: `reports/phase5/phase5_report.md`.

---

## Phần B — Chi tiết Pha 5b (Theo Từng Prompt)

1. **Prompt 1 — Bước 0 & Bước 1 (`6547f60`):**
   - Đóng băng baseline Pha 5 tại `reports/phase5/baseline_frozen/`, gắn nhãn `eval/handwritten_queries.json` là **DEV SET** (chỉ chẩn đoán).
   - Chẩn đoán `candidate_pool` 100 → 1.290 giúp `mAP` tăng `+0.0302` (`0.5537 → 0.5839`) và `R@10` lên `0.8721`; phân loại 13 truy vấn thất bại (`reports/phase5b/diagnostics_summary.md`).
2. **Prompt 2 — Bước 1b (`5b8b219`):**
   - Xác nhận 2 ID `in_scope=False` trong `reference_gallery` (`43230`, `42066`), chạy Oracle Parser, phân tích 13 câu thất bại, quét `gamma` chẩn đoán và phân tích tính vòng tròn GT (`reports/phase5b/diagnostics_1b_report.md`).
3. **Prompt 3 — Bước 1c & Bước 2 nháp (`52ef61f`):**
   - Dựng `src/eval/canonical_eval.py`, giải thích lệch `R@1` (`0.2907` vs `0.3023` do Query 56 giữa HNSW và Exact), bổ sung `ERRATUM.md`, gắn cờ spec ẩn, chứng minh lỗi cấu trúc của `gamma=0.3` (`reports/phase5b/diagnostics_1c.md`).
   - Dựng `index/val_gallery/` (**1.884** sản phẩm val sạch) và nháp Tập A/B/C.
4. **Prompt 4 — Bước 2b (`92ca6c5`):**
   - Áp dụng `tie_rule="conservative"` ($\epsilon = 10^{-5}$) + `force_exact=True`.
   - Xác nhận 6/6 ca `baseColour` đoán dư trên DEV đều là parser bịa do đồng âm bỏ dấu (`bé→be`, `đồ→do`, `cầm/cam→cam`).
   - Ghi chú phóng đại cosine trên `val_gallery` (yêu cầu z-score ở Bước 3e); sửa Tập A, tách Tập B (`B-synth` + `B-human` trống), dựng Tập C, khóa `selection_rule_v2` (`31/31` pytest passed).
5. **Prompt 5 — Bước 2c (`8181ff9`, tag `phase5b-rule-v3`):**
   - Đóng băng `src/serve/parser_v1.py` (`SHA-256: 111002f69185514f2a9fef86438e1fbd25e6108b0d92f16e7ef1e70e74fe58eb`).
   - Kiểm toán `parser_v1` (`reports/phase5b/parser_v1_audit.json`): `baseColour` FP trên B-synth là **32.65%/query** (**42.11%/dự đoán**), xác nhận **`38` câu** bị `"tìm mua"`/`"tim mua"` gán nhầm `Purple`; lập bảng đồng âm nguồn độc lập (`reports/phase5b/homograph_table_2c1c.md`).
   - Unit test `test_tie_rule_rank1_conservative_r1_zero` đạt (`R@1 = 0.0` khi hòa rank 1); đối chiếu DEV Exact+tie-rule (`model_only R@1=0.3023`, `system R@1=0.6047, mAP=0.5540`).
   - Chốt Tập A (`240` câu, `118` cụm, SHA-256: `4fd30c20...`), Tập B-synth (`245` câu = `165` sạch + `80` typo, `166` cụm, SHA-256: `41739174...`), Tập C lồng nhau (`160` câu, 4 mức lỗi thực `0% / 8.75% / 15.62% / 32.50%` + `parsed_real_v1`, SHA-256: `dd843eba...`).
   - Tạo công cụ `scripts/check_b_human.py` và khóa quy tắc chọn **v3** (22 cấu hình, CV 5-fold theo cụm trên A + B-synth sạch, cổng B-human) tại `reports/phase5b/selection_rule_v3.json` (`SHA-256: fa4e0df3...`).
   - **Pytest hiện tại:** **35/35 PASSED**.

---

## Điểm Dừng Hiện Tại
Chờ bạn điền $\ge 60$ câu vào `eval/val_queries_b_human.json` (theo `eval/val_queries_b_human_schema.md`) và duyệt **Bước 2c + quy tắc chọn v3** để chạy `scripts/check_b_human.py` và chuyển sang **Bước 3–6**.
