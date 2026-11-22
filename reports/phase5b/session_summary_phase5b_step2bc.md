# Tóm tắt Phiên Làm việc — Pha 5b (Bước 2b & Bước 2c)

- **Dự án:** SigLIP + Dense Captioning cho hệ thống tra cứu sản phẩm thương mại điện tử (`d:\Kì 7\Hoc_sau_ver4`)
- **Nhánh Git:** `phase7a`
- **Ràng buộc tuân thủ:** Không train/fine-tune hay sửa `best_model.pt` (`SHA-256: e2f82fee...`); seed cố định `42`; `eval/handwritten_queries.json` giữ nhãn **DEV SET** (chỉ chẩn đoán/so sánh tương đối).

---

## 1. Tiến độ Các Bước & Commit Hash

| Bước | Nội dung chính | Commit Hash |
|---|---|---|
| Pha 5 | Đánh giá baseline gốc (`R@10=0.8488`, `mAP=0.5537` trên DEV / Test Gallery) | `bcc5284` |
| Bước 0–1 | Đóng băng baseline (`reports/phase5/baseline_frozen/`), gắn nhãn DEV SET, chẩn đoán ban đầu | `6547f60` |
| Bước 1b | Chẩn đoán bổ sung, Oracle parser, phân tích tính vòng tròn GT | `5b8b219` |
| Bước 1c + 2 (draft) | Hàm đánh giá canonical, phân tích spec ẩn, dựng `index/val_gallery/` & nháp Tập A/B/C | `52ef61f` |
| **Bước 2b** | Tie-rule bảo thủ `1e-5`, ghi chú `val_gallery`, sửa A/B-synth/C, `selection_rule_v2` | `92ca6c5` |
| **Bước 2c** | Đóng băng `parser_v1`, kiểm toán homograph, Tập C lồng nhau, sửa A/B-synth, `check_b_human.py`, khóa `selection_rule_v3` | `8181ff9` (tag: `phase5b-rule-v3`) |

---

## 2. Kết quả Chính Trong Phiên (Bước 2b → Bước 2c)

### 2.1. Tie-rule bảo thủ (`1e-5`) & Exact Search (`src/eval/canonical_eval.py`)
- Áp dụng `apply_conservative_tie_break(epsilon=1e-5)` (trong nhóm hòa chênh $\le 10^{-5}$, xếp ảnh `rel=0` lên trước `rel=1`) và mặc định `force_exact=True` (`q @ E.T`).
- Unit test `test_tie_rule_rank1_conservative_r1_zero` xác nhận cặp hòa ở rank 1 cho `R@1 = 0.0`.
- Đối chiếu baseline Pha 5 (`parser_v1`, $\gamma=0.3$, `pool=100`) trên DEV SET (không sửa `baseline_frozen/`):
  - **HNSW (Phase 5 baseline):** `model_only` `R@1=0.2907, R@5=0.5698, R@10=0.7093, mAP=0.3279` | `system` `R@1=0.6047, R@5=0.8256, R@10=0.8488, mAP=0.5537`
  - **Exact + conservative tie-rule (`1e-5`):** `model_only` `R@1=0.3023, R@5=0.5698, R@10=0.7093, mAP=0.3282` | `system` `R@1=0.6047, R@5=0.8256, R@10=0.8488, mAP=0.5540`

### 2.2. Đóng băng `parser_v1` & Kiểm toán Homograph (`2c.1`)
- Đóng băng `src/serve/parser_v1.py` (`SHA-256: 111002f69185514f2a9fef86438e1fbd25e6108b0d92f16e7ef1e70e74fe58eb`).
- Kết quả kiểm toán `parser_v1` (`reports/phase5b/parser_v1_audit.json`):
  - **DEV (100 câu):** FP/truy vấn `baseColour = 12.00%` (FP/dự đoán `15.79%`), `articleType = 1.00%`, `gender = 0.00%`. `"tìm mua"` → `Purple`: `0` câu.
  - **Val A (240 câu):** FP/truy vấn `baseColour = 4.17%` (FP/dự đoán `4.98%`), `articleType = 0.42%`, `gender = 0.00%`. `"tìm mua"` → `Purple`: `0` câu.
  - **Val B-synth (245 câu | 165 sạch):** FP/truy vấn `baseColour = 32.65%` (`32.12%` câu sạch; FP/dự đoán `42.11%` / `40.15%`), `articleType = 0.00%`, `gender = 0.00%`.
  - **Xác nhận lỗi `"tìm mua"`/`"tim mua"` → `Purple`:** Có xảy ra trên **`38` câu** B-synth (`26` câu sạch).
- Lập bảng đồng âm không dấu nguồn độc lập (âm tiết tiếng Việt 6 thanh & từ điển mở `Viet74K`) tại `reports/phase5b/homograph_table_2c1c.md`.

### 2.3. Chuẩn hóa & Khóa Bộ Tuning Độc lập trên `index/val_gallery/` (`1.884` sp)
- **Tập A (`eval/val_queries_a_synthetic.json`):** `240` câu, `118` `spec_cluster_id` độc nhất; sửa `sarees` → `"sari Ấn Độ"` / `"sari an do"` (4 câu, không dùng `"áo dài"`); ghi nhận `86` `articleType` & `6` `baseColour` thiếu Việt hóa ở `reports/phase5b/catalog_vi_gaps.md`.
  - SHA-256: `4fd30c2028a2b8c63ea2fac0d0147297e4d11f8b1eb40ee4ff2f1f8a6cf4f2b4`
- **Tập B-synth (`eval/val_queries_b_synth.json`):** `245` câu (`165` sạch + `80` typo), `166` `spec_cluster_id` (`165` phần sạch). Chuyển các khái niệm đa nghĩa sang `spec.articleType` đa trị; tính lại `dia_class` cho typo (`wrong_tone` không còn `"full"`); bổ sung `35` câu typo (`telex_raw`, `vni_raw`, `sticky_keys`, `wrong_tone`, `seed=42`). Changelog tại `reports/phase5b/val_ab_changelog.md`.
  - SHA-256: `41739174392fd1fe4cb14c76bddefc33ff23cf7dd7e885d9e999987db3bea6a1`
- **Tập C (`eval/val_queries_c_error_injection.json`, `src/eval/error_injection.py`):** `160` câu, tiêm lỗi **lồng nhau** có hiệu lực ($u_i \sim U(0,1)$, `seed=42`) kèm hàng tham chiếu `"parsed_real_v1"`. Tỉ lệ lỗi thực: `0%` (`0.0000`, 0 câu), `10%` (`0.0875`, 14 câu), `20%` (`0.1562`, 25 câu), `30%` (`0.3250`, 52 câu).
  - SHA-256: `dd843ebae1e050962f9876b1e304ffe49260dd8732e2a4209e1fa0e1fdc9089c`
- **Tập B-human (`eval/val_queries_b_human.json`):** Giữ trống `[]` kèm hướng dẫn `eval/val_queries_b_human_schema.md` và công cụ kiểm tra `scripts/check_b_human.py` (không nạp model; kiểm tra $\ge 60$ câu, `n_relevant < 5`, `dia_class`, Jaccard $\ge 0.8$, token ngoài lexicon v1).

### 2.4. Quy tắc Chọn Cấu hình v3 (`2c.6`)
- Khóa tại `reports/phase5b/selection_rule_v3.json` (`SHA-256: fa4e0df3469b0f83a458c38a466efaec3ed3b4d951ee52b80c36dc4c783de07e`) và `reports/phase5b/selection_rule_v3.md` (`SHA-256: d9e7b56d3301b8e0b839706844651ed5a3a00c5516d194898303f1c6e39db4eb`), tag git `phase5b-rule-v3`.
- Liệt kê tường minh **22** cấu hình ($\le 24$), giữ thứ tự đơn giản, CV 5-fold theo cụm trên `A + B-synth sạch`, cổng xác nhận `B-human`, 5 ràng buộc (C 20%, tụt thảm họa $\le 5\%$, R@10, độ trễ p95, FP parser $\le$ v1), báo song song cấu hình Tốt nhất & An toàn, và ablation bắt buộc `cfg_02` (`parser_v2 + γ=0.3 đều`).

---

## 3. Kiểm thử (`pytest`)
- Lệnh: `pytest tests/ -v`
- Kết quả cuối cùng: **35 passed, 0 failed** (gồm `tests/test_phase1.py`, `test_phase2_caption.py`, `test_phase3_parity.py`, `test_phase4_index.py`, `test_phase5_eval.py`, `test_phase5b_step2b.py`, `test_phase5b_step2c.py`).

---

## 4. Điểm Dừng Hiện Tại & Bước Tiếp Theo
1. Người dùng điền $\ge 60$ câu vào `eval/val_queries_b_human.json` theo `eval/val_queries_b_human_schema.md` và duyệt Bước 2c + quy tắc chọn v3.
2. Agent chạy `python scripts/check_b_human.py` để kiểm tra `B-human`, khóa SHA-256, rồi mới triển khai Bước 3–6 (`parser_v2`, thử nghiệm lưới cấu hình trên CV, xác nhận cổng B-human, đánh giá trên Test Gallery & độ trễ).
