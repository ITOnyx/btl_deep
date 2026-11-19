# Báo cáo Kiểm định Độc lập — Pha 1: Tái lập Trạng thái và Toàn vẹn

- **Thời điểm chạy (UTC):** `2026-11-19T14:27:27Z`
- **Git HEAD lúc kiểm định:** `0c8e49e2afe14328bcd272fa2a50e184389702ea` (nhánh `add/test-crawl-img`)
- **Seed:** `42`
- **Bằng chứng:**
  - `reports/verification/run_manifest.json`
  - `reports/verification/phase_1/run_phase1.py`
  - `reports/verification/phase_1/phase1_results.json`
  - `reports/verification/phase_1/phase1_stdout.txt`
  - `reports/verification/phase_1/pytest_stdout.txt`
  - `reports/verification/phase_1/tampered_index/manifest.json`

## Lệnh đã chạy
1. `python reports/verification/phase_1/run_phase1.py | Tee-Object -FilePath reports/verification/phase_1/phase1_stdout.txt`
2. `pytest -v 2>&1 | Tee-Object -FilePath reports/verification/phase_1/pytest_stdout.txt`

## Bảng kết quả Pha 1

| Mục | Tuyên bố báo cáo | Giá trị đo lại thực tế | Chênh lệch | Bằng chứng | Kết luận |
|---|---|---|---|---|---|
| 1. Git HEAD & WIP | HEAD `0c8e49e2...`, nhánh `add/test-crawl-img`, có WIP chưa commit | HEAD `0c8e49e2afe14328bcd272fa2a50e184389702ea`, nhánh `add/test-crawl-img`, 47 mục WIP (18 `M`, 29 `??`, chưa tính `reports/verification/`) | Khớp HEAD & nhánh | `phase1_results.json` (`1_git`) | **PASS** |
| 2. SHA-256 checkpoint | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` (3,549,699,293 bytes) | 0 | `phase1_results.json` (`2_checkpoint`) | **PASS** |
| 3. SHA-256 tập eval | `test_split.json`: `d61bd87b…`; `test_queries_heldout_v2.json`: `96c75433…`; `cir_eval_test.json`: `d6e33f4e…` | Đều khớp cả tiền tố và toàn bộ 64 ký tự trong `*.sha256` của repo (`d61bd87b7ae495cb...`, `96c75433f911f212...`, `d6e33f4ee9141526...`) | 0 (lưu ý: `config.yaml` dòng 210 ghi nhầm tên `eval/heldout_queries_v2.json` thay vì `eval/test_queries_heldout_v2.json`) | `phase1_results.json` (`3_eval_hashes`) | **PASS** |
| 4. `index/*/manifest.json` & Hash Guard | Tất cả manifest khớp checkpoint; Hash Guard từ chối nạp khi lệch | 8/8 `index/*/manifest.json` khớp SHA-256 checkpoint. Thử trên bản sao `tampered_index/manifest.json` (`checkpoint_hash="0"*64`): `ECommerceSearchEngine` ném `ValueError: HASH GUARD LỆCH (Luật 6)` | 0 | `phase1_results.json` (`4_manifests_and_hash_guard`) | **PASS** |
| 5. Môi trường & GPU | Python 3.12.0, torch 2.6.0+cu124, faiss 1.15.1, bitsandbytes 0.50.2, RTX 2050 4 GB | Python 3.12.0, torch 2.6.0+cu124, faiss 1.15.1, bitsandbytes 0.50.2, NVIDIA GeForce RTX 2050 (4096 MiB / 4095.6 MB) | 0 (GPU nền lúc đo: 57°C, 30%) | `phase1_results.json` (`5_environment`), `run_manifest.json` | **PASS** |
| 6. Toàn bộ `pytest` | 105/105 PASS (Pha 6) | **106 passed, 1 failed, 0 skipped** (tổng 107 test, 137.51s). Fail: `tests/test_app_structure.py::test_app_imports` (`AssertionError`: `app.components.product_card` thiếu `render_product_grid`) | +2 test mới trong WIP (`tests/test_phase6_app.py`), **1 FAIL** do sửa WIP ở `app/components/product_card.py` | `pytest_stdout.txt` | **FAIL** |

## Ghi chú chênh lệch về luật & cấu hình
- Số thứ tự luật trong `reports/phase7/final_project_report.md` khớp `.agents/AGENTS.md` (Luật 1–7; Hash Guard = Luật 6).
- `config.yaml` dòng 210 ghi `heldout_queries_v2: "eval/heldout_queries_v2.json"`, trong khi tệp thực tế trên đĩa là `eval/test_queries_heldout_v2.json`.

## Điểm mơ hồ cần người dùng quyết định
1. Sau khi commit `reports/verification` cho Pha 1 (`git commit -m "verify phase 1"`), `git rev-parse HEAD` sẽ dịch khỏi `0c8e49e2...` sang commit kiểm định mới (và nhánh vẫn giữ nguyên 47 file WIP chưa commit). Xác nhận tiếp tục commit theo từng pha như hướng dẫn hay giữ nguyên HEAD?
2. `tests/test_app_structure.py::test_app_imports` FAIL trên working tree WIP hiện tại (`app/components/product_card.py` bị sửa). Theo Luật K1 chúng tôi không sửa mã và ghi nhận FAIL; xác nhận chuyển sang Pha 2?
