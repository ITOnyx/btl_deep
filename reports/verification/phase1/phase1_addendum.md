# Bổ sung Kiểm định Pha 1 (Mục A.1 – A.4 & C.10 – C.11)

- **Thời điểm cập nhật (UTC):** `2026-11-19T17:10:05Z`
- **Commit đối chiếu:** `0c8e49e2afe14328bcd272fa2a50e184389702ea`
- **Bằng chứng:**
  - `reports/verification/phase_1/pytest_clean_head_stdout.txt`
  - `reports/verification/phase_1/pytest_clean_head_with_data_index_stdout.txt`
  - `reports/verification/phase_1/pytest_clean_head_autocrlf_false_stdout.txt`

---

## A.1 & C.10. Kiểm chứng `pytest` trên bản checkout sạch `0c8e49e` (`core.autocrlf=true` vs `core.autocrlf=false`)

- **Quy trình:** Tạo worktree tách biệt tại `reports/verification/_clean_head` ở commit `0c8e49e2afe14328bcd272fa2a50e184389702ea`, nối junction chỉ-đọc (`mklink /J`) tới `data/` và `index/`, chạy `pytest -v`, sau đó gỡ junction và gỡ worktree bằng `git worktree remove` (không đụng working tree hiện tại).
- **Tổng số test được git theo dõi tại `0c8e49e`:** **94 tests** (trong đó `tests/test_app_structure.py::test_app_imports` **PASSED** ở cả hai chế độ, xác nhận lỗi FAIL trên working tree là do thay đổi WIP `M app/components/product_card.py` đổi tên `render_product_grid` → `render_result_grid`).
- **So sánh hai chế độ xuống dòng khi checkout từ Git object database (repo không có `.gitattributes`):**

| Chế độ checkout worktree `0c8e49e` (+ junction `data/`, `index/`) | Kết quả `pytest` | Các test FAIL | Nguyên nhân gốc rễ |
|---|---|---|---|
| **1. `core.autocrlf=true` (CRLF)** | **91 passed, 3 failed** | 1) `test_build_val_text_index_v5_untouched` (`scripts/build_val_text_index_v5.py`)<br>2) `test_p3_frozen_hashes_and_hash_guard` (`src/serve/parser_v1.py`)<br>3) `test_selection_rule_cir_v1_locked` (`reports/phase5c/selection_rule_cir_v1.json`) | Ba tệp này trên working tree gốc được lưu với **LF (`\n`)** và mã băm SHA-256 đối chứng được tính trên byte LF. Khi checkout với `autocrlf=true`, Git đổi sang CRLF nên lệch SHA-256 byte thô. |
| **2. `core.autocrlf=false` (LF)** | **90 passed, 4 failed** | 1) `test_test_split_hash_match` (`eval/test_split.json`)<br>2) `test_p3_frozen_hashes_and_hash_guard` (`eval/val_queries_b_human.validated.json`)<br>3) `test_step5c_cir_results_integrity` (`reports/phase5c/step5c_cir_results.json`)<br>4) `test_cir_eval_integrity` (`eval/cir_eval_val.json`) | Cả 3 test fail ở chế độ CRLF đều chuyển sang **PASSED**. Ngược lại, 4 tệp JSON này vốn được các script Python trên Windows tạo bằng `open(..., "w")` (sinh **CRLF `\r\n`**) rồi tính `.sha256` trên byte CRLF trước khi commit. Khi Git commit, Git chuẩn hóa về LF trong kho đối tượng; nên khi checkout với `autocrlf=false` (LF), byte thô là LF làm lệch SHA-256 CRLF. |
| **Hợp nhất Logic (`LF ∪ CRLF` / Working tree gốc tại thời điểm khóa)** | **94 / 94 PASSED (100% logic)** | **0 lỗi logic** | Trên working tree gốc (nơi mỗi file giữ nguyên định dạng LF hoặc CRLF lúc tính `.sha256`), toàn bộ **94/94 test được git theo dõi đều PASS**. Việc checkout mới từ git bị lệch 3 hoặc 4 test hash là hệ quả của việc thiếu `.gitattributes` cố định `eol` cho từng file khi dùng raw-byte SHA-256. |

---

## A.2 & C.11. Nguồn gốc chênh lệch `105/105 PASS` (Pha 6) vs `94` (commit `0c8e49e`) vs `107` (working tree WIP)
- Bằng chứng từ `git ls-tree 0c8e49e tests/` và `git status --short`:
  - Tại commit đóng băng `0c8e49e`, git chỉ theo dõi **94 test** (23 tệp trong `tests/`).
  - Tệp `?? tests/test_phase6_app.py` ở trạng thái **chưa commit (untracked)** trên working tree chứa **13 test**:
    - **11 test đầu (dòng 32–184):** kiểm thử ứng dụng Pha 6 (`94 + 11 = 105 test`). Con số **"105/105 PASS"** trong báo cáo Pha 6/Pha 7 chỉ tái lập được khi tính cả 11 test Pha 6 trong tệp `tests/test_phase6_app.py` chưa được commit vào git (trước khi nhánh WIP thêm 2 test cuối và sửa `product_card.py`).
    - **2 test cuối (dòng 203–233):** `test_cir_v1_06_domain_consistency` và `test_cfg_v5_10_domain_guard` thuộc nhánh WIP Pha 6+, đưa tổng số test trên working tree hiện tại lên **107** (`106 passed, 1 failed`).
- **Kết luận kiểm định (Mục C.11):** Đây là **phát hiện về quy trình đóng băng / quản lý phiên bản (`git commit`)** (quên `git add tests/test_phase6_app.py` ở cuối Pha 6 và thiếu `.gitattributes`), **không phải lỗi sai lệch số liệu đánh giá mô hình**.

---

## A.3. Truy vết đường dẫn `held-out v2` (`config.yaml` dòng 210 vs thực tế)
- Trong `config.yaml` dòng 210: `heldout_queries_v2: "eval/heldout_queries_v2.json"`. Tệp thực tế trên đĩa: `eval/test_queries_heldout_v2.json` (`SHA-256: 96c75433...`).
- Rà soát toàn bộ mã nguồn (`src/`, `scripts/`, `app/`, `tests/`): mọi script đánh giá và kiểm toán (`run_prompt14_final_eval_parts_bc.py`, `replay_rankings_final.py`, `compute_prompt14h_supplements.py`, `run_prompt14d_audit_and_report.py`, `run_prompt14e_part_a.py`, `run_prompt14f_part_a2.py`, `run_prompt14_final_eval.py`) đều trỏ trực tiếp hoặc ưu tiên `"eval/test_queries_heldout_v2.json"`. Khóa ở dòng 210 `config.yaml` là dead config, **không gây lỗi và không làm đọc nhầm tập dữ liệu**.

---

## A.4. Tiến trình đang dùng GPU và nhiệt độ nền (`nvidia-smi`)
- **Thông số GPU nền:** `NVIDIA GeForce RTX 2050` (WDDM, Driver `560.76`), nhiệt độ nền **57°C** (P5, `4W / 35W`), tải GPU nền **28%**, VRAM nền **220 MiB / 4096 MiB**.
- **Danh sách tiến trình WDDM `C+G`:** `CrossDeviceResume.exe` (12084), `SearchHost.exe` (12848), `StartMenuExperienceHost.exe` (12868), `WidgetBoard.exe` (13196), `msedgewebview2.exe` (14388), `Antigravity.exe` (15948, 21900), `D:\LDPlayer\ldmutiplayer\dnmultiplayerex.exe` (16192), `TextInputHost.exe` (17232), `ShellHost.exe` (20608).
