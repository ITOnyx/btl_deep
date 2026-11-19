# Bổ sung Kiểm định Pha 1 (Mục A.1 – A.4)

- **Thời điểm cập nhật (UTC):** `2026-11-19T14:35:20Z`
- **Commit đối chiếu:** `0c8e49e2afe14328bcd272fa2a50e184389702ea`
- **Bằng chứng:**
  - `reports/verification/phase_1/pytest_clean_head_stdout.txt`
  - `reports/verification/phase_1/pytest_clean_head_with_data_index_stdout.txt`

---

## A.1. Kiểm chứng `pytest` trên bản checkout sạch của commit `0c8e49e` (`git worktree`)
- **Quy trình:** Tạo worktree tách biệt bằng `git worktree add --detach "reports/verification/_clean_head" 0c8e49e2afe14328bcd272fa2a50e184389702ea`, chạy `pytest -v`, sau đó gỡ sạch bằng `git worktree remove "reports/verification/_clean_head"` (không đụng working tree hiện tại, không xóa thủ công).
- **Kết quả chạy 1 — Worktree `0c8e49e` thuần (không có `data/` và `index/` do nằm trong `.gitignore`):**
  - Thu thập: **94 tests** (vì `tests/test_phase6_app.py` gồm 13 test là file untracked `??`).
  - `tests/test_app_structure.py::test_app_imports`: **PASSED** (xác nhận 100% bằng thực nghiệm rằng lỗi FAIL ở working tree hiện tại là do chỉnh sửa WIP `M app/components/product_card.py` đổi tên `render_product_grid` thành `render_result_grid` + `render_result_card`).
  - Tổng: **78 passed, 9 failed, 5 skipped, 2 errors** (các lỗi/skip do thiếu thư mục gitignored `data/` và `index/`).
- **Kết quả chạy 2 — Worktree `0c8e49e` kèm junction chỉ-đọc (`mklink /J`) tới `data/` và `index/`:**
  - Thu thập: **94 tests**.
  - `tests/test_app_structure.py::test_app_imports`: **PASSED**.
  - Tổng: **91 passed, 3 failed, 0 skipped**. Ba test fail (`test_build_val_text_index_v5_untouched`, `test_p3_frozen_hashes_and_hash_guard`, `test_selection_rule_cir_v1_locked`) do kiểm tra SHA-256 byte thô của 3 file text (`scripts/build_val_text_index_v5.py`, `src/serve/parser_v1.py`, `reports/phase5c/selection_rule_cir_v1.json`): trên Windows, `git worktree add` với `core.autocrlf=true` chuyển LF sang CRLF làm đổi SHA-256 byte thô, trong khi bản working tree gốc lưu LF nên PASS cả 3 test này.

---

## A.2. Nguồn gốc chênh lệch 2 test (107 so với 105)
- Tại commit `0c8e49e`, git chỉ theo dõi **94 test** trong `tests/`.
- Tệp `?? tests/test_phase6_app.py` (chưa commit, untracked) chứa tổng cộng **13 test**:
  - **11 test đầu (dòng 32–184):** bộ kiểm thử ứng dụng Pha 6 (`94 + 11 = 105 test` như báo cáo Pha 6 ghi nhận).
  - **2 test cuối (dòng 203–233):**
    1. `tests/test_phase6_app.py::test_cir_v1_06_domain_consistency` (dòng 203)
    2. `tests/test_phase6_app.py::test_cfg_v5_10_domain_guard` (dòng 219)
  - Hai test này thuộc nhánh WIP Pha 6+ (`cir_v1_06` và `cfg_v5_10`), làm tổng số test trên working tree tăng từ **105 lên 107**.

---

## A.3. Truy vết đường dẫn `held-out v2` (`config.yaml` dòng 210 vs thực tế)
- Trong `config.yaml` dòng 210: `heldout_queries_v2: "eval/heldout_queries_v2.json"`.
- Tệp thực tế trên đĩa: `eval/test_queries_heldout_v2.json` (`SHA-256: 96c75433...`).
- **Kết quả rà soát toàn bộ mã nguồn (`src/`, `scripts/`, `app/`, `tests/`):**
  - **Không có module nào** đọc `cfg["eval"]["heldout_queries_v2"]` mà không có cơ chế ưu tiên file thật:
    - `scripts/run_prompt14_final_eval_parts_bc.py` (dòng 67, 246), `scripts/replay_rankings_final.py` (dòng 104), `scripts/compute_prompt14h_supplements.py` (dòng 83), `scripts/run_prompt14d_audit_and_report.py` (dòng 115), `scripts/run_prompt14e_part_a.py` (dòng 1923), `scripts/run_prompt14f_part_a2.py` (dòng 688), `scripts/finalize_heldout_v2.py` (dòng 12) đều trỏ trực tiếp tới `"eval/test_queries_heldout_v2.json"`.
    - `scripts/run_prompt14_final_eval.py` (dòng 226–228) kiểm tra `eval/test_queries_heldout_v2.json` trước, chỉ fallback sang `eval/heldout_queries_v2.json` nếu file chính không tồn tại.
    - Script WIP `scripts/run_heldout_eval.py` tự sinh 50 câu T2I + 50 cặp CIR trong bộ nhớ (`seed=99`), không đọc khóa này.
  - **Kết luận:** Dòng 210 trong `config.yaml` là khóa cấu hình chết (dead config), **không gây lỗi và không làm đọc nhầm tập dữ liệu** ở bất kỳ kịch bản đánh giá nào.

---

## A.4. Tiến trình đang dùng GPU và nhiệt độ nền (`nvidia-smi`)
- **Thông số GPU nền (`2026-11-19 21:32:12+07:00`):**
  - GPU: `NVIDIA GeForce RTX 2050` (WDDM, Driver `560.76`, CUDA `12.6`)
  - Nhiệt độ nền: **57°C** (P5 state, `4W / 35W`)
  - Tải GPU nền (`utilization.gpu`): **28%** (`utilization.memory`: `2%`, VRAM đang dùng: **220 MiB / 4096 MiB**)
- **Danh sách tiến trình (`nvidia-smi` / `--query-compute-apps`, chế độ WDDM `C+G`):**
  1. `PID 12084`: `C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\CrossDeviceResume.exe`
  2. `PID 12848`: `C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe`
  3. `PID 12868`: `C:\Windows\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe`
  4. `PID 13196`: `C:\Program Files\WindowsApps\MicrosoftWindows.Client.WebExperience_526.21100.40.0_x64__cw5n1h2txyewy\WidgetBoard.exe`
  5. `PID 14388`: `C:\Program Files (x86)\Microsoft\EdgeWebView\Application\157.0.4341.46\msedgewebview2.exe`
  6. `PID 15948` & `PID 21900`: `C:\Users\nguye\AppData\Local\Programs\antigravity\Antigravity.exe`
  7. `PID 16192`: `D:\LDPlayer\ldmutiplayer\dnmultiplayerex.exe`
  8. `PID 17232`: `C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TextInputHost.exe`
  9. `PID 20608`: `C:\Windows\System32\ShellHost.exe`
  10. `PID 1776`: `[Insufficient Permissions]` (System/DWM)
