# BÁO CÁO ĐÁNH GIÁ CHÍNH THỨC TRÊN `index/test_gallery/` & ĐỘ TRỄ 4 LUỒNG (PROMPT 14 / 14b / 14c)

- **Trạng thái hiện tại:** `HALTED_PROMPT14C_STEP2_GPU_PROCESSES_STILL_ACTIVE`
- **Bản lưu các lần dừng trước (giữ nguyên không xóa):**
  - `HALTED_PREFLIGHT` (Prompt 14): `reports/phase5c/final_eval_report_halted_preflight.md` & `reports/phase5c/final_eval_results_halted_preflight.json`
  - `HALTED_PROMPT14B_STEP1_AND_STEP2` (Prompt 14b): `reports/phase5c/final_eval_report_halted_prompt14b.md` & `reports/phase5c/final_eval_results_halted_prompt14b.json`
- **Branch:** `phase7a`
- **Commit khởi điểm Prompt 14c:** `1d30dd62111d49ad78e8e4cf3f44705abbf61462`
- **Seed cố định:** `42`

---

## 1. Kết quả Bước 0 — Kiểm tra điều kiện & Khóa định nghĩa (`PASS` phần hạ tầng, thiếu `heldout_v2`)

### 1.1 Đối chiếu SHA-256 các tệp khóa (Mục 0.1 & Ràng buộc bất biến)

| Tệp khóa | SHA-256 thực tế | Khớp kỳ vọng |
|---|---|---|
| `data/checkpoints_siglip-vit/best_model.pt` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | **PASS** |
| `reports/phase5c/selection_rule_v5.json` | `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | **PASS** |
| `reports/phase5c/selection_rule_cir_v1.json` | `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067` | **PASS** |
| `eval/cir_eval_test.json` (chỉ kiểm hash, không mở) | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | **PASS** |
| `eval/test_split.json` (`1,290` mục test) | `d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681` | **PASS** |
| `data/catalog/catalog_rebuild.parquet` | `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` | **PASS** |
| `src/eval/locked_final_definitions.py` | `d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7` | **PASS** |
| `scripts/build_val_text_index_v5.py` (không sửa) | `da87e18c0027c769a072d99d59fe47e7c5e2fc9a71f1a4116c9280d232d2e867` | **PASS** |
| `eval/test_queries_heldout_v2.json` | *(Chưa có tệp — chỉ chạy Bước 1, Bước 2, Phần A rồi dừng)* | **MISSING** |

### 1.2 Kiểm định tiền xử lý Luật 4 trên `index/test_gallery/` (Mục 0.3)
- Tái mã hóa ngẫu nhiên `200` ảnh (`seed = 42`) từ `index/test_gallery/` qua `SigLIPPreprocessor`:
  - `min_cosine = 0.999978 >= 0.999` (**PASS**, khớp đối chứng Pha 3 `0.999660`).

---

## 2. Kết quả Bước 1 (Prompt 14c — Phương án 1B): Dựng `index/test_gallery_text_v5/` (`HOÀN TẤT — PASS`)

Theo quyết định của người dùng ở Prompt 14c (Phương án 1B):
- Giữ nguyên không sửa `scripts/build_val_text_index_v5.py` (`SHA-256: da87e18c0027c769a072d99d59fe47e7c5e2fc9a71f1a4116c9280d232d2e867`).
- Tạo script mới `scripts/build_test_text_index_v5.py` (có `argparse`) import trực tiếp `_encode_batch_fast`, `_sha256_file`, `_get_git_commit` từ `scripts.build_val_text_index_v5` và sử dụng cùng `SiglipProcessor(max_length=128, padding="max_length")` + tokenizer của `ECommerceSearchEngine`.

### 2.1 Kiểm chứng bắt buộc trên `val`: `index/_verify_val_text_v5/` so với `index/val_gallery_text_v5/`
- Quá trình kiểm chứng ban đầu phát hiện nếu dùng glob `shard_val_*.parquet` sẽ quét nhầm thêm `shard_val_core_0000..0003.parquet` (các shard chỉ có `caption_A`, làm `caption_B` bị ghi đè). Sau khi dùng đúng glob `shard_val_[0-9]*.parquet` như dòng 53 của `scripts/build_val_text_index_v5.py`, kết quả đối chiếu từng hàng (`1,884` hàng) giữa `index/_verify_val_text_v5/` và `index/val_gallery_text_v5/` khớp tuyệt đối tới từng byte:

| Tệp `.npy` (`1,884 × 1,152`, `float32`) | `min_cosine` | `mean_cosine` | `max_abs_diff` | Số hàng `cos < 0.9999` | SHA-256 (`ref` == `cand`) | Kết luận |
|---|---:|---:|---:|---:|---|---|
| `internvl2_only_embeddings.npy` | **0.99999988** | **1.00000000** | `0.0` | `0` | `a1dd96ab2efa293e0748f02934148a547522e94c8124714e1b6f631a1e89c5a2` | **PASS (`≥ 0.9999`, exact byte match)** |
| `internvl2_plus_aux_embeddings.npy` | **0.99999976** | **1.00000000** | `0.0` | `0` | `ea0df85b52dc69d655b1230c55642bfa475f462c5fff1f56c44b1e1481a50f63` | **PASS (`≥ 0.9999`, exact byte match)** |
| `metadata_aux_embeddings.npy` | **0.99999982** | **1.00000000** | `0.0` | `0` | `faf6880e9d3bb7b01544c91b6888e8afc07ea770f964659bcc99c632ffe7b04b` | **PASS (`≥ 0.9999`, exact byte match)** |

### 2.2 Kết quả dựng `index/test_gallery_text_v5/`, Hash Guard (Luật 6) & Sanity Check
- Nguồn dữ liệu: `data/catalog/captions/shard_test_0000..0002.parquet` (`1,500` dòng tổng, khớp `1,290 / 1,290` mục của `index/test_gallery/id_map.json`):
  - `shard_test_0000.parquet`: `9cf154b9c1f8f33adadb209fb234ac731a03b685c7988a42c8ffda295581d690`
  - `shard_test_0001.parquet`: `32db89eebeb87982714add0af4be9126644a77e3f4502698da5044e03c10c03c`
  - `shard_test_0002.parquet`: `e888366e043a5ba13d751669ddcdd032925eeb29073d7a0e08d92c8b4d9af7c2`
- `index/test_gallery_text_v5/manifest.json` (`SHA-256: 286fe59ed6c2348dd9815263085269fd688993c83e1ede91d9152344184a2477`):
  - `checkpoint_hash`: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` (**PASS Hash Guard**)
  - `catalog_hash`: `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` (**PASS Hash Guard**)
  - `n_vectors`: `1290`, `0` ID trùng, thứ tự `id_order` khớp 100% với `index/test_gallery/id_map.json` (`id_order_sha256: 6ca5bbf0b268103da2b901538effc004057faa817e7473d96dd49395418ee282`).
  - Provenance (Luật 5 & D.1): `caption_model = "OpenGVLab/InternVL2-2B + 5CD-AI/Vintern-1B-v3_5"`, `model_revision = "e4f6747bd20f139e637642c6a058c6bd00b36919 + b98f263eab246eb5269ade64edbdca8a887dc44d"`, `combination_method = "merge_v1"`, `generation_params = {"max_new_tokens": 96, "do_sample": false, "num_beams": 1, "quantization": "8bit"}`, `seed = 42`.

| Tệp `.npy` trong `index/test_gallery_text_v5/` | Shape & Dtype | Số phần tử NaN/Inf | Chuẩn L2 (`min / mean / max`) | SHA-256 |
|---|---|---:|---|---|
| `internvl2_only_embeddings.npy` | `(1290, 1152)`, `float32` | `0` | `0.999605 / 1.000035 / 1.000364` | `7d4344b33194cc01b8a211b978e82357ef7b04ec65f3a435895e8b4efb58aaf3` |
| `internvl2_plus_aux_embeddings.npy` | `(1290, 1152)`, `float32` | `0` | `0.999523 / 1.000003 / 1.000471` | `53c75ab8a110a0053fc5e1d74d8a0407c14aebe2f9744d4f65f104fb1084d450` |
| `metadata_aux_embeddings.npy` | `(1290, 1152)`, `float32` | `0` | `0.999522 / 1.000008 / 1.000470` | `0339efc1e3799fe2d449b92c412f7a06742190b44f9d716bbf9b2f30c5eb101c` |

### 2.3 So sánh phân phối `token_length` và tỉ lệ bị cắt ở `max_length = 128` giữa `val` (`max_new_tokens=48`) và `test` (`max_new_tokens=96`)

Đo trực tiếp bằng tokenizer SigLIP (`truncation=False`) trên toàn bộ `1,884` mục `val_gallery` và `1,290` mục `test_gallery`:

| Trường văn bản | Split (`gen_params`) | `N` | `min` | `p25` | `median` | `mean` | `p95` | `p99` | `max` | Số mục `> 128` (bị cắt bởi `SiglipProcessor`) | Số mục `= 128` (chạm trần ngân sách `merge_v1`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`caption_A`** (`internvl2_only`) | `val` (`max_new_tokens=48`) | `1,884` | `49` | `68.0` | `68.0` | `67.91` | `68.0` | `68.0` | `68` | **`0` (`0.00%`)** | `0` (`0.00%`) |
| **`caption_A`** (`internvl2_only`) | `test` (`max_new_tokens=96`) | `1,290` | `4` | `6.0` | `68.0` | `49.31` | `68.0` | `123.11` | `131` | **`2` (`0.155%`)** | `1` (`0.078%`) |
| **`caption_B`** (`internvl2_plus_aux` = `merge_v1`) | `val` (`max_new_tokens=48`) | `1,884` | `97` | `127.0` | `128.0` | `127.39` | `128.0` | `128.0` | `128` | **`0` (`0.00%`)** | `1,055` (`56.00%`) |
| **`caption_B`** (`internvl2_plus_aux` = `merge_v1`) | `test` (`max_new_tokens=96`) | `1,290` | `111` | `127.0` | `128.0` | `127.37` | `128.0` | `128.0` | `128` | **`0` (`0.00%`)** | `688` (`53.33%`) |
| **`metadata_aux`** (`title + 3 slot`) | `val` (`id_map.json` có `*_vi`) | `1,884` | `9` | `19.0` | `22.0` | `22.36` | `30.0` | `33.0` | `36` | **`0` (`0.00%`)** | `0` (`0.00%`) |
| **`metadata_aux`** (`title + 3 slot`) | `test` (`id_map.json` chỉ có tiếng Anh) | `1,290` | `5` | `5.0` | `7.0` | `7.45` | `13.0` | `13.0` | `15` | **`0` (`0.00%`)** | `0` (`0.00%`) |

**Nhận xét kỹ thuật trung thực (Luật 5 & Luật 7):**
1. **`caption_B` (`internvl2_plus_aux_embeddings.npy`, dùng cho cấu hình tham khảo `cfg_v5_08b`):** Cả `val` (`mean = 127.39`, `median = 128.0`) và `test` (`mean = 127.37`, `median = 128.0`) đều có **0 mục vượt quá 128 token (`0.00%` bị cắt mất hậu tố bởi `SiglipProcessor`)** vì hàm hợp nhất `merge_v1` đã chủ động giới hạn độ dài chuỗi hợp nhất ở đúng `<= 128` token theo Luật D.3 (đặt thuộc tính phân biệt `loại + giới + màu` ở đầu câu).
2. **`caption_A` (`internvl2_only_embeddings.npy`):**
   - Trên `test` (`max_new_tokens=96`), chỉ có `2 / 1,290` mục (`0.155%`) vượt quá `128` token (`max = 131` token).
   - Tuy nhiên, trên `shard_test_0000..0002.parquet` (sinh ngày 12–13/11/2026 ở Pha 2 trước khi sửa bộ tách câu cho `shard_val`), có `393 / 1,290` mục (`30.47%`) bị bộ tách câu cắt ngắn `caption_A` ở dấu chấm sau số thứ tự (ví dụ `"1. Thô"`, dài `4–6` token, làm `p25 = 6.0` và `mean = 49.31` dù `median = 68.0`). Ở `caption_B` (`merge_v1`), phần mô tả Vintern + prefix thuộc tính bù đắp nên 100% mục `test` vẫn có độ dài `111–128` token.
3. **`metadata_aux` (`metadata_aux_embeddings.npy`):**
   - Trong `index/val_gallery/id_map.json`, mỗi mục lưu sẵn `articleType_vi`, `gender_vi`, `baseColour_vi` (tiếng Việt có dấu, khi qua SentencePiece của SigLIP tách thành trung vị `22` subword token).
   - Trong `index/test_gallery/id_map.json` (tệp khóa từ Pha 4 theo Luật 3), mỗi mục chỉ lưu `articleType`, `gender`, `baseColour` bằng tiếng Anh (`"Shirts Shirts Men Navy Blue"`), nên hàm tạo `texts_meta` của `build_val_text_index_v5.py` rơi về nhánh fallback tiếng Anh (`m.get("articleType_vi") or m.get("articleType")`), có trung vị `7.0` token (`0%` bị cắt).

---

## 3. Kết quả Bước 2 (Prompt 14c): Kiểm tra lại môi trường GPU (`HALT — CÒN TIẾN TRÌNH GIẢ LẬP/GAME TRÊN GPU`)

Thực hiện đúng Bước 2 của Prompt 14c lúc `2026-11-14 21:27:45` (khi agent **chưa nạp model**, PID của tiến trình Python kiểm tra là `PID 988`):

### 3.1 Bảng tiến trình GPU từ `nvidia-smi` lúc `21:27:45`

| GPU | PID | Loại | Tên tiến trình (`nvidia-smi` / `Win32_Process`) | Thời điểm khởi tạo (`CreationDate`) | Phân loại theo Bước 2 |
|---:|---:|---|---|---|---|
| 0 | `1596` | `C+G` | `...\Programs\antigravity\Antigravity.exe` | Tiến trình IDE | Cho phép |
| 0 | `2912` | `C+G` | `...\Programs\antigravity\Antigravity.exe` | Tiến trình IDE | Cho phép |
| 0 | **`3588`** | **`C+G`** | **`C:\Program Files\WindowsApps\Microsoft.Edge.GameAssist_...\EdgeGameAssist.exe`** | **`11/14/2026 6:37:24 PM` (`ParentPID: 1540`)** | **CẤM (Tiến trình Game Assist)** |
| 0 | `4724` | `C+G` | `C:\Windows\System32\ShellHost.exe` | Tiến trình hệ thống Windows | Cho phép |
| 0 | `12004` | `C+G` | `...\CrossDeviceResume.exe` | Tiến trình hệ thống Windows | Cho phép |
| 0 | `12964` | `C+G` | `...\SearchHost.exe` | Tiến trình hệ thống Windows | Cho phép |
| 0 | `12976` | `C+G` | `...\StartMenuExperienceHost.exe` | Tiến trình hệ thống Windows | Cho phép |
| 0 | `13176` | `C+G` | `...\WidgetBoard.exe` | Tiến trình hệ thống Windows | Cho phép |
| 0 | `13356` | `C+G` | `...\msedgewebview2.exe` | WebView2 nền | Nên tắt cùng `EdgeGameAssist` |
| 0 | **`16284`** | **`C+G`** | **`D:\LDPlayer\ldmutiplayer\dnmultiplayerex.exe`** | **`11/14/2026 3:22:21 PM` (`ParentPID: 4840`)** | **CẤM (Trình giả lập LDMultiPlayer)** |
| 0 | `17140` | `C+G` | `...\TextInputHost.exe` | Tiến trình hệ thống Windows | Cho phép |
| 0 | `21960` | `C+G` | `...\PhoneExperienceHost.exe` | Tiến trình hệ thống Windows | Cho phép |

### 3.2 Mẫu đo `GPU-Util` nghỉ 30 giây (`21:27:45` – `21:28:18`, 30 mẫu cách nhau 1 giây, agent không nạp model)
- **GPU-Util (%):** `median = 31.0%`, `mean = 31.97%`, `min = 12.0%`, `max = 59.0%`.
- **Nhiệt độ & Trạng thái GPU:** `Temp = 71.0–72.0°C` (`median = 71.0°C`), P-state `P5`, công suất `3.67–4.32 W / 35 W`, VRAM sử dụng `332 MiB / 4096 MiB`.
- Chi tiết toàn bộ 30 mẫu được lưu tại `reports/phase5c/step2_gpu_env_check_prompt14c.json`.

### 3.3 Nguyên nhân vì sao `dnmultiplayerex.exe` và `EdgeGameAssist.exe` vẫn còn sau khi đóng cửa sổ & Cách tắt dứt điểm
1. **`dnmultiplayerex.exe` (`PID 16284`, `D:\LDPlayer\ldmutiplayer\dnmultiplayerex.exe`):** Khởi tạo từ `15:22:21` chiều nay. Khi bấm nút `[X]` trên cửa sổ LDMultiPlayer, ứng dụng mặc định thu nhỏ xuống **khay hệ thống (System Tray)** ở góc phải dưới thanh Taskbar (hoặc chạy với quyền Administrator) nên tiến trình `PID 16284` không thoát.
2. **`EdgeGameAssist.exe` (`PID 3588`):** Là dịch vụ overlay Game Bar của Microsoft Edge (`ParentPID: 1540` — `svchost.exe`), tự giữ tiến trình nền ngay cả khi đã đóng cửa sổ.
3. **Tuân thủ nghiêm ngặt quy tắc Bước 2 của Prompt 14c:**
   > *"Nếu còn `dnmultiplayerex.exe`, `EdgeGameAssist.exe`, `RobloxPlayerBeta.exe` hoặc tiến trình game/giả lập/đồ họa nặng khác: DỪNG, báo, KHÔNG tự kill."*
   Do đó, agent **dừng ngay trước Phần A** và không tự gọi `taskkill`.

**Hướng dẫn tắt dứt điểm 2 tiến trình này (người dùng chỉ cần chạy 1 lệnh trong PowerShell quyền Administrator):**
```powershell
Stop-Process -Id 16284, 3588, 13356 -Force
```
*(Hoặc mở **Task Manager** $\to$ tab **Details (Chi tiết)** $\to$ tìm `dnmultiplayerex.exe` (`PID 16284`) và `EdgeGameAssist.exe` (`PID 3588`) $\to$ chuột phải chọn **End process tree / Kết thúc cây tiến trình**, sau đó kiểm tra lại bằng `nvidia-smi` để chắc chắn hai PID này đã biến mất).*

---

## 4. Kiểm thử tự động (`pytest`)
- Chạy `pytest tests/test_build_test_text_index_v5.py -q`: **`3 passed`**.
- Toàn bộ bộ kiểm thử hiện tại sẽ được chạy và ghi nhận trước khi commit.
