# Báo cáo Kiểm định Prompt 14b — Điểm dừng Bước 1 (Script Hardcode) & Bước 2 (GPU Chưa Sạch)

- **Trạng thái thực thi:** `HALTED_PROMPT14B_STEP1_AND_STEP2`
- **Bản lưu `HALTED_PREFLIGHT` trước đó (đã đổi tên bảo toàn theo đúng yêu cầu):**
  - [final_eval_report_halted_preflight.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_report_halted_preflight.md)
  - [final_eval_results_halted_preflight.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_results_halted_preflight.json)
- **Tệp kết quả JSON hiện hành:** [final_eval_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_results.json)
- **Tệp định nghĩa chỉ số đã khóa (giữ nguyên):** [locked_final_definitions.py](file:///d:/Kì%207/Hoc_sau_ver4/src/eval/locked_final_definitions.py) (`SHA-256: d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7`)

---

## 1. Kiểm tra Bước 1 (`index/test_gallery_text_v5/`) — Điểm dừng do Script Hardcode

Yêu cầu của **Bước 1** trong Prompt 14b:
> *"Dùng ĐÚNG script, tokenizer, tham số và đường tiền xử lý text (`max_length=128`, `padding="max_length"`) đã dùng để dựng `index/val_gallery_text_v5/`. **Không sửa script. Nếu phải sửa thì DỪNG và báo.**"*

Kiểm tra trực tiếp [scripts/build_val_text_index_v5.py](file:///d:/Kì%207/Hoc_sau_ver4/scripts/build_val_text_index_v5.py#L47-L132) cho thấy:
1. Hàm `main()` **hardcode cố định** các đường dẫn của `val_gallery` và **không nhận tham số dòng lệnh (`argparse`)**:
   - Dòng 48: `val_dir = PROJECT_ROOT / "index/val_gallery"`
   - Dòng 49: `out_dir = PROJECT_ROOT / "index/val_gallery_text_v5"`
   - Dòng 53: `shard_files = sorted(captions_dir.glob("shard_val_[0-9]*.parquet"))`
   - Dòng 114: `"gallery_type": "val_tuning_split_text_v5"`
2. Ngoài ra, `manifest.json` sinh bởi `build_val_text_index_v5.py` hiện tại chưa ghi các trường mở rộng mà Bước 1 yêu cầu bổ sung (`hash từng shard caption`, `thứ tự id`, `caption_model`, `model_revision`, `phương pháp kết hợp merge_v1`), cũng như đoạn kiểm tra đối chứng phân bố độ dài token & tỷ lệ cắt giữa `val_gallery_text_v5` và `test_gallery_text_v5`.
3. **Tuân thủ tuyệt đối quy tắc Bước 1 (*"Không sửa script. Nếu phải sửa thì DỪNG và báo"*):** Agent không tự ý sửa `scripts/build_val_text_index_v5.py` hay tạo script mới mà dừng lại báo cáo để người dùng duyệt **1 trong 2 phương án**:
   - **Phương án 1A:** Cho phép bổ sung tham số CLI (`--gallery-dir`, `--out-dir`, `--shard-glob`, `--verify-both`) và các trường manifest mới vào `scripts/build_val_text_index_v5.py` (giữ nguyên 100% hàm `_encode_batch_fast` và đường tiền xử lý `SiglipProcessor(max_length=128, padding="max_length")`).
   - **Phương án 1B:** Giữ nguyên tuyệt đối `scripts/build_val_text_index_v5.py` (không sửa 1 byte), và tạo `scripts/build_test_text_index_v5.py` import/dùng chung đúng hàm `_encode_batch_fast` + `ECommerceSearchEngine` để xuất `index/test_gallery_text_v5/` và kiểm định đồng thời cả `val_gallery_text_v5` lẫn `test_gallery_text_v5`.

---

## 2. Kiểm tra Bước 2 (Kiểm tra GPU Sạch & Lấy mẫu `GPU-Util` Nghỉ 30 Giây) — Điểm dừng Bắt buộc

Yêu cầu của **Bước 2** trong Prompt 14b:
> - *"Chạy `nvidia-smi`, ghi toàn bộ tiến trình. Ghi riêng PID của tiến trình Python do chính agent chạy (loại khỏi phán định 'tiến trình nền', nhưng phải ghi lại)."*
> - *"Nếu còn `dnmultiplayerex.exe`, `EdgeGameAssist.exe`, `RobloxPlayerBeta.exe` hoặc tiến trình game/giả lập/đồ họa nặng khác: **DỪNG, báo người dùng, KHÔNG tự kill.**"*
> - *"Lấy mẫu `GPU-Util` nghỉ mỗi 1 giây trong 30 giây khi agent không chạy gì; ghi trung vị và max. **Nếu trung vị > 10% thì báo là ứng viên gây nhiễu và DỪNG hỏi người dùng.**"*

### 2.1. Danh sách toàn bộ tiến trình trên GPU (`nvidia-smi` lúc `20:57:46`)
- **PID tiến trình Python do chính Agent chạy (chỉ đọc `nvidia-smi`, không khởi tạo CUDA):** `PID 15456` (không xuất hiện trong bảng `nvidia-smi` vì không mở CUDA context).
- **Bảng toàn bộ tiến trình trên GPU (`NVIDIA GeForce RTX 2050`):**

| PID | Loại | Tên tiến trình | Phân loại theo Bước 2 |
|---:|---|---|---|
| **`16284`** | `C+G` | `...er\ldmutiplayer\dnmultiplayerex.exe` | ❌ **CÒN SỐNG — Tiến trình giả lập nền (chạy ngầm ở System Tray)** |
| **`3588`** | `C+G` | `...4__8wekyb3d8bbwe\EdgeGameAssist.exe` | ❌ **CÒN SỐNG — Tiến trình Game Assist nền** |
| `1596` | `C+G` | `...rograms\antigravity\Antigravity.exe` | ✅ Chấp nhận (IDE) |
| `2912` | `C+G` | `...rograms\antigravity\Antigravity.exe` | ✅ Chấp nhận (IDE) |
| `4724` | `C+G` | `C:\Windows\System32\ShellHost.exe` | ✅ Chấp nhận (Windows) |
| `12004` | `C+G` | `...cw5n1h2txyewy\CrossDeviceResume.exe` | ✅ Chấp nhận (Windows) |
| `12964` | `C+G` | `...nt.CBS_cw5n1h2txyewy\SearchHost.exe` | ✅ Chấp nhận (Windows) |
| `12976` | `C+G` | `...2txyewy\StartMenuExperienceHost.exe` | ✅ Chấp nhận (Windows) |
| `13176` | `C+G` | `..._x64__cw5n1h2txyewy\WidgetBoard.exe` | ✅ Chấp nhận (Windows) |
| `13356` | `C+G` | `...on\157.0.4341.46\msedgewebview2.exe` | ✅ Chấp nhận (Windows/WebView) |
| `17140` | `C+G` | `...CBS_cw5n1h2txyewy\TextInputHost.exe` | ✅ Chấp nhận (Windows) |

> [!WARNING]
> **Nguyên nhân 2 tiến trình vẫn còn trên GPU:** Trên Windows, khi đóng cửa sổ `LDPlayer Multi-Player` (`dnmultiplayerex.exe`, `PID 16284`) và `EdgeGameAssist.exe` (`PID 3588`), ứng dụng chỉ thu nhỏ xuống khay hệ thống (System Tray) hoặc giữ tiến trình nền mà chưa kết thúc hẳn.

### 2.2. Kết quả lấy mẫu `GPU-Util` nghỉ mỗi 1 giây trong 30 giây (khi Agent không chạy tác vụ GPU)
- **30 mẫu `GPU-Util (%)` liên tiếp (mỗi 1 giây):**
  `[28.0, 28.0, 28.0, 28.0, 28.0, 55.0, 28.0, 30.0, 28.0, 28.0, 28.0, 29.0, 28.0, 28.0, 29.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 29.0, 28.0, 55.0]`
- **Trung vị (`median`) `GPU-Util` nghỉ:** **`28.0%`** ❌ *(vượt ngưỡng `> 10%` của Bước 2)*
- **Cực đại (`max`) `GPU-Util` nghỉ:** **`55.0%`** (`min = 28.0%`, `mean = 29.97%`)
- **Nhiệt độ GPU nghỉ trong 30 giây:** Trung vị **`56.0°C`**, max **`56.0°C`** (`< 60°C` ✅).

---

## 3. Hành động Cần Người Dùng Thực Hiện / Phê Duyệt Để Chạy Tiếp

1. **Về Bước 1 (`scripts/build_val_text_index_v5.py`):**
   - Vui lòng xác nhận chọn **Phương án 1A** (cho phép thêm tham số CLI + trường manifest vào `scripts/build_val_text_index_v5.py`) hoặc **Phương án 1B** (giữ nguyên `build_val_text_index_v5.py`, tạo `scripts/build_test_text_index_v5.py` dùng chung hàm `_encode_batch_fast`).
2. **Về Bước 2 (Tắt hẳn `dnmultiplayerex.exe` và `EdgeGameAssist.exe` khỏi tiến trình nền):**
   - Người dùng có thể mở PowerShell / CMD và chạy lệnh sau để buộc đóng hẳn 2 tiến trình ngầm này:
     ```powershell
     taskkill /F /IM dnmultiplayerex.exe /IM EdgeGameAssist.exe
     ```
     *(Hoặc chỉ định rõ bằng văn bản: **"Cho phép Agent chạy `taskkill` tắt `dnmultiplayerex.exe` và `EdgeGameAssist.exe`"** nếu muốn Agent thực hiện lệnh đó).*
   - Ngoài ra, nếu sau khi tắt 2 tiến trình trên mà `GPU-Util` nghỉ ở trạng thái `P5 / P8` (xung nhịp thấp `442 MHz`) của Windows Desktop/Antigravity vẫn dao động trên `10%`, vui lòng cho biết có cho phép tiếp tục đo Phần A khi đã tắt sạch `dnmultiplayerex.exe` và `EdgeGameAssist.exe` hay không.
