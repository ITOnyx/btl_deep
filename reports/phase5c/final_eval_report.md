# BÁO CÁO ĐÁNH GIÁ CHÍNH THỨC TRÊN `index/test_gallery/` & KIỂM TOÁN CAPTION TEST (PROMPT 14d)

- **Trạng thái hiện tại:** `HALTED_PROMPT14D_STEP2_GPU_PROCESSES_STILL_ACTIVE`
- **Branch:** `phase7a`
- **Commit khởi điểm:** `f22b57033e083bbf900ca05c7464187a17436993`
- **Seed cố định:** `42`
- **Thời điểm tạo báo cáo:** `2026-11-14 22:12:18 +07:00`
- **Các bản lưu dừng trước (giữ nguyên không xóa):**
  - Prompt 14 (`HALTED_PREFLIGHT`): `reports/phase5c/final_eval_report_halted_preflight.md` & `reports/phase5c/final_eval_results_halted_preflight.json`
  - Prompt 14b (`HALTED_PROMPT14B_STEP1_AND_STEP2`): `reports/phase5c/final_eval_report_halted_prompt14b.md` & `reports/phase5c/final_eval_results_halted_prompt14b.json`
  - Prompt 14c (`HALTED_PROMPT14C_STEP2_GPU_PROCESSES_STILL_ACTIVE`): `reports/phase5c/final_eval_report_halted_prompt14c.md` & `reports/phase5c/final_eval_results_halted_prompt14c.json`

---

## 1. Kết quả Bước 1a — Kiểm Hash & Hash Guard Luật 6 (`PASS`, không dựng lại)

### 1.1 Đối chiếu SHA-256 các tệp khóa đầu vào

| Khóa định danh | Đường dẫn tệp | SHA-256 thực tế | Khớp kỳ vọng |
|---|---|---|---|
| `best_model_pt` | `data/checkpoints_siglip-vit/best_model.pt` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | **PASS** |
| `selection_rule_v5` | `reports/phase5c/selection_rule_v5.json` | `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | **PASS** |
| `selection_rule_cir_v1` | `reports/phase5c/selection_rule_cir_v1.json` | `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067` | **PASS** |
| `cir_eval_test` | `eval/cir_eval_test.json` | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | **PASS** |
| `test_split` | `eval/test_split.json` | `d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681` | **PASS** |
| `catalog_rebuild` | `data/catalog/catalog_rebuild.parquet` | `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` | **PASS** |
| `locked_final_definitions_py` | `src/eval/locked_final_definitions.py` | `d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7` | **PASS** |
| `build_val_text_index_v5_py` | `scripts/build_val_text_index_v5.py` | `da87e18c0027c769a072d99d59fe47e7c5e2fc9a71f1a4116c9280d232d2e867` | **PASS** |
| `test_queries_heldout_v2` | `eval/test_queries_heldout_v2.json` | *(Chưa có tệp)* | **MISSING** |

### 1.2 Kiểm tra `index/test_gallery_text_v5/` (Đã dựng ở Prompt 14c — chỉ đọc & kiểm Hash Guard)

- Đường dẫn manifest: `index/test_gallery_text_v5/manifest.json`
- Manifest SHA-256: `286fe59ed6c2348dd9815263085269fd688993c83e1ede91d9152344184a2477` (khớp tiền tố `286fe59e`: `True`)
- Dựng lại hoặc chỉnh sửa trong Prompt 14d: `False`
- Hash Guard `checkpoint_hash` khớp `config.yaml` & `index/test_gallery/manifest.json`: `True`
- Hash Guard `catalog_hash` khớp `index/test_gallery/manifest.json`: `True`
- Số vector: `1290` (kỳ vọng: `1290`, khớp: `True`)
- Số ID trùng: `0` | Thứ tự `id_order` khớp `id_map.json`: `True` (`id_order_sha256`: `6ca5bbf0b268103da2b901538effc004057faa817e7473d96dd49395418ee282`)

| Khóa tệp `.npy` | Tên tệp | Shape | Dtype | Số NaN/Inf | L2 norm (`min / mean / max`) | SHA-256 thực tế | Khớp manifest |
|---|---|---|---|---:|---|---|---|
| `internvl2_only_embeddings` | `internvl2_only_embeddings.npy` | `(1290, 1152)` | `float32` | `0` | `0.999605 / 1.000035 / 1.000364` | `7d4344b33194cc01b8a211b978e82357ef7b04ec65f3a435895e8b4efb58aaf3` | `True` |
| `internvl2_plus_aux_embeddings` | `internvl2_plus_aux_embeddings.npy` | `(1290, 1152)` | `float32` | `0` | `0.999523 / 1.000003 / 1.000471` | `53c75ab8a110a0053fc5e1d74d8a0407c14aebe2f9744d4f65f104fb1084d450` | `True` |
| `metadata_aux_embeddings` | `metadata_aux_embeddings.npy` | `(1290, 1152)` | `float32` | `0` | `0.999522 / 1.000008 / 1.000470` | `0339efc1e3799fe2d449b92c412f7a06742190b44f9d716bbf9b2f30c5eb101c` | `True` |

---

## 2. Kết quả Bước 1b — Kiểm toán Caption Test (Chỉ đọc, không dùng nhãn/truy vấn test, không chạy model)

### 2.1 Mục 1 — Chuỗi sinh thô (`raw_caption`) của `shard_test_0000..0002` còn được lưu không?
- Có cột `raw_caption` trong `shard_test_0000..0002.parquet`: `False`
- Số tệp `cache_core_test_*.json` / `cache_aux_test_*.json` còn trên đĩa: `0`
- Số mục có `raw_caption` còn lưu trong `1,500` dòng `shard_test_0000..0002`: `0`
- Số mục có `raw_caption` còn lưu trong `1,290` mục `index/test_gallery`: `0`
- Số mục giao giữa `eval/cache_internvl2_pilot.json` (`200` ảnh pilot Pha 2) với `shard_test` (`1,500` mục) / `test_gallery` (`1,290` mục): `0` / `0`
- **Nguyên nhân kỹ thuật:** Trong `src/caption/batch_runner.py`, dòng 135–136 và 165–170 chỉ ghi `cleaned_caption` vào cột `caption_A` và `caption_C` (cùng bản hợp nhất `merge_v1` vào `caption_B` / `caption_vi`), sau đó tại dòng 190–193 gọi `shard_cache_core_path.unlink()` và `shard_cache_aux_path.unlink()` xóa tệp cache JSON tạm ngay khi hoàn tất mỗi shard.

### 2.2 Mục 2 — So sánh mã làm sạch tiền tố / sinh caption giữa `shard_test` (12–13/11/2026) và `shard_val` (13–14/11/2026)
- Tệp chứa hàm làm sạch tiền tố `clean_introductory_phrases()` và hợp nhất `merge_caption_v1()`: `src/caption/merger.py`
- Commit cuối cùng của `src/caption/merger.py`: `dfc150ad90edc9d426892a4cbf32eb0221b24955` (`2026-11-12 09:21:45 +0700`)
- `src/caption/merger.py` có thay đổi giữa `shard_test` và `shard_val` không: `False`
- Tệp sinh caption thay đổi giữa `shard_test_0000` và `shard_test_0001..0002` / `shard_val`: `src/caption/generator.py`
  - Trước `shard_test_0001`: `98aed711125e57d75c5cedb796ca9e4b4d0adcb1 (2026-11-12 09:34:50 +0700, max_new_tokens=96, max_num=2, use_thumbnail=True)`
  - Từ `shard_test_0001` và toàn bộ `shard_val`: `2e057d6bf870b4281fe556e03f2e0c2bf9d5a076 (2026-11-12 17:43:50 +0700, max_new_tokens=48, max_num=1, use_thumbnail=False)`
- Số mục `val_gallery` (`1,884` mục) có `caption_A` là chuỗi echo đề bài dài đúng `68` token (`"Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng"`): `1859`
- Số mục `test_gallery` (`1,290` mục) có `caption_A` là chuỗi echo đề bài dài đúng `68` token: `841`

| Tên shard | Thời điểm ghi (`mtime`) | Tổng dòng | Số dòng thuộc `test_gallery` | Số mục `test_gallery` có `caption_A <= 6` token | Số mục `test_gallery` có `caption_A` bắt đầu bằng `1.` | Số mục `test_gallery` có `caption_A == 68` token | Số mục `test_gallery` có `caption_B` chứa `1. Th` |
|---|---|---:|---:|---:|---:|---:|---:|
| `shard_test_0000.parquet` | `2026-11-12 18:57:06` | `500` | `442` | `411` | `413` | `0` | `0` |
| `shard_test_0001.parquet` | `2026-11-12 23:31:43` | `500` | `419` | `0` | `0` | `418` | `0` |
| `shard_test_0002.parquet` | `2026-11-13 01:52:11` | `500` | `429` | `0` | `0` | `425` | `0` |

- **Giải thích nguyên nhân gốc rễ (đính chính nhận định ở báo cáo 14c):** Hàm clean_introductory_phrases() và merge_caption_v1() trong src/caption/merger.py KHÔNG thay đổi giữa shard_test và shard_val (cùng khóa từ commit dfc150a lúc 09:21:45 ngày 12/11/2026). Sự khác biệt nằm ở commit 2e057d6 (17:43:50 ngày 12/11/2026) trong src/caption/generator.py khi giảm max_new_tokens từ 96 xuống 48 (và max_num từ 2 xuống 1): (1) Ở shard_test_0000 (chạy Stage 1 trước 17:43:50 với max_new_tokens=96), InternVL2-2B lặp lại (echo) đề bài tới cụm 'YÊU CẦU BẮT BUỘC:\n\n1. Thô' (có dấu hai chấm ':'), nên regex số 3 r'^(mô tả ngắn gọn...|yêu cầu bắt buộc)[\s\S]*?:\s*' trong clean_introductory_phrases() cắt toàn bộ phần trước dấu ':', để lại '1. Thô' (393/1,290 mục <= 6 token; 413/1,290 mục bắt đầu bằng '1.'). (2) Ở shard_test_0001, shard_test_0002 và toàn bộ shard_val_0000..0003 (chạy với max_new_tokens=48), chuỗi echo đề bài dừng ở 'Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng' (68 token SigLIP, chưa tới dấu ':'), nên regex số 3 không khớp và giữ nguyên 68 token echo đề bài (841/1,290 mục test và 1,859/1,884 mục val). (3) Với caption_B (merge_v1), cả ở val lẫn test, phần thân lấy từ caption_C (Vintern-1B-v3_5) đứng trước và giới hạn ở <= 128 token nên 0/1,290 mục test chứa '1. Th' trong caption_B.

### 2.3 Mục 3 — Kết quả đếm khôi phục trên các mục bị cắt cụt (không ghi đè shard, không ghi index)
- Tổng số mục trong `test_gallery` (`1,290` mục) có `caption_A` bị cắt ở đầu mục `"1."`: `413` mục (toàn bộ nằm ở `shard_test_0000.parquet`), gồm: `"1. Thô"` (`405` mục), `"1. **Th"` (`6` mục), `"1. Thông"` (`2` mục); tổng số mục `<= 6` token SigLIP theo `AutoTokenizer` là `411` mục.
- Số mục khôi phục được từ dữ liệu lưu trên đĩa mà không chạy lại model: `0`
- Số mục thay đổi khi áp lại `clean_introductory_phrases()` hiện tại lên `caption_A` đã lưu: `0`
- Độ dài token SigLIP của `caption_A` sau khi áp lại (`min / median / mean`): `4 / 68.0 / 49.31`
- Đối chứng tham khảo trên `eval/cache_internvl2_pilot.json` (`200` ảnh pilot Pha 2 có lưu `raw_caption` khi `max_new_tokens=96`): có `185` mục mà sau khi qua `clean_introductory_phrases()` trở thành `"1. Thô"`; toàn bộ `raw_caption` gốc của các mục đó đều là chuỗi echo đề bài dài `120` token SigLIP (`"Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:\n\n1. Thô"`).

### 2.4 Mục 4 — Danh sách `cfg_v5_*` dùng / không dùng index caption (`text_gallery != none`)
- Tổng số cấu hình trong `selection_rule_v5.json`: `12`
- Xác nhận `cfg_v5_09` (Official Winner) phụ thuộc `index/test_gallery_text_v5/`: `False`
- Xác nhận `cfg_v5_01` (Safe Config) phụ thuộc `index/test_gallery_text_v5/`: `False`
- Xác nhận `cfg_v5_06` (Reference Config) phụ thuộc `index/test_gallery_text_v5/`: `False`
- Xác nhận `model_only` phụ thuộc `index/test_gallery_text_v5/`: `False`

| `config_id` | Vai trò (`role`) | `gamma` | `i3_mode` | `conf_gating` | `fuzzy_typo` | `text_gallery` | `text_alpha` | Phụ thuộc `test_gallery_text_v5` | Tệp `.npy` sử dụng |
|---|---|---:|---|---|---|---|---:|---|---|
| `cfg_v5_01` | `v4_winner_baseline` | `1.5` | `none` | `False` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_02` | `i3_direct_equivalence_latency_only` | `1.5` | `direct` | `False` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_03` | `i3_cached_equivalence_latency_only` | `1.5` | `cached` | `False` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_04` | `tier_soft_plus_i3_cached` | `1.5` | `cached` | `False` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_05` | `ablation_I2_on_04` | `1.5` | `cached` | `True` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_06` | `ablation_I7_on_04` | `1.5` | `cached` | `False` | `True` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_07` | `combo_I2_I7_on_04` | `1.5` | `cached` | `True` | `True` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_08a` | `ablation_I5_internvl2_only_on_04` | `1.5` | `cached` | `False` | `False` | `internvl2_only` | `0.85` | `True` | `internvl2_only_embeddings.npy (caption_A)` |
| `cfg_v5_08b` | `ablation_I5_internvl2_plus_aux_on_04` | `1.5` | `cached` | `False` | `False` | `internvl2_plus_aux` | `0.85` | `True` | `internvl2_plus_aux_embeddings.npy (caption_B = merge_v1)` |
| `cfg_v5_09` | `gamma_sweep_tier_soft_1.0` | `1.0` | `cached` | `False` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_10` | `gamma_sweep_tier_soft_2.0` | `2.0` | `cached` | `False` | `False` | `none` | `1.0` | `False` | `none` |
| `cfg_v5_11` | `gamma_sweep_tier_soft_3.0` | `3.0` | `cached` | `False` | `False` | `none` | `1.0` | `False` | `none` |

### 2.5 Mục 5 — Tác động của việc `id_map.json` của `test` thiếu `*_vi` lên `metadata_aux` & Nguồn tên tiếng Việt khác
- Số cấu hình trong `v5` hoặc `cir_v1` sử dụng `metadata_aux_embeddings.npy`: `0`
- Tác động thực tế: Không ảnh hưởng (0 cấu hình trong selection_rule_v5.json và selection_rule_cir_v1.json sử dụng metadata_aux_embeddings.npy; selection_rule_v5.json dòng 40 khóa forbid_title_vi=true).

| Cột tiếng Việt | Số mục có giá trị trong `index/test_gallery/id_map.json` (`/ 1290`) | Số mục có giá trị trong `data/catalog/catalog_rebuild.parquet` (`/ 1290`) | Số mục có giá trị trong `shard_test_0000..0002.parquet` (`/ 1290`) |
|---|---:|---:|---:|
| `title_vi` | `0` | `1290` | `1290` |
| `articleType_vi` | `0` | `1290` | `1290` |
| `gender_vi` | `0` | `1290` | `1290` |
| `baseColour_vi` | `0` | `1290` | `1290` |

### 2.6 Mục 6 — Những điều "không xác minh được" (Luật 7)
1. Không xác minh được nội dung từng ký tự của raw_caption (trước bước clean_introductory_phrases) trên 1,500 mục của shard_test_0000..0002.parquet (1,290 mục test_gallery), vì src/caption/batch_runner.py không lưu cột raw_caption vào tệp .parquet và đã xóa (unlink) tệp cache_core_test_0000..0002.json ngay sau khi hoàn tất mỗi shard vào ngày 12–13/11/2026.
2. Không xác minh được nếu chạy lại InternVL2-2B với prompt khác hoặc giải pháp chống lặp đề bài trên 1,290 ảnh test_gallery thì caption_A (internvl2_only) sẽ thay đổi điểm số của cfg_v5_08a bao nhiêu nếu không thực sự chạy lại model VLM.

---

## 3. Kết quả Bước 2 — Kiểm tra GPU & Mẫu nghỉ 30 giây (`HALT — CÒN TIẾN TRÌNH CẤM TRÊN GPU`)

- Thời điểm kiểm tra: `2026-11-14 22:06:14`
- PID tiến trình Python của agent (chỉ thu thập `nvidia-smi`, không nạp model): `15912`
- Số tiến trình thuộc danh sách cấm đang giữ context trên GPU 0: `2`

### 3.1 Danh sách toàn bộ tiến trình trên GPU 0 từ `nvidia-smi`

| PID | Loại | Tên tiến trình | Đường dẫn (`Win32_Process` / `nvidia-smi`) | `ParentPID` | Bộ nhớ GPU | Phân loại theo Bước 2 |
|---:|---|---|---|---:|---|---|
| `1560` | `C+G` | `ShellHost.exe` | `C:\Windows\System32\ShellHost.exe` | `10500` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `1596` | `C+G` | `Antigravity.exe` | `C:\Users\nguye\AppData\Local\Programs\antigravity\Antigravity.exe` | `12056` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `2912` | `C+G` | `Antigravity.exe` | `C:\Users\nguye\AppData\Local\Programs\antigravity\Antigravity.exe` | `12056` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `3588` | `C+G` | `EdgeGameAssist.exe` | `C:\Program Files\WindowsApps\Microsoft.Edge.GameAssist_1.0.4019.0_x64__8wekyb3d8bbwe\EdgeGameAssist.exe` | `1540` | `N/A` | `FORBIDDEN_GAME_OR_EMULATOR` |
| `12004` | `C+G` | `CrossDeviceResume.exe` | `C:\WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\CrossDeviceResume.exe` | `10500` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `12964` | `C+G` | `SearchHost.exe` | `C:\WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe` | `1540` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `12976` | `C+G` | `StartMenuExperienceHost.exe` | `C:\WINDOWS\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe` | `1540` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `13176` | `C+G` | `WidgetBoard.exe` | `C:\Program Files\WindowsApps\MicrosoftWindows.Client.WebExperience_526.21100.40.0_x64__cw5n1h2txyewy\WidgetBoard.exe` | `1540` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `13356` | `C+G` | `msedgewebview2.exe` | `C:\Program Files (x86)\Microsoft\EdgeWebView\Application\157.0.4341.46\msedgewebview2.exe` | `13784` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `16284` | `C+G` | `dnmultiplayerex.exe` | `D:\LDPlayer\ldmutiplayer\dnmultiplayerex.exe` | `4840` | `N/A` | `FORBIDDEN_GAME_OR_EMULATOR` |
| `17140` | `C+G` | `TextInputHost.exe` | `C:\WINDOWS\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TextInputHost.exe` | `1540` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |
| `22288` | `C+G` | `PhoneExperienceHost.exe` | `...ekyb3d8bbwe\PhoneExperienceHost.exe` | `N/A` | `N/A` | `ALLOWED_WINDOWS_OR_IDE` |

### 3.2 Thống kê 30 mẫu `GPU-Util` và cảm biến GPU khi nghỉ (`1 giây × 30 giây`, agent không chạy gì)

- Số mẫu: `30` (khoảng cách `1.0` giây)
- Phân bố P-state trong 30 mẫu: `{"P5": 26, "P8": 4}`

| Chỉ số cảm biến (`nvidia-smi`) | `median` | `mean` | `min` | `max` |
|---|---:|---:|---:|---:|
| `gpu_util_pct` (%) | `29.5` | `26.23` | `0.0` | `60.0` |
| `mem_util_pct` (%) | `2.0` | `1.73` | `0.0` | `4.0` |
| `temp_c` (°C) | `59.0` | `58.97` | `58.0` | `59.0` |
| `power_w` (W) | `4.11` | `4.06` | `3.47` | `4.4` |
| `gr_clock_mhz` (MHz) | `487.0` | `508.03` | `345.0` | `615.0` |
| `mem_clock_mhz` (MHz) | `810.0` | `756.0` | `405.0` | `810.0` |

### 3.3 Lý do dừng tại Bước 2 & Hướng dẫn tắt chỉ các tiến trình thuộc danh sách cấm
- Theo đúng quy tắc Bước 2 của Prompt 14d:
  > *"Còn `dnmultiplayerex.exe`, `EdgeGameAssist.exe`, `RobloxPlayerBeta.exe` hoặc tiến trình game/giả lập/đồ họa nặng khác: DỪNG, báo, KHÔNG tự kill. Khi hướng dẫn tắt, chỉ nêu các tiến trình thuộc danh sách cấm; không đề xuất kill `msedgewebview2.exe` hay tiến trình hệ thống/IDE."*
- Hiện tại trên GPU 0 vẫn còn đúng 2 tiến trình thuộc danh sách cấm:
  - `EdgeGameAssist.exe` (PID `3588`, đường dẫn `C:\Program Files\WindowsApps\Microsoft.Edge.GameAssist_1.0.4019.0_x64__8wekyb3d8bbwe\EdgeGameAssist.exe`)
  - `dnmultiplayerex.exe` (PID `16284`, đường dẫn `D:\LDPlayer\ldmutiplayer\dnmultiplayerex.exe`)
- Để tắt dứt điểm **chỉ 2 tiến trình thuộc danh sách cấm này** (không chạm vào các tiến trình hệ thống Windows hay IDE), vui lòng mở PowerShell (Run as Administrator) và chạy:
```powershell
Stop-Process -Id 3588, 16284 -Force
```
sau đó kiểm tra lại bằng `nvidia-smi` để xác nhận hai PID trên không còn trong bảng `Processes`.

---

## 4. Trạng thái Phần A (Đo độ trễ) & Chốt CIR
- **Phần A (Độ trễ 4 luồng):** `NOT_RUN_HALTED_AT_STEP_2_FORBIDDEN_GPU_PROCESSES` — Chưa chạy do Bước 2 dừng bởi tiến trình cấm trên GPU.
- **Chốt CIR (`selection_rule_cir_v1.json`):** `PENDING_PART_A_CLEAN_GPU_LATENCY` — Giữ nguyên trạng thái tạm thời (`cir_v1_05` chờ số đo độ trễ GPU sạch của Phần A).
- **Phần B & C (Chất lượng trên `test_gallery`):** `NOT_RUN_MISSING_HELDOUT_V2_AND_HALTED_AT_STEP_2` — Không chạy (chưa có `eval/test_queries_heldout_v2.json`, không mở `eval/cir_eval_test.json`).

---

## 5. Kết quả đối chiếu tự động `.md` ↔ `.json` (Quy tắc số liệu mới Prompt 14d)
- Phương thức: Toàn bộ bảng và con số trong `reports/phase5c/final_eval_report.md` được sinh tự động bằng hàm `render_markdown_from_json()` trong `scripts/run_prompt14d_audit_and_report.py` trực tiếp từ `reports/phase5c/final_eval_results.json`.
- Tổng số điểm dữ liệu (số học & mã băm SHA-256) được kiểm tra đối chiếu tự động: `56`
- Số điểm khớp tuyệt đối `.md == .json`: `56`
- Số điểm lệch: `0`
- Kết luận đối chiếu: **`PASS_EXACT_MATCH`**

---

## 6. Kiểm thử tự động (`pytest`) & Tệp thay đổi
- Kết quả chạy `pytest`: `70 passed, 2 warnings in 89.38s` (`passed = 70`, `failed = 0`).
