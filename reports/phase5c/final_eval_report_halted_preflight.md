# Báo cáo Kiểm định Bước 0 & Điểm dừng Trước Phần A/B — Prompt 14 (`index/test_gallery/`)

- **Trạng thái thực thi:** `HALTED_PREFLIGHT` *(Tuân thủ nghiêm ngặt quy tắc dừng ở Bước 0.2, Bước 0.4 và Phần A Pre-check; chưa chạy đánh giá chất lượng trên `index/test_gallery/` và không tự ý kill tiến trình nền)*.
- **Branch / Commit khởi điểm:** `phase7a` / `b4d5300b46aca72529254842067eab9c1d34d48c`
- **Seed cố định:** `42`
- **Tệp kết quả JSON:** [final_eval_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/final_eval_results.json)
- **Tệp định nghĩa chỉ số đã khóa (Mục 0.6):** [locked_final_definitions.py](file:///d:/Kì%207/Hoc_sau_ver4/src/eval/locked_final_definitions.py) (`SHA-256: d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7`)

---

## 1. Kết quả Kiểm tra Điều kiện Bước 0 (`0.1` – `0.6`)

### 0.1. Kiểm tra SHA-256 toàn bộ đầu vào
| Đầu vào | Đường dẫn | SHA-256 Kỳ vọng | SHA-256 Thực tế | Trạng thái |
|---|---|---|---|---|
| Checkpoint `best_model.pt` | `data/checkpoints_siglip-vit/best_model.pt` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` | ✅ Khớp |
| Quy tắc T2I `selection_rule_v5.json` | `reports/phase5c/selection_rule_v5.json` | `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | `e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c` | ✅ Khớp |
| Quy tắc CIR `selection_rule_cir_v1.json` | `reports/phase5c/selection_rule_cir_v1.json` | `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067` | `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067` | ✅ Khớp |
| Tập CIR Test `cir_eval_test.json` | `eval/cir_eval_test.json` | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | ✅ Khớp |
| Danh sách Test Split `test_split.json` | `eval/test_split.json` | `d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681` | `d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681` | ✅ Khớp |
| Catalog chuẩn hóa `catalog_rebuild.parquet` | `data/catalog/catalog_rebuild.parquet` | `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` | `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` | ✅ Khớp |
| Định nghĩa chỉ số khóa `locked_final_definitions.py` | `src/eval/locked_final_definitions.py` | *(Khóa tại Bước 0.6)* | `d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7` | ✅ Đã khóa |
| Tập truy vấn Unseen `test_queries_heldout_v2.json` | `eval/test_queries_heldout_v2.json` | *(Bắt buộc ở Mục 0.4)* | `None` *(Không tồn tại trên đĩa)* | ❌ **THIẾU** |

### 0.2. Kiểm tra Hash Guard `index/test_gallery/` & Các Index Caption (`Config A / Config B / metadata-only`)
1. **`index/test_gallery/manifest.json` (`SHA-256: 57348838c4aa920e853ba004b60f5f4a1b6fdebbc1d59043cd122691867f92aa`):**
   - `checkpoint_hash`: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` (Khớp 100% ✅).
   - `catalog_hash`: `aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab` (Khớp 100% ✅).
   - Số vector: `n_vectors = 1290`, `image_embeddings.npy` có shape `(1290, 1152)`, `len(id_map) = 1290` (Khớp 100% ✅).
2. **Kiểm tra Index Caption `Config A / Config B / metadata-only`:**
   - Trên `val_gallery`, thư mục `index/val_gallery_text_v5/manifest.json` (`SHA-256: 6d44fe7dc702257d294e375f409858b1ed53b1814d0363ea21c0b2cbce258c15`, `1,884` vector) đã tồn tại và hợp lệ.
   - **Điểm dừng Mục 0.2 trên `test_gallery`:** Thư mục **`index/test_gallery_text_v5/manifest.json`** (chứa vector nhúng `internvl2_only_embeddings.npy`, `internvl2_plus_aux_embeddings.npy`, `metadata_aux_embeddings.npy` cho `1,290` mục của `test_gallery`, cần thiết cho `cfg_v5_08b` ở Phần A/B1(iii) và đánh giá Luật 5 ở B1(iv)) **chưa được dựng trên đĩa**.
   - *(Ghi chú: Dữ liệu caption nguồn cho `test_split` tại `data/catalog/captions/shard_test_0000..0002.parquet` đã có sẵn đầy đủ `1,290/1,290` sản phẩm với cả `caption_A` và `caption_B`. Tuy nhiên, theo đúng quy tắc Bước 0 "sai mục nào thì DỪNG và báo, không tự sửa", Agent không tự ý dựng `index/test_gallery_text_v5/` khi chưa có lệnh của người dùng).*

### 0.3. Kiểm định Luật 4 (Tái encode 200 ảnh mẫu, `cosine min >= 0.999`)
- Đã tái encode trực tiếp **200 ảnh** lấy mẫu ngẫu nhiên (`seed = 42`) từ `index/test_gallery/` bằng đường tiền xử lý duy nhất (`SigLIPPreprocessor` + `load_siglip_model`) và đối chiếu với `index/test_gallery/image_embeddings.npy`:
  - **`min_cosine = 0.999978`** (`>= 0.999` ✅ **ĐẠT**)
  - **`mean_cosine = 1.000000`**, **`median_cosine = 1.000000`**
- Đối chiếu tham chiếu Pha 3 ([parity_report.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase3/parity_report.json)): `min_cosine = 0.999660` (`>= 0.999` ✅).

### 0.4. Kiểm tra `eval/test_queries_heldout_v2.json`
- Tệp `eval/test_queries_heldout_v2.json` (hoặc `eval/heldout_queries_v2.json`) **chưa tồn tại trên đĩa** (`exists = False`; hiện chỉ có phiếu lấy mẫu [heldout_v2_worksheet.csv](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/heldout_v2_worksheet.csv)).
- Theo đúng mệnh lệnh Mục 0.4:
  > *"Nếu thiếu hoặc không đạt: KHÔNG chạy bất kỳ đánh giá chất lượng nào trên test_gallery. Chỉ chạy Phần A (độ trễ, không dùng nhãn test) rồi báo và DỪNG."*

### 0.5. Xác nhận Duyệt Cấu hình T2I (Prompt 12) và CIR (Prompt 13)
- **T2I:** Đã xác nhận trong [step5c_selection_report.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/step5c_selection_report.md): cấu hình chính thức **`cfg_v5_09`**, cấu hình an toàn **`cfg_v5_01`**, hàng tham khảo khai báo trước **`cfg_v5_06`** và **`cfg_v5_08b`**.
- **CIR:** Đã xác nhận trong [step5c_cir_report.md](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/step5c_cir_report.md): duyệt tạm **`cir_v1_05`** (kèm baseline **`cir_v1_01`**), chốt chính thức sau khi đo độ trễ Phần A theo đúng `selection_rule_cir_v1.json`.

### 0.6. Khóa Định nghĩa Chỉ số, Bootstrap, GT phụ và Quy ước Kết luận trước khi chạy
- Đã khóa trong [locked_final_definitions.py](file:///d:/Kì%207/Hoc_sau_ver4/src/eval/locked_final_definitions.py) (`SHA-256: d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7`, lưu tại [locked_final_definitions.sha256](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/locked_final_definitions.sha256)):
  1. `Recall@K (hit-rate)`: `1.0` nếu có ít nhất 1 mục thuộc GT trong Top-`K`, ngược lại `0.0`.
  2. `true-recall@10` (bổ sung): `sum(rels[:10]) / max(1, total_relevant_in_gallery)`.
  3. `mAP`: Trung bình `AP` tính trên toàn bộ danh sách xếp hạng `1,290` mục của `test_gallery`.
  4. `Bootstrap`: Theo đơn vị truy vấn (`query-level`), `1,000` lần lấy mẫu có hoàn lại với `seed = 42` (`paired` khi so sánh chênh lệch giữa hai cấu hình).
  5. `Chỉ số GT phụ non-circular (usage/season)`: Giữ nguyên thứ tự xếp hạng của hệ thống bên trong tập con `S_primary` (các mục khớp thuộc tính chính `articleType, gender [hybrid], baseColour`), tính `within_group_mAP` và `within_group_R@5` đối với các mục trong `S_primary` khớp thuộc tính phụ `usage/season`.
  6. `Quy ước kết luận độ trễ (p95 < 200 ms)` và `Quy ước kết luận chất lượng (R@10 >= 0.85, mAP >= 0.75)`: Khóa hàm phân loại tự động trong `locked_final_definitions.py`.

---

## 2. Kiểm tra Trước Phần A (`nvidia-smi`) — Điểm dừng Bắt buộc do Tiến trình Đồ họa Nền

Khi chuyển sang thực hiện **Phần A (Độ trễ 4 luồng)** theo chỉ dẫn của Mục 0.4, bước kiểm tra danh sách tiến trình GPU từ `nvidia-smi` ghi nhận:

| PID | Loại | Tên tiến trình trên GPU | Ghi chú |
|---:|---|---|---|
| `16284` | `C+G` | `...\ldmutiplayer\dnmultiplayerex.exe` | ❌ **Tiến trình giả lập đồ họa nền (LDPlayer Multi-Player)** |
| `3588` | `C+G` | `...4__8wekyb3d8bbwe\EdgeGameAssist.exe` | ❌ **Tiến trình hỗ trợ game nền (Edge Game Assist)** |
| `1596`, `2912` | `C+G` | `...\antigravity\Antigravity.exe` | Tiến trình IDE hiện tại |
| `4724`, `12004`, `12964`, `12976`, `13176`, `13356`, `17140` | `C+G` | Các tiến trình giao diện Windows (`ShellHost`, `SearchHost`, `StartMenuExperienceHost`, `WidgetBoard`, `msedgewebview2`, `TextInputHost`) | Hệ thống Windows |

- **Thông số GPU tại thời điểm kiểm tra:** `NVIDIA GeForce RTX 2050`, `GPU-Util = 28%` ở trạng thái nghỉ (lên `84%` khi kiểm định 200 ảnh), `temperature = 58–70°C`.
- **Tuân thủ mệnh lệnh Phần A:**
  > *"Trước/sau: ghi tiến trình GPU (nvidia-smi). **Còn tiến trình đồ họa nền thì DỪNG, báo người dùng, không tự kill.**"*
- Do `dnmultiplayerex.exe` (PID `16284`) và `EdgeGameAssist.exe` (PID `3588`) vẫn đang chạy trên GPU, Agent **DỪNG NGAY**, không tự kill tiến trình và không chạy phép đo độ trễ Phần A để tránh lặp lại sai số nhiệt/xung nhịp của Prompt 13.

---

## 3. Các Điểm Cần Người Dùng Xử Lý / Xác Nhận Trước Khi Chạy Tiếp Prompt 14

1. **Đóng tiến trình đồ họa nền trên GPU (để chạy Phần A):**
   - Vui lòng đóng `dnmultiplayerex.exe` (LDPlayer Multi-Player, PID `16284`) và `EdgeGameAssist.exe` (PID `3588`) để GPU hoàn toàn sạch (`GPU-Util = 0%`, nhiệt độ `< 60°C`).
2. **Dựng `index/test_gallery_text_v5/` kèm `manifest.json` (Mục 0.2):**
   - Để đo độ trễ riêng cho `cfg_v5_08b` ở Phần A (theo quyết định sau Prompt 12: đo riêng `cfg_v5_01, 06, 09, 08b` trên `test_gallery`) và chạy B1(iii)/B1(iv), xác nhận cho phép Agent encode các cột `caption_A` / `caption_B` / `metadata` từ `data/catalog/captions/shard_test_0000..0002.parquet` (đã có sẵn `1,290/1,290` mục) thành `index/test_gallery_text_v5/` kèm `manifest.json` + Hash Guard.
3. **Cung cấp `eval/test_queries_heldout_v2.json` (Mục 0.4 — để chạy Phần B & C):**
   - Hiện tại chưa có `eval/test_queries_heldout_v2.json` trên đĩa. Nếu người dùng chưa cung cấp tệp này ngay, sau khi người dùng đóng tiến trình GPU nền (và duyệt dựng `index/test_gallery_text_v5/`), Agent sẽ chạy **chỉ Phần A (Độ trễ 4 luồng)** rồi dừng theo đúng Mục 0.4; khi có `eval/test_queries_heldout_v2.json` đạt đủ tiêu chuẩn Mục 0.4 sẽ chạy **Phần B & Phần C** đúng 1 lần duy nhất.
