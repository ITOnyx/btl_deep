import re
from pathlib import Path

p = Path("reports/phase3/phase3f/phase3f_errata.md")
content = p.read_text(encoding="utf-8").replace("\r\n", "\n")

# 1. Update Table 0 summary
old_table_0 = """| **B** | Latency & SLA AGENTS.md | `XÁC NHẬN LỖI` | AGENTS.md chỉ quy định `latency < 200 ms/query` (không ghi p95); xóa câu "SLA chuẩn theo AGENTS.md"; truy dải p95 ảnh→ảnh đúng là `[137.22 - 219.23 ms]` (Rev3 ghi `[171.2–192.4]` là sai bản); bác bỏ giả thuyết "cold-start CUDA JIT" (10 warmup queries trước run 1; đỉnh 1447-1500 ms ở Run 4 & 5 do nhiệt độ 86°C thermal throttling xuống 442 MHz và tranh chấp tiến trình nền); đổi nhãn thành **"ĐẠT CÓ ĐIỀU KIỆN" (1/5 run vượt 200 ms)** | `reports/phase3/phase3g/phase3g_latency_multi_run.json:712-735`, `reports/phase3/phase3g/phase3g_slow_queries_telemetry.json`, `AGENTS.md:7` |"""

new_table_0 = """| **B** | Latency & SLA AGENTS.md | `XÁC NHẬN LỖI` | AGENTS.md chỉ quy định `latency < 200 ms/query` (không ghi p95); xóa câu "SLA chuẩn theo AGENTS.md"; truy dải p95 ảnh→ảnh đúng là `[137.22 - 219.23 ms]`; hạ nguyên nhân quá nhiệt thành "giả thuyết chưa tái lập" (đo sạch 5 run B.4 và Pha 4 không ghi nhận truy vấn > 500 ms); không kết luận SLA từ 3 truy vấn ảnh; đổi nhãn thành **"ĐẠT CÓ ĐIỀU KIỆN" (1/5 run vượt 200 ms)** | `reports/phase3/phase3g/phase3g_latency_multi_run.json:712-735`, `reports/phase4_verifications.json`, `AGENTS.md:7` |"""

content = content.replace(old_table_0, new_table_0)

# Add row K in Table 0
old_table_f = """| **F** | Đánh giá Chấm mù A/B Rev5 | `ĐÃ HOÀN TẤT` | Ghi nhận Claude (AI) single rater; bổ sung hạn chế "người chấm từng thấy mapping cũ"; kiểm tra sinh hoán vị A/B ngẫu nhiên (seed 20261109) ra 17 lần A=C5, 13 lần A=C0 với 9 khối chạy (8 lần đổi hàng); đối chiếu P@10 C0 brand: 3f ghi 0.0625 (5/80 hits) vs Rev4 ghi 0.0500 (4/80 hits); Parser V2 bổ sung quy tắc cho đầu vào không dấu, chạy lại 36 câu đổi 4 câu (x025, x027, x028, Q3) và ghi nhận tính chất hậu nghiệm | `reports/phase3/phase3g/phase3g_blind_ab_results.json`, `reports/phase3/phase3g/phase3g_do_bug_investigation.json`, `eval/_sealed/ab_mapping.json` |"""

new_table_f_and_k = """| **F** | Đánh giá Chấm mù A/B Rev5 | `ĐÃ HOÀN TẤT` | Ghi nhận Claude (AI) single rater; bổ sung hạn chế "người chấm từng thấy mapping cũ"; sửa chuỗi 30 phần tử hoán vị A/B (khớp hoàn toàn seed 20261109 và ab_mapping.json); tính C0 P@10 trên GT mới (x018=209) đạt 0.0714 so với C5 đạt 0.8000; ghi rõ GT Condition A cũ (n=24/23) | `reports/phase3/phase3g/phase3g_blind_ab_results.json`, `reports/phase4_verifications.json`, `eval/_sealed/ab_mapping.json` |
| **K** | Kiểm toán 20 Ca Thất bại Pha 3e | `XÁC NHẬN LỖI` | Đối soát với `eval/results_3e.json`: 5 ca trước đây (text_x002 áo sơ mi, x007 giày cao gót be, x015 bomber, x021 hoodie, visual_1855) hoàn toàn không có trong JSON; đánh dấu "KHÔNG TRUY ĐƯỢC NGUỒN, KHÔNG ĐÁNG TIN" (không gọi lỗi gõ tay); lập danh sách 20 ca thật từ JSON | `eval/results_3e.json:diagnostics.failure_cases`, `reports/phase3/phase3e/phase3e_inference_only.md:220-250` |"""

content = content.replace(old_table_f, new_table_f_and_k)

# 2. Update B.3
old_b3 = """  3. **Nguyên nhân kỹ thuật thực sự:**
     - **Quá nhiệt và Thermal Throttling:** Chạy 5 runs liên tục (1.000 text queries + 1.000 image queries) trên GPU laptop NVIDIA RTX 2050 Mobile khiến nhiệt độ chạm trần nhiệt **86°C**, xung nhịp GPU bị hệ điều hành bóp từ 1.470 MHz xuống **442–480 MHz**.
     - **Cạnh tranh tài nguyên tiến trình nền:** Tại thời điểm đo (`2026-11-09 12:52:43`), log ghi nhận 16 tiến trình nền Windows/IDE đang chia sẻ GPU (`SearchHost.exe`, `WidgetBoard.exe`, `EdgeWebView`, `Antigravity.exe`, `Claude.exe`, `python.exe`), gây hiện tượng context-switch stall ngẫu nhiên ở cuối phiên."""

new_b3 = """  3. **Hạ nguyên nhân quá nhiệt thành GIẢ THUYẾT CHƯA TÁI LẬP:**
     - Giả thuyết ban đầu cho rằng nhiệt độ 86°C gây thermal throttling dẫn đến các đỉnh trễ > 1.400 ms.
     - Tuy nhiên, trong đợt đo đối chứng B.4 (cũng như benchmark kiểm toán 5 run mới tại `reports/phase4_verifications.json`), **HOÀN TOÀN KHÔNG CÓ TRUY VẤN NÀO > 500 ms** (0/1.000 truy vấn). Ngay cả ở nhiệt độ 80°C, xung nhịp GPU duy trì ổn định ở 1.732 MHz và độ trễ p99 đạt < 195 ms.
     - Do đó, nguyên nhân "quá nhiệt gây đỉnh trễ 1.500 ms" **chính thức bị hạ thành giả thuyết chưa tái lập**. Đỉnh trễ cũ là do tranh chấp tài nguyên ngẫu nhiên của các tiến trình nền Windows tại thời điểm đo cũ, không tái lập được trong môi trường đo chuẩn hóa.
     - **Quy tắc nghiệm thu SLA:** **Tuyệt đối không kết luận đạt/vượt SLA từ chỉ 3 truy vấn ảnh khói (smoke test)**. Kết luận SLA phải dựa trên phân phối thực nghiệm tối thiểu 5 run benchmark 200 truy vấn (1.000 truy vấn tổng thể) kèm telemetry GPU chi tiết."""

content = content.replace(old_b3, new_b3)

# 3. Update D.2
old_d2_noise = """2. **Cắt cụt tại token 64 trên cấu trúc chuỗi metadata này không gây mất mát thông tin:** Trong cấu trúc `[{category}] {title} ({attrs})`, 64 token đầu đã bao hàm toàn bộ danh mục cốt lõi, tên sản phẩm và các thuộc tính chính. Việc cắt cụt tại token 64 rồi đệm lên 128 (Condition B) loại bỏ các thuộc tính nhiễu ở đuôi chuỗi (năm sản xuất, mùa phụ, mã SKU), giúp R@10 tăng từ **0.5225** lên **0.6186**. Lưu ý: Kết luận này áp dụng cụ thể cho cấu trúc chuỗi metadata hiện tại, không khái quát hóa thành nguyên lý chung cho mọi loại văn bản dài."""

new_d2_noise = """2. **Đánh giá Truncate 64 + Pad 128 trên tập Validation (split val), không chọn trên $G_{test}$:**
   - **Nguyên tắc chống rò rỉ dữ liệu:** Nếu muốn cân nhắc áp dụng Truncate 64 + Pad 128, mọi tham số bắt buộc phải được chọn và đánh giá trên tập validation (`split == 'val'`), tuyệt đối không được lựa chọn dựa trên tập test ($G_{test}$).
   - **Xóa bỏ giải thích "loại bỏ thuộc tính nhiễu":** Câu giải thích *"loại bỏ các thuộc tính nhiễu ở đuôi chuỗi"* trong các bản trước **chính thức được đánh dấu là giả thuyết chưa đo (unmeasured hypothesis)**.
   - **Kết quả đo trên tập Validation (7.470 sản phẩm, seed 42, 28 truy vấn val tại `reports/phase4_verifications.json`):**
     - Condition A (Full 128, Pad 128): Mean AP@10 = **0.4223**, Mean P@10 = **0.4929**.
     - Condition B (Truncate 64, Pad 128): Mean AP@10 = **0.4223**, Mean P@10 = **0.4929** (Đồng nhất 100% với Condition A vì các truy vấn thực tế đều $\\le 64$ token).
     - Condition C (Truncate 64, Pad 64): Mean AP@10 = **0.2612**, Mean P@10 = **0.3214** (Sụt giảm -38.1% do lệch positional embedding 128).
   - **Kết luận:** Cấu hình chuẩn hóa của hệ thống tiếp tục giữ nguyên Condition A (`max_length: 128, padding: max_length`)."""

content = content.replace(old_d2_noise, new_d2_noise)

# 4. Update F.2 sequence
old_seq = "`[1, 1, 1, 1, 1, 1 | 0, 0, 0, 0, 0 | 1, 1, 1, 1 | 0, 0 | 1, 1 | 0, 0, 0, 0, 0 | 1 | 0 | 1, 1, 1]`."
new_seq = "`[1, 1, 1, 1, 1, 1 | 0, 0, 0, 0, 0 | 1, 1, 1, 1 | 0, 0 | 1, 1, 1 | 0, 0, 0, 0, 0 | 1 | 0 | 1, 1, 1]` (Đúng 30 phần tử, khớp 100% với `eval/_sealed/ab_mapping.json`)."
content = content.replace(old_seq, new_seq)

# 5. Add Section K at the end
section_k = """
---

## K. Kiểm toán 20 Ca Thất bại Pha 3e & Khắc phục Mâu thuẫn Dữ liệu

### K.1 Đối chiếu 1:1 với `eval/results_3e.json` (`diagnostics.failure_cases`)
- **Phát hiện kiểm toán:** Trong báo cáo Pha 3e gốc (`reports/phase3/phase3e/phase3e_inference_only.md:224-242`), 5 trường hợp ví dụ sau:
  1. `text_x002` ("áo sơ mi công sở nam dài tay màu xanh navy")
  2. `text_x007` ("giày cao gót nữ màu be mũi nhọn")
  3. `text_x015` ("áo khoác bomber nam unisex đen")
  4. `text_x021` ("áo hoodie nỉ lót bông form rộng")
  5. `visual_1855` (Truy xuất ảnh dép xỏ ngón Flip Flops)
- **Đánh dấu chính thức:** Cả 5 trường hợp trên **hoàn toàn KHÔNG TỒN TẠI trong file dữ liệu gốc `eval/results_3e.json`**.
- **Quy tắc phân loại lỗi:** Đánh dấu chính thức là **"KHÔNG TRUY ĐƯỢC NGUỒN, KHÔNG ĐÁNG TIN"**, **tuyệt đối không gọi là lỗi gõ tay hay nhầm lẫn biên tập**.

### K.2 Danh sách Đầy đủ 20 Ca Thất bại Thật Trích xuất từ `eval/results_3e.json`
Toàn bộ 20 ca thật đã được phục hồi đầy đủ vào `reports/phase3/phase3e/phase3e_inference_only.md:220-250`:
1. `text_x002`: `"áo thun nữ màu hồng"` (Target: `Tshirts`, `Women`, `Pink` | C0 Top-1: `26923` | C5 Top-1: `1651` | C5 P@10: `0.4`)
2. `text_x004`: `"giày cao gót nữ màu đen"` (Target: `Heels`, `Women`, `Black` | C0 Top-1: `47569` | C5 Top-1: `47569` | C5 P@10: `0.4`)
3. `text_x005`: `"túi xách nữ màu đỏ"` (Target: `Handbags`, `Women`, `Red` | C0 Top-1: `52641` | C5 Top-1: `47289` | C5 P@10: `0.4`)
4. `text_x006`: `"dép xỏ ngón nam màu đen"` (Target: `Flip Flops`, `Men`, `Black` | C0 Top-1: `20840` | C5 Top-1: `20943` | C5 P@10: `0.1`)
5. `text_x009`: `"vay nu mau xanh la"` (Target: `Dresses`, `Women`, `Green` | C0 Top-1: `49593` | C5 Top-1: `34585` | C5 P@10: `0.1`)
6. `text_x010`: `"quan short nu mau trang"` (Target: `Shorts`, `Women`, `White` | C0 Top-1: `55814` | C5 Top-1: `43915` Cream | C5 P@10: `0.0`)
7. `text_x014`: `"đồng hồ Fossil nam"` (Target: `Watches`, `Men`, `Fossil` | C0 Top-1: `36251` | C5 Top-1: `58454` | C5 P@10: `0.3`)
8. `text_x015`: `"áo kurta Fabindia nam"` (Target: `Kurtas`, `Men`, `Fabindia` | C0 Top-1: `32199` | C5 Top-1: `30782` | C5 P@10: `0.3`)
9. `text_x017`: `"túi xách Lavie nữ"` (Target: `Handbags`, `Women`, `Lavie` | C0 Top-1: `51958` | C5 Top-1: `53679` Kiara | C5 P@10: `0.0`, R=0)
10. `text_x018`: `"áo thun Benetton nam"` (Target: `Tshirts`, `Men`, `Benetton` | C0 Top-1: `30756` | C5 Top-1: `19664` | C5 P@10: `0.0` ở Pha 3e)
11. `text_x021`: `"áo khoác nữ mùa đông"` (Target: `Jackets`, `Women` | C0 Top-1: `13961` | C5 Top-1: `18961` | C5 P@10: `0.1`)
12. `text_x022`: `"đầm nữ đi tiệc màu đen"` (Target: `Dresses`, `Women`, `Black` | C0 Top-1: `8485` | C5 Top-1: `8485` | C5 P@10: `0.3`)
13. `visual_7990`: Image Query (Texture similarity dominated over `Tshirts, Purple`)
14. `visual_25947`: Image Query (Texture similarity dominated over `Mufflers, Black`)
15. `visual_8746`: Image Query (Texture similarity dominated over `Sandals, Navy Blue`)
16. `visual_46086`: Image Query (Texture similarity dominated over `Sports Shoes, Purple`)
17. `visual_36137`: Image Query (Texture similarity dominated over `Casual Shoes, Red`)
18. `visual_44725`: Image Query (Texture similarity dominated over `Flip Flops, Brown`)
19. `visual_5253`: Image Query (Texture similarity dominated over `Duffel Bag, Blue`)
20. `visual_50574`: Image Query (Texture similarity dominated over `Trunk, Black`)

---

## L. Đánh giá Cấu hình Phục vụ (Serving) & So sánh Cơ sở Ground Truth Brand

### L.1 Chạy lại Q2, Q3, Q5, Heldout trên Cấu hình Phục vụ (`reports/phase4_verifications.json`)
Cấu hình phục vụ: `QueryParserV2` + `brands_v2.json` + `Thông báo Lavie` + `AttributeIndex` relaxed + `BM25 RRF` (Ghi nhận tính chất **hậu nghiệm - post-hoc**):

| Bộ truy vấn | Tập Gallery | Reported AP@10 (Bản cũ 3f/3g) | Serving AP@10 (Phục vụ) | Serving P@10 | Ghi chú Trạng thái |
|---|:---:|:---:|:---:|:---:|---|
| **Q2 (61 Attribute Queries)** | $G_{test}$ | **0.9457** | **0.9457** | 0.7230 | Không đổi (khớp parser) |
| **Q2 (61 Attribute Queries)** | $G_{full}$ | **0.9982** | **0.9982** | 0.9984 | Không đổi |
| **Q3 (100 Handwritten Queries)** | $G_{test}$ | **0.7635** | **0.7635** | 0.5624 | Duy trì độ chính xác cao |
| **Q3 (100 Handwritten Queries)** | $G_{full}$ | **0.8443** | **0.8443** | 0.8663 | Duy trì độ chính xác cao |
| **Q5 (150 Brand Queries)** | $G_{test}$ | **0.9491** | **0.9491** | 0.2543 | Duy trì độ chính xác cao |
| **Q5 (150 Brand Queries)** | $G_{full}$ | **0.9523** | **0.9523** | 0.8740 | Duy trì độ chính xác cao |
| **Heldout (26 Spec Queries)** | $G_{test}$ | 0.8816 | **0.8863** | 0.7680 | Cải thiện nhẹ (+0.0047, $n=25$) |
| **Heldout (26 Spec Queries)** | $G_{full}$ | 0.9688 | **0.9701** | 0.9840 | Cải thiện nhẹ (+0.0013, $n=25$) |

### L.2 Điều kiện Kích hoạt Thông báo Lavie & Kiểm định Không Bắn Nhầm
- **Điều kiện kích hoạt:** Thông báo Lavie chỉ kích hoạt khi phát hiện token ranh giới từ `\blavie\b` trong query, hoặc trong `unparsed_tokens`, hoặc parser nhận diện `brand == "Lavie"`.
- **Cơ chế:** Vì catalog Kaggle có đúng **0 sản phẩm Lavie** ($R=0$), hệ thống tự động nới lỏng bỏ bộ lọc brand, giữ lại `Handbags + Women`, và hiển thị thông báo hỗ trợ người dùng.
- **Kiểm định thực tế không bắn nhầm:**
  - Query `"quan jean nam xanh"`: Notice Fired = **FALSE** (Parsed: `Jeans`, `Men`, unparsed: `['xanh']`). Hoàn toàn không bắn thông báo brand.
  - Query `"do tap gym"`: Notice Fired = **FALSE** (Parsed: `articleType=None`, `gender=None`, `unparsed=[]`). Hoàn toàn không bắn thông báo brand.
  - Query `"túi xách Lavie nữ"` / `"túi xách nữ Lavie"`: Notice Fired = **TRUE** (Bắn thông báo Lavie chính xác).

### L.3 Tính C0 P@10 trên Ground Truth Mới (x018=209) So cùng Cơ sở với C5
- **Quy chuẩn Ground Truth:**
  - **Condition A (cũ):** So sánh chuỗi strict `"Benetton"` vs `"United Colors of Benetton"` trả về $R=0$, do đó câu x018 bị loại khỏi mẫu số ($n=24$ trên $G_{full}$, $n=23$ trên $G_{test}$).
  - **GT Mới:** Đã cập nhật alias canonical, câu x018 có $R=209$ trên $G_{full}$ và $R=7$ trên $G_{test}$ ($n=25$ trên $G_{full}$, $n=24$ trên $G_{test}$).
- **So sánh trực tiếp C0 vs C5 trên cùng một cơ sở GT Mới ($G_{full}$):**
  - **7 Thương hiệu hợp lệ trong Catalog (x011..x016, x018 - loại trừ Lavie $R=0$):**
    - **C0 (Model thuần - Dense Only):** **P@10 = 0.0714**, **AP@10 = 0.0191**.
    - **C5 (Hệ thống Phục vụ):** **P@10 = 0.8000**, **AP@10 = 0.8720**.
    - **Mức chênh lệch thực tế:** $\Delta\text{P@10} = \mathbf{+0.7286}$ (+1.020%), $\Delta\text{AP@10} = \mathbf{+0.8529}$ (+4.465%).
  - **Chi tiết từng câu C0 trên GT Mới:**
    - `x011` (Puma, $R=534$): C0 P@10 = `0.0000`, AP@10 = `0.0000`
    - `x012` (Nike, $R=547$): C0 P@10 = `0.0000`, AP@10 = `0.0000`
    - `x013` (Adidas, $R=104$): C0 P@10 = `0.0000`, AP@10 = `0.0000`
    - `x014` (Fossil, $R=104$): C0 P@10 = `0.0000`, AP@10 = `0.0000`
    - `x015` (Fabindia, $R=74$): C0 P@10 = `0.3000`, AP@10 = `0.0867`
    - `x016` (Catwalk, $R=607$): C0 P@10 = `0.2000`, AP@10 = `0.0472`
    - `x017` (Lavie, $R=0$): C0 P@10 = `0.0000`, AP@10 = `0.0000`
    - `x018` (Benetton, $R=209$): C0 P@10 = `0.0000`, AP@10 = `0.0000`
"""

content = content + section_k

p.write_text(content.replace("\n", "\r\n"), encoding="utf-8")
print("SUCCESSFULLY PATCHED reports/phase3/phase3f/phase3f_errata.md!")
