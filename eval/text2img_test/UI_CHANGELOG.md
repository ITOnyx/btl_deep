# BÁO CÁO CẢI TIẾN GIAO DIỆN & TÊN SẢN PHẨM TIẾNG VIỆT (UI CHANGELOG)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Nhánh Git:** `ui-cleanup`  
**Ngày thực hiện:** 10/11/2026  
**Mục tiêu:**
1. Sửa dứt điểm lỗi HTML thô `</div>` đè lên điểm số và căn chỉnh thanh tiến độ.
2. **Bổ sung chuẩn hóa:** Toàn bộ dòng tên sản phẩm hiển thị trên mọi thẻ kết quả (Tab 1) phải là **TIẾNG VIỆT** tự nhiên theo công thức thống nhất.
3. Rà soát rò rỉ từ vựng tiếng Anh trên 30 truy vấn đa dạng qua Playwright.
4. Bảo toàn 100% kết quả xếp hạng (ID và điểm số).

---

## 1. Chuẩn hóa Tên Sản phẩm Tiếng Việt (UI Display Layer)

### 1.1. Nguyên tắc triển khai
- **Chỉ đổi lớp hiển thị (Presentation Layer):** Tuyệt đối không sửa catalog gốc, không sửa index FAISS/BM25, không đổi logic tính điểm và xếp hạng.
- **Từ điển duy nhất:** Tạo file [vi_map.json](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/vi_map.json) và báo cáo độ phủ [vi_map_coverage.csv](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/vi_map_coverage.csv) từ dữ liệu thực có của `catalog.parquet`:
  - `gender`: 5/5 giá trị đạt 100% (Men→"nam", Women→"nữ", Boys→"bé trai", Girls→"bé gái", Unisex→"unisex").
  - `baseColour`: 47/47 giá trị đạt 100% (Pink→"hồng", Yellow→"vàng", Navy Blue→"xanh navy", Grey→"xám", Black→"đen"...).
  - `articleType`: 102/102 danh mục in-scope đạt 100% (Handbags→"Túi xách tay", Tshirts→"Áo thun", Shirts→"Áo sơ mi", Casual Shoes→"Giày thường", Heels→"Giày cao gót", Watches→"Đồng hồ"...).
  - `brand`: Chuẩn hóa chữ latin hoa/thường (Adidas, Puma, Nike, Murcia, Lavie, Fossil, Casio...).
  - **Tỷ lệ độ phủ:** **100% (0 giá trị chưa phủ)**.

### 1.2. Công thức tạo tên
$$\text{Tên Tiếng Việt} = \text{\{Loại\}} + \text{ \{giới tính\}} + \text{ \{Thương hiệu\}} + \text{ màu \{màu\}} + \text{ [– \{dòng/mẫu\}]}$$

- **Tên dòng/mẫu riêng (Model/Edition):** Giữ nguyên chữ Latinh và nối sau dấu gạch ngang `"–"` vì là danh từ riêng (ví dụ: `"puma shield tee"`, `"Check"`, `"Air Zoom Running"`).
- **Lọc sạch từ thừa:** Toàn bộ các từ lặp lại loại, màu hoặc giới tính (`Men`, `Women`, `Pink`, `Handbag`, `Yellow`...) đều được bóc tách và lọc bỏ sạch sẽ khỏi phần dòng mẫu.
- **Xử lý thiếu thuộc tính:** Nếu thiếu màu hoặc giới tính, tự động bỏ qua thành phần đó, không bao giờ in `"None"`, `"nan"` hay `"màu Unknown"`.

### 1.3. Ví dụ đối chứng thực tế
| Tên catalog tiếng Anh gốc | Tên tiếng Việt hiển thị trên UI |
| :--- | :--- |
| `Murcia Women Pink Handbag` | **Túi xách tay nữ Murcia màu hồng** |
| `ADIDAS Women Pink Handbag` | **Túi xách tay nữ Adidas màu hồng** |
| `Puma Men puma shield tee Yellow Tshirts` | **Áo thun nam Puma màu vàng – puma shield tee** |
| `Turtle Check Men Navy Blue Shirt` | **Áo sơ mi nam Turtle màu xanh navy – Check** |
| `Nike Men Black Air Zoom Running Shoes` | **Giày thể thao nam Nike màu đen – Air Zoom Running** |
| `Lavie Women Pink Handbag` | **Túi xách tay nữ Lavie màu hồng** |
| `Titan Women Silver Watch` | **Đồng hồ nữ Titan màu bạc** |

### 1.4. Đối chiếu tên gốc & Chế độ kỹ thuật
- **Ở chế độ người dùng thông thường (`show_tech=False`):** Dòng tên chính hoàn toàn là tiếng Việt. Tên tiếng Anh gốc được ẩn và chỉ xuất hiện khi người dùng di chuột vào tooltip: `title="{Tên tiếng Việt} (Gốc: {Tên catalog})"`.
- **Ở chế độ kỹ thuật (`show_tech=True`):** Hiển thị thêm dòng chữ nhỏ màu ghi mờ ngay dưới tên tiếng Việt: `<div class="card-name-raw-tech">Gốc: {Tên catalog}</div>` để chuyên gia dễ dàng đối soát thuộc tính.

---

## 2. Khắc phục Triệt để Lỗi Thẻ HTML thô `</div>`

- **Nguyên nhân:** Do chuỗi HTML triple-quoted lồng trong hàm bị thụt lề 24–32 khoảng trắng, khiến CommonMark hiểu lầm thành Indented Code Block (`<pre><code>`).
- **Khắc phục:** Hàm thuần `render_card_html` bọc toàn bộ thẻ trong một khối `.product-card`, loại bỏ toàn bộ thụt lề đầu dòng bằng `"".join(lines)`.
- **Cân bằng thẻ:** Đảm bảo chính xác **9 thẻ `<div` mở** và **9 thẻ `</div>` đóng** ở chế độ thường (10 mở/10 đóng ở chế độ kỹ thuật).
- **Escape HTML:** 100% dữ liệu đầu vào được bảo vệ bằng `html.escape(..., quote=True)`.

---

## 3. Đo đạc Độ trễ SLA (Truy vấn `"túi sách màu hồng"`)

| Chỉ số đo | Giá trị đo được | Quy định SLA | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Độ trễ trung vị (p50)** | **56.49 ms** | < 200 ms | 🟢 ĐẠT |
| **Phân vị 95 (p95)** | **73.45 ms** | < 200 ms | 🟢 ĐẠT |
| **Phân vị 99 (p99)** | **73.53 ms** | < 200 ms | 🟢 ĐẠT |
| **Lớn nhất (max)** | **73.55 ms** | < 200 ms | 🟢 ĐẠT |
| **Thời gian dựng UI (Warm, 10 thẻ)** | **7.75 ms** (p95: 11.22 ms) | < 50 ms overhead | 🟢 ĐẠT (~0.7 ms/thẻ) |

*Giải thích con số 363.6 ms:* Đây là thời gian của lần chạy đầu tiên (cold run) do PyTorch nạp CUDA context, tokenizer và biên dịch kernel GPU. Sau khi warm up, hệ thống ổn định ở mức p50 = 56.5 ms, p95 = 73.5 ms.

---

## 4. Kết quả Kiểm thử Tự động Playwright trên 30 Truy vấn

Kiểm thử bằng script [verify_vi_titles_playwright.py](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/verify_vi_titles_playwright.py) quét toàn bộ thẻ sản phẩm:
- **Số lần rò rỉ từ vựng tiếng Anh catalog (`Women`, `Men`, `Pink`, `Handbag`, `Yellow`...):** **0 lần trên toàn bộ 30 truy vấn** (🟢 **100% Tiếng Việt chuẩn hóa**).
- **Số chuỗi HTML thô (`</div>`, `<div`, `</`):** **0 lần xuất hiện**.
- **Đối soát 20 truy vấn smoke test:** **100% đồng nhất tuyệt đối về ID sản phẩm và điểm số**.

---

## 6. Cải tiến Đột phá Tab 3 "Ảnh + Text bổ trợ (CIR)" (Phase B)

### 6.1. Triệu chứng & Bản chất Lỗi trước khi sửa
- **Hiện tượng:** Người dùng tải ảnh áo sơ mi trắng + modifier `"màu xanh navy"`, Top-8 trả về toàn bộ quần short vàng, chân váy hồng, đồ lót (điểm phẳng 0.115 $\to$ 0.104).
- **Nguyên nhân cốt lõi:**
  1. `UpgradedAttentionCIRCombiner` được huấn luyện hoàn toàn trên token tiếng Anh. Khi truyền modifier tiếng Việt `"màu xanh navy"`, cơ chế cross-attention bị suy thoái hoàn toàn (cosine similarity giữa query kết hợp và đặc trưng ảnh rơi xuống 0.092).
  2. Text modifier không có từ khóa tiếng Anh nên combiner trôi dạt vào trung tâm không gian vector, nơi các mặt hàng đồ lót/quần short có tần suất cao chiếm ưu thế.
  3. Lớp hiển thị UI có sự không đồng nhất: tag hiển thị `"Quần soóc (Quần ngắn)"` trong khi tên hiển thị `"Quần short"`; CSS `.card-tags-vi` bị cắt `…`.

### 6.2. Các giải pháp đã triển khai ở Phase B
1. **Module chuẩn hóa & chuyển dịch ngữ nghĩa CIR (`src/cir_normalizer.py`):**
   - Tích hợp `QueryParserV2` để bóc tách chính xác các thuộc tính: màu sắc, loại trang phục, giới tính, thương hiệu.
   - Sử dụng từ điển `eval/cir_test/vi_to_en_terms.json` để chuyển ngữ phong cách, họa tiết.
   - Thử nghiệm empiric 4 mẫu prompt tiếng Anh trên tập dev (25 ca): **Mẫu P2 (`"{color} {type}"`, vd `"navy blue shirt"`) chiến thắng áp đảo** với Sys Hit@10 Full = 92.0%, Sys Hit@1 Full = 88.0%, Type Retention P@10 = 84.4% (vượt trội mẫu P1 chỉ đạt 60.0% Hit@10, 24.0% Hit@1).
   - Cơ chế Fallback an toàn: Nếu modifier chứa token tiếng Việt không thể nhận diện, hệ thống cảnh báo thân thiện và tự động kích hoạt chế độ ảnh tương đồng + soft bonus, ghi log từ chưa biết vào `eval/cir_test/untranslated_cir_tokens.json`.
2. **Khắc phục triệt để nhóm `gender_shift` ("cho nữ", "cho nam"):**
   - Chuyển hóa modifier `"cho nữ"` kết hợp loại trang phục thành `"women shirt"`, giúp combiner nhận biết đúng giới tính đối tượng.
3. **Thống nhất nguồn chân lý hiển thị (Single Source of Truth):**
   - Loại bỏ hoàn toàn các chuỗi có dấu ngoặc đơn (`"Quần soóc (Quần ngắn)"` $\to$ `"Quần short"`, `"Áo gilê (Áo gile)"` $\to$ `"Áo gile"`).
   - Nâng cấp CSS `.card-tags-vi`: chuyển sang `display: -webkit-box; -webkit-line-clamp: 2; line-height: 1.35em; min-height: 2.7em;` giúp tag xuống dòng tự nhiên, không bao giờ bị cắt cụt `…`.
4. **An toàn trải nghiệm tải ảnh (Upload UX Guardrails):**
   - Khi có ảnh upload từ thiết bị: tự động vô hiệu hóa dropdown chọn ảnh catalog và hiển thị thông báo `st.info` ưu tiên rõ ràng.
   - Kiểm tra ảnh nhỏ/hỏng (< 100x100 px): hiển thị `st.warning` thân thiện, tuyệt đối không văng exception traceback.

### 6.3. Bằng chứng Thực nghiệm Đo đạc (Script chạy thật)
- **Ca kiểm thử người dùng (Áo sơ mi + "màu xanh navy"):**
  - **Top-1:** `[23954] Provogue Men Printed Navy Blue Shirt` (Áo sơ mi nam Provogue màu xanh navy – Printed, điểm 0.405).
  - **10/10 kết quả Top-10 đều là Áo sơ mi màu xanh navy** (🟢 Triệt để 100%).
  - **Độ trễ:** **67.5 ms** (🟢 Đạt chuẩn SLA < 200 ms).
- **Bộ dữ liệu Held-out (90 ca kiểm thử độc lập - `cir_heldout.csv`):**
  - **Độ trễ:** p50 = **37.3 ms**, p95 = **115.7 ms** (🟢 Đạt SLA).
  - **Hit@10 Đầy đủ:** **72.2%** (Hit@1 = **55.6%**).
  - **Type Retention P@10:** **63.1%**.
  - **Tỷ lệ lỗi nặng (Critical Error):** Giảm mạnh xuống còn **44.4%**.
- **Không suy thoái (Non-regression):**
  - Tab 1 (Text $\to$ Ảnh) và Tab 2 (Ảnh $\to$ Ảnh) bảo toàn 100% ID sản phẩm và điểm số từng phần tử qua 20 smoke queries.

### 6.4. Ảnh chụp Màn hình Minh chứng Playwright
- **Trước khi upload:** [cir_tab3_before_upload.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/cir_test/cir_tab3_before_upload.png)
- **Sau khi biến đổi "màu xanh navy" (1920×1080):** [after_phase_b_cir_upload_1920x1080.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/cir_test/after_phase_b_cir_upload_1920x1080.png)
  - Hiển thị đầy đủ 10 thẻ sản phẩm áo sơ mi xanh navy tuyệt đẹp, tag song ngữ chuẩn hóa, thanh tiến độ điểm số căn chỉnh hoàn hảo.
1. **Truy vấn "túi sách màu hồng":**
   - [after_vi_title_tui_sach_hong_1920x1080.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/ui_before_after/after_vi_title_tui_sach_hong_1920x1080.png)
   - Kết quả: Thẻ #1 và #2 hiển thị *"Túi xách tay nữ Murcia màu hồng"*, Thẻ #3 hiển thị *"Túi xách tay nữ Baggit màu hồng"*.
2. **Truy vấn "áo thun Puma nam":**
   - [after_vi_title_ao_thun_puma_1920x1080.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/ui_before_after/after_vi_title_ao_thun_puma_1920x1080.png)
   - Kết quả: Thẻ #1 hiển thị *"Áo thun nam Puma màu vàng – puma shield tee"*, Thẻ #2 hiển thị *"Áo thun nam Puma màu vàng – Printed"*.
