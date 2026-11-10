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

## 5. Ảnh chụp Màn hình Đối chứng (1920×1080)
1. **Truy vấn "túi sách màu hồng":**
   - [after_vi_title_tui_sach_hong_1920x1080.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/ui_before_after/after_vi_title_tui_sach_hong_1920x1080.png)
   - Kết quả: Thẻ #1 và #2 hiển thị *"Túi xách tay nữ Murcia màu hồng"*, Thẻ #3 hiển thị *"Túi xách tay nữ Baggit màu hồng"*.
2. **Truy vấn "áo thun Puma nam":**
   - [after_vi_title_ao_thun_puma_1920x1080.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/ui_before_after/after_vi_title_ao_thun_puma_1920x1080.png)
   - Kết quả: Thẻ #1 hiển thị *"Áo thun nam Puma màu vàng – puma shield tee"*, Thẻ #2 hiển thị *"Áo thun nam Puma màu vàng – Printed"*.
