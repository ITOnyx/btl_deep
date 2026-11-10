# NHẬT KÝ THAY ĐỔI GIAO DIỆN (UI_CHANGELOG.md)
**Dự án:** SigLIP E-commerce Visual Search (Hoc_sau_ver4)  
**Nhánh git:** `ui-cleanup` | **Thời điểm thực hiện:** 2026-11-10  
**Bản sao lưu gốc:** `_backup_ui/20261110_194135/main.py`

---

## 1. TỔNG QUAN CÁC THAY ĐỔI & LÝ DO THỰC HIỆN

| Thành phần | Trước khi sửa | Sau khi sửa | Lý do & Chuẩn mực |
| :--- | :--- | :--- | :--- |
| **Chế độ hiển thị** | Hiển thị tất cả thông số kỹ thuật (C5/C0, SHA-256, GPU FP16, GT policy, v.v.) trực tiếp trên giao diện chính và sidebar. | Tách làm 2 chế độ: Mặc định là **Chế độ Người dùng** tinh gọn; bổ sung công tắc **"Chế độ kỹ thuật"** trong Sidebar. Thông tin kiểm toán Luật 7 luôn xem được qua 1 lần click expander. | Giúp người dùng thông thường tập trung mua sắm và tra cứu thời trang; chuyên gia/kỹ sư kiểm toán vẫn xem đủ thông số khi cần. |
| **Tiêu đề & Đầu trang** | Tiêu đề chìm trong nền tối; lộ mã kỹ thuật `( flow: text_to_image )`; lạm dụng emoji (🛍️🔍🖼️🎨📊🎯🛡️✨...). | Tiêu đề màu `#F8FAFC`, tương phản cao đạt chuẩn **WCAG AA**; bỏ hoàn toàn `( flow: text_to_image )`; bỏ emoji rác, chỉ giữ 1 biểu tượng trang nhã cho tên tab. | Tăng tính chuyên nghiệp, thẩm mỹ và khả năng tiếp cận (Accessibility WCAG AA). |
| **Điều khiển tìm kiếm (Tab 1)** | Nhãn radio rất dài ("Hệ thống (Parser + Metadata + BM25) [Cấu hình C5 Hybrid]"); khối mô tả xanh lá chữ dày đặc; xung đột ngầm giữa radio Tab 1 và selectbox Sidebar. | Radio rút gọn thành **"Hệ thống đầy đủ"** / **"Model thuần"**; chi tiết đưa vào tooltip `help`; khối mô tả thu gọn 1 dòng caption; đưa cấu hình nâng cao (C2, C4, $\gamma$) vào `Tùy chọn nâng cao` (expander). Đồng bộ hóa 1 nguồn sự thật. | Giao diện gọn gàng, loại bỏ xung đột logic giữa Tab 1 và Sidebar. |
| **Gợi ý mẫu (Thử ngay)** | 7 nút xếp hàng ngang kích thước không đều, bấm vào chỉ điền chữ nhưng không tự tìm kiếm. | 7 nút chia cột đều đặn với nhãn **"Thử ngay:"**; bấm nút là tự động điền và **tự động kích hoạt tìm kiếm ngay**. | Nâng cao trải nghiệm người dùng (1-click search). |
| **Thanh tìm kiếm & Top-K** | Ô nhập và nút "Tìm kiếm" màu đỏ rực tách làm 2 hàng riêng biệt; Top-K bị giấu trong sidebar đã thu gọn. | Đưa ô nhập văn bản, ô chọn **Top-K (5, 10, 20, 50)** và nút **Tìm kiếm** (màu xanh navy dịu) lên **cùng một hàng ngang**; hỗ trợ nhấn phím Enter để tìm kiếm. | Thao tác tập trung, tiện lợi, đúng thói quen của người dùng eCommerce. |
| **Thẻ sản phẩm (Grid)** | Chiều cao thẻ lệch nhau do tên 1 dòng vs 2 dòng; lặp dòng `Catalog: ...`; điểm số hiển thị dạng mã monospace thô `0.0311`; tag hiển thị tiếng Anh lộn xộn (Tshirts • Yellow • Men). | Cố định chiều cao thẻ; tên tiếng Việt hiển thị **tối đa 2 dòng kèm dấu ba chấm (`...`)** và tooltip hiển thị tên đầy đủ; bỏ dòng `Catalog: ...` thừa (đưa vào tooltip); 3 tag chuẩn hóa hoàn toàn sang **tiếng Việt** (`vi_map.json`: Loại • Màu • Giới tính); điểm làm tròn 3 chữ số (`0.031`) kèm **thanh tiến độ mini trực quan**. | Thẻ sản phẩm thẳng hàng tăm tắp, cân đối, thuần Việt và hiện đại. |
| **Thẻ SLA & Pipeline** | Hàng 6 thẻ nền trắng chữ xám mờ lệch tông với theme tối, chiếm diện tích lớn ngay trên kết quả. | Chuyển thành khối **"Thông tin hiệu năng & SLA chi tiết"** (expander) đặt **SAU** kết quả tìm kiếm, thiết kế nền tối `#1E293B`, chữ sáng rõ ràng. | Ưu tiên hiển thị sản phẩm trước; chỉ mở xem hiệu năng kỹ thuật khi cần. |
| **Cấu hình Streamlit Chrome** | Hiện thanh "Deploy" và menu ba chấm của Streamlit gây xao nhãng. | Thêm `.streamlit/config.toml` cấu hình `toolbarMode="minimal"`, ẩn thanh deploy và thiết lập bảng màu dark theme WCAG AA. | Giao diện chuẩn sản phẩm đóng gói (production-ready). |

---

## 2. DANH SÁCH TỆP THAY ĐỔI (DIFF)
1. `app/main.py`: Tái cấu trúc lớp giao diện người dùng, áp dụng CSS Dark Mode hiện đại, chuẩn hóa thẻ sản phẩm, đồng bộ điều khiển.
2. `.streamlit/config.toml`: Tệp cấu hình giao diện Streamlit (toolbarMode, theme tối).
3. `eval/text2img_test/ui_before_after/`: 16 ảnh chụp màn hình kiểm chứng trước và sau khi làm sạch ở cả 2 độ phân giải (1920×1080 và 1280×720).

---

## 3. HƯỚNG DẪN HOÀN TÁC (ROLLBACK)
Nếu muốn hoàn tác giao diện về trạng thái trước khi làm sạch:
```powershell
# Cách 1: Chuyển lại nhánh master
git checkout master

# Cách 2: Phục hồi tệp từ thư mục sao lưu
Copy-Item -Path _backup_ui/20261110_194135/main.py -Destination app/main.py -Force
Remove-Item -Path .streamlit/config.toml -Force
```

---

## 4. KẾT QUẢ KIỂM ĐỊNH SAU KHI SỬA (VERIFICATION)
1. **Đối soát 20 truy vấn đại diện (Smoke Test):**
   - Chạy kiểm thử tự động so sánh trực tiếp với `eval/text2img_test/results_raw.parquet`.
   - Kết quả: **20/20 câu truy vấn (100%) có danh sách Top-K ID và Điểm số TRÙNG KHỚP TUYỆT ĐỐI** so với trước khi sửa UI (không lệch dù chỉ 1 ID hay 1 chữ số thập phân).
2. **Độ bền các Tab 2, Tab 3, Tab 4:**
   - Đã kiểm tra mở lần lượt Tab 2 (Ảnh → Ảnh), Tab 3 (CIR Đa phương thức) và Tab 4 (Đối soát SLA & GT).
   - Tất cả các tab hoạt động bình thường, bố cục không bị vỡ.
3. **Kiểm toán Console Trình duyệt (Playwright):**
   - Ghi nhận: **0 lỗi console, 0 pageerror**.

---

## 5. ĐỐI CHIẾU DANH MỤC YÊU CẦU MỤC 10

- [x] **A. Hai chế độ hiển thị:** Mặc định "Người dùng", công tắc "Chế độ kỹ thuật" trong sidebar; khối kiểm toán Luật 7 truy cập 1 chạm.
- [x] **B. Đầu trang:** Tiêu đề sáng WCAG AA; bỏ `flow: text_to_image`; bỏ emoji thừa, biểu tượng tab nhất quán.
- [x] **C. Điều khiển tìm kiếm:** Radio rút gọn "Hệ thống đầy đủ" / "Model thuần"; khối mô tả 1 dòng; gợi ý mẫu xếp lưới đều 7 cột bấm tự tìm; ô nhập + Top-K + nút tìm kiếm cùng 1 hàng; tùy chọn nâng cao trong expander; đồng bộ điều khiển.
- [x] **D. Thẻ sản phẩm:** Chiều cao cố định đồng nhất; tên tiếng Việt 2 dòng cắt bằng `...` kèm tooltip; bỏ dòng Catalog trùng; 3 tag tiếng Việt (Loại • Màu • Giới tính); điểm làm tròn 3 chữ số kèm thanh tiến độ; ID để mờ ở chế độ kỹ thuật.
- [x] **E. Bố cục & Trạng thái:** Lưới 5 cột co giãn; spinner khi tìm; thông báo thân thiện khi rỗng, không hiện traceback; hiện thời gian truy vấn (ms) nhỏ cạnh tiêu đề.
- [x] **F. Thẻ SLA/pipeline:** Chuyển vào khối expander sau kết quả, nền tối đồng bộ, hiện số đo thật.
- [x] **G. Streamlit chrome:** `toolbarMode="minimal"`, ẩn Deploy/menu, layout="wide".
- [x] **H. Nhất quán:** Toàn bộ nhãn tiếng Việt chuẩn; giữ dark theme; không reload model khi tương tác; độ trễ UI thêm = 0 ms.
- *Không có mục nào trong Mục 10 bị bỏ sót.*
