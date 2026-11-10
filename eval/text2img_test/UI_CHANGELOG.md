# BÁO CÁO CẢI TIẾN GIAO DIỆN & SỬA LỖI THẺ SẢN PHẨM (UI CHANGELOG)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Nhánh Git:** `ui-cleanup`  
**Ngày thực hiện:** 10/11/2026  
**Mục tiêu:** Sửa dứt điểm lỗi HTML thô `</div>` đè lên điểm số, chuẩn hóa thẻ sản phẩm bằng hàm thuần `render_card_html`, đối soát độ trễ SLA và bảo toàn 100% độ chính xác mô hình.

---

## 1. Triệu chứng & Nguyên nhân Gốc rễ (Root Cause)

### 1.1. Triệu chứng
- Ở mỗi thẻ sản phẩm (Tab 1, Text→Ảnh), ngay dòng điểm số (ví dụ `0.031` + thanh tiến độ) xuất hiện một khung code block màu tối chứa chữ `</div>` cùng biểu tượng copy của Streamlit, đè trực tiếp lên huy hiệu điểm.
- Lỗi xuất hiện ở cả 10 thẻ kết quả của truy vấn `"túi sách màu hồng"` và lặp lại ở các truy vấn khác.

### 1.2. Nguyên nhân kỹ thuật chính xác
- **Thụt lề $\ge 4$ khoảng trắng trong chuỗi Markdown:** Trong `app/main.py` (trước khi sửa), khối HTML thẻ sản phẩm được viết dưới dạng multiline f-string lồng trong vòng lặp hiển thị. Các dòng con như:
  ```html
                              <div class="progress-bar-container">
                                  <div class="progress-bar-fill" style="width: 77%;"></div>
                              </div>
                          </div>
                      </div>
  ```
  có độ thụt lề từ **24 đến 32 khoảng trắng** ở đầu dòng. Theo đặc tả Markdown (CommonMark), bất kỳ dòng nào bắt đầu bằng $\ge 4$ khoảng trắng sẽ tự động bị phân tích cú pháp thành **Indented Code Block** (`<pre><code>...</code></pre>`).
- **Phân mảnh lời gọi Markdown:** Thẻ sản phẩm trước đó bị chia cắt thành 4 lời gọi `st.markdown` rời rạc xen kẽ với 1 lời gọi `st.image`, không có container HTML bao bọc thống nhất (`.product-card`), khiến layout flexbox không kiểm soát được chiều cao và vị trí các thành phần.

---

## 2. Giải pháp Kỹ thuật & Cải tiến Thực hiện

### 2.1. Hàm thuần `render_card_html(prod, show_tech)`
- **Đóng gói container duy nhất:** Mỗi thẻ sản phẩm là một khối HTML độc lập `<div class="product-card">...</div>`.
- **Triệt tiêu thụt lề:** Toàn bộ chuỗi HTML được ghép bằng `"".join(lines)`, đảm bảo **không có bất kỳ khoảng trắng thụt lề nào $\ge 4$ spaces** và **không có dòng trống**.
- **Cân bằng thẻ mở/đóng tuyệt đối:** Hàm được kiểm tra tự động qua unit test: chính xác **9 thẻ `<div` mở** và **9 thẻ `</div>` đóng**.
- **Bảo mật & Tránh vỡ layout:** Tất cả dữ liệu động (`name_vi`, `name`, `brand`, `articleType`, `baseColour`, `gender`, `item_id`) đều được xử lý qua `html.escape(..., quote=True)`. Các ký tự nhạy cảm như `<`, `>`, `&`, `'`, `"` không thể làm vỡ cấu trúc thẻ.
- **Một lời gọi duy nhất:** Trong `render_product_grid`, mỗi thẻ chỉ sử dụng đúng **MỘT lời gọi `st.markdown(card_html, unsafe_allow_html=True)`**.

### 2.2. Tối ưu ảnh với In-Memory LRU Base64 Cache
- Triển khai `@lru_cache(maxsize=4096) def get_image_base64_data_uri(path_str: str) -> str`.
- Ảnh sản phẩm được đọc trực tiếp và nhúng an toàn qua Base64 Data URI trong thẻ `<img class="card-img" ... loading="lazy"/>`.
- Thẻ có container cố định `.card-img-wrap` (chiều cao 190px, background `#0B132B`, `object-fit: contain`). Nếu thiếu ảnh, hiển thị placeholder sạch đẹp thay vì làm hỏng layout.

### 2.3. Căn chỉnh layout & Responsive (1920×1080 và 1280×720)
- Container `.product-card` dùng `display: flex; flex-direction: column; justify-content: space-between; min-height: 380px;`.
- Phần điểm số `.score-badge` và thanh tiến độ `.progress-bar-container` được đưa vào `.card-meta-row` với `margin-top: auto`. Điều này đảm bảo huy hiệu điểm và thanh tiến độ luôn **thẳng hàng tăm tắp** giữa các thẻ trong cùng một hàng, bất kể tiêu đề sản phẩm dài 1 hay 2 dòng.

---

## 3. Đo đạc Độ trễ Độc lập (SLA Benchmark: `"túi sách màu hồng"`)

Được thực hiện độc lập bởi script [benchmark_card_latency.py](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/benchmark_card_latency.py) và lưu tại [latency_benchmark_tui_sach_hong.json](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/latency_benchmark_tui_sach_hong.json):

### 3.1. Bản chất con số hiển thị trên tiêu đề kết quả (ví dụ 363.6 ms)
- Con số trên tiêu đề lấy trực tiếp từ `meta["breakdown"]["total_ms"]`, đo **thời gian thực thi E2E của backend `service.search_text()`** (gồm QueryParser V2, lọc ứng viên đa tầng, SigLIP Text Tower encode và Hybrid/BM25/Top-K).
- Ở lần chạy đầu tiên (cold run) ngay sau khi khởi động app, PyTorch nạp CUDA context, tokenizer và biên dịch kernel GPU gây ra độ trễ khởi động ~350-380 ms.

### 3.2. Kết quả 20 lần chạy ấm liên tiếp (Warm Runs)
| Chỉ số đo | Giá trị đo được | Quy định SLA | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Độ trễ trung vị (p50)** | **56.49 ms** | < 200 ms | 🟢 ĐẠT |
| **Phân vị 95 (p95)** | **73.45 ms** | < 200 ms | 🟢 ĐẠT |
| **Phân vị 99 (p99)** | **73.53 ms** | < 200 ms | 🟢 ĐẠT |
| **Lớn nhất (max)** | **73.55 ms** | < 200 ms | 🟢 ĐẠT |
| **Nhỏ nhất (min)** | **45.81 ms** | < 200 ms | 🟢 ĐẠT |
| **Trung bình (mean)** | **59.16 ms** | < 200 ms | 🟢 ĐẠT |

**Phân rã trung bình từng thành phần (Warm):**
1. `QueryParser V2`: 1.30 ms (p95: 1.75 ms)
2. `Lọc ứng viên L0..L6`: 0.26 ms (p95: 0.39 ms)
3. `SigLIP Text Tower GPU`: 51.13 ms (p95: 62.96 ms)
4. `Hybrid/BM25/TopK`: 6.40 ms (p95: 8.13 ms)

### 3.3. Thời gian dựng giao diện UI (`render_card_html` cho 10 thẻ)
- **Lần đầu (Cold render - đọc đĩa 10 ảnh JPEG gốc):** 97.36 ms (trung bình ~9.7 ms/ảnh).
- **Lần ấm (Warm render - có cache Base64):** p50 = **7.75 ms**, p95 = **11.22 ms** (trung bình **~0.7 ms/thẻ**).
- **Kết luận:** Lớp UI mới chỉ tốn ~7.75 ms cho toàn bộ 10 thẻ, hoàn toàn không gây trễ thêm quá 50 ms so với phiên bản gốc.

---

## 4. Kết quả Kiểm thử Tự động & Nghiệm thu

1. **Kiểm thử Playwright ([verify_playwright_fix.py](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/verify_playwright_fix.py)):**
   - Quét toàn bộ DOM và text của trang web: **0 ký tự `</div>`, `<div`, `</span>` hay `</` nào xuất hiện dưới dạng văn bản**.
   - Cả 10 thẻ đều có đầy đủ huy hiệu điểm (`.score-badge`), thanh tiến độ (`.progress-bar-fill`) và ảnh hiển thị rõ ràng.
   - Kiểm tra trên 7 câu gợi ý mẫu: Toàn bộ **7/7 câu đạt chuẩn 100% (0 lỗi HTML thô)**.
2. **Kiểm thử các ca biên ([test_card_edge_cases.py](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/test_card_edge_cases.py)):**
   - Tên cực dài (>200 ký tự): Thẻ tự co giãn hoặc cắt bằng ellipsis, không vỡ layout.
   - Tên chứa XSS/HTML (`<script>`, `<b>`, `&`, `'`): Thoát mã an toàn 100%.
   - Thương hiệu rỗng: Hiển thị nhãn trống ẩn, không xô lệch rank tag.
   - Thiếu ảnh: Hiển thị placeholder chuẩn mực.
   - Chế độ kỹ thuật bật/tắt: ID sản phẩm hiển thị đúng vị trí.
3. **Bảo toàn 100% kết quả mô hình ([verify_post_cleanup.py](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/verify_post_cleanup.py)):**
   - Chạy đối soát lại 20 truy vấn smoke test: **100% khớp tuyệt đối cả về ID sản phẩm và điểm số** so với trước khi sửa. Không có bất kỳ sự thay đổi nào đối với logic tìm kiếm.
4. **Ảnh chụp đối chứng:**
   - 1920×1080: [after_fix_tui_sach_hong_1920x1080.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/ui_before_after/after_fix_tui_sach_hong_1920x1080.png)
   - 1280×720: [after_fix_tui_sach_hong_1280x720.png](file:///d:/Kì 7/Hoc_sau_ver4/eval/text2img_test/ui_before_after/after_fix_tui_sach_hong_1280x720.png)
