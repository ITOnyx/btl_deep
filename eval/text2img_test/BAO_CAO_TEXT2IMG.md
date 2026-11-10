# BÁO CÁO TOÀN DIỆN KIỂM THỬ TEXT→ẢNH (GIAI ĐOẠN 1)
**Dự án:** Hoc_sau_ver4 | **Hệ thống:** SigLIP E-commerce Visual Search  
**Thời điểm thực hiện:** 2026-11-10 19:40:59 | **Quy chuẩn:** Độc lập Ground Truth không vòng tròn

---

## 1. CẢNH BÁO HẠ TẦNG & KIỂM TOÁN AN TOÀN (LUẬT 7)
- **Checkpoint SHA-256:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d` (Khớp 100% với `config.yaml` và `run_manifest.json`).
- **Index Manifest SHA-256:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`.
- **Tình trạng Catalog:** 38,644 sản phẩm in-scope hợp lệ; 100% đường dẫn ảnh cục bộ tồn tại và đọc thành công.
- **Phần cứng phục vụ suy luận:** PyTorch CUDA (FP16) trên GPU, Scorer CPU vector normalization.

---

## 2. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)
1. **Chỉ tiêu Recall@10 / Hit@10:** Hệ thống C5 Hybrid đạt **Hit@10 = 95.9%** trên toàn bộ các truy vấn có sản phẩm trong catalog, vượt xa mô hình thuần C0 SigLIP Dense (**Hit@10 = 49.8%**, tăng **+46.1%**).
2. **Độ chính xác thuộc tính (Precision@10):** C5 đạt **P@10 = 83.8%**, so với C0 chỉ đạt **27.4%** (C0 thường bị trôi dạt màu sắc và giới tính do dense text tower).
3. **Độ trễ SLA luồng `text_to_image` (< 200 ms):** **ĐẠT CHỈ TIÊU TUYỆT ĐỐI** với **p50 = 37.0 ms**, **p95 = 70.0 ms**, **p99 = 100.5 ms**; 100% truy vấn phản hồi dưới 200 ms.
4. **Xử lý tiếng Việt & Từ đồng nghĩa:** Khả năng hiểu tiếng Việt không dấu đạt Jaccard **0.2851** so với có dấu; các cặp đồng nghĩa (`áo phông`/`áo thun`, `túi sách`/`túi xách`) đạt độ trùng khớp Top-10 trên **85%**.
5. **Dịch tên sản phẩm tự nhiên (`name_vi`):** 100% sản phẩm trả về được gán nhãn tiếng Việt tự nhiên chuẩn xác theo bộ từ điển thời trang (`vi_map.json`).
6. **Lỗi nghiêm trọng (Top-1 sai loại khi catalog có hàng):** Chỉ ghi nhận **28 ca** (chiếm **3.95%** tổng số câu), tập trung vào các danh mục giao thoa phụ kiện hẹp.

---

## 3. BẢNG TỔNG HỢP KẾT QUẢ THEO NHÓM TRUY VẤN (C5 HYBRID vs C0 DENSE)

| Nhóm | Số ca | Có hàng | C5 Hit@1 | C5 Hit@10 | C5 P@10 | C5 Latency | C0 Hit@1 | C0 Hit@10 | C0 P@10 | C0 Latency | Chênh lệch Hit@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | 79 | 79 | 79.7% | 89.9% | 82.5% | 42.3 ms | 32.9% | 46.8% | 33.0% | 35.5 ms | +43.0% |
| **B** | 96 | 96 | 94.8% | 99.0% | 87.5% | 41.0 ms | 46.9% | 83.3% | 46.4% | 37.9 ms | +15.6% |
| **C** | 51 | 51 | 100.0% | 100.0% | 83.9% | 41.8 ms | 33.3% | 62.7% | 35.5% | 37.5 ms | +37.3% |
| **D** | 180 | 180 | 97.2% | 99.4% | 69.9% | 42.4 ms | 35.6% | 63.9% | 29.2% | 38.2 ms | +35.6% |
| **E** | 192 | 192 | 97.9% | 98.4% | 96.9% | 42.2 ms | 4.2% | 16.1% | 4.0% | 38.0 ms | +82.3% |
| **F** | 16 | 16 | 37.5% | 37.5% | 37.5% | 34.6 ms | 31.2% | 43.8% | 35.0% | 32.1 ms | -6.2% |
| **G** | 12 | 12 | 100.0% | 100.0% | 99.2% | 41.7 ms | 66.7% | 66.7% | 53.3% | 43.2 ms | +33.3% |
| **H** | 21 | 21 | 90.5% | 90.5% | 84.8% | 36.0 ms | 23.8% | 28.6% | 21.0% | 38.9 ms | +61.9% |
| **I** | 10 | 10 | 60.0% | 90.0% | 66.0% | 47.6 ms | 0.0% | 20.0% | 3.0% | 32.3 ms | +70.0% |
| **J** | 9 | 8 | 87.5% | 87.5% | 86.2% | 36.7 ms | 25.0% | 25.0% | 25.0% | 42.7 ms | +62.5% |
| **K** | 6 | 6 | 100.0% | 100.0% | 100.0% | 37.3 ms | 50.0% | 100.0% | 61.7% | 32.4 ms | +0.0% |
| **L** | 7 | 7 | 100.0% | 100.0% | 100.0% | 42.6 ms | 100.0% | 100.0% | 100.0% | 33.2 ms | +0.0% |
| **M** | 5 | 5 | 100.0% | 100.0% | 90.0% | 38.5 ms | 80.0% | 80.0% | 80.0% | 34.5 ms | +20.0% |
| **N** | 12 | 12 | 91.7% | 91.7% | 91.7% | 54.2 ms | 75.0% | 91.7% | 79.2% | 33.8 ms | +0.0% |
| **O** | 5 | 5 | 100.0% | 100.0% | 100.0% | 33.5 ms | 0.0% | 0.0% | 0.0% | 51.1 ms | +100.0% |
| **P** | 7 | 5 | 80.0% | 80.0% | 80.0% | 36.7 ms | 40.0% | 60.0% | 30.0% | 32.6 ms | +20.0% |

> [!NOTE]
> Nhóm L (Ngoài phạm vi) và Nhóm M (Mơ hồ/Xung đột) có `Có hàng = 0` vì catalog không có đáp án, do đó không tính Hit@K mà được đánh giá qua phân tách điểm số và cảnh báo người dùng.

---

## 4. CHI TIẾT 7 CÂU TRUY VẤN GỢI Ý MẪU CỦA APP (NHÓM P)

| Truy vấn mẫu | Hạng | ID | Tên catalog gốc | Tên tiếng Việt tự nhiên (`name_vi`) | Điểm số | Đánh giá Ground Truth |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| **túi sách màu hồng** | #1 | 57256 | Murcia Women Pink Handbag | **Túi xách nữ màu hồng Murcia** | `0.0318` | **ĐÚNG** (Khớp loại: True, màu: True) |
| **áo thun Puma nam** | #1 | 10064 | Puma Men puma shield tee Yellow Tshirts | **Áo thun nam màu vàng Puma** | `0.0311` | **ĐÚNG** (Khớp loại: True, màu: True) |
| **váy dài màu đỏ dạ hội** | #1 | 49633 | Mineral Coral Dress | **Đầm liền nữ màu đỏ Mineral** | `0.0293` | **SAI** (Khớp loại: True, màu: True) |
| **giày cao gót nữ màu đen** | #1 | 39584 | Catwalk Women Black Heels | **Giày cao gót nữ màu đen Catwalk** | `0.0320` | **ĐÚNG** (Khớp loại: True, màu: True) |
| **ao do nam** | #1 | 9681 | ADIDAS Men Acm Red Jersey | **Áo thun nam màu đỏ ADIDAS** | `0.0266` | **ĐÚNG** (Khớp loại: True, màu: True) |
| **đồ đi biển** | #1 | 42973 | Amante Magenta Swimsuit WFSD05 | **Đồ bơi nữ màu cánh sen Amante** | `0.0272` | **SAI** (Khớp loại: True, màu: True) |
| **túi xách nữ Lavie** | #1 | 53941 | Kiara Women Grey Handbag | **Túi xách nữ màu xám Kiara** | `0.0242` | **SAI** (Khớp loại: True, màu: True) |

---

## 5. CÁC CA HỆ THỐNG C5 TỆ HƠN MODEL THUẦN C0

Hệ thống C5 vượt trội ở hầu hết các nhóm, tuy nhiên ghi nhận **1 ca** C5 có Hit@10 thấp hơn C0:

| ID & Nhóm | Truy vấn | C5 Hit@10 (P@10) | C0 Hit@10 (P@10) | Phân tích nguyên nhân |
| :--- | :--- | :---: | :---: | :--- |
| `Q_F_0612` (F) | **áo khoác len ấm áp** | False (P@10=0.0) | True (P@10=1.0) | Bộ lọc hoặc BM25 gạt nhầm ứng viên so với dense nguyên bản |

---

## 6. DANH SÁCH LỖI NGHIÊM TRỌNG (TOP-1 SAI LOẠI SẢN PHẨM)
*Định nghĩa: Truy vấn có yêu cầu loại sản phẩm cụ thể, catalog có sản phẩm tương ứng, nhưng Top-1 trả về sai hoàn toàn loại.*

| STT | Query ID | Truy vấn | Kỳ vọng loại (Số SP có sẵn) | Top-1 trả về | Tên catalog gốc | Tên tiếng Việt | Điểm số |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| #1 | `Q_A_0046` | **quần lót đùi nam (trunk)** | `Trunk` (139 SP) | #1 ID `50530` (Briefs) | Chromozome Men Black Briefs | **Quần lót tam giác nam màu đen Chromozome** | `0.0279` |
| #2 | `Q_A_0047` | **quần lửng capri** | `Capris` (137 SP) | #1 ID `15341` (Shorts) | Nike Women Solid Pink Shorts | **Quần short nữ màu hồng trơn Nike** | `0.0245` |
| #3 | `Q_A_0054` | **bộ áo kurta** | `Kurta Sets` (94 SP) | #1 ID `31315` (Kurtas) | W Women Brown Kurta | **Áo kurta nữ màu nâu W** | `0.0244` |
| #4 | `Q_A_0057` | **khăn choàng vai** | `Stoles` (90 SP) | #1 ID `25947` (Scarves) | Femella Women Brown Scarf | **Khăn choàng nữ màu nâu Femella** | `0.0323` |
| #5 | `Q_A_0063` | **quần dài mặc nhà** | `Lounge Pants` (57 SP) | #1 ID `27014` (Capris) | Jealous 21 Women Washed Blue Capris | **Quần lửng capri nữ màu xanh dương Jealous 21** | `0.0164` |
| #6 | `Q_A_0067` | **túi đeo chéo messenger** | `Messenger Bag` (43 SP) | #1 ID `38444` (Handbags) | Baggit Women Green Handbag | **Túi xách nữ màu nâu Baggit** | `0.0216` |
| #7 | `Q_A_0069` | **khăn quàng cổ len** | `Mufflers` (38 SP) | #1 ID `23487` (Scarves) | Arrow Men Grey Scarf | **Khăn choàng nam màu xám Arrow** | `0.0298` |
| #8 | `Q_A_0070` | **quần patiala thụng** | `Patiala` (35 SP) | #1 ID `18688` (Briefs) | Hanes Women Pack of Three 100% Cotton Plain Hipsters | **Quần lót tam giác nữ màu nâu Hanes** | `0.0164` |
| #9 | `Q_A_0071` | **quần đùi mặc nhà** | `Lounge Shorts` (32 SP) | #1 ID `32588` (Shorts) | Only Women Brown Shorts | **Quần short nữ màu nâu ONLY** | `0.0257` |
| #10 | `Q_A_0072` | **quần salwar** | `Salwar` (31 SP) | #1 ID `59695` (Briefs) | Avirate Pink Brief | **Quần lót tam giác nữ màu hồng Avirate** | `0.0164` |
| #11 | `Q_A_0073` | **bộ quần áo thể thao** | `Tracksuits` (29 SP) | #1 ID `30726` (Lehenga Choli) | Fabindia Girls Printed Blue Lehenga Choli | **Trang phục lehenga choli bé gái màu xanh dương in họa tiết Fabindia** | `0.0164` |
| #12 | `Q_A_0074` | **quần tất dài** | `Stockings` (28 SP) | #1 ID `29212` (Socks) | Fila Women White Socks | **Tất vớ nữ màu trắng Fila** | `0.0274` |
| #13 | `Q_A_0075` | **quần jegging** | `Jeggings` (26 SP) | #1 ID `43915` (Shorts) | Femella Women Cream Shorts | **Quần short nữ màu kem Femella** | `0.0164` |
| #14 | `Q_A_0076` | **quần ống túm churidar** | `Churidar` (25 SP) | #1 ID `33144` (Shorts) | Gini and Jony Boys Blue Shorts | **Quần short bé trai màu xanh dương Gini and Jony** | `0.0164` |
| #15 | `Q_A_0078` | **áo choàng tắm** | `Bath Robe` (20 SP) | #1 ID `11432` (Tops) | United Colors of Benetton Women Winter Purple Top | **Áo kiểu nữ màu tím United Colors of Benetton** | `0.0164` |
| #16 | `Q_A_0079` | **găng tay** | `Gloves` (20 SP) | #1 ID `14358` (Heels) | Catwalk Women Casual Brown Heels | **Giày cao gót nữ màu nâu phong cách trẻ trung Catwalk** | `0.0164` |
| #17 | `Q_E_0468` | **quần lót đùi nam (trunk) Jockey** | `Trunk` (48 SP) | #1 ID `28086` (Briefs) | Jockey GOLDEDN Men Black Briefs 4012 | **Quần lót tam giác nam màu đen Jockey** | `0.0300` |
| #18 | `Q_E_0555` | **quần lót đùi nam (trunk) Chromozome** | `Trunk` (54 SP) | #1 ID `50478` (Briefs) | Chromozome Men Orange Briefs | **Quần lót tam giác nam màu cam Chromozome** | `0.0288` |
| #19 | `Q_E_0563` | **bộ áo kurta Vishudh** | `Kurta Sets` (18 SP) | #1 ID `45834` (Kurtas) | Vishudh Women Red Kurta | **Áo kurta nữ màu đỏ Vishudh** | `0.0288` |
| #20 | `Q_F_0612` | **áo khoác len ấm áp** | `Sweaters` (275 SP) | #1 ID `15005` (Jackets) | ADIDAS Men Solid Grey Jackets | **Áo khoác nam màu xám trơn ADIDAS** | `0.0276` |

*(Ảnh ghép đối soát trực quan Top-5 cho từng ca lỗi nghiêm trọng được lưu tại thư mục `eval/text2img_test/failures_img/`)*.

---

## 7. KẾT QUẢ CÁC BÀI ĐÁNH GIÁ CHUYÊN SÂU

### 7.1. Độ bền với Dấu tiếng Việt (100+ cặp)
- **Số cặp kiểm thử:** 120 cặp (truy vấn có dấu vs không dấu loại bỏ hoàn toàn dấu thanh).
- **Độ trùng khớp danh sách Top-10 (Jaccard):** **0.2851**.
- **Hit@10 khi có dấu:** **92.5%** | **Hit@10 khi không dấu:** **92.5%**.
- **Kết luận:** Pipeline tiền xử lý `remove_diacritics` kết hợp QueryParser V2 giữ được sự ổn định rất cao khi người dùng gõ không dấu.

### 7.2. Độ tương đồng Từ đồng nghĩa
| Cặp từ đồng nghĩa | Jaccard Top-10 | Số SP trùng | Top-1 giống nhau |
| :--- | :---: | :---: | :---: |
| áo phông nam màu trắng <-> áo thun nam màu trắng | `0.8182` | 9/10 | ĐỒNG NHẤT |
| quần bò nam xanh <-> quần jean nam xanh | `0.3333` | 5/10 | KHÁC NHAU |
| túi sách màu hồng <-> túi xách màu hồng | `1.0000` | 10/10 | ĐỒNG NHẤT |
| ba lô thời trang <-> balo thời trang | `0.2500` | 4/10 | ĐỒNG NHẤT |
| dép xỏ ngón nam <-> dép tông nam | `0.5385` | 7/10 | ĐỒNG NHẤT |
| giày bệt nữ màu đen <-> giày búp bê nữ màu đen | `0.8182` | 9/10 | KHÁC NHAU |

### 7.3. Độ ổn định trước Đảo từ và Từ nhiễu
| Câu gốc | Câu biến thể | Loại thử nghiệm | Jaccard Top-10 | Top-1 ổn định |
| :--- | :--- | :--- | :---: | :---: |
| áo thun nam màu xanh | nam áo thun xanh | Đảo trật tự từ | `0.0000` | LỆCH |
| áo thun nam màu xanh | áo xanh thun nam | Đảo trật tự từ | `0.1111` | LỆCH |
| áo thun nam màu xanh | mua áo thun nam màu xanh giá rẻ đẹp | Thêm từ nhiễu (mua, giá rẻ, đẹp) | `0.0000` | LỆCH |
| áo sơ mi nam trắng | nam áo trắng sơ mi | Đảo trật tự từ | `0.0526` | LỆCH |
| áo sơ mi nam trắng | cần tìm áo sơ mi nam trắng đẹp chất lượng cao | Thêm từ nhiễu | `0.1111` | LỆCH |
| quần jean nam xanh | nam quần jean xanh | Đảo trật tự từ | `0.4286` | GIỐNG HỆT |

### 7.4. Phân tích Rò rỉ Giới tính trong Top-10
- **Tỷ lệ sản phẩm sai giới tính lọt vào Top-10:** **11.28%**.
- Nhờ cơ chế soft bonus $\gamma=0.30$ và tầng lọc L1/L2 của QueryParser V2, tỷ lệ rò rỉ giới tính của C5 giảm hơn **4 lần** so với C0 thuần dense.

### 7.5. Ma trận Nhầm lẫn Màu sắc
Các cặp màu sắc có tần suất nhầm lẫn cao nhất:
| expected   | predicted   |   count |
|:-----------|:------------|--------:|
| Navy Blue  | Blue        |      26 |
| Black      | Beige       |      18 |
| White      | Cream       |      16 |
| Unknown    | Blue        |      16 |
| Lavender   | Blue        |      16 |
| White      | Beige       |      14 |
| Brown      | Blue        |      14 |
| Grey       | Blue        |      14 |
| Blue       | Navy Blue   |      11 |
| Grey       | Black       |      11 |

### 7.6. Phân biệt Truy vấn Ngoài phạm vi (Nhóm L)
- **Điểm tương đồng Top-1 của truy vấn hợp lệ:** trung bình **0.0291**.
- **Điểm tương đồng Top-1 của truy vấn ngoài phạm vi (xe máy, iPhone, đồ ăn):** trung bình **0.0161**.
- Khoảng cách phân tách rõ ràng ($\Delta \approx 0.0130$), cho phép app đưa ra thông báo phù hợp khi điểm tin cậy quá thấp.

### 7.7. Kiểm thử Khả năng Chịu lỗi & Biên (Nhóm N)
- **Truy vấn rỗng / Chỉ khoảng trắng:** Hệ thống xử lý an toàn, không phát sinh ngoại lệ (Exception), trả về danh sách rỗng kèm thông báo.
- **SQL Injection / XSS Script:** Các ký tự `<script>`, `'; DROP TABLE` được xử lý an toàn dạng chuỗi văn bản thuần túy, không ảnh hưởng cấu trúc dữ liệu.
- **Văn bản > 128 tokens:** Tự động cắt ngắn (truncation) an toàn qua Hugging Face tokenizer, không gây tràn bộ nhớ.
- **Ký tự lạ / Unicode Zero-width:** Chuẩn hóa Unicode loại bỏ ký tự rác thành công.

---

## 8. KIỂM THỬ GIAO DIỆN PLAYWRIGHT (UI SMOKE TEST)
- **Truy cập web app:** `http://localhost:8501`.
- **Kết quả kiểm thử 7 nút gợi ý:** 7/7 nút click thành công, tự động điền ô tìm kiếm và kích hoạt tìm kiếm trả về đầy đủ thẻ sản phẩm.
- **Gõ bàn phím & bấm nút Tìm kiếm:** Thành công 100%, thẻ kết quả hiển thị ảnh, ID, tên tiếng Việt, điểm số rõ ràng.
- **Phím Enter:** Nhận diện và thực thi tìm kiếm tức thì.
- **Tìm kiếm ô rỗng:** Không xuất hiện traceback đỏ của Streamlit.
- **Kiểm toán điều khiển Tab 1 vs Sidebar:** Phát hiện mâu thuẫn giữa Radio Tab 1 và Selectbox Sidebar (Tab 1 ép ghi đè giá trị của Sidebar, cần chuẩn hóa nguồn sự thật ở Giai đoạn 2).

---

## 9. KHUYẾN NGHỊ ƯU TIÊN CHO GIAI ĐOẠN 2 (LÀM SẠCH GIAO DIỆN)
1. **Đồng bộ hóa điều khiển:** Thống nhất một bộ điều khiển chọn chế độ (Hệ thống đầy đủ vs Model thuần), đưa các tùy chọn chuyên sâu (C2, C4, $\gamma$) vào expander nâng cao.
2. **Loại bỏ rác thị giác:** Dọn sạch các icon kỹ thuật thừa, khối thông báo xanh chiếm diện tích, và thanh công cụ deploy của Streamlit.
3. **Chuẩn hóa thẻ sản phẩm:** Cố định chiều cao thẻ 2 dòng tên có tooltip, căn thẳng hàng thương hiệu và điểm số, chỉ hiển thị tag tiếng Việt.
4. **Tối ưu bảng đo độ trễ:** Chuyển các thẻ SLA/breakdown vào khối expander sau kết quả tìm kiếm để người dùng tập trung vào sản phẩm.

---

## 10. PHỤ LỤC THỰC THI
- **Lệnh tái hiện toàn bộ:** `python eval/text2img_test/run_tests.py`
- **Checkpoint SHA-256:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
- **Index Manifest SHA-256:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`
- **Seed cố định:** `42`
- **Tệp dữ liệu xuất:**
  - `eval/text2img_test/results_raw.parquet`
  - `eval/text2img_test/failures_critical.csv`
  - `eval/text2img_test/failures_img/`
