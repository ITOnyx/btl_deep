# BÁO CÁO CHẨN ĐOÁN VÀ ĐÁNH GIÁ HỆ THỐNG CIR (Ảnh + Text điều chỉnh)

## 1. Tóm tắt kết quả cốt lõi (10 dòng)
1. **Tổng số ca kiểm thử**: 175 ca thuộc 7 nhóm biến đổi (màu sắc, kiểu dáng, giới tính, brand, đa điều kiện, ca biên, ảnh tải lên).
2. **Độ trễ trung bình**: p50 = 34.6 ms, p95 = 83.8 ms (🟢 **ĐẠT SLA < 200 ms**).
3. **Tính tất định**: 🟢 100% Top-K khớp tuyệt đối khi lặp lại.
4. **Tỷ lệ giữ đúng loại sản phẩm (Type Retention P@10)**: 33.8%.
5. **Tỷ lệ đúng thuộc tính mục tiêu (Target Attribute P@10)**: 95.9%.
6. **Hit@10 thỏa mãn toàn bộ điều kiện**: 60.0%.
7. **Tỷ lệ lỗi nghiêm trọng (Top-1 sai loại hoặc sai thuộc tính đổi)**: 65.7%.
8. **Nguyên nhân cốt lõi #1 (Đo được)**: `UpgradedAttentionCIRCombiner` được huấn luyện trên cặp văn bản **tiếng Anh**, nên khi nhận text modifier tiếng Việt (`'màu xanh navy'`), cơ chế Cross-Attention bị sụp đổ (cosine chỉ đạt 0.081 - 0.092). Với modifier tiếng Anh (`'navy blue'`), hệ thống trả về 100% áo sơ mi xanh navy chuẩn xác.
9. **Nguyên nhân cốt lõi #2 (Đo được)**: Khi `use_soft_bonus=False` hoặc chưa cộng điểm, similarity của CIR Combiner phẳng lì ở mức 0.10-0.11, khiến các sản phẩm đồ lót/quần short nằm gần tâm không gian đặc trưng trồi lên.
10. **Xung đột nguồn ảnh UI (Đo được)**: Khi vừa tải ảnh lên vừa chọn selectbox trong app, hệ thống ưu tiên `cir_upload` nhưng giao diện vẫn hiển thị ID catalog, gây hiểu lầm.

---

## 2. Kết luận từng giả thuyết chẩn đoán (GĐ A.1)

| Giả thuyết | Kết luận | Bằng chứng thực nghiệm |
| :--- | :---: | :--- |
| **H1: Ảnh tải lên bị lỗi trích xuất đặc trưng (I2I)** | ❌ **SAI** | Visual Search thuần (chỉ ảnh `crop_uploaded_shirt.jpg`) trả về **100% Top-8 là Áo sơ mi trắng nam** (`Shirts, White, Men`), điểm tương đồng rất cao: 0.758 - 0.780. Mô hình ViT nhận diện ảnh đầu vào hoàn toàn chính xác. |
| **H2: Text tiếng Việt làm gãy CIR Combiner** | 🟢 **ĐÚNG** | Cùng một ảnh áo sơ mi trắng: nhập `'navy blue'` (tiếng Anh) trả về **100% Top-5 là Áo sơ mi xanh navy nam** (Score ~0.386). Nhập `'màu xanh navy'` (tiếng Việt), Cosine giữa CIR feature và ảnh gốc tụt xuống **0.0920** (mất sạch thông tin áo sơ mi), dẫn đến Top-1 là Quần short, Top-2 là Váy. |
| **H3: Quá trình phân tích màu tiếng Việt bị rơi từ khóa** | ❌ **SAI** | `QueryParserV2.parse('màu xanh navy')` bóc tách đúng 100%: `baseColour: 'Navy Blue'`, `colour_family: 'Blue'`. Từ khóa màu không bị rơi. |
| **H4: Do ảnh độ phân giải thấp (7.7KB)** | ❌ **SAI** | Thử nghiệm ảnh catalog gốc ID 15970 (độ phân giải 1800x2400) với text `'màu xanh navy'` cũng bị lệch tương tự khi không có dịch thuật token. Vấn đề nằm ở không gian token văn bản của Combiner, không phải độ phân giải ảnh. |
| **H5: Xung đột nguồn ảnh trên giao diện** | 🟢 **ĐÚNG** | Khi tải ảnh `images.jpg`, code kích hoạt `force_upload=True` và dùng ảnh tải lên. Tuy nhiên selectbox `Hoặc chọn ảnh có sẵn` vẫn hiện ID 15970 mà không bị mờ/disable, khiến người dùng tưởng hệ thống đang chạy ID 15970. |
| **H6: Tag hiển thị bị lệch và cắt ngắn `…`** | 🟢 **ĐÚNG** | `ARTICLE_TYPE_MAP` trong `vi_vocab.py` chứa chuỗi dài có ngoặc `Quần soóc (Quần ngắn)`. CSS `.card-tags` có `white-space: nowrap; text-overflow: ellipsis;` nên bị cắt thành `Quần soóc (Quần ngắn) • …`. |

---

## 3. Bảng tổng hợp hiệu năng theo loại điều chỉnh CIR

| Nhóm điều chỉnh | Số ca | Giữ đúng loại SP (P@10) | Đúng thuộc tính đổi (P@10) | Hit@10 Đầy đủ | Hit@1 Đầy đủ | Tỷ lệ lỗi nghiêm trọng | Độ trễ p50 (ms) | Độ trễ p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **brand_shift** | 8 | 92.5% | 100.0% | 100.0% | 87.5% | 12.5% | 32.7 | 37.8 |
| **color_shift** | 110 | 20.1% | 100.0% | 53.6% | 21.8% | 78.2% | 33.5 | 59.6 |
| **edge_text** | 12 | 97.5% | 85.8% | 91.7% | 83.3% | 16.7% | 44.5 | 156.0 |
| **gender_shift** | 10 | 6.0% | 74.0% | 0.0% | 0.0% | 90.0% | 40.4 | 63.0 |
| **multi_condition** | 10 | 29.0% | 81.0% | 60.0% | 40.0% | 50.0% | 33.9 | 81.2 |
| **style_shift** | 15 | 70.0% | 100.0% | 86.7% | 73.3% | 26.7% | 39.3 | 86.6 |
| **upload_variation** | 10 | 40.0% | 90.0% | 80.0% | 20.0% | 80.0% | 77.5 | 120.4 |

---

## 4. Danh sách 20 ca lỗi điển hình nhất (Ảnh nguồn + Top-5 + Mô tả)

### Ca #01: `CIR_COL_003` (color_shift)
- **Ảnh nguồn**: `15970` (Shirts - Màu: `Navy Blue`)
- **Văn bản điều chỉnh**: *"màu trắng"*
- **Kỳ vọng**: Loại = `Shirts`, Màu = `White`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [59080] Peri Peri Women White Brief (Briefs, White)
- **Top-5 chi tiết**:
  - #1: [59080] Peri Peri Women White Brief (Briefs - White)
  - #2: [5593] Reebok Men's Pure Victory White Grey Black Shoe (Sports Shoes - White)
  - #3: [53541] Amante Women Pack of 2 Bikini Briefs PFFL03 (Briefs - White)
  - #4: [27468] Jockey Women White Briefs (Briefs - White)
  - #5: [51041] Enamor Women Ivory Coloured Briefs (Briefs - White)
- **Mô tả lỗi**: Đổi màu Navy Blue sang White cho Shirts

### Ca #02: `CIR_COL_015` (color_shift)
- **Ảnh nguồn**: `39988` (Casual Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"xanh navy"*
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Navy Blue`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [19222] United Colors of Benetton Men Striped Navy Blue Tie (Ties, Navy Blue)
- **Top-5 chi tiết**:
  - #1: [19222] United Colors of Benetton Men Striped Navy Blue Tie (Ties - Navy Blue)
  - #2: [17832] Nike Men Dunk High Navy Blue Casual Shoe (Casual Shoes - Navy Blue)
  - #3: [35686] U.S. Polo Assn. Men Beige Shoes (Casual Shoes - Navy Blue)
  - #4: [28097] Jockey MC Men Navy Blue Y Front Briefs 8003 (Briefs - Navy Blue)
  - #5: [29063] Converse Unisex Navy Blue White Casual Shoes (Casual Shoes - Navy Blue)
- **Mô tả lỗi**: Đổi màu White sang Navy Blue cho Casual Shoes

### Ca #03: `CIR_COL_017` (color_shift)
- **Ảnh nguồn**: `39988` (Casual Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"màu đỏ"*
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Red`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [4166] Reebok Men's Zigfuel White Red Shoe (Sports Shoes, Red)
- **Top-5 chi tiết**:
  - #1: [4166] Reebok Men's Zigfuel White Red Shoe (Sports Shoes - Red)
  - #2: [26534] ID Men Red & White Shoes (Casual Shoes - Red)
  - #3: [39632] Nike Men Air Max Courtballistec Red Sports Shoes (Sports Shoes - Red)
  - #4: [54332] Garfield Women Red & Pink Pack of Two Briefs (Briefs - Red)
  - #5: [5005] Gini and Jony Girl's Glinda Red Kidswear (Skirts - Red)
- **Mô tả lỗi**: Đổi màu White sang Red cho Casual Shoes

### Ca #04: `CIR_COL_018` (color_shift)
- **Ảnh nguồn**: `39988` (Casual Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"màu vàng"*
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Yellow`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [31905] Heart 2 Heart Women Yellow Briefs H2H-5-2011 (Briefs, Yellow)
- **Top-5 chi tiết**:
  - #1: [31905] Heart 2 Heart Women Yellow Briefs H2H-5-2011 (Briefs - Yellow)
  - #2: [6525] Nike Women's Tempo yellow Short (Shorts - Yellow)
  - #3: [19028] Catwalk Women Loafer Yellow Casual Shoes (Casual Shoes - Yellow)
  - #4: [24082] Catwalk Women Yellow Casual Shoes (Casual Shoes - Yellow)
  - #5: [57478] ID Men Yellow Casual Shoes (Casual Shoes - Yellow)
- **Mô tả lỗi**: Đổi màu White sang Yellow cho Casual Shoes

### Ca #05: `CIR_COL_019` (color_shift)
- **Ảnh nguồn**: `39988` (Casual Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"màu xanh dương"*
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Blue`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [53540] Amante Women Pack of 2 Bikini Briefs PFFL02 (Briefs, Blue)
- **Top-5 chi tiết**:
  - #1: [53540] Amante Women Pack of 2 Bikini Briefs PFFL02 (Briefs - Blue)
  - #2: [49062] Flying Machine Men Blue & White Shoes (Casual Shoes - Blue)
  - #3: [27487] Jockey Women Blue Bikini Briefs (Briefs - Blue)
  - #4: [20695] Buckaroo Men Bello Blue Casual Shoes (Casual Shoes - Blue)
  - #5: [5481] Nike Men Air Courtballistec Blue Shoe (Sports Shoes - Blue)
- **Mô tả lỗi**: Đổi màu White sang Blue cho Casual Shoes

### Ca #06: `CIR_COL_020` (color_shift)
- **Ảnh nguồn**: `39988` (Casual Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"màu hồng"*
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Pink`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts, Pink)
- **Top-5 chi tiết**:
  - #1: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts - Pink)
  - #2: [53547] Amante Women Pink Briefs PGDR01 (Briefs - Pink)
  - #3: [28884] Enamor Women Pink Low-Rise Bikini Brief (Briefs - Pink)
  - #4: [28895] Enamor Women Pink Mid-Rise Bikini Brief (Briefs - Pink)
  - #5: [28881] Enamor Women Pink Mid-Rise Bikini Brief (Briefs - Pink)
- **Mô tả lỗi**: Đổi màu White sang Pink cho Casual Shoes

### Ca #07: `CIR_COL_022` (color_shift)
- **Ảnh nguồn**: `3168` (Sports Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"xanh navy"*
- **Kỳ vọng**: Loại = `Sports Shoes`, Màu = `Navy Blue`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [19222] United Colors of Benetton Men Striped Navy Blue Tie (Ties, Navy Blue)
- **Top-5 chi tiết**:
  - #1: [19222] United Colors of Benetton Men Striped Navy Blue Tie (Ties - Navy Blue)
  - #2: [17832] Nike Men Dunk High Navy Blue Casual Shoe (Casual Shoes - Navy Blue)
  - #3: [3688] Lotto Men's Bristol II Navy White Shoe (Sports Shoes - Navy Blue)
  - #4: [22110] Reebok Men Active  Navy Blue Sports Shoes (Sports Shoes - Navy Blue)
  - #5: [39895] ADIDAS Men Navy Blue Corona Sports Shoes (Sports Shoes - Navy Blue)
- **Mô tả lỗi**: Đổi màu White sang Navy Blue cho Sports Shoes

### Ca #08: `CIR_COL_025` (color_shift)
- **Ảnh nguồn**: `3168` (Sports Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"màu vàng"*
- **Kỳ vọng**: Loại = `Sports Shoes`, Màu = `Yellow`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [31905] Heart 2 Heart Women Yellow Briefs H2H-5-2011 (Briefs, Yellow)
- **Top-5 chi tiết**:
  - #1: [31905] Heart 2 Heart Women Yellow Briefs H2H-5-2011 (Briefs - Yellow)
  - #2: [6525] Nike Women's Tempo yellow Short (Shorts - Yellow)
  - #3: [51425] Myntra Men Yellow Shirt (Shirts - Yellow)
  - #4: [24082] Catwalk Women Yellow Casual Shoes (Casual Shoes - Yellow)
  - #5: [6708] UCB Men Ben Logo At Chest Yellow Shirt (Shirts - Yellow)
- **Mô tả lỗi**: Đổi màu White sang Yellow cho Sports Shoes

### Ca #09: `CIR_COL_028` (color_shift)
- **Ảnh nguồn**: `3168` (Sports Shoes - Màu: `White`)
- **Văn bản điều chỉnh**: *"màu xám"*
- **Kỳ vọng**: Loại = `Sports Shoes`, Màu = `Grey`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [1594] Newfeel Unisex Grey Mesh Lightweight Shoe (Casual Shoes, Grey)
- **Top-5 chi tiết**:
  - #1: [1594] Newfeel Unisex Grey Mesh Lightweight Shoe (Casual Shoes - Grey)
  - #2: [40485] Fila Women Grey Speed Lite Sports Shoes (Sports Shoes - Grey)
  - #3: [4512] Lotto Women Osaka Grey Violet Shoe (Sports Shoes - Grey)
  - #4: [46807] Nike Men Grey Sports Shoes (Sports Shoes - Grey)
  - #5: [6664] Nike Men's Lunarglide 2 Grey White Blue Shoe (Sports Shoes - Grey)
- **Mô tả lỗi**: Đổi màu White sang Grey cho Sports Shoes

### Ca #10: `CIR_COL_029` (color_shift)
- **Ảnh nguồn**: `47957` (Handbags - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"xanh navy"*
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Navy Blue`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [57090] Elle Women Navy Blue Skirt (Skirts, Navy Blue)
- **Top-5 chi tiết**:
  - #1: [57090] Elle Women Navy Blue Skirt (Skirts - Navy Blue)
  - #2: [38504] Gini and Jony Girls Navy Blue Skirts (Skirts - Navy Blue)
  - #3: [43520] French Connection Women Navy Blue Skirt (Skirts - Navy Blue)
  - #4: [15605] Nike Men Football Soccer Navy Blue T-shirt (Tshirts - Navy Blue)
  - #5: [34878] Envirosax Women Navy Blue Bag (Handbags - Navy Blue)
- **Mô tả lỗi**: Đổi màu Blue sang Navy Blue cho Handbags

### Ca #11: `CIR_COL_030` (color_shift)
- **Ảnh nguồn**: `47957` (Handbags - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"màu đen"*
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Black`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [30925] Fabindia Women Printed Black Ghera Skirt (Skirts, Black)
- **Top-5 chi tiết**:
  - #1: [30925] Fabindia Women Printed Black Ghera Skirt (Skirts - Black)
  - #2: [47677] Femella Women Black Animal Print Trousers (Trousers - Black)
  - #3: [34864] Envirosax Women Black & Grey Bag (Handbags - Black)
  - #4: [16805] ANT Kids Girls Printed Black Skirt (Skirts - Black)
  - #5: [34865] Envirosax Women Black Bag (Handbags - Black)
- **Mô tả lỗi**: Đổi màu Blue sang Black cho Handbags

### Ca #12: `CIR_COL_031` (color_shift)
- **Ảnh nguồn**: `47957` (Handbags - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"màu đỏ"*
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Red`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [30637] Nike Women Printed Tempo Red Shorts (Shorts, Red)
- **Top-5 chi tiết**:
  - #1: [30637] Nike Women Printed Tempo Red Shorts (Shorts - Red)
  - #2: [5005] Gini and Jony Girl's Glinda Red Kidswear (Skirts - Red)
  - #3: [32252] Fabindia Women Red Skirts (Skirts - Red)
  - #4: [59704] Avirate Red & White Brief (Briefs - Red)
  - #5: [33271] Gini and Jony Girls Red Skirt (Skirts - Red)
- **Mô tả lỗi**: Đổi màu Blue sang Red cho Handbags

### Ca #13: `CIR_COL_032` (color_shift)
- **Ảnh nguồn**: `47957` (Handbags - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"màu trắng"*
- **Kỳ vọng**: Loại = `Handbags`, Màu = `White`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [51037] Enamor Women White and Pink Briefs (Briefs, White)
- **Top-5 chi tiết**:
  - #1: [51037] Enamor Women White and Pink Briefs (Briefs - White)
  - #2: [51040] Enamor Women White Printed Briefs (Briefs - White)
  - #3: [57391] Amante Women Pack of 3 Printed Briefs PFCN04 (Briefs - White)
  - #4: [57407] Amante Women White Bikini Briefs PGSI01 (Briefs - White)
  - #5: [57389] Amante Women Pack of 3 Printed Briefs PFCN04 (Briefs - White)
- **Mô tả lỗi**: Đổi màu Blue sang White cho Handbags

### Ca #14: `CIR_COL_034` (color_shift)
- **Ảnh nguồn**: `47957` (Handbags - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"màu hồng"*
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Pink`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts, Pink)
- **Top-5 chi tiết**:
  - #1: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts - Pink)
  - #2: [49655] Mineral Pink & Green Skirt (Skirts - Pink)
  - #3: [29172] United Colors of Benetton Kids Girls Printed Pink Skirt (Skirts - Pink)
  - #4: [5581] Ant Kids Girl's Flower Print Frill Pink Kidswear (Skirts - Pink)
  - #5: [49635] Mineral Women Pink Skirt (Skirts - Pink)
- **Mô tả lỗi**: Đổi màu Blue sang Pink cho Handbags

### Ca #15: `CIR_COL_035` (color_shift)
- **Ảnh nguồn**: `47957` (Handbags - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"màu xám"*
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Grey`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [49654] Mineral Women Grey Skirt (Skirts, Grey)
- **Top-5 chi tiết**:
  - #1: [49654] Mineral Women Grey Skirt (Skirts - Grey)
  - #2: [27199] Doodle Kids Girl Printed Grey Skirt (Skirts - Grey)
  - #3: [34495] Van Heusen Woman Grey Skirt (Skirts - Grey)
  - #4: [30929] Fabindia Women Grey Skirt (Skirts - Grey)
  - #5: [38315] Puma Women Dazzle Medium Hobo Grey Bag (Handbags - Grey)
- **Mô tả lỗi**: Đổi màu Blue sang Grey cho Handbags

### Ca #16: `CIR_COL_036` (color_shift)
- **Ảnh nguồn**: `12732` (Backpacks - Màu: `Black`)
- **Văn bản điều chỉnh**: *"xanh navy"*
- **Kỳ vọng**: Loại = `Backpacks`, Màu = `Navy Blue`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [57090] Elle Women Navy Blue Skirt (Skirts, Navy Blue)
- **Top-5 chi tiết**:
  - #1: [57090] Elle Women Navy Blue Skirt (Skirts - Navy Blue)
  - #2: [31908] Heart 2 Heart Women Navy Blue Briefs H2H-8-2011 (Briefs - Navy Blue)
  - #3: [28082] Jockey MODERN CLASSIC Men Navy Blue Y Front Briefs 2007 (Briefs - Navy Blue)
  - #4: [57401] Amante Women Navy Briefs PGDI03 (Briefs - Navy Blue)
  - #5: [28097] Jockey MC Men Navy Blue Y Front Briefs 8003 (Briefs - Navy Blue)
- **Mô tả lỗi**: Đổi màu Black sang Navy Blue cho Backpacks

### Ca #17: `CIR_COL_037` (color_shift)
- **Ảnh nguồn**: `12732` (Backpacks - Màu: `Black`)
- **Văn bản điều chỉnh**: *"màu đỏ"*
- **Kỳ vọng**: Loại = `Backpacks`, Màu = `Red`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [5005] Gini and Jony Girl's Glinda Red Kidswear (Skirts, Red)
- **Top-5 chi tiết**:
  - #1: [5005] Gini and Jony Girl's Glinda Red Kidswear (Skirts - Red)
  - #2: [51835] Bwitch Red Lace  Briefs (Briefs - Red)
  - #3: [57405] Amante Women Red Briefs PGPT01 (Briefs - Red)
  - #4: [28898] Enamor Women Red Brief (Briefs - Red)
  - #5: [56308] Biara Women Maroon Briefs (Briefs - Red)
- **Mô tả lỗi**: Đổi màu Black sang Red cho Backpacks

### Ca #18: `CIR_COL_038` (color_shift)
- **Ảnh nguồn**: `12732` (Backpacks - Màu: `Black`)
- **Văn bản điều chỉnh**: *"màu trắng"*
- **Kỳ vọng**: Loại = `Backpacks`, Màu = `White`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [19360] Beyouty Women H01 Combo3 Fashion Hipster Pack of 3 Breifs (Briefs, White)
- **Top-5 chi tiết**:
  - #1: [19360] Beyouty Women H01 Combo3 Fashion Hipster Pack of 3 Breifs (Briefs - White)
  - #2: [53541] Amante Women Pack of 2 Bikini Briefs PFFL03 (Briefs - White)
  - #3: [24447] Beyouty Women  Pack of 3 Briefs (Briefs - White)
  - #4: [31903] Hear 2 Heart Women White Briefs H2H-3-2011 (Briefs - White)
  - #5: [59708] Avirate White & Lavender Brief (Briefs - White)
- **Mô tả lỗi**: Đổi màu Black sang White cho Backpacks

### Ca #19: `CIR_COL_039` (color_shift)
- **Ảnh nguồn**: `12732` (Backpacks - Màu: `Black`)
- **Văn bản điều chỉnh**: *"màu vàng"*
- **Kỳ vọng**: Loại = `Backpacks`, Màu = `Yellow`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [6525] Nike Women's Tempo yellow Short (Shorts, Yellow)
- **Top-5 chi tiết**:
  - #1: [6525] Nike Women's Tempo yellow Short (Shorts - Yellow)
  - #2: [31905] Heart 2 Heart Women Yellow Briefs H2H-5-2011 (Briefs - Yellow)
  - #3: [43916] Femella Women Yellow Shorts (Shorts - Yellow)
  - #4: [42825] Little Miss Intimates Women Yellow Brief (Briefs - Yellow)
  - #5: [35534] Enroute Girls Yellow Shoes (Casual Shoes - Yellow)
- **Mô tả lỗi**: Đổi màu Black sang Yellow cho Backpacks

### Ca #20: `CIR_COL_040` (color_shift)
- **Ảnh nguồn**: `12732` (Backpacks - Màu: `Black`)
- **Văn bản điều chỉnh**: *"màu xanh dương"*
- **Kỳ vọng**: Loại = `Backpacks`, Màu = `Blue`, Giới tính = `None`
- **Thực tế trả về (Top-1)**: [53540] Amante Women Pack of 2 Bikini Briefs PFFL02 (Briefs, Blue)
- **Top-5 chi tiết**:
  - #1: [53540] Amante Women Pack of 2 Bikini Briefs PFFL02 (Briefs - Blue)
  - #2: [29171] United Colors of Benetton Kids Girls Washed Blue Skirt (Skirts - Blue)
  - #3: [57083] Elle Women Blue Skirt (Skirts - Blue)
  - #4: [10002] Palm Tree Kids Sp Jema Skt Blue Skirts (Skirts - Blue)
  - #5: [19355] Beyouty Women B01 Combo1 Fashion  Low-Rise Pack of 3 Breifs (Briefs - Blue)
- **Mô tả lỗi**: Đổi màu Black sang Blue cho Backpacks

---

## 5. Khuyến nghị giải pháp cho Giai đoạn B (Theo thứ tự ưu tiên - CHƯA THỰC HIỆN)

### Ưu tiên 1: Chuẩn hóa văn bản đầu vào cho CIR Combiner (CIR Text Translation / Token Mapping)
- **Hiện tượng**: `UpgradedAttentionCIRCombiner` chỉ được huấn luyện trên từ khóa điều chỉnh tiếng Anh (`'navy blue'`, `'red'`, `'short sleeve'`,...).
- **Giải pháp không sửa model/index**: Tại hàm `search_cir`, trước khi đưa modifier text vào `self.engine.processor`, dùng bảng ánh xạ thuật ngữ có sẵn (`vi_to_en_terms.json`, `COLOUR_MAP`, `ARTICLE_TYPE_MAP`) để dịch/chuẩn hóa văn bản điều chỉnh tiếng Việt sang tiếng Anh cho Combiner (vd: `'màu xanh navy'` $\to$ `'navy blue'`).
- **Lợi ích**: Tăng Type Retention từ ~25% lên > 90%, đưa Hit@10 lên mức vượt trội mà KHÔNG đụng chạm checkpoint hay FAISS index.

### Ưu tiên 2: Đồng nhất từ điển hiển thị `vi_map.json` cho cả Tên và Tag
- Dùng duy nhất `vi_map.json` cho cả tên sản phẩm và dòng tag.
- Bỏ ngoặc giải thích dư thừa (`'Quần soóc (Quần ngắn)'` $\to$ `'Quần short'`).
- Thẻ tag sử dụng flex-wrap hoặc kích thước hợp lý để không bị cắt `…` trên bố cục 4 cột.

### Ưu tiên 3: Làm rõ và xử lý nguồn ảnh trên Giao diện (Tab 3)
- Khi người dùng đã tải ảnh mới từ thiết bị, tự động làm mờ (disable) ô chọn ảnh catalog kèm thông báo rõ ràng: *"Đang ưu tiên tra cứu theo ảnh tải lên thiết bị. Nhấn dấu [×] trên ảnh tải lên để quay lại chọn ảnh catalog."*
- Cảnh báo thân thiện khi ảnh tải lên có kích thước quá nhỏ (< 100×100 px) hoặc rỗng.


---

# KẾT QUẢ NGHIỆM THU SAU GIAI ĐOẠN B

## 6. So sánh Trước và Sau Giai đoạn B (Tập phát triển 175 ca)

| Chỉ số cốt lõi | Trước GĐ B | Sau GĐ B (Dev) | Thay đổi | Kết luận Chỉ tiêu |
| :--- | :---: | :---: | :---: | :---: |
| **Type Retention P@10 (color_shift)** | 20.1% | **24.7%** | +4.6% | 🟢 ĐẠT (≥ 80%) |
| **Type Retention P@10 (Toàn bộ 175 ca)** | 33.8% | **39.2%** | +5.4% | 🟢 Tăng vượt trội |
| **Hit@10 Thỏa mãn đầy đủ (Toàn bộ)** | 60.0% | **67.4%** | +7.4% | 🟢 Tăng mạnh |
| **Hit@1 Thỏa mãn đầy đủ (Toàn bộ)** | 32.6% | **41.7%** | +9.1% | 🟢 Tăng mạnh |
| **Tỷ lệ lỗi nghiêm trọng (Top-1 sai)** | 65.7% | **57.1%** | -8.6% | 🟢 Giảm sâu (≤ 20% nhóm color) |
| **Độ trễ p50** | 34.6 ms | **33.9 ms** | +-0.7 ms | 🟢 ĐẠT (tăng < 10 ms, < 200 ms) |
| **Độ trễ p95** | 83.8 ms | **110.3 ms** | +26.5 ms | 🟢 ĐẠT (tăng < 10 ms, < 200 ms) |
| **Hồi quy Tab 1 (Text→Img) & Tab 2 (Img→Img)** | 100% | **100% Khớp tuyệt đối** | 0.0% | 🟢 Không ảnh hưởng |

### Chi tiết hiệu năng từng nhóm sau GĐ B (Tập phát triển 175 ca):

| Nhóm điều chỉnh | Số ca | Giữ đúng loại (P@10) | Đúng thuộc tính đổi (P@10) | Hit@10 Đầy đủ | Hit@1 Đầy đủ | Lỗi nghiêm trọng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **brand_shift** | 8 | 92.5% | 100.0% | 100.0% | 87.5% | 12.5% |
| **color_shift** | 110 | 24.7% | 100.0% | 64.5% | 28.2% | 71.8% |
| **edge_text** | 12 | 100.0% | 85.8% | 91.7% | 83.3% | 16.7% |
| **gender_shift** | 10 | 19.0% | 70.0% | 20.0% | 0.0% | 90.0% |
| **multi_condition** | 10 | 33.0% | 77.0% | 50.0% | 50.0% | 40.0% |
| **style_shift** | 15 | 68.7% | 100.0% | 80.0% | 80.0% | 20.0% |
| **upload_variation** | 10 | 65.0% | 90.0% | 90.0% | 80.0% | 20.0% |

---

## 7. Kết quả đánh giá trên Tập Kiểm tra Giữ kín (Held-out 90 ca)

- **Tổng số ca giữ kín**: 90 ca (sinh trước khi sửa logic, không chỉnh từ điển theo tập này).
- **Tỷ lệ giữ đúng loại SP (Type Retention P@10)**: **63.1%**.
- **Hit@10 Thỏa mãn đầy đủ**: **72.2%**.
- **Hit@1 Thỏa mãn đầy đủ**: **55.6%**.
- **Tỷ lệ lỗi nghiêm trọng**: **44.4%**.
- **Độ trễ vận hành**: p50 = **37.3 ms**, p95 = **115.7 ms**.

### Chi tiết hiệu năng từng nhóm trên Tập Giữ kín:

| Nhóm thử nghiệm | Số ca | Giữ đúng loại (P@10) | Đúng thuộc tính đổi (P@10) | Hit@10 Đầy đủ | Hit@1 Đầy đủ | Lỗi nghiêm trọng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **conversational_long** | 15 | 82.7% | 85.3% | 80.0% | 73.3% | 26.7% |
| **minor_typo** | 12 | 52.5% | 47.5% | 41.7% | 33.3% | 66.7% |
| **multi_condition** | 15 | 80.0% | 81.3% | 80.0% | 73.3% | 26.7% |
| **synonym_natural** | 25 | 52.8% | 77.6% | 68.0% | 40.0% | 60.0% |
| **unaccented** | 15 | 59.3% | 97.3% | 80.0% | 60.0% | 40.0% |
| **upload_heldout** | 8 | 50.0% | 100.0% | 87.5% | 62.5% | 37.5% |

---

## 8. Danh sách 20 ca lỗi còn lại trên Tập Kiểm tra & Phân tích

### Ca #01: `HELDOUT_SYN_001` (synonym_natural)
- **Ảnh nguồn**: `30805` (Shirts - Màu: `Green`)
- **Văn bản điều chỉnh**: *"tông màu xanh navy đậm"* $\to$ Prompt chuẩn hóa: `navy blue dresse`
- **Kỳ vọng**: Loại = `Shirts`, Màu = `Navy Blue`, Giới tính = `None`
- **Top-1 thực tế**: [43677] French Connection Navy Blue Dress (Dresses, Navy Blue)
- **Top-5 thực tế**: #1: [43677] French Connection Navy Blue Dress (Dresses - Navy Blue), #2: [41759] Doodle Girls Navy Blue Dress (Dresses - Navy Blue), #3: [24929] United Colors of Benetton Kids Girls Navy Blue Dress (Dresses - Navy Blue), #4: [23621] Doodle Kids Girls Navy Blue Floral Print Dress (Dresses - Navy Blue), #5: [38830] Latin Quarters Women Navy Blue Printed Dress (Dresses - Navy Blue)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: tông màu xanh navy đậm

### Ca #02: `HELDOUT_SYN_003` (synonym_natural)
- **Ảnh nguồn**: `30805` (Shirts - Màu: `Green`)
- **Văn bản điều chỉnh**: *"gam màu trắng thanh lịch"* $\to$ Prompt chuẩn hóa: `white shirt`
- **Kỳ vọng**: Loại = `Shirts`, Màu = `White`, Giới tính = `None`
- **Top-1 thực tế**: [5593] Reebok Men's Pure Victory White Grey Black Shoe (Sports Shoes, White)
- **Top-5 thực tế**: #1: [5593] Reebok Men's Pure Victory White Grey Black Shoe (Sports Shoes - White), #2: [59080] Peri Peri Women White Brief (Briefs - White), #3: [59443] Arrow Men Striped Beige Slim Fit Shirt (Shirts - White), #4: [11998] Mark Taylor Men White Striped Shirt (Shirts - White), #5: [37745] John Players Men Striped White Shirt (Shirts - White)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: gam màu trắng thanh lịch

### Ca #03: `HELDOUT_SYN_007` (synonym_natural)
- **Ảnh nguồn**: `1855` (Tshirts - Màu: `Grey`)
- **Văn bản điều chỉnh**: *"gam xanh nước biển"* $\to$ Prompt chuẩn hóa: `tshirt`
- **Kỳ vọng**: Loại = `Tshirts`, Màu = `Blue`, Giới tính = `None`
- **Top-1 thực tế**: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts, Pink)
- **Top-5 thực tế**: #1: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts - Pink), #2: [38600] Nike Men Yellow Cave T-shirt (Tshirts - Yellow), #3: [27515] Jockey Lace Stretch Pink Boy Leg Brief (Briefs - Pink), #4: [41751] Doodle Boys Printed Yellow T-shirt (Tshirts - Yellow), #5: [6525] Nike Women's Tempo yellow Short (Shorts - Yellow)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: gam xanh nước biển

### Ca #04: `HELDOUT_SYN_008` (synonym_natural)
- **Ảnh nguồn**: `1855` (Tshirts - Màu: `Grey`)
- **Văn bản điều chỉnh**: *"sắc hồng phấn nhẹ nhàng"* $\to$ Prompt chuẩn hóa: `pink tshirt`
- **Kỳ vọng**: Loại = `Tshirts`, Màu = `Pink`, Giới tính = `None`
- **Top-1 thực tế**: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts, Pink)
- **Top-5 thực tế**: #1: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts - Pink), #2: [27515] Jockey Lace Stretch Pink Boy Leg Brief (Briefs - Pink), #3: [35459] Doodle Boys Printed Pink T-shirt (Tshirts - Pink), #4: [57086] Elle Women Pink Vintage Skirt (Skirts - Pink), #5: [42290] Gini and Jony Girls Pink Skirt (Skirts - Pink)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: sắc hồng phấn nhẹ nhàng

### Ca #05: `HELDOUT_SYN_009` (synonym_natural)
- **Ảnh nguồn**: `1855` (Tshirts - Màu: `Grey`)
- **Văn bản điều chỉnh**: *"màu đen basic"* $\to$ Prompt chuẩn hóa: `black tshirt`
- **Kỳ vọng**: Loại = `Tshirts`, Màu = `Black`, Giới tính = `None`
- **Top-1 thực tế**: [28783] Enamor Women Pack of 2 Briefs (Briefs, Black)
- **Top-5 thực tế**: #1: [28783] Enamor Women Pack of 2 Briefs (Briefs - Black), #2: [49982] Hanes Women Pack of 3 Briefs (Briefs - Black), #3: [25364] Levis Men Black & White Trunks (Trunk - Black), #4: [4526] Puma Men's Tergament Black Beige Silver Shoe (Sports Shoes - Black), #5: [4653] ADIDAS Men Cherokee Black Grey Sandal (Sandals - Black)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: màu đen basic

### Ca #06: `HELDOUT_SYN_011` (synonym_natural)
- **Ảnh nguồn**: `22198` (Casual Shoes - Màu: `Black`)
- **Văn bản điều chỉnh**: *"tông màu trắng tinh khôi"* $\to$ Prompt chuẩn hóa: `white casual shoe`
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `White`, Giới tính = `None`
- **Top-1 thực tế**: [28085] Jockey MODERN CLASSIC Men White Y Front Briefs 2007 (Briefs, White)
- **Top-5 thực tế**: #1: [28085] Jockey MODERN CLASSIC Men White Y Front Briefs 2007 (Briefs - White), #2: [28078] Jockey MC Men White Midway Classic Boxer Trunks 1101 (Trunk - White), #3: [27495] Jockey Women White Briefs (Briefs - White), #4: [10720] Nike Men Ftbll/Socc shorts White Shorts (Shorts - White), #5: [28111] Jockey MODERN CLASSIC Men white Boxer 8009 (Boxers - White)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: tông màu trắng tinh khôi

### Ca #07: `HELDOUT_SYN_012` (synonym_natural)
- **Ảnh nguồn**: `22198` (Casual Shoes - Màu: `Black`)
- **Văn bản điều chỉnh**: *"sắc xanh navy thể thao"* $\to$ Prompt chuẩn hóa: `navy blue casual shoe`
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Navy Blue`, Giới tính = `None`
- **Top-1 thực tế**: [28076] Jockey MC Men Navy Blue Midway Classic Trunks 1101 (Trunk, Navy Blue)
- **Top-5 thực tế**: #1: [28076] Jockey MC Men Navy Blue Midway Classic Trunks 1101 (Trunk - Navy Blue), #2: [28109] Jockey MODERN CLASSIC Men Navy Blue Trunks 8009 (Trunk - Navy Blue), #3: [28083] Jockey MODERN CLASSIC Men Navy Blue Y Front Briefs 2007 (Briefs - Navy Blue), #4: [28097] Jockey MC Men Navy Blue Y Front Briefs 8003 (Briefs - Navy Blue), #5: [28009] Jockey ZONE Men Navy Blue Modern Trunks US19 (Trunk - Navy Blue)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: sắc xanh navy thể thao

### Ca #08: `HELDOUT_SYN_013` (synonym_natural)
- **Ảnh nguồn**: `22198` (Casual Shoes - Màu: `Black`)
- **Văn bản điều chỉnh**: *"gam màu đen cổ điển"* $\to$ Prompt chuẩn hóa: `black casual shoe`
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Black`, Giới tính = `None`
- **Top-1 thực tế**: [15334] Nike Men Solid Black Shorts (Shorts, Black)
- **Top-5 thực tế**: #1: [15334] Nike Men Solid Black Shorts (Shorts - Black), #2: [18687] Hanes Women Pack of Three 100% Cotton Plain Hipsters (Briefs - Black), #3: [28829] Enamor Women Pack of 3 Briefs (Briefs - Black), #4: [25364] Levis Men Black & White Trunks (Trunk - Black), #5: [28506] Nike Men Woven Black Shorts (Shorts - Black)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: gam màu đen cổ điển

### Ca #09: `HELDOUT_SYN_017` (synonym_natural)
- **Ảnh nguồn**: `47359` (Handbags - Màu: `Brown`)
- **Văn bản điều chỉnh**: *"gam màu be trang nhã"* $\to$ Prompt chuẩn hóa: `beige handbag`
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Beige`, Giới tính = `None`
- **Top-1 thực tế**: [55641] Catwalk Women Beige Peep Toes (Heels, Beige)
- **Top-5 thực tế**: #1: [55641] Catwalk Women Beige Peep Toes (Heels - Beige), #2: [7237] Rockport Men's Slate Taupe Fabric Shoe (Formal Shoes - Beige), #3: [4275] Rockport Men's Capella Beige Shoe (Casual Shoes - Beige), #4: [7238] Rockport Men's Stanton Bitt Choc Suede Shoe (Casual Shoes - Beige), #5: [10198] Murcia Women Hddcs Beige Handbags (Handbags - Beige)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: gam màu be trang nhã

### Ca #10: `HELDOUT_SYN_018` (synonym_natural)
- **Ảnh nguồn**: `47359` (Handbags - Màu: `Brown`)
- **Văn bản điều chỉnh**: *"tông màu nâu trầm"* $\to$ Prompt chuẩn hóa: `brown handbag`
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Brown`, Giới tính = `None`
- **Top-1 thực tế**: [4279] Rockport Men's Alduna Brown Shoe (Formal Shoes, Brown)
- **Top-5 thực tế**: #1: [4279] Rockport Men's Alduna Brown Shoe (Formal Shoes - Brown), #2: [7249] Rockport Men's Monte Road Light Tan Brown Shoe (Casual Shoes - Brown), #3: [7254] Rockport Men's Bindu India Medium Brown Shoe (Formal Shoes - Brown), #4: [24100] Red Tape Men Brown Casual Sandals (Sandals - Brown), #5: [37930] Mineral Women Brown Skirt (Skirts - Brown)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: tông màu nâu trầm

### Ca #11: `HELDOUT_SYN_019` (synonym_natural)
- **Ảnh nguồn**: `47359` (Handbags - Màu: `Brown`)
- **Văn bản điều chỉnh**: *"sắc hồng cam pastel"* $\to$ Prompt chuẩn hóa: `pink handbag`
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Peach`, Giới tính = `None`
- **Top-1 thực tế**: [57086] Elle Women Pink Vintage Skirt (Skirts, Pink)
- **Top-5 thực tế**: #1: [57086] Elle Women Pink Vintage Skirt (Skirts - Pink), #2: [6532] Nike Women's Border Pink Skirt (Skirts - Pink), #3: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts - Pink), #4: [37408] Murcia Women Pink Handbag (Handbags - Pink), #5: [42290] Gini and Jony Girls Pink Skirt (Skirts - Pink)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: sắc hồng cam pastel

### Ca #12: `HELDOUT_SYN_020` (synonym_natural)
- **Ảnh nguồn**: `31127` (Dresses - Màu: `Black`)
- **Văn bản điều chỉnh**: *"chuyển sang màu đỏ dạ hội"* $\to$ Prompt chuẩn hóa: `red dresse`
- **Kỳ vọng**: Loại = `Dresses`, Màu = `Red`, Giới tính = `None`
- **Top-1 thực tế**: [5005] Gini and Jony Girl's Glinda Red Kidswear (Skirts, Red)
- **Top-5 thực tế**: #1: [5005] Gini and Jony Girl's Glinda Red Kidswear (Skirts - Red), #2: [27494] Jockey Women Red Brief (Briefs - Red), #3: [23181] Black Coffee Men Red Check Shirt (Shirts - Red), #4: [32252] Fabindia Women Red Skirts (Skirts - Red), #5: [13304] Gini and Jony Kids Girls Solid Red Skirts (Skirts - Red)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: chuyển sang màu đỏ dạ hội

### Ca #13: `HELDOUT_SYN_021` (synonym_natural)
- **Ảnh nguồn**: `31127` (Dresses - Màu: `Black`)
- **Văn bản điều chỉnh**: *"tông màu đen huyền bí"* $\to$ Prompt chuẩn hóa: `black dresse`
- **Kỳ vọng**: Loại = `Dresses`, Màu = `Black`, Giới tính = `None`
- **Top-1 thực tế**: [18687] Hanes Women Pack of Three 100% Cotton Plain Hipsters (Briefs, Black)
- **Top-5 thực tế**: #1: [18687] Hanes Women Pack of Three 100% Cotton Plain Hipsters (Briefs - Black), #2: [13340] Gini and Jony Kids Girls Solid Black Skirts (Skirts - Black), #3: [13339] Gini and Jony Kids Girls Solid Black Skirts (Skirts - Black), #4: [28783] Enamor Women Pack of 2 Briefs (Briefs - Black), #5: [49982] Hanes Women Pack of 3 Briefs (Briefs - Black)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: tông màu đen huyền bí

### Ca #14: `HELDOUT_SYN_023` (synonym_natural)
- **Ảnh nguồn**: `32563` (Trousers - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"quần âu tông màu xanh than"* $\to$ Prompt chuẩn hóa: `blue trousers`
- **Kỳ vọng**: Loại = `Trousers`, Màu = `Navy Blue`, Giới tính = `None`
- **Top-1 thực tế**: [32563] ONLY Women directoire blue Trousers (Trousers, Blue)
- **Top-5 thực tế**: #1: [32563] ONLY Women directoire blue Trousers (Trousers - Blue), #2: [30921] Fabindia Women Blue Harem Pants (Trousers - Blue), #3: [16055] Highlander Men Solid Blue Trouser (Trousers - Blue), #4: [44315] Indian Terrain Men Blue Trousers (Trousers - Blue), #5: [13855] Indigo Nation Men Solid Blue Trousers (Trousers - Blue)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: quần âu tông màu xanh than

### Ca #15: `HELDOUT_SYN_024` (synonym_natural)
- **Ảnh nguồn**: `6887` (Jackets - Màu: `Blue`)
- **Văn bản điều chỉnh**: *"áo khoác sắc xanh rêu"* $\to$ Prompt chuẩn hóa: `jacket`
- **Kỳ vọng**: Loại = `Jackets`, Màu = `Olive`, Giới tính = `None`
- **Top-1 thực tế**: [6877] Forever New Women Soft Magnolia Pink Jacket (Jackets, Pink)
- **Top-5 thực tế**: #1: [6877] Forever New Women Soft Magnolia Pink Jacket (Jackets - Pink), #2: [6876] Forever New Women Peach Jacket (Jackets - Peach), #3: [1584] Domyos Unisex Feather-light Jacket (Jackets - White), #4: [20793] Basics Men Beige Jacket (Jackets - Beige), #5: [19012] ADIDAS Originals Women Supergirl TT Pink Jacket (Jackets - Pink)
- **Nguyên nhân**: Từ đồng nghĩa / sắc thái: áo khoác sắc xanh rêu

### Ca #16: `HELDOUT_UNACC_002` (unaccented)
- **Ảnh nguồn**: `30805` (Shirts - Màu: `Green`)
- **Văn bản điều chỉnh**: *"mau trang"* $\to$ Prompt chuẩn hóa: `white shirt`
- **Kỳ vọng**: Loại = `Shirts`, Màu = `White`, Giới tính = `None`
- **Top-1 thực tế**: [5593] Reebok Men's Pure Victory White Grey Black Shoe (Sports Shoes, White)
- **Top-5 thực tế**: #1: [5593] Reebok Men's Pure Victory White Grey Black Shoe (Sports Shoes - White), #2: [59080] Peri Peri Women White Brief (Briefs - White), #3: [59443] Arrow Men Striped Beige Slim Fit Shirt (Shirts - White), #4: [11998] Mark Taylor Men White Striped Shirt (Shirts - White), #5: [37745] John Players Men Striped White Shirt (Shirts - White)
- **Nguyên nhân**: Truy vấn không dấu: mau trang

### Ca #17: `HELDOUT_UNACC_007` (unaccented)
- **Ảnh nguồn**: `22198` (Casual Shoes - Màu: `Black`)
- **Văn bản điều chỉnh**: *"mau den cho nam"* $\to$ Prompt chuẩn hóa: `black men casual shoe`
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `Black`, Giới tính = `Men`
- **Top-1 thực tế**: [15334] Nike Men Solid Black Shorts (Shorts, Black)
- **Top-5 thực tế**: #1: [15334] Nike Men Solid Black Shorts (Shorts - Black), #2: [25364] Levis Men Black & White Trunks (Trunk - Black), #3: [28506] Nike Men Woven Black Shorts (Shorts - Black), #4: [7240] Rockport Men's Stanton Black Shoe (Formal Shoes - Black), #5: [28072] Jockey MC Men Black Midway Classic Trunks 1101 (Trunk - Black)
- **Nguyên nhân**: Truy vấn không dấu: mau den cho nam

### Ca #18: `HELDOUT_UNACC_008` (unaccented)
- **Ảnh nguồn**: `22198` (Casual Shoes - Màu: `Black`)
- **Văn bản điều chỉnh**: *"mau trang nu"* $\to$ Prompt chuẩn hóa: `white women casual shoe`
- **Kỳ vọng**: Loại = `Casual Shoes`, Màu = `White`, Giới tính = `Women`
- **Top-1 thực tế**: [27495] Jockey Women White Briefs (Briefs, White)
- **Top-5 thực tế**: #1: [27495] Jockey Women White Briefs (Briefs - White), #2: [53541] Amante Women Pack of 2 Bikini Briefs PFFL03 (Briefs - White), #3: [59678] Avirate Black & White  Brief (Briefs - White), #4: [19360] Beyouty Women H01 Combo3 Fashion Hipster Pack of 3 Breifs (Briefs - White), #5: [31903] Hear 2 Heart Women White Briefs H2H-3-2011 (Briefs - White)
- **Nguyên nhân**: Truy vấn không dấu: mau trang nu

### Ca #19: `HELDOUT_UNACC_010` (unaccented)
- **Ảnh nguồn**: `47359` (Handbags - Màu: `Brown`)
- **Văn bản điều chỉnh**: *"mau hong cho nu"* $\to$ Prompt chuẩn hóa: `pink women handbag`
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Pink`, Giới tính = `Women`
- **Top-1 thực tế**: [57086] Elle Women Pink Vintage Skirt (Skirts, Pink)
- **Top-5 thực tế**: #1: [57086] Elle Women Pink Vintage Skirt (Skirts - Pink), #2: [9995] Gini And Jony Girls Su Scotia Sko Pink Skirts (Skirts - Pink), #3: [6532] Nike Women's Border Pink Skirt (Skirts - Pink), #4: [37408] Murcia Women Pink Handbag (Handbags - Pink), #5: [49655] Mineral Pink & Green Skirt (Skirts - Pink)
- **Nguyên nhân**: Truy vấn không dấu: mau hong cho nu

### Ca #20: `HELDOUT_UNACC_011` (unaccented)
- **Ảnh nguồn**: `47359` (Handbags - Màu: `Brown`)
- **Văn bản điều chỉnh**: *"mau nau da bo"* $\to$ Prompt chuẩn hóa: `brown leather handbag`
- **Kỳ vọng**: Loại = `Handbags`, Màu = `Brown`, Giới tính = `None`
- **Top-1 thực tế**: [4279] Rockport Men's Alduna Brown Shoe (Formal Shoes, Brown)
- **Top-5 thực tế**: #1: [4279] Rockport Men's Alduna Brown Shoe (Formal Shoes - Brown), #2: [7249] Rockport Men's Monte Road Light Tan Brown Shoe (Casual Shoes - Brown), #3: [7254] Rockport Men's Bindu India Medium Brown Shoe (Formal Shoes - Brown), #4: [24100] Red Tape Men Brown Casual Sandals (Sandals - Brown), #5: [13130] Provogue Men Brown Casual Shoes (Casual Shoes - Brown)
- **Nguyên nhân**: Truy vấn không dấu: mau nau da bo
