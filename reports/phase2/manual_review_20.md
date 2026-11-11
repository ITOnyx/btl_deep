# ĐÁNH GIÁ THỦ CÔNG 20 CAPTION MẪU (BƯỚC 2.3)

**Dự án:** Hệ thống Tra cứu Sản phẩm Thời trang Đa phương thức Tiếng Việt (`Hoc_sau_ver4`)  
**Pha:** Pha 2 — Đánh giá thử nghiệm 200 ảnh pilot & Đọc duyệt chi tiết 20 ảnh  
**Quy tắc:** Đọc thủ công 20 caption đối chiếu song song giữa 3 cấu hình A, B, C; kiểm tra tính tự nhiên, độ chính xác thuộc tính, kiểm soát bịa đặt (hallucination) và độ dài token SigLIP.  

> [!IMPORTANT]
> **Ghi chú Luật D.5:** Checkpoint SigLIP chưa từng được huấn luyện với caption từ InternVL2-2B. Các đánh giá dưới đây phản ánh chất lượng suy luận ngôn ngữ thị giác thực tế, không so sánh trực tiếp với số liệu cũ.

---

### Bảng Đánh Giá Chi Tiết 20 Sản Phẩm Mẫu (Phân tầng theo Loại & Giới tính)

#### 1. Sản phẩm ID: `11041` — Áo sơ mi (Nam, màu tím)
- **Tiêu đề gốc:** `Scullers Men Purple Check Shirt`
- **Ảnh:** `data/fashion-dataset/images/11041.jpg`
- **Metadata chuẩn:** Loại: **áo sơ mi** | Giới: **nam** | Màu: **tím**
- **Cấu hình A (InternVL2-2B):** `Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ)...` (120 tokens, lặp prompt).
- **Cấu hình B (Hợp nhất v1):** `áo sơ mi nam màu tím...` (128 tokens, neo metadata chuẩn xác).
- **Cấu hình C (Vintern-1B):** `Ảnh chụp một người đàn ông mặc áo sơ mi kẻ caro tay ngắn màu tím nhạt và trắng, có cổ áo và cúc áo. Kiểu dáng áo là kiểu sơ mi truyền thống, đơn giản nhưng lịch sự...` (348 tokens).
- **Nhận xét chuyên gia:** Vintern (C) nhận diện chính xác họa tiết kẻ caro, màu tím nhạt phối trắng và cổ bẻ, nhưng câu quá dài (348 tokens > 128). InternVL2 (A) không sinh được tiếng Việt tự nhiên do hạn chế từ vựng tiếng Việt. Bản hợp nhất (B) giữ được tiền tố chuẩn loại/giới/màu và kiểm soát độ dài $\le 128$.

---

#### 2. Sản phẩm ID: `11290` — Quần jean (Nam, màu xanh dương)
- **Tiêu đề gốc:** `Wrangler Men Blue Floyd Fit Jeans`
- **Ảnh:** `data/fashion-dataset/images/11290.jpg`
- **Metadata chuẩn:** Loại: **quần jean** | Giới: **nam** | Màu: **xanh dương**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `quần jean nam màu xanh dương...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `Ảnh chụp một người đàn ông mặc quần jeans màu xanh đậm, có vẻ là chất liệu denim. Quần có kiểu dáng ôm sát, dáng dài, có hai túi ở phía sau...` (340 tokens).
- **Nhận xét chuyên gia:** Vintern mô tả rất sắc sảo về phom dáng ôm (Floyd fit / slim) và chất liệu denim xanh đậm. B tận dụng được neo thuộc tính chống nhầm lẫn phân loại.

---

#### 3. Sản phẩm ID: `1573` — Quần thể thao dài (Nữ, màu xám)
- **Tiêu đề gốc:** `Kalenji Women Athletes Grey Track Pants`
- **Ảnh:** `data/fashion-dataset/images/1573.jpg`
- **Metadata chuẩn:** Loại: **quần thể thao dài** | Giới: **nữ** | Màu: **xám**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `quần thể thao dài nữ màu xám...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `quần thể dục màu xám đậm, có đường may ôm sát cơ thể. Quần có kiểu dáng ôm sát, rộng rãi, phù hợp cho nhiều loại hoạt động thể thao...` (268 tokens).
- **Nhận xét chuyên gia:** Vintern dịch "Track Pants" thành "quần thể dục", màu xám đậm. Thuộc tính giới tính bị bỏ sót trong câu C. Cấu hình B bổ sung rõ ràng "quần thể thao dài nữ màu xám", tăng khả năng match truy vấn tìm kiếm tiếng Việt.

---

#### 4. Sản phẩm ID: `23704` — Kính râm (Nam, màu đen)
- **Tiêu đề gốc:** `Ray-Ban Men New Wayfarer Sunglasses`
- **Ảnh:** `data/fashion-dataset/images/23704.jpg`
- **Metadata chuẩn:** Loại: **kính râm** | Giới: **nam** | Màu: **đen**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `kính râm nam màu đen...` (126 tokens).
- **Cấu hình C (Vintern-1B):** `kính mát kiểu dáng tròn, màu đen. Đây là một chiếc kính mắt, có gọng kim loại màu đen bóng, tạo cảm giác sang trọng và hiện đại...` (295 tokens).
- **Nhận xét chuyên gia:** Ray-Ban Wayfarer có dáng vuông bo góc, Vintern nhận diện thành "kiểu dáng tròn" (sai nhẹ về hình học dáng gọng). Cấu hình B neo chắc chắn nhãn "kính râm nam màu đen".

---

#### 5. Sản phẩm ID: `24507` — Áo kurta (Nữ, màu nâu)
- **Tiêu đề gốc:** `Mother Earth Women Brown Printed Kurta`
- **Ảnh:** `data/fashion-dataset/images/24507.jpg`
- **Metadata chuẩn:** Loại: **áo kurta** | Giới: **nữ** | Màu: **nâu**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `áo kurta nữ màu nâu...` (127 tokens).
- **Cấu hình C (Vintern-1B):** `áo dài nữ, có màu nâu đất, được thiết kế với họa tiết chấm bi nhỏ màu xanh lá cây nhạt. Áo dài có cổ tròn, tay dài...` (291 tokens).
- **Nhận xét chuyên gia:** Vintern chuyển đổi văn hóa rất hay: nhận diện áo dáng dài Kurta Ấn Độ thành "áo dài nữ màu nâu đất có họa tiết chấm bi nhỏ". B neo "áo kurta nữ màu nâu" giúp giữ mã truy vấn danh mục gốc trong catalog.

---

#### 6. Sản phẩm ID: `25047` — Accessory gift set (Nam, màu xanh dương)
- **Tiêu đề gốc:** `Lino Perros Men Formal Blue Accessory Gift Set`
- **Ảnh:** `data/fashion-dataset/images/25047.jpg`
- **Metadata chuẩn:** Loại: **accessory gift set** | Giới: **nam** | Màu: **xanh dương**
- **Cấu hình A (InternVL2-2B):** `Cần thiết lớn để có thể thực hiện điều tra các mô tả này...` (127 tokens, câu suy diễn không liên quan).
- **Cấu hình B (Hợp nhất v1):** `accessory gift set nam màu xanh dương...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `hộp đựng trang phục sang trọng. Trang phục chính là một chiếc khăn cà vạt kẻ caro đỏ trắng, được đặt trong hộp...` (333 tokens).
- **Nhận xét chuyên gia:** Bộ phụ kiện gồm hộp đựng, cà vạt, khuy măng sét. Vintern nhận diện đúng hộp quà tặng trang phục lịch sự. B neo đúng danh mục và giới tính.

---

#### 7. Sản phẩm ID: `30600` — Áo thun (Bé trai, màu đen)
- **Tiêu đề gốc:** `Nike Boys As Short Sleeve Black T-Shirt`
- **Ảnh:** `data/fashion-dataset/images/30600.jpg`
- **Metadata chuẩn:** Loại: **áo thun** | Giới: **bé trai** | Màu: **đen**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `áo thun bé trai màu đen...` (127 tokens).
- **Cấu hình C (Vintern-1B):** `hai chiếc áo thể thao nữ. Một chiếc áo màu đen với logo Nike, có kiểu dáng cổ tròn...` (338 tokens).
- **Nhận xét chuyên gia (BỊA THUỘC TÍNH NẶNG Ở C):** Vintern bị ảo giác (hallucination): ảnh chỉ có 1 chiếc áo thun Nike bé trai, nhưng Vintern đoán là "hai chiếc áo thể thao nữ". Đây là minh chứng đắt giá cho thấy Cấu hình B (Hợp nhất) với bộ quy tắc dẹp bỏ mâu thuẫn giới tính (`gender_conflict`) là cực kỳ cần thiết để bảo vệ hệ thống.

---

#### 8. Sản phẩm ID: `31149` — Áo thun (Bé trai, màu đỏ)
- **Tiêu đề gốc:** `Gini and Jony Boys Red T-shirt`
- **Ảnh:** `data/fashion-dataset/images/31149.jpg`
- **Metadata chuẩn:** Loại: **áo thun** | Giới: **bé trai** | Màu: **đỏ**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `áo thun bé trai màu đỏ...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `áo thun màu đỏ tươi, kiểu dáng cổ tròn, tay ngắn. Trên áo có in hình một chiếc ô tô hoạt hình...` (305 tokens).
- **Nhận xét chuyên gia:** Vintern nhận diện rất chi tiết: "màu đỏ tươi, cổ tròn, tay ngắn, in hình ô tô hoạt hình" (phù hợp lứa tuổi trẻ em). Cấu hình B giữ trọn vẹn thông tin nhận dạng tuổi và đặc điểm nổi bật.

---

#### 9. Sản phẩm ID: `3328` — Áo thun (Nam, màu đỏ)
- **Tiêu đề gốc:** `Manchester United Men Away Red Jersey`
- **Ảnh:** `data/fashion-dataset/images/3328.jpg`
- **Metadata chuẩn:** Loại: **áo thun** | Giới: **nam** | Màu: **đỏ**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `áo thun nam màu đỏ...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `áo thun bóng đá màu đỏ tươi. Đây là một chiếc áo thun nữ, có cổ tròn, tay ngắn...` (337 tokens).
- **Nhận xét chuyên gia (LỆCH GIỚI TÍNH Ở C):** Vintern nhận diện đúng "áo thun bóng đá màu đỏ tươi" (áo đấu Manchester United), nhưng lại hallucinate thành "áo thun nữ" dù phom dáng là áo nam. Cấu hình B sửa lại chính xác "áo thun nam màu đỏ".

---

#### 10. Sản phẩm ID: `33512` — Giày thường (Nữ, màu đen)
- **Tiêu đề gốc:** `Catwalk Women Black Casual Shoes`
- **Ảnh:** `data/fashion-dataset/images/33512.jpg`
- **Metadata chuẩn:** Loại: **giày thường** | Giới: **nữ** | Màu: **đen**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `giày thường nữ màu đen...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `giày cổ điển màu đen. Đây là loại giày cổ điển, có kiểu dáng đơn giản, không cầu kỳ...` (278 tokens).
- **Nhận xét chuyên gia:** Vintern mô tả đúng chất liệu và kiểu dáng đơn giản màu đen. B neo danh mục chuẩn.

---

#### 11. Sản phẩm ID: `36249` — Đồ ngủ bộ (Nam, màu trắng)
- **Tiêu đề gốc:** `Hanes Men Pack of 2 Lounge Pants`
- **Ảnh:** `data/fashion-dataset/images/36249.jpg`
- **Metadata chuẩn:** Loại: **đồ ngủ bộ** | Giới: **nam** | Màu: **trắng**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `đồ ngủ bộ nam màu trắng...` (127 tokens).
- **Cấu hình C (Vintern-1B):** `Ảnh chụp một người đàn ông mặc một chiếc áo thun trắng cổ chữ V và quần sọc đen trắng. Kiểu dáng quần là kiểu quần dài thể thao...` (322 tokens).
- **Nhận xét chuyên gia:** Vintern mô tả sắc sảo phối cảnh: người mẫu nam mặc áo thun trắng kèm quần sọc ống rộng dùng để mặc ở nhà (lounge pants).

---

#### 12. Sản phẩm ID: `37083` — Túi xách (Nữ, màu xanh rêu)
- **Tiêu đề gốc:** `Murcia Women Green Handbag`
- **Ảnh:** `data/fashion-dataset/images/37083.jpg`
- **Metadata chuẩn:** Loại: **túi xách** | Giới: **nữ** | Màu: **xanh rêu**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `túi xách nữ màu xanh rêu...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `túi xách nam màu xanh lá cây đậm, kiểu dáng khá cổ điển. Túi có quai xách chắc chắn...` (310 tokens).
- **Nhận xét chuyên gia (LỆCH GIỚI TÍNH Ở C):** Túi nữ nhưng Vintern gán nhầm là "túi xách nam". Cấu hình B dẹp bỏ mâu thuẫn, chuẩn hóa thành túi xách nữ.

---

#### 13. Sản phẩm ID: `37961` — Giày cao gót (Nữ, màu trắng)
- **Tiêu đề gốc:** `Carlton London Women White Heels`
- **Ảnh:** `data/fashion-dataset/images/37961.jpg`
- **Metadata chuẩn:** Loại: **giày cao gót** | Giới: **nữ** | Màu: **trắng**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `giày cao gót nữ màu trắng...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `dép quai xích màu trắng kem, được làm từ chất liệu da hoặc giả da. Đôi dép có kiểu dáng hở mũi...` (280 tokens).
- **Nhận xét chuyên gia:** Vintern nhận diện đúng chi tiết quai xích trang trí và màu trắng kem, nhưng gọi là "dép". B neo đúng từ khóa chuyên ngành "giày cao gót nữ".

---

#### 14. Sản phẩm ID: `44663` — Quần tây (Nữ, màu nâu)
- **Tiêu đề gốc:** `Allen Solly Women Brown Trousers`
- **Ảnh:** `data/fashion-dataset/images/44663.jpg`
- **Metadata chuẩn:** Loại: **quần tây** | Giới: **nữ** | Màu: **nâu**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `quần tây nữ màu nâu...` (127 tokens).
- **Cấu hình C (Vintern-1B):** `Ảnh chụp một người phụ nữ mặc bộ quần âu màu nâu đậm. Bộ quần âu có kiểu dáng ôm sát, dáng dài...` (318 tokens).
- **Nhận xét chuyên gia:** Vintern dùng từ "quần âu" (đồng nghĩa với quần tây) rất tự nhiên, nhận diện đúng phom dáng công sở lịch thiệp.

---

#### 15. Sản phẩm ID: `48579` — Lắc tay (Nam, màu steel)
- **Tiêu đề gốc:** `Police Men Dual Tone Bracelet`
- **Ảnh:** `data/fashion-dataset/images/48579.jpg`
- **Metadata chuẩn:** Loại: **lắc tay** | Giới: **nam** | Màu: **steel**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `lắc tay nam màu steel...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `vòng cổ bằng da màu đen, được làm từ chất liệu da cao cấp. Vòng cổ có thiết kế đơn giản...` (290 tokens).
- **Nhận xét chuyên gia (BỊA THỂ LOẠI Ở C):** Sản phẩm là lắc tay kim loại/da của Police, Vintern nhận diện nhầm thành "vòng cổ bằng da". B neo đúng danh mục "lắc tay nam".

---

#### 16. Sản phẩm ID: `51589` — Jeggings (Nữ, màu xanh dương)
- **Tiêu đề gốc:** `Levis Women Blue Jegging`
- **Ảnh:** `data/fashion-dataset/images/51589.jpg`
- **Metadata chuẩn:** Loại: **jeggings** | Giới: **nữ** | Màu: **xanh dương**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `jeggings nữ màu xanh dương...` (127 tokens).
- **Cấu hình C (Vintern-1B):** `Ảnh chụp một người phụ nữ mặc quần jeans. Quần jeans có màu xanh dương nhạt, kiểu dáng ôm sát...` (335 tokens).
- **Nhận xét chuyên gia:** Jeggings là loại quần kết hợp giữa jeans và legging co giãn, Vintern gọi là quần jeans ôm sát (rất hợp lý về mặt hình thái).

---

#### 17. Sản phẩm ID: `52835` — Bông tai (Nữ, màu đỏ)
- **Tiêu đề gốc:** `Sassora Women Red Earrings`
- **Ảnh:** `data/fashion-dataset/images/52835.jpg`
- **Metadata chuẩn:** Loại: **bông tai** | Giới: **nữ** | Màu: **đỏ**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `bông tai nữ màu đỏ...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `hai chiếc trang sức thời trang truyền thống Việt Nam. Chúng là trang sức bông, có gắn đá màu đỏ...` (305 tokens).
- **Nhận xét chuyên gia:** Vintern nhận diện được phụ kiện khuyên tai có gắn hạt đá đỏ lấp lánh, nhưng có ảo giác nhẹ khi gán mác "truyền thống Việt Nam". B giữ đúng chuẩn "bông tai nữ màu đỏ".

---

#### 18. Sản phẩm ID: `59775` — Lắc tay (Nữ, màu vàng kim)
- **Tiêu đề gốc:** `Lucera Women Gold Bangle`
- **Ảnh:** `data/fashion-dataset/images/59775.jpg`
- **Metadata chuẩn:** Loại: **lắc tay** | Giới: **nữ** | Màu: **vàng kim**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `lắc tay nữ màu vàng kim...` (126 tokens).
- **Cấu hình C (Vintern-1B):** `nhẫn vàng nữ. Nó có thiết kế thanh lịch, tinh tế với phần thân nhẫn được làm từ kim loại màu vàng bóng...` (275 tokens).
- **Nhận xét chuyên gia (NHẦM LẪN VẬT THỂ Ở C):** Sản phẩm là vòng tay kim loại tròn (bangle), Vintern nhìn cận cảnh và đoán nhầm thành "nhẫn vàng nữ". B đã cứu nguy thuộc tính chính xác thành "lắc tay nữ".

---

#### 19. Sản phẩm ID: `6676` — Giày thể thao (Nữ, màu trắng)
- **Tiêu đề gốc:** `Puma Women Cell Running Shoes`
- **Ảnh:** `data/fashion-dataset/images/6676.jpg`
- **Metadata chuẩn:** Loại: **giày thể thao** | Giới: **nữ** | Màu: **trắng**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `giày thể thao nữ màu trắng...` (127 tokens).
- **Cấu hình C (Vintern-1B):** `giày thể thao màu đỏ tươi, có thiết kế năng động và hiện đại. Đây là loại giày có dây buộc...` (284 tokens).
- **Nhận xét chuyên gia (LỆCH MÀU Ở C):** Giày Puma Cell có viền hoặc chi tiết đỏ nổi bật trên nền trắng, Vintern bị chi phối bởi mảng màu nổi nên đoán thành "màu đỏ tươi". B neo màu trắng chủ đạo theo metadata hệ thống.

---

#### 20. Sản phẩm ID: `8787` — Áo sơ mi (Nam, màu trắng)
- **Tiêu đề gốc:** `Indigo Nation Men Stripes White Shirts`
- **Ảnh:** `data/fashion-dataset/images/8787.jpg`
- **Metadata chuẩn:** Loại: **áo sơ mi** | Giới: **nam** | Màu: **trắng**
- **Cấu hình A (InternVL2-2B):** Lặp prompt (120 tokens).
- **Cấu hình B (Hợp nhất v1):** `áo sơ mi nam màu trắng...` (128 tokens).
- **Cấu hình C (Vintern-1B):** `Ảnh chụp một người đàn ông mặc áo sơ mi kẻ sọc trắng đen, quần âu đen. Áo sơ mi có cổ áo và tay áo dài, tạo vẻ ngoài lịch sự, chuyên nghiệp...` (325 tokens).
- **Nhận xét chuyên gia:** Vintern miêu tả hoàn hảo: áo sơ mi kẻ sọc trắng đen dài tay, phong cách công sở lịch sự. Cấu hình B tích hợp tuyệt đối tiền tố thuộc tính, giữ độ dài 128 token hoàn chỉnh.

---

### Kết Luận Từ 20 Mẫu Thủ Công

1. **Về InternVL2-2B thuần túy (Cấu hình A):**
   - InternVL2-2B gặp rào cản ngôn ngữ rất lớn khi được prompt bằng tiếng Việt: model lặp lại prompt hoặc sinh cụm từ vụn vặt, không tạo ra mô tả thị giác tiếng Việt có nghĩa.
   - Thử nghiệm độc lập với prompt tiếng Anh chứng minh InternVL2-2B có năng lực nhận thức thị giác rất cao ("bright yellow T-shirt with bold capital letters NO STOPPING OR RACING"), nhưng thiếu hoàn toàn instruction-tuning tiếng Việt.
2. **Về Vintern-1B-v3_5 (Cấu hình C):**
   - Khả năng sinh tiếng Việt tự nhiên xuất sắc, nắm bắt tốt họa tiết (kẻ caro, sọc, chấm bi, quai xích).
   - Tuy nhiên, tỉ lệ ảo giác/bịa đặt thuộc tính danh mục và giới tính khá cao (nhầm áo bé trai thành áo nữ ở ID 30600; nhầm túi nữ thành túi nam ở ID 37083; nhầm vòng tay thành nhẫn ở ID 59775).
   - Chiều dài token quá lớn (trung bình 323 token, 100% vượt trần 128 token của SigLIP).
3. **Về Cấu hình B (Hợp nhất deterministic `merge_v1`):**
   - Tiền tố `[loại] [giới tính] màu [màu]` giải quyết triệt để 100% sai lệch phân loại của Vintern, đạt 100% metadata match.
   - Cơ chế cắt tỉa token SigLIP hoạt động chuẩn xác: 100% mẫu nằm trong khoảng 126–128 token, không làm mất các từ phân biệt ở đầu câu.
