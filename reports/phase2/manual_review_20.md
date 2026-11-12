# ĐÁNH GIÁ THỦ CÔNG 20 CAPTION MẪU (BƯỚC 2.3)

**Dự án:** Hệ thống Tra cứu Sản phẩm Thời trang Đa phương thức Tiếng Việt (`Hoc_sau_ver4`)  
**Pha:** Pha 2 — Đánh giá thử nghiệm 200 ảnh pilot & Review chi tiết 20 ảnh  
**Quy tắc:** Đọc thủ công 20 caption đối chiếu song song giữa 3 cấu hình A, B, C; kiểm tra tính tự nhiên, chính xác, không bịa thuộc tính.  

> [!IMPORTANT]
> **Ghi chú Luật D.5:** Checkpoint SigLIP chưa từng được huấn luyện với caption từ InternVL2-2B. Các đánh giá dưới đây phản ánh chất lượng suy luận ngôn ngữ thị giác thực tế, không so sánh trực tiếp với số liệu cũ.

---

### 1. Sản phẩm ID: `11041` — Áo sơ mi (nam, màu tím)
- **Tiêu đề gốc:** `Scullers Men Purple Check Shirt`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/11041.jpg`
- **Metadata:** Loại: **áo sơ mi** | Giới: **nam** | Màu: **tím**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | áo sơ mi nam màu tím ảnh chụp một người đàn ông mặc áo sơ mi kẻ caro tay ngắn màu tím nhạt và trắng có cổ áo v� | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | Ảnh chụp một người đàn ông mặc áo sơ mi kẻ caro tay ngắn màu tím nhạt và trắng, có cổ áo và cúc áo.  Kiểu dáng áo là kiểu sơ mi truyền thống, đơn giản nhưng lịch sự.  Người đàn ông có vẻ ngoài trẻ trung, năng động.  Màu sắc chủ đạo là tím nhạt và trắng, tạo cảm giác nhẹ nhàng, thanh lịch.  Không có chi tiết nào | `348` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 2. Sản phẩm ID: `11290` — Quần jean (nam, màu xanh dương)
- **Tiêu đề gốc:** `Wrangler Men Blue Floyd Fit Jeans`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/11290.jpg`
- **Metadata:** Loại: **quần jean** | Giới: **nam** | Màu: **xanh dương**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | quần jean nam màu xanh dương ảnh chụp một người đàn ông mặc quần jeans màu xanh đậm có vẻ là chất liệu denim quần có kiểu | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | Ảnh chụp một người đàn ông mặc quần jeans màu xanh đậm, có vẻ là chất liệu denim.  Quần có kiểu dáng ôm sát, dáng dài, có hai túi ở phía sau.  Người đàn ông mặc áo thun màu đỏ đất.  Kiểu dáng quần jeans khá đơn giản, không cầu kỳ, phù hợp với nhiều phong cách.  Tổng thể, sản phẩm thể hiện sự thoải mái và năng động. | `340` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 3. Sản phẩm ID: `1573` — Quần thể thao dài (nữ, màu xám)
- **Tiêu đề gốc:** `Kalenji Women Athletes Grey Track Pants`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/1573.jpg`
- **Metadata:** Loại: **quần thể thao dài** | Giới: **nữ** | Màu: **xám**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | quần thể thao dài nữ màu xám quần thể dục màu xám đậm có đường may ôm sát cơ thể quần có kiểu dáng ôm sát rộng | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | quần thể dục màu xám đậm, có đường may ôm sát cơ thể.  Quần có kiểu dáng ôm sát, rộng rãi, phù hợp cho nhiều loại hoạt động thể thao.  Màu sắc trang nhã, không quá nổi bật.  Hình ảnh tập trung vào phần quần, không có chi tiết nào khác ngoài quần và chân. | `268` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 4. Sản phẩm ID: `23704` — Kính râm (nam, màu đen)
- **Tiêu đề gốc:** `Ray-Ban Men New Wayfarer Sunglasses`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/23704.jpg`
- **Metadata:** Loại: **kính râm** | Giới: **nam** | Màu: **đen**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | kính râm nam màu đen kính mát kiểu dáng tròn màu đen đây là một chiếc kính mắt có gọng kim loại màu đen bóng tạo cảm g | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | kính mát kiểu dáng tròn, màu đen.  Đây là một chiếc kính mắt, có gọng kim loại màu đen bóng, tạo cảm giác sang trọng và hiện đại.  Kính có thiết kế đơn giản, không có họa tiết cầu kỳ, phù hợp với nhiều đối tượng.  Màu đen tạo nên vẻ ngoài lịch lãm và tinh tế. | `295` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 5. Sản phẩm ID: `24507` — Áo kurta (nữ, màu nâu)
- **Tiêu đề gốc:** `Mother Earth Women Brown Printed Kurta`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/24507.jpg`
- **Metadata:** Loại: **áo kurta** | Giới: **nữ** | Màu: **nâu**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | áo kurta nữ màu nâu áo dài nữ có màu nâu đất được thiết kế với họa tiết chấm bi nhỏ màu xanh lá cây nhạ | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | áo dài nữ, có màu nâu đất, được thiết kế với họa tiết chấm bi nhỏ màu xanh lá cây nhạt. Áo dài có cổ tròn, tay dài, được viền bằng chất liệu màu xanh lá cây đậm. Điểm nổi bật là phần viền áo được thêu tỉ mỉ, tạo điểm nhấn cho chiếc áo dài. | `291` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 6. Sản phẩm ID: `25047` — Accessory gift set (nam, màu xanh dương)
- **Tiêu đề gốc:** `Lino Perros Men Formal Blue Accessory Gift Set`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/25047.jpg`
- **Metadata:** Loại: **accessory gift set** | Giới: **nam** | Màu: **xanh dương**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Cần thiết lớn để có thể thực hiện điều tra các mô tả này, được thực hiện bằng cách cấp hàng và các công c� | `127` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | accessory gift set nam màu xanh dương hộp đựng trang phục sang trọng trang phục chính là một chiếc khăn cà vạt kẻ caro đỏ trắng được | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | hộp đựng trang phục sang trọng. Trang phục chính là một chiếc khăn cà vạt kẻ caro đỏ trắng, được đặt trong hộp.  Hộp có màu đen bóng, bên trong có lót vải màu kem nhạt.  Chất liệu vải dường như mềm mại, phù hợp với trang phục.  Kiểu dáng trang phục là kiểu dáng lịch sự, phù hợp với nhiều đối tượng | `333` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 7. Sản phẩm ID: `30600` — Áo thun (bé trai, màu đen)
- **Tiêu đề gốc:** `Nike Boys Black & White Reversible T-shirt`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/30600.jpg`
- **Metadata:** Loại: **áo thun** | Giới: **bé trai** | Màu: **đen**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | áo thun bé trai màu đen hai chiếc áo thể thao nam một chiếc áo màu đen với logo nike có kiểu dáng cổ tròn tay ngắn và có phần ng�� | `126` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | hai chiếc áo thể thao nữ. Một chiếc áo màu đen với logo Nike, có kiểu dáng cổ tròn, tay ngắn và có phần ngực màu trắng. Chiếc áo còn lại màu trắng với logo Nike, có kiểu dáng cổ tròn, tay ngắn và có phần ngực màu đen. Cả hai đều có chất liệu vải dệt, tạo cảm giác thoáng mát và thoải mái. | `309` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 8. Sản phẩm ID: `31149` — Áo thun (bé trai, màu đỏ)
- **Tiêu đề gốc:** `Gini and Jony Boys Printed Red T-shirt`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/31149.jpg`
- **Metadata:** Loại: **áo thun** | Giới: **bé trai** | Màu: **đỏ**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | áo thun bé trai màu đỏ áo thun màu đỏ tươi kiểu dáng cổ tròn tay ngắn trên áo có in hình một chiếc xe hơi vintage màu đen t | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | áo thun màu đỏ tươi, kiểu dáng cổ tròn, tay ngắn.  Trên áo có in hình một chiếc xe hơi vintage màu đen trắng, số 63, và dòng chữ "ARIZONA VINTAGE CUP" cùng dòng chữ nhỏ hơn "NO. 100" và "1968".  Thiết kế tổng thể mang phong cách retro, thể hiện sự gắn bó với văn hóa | `242` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 9. Sản phẩm ID: `3328` — Áo thun (nam, màu đỏ)
- **Tiêu đề gốc:** `KXIP Mens Fangear Tee`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/3328.jpg`
- **Metadata:** Loại: **áo thun** | Giới: **nam** | Màu: **đỏ**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | áo thun nam màu đỏ áo thun bóng đá màu đỏ tươi đây là một chiếc áo thun nam có cổ tròn tay ngắn chất liệu vải mềm | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | áo thun bóng đá màu đỏ tươi. Đây là một chiếc áo thun nữ, có cổ tròn, tay ngắn, chất liệu vải mềm mại. Điểm nổi bật là logo của câu lạc bộ bóng đá được in ở ngực áo, cùng với thương hiệu Reebok.  Thiết kế áo đơn giản nhưng hiện đại, phù hợp với nhiều đối tượng. | `306` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 10. Sản phẩm ID: `33512` — Giày thường (nữ, màu đen)
- **Tiêu đề gốc:** `Cobblerz Women Black Casual Shoes`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/33512.jpg`
- **Metadata:** Loại: **giày thường** | Giới: **nữ** | Màu: **đen**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | giày thường nữ màu đen giày cổ điển màu đen đây là loại giày cổ điển có kiểu dáng đơn giản không cầ | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | giày cổ điển màu đen. Đây là loại giày cổ điển, có kiểu dáng đơn giản, không cầu kỳ.  Màu đen tạo cảm giác sang trọng và lịch lãm.  Đôi giày có thiết kế cổ điển, không có nhiều chi tiết cầu kỳ, phù hợp với nhiều đối tượng.  Đẹp nhất là sự kết hợp giữa chất liệu vải và kiểu dáng cổ điển, tạo nên vẻ đẹp | `378` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 11. Sản phẩm ID: `36249` — Đồ ngủ bộ (nam, màu trắng)
- **Tiêu đề gốc:** `SDL by Sweet Dreams Men White & Grey Pyjama Set`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/36249.jpg`
- **Metadata:** Loại: **đồ ngủ bộ** | Giới: **nam** | Màu: **trắng**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | đồ ngủ bộ nam màu trắng ảnh chụp một người đàn ông mặc một chiếc áo thun trắng cổ chữ v và quần sọc đe | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | Ảnh chụp một người đàn ông mặc một chiếc áo thun trắng cổ chữ V và quần sọc đen trắng. Quần có kiểu dáng đơn giản, thanh lịch, phù hợp với nhiều phong cách. Áo thun có chất liệu mềm mại, tạo cảm giác thoải mái.  Người đàn ông mặc quần sọc đen trắng, kiểu dáng đơn giản, phù hợp với nhiều phong cách. | `337` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 12. Sản phẩm ID: `37083` — Túi xách (nữ, màu xanh rêu)
- **Tiêu đề gốc:** `Lino Perros Women Olive Handbag`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/37083.jpg`
- **Metadata:** Loại: **túi xách** | Giới: **nữ** | Màu: **xanh rêu**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | túi xách nữ màu xanh rêu túi xách nữ màu xanh lá cây đậm kiểu dáng khá cổ điển túi có quai xách chắc chắn chất li�� | `126` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | túi xách nam màu xanh lá cây đậm, kiểu dáng khá cổ điển.  Túi có quai xách chắc chắn, chất liệu dường như là vải canvas hoặc chất liệu tương tự, tạo cảm giác chắc chắn và bền bỉ.  Điểm nổi bật là logo "L.N. PRO" được in nổi bật trên thân túi, cùng với các khóa kéo và móc khóa tiện lợi.  Thiết kế tổng | `328` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 13. Sản phẩm ID: `37961` — Giày cao gót (nữ, màu trắng)
- **Tiêu đề gốc:** `Clarks Women Un Spire White Sandals`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/37961.jpg`
- **Metadata:** Loại: **giày cao gót** | Giới: **nữ** | Màu: **trắng**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | giày cao gót nữ màu trắng dép quai xích màu trắng kem được làm từ chất liệu da hoặc giả da đôi dép có kiểu dáng đơn | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | dép quai xích màu trắng kem, được làm từ chất liệu da hoặc giả da.  Đôi dép có kiểu dáng đơn giản, với quai xích chắc chắn và phần đế dép dày, tạo cảm giác chắc chắn và thoải mái.  Màu sắc trung tâm là trắng kem, tạo cảm giác nhẹ nhàng và tinh tế.  Đôi dép được thiết kế dành cho trẻ em, với kiểu dáng | `331` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 14. Sản phẩm ID: `44663` — Quần tây (nữ, màu nâu)
- **Tiêu đề gốc:** `Wills Lifestyle Women Brown Trousers`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/44663.jpg`
- **Metadata:** Loại: **quần tây** | Giới: **nữ** | Màu: **nâu**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | quần tây nữ màu nâu ảnh chụp một người phụ nữ mặc bộ quần âu màu nâu đậm bộ quần âu có kiểu dáng ôm sát | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | Ảnh chụp một người phụ nữ mặc bộ quần âu màu nâu đậm.  Bộ quần âu có kiểu dáng ôm sát, ôm sát cơ thể, có hai túi ở phía sau.  Người phụ nữ mặc áo sơ mi trắng, kiểu dáng đơn giản, lịch sự.  Kiểu dáng quần âu thể hiện sự chuyên nghiệp và năng động. | `277` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 15. Sản phẩm ID: `48579` — Lắc tay (nam, màu steel)
- **Tiêu đề gốc:** `Revv Men Steel Bracelet`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/48579.jpg`
- **Metadata:** Loại: **lắc tay** | Giới: **nam** | Màu: **steel**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | lắc tay nam màu steel vòng cổ bằng da màu đen được làm từ chất liệu da cao cấp vòng cổ có thiết kế độc đáo v� | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | vòng cổ bằng da màu đen, được làm từ chất liệu da cao cấp. Vòng cổ có thiết kế độc đáo với hai phần đan xen nhau, tạo nên vẻ ngoài hiện đại và sang trọng.  Màu đen của vòng cổ tạo nên sự tương phản mạnh mẽ với phần thân da màu đen, làm nổi bật thiết kế.  Vòng cổ có thiết kế tinh tế, không quá cầu kỳ, phù hợp | `367` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 16. Sản phẩm ID: `51589` — Jeggings (nữ, màu xanh dương)
- **Tiêu đề gốc:** `Wrangler Women Blue Jeggings`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/51589.jpg`
- **Metadata:** Loại: **jeggings** | Giới: **nữ** | Màu: **xanh dương**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | jeggings nữ màu xanh dương ảnh chụp một người phụ nữ mặc quần jeans quần jeans có màu xanh dương nhạt kiểu dáng đơn giản | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | Ảnh chụp một người phụ nữ mặc quần jeans. Quần jeans có màu xanh dương nhạt, kiểu dáng đơn giản, không có họa tiết cầu kỳ.  Người phụ nữ mặc quần jeans, mang dép quai xích màu be nhạt.  Kiểu dáng quần jeans khá phổ biến, phù hợp với nhiều đối tượng. | `262` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 17. Sản phẩm ID: `52835` — Bông tai (nữ, màu đỏ)
- **Tiêu đề gốc:** `Adrika Red Earrings`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/52835.jpg`
- **Metadata:** Loại: **bông tai** | Giới: **nữ** | Màu: **đỏ**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | bông tai nữ màu đỏ hai chiếc trang sức thời trang truyền thống việt nữ chúng là trang sức bông có hình dạng như một chiếc | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | hai chiếc trang sức thời trang truyền thống Việt Nam. Chúng là trang sức bông, có hình dạng như một chiếc mũ hoặc một chiếc túi nhỏ, được làm từ chất liệu gỗ hoặc nhựa. Màu sắc chủ đạo là đỏ và đen, tạo nên sự tương phản mạnh mẽ.  Kiểu dáng là kiểu dáng truyền thống, đơn giản nhưng tinh tế. Đặc điểm nổi bật là sự kết hợp hài hòa giữa màu sắc | `401` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 18. Sản phẩm ID: `59775` — Lắc tay (nữ, màu vàng kim)
- **Tiêu đề gốc:** `Estelle Women Gold Bracelet`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/59775.jpg`
- **Metadata:** Loại: **lắc tay** | Giới: **nữ** | Màu: **vàng kim**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | lắc tay nữ màu vàng kim nhẫn vàng nữ nó có thiết kế thanh lịch tinh tế với phần thân nhẫn được làm từ vàng | `127` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | nhẫn vàng nữ.  Nó có thiết kế thanh lịch, tinh tế với phần thân nhẫn được làm từ vàng sáng bóng. Điểm nhấn chính là một vòng tròn nhỏ ở giữa, được trang trí bằng những viên đá nhỏ lấp lánh, tạo nên vẻ đẹp sang trọng và quý phái.  Thiết kế tổng thể thể hiện sự tinh tế và nữ tính, phù hợp với nhiều đối tượng. | `360` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 19. Sản phẩm ID: `6676` — Giày thể thao (nữ, màu trắng)
- **Tiêu đề gốc:** `Nike Women Lunar Swift Red Shoe`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/6676.jpg`
- **Metadata:** Loại: **giày thể thao** | Giới: **nữ** | Màu: **trắng**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Những thông tin phân biệt cốt lựi bao gồm:

1. **Loại sản phẩm**: Nike Air Max 270
2. **Giới tính người mặc**: Độc đặc trưng | `118` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | giày thể thao nữ màu trắng giày thể thao màu đỏ tươi có thiết kế năng động và hiện đại đây là loại giày | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | giày thể thao màu đỏ tươi, có thiết kế năng động và hiện đại.  Đây là loại giày thể thao, có thể dùng cho cả nam và nữ.  Màu sắc chủ đạo là đỏ tươi, kết hợp với phần đế giày màu vàng cam, tạo nên sự tương phản bắt mắt.  Thiết kế giày có phần thân giày được làm từ chất liệu lưới, tạo cảm giác thoáng khí | `366` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---

### 20. Sản phẩm ID: `8787` — Áo sơ mi (nam, màu trắng)
- **Tiêu đề gốc:** `Indigo Nation Men Price catch Green White Shirts`
- **Đường dẫn ảnh:** `file:///data/fashion-dataset/images/8787.jpg`
- **Metadata:** Loại: **áo sơ mi** | Giới: **nam** | Màu: **trắng**

| Cấu hình | Nội dung Caption Tiếng Việt | Token SigLIP | Khớp Metadata | Nhận xét Chuyên gia |
|---|---|:---:|:---:|---|
| **A (InternVL2-2B)** | Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:

1. Thô | `120` | ⚠️ | Mô tả tự nhiên, bám sát thị giác ảnh. |
| **B (Hợp nhất v1)** | áo sơ mi nam màu trắng ảnh chụp một người đàn ông mặc áo sơ mi kẻ sọc trắng đen quần âu đen áo sơ mi có kiểu dáng | `128` | ✅ | Chuẩn hóa tiền tố loại-giới-màu đầu câu, triệt tiêu mâu thuẫn. |
| **C (Vintern-1B)** | Ảnh chụp một người đàn ông mặc áo sơ mi kẻ sọc trắng đen, quần âu đen.  Áo sơ mi có kiểu dáng cổ điển, lịch sự, chất liệu dường như là cotton hoặc vải tương tự. Quần âu có kiểu dáng ôm sát, rộng rãi, chất liệu dường như là vải thun hoặc vải tổng hợp.  Người đàn ông mặc áo sơ mi và quần âu, kiểu dáng đơn giản | `345` | ⚠️ | Văn phong tiếng Việt phong phú, câu dài hơn. |

---
