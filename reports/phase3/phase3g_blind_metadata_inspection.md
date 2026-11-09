# Báo cáo Kiểm toán Metadata Chấm mù (Phase 3g Rev3)

**Dự án:** SigLIP E-commerce Visual Search (`Hoc_sau_ver4`)  
**Mục đích:** Kiểm toán khách quan chất lượng kết quả truy vấn trên 30 câu heldout với nhãn ẩn `Row_1` / `Row_2` (tương ứng với Row A và Row B trên contact sheet), hoàn toàn **không tiết lộ mô hình nào ứng với hàng nào**.  
**Tình trạng Contact Sheet:** Đã xác minh bằng code pixel analysis: **30/30 sheet đạt chuẩn** (mỗi sheet có đúng 10 ảnh ở Row A và 10 ảnh ở Row B, 0 ô trống).

---

## 1. Bảng Tổng hợp Hit@10 theo Metadata (26 câu có Spec chuẩn)

| Query ID | Nội dung Truy vấn | Số SP chuẩn trong Catalog | Parsed ArticleType | Parsed Gender | Tier L0-L6 | Row_1 Đúng/10 | Row_2 Đúng/10 | Hàng có Hit@10 = 0 |
|---|---|---|---|---|---|---|---|---|
| `x001` | sơ mi nam xanh navy | 166 sp | `Shirts` | `Men` | `L1_type_gender_colour` | 10/10 | 0/10 | **Row_2** |
| `x002` | áo thun nữ màu hồng | 315 sp | `Tshirts` | `Women` | `L1_type_gender_colour` | 10/10 | 1/10 | **Không có (cả 2 đều > 0)** |
| `x003` | quần jean nam màu xanh | 224 sp | `Jeans` | `Men` | `L1_type_gender_colour` | 10/10 | 2/10 | **Không có (cả 2 đều > 0)** |
| `x004` | giày cao gót nữ màu đen | 320 sp | `Heels` | `Women` | `L1_type_gender_colour` | 10/10 | 3/10 | **Không có (cả 2 đều > 0)** |
| `x005` | túi xách nữ màu đỏ | 106 sp | `Handbags` | `Women` | `L1_type_gender_colour` | 10/10 | 5/10 | **Không có (cả 2 đều > 0)** |
| `x006` | dép xỏ ngón nam màu đen | 168 sp | `Flip Flops` | `Men` | `L1_type_gender_colour` | 10/10 | 3/10 | **Không có (cả 2 đều > 0)** |
| `x007` | ao thun nam mau xam | 536 sp | `Tshirts` | `Men` | `L1_type_gender_colour` | 4/10 | 10/10 | **Không có (cả 2 đều > 0)** |
| `x008` | giay the thao nam mau den | 440 sp | `Sports Shoes` | `Men` | `L1_type_gender_colour` | 0/10 | 10/10 | **Row_1** |
| `x009` | vay nu mau xanh la | 13 sp | `Dresses` | `Women` | `L3_type_gender` | 1/10 | 8/10 | **Không có (cả 2 đều > 0)** |
| `x010` | quan short nu mau trang | 6 sp | `Shorts` | `Women` | `L3_type_gender` | 0/10 | 5/10 | **Row_1** |
| `x011` | áo thun Puma nam | 536 sp | `Tshirts` | `Men` | `L0_exact_with_brand` | 0/10 | 10/10 | **Row_1** |
| `x012` | giày Nike nam | 550 sp | `-` | `Men` | `L0_exact_with_brand` | 9/10 | 0/10 | **Row_2** |
| `x013` | giày Adidas nữ | 132 sp | `-` | `Women` | `L0_exact_with_brand` | 8/10 | 0/10 | **Row_2** |
| `x014` | đồng hồ Fossil nam | 105 sp | `Watches` | `Men` | `L0_exact_with_brand` | 10/10 | 0/10 | **Row_2** |
| `x015` | áo kurta Fabindia nam | 74 sp | `Kurtas` | `Men` | `L0_exact_with_brand` | 10/10 | 2/10 | **Không có (cả 2 đều > 0)** |
| `x016` | giày cao gót Catwalk | 607 sp | `Heels` | `-` | `L0_exact_with_brand` | 0/10 | 10/10 | **Row_1** |
| `x017` | túi xách Lavie nữ | 0 sp | `Handbags` | `Women` | `L3_type_gender` | 0/10 | 0/10 | **Row_1, Row_2** |
| `x018` | áo thun Benetton nam | 0 sp | `Tshirts` | `Men` | `L3_type_gender` | 0/10 | 0/10 | **Row_1, Row_2** |
| `x019` | áo sơ mi nam dài tay đi làm | 2597 sp | `Shirts` | `Men` | `L3_type_gender` | 10/10 | 2/10 | **Không có (cả 2 đều > 0)** |
| `x020` | giày chạy bộ nam | 1460 sp | `Sports Shoes` | `Men` | `L3_type_gender` | 10/10 | 0/10 | **Row_2** |
| `x021` | áo khoác nữ mùa đông | 56 sp | `Jackets` | `Women` | `L3_type_gender` | 1/10 | 10/10 | **Không có (cả 2 đều > 0)** |
| `x022` | đầm nữ đi tiệc màu đen | 103 sp | `Dresses` | `Women` | `L1_type_gender_colour` | 1/10 | 10/10 | **Không có (cả 2 đều > 0)** |
| `x023` | quần short nam thể thao màu xám | 45 sp | `Shorts` | `Men` | `L1_type_gender_colour` | 1/10 | 10/10 | **Không có (cả 2 đều > 0)** |
| `x024` | ba lô đi học màu xanh dương | 148 sp | `Backpacks` | `-` | `L4_type_only` | 0/10 | 10/10 | **Row_1** |
| `x025` | đồ đi biển | Đánh giá tay | `-` | `-` | `L6_fallback_all` | N/A | N/A | **Không có (cả 2 đều > 0)** |
| `x026` | quà tặng cho bố | Đánh giá tay | `-` | `-` | `L6_fallback_all` | N/A | N/A | **Không có (cả 2 đều > 0)** |
| `x027` | đồ tập gym | Đánh giá tay | `-` | `-` | `L6_fallback_all` | N/A | N/A | **Không có (cả 2 đều > 0)** |
| `x028` | đồ mặc ở nhà cho nữ | Đánh giá tay | `-` | `Women` | `L6_fallback_all` | N/A | N/A | **Không có (cả 2 đều > 0)** |
| `x029` | cà vạt nam | 216 sp | `Ties` | `Men` | `L3_type_gender` | 10/10 | 0/10 | **Row_2** |
| `x030` | khăn quàng cổ nữ | 163 sp | `Scarves` | `Women` | `L3_type_gender` | 10/10 | 4/10 | **Không có (cả 2 đều > 0)** |

---

## 2. Phân tích Chuyên sâu 5 Câu Trọng điểm (Deep Dive)

### 2.1 Query `x024`: "ba lô đi học màu xanh dương"
- **Ground Truth Spec:** `{'articleType': ['Backpacks'], 'colour': ['Blue', 'Navy Blue']}`
- **Số sản phẩm hợp lệ trong Catalog:** 148 sản phẩm.
- **Kết quả QueryParser:**
  - `articleType`: `Backpacks`
  - `gender`: `None`
  - `colour`: `Blue`
  - `brand`: `None`
- **Tier lọc ứng viên đã kích hoạt:** `L4_type_only`
- **Kết quả theo Metadata:** Row_1 có 0/10 sp đúng; Row_2 có 10/10 sp đúng.
- **Top 5 sản phẩm của Hệ thống:**
  | Rank | Item ID | Tên sản phẩm | articleType | gender | baseColour | Khớp Spec? |
  |---|---|---|---|---|---|---|
  | 1 | `25235` | Lino Perros Women Blue Backpacks | Backpacks | Women | Blue | **ĐÚNG** |
  | 2 | `21437` | Roxy Women Blue Backpack | Backpacks | Women | Blue | **ĐÚNG** |
  | 3 | `21189` | Quiksilver Unisex Blue Backpack | Backpacks | Unisex | Blue | **ĐÚNG** |
  | 4 | `24324` | ADIDAS Unisex Blue Backpack | Backpacks | Unisex | Blue | **ĐÚNG** |
  | 5 | `38131` | Wildcraft Unisex Blue Backpack | Backpacks | Unisex | Blue | **ĐÚNG** |

### 2.2 Query `x027`: "đồ tập gym"
- **Ground Truth Spec:** `None`
- **Số sản phẩm hợp lệ trong Catalog:** None sản phẩm.
- **Kết quả QueryParser:**
  - `articleType`: `None`
  - `gender`: `None`
  - `colour`: `Red`
  - `brand`: `None`
- **Tier lọc ứng viên đã kích hoạt:** `L6_fallback_all`
- **Kết quả theo Metadata:** Row_1 có None/10 sp đúng; Row_2 có None/10 sp đúng.
- **Top 5 sản phẩm của Hệ thống:**
  | Rank | Item ID | Tên sản phẩm | articleType | gender | baseColour | Khớp Spec? |
  |---|---|---|---|---|---|---|
  | 1 | `52417` | Red Rose Women Red Pyjama Set | Night suits | Women | Red | **N/A** |
  | 2 | `52442` | Red Rose Red Nightdress | Nightdress | Women | Red | **N/A** |
  | 3 | `52501` | Red Rose Red Nightdress | Baby Dolls | Women | Red | **N/A** |
  | 4 | `52508` | Red Rose Red Nightdress | Baby Dolls | Women | Red | **N/A** |
  | 5 | `33961` | Femella Women Red Leggings | Leggings | Women | Red | **N/A** |

### 2.3 Query `x029`: "cà vạt nam"
- **Ground Truth Spec:** `{'articleType': ['Ties'], 'gender': 'Men'}`
- **Số sản phẩm hợp lệ trong Catalog:** 216 sản phẩm.
- **Kết quả QueryParser:**
  - `articleType`: `Ties`
  - `gender`: `Men`
  - `colour`: `None`
  - `brand`: `None`
- **Tier lọc ứng viên đã kích hoạt:** `L3_type_gender`
- **Kết quả theo Metadata:** Row_1 có 10/10 sp đúng; Row_2 có 0/10 sp đúng.
- **Top 5 sản phẩm của Hệ thống:**
  | Rank | Item ID | Tên sản phẩm | articleType | gender | baseColour | Khớp Spec? |
  |---|---|---|---|---|---|---|
  | 1 | `58911` | Hakashi Men Grey Silk Tie | Ties | Men | Grey | **ĐÚNG** |
  | 2 | `49715` | Park Avenue Men Black Tie | Ties | Men | Black | **ĐÚNG** |
  | 3 | `50284` | Parx Men Black Tie | Ties | Men | Black | **ĐÚNG** |
  | 4 | `50285` | Parx Men Purple Tie | Ties | Men | Purple | **ĐÚNG** |
  | 5 | `49697` | Park Avenue Men Red Tie | Ties | Men | Red | **ĐÚNG** |

### 2.4 Query `x008`: "giay the thao nam mau den"
- **Ground Truth Spec:** `{'articleType': ['Sports Shoes'], 'gender': 'Men', 'colour': ['Black']}`
- **Số sản phẩm hợp lệ trong Catalog:** 440 sản phẩm.
- **Kết quả QueryParser:**
  - `articleType`: `Sports Shoes`
  - `gender`: `Men`
  - `colour`: `Black`
  - `brand`: `None`
- **Tier lọc ứng viên đã kích hoạt:** `L1_type_gender_colour`
- **Kết quả theo Metadata:** Row_1 có 0/10 sp đúng; Row_2 có 10/10 sp đúng.
- **Top 5 sản phẩm của Hệ thống:**
  | Rank | Item ID | Tên sản phẩm | articleType | gender | baseColour | Khớp Spec? |
  |---|---|---|---|---|---|---|
  | 1 | `34837` | ADIDAS Men Black Sports Shoes | Sports Shoes | Men | Black | **ĐÚNG** |
  | 2 | `4662` | ADIDAS Men  Elroy Low Black Shoes | Sports Shoes | Men | Black | **ĐÚNG** |
  | 3 | `15284` | Fila Men Tenet Black Sports Shoes | Sports Shoes | Men | Black | **ĐÚNG** |
  | 4 | `29227` | ADIDAS Men Mactelo Black Sports Shoes | Sports Shoes | Men | Black | **ĐÚNG** |
  | 5 | `44401` | Force 10 Men Black Shoes | Sports Shoes | Men | Black | **ĐÚNG** |

### 2.5 Query `x010`: "quan short nu mau trang"
- **Ground Truth Spec:** `{'articleType': ['Shorts'], 'gender': 'Women', 'colour': ['White']}`
- **Số sản phẩm hợp lệ trong Catalog:** 6 sản phẩm.
- **Kết quả QueryParser:**
  - `articleType`: `Shorts`
  - `gender`: `Women`
  - `colour`: `White`
  - `brand`: `None`
- **Tier lọc ứng viên đã kích hoạt:** `L3_type_gender`
- **Kết quả theo Metadata:** Row_1 có 0/10 sp đúng; Row_2 có 5/10 sp đúng.
- **Top 5 sản phẩm của Hệ thống:**
  | Rank | Item ID | Tên sản phẩm | articleType | gender | baseColour | Khớp Spec? |
  |---|---|---|---|---|---|---|
  | 1 | `5789` | Jealous 21 Women's Blue White Checkered Short | Shorts | Women | White | **ĐÚNG** |
  | 2 | `5788` | Jealous 21 Women's Checkered Pink White Short | Shorts | Women | White | **ĐÚNG** |
  | 3 | `6525` | Nike Women's Tempo yellow Short | Shorts | Women | Yellow | **SAI** |
  | 4 | `21431` | Roxy Women White Shorts | Shorts | Women | White | **ĐÚNG** |
  | 5 | `57896` | Femella Women Off White Shorts | Shorts | Women | Off White | **SAI** |
