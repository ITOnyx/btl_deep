# TỔNG KẾT BỘ DỮ LIỆU THỜI TRANG VIỆT NAM (`crawl_v2` — Out-of-Domain Showcase)

- **Ngày hoàn tất (UTC):** `2026-11-20T07:55:52+00:00`
- **Git HEAD:** `4daaf1ba996a19bf18dbab018d9062ef9285fb39`
- **Seed cố định:** `42`
- **User-Agent học thuật:** `AcademicResearchBot/2.0 (Hoc_sau_ver4 Vietnamese Fashion Visual Search; non-commercial academic evaluation; contact: nguyen-student@university.edu.vn)`
- **Tổng số ảnh sạch bàn giao (`filter_flag == pass`):** **10,429 ảnh** (1 ảnh đại diện chính thức / sản phẩm)
- **Kiểm tra hiển thị Thumbnail trong Gallery:** **10,429 / 10,429 hiển thị thành công (`0` link hỏng — đạt 100%)**
- **Tổng dung lượng ảnh gốc (nguyên byte trên đĩa):** **2827.07 MB** (`2.761 GB`)
- **Tổng dung lượng thumbnail (`<= 320px`):** **95.68 MB**
- **Tỷ lệ trạng thái nhãn:** **10,057 `ok` (96.43%)** | **372 `needs_review` (3.57%)**
- **Phiên bản thư viện:** `{"requests": "2.31.0", "Pillow": "10.2.0", "opencv-python": "5.0.0", "numpy": "2.5.3", "pandas": "3.0.0", "imagehash": "4.3.2", "PyYAML": "6.0.3"}`

---

## 1. Phân bố theo Thương hiệu (`brand`) & Trạng thái `robots.txt` / Điều khoản

Tất cả 11 thương hiệu đều được kiểm tra `robots.txt` trước khi crawl (bản chụp lưu tại `reports/crawl_v2/robots/`), tuân thủ tốc độ `<= 1 request/giây/domain`, `<= 2 kết nối đồng thời/domain`, không vượt tường đăng nhập hay CAPTCHA.

| Thương hiệu (`brand`) | Domain | Trạng thái `robots.txt` | Số ảnh sạch | Tỷ lệ (%) | Số ca `ok` | Số ca `needs_review` |
|---|---|---|---:|---:|---:|---:|
| **IVY moda** | `https://ivymoda.com` | HTTP 200 (Allow product/collection) | **2,473** | **23.71%** | 2,473 | 0 |
| **Aristino** | `https://aristino.com` | HTTP 200 (Allow product/collection) | **1,611** | **15.45%** | 1,501 | 110 |
| **Biti's** | `https://bitis.com.vn` | HTTP 200 (Allow product/collection) | **1,420** | **13.62%** | 1,337 | 83 |
| **Mắt Kính Shady** | `https://matkinhshady.com` | HTTP 200 (Allow product/collection) | **1,336** | **12.81%** | 1,287 | 49 |
| **WatchStore** | `https://www.watchstore.vn` | HTTP 200 (Allow product/collection) | **976** | **9.36%** | 976 | 0 |
| **Rabity** | `https://rabity.vn` | HTTP 200 (Allow product/collection) | **844** | **8.09%** | 817 | 27 |
| **YaMe** | `https://yame.vn` | HTTP 200 (Allow product/collection) | **689** | **6.61%** | 663 | 26 |
| **Juno** | `https://juno.vn` | HTTP 200 (Allow product/collection) | **672** | **6.44%** | 659 | 13 |
| **MIA.vn** | `https://mia.vn` | HTTP 200 (Allow product/collection) | **220** | **2.11%** | 220 | 0 |
| **Leonardo** | `https://www.leonardo.vn` | HTTP 200 (Allow product/collection) | **145** | **1.39%** | 87 | 58 |
| **Coolmate** | `https://www.coolmate.me` | HTTP 200 (Allow product/collection) | **43** | **0.41%** | 37 | 6 |

## 2. Phân bố theo Giới tính (`gender`)

| Giới tính (`gender`) | Số lượng ảnh | Tỷ lệ (%) |
|---|---:|---:|
| `Women` | 4,279 | 41.03% |
| `Men` | 3,694 | 35.42% |
| `Unisex` | 1,200 | 11.51% |
| `Girls` | 620 | 5.94% |
| `Boys` | 567 | 5.44% |
| `needs_review` | 69 | 0.66% |

## 3. Phân bố theo Loại sản phẩm (`articleType`)

| Loại sản phẩm (`articleType`) | Số lượng ảnh | Tỷ lệ (%) | Nguồn suy nhãn chủ đạo (`label_source`) |
|---|---:|---:|---|
| `Sunglasses` | 1,306 | 12.52% | `breadcrumb` |
| `Tshirts` | 1,218 | 11.68% | `breadcrumb` |
| `Shirts` | 1,028 | 9.86% | `breadcrumb` |
| `Watches` | 976 | 9.36% | `breadcrumb` |
| `Dresses` | 889 | 8.52% | `breadcrumb` |
| `Sports Shoes` | 503 | 4.82% | `breadcrumb` |
| `Sandals` | 502 | 4.81% | `breadcrumb` |
| `Trousers` | 474 | 4.55% | `breadcrumb` |
| `Jackets` | 447 | 4.29% | `breadcrumb` |
| `Skirts` | 439 | 4.21% | `breadcrumb` |
| `needs_review` | 320 | 3.07% | `breadcrumb` |
| `Flip Flops` | 281 | 2.69% | `breadcrumb` |
| `Handbags` | 264 | 2.53% | `breadcrumb` |
| `Shorts` | 249 | 2.39% | `breadcrumb` |
| `Backpacks` | 180 | 1.73% | `breadcrumb` |
| `Flats` | 109 | 1.05% | `breadcrumb` |
| `Heels` | 104 | 1.00% | `breadcrumb` |
| `Clothing Set` | 101 | 0.97% | `breadcrumb` |
| `Belts` | 91 | 0.87% | `breadcrumb` |
| `Socks` | 86 | 0.82% | `breadcrumb` |
| `Wallets` | 85 | 0.82% | `breadcrumb` |
| `Trolley Bag` | 81 | 0.78% | `breadcrumb` |
| `Sweaters` | 70 | 0.67% | `breadcrumb` |
| `Jeans` | 54 | 0.52% | `breadcrumb` |
| `Formal Shoes` | 52 | 0.50% | `breadcrumb` |
| `Casual Shoes` | 47 | 0.45% | `breadcrumb` |
| `Ties` | 47 | 0.45% | `breadcrumb` |
| `Travel Accessory` | 45 | 0.43% | `breadcrumb` |
| `Boxers` | 43 | 0.41% | `breadcrumb` |
| `Messenger Bag` | 41 | 0.39% | `title` |
| `Briefs` | 40 | 0.38% | `breadcrumb` |
| `Duffel Bag` | 39 | 0.37% | `breadcrumb` |
| `Tops` | 34 | 0.33% | `breadcrumb` |
| `Clutches` | 27 | 0.26% | `breadcrumb` |
| `Bracelet` | 27 | 0.26% | `title` |
| `Blazers` | 25 | 0.24% | `breadcrumb` |
| `Caps` | 23 | 0.22% | `breadcrumb` |
| `Jumpsuit` | 16 | 0.15% | `breadcrumb` |
| `Sweatshirts` | 15 | 0.14% | `breadcrumb` |
| `Swimwear` | 12 | 0.12% | `breadcrumb` |
| `Scarves` | 9 | 0.09% | `breadcrumb` |
| `Waistcoat` | 7 | 0.07% | `breadcrumb` |
| `Gloves` | 6 | 0.06% | `breadcrumb` |
| `Ring` | 6 | 0.06% | `title` |
| `Trunk` | 3 | 0.03% | `breadcrumb` |
| `Bra` | 2 | 0.02% | `breadcrumb` |
| `Track Pants` | 2 | 0.02% | `breadcrumb` |
| `Hat` | 2 | 0.02% | `title` |
| `Booties` | 1 | 0.01% | `breadcrumb` |
| `Waist Pouch` | 1 | 0.01% | `breadcrumb` |

## 4. Phân bố theo Màu sắc (`baseColour` — Top 20)

| Màu sắc (`baseColour`) | Số lượng ảnh | Tỷ lệ (%) |
|---|---:|---:|
| `(Trống / Không ghi trên nguồn)` | 2,967 | 28.45% |
| `Black` | 1,867 | 17.90% |
| `Blue` | 1,178 | 11.30% |
| `White` | 728 | 6.98% |
| `Grey` | 676 | 6.48% |
| `Pink` | 522 | 5.01% |
| `Brown` | 415 | 3.98% |
| `Beige` | 269 | 2.58% |
| `Bronze` | 259 | 2.48% |
| `Cream` | 199 | 1.91% |
| `Red` | 190 | 1.82% |
| `Yellow` | 164 | 1.57% |
| `Green` | 160 | 1.53% |
| `Purple` | 156 | 1.50% |
| `Silver` | 152 | 1.46% |
| `Navy Blue` | 91 | 0.87% |
| `Orange` | 89 | 0.85% |
| `Off White` | 81 | 0.78% |
| `Skin` | 74 | 0.71% |
| `Olive` | 60 | 0.58% |

## 5. Giới hạn đã biết (Ghi nhận trung thực theo Luật L5)

1. **Thiên lệch danh mục theo thương hiệu Việt Nam:** Bộ dữ liệu phản ánh cơ cấu mặt hàng thực tế của 11 thương hiệu thời trang Việt Nam (áo thun, áo sơ mi, đầm, quần tây, giày thể thao, kính mát, đồng hồ, balo/túi xách chiếm tỷ trọng lớn). Các trang phục truyền thống Ấn Độ trong catalog gốc Myntra (`Sarees`, `Kurtas`, `Dupatta`, `Lehenga Choli`, `Salwar`) không tồn tại ở thị trường Việt Nam.
2. **Nhãn suy diễn tự động từ metadata trang nguồn (`breadcrumb` / `title`):** Hoàn toàn không dùng mô hình SigLIP hay VLM để tránh rò rỉ vòng tròn. Các sản phẩm mà trang nguồn không ghi rõ giới tính hoặc loại sản phẩm cụ thể được giữ nguyên trạng thái `label_status = needs_review` (liệt kê tại `reports/crawl_v2/needs_review_list.csv`), không tự ý đoán hay sửa hàng loạt.
3. **Đặc điểm hình ảnh ngoài miền (Out-of-Domain Visual Shift):** Ảnh sản phẩm thương mại điện tử Việt Nam bao gồm cả ảnh chụp phẳng (flat-lay), ảnh chụp ma-nơ-canh và ảnh người mẫu mặc thực tế (lookbook studio hoặc ngoại cảnh), khác biệt rõ với phong cách studio nền trắng đồng nhất của Myntra/Kaggle.
4. **Thuộc tính màu sắc (`baseColour`):** Một số trang sản phẩm không công bố tên màu trong tiêu đề hoặc JSON biến thể; các trường hợp này được để trống `""` thay vì dùng mô hình thị giác tự gán màu.
