# Báo cáo Thiết kế Pha 1 — Mục tiêu, Phân bổ Nguồn & Schema (`crawl_v2`)

## 1. Mục tiêu quy mô & Ước tính phễu tải thô (1 ảnh đại diện / sản phẩm)
- **Mục tiêu ảnh sạch bàn giao:** `>= 10,000` ảnh sản phẩm độc lập (1 ảnh đại diện chính thức / trang sản phẩm) **SAU KHI** qua toàn bộ các bước:
  1. Lọc phạm vi thời trang (`Apparel`, `Footwear`, `Accessories` — loại mỹ phẩm, quà tặng, hộp/túi giấy, gia dụng, dụng cụ thể thao).
  2. Lọc chất lượng ảnh dự án: `short_side >= 224 px`, `blur_score (Laplacian variance) >= 12.0`, không phải banner/bảng size/placeholder, định dạng hợp lệ.
  3. Khử trùng nội bộ trên toàn bộ tập crawl: `SHA-1` chính xác + `pHash 64-bit Hamming <= 1` (không chia theo nhóm nhãn).
  4. Khử trùng chéo (anti-leakage) với catalog hiện có (`data/fashion-dataset`: train, val, test): loại mọi ảnh trùng `SHA-1` hoặc `pHash Hamming <= 1` vào `_quarantine/overlap_catalog/`.
- **Ước tính tỷ lệ hao hụt:**
  - Tỷ lệ lọc chất lượng + ngoài phạm vi dự kiến: ~8–12%.
  - Tỷ lệ trùng lặp nội bộ (`SHA-1` / `pHash <= 1` giữa các mã sản phẩm cùng ảnh đại diện): ~10–14%.
  - Tỷ lệ trùng chéo với `data/fashion-dataset` (Myntra): < 0.5%.
  - **Tổng hạn ngạch cần tải thô (raw target):** **13,000 sản phẩm** (dự kiến thu được **~10,300 – 11,000 ảnh sạch** sau Pha 4).

## 2. Phân bổ hạn ngạch đề xuất theo 11 thương hiệu (Ràng buộc: không brand nào > 30%)

Tổng số URL sản phẩm có sẵn trong sitemap của 11 nguồn là **29,987 URL**. Phân bổ hạn ngạch tải thô (`raw_quota`) và dự kiến ảnh sạch (`expected_clean`) như sau:

| STT | Brand (`brand`) | Mã nguồn (`source`) | Tổng URL Sitemap | Hạn ngạch tải thô (`raw_quota`) | Dự kiến ảnh sạch (`~82%`) | Tỷ lệ dự kiến (%) | Nhóm giới tính chính | Nhóm sản phẩm chính (`articleType`) |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | **IVY moda** | `ivymoda` | 3,839 | **2,500** | ~2,050 | **19.2%** | Nữ, Nam, Trẻ em | Đầm (`Dresses`), Áo sơ mi (`Shirts`), Áo kiểu (`Tops`), Chân váy (`Skirts`), Quần tây (`Trousers`), Blazer (`Blazers`), Áo khoác (`Jackets`), Jumpsuit |
| 2 | **Aristino** | `aristino` | 1,969 | **1,650** | ~1,350 | **12.6%** | Nam | Áo sơ mi (`Shirts`), Áo polo/thun (`Tshirts`), Quần âu (`Trousers`), Quần kaki/jeans, Suit/Blazer, Áo khoác, Ví da, Thắt lưng, Giày da (`Formal Shoes`) |
| 3 | **Biti's** | `bitis` | 1,627 | **1,450** | ~1,200 | **11.2%** | Nam, Nữ, Bé trai, Bé gái | Giày thể thao (`Sports Shoes`), Giày thường (`Casual Shoes`), Sandal (`Sandals`), Dép (`Flip Flops`), Giày tây, Giày búp bê (`Flats`), Balo |
| 4 | **YaMe** | `yame` | 1,552 | **1,350** | ~1,100 | **10.3%** | Nam, Unisex | Áo thun/polo (`Tshirts`), Áo khoác (`Jackets`), Hoodie (`Sweatshirts`), Quần jeans (`Jeans`), Quần short (`Shorts`), Quần dài (`Trousers`), Balo, Nón, Ví, Dép |
| 5 | **Mắt Kính Shady** | `matkinhshady` | 2,224 | **1,350** | ~1,100 | **10.3%** | Nam, Nữ, Unisex | Kính mát / Mắt kính thời trang (`Sunglasses`) |
| 6 | **WatchStore** | `watchstore` | 14,001 | **1,350** | ~1,100 | **10.3%** | Nam, Nữ, Unisex | Đồng hồ đeo tay (`Watches`), Dây chuyền (`Necklace and Chains`) |
| 7 | **Juno** | `juno` | 1,387 | **1,100** | ~900 | **8.4%** | Nữ | Giày cao gót (`Heels`), Giày búp bê (`Flats`), Sandal nữ (`Sandals`), Túi xách (`Handbags`), Ví cầm tay (`Clutches`), Balo, Kính mát |
| 8 | **Coolmate** | `coolmate` | 1,272 | **1,000** | ~820 | **7.7%** | Nam, Nữ, Unisex | Áo thun/polo (`Tshirts`), Quần short (`Shorts`), Quần lót (`Boxers`, `Briefs`), Áo khoác, Quần dài/jogger (`Track Pants`), Tất (`Socks`), Mũ (`Caps`) |
| 9 | **Rabity** | `rabity` | 1,024 | **850** | ~700 | **6.5%** | Bé trai (`Boys`), Bé gái (`Girls`) | Áo thun, Đầm bé gái, Đồ bộ (`Clothing Set`), Áo khoác, Quần short/dài trẻ em, Giày dép trẻ em |
| 10 | **MIA.vn** | `mia` | 897 | **220** | ~190 | **1.8%** | Unisex | Vali (`Trolley Bag`), Balo (`Backpacks`), Túi du lịch (`Duffel Bag`), Túi laptop (`Laptop Bag`), Phụ kiện du lịch |
| 11 | **Leonardo** | `leonardo` | 195 | **180** | ~160 | **1.5%** | Nam, Nữ | Ví da (`Wallets`), Thắt lưng (`Belts`), Túi đeo chéo (`Messenger Bag`), Túi xách, Balo, Giày/Dép da |
| — | **TỔNG CỘNG** | **11 nguồn** | **29,987** | **13,000** | **~10,670** | **100.0%** | **Max brand = 19.2% (< 30%)** | **Phủ >35 `articleType` thuộc cả 3 `masterCategory`** |

### Phân bố dự kiến theo Giới tính & Nhóm sản phẩm
- **Theo giới tính (`gender`):**
  - `Men` (Nam): ~42% (Aristino, Coolmate, YaMe, Biti's nam, WatchStore nam, Shady nam, Leonardo).
  - `Women` (Nữ): ~32% (IVY moda nữ, Juno, Biti's nữ, WatchStore nữ, Shady nữ).
  - `Unisex`: ~18% (YaMe unisex, MIA.vn, Shady unisex, Coolmate phụ kiện).
  - `Boys` & `Girls` (Trẻ em): ~8% (Rabity, Biti's bé trai/bé gái, IVY kids).
- **Theo nhóm danh mục lớn (`masterCategory`):**
  - `Apparel` (Áo thun, sơ mi, áo kiểu, áo khoác, blazer, hoodie, đầm, chân váy, quần tây, quần jeans, quần short, jogger, đồ bộ, đồ lót, đồ bơi): ~58%.
  - `Accessories` (Kính mát, đồng hồ, túi xách, balo, ví, vali, thắt lưng, tất/vớ, mũ/nón, cà vạt): ~27%.
  - `Footwear` (Giày thể thao, giày cao gót, giày búp bê, sandal, dép, giày tây): ~15%.

## 3. Schema bản ghi chuẩn (20 trường bắt buộc theo đề bài)

Mọi bản ghi trong `reports/crawl_v2/metadata.csv` và `reports/crawl_v2/metadata.jsonl` tuân thủ chính xác 20 cột:

| STT | Tên trường | Kiểu dữ liệu | Mô tả & Quy tắc gán giá trị |
|---:|---|---|---|
| 1 | `crawl_id` | `string` | Định danh duy nhất, định dạng `cr2_000001` .. `cr2_013000`. |
| 2 | `brand` | `string` | Tên thương hiệu chuẩn (`Coolmate`, `Juno`, `Biti's`, `IVY moda`, `YaMe`, `Aristino`, `Rabity`, `MIA.vn`, `Leonardo`, `Mắt Kính Shady`, `WatchStore`). |
| 3 | `product_url` | `string` | URL trang chi tiết sản phẩm chính thức. |
| 4 | `image_url` | `string` | URL ảnh sản phẩm chính thức gốc trên CDN của thương hiệu. |
| 5 | `local_path` | `string` | Đường dẫn tương đối tới tệp ảnh gốc lưu nguyên byte (`data/crawl_v2/images/<source>/<crawl_id>.<ext>`). |
| 6 | `product_name` | `string` | Tên sản phẩm tiếng Việt lấy từ `<h1>` / JSON-LD / API sản phẩm. |
| 7 | `category_raw` | `string` | Chuỗi danh mục gốc (breadcrumb / `product_type` / tên collection của trang). |
| 8 | `articleType` | `string` | Nhãn loại sản phẩm ánh xạ theo taxonomy của catalog hiện có (ví dụ `Tshirts`, `Shirts`, `Dresses`, `Sports Shoes`, `Watches`...). Nếu không chắc chắn: ghi `needs_review`. |
| 9 | `gender` | `string` | Giới tính ánh xạ (`Men`, `Women`, `Unisex`, `Boys`, `Girls`). Nếu không xác định được từ metadata: ghi `needs_review`. |
| 10 | `baseColour` | `string` | Màu cơ bản ánh xạ theo `vi_map.json` (`Black`, `White`, `Blue`, `Navy Blue`, `Beige`...). Để trống `""` nếu trang nguồn không ghi màu. |
| 11 | `price` | `string` | Giá bán sản phẩm (VND, chuỗi số nguyên hoặc định dạng từ JSON/HTML, để trống `""` nếu không có). |
| 12 | `label_source` | `string` | Nguồn suy nhãn: `breadcrumb` (ưu tiên 1: từ breadcrumb/collection/product_type), `title` (ưu tiên 2: từ tiêu đề sản phẩm), hoặc `manual` (nếu chỉnh tay). Tuyệt đối không dùng SigLIP/VLM. |
| 13 | `label_status` | `string` | `ok` (khi có đủ cả `articleType` và `gender` hợp lệ từ luật xác định) hoặc `needs_review` (khi thiếu/mơ hồ ít nhất 1 trong 2 trường). |
| 14 | `crawled_at` | `string` | Thời điểm tải ảnh theo chuẩn ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SS+00:00`). |
| 15 | `http_status` | `int` | Mã HTTP khi tải ảnh gốc (`200`). |
| 16 | `sha1` | `string` | Mã băm SHA-1 (40 ký tự hex) tính trực tiếp trên **byte gốc** của tệp ảnh tải về. |
| 17 | `phash` | `string` | Perceptual hash 64-bit (16 ký tự hex) tính bằng `imagehash.phash(img.convert("L"))`. |
| 18 | `width` | `int` | Chiều rộng ảnh gốc (pixel). |
| 19 | `height` | `int` | Chiều cao ảnh gốc (pixel). |
| 20 | `filter_flag` | `string` | `pass` (đạt chuẩn giữ lại) hoặc mã lý do loại (`low_res`, `blurry`, `banner`, `corrupt`, `out_of_scope`, `dup_sha1`, `dup_phash`, `overlap_with_catalog`). |
