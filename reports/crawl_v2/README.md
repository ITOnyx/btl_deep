# README — Bộ Dữ Liệu Ảnh Sản Phẩm Thời Trang Việt Nam (`crawl_v2`)

## 1. Mục đích & Phạm vi sử dụng
- Bộ dữ liệu này được thu thập nhằm **(a) trình bày trực quan cho giảng viên kiểm tra** và **(b) làm tập kiểm thử ngoài miền (out-of-domain evaluation)** trong tương lai cho dự án `Hoc_sau_ver4` (Hệ thống tra cứu sản phẩm thời trang tiếng Việt dùng SigLIP + FAISS HNSW).
- **Cam kết tách biệt (Luật bất biến L1 & L3):** Bộ dữ liệu này **KHÔNG** dùng để huấn luyện, **KHÔNG** trộn vào `train` / `val` / `test_gallery` / `reference_gallery`, và **KHÔNG** dùng để tinh chỉnh hay chạy lại các đánh giá đã đóng băng.

## 2. Bản quyền & Hạn chế sử dụng
- Bản quyền hình ảnh và tên thương hiệu thuộc về các chủ sở hữu thương hiệu tương ứng (`IVY moda`, `Aristino`, `Biti's`, `YaMe`, `Mắt Kính Shady`, `WatchStore`, `Juno`, `Coolmate`, `Rabity`, `MIA.vn`, `Leonardo`).
- Bộ dữ liệu **chỉ sử dụng nội bộ cho mục đích học tập, giảng dạy và nghiên cứu học thuật phi thương mại**.
- **KHÔNG** đăng tải lên các kho dữ liệu công khai (Kaggle, HuggingFace, GitHub public...) và **KHÔNG** phát tán lại hoặc sử dụng cho mục đích thương mại.

## 3. Cấu trúc thư mục bàn giao
- `reports/crawl_v2/gallery/index.html`: Trang web tĩnh xem gallery **OFFLINE 100%** (không cần Internet, không dùng CDN ngoài, không cần chạy web server).
- `reports/crawl_v2/gallery/thumbs/`: Thư mục chứa ảnh thumbnail (`<= 320 px`) dùng cho gallery.
- `reports/crawl_v2/metadata.csv` & `reports/crawl_v2/metadata.jsonl`: Dữ liệu metadata đầy đủ 20 trường chuẩn theo thiết kế Pha 1.
- `reports/crawl_v2/dataset_summary.md`: Báo cáo tổng hợp số liệu thật trên đĩa theo thương hiệu, loại sản phẩm, giới tính, màu sắc.
- `reports/crawl_v2/robots/`: Bản chụp `robots.txt` kèm ngày giờ UTC của toàn bộ 11 domain.
- `reports/crawl_v2/run_manifest.json`: Thông tin cấu hình, seed `42`, phiên bản thư viện và `git HEAD`.
- `reports/crawl_v2/vietnamese_fashion_crawl_v2_handover.zip`: Gói nén bàn giao gọn nhẹ gửi giảng viên (gồm gallery + thumbnails + metadata + báo cáo; không gồm ảnh gốc).
- `data/crawl_v2/images/`: Thư mục chứa ảnh gốc lưu nguyên byte tải về.
- `data/crawl_v2/_quarantine/`: Thư mục lưu các tệp ảnh bị loại (`low_res`, `blurry`, `banner`, `corrupt`, `dup_sha1`, `dup_phash`, `overlap_catalog`) kèm log `reports/crawl_v2/quarantine_log.csv`.

## 4. Hướng dẫn mở Gallery xem Offline trên Windows
1. Mở File Explorer tới thư mục `D:\Kì 7\Hoc_sau_ver4\reports\crawl_v2\gallery\` (hoặc giải nén tệp `vietnamese_fashion_crawl_v2_handover.zip` ra bất kỳ thư mục nào trên máy tính).
2. Nhấp đúp chuột trực tiếp vào tệp **`index.html`** để mở bằng trình duyệt bất kỳ (Edge, Chrome, Firefox).
3. Sử dụng thanh tìm kiếm tiếng Việt và các hộp chọn **Thương hiệu**, **Loại sản phẩm (`articleType`)**, **Giới tính (`gender`)**, **Màu sắc (`baseColour`)**, **Trạng thái nhãn (`ok` / `needs_review`)** để lọc và duyệt qua các trang.
