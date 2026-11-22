# BÁO CÁO PHA 0 — KHẢO SÁT HIỆN TRẠNG & PHÁP LÝ (CHỈ ĐỌC)

- **Ngày thực hiện (UTC):** `2026-11-20T00:08:02Z`
- **Nhánh Git:** `add/test-crawl-img`
- **Seed cố định:** `42`
- **User-Agent học thuật đã chốt:** `AcademicResearchBot/2.0 (Hoc_sau_ver4 Vietnamese Fashion Visual Search; non-commercial academic evaluation; contact: nguyen-student@university.edu.vn)`

## 1. Tóm tắt luật liên quan (`.agents/AGENTS.md` & `config.yaml`)
- **L1 (Chỉ đọc lõi):** Không train, không sửa checkpoint `best_model.pt` (`e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`), không sửa `index/`, `eval/`, `notebooks/`, `docs/`, `data/catalog/`, `data/fashion-dataset/`.
- **L2 (Quarantine):** Không xóa vĩnh viễn; mọi ảnh bị loại chuyển vào `data/crawl_v2/_quarantine/<lý_do>/` và ghi `reports/crawl_v2/quarantine_log.csv`.
- **L3 (Phạm vi ghi):** Chỉ ghi vào `data/crawl_v2/`, `reports/crawl_v2/`, `src/crawl/`, `config.yaml` (chỉ thêm mục `crawl_v2`), `.gitignore`.
- **L4 (Nguyên vẹn byte):** Không crop, không sửa, không nén lại ảnh gốc. Ảnh gốc lưu nguyên byte tải về; thumbnail (`<= 320px`) cho gallery lưu riêng.
- **L5–L7:** Số liệu đếm thật trên đĩa; seed `42`; hỗ trợ đường dẫn tiếng Việt có dấu cách (`D:\Kì 7\Hoc_sau_ver4`); có checkpoint resume sau mỗi sản phẩm/trang.

## 2. Phân tích crawler cũ trên nhánh `add/test-crawl-img`
- `data/images_crawled`: Không tồn tại (0 ảnh).
- `data/crawl` (lần chạy thử đầu): 821 bản ghi trong `raw_crawled.jsonl` (5 thương hiệu đầu: Coolmate, Juno, Biti's, IVY moda, YaMe đạt 94–574 bản ghi; tổng 1,978 tệp ảnh gồm cả `_alt.jpg`).
- `data/external_test` (lần chạy mục tiêu 1,500 sp): thu 1,373 bản ghi → 1,149 ảnh qua lọc → chỉ còn **416** sản phẩm giữ lại (86 `clean`, 330 `suspect`).
- **Nguyên nhân ít ảnh & lỗi cũ:**
  1. Đặt hạn ngạch thấp (`150` sp/nguồn).
  2. Bước khử trùng cũ (`dedup_and_clean.py`) dùng ngưỡng quá gắt (`pHash Hamming <= 8` và `SigLIP cosine >= 0.95`) làm loại nhầm 733/1,149 ảnh (63.8%) có nền trắng studio tương tự ảnh catalog.
  3. Hàm `download_and_save_image` trong `common.py` cũ decode PIL và nén lại `JPEG quality=92`, vi phạm L4 (lưu nguyên byte).
- **Thành phần tái sử dụng được:** `PoliteHttpClient` (rate-limit theo domain, kiểm tra `robots.txt`, backoff 429/5xx), `TaxonomyMapper` (ánh xạ xác định từ breadcrumb/tiêu đề sang `articleType`, `gender`, `baseColour` qua `vi_map.json`), 11 scraper Sitemap & Haravan/Shopify JSON, hàm tính `min_hamming_batch` vector hóa 64-bit.

## 3. Tài nguyên hệ thống & Ước tính dung lượng (Đo thật)
- **Ổ đĩa `D:`:** Trống **62.50 GB** / 209.71 GB.
- **RAM:** Trống **7.26 GB** / 16.87 GB (57.0% đang dùng).
- **Mạng & kích thước ảnh thực tế:** Trung bình **186.2 KB/ảnh** (đo trên 2,394 ảnh đã tải); thời gian phản hồi `robots.txt` 577–837 ms/domain.
- **Ước tính cho 10,000 ảnh sạch (~13,500 ảnh tải thô + 10,000 thumbnail <= 320px):** ~**2.8 – 3.2 GB** (chỉ chiếm ~5% dung lượng trống ổ `D:`).

## 4. Kiểm kê Sitemap & `robots.txt` của 11 nguồn đã chọn
Bản chụp `robots.txt` kèm timestamp UTC đã lưu tại `reports/crawl_v2/robots/` và `reports/crawl_v2/robots/robots_manifest.json`.

| Nguồn (`brand`) | Domain | `robots.txt` | Số URL sản phẩm trong Sitemap (đếm thật) | Phương thức thu thập |
|---|---|---|---:|---|
| **Coolmate** | `https://www.coolmate.me` | HTTP 200 (Allow `/product/`) | **1,272** | Sitemap `product0/1.xml` + HTML/JSON |
| **Juno** | `https://juno.vn` | HTTP 200 (Allow `/products/`) | **1,387** | Sitemap 15 trang + JSON-LD/Haravan JSON |
| **Biti's** | `https://bitis.com.vn` | HTTP 200 (Allow `/collections/`, `/products/`) | **1,627** | Haravan JSON `/collections/*/products.json` + Sitemap |
| **IVY moda** | `https://ivymoda.com` | HTTP 200 (Allow `/sanpham/`) | **3,839** | Sitemap 8 trang `/sitemap/product/*` + HTML |
| **YaMe** | `https://yame.vn` | HTTP 200 (Allow `/collections/`, `/products/`) | **1,552** | Shopify JSON `/collections/*/products.json` + Sitemap |
| **Aristino** | `https://aristino.com` | HTTP 200 (Allow `/collections/`, `/products/`) | **1,969** | Haravan JSON `/collections/*/products.json` + Sitemap |
| **Rabity** | `https://rabity.vn` | HTTP 200 (Allow `/collections/`, `/products/`) | **1,024** | Haravan JSON `/collections/*/products.json` + Sitemap |
| **MIA.vn** | `https://mia.vn` | HTTP 200 (`Allow: /`, Disallow `/tim-kiem`) | **897** (~250 mẫu gốc) | Sitemap `products.xml` + HTML |
| **Leonardo** | `https://www.leonardo.vn` | HTTP 200 (`Allow: /`) | **195** | Shopify JSON `/collections/*/products.json` + Sitemap |
| **Mắt Kính Shady** | `https://matkinhshady.com` | HTTP 200 (Allow `/collections/`, `/products/`) | **2,224** | Haravan JSON `/collections/*/products.json` + Sitemap |
| **WatchStore** | `https://www.watchstore.vn` | HTTP 200 (`Allow: /`) | **14,001** | Sitemap / Collections + HTML/JSON |
| **TỔNG CỘNG** | **11 domain** | **100% hợp lệ `robots.txt`** | **29,987 URL sản phẩm** | **1 ảnh đại diện / sản phẩm** |
