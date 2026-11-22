# BÁO CÁO PHA 4 — LÀM SẠCH, LỌC CHẤT LƯỢNG & KHỬ TRÙNG TOÀN CỤC

- **Thời điểm hoàn tất (UTC):** `2026-11-20T07:47:24+00:00`
- **Tiêu chí chất lượng áp dụng:** `short_side >= 224 px`, `blur_score (Laplacian) >= 12.0`, `detect_banner_or_placeholder`, kiểm tra định dạng/byte gốc.
- **Tiêu chí khử trùng nội bộ (toàn cục, không chia theo nhãn):** `SHA-1` chính xác + `pHash 64-bit Hamming <= 1`.
- **Tiêu chí khử trùng chéo với Catalog (`data/fashion-dataset` train/val/test):** `SHA-1` chính xác + `pHash 64-bit Hamming <= 1` trên `140,077` ảnh catalog.

## 1. Bảng phễu số lượng ảnh qua từng bước (Đếm thật trên đĩa)

| Bước xử lý | Số lượng bản ghi / ảnh | Ghi chú / Thư mục đích |
|---|---:|---|
| 0. Tổng ứng viên sản phẩm đi qua crawler | **11,691** | `data/crawl_v2/raw_crawled.jsonl` |
| — Loại ngoài phạm vi thời trang (`out_of_scope`) | 792 | Mỹ phẩm, quà tặng, lens, bình nước (không tải ảnh) |
| — Lỗi tải HTTP / URL từ khóa banner | 24 | Không lưu tệp |
| **1. Tổng ảnh tải thô xuống đĩa** | **10,875** | Đã kiểm tra byte gốc |
| — Loại do độ phân giải thấp (`low_res < 224px`) | 1 | `data/crawl_v2/_quarantine/low_res/` |
| — Loại do ảnh mờ (`blurry Laplacian < 12.0`) | 215 | `data/crawl_v2/_quarantine/blurry/` |
| — Loại do ảnh banner / chữ khuyến mãi / khung cực đoan (`banner`) | 50 | `data/crawl_v2/_quarantine/banner/` |
| — Loại do tệp hỏng / định dạng (`corrupt` / `missing`) | 0 | `data/crawl_v2/_quarantine/corrupt/` |
| **2. Còn lại sau lọc chất lượng** | **10,437** | 100% ảnh sắc nét, nguyên byte gốc |
| — Loại do trùng `SHA-1` nội bộ toàn cục (`dup_sha1`) | 0 | `data/crawl_v2/_quarantine/dup_sha1/` |
| — Loại do trùng `pHash Hamming <= 1` nội bộ toàn cục (`dup_phash`) | 0 | `data/crawl_v2/_quarantine/dup_phash/` |
| **3. Còn lại sau khử trùng nội bộ** | **10,437** | 1 bản đại diện duy nhất / cụm |
| — Loại do trùng chéo với Catalog hiện có (`overlap_with_catalog`) | 8 | `data/crawl_v2/_quarantine/overlap_catalog/` |
| **4. TỔNG SỐ ẢNH SẠCH CUỐI CÙNG (`filter_flag == pass`)** | **10,429** | Lưu tại `reports/crawl_v2/metadata.csv` & `.jsonl` |
| — Đối chiếu số tệp ảnh thật trong `data/crawl_v2/images/` | **10,429** | Lệch = `0` |
| — Tổng số tệp đang lưu trong `data/crawl_v2/_quarantine/` | **446** | Kèm log `reports/crawl_v2/quarantine_log.csv` |

