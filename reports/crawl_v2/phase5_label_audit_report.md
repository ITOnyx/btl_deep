# BÁO CÁO PHA 5 — KIỂM TRA NHÃN PHÂN TẦNG (SEED 42) & DANH SÁCH `needs_review`

- **Thời điểm hoàn tất (UTC):** `2026-11-20T07:48:39+00:00`
- **Phương pháp lấy mẫu:** Lấy mẫu ngẫu nhiên có phân tầng theo `(brand, articleType)` với `seed = 42`.
- **Cỡ mẫu kiểm tra:** **`n = 300`** ảnh trên tổng số **`10,429`** ảnh sạch.
- **Tệp chi tiết 300 mẫu:** `reports/crawl_v2/phase5_sample_300_audit.csv`
- **Tệp danh sách toàn bộ `needs_review` (372 bản ghi):** `reports/crawl_v2/needs_review_list.csv` (giữ nguyên, KHÔNG tự sửa hàng loạt).

## 1. Ước lượng tỷ lệ nhãn sai / mơ hồ trên mẫu `n = 300` (Kèm khoảng tin cậy 95% Wilson)

> **Lưu ý học thuật (Luật L5 & Quy tắc Pha 5):** Các con số dưới đây là **ước lượng thống kê từ mẫu phân tầng `n = 300`**, đi kèm khoảng tin cậy 95% (Wilson score interval); **không khẳng định tuyệt đối cho toàn bộ tập dữ liệu**.

| Tập mẫu đánh giá | Cỡ mẫu (`n`) | Số ca mâu thuẫn / `needs_review` (`k`) | Tỷ lệ ước lượng (`k/n`) | Khoảng tin cậy 95% (Wilson CI) |
|---|---:|---:|---:|---|
| Toàn bộ mẫu phân tầng (gồm cả `ok` và `needs_review`) | **300** | **54** | **18.00%** | **`[14.07%, 22.74%]`** |
| Chỉ tính các bản ghi có `label_status == ok` trong mẫu | **273** | **27** | **9.89%** | **`[6.89%, 14.01%]`** |

## 2. Danh sách 6 tờ Contact Sheet (300 ảnh kèm nhãn để giảng viên duyệt nhanh)

- `reports/crawl_v2/contact_sheets_phase5/phase5_sample300_sheet_01.png`
- `reports/crawl_v2/contact_sheets_phase5/phase5_sample300_sheet_02.png`
- `reports/crawl_v2/contact_sheets_phase5/phase5_sample300_sheet_03.png`
- `reports/crawl_v2/contact_sheets_phase5/phase5_sample300_sheet_04.png`
- `reports/crawl_v2/contact_sheets_phase5/phase5_sample300_sheet_05.png`
- `reports/crawl_v2/contact_sheets_phase5/phase5_sample300_sheet_06.png`

## 3. Thống kê bản ghi `needs_review` trên toàn bộ tập sạch (Không tự sửa hàng loạt)

- Tổng số bản ghi `label_status == 'ok'`: **10,057** (`96.43%`)
- Tổng số bản ghi `label_status == 'needs_review'`: **372** (`3.57%`)

| Thương hiệu (`brand`) | Tổng ảnh sạch | Số ca `ok` | Số ca `needs_review` | Tỷ lệ `needs_review` (%) |
|---|---:|---:|---:|---:|
| Aristino | 1,611 | 1,501 | 110 | 6.8% |
| Biti's | 1,420 | 1,337 | 83 | 5.8% |
| Coolmate | 43 | 37 | 6 | 14.0% |
| IVY moda | 2,473 | 2,473 | 0 | 0.0% |
| Juno | 672 | 659 | 13 | 1.9% |
| Leonardo | 145 | 87 | 58 | 40.0% |
| MIA.vn | 220 | 220 | 0 | 0.0% |
| Mắt Kính Shady | 1,336 | 1,287 | 49 | 3.7% |
| Rabity | 844 | 817 | 27 | 3.2% |
| WatchStore | 976 | 976 | 0 | 0.0% |
| YaMe | 689 | 663 | 26 | 3.8% |
