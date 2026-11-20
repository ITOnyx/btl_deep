# BÁO CÁO PHA 2 — PILOT 200 ẢNH TRẢI ĐỀU 11 NGUỒN

- **Thời điểm hoàn tất (UTC):** `2026-11-20T05:57:16+00:00`
- **Seed cố định:** `42`
- **Contact sheet 50 ảnh:** `reports/crawl_v2/phase2_pilot_contact_sheet_50.png`

## 1. Số liệu thực tế đo trên đĩa (Pilot)

| Chỉ số | Giá trị đo thật | Ghi chú |
|---|---:|---|
| Tổng ứng viên đi qua pipeline | **224** | 11 nguồn thương hiệu VN |
| Số ảnh hợp lệ (`filter_flag == pass`) trên đĩa | **200** | Lưu nguyên byte gốc (Luật L4) |
| Tốc độ thu thập thực tế (song song 11 domain) | **1.11 ảnh/giây** | Tổng thời gian tường: `180.0s` |
| Tỷ lệ lỗi tải HTTP (`download_failed`) | **2.68%** (6/224) | Ngưỡng dừng an toàn: < 10% |
| Tỷ lệ loại ngoài phạm vi (`out_of_scope`) | **4.91%** (11/224) | Mỹ phẩm/quà tặng/lens |
| Tỷ lệ loại chất lượng ảnh (`low_res`/`blurry`/`banner`) | **3.12%** (7/224) | Đã chuyển vào `_quarantine/` |
| Tỷ lệ nhãn `ok` / `needs_review` | **95.5% `ok`** / **4.5% `needs_review`** | `191` ok, `9` needs_review |
| Dung lượng trung bình 1 ảnh gốc | **341.9 KB** | Tổng `66.77 MB` cho `200` ảnh |
| Ước tính thời gian cho 13,000 ảnh (Pha 3) | **~195.0 phút** | Chạy song song 11 domain, <=1 req/s/domain |
| Ước tính dung lượng đĩa cho 13,000 ảnh gốc | **~4.24 GB** | Ổ `D:` đang trống `72.66 GB` |

## 2. Thống kê theo từng nguồn trong Pilot

| Nguồn (`source`) | Mục tiêu Pilot | Đạt `pass` | Đếm file trên đĩa | Lệch đĩa | Ngoài phạm vi | Loại chất lượng | Lỗi tải | Thời gian (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `aristino` | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0.0 |
| `bitis` | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 0.0 |
| `coolmate` | 18 | 18 | 18 | 0 | 0 | 2 | 6 | 0.0 |
| `ivymoda` | 22 | 22 | 22 | 0 | 0 | 0 | 0 | 0.0 |
| `juno` | 18 | 18 | 18 | 0 | 3 | 1 | 0 | 0.0 |
| `leonardo` | 15 | 15 | 15 | 0 | 7 | 3 | 0 | 0.0 |
| `matkinhshady` | 18 | 18 | 18 | 0 | 0 | 0 | 0 | 0.0 |
| `mia` | 15 | 15 | 15 | 0 | 0 | 0 | 0 | 0.0 |
| `rabity` | 18 | 18 | 18 | 0 | 0 | 1 | 0 | 0.0 |
| `watchstore` | 18 | 18 | 18 | 0 | 0 | 0 | 0 | 0.0 |
| `yame` | 18 | 18 | 18 | 0 | 1 | 0 | 0 | 0.0 |
