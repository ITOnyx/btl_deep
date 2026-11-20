# BÁO CÁO PHA 3 — CRAWL ĐẦY ĐỦ & ĐỐI CHIẾU TỆP TRÊN ĐĨA

- **Thời điểm hoàn tất (UTC):** `2026-11-20T07:44:00+00:00`
- **Tổng thời gian chạy Pha 3:** `1074.5 giây` (`17.9 phút`)
- **Tổng bản ghi trong `raw_crawled.jsonl`:** **11,691**
- **Tổng ảnh hợp lệ (`filter_flag == pass`) tồn tại trên đĩa:** **10,609**
- **Tổng dung lượng ảnh gốc trên đĩa:** **2.794 GB** (`3,000,186,506 bytes`)
- **Dung lượng ổ `D:` còn trống:** **67.41 GB**
- **Phân bố `filter_flag`:** `{'pass': 10609, 'out_of_scope': 792, 'banner': 50, 'download_failed': 11, 'banner_url_keyword': 13, 'blurry': 215, 'low_res': 1}`

## Đối chiếu số ảnh thật trên đĩa với log theo từng nguồn

| Nguồn (`source`) | Hạn ngạch (`target`) | Số bản ghi `pass` | Số file thật trên đĩa | Lệch (Đĩa - Log) | Ngoài phạm vi | Loại chất lượng | Lỗi tải | Trạng thái dừng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `watchstore` | 1350 | 1012 | 1012 | 0 | 0 | 2 | 0 | `completed_ok` |
| `yame` | 1350 | 710 | 710 | 0 | 4 | 14 | 0 | `completed_ok` |
