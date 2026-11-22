# Đánh giá ngoài miền trên crawl_v2 (lần chạy đóng băng)

- Gallery: `index/crawl_v2_gallery` (10,057 ảnh, 11 thương hiệu Việt Nam, nhãn `ok`), top-10.
- Truy vấn sinh từ nhãn trước khi chạy cấu hình nào (seed 42), SHA-256:
```
07a42f34f6579dca6bcafadbb54e22f4143fda0d3ff5f5906c1f264a91551a2d  crawl_v2_t2i_queries.json
335038775b60c695be3ff6dcc60c70015498bdbfa8882f36c225c713b02555b4  crawl_v2_cir_pairs.json
```
- GT: `is_item_relevant`, giới tính `hybrid`. mAP@10 = AP chia cho min(số liên quan, 10). Bootstrap 1.000 lần, seed 42.

## Text → Ảnh (250 câu: 150 ghép nhãn, 100 tên sản phẩm thật; 50 câu không dấu)

| Cấu hình | R@1 | R@10 | mAP@10 | R@10 tên SP thật | R@10 không dấu | Đúng loại @10 | Đúng giới @10 | Đúng màu @10 | Sai lĩnh vực @10 | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `model_only` | 0.272 | 0.512 | 0.181 | 0.540 | 0.220 | 0.426 | 0.731 | 0.248 | 0.184 | 41.4 |
| `cfg_v5_01` | 0.568 | 0.676 | 0.409 | 0.580 | 0.500 | 0.584 | 0.935 | 0.599 | 0.170 | 37.7 |
| `cfg_v5_09` | 0.568 | 0.676 | 0.409 | 0.580 | 0.500 | 0.585 | 0.935 | 0.598 | 0.170 | 34.2 |
| `cfg_v5_10` | 0.640 | 0.720 | 0.499 | 0.600 | 0.660 | 0.679 | 0.859 | 0.647 | 0.060 | 36.8 |
| `cfg_v5_11` | 0.780 | 0.844 | 0.689 | 0.740 | 0.900 | 0.820 | 0.998 | 0.650 | 0.047 | 43.3 |

## CIR catalog (120 cặp: 80 đổi màu, 40 đổi giới tính; loại ảnh tham chiếu)

| Cấu hình | R@1 | R@10 | mAP@10 | R@10 đổi màu | R@10 đổi giới | Đúng loại @10 | Đúng giới @10 | Đúng màu @10 | Sai lĩnh vực @10 | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `cir_v1_01` | 0.008 | 0.158 | 0.017 | 0.138 | 0.200 | 0.324 | 0.667 | 0.143 | 0.420 | 36.9 |
| `cir_v1_05` | 0.158 | 0.358 | 0.087 | 0.400 | 0.275 | 0.342 | 0.790 | 0.368 | 0.438 | 44.4 |
| `cir_v1_06` | 0.325 | 0.425 | 0.173 | 0.400 | 0.475 | 0.626 | 0.765 | 0.254 | 0.125 | 147.1 |
| `cir_v1_07` | 0.717 | 0.717 | 0.717 | 0.575 | 1.000 | 0.767 | 0.873 | 0.670 | 0.125 | 51.1 |

## Chênh lệch so với cấu hình chính thức (95% CI)

| So sánh | ΔR@10 | ΔmAP@10 |
|---|---|---|
| `t2i:model_only - cfg_v5_09` | -0.164 [-0.216, -0.112] | -0.228 [-0.269, -0.189] |
| `t2i:cfg_v5_01 - cfg_v5_09` | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| `t2i:cfg_v5_10 - cfg_v5_09` | +0.044 [+0.000, +0.092] | +0.091 [+0.062, +0.123] |
| `t2i:cfg_v5_11 - cfg_v5_09` | +0.168 [+0.124, +0.220] | +0.280 [+0.235, +0.327] |
| `cir:cir_v1_01 - cir_v1_05` | -0.200 [-0.275, -0.133] | -0.070 [-0.102, -0.042] |
| `cir:cir_v1_06 - cir_v1_05` | +0.067 [-0.017, +0.150] | +0.086 [+0.044, +0.130] |
| `cir:cir_v1_07 - cir_v1_05` | +0.358 [+0.242, +0.475] | +0.630 [+0.537, +0.714] |

## Hạn chế

- GT dựa trên thuộc tính (loại/giới/màu) nên ưu ái cấu hình ràng buộc thuộc tính (vòng tròn GT như báo cáo chính thức); nhóm tên sản phẩm thật là kiểm tra độc lập hơn.
- Câu ghép dùng tên tiếng Việt từ `app/i18n_vi.py` (gần với từ điển parser) → lợi cho cấu hình có parser so với `model_only`.
- Nhãn crawl_v2 suy từ breadcrumb (audit báo 3.6% `needs_review`, đã loại khỏi gallery).
- Lỗi phát hiện nhờ bộ này: 'giống vậy' bị đọc thành 'váy' (Dresses) — 24/34 ca thất bại của `cir_v1_07` có parser ra Dresses; mọi số sau khi sửa lỗi này không còn là số 'chưa thấy'.
- Độ trễ đo khi app desktop đang chạy song song trên cùng GPU 4 GB.

## Sau khi sửa lỗi 'giống vậy' → 'váy' (KHÔNG còn là số chưa thấy)

Sửa: thêm 'vậy/vây/vẫy…' vào danh sách đồng dạng của 'váy' (kèm 'tụi'≠'túi', 'quân/quấn/quận'≠'quần'),
che cụm 'giống vậy'/'như vậy'/'vậy thì' (cả không dấu) trong parser_v3 và taxonomy, và không khôi phục 'giong vay' thành 'giống váy'.
Kiểm tra hồi quy: 0/715 truy vấn dev đổi kết quả phân tích; 194 test pass. Kết quả chạy lại: `after_fix_vay/`.

- Chỉ `cir_v1_07` thay đổi (các cấu hình khác dùng parser v1/v2 đóng băng): R@10 **0.950** (đổi màu 0.925, đổi giới 1.000),
  mAP@10 0.950, sai lĩnh vực @10 0.000; so với `cir_v1_05`: ΔR@10 = +0.592 [+0.500, +0.683].
- 6 ca còn trượt đều do lệch tên màu giữa từ điển parser và nhãn ('xanh rêu' → Green thay vì Olive, 'kem' → Off White thay vì Cream,
  chưa có 'oải hương', 'màu đào', 'hồng hoa hồng'). Không sửa để tránh tinh chỉnh theo tập test; cần một tập khác để làm việc này.
- Lỗi nhỏ của bộ sinh truy vấn: màu Peach sinh ra 'màu màu đào' (tên tiếng Việt đã có chữ 'màu'); giữ nguyên vì bộ đã đóng băng.
