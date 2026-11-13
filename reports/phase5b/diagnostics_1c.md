# Báo cáo Chẩn đoán Bước 1c — Chuẩn hóa Canonical & Phân tích Spec ẩn

- **Hàm đánh giá Canonical duy nhất:** [`src/eval/canonical_eval.py`](file:///d:/Kì%207/Hoc_sau_ver4/src/eval/canonical_eval.py)
- **Erratum Baseline Pha 5:** [`reports/phase5/baseline_frozen/ERRATUM.md`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5/baseline_frozen/ERRATUM.md)

## 1c.1. Xác nhận tái tạo Baseline Pha 5 bằng Hàm Canonical

- **Model thuần (`pool=100`):** R@1 = `0.2907`, R@10 = `0.7093`, mAP = `0.3279` (khớp 100% `baseline_frozen`).
- **Hệ thống (`pool=100, gamma=0.3`):** R@1 = `0.6047`, R@10 = `0.8488`, mAP = `0.5537` (khớp 100% `baseline_frozen`).

## 1c.2. Phân tích Spec ẩn trên DEV SET (`N=100`)

- **Tỉ lệ truy vấn có spec ẩn:** **20/100 (20%)**
  - Ẩn `gender` (spec có nhưng câu không nêu): **12/100**
  - Ẩn `baseColour` (spec có nhưng câu không nêu): **8/100** (ví dụ: câu 83 *dong ho nam day kim loai*, câu 88 *đồng hồ nam mặt vuông dây cao su*, câu 94 *day nit nam da bo*)

### Kết quả 'Oracle chỉ thuộc tính nêu rõ' (`pool=1290, cut@100`):

| Cấu hình | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|
| Oracle chỉ nêu rõ (`gamma=0.3`) | `0.6628` | `0.8953` | `0.9070` | `0.6388` |
| Oracle chỉ nêu rõ (`hard-tier`) | `0.9070` | `0.9884` | `0.9884` | `0.9398` |

> **Nhận xét:** Khi loại bỏ các thuộc tính ẩn khỏi Oracle, `mAP` ở `gamma=0.3` đạt `0.6388` (và `hard-tier` đạt `0.9398`), cho thấy ở Bước 5.1, quy tắc soạn tập held-out v2 **chỉ ghi thuộc tính nêu rõ trong câu, không suy diễn** là hoàn toàn đúng đắn.

## 1c.3. Độ chính xác Parser theo từng Slot (86 truy vấn có GT)

| Slot | Có trong spec | Đúng | Thiếu (ít hại) | Sai (hại nặng) | Đoán dư (spec trống) |
|---|---:|---:|---:|---:|---:|
| `articleType` | 86 | 77 (89.5%) | 8 | 1 | 0 |
| `gender` | 86 | 77 (89.5%) | 9 | 0 | 0 |
| `baseColour` | 64 | 57 (89.1%) | 7 | 0 | 6 |

- **Bổ sung phân loại lỗi (D) GT mơ hồ vào 13 truy vấn thất bại:**
  - Câu **41 (`quần lót boxer nam co giãn tốt`)**: GT quy định `['Trunk', 'Boxers']` nhưng catalog có nhiều ảnh `Briefs` giống hệt boxer (lỗi D kết hợp C).
  - Các câu **83, 88, 94**: có **spec ẩn màu sắc** (ví dụ câu 88 *mặt vuông dây cao su* không hề ghi màu đen nhưng spec yêu cầu `Black`, câu 83 *dây kim loại* không ghi màu nhưng spec yêu cầu `Steel`/`Silver`, câu 94 *da bò* yêu cầu `Brown`/`Black`).

## 1c.4. Chứng minh bằng số liệu Lỗi cấu trúc của Bonus (`gamma=0.3`)

Với cơ chế chia đều 3 slot của baseline, sai `articleType` nhưng khớp `gender + baseColour` vẫn nhận `2/3 * 0.3 = 0.200`, tức chỉ bị phạt **`0.100`** so với ảnh đúng toàn phần (`0.300`).

- Trên toàn bộ câu mà Model thuần xếp sai ở Top-1 (`n=60`): chênh lệch cosine `(cos_best_wrong - cos_best_rel)` có **trung bình = `0.1193`**, **trung vị = `0.0719`**.
- Riêng trên 13 câu thất bại ở Baseline (`n=13`): chênh lệch cosine có **trung bình = `0.2637`**, **trung vị = `0.2044`**.
- **Kết luận cấu trúc:** Mức phạt `0.100` cho sai `articleType` nhỏ hơn đáng kể so với độ chênh cosine trung bình (`> 0.14–0.19`) khi mô hình SigLIP gặp câu không dấu hoặc từ khóa hiếm. Do đó, cần thiết kế **Bonus có trọng số theo slot** (ưu tiên `articleType` cao hơn `gender`/`baseColour`) ở Bước 3.
