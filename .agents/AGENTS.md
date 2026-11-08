# Dự án: SigLIP E-commerce Visual Search (Hoc_sau_ver4)

Đọc file này và `config.yaml` ở đầu MỖI phiên làm việc. Mọi đường dẫn và tham số lấy từ `config.yaml`, không tự gõ cứng.

## Mục tiêu
Tra cứu sản phẩm thời trang: text tiếng Việt → ảnh, ảnh → ảnh, ảnh + text (CIR).
Chỉ tiêu: Recall@10 ≥ 0.85, mAP ≥ 0.75 (text→ảnh, KHÔNG lọc thuộc tính), latency < 200 ms/query.

## Luật bất biến
1. Không train lại, không ghi đè checkpoint SigLIP (chỉ đọc).
2. Không xóa file vĩnh viễn. Ảnh bị loại chuyển vào `data/_quarantine/<lý_do>/` và ghi `reports/quarantine_log.csv`.
3. Không sửa `notebooks/` và `docs/`. Chỉ ghi vào: `data/catalog/`, `data/_quarantine/`, `index/`, `reports/`, `eval/`, `src/`, `app/`, `config.yaml`, `README_run.md`. Ngoài dự án chỉ đọc.
4. Tiền xử lý ảnh/text khi tạo embedding phải giống hệt lúc train. Không xác định được thì dừng và hỏi.
5. Mọi số liệu (Recall, mAP, latency) phải do script chạy thật. Không bịa, không làm tròn có lợi. Không đạt thì nói thẳng.
6. Không crop hay sửa ảnh gốc. Ảnh chỉ bị loại vì chất lượng (mờ, nhỏ, trùng, banner), không vì "khó truy vấn". Không loại ảnh khỏi bộ test để tăng điểm.
7. Index phải khớp checkpoint: manifest lưu hash checkpoint; app từ chối nạp nếu lệch.
8. Seed cố định; ghi phiên bản thư viện và tham số vào `reports/run_manifest.json`.
9. Windows: dùng `pathlib`, `encoding="utf-8"`, đường dẫn có dấu cách/ký tự có dấu phải chạy được. Tác vụ dài phải resume được.

## Quy tắc đánh giá (tránh điểm cao giả)
- Chỉ tiêu chính chấm trên kết quả KHÔNG lọc thuộc tính (model thuần).
- Kết quả có lọc thuộc tính báo cáo riêng, coi là chức năng sản phẩm.
- Có tập truy vấn viết tay (`eval/handwritten_queries.json`) độc lập với metadata; báo cáo riêng.
- Nếu checkpoint từng train trên dataset này, nêu rõ khả năng số liệu lạc quan; tách tập test khỏi tập train nếu xác định được.
- Tập test cố định (seed), không đổi giữa các lần chạy.

## Quy trình
- Làm từng pha, hết pha dừng, tóm tắt tối đa ~15 dòng, chờ tôi xác nhận.
- Ngưỡng lọc chất lượng ảnh KHÔNG tự chốt: xuất contact sheet cho tôi duyệt rồi mới áp dụng.
- Điểm mơ hồ có thể ảnh hưởng kết quả: dừng và hỏi.
- Commit git sau mỗi pha (`git add -A && git commit -m "phase N: ..."`); không commit ảnh, embedding, index, checkpoint (dùng `.gitignore`).