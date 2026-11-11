# AGENTS.md — Quy tắc cho agent (phiên bản xây lại, dùng lại checkpoint cũ)

Dự án: SigLIP + Dense Captioning cho hệ thống tra cứu sản phẩm thương mại điện tử.
Phạm vi: làm lại pipeline từ Pha 1 theo đúng `context.md` (đề cương giáo viên).
**Không huấn luyện lại.** Dùng nguyên checkpoint SigLIP đã có.

Checkpoint dùng lại (chỉ đọc):
- Tệp: `best_model.pt`
- SHA-256: `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`

Seed cố định: **42** cho mọi phân chia dữ liệu và mọi thao tác ngẫu nhiên.

---

## A. Chỉ tiêu và công nghệ bắt buộc (từ `context.md`)

- Recall@10 ≥ 0.85; Text→Image mAP ≥ 0.75; độ trễ < 200 ms/query (p95, đo riêng từng luồng).
- Backbone: SigLIP ViT-SO400M, vector 1152 chiều, L2-normalized.
- Dense captioning: **InternVL2-2B** (`OpenGVLab/InternVL2-2B`). Đây là model sinh caption duy nhất của dự án.
- Chỉ mục: FAISS HNSW (`IndexHNSWFlat`, M=32, efConstruction=200, efSearch=128).
- Giao diện: Streamlit. Hỗ trợ Text→Ảnh, Ảnh→Ảnh, CIR (ảnh + text chỉnh sửa).
- Truy vấn bằng tiếng Việt, có dấu và không dấu. Hiển thị Top-10 kèm độ tương đồng.
- Chỉ số báo cáo: Recall@1, Recall@5, Recall@10, mAP.

---

## B. Luật bất biến

**Luật 1 — Không huấn luyện, không sửa checkpoint.**
Cấm train, fine-tune, đổi trọng số, thêm adapter hay lưu đè `best_model.pt`.
Mọi cải tiến chỉ nằm ở tầng suy luận (parser, lexicon, hybrid ranking, index).
Nếu cần train mới để đạt chỉ tiêu, dừng lại và báo cáo, không tự làm.

**Luật 2 — Không xóa vĩnh viễn.**
Mọi ảnh/tệp bị loại phải chuyển vào `data/_quarantine/<lý_do>/` và ghi `reports/quarantine_log.csv`.

**Luật 3 — Chỉ ghi vào thư mục được phép.**
Được ghi: `src/`, `app/`, `reports/`, `eval/`, `index/` (thư mục index mới), `data/catalog/`, `config.yaml`, `README_run.md`.
Cấm sửa: `notebooks/`, `docs/`, checkpoint, và mọi `index_v*/`, `catalog.parquet` đã tồn tại từ trước.
Index mới dựng ở thư mục riêng; không ghi đè index cũ.

**Luật 4 — Tiền xử lý khớp lúc train.**
- Ảnh: `Resize((224,224), BILINEAR)` rồi `SiglipProcessor`.
- Text: `max_length=128`, `padding="max_length"`.
- Chỉ có **một** đường code tiền xử lý dùng chung cho embed index, truy vấn và đánh giá.
- Trước khi embed số lượng lớn, tái encode 200 ảnh mẫu từ embedding đã lưu của checkpoint: cosine nhỏ nhất phải ≥ 0.999, nếu không thì **dừng ngay**.

**Luật 5 — Có thể kết hợp các phương án khác cùng với InternVL2-2B.**
- `InternVL2-2B` (`OpenGVLab/InternVL2-2B`) giữ vai trò mô hình sinh dense caption chính theo `context.md`; không loại bỏ hay thay thế hoàn toàn bằng mô hình khác.
- Có thể sử dụng `Vintern` (ví dụ `Vintern-1B-v3.5`) hoặc các phương án bổ trợ khác (hậu xử lý, trích xuất thuộc tính, đối chiếu/hợp nhất caption) để nâng cao chất lượng mô tả tiếng Việt cho `InternVL2-2B`.
- Mọi bản ghi caption phải ghi rõ `caption_model`, `model_revision`, `seed`, tham số sinh và phương pháp kết hợp/chọn lọc (nếu có) để đảm bảo truy vết và tái lập.
- Trong đánh giá và báo cáo (Luật 7), phải tách riêng số liệu **chỉ dùng InternVL2-2B** và **kết hợp bổ trợ** (kèm bootstrap CI nếu so sánh hai cấu hình), không dùng nguồn ngoài đề cương để ngầm thay thế baseline.

**Luật 6 — Hash Guard.**
Engine so SHA-256 của checkpoint trong `config.yaml` với `manifest.json` của index trước khi khởi động. Lệch thì từ chối nạp.
Mỗi index có `manifest.json` gồm: `checkpoint_hash`, `catalog_hash`, số vector, tham số HNSW, commit hash, seed.

**Luật 7 — Số liệu thật, không chỉnh để đạt.**
- Mọi số liệu lấy từ script chạy thật, kèm commit hash và seed.
- Không đổi định nghĩa chỉ số, gallery hay bộ test sau khi xem kết quả để "đạt chỉ tiêu".
- Không đoán khi thiếu bằng chứng. Ghi rõ "không xác minh được".

---

## C. Quy tắc đánh giá

1. Chia train/val/test **theo sản phẩm** trước mọi bước khác. Danh sách test được khóa và lưu hash.
2. Chỉ tiêu chính thức chỉ đối chiếu trên **gallery gồm test split**.
3. Số trên gallery lớn có chứa ảnh train phải ghi rõ là **ước lượng lạc quan (tham khảo)**.
4. Báo cáo tách riêng **"Model thuần"** (checkpoint + cosine) và **"Hệ thống"** (có parser, lexicon, soft bonus).
5. Bộ truy vấn unseen chỉ chạy một lần. Không chỉnh parser hay lexicon theo kết quả của bộ đó. Nếu có điều chỉnh sau khi xem, bộ đó chuyển nhãn **DEV SET**.
6. Số liệu theo slot mỏng, theo màu và theo giới tính phải tách riêng, không chỉ báo trung bình chung.
7. Khoảng tin cậy bằng bootstrap 1.000 lần khi so sánh hai cấu hình.
8. Đo hubness (ảnh xuất hiện trong Top-10 của quá nhiều truy vấn). Nếu có, thử mean-centering và báo cáo trước/sau.
9. Độ trễ: đo riêng từng luồng (Text→Ảnh, Ảnh→Ảnh, CIR catalog, CIR upload), ≥ 5 run × 200 truy vấn, warmup 20, `torch.cuda.synchronize()` quanh forward, ghi nhiệt độ GPU, báo p50/p95/p99.
10. Chưa đạt chỉ tiêu thì báo cáo trung thực, kèm nguyên nhân và đề xuất. Không tự hạ chuẩn.

---

## D. Quy tắc dense caption (InternVL2-2B)

1. Ghi cho mỗi caption: `caption_model`, `model_revision`, `seed`, tham số sinh.
2. Caption viết bằng tiếng Việt, súc tích. Thông tin phân biệt (loại, giới, màu, kiểu dáng) đứng **đầu** câu.
3. Đo độ dài bằng đúng tokenizer SigLIP. Chuỗi text ghép phải ≤ 128 token và không bị cắt mất phần phân biệt.
4. Trước khi chạy toàn bộ: thử 200 ảnh, đọc thủ công 20 caption, kiểm tra tiếng Việt tự nhiên, không lẫn tiếng Anh/Trung, không bịa thuộc tính.
5. Checkpoint chưa được train với caption InternVL2, nên điểm paired retrieval dùng caption mới **không so trực tiếp** với số liệu cũ. Phải ghi rõ điều này trong báo cáo.
6. Lưu theo shard, có resume. Ảnh lỗi ghi log riêng, không bỏ qua im lặng.
7. Caption cũ không xóa. Chuyển vào `data/_quarantine/caption_vintern_replaced/` kèm log.
8. Kiểm tra VRAM thật trước khi chạy. Nếu không nạp nổi ở FP16, báo cáo phương án (8-bit/4-bit/CPU) và đánh đổi. Không tự chọn phương án làm giảm chất lượng mà không báo.

---

## E. Quy trình các pha (mỗi pha kết thúc bằng báo cáo ngắn và dừng chờ duyệt)

| Pha | Nội dung | Điểm dừng |
|---|---|---|
| 1 | Thu thập (Shopping100k, Shopee/Kaggle, crawl Tiki/Shopee), kiểm kê, lọc chất lượng, khử trùng SHA-1/pHash, chuẩn hóa nhãn, chia train/val/test | Duyệt nguồn dữ liệu và bảng kiểm kê |
| 2 | Dense caption bằng InternVL2-2B (thử 200 ảnh rồi mới chạy toàn bộ) | Duyệt kết quả thử 200 ảnh |
| 3 | Nạp checkpoint cũ, kiểm hash, kiểm định tiền xử lý (Luật 4). **Không train** | Duyệt kết quả cosine ≥ 0.999 |
| 4 | Embed gallery, dựng FAISS HNSW, manifest, sanity check | Duyệt kết quả sanity check |
| 5 | Đánh giá, hubness, SLA độ trễ | Duyệt số liệu và nhận xét đạt/chưa đạt |
| 6 | Ứng dụng Streamlit, chạy thử 4 luồng | Duyệt demo |
| 7 | Báo cáo tổng kết, `README_run.md`, pytest, commit | Duyệt cuối |

Sanity check Pha 4 bắt buộc: 0 id trùng; 0 ảnh test lọt vào dữ liệu train; truy vấn lại một ảnh phải trả chính nó ở rank 1 (cosine 1.0000); số vector khớp manifest.

---

## F. Giới hạn đã biết (ghi trung thực, không che)

1. Checkpoint được train với dense captions từ Vintern-1B-v3.5. Đây là sự kiện lịch sử, không phải cấu hình đang dùng. Cách ghi trong báo cáo do người dùng quyết định: (A) giữ một dòng ghi chú trung thực, hoặc (B) chỉ giữ trong lịch sử git. Agent không tự quyết.
2. `UpgradedAttentionCIRCombiner` được train với modifier **tiếng Anh**. Luật 1 cấm train lại nên giới hạn này được ghi nhận, chưa sửa. Cải thiện chỉ ở tầng suy luận.
3. Model thuần có thể không đạt Recall@10 ≥ 0.85 hoặc mAP ≥ 0.75 trên gallery chỉ gồm test split. Nếu vậy, báo cáo phải nói rõ phần nào do model, phần nào do hệ thống.
4. Cấu hình train trong `context.md` (batch 256, AdamW, lr 1e-4, warmup 1.000 step, 10 epoch) không áp dụng khi dùng lại checkpoint. Báo cáo phải nêu rõ điều này.

---

## G. Cách báo cáo

Mỗi báo cáo pha gồm: kết quả chính, số liệu thật, vấn đề gặp phải, tệp đã thay đổi, số test pytest, commit hash.
Không sang pha sau khi người dùng chưa duyệt.
