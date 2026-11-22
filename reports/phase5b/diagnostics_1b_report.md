# Báo cáo Chẩn đoán Bổ sung — Bước 1b (Pha 5b)

- **Trạng thái:** Hoàn thành Bước 1b — **DỪNG CHỜ DUYỆT TRƯỚC KHI SANG BƯỚC 2**
- **Lưu ý quy tắc (C.5):** Tập `eval/handwritten_queries.json` là **DEV SET**, các số liệu dưới đây chỉ dùng để chẩn đoán nguyên nhân và tách lỗi, **KHÔNG** dùng để chọn tham số hay làm bằng chứng đạt chỉ tiêu.
- **Seed cố định:** `42` | **Checkpoint SHA-256:** `e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d`

---

## 1b.1. Kiểm tra `git diff` và Đính chính Pha 4

### 1. Kết quả `git diff 2e057d6..bcc5284 -- src/serve src/eval config.yaml`
- `config.yaml`: **Không có thay đổi**.
- `src/serve/` và `src/eval/`: Toàn bộ 6 module đánh giá và suy luận mới của Pha 5 (`src/eval/eval_captions.py`, `eval_latency.py`, `eval_retrieval.py`, `metrics.py`, `src/serve/parser.py`, `search_engine.py`) được thêm mới tại commit `bcc5284` (cùng chỉnh sửa 16 dòng ở `src/serve/__init__.py`).
- **Kết luận:** Commit `2e057d6` là HEAD trước lúc chạy eval; toàn bộ mã nguồn và kết quả Pha 5 được đóng gói đầy đủ tại commit `bcc5284`.

### 2. Đính chính chính thức về `index/reference_gallery/` (Pha 4)
- **(a) Thành phần sản phẩm:** Báo cáo Pha 4 ghi nhận `reference_gallery` chứa "38.646 sản phẩm in-scope sạch", tuy nhiên kiểm tra thực tế trên `id_map.json` cho thấy có **38.644** sản phẩm `in_scope=True` và **2 sản phẩm `in_scope=False`** (nhưng đạt chuẩn chất lượng ảnh `filter_flag='pass'`) là ID `43230` (*Wildcraft Unisex Black & Grey Carabiner Key Chain*) và ID `42066` (*American Tourister Unisex Black Key Case*), vốn nằm trong tập `split='train'` của checkpoint cũ.
- **(b) Nguồn vector:** Vector của `reference_gallery` được **nạp trực tiếp từ tệp embedding lịch sử** (`data/_quarantine/legacy_rebuild_20261111/untouchable/index/image_embeddings.npy`) trong `src/index/build_index.py`, **không phải** embed lại bằng pipeline mới. Chỉ có `index/test_gallery/` (1.290 sản phẩm) được embed lại bằng pipeline tiền xử lý chuẩn của Pha 3–4.

---

## 1b.2. Kiểm tra Đối xứng Định nghĩa (`cut@100` vs `full@1290`)

Trong `src/serve/search_engine.py` (dòng 169):
- Đường `mode == "model_only"` thiết lập `fetch_k = top_k` (bỏ qua tham số `candidate_pool`). Khi `eval_retrieval.py` gọi với `eval_top_k = 100`, `model_only` luôn cắt danh sách tại Top-100 bất kể truyền `candidate_pool=100` hay `1290`.
- Đường `mode == "system"` lấy `fetch_k = max(top_k, candidate_pool)` để re-rank rồi mới cắt `results[:top_k]`.
- Vì mẫu số AP luôn chia cho tổng số ảnh đúng trên toàn gallery (`total_relevant`), việc cắt ở 100 làm các ảnh đúng ở rank 101..1290 không được cộng vào tử số AP.

Bảng đối chiếu định nghĩa đối xứng trên **DEV SET (86 truy vấn có GT, test gallery N=1.290)**:

| Phương pháp | Phạm vi tính xếp hạng | Recall@1 | Recall@10 | mAP |
|---|---|---:|---:|---:|
| **Model thuần** (`cut@100` — định nghĩa Pha 5) | Top-100 cosine | `0.3023` | `0.7093` | `0.3282` |
| **Model thuần** (`full@1290` — đối xứng toàn gallery) | Toàn bộ 1.290 ảnh | `0.3023` | `0.7093` | `0.3389` |
| **Hệ thống (`pool=100`, `gamma=0.3`, `cut@100`)** | Pool 100 -> Top-100 | `0.6047` | `0.8488` | `0.5537` |
| **Hệ thống (`pool=1290`, `gamma=0.3`, `cut@100`)** | Pool 1290 -> Top-100 | `0.6047` | `0.8721` | `0.5852` |
| **Hệ thống (`pool=1290`, `gamma=0.3`, `full@1290`)** | Toàn bộ 1.290 ảnh | `0.6047` | `0.8721` | `0.5942` |

*(Ghi chú: Không thay đổi định nghĩa chính thức `cut@100` của Pha 5; số `full@1290` chỉ báo cáo thêm để kiểm chứng tính đối xứng — chênh lệch cắt top-100 khoảng ~0.01 mAP).*

---

## 1b.3. Kết quả ORACLE PARSER (DEV SET, `pool=1290`, `gamma=0.3`, `cut@100`)

Thay thế đầu ra của `VietnameseQueryParser` bằng nhãn chuẩn từ `spec` (`articleType`, `gender`, `baseColour`) để tách rạch ròi phần mất điểm do **Parser/Lexicon** vs phần do **Trọng số bonus / Model SigLIP**:

### Tổng thể (86 truy vấn có GT)
| Cấu hình (`pool=1290`, `gamma=0.3`) | Recall@1 | Recall@5 | Recall@10 | mAP (`cut@100`) | mAP (`full@1290`) |
|---|---:|---:|---:|---:|---:|
| **Model thuần** | `0.3023` | `0.5698` | `0.7093` | `0.3282` | `0.3389` |
| **Hệ thống (Parser hiện tại)** | `0.6047` | `0.8488` | `0.8721` | `0.5852` | `0.5942` |
| **Hệ thống (Oracle Parser)** | **`0.7326`** | **`0.8953`** | **`0.9070`** | **`0.6561`** | **`0.6645`** |

### Phân tách theo Có dấu / Không dấu
| Nhóm | Cấu hình (`pool=1290`) | n_queries | Recall@1 | Recall@5 | Recall@10 | mAP (`cut@100`) |
|---|---|---:|---:|---:|---:|---:|
| **Có dấu** | Parser hiện tại | 65 | `0.7077` | `0.8923` | `0.9077` | `0.6782` |
| **Có dấu** | **Oracle Parser** | 65 | **`0.8462`** | **`0.9538`** | **`0.9538`** | **`0.7589`** |
| **Không dấu** | Parser hiện tại | 21 | `0.2857` | `0.7143` | `0.7619` | `0.2973` |
| **Không dấu** | **Oracle Parser** | 21 | **`0.3810`** | **`0.7143`** | **`0.7619`** | **`0.3379`** |

### Phân tách theo Slot mỏng (`category`)
| Slot mỏng | n | Parser hiện tại (R@10 / mAP) | Oracle Parser (R@10 / mAP) | Chênh lệch mAP (lỗi Parser) |
|---|---:|---:|---:|---:|
| **Áo** | 19 | `1.0000` / `0.5906` | `1.0000` / `0.6513` | `+0.0607` |
| **Giày dép** | 17 | `0.9412` / `0.6424` | `0.9412` / `0.6880` | `+0.0456` |
| **Quần & Váy** | 16 | `0.8125` / `0.5725` | `0.8125` / `0.6522` | `+0.0797` |
| **Túi & Balo** | 14 | `0.8571` / `0.5827` | `1.0000` / `0.6774` | **`+0.0947`** |
| **Phụ kiện** | 10 | `0.9000` / `0.6716` | `0.9000` / `0.6772` | `+0.0056` |
| **Đồng hồ** | 10 | `0.6000` / `0.4146` | `0.7000` / `0.5665` | **`+0.1519`** |

> **Kết luận tách lỗi:**
> 1. Lỗi **Parser/Lexicon** làm mất **`0.0709` mAP** và **`0.0349` R@10** (nặng nhất ở **Đồng hồ** `+0.1519` và **Túi & Balo** `+0.0947`).
> 2. Ở nhóm **Có dấu**, khi khắc phục được Parser (Oracle), mô hình đạt **R@10 = 0.9538** và **mAP = 0.7589** (vượt cả 2 ngưỡng chỉ tiêu!).
> 3. Ở nhóm **Không dấu**, ngay cả với **Oracle Parser** (`gamma=0.3`), mAP vẫn chỉ đạt **`0.3379`** và R@10 = **`0.7619`**. Điều này chứng minh có một vấn đề rất lớn nằm ở **độ lệch điểm cosine của câu không dấu** so với trọng số `gamma=0.3`.

---

## 1b.4. Phân tích chi tiết 13 Truy vấn Thất bại & Nguyên nhân cho 4 câu (26, 29, 72, 94)

| ID | Truy vấn | Parsed vs Spec | Rank (Base / Model / SysFull) | Ảnh đúng #1 (`cos + bon = tot`) | Ảnh sai Top-1 (`cos + bon = tot`) |
|---:|---|---|---:|---|---|
| **26** | `quan jean nam xanh` | Đúng 3/3 slot | `>100` / `394` / `51` | `0.1214 + 0.300 = 0.4214` | `0.4256 + 0.200 = 0.6256` (`Backpacks \| Unisex \| Blue`) |
| **29** | `quan tay nam cong so mau den` | Đúng 3/3 slot | `>100` / `162` / `3` | `0.3357 + 0.300 = 0.6357` | `0.4866 + 0.200 = 0.6866` (`Casual Shoes \| Unisex \| Black`) |
| **41** | `quần lót boxer nam co giãn tốt` | Thiếu `articleType` (`[]` vs `['Trunk', 'Boxers']`) | `14` / `14` / `14` | `0.5953 + 0.300 = 0.8953` | `0.7022 + 0.300 = 1.0022` (`Briefs \| Men \| Olive`) |
| **44** | `áo liền quần jumpsuit nữ thanh lịch` | Thiếu `articleType` (`[]` vs `['Jumpsuit']`) | `>100` / `360` / `265` | `0.1859 + 0.300 = 0.4859` | `0.7408 + 0.300 = 1.0408` (`Tops \| Women \| Pink`) |
| **63** | `giày đá bóng sân cỏ nhân tạo nam` | Thiếu `articleType` (`[]` vs `['Sports Shoes']`) | `48` / `62` / `48` | `0.4837 + 0.300 = 0.7837` | `0.6880 + 0.300 = 0.9880` (`Formal Shoes \| Men \| Brown`) |
| **72** | `vi da nam mau den` | Đúng 3/3 slot | `>100` / `139` / `5` | `0.3016 + 0.300 = 0.6016` | `0.4521 + 0.200 = 0.6521` (`Rain Jacket \| Men \| Black`) |
| **76** | `túi đeo chéo nam đựng ipad thể thao` | Sai `articleType` (`Handbags` vs `Messenger Bag`) | `40` / `44` / `40` | `0.5507 + 0.150 = 0.7007` | `0.5848 + 0.300 = 0.8848` (`Handbags \| Unisex \| Khaki`) |
| **77** | `túi trống thể thao tập gym đựng đồ` | Thiếu `articleType` & `gender` | `12` / `5` / `18` | `0.6941 + 0.000 = 0.6941` | `0.6136 + 0.300 = 0.9136` (`Socks \| Unisex \| Red`) |
| **81** | `dong ho nam day da mau den` | Đúng 3/3 slot | `>100` / `616` / `210` | `0.1271 + 0.300 = 0.4271` | `0.7125 + 0.100 = 0.8125` (`Casual Shoes \| Men \| Beige`) |
| **83** | `dong ho nam day kim loai` | Thiếu `baseColour` (`[]` vs `['Steel', 'Silver']`) | `>100` / `409` / `139` | `0.2567 + 0.300 = 0.5567` | `0.6725 + 0.150 = 0.8225` (`Sandals \| Men \| Tan`) |
| **85** | `dong ho nu mau trang thanh lich` | Đúng 3/3 slot | `>100` / `328` / `52` | `0.3326 + 0.300 = 0.6326` | `0.6650 + 0.200 = 0.8650` (`Heels \| Women \| White`) |
| **88** | `đồng hồ nam mặt vuông dây cao su` | Thiếu `baseColour` (`[]` vs `['Black']`) | `16` / `16` / `16` | `0.5847 + 0.300 = 0.8847` | `0.6788 + 0.300 = 0.9788` (`Watches \| Men \| Cream`) |
| **94** | `day nit nam da bo` | Thiếu `baseColour` (`[]` vs `['Brown', 'Black']`) | `>100` / `184` / `16` | `0.3167 + 0.300 = 0.6167` | `0.5627 + 0.150 = 0.7127` (`Jackets \| Men \| Olive`) |

### Giải thích nguyên nhân cho 4 câu không dấu (26, 29, 72, 94) dù `pool=1290`:
1. Trong cả 4 câu, SigLIP Text Encoder **không hiểu được cụm từ khóa không dấu** (`quan jean`, `quan tay`, `vi da`, `day nit`), dẫn đến vector text bị lệch hẳn hướng khỏi vùng embedding của ảnh đúng: điểm `cosine` của ảnh đúng chỉ đạt **`0.1214` – `0.3357`** (so với ~0.60 ở câu có dấu).
2. Trong khi đó, các sản phẩm sai (như `Backpacks`, `Casual Shoes`, `Rain Jacket`, `Jackets`) chỉ tình cờ khớp **giới tính + màu** (`bonus = 0.150 – 0.200`) nhưng lại có điểm `cosine` ngẫu nhiên cao hơn hẳn (**`0.4256` – `0.5627`**).
3. Với `gamma = 0.3`, chênh lệch bonus giữa ảnh đúng (khớp 3/3 slot: `+0.300`) và ảnh sai (khớp 2/3 slot: `+0.200`) chỉ là **`+0.100`** — không đủ để bù đắp mức chênh lệch cosine **`0.150 – 0.304`**! Vì vậy, ngay cả khi lấy `pool = 1290`, câu 26 vẫn kẹt ở rank 51 và câu 94 vẫn kẹt ở rank 16 (câu 29 lên rank 3, câu 72 lên rank 5 nhưng bị tụt AP nghiêm trọng).

---

## 1b.5. Quét `gamma` và `hard-tier` (CHỈ CHẨN ĐOÁN, DEV SET, `pool=1290`, `cut@100`)

| Parser | Trọng số / Phương pháp | Recall@1 | Recall@10 | mAP (`cut@100`) | mAP (`full@1290`) |
|---|---|---:|---:|---:|---:|
| Parser hiện tại | `gamma = 0.3` | `0.6047` | `0.8721` | `0.5852` | `0.5942` |
| Parser hiện tại | `gamma = 0.5` | `0.7209` | `0.8953` | `0.6881` | `0.6958` |
| Parser hiện tại | `gamma = 1.0` | `0.7791` | `0.9186` | `0.7727` | `0.7781` |
| Parser hiện tại | `gamma = 2.0` | `0.8023` | `0.9186` | `0.8216` | `0.8268` |
| Parser hiện tại | `hard-tier` (khớp thuộc tính trước, cosine sau) | `0.8023` | `0.9186` | `0.8246` | `0.8298` |
| **Oracle Parser** | `hard-tier` | **`1.0000`** | **`1.0000`** | **`0.9962`** | **`1.0000`** |

*(Ghi chú: Bảng này chỉ dùng để chẩn đoán độ nhạy của `gamma` đối với hàm mục tiêu, tuyệt đối không chọn tham số cuối từ DEV SET).*

---

## 1b.6. Hạn chế của Ground Truth (Tính vòng tròn) & Đề xuất GT phụ không vòng tròn

### 1. Tính vòng tròn (Circularity) của GT hiện tại
- Bộ đánh giá chính thức sử dụng GT `hybrid` được suy ra từ 3 trường metadata có cấu trúc: `(articleType, gender, baseColour)`.
- Đồng thời, tầng suy luận `mode="system"` cũng sử dụng `VietnameseQueryParser` để trích xuất đúng 3 trường `(articleType, gender, baseColour)` từ câu truy vấn và cộng điểm `gamma * match_frac`.
- Hệ quả được phơi bày rõ ở mục 1b.5: Nếu tăng `gamma >= 1.0` hoặc dùng `hard-tier`, hệ thống thực chất biến thành bộ lọc metadata cứng theo đúng 3 trường định nghĩa của GT (khi Oracle Parser đạt `mAP = 1.0000`), làm lu mờ vai trò đo lường năng lực xếp hạng thị giác/ngữ nghĩa thực sự của mô hình SigLIP.

### 2. Đề xuất GT phụ không vòng tròn (Non-circular Secondary GT)
- Kiểm tra trên tệp nguồn `data/fashion-dataset/styles.csv` cho thấy có 2 trường metadata ngữ cảnh phong phú mà **hệ thống tìm kiếm hiện tại hoàn toàn KHÔNG sử dụng để phân tích hay cộng điểm**:
  1. **`usage`** (`Casual`: 34.406, `Sports`: 4.025, `Ethnic`: 3.208, `Formal`: 2.345, `Smart Casual`: 67 — chỉ 317/44.424 null).
  2. **`season`** (`Summer`: 21.472, `Fall`: 11.431, `Winter`: 8.517, `Spring`: 2.983 — chỉ 21 null).
- **Cách áp dụng song song (không thay thế GT chính thức):**
  - Với các truy vấn có chỉ định ngữ cảnh mục đích sử dụng hoặc mùa (ví dụ: *công sở / đi làm* $\rightarrow$ `Formal` / `Smart Casual`; *thể thao / tập gym / đá bóng / chạy bộ* $\rightarrow$ `Sports`; *mùa hè / đi biển* $\rightarrow$ `Summer`; *mùa đông / ấm áp* $\rightarrow$ `Winter`), thiết lập thêm cột đánh giá **Non-Circular GT** yêu cầu sản phẩm đúng phải thỏa mãn cả `(articleType, gender, baseColour)` **VÀ** trường `usage`/`season` tương ứng.
  - Vì parser không trích xuất và không cộng điểm cho `usage`/`season`, mọi cải thiện trên chỉ số GT phụ này đều phản ánh **năng lực hiểu ngữ nghĩa thị giác thật sự của SigLIP** (hoặc của việc khôi phục dấu giúp SigLIP hiểu đúng ngữ cảnh "cong so", "the thao", "mua dong").
