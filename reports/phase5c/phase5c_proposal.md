# Đề xuất Cải thiện Tầng Suy luận — Pha 5c (`selection_rule_v4` Draft)

- **Nhánh:** `phase7a` | **Mốc tiền đề:** `phase5b-step3` (commit `2a0de876d89432587250ea4eb4372a3966fc473f`) | **Seed:** `42`
- **Trạng thái:** **CHỈ THIẾT KẾ — CHƯA chạy thực nghiệm, CHƯA sửa code, CHƯA chạy lưới mới.** Chờ người dùng duyệt trước khi khóa `selection_rule_v4.json/.md`.
- **Bất biến giữ nguyên:** `best_model.pt` (`SHA-256: e2f82fee...`, Luật 1), `parser_v1.py` (`111002f6...`), `parser_v2.1` (`1262383c...`), `selection_rule_v3.*`, các tập `DEV / Val A / Val B-synth / Val C / B-human`, và `index/test_gallery/` (chỉ chạy đúng 1 lần ở Bước 5).

---

## 1. Bối cảnh từ Bước 3 (`v3`)

Trên lưới `v3` (`index/val_gallery/`, `1.884` sản phẩm, `exact + conservative tie`, `zscore_cosine=true`):
- **`cfg_13` (thắng cuộc sau phá hòa CI):** `parser_v2.1`, $\gamma=1.5$ đều, `query_rewrite=true` — CV Score **`0.5053`** (`A mAP(cl)=0.7546`, `Bc mAP(cl)=0.3391`, `Bt mAP(cl)=0.2522`, `C20 mAP(cl)=0.4966`, `p95=54.0 ms`), B-human gate **`0.6285`** (`R@1=0.7538`, `R@10=0.8615`, `ΔmAP vs cfg_01 = +0.3036`, `CatDrop=3.08%`).
- **`cfg_19` (tốt nhất theo Score thô):** `CV=0.5391` (`bm25=true`, `p95=55.7 ms`, nằm trong 95% CI của `cfg_13`).
- **Ba nút thắt lộ rõ từ `v3`:**
  1. **Thang z-score & biên lưới $\gamma=1.5$:** Độ lệch chuẩn cosine trên Val A là `0.1622`; `zscore_cosine=true` khiến $\gamma=0.3$ yếu đi ~`6.2×` (`cfg_01` DEV mAP `0.3749` vs `0.5852`), đẩy nghiệm thắng ra biên $\gamma=1.5$ (CV Score vẫn tăng đơn điệu).
  2. **Hổng mAP trên `B-synth` (`Bc=0.3391`, `Bt=0.2522`):** Do (i) câu `B-synth` không dấu/typo làm parser bỏ sót hoặc model thuần xếp hạng sâu kém (`model_only Bc mAP=0.1071`), và (ii) khi nhiều sản phẩm cùng khớp 3/3 thuộc tính, thứ tự bên trong nhóm phụ thuộc hoàn toàn vào cosine thuần của checkpoint.
  3. **Rủi ro hubness & lỗi gõ:** Chưa đo/khử hubness bằng CSLS trong lưới và `parser_v2.1` chưa có cơ chế khớp mờ cho typo (`Bt mAP=0.2522` vs `Bc mAP=0.3391`).

---

## 2. Phân tích 8 Ý tưởng Ứng viên (I1–I8)

### I1. Bonus theo thang xếp hạng / phân vị thay vì cộng cosine thô
- **(a) Cơ chế:** Thay vì cộng $\gamma \cdot s_{\text{attr}}$ trực tiếp vào $z(\cos)$, định nghĩa hệ thống **xếp tầng (tier)** theo số slot khớp chắc chắn (hoặc trọng số slot), và dùng biên độ tỉ lệ với $\sigma_q$ (hoặc theo khoảng cách phân vị cosine trong pool, ví dụ $s_{\text{final}} = \cos(q, d) + \gamma_{\text{rel}} \cdot \sigma_q \cdot s_{\text{attr}}$ hoặc tier lexicographic chỉ cho slot tin cậy cao). Khi slot không chắc chắn thì không gating cứng.
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Chuẩn hóa biên độ bonus theo $\sigma_q$ (hoặc phân vị) sẽ giải quyết hiện tượng sụt `cfg_01` dưới `zscore_cosine=true` và giảm sự phụ thuộc vào biên $\gamma=1.5$, tăng `B-synth clean mAP(cl)` $\ge +0.02$ so với `cfg_13` mà không làm tăng `CatDrop(B)` vượt `5%`.
  - *Bác bỏ:* Nếu trên 5-fold CV `A + B-synth clean`, `I1` không vượt `cfg_13` ngoài 95% paired cluster bootstrap CI hoặc vi phạm ràng buộc `5(i)` (`C 20%`).
- **(c) Lưới cấu hình tối đa:** `4` cấu hình (`tier_soft` với $\gamma_{\text{rel}} \in \{1.0, 1.5, 2.0\}$ + `quantile_bonus` `1` mức).
- **(d) Chi phí độ trễ T2I ước lượng:** `+0.2` đến `+0.8 ms` (tính $\sigma_q$ / phân vị trên vector `1.884` phần tử bằng NumPy). Tổng p95 ước lượng `~54–55 ms` (đạt `< 200 ms`, tăng `<= 20 ms`).
- **(e) Rủi ro rò rỉ / vòng tròn GT / winner's curse:** Không dùng thông tin nhãn test; nhưng nếu chọn hệ số $\gamma_{\text{rel}}$ trên cùng `A + B-synth` đã quan sát ở `v3`, có rủi ro *winner's curse* nhẹ (cần cổng xác nhận mới).
- **(f) Tệp thêm/sửa & test:** Sửa `src/eval/canonical_eval.py` (thêm `bonus_mode="zscore"|"sigma_scaled"|"tier_soft"`), thêm unit test trong `tests/test_phase5c_v4.py` kiểm tra tính bất biến theo $\sigma_q$ và không phá tie-rule conservative.
- **(g) Tuân thủ luật:** Luật 1 (không đụng trọng số), Luật 4 (không đổi tiền xử lý), Luật 7 & C.5.

### I2. Gating theo độ tin cậy parser (`parser_confidence_gating`)
- **(a) Cơ chế:** `parser_v2.1` xuất thêm mức tin cậy cho từng slot: `high` (khớp từ khóa có dấu hoặc có tiền tố `"màu "`/`"mau "` hoặc cụm danh từ đầy đủ nhiều âm tiết) vs `low` (khớp không dấu đơn âm / fallback). Chỉ áp dụng tier/gating mạnh cho slot `high`; slot `low` chỉ nhận bonus mềm nhỏ ($\gamma_{\text{low}} = 0.2\gamma$).
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Khi $\gamma$ lớn ($\ge 1.5$), phần lớn `CatDrop` đến từ số ít trường hợp parser khớp nhầm ở ngữ cảnh không dấu/mơ hồ; gating theo độ tin cậy sẽ giảm `CatDrop(B)` xuống $\le 0.5\%$ và cải thiện `C 20%` (`mAP` tăng $\ge +0.01$ so với gating đồng nhất).
  - *Bác bỏ:* Nếu `CatDrop(B)` không giảm hoặc `A/B-synth mAP` giảm ngoài 95% CI do bỏ lỡ nhiều câu không dấu thật (`dia_class="none"`).
- **(c) Lưới cấu hình tối đa:** `3` cấu hình (`gating_high_only`, `gating_high_soft_low` ở $\gamma \in \{1.5, 2.0\}$).
- **(d) Chi phí độ trễ T2I ước lượng:** `+0.1 ms` (trả về dict confidence ngay trong `parse()`). Tổng p95 `~54 ms`.
- **(e) Rủi ro:** Không dùng GT; tránh tinh chỉnh ngưỡng confidence theo câu cụ thể của `A/B-synth/B-human`.
- **(f) Tệp thêm/sửa & test:** Tạo `src/serve/parser_v2_conf.py` (kế thừa/bọc `parser_v2.1`, KHÔNG sửa `src/serve/parser_v2.py` đã đóng băng); test chứng minh câu không dấu có ngữ cảnh `"mau "` được gắn `high`, đơn âm mơ hồ gắn `low`.
- **(g) Tuân thủ luật:** Luật 1, Luật 3, giữ nguyên `parser_v2.1` đóng băng.

### I3. Mở rộng truy vấn phía text (đa mẫu chuẩn hóa từ catalog)
- **(a) Cơ chế:** Mở rộng `query_rewrite` + `dual_encode` của `v3`: tạo trung bình L2-normalized của embedding câu gốc, câu khôi phục dấu, và 2 mẫu chuẩn hóa sinh **thuần túy từ nhãn catalog đã parse** (ví dụ `"[articleType_vi] [gender_vi] màu [baseColour_vi]"` nếu đủ slot tin cậy).
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* SigLIP được pretrain/đánh giá với caption mô tả chuẩn tiếng Việt (`[Loại] [giới] màu [màu]`); trung bình hóa với mẫu chuẩn sẽ nâng cosine của ảnh đúng khi câu truy vấn tự nhiên viết tắt hoặc đảo trật tự, tăng `B-synth clean mAP(cl)` $\ge +0.02$.
  - *Bác bỏ:* Nếu làm giảm `mAP` ở câu có mô tả phụ (kiểu dáng, chất liệu) do bị pha loãng vector, hoặc vượt ngưỡng độ trễ `+20 ms` (vì encode 3–4 chuỗi thay vì 1).
- **(c) Lưới cấu hình tối đa:** `3` cấu hình (`2` mẫu, `3` mẫu, hoặc chỉ thêm mẫu chuẩn hóa khi `parser` đủ `>= 2` slot).
- **(d) Chi phí độ trễ T2I ước lượng:** **CAO** nếu gọi `encode_text_query` tuần tự (`dual_encode` 2 lần đã tăng từ `~35 ms` lên `~100 ms` p95 ở `cfg_14..16`!). Nếu **batch encode** 1 lần (`batch_size=3–4` trên GPU), ước lượng tăng `+10` đến `+18 ms`, p95 `~50–55 ms` (vẫn có nguy cơ chạm ngưỡng `+20 ms` so với `cfg_01` `35.9 ms`).
- **(e) Rủi ro:** Mẫu sinh chỉ lấy từ bảng ánh xạ nhãn `catalog_rebuild.parquet`, tuyệt đối không lấy mẫu câu từ `DEV/A/B-synth/B-human`.
- **(f) Tệp thêm/sửa & test:** Mở rộng `src/serve/query_rewriter.py` (hàm `multi_template_encode_texts` dùng batch forward); test xác nhận shape `(N, 1152)`, L2-norm `= 1.0`, và `batch` khớp `single` cosine $\ge 0.9999$.
- **(g) Tuân thủ luật:** Luật 4 (cùng `SiglipProcessor` `max_length=128`), ràng buộc `5(iv)` độ trễ.

### I4. Pseudo-Relevance Feedback (Rocchio) trong không gian embedding ảnh
- **(a) Cơ chế:** Sau lượt tìm kiếm đầu tiên, lấy top-$k$ ảnh ($k \in \{3, 5\}$) có điểm cao nhất trong pool, tính $q' = \text{norm}(q + \beta \cdot \frac{1}{k}\sum_{j=1}^k v_{d_j})$ ($\beta \in \{0.2, 0.4\}$), rồi chấm lại cosine trên pool/gallery.
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Các truy vấn có nhiều ảnh đúng (`n_relevant >= 5`) thường có một số ảnh đúng nằm sâu ở hạng 15–50 vì khác góc chụp/ánh sáng; kéo $q'$ về phía top-$k$ ảnh đúng sẽ gom cụm ảnh cùng loại lên top-10, tăng `mAP` $\ge +0.03$.
  - *Bác bỏ:* Nếu top-$k$ ban đầu chứa ảnh sai (đặc biệt trên `B-synth` khi `model_only R@1=0.109`), Rocchio sẽ **trôi chủ đề (query drift)** và làm tăng **hubness** + `CatDrop(B) > 5%`.
- **(c) Lưới cấu hình tối đa:** `3` cấu hình: `(k=3, β=0.2)`, `(k=5, β=0.2)`, `(k=5, β=0.4)` (chỉ kích hoạt trên top-$k$ đã qua lọc `parser` nếu dùng hệ thống).
- **(d) Chi phí độ trễ T2I ước lượng:** `+0.5` đến `+1.5 ms` (matmul lần 2 trên `1.884 × 1152` float32). Tổng p95 `~55 ms` (đạt `5(iv)`).
- **(e) Rủi ro:** Nguy cơ tăng hubness rất cao (đo bắt buộc theo C.8); không dùng GT.
- **(f) Tệp thêm/sửa & test:** Thêm `apply_rocchio_prf()` trong `src/eval/canonical_eval.py`; test kiểm tra $q'$ L2-normalized, không làm thay đổi kết quả khi $\beta=0$.
- **(g) Tuân thủ luật:** Luật 1, C.8 (đo hubness trước/sau).

### I5. Hợp nhất Text–Ảnh phía Gallery (Index văn bản phụ trợ)
- **(a) Cơ chế:** Mã hóa văn bản mô tả sản phẩm ở gallery (gồm caption `InternVL2-2B` chính, cấu hình kết hợp `Vintern` theo Luật 5, hoặc metadata tiếng Việt) bằng chính text tower của `best_model.pt` (qua cùng `SiglipProcessor`, `max_length=128`), lưu vào thư mục index mới riêng biệt (ví dụ `index/val_gallery_text_v4/` có `manifest.json` + Hash Guard). Điểm cuối: $s = \alpha \cos(q_{\text{txt}}, d_{\text{img}}) + (1-\alpha)\cos(q_{\text{txt}}, d_{\text{txt}})$ (kết hợp bonus thuộc tính).
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Đối chiếu `text↔text` bổ trợ cho `text↔image` ở các khái niệm tiếng Việt khó (`B-synth` không dấu/typo, từ hiếm như `Steel`, `Dupatta`), giúp cải thiện sâu thứ hạng bên trong nhóm cùng thuộc tính (nút thắt của `mAP`).
  - *Bác bỏ:* Nếu text tower của checkpoint chưa căn chỉnh tốt cho `text↔text` tiếng Việt hoặc làm giảm `mAP` trên câu tự nhiên ngắn (`B-human`), hoặc vượt ngưỡng độ trễ.
- **(c) Lưới cấu hình tối đa:** `3` cấu hình ($\alpha \in \{0.7, 0.85\}$ cho `internvl2_only`, và `1` cấu hình `internvl2_plus_auxiliary` tách riêng theo Luật 5).
- **(d) Chi phí độ trễ T2I ước lượng:** `+0.5` đến `+1.5 ms` (thêm 1 phép nhân ma trận `1.884 × 1152` trên RAM/GPU). Tổng p95 `~55 ms`.
- **(e) Rủi ro rò rỉ / vòng tròn GT:** **CỰC KỲ CẨN TRỌNG.** Nếu văn bản gallery chứa nguyên văn `articleType_vi` / `baseColour_vi` / `gender_vi` mà tập A/B-synth cũng dùng để sinh truy vấn và làm GT hybrid, thì `cos(q_txt, d_txt)` trở thành khớp chuỗi/ngữ nghĩa **vòng tròn với GT**! Cần báo cáo song song với **I8 (GT phụ không vòng tròn)** và tách rõ `metadata-only` vs `internvl2_only` vs `internvl2_plus_auxiliary` (Luật 5).
- **(f) Tệp thêm/sửa & test:** Script dựng index văn bản mới `scripts/build_val_text_index.py` ghi vào `index/val_gallery_text_v4/` (Luật 3, Luật 6 `manifest.json`), cập nhật `canonical_eval.py` nhận `text_cos_matrix`; test Hash Guard và tách biệt `internvl2_only`.
- **(g) Tuân thủ luật:** Luật 1, Luật 3 (thư mục index mới, không đụng index cũ), Luật 4 (cùng preprocessor), Luật 5 (tách `InternVL2-2B` vs bổ trợ + bootstrap CI), Luật 6 (manifest + Hash Guard).

### I6. Đo & Khử Hubness (Mean-Centering + CSLS)
- **(a) Cơ chế:** Đo phân bố tần suất xuất hiện trong Top-10 (`hubness_top10`: max count, Gini, % hub > 5× kỳ vọng) theo C.8 trên `val_gallery`. Thử (1) `mean-centering` (đã có trong `cfg_20..22`, vốn làm tụt CV ở `v3`) và (2) **CSLS (Cross-domain Similarity Local Scaling)**: $\text{CSLS}(q, d) = 2\cos(q, d) - r_T(q) - r_S(d)$, với $r_S(d)$ là trung bình cosine của ảnh $d$ tới $k$ truy vấn lân cận (tính từ tập truy vấn/anchor pool độc lập, không lấy từ test) và $r_T(q)$ là trung bình top-$k$ cosine của truy vấn $q$ tới gallery.
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Nếu `val_gallery` có hubness cao, CSLS sẽ giảm tỉ lệ hub xuất hiện lặp lại mà không phá cấu trúc cosine gốc như mean-centering toàn cục.
  - *Bác bỏ:* Nếu hubness thực tế trên `val_gallery` thấp hoặc CSLS làm giảm `mAP` so với `cfg_13`.
- **(c) Lưới cấu hình tối đa:** `2` cấu hình (`CSLS k=10` trên `cfg_13`, `CSLS k=20` trên `cfg_13`).
- **(d) Chi phí độ trễ T2I ước lượng:** `+0.3 ms` ($r_S(d)$ tiền tính offline trên anchor pool; $r_T(q)$ tính `top-k` trên `1.884` phần tử). Tổng p95 `~54 ms`.
- **(e) Rủi ro:** Anchor pool để tính $r_S(d)$ tuyệt đối không được chứa truy vấn từ `test` hay tập xác nhận mới; phải dùng tập truy vấn train/dev hoặc mẫu caption catalog.
- **(f) Tệp thêm/sửa & test:** Thêm `compute_csls_scores()` vào `src/eval/canonical_eval.py` + test kiểm tra $r_S(d)$ làm giảm độ lệch chuẩn tần suất xuất hiện Top-10.
- **(g) Tuân thủ luật:** Luật 7, mục C.8 (báo hubness trước/sau).

### I7. Parser chịu lỗi gõ (Edit-distance giới hạn + bảo vệ homograph)
- **(a) Cơ chế:** Chỉ kích hoạt khi token/cụm trong câu truy vấn **không khớp chính xác** với bất kỳ từ khóa nào của `parser_v2.1`. Cho phép khớp mờ với `Damerau-Levenshtein distance == 1` đối với từ khóa dài $\ge 5$ ký tự trong lexicon catalog (tuyệt đối **không** áp dụng cho từ đơn âm ngắn hoặc danh sách `HOMOGRAPH_COLOUR_TOKENS` như `tim/do/cam/be/den/da/...`).
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Sẽ khôi phục được một phần lớn lỗi gõ trên `B-synth typo` (`Bt mAP` đang tụt từ `0.3391` xuống `0.2522`, chênh `-0.0869`), đưa `Bt mAP` tiệm cận `Bc mAP` mà **không** làm tăng FP trên `DEV / Val A / B-synth clean / B-human` (đạt ràng buộc `5(v)`).
  - *Bác bỏ:* Nếu gây dù chỉ 1 FP mới vượt `parser_v1` trên bất kỳ tập nào (vi phạm `5(v)`), hoặc cải thiện trên `B-synth typo` $< +0.01$ mAP.
- **(c) Lưới cấu hình tối đa:** `2` cấu hình (bật `fuzzy_typo=true` trên `cfg_13`, và kết hợp `I1/I2`).
- **(d) Chi phí độ trễ T2I ước lượng:** `+0.5` đến `+1.5 ms` (chỉ chạy khi có token ngoài từ điển trên câu ngắn ~5–10 từ). Tổng p95 `~55 ms`.
- **(e) Rủi ro:** Rủi ro tăng FP nếu ngưỡng độ dài `< 5` hoặc chạm homograph; bắt buộc chạy lại toàn bộ bảng kiểm toán `P2 (5v)` như Preflight.
- **(f) Tệp thêm/sửa & test:** Tạo `src/serve/parser_v2_fuzzy.py` bọc `parser_v2.1`; test unit riêng cho 4 loại lỗi gõ (thiếu ký tự, thừa ký tự, đảo ký tự, sai 1 ký tự) và test khẳng định các từ homograph ngắn không bao giờ bị khớp mờ.
- **(g) Tuân thủ luật:** Luật 7, ràng buộc `5(v)`.

### I8. GT phụ không vòng tròn (`usage` / `season`) — Chỉ số báo cáo song song
- **(a) Cơ chế:** Xây dựng bộ chỉ số đánh giá song song sử dụng thuộc tính `usage` / `season` (hoặc `masterCategory` / `subCategory` không nằm trong 3 slot parser `articleType/gender/baseColour`) cho các truy vấn có nêu ngữ cảnh sử dụng (ví dụ: thể thao, công sở, dự tiệc, đi biển, mùa hè/đông). Không dùng để chấm điểm chọn cấu hình thay thế GT chính, mà làm thước đo chẩn đoán tách **"hệ thống khớp metadata 3 slot"** khỏi **"model hiểu ngữ nghĩa ảnh"**.
- **(b) Giả thuyết & cách bác bỏ:**
  - *Giả thuyết:* Nếu `I5` (text–ảnh) hoặc `I3` (đa mẫu) thực sự cải thiện hiểu ngữ nghĩa ngoài 3 slot parser, điểm trên GT phụ `usage/season` cũng phải tăng; ngược lại nếu chỉ tăng trên GT 3 slot mà giữ nguyên/giảm trên `usage/season`, phần tăng đó chủ yếu là khớp metadata.
  - *Bác bỏ:* Nếu số lượng sản phẩm có nhãn `usage`/`season` sạch trong `catalog_rebuild.parquet` quá ít hoặc độ nhiễu nhãn gốc cao (cần kiểm kê trước khi dùng).
- **(c) Lưới cấu hình tối đa:** `0` cấu hình (đây là thước đo báo cáo song song, không phải tham số mô hình).
- **(d) Chi phí độ trễ T2I ước lượng:** `0 ms` (chỉ tính offline khi đánh giá).
- **(e) Rủi ro:** Không gây rò rỉ; giúp **giảm** rủi ro tự đánh lừa bởi tính vòng tròn của GT 3 slot.
- **(f) Tệp thêm/sửa & test:** Script `scripts/eval_noncircular_gt.py` + kiểm tra cột `usage`/`season` trong `catalog_rebuild.parquet`.
- **(g) Tuân thủ luật:** Luật 7, C.5, C.10 (không thay đổi GT chính thức khi báo cáo chỉ tiêu đề cương).

---

## 3. Xếp hạng Ưu tiên (Kỳ vọng Cải thiện / Rủi ro)

| Hạng | Ý tưởng | Kỳ vọng tác động | Rủi ro (rò rỉ / FP / độ trễ / hubness) | Ghi chú |
|---:|---|---|---|---|
| **1** | **I1** (Bonus theo $\sigma_q$ / tier) | Rất cao (khắc phục nút thắt z-score ở `v3`) | Thấp (độ trễ `+0.5 ms`, không tăng FP) | Giải quyết trực tiếp cảnh báo biên $\gamma=1.5$ |
| **2** | **I2** (Gating theo độ tin cậy parser) | Trung bình–Cao (giảm `CatDrop` khi $\gamma$ lớn) | Thấp (bọc ngoài `parser_v2.1`, độ trễ `+0.1 ms`) | Bổ trợ trực tiếp cho I1 |
| **3** | **I7** (Parser chịu lỗi gõ có bảo vệ) | Cao trên `B-synth typo` (`Bt mAP`) | Trung bình (phải qua cổng `5(v)` nghiêm ngặt) | Thu hẹp khoảng cách `Bt` vs `Bc` |
| **4** | **I5** (Hợp nhất text–ảnh gallery) | Trung bình–Cao trên `mAP` | Trung bình–Cao (vòng tròn GT nếu dùng metadata; phải tách Luật 5) | Cần I8 để kiểm soát vòng tròn |
| **5** | **I6** (Hubness + CSLS) | Trung bình | Thấp–Trung bình (`+0.3 ms`; cần anchor pool sạch) | Theo đúng C.8 |
| **6** | **I3** (Mở rộng text đa mẫu batch) | Trung bình | Trung bình–Cao (rủi ro vượt `+20 ms` p95) | Chỉ thử nếu batch forward đạt `5(iv)` |
| **7** | **I4** (Rocchio PRF) | Không chắc chắn | Cao (dễ trôi chủ đề khi `R@1` model thuần thấp, tăng hubness) | Thử cuối hoặc loại nếu `CatDrop > 5%` |
| **—** | **I8** (GT phụ `usage/season`) | Thước đo chẩn đoán (không tối ưu) | Rất thấp | Chạy song song để kiểm chứng I3/I5 |

---

## 4. Thiết kế Đánh giá cho `selection_rule_v4` (Đề xuất, chờ duyệt)

1. **Lưới `v4` tường minh ($\le 24$ cấu hình, đề xuất `18` cấu hình):**
   - `cfg_v4_01`: Baseline kế thừa từ `v3` (`cfg_13`: `parser_v2.1`, $\gamma=1.5$, `query_rewrite=true`, `zscore=true`).
   - **Ablation từng ý tưởng riêng trên nền `cfg_13`:**
     - `I1` (4 cấu hình): `sigma_scaled` ($\gamma_{\text{rel}} \in \{1.0, 1.5, 2.0\}$) + `tier_soft`.
     - `I2` (2 cấu hình): `parser_conf_gating` ($\gamma_{\text{rel}} \in \{1.5, 2.0\}$).
     - `I7` (1 cấu hình): `fuzzy_typo` trên `cfg_13`.
     - `I6` (2 cấu hình): `CSLS` ($k \in \{10, 20\}$) trên `cfg_13`.
     - `I5` (3 cấu hình): `text_gallery` (`internvl2_only` $\alpha \in \{0.7, 0.85\}$, và `internvl2_plus_auxiliary` tách riêng theo Luật 5).
     - `I3` (2 cấu hình): `multi_template_batch` (`2` mẫu, `3` mẫu).
     - `I4` (1 cấu hình): `Rocchio` ($k=3, \beta=0.2$).
   - **Tổ hợp có kiểm soát (tối đa `2` cấu hình):** chỉ ghép các thành phần đơn lẻ đã vượt `cfg_13` và thỏa `5(i)–(v)` (ví dụ `I1 + I2 + I7`, và nếu I5/I6 đạt thì thêm 1 tổ hợp).
   - **Thứ tự đơn giản (`simplicity_order` v4):**
     `không PRF/CSLS < có CSLS < có PRF` → `không text-gallery < có text-gallery` → `query_rewrite 1 câu < multi-template` → `không fuzzy < fuzzy` → `không conf-gating < conf-gating` → `zscore/sigma nhỏ hơn < lớn hơn`.
2. **Giao thức & Ràng buộc:** Giữ nguyên cấu trúc `v3`:
   - `5-fold CV` theo `spec_cluster_id` trên `A + B-synth clean` (dùng lại `folds.json` đã khóa hoặc nested CV nếu số cấu hình vượt 24).
   - Đầy đủ ràng buộc `5(i)–(v)` (`C 20%`, `CatDrop(B) <= 5%`, `ΔR@10(AB) >= -0.01`, `T2I p95 < 200 ms` và tăng `<= 20 ms` so với `cfg_01` `35.9 ms`, `parser FP <= v1`).
   - Phá hòa bằng paired cluster bootstrap 1.000 lần theo `simplicity_order`; báo song song **Tốt nhất theo Score** và **An toàn**.
3. **Xử lý *Winner's Curse* (Tập xác nhận MỚI):**
   - Ở `v3`, các tập `Val A`, `Val B-synth` và `B-human` đều đã được quan sát (riêng `parser_v2.1` đã sửa sau audit Tập A, và `B-human` toàn bộ do AI soạn với `61/65` cụm đơn lẻ).
   - **Đề xuất bắt buộc cho `v4`:** Người dùng tự soạn một tập xác nhận **MỚI (`B-human-v4`)** (tối thiểu `60–80` câu, có đa dạng cụm $\ge 2$ câu/cụm càng tốt, khóa `SHA-256` trước khi chạy `v4`). Cổng xác nhận `v4` chỉ chạy trên tập mới này sau khi đã khóa cấu hình thắng từ lưới `v4`.
   - **Cam kết minh bạch:** Nếu chưa có tập xác nhận mới, mọi kết luận của `v4` trên `B-human` cũ chỉ được gắn nhãn **"tham khảo (đã quan sát ở v3), không phải kiểm chứng độc lập"**.
4. **Quy ước báo cáo điểm & CI (3 mức):**
   - Mọi so sánh chính đều kèm điểm ước lượng + 95% cluster bootstrap CI (1.000 lần) và kết luận theo 3 mức:
     - **Đạt chắc chắn:** điểm ước lượng đạt ngưỡng và toàn bộ 95% CI nằm phía đạt ngưỡng.
     - **Đạt điểm số nhưng CI chứa ngưỡng:** điểm ước lượng đạt nhưng cận dưới/trên CI cắt qua ngưỡng.
     - **Chưa đạt:** điểm ước lượng không đạt ngưỡng.
5. **Tách 3 hàng & cột bắt buộc trong bảng kết quả `v4`:**
   - Tách riêng 3 hàng: **`Model thuần`** / **`Hệ thống soft`** / **`Hệ thống gated`**.
   - Bắt buộc có các cột: `CatDrop(B)`, `Parser FP (q / pred)`, `T2I p95 (ms)` + `Δp95`, và tách riêng `InternVL2-2B` vs bổ trợ (Luật 5).

---

## 5. Giới hạn Trung thực (Khoảng cách tới R@10 ≥ 0.85 và mAP ≥ 0.75)

1. **Phần do Model thuần (`best_model.pt`):**
   - Trên `Val B-synth clean` (`index/val_gallery/`), `model_only` chỉ đạt `R@10 = 0.2485` và `mAP = 0.1071` (`B-human` cluster `mAP = 0.2680`). Ngay cả khi parser đúng 100% và gating mạnh, khi có nhiều sản phẩm cùng khớp 3/3 thuộc tính (`n_relevant >= 5..30`), thứ tự xếp hạng **bên trong** nhóm đó phụ thuộc vào biểu diễn thị giác–ngôn ngữ của checkpoint. Do `UpgradedAttentionCIRCombiner` / checkpoint cũ chưa được tối ưu cho truy vấn tiếng Việt ngắn/không dấu/typo, khả năng đạt **`mAP >= 0.75`** chỉ bằng tầng suy luận là rất thấp nếu không cải thiện được xếp hạng sâu.
2. **Phần do Parser:**
   - Parser dựa trên từ điển không tránh khỏi thiếu từ đồng nghĩa mới hoặc lỗi gõ nặng (`Bt mAP = 0.2522` vs `Bc = 0.3391`), và có giới hạn ở các khái niệm đa nghĩa (`"xanh rêu"` vs `Olive`).
3. **Phần do Định nghĩa GT:**
   - GT hybrid 3 slot coi mọi sản phẩm khớp `(articleType, gender, baseColour)` đều đúng như nhau, trong khi người dùng có thể kỳ vọng thêm kiểu dáng/chất liệu (hoặc ngược lại, `spec` ẩn làm `model_only`/`system` bị phạt ở các câu không nêu màu/giới).
4. **Lựa chọn nếu tầng suy luận không đủ đạt chỉ tiêu (Luật 1):**
   - Tuân thủ **Luật 1**, agent **KHÔNG** tự đề xuất bắt đầu huấn luyện và **KHÔNG** hạ chuẩn (`R@10 >= 0.85`, `mAP >= 0.75`, `p95 < 200 ms`). Nếu sau khi thử nghiệm `v4` (hoặc đánh giá Bước 5 trên `test_gallery` ở `v3`) các chỉ tiêu vẫn chưa đạt, báo cáo sẽ ghi rõ **"cần train / fine-tune"** như một lựa chọn để **NGƯỜI DÙNG quyết định**.

---

## 6. Kế hoạch Triển khai & Kiểm thử (chờ duyệt trước khi thực hiện)

1. **Điểm dừng hiện tại:** Chờ người dùng duyệt bản đề xuất này (chọn các ý tưởng I1–I8 muốn đưa vào `v4`, xác nhận có tạo tập xác nhận mới `B-human-v4` hay không).
2. **Sau khi được duyệt:**
   - Soạn & khóa `reports/phase5c/selection_rule_v4.json` + `selection_rule_v4.md` (commit + `SHA-256` + tag `phase5c-rule-v4-frozen`) **TRƯỚC** khi viết/chạy bất kỳ thực nghiệm nào.
   - Xây dựng các module/test tối thiểu cho từng ý tưởng được duyệt (luôn giữ `parser_v1.py` và `parser_v2.py` `v2.1` nguyên vẹn), chạy `pytest` xanh 100% trước khi chạy lưới `v4`.
