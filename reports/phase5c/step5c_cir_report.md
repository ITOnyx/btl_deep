# Báo cáo Đánh giá & Chọn Cấu hình CIR — Pha 5c (Prompt 13 & Cập nhật Quyết định Duyệt Tạm)

- **Rule SHA-256 (`selection_rule_cir_v1.json` — giữ nguyên, không đổi):** `4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067`
- **Tập tuning (`eval/cir_eval_val.json` — trên `index/val_gallery`, `N = 1,884`):** `120` cặp (`80` đổi màu, `40` đổi giới; `SHA-256: c735c7742c5277cffc4113c68dd8ee59fd3475c742284c8fe9e098af0819062a`)
- **Kết quả JSON (`reports/phase5c/step5c_cir_results.json`):** `SHA-256: a4448834f69f65687e8e6c58b233594c67d9c2d465bdb24d2f91d1647b7136ab`
- **Quyết định cấu hình CIR sau Prompt 13:**
  - **Duyệt tạm (`provisional winner` theo `mAP`):** **`cir_v1_05`** (`c3_mod_bonus_only`: `c1_vi_to_en=False`, `beta=1.0`, `c3_mod_bonus=True`, `gamma=1.0`, `c4_rocchio=False`; `mAP = 0.2902`, `ΔmAP vs cir_v1_01 = +0.2114`, 95% CI: `[+0.1615, +0.2684]`).
  - **Nhóm ứng viên hòa trong 95% cluster bootstrap CI:** `['cir_v1_05', 'cir_v1_06', 'cir_v1_08']` (chọn `cir_v1_05` theo `simplicity_order`).
  - **Điều kiện ràng buộc độ trễ:** Trong lần chạy Prompt 13, **không cấu hình nào đạt ràng buộc `p95 < 200 ms` cho luồng `cir_upload`** (`cir_v1_01`: `415.29 ms`; `cir_v1_05`: `446.27 ms` — đo trong điều kiện vi phạm protocol do có tiến trình đồ họa nền). Vì vậy, **lựa chọn chính thức phụ thuộc vào lần đo lại độ trễ trên GPU sạch/nguội ở Prompt 14**. Không thay đổi `selection_rule_cir_v1.json`.

---

## 1. Tách riêng "Combiner thuần" (`cir_v1_01`) và "Hệ thống" (`cir_v1_05`) trên `val_gallery` (`N = 1,884`)

| Phân loại (Luật C.4) | Config ID | Cơ chế suy luận | R@1 | R@5 | R@10 | mAP | 95% CI (mAP) | mAP Đổi màu (`n=80`) | mAP Đổi giới (`n=40`) | mAP Có dấu (`n=60`) | mAP Không dấu (`n=60`) |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **Combiner thuần (Model-only)** | `cir_v1_01` | `UpgradedAttentionCIRCombiner(ref_img, mod_txt_vi)` + cosine | 0.0750 | 0.1917 | 0.3083 | **0.0910** | `[0.0667, 0.1222]` | 0.0902 | 0.0927 | 0.0803 | 0.1017 |
| **Hệ thống (System — Duyệt tạm)** | `cir_v1_05` | Combiner thuần + `C3` (`parser_v2.1` `tier_soft γ=1.0` trên modifier) | 0.2583 | 0.4833 | 0.6250 | **0.2902** | `[0.2441, 0.3663]` | 0.3634 | 0.1437 | 0.2747 | 0.3057 |
| **Độ lệch (`Hệ thống - Thuần`)** | `Δ` | Đóng góp thuần của tầng suy luận `C3` | **+0.1833** | **+0.2916** | **+0.3167** | **+0.2114** | `[+0.1615, +0.2684]` | **+0.2732** | **+0.0510** | **+0.1944** | **+0.2040** |

> [!IMPORTANT]
> **Hai lưu ý phương pháp luận bắt buộc khi diễn giải bảng trên:**
> 1. **Tính tuần hoàn một phần của Ground Truth (`Catalog-Attribute Circularity`):** Tập `eval/cir_eval_val.json` được sinh tự động từ thuộc tính catalog (`target_attrs = {articleType, gender, baseColour}`), trong khi mô-đun `C3` (`cir_v1_05`) cũng chấm điểm khớp thuộc tính catalog (`baseColour` hoặc `gender`) trích từ câu modifier. Mặc dù `C3` chỉ biết 1 thuộc tính thay đổi từ câu modifier (không biết `articleType` và thuộc tính giữ nguyên của ảnh tham chiếu — phần này vẫn do `combiner` xếp hạng), việc cả GT lẫn `C3` cùng dựa trên nhãn thuộc tính catalog tạo ra lợi thế cấu trúc cho `C3` so với truy vấn tự nhiên ngoài đời thực.
> 2. **Phạm vi đánh giá (`val_gallery`):** Mọi số liệu trong báo cáo này được đo trên `index/val_gallery` (`1,884` ảnh, tập tuning). Chỉ tiêu chính thức trên `index/test_gallery` (`1,290` ảnh) với `eval/cir_eval_test.json` chỉ được chạy đúng 1 lần ở **Prompt 14**.

---

## 2. Bảng Kết quả Đầy đủ Lưới 8 Cấu hình trên `CIR-eval val` (`120` cặp)

| Config ID | Vai trò | R@1 | R@5 | R@10 | mAP | 95% CI (mAP) | ΔmAP vs `cir_v1_01` (95% CI) | mAP Màu (`n=80`) | mAP Giới (`n=40`) | mAP Có dấu (`n=60`) | mAP Không dấu (`n=60`) |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `cir_v1_01` | `baseline_combiner_vi` | 0.0750 | 0.1917 | 0.3083 | **0.0910** | `[0.0667, 0.1222]` | `+0.0000 [+0.0000, +0.0000]` | 0.0902 | 0.0927 | 0.0803 | 0.1017 |
| `cir_v1_02` | `c1_vi_to_en_only` | 0.0333 | 0.0917 | 0.2167 | **0.0582** | `[0.0426, 0.0773]` | `-0.0335 [-0.0541, -0.0169]` | 0.0570 | 0.0608 | 0.0433 | 0.0731 |
| `cir_v1_03` | `c2_text_fusion_b085` | 0.0417 | 0.1417 | 0.1833 | **0.0654** | `[0.0405, 0.0910]` | `-0.0277 [-0.0462, -0.0123]` | 0.0619 | 0.0723 | 0.0583 | 0.0724 |
| `cir_v1_04` | `c2_text_fusion_b070` | 0.0000 | 0.0417 | 0.0833 | **0.0253** | `[0.0174, 0.0321]` | `-0.0679 [-0.0960, -0.0454]` | 0.0218 | 0.0324 | 0.0254 | 0.0253 |
| `cir_v1_05` | `c3_mod_bonus_only` | 0.2583 | 0.4833 | 0.6250 | **0.2902** | `[0.2441, 0.3663]` | `+0.2114 [+0.1615, +0.2684]` | 0.3634 | 0.1437 | 0.2747 | 0.3057 |
| `cir_v1_06` | `c1_plus_c3` | 0.2250 | 0.4833 | 0.5750 | **0.2788** | `[0.2392, 0.3702]` | `+0.2095 [+0.1517, +0.2713]` | 0.3462 | 0.1440 | 0.2675 | 0.2901 |
| `cir_v1_07` | `c1_c2_b085_c3` | 0.1167 | 0.3000 | 0.4000 | **0.1768** | `[0.1372, 0.2348]` | `+0.0896 [+0.0503, +0.1344]` | 0.2216 | 0.0873 | 0.1836 | 0.1701 |
| `cir_v1_08` | `c1_c2_b085_c3_c4_rocchio` | 0.2250 | 0.3917 | 0.4750 | **0.2664** | `[0.2224, 0.3604]` | `+0.1953 [+0.1358, +0.2612]` | 0.3367 | 0.1260 | 0.2732 | 0.2597 |

### Chẩn đoán Toàn bộ 120 cặp Có dấu (`all_full`) vs 120 cặp Không dấu (`all_none`)

| Config ID | `all_full (n=120)` R@1 | `all_full` R@10 | `all_full` mAP | `all_none (n=120)` R@1 | `all_none` R@10 | `all_none` mAP |
|---|---:|---:|---:|---:|---:|---:|
| `cir_v1_01` | 0.0583 | 0.3083 | 0.0989 | 0.0583 | 0.3667 | 0.0961 |
| `cir_v1_02` | 0.0167 | 0.2083 | 0.0548 | 0.0333 | 0.1750 | 0.0541 |
| `cir_v1_03` | 0.0333 | 0.2000 | 0.0673 | 0.0417 | 0.1833 | 0.0642 |
| `cir_v1_04` | 0.0000 | 0.0833 | 0.0268 | 0.0000 | 0.0667 | 0.0246 |
| `cir_v1_05` | 0.2333 | 0.6083 | 0.3019 | 0.2667 | 0.6167 | 0.3003 |
| `cir_v1_06` | 0.2167 | 0.5667 | 0.2710 | 0.1917 | 0.5833 | 0.2657 |
| `cir_v1_07` | 0.1167 | 0.3833 | 0.1766 | 0.1083 | 0.3917 | 0.1716 |
| `cir_v1_08` | 0.2250 | 0.5083 | 0.2707 | 0.1833 | 0.4333 | 0.2463 |

---

## 3. Đối chiếu Premise F.2 với `notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb` (Dòng 970–985)

Theo yêu cầu mục 3, đối chiếu trực tiếp mã nguồn tại [siglip_vietnamese_ecommerce_colab_ver2.ipynb](file:///d:/Kì%207/Hoc_sau_ver4/notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb#L970-L985) (thuộc lớp `ECommerceMultimodalDataset`, Cell 13):

### 3.1. Nguyên văn hàm `_synthesize_cir_modifier` (dòng 970–985)
```python
    def _synthesize_cir_modifier(self, ref_item: dict, tar_item: dict) -> str:
        ref_attrs = ref_item.get("attributes") or {}
        tar_attrs = tar_item.get("attributes") or {}
        changes = []
        for attr in ["color", "pattern", "sleeve", "material", "collar", "style"]:
            v_ref = ref_attrs.get(attr)
            v_tar = tar_attrs.get(attr)
            if v_tar and v_tar != v_ref:
                changes.append(v_tar)
        if changes:
            if len(changes) == 1:
                tmpls = ["đổi sang {v}", "thay bằng {v}", "chuyển thành {v}", "kiểu này nhưng là {v}"]
                return self.rng.choice(tmpls).format(v=changes[0])
            tmpls = ["kiểu này nhưng đổi sang {c}", "giống mẫu này nhưng chuyển sang {c}", "phiên bản có {c}"]
            return self.rng.choice(tmpls).format(c=", ".join(changes))
        return f"phiên bản tương tự phong cách {tar_item.get('title', '')[:40]}"
```

### 3.2. Phân tích ngôn ngữ mẫu câu, giá trị thuộc tính và trạng thái huấn luyện thực tế
1. **Tên hàm và các mẫu câu (`tmpls`):**
   - Hàm `_synthesize_cir_modifier(self, ref_item: dict, tar_item: dict) -> str` dùng **100% khung câu tiếng Việt**:
     - Khi đổi 1 thuộc tính (`len(changes) == 1`): `"đổi sang {v}"`, `"thay bằng {v}"`, `"chuyển thành {v}"`, `"kiểu này nhưng là {v}"`.
     - Khi đổi từ 2 thuộc tính trở lên (`len(changes) >= 2`): `"kiểu này nhưng đổi sang {c}"`, `"giống mẫu này nhưng chuyển sang {c}"`, `"phiên bản có {c}"`.
     - Khi không có thuộc tính khác biệt (`fallback`): `"phiên bản tương tự phong cách {tar_item.get('title', '')[:40]}"` (`title` đã được Việt hóa bởi `clean_record_anti_overfit`, dòng 481–500).
2. **Ngôn ngữ của giá trị `{v}` / `{c}` trong `item.get("attributes")` và tỷ lệ nguồn tập train:**
   - Theo log kiểm kê tập train (`153,126` mẫu, Cell 12 dòng 686), phân bố 3 nguồn dữ liệu là:
     - `shop100k`: `100,713` mẫu (**`65.77%`**)
     - `myntra`: `41,441` mẫu (**`27.06%`**)
     - `tiki`: `10,972` mẫu (**`7.17%`**)
   - Trong `clean_record_anti_overfit` (dòng 463–500), các quy tắc dịch `EN_VI_MAP` (`"men"` $\to$ `"nam"`, `"women"` $\to$ `"nữ"`, `"striped"` $\to$ `"kẻ sọc"`, `"checked"` $\to$ `"kẻ caro"`, `"solid"` $\to$ `"trơn"`, `"slim fit"` $\to$ `"dáng ôm"`) chỉ áp dụng lên trường `title`, **không** ghi đè vào dict `item["attributes"]`. Do tệp `train_clean.jsonl` nằm trên Google Drive (`/content/drive/MyDrive/Hoc_sau_ver4/data/train_clean.jsonl`) và không lưu trong repo cục bộ, tỷ lệ chính xác bao nhiêu phần trăm giá trị `{v}` bên trong `item["attributes"]` của `shop100k` + `myntra` (`92.83%` tập train) còn giữ nguyên tiếng Anh (ví dụ `"Black"`, `"Solid"`, `"Long Sleeves"`) so với đã Việt hóa trước khi tạo `train.jsonl` là **không xác minh trực tiếp được từ đĩa cục bộ**. Tuy nhiên, kể cả khi `{v}` chứa từ khóa thuộc tính tiếng Anh từ `shop100k`/`myntra`, câu modifier sinh ra là **câu pha trộn khung tiếng Việt + giá trị `{v}`** (không phải câu tiếng Anh thuần túy).
3. **Phát hiện quan trọng từ mã huấn luyện thực tế trong Notebook và trọng số `best_model.pt`:**
   - Trong `notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb`, lớp `ECommerceMultimodalDataset` (Cell 13) mặc định đặt `is_cir_mode: bool = False`, còn vòng lặp huấn luyện chính thức ở Cell 32 (dòng 2809) sử dụng `OptimizedTextToImageDataset` (chỉ trả về `image` và `text`, không sinh `modifier_text`) và `MultimodalTrainer.train_epoch` (Cell 27 dòng 2255–2276) chỉ gọi `img_emb, txt_emb = self.model(pixel_values, input_ids, attention_mask)` mà **không gọi `self.model.cir_combiner`**.
   - Kiểm tra trực tiếp trọng số `cir_combiner` trong `models/best_model.pt` xác nhận: `norm_att.weight` bằng đúng `1.000000` (`std = 0.000000`), `norm_att.bias = 0.000000`, `norm_final.weight = 1.000000` (`std = 0.000000`), `norm_final.bias = 0.000000`, và cổng `gate` luôn xuất `alpha ≈ 0.5062` (`sigmoid(0.0208)`). Điều này chứng minh `UpgradedAttentionCIRCombiner` trong `best_model.pt` chưa từng được cập nhật bởi hàm loss CIR chuyên biệt; khả năng truy vấn của combiner thuần (`mAP = 0.0910`) hoàn toàn đến từ đường cộng tắt (residual connection) `alpha * text_feat_norm + (1 - alpha) * img_feat_norm` (dòng 1443), trong khi Text Tower của SigLIP đã được fine-tune sâu bằng tiếng Việt. Đó là lý do thực sự khiến `C1` (dịch câu modifier sang tiếng Anh) làm giảm `mAP` từ `0.0910` xuống `0.0582`.
   - *(Tuân thủ chỉ đạo: Agent chỉ ghi nhận nguyên văn bằng chứng vào báo cáo này, không tự sửa `AGENTS.md` hay `config.yaml`).*

---

## 4. Phân tích Chuyên sâu Truy vấn Đổi giới (`mAP = 0.1437`, `n=40`) vs Đổi màu (`mAP = 0.3634`, `n=80`) ở `cir_v1_05`

Trên tập `eval/cir_eval_val.json` (`120` cặp), `cir_v1_05` (`C3`) nâng mạnh `mAP` của nhóm **Đổi màu** (`n=80`) từ `0.0902` lên **`0.3634`** (`+0.2732`), nhưng nhóm **Đổi giới** (`n=40`: `22 Men`, `14 Women`, `3 Unisex`, `1 Girls`) chỉ tăng từ `0.0927` lên **`0.1437`** (`+0.0510`). Kết quả chẩn đoán định lượng trên `index/val_gallery` tách bạch 3 nguyên nhân gốc rễ sau:

| Cấu hình chẩn đoán trên 40 cặp Đổi giới (`all_full`) | Chính sách `gender` của `C3` | Vector xếp hạng bên trong tầng `mf = 1.0` | Kích thước trung bình tầng `mf = 1.0` (`/ 1,884`) | R@1 | R@10 | mAP | Số mục Top-10 sai `articleType` | Số mục Top-10 sai `baseColour` | Số mục Top-10 lệch `gender` GT (`Unisex`/khác) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Đổi màu (n=80)` — tham chiếu | `hybrid` | `cir_combiner(ref_img, mod_txt)` | `219.7` (`11.66%`) | 0.3125 | 0.7500 | **0.3870** | `6.95 / 10` | `0.00 / 10` | `2.79 / 10` |
| `Đổi giới (n=40)` — `cir_v1_05` hiện tại | `hybrid` | `cir_combiner(ref_img, mod_txt)` | `902.3` (`47.89%`) | 0.0750 | 0.3250 | **0.1324** | `7.25 / 10` | `7.88 / 10` | `0.78 / 10` |
| `Đổi giới (n=40)` — đổi sang `strict` gender | `strict` (khớp tuyệt đối) | `cir_combiner(ref_img, mod_txt)` | `805.2` (`42.74%`) | 0.1250 | 0.3250 | **0.1580** | `7.25 / 10` | `7.95 / 10` | `0.00 / 10` |
| `Đổi giới (n=40)` — xếp hạng bằng `ref_img` (`hybrid`) | `hybrid` | `ref_img` (`cos(ref_img, gal_img)`) | `902.3` (`47.89%`) | 0.4500 | 0.8000 | **0.3988** | `2.55 / 10` | `5.60 / 10` | `1.65 / 10` |
| `Đổi giới (n=40)` — xếp hạng bằng `ref_img` (`strict`) | `strict` (khớp tuyệt đối) | `ref_img` (`cos(ref_img, gal_img)`) | `805.2` (`42.74%`) | 0.5500 | 0.8500 | **0.4376** | `2.25 / 10` | `5.55 / 10` | `0.00 / 10` |

### Ba nguyên nhân định lượng:
1. **Kiểm tra lỗi `parser_v2.1` và khôi phục dấu trên `"nam nữ"` / `"mau danh cho..."`:**
   - Trên cả 3 chế độ (`primary`, `all_full`, `all_none`), `VietnameseQueryParserV2Conf` (`parser_v2.1`) nhận diện **chính xác `40/40 (100%)` nhãn `gender` mục tiêu** (kể cả 3/3 cặp `"mẫu dành cho nam nữ"` / `"doi sang cho nam nu"` $\to$ `Unisex`, và `"phiên bản dành cho bé gái"` $\to$ `Girls`), đồng thời **`0/40` lần trích nhầm `baseColour` hay `articleType`**. Như vậy, `parser_v2.1` **không** làm rơi bất kỳ nhãn giới tính nào ở bước chấm điểm `C3`.
   - Tuy nhiên, ở bước khôi phục dấu `restore_diacritics_for_query` trên các câu không dấu bắt đầu bằng `"mau danh cho..."` (`6/40` câu ở `primary`, `16/40` câu ở `all_none`), từ `"mau"` (mẫu) bị khôi phục thành `"màu"` (`"màu danh cho nam"`, `"màu danh cho nữ"`, `"màu danh cho nam nữ"`). Dù `parser_v2.1` không bị ảnh hưởng (vì `"màu"` đứng một mình không phải tên màu), chuỗi có từ `"màu"` đi vào `encode_text` làm nhiễu nhẹ biểu diễn văn bản của combiner (và ở `C1` bị dịch nhầm thành `"color"`).
2. **Độ rộng tầng ứng viên `mf = 1.0` (`47.89%` gallery) và độ lệch giữa chính sách `gender_policy="hybrid"` với GT khớp tuyệt đối:**
   - Khi đổi màu (`baseColour`), `C3` lọc tầng đỉnh (`mf = 1.0`) xuống trung bình chỉ còn **`219.7 / 1,884` sản phẩm (`11.66%` gallery)**. Ngược lại, khi đổi giới (`gender`), vì chỉ có 5 nhãn giới tính (trong đó `Men` và `Women` chiếm đa số kho hàng), tầng đỉnh (`mf = 1.0`) rộng tới **`902.3 / 1,884` sản phẩm (`47.89%` gallery — rộng gấp `4.11×` so với đổi màu)**.
   - Bên cạnh đó, GT của `eval/cir_eval_val.json` yêu cầu khớp tuyệt đối `gender == target_attr_value` (ví dụ `target = Men` thì các mục `Unisex` không thuộc GT), trong khi `C3` dùng `gender_policy="hybrid"` (cho phép các mục `Unisex` ở ngành hàng không nghiêm ngặt nhận trọn `mf = 1.0`). Điều này đẩy thêm trung bình `97.1` mục `Unisex` phi-GT vào tầng `mf = 1.0` (chiếm `0.78/10` vị trí Top-10). Khi thử thay `hybrid` bằng `strict` (khớp giới tính tuyệt đối), `mAP` Đổi giới tăng từ `0.1324` lên **`0.1580`** (`+0.0256`, `R@1` tăng từ `0.0750` lên `0.1250`).
3. **Suy giảm bảo toàn đặc trưng thị giác (`articleType` + `baseColour`) của `UpgradedAttentionCIRCombiner` bên trong tầng `mf = 1.0` (Nguyên nhân quyết định):**
   - Trong tầng `902.3` sản phẩm cùng giới tính (`mf = 1.0`), mọi sản phẩm đều nhận cùng mức thưởng `+1.0`, nên thứ tự xếp hạng hoàn toàn do `cos(cir_combiner(ref_img, mod_txt), gal_img)` quyết định để giữ lại **đồng thời 2 thuộc tính của ảnh gốc** (`articleType` và `baseColour`).
   - Do nhánh `fused` (Cross-Attention + MLP) của `UpgradedAttentionCIRCombiner` trong `best_model.pt` chưa được huấn luyện (và ngay trong hàm `_synthesize_cir_modifier` lúc thiết kế cũng không có khóa `"gender"` trong 6 thuộc tính `["color", "pattern", "sleeve", "material", "collar", "style"]`), vector đầu ra `out` của combiner bị nhiễu mạnh, chỉ còn tương quan rất yếu với ảnh gốc (`mean cos(out, ref_img) = 0.0769`). Hệ quả là trong Top-10 của `cir_v1_05` trên nhóm Đổi giới, có trung bình tới **`7.25 / 10` sản phẩm sai `articleType`** và **`7.88 / 10` sản phẩm sai `baseColour`**!
   - **Đối chứng định lượng:** Nếu giữ nguyên điểm thưởng `C3` nhưng chỉ dùng trực tiếp vector ảnh tham chiếu `ref_img` (`cos(ref_img, gal_img)`) để xếp hạng bên trong tầng `mf = 1.0`, số mục sai `articleType` trong Top-10 giảm gần 3 lần (từ `7.25` xuống **`2.55 / 10`**) và `mAP` Đổi giới tăng vọt hơn **3 lần**: từ **`0.1324` lên `0.3988`** (`R@1 = 0.4500`, `R@10 = 0.8000` với `hybrid`; và đạt **`mAP = 0.4376, R@1 = 0.5500, R@10 = 0.8500`** với `strict`). Theo đúng quy tắc khóa lưới, phát hiện này chỉ ghi nhận để giải thích bản chất mô hình, **không** dùng để sửa lại lưới `selection_rule_cir_v1`.

---

## 5. Độ trễ C.9 trên `val_gallery` (Vi phạm Protocol — KHÔNG dùng làm số chính thức) & Giải thích Chậm `2.5–3.3×` so với Pha 5

| Luồng | Config ID | p50 (ms) | p95 median ± IQR (ms) | p99 (ms) | Pha 5 (`test_gallery`, `p95` ms) | Tỷ lệ chậm đi (`Prompt 13 / Pha 5`) | Trạng thái số liệu |
|---|---|---:|---:|---:|---:|---:|---|
| `cir_catalog` | `cir_v1_01` | 72.86 | `121.42 ± 9.34` | 215.70 | `36.98` | `3.28×` | ⚠️ **Vi phạm protocol GPU (Không dùng chính thức)** |
| `cir_catalog` | `cir_v1_05` | 77.56 | `135.54 ± 16.17` | 215.14 | — | `3.67×` | ⚠️ **Vi phạm protocol GPU (Không dùng chính thức)** |
| `cir_upload` | `cir_v1_01` | 267.19 | `415.29 ± 22.77` | 505.96 | `165.49` | `2.51×` | ⚠️ **Vi phạm protocol GPU (Không dùng chính thức)** |
| `cir_upload` | `cir_v1_05` | 270.38 | `446.27 ± 21.82` | 535.76 | — | `2.70×` | ⚠️ **Vi phạm protocol GPU (Không dùng chính thức)** |

> [!WARNING]
> **Ghi nhận vi phạm protocol C.9 ở lần đo Prompt 13:**
> Trong suốt quá trình đo độ trễ ở Prompt 13, trên máy có các tiến trình đồ họa nền ngoài dự án (`RobloxPlayerBeta.exe`, `dnmultiplayerex.exe`) chiếm `~27–32%` tải GPU, làm nhiệt độ GPU RTX 2050 giữ ở mức **`85–86°C`**, công suất bị ghìm ở **`8.27–8.33 W`** và xung nhịp đồ họa giảm xuống **`900–1425 MHz`**. Do đó, toàn bộ các con số độ trễ trên **KHÔNG được dùng làm số liệu chính thức**.

### 5.1. Giải thích định lượng vì sao toàn bộ luồng ở Prompt 13 chậm khoảng `2.5–3.3×` so với Pha 5 (`cir_catalog: 36.98 -> 121.42 ms`, `cir_upload: 165.49 -> 415.29 ms`)
Đối chiếu trực tiếp mã nguồn Pha 5 ([eval_latency.py](file:///d:/Kì%207/Hoc_sau_ver4/src/eval/eval_latency.py#L149-L218) và [latency_sla_results.json](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5/latency_sla_results.json)) với mã đo Prompt 13 ([run_step5c_cir_grid.py](file:///d:/Kì%207/Hoc_sau_ver4/scripts/run_step5c_cir_grid.py#L310-L346)) cho thấy sự chênh lệch `2.5–3.3×` đến từ **3 nguyên nhân cộng hưởng**:
1. **Thắt cổ chai nhiệt & ghìm xung nhịp GPU (`Thermal & Power Throttling` do tiến trình game nền):**
   - Ở Pha 5 (`latency_sla_results.json`), GPU bắt đầu ở `75.0°C` (`utilization = 9%`), không bị tiến trình 3D nền cạnh tranh nên xung nhịp boost duy trì cao.
   - Ở Prompt 13, `RobloxPlayerBeta.exe` và `dnmultiplayerex.exe` ghìm GPU ở ngưỡng nhiệt `85–86°C` (`~8.3 W`, xung nhịp tụt xuống `900–1425 MHz`), cộng với tranh chấp hàng đợi lệnh CUDA và băng thông PCIe trước mỗi lệnh `torch.cuda.synchronize()`, làm thời gian thực thi của mọi nhân CUDA (`encode_text`, `encode_image`, `cir_combiner`) tăng gấp `~2.2–2.8×`.
2. **Đọc đĩa + giải mã ảnh JPEG nằm bên trong vùng bấm giờ đối với `cir_upload`:**
   - Ở Pha 5 (`src/eval/eval_latency.py` dòng 154–161 và 207–213), 200 ảnh đã được mở từ đĩa và giải mã sang `PIL.Image` (`im.convert("RGB")`) lưu sẵn trong RAM (`pil_pool`) **trước khi bắt đầu vòng lặp đo**.
   - Ở Prompt 13 (`scripts/run_step5c_cir_grid.py` dòng 328–330), hàm `_run_cir_end_to_end` truyền đường dẫn chuỗi `item["ref_image_path"]` vào `preprocess_single_image`, khiến mỗi truy vấn `cir_upload` phải thực hiện thêm **đọc tệp từ đĩa + giải mã JPEG + chuyển `RGB`** ngay bên trong thời gian đo.
3. **Chấm điểm toàn bộ gallery (`N = 1,884`) + khôi phục dấu/parser thay vì `FAISS HNSW candidate_pool = 100` (`N = 1,290`):**
   - Ở Pha 5 (`src/serve/search_engine.py` dòng 169–198), `engine.search_cir` chỉ truy vấn đồ thị `FAISS HNSW` lấy `fetch_k = 100` ứng viên trên `test_gallery` (`1,290` mục) và chỉ chạy vòng lặp Python `compute_soft_attribute_match` trên đúng **`100` mục**.
   - Ở Prompt 13 (`scripts/run_step5c_cir_grid.py` dòng 312–346), mỗi truy vấn đều chạy thêm `restore_diacritics_for_query` + `parser.parse_with_confidence` và gọi `score_cir_single_query` chấm điểm trên **toàn bộ `1,884` mục của `val_gallery`** (khi bật `C3` ở `cir_v1_05`, vòng lặp Python `compute_weighted_attribute_match` chạy `1,884` lần/truy vấn — nhiều gấp **`18.84×`** so với `candidate_pool = 100` của Pha 5).

### 5.2. Giao thức Đo lại Độ trễ Chính thức tại Prompt 14
Tại **Prompt 14**, độ trễ CIR (`cir_v1_01` và `cir_v1_05`) sẽ được đo lại nghiêm ngặt theo đúng quy chuẩn đã khóa:
- **Môi trường GPU sạch & nguội:** Đóng toàn bộ tiến trình đồ họa/game nền; ghi nhận danh sách tiến trình GPU từ `nvidia-smi` trước và sau khi đo; đảm bảo nhiệt độ GPU `< 60°C` trước mỗi luồng (hoặc ghi nhận rõ vi phạm nếu phần cứng không hạ xuống dưới 60°C).
- **Thiết kế phép đo:** Đo xen kẽ giữa các cấu hình (`interleaved`), `>= 5 run × 200 truy vấn`, `warmup 20`, có `torch.cuda.synchronize()` quanh mỗi bước.
- **Chỉ số báo cáo:** `p50 / p95 / p99`, `trung vị ± IQR của p95` qua 5 run, **kèm bảng phân rã thời gian từng bước (`per-stage breakdown`) của `cir_upload`**:
  1. `Đọc đĩa & giải mã ảnh (disk read + RGB decode)`
  2. `Resize (224×224 BILINEAR) + SiglipProcessor`
  3. `Vision Tower (encode_image)`
  4. `Text Tower (restore_diacritics + parser_v2.1 + encode_text)`
  5. `Combiner (UpgradedAttentionCIRCombiner)`
  6. `HNSW / Cosine + Điểm thưởng thuộc tính (C3 bonus)`

---

## 6. Kiểm tra Trạng thái Khóa Dữ liệu Chuẩn bị cho Prompt 14 (Chưa chạy `test_gallery`)

| Tập đánh giá cho Prompt 14 | Đường dẫn | Trạng thái trên đĩa | SHA-256 đã khóa | Ghi chú |
|---|---|---|---|---|
| **CIR-eval test (`120` cặp: `80` màu, `40` giới)** | `eval/cir_eval_test.json` | ✅ **Đã có sẵn & đã khóa** | `d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b` | Kèm tệp `eval/cir_eval_test.sha256`; tuyệt đối chưa chạy trên `test_gallery`. |
| **Phiếu lấy mẫu `held-out v2`** | `reports/phase5c/heldout_v2_worksheet.csv` | ✅ **Đã có sẵn** | `f1da994630e522f52cf8580e0a6eb1bcbb77be7c00865d68c083ab102f8c709e` | `60` cụm mục tiêu trên `test_split` chờ điền câu truy vấn. |
| **Tập truy vấn `held-out v2` (`test_queries_heldout_v2`)** | `eval/test_queries_heldout_v2.json` | ⏳ **Chưa có trên đĩa** | *(Chờ người dùng cung cấp & chạy `check_query_set.py` khóa SHA-256)* | Cần có và khóa SHA-256 trước khi chạy đánh giá một lần ở Prompt 14. |
