# Quy tắc Chọn Cấu hình v4 (Khóa trước thực nghiệm Pha 5c)

- **Nhánh:** `phase7a` | **Seed:** `42` | **Tag khóa:** `phase5c-rule-v4-frozen`
- **Tệp JSON chuẩn:** [`reports/phase5c/selection_rule_v4.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5c/selection_rule_v4.json)

## 1. Lưới cấu hình tường minh (`18` cấu hình $\le 24$)
- `cfg_v4_01`: Baseline kế thừa từ `v3` (`cfg_13`: `parser_v2.1`, $\gamma=1.5$, `query_rewrite=true`, `zscore=true`).
- `cfg_v4_02..05`: Ablation **I1** (`sigma_scaled` $\gamma \in \{1.0, 1.5, 2.0\}$ và `tier_soft`).
- `cfg_v4_06..07`: Ablation **I2** (`conf_gating` + `sigma_scaled`, $\gamma \in \{1.5, 2.0\}$).
- `cfg_v4_08`: Ablation **I7** (`fuzzy_typo`, `Damerau-Levenshtein <= 1` cho từ $\ge 5$ ký tự, chặn homograph).
- `cfg_v4_09..10`: Ablation **I6** (`CSLS` $k \in \{10, 20\}$).
- `cfg_v4_11..13`: Ablation **I5** (hợp nhất text–ảnh gallery, tách riêng `internvl2_only` và `internvl2_plus_aux` theo Luật 5).
- `cfg_v4_14..15`: Ablation **I3** (`multi_template_batch` 2–3 mẫu chuẩn catalog).
- `cfg_v4_16`: Ablation **I4** (`Rocchio PRF` $k=3, \beta=0.2$).
- `cfg_v4_17..18`: Tổ hợp có kiểm soát (`I1+I2+I7` và `I1+I2+I7+I5`).

## 2. Giao thức, Ràng buộc & Phá hòa
- Giữ nguyên `exact search` toàn `index/val_gallery/` (`1.884` sp), `tie_rule="conservative"` ($\epsilon = 10^{-5}$), 5-fold CV theo `spec_cluster_id` dùng [`reports/phase5b/folds.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/folds.json) (`SHA-256: dfd30172...`).
- Ràng buộc `5(i)–(v)` giữ nguyên như `v3` (so với `cfg_01` / `model_only` / `parser_v1`).
- Phá hòa bằng paired cluster bootstrap 1.000 lần theo `simplicity_order` định nghĩa trước trong `selection_rule_v4.json`.
- **Cổng xác nhận & Winner's Curse:** Nếu chưa có tập `B-human-v4` mới do người dùng viết và khóa hash, mọi kết quả trên `B-human validated` cũ được gắn nhãn **"tham khảo (đã quan sát ở v3), không phải kiểm chứng độc lập"**.
- **Báo cáo 3 hàng & I8:** Tách `Model thuần / Hệ thống soft / Hệ thống gated`, kèm phân tích hubness (C.8), phân tích 3 mức CI, và kiểm tra thuộc tính phụ không nằm trong 3 slot parser (`masterCategory` / `subCategory`, lưu ý `catalog_rebuild.parquet` không có cột `usage`/`season`).
