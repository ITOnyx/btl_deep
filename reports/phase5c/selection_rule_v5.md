# Quy tắc Chọn Cấu hình v5 (`selection_rule_v5`) — Pha 5c

- **Trạng thái:** `LOCKED_BEFORE_EXPERIMENTS` | **Seed:** `42`
- **Folds:** `reports/phase5b/folds.json` (`SHA-256: dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25`)

## 1. Quy định riêng v5 (theo duyệt Bước 12.1)
1. **Cache I3 (`index/template_cache_v5/`):** Khóa không có trong `template_key_map.json` sẽ **FALLBACK về 1 forward** (như `cfg_v4_01`), không báo lỗi; bắt buộc ghi tỉ lệ rơi vào fallback.
2. **Đối chứng I3 (`cfg_v5_02` & `cfg_v5_03`):** Chỉ dùng để kiểm tra tương đương và đo độ trễ, **KHÔNG** tranh hạng chọn cấu hình (`competes_for_winner=false`).
3. **Mở rộng $\gamma$ cho `tier_soft`:** Thêm `cfg_v5_09..11` = `cfg_v5_04` với $\gamma \in \{1.0, 2.0, 3.0\}$.
4. **I5 (`cfg_v5_08a` & `cfg_v5_08b`):** Dùng caption **THẬT** của `val_gallery` tại `index/val_gallery_text_v5/` (tách riêng `internvl2_only` và `internvl2_plus_aux` theo Luật 5), **không** dùng `title_vi`. Tổng lưới: **12 cấu hình** (`<= 24`).
5. **Ràng buộc & Độ trễ:** Giữ 100% ràng buộc `5(i)-(v)` của v4. Đo độ trễ theo giao thức **C.9** (GPU nguội, xen kẽ, báo trung vị + IQR của p95).

## 2. Lưới 12 Cấu hình
| Config ID | Tranh hạng | Vai trò | `bonus_mode` | $\gamma$ | `i3_mode` | `conf_gating` | `fuzzy_typo` | `text_gallery` |
|---|:---:|---|---|---:|---|:---:|:---:|---|
| `cfg_v5_01` | ✅ | Baseline v4 winner (`cfg_v4_05`) | `tier_soft` | 1.5 | `none` | `false` | `false` | `none` |
| `cfg_v5_02` | ❌ | Đối chứng I3 direct | `zscore` | 1.5 | `direct` | `false` | `false` | `none` |
| `cfg_v5_03` | ❌ | Đối chứng I3 cached | `zscore` | 1.5 | `cached` | `false` | `false` | `none` |
| `cfg_v5_04` | ✅ | `tier_soft` + `I3_cached` | `tier_soft` | 1.5 | `cached` | `false` | `false` | `none` |
| `cfg_v5_05` | ✅ | `+ I2` trên `04` | `tier_soft` | 1.5 | `cached` | `true` | `false` | `none` |
| `cfg_v5_06` | ✅ | `+ I7` trên `04` | `tier_soft` | 1.5 | `cached` | `false` | `true` | `none` |
| `cfg_v5_07` | ✅ | `+ I2 + I7` trên `04` | `tier_soft` | 1.5 | `cached` | `true` | `true` | `none` |
| `cfg_v5_08a` | ✅ | `+ I5` (`internvl2_only`, $\alpha=0.85$) | `tier_soft` | 1.5 | `cached` | `false` | `false` | `internvl2_only` |
| `cfg_v5_08b` | ✅ | `+ I5` (`internvl2_plus_aux`, $\alpha=0.85$) | `tier_soft` | 1.5 | `cached` | `false` | `false` | `internvl2_plus_aux` |
| `cfg_v5_09` | ✅ | Quét $\gamma=1.0$ trên `04` | `tier_soft` | 1.0 | `cached` | `false` | `false` | `none` |
| `cfg_v5_10` | ✅ | Quét $\gamma=2.0$ trên `04` | `tier_soft` | 2.0 | `cached` | `false` | `false` | `none` |
| `cfg_v5_11` | ✅ | Quét $\gamma=3.0$ trên `04` | `tier_soft` | 3.0 | `cached` | `false` | `false` | `none` |
