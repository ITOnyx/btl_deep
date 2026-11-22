# Báo cáo Preflight Bước 3 (P1–P4) — Pha 5b

- **Nhánh:** `phase7a` | **Seed:** `42`
- **Quyết định:** **DỪNG tại P4 — KHÔNG chạy lưới 22 cấu hình.**
- **Lý do dừng:**
  1. **P2 vi phạm ràng buộc 5(v):** trên `Val A (synthetic)`, `parser_v2` có `gender` FP `0.0333` (`8/240` câu) so với `parser_v1` (`0.0000`) do `"cho nam"` (`Men`, 7 ký tự) dài hơn `"nam nữ"` (`Unisex`, 6 ký tự).
  2. **P1 phát hiện mục chưa truy được nguồn trực tiếp trước B-human:** `18/398` mục trong `parser_v2.py` chỉ xuất hiện trong `src/` từ commit `ec87034` (sau khi `B-human` xuất hiện ở `92ca6c5`/`7b2a652`), trong đó **2 cụm (`"bộ kurta"`, `"quần capri"`)** vừa không có nguyên văn trong `catalog_rebuild.parquet`/`catalog_vi_gaps.md` vừa xuất hiện trong `B-human` (dù nhãn gốc `Kurta Sets`/`Capris` và từ `kurta`/`capri` có trong catalog).

---

## P1. Truy vết Lexicon `parser_v2`

- **Tệp xuất:** [`reports/phase5b/parser_v2_lexicon_provenance.csv`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/parser_v2_lexicon_provenance.csv) (`398` mục: `ARTICLE_TYPE_LEXICON_V2`, `GENDER_LEXICON_V2`, `COLOUR_LEXICON_V2`, `PROTECTED_NON_COLOUR_PHRASES`, `HOMOGRAPH_COLOUR_TOKENS`).
- **Phân bổ nguồn:**
  - `324` mục truy trực tiếp về `parser_v1`, `catalog_rebuild.parquet` (`articleType`, `articleType_vi`, `baseColour`, `baseColour_vi`, `gender`, `gender_vi`, `title_vi`, `caption_vi`), `catalog_vi_gaps.md`, hoặc `homograph_table_2c1c.md`.
  - `56` mục truy về các commit trong `src/` **trước** `ec87034` (`35091c2`, `64aba60`, `92ca6c5`, v.v.).
  - `18` mục chỉ xuất hiện trong `src/` tại `ec87034` (`2026-11-13 15:23:40 +0700`, sau `B-human` `92ca6c5` `11:26:31` / `7b2a652` `15:16:06`):
    - **2 mục xuất hiện trong `B-human` (`in_b_human=True`) và không có nguyên văn trong catalog:**
      - `('articleType', 'Kurta Sets', 'bộ kurta')` — nhãn `Kurta Sets` và `áo kurta` có trong catalog; ghép cụm `bộ kurta` xuất hiện ở `ec87034`.
      - `('articleType', 'Capris', 'quần capri')` — nhãn `Capris`/`capri` có trong catalog & `catalog_vi_gaps.md`; ghép cụm `quần capri` xuất hiện ở `ec87034`.
    - **15 mục không xuất hiện trong `B-human` (`in_b_human=False`)** (dịch tiếng Việt cho nhãn đã có trong `catalog_rebuild.parquet`): `áo khoác đi mưa`, `áo mưa`, `váy quấn sari`, `quần lửng capri`, `quần tất`, `đầm ngủ`, `sandal thể thao`, `dép sandal thể thao`, `túi đưa thư`, `balo leo núi`, `quần trunk`, `xanh ngọc lam`, `xanh oliu`, `đỏ burgundy`, `tím oải hương`.
    - **1 mục không dấu của cụm homograph:** `da that` (dạng không dấu của `da thật`).

### Truy vết 8 mục chỉ định (so với mốc `B-human` `92ca6c5` `2026-11-13 11:26:31`)
| Mục | Nguồn cụ thể | `git log -S` trong `src/` sớm nhất | So với `B-human` |
|---|---|---|---|
| `thép` | `catalog_rebuild.parquet` (`baseColour="Steel"`), `catalog_vi_gaps.md` | `b27dc7a` / `35091c2` (`2026-11-11`) | Trước `B-human` (`màu thép` từ `92ca6c5`) |
| `xanh rêu` | `catalog_rebuild.parquet` (`baseColour_vi="xanh rêu"` → `Green`) | `6adf41e` / `35091c2` (`2026-11-11`) | Trước `B-human` |
| `tunic` | `catalog_rebuild.parquet` (`articleType="Tunics"`) | `b27dc7a` / `35091c2` (`2026-11-11`) | Trước `B-human` |
| `trunk` | `catalog_rebuild.parquet` (`articleType="Trunk"`) | `6adf41e` / `35091c2` (`2026-11-11`) | Trước `B-human` |
| `kurti` | `catalog_rebuild.parquet` (`articleType="Kurtis"`, `articleType_vi="áo kurti"`) | `b27dc7a` / `35091c2` (`2026-11-11`) | Trước `B-human` |
| `dupatta` | `catalog_rebuild.parquet` (`articleType="Dupatta"`), `catalog_vi_gaps.md` | `b27dc7a` / `35091c2` (`2026-11-11`) | Trước `B-human` |
| `capri` | `catalog_rebuild.parquet` (`articleType="Capris"`), `catalog_vi_gaps.md` | `b27dc7a` / `35091c2` (`2026-11-11`) | Trước `B-human` |
| `pyjama` | Không có trong `parser_v2.py` | Không xuất hiện trong `src/` | N/A |

---

## P2. Kiểm toán Dương tính giả `parser_v2` vs `parser_v1`

- **Tệp xuất:** [`reports/phase5b/parser_v2_audit.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/parser_v2_audit.json)
- **Câu `"tìm mua"` / `"tim mua"` / `"cần tìm"` bị gán `Purple`:**
  - `DEV` (`n=100`): `0 -> 0`
  - `Val A` (`n=240`): `0 -> 0`
  - `Val B-synth (all)` (`n=245`): `73 -> 0` (clean `n=165`: `49 -> 0`)
  - `B-human validated` (`n=65`): `4 -> 0`

| Tập | Slot | `v1` fp_q | `v2` fp_q | `v1` fp_p | `v2` fp_p | Đạt 5(v) (`v2 <= v1`)? |
|---|---|---:|---:|---:|---:|---|
| `DEV` (100) | `articleType` | `0.0100` | `0.0000` | `0.0114` | `0.0000` | Đạt |
| `DEV` (100) | `gender` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Đạt |
| `DEV` (100) | `baseColour` | `0.0700` | `0.0100` | `0.0921` | `0.0149` | Đạt |
| `Val A` (240) | `articleType` | `0.0042` | `0.0042` | `0.0044` | `0.0042` | Đạt |
| **`Val A` (240)** | **`gender`** | **`0.0000`** | **`0.0333`** | **`0.0000`** | **`0.0333`** | **VI PHẠM (8/240 câu `"cho nam nữ"` → `Men`)** |
| `Val A` (240) | `baseColour` | `0.0167` | `0.0000` | `0.0199` | `0.0000` | Đạt |
| `Val B-synth all` (245) | `articleType` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Đạt |
| `Val B-synth all` (245) | `gender` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Đạt |
| `Val B-synth all` (245) | `baseColour` | `0.1306` | `0.0204` | `0.1684` | `0.0311` | Đạt |
| `Val B-synth clean` (165) | `articleType` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Đạt |
| `Val B-synth clean` (165) | `gender` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Đạt |
| `Val B-synth clean` (165) | `baseColour` | `0.1030` | `0.0182` | `0.1288` | `0.0252` | Đạt |
| `B-human validated` (65) | `articleType` | `0.0462` | `0.0154` | `0.0545` | `0.0161` | Đạt |
| `B-human validated` (65) | `gender` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | Đạt |
| `B-human validated` (65) | `baseColour` | `0.1077` | `0.0154` | `0.1944` | `0.0333` | Đạt |

---

## P3. Hồi quy `cfg_01`, Cờ & Hash Khóa

- **Hồi quy `cfg_01` trên `DEV` (`index/test_gallery`, `n_valid=86`):**
  - `Pha 5 (pool=100, zscore=False)`:
    - `model_only`: R@1=`0.2907`, R@10=`0.7093`, mAP=`0.3279` (khớp `baseline_frozen`).
    - `cfg_01`: R@1=`0.6047`, R@10=`0.8488`, mAP=`0.5537` (khớp `baseline_frozen`).
  - `Exact full (pool=1290, tie_rule=conservative)`:
    - `model_only`: R@1=`0.3023`, R@10=`0.7093`, mAP=`0.3282`.
    - `cfg_01 (giao thức Pha 5, zscore=False)`: R@1=`0.6047`, R@10=`0.8721`, mAP=`0.5852`.
    - `cfg_01 (giao thức v3, zscore=True)`: R@1=`0.3605`, R@10=`0.7326`, mAP=`0.3749`.
- **SHA-256 tệp đóng băng (đều khớp):**
  - `src/serve/parser_v1.py`: `111002f69185514f2a9fef86438e1fbd25e6108b0d92f16e7ef1e70e74fe58eb`
  - `reports/phase5b/selection_rule_v3.json`: `fa4e0df3469b0f83a458c38a466efaec3ed3b4d951ee52b80c36dc4c783de07e`
  - `reports/phase5b/selection_rule_v3.md`: `d9e7b56d3301b8e0b839706844651ed5a3a00c5516d194898303f1c6e39db4eb`
  - `eval/val_queries_b_human.validated.json`: `19f81e8aa35a034f1528dc101e108ad0469201d50372e16afbc5286e78bd8deb`
- **Tests:** [`tests/test_phase5b_preflight_p3.py`](file:///d:/Kì%207/Hoc_sau_ver4/tests/test_phase5b_preflight_p3.py) kiểm tra đầy đủ hash, Hash Guard, 6 cờ độc lập (`zscore`, `slot_weights`, `bm25`, `mean_centering`, `dual_encode`, `query_rewrite`) và 22 cấu hình `cfg_01..cfg_22`.
