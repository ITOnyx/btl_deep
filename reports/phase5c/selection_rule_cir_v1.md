# Quy tắc Chọn Cấu hình CIR (`selection_rule_cir_v1` — Khóa trước khi chạy)

- **Phiên bản:** `cir_v1` (`LOCKED_BEFORE_EXPERIMENTS`)
- **Seed cố định:** `42`
- **Tập tuning:** `eval/cir_eval_val.json` (`120` cặp: `80` đổi màu, `40` đổi giới; `SHA-256: c735c7742c5277cffc4113c68dd8ee59fd3475c742284c8fe9e098af0819062a`).
- **Tuyệt đối không chạm:** `eval/cir_eval_test.json` và `index/test_gallery/`.

## 1. Lưới 8 Cấu hình Ứng viên (`<= 8`)

| Config ID | Vai trò | C1 (`vi->en`) | C2 (`β`) | C3 (`mod bonus γ=1.0`) | C4 (`PRF Rocchio`) |
|---|---|:---:|---:|:---:|:---:|
| `cir_v1_01` | Baseline combiner tiếng Việt | ❌ | `1.00` | ❌ | ❌ |
| `cir_v1_02` | Chỉ C1 (dịch modifier sang tiếng Anh từ bảng thuộc tính catalog) | ✅ | `1.00` | ❌ | ❌ |
| `cir_v1_03` | Chỉ C2 (`β = 0.85`) | ❌ | `0.85` | ❌ | ❌ |
| `cir_v1_04` | Chỉ C2 (`β = 0.70`) | ❌ | `0.70` | ❌ | ❌ |
| `cir_v1_05` | Chỉ C3 (`parser_v2.1` bonus trên phần modifier, không gating) | ❌ | `1.00` | ✅ | ❌ |
| `cir_v1_06` | Kết hợp `C1 + C3` | ✅ | `1.00` | ✅ | ❌ |
| `cir_v1_07` | Kết hợp `C1 + C2 (β=0.85) + C3` | ✅ | `0.85` | ✅ | ❌ |
| `cir_v1_08` | Kết hợp `C1 + C2 (β=0.85) + C3 + C4 (k=3, β_prf=0.15)` | ✅ | `0.85` | ✅ | ✅ |

## 2. Quy tắc Chọn & Tie-Break
1. Đánh giá trên `eval/cir_eval_val.json` (120 cặp với quy tắc 50/50 có dấu `idx % 2 == 0` / không dấu `idx % 2 == 1`, đồng thời báo cáo bảng tách 120 cặp có dấu và 120 cặp không dấu).
2. Cụm bootstrap: bộ ba thuộc tính đích `target_attrs` (`articleType__gender__baseColour`), 1.000 lần lấy mẫu lại.
3. Ràng buộc độ trễ C.9: `p95 < 200 ms` cho cả hai luồng **CIR catalog** và **CIR upload** (cảnh báo nếu `p99 CIR upload > 200 ms`).
4. C4 (`cir_v1_08`) chỉ được giữ nếu `ΔmAP` so với `cir_v1_07` có 95% bootstrap CI nằm hoàn toàn trên `0` và đạt `5(iv)`.
5. Nếu chênh lệch `mAP` so với cấu hình cao nhất có 95% paired cluster bootstrap CI chứa `0`, chọn cấu hình đơn giản hơn theo `simplicity_order`.
