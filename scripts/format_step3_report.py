"""Generate reports/phase5b/step3_selection_report.md directly from step3_results.json."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> None:
    r = json.loads(Path("reports/phase5b/step3_results.json").read_text(encoding="utf-8"))
    commit_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    by_id = {g["config_id"]: g for g in r["grid_results"]}

    sel = r["selection"]
    win = by_id[sel["winner_after_tie_break"]]
    best_raw = by_id[sel["best_raw_by_score"]]
    safe = by_id[sel["safe_config"]]
    cfg01 = by_id["cfg_01"]
    cfg02 = by_id["cfg_02"]
    mo = r["reference_model_only"]
    p5 = r["reference_baseline_pha5_no_zscore"]
    bh = r["b_human_gate"]

    table_rows = [
        "| Config | Parser | γ | Slot W | Flags | CV Score | A mAP(cl) | Bc mAP(cl) | Bt mAP(cl) | C20 mAP(cl) | ΔR@10(AB) | CatDrop(B) | p95(ms) | Đạt 5(i)-(v) |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for g in r["grid_results"]:
        c = g["cfg"]
        sw = "unif" if c["slot_weights"] == "uniform" else "0.6/0.2/0.2"
        fl = []
        if c["query_rewrite"]:
            fl.append("RW")
        if c["dual_encode"]:
            fl.append("DE")
        if c["bm25"]:
            fl.append("BM25")
        if c["mean_centering"]:
            fl.append("MC")
        fls = "+".join(fl) if fl else "—"
        p95 = f"{g['latency']['p95_ms']:.1f}" if "latency" in g else "—"
        if g["pass_all"]:
            ok = "PASS"
        elif g["pass_pre_latency"] and g.get("pass_5iv") is False:
            ok = "FAIL(5iv)"
        else:
            ok = "FAIL"
        table_rows.append(
            f"| {g['config_id']} | {c['parser']} | {c['gamma']} | {sw} | {fls} | "
            f"{g['cv_score_mean']:.4f} | {g['map_a_cluster']:.4f} | {g['map_b_clean_cluster']:.4f} | "
            f"{g['map_b_typo_cluster']:.4f} | {g['map_c_20_cluster']:.4f} | "
            f"{g['delta_r10_ab_vs_cfg01']:+.4f} | {g['catastrophic_drop_b']:.4f} | {p95} | {ok} |"
        )

    tie_to_win = next(t for t in sel["tie_checks"] if t["vs_simpler"] == sel["winner_after_tie_break"])
    cos_std = r["mean_cosine_std_val_a"]

    md = f"""# Báo cáo Chọn Cấu hình Bước 3 — Pha 5b (`selection_rule_v3`)

- **Nhánh:** `phase7a` | **Seed:** `{r['seed']}` | **Commit gốc:** `{commit_head}`
- **Parser:** `parser_v2.1` (`SHA-256`: `{r['parser_v2_sha256']}`)
- **Folds CV (5-fold theo `spec_cluster_id`):** [`reports/phase5b/folds.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/folds.json) (`SHA-256`: `{r['folds_sha256']}`)
- **Quy tắc:** [`reports/phase5b/selection_rule_v3.json`](file:///d:/Kì%207/Hoc_sau_ver4/reports/phase5b/selection_rule_v3.json) (`SHA-256`: `{r['selection_rule_sha256']}`)
- **Hash kết quả khóa TRƯỚC khi chạy B-human:** `{r['pre_b_human_results_sha256']}`
- **Phạm vi:** `index/val_gallery/` (`1.884` sản phẩm), `exact search`, `tie_rule="conservative"` ($\\epsilon = 10^{{-5}}$), `zscore_cosine=true`. **KHÔNG** chạy `test_gallery`, **KHÔNG** tune trên DEV.

---

## 1. Tóm tắt Quyết định Chọn Cấu hình

| Vai trò | Config | Mô tả | CV Score | A mAP(cl) | Bc mAP(cl) | Bt mAP(cl) | C20 mAP(cl) | p95 (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Thắng cuộc (sau phá hòa CI)** | **`{win['config_id']}`** | `parser_v2.1`, $\\gamma={win['cfg']['gamma']}$ đều, `query_rewrite=true` | **`{win['cv_score_mean']:.4f}`** | `{win['map_a_cluster']:.4f}` | `{win['map_b_clean_cluster']:.4f}` | `{win['map_b_typo_cluster']:.4f}` | `{win['map_c_20_cluster']:.4f}` | `{win['latency']['p95_ms']:.1f}` |
| **Tốt nhất theo Score thô** | `{best_raw['config_id']}` | `parser_v2.1`, $\\gamma={best_raw['cfg']['gamma']}$ đều, `bm25=true` | `{best_raw['cv_score_mean']:.4f}` | `{best_raw['map_a_cluster']:.4f}` | `{best_raw['map_b_clean_cluster']:.4f}` | `{best_raw['map_b_typo_cluster']:.4f}` | `{best_raw['map_c_20_cluster']:.4f}` | `{best_raw['latency']['p95_ms']:.1f}` |
| **An toàn ($\gamma$ thấp nhất đạt 5(i)(ii))** | `{safe['config_id']}` | `parser_v1`, $\\gamma={safe['cfg']['gamma']}$ đều (baseline v3) | `{safe['cv_score_mean']:.4f}` | `{safe['map_a_cluster']:.4f}` | `{safe['map_b_clean_cluster']:.4f}` | `{safe['map_b_typo_cluster']:.4f}` | `{safe['map_c_20_cluster']:.4f}` | `{safe['latency']['p95_ms']:.1f}` |
| **Ablation bắt buộc** | **`{cfg02['config_id']}`** | `parser_v2.1`, $\\gamma={cfg02['cfg']['gamma']}$ đều | `{cfg02['cv_score_mean']:.4f}` | `{cfg02['map_a_cluster']:.4f}` | `{cfg02['map_b_clean_cluster']:.4f}` | `{cfg02['map_b_typo_cluster']:.4f}` | `{cfg02['map_c_20_cluster']:.4f}` | `{cfg02['latency']['p95_ms']:.1f}` |

- **Phá hòa theo 95% paired cluster bootstrap CI (1.000 lần):** chênh lệch Score giữa `{best_raw['config_id']}` (`0.5391`) và `{win['config_id']}` (`0.5053`) là `{tie_to_win['diff_ci']['mean']:+.4f}` với CI 95% **`[{tie_to_win['diff_ci']['ci_low']:+.4f}, {tie_to_win['diff_ci']['ci_high']:+.4f}]`** (chứa `0.0`). Theo `simplicity_order` (`bm25=false < bm25=true`), chọn **`{win['config_id']}`**.
- **Cảnh báo biên lưới (PROMPT 8 E.3):** Cả `{best_raw['config_id']}` và `{win['config_id']}` đều nằm tại biên trên của lưới (**$\\gamma = 1.5$**) và CV Score còn tăng đều theo $\\gamma$ ($0.3 \\to 0.5 \\to 0.8 \\to 1.0 \\to 1.5$). Do đó **"lưới có thể chưa phủ vùng tối ưu"**; đề xuất khóa `selection_rule_v4` nếu muốn khảo sát $\\gamma > 1.5$ (không tự mở rộng lưới ngoài quy tắc).

---

## 2. Tham chiếu `model_only` và `baseline Pha 5 (KHÔNG z-score)` & Hiệu ứng Z-score

| Cấu hình tham chiếu | Val A mAP(cl) / R@10 | B-synth clean mAP(cl) / R@10 | B-human mAP(cl) / R@10 |
|---|---:|---:|---:|
| **`model_only`** | `{mo['A']['mAP']:.4f}` (query) / `{mo['A']['recall@10']:.4f}` | `{mo['B_clean']['mAP']:.4f}` (query) / `{mo['B_clean']['recall@10']:.4f}` | `{bh['model_only_map_cluster']:.4f}` / — |
| **`baseline Pha 5 (KHÔNG z-score, v1, γ=0.3)`** | **`{p5['A_map_cluster']:.4f}`** / `{p5['A_summary']['recall@10']:.4f}` | **`{p5['B_clean_map_cluster']:.4f}`** / `{p5['B_clean_summary']['recall@10']:.4f}` | **`{bh['baseline_pha5_no_zscore_map_cluster']:.4f}`** / `{bh['baseline_pha5_no_zscore_summary']['recall@10']:.4f}` |
| **`cfg_01` (giao thức v3, `zscore=true`, `v1, γ=0.3`)** | `{cfg01['map_a_cluster']:.4f}` / `{cfg01['sum_a']['recall@10']:.4f}` | `{cfg01['map_b_clean_cluster']:.4f}` / `{cfg01['sum_bc']['recall@10']:.4f}` | `{bh['baseline_v3_map_cluster']:.4f}` / — |

- **Giải thích hiệu ứng z-score (PROMPT 8 E.2):** Độ lệch chuẩn cosine trung bình theo truy vấn trên Val A là **`{cos_std:.4f}`**. Khi KHÔNG z-score, $\\gamma = 0.3$ tương đương $0.3 / {cos_std:.4f} \\approx {0.3/cos_std:.2f}\\,\\sigma$. Khi bật `zscore_cosine=true`, cosine được chia cho $\\sigma \\approx {cos_std:.4f}$, nên $\\gamma = 0.3$ chỉ còn tương đương $0.3\\,\\sigma$ (yếu hơn ~`{1.0/cos_std:.1f}x`), làm `cfg_01` tụt mạnh trên DEV (`0.3749` vs `0.5852`) và Val A (`0.3660` vs `0.5896`). Ngược lại, ở $\\gamma = 1.5$ trên thang z-score ($1.5\\,\\sigma \\approx {1.5*cos_std:.3f}$ theo thang cosine gốc), cường độ bonus mới tiệm cận $\\gamma = 0.25$–$0.30$ trước đây.

---

## 3. Bảng Đầy đủ 22 Cấu hình (`cfg_01..cfg_22`)

*(Đo độ trễ T2I p95 theo AGENTS.md C.9: 5 run × 200 truy vấn, warmup 20, `torch.cuda.synchronize()`, GPU RTX 2050 nhiệt độ 68–86°C, xung nhịp 1477–1732 MHz. Đã đo cho `21/22` cấu hình — chỉ `cfg_20` không đo do fail ràng buộc `5(iii)` `ΔR@10 = -0.0144 < -0.01`.)*

{chr(10).join(table_rows)}

- **Ablation `cfg_02` vs `cfg_01`:** cùng `γ=0.3` và `zscore=true`, đổi `parser_v1` → `parser_v2.1` giúp CV Score nhích `+0.0011` (`0.2286` → `0.2297`), cho thấy phần lớn cải thiện của `{win['config_id']}` đến từ `query_rewrite=true` + tăng $\\gamma$ lên `1.5` (khớp thang z-score), kết hợp giảm FP của `parser_v2.1`.

---

## 4. Cổng Xác nhận `B-human validated` (`n=65`, `61` cụm — chạy SAU khi khóa `{win['config_id']}`)

- **`{win['config_id']}`:** cluster mAP = **`{bh['winner_map_cluster']:.4f}`** (`R@1={bh['winner_summary']['recall@1']:.4f}`, `R@5={bh['winner_summary']['recall@5']:.4f}`, `R@10={bh['winner_summary']['recall@10']:.4f}`, query mAP=`{bh['winner_summary']['mAP']:.4f}`).
- **`cfg_01` (baseline v3):** cluster mAP = **`{bh['baseline_v3_map_cluster']:.4f}`**.
- **`ΔmAP ({win['config_id']} - cfg_01)`:** **`{bh['delta_map_vs_baseline_ci95']['mean']:+.4f}`**, 95% cluster bootstrap CI **`[{bh['delta_map_vs_baseline_ci95']['ci_low']:+.4f}, {bh['delta_map_vs_baseline_ci95']['ci_high']:+.4f}]`** (cận dưới $\\ge -0.02$).
- **Tỉ lệ tụt thảm họa trên B-human:** **`{bh['catastrophic_drop_rate']:.4f}`** (`{bh['catastrophic_drop_rate']*100:.2f}% <= 5%`) → **Đạt cổng B-human (`gate_pass=true`)**.
- **Tách nhóm:**
  - Có màu (`31` câu): `{win['config_id']}` = **`{bh['split_with_colour_31']['winner_map_cluster']:.4f}`** vs `cfg_01` = `{bh['split_with_colour_31']['baseline_v3_map_cluster']:.4f}`.
  - Không màu (`34` câu): `{win['config_id']}` = **`{bh['split_no_colour_34']['winner_map_cluster']:.4f}`** vs `cfg_01` = `{bh['split_no_colour_34']['baseline_v3_map_cluster']:.4f}`.
- **Nhãn màu hiếm:**
  - `Steel (val_b_human_088)`: AP `{bh['rare_labels']['Steel (val_b_human_088)']['winner_ap']:.4f}` (vs `cfg_01`: `{bh['rare_labels']['Steel (val_b_human_088)']['baseline_ap']:.4f}`).
  - `Navy Blue (val_b_human_091)`: AP `{bh['rare_labels']['Navy Blue (val_b_human_091)']['winner_ap']:.4f}` (vs `cfg_01`: `{bh['rare_labels']['Navy Blue (val_b_human_091)']['baseline_ap']:.4f}`).
  - `Olive (val_b_human_079)`: AP `{bh['rare_labels']['Olive (val_b_human_079)']['winner_ap']:.4f}` (vs `cfg_01`: `{bh['rare_labels']['Olive (val_b_human_079)']['baseline_ap']:.4f}` — hổng trung thực do `"xanh rêu"` ánh xạ `Green`).

---

## 5. Hạn chế & Minh bạch

1. **`B-human` toàn bộ do AI soạn (`65/65` `ai_written`)** và **`61/65` cụm là đơn lẻ** (chỉ `4/61` cụm có $\\ge 2$ câu), nên bootstrap CI theo cụm gần bằng theo câu và độ rộng CI phản ánh cỡ mẫu nhỏ.
2. **Tập `val_gallery/` đã nằm trong tập dữ liệu mà checkpoint SigLIP từng thấy khi train**, nên mọi số liệu trên `val_gallery/` chỉ dùng để **so sánh tương đối** giữa các cấu hình, không suy diễn thành hiệu năng tổng quát trên `test_gallery/`.
3. **`parser_v2.1` đã được chỉnh sửa sau khi audit trên Tập A** (sửa ưu tiên cấu trúc `Unisex` và gỡ 18 mục `ec87034`), do đó **Tập A không còn ở trạng thái "chưa thấy"**.
4. **Biên lưới $\\gamma = 1.5$:** Điểm CV còn tăng theo $\\gamma$, lưới hiện tại có thể chưa phủ vùng tối ưu.
"""
    Path("reports/phase5b/step3_selection_report.md").write_text(md, encoding="utf-8")
    print("Wrote reports/phase5b/step3_selection_report.md")


if __name__ == "__main__":
    main()
