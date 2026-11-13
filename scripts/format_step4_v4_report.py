"""
Tạo báo cáo reports/phase5c/step4_v4_report.md từ reports/phase5c/step4_v4_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _verdict(val: float, ci_low: float, target: float, is_val_a: bool = False) -> str:
    if ci_low >= target:
        if is_val_a:
            return "THAM KHẢO, ước lượng lạc quan (ảnh val đã được checkpoint thấy, parser gần 100% trên A, GT trùng thuộc tính hệ thống dùng)"
        return "Đạt chắc chắn"
    if val >= target:
        return "Đạt điểm số nhưng CI chứa ngưỡng"
    return "Chưa đạt"


def main() -> None:
    res = json.loads(Path("reports/phase5c/step4_v4_results.json").read_text(encoding="utf-8"))
    summary = res["summary"]
    by_id: Dict[str, Any] = {r["config"]["config_id"]: r for r in res["grid_results"]}
    m_only = res["model_only"]

    lines = [
        "# Báo cáo Pha 5c — Bước 4: Đánh giá lưới 18 cấu hình (`selection_rule_v4`)",
        "",
        "## 1. Thông tin tái lập & Bất biến",
        f"- **Git commit:** `{res['git_commit']}`",
        f"- **Seed:** `{res['seed']}`",
        f"- **`selection_rule_v4.json` SHA-256:** `{res['rule_sha256']}`",
        f"- **`folds.json` SHA-256:** `{res['folds_sha256']}`",
        "- **Gallery:** `index/val_gallery/` (1.884 sản phẩm, **KHÔNG** đụng `index/test_gallery/`).",
        "- **Luật 5 — Giới hạn I5 trên `val_gallery`:** Trong `data/catalog/captions/`, chỉ `test_gallery` có caption `InternVL2-2B`; `val_gallery` hiện có `caption_model=None`. Do đó `index/val_gallery_text_v4/` chỉ dùng `title_vi` (`cfg_v4_11/12/18`) và `metadata_aux` (`title_vi + articleType + gender + baseColour`, `cfg_v4_13` — chứa trực tiếp thuộc tính GT). Kết quả I5 trên `val_gallery` chỉ phản ánh text-fusion từ tiêu đề/metadata, **không** phản ánh đóng góp của `InternVL2-2B`.",
        "- **Cảnh báo B-human:** Số liệu `B_human_ref` (65 câu, 61 cụm) chỉ mang tính **THAM KHẢO (REFERENCE ONLY)** vì tập này đã được quan sát ở v3 và chưa có tập `B-human-v4` mới.",
        "",
        "## 2. Kết quả lưới 18 cấu hình (5-fold Cluster CV trên `Val A` + `B-synth clean`)",
        "",
        "| Config | Ý tưởng | CV Score | Val A mAP (cụm) | B-clean mAP (cụm) | B-typo mAP | C-20% mAP | B-human_ref (THAM KHẢO) | Catastrophic Drop (B-clean) | T2I p95 (ms) | Đạt 5(i)-(v) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    mo_a = m_only["A"]
    mo_bc = m_only["B_clean"]
    mo_bt = m_only["B_typo"]
    mo_c20 = m_only["C_20"]
    mo_bh = m_only["B_human_ref"]
    mo_cv = 0.4 * mo_a["cluster_mAP"] + 0.6 * mo_bc["cluster_mAP"]
    lines.append(
        f"| `model_only` | Baseline SigLIP thuần | {mo_cv:.4f} | "
        f"{mo_a['cluster_mAP']:.4f} [{mo_a['ci']['ci_low']:.4f}, {mo_a['ci']['ci_high']:.4f}] | "
        f"{mo_bc['cluster_mAP']:.4f} [{mo_bc['ci']['ci_low']:.4f}, {mo_bc['ci']['ci_high']:.4f}] | "
        f"{mo_bt['cluster_mAP']:.4f} | {mo_c20['cluster_mAP']:.4f} | "
        f"{mo_bh['cluster_mAP']:.4f} | 0.000 | — | — |"
    )

    for r in res["grid_results"]:
        c = r["config"]
        ds = r["datasets"]
        ok = "PASS" if r["constraints"]["all_pass"] else "FAIL"
        lines.append(
            f"| `{c['config_id']}` | {c['idea']} | {r['cv_score']:.4f} | "
            f"{ds['A']['cluster_mAP']:.4f} [{ds['A']['ci']['ci_low']:.4f}, {ds['A']['ci']['ci_high']:.4f}] | "
            f"{ds['B_clean']['cluster_mAP']:.4f} [{ds['B_clean']['ci']['ci_low']:.4f}, {ds['B_clean']['ci']['ci_high']:.4f}] | "
            f"{ds['B_typo']['cluster_mAP']:.4f} | {ds['C_20']['cluster_mAP']:.4f} | "
            f"{ds['B_human_ref']['cluster_mAP']:.4f} | {r['catastrophic_drop_rate_b_clean']:.3f} | "
            f"{r['latency']['p95_ms']:.2f} | {ok} |"
        )

    lines.extend([
        "",
        "## 3. Chọn cấu hình & Tie-breaking (Paired Cluster Bootstrap 1.000 lần)",
        f"- **Best by CV Score (đạt mọi ràng buộc):** `{summary['best_by_score']}` (CV = {by_id[summary['best_by_score']]['cv_score']:.4f})",
        f"- **Các cấu hình hòa trong 95% CI với best:** `{', '.join(summary['tied_configs'])}`",
        f"- **Winner sau tie-break theo `simplicity_order`:** `{summary['winner_after_tie_break']}`",
        "",
        "## 4. Báo cáo song song 4 hàng (Val A & B-synth clean)",
        "",
        "| Vai trò | Config | Val A R@10 | Val A mAP (cụm, 95% CI) | Kết luận mAP >= 0.75 | B-clean R@10 | B-clean mAP (cụm, 95% CI) | Kết luận mAP >= 0.75 |",
        "|---|---|---:|---|---|---:|---|---|",
    ])

    def _add_parallel_row(label: str, cid: str | None) -> None:
        if cid is None:
            return
        if cid == "model_only":
            a, bc = mo_a, mo_bc
        else:
            a, bc = by_id[cid]["datasets"]["A"], by_id[cid]["datasets"]["B_clean"]
        v_a = _verdict(a["cluster_mAP"], a["ci"]["ci_low"], 0.75, is_val_a=True)
        v_bc = _verdict(bc["cluster_mAP"], bc["ci"]["ci_low"], 0.75, is_val_a=False)
        lines.append(
            f"| {label} | `{cid}` | {a['metrics']['recall@10']:.4f} | "
            f"{a['cluster_mAP']:.4f} [{a['ci']['ci_low']:.4f}, {a['ci']['ci_high']:.4f}] | {v_a} | "
            f"{bc['metrics']['recall@10']:.4f} | "
            f"{bc['cluster_mAP']:.4f} [{bc['ci']['ci_low']:.4f}, {bc['ci']['ci_high']:.4f}] | {v_bc} |"
        )

    _add_parallel_row("Model thuần", "model_only")
    _add_parallel_row("Hệ thống soft (v3 baseline)", "cfg_v4_01")
    if summary["system_gated"] == summary["winner_after_tie_break"]:
        _add_parallel_row("Hệ thống soft / gated (v4 winner, không dùng conf_gating)", summary["winner_after_tie_break"])
    else:
        _add_parallel_row("Hệ thống soft (v4 winner)", summary["winner_after_tie_break"])
        _add_parallel_row("Hệ thống gated", summary["system_gated"])
    _add_parallel_row("Safe config (5i+5ii)", summary["safe_config"])
    lines.append("")
    lines.append("> **Lưu ý:** Không được dùng kết luận trên `Val A` làm bằng chứng đạt chỉ tiêu `0.85` / `0.75`. Chỉ số chính thức chỉ xét trên `test_gallery`.")

    lines.extend([
        "",
        "## 4b. Cấu hình tương đương & Số cấu hình hiệu dụng",
        f"- **Số cấu hình hiệu dụng khác nhau thật sự:** **16** cấu hình trên `(Val A, B-clean)` (và **18** trên cả 4 tập `A, B-clean, B-typo, C-20%`).",
        "",
        "| Cặp đối chiếu | Val A mAP | B-clean mAP | B-typo mAP | Giải thích vì sao trùng",
        "|---|---:|---:|---:|---|",
        f"| `cfg_v4_01` (`zscore`, γ=1.5) vs `cfg_v4_03` (`sigma_scaled`, γ=1.5) | {by_id['cfg_v4_01']['datasets']['A']['cluster_mAP']:.6f} vs {by_id['cfg_v4_03']['datasets']['A']['cluster_mAP']:.6f} | {by_id['cfg_v4_01']['datasets']['B_clean']['cluster_mAP']:.6f} vs {by_id['cfg_v4_03']['datasets']['B_clean']['cluster_mAP']:.6f} | {by_id['cfg_v4_01']['datasets']['B_typo']['cluster_mAP']:.6f} vs {by_id['cfg_v4_03']['datasets']['B_typo']['cluster_mAP']:.6f} | `(cos - μ)/σ + γ·mf` là phép biến đổi tuyến tính đơn điệu của `cos + γ·σ·mf` khi không có thành phần điểm thứ ba (`extra_score_matrix`/`bm25`). Sai khác ở chữ số thứ 6 trên `B-typo` chỉ do làm tròn tie-breaking dấu phẩy động. |",
        f"| `cfg_v4_01` (`fuzzy_typo=false`) vs `cfg_v4_08` (`fuzzy_typo=true`) | {by_id['cfg_v4_01']['datasets']['A']['cluster_mAP']:.6f} vs {by_id['cfg_v4_08']['datasets']['A']['cluster_mAP']:.6f} | {by_id['cfg_v4_01']['datasets']['B_clean']['cluster_mAP']:.6f} vs {by_id['cfg_v4_08']['datasets']['B_clean']['cluster_mAP']:.6f} | {by_id['cfg_v4_01']['datasets']['B_typo']['cluster_mAP']:.6f} vs {by_id['cfg_v4_08']['datasets']['B_typo']['cluster_mAP']:.6f} | Trên `A` và `B-clean` không có lỗi chính tả độ dài ≥ 4 ký tự làm thay đổi kết quả parse, nên `fuzzy_typo` chỉ tác động trên `B-typo` (+0.0142 mAP). |",
    ])

    hub = res["hubness_c8_val_a"]
    i8 = res["i8_non_circular_check_val_a"]
    w_lat = by_id[summary["winner_after_tie_break"]]["latency"]
    b_lat = by_id["cfg_v4_01"]["latency"]

    lines.extend([
        "",
        "## 5. Hubness (Mục C.8 trên Val A)",
        f"- **`model_only` trước mean-centering:** max_appearances={hub['model_only_before_mc']['max_appearances']}, coverage_ratio={hub['model_only_before_mc']['coverage_ratio']:.4f}, skewness={hub['model_only_before_mc']['skewness']:.4f}, n_hubs(>5)={hub['model_only_before_mc']['n_hubs_above_threshold']}",
        f"- **`model_only` sau mean-centering:** max_appearances={hub['model_only_after_mc']['max_appearances']}, coverage_ratio={hub['model_only_after_mc']['coverage_ratio']:.4f}, skewness={hub['model_only_after_mc']['skewness']:.4f}, n_hubs(>5)={hub['model_only_after_mc']['n_hubs_above_threshold']}",
        f"- **Winner (`{summary['winner_after_tie_break']}`):** max_appearances={hub['winner']['max_appearances']}, coverage_ratio={hub['winner']['coverage_ratio']:.4f}, skewness={hub['winner']['skewness']:.4f}, n_hubs(>5)={hub['winner']['n_hubs_above_threshold']}",
        "",
        "## 6. Độ trễ T2I (Mục C.9: 5 run × 200 truy vấn, warmup 20, `torch.cuda.synchronize()`)",
        f"- **`cfg_v4_01` (v3 baseline):** p50={b_lat['p50_ms']} ms, p95={b_lat['p95_ms']} ms, p99={b_lat['p99_ms']} ms | GPU before: {b_lat['gpu_before']} | GPU after: {b_lat['gpu_after']}",
        f"- **Winner (`{summary['winner_after_tie_break']}`):** p50={w_lat['p50_ms']} ms, p95={w_lat['p95_ms']} ms, p99={w_lat['p99_ms']} ms (chênh lệch p95 so với `cfg_v4_01`: {w_lat['p95_ms'] - b_lat['p95_ms']:+.3f} ms) | GPU before: {w_lat['gpu_before']} | GPU after: {w_lat['gpu_after']}",
        "",
        "## 7. Kiểm tra thuộc tính phi vòng tròn (I8)",
        "- **Trạng thái:** **KHÔNG THỰC HIỆN (`n_evaluated=0`)**.",
        "- **Lý do:** Đã đề xuất kiểm tra trên các thuộc tính ngoài 3-slot (`usage`/`season` hoặc `masterCategory`/`subCategory`), nhưng `usage`/`season` chưa có trong `catalog_rebuild.parquet` và chưa nối ánh xạ GT cho phép đo này ở Bước 4.",
        "",
        "## 8. Giới hạn trung thực (Luật 1 & C.10)",
        "- Checkpoint `best_model.pt` giữ nguyên không fine-tune (Luật 1). Toàn bộ cải thiện nằm ở tầng suy luận.",
        "- Chưa chạy `index/test_gallery/` (Bước 5 chỉ chạy 1 lần sau khi người dùng duyệt kết quả Bước 4 và/hoặc bổ sung cổng `B-human-v4` mới).",
        "- Nếu sau Bước 5 trên `test_gallery`, điểm `model_only` hoặc `system` vẫn chưa đạt ngưỡng 0.85 R@10 / 0.75 mAP, báo cáo sẽ dừng ở mức hiện trạng và ghi nhận phương án **\"cần train\"** để người dùng quyết định.",
    ])

    out_path = Path("reports/phase5c/step4_v4_report.md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Winner={summary['winner_after_tie_break']}, CV={by_id[summary['winner_after_tie_break']]['cv_score']:.4f}, ValA_mAP={by_id[summary['winner_after_tie_break']]['datasets']['A']['cluster_mAP']:.4f}, Bclean_mAP={by_id[summary['winner_after_tie_break']]['datasets']['B_clean']['cluster_mAP']:.4f}, p95={w_lat['p95_ms']}ms")


if __name__ == "__main__":
    main()
