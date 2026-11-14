"""
Phase 5c Prompt 13: Đánh giá và chọn cấu hình CIR (cir_v1_01..cir_v1_08) trên eval/cir_eval_val.json.
Tuân thủ reports/phase5c/selection_rule_cir_v1.json, seed=42.
TUYỆT ĐỐI KHÔNG chạm eval/cir_eval_test.json hoặc index/test_gallery/.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_step3_grid import (
    cluster_bootstrap_ci,
    gpu_telemetry,
    paired_cluster_bootstrap_diff,
)
from src.eval.metrics import compute_average_precision, compute_recall_at_k
from src.serve.cir_enhancer import CatalogModifierTranslator, score_cir_single_query
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import restore_diacritics_for_query
from src.serve.search_engine import ECommerceSearchEngine
from src.serve.template_cache import TemplateCacheV5

SEED = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def simplicity_rank_cir(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Thứ tự đơn giản (nhỏ hơn = đơn giản hơn) theo selection_rule_cir_v1.json:
    1. c4_rocchio=False < True
    2. Số mô-đun tích cực trong (C1, C2, C3) ít hơn
    3. beta gần 1.0 hơn (1.0 < 0.85 < 0.70 -> 1.0 - beta nhỏ hơn)
    4. c1_vi_to_en=False < True
    5. c3_mod_bonus=False < True
    """
    c1 = bool(cfg.get("c1_vi_to_en", False))
    beta = float(cfg.get("beta", 1.0))
    c2 = beta < 1.0
    c3 = bool(cfg.get("c3_mod_bonus", False))
    c4 = bool(cfg.get("c4_rocchio", False))
    n_active = int(c1) + int(c2) + int(c3)
    return (
        1 if c4 else 0,
        n_active,
        round(1.0 - beta, 4),
        1 if c1 else 0,
        1 if c3 else 0,
        cfg["config_id"],
    )


def compute_subset_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mAP": 0.0}
    return {
        "n": len(records),
        "recall@1": round(float(np.mean([r["r1"] for r in records])), 4),
        "recall@5": round(float(np.mean([r["r5"] for r in records])), 4),
        "recall@10": round(float(np.mean([r["r10"] for r in records])), 4),
        "mAP": round(float(np.mean([r["ap"] for r in records])), 4),
    }


def main() -> int:
    rule_path = PROJECT_ROOT / "reports/phase5c/selection_rule_cir_v1.json"
    rule_sha_path = PROJECT_ROOT / "reports/phase5c/selection_rule_cir_v1.sha256"
    rule_sha = sha256_file(rule_path)
    expected_rule_sha = rule_sha_path.read_text(encoding="utf-8").strip().split()[0]
    if rule_sha != expected_rule_sha:
        raise RuntimeError(f"selection_rule_cir_v1.json SHA-256 mismatch: {rule_sha} != {expected_rule_sha}")

    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    val_cir_path = PROJECT_ROOT / rule["tuning_dataset"]["path"]
    if sha256_file(val_cir_path) != rule["tuning_dataset"]["sha256"]:
        raise RuntimeError("eval/cir_eval_val.json SHA-256 mismatch!")

    val_cir_obj = json.loads(val_cir_path.read_text(encoding="utf-8"))
    pairs: List[Dict[str, Any]] = val_cir_obj["pairs"]
    grid: List[Dict[str, Any]] = rule["config_grid"]

    eng = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    val_manifest = json.loads((PROJECT_ROOT / "index/val_gallery/manifest.json").read_text(encoding="utf-8"))
    tpl_cache = TemplateCacheV5.load(
        PROJECT_ROOT / "index/template_cache_v5",
        expected_checkpoint_hash=val_manifest["checkpoint_hash"],
    )
    parser = VietnameseQueryParserV2Conf()
    translator = CatalogModifierTranslator(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    gpol = json.loads((PROJECT_ROOT / "eval/gender_gt_policy.json").read_text(encoding="utf-8"))
    strict_cats = gpol["policies"]["hybrid"]["strict_categories"]

    # Map product_id -> gallery index in val_gallery
    pid_to_idx: Dict[str, int] = {}
    for idx, meta in enumerate(eng.id_map):
        pid = str(meta.get("product_id") or meta.get("item_id"))
        pid_to_idx[pid] = idx

    # Precompute GPU embeddings for all 120 val pairs (both full and none diacritic variants, VI and EN)
    print(f"[Pre-encode CIR Val] {len(pairs)} pairs...", flush=True)
    precomp: List[Dict[str, Any]] = []
    for p_idx, p in enumerate(pairs):
        ref_pid = str(p["ref_product_id"])
        ref_gal_idx = pid_to_idx[ref_pid]
        ref_vec_np = eng.embeddings[ref_gal_idx : ref_gal_idx + 1]
        ref_img_gpu = torch.from_numpy(ref_vec_np).to(eng.device)
        if eng.device == "cuda":
            ref_img_gpu = ref_img_gpu.half()

        mod_full = str(p["modifier_vi"])
        mod_none = strip_diacritics(mod_full)
        primary_dia = "full" if (p_idx % 2 == 0) else "none"
        mod_primary = mod_full if primary_dia == "full" else mod_none

        variants = {}
        for v_key, raw_mod in (("primary", mod_primary), ("full", mod_full), ("none", mod_none)):
            qt_vi = restore_diacritics_for_query(raw_mod)
            en_mod = translator.translate(raw_mod)
            parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)

            in_vi = eng.preprocessor.preprocess_single_text(qt_vi, device=eng.device)
            in_en = eng.preprocessor.preprocess_single_text(en_mod, device=eng.device)
            with torch.no_grad():
                emb_vi = eng.model.encode_text(
                    input_ids=in_vi["input_ids"], attention_mask=in_vi.get("attention_mask")
                )
                emb_en = eng.model.encode_text(
                    input_ids=in_en["input_ids"], attention_mask=in_en.get("attention_mask")
                )
            variants[v_key] = {
                "raw_mod": raw_mod,
                "qt_vi": qt_vi,
                "en_mod": en_mod,
                "parsed_mod": parsed_mod,
                "emb_vi_gpu": emb_vi,
                "emb_en_gpu": emb_en,
            }

        t_attrs = p["target_attrs"]
        cluster_id = f"{t_attrs['articleType']}__{t_attrs['gender']}__{t_attrs['baseColour']}"
        gt_indices = {pid_to_idx[str(g)] for g in p["gt_product_ids"] if str(g) in pid_to_idx}

        precomp.append({
            "pair_idx": p_idx,
            "id": p["id"],
            "ref_gal_idx": ref_gal_idx,
            "ref_image_path": str(PROJECT_ROOT / p["ref_image_path"]),
            "ref_img_gpu": ref_img_gpu,
            "changed_attr": p["changed_attr"],
            "primary_dia": primary_dia,
            "cluster_id": cluster_id,
            "gt_indices": gt_indices,
            "variants": variants,
        })

    def _eval_config_variant(cfg: Dict[str, Any], variant_key: str) -> Dict[str, Any]:
        c1 = bool(cfg.get("c1_vi_to_en", False))
        beta = float(cfg.get("beta", 1.0))
        c3 = bool(cfg.get("c3_mod_bonus", False))
        gamma = float(cfg.get("gamma", 1.0))
        c4 = bool(cfg.get("c4_rocchio", False))
        rk = int(cfg.get("rocchio_k", 0)) if c4 else 0
        rb = float(cfg.get("rocchio_beta", 0.0)) if c4 else 0.0

        records: List[Dict[str, Any]] = []
        by_cluster: Dict[str, List[float]] = defaultdict(list)

        for item in precomp:
            var = item["variants"][variant_key]
            mod_gpu = var["emb_en_gpu"] if c1 else var["emb_vi_gpu"]
            _, ranked = score_cir_single_query(
                engine=eng,
                ref_img_emb_gpu=item["ref_img_gpu"],
                mod_txt_emb_gpu=mod_gpu,
                parsed_mod=var["parsed_mod"],
                tpl_cache=tpl_cache,
                strict_cats=strict_cats,
                beta=beta,
                use_c3_bonus=c3,
                gamma=gamma,
                rocchio_k=rk,
                rocchio_beta=rb,
            )
            gt_set = item["gt_indices"]
            rels = [1 if idx in gt_set else 0 for idx in ranked]
            r1 = compute_recall_at_k(rels, 1)
            r5 = compute_recall_at_k(rels, 5)
            r10 = compute_recall_at_k(rels, 10)
            ap = compute_average_precision(rels, total_relevant=len(gt_set))
            rec = {
                "id": item["id"],
                "cluster_id": item["cluster_id"],
                "changed_attr": item["changed_attr"],
                "dia_class": item["primary_dia"] if variant_key == "primary" else variant_key,
                "r1": r1,
                "r5": r5,
                "r10": r10,
                "ap": ap,
            }
            records.append(rec)
            by_cluster[item["cluster_id"]].append(float(ap))

        cluster_map = {k: float(np.mean(v)) for k, v in by_cluster.items()}
        overall = compute_subset_metrics(records)
        by_colour = compute_subset_metrics([r for r in records if r["changed_attr"] == "baseColour"])
        by_gender = compute_subset_metrics([r for r in records if r["changed_attr"] == "gender"])
        by_dia_full = compute_subset_metrics([r for r in records if r["dia_class"] == "full"])
        by_dia_none = compute_subset_metrics([r for r in records if r["dia_class"] == "none"])

        return {
            "overall": overall,
            "by_modifier_type": {"baseColour": by_colour, "gender": by_gender},
            "by_dia_class": {"full": by_dia_full, "none": by_dia_none},
            "cluster_map": cluster_map,
            "records": records,
        }

    results_by_cfg: Dict[str, Dict[str, Any]] = {}
    for cfg in grid:
        cid = cfg["config_id"]
        print(f"[CIR Grid Eval] Evaluating {cid}...", flush=True)
        res_prim = _eval_config_variant(cfg, "primary")
        res_full120 = _eval_config_variant(cfg, "full")
        res_none120 = _eval_config_variant(cfg, "none")
        ci_prim = cluster_bootstrap_ci(res_prim["cluster_map"], n_resamples=1000, seed=SEED)
        results_by_cfg[cid] = {
            "config": cfg,
            "primary_120": {
                "overall": res_prim["overall"],
                "cluster_mAP_ci95": [round(ci_prim["ci_low"], 4), round(ci_prim["ci_high"], 4)],
                "by_modifier_type": res_prim["by_modifier_type"],
                "by_dia_class": res_prim["by_dia_class"],
            },
            "diagnostic_all_full_120": res_full120["overall"],
            "diagnostic_all_none_120": res_none120["overall"],
            "_cluster_map": res_prim["cluster_map"],
        }

    base_cmap = results_by_cfg["cir_v1_01"]["_cluster_map"]
    for cid, v in results_by_cfg.items():
        diff_vs_base = paired_cluster_bootstrap_diff(v["_cluster_map"], base_cmap, n_resamples=1000, seed=SEED)
        v["primary_120"]["delta_mAP_vs_cir_v1_01"] = round(diff_vs_base["mean"], 4)
        v["primary_120"]["delta_CI95_vs_cir_v1_01"] = [
            round(diff_vs_base["ci_low"], 4),
            round(diff_vs_base["ci_high"], 4),
        ]

    # Check C4 rule: cir_v1_08 vs cir_v1_07
    diff_08_vs_07 = paired_cluster_bootstrap_diff(
        results_by_cfg["cir_v1_08"]["_cluster_map"],
        results_by_cfg["cir_v1_07"]["_cluster_map"],
        n_resamples=1000,
        seed=SEED,
    )
    c4_ci_excludes_zero = diff_08_vs_07["ci_low"] > 0.0

    # Select winner before latency check, then verify latency SLA (< 200ms)
    eligible = [
        cid for cid in results_by_cfg
        if not (results_by_cfg[cid]["config"].get("c4_rocchio") and not c4_ci_excludes_zero)
    ]
    eligible.sort(
        key=lambda cid: (
            -results_by_cfg[cid]["primary_120"]["overall"]["mAP"],
            simplicity_rank_cir(results_by_cfg[cid]["config"]),
        )
    )
    top_cid = eligible[0]
    top_cmap = results_by_cfg[top_cid]["_cluster_map"]
    tied_cids = [top_cid]
    for cid in eligible[1:]:
        d_res = paired_cluster_bootstrap_diff(top_cmap, results_by_cfg[cid]["_cluster_map"], n_resamples=1000, seed=SEED)
        if d_res["ci_low"] <= 0.0 <= d_res["ci_high"]:
            tied_cids.append(cid)

    tied_cids.sort(key=lambda cid: simplicity_rank_cir(results_by_cfg[cid]["config"]))
    winner_cid = tied_cids[0]
    winner_cfg = results_by_cfg[winner_cid]["config"]

    # C.9 Latency measurement on CIR catalog and CIR upload (5 runs x 200 queries, warmup 20)
    # Measure both baseline (cir_v1_01) and winner (winner_cid) interleaved per query
    lat_configs = [results_by_cfg["cir_v1_01"]["config"]]
    if winner_cid != "cir_v1_01":
        lat_configs.append(winner_cfg)

    queries_200 = [precomp[i % len(precomp)] for i in range(200)]

    def _run_cir_end_to_end(cfg: Dict[str, Any], item: Dict[str, Any], is_catalog: bool) -> None:
        raw_mod = item["variants"]["primary"]["raw_mod"]
        qt_vi = restore_diacritics_for_query(raw_mod)
        parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
        mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
        mod_inputs = eng.preprocessor.preprocess_single_text(mod_str, device=eng.device)
        with torch.no_grad():
            mod_txt_emb = eng.model.encode_text(
                input_ids=mod_inputs["input_ids"],
                attention_mask=mod_inputs.get("attention_mask"),
            )
            if is_catalog:
                idx = item["ref_gal_idx"]
                ref_vec = eng.embeddings[idx : idx + 1]
                ref_img_emb = torch.from_numpy(ref_vec).to(eng.device)
                if eng.device == "cuda":
                    ref_img_emb = ref_img_emb.half()
            else:
                px = eng.preprocessor.preprocess_single_image(
                    item["ref_image_path"], device=eng.device, half_precision=(eng.device == "cuda")
                )
                ref_img_emb = eng.model.encode_image(px)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = score_cir_single_query(
                engine=eng,
                ref_img_emb_gpu=ref_img_emb,
                mod_txt_emb_gpu=mod_txt_emb,
                parsed_mod=parsed_mod,
                tpl_cache=tpl_cache,
                strict_cats=strict_cats,
                beta=float(cfg.get("beta", 1.0)),
                use_c3_bonus=bool(cfg.get("c3_mod_bonus", False)),
                gamma=float(cfg.get("gamma", 1.0)),
                rocchio_k=int(cfg.get("rocchio_k", 0)) if cfg.get("c4_rocchio") else 0,
                rocchio_beta=float(cfg.get("rocchio_beta", 0.0)) if cfg.get("c4_rocchio") else 0.0,
            )

    latency_c9: Dict[str, Dict[str, Any]] = {}
    for stream_name, is_cat in (("cir_catalog", True), ("cir_upload", False)):
        p95_runs: Dict[str, List[float]] = {c["config_id"]: [] for c in lat_configs}
        all_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in lat_configs}
        tele_log: List[Dict[str, Any]] = []

        for r_idx in range(5):
            print(f"[C.9 Latency {stream_name}] Run {r_idx + 1}/5...", flush=True)
            time.sleep(1.0)
            t_before = gpu_telemetry()
            for q_item in queries_200[:20]:
                for cfg in lat_configs:
                    _run_cir_end_to_end(cfg, q_item, is_catalog=is_cat)
            run_ms_by_cid: Dict[str, List[float]] = {c["config_id"]: [] for c in lat_configs}
            for q_idx, q_item in enumerate(queries_200):
                order = list(lat_configs) if (r_idx + q_idx) % 2 == 0 else list(reversed(lat_configs))
                for cfg in order:
                    cid = cfg["config_id"]
                    t0 = time.perf_counter()
                    _run_cir_end_to_end(cfg, q_item, is_catalog=is_cat)
                    run_ms_by_cid[cid].append((time.perf_counter() - t0) * 1000.0)
            t_after = gpu_telemetry()
            tele_log.append({"before": t_before, "after": t_after})
            for cfg in lat_configs:
                cid = cfg["config_id"]
                p95_runs[cid].append(float(np.percentile(run_ms_by_cid[cid], 95)))
                all_ms[cid].extend(run_ms_by_cid[cid])

        latency_c9[stream_name] = {}
        for cfg in lat_configs:
            cid = cfg["config_id"]
            p95s = p95_runs[cid]
            arr = np.array(all_ms[cid], dtype=np.float64)
            q25, q75 = np.percentile(p95s, [25, 75])
            p99_val = float(np.percentile(arr, 99))
            p95_med = float(statistics.median(p95s))
            latency_c9[stream_name][cid] = {
                "n_runs": 5,
                "n_queries_per_run": 200,
                "p95_runs_ms": [round(x, 3) for x in p95s],
                "p95_median_ms": round(p95_med, 3),
                "p95_iqr_ms": round(float(q75 - q25), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p99_ms": round(p99_val, 3),
                "sla_p95_under_200ms": p95_med < 200.0,
                "warn_upload_p99_over_200ms": (not is_cat) and (p99_val > 200.0),
                "telemetry": tele_log,
            }

    # Sample C1 translations for inspection
    sample_translations = [
        {
            "raw_full": item["variants"]["full"]["raw_mod"],
            "en_from_full": item["variants"]["full"]["en_mod"],
            "raw_none": item["variants"]["none"]["raw_mod"],
            "en_from_none": item["variants"]["none"]["en_mod"],
        }
        for item in precomp[:6]
    ]

    summary = {
        "rule_sha256": rule_sha,
        "val_cir_sha256": rule["tuning_dataset"]["sha256"],
        "seed": SEED,
        "c4_rocchio_check": {
            "delta_mAP_08_vs_07": round(diff_08_vs_07["mean"], 4),
            "ci95": [round(diff_08_vs_07["ci_low"], 4), round(diff_08_vs_07["ci_high"], 4)],
            "ci_excludes_zero": c4_ci_excludes_zero,
        },
        "top_by_mAP": top_cid,
        "tied_within_95ci": tied_cids,
        "winner_config_id": winner_cid,
        "winner_config": winner_cfg,
        "sample_c1_translations": sample_translations,
        "latency_c9": latency_c9,
        "configs": {
            cid: {k: val for k, val in v.items() if not k.startswith("_")}
            for cid, v in results_by_cfg.items()
        },
    }

    out_json = PROJECT_ROOT / "reports/phase5c/step5c_cir_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_sha = sha256_file(out_json)
    out_json.with_suffix(".sha256").write_text(f"{out_sha}  {out_json.name}\n", encoding="utf-8")

    # Build Markdown report
    w_res = results_by_cfg[winner_cid]["primary_120"]
    b_res = results_by_cfg["cir_v1_01"]["primary_120"]
    md = [
        "# Báo cáo Đánh giá & Chọn Cấu hình CIR — Pha 5c (Prompt 13)",
        "",
        f"- **Rule SHA-256 (`selection_rule_cir_v1.json`):** `{rule_sha}`",
        f"- **Tập tuning (`eval/cir_eval_val.json`):** `120` cặp (`80` đổi màu, `40` đổi giới; `SHA-256: {rule['tuning_dataset']['sha256']}`)",
        f"- **Kết quả JSON:** `reports/phase5c/step5c_cir_results.json` (`SHA-256: {out_sha}`)",
        f"- **Cấu hình CIR cuối (`winner_config_id`):** **`{winner_cid}`** (`mAP = {w_res['overall']['mAP']:.4f}`, `ΔmAP vs cir_v1_01 = {w_res['delta_mAP_vs_cir_v1_01']:+.4f}`, 95% CI: `[{w_res['delta_CI95_vs_cir_v1_01'][0]:+.4f}, {w_res['delta_CI95_vs_cir_v1_01'][1]:+.4f}]`)",
        f"- **Nhóm ứng viên hòa trong 95% cluster bootstrap CI:** `{tied_cids}` (chọn `{winner_cid}` theo `simplicity_order`).",
        "",
        "## 1. Bảng Kết quả Lưới 8 Cấu hình trên `CIR-eval val` (120 cặp: 60 có dấu + 60 không dấu)",
        "",
        "| Config ID | Vai trò | R@1 | R@5 | R@10 | mAP | 95% CI (mAP) | ΔmAP vs `cir_v1_01` (95% CI) | mAP Màu (`n=80`) | mAP Giới (`n=40`) | mAP Có dấu (`n=60`) | mAP Không dấu (`n=60`) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in grid:
        cid = cfg["config_id"]
        p120 = results_by_cfg[cid]["primary_120"]
        ov = p120["overall"]
        ci = p120["cluster_mAP_ci95"]
        d_ci = p120["delta_CI95_vs_cir_v1_01"]
        mc = p120["by_modifier_type"]["baseColour"]["mAP"]
        mg = p120["by_modifier_type"]["gender"]["mAP"]
        dfull = p120["by_dia_class"]["full"]["mAP"]
        dnone = p120["by_dia_class"]["none"]["mAP"]
        md.append(
            f"| `{cid}` | {cfg['role']} | {ov['recall@1']:.4f} | {ov['recall@5']:.4f} | {ov['recall@10']:.4f} | "
            f"**{ov['mAP']:.4f}** | `[{ci[0]:.4f}, {ci[1]:.4f}]` | `{p120['delta_mAP_vs_cir_v1_01']:+.4f} [{d_ci[0]:+.4f}, {d_ci[1]:+.4f}]` | "
            f"{mc:.4f} | {mg:.4f} | {dfull:.4f} | {dnone:.4f} |"
        )

    md.extend([
        "",
        "## 2. Chẩn đoán Toàn bộ 120 cặp Có dấu (`all_full`) vs 120 cặp Không dấu (`all_none`)",
        "",
        "| Config ID | `all_full (n=120)` R@1 | `all_full` R@10 | `all_full` mAP | `all_none (n=120)` R@1 | `all_none` R@10 | `all_none` mAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cfg in grid:
        cid = cfg["config_id"]
        ff = results_by_cfg[cid]["diagnostic_all_full_120"]
        nn = results_by_cfg[cid]["diagnostic_all_none_120"]
        md.append(
            f"| `{cid}` | {ff['recall@1']:.4f} | {ff['recall@10']:.4f} | {ff['mAP']:.4f} | "
            f"{nn['recall@1']:.4f} | {nn['recall@10']:.4f} | {nn['mAP']:.4f} |"
        )

    md.extend([
        "",
        "## 3. Kiểm tra C4 (PRF Rocchio) & C1 (Dịch Modifier Tiếng Việt -> Tiếng Anh)",
        f"- **Kiểm định C4 (`cir_v1_08` vs `cir_v1_07`):** `ΔmAP = {summary['c4_rocchio_check']['delta_mAP_08_vs_07']:+.4f}`, "
        f"95% CI = `{summary['c4_rocchio_check']['ci95']}`, `ci_excludes_zero = {c4_ci_excludes_zero}` "
        f"=> **{'Giữ C4' if c4_ci_excludes_zero else 'LOẠI C4 (không cải thiện có ý nghĩa thống kê)'}**.",
        f"- **Đối chiếu C1 thuần (`cir_v1_02` vs `cir_v1_01`):** `mAP` tăng từ `{b_res['overall']['mAP']:.4f}` lên `{results_by_cfg['cir_v1_02']['primary_120']['overall']['mAP']:.4f}` (`ΔmAP = {results_by_cfg['cir_v1_02']['primary_120']['delta_mAP_vs_cir_v1_01']:+.4f}`, 95% CI `{results_by_cfg['cir_v1_02']['primary_120']['delta_CI95_vs_cir_v1_01']}`).",
        "",
        "## 4. Độ trễ C.9 cho Luồng `CIR catalog` và `CIR upload` (5 run × 200 truy vấn, warmup 20)",
        "",
        "| Luồng | Config ID | p50 (ms) | p95 median ± IQR (ms) | p99 (ms) | Đạt `p95 < 200 ms` | Cảnh báo `p99 upload > 200 ms` |",
        "|---|---|---:|---:|---:|:---:|:---:|",
    ])
    for s_name in ("cir_catalog", "cir_upload"):
        for cid, lstat in latency_c9[s_name].items():
            ok_sla = "✅" if lstat["sla_p95_under_200ms"] else "❌"
            warn_p99 = "⚠️ CÓ (`> 200 ms`)" if lstat["warn_upload_p99_over_200ms"] else "Không"
            md.append(
                f"| `{s_name}` | `{cid}` | {lstat['p50_ms']:.2f} | **{lstat['p95_median_ms']:.2f} ± {lstat['p95_iqr_ms']:.2f}** | "
                f"{lstat['p99_ms']:.2f} | {ok_sla} | {warn_p99} |"
            )

    (PROJECT_ROOT / "reports/phase5c/step5c_cir_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"CIR Winner: {winner_cid} | mAP={w_res['overall']['mAP']:.4f} | SHA256={out_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
