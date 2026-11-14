"""
Phase 5c Step 5: 5-fold Cluster CV Grid Selection for v5 (cfg_v5_01..cfg_v5_11).
Tuân thủ reports/phase5c/selection_rule_v5.json, seed=42, KHÔNG chạm index/test_gallery/.

Yêu cầu đầu vào trước khi chạy:
- index/template_cache_v5/ (manifest + Hash Guard)
- index/val_gallery_text_v5/ (caption THẬT của val_gallery từ InternVL2-2B + merge_v1; KHÔNG dùng title_vi)
- selection_rule_v5.json đã commit + SHA-256 + tag phase5c-rule-v5
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_step3_grid import (
    cluster_bootstrap_ci,
    cluster_map_from_per_query,
    gpu_telemetry,
    paired_cluster_bootstrap_diff,
    paired_score_bootstrap_ci,
)
from src.eval.canonical_eval import evaluate_t2i_canonical
from src.eval.metrics import compute_hubness_stats
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import multi_template_encode_texts, restore_diacritics_for_query
from src.serve.search_engine import ECommerceSearchEngine
from src.serve.template_cache import TemplateCacheV5, encode_texts_with_template_cache

SEED = 42
FOLDS_SHA256 = "dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplicity_rank_key_v5(cfg: Dict[str, Any]) -> Tuple[int, int, int, int, int, float]:
    tg_map = {"none": 0, "internvl2_only": 1, "internvl2_plus_aux": 2}
    i3_map = {"none": 0, "cached": 1, "direct": 2}
    b_map = {"zscore": 0, "tier_soft": 1}
    return (
        tg_map.get(cfg.get("text_gallery", "none"), 9),
        i3_map.get(cfg.get("i3_mode", "none"), 9),
        1 if bool(cfg.get("fuzzy_typo", False)) else 0,
        1 if bool(cfg.get("conf_gating", False)) else 0,
        b_map.get(cfg.get("bonus_mode", "zscore"), 9),
        float(cfg.get("gamma", 1.5)),
    )


def _oracle_parsed_from_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    arts = list(spec.get("articleType") or [])
    gd = spec.get("gender")
    cols = list(spec.get("baseColour") or [])
    return {
        "articleType": arts,
        "gender": gd,
        "baseColour": cols,
        "slot_confidence": {
            "articleType": "high" if arts else "none",
            "gender": "high" if gd else "none",
            "baseColour": "high" if cols else "none",
        },
    }


def measure_c9_interleaved_latency(
    engine: ECommerceSearchEngine,
    parser: VietnameseQueryParserV2Conf,
    tpl_cache: TemplateCacheV5,
    configs: List[Dict[str, Any]],
    queries_200: List[str],
    embs_a: np.ndarray,
    embs_b: np.ndarray,
    n_runs: int = 5,
    n_warmup: int = 20,
    cooldown_sec: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """Protocol C.9: GPU nguội, xen kẽ giữa các cấu hình, báo trung vị + IQR của p95."""
    gal_vecs = engine.embeddings

    def _profile_key(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            cfg.get("i3_mode", "none"),
            bool(cfg.get("fuzzy_typo", False)),
            cfg.get("text_gallery", "none"),
        )

    rep_configs: List[Dict[str, Any]] = []
    seen_profiles: Dict[Tuple[Any, ...], str] = {}
    cid_to_rep: Dict[str, str] = {}
    for cfg in configs:
        pk = _profile_key(cfg)
        cid = cfg["config_id"]
        if pk not in seen_profiles:
            seen_profiles[pk] = cid
            rep_configs.append(cfg)
        cid_to_rep[cid] = seen_profiles[pk]

    def _run_one(cfg: Dict[str, Any], q: str) -> None:
        qt = restore_diacritics_for_query(q)
        fuzzy = bool(cfg.get("fuzzy_typo", False))
        parsed = parser.parse_with_confidence(qt, fuzzy_typo=fuzzy)
        i3_mode = cfg.get("i3_mode", "none")
        if i3_mode == "direct":
            qv = multi_template_encode_texts(engine, [q], [parsed], n_templates=2)
        elif i3_mode == "cached":
            qv, _ = encode_texts_with_template_cache(engine, [q], [parsed], tpl_cache)
        else:
            qv = engine.encode_text_query(qt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sims = (gal_vecs @ qv[0]).astype(np.float64)
        tg = cfg.get("text_gallery", "none")
        t_alpha = float(cfg.get("text_alpha", 1.0))
        if tg == "internvl2_only":
            sims = t_alpha * sims + (1.0 - t_alpha) * (embs_a @ qv[0]).astype(np.float64)
        elif tg == "internvl2_plus_aux":
            sims = t_alpha * sims + (1.0 - t_alpha) * (embs_b @ qv[0]).astype(np.float64)
        _ = np.argsort(-sims, kind="mergesort")[:10]

    per_cfg_p95_runs: Dict[str, List[float]] = {c["config_id"]: [] for c in rep_configs}
    per_cfg_all_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in rep_configs}
    tele_log: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in rep_configs}

    for r_idx in range(n_runs):
        print(f"[C.9 Latency] Run {r_idx + 1}/{n_runs} (per-query interleaved across {len(rep_configs)} profiles)...", flush=True)
        time.sleep(cooldown_sec)
        tele_before = gpu_telemetry()
        for q in queries_200[:n_warmup]:
            for cfg in rep_configs:
                _run_one(cfg, q)
        run_ms_by_cid: Dict[str, List[float]] = {c["config_id"]: [] for c in rep_configs}
        for q_idx, q in enumerate(queries_200):
            order = list(rep_configs) if (r_idx + q_idx) % 2 == 0 else list(reversed(rep_configs))
            for cfg in order:
                cid = cfg["config_id"]
                t0 = time.perf_counter()
                _run_one(cfg, q)
                run_ms_by_cid[cid].append((time.perf_counter() - t0) * 1000.0)
        tele_after = gpu_telemetry()
        for cfg in rep_configs:
            cid = cfg["config_id"]
            rms = run_ms_by_cid[cid]
            per_cfg_p95_runs[cid].append(float(np.percentile(rms, 95)))
            per_cfg_all_ms[cid].extend(rms)
            tele_log[cid].append({"before": tele_before, "after": tele_after})

    out: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        cid = cfg["config_id"]
        rcid = cid_to_rep[cid]
        p95s = per_cfg_p95_runs[rcid]
        arr = np.array(per_cfg_all_ms[rcid], dtype=np.float64)
        q25, q75 = np.percentile(p95s, [25, 75])
        out[cid] = {
            "profile_rep_config_id": rcid,
            "n_runs": n_runs,
            "n_queries_per_run": len(queries_200),
            "p95_runs_ms": [round(x, 3) for x in p95s],
            "p95_median_ms": round(float(statistics.median(p95s)), 3),
            "p95_iqr_ms": round(float(q75 - q25), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "telemetry": tele_log[rcid],
        }
    return out


def main() -> int:
    rule_path = PROJECT_ROOT / "reports/phase5c/selection_rule_v5.json"
    folds_path = PROJECT_ROOT / "reports/phase5b/folds.json"
    if sha256_file(folds_path) != FOLDS_SHA256:
        raise RuntimeError("folds.json hash mismatch!")

    txt_v5_dir = PROJECT_ROOT / "index/val_gallery_text_v5"
    if not (txt_v5_dir / "manifest.json").exists():
        raise FileNotFoundError(
            "Thiếu index/val_gallery_text_v5/manifest.json! "
            "Theo Bước 12.1.4, phải caption thật toàn bộ val_gallery và chạy scripts/build_val_text_index_v5.py trước."
        )

    tpl_cache_dir = PROJECT_ROOT / "index/template_cache_v5"
    val_manifest = json.loads((PROJECT_ROOT / "index/val_gallery/manifest.json").read_text(encoding="utf-8"))
    tpl_cache = TemplateCacheV5.load(tpl_cache_dir, expected_checkpoint_hash=val_manifest["checkpoint_hash"])

    txt_manifest = json.loads((txt_v5_dir / "manifest.json").read_text(encoding="utf-8"))
    if txt_manifest["checkpoint_hash"] != val_manifest["checkpoint_hash"]:
        raise RuntimeError("val_gallery_text_v5 checkpoint_hash mismatch!")
    if not txt_manifest.get("internvl2_captions_present"):
        raise RuntimeError("val_gallery_text_v5 phải chứa caption thật InternVL2-2B!")

    embs_a = np.load(txt_v5_dir / "internvl2_only_embeddings.npy")
    embs_b = np.load(txt_v5_dir / "internvl2_plus_aux_embeddings.npy")

    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    rule_sha = sha256_file(rule_path)
    folds_data = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_a = folds_data["fold_of_cluster_a"]
    fold_b = folds_data["fold_of_cluster_b_clean"]
    gpol = json.loads((PROJECT_ROOT / "eval/gender_gt_policy.json").read_text(encoding="utf-8"))
    grid = rule["config_grid"]

    val_a = json.loads((PROJECT_ROOT / "eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    val_b_all = json.loads((PROJECT_ROOT / "eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    val_b_clean = [q for q in val_b_all if not q.get("is_typo")]
    val_b_typo = [q for q in val_b_all if q.get("is_typo")]
    val_c_obj = json.loads((PROJECT_ROOT / "eval/val_queries_c_error_injection.json").read_text(encoding="utf-8"))
    val_c_20 = val_c_obj["queries"]
    parsed_c_20 = [q["parsed_by_rate"]["20%"] for q in val_c_20]

    b_human_v4_val_path = PROJECT_ROOT / "eval/val_queries_b_human_v4.validated.json"
    b_human_v4_path = PROJECT_ROOT / "eval/val_queries_b_human_v4.json"
    if b_human_v4_val_path.exists():
        bh_raw = json.loads(b_human_v4_val_path.read_text(encoding="utf-8"))
        b_human_queries = bh_raw["queries"] if isinstance(bh_raw, dict) else bh_raw
        has_ai = any(q.get("source") == "ai_written" for q in b_human_queries)
        b_human_label = (
            "B-human-v4 (kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối)"
            if has_ai
            else "B-human-v4 (độc lập, locked)"
        )
        b_human_is_ref_only = False
    elif b_human_v4_path.exists():
        bh_raw = json.loads(b_human_v4_path.read_text(encoding="utf-8"))
        b_human_queries = bh_raw["queries"] if isinstance(bh_raw, dict) else bh_raw
        b_human_label = "B-human-v4 (kiểm chứng ngoài mẫu, chưa độc lập tuyệt đối)"
        b_human_is_ref_only = False
    else:
        bh_raw = json.loads((PROJECT_ROOT / "eval/val_queries_b_human.validated.json").read_text(encoding="utf-8"))
        b_human_queries = bh_raw["queries"] if isinstance(bh_raw, dict) else bh_raw
        b_human_label = "B-human cũ (THAM KHẢO — thiếu B-human-v4)"
        b_human_is_ref_only = True

    eng = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    parser = VietnameseQueryParserV2Conf()

    datasets = {
        "A": val_a,
        "B_clean": val_b_clean,
        "B_typo": val_b_typo,
        "C_20": val_c_20,
        "B_human": b_human_queries,
    }

    cache: Dict[str, Dict[str, Any]] = {}
    fallback_stats: Dict[str, Any] = {}
    for dname, ds in datasets.items():
        print(f"[Pre-encode] Dataset {dname} ({len(ds)} queries)...", flush=True)
        raw_texts = [q["query"] for q in ds]
        rw_texts = [restore_diacritics_for_query(t) for t in raw_texts]
        vecs_rw = np.vstack([eng.encode_text_query(t) for t in rw_texts])
        parsed_std = [parser.parse_with_confidence(t, fuzzy_typo=False) for t in rw_texts]
        parsed_fuz = [parser.parse_with_confidence(t, fuzzy_typo=True) for t in rw_texts]
        vecs_mt2 = multi_template_encode_texts(eng, raw_texts, parsed_std, n_templates=2)
        vecs_cached_std, fb_std = encode_texts_with_template_cache(eng, raw_texts, parsed_std, tpl_cache)
        vecs_cached_fuz, fb_fuz = encode_texts_with_template_cache(eng, raw_texts, parsed_fuz, tpl_cache)
        fallback_stats[dname] = {"std": fb_std, "fuzzy": fb_fuz}

        def _zscore_rows(m: np.ndarray) -> np.ndarray:
            mu = np.mean(m, axis=1, keepdims=True)
            sd = np.maximum(np.std(m, axis=1, keepdims=True), 1e-8)
            return ((m - mu) / sd).astype(np.float32)

        cache[dname] = {
            "vecs_rw": vecs_rw,
            "vecs_mt2": vecs_mt2,
            "vecs_cached_std": vecs_cached_std,
            "vecs_cached_fuz": vecs_cached_fuz,
            "parsed_std": parsed_std,
            "parsed_fuz": parsed_fuz,
            "confs_std": [p["slot_confidence"] for p in parsed_std],
            "confs_fuz": [p["slot_confidence"] for p in parsed_fuz],
            "z_a_rw": _zscore_rows(vecs_rw @ embs_a.T),
            "z_b_rw": _zscore_rows(vecs_rw @ embs_b.T),
            "z_a_cached": _zscore_rows(vecs_cached_std @ embs_a.T),
            "z_b_cached": _zscore_rows(vecs_cached_std @ embs_b.T),
        }

    def _eval_cfg(
        cfg: Dict[str, Any],
        dname: str,
        mode: str = "system",
        use_c_oracle: bool = False,
        oracle_spec: bool = False,
    ) -> Dict[str, Any]:
        ds = datasets[dname]
        c = cache[dname]
        fuzzy = bool(cfg.get("fuzzy_typo", False))
        i3_mode = cfg.get("i3_mode", "none")

        if i3_mode == "direct":
            q_vecs = c["vecs_mt2"]
        elif i3_mode == "cached":
            q_vecs = c["vecs_cached_fuz"] if fuzzy else c["vecs_cached_std"]
        else:
            q_vecs = c["vecs_rw"]

        if oracle_spec:
            parsed = [_oracle_parsed_from_spec(q.get("spec") or q.get("expected_spec") or {}) for q in ds]
            confs = [p["slot_confidence"] for p in parsed] if cfg.get("conf_gating") else None
        elif use_c_oracle:
            parsed = parsed_c_20
            confs = None
        else:
            parsed = c["parsed_fuz"] if fuzzy else c["parsed_std"]
            confs = (c["confs_fuz"] if fuzzy else c["confs_std"]) if cfg.get("conf_gating") else None

        bmode = cfg.get("bonus_mode", "zscore")
        zscore_cos = (bmode == "zscore")
        tg = cfg.get("text_gallery", "none")
        t_alpha = float(cfg.get("text_alpha", 1.0))
        extra_mat = None
        extra_w = 0.0
        if tg == "internvl2_only":
            extra_mat = c["z_a_cached"] if i3_mode == "cached" else c["z_a_rw"]
            extra_w = (1.0 - t_alpha) / t_alpha
        elif tg == "internvl2_plus_aux":
            extra_mat = c["z_b_cached"] if i3_mode == "cached" else c["z_b_rw"]
            extra_w = (1.0 - t_alpha) / t_alpha

        return evaluate_t2i_canonical(
            engine=eng,
            queries=ds,
            parsed_queries=parsed,
            query_vecs=q_vecs,
            mode=mode,
            gamma=float(cfg.get("gamma", 1.5)),
            gender_policy_cfg=gpol,
            force_exact=True,
            tie_rule="conservative",
            zscore_cosine=zscore_cos,
            bonus_mode=bmode,
            slot_confidences=confs,
            extra_score_matrix=extra_mat,
            extra_score_weight=extra_w,
        )

    def _unpack_eval(ds: List[Dict[str, Any]], res: Dict[str, Any]) -> Dict[str, Any]:
        hyb = res["policies"]["hybrid"]
        pqv = hyb["per_query_valid"]
        if ds and "spec_cluster_id" in ds[0]:
            macro_c, cl_map = cluster_map_from_per_query(ds, res)
        else:
            macro_c, cl_map = hyb["summary"]["mAP"], {}
        ap_by_idx = dict(zip(pqv["indices"], pqv["ap"]))
        return {
            "summary": hyb["summary"],
            "macro_cluster_mAP": macro_c,
            "cluster_map": cl_map,
            "valid_indices": pqv["indices"],
            "ap_list": pqv["ap"],
            "ap_by_idx": ap_by_idx,
            "ranked_indices": res["ranked_indices"],
        }

    # Model-only baseline
    model_a = _unpack_eval(val_a, _eval_cfg(grid[0], "A", mode="model_only"))
    model_bc = _unpack_eval(val_b_clean, _eval_cfg(grid[0], "B_clean", mode="model_only"))
    model_c20 = _unpack_eval(val_c_20, _eval_cfg(grid[0], "C_20", mode="model_only"))
    model_bh = _unpack_eval(b_human_queries, _eval_cfg(grid[0], "B_human", mode="model_only"))

    results_by_cfg: Dict[str, Any] = {}
    for cfg in grid:
        cid = cfg["config_id"]
        print(f"[Grid Eval] Evaluating {cid}...", flush=True)
        res_a = _unpack_eval(val_a, _eval_cfg(cfg, "A"))
        res_bc = _unpack_eval(val_b_clean, _eval_cfg(cfg, "B_clean"))
        res_bt = _unpack_eval(val_b_typo, _eval_cfg(cfg, "B_typo"))
        res_c20 = _unpack_eval(val_c_20, _eval_cfg(cfg, "C_20", use_c_oracle=True))
        res_bh = _unpack_eval(b_human_queries, _eval_cfg(cfg, "B_human"))

        # 5-fold CV score
        fold_scores = []
        for f in range(5):
            va_f = [val for k, val in res_a["cluster_map"].items() if fold_a.get(k) == f]
            vb_f = [val for k, val in res_bc["cluster_map"].items() if fold_b.get(k) == f]
            ma = float(np.mean(va_f)) if va_f else 0.0
            mb = float(np.mean(vb_f)) if vb_f else 0.0
            fold_scores.append(0.4 * ma + 0.6 * mb)

        cv_score = float(np.mean(fold_scores))
        cat_drop_bc = float(
            np.mean([
                (res_bc["ap_by_idx"][i] - model_bc["ap_by_idx"][i]) < -0.10
                for i in res_bc["valid_indices"]
                if i in model_bc["ap_by_idx"]
            ])
        )
        hub_bc = compute_hubness_stats(res_bc["ranked_indices"], len(eng.id_map), top_k=10)

        results_by_cfg[cid] = {
            "config": cfg,
            "cv_score": round(cv_score, 4),
            "fold_scores": [round(x, 4) for x in fold_scores],
            "A_mAP_cluster": res_a["macro_cluster_mAP"],
            "A_R10": res_a["summary"]["recall@10"],
            "Bc_mAP_cluster": res_bc["macro_cluster_mAP"],
            "Bc_R10": res_bc["summary"]["recall@10"],
            "Bt_mAP_cluster": res_bt["macro_cluster_mAP"],
            "C20_mAP": res_c20["summary"]["mAP"],
            "cat_drop_Bc": round(cat_drop_bc, 4),
            "hubness_Bc": hub_bc,
            "B_human_mAP": res_bh["summary"]["mAP"],
            "B_human_R1": res_bh["summary"]["recall@1"],
            "B_human_R10": res_bh["summary"]["recall@10"],
            "_u_A": res_a,
            "_u_Bc": res_bc,
            "_u_Bh": res_bh,
        }

    # C.9 latency measurement across all configs
    queries_200 = [q["query"] for q in val_a[:200]]
    lat_results = measure_c9_interleaved_latency(
        eng, parser, tpl_cache, grid, queries_200, embs_a, embs_b, n_runs=5, n_warmup=20
    )

    # Parser FP check (5(v)): compare parser_v2_conf (std vs fuzzy) on A+B_clean+B_human
    fp_check_queries = val_a + val_b_clean + b_human_queries
    fp_std, fp_fuz = 0, 0
    for q in fp_check_queries:
        spec = q.get("spec") or {}
        gt_arts = set(spec.get("articleType") or [])
        gt_gd = spec.get("gender")
        gt_cols = set(spec.get("baseColour") or [])
        qt = restore_diacritics_for_query(q["query"])
        pr_std = parser.parse_with_confidence(qt, fuzzy_typo=False)
        pr_fuz = parser.parse_with_confidence(qt, fuzzy_typo=True)
        for pr, is_fuz in ((pr_std, False), (pr_fuz, True)):
            has_fp = False
            if pr["articleType"] and gt_arts and not (set(pr["articleType"]) & gt_arts):
                has_fp = True
            if pr["gender"] and gt_gd and pr["gender"] != gt_gd:
                has_fp = True
            if pr["baseColour"] and gt_cols and not (set(pr["baseColour"]) & gt_cols):
                has_fp = True
            if has_fp:
                if is_fuz:
                    fp_fuz += 1
                else:
                    fp_std += 1
    fp_rate_std = round(fp_std / len(fp_check_queries), 4)
    fp_rate_fuz = round(fp_fuz / len(fp_check_queries), 4)

    # Evaluate constraints 5(i)-(v) per config
    base_01 = results_by_cfg["cfg_v5_01"]
    base_p95 = lat_results["cfg_v5_01"]["p95_median_ms"]
    for cid, v in results_by_cfg.items():
        cfg = v["config"]
        p95_med = lat_results[cid]["p95_median_ms"]
        fp_rate = fp_rate_fuz if cfg.get("fuzzy_typo") else fp_rate_std
        c_5i = (v["C20_mAP"] - model_c20["summary"]["mAP"]) >= -0.03 and (v["C20_mAP"] - base_01["C20_mAP"]) >= -0.05
        c_5ii = v["cat_drop_Bc"] <= 0.05
        c_5iii = (v["A_R10"] - base_01["A_R10"]) >= -0.01 and (v["Bc_R10"] - base_01["Bc_R10"]) >= -0.01
        c_5iv = (p95_med < 200.0) and ((p95_med - base_p95) <= 20.0)
        c_5v = fp_rate <= fp_rate_std
        v["parser_fp_rate"] = fp_rate
        v["latency_p95_median_ms"] = p95_med
        v["latency_p95_iqr_ms"] = lat_results[cid]["p95_iqr_ms"]
        v["constraints"] = {
            "5i_c20": c_5i,
            "5ii_cat_drop": c_5ii,
            "5iii_r10": c_5iii,
            "5iv_latency": c_5iv,
            "5v_parser_fp": c_5v,
            "all_passed": c_5i and c_5ii and c_5iii and c_5iv and c_5v,
        }

    # Select winner among competes_for_winner=True & all_passed (with 1000x paired cluster bootstrap tie-break)
    competitors = [
        cid for cid, v in results_by_cfg.items()
        if v["config"].get("competes_for_winner", True) and v["constraints"]["all_passed"]
    ]
    if not competitors:
        competitors = [cid for cid, v in results_by_cfg.items() if v["config"].get("competes_for_winner", True)]
    competitors.sort(key=lambda cid: (-results_by_cfg[cid]["cv_score"], simplicity_rank_key_v5(results_by_cfg[cid]["config"])))

    top_cid = competitors[0]
    top_ca = results_by_cfg[top_cid]["_u_A"]["cluster_map"]
    top_cb = results_by_cfg[top_cid]["_u_Bc"]["cluster_map"]
    tied_candidates = [top_cid]
    for cid in competitors[1:]:
        cand_ca = results_by_cfg[cid]["_u_A"]["cluster_map"]
        cand_cb = results_by_cfg[cid]["_u_Bc"]["cluster_map"]
        diff_res = paired_score_bootstrap_ci(top_ca, top_cb, cand_ca, cand_cb, n_resamples=1000, seed=SEED)
        if diff_res["ci_low"] <= 0.0 <= diff_res["ci_high"]:
            tied_candidates.append(cid)

    tied_candidates.sort(key=lambda cid: simplicity_rank_key_v5(results_by_cfg[cid]["config"]))
    winner_cid = tied_candidates[0]

    safe_candidates = [
        cid for cid, v in results_by_cfg.items()
        if v["config"].get("competes_for_winner", True) and v["constraints"]["5i_c20"] and v["constraints"]["5ii_cat_drop"]
    ]
    safe_candidates.sort(key=lambda cid: simplicity_rank_key_v5(results_by_cfg[cid]["config"]))
    safe_cid = safe_candidates[0] if safe_candidates else winner_cid

    # B-human-v4 gate analysis (after selection, NEVER re-select)
    bh_valid_idx = results_by_cfg["cfg_v5_01"]["_u_Bh"]["valid_indices"]
    bh_aps_model = [model_bh["ap_by_idx"][i] for i in bh_valid_idx]

    def _bh_cluster_dict(u_bh: Dict[str, Any]) -> Dict[str, float]:
        by_c: Dict[str, List[float]] = defaultdict(list)
        for i in bh_valid_idx:
            ck = str(
                b_human_queries[i].get("cluster_id")
                or b_human_queries[i].get("spec_cluster_id")
                or b_human_queries[i].get("id")
                or i
            )
            by_c[ck].append(float(u_bh["ap_by_idx"][i]))
        return {k: float(np.mean(v)) for k, v in by_c.items()}

    bh_cdict_base = _bh_cluster_dict(results_by_cfg["cfg_v5_01"]["_u_Bh"])

    def _bh_gate_detail(cid: str) -> Dict[str, Any]:
        u_bh = results_by_cfg[cid]["_u_Bh"]
        aps = [u_bh["ap_by_idx"][i] for i in bh_valid_idx]
        diff_vs_base = paired_cluster_bootstrap_diff(_bh_cluster_dict(u_bh), bh_cdict_base, n_resamples=1000, seed=SEED)
        cat_drop_bh = float(np.mean([(a - m) < -0.10 for a, m in zip(aps, bh_aps_model)]))
        with_col = [a for i, a in zip(bh_valid_idx, aps) if (b_human_queries[i].get("spec") or {}).get("baseColour")]
        no_col = [a for i, a in zip(bh_valid_idx, aps) if not (b_human_queries[i].get("spec") or {}).get("baseColour")]
        by_dia: Dict[str, List[float]] = defaultdict(list)
        for i, a in zip(bh_valid_idx, aps):
            by_dia[str(b_human_queries[i].get("dia_class") or "unknown")].append(a)
        return {
            "mAP": round(float(np.mean(aps)), 4),
            "delta_mAP_vs_cfg_v5_01": round(diff_vs_base["mean"], 4),
            "delta_CI95": [round(diff_vs_base["ci_low"], 4), round(diff_vs_base["ci_high"], 4)],
            "catastrophic_drop_rate": round(cat_drop_bh, 4),
            "mAP_with_colour": round(float(np.mean(with_col)), 4) if with_col else None,
            "n_with_colour": len(with_col),
            "mAP_without_colour": round(float(np.mean(no_col)), 4) if no_col else None,
            "n_without_colour": len(no_col),
            "mAP_by_dia_class": {k: {"n": len(v), "mAP": round(float(np.mean(v)), 4)} for k, v in sorted(by_dia.items())},
        }

    bh_gate = {
        "winner": _bh_gate_detail(winner_cid),
        "safe": _bh_gate_detail(safe_cid),
        "baseline_cfg_v5_01": _bh_gate_detail("cfg_v5_01"),
        "cfg_v5_06_fuzzy": _bh_gate_detail("cfg_v5_06"),
        "cfg_v5_08a_internvl2_only": _bh_gate_detail("cfg_v5_08a"),
        "cfg_v5_08b_internvl2_plus_aux": _bh_gate_detail("cfg_v5_08b"),
    }

    # Check identical metrics across configs (exact 4-dataset and CV+B_human 4-decimal)
    metric_sigs_all = defaultdict(list)
    metric_sigs_cv_bh = defaultdict(list)
    for cid, v in results_by_cfg.items():
        sig_all = (
            round(v["A_mAP_cluster"], 4),
            round(v["Bc_mAP_cluster"], 4),
            round(v["Bt_mAP_cluster"], 4),
            round(v["C20_mAP"], 4),
            round(v["B_human_mAP"], 4),
        )
        sig_cv_bh = (
            round(v["cv_score"], 4),
            round(v["A_mAP_cluster"], 4),
            round(v["Bc_mAP_cluster"], 4),
            round(v["Bt_mAP_cluster"], 4),
            round(v["B_human_mAP"], 4),
        )
        metric_sigs_all[sig_all].append(cid)
        metric_sigs_cv_bh[sig_cv_bh].append(cid)
    identical_groups = {
        "exact_all_datasets_4dec": [cids for cids in metric_sigs_all.values() if len(cids) > 1],
        "identical_on_A_Bc_Bt_Bhuman_4dec": [cids for cids in metric_sigs_cv_bh.values() if len(cids) > 1],
    }

    # Oracle parser on B-clean and B-human for winner
    w_cfg = results_by_cfg[winner_cid]["config"]
    oracle_bc = _unpack_eval(val_b_clean, _eval_cfg(w_cfg, "B_clean", oracle_spec=True))
    oracle_bh = _unpack_eval(b_human_queries, _eval_cfg(w_cfg, "B_human", oracle_spec=True))

    # Hubness before (model_only) and after (winner) on B_clean & B_human (C.8)
    hub_model_bc = compute_hubness_stats(model_bc["ranked_indices"], len(eng.id_map), top_k=10)
    hub_model_bh = compute_hubness_stats(model_bh["ranked_indices"], len(eng.id_map), top_k=10)
    hub_win_bh = compute_hubness_stats(
        _eval_cfg(w_cfg, "B_human")["ranked_indices"], len(eng.id_map), top_k=10
    )

    # Cosine std stats (bonus size vs cosine std)
    cos_stds_a = [float(np.std(cache["A"]["vecs_rw"][i] @ eng.embeddings.T)) for i in range(len(val_a))]

    summary = {
        "rule_sha256": rule_sha,
        "folds_sha256": FOLDS_SHA256,
        "seed": SEED,
        "b_human_label": b_human_label,
        "b_human_is_ref_only": b_human_is_ref_only,
        "fallback_stats": fallback_stats,
        "top_by_cv_score": top_cid,
        "tied_within_95ci": tied_candidates,
        "winner_config_id": winner_cid,
        "safe_config_id": safe_cid,
        "b_human_gate": bh_gate,
        "identical_metric_groups": identical_groups,
        "cosine_std_mean_A": round(float(np.mean(cos_stds_a)), 4),
        "oracle_parser_winner": {
            "Bc_mAP_cluster": oracle_bc["macro_cluster_mAP"],
            "B_human_mAP": oracle_bh["summary"]["mAP"],
        },
        "hubness_c8": {
            "B_clean_model_only": hub_model_bc,
            "B_clean_winner": results_by_cfg[winner_cid]["hubness_Bc"],
            "B_human_model_only": hub_model_bh,
            "B_human_winner": hub_win_bh,
        },
        "model_only": {
            "A_mAP_cluster": model_a["macro_cluster_mAP"],
            "A_R10": model_a["summary"]["recall@10"],
            "Bc_mAP_cluster": model_bc["macro_cluster_mAP"],
            "Bc_R10": model_bc["summary"]["recall@10"],
            "C20_mAP": model_c20["summary"]["mAP"],
            "B_human_mAP": model_bh["summary"]["mAP"],
            "B_human_R1": model_bh["summary"]["recall@1"],
            "B_human_R10": model_bh["summary"]["recall@10"],
        },
        "latency_c9": lat_results,
        "configs": {
            cid: {k: val for k, val in v.items() if not k.startswith("_u_")}
            for cid, v in results_by_cfg.items()
        },
    }

    out_json = PROJECT_ROOT / "reports/phase5c/step5c_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Format reports/phase5c/step5c_selection_report.md
    md_lines = [
        "# Báo cáo Chọn Cấu hình v5 — Pha 5c (`step5c_selection_report`)",
        "",
        f"- **Rule SHA-256:** `{rule_sha}` | **Folds SHA-256:** `{FOLDS_SHA256}` | **Seed:** `{SEED}`",
        f"- **Cổng B-human-v4:** `{b_human_label}` ({len(b_human_queries)} câu)",
        f"- **Tốt nhất (`winner_config_id`):** **`{winner_cid}`** (CV Score `{results_by_cfg[winner_cid]['cv_score']}`, tie-break bootstrap 1.000 lần trong nhóm: `{tied_candidates}`)",
        f"- **An toàn (`safe_config_id`):** **`{safe_cid}`** (đạt `5(i)` & `5(ii)` đơn giản nhất)",
        "",
        "## 1. Bảng So sánh Model thuần vs Hệ thống Soft vs Hệ thống Gated (12 Cấu hình v5)",
        "",
        "| Nhóm / Config | Tranh hạng | CV Score | A mAP(cl) | Bc mAP(cl) | Bt mAP(cl) | C20 mAP | Tụt thảm họa | Parser FP | p95 C.9 (med ± IQR ms) | 5(i)-(v) |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        f"| `model_only` | — | — | {model_a['macro_cluster_mAP']:.4f} | {model_bc['macro_cluster_mAP']:.4f} | — | {model_c20['summary']['mAP']:.4f} | 0.0000 | — | — | — |",
    ]
    for cfg in grid:
        cid = cfg["config_id"]
        v = results_by_cfg[cid]
        grp = "system_gated" if cfg.get("conf_gating") else ("eq_ref" if not cfg.get("competes_for_winner", True) else "system_soft")
        comp = "✅" if cfg.get("competes_for_winner", True) else "❌"
        ok = "✅" if v["constraints"]["all_passed"] else "❌"
        md_lines.append(
            f"| `{cid}` ({grp}) | {comp} | {v['cv_score']:.4f} | {v['A_mAP_cluster']:.4f} | {v['Bc_mAP_cluster']:.4f} | "
            f"{v['Bt_mAP_cluster']:.4f} | {v['C20_mAP']:.4f} | {v['cat_drop_Bc']:.4f} | {v['parser_fp_rate']:.4f} | "
            f"{v['latency_p95_median_ms']:.2f} ± {v['latency_p95_iqr_ms']:.2f} | {ok} |"
        )

    w_bh = bh_gate["winner"]
    md_lines.extend([
        "",
        f"## 2. Cổng Xác nhận `{b_human_label}` (chạy 1 lần sau khi chọn, KHÔNG chọn lại)",
        f"- **`{winner_cid}` (Tốt nhất):** `mAP={w_bh['mAP']}`, `ΔmAP vs cfg_v5_01 = {w_bh['delta_mAP_vs_cfg_v5_01']:+.4f}` "
        f"(95% CI: `[{w_bh['delta_CI95'][0]:+.4f}, {w_bh['delta_CI95'][1]:+.4f}]`), tụt thảm họa `{w_bh['catastrophic_drop_rate']}`.",
        f"- **Tách có màu / không màu:** Có màu (`n={w_bh['n_with_colour']}`) `mAP={w_bh['mAP_with_colour']}` | Không màu (`n={w_bh['n_without_colour']}`) `mAP={w_bh['mAP_without_colour']}`.",
        f"- **Tách theo `dia_class`:** `{w_bh['mAP_by_dia_class']}`.",
        "",
        "## 3. Phân tích Chẩn đoán (Oracle Parser, Hubness C.8, Bonus vs Cosine Std, Fallback & Trùng lặp)",
        f"- **Oracle Parser (`{winner_cid}`):** `B-clean mAP(cl)={oracle_bc['macro_cluster_mAP']:.4f}` (vs parser `{results_by_cfg[winner_cid]['Bc_mAP_cluster']:.4f}`), "
        f"`B-human-v4 mAP={oracle_bh['summary']['mAP']:.4f}` (vs parser `{w_bh['mAP']:.4f}`).",
        f"- **Hubness C.8 (`B-clean`):** Model thuần `{hub_model_bc}` → Winner `{results_by_cfg[winner_cid]['hubness_Bc']}`.",
        f"- **Cosine std trung bình trên Val A:** `{summary['cosine_std_mean_A']}` (so sánh với biên độ bonus `tier_soft` / `gamma`).",
        f"- **I3 Cache Fallback:** `{fallback_stats}`.",
        f"- **Các cấu hình trùng số liệu:** `{identical_groups}`.",
    ])

    (PROJECT_ROOT / "reports/phase5c/step5c_selection_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Winner: {winner_cid} | Safe: {safe_cid} | CV Score: {results_by_cfg[winner_cid]['cv_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
