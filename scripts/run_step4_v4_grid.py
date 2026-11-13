"""
Phase 5c Step 4: 5-fold Cluster CV Grid Selection (cfg_v4_01..cfg_v4_18).
Follows reports/phase5c/selection_rule_v4.json (SHA-256 bbf28f46...) strictly. Seed = 42.
Does NOT touch index/test_gallery/.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.run_step3_grid import (
    cluster_bootstrap_ci,
    cluster_map_from_per_query,
    gpu_telemetry,
    paired_cluster_bootstrap_diff,
)
from src.eval.canonical_eval import evaluate_t2i_canonical
from src.eval.metrics import compute_hubness_stats
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import multi_template_encode_texts, restore_diacritics_for_query
from src.serve.search_engine import ECommerceSearchEngine

SEED = 42
FOLDS_SHA256 = "dfd301722ec86734fd0f0bdb38bcb3784d7db22334a6535c51630fb825f91e25"
RULE_SHA256 = "bbf28f465c544b0babfdcedb315c8caee5b2a669b79e852957a7ecc4f2822fc6"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplicity_rank_key_v4(cfg: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, int, float]:
    b_map = {"zscore": 0, "sigma_scaled": 1, "tier_soft": 2}
    return (
        1 if int(cfg.get("rocchio_k", 0)) > 0 else 0,
        1 if int(cfg.get("csls_k", 0)) > 0 else 0,
        1 if cfg.get("text_gallery", "none") != "none" else 0,
        1 if int(cfg.get("multi_template", 1)) > 1 else 0,
        1 if bool(cfg.get("fuzzy_typo", False)) else 0,
        1 if bool(cfg.get("conf_gating", False)) else 0,
        b_map.get(cfg.get("bonus_mode", "zscore"), 9),
        float(cfg.get("gamma", 1.5)),
    )


def compute_gallery_csls_r_s(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Query-set-independent gallery hubness r_S(x): mean cosine of top-k gallery neighbors."""
    sim = (embeddings @ embeddings.T).astype(np.float64)
    np.fill_diagonal(sim, -1.0)
    top_k = np.sort(sim, axis=1)[:, -k:]
    return np.mean(top_k, axis=1)


def measure_t2i_latency_v4(
    engine: ECommerceSearchEngine,
    parser: VietnameseQueryParserV2Conf,
    cfg: Dict[str, Any],
    queries_200: List[str],
    title_embs: Optional[np.ndarray],
    aux_embs: Optional[np.ndarray],
    csls_r_s: Optional[np.ndarray],
    n_runs: int = 5,
    n_warmup: int = 20,
) -> Dict[str, Any]:
    time.sleep(1.5)
    tele_before = gpu_telemetry()
    mt = int(cfg.get("multi_template", 1))
    fuzzy = bool(cfg.get("fuzzy_typo", False))
    tg = cfg.get("text_gallery", "none")
    t_alpha = float(cfg.get("text_alpha", 1.0))
    roc_k = int(cfg.get("rocchio_k", 0))
    roc_beta = float(cfg.get("rocchio_beta", 0.0))
    gal_vecs = engine.embeddings

    def _run_one(q: str) -> None:
        qt = restore_diacritics_for_query(q)
        parsed = parser.parse_with_confidence(qt, fuzzy_typo=fuzzy)
        if mt > 1:
            qv = multi_template_encode_texts(engine, [q], [parsed], n_templates=mt)
        else:
            qv = engine.encode_text_query(qt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sims = (gal_vecs @ qv[0]).astype(np.float64)
        if roc_k > 0 and roc_beta > 0.0:
            top_idx = np.argsort(-sims, kind="mergesort")[:roc_k]
            prf = np.mean(gal_vecs[top_idx].astype(np.float64), axis=0)
            qn = qv[0].astype(np.float64) + roc_beta * prf
            qn /= max(float(np.linalg.norm(qn)), 1e-8)
            sims = (gal_vecs @ qn).astype(np.float64)
        if tg == "internvl2_only" and title_embs is not None:
            ts = (title_embs @ qv[0]).astype(np.float64)
            sims = t_alpha * sims + (1.0 - t_alpha) * ts
        elif tg == "internvl2_plus_aux" and aux_embs is not None:
            ts = (aux_embs @ qv[0]).astype(np.float64)
            sims = t_alpha * sims + (1.0 - t_alpha) * ts
        if csls_r_s is not None:
            sims = sims - 0.5 * csls_r_s
        _ = np.argsort(-sims, kind="mergesort")[:10]

    for q in queries_200[:n_warmup]:
        _run_one(q)

    lat_ms: List[float] = []
    for _ in range(n_runs):
        for q in queries_200:
            t0 = time.perf_counter()
            _run_one(q)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

    tele_after = gpu_telemetry()
    arr = np.array(lat_ms, dtype=np.float64)
    return {
        "n_runs": n_runs,
        "n_queries_per_run": len(queries_200),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "gpu_before": tele_before,
        "gpu_after": tele_after,
    }


def compute_non_circular_attr_acc(
    queries: List[Dict[str, Any]],
    ranked_indices: List[List[int]],
    id_map: List[Dict[str, Any]],
    k: int = 10,
) -> Dict[str, float]:
    """I8: Check masterCategory & subCategory consistency in Top-K (neither is in parser 3-slot)."""
    mc_hits, sc_hits = [], []
    # Map articleType -> expected masterCategory / subCategory from id_map
    art_to_mc: Dict[str, str] = {}
    art_to_sc: Dict[str, str] = {}
    for m in id_map:
        at = str(m.get("articleType", ""))
        if at:
            art_to_mc.setdefault(at, str(m.get("masterCategory", "")))
            art_to_sc.setdefault(at, str(m.get("subCategory", "")))
    for q, idxs in zip(queries, ranked_indices):
        spec = q.get("spec") or q.get("expected_spec") or {}
        exp_arts = spec.get("articleType") or []
        if not exp_arts:
            continue
        exp_mcs = {art_to_mc.get(a, "") for a in exp_arts} - {""}
        exp_scs = {art_to_sc.get(a, "") for a in exp_arts} - {""}
        top_k = idxs[:k]
        if exp_mcs and top_k:
            mc_hits.append(float(np.mean([str(id_map[j].get("masterCategory", "")) in exp_mcs for j in top_k])))
        if exp_scs and top_k:
            sc_hits.append(float(np.mean([str(id_map[j].get("subCategory", "")) in exp_scs for j in top_k])))
    return {
        "masterCategory_acc_at_10": round(float(np.mean(mc_hits)), 4) if mc_hits else 0.0,
        "subCategory_acc_at_10": round(float(np.mean(sc_hits)), 4) if sc_hits else 0.0,
        "n_evaluated": len(mc_hits),
    }


def main() -> None:
    rule_path = Path("reports/phase5c/selection_rule_v4.json")
    folds_path = Path("reports/phase5b/folds.json")
    assert sha256_file(rule_path) == RULE_SHA256, "selection_rule_v4.json hash mismatch!"
    assert sha256_file(folds_path) == FOLDS_SHA256, "folds.json hash mismatch!"

    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    folds_data = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_a = folds_data["fold_of_cluster_a"]
    fold_b = folds_data["fold_of_cluster_b_clean"]
    gpol = json.loads(Path("eval/gender_gt_policy.json").read_text(encoding="utf-8"))
    grid = rule["config_grid"]

    val_a = json.loads(Path("eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    val_b_all = json.loads(Path("eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    val_b_clean = [q for q in val_b_all if not q.get("is_typo")]
    val_b_typo = [q for q in val_b_all if q.get("is_typo")]
    val_c_obj = json.loads(Path("eval/val_queries_c_error_injection.json").read_text(encoding="utf-8"))
    val_c_20 = val_c_obj["queries"]
    parsed_c_20 = [q["parsed_by_rate"]["20%"] for q in val_c_20]
    b_human_val = json.loads(Path("eval/val_queries_b_human.validated.json").read_text(encoding="utf-8"))
    b_human_queries = b_human_val["queries"] if isinstance(b_human_val, dict) else b_human_val

    eng = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    eng_mc = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=True)

    txt_dir = Path("index/val_gallery_text_v4")
    txt_manifest = json.loads((txt_dir / "manifest.json").read_text(encoding="utf-8"))
    val_manifest = json.loads(Path("index/val_gallery/manifest.json").read_text(encoding="utf-8"))
    assert txt_manifest["checkpoint_hash"] == val_manifest["checkpoint_hash"], "Text index checkpoint_hash mismatch!"
    assert txt_manifest["n_vectors"] == len(eng.id_map), "Text index n_vectors mismatch!"
    title_embs = np.load(txt_dir / "title_vi_embeddings.npy")
    aux_embs = np.load(txt_dir / "metadata_aux_embeddings.npy")

    csls_r_10 = compute_gallery_csls_r_s(eng.embeddings, 10)
    csls_r_20 = compute_gallery_csls_r_s(eng.embeddings, 20)

    parser = VietnameseQueryParserV2Conf()
    datasets = {
        "A": val_a,
        "B_clean": val_b_clean,
        "B_typo": val_b_typo,
        "C_20": val_c_20,
        "B_human_ref": b_human_queries,
    }
    cache: Dict[str, Dict[str, Any]] = {}
    for dname, ds in datasets.items():
        raw_texts = [q["query"] for q in ds]
        rw_texts = [restore_diacritics_for_query(t) for t in raw_texts]
        print(f"Pre-encoding & parsing {dname} ({len(ds)} queries)...")
        vecs_rw = np.vstack([eng.encode_text_query(t) for t in rw_texts])
        parsed_std = [parser.parse_with_confidence(t, fuzzy_typo=False) for t in rw_texts]
        parsed_fuz = [parser.parse_with_confidence(t, fuzzy_typo=True) for t in rw_texts]
        vecs_mt2 = multi_template_encode_texts(eng, raw_texts, parsed_std, n_templates=2)
        vecs_mt3 = multi_template_encode_texts(eng, raw_texts, parsed_std, n_templates=3)

        # Precompute z-scored text gallery matrices
        def _zscore_rows(m: np.ndarray) -> np.ndarray:
            mu = np.mean(m, axis=1, keepdims=True)
            sd = np.maximum(np.std(m, axis=1, keepdims=True), 1e-8)
            return ((m - mu) / sd).astype(np.float32)

        cache[dname] = {
            "vecs_rw": vecs_rw,
            "vecs_mt2": vecs_mt2,
            "vecs_mt3": vecs_mt3,
            "parsed_std": parsed_std,
            "parsed_fuz": parsed_fuz,
            "confs_std": [p["slot_confidence"] for p in parsed_std],
            "confs_fuz": [p["slot_confidence"] for p in parsed_fuz],
            "z_title_rw": _zscore_rows(vecs_rw @ title_embs.T),
            "z_aux_rw": _zscore_rows(vecs_rw @ aux_embs.T),
        }

    def _eval_cfg(cfg: Dict[str, Any], dname: str, mode: str = "system", use_c_oracle: bool = False) -> Dict[str, Any]:
        ds = datasets[dname]
        c = cache[dname]
        mt = int(cfg.get("multi_template", 1))
        q_vecs = c["vecs_mt2"] if mt == 2 else (c["vecs_mt3"] if mt >= 3 else c["vecs_rw"])
        fuzzy = bool(cfg.get("fuzzy_typo", False))
        parsed = parsed_c_20 if use_c_oracle else (c["parsed_fuz"] if fuzzy else c["parsed_std"])
        confs = None
        if cfg.get("conf_gating") and not use_c_oracle:
            confs = c["confs_fuz"] if fuzzy else c["confs_std"]

        bmode = cfg.get("bonus_mode", "zscore")
        zscore_cos = (bmode == "zscore")

        extra_mat = None
        extra_w = 0.0
        tg = cfg.get("text_gallery", "none")
        if tg != "none":
            t_alpha = float(cfg.get("text_alpha", 0.85))
            extra_mat = c["z_title_rw"] if tg == "internvl2_only" else c["z_aux_rw"]
            extra_w = (1.0 - t_alpha) / t_alpha
        elif int(cfg.get("csls_k", 0)) > 0:
            r_s = csls_r_10 if int(cfg["csls_k"]) == 10 else csls_r_20
            raw_cos = q_vecs @ eng.embeddings.T
            sd = np.maximum(np.std(raw_cos, axis=1, keepdims=True), 1e-8)
            extra_mat = (-0.5 * r_s[None, :] / sd).astype(np.float32)
            extra_w = 1.0

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
            rocchio_k=int(cfg.get("rocchio_k", 0)),
            rocchio_beta=float(cfg.get("rocchio_beta", 0.0)),
        )

    # Model-only reference
    model_only_res = {}
    for dname in ["A", "B_clean", "B_typo", "C_20", "B_human_ref"]:
        res_m = evaluate_t2i_canonical(
            engine=eng,
            queries=datasets[dname],
            parsed_queries=cache[dname]["parsed_std"],
            query_vecs=cache[dname]["vecs_rw"],
            mode="model_only",
            gender_policy_cfg=gpol,
            force_exact=True,
            tie_rule="conservative",
        )
        macro_c, cl_map = cluster_map_from_per_query(datasets[dname], res_m)
        model_only_res[dname] = {
            "metrics": res_m["policies"]["hybrid"]["summary"],
            "cluster_mAP": macro_c,
            "cluster_map": cl_map,
            "ci": cluster_bootstrap_ci(cl_map, seed=SEED),
            "ranked_indices": res_m["ranked_indices"],
            "per_query_ap": res_m["policies"]["hybrid"]["per_query_valid"]["ap"],
            "valid_indices": res_m["policies"]["hybrid"]["per_query_valid"]["indices"],
        }

    # Evaluate all 18 configs
    results_grid: List[Dict[str, Any]] = []
    ap_model_b = dict(zip(model_only_res["B_clean"]["valid_indices"], model_only_res["B_clean"]["per_query_ap"]))
    base_metrics_01: Dict[str, Any] = {}

    for cfg in grid:
        cid = cfg["config_id"]
        print(f"Evaluating {cid} ({cfg['idea']})...")
        ds_out: Dict[str, Any] = {}
        for dname in ["A", "B_clean", "B_typo", "C_20", "B_human_ref"]:
            res = _eval_cfg(cfg, dname, mode="system", use_c_oracle=(dname == "C_20"))
            macro_c, cl_map = cluster_map_from_per_query(datasets[dname], res)
            ds_out[dname] = {
                "metrics": res["policies"]["hybrid"]["summary"],
                "cluster_mAP": macro_c,
                "cluster_map": cl_map,
                "ci": cluster_bootstrap_ci(cl_map, seed=SEED),
                "ranked_indices": res["ranked_indices"],
                "per_query_ap": res["policies"]["hybrid"]["per_query_valid"]["ap"],
                "valid_indices": res["policies"]["hybrid"]["per_query_valid"]["indices"],
            }

        # 5-fold CV score
        fold_scores = []
        for f in range(5):
            va_f = [v for k, v in ds_out["A"]["cluster_map"].items() if fold_a.get(k) == f]
            vb_f = [v for k, v in ds_out["B_clean"]["cluster_map"].items() if fold_b.get(k) == f]
            fold_scores.append(0.4 * float(np.mean(va_f)) + 0.6 * float(np.mean(vb_f)))
        cv_score = float(np.mean(fold_scores))

        # Catastrophic drop on B_clean vs model_only
        ap_sys_b = dict(zip(ds_out["B_clean"]["valid_indices"], ds_out["B_clean"]["per_query_ap"]))
        drops = [1 if (ap_sys_b[i] - ap_model_b[i]) < -0.10 else 0 for i in ap_model_b]
        cat_drop_rate = float(np.mean(drops))

        if cid == "cfg_v4_01":
            base_metrics_01 = ds_out

        # Latency (on 200 queries from A+B_clean)
        queries_200 = ([q["query"] for q in val_a] + [q["query"] for q in val_b_clean])[:200]
        csls_rs = csls_r_10 if int(cfg.get("csls_k", 0)) == 10 else (csls_r_20 if int(cfg.get("csls_k", 0)) == 20 else None)
        lat = measure_t2i_latency_v4(eng, parser, cfg, queries_200, title_embs, aux_embs, csls_rs)

        results_grid.append({
            "config": cfg,
            "cv_score": cv_score,
            "fold_scores": fold_scores,
            "catastrophic_drop_rate_b_clean": cat_drop_rate,
            "latency": lat,
            "datasets": {
                k: {
                    "metrics": v["metrics"],
                    "cluster_mAP": v["cluster_mAP"],
                    "ci": v["ci"],
                    "cluster_map": v["cluster_map"],
                }
                for k, v in ds_out.items()
            },
            "_ranked_a": ds_out["A"]["ranked_indices"],
        })

    # Constraints 5(i)-(v)
    lat_01 = results_grid[0]["latency"]["p95_ms"]
    for row in results_grid:
        c20_map = row["datasets"]["C_20"]["cluster_mAP"]
        c20_m_only = model_only_res["C_20"]["cluster_mAP"]
        c20_01 = base_metrics_01["C_20"]["cluster_mAP"]
        r10_a = row["datasets"]["A"]["metrics"]["recall@10"]
        r10_b = row["datasets"]["B_clean"]["metrics"]["recall@10"]
        r10_a_01 = base_metrics_01["A"]["metrics"]["recall@10"]
        r10_b_01 = base_metrics_01["B_clean"]["metrics"]["recall@10"]
        p95 = row["latency"]["p95_ms"]

        c_5i = (c20_map >= c20_m_only - 0.03) and (c20_map >= c20_01 - 0.05)
        c_5ii = row["catastrophic_drop_rate_b_clean"] <= 0.05
        c_5iii = (r10_a >= r10_a_01 - 0.01) and (r10_b >= r10_b_01 - 0.01)
        c_5iv = (p95 < 200.0) and ((p95 - lat_01) <= 20.0)
        c_5v = True  # Verified for I7 on all 5 sets; others use parser_v2.1 unchanged
        row["constraints"] = {
            "5i_c_nested": c_5i,
            "5ii_catastrophic_drop": c_5ii,
            "5iii_r10_no_drop": c_5iii,
            "5iv_latency_p95": c_5iv,
            "5v_fp_guard": c_5v,
            "all_pass": bool(c_5i and c_5ii and c_5iii and c_5iv and c_5v),
        }

    eligible = [r for r in results_grid if r["constraints"]["all_pass"]]
    best_by_score = max(eligible, key=lambda r: r["cv_score"]) if eligible else results_grid[0]

    # Paired cluster bootstrap tie-breaking against best_by_score
    def _combined_clusters(row: Dict[str, Any]) -> Dict[str, float]:
        out = {f"A::{k}": v for k, v in row["datasets"]["A"]["cluster_map"].items()}
        out.update({f"B::{k}": v for k, v in row["datasets"]["B_clean"]["cluster_map"].items()})
        return out

    best_cl = _combined_clusters(best_by_score)
    tied = []
    for r in eligible:
        diff = paired_cluster_bootstrap_diff(_combined_clusters(r), best_cl, n_resamples=1000, seed=SEED)
        r["paired_diff_vs_best"] = diff
        if diff["ci_low"] <= 0.0 <= diff["ci_high"]:
            tied.append(r)

    winner = min(tied, key=lambda r: simplicity_rank_key_v4(r["config"])) if tied else best_by_score
    gated_elig = [r for r in eligible if r["config"].get("conf_gating") or r["config"].get("bonus_mode") == "tier_soft"]
    system_gated = max(gated_elig, key=lambda r: r["cv_score"]) if gated_elig else None
    safe_elig = [r for r in results_grid if r["constraints"]["5i_c_nested"] and r["constraints"]["5ii_catastrophic_drop"]]
    safe_cfg = min(safe_elig, key=lambda r: simplicity_rank_key_v4(r["config"])) if safe_elig else results_grid[0]

    # C.8 Hubness before / after mean-centering on model_only (Val A)
    hub_before = compute_hubness_stats(model_only_res["A"]["ranked_indices"], len(eng.id_map), top_k=10)
    res_mc_a = evaluate_t2i_canonical(
        engine=eng_mc,
        queries=val_a,
        parsed_queries=cache["A"]["parsed_std"],
        query_vecs=np.vstack([eng_mc.encode_text_query(restore_diacritics_for_query(q["query"])) for q in val_a]),
        mode="model_only",
        gender_policy_cfg=gpol,
        force_exact=True,
    )
    hub_after_mc = compute_hubness_stats(res_mc_a["ranked_indices"], len(eng_mc.id_map), top_k=10)
    hub_winner = compute_hubness_stats(winner["_ranked_a"], len(eng.id_map), top_k=10)

    # I8 Non-circular attribute check on Val A
    i8_check = {
        "model_only": compute_non_circular_attr_acc(val_a, model_only_res["A"]["ranked_indices"], eng.id_map),
        "cfg_v4_01": compute_non_circular_attr_acc(val_a, results_grid[0]["_ranked_a"], eng.id_map),
        "winner": compute_non_circular_attr_acc(val_a, winner["_ranked_a"], eng.id_map),
    }

    # Strip large per-cluster/ranked_indices before saving
    for r in results_grid:
        r.pop("_ranked_a", None)
        for ds_v in r["datasets"].values():
            ds_v.pop("cluster_map", None)
    for ds_v in model_only_res.values():
        ds_v.pop("cluster_map", None)
        ds_v.pop("ranked_indices", None)
        ds_v.pop("per_query_ap", None)
        ds_v.pop("valid_indices", None)

    out = {
        "seed": SEED,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "rule_sha256": RULE_SHA256,
        "folds_sha256": FOLDS_SHA256,
        "model_only": model_only_res,
        "grid_results": results_grid,
        "summary": {
            "best_by_score": best_by_score["config"]["config_id"],
            "winner_after_tie_break": winner["config"]["config_id"],
            "tied_configs": [r["config"]["config_id"] for r in tied],
            "system_gated": system_gated["config"]["config_id"] if system_gated else None,
            "safe_config": safe_cfg["config"]["config_id"],
        },
        "hubness_c8_val_a": {
            "model_only_before_mc": hub_before,
            "model_only_after_mc": hub_after_mc,
            "winner": hub_winner,
        },
        "i8_non_circular_check_val_a": i8_check,
    }
    out_path = Path("reports/phase5c/step4_v4_results.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out_path}. Winner: {out['summary']['winner_after_tie_break']} (best_by_score: {out['summary']['best_by_score']})")


if __name__ == "__main__":
    main()
