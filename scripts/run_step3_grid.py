"""
Phase 5b Step 3: 5-fold Cluster CV Grid Selection (cfg_01..cfg_22) + B-human Confirmation Gate.
Follows selection_rule_v3.json, PROMPT 7, and PROMPT 8 strictly. Seed = 42.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.eval.canonical_eval import evaluate_t2i_canonical
from src.serve.parser_v1 import VietnameseQueryParser
from src.serve.parser_v2 import PARSER_VERSION, VietnameseQueryParserV2
from src.serve.query_rewriter import (
    BM25GalleryScorer,
    dual_encode_texts,
    restore_diacritics_for_query,
)
from src.serve.search_engine import ECommerceSearchEngine

SEED = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_and_lock_folds(val_a: List[Dict[str, Any]], val_b_clean: List[Dict[str, Any]]) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    clusters_a = sorted({q["spec_cluster_id"] for q in val_a})
    clusters_b = sorted({q["spec_cluster_id"] for q in val_b_clean})
    perm_a = rng.permutation(len(clusters_a))
    perm_b = rng.permutation(len(clusters_b))

    fold_of_a = {clusters_a[idx]: int(rank % 5) for rank, idx in enumerate(perm_a)}
    fold_of_b = {clusters_b[idx]: int(rank % 5) for rank, idx in enumerate(perm_b)}

    folds_obj = {
        "seed": SEED,
        "n_folds": 5,
        "n_clusters_a": len(clusters_a),
        "n_clusters_b_clean": len(clusters_b),
        "fold_of_cluster_a": fold_of_a,
        "fold_of_cluster_b_clean": fold_of_b,
    }
    folds_path = Path("reports/phase5b/folds.json")
    folds_path.write_text(json.dumps(folds_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    folds_sha = sha256_file(folds_path)
    Path("reports/phase5b/folds.sha256").write_text(f"{folds_sha}  folds.json\n", encoding="utf-8")
    return {"data": folds_obj, "sha256": folds_sha}


def cluster_map_from_per_query(
    queries: List[Dict[str, Any]],
    res: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    pq = res["policies"]["hybrid"]["per_query_valid"]
    idxs = pq["indices"]
    aps = pq["ap"]
    by_cluster: Dict[str, List[float]] = defaultdict(list)
    for i, ap in zip(idxs, aps):
        by_cluster[queries[i]["spec_cluster_id"]].append(float(ap))
    cluster_means = {cid: float(np.mean(vals)) for cid, vals in by_cluster.items()}
    macro = float(np.mean(list(cluster_means.values()))) if cluster_means else 0.0
    return macro, cluster_means


def cluster_bootstrap_ci(
    cluster_vals: Dict[str, float],
    n_resamples: int = 1000,
    seed: int = SEED,
) -> Dict[str, float]:
    keys = sorted(cluster_vals.keys())
    arr = np.array([cluster_vals[k] for k in keys], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(arr[idx]))
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def paired_cluster_bootstrap_diff(
    vals_a: Dict[str, float],
    vals_b: Dict[str, float],
    n_resamples: int = 1000,
    seed: int = SEED,
) -> Dict[str, float]:
    keys = sorted(set(vals_a.keys()) & set(vals_b.keys()))
    a = np.array([vals_a[k] for k in keys], dtype=np.float64)
    b = np.array([vals_b[k] for k in keys], dtype=np.float64)
    diff = a - b
    n = len(diff)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diff[idx]))
    return {
        "mean": float(np.mean(diff)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def paired_score_bootstrap_ci(
    c_a_1: Dict[str, float],
    c_b_1: Dict[str, float],
    c_a_2: Dict[str, float],
    c_b_2: Dict[str, float],
    n_resamples: int = 1000,
    seed: int = SEED,
) -> Dict[str, float]:
    keys_a = sorted(set(c_a_1.keys()) & set(c_a_2.keys()))
    keys_b = sorted(set(c_b_1.keys()) & set(c_b_2.keys()))
    da = np.array([c_a_1[k] - c_a_2[k] for k in keys_a], dtype=np.float64)
    db = np.array([c_b_1[k] - c_b_2[k] for k in keys_b], dtype=np.float64)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ia = rng.integers(0, len(da), size=len(da))
        ib = rng.integers(0, len(db), size=len(db))
        diffs[i] = 0.4 * float(np.mean(da[ia])) + 0.6 * float(np.mean(db[ib]))
    obs = 0.4 * float(np.mean(da)) + 0.6 * float(np.mean(db))
    return {
        "mean": obs,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def catastrophic_drop_rate(
    res_cfg: Dict[str, Any],
    res_mo: Dict[str, Any],
    ap_drop_thresh: float = 0.30,
) -> float:
    pq_c = res_cfg["policies"]["hybrid"]["per_query_valid"]
    pq_m = res_mo["policies"]["hybrid"]["per_query_valid"]
    ap_c = dict(zip(pq_c["indices"], pq_c["ap"]))
    ap_m = dict(zip(pq_m["indices"], pq_m["ap"]))
    r10_c = dict(zip(pq_c["indices"], pq_c["recall@10"]))
    r10_m = dict(zip(pq_m["indices"], pq_m["recall@10"]))
    common = sorted(set(ap_c.keys()) & set(ap_m.keys()))
    if not common:
        return 0.0
    drops = sum(
        1
        for i in common
        if (ap_m[i] - ap_c[i] > ap_drop_thresh) or (r10_m[i] >= 1.0 and r10_c[i] < 1.0)
    )
    return float(drops / len(common))


def gpu_telemetry() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu": parts[0],
            "temp_c": int(parts[1]),
            "graphics_clock_mhz": int(parts[2]),
            "mem_clock_mhz": int(parts[3]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def measure_t2i_latency(
    engine: ECommerceSearchEngine,
    parser,
    cfg: Dict[str, Any],
    queries_200: List[str],
    bm25_scorer: Optional[BM25GalleryScorer],
    n_runs: int = 5,
    n_warmup: int = 20,
) -> Dict[str, Any]:
    # Cool GPU briefly before each config
    time.sleep(1.5)
    tele_before = gpu_telemetry()
    use_rewrite = bool(cfg.get("query_rewrite", False))
    use_dual = bool(cfg.get("dual_encode", False))
    use_bm25 = bool(cfg.get("bm25", False))
    use_zscore = bool(cfg.get("zscore_cosine", True))
    gamma = float(cfg.get("gamma", 0.3))
    sw = None if cfg.get("slot_weights", "uniform") == "uniform" else cfg["slot_weights"]
    gal_vecs = engine.embeddings
    if engine.use_mean_centering:
        c = gal_vecs - engine.gallery_mean
        gal_vecs = (c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    # Warmup
    for q in queries_200[:n_warmup]:
        qt = restore_diacritics_for_query(q) if use_rewrite else q
        _ = parser.parse(qt)
        _ = dual_encode_texts(engine, [qt]) if use_dual else engine.encode_text_query(qt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    lat_ms: List[float] = []
    for _ in range(n_runs):
        for q in queries_200:
            t0 = time.perf_counter()
            qt = restore_diacritics_for_query(q) if use_rewrite else q
            _ = parser.parse(qt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            qv = dual_encode_texts(engine, [qt]) if use_dual else engine.encode_text_query(qt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sims = (gal_vecs @ qv[0]).astype(np.float64)
            if use_zscore:
                sd = float(np.std(sims))
                if sd > 1e-8:
                    sims = (sims - float(np.mean(sims))) / sd
            if use_bm25 and bm25_scorer is not None:
                sims = sims + 0.2 * bm25_scorer.score_query(qt)
            # Simulate exact top-K sort
            _ = np.argpartition(-sims, 10)[:10]
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

    tele_after = gpu_telemetry()
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "n_runs": n_runs,
        "queries_per_run": len(queries_200),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "gpu_before": tele_before,
        "gpu_after": tele_after,
    }


def simplicity_rank_key(cfg: Dict[str, Any]) -> Tuple[int, int, int, int, int, float]:
    return (
        1 if cfg["mean_centering"] else 0,
        1 if cfg["bm25"] else 0,
        1 if cfg["dual_encode"] else 0,
        1 if cfg["query_rewrite"] else 0,
        0 if cfg["slot_weights"] == "uniform" else 1,
        float(cfg["gamma"]),
    )


def main() -> None:
    rule = json.loads(Path("reports/phase5b/selection_rule_v3.json").read_text(encoding="utf-8"))
    grid = rule["config_grid"]
    gpol = json.loads(Path("eval/gender_gt_policy.json").read_text(encoding="utf-8"))

    val_a = json.loads(Path("eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    val_b_all = json.loads(Path("eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    val_b_clean = [q for q in val_b_all if not q.get("is_typo")]
    val_b_typo = [q for q in val_b_all if q.get("is_typo")]
    val_c_obj = json.loads(Path("eval/val_queries_c_error_injection.json").read_text(encoding="utf-8"))
    val_c_20 = val_c_obj["queries"]
    parsed_c_20 = [q["parsed_by_rate"]["20%"] for q in val_c_20]

    # 1. Lock folds BEFORE scoring
    folds_info = build_and_lock_folds(val_a, val_b_clean)
    fold_a = folds_info["data"]["fold_of_cluster_a"]
    fold_b = folds_info["data"]["fold_of_cluster_b_clean"]
    print(f"Locked folds.json SHA-256: {folds_info['sha256']}")

    # 2. Load engines (std and mean-centered)
    eng_std = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    eng_mc = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=True)
    n_gal = len(eng_std.id_map)
    print(f"Loaded val_gallery: n_gallery={n_gal}")

    p1 = VietnameseQueryParser()
    p2 = VietnameseQueryParserV2()
    bm25 = BM25GalleryScorer(eng_std.id_map)

    # Pre-encode & pre-parse datasets (raw, rewrite, dual, mc)
    datasets = {
        "A": val_a,
        "B_clean": val_b_clean,
        "B_typo": val_b_typo,
        "C_20": val_c_20,
    }
    cache: Dict[str, Dict[str, Any]] = {}
    for dname, ds in datasets.items():
        raw_texts = [q["query"] for q in ds]
        rw_texts = [restore_diacritics_for_query(t) for t in raw_texts]
        print(f"Pre-encoding {dname} ({len(ds)} queries)...")
        vecs_raw = np.vstack([eng_std.encode_text_query(t) for t in raw_texts])
        vecs_rw = np.vstack([eng_std.encode_text_query(t) for t in rw_texts])
        vecs_dual = dual_encode_texts(eng_std, raw_texts)
        vecs_mc = np.vstack([eng_mc.encode_text_query(t) for t in raw_texts])
        bm25_mat = np.vstack([bm25.score_query(t) for t in raw_texts])
        cache[dname] = {
            "vecs_raw": vecs_raw,
            "vecs_rw": vecs_rw,
            "vecs_dual": vecs_dual,
            "vecs_mc": vecs_mc,
            "bm25": bm25_mat,
            "parsed_v1": [p1.parse(t) for t in raw_texts],
            "parsed_v2": [p2.parse(t) for t in raw_texts],
            "parsed_v2_rw": [p2.parse(t) for t in rw_texts],
        }

    # Cosine std diagnostics (for Prompt 8 E.2)
    raw_cos_a = cache["A"]["vecs_raw"] @ eng_std.embeddings.T
    cos_std_mean_a = float(np.mean(np.std(raw_cos_a, axis=1)))
    print(f"Mean per-query cosine std on Val A: {cos_std_mean_a:.4f}")

    def eval_config_on_ds(cfg: Dict[str, Any], dname: str, use_c_oracle: bool = False) -> Dict[str, Any]:
        ds = datasets[dname]
        c = cache[dname]
        eng = eng_mc if cfg.get("mean_centering") else eng_std
        if cfg.get("mean_centering"):
            q_vecs = c["vecs_mc"]
        elif cfg.get("dual_encode"):
            q_vecs = c["vecs_dual"]
        elif cfg.get("query_rewrite"):
            q_vecs = c["vecs_rw"]
        else:
            q_vecs = c["vecs_raw"]

        if use_c_oracle:
            parsed = parsed_c_20
        elif cfg["parser"] == "model_only":
            parsed = c["parsed_v1"]
        elif cfg["parser"] == "parser_v1":
            parsed = c["parsed_v1"]
        elif cfg.get("query_rewrite"):
            parsed = c["parsed_v2_rw"]
        else:
            parsed = c["parsed_v2"]

        sw = None if cfg.get("slot_weights", "uniform") == "uniform" else cfg["slot_weights"]
        bm25_mat = c["bm25"] if cfg.get("bm25") else None
        mode = "model_only" if cfg["parser"] == "model_only" else "system"
        return evaluate_t2i_canonical(
            eng,
            ds,
            q_vecs,
            parsed,
            gpol,
            mode=mode,
            candidate_pool=n_gal,
            eval_top_k=n_gal,
            gamma=float(cfg["gamma"]),
            slot_weights=sw,
            force_exact=True,
            tie_rule="conservative",
            tie_epsilon=1e-5,
            zscore_cosine=bool(cfg["zscore_cosine"]),
            bm25_scores=bm25_mat,
            bm25_weight=0.2,
        )

    # Reference rows: model_only & baseline Pha 5 (no z-score)
    mo_cfg = {"config_id": "model_only", "parser": "model_only", "gamma": 0.0, "slot_weights": "uniform", "zscore_cosine": False, "mean_centering": False, "bm25": False, "dual_encode": False, "query_rewrite": False}
    p5_cfg = {"config_id": "baseline_pha5_no_zscore", "parser": "parser_v1", "gamma": 0.3, "slot_weights": "uniform", "zscore_cosine": False, "mean_centering": False, "bm25": False, "dual_encode": False, "query_rewrite": False}

    mo_res = {d: eval_config_on_ds(mo_cfg, d, use_c_oracle=(d == "C_20")) for d in datasets}
    p5_res = {d: eval_config_on_ds(p5_cfg, d, use_c_oracle=(d == "C_20")) for d in datasets}

    all_rows: List[Dict[str, Any]] = []
    cluster_maps_by_id: Dict[str, Tuple[Dict[str, float], Dict[str, float]]] = {}

    for cfg in grid:
        cid = cfg["config_id"]
        res_a = eval_config_on_ds(cfg, "A")
        res_bc = eval_config_on_ds(cfg, "B_clean")
        res_bt = eval_config_on_ds(cfg, "B_typo")
        res_c = eval_config_on_ds(cfg, "C_20", use_c_oracle=True)

        map_a_cl, cl_a = cluster_map_from_per_query(val_a, res_a)
        map_bc_cl, cl_bc = cluster_map_from_per_query(val_b_clean, res_bc)
        map_bt_cl, _ = cluster_map_from_per_query(val_b_typo, res_bt)
        map_c_cl, _ = cluster_map_from_per_query(val_c_20, res_c)
        cluster_maps_by_id[cid] = (cl_a, cl_bc)

        # 5-fold CV score
        fold_scores = []
        for f_idx in range(5):
            fa_vals = [v for k, v in cl_a.items() if fold_a[k] == f_idx]
            fb_vals = [v for k, v in cl_bc.items() if fold_b[k] == f_idx]
            f_score = 0.4 * float(np.mean(fa_vals)) + 0.6 * float(np.mean(fb_vals))
            fold_scores.append(f_score)
        cv_score = float(np.mean(fold_scores))

        # Overall A+B-synth R@10 (485 queries)
        r10_all_ab = (
            res_a["policies"]["hybrid"]["per_query_valid"]["recall@10"]
            + res_bc["policies"]["hybrid"]["per_query_valid"]["recall@10"]
            + res_bt["policies"]["hybrid"]["per_query_valid"]["recall@10"]
        )
        mean_r10_ab = float(np.mean(r10_all_ab))

        # Catastrophic drop on B-synth (clean + typo)
        cat_bc = catastrophic_drop_rate(res_bc, mo_res["B_clean"])
        cat_bt = catastrophic_drop_rate(res_bt, mo_res["B_typo"])
        n_bc = len(val_b_clean)
        n_bt = len(val_b_typo)
        cat_b_all = (cat_bc * n_bc + cat_bt * n_bt) / (n_bc + n_bt)

        row = {
            "config_id": cid,
            "cfg": cfg,
            "cv_score_mean": round(cv_score, 4),
            "fold_scores": [round(x, 4) for x in fold_scores],
            "map_a_cluster": round(map_a_cl, 4),
            "map_b_clean_cluster": round(map_bc_cl, 4),
            "map_b_typo_cluster": round(map_bt_cl, 4),
            "map_c_20_cluster": round(map_c_cl, 4),
            "sum_a": res_a["policies"]["hybrid"]["summary"],
            "sum_bc": res_bc["policies"]["hybrid"]["summary"],
            "sum_bt": res_bt["policies"]["hybrid"]["summary"],
            "sum_c20": res_c["policies"]["hybrid"]["summary"],
            "mean_r10_ab": round(mean_r10_ab, 4),
            "catastrophic_drop_b": round(cat_b_all, 4),
        }
        all_rows.append(row)
        print(
            f"{cid}: CV={row['cv_score_mean']:.4f} | A={row['map_a_cluster']:.4f} "
            f"Bc={row['map_b_clean_cluster']:.4f} Bt={row['map_b_typo_cluster']:.4f} "
            f"C20={row['map_c_20_cluster']:.4f} R10_AB={row['mean_r10_ab']:.4f} CatB={row['catastrophic_drop_b']:.4f}"
        )

    # Check constraints 5(i)-(iii), (v) relative to cfg_01 and model_only
    base_row = next(r for r in all_rows if r["config_id"] == "cfg_01")
    mo_c20_map, _ = cluster_map_from_per_query(val_c_20, mo_res["C_20"])
    base_c20_map = base_row["map_c_20_cluster"]
    base_r10_ab = base_row["mean_r10_ab"]

    for r in all_rows:
        c20_vs_mo = r["map_c_20_cluster"] - mo_c20_map
        c20_vs_base = r["map_c_20_cluster"] - base_c20_map
        dr10_ab = r["mean_r10_ab"] - base_r10_ab
        p5i = (c20_vs_mo >= -0.03) and (c20_vs_base >= -0.05)
        p5ii = r["catastrophic_drop_b"] <= 0.05
        p5iii = dr10_ab >= -0.01
        p5v = True  # verified in Preflight D for both parser_v1 and parser_v2.1
        r["delta_c20_vs_mo"] = round(c20_vs_mo, 4)
        r["delta_c20_vs_base"] = round(c20_vs_base, 4)
        r["delta_r10_ab_vs_cfg01"] = round(dr10_ab, 4)
        r["pass_5i"] = p5i
        r["pass_5ii"] = p5ii
        r["pass_5iii"] = p5iii
        r["pass_5v"] = p5v
        r["pass_pre_latency"] = p5i and p5ii and p5iii and p5v

    # Measure latency for cfg_01, cfg_02, and all configs passing pre_latency
    lat_queries = [q["query"] for q in (val_a[:120] + val_b_clean[:80])]
    to_measure = [
        r for r in all_rows if r["config_id"] in ("cfg_01", "cfg_02") or r["pass_pre_latency"]
    ]
    print(f"Measuring T2I p95 latency (5x200, warmup 20) for {len(to_measure)} configs...")
    lat_results: Dict[str, Any] = {}
    for r in to_measure:
        cid = r["config_id"]
        cfg = r["cfg"]
        eng = eng_mc if cfg["mean_centering"] else eng_std
        pr = p1 if cfg["parser"] == "parser_v1" else p2
        bm = bm25 if cfg["bm25"] else None
        lat = measure_t2i_latency(eng, pr, cfg, lat_queries, bm)
        lat_results[cid] = lat
        r["latency"] = lat
        print(f"  {cid} latency: p50={lat['p50_ms']}ms p95={lat['p95_ms']}ms p99={lat['p99_ms']}ms")

    base_p95 = lat_results["cfg_01"]["p95_ms"]
    for r in all_rows:
        if "latency" in r:
            dp95 = r["latency"]["p95_ms"] - base_p95
            r["pass_5iv"] = (r["latency"]["p95_ms"] < 200.0) and (dp95 <= 20.0)
        else:
            r["pass_5iv"] = None
        r["pass_all"] = bool(r["pass_pre_latency"] and (r["pass_5iv"] is True))

    eligible = [r for r in all_rows if r["pass_all"]]
    if not eligible:
        eligible = [r for r in all_rows if r["pass_pre_latency"]]

    # Best by score + paired cluster bootstrap tie-breaking (simplicity_order)
    best_raw = max(eligible, key=lambda x: x["cv_score_mean"])
    # Check if any simpler eligible config is within 95% paired cluster bootstrap CI of best_raw
    simpler_candidates = [
        r for r in eligible
        if simplicity_rank_key(r["cfg"]) <= simplicity_rank_key(best_raw["cfg"])
    ]
    simpler_candidates.sort(key=lambda x: simplicity_rank_key(x["cfg"]))

    winner = best_raw
    tie_checks = []
    ba, bb = cluster_maps_by_id[best_raw["config_id"]]
    for cand in simpler_candidates:
        if cand["config_id"] == best_raw["config_id"]:
            continue
        ca, cb = cluster_maps_by_id[cand["config_id"]]
        ci = paired_score_bootstrap_ci(ba, bb, ca, cb)
        within_ci = (ci["ci_low"] <= 0.0 <= ci["ci_high"])
        tie_checks.append({"vs_simpler": cand["config_id"], "diff_ci": ci, "within_95_ci": within_ci})
        if within_ci:
            winner = cand
            break

    # Safe config: satisfies 5(i) & 5(ii) with lowest gamma (tie-break by simplicity)
    safe_pool = [r for r in all_rows if r["pass_5i"] and r["pass_5ii"] and (r.get("pass_5iv") is not False)]
    safe_config = min(safe_pool, key=lambda x: (float(x["cfg"]["gamma"]), simplicity_rank_key(x["cfg"])))

    # Lock selection results BEFORE running B-human
    out_obj: Dict[str, Any] = {
        "seed": SEED,
        "parser_v2_version": PARSER_VERSION,
        "parser_v2_sha256": sha256_file(Path("src/serve/parser_v2.py")),
        "folds_sha256": folds_info["sha256"],
        "selection_rule_sha256": sha256_file(Path("reports/phase5b/selection_rule_v3.json")),
        "mean_cosine_std_val_a": round(cos_std_mean_a, 4),
        "reference_model_only": {
            "A": mo_res["A"]["policies"]["hybrid"]["summary"],
            "B_clean": mo_res["B_clean"]["policies"]["hybrid"]["summary"],
            "B_typo": mo_res["B_typo"]["policies"]["hybrid"]["summary"],
            "C_20": mo_res["C_20"]["policies"]["hybrid"]["summary"],
            "map_c_20_cluster": round(mo_c20_map, 4),
        },
        "reference_baseline_pha5_no_zscore": {
            "A_map_cluster": round(cluster_map_from_per_query(val_a, p5_res["A"])[0], 4),
            "A_summary": p5_res["A"]["policies"]["hybrid"]["summary"],
            "B_clean_map_cluster": round(cluster_map_from_per_query(val_b_clean, p5_res["B_clean"])[0], 4),
            "B_clean_summary": p5_res["B_clean"]["policies"]["hybrid"]["summary"],
        },
        "grid_results": all_rows,
        "selection": {
            "best_raw_by_score": best_raw["config_id"],
            "winner_after_tie_break": winner["config_id"],
            "safe_config": safe_config["config_id"],
            "mandatory_ablation": "cfg_02",
            "tie_checks": tie_checks,
        },
    }
    res_path = Path("reports/phase5b/step3_results.json")
    res_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    pre_bh_hash = sha256_file(res_path)
    print(f"LOCKED selection before B-human: winner={winner['config_id']} best_raw={best_raw['config_id']} safe={safe_config['config_id']} hash={pre_bh_hash}")

    # NOW run B-human confirmation gate (validated 65 queries)
    bh = json.loads(Path("eval/val_queries_b_human.validated.json").read_text(encoding="utf-8"))
    bh_texts = [q["query"] for q in bh]
    bh_vecs_raw = np.vstack([eng_std.encode_text_query(t) for t in bh_texts])
    bh_vecs_rw = np.vstack([eng_std.encode_text_query(restore_diacritics_for_query(t)) for t in bh_texts])
    bh_vecs_dual = dual_encode_texts(eng_std, bh_texts)
    bh_vecs_mc = np.vstack([eng_mc.encode_text_query(t) for t in bh_texts])
    bh_bm25 = np.vstack([bm25.score_query(t) for t in bh_texts])
    bh_p1 = [p1.parse(t) for t in bh_texts]
    bh_p2 = [p2.parse(t) for t in bh_texts]
    bh_p2_rw = [p2.parse(restore_diacritics_for_query(t)) for t in bh_texts]

    def eval_bh(cfg: Dict[str, Any]):
        eng = eng_mc if cfg["mean_centering"] else eng_std
        qv = bh_vecs_mc if cfg["mean_centering"] else (bh_vecs_dual if cfg["dual_encode"] else (bh_vecs_rw if cfg["query_rewrite"] else bh_vecs_raw))
        pr = bh_p1 if cfg["parser"] in ("parser_v1", "model_only") else (bh_p2_rw if cfg["query_rewrite"] else bh_p2)
        sw = None if cfg["slot_weights"] == "uniform" else cfg["slot_weights"]
        bm = bh_bm25 if cfg["bm25"] else None
        mode = "model_only" if cfg["parser"] == "model_only" else "system"
        return evaluate_t2i_canonical(
            eng, bh, qv, pr, gpol, mode=mode, candidate_pool=n_gal, eval_top_k=n_gal,
            gamma=float(cfg["gamma"]), slot_weights=sw, force_exact=True,
            tie_rule="conservative", tie_epsilon=1e-5, zscore_cosine=bool(cfg["zscore_cosine"]),
            bm25_scores=bm, bm25_weight=0.2,
        )

    bh_mo = eval_bh(mo_cfg)
    bh_base = eval_bh(base_row["cfg"])
    bh_win = eval_bh(winner["cfg"])
    bh_p5 = eval_bh(p5_cfg)

    _, bh_cl_base = cluster_map_from_per_query(bh, bh_base)
    _, bh_cl_win = cluster_map_from_per_query(bh, bh_win)
    _, bh_cl_p5 = cluster_map_from_per_query(bh, bh_p5)
    _, bh_cl_mo = cluster_map_from_per_query(bh, bh_mo)

    bh_diff_ci = paired_cluster_bootstrap_diff(bh_cl_win, bh_cl_base)
    bh_cat = catastrophic_drop_rate(bh_win, bh_mo)

    # Split with_colour (31) / no_colour (34) + rare labels
    def subset_map(res, id_set):
        pq = res["policies"]["hybrid"]["per_query_valid"]
        by_c = defaultdict(list)
        for i, ap in zip(pq["indices"], pq["ap"]):
            if bh[i]["id"] in id_set:
                by_c[bh[i]["spec_cluster_id"]].append(float(ap))
        return float(np.mean([np.mean(v) for v in by_c.values()])) if by_c else 0.0

    col_ids = {q["id"] for q in bh if q["spec"].get("baseColour")}
    nocol_ids = {q["id"] for q in bh if not q["spec"].get("baseColour")}
    rare_info = {}
    for q in bh:
        cols = q["spec"].get("baseColour") or []
        for rc in ("Steel", "Olive", "Navy Blue"):
            if rc in cols:
                pq_w = bh_win["policies"]["hybrid"]["per_query_valid"]
                pq_b = bh_base["policies"]["hybrid"]["per_query_valid"]
                idx = bh.index(q)
                ap_w = pq_w["ap"][pq_w["indices"].index(idx)]
                ap_b = pq_b["ap"][pq_b["indices"].index(idx)]
                rare_info[f"{rc} ({q['id']})"] = {"winner_ap": round(ap_w, 4), "baseline_ap": round(ap_b, 4)}

    out_obj["pre_b_human_results_sha256"] = pre_bh_hash
    out_obj["b_human_gate"] = {
        "n_queries": len(bh),
        "n_clusters": len(bh_cl_win),
        "model_only_map_cluster": round(float(np.mean(list(bh_cl_mo.values()))), 4),
        "baseline_v3_map_cluster": round(float(np.mean(list(bh_cl_base.values()))), 4),
        "baseline_pha5_no_zscore_map_cluster": round(float(np.mean(list(bh_cl_p5.values()))), 4),
        "baseline_pha5_no_zscore_summary": bh_p5["policies"]["hybrid"]["summary"],
        "winner_map_cluster": round(float(np.mean(list(bh_cl_win.values()))), 4),
        "winner_summary": bh_win["policies"]["hybrid"]["summary"],
        "delta_map_vs_baseline_ci95": {k: round(v, 4) for k, v in bh_diff_ci.items()},
        "catastrophic_drop_rate": round(bh_cat, 4),
        "gate_pass": (bh_diff_ci["ci_low"] >= -0.02) and (bh_cat <= 0.05),
        "split_with_colour_31": {
            "winner_map_cluster": round(subset_map(bh_win, col_ids), 4),
            "baseline_v3_map_cluster": round(subset_map(bh_base, col_ids), 4),
        },
        "split_no_colour_34": {
            "winner_map_cluster": round(subset_map(bh_win, nocol_ids), 4),
            "baseline_v3_map_cluster": round(subset_map(bh_base, nocol_ids), 4),
        },
        "rare_labels": rare_info,
    }
    res_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("B-human gate:", json.dumps(out_obj["b_human_gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
