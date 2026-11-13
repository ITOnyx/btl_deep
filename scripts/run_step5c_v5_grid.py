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
    cooldown_sec: float = 3.0,
) -> Dict[str, Dict[str, Any]]:
    """Protocol C.9: GPU nguội, xen kẽ giữa các cấu hình, báo trung vị + IQR của p95."""
    gal_vecs = engine.embeddings

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

    per_cfg_p95_runs: Dict[str, List[float]] = {c["config_id"]: [] for c in configs}
    per_cfg_all_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in configs}
    tele_log: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in configs}

    for r_idx in range(n_runs):
        order = list(configs) if r_idx % 2 == 0 else list(reversed(configs))
        for cfg in order:
            cid = cfg["config_id"]
            time.sleep(cooldown_sec)
            tele_before = gpu_telemetry()
            for q in queries_200[:n_warmup]:
                _run_one(cfg, q)
            run_ms: List[float] = []
            for q in queries_200:
                t0 = time.perf_counter()
                _run_one(cfg, q)
                run_ms.append((time.perf_counter() - t0) * 1000.0)
            tele_after = gpu_telemetry()
            per_cfg_p95_runs[cid].append(float(np.percentile(run_ms, 95)))
            per_cfg_all_ms[cid].extend(run_ms)
            tele_log[cid].append({"before": tele_before, "after": tele_after})

    out: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        cid = cfg["config_id"]
        p95s = per_cfg_p95_runs[cid]
        arr = np.array(per_cfg_all_ms[cid], dtype=np.float64)
        q25, q75 = np.percentile(p95s, [25, 75])
        out[cid] = {
            "n_runs": n_runs,
            "n_queries_per_run": len(queries_200),
            "p95_runs_ms": [round(x, 3) for x in p95s],
            "p95_median_ms": round(float(statistics.median(p95s)), 3),
            "p95_iqr_ms": round(float(q75 - q25), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "telemetry": tele_log[cid],
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

    b_human_v4_path = PROJECT_ROOT / "eval/val_queries_b_human_v4.json"
    if b_human_v4_path.exists():
        bh_raw = json.loads(b_human_v4_path.read_text(encoding="utf-8"))
        b_human_queries = bh_raw["queries"] if isinstance(bh_raw, dict) else bh_raw
        b_human_label = "B-human-v4 (locked)"
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

        tg = cfg.get("text_gallery", "none")
        t_alpha = float(cfg.get("text_alpha", 1.0))
        extra_text_mat = None
        if tg == "internvl2_only":
            extra_text_mat = c["z_a_cached"] if i3_mode == "cached" else c["z_a_rw"]
        elif tg == "internvl2_plus_aux":
            extra_text_mat = c["z_b_cached"] if i3_mode == "cached" else c["z_b_rw"]

        return evaluate_t2i_canonical(
            queries=ds,
            query_vectors=q_vecs,
            gallery_embeddings=eng.embeddings,
            id_map=eng.id_map,
            gender_policy=gpol,
            mode=mode,
            parsed_queries=parsed,
            gamma=float(cfg.get("gamma", 1.5)),
            zscore_cosine=True,
            bonus_mode=cfg.get("bonus_mode", "zscore"),
            slot_confidences=confs,
            extra_text_cos_matrix=extra_text_mat,
            text_alpha=t_alpha,
        )

    # Model-only baseline
    model_a = _eval_cfg(grid[0], "A", mode="model_only")
    model_bc = _eval_cfg(grid[0], "B_clean", mode="model_only")
    model_c20 = _eval_cfg(grid[0], "C_20", mode="model_only")
    model_bh = _eval_cfg(grid[0], "B_human", mode="model_only")

    results_by_cfg: Dict[str, Any] = {}
    for cfg in grid:
        cid = cfg["config_id"]
        res_a = _eval_cfg(cfg, "A")
        res_bc = _eval_cfg(cfg, "B_clean")
        res_bt = _eval_cfg(cfg, "B_typo")
        res_c20 = _eval_cfg(cfg, "C_20", use_c_oracle=True)
        res_bh = _eval_cfg(cfg, "B_human")

        # 5-fold CV score
        fold_scores = []
        for f in range(5):
            pa = [r for q, r in zip(val_a, res_a["per_query"]) if fold_a.get(q.get("spec_cluster_id")) == f]
            pb = [r for q, r in zip(val_b_clean, res_bc["per_query"]) if fold_b.get(q.get("spec_cluster_id")) == f]
            ma = cluster_map_from_per_query(pa, val_a) if pa else 0.0
            mb = cluster_map_from_per_query(pb, val_b_clean) if pb else 0.0
            fold_scores.append(0.4 * ma + 0.6 * mb)

        cv_score = float(np.mean(fold_scores))
        cat_drop_bc = float(
            np.mean([
                (rb["ap"] - rm["ap"]) < -0.10
                for rb, rm in zip(res_bc["per_query"], model_bc["per_query"])
            ])
        )
        hub_bc = compute_hubness_stats(res_bc["ranked_indices"], len(eng.id_map), top_k=10)

        results_by_cfg[cid] = {
            "config": cfg,
            "cv_score": round(cv_score, 4),
            "fold_scores": [round(x, 4) for x in fold_scores],
            "A_mAP_cluster": res_a["macro_cluster_mAP"],
            "A_R10": res_a["overall"]["Recall@10"],
            "Bc_mAP_cluster": res_bc["macro_cluster_mAP"],
            "Bc_R10": res_bc["overall"]["Recall@10"],
            "Bt_mAP_cluster": res_bt["macro_cluster_mAP"],
            "C20_mAP": res_c20["overall"]["mAP"],
            "cat_drop_Bc": round(cat_drop_bc, 4),
            "hubness_Bc": hub_bc,
            "B_human_mAP": res_bh["overall"]["mAP"],
            "B_human_R1": res_bh["overall"]["Recall@1"],
            "B_human_R10": res_bh["overall"]["Recall@10"],
            "per_query_A": res_a["per_query"],
            "per_query_Bc": res_bc["per_query"],
            "per_query_Bh": res_bh["per_query"],
        }

    # C.9 latency measurement across all configs
    queries_200 = [q["query"] for q in val_a[:200]]
    lat_results = measure_c9_interleaved_latency(
        eng, parser, tpl_cache, grid, queries_200, embs_a, embs_b, n_runs=5, n_warmup=20
    )

    # Select winner among competes_for_winner=True
    competitors = [cid for cid, v in results_by_cfg.items() if v["config"].get("competes_for_winner", True)]
    competitors.sort(key=lambda cid: (-results_by_cfg[cid]["cv_score"], simplicity_rank_key_v5(results_by_cfg[cid]["config"])))
    winner_cid = competitors[0]

    # Check identical metrics across configs
    metric_sigs = defaultdict(list)
    for cid, v in results_by_cfg.items():
        sig = (v["A_mAP_cluster"], v["Bc_mAP_cluster"], v["Bt_mAP_cluster"], v["C20_mAP"])
        metric_sigs[sig].append(cid)
    identical_groups = [cids for cids in metric_sigs.values() if len(cids) > 1]

    # Oracle parser on B-clean and B-human for winner
    w_cfg = results_by_cfg[winner_cid]["config"]
    oracle_bc = _eval_cfg(w_cfg, "B_clean", oracle_spec=True)
    oracle_bh = _eval_cfg(w_cfg, "B_human", oracle_spec=True)

    # Cosine std stats (bonus size vs cosine std)
    cos_stds_a = [float(np.std(cache["A"]["vecs_rw"][i] @ eng.embeddings.T)) for i in range(len(val_a))]

    summary = {
        "rule_sha256": rule_sha,
        "folds_sha256": FOLDS_SHA256,
        "seed": SEED,
        "b_human_label": b_human_label,
        "b_human_is_ref_only": b_human_is_ref_only,
        "fallback_stats": fallback_stats,
        "winner_config_id": winner_cid,
        "identical_metric_groups": identical_groups,
        "cosine_std_mean_A": round(float(np.mean(cos_stds_a)), 4),
        "oracle_parser_winner": {
            "Bc_mAP_cluster": oracle_bc["macro_cluster_mAP"],
            "B_human_mAP": oracle_bh["overall"]["mAP"],
        },
        "model_only": {
            "A_mAP_cluster": model_a["macro_cluster_mAP"],
            "Bc_mAP_cluster": model_bc["macro_cluster_mAP"],
            "C20_mAP": model_c20["overall"]["mAP"],
            "B_human_mAP": model_bh["overall"]["mAP"],
        },
        "latency_c9": lat_results,
        "configs": {
            cid: {k: val for k, val in v.items() if not k.startswith("per_query_")}
            for cid, v in results_by_cfg.items()
        },
    }

    out_json = PROJECT_ROOT / "reports/phase5c/step5c_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Winner: {winner_cid} | CV Score: {results_by_cfg[winner_cid]['cv_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
