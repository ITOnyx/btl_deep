"""
Chẩn đoán sau Bước 4 (PROMPT 10 Phần 1.4 & Phần 2):
- Phần 1.4: Kiểm chứng cấu hình tương đương & đếm số cấu hình hiệu dụng khác nhau thật sự.
- Phần 2.1: Oracle parser (spec) cho cfg_v4_01, cfg_v4_05, cfg_v4_14 trên B-clean, B-typo, B-human_ref, Val A.
- Phần 2.2: I3 (cfg_v4_14 vs cfg_v4_01) trên B-synth (clean, typo) vs B-human_ref theo dia_class và nhóm trong/ngoài lexicon (+ paired cluster bootstrap 1.000 lần).
- Phần 2.3: Phân tích hành vi theo gamma của nhóm I1 (cfg_v4_02/03/04) và cfg_v4_05, kích thước bonus vs cosine std.
Seed = 42. KHÔNG chạy trên test_gallery.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, ".")
from scripts.run_step3_grid import cluster_map_from_per_query, paired_cluster_bootstrap_diff
from src.eval.canonical_eval import evaluate_t2i_canonical
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import multi_template_encode_texts, restore_diacritics_for_query
from src.serve.search_engine import ECommerceSearchEngine

SEED = 42


def _subset_cluster_diff(
    queries: List[Dict[str, Any]],
    cl_a: Dict[str, float],
    cl_b: Dict[str, float],
    idxs: List[int],
) -> Dict[str, Any]:
    """Paired cluster bootstrap diff (cl_a - cl_b) restricted to queries in idxs."""
    if not idxs:
        return {"n_queries": 0, "n_clusters": 0, "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    cids = sorted({queries[i]["spec_cluster_id"] for i in idxs if queries[i]["spec_cluster_id"] in cl_a and queries[i]["spec_cluster_id"] in cl_b})
    if not cids:
        return {"n_queries": len(idxs), "n_clusters": 0, "mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    sub_a = {c: cl_a[c] for c in cids}
    sub_b = {c: cl_b[c] for c in cids}
    diff = paired_cluster_bootstrap_diff(sub_a, sub_b, n_resamples=1000, seed=SEED)
    return {
        "n_queries": len(idxs),
        "n_clusters": len(cids),
        "mean_a": float(np.mean(list(sub_a.values()))),
        "mean_b": float(np.mean(list(sub_b.values()))),
        "mean_diff": diff["mean"],
        "ci_low": diff["ci_low"],
        "ci_high": diff["ci_high"],
    }


def _per_query_ap_by_cluster(queries: List[Dict[str, Any]], res: Dict[str, Any], subset_idxs: List[int]) -> Dict[str, float]:
    pq = res["policies"]["hybrid"]["per_query_valid"]
    ap_map = dict(zip(pq["indices"], pq["ap"]))
    by_cl: Dict[str, List[float]] = defaultdict(list)
    for i in subset_idxs:
        if i in ap_map:
            by_cl[queries[i]["spec_cluster_id"]].append(float(ap_map[i]))
    return {c: float(np.mean(v)) for c, v in by_cl.items()}


def main() -> None:
    v4_res = json.loads(Path("reports/phase5c/step4_v4_results.json").read_text(encoding="utf-8"))
    gpol = json.loads(Path("eval/gender_gt_policy.json").read_text(encoding="utf-8"))

    val_a = json.loads(Path("eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    val_b_all = json.loads(Path("eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    val_b_clean = [q for q in val_b_all if not q.get("is_typo")]
    val_b_typo = [q for q in val_b_all if q.get("is_typo")]
    b_human_val = json.loads(Path("eval/val_queries_b_human.validated.json").read_text(encoding="utf-8"))
    b_human_queries = b_human_val["queries"] if isinstance(b_human_val, dict) else b_human_val

    # 1. Equivalence & effective distinct configs from step4_v4_results.json
    grid_rows = v4_res["grid_results"]
    sigs = defaultdict(list)
    for r in grid_rows:
        ds = r["datasets"]
        sig = (
            round(ds["A"]["cluster_mAP"], 6),
            round(ds["B_clean"]["cluster_mAP"], 6),
            round(ds["B_typo"]["cluster_mAP"], 6),
            round(ds["C_20"]["cluster_mAP"], 6),
        )
        sigs[sig].append(r["config"]["config_id"])
    sigs_ab = defaultdict(list)
    for r in grid_rows:
        ds = r["datasets"]
        sig_ab = (round(ds["A"]["cluster_mAP"], 6), round(ds["B_clean"]["cluster_mAP"], 6))
        sigs_ab[sig_ab].append(r["config"]["config_id"])

    by_id = {r["config"]["config_id"]: r for r in grid_rows}
    equiv_checks = {
        "cfg_v4_03_vs_01": {
            "A_mAP_01": by_id["cfg_v4_01"]["datasets"]["A"]["cluster_mAP"],
            "A_mAP_03": by_id["cfg_v4_03"]["datasets"]["A"]["cluster_mAP"],
            "B_clean_mAP_01": by_id["cfg_v4_01"]["datasets"]["B_clean"]["cluster_mAP"],
            "B_clean_mAP_03": by_id["cfg_v4_03"]["datasets"]["B_clean"]["cluster_mAP"],
            "B_typo_mAP_01": by_id["cfg_v4_01"]["datasets"]["B_typo"]["cluster_mAP"],
            "B_typo_mAP_03": by_id["cfg_v4_03"]["datasets"]["B_typo"]["cluster_mAP"],
        },
        "cfg_v4_08_vs_01": {
            "A_mAP_01": by_id["cfg_v4_01"]["datasets"]["A"]["cluster_mAP"],
            "A_mAP_08": by_id["cfg_v4_08"]["datasets"]["A"]["cluster_mAP"],
            "B_clean_mAP_01": by_id["cfg_v4_01"]["datasets"]["B_clean"]["cluster_mAP"],
            "B_clean_mAP_08": by_id["cfg_v4_08"]["datasets"]["B_clean"]["cluster_mAP"],
            "B_typo_mAP_01": by_id["cfg_v4_01"]["datasets"]["B_typo"]["cluster_mAP"],
            "B_typo_mAP_08": by_id["cfg_v4_08"]["datasets"]["B_typo"]["cluster_mAP"],
        },
        "n_distinct_all_4_sets": len(sigs),
        "n_distinct_a_bclean": len(sigs_ab),
        "equivalence_groups_all_4_sets": list(sigs.values()),
        "equivalence_groups_a_bclean": list(sigs_ab.values()),
    }

    # 2. Load engine & parser for Oracle & I3 breakdown
    eng = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    parser = VietnameseQueryParserV2Conf()

    datasets = {
        "A": val_a,
        "B_clean": val_b_clean,
        "B_typo": val_b_typo,
        "B_human_ref": b_human_queries,
    }

    oracle_results: Dict[str, Any] = {}
    parser_coverage: Dict[str, Any] = {}
    cosine_std_stats: Dict[str, Any] = {}
    i3_data: Dict[str, Any] = {}

    for dname, ds in datasets.items():
        raw_texts = [q["query"] for q in ds]
        rw_texts = [restore_diacritics_for_query(t) for t in raw_texts]
        parsed_std = [parser.parse_with_confidence(t, fuzzy_typo=False) for t in rw_texts]
        oracle_specs = [q["spec"] for q in ds]

        n_art_cov = sum(1 for p in parsed_std if p.get("articleType"))
        n_gen_cov = sum(1 for p in parsed_std if p.get("gender"))
        n_col_cov = sum(1 for p in parsed_std if p.get("baseColour"))
        parser_coverage[dname] = {
            "n_queries": len(ds),
            "articleType_parsed": n_art_cov,
            "articleType_rate": round(n_art_cov / len(ds), 4),
            "gender_parsed": n_gen_cov,
            "baseColour_parsed": n_col_cov,
        }

        vecs_rw = np.vstack([eng.encode_text_query(t) for t in rw_texts])
        vecs_mt2_sys = multi_template_encode_texts(eng, raw_texts, parsed_std, n_templates=2)
        vecs_mt2_ora = multi_template_encode_texts(eng, raw_texts, oracle_specs, n_templates=2)

        raw_cos = vecs_rw @ eng.embeddings.T
        sds = np.std(raw_cos, axis=1)
        cosine_std_stats[dname] = {
            "mean": round(float(np.mean(sds)), 5),
            "median": round(float(np.median(sds)), 5),
            "min": round(float(np.min(sds)), 5),
            "max": round(float(np.max(sds)), 5),
        }

        # Evaluate system vs oracle for cfg_v4_01, cfg_v4_05, cfg_v4_14
        cfgs_to_check = [
            ("cfg_v4_01", vecs_rw, "zscore", 1.5),
            ("cfg_v4_05", vecs_rw, "tier_soft", 1.5),
            ("cfg_v4_14", vecs_mt2_sys, "zscore", 1.5),
        ]
        oracle_results[dname] = {}
        res_sys_by_cfg = {}
        for cid, qv_sys, bmode, gamma in cfgs_to_check:
            qv_ora = vecs_mt2_ora if cid == "cfg_v4_14" else vecs_rw
            r_sys = evaluate_t2i_canonical(
                engine=eng, queries=ds, parsed_queries=parsed_std, query_vecs=qv_sys,
                gender_policy_cfg=gpol, mode="system", gamma=gamma, force_exact=True,
                zscore_cosine=(bmode == "zscore"), bonus_mode=bmode,
            )
            r_ora = evaluate_t2i_canonical(
                engine=eng, queries=ds, parsed_queries=oracle_specs, query_vecs=qv_ora,
                gender_policy_cfg=gpol, mode="system", gamma=gamma, force_exact=True,
                zscore_cosine=(bmode == "zscore"), bonus_mode=bmode,
            )
            m_sys, _ = cluster_map_from_per_query(ds, r_sys)
            m_ora, _ = cluster_map_from_per_query(ds, r_ora)
            oracle_results[dname][cid] = {
                "sys_cluster_mAP": round(m_sys, 4),
                "oracle_cluster_mAP": round(m_ora, 4),
                "parser_loss": round(m_ora - m_sys, 4),
                "model_gap_to_1": round(1.0 - m_ora, 4),
                "sys_r10": round(r_sys["policies"]["hybrid"]["summary"]["recall@10"], 4),
                "oracle_r10": round(r_ora["policies"]["hybrid"]["summary"]["recall@10"], 4),
            }
            res_sys_by_cfg[cid] = r_sys

        if dname in ("B_clean", "B_typo", "B_human_ref"):
            # I3 breakdown: cfg_v4_14 vs cfg_v4_01 by dia_class & by in_lexicon (articleType parsed vs not)
            r01 = res_sys_by_cfg["cfg_v4_01"]
            r14 = res_sys_by_cfg["cfg_v4_14"]
            bd = {"by_dia_class": {}, "by_articleType_in_lexicon": {}}
            for dc in ("full", "partial", "none"):
                idxs = [i for i, q in enumerate(ds) if q.get("dia_class") == dc]
                if idxs:
                    c01 = _per_query_ap_by_cluster(ds, r01, idxs)
                    c14 = _per_query_ap_by_cluster(ds, r14, idxs)
                    bd["by_dia_class"][dc] = _subset_cluster_diff(ds, c14, c01, idxs)
            in_lex = [i for i, p in enumerate(parsed_std) if p.get("articleType")]
            out_lex = [i for i, p in enumerate(parsed_std) if not p.get("articleType")]
            for label, idxs in [("in_lexicon", in_lex), ("out_of_lexicon", out_lex)]:
                if idxs:
                    c01 = _per_query_ap_by_cluster(ds, r01, idxs)
                    c14 = _per_query_ap_by_cluster(ds, r14, idxs)
                    bd["by_articleType_in_lexicon"][label] = _subset_cluster_diff(ds, c14, c01, idxs)
            i3_data[dname] = bd

    # Gamma behavior for I1 & cfg_v4_05
    gamma_analysis = {
        "I1_sigma_scaled": [
            {"config_id": "cfg_v4_02", "gamma": 1.0, "cv_score": by_id["cfg_v4_02"]["cv_score"], "pass": by_id["cfg_v4_02"]["constraints"]["all_pass"]},
            {"config_id": "cfg_v4_03", "gamma": 1.5, "cv_score": by_id["cfg_v4_03"]["cv_score"], "pass": by_id["cfg_v4_03"]["constraints"]["all_pass"]},
            {"config_id": "cfg_v4_04", "gamma": 2.0, "cv_score": by_id["cfg_v4_04"]["cv_score"], "pass": by_id["cfg_v4_04"]["constraints"]["all_pass"]},
        ],
        "I1_tier_soft": [
            {"config_id": "cfg_v4_05", "gamma": 1.5, "cv_score": by_id["cfg_v4_05"]["cv_score"], "note": "Single point in v4 grid (boundary of tier_soft grid)"},
        ],
        "cosine_std": cosine_std_stats,
    }

    out = {
        "seed": SEED,
        "equivalence_checks": equiv_checks,
        "parser_coverage": parser_coverage,
        "oracle_results": oracle_results,
        "i3_breakdown": i3_data,
        "gamma_analysis": gamma_analysis,
    }
    Path("reports/phase5c/step4_diagnostics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved reports/phase5c/step4_diagnostics.json")


if __name__ == "__main__":
    main()
