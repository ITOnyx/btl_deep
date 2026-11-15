"""
Replay and save per-query full rankings on index/test_gallery/ (N=1,290)
for Prompt 14h Step 1, with strict numerical verification against reports/phase5c/final_eval_results.json.

Strict Invariants:
- Read-only replay: NO retraining, NO checkpoint modification, NO code changes to inference.
- Exact locked inference pipelines, tie-breaking (epsilon=1e-5), seed 42, and locked definitions.
- Saves full ranked lists of 1,290 product IDs and scores for each query.
- Compares point estimates against final_eval_results.json (tolerance 1e-9).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.canonical_eval import (
    apply_conservative_tie_break,
    compute_weighted_attribute_match,
)
from src.eval.locked_final_definitions import (
    GALLERY_SIZE_TEST_LOCKED,
    SEED_LOCKED,
    compute_full_ranking_ap,
    compute_hit_rate_at_k,
    compute_secondary_within_group_metrics,
    compute_true_recall_at_k,
)
from src.eval.metrics import is_item_relevant, load_gender_gt_policy
from src.serve.cir_enhancer import CatalogModifierTranslator, score_cir_single_query
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import restore_diacritics_for_query
from src.serve.search_engine import ECommerceSearchEngine
from src.serve.template_cache import TemplateCacheV5, encode_texts_with_template_cache


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _zscore_rows(m: np.ndarray) -> np.ndarray:
    mu = np.mean(m, axis=1, keepdims=True)
    sd = np.maximum(np.std(m, axis=1, keepdims=True), 1e-8)
    return ((m - mu) / sd).astype(np.float32)


def main() -> int:
    start_time = time.time()
    print("================================================================================")
    print("PROMPT 14h: REPLAY PER-QUERY RANKINGS & NUMERICAL VERIFICATION (N=1,290)")
    print(f"Seed: {SEED_LOCKED}")
    print("================================================================================")

    rankings_dir = PROJECT_ROOT / "reports/phase5c/rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load resources
    engine = ECommerceSearchEngine(index_dir="index/test_gallery", use_mean_centering=False)
    id_map = engine.id_map
    n_gallery = len(id_map)
    assert n_gallery == GALLERY_SIZE_TEST_LOCKED, f"Gallery size {n_gallery} != {GALLERY_SIZE_TEST_LOCKED}"
    pid_to_idx: Dict[str, int] = {str(item.get("product_id") or item.get("item_id")): idx for idx, item in enumerate(id_map)}
    idx_to_pid: Dict[int, str] = {idx: str(item.get("product_id") or item.get("item_id")) for idx, item in enumerate(id_map)}

    tpl_cache = TemplateCacheV5.load(
        PROJECT_ROOT / "index/template_cache_v5",
        expected_checkpoint_hash="e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d",
    )
    parser = VietnameseQueryParserV2Conf()
    gpol = json.loads((PROJECT_ROOT / "eval/gender_gt_policy.json").read_text(encoding="utf-8"))
    strict_cats = gpol["policies"]["hybrid"]["strict_categories"]

    styles_df = pd.read_csv(PROJECT_ROOT / "data/fashion-dataset/styles.csv", on_bad_lines="skip")
    styles_df["id"] = styles_df["id"].astype(str)
    styles_map = styles_df.set_index("id").to_dict(orient="index")

    # Load caption embeddings
    embs_caption_a = np.load(PROJECT_ROOT / "index/test_gallery_text_v5/internvl2_only_embeddings.npy")
    embs_caption_b = np.load(PROJECT_ROOT / "index/test_gallery_text_v5/internvl2_plus_aux_embeddings.npy")

    # 2. Text queries
    heldout_v2_obj = json.loads((PROJECT_ROOT / "eval/test_queries_heldout_v2.json").read_text(encoding="utf-8"))
    queries: List[Dict[str, Any]] = heldout_v2_obj["queries"]
    n_t2i = len(queries)
    print(f"Total Held-out v2 Queries: {n_t2i}")

    raw_texts = [q["query"] for q in queries]
    rewritten_texts = [restore_diacritics_for_query(q) for q in raw_texts]
    parsed_std = [parser.parse_with_confidence(qt, fuzzy_typo=False) for qt in rewritten_texts]
    parsed_fuz = [parser.parse_with_confidence(qt, fuzzy_typo=True) for qt in rewritten_texts]

    # Pre-encode raw queries
    device = engine.device
    raw_vecs_list = []
    batch_sz = 32
    for b_start in range(0, n_t2i, batch_sz):
        b_txts = rewritten_texts[b_start : b_start + batch_sz]
        inp = engine.preprocessor.preprocess_texts(b_txts, device=device)
        with torch.no_grad():
            v = engine.model.encode_text(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
            raw_vecs_list.append(v.float().cpu().numpy())
    vecs_raw = np.vstack(raw_vecs_list)
    vecs_raw = vecs_raw / np.maximum(np.linalg.norm(vecs_raw, axis=1, keepdims=True), 1e-12)

    vecs_cached_std, _ = encode_texts_with_template_cache(engine, raw_texts, parsed_std, tpl_cache)
    vecs_cached_fuz, _ = encode_texts_with_template_cache(engine, raw_texts, parsed_fuz, tpl_cache)

    z_a_cached = _zscore_rows(vecs_cached_std @ embs_caption_a.T)
    z_b_cached = _zscore_rows(vecs_cached_std @ embs_caption_b.T)

    t2i_configs = [
        {"config_id": "model_only", "is_model_only": True, "gamma": 0.0, "i3": "none", "fuzzy": False, "tg": "none"},
        {"config_id": "cfg_v5_01", "is_model_only": False, "gamma": 1.5, "i3": "none", "fuzzy": False, "tg": "none"},
        {"config_id": "cfg_v5_06", "is_model_only": False, "gamma": 1.5, "i3": "cached", "fuzzy": True, "tg": "none"},
        {"config_id": "cfg_v5_09", "is_model_only": False, "gamma": 1.0, "i3": "cached", "fuzzy": False, "tg": "none"},
        {"config_id": "cfg_v5_08b", "is_model_only": False, "gamma": 1.5, "i3": "cached", "fuzzy": False, "tg": "internvl2_plus_aux", "alpha": 0.85},
        {"config_id": "cfg_v5_08a", "is_model_only": False, "gamma": 1.5, "i3": "cached", "fuzzy": False, "tg": "internvl2_only", "alpha": 0.85},
    ]

    replayed_t2i_metrics: Dict[str, Dict[str, float]] = {}
    saved_ranking_hashes: Dict[str, str] = {}

    for cfg in t2i_configs:
        cid = cfg["config_id"]
        is_model_only = cfg["is_model_only"]
        fuzzy = cfg["fuzzy"]
        i3_mode = cfg["i3"]
        gamma = cfg["gamma"]
        tg = cfg["tg"]
        t_alpha = cfg.get("alpha", 1.0)

        if is_model_only:
            q_vecs = vecs_raw
            parsed_list = parsed_std
        elif i3_mode == "cached":
            q_vecs = vecs_cached_fuz if fuzzy else vecs_cached_std
            parsed_list = parsed_fuz if fuzzy else parsed_std
        else:
            q_vecs = vecs_raw
            parsed_list = parsed_fuz if fuzzy else parsed_std

        extra_mat = None
        extra_w = 0.0
        if tg == "internvl2_plus_aux":
            extra_mat = z_b_cached
            extra_w = (1.0 - t_alpha) / t_alpha
        elif tg == "internvl2_only":
            extra_mat = z_a_cached
            extra_w = (1.0 - t_alpha) / t_alpha

        cos_matrix = (q_vecs @ engine.embeddings.T).astype(np.float64)

        per_q_r1, per_q_r5, per_q_r10, per_q_true_r10, per_q_ap = [], [], [], [], []
        per_q_sec_ap, per_q_sec_r1, per_q_sec_r5 = [], [], []
        saved_ranking_records: List[Dict[str, Any]] = []

        for q_idx, q in enumerate(queries):
            base_cos_row = cos_matrix[q_idx]
            gt_pids = set(map(str, q["relevant_ids"]))
            gt_indices = {pid_to_idx[pid] for pid in gt_pids if pid in pid_to_idx}

            if is_model_only:
                scored = [(float(base_cos_row[i]), i) for i in range(n_gallery)]
            else:
                parsed = parsed_list[q_idx]
                scored = []
                for idx in range(n_gallery):
                    s_val = float(base_cos_row[idx])
                    meta = id_map[idx]
                    mf = compute_weighted_attribute_match(
                        parsed, meta, weights=None, gender_policy="hybrid", strict_categories=strict_cats
                    )
                    if mf > 0.0 and gamma > 0.0:
                        tier_b = 0.5 if mf >= 0.999 else 0.0
                        s_val += gamma * (mf + tier_b)
                    if extra_mat is not None and extra_w != 0.0:
                        s_val += extra_w * float(extra_mat[q_idx, idx])
                    scored.append((s_val, idx))

            tied_ranked_indices = apply_conservative_tie_break(
                scored,
                is_relevant_fn=lambda idx: (idx in gt_indices),
                epsilon=1e-5,
            )

            rels = [1 if idx in gt_indices else 0 for idx in tied_ranked_indices]
            r1 = compute_hit_rate_at_k(rels, 1)
            r5 = compute_hit_rate_at_k(rels, 5)
            r10 = compute_hit_rate_at_k(rels, 10)
            true_r10 = compute_true_recall_at_k(rels, 10, total_relevant=len(gt_indices))
            ap = compute_full_ranking_ap(rels, total_relevant=len(gt_indices))

            per_q_r1.append(r1)
            per_q_r5.append(r5)
            per_q_r10.append(r10)
            per_q_true_r10.append(true_r10)
            per_q_ap.append(ap)

            # Secondary GT metrics
            sec_res = {"within_group_ap": 0.0, "within_group_r1": 0.0, "within_group_r5": 0.0}
            if q.get("has_usage_season_signal") and q.get("secondary"):
                sec_info = q["secondary"]
                sec_kind = sec_info["kind"]
                sec_val = str(sec_info["value"]).strip().lower()
                sec_indices = set()
                for pid in gt_pids:
                    row_info = styles_map.get(str(pid))
                    if row_info and str(row_info.get(sec_kind, "")).strip().lower() == sec_val:
                        sec_indices.add(pid_to_idx[str(pid)])
                sec_res = compute_secondary_within_group_metrics(tied_ranked_indices, gt_indices, sec_indices)
                per_q_sec_ap.append(sec_res["within_group_ap"])
                per_q_sec_r1.append(sec_res["within_group_r1"])
                per_q_sec_r5.append(sec_res["within_group_r5"])

            # Map scores to ranked product IDs
            scored_dict = {idx: s for s, idx in scored}
            ranked_pids = [idx_to_pid[idx] for idx in tied_ranked_indices]
            ranked_scores = [round(float(scored_dict[idx]), 6) for idx in tied_ranked_indices]

            saved_ranking_records.append({
                "query_id": q["query_id"],
                "query": q["query"],
                "ranked_product_ids": ranked_pids,
                "scores": ranked_scores,
                "top10_ids": ranked_pids[:10],
                "r1": r1,
                "r5": r5,
                "r10": r10,
                "true_r10": true_r10,
                "ap": ap,
                "secondary_within_group_ap": sec_res["within_group_ap"],
            })

        replayed_t2i_metrics[cid] = {
            "recall@1": float(np.mean(per_q_r1)),
            "recall@5": float(np.mean(per_q_r5)),
            "recall@10": float(np.mean(per_q_r10)),
            "true_recall@10": float(np.mean(per_q_true_r10)),
            "mAP": float(np.mean(per_q_ap)),
            "sec_ap": float(np.mean(per_q_sec_ap)) if per_q_sec_ap else 0.0,
            "sec_r1": float(np.mean(per_q_sec_r1)) if per_q_sec_r1 else 0.0,
            "sec_r5": float(np.mean(per_q_sec_r5)) if per_q_sec_r5 else 0.0,
        }

        # Save rankings JSON
        out_f = rankings_dir / f"t2i_{cid}.json"
        out_f.write_text(json.dumps(saved_ranking_records, indent=2, ensure_ascii=False), encoding="utf-8")
        h_f = sha256_file(out_f)
        saved_ranking_hashes[f"t2i_{cid}"] = h_f
        print(f"  [Saved T2I Ranking] {cid} -> {out_f.name} (SHA-256: {h_f[:16]}...)")

    # 3. CIR Replay
    cir_test_obj = json.loads((PROJECT_ROOT / "eval/cir_eval_test.json").read_text(encoding="utf-8"))
    cir_pairs: List[Dict[str, Any]] = cir_test_obj["pairs"]
    n_cir = len(cir_pairs)
    print(f"\nTotal Test CIR Pairs: {n_cir}")

    translator = CatalogModifierTranslator(PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet")
    cir_precomp: List[Dict[str, Any]] = []
    for p_idx, p in enumerate(cir_pairs):
        ref_pid = str(p["ref_product_id"])
        ref_gal_idx = pid_to_idx[ref_pid]
        ref_vec_np = engine.embeddings[ref_gal_idx : ref_gal_idx + 1]
        ref_img_gpu = torch.from_numpy(ref_vec_np).to(device)
        if device == "cuda":
            ref_img_gpu = ref_img_gpu.half()

        mod_full = str(p["modifier_vi"])
        mod_none = strip_diacritics(mod_full)
        primary_dia = "full" if (p_idx % 2 == 0) else "none"
        mod_primary = mod_full if primary_dia == "full" else mod_none

        variants: Dict[str, Any] = {}
        for v_key, raw_mod in (("primary", mod_primary), ("full", mod_full), ("none", mod_none)):
            qt_vi = restore_diacritics_for_query(raw_mod)
            en_mod = translator.translate(raw_mod)
            parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)

            in_vi = engine.preprocessor.preprocess_single_text(qt_vi, device=device)
            with torch.no_grad():
                emb_vi = engine.model.encode_text(
                    input_ids=in_vi["input_ids"], attention_mask=in_vi.get("attention_mask")
                )
            variants[v_key] = {
                "raw_mod": raw_mod,
                "qt_vi": qt_vi,
                "en_mod": en_mod,
                "parsed_mod": parsed_mod,
                "emb_vi_gpu": emb_vi,
            }

        t_attrs = p["target_attrs"]
        cluster_id = f"{t_attrs['articleType']}__{t_attrs['gender']}__{t_attrs['baseColour']}"
        gt_indices = {pid_to_idx[str(g)] for g in p["gt_product_ids"] if str(g) in pid_to_idx}

        cir_precomp.append({
            "pair_idx": p_idx,
            "id": p["id"],
            "ref_gal_idx": ref_gal_idx,
            "ref_image_path": str(PROJECT_ROOT / p["ref_image_path"]),
            "ref_img_gpu": ref_img_gpu,
            "ref_vec_np": ref_vec_np.flatten(),
            "changed_attr": p["changed_attr"],
            "primary_dia": primary_dia,
            "cluster_id": cluster_id,
            "gt_indices": gt_indices,
            "variants": variants,
        })

    cir_configs = [
        {"config_id": "cir_v1_01", "c3": False, "gamma": 0.0},
        {"config_id": "cir_v1_05", "c3": True, "gamma": 1.0},
    ]

    replayed_cir_metrics: Dict[str, Dict[str, float]] = {}

    for cfg in cir_configs:
        cid = cfg["config_id"]
        use_c3 = cfg["c3"]
        gamma = cfg["gamma"]

        per_pair_r1, per_pair_r5, per_pair_r10, per_pair_ap = [], [], [], []
        saved_cir_records: List[Dict[str, Any]] = []

        for item in cir_precomp:
            var = item["variants"]["primary"]
            final_scores, raw_ranked = score_cir_single_query(
                engine=engine,
                ref_img_emb_gpu=item["ref_img_gpu"],
                mod_txt_emb_gpu=var["emb_vi_gpu"],
                parsed_mod=var["parsed_mod"],
                tpl_cache=tpl_cache,
                strict_cats=strict_cats,
                beta=1.0,
                use_c3_bonus=use_c3,
                gamma=gamma,
                rocchio_k=0,
                rocchio_beta=0.0,
            )
            gt_set = item["gt_indices"]

            scored_items = [(float(final_scores[idx]), idx) for idx in range(n_gallery)]
            tied_ranked = apply_conservative_tie_break(
                scored_items,
                is_relevant_fn=lambda idx: idx in gt_set,
                epsilon=1e-5,
            )

            rels = [1 if idx in gt_set else 0 for idx in tied_ranked]
            r1 = compute_hit_rate_at_k(rels, 1)
            r5 = compute_hit_rate_at_k(rels, 5)
            r10 = compute_hit_rate_at_k(rels, 10)
            ap = compute_full_ranking_ap(rels, total_relevant=len(gt_set))

            per_pair_r1.append(r1)
            per_pair_r5.append(r5)
            per_pair_r10.append(r10)
            per_pair_ap.append(ap)

            scored_dict = {idx: s for s, idx in scored_items}
            ranked_pids = [idx_to_pid[idx] for idx in tied_ranked]
            ranked_scores = [round(float(scored_dict[idx]), 6) for idx in tied_ranked]

            saved_cir_records.append({
                "pair_id": item["id"],
                "changed_attr": item["changed_attr"],
                "dia_class": item["primary_dia"],
                "ranked_product_ids": ranked_pids,
                "scores": ranked_scores,
                "top10_ids": ranked_pids[:10],
                "r1": r1,
                "r5": r5,
                "r10": r10,
                "ap": ap,
            })

        replayed_cir_metrics[cid] = {
            "recall@1": float(np.mean(per_pair_r1)),
            "recall@5": float(np.mean(per_pair_r5)),
            "recall@10": float(np.mean(per_pair_r10)),
            "mAP": float(np.mean(per_pair_ap)),
        }

        out_f = rankings_dir / f"{cid}_primary.json"
        out_f.write_text(json.dumps(saved_cir_records, indent=2, ensure_ascii=False), encoding="utf-8")
        h_f = sha256_file(out_f)
        saved_ranking_hashes[f"{cid}_primary"] = h_f
        print(f"  [Saved CIR Ranking] {cid} -> {out_f.name} (SHA-256: {h_f[:16]}...)")

    # 4. Step 3d: CIR Reference Image Only (I2I Baseline) on cir_eval_test.json
    print("\n  [Evaluating CIR Reference-Only Baseline (I2I on cir_eval_test)]...")
    per_pair_r1_i2i, per_pair_r5_i2i, per_pair_r10_i2i, per_pair_ap_i2i = [], [], [], []
    saved_cir_i2i_records: List[Dict[str, Any]] = []

    for item in cir_precomp:
        ref_vec = item["ref_vec_np"]
        cos_scores = (engine.embeddings @ ref_vec).astype(np.float64)
        gt_set = item["gt_indices"]

        scored_items = [(float(cos_scores[idx]), idx) for idx in range(n_gallery)]
        tied_ranked = apply_conservative_tie_break(
            scored_items,
            is_relevant_fn=lambda idx: idx in gt_set,
            epsilon=1e-5,
        )

        rels = [1 if idx in gt_set else 0 for idx in tied_ranked]
        r1 = compute_hit_rate_at_k(rels, 1)
        r5 = compute_hit_rate_at_k(rels, 5)
        r10 = compute_hit_rate_at_k(rels, 10)
        ap = compute_full_ranking_ap(rels, total_relevant=len(gt_set))

        per_pair_r1_i2i.append(r1)
        per_pair_r5_i2i.append(r5)
        per_pair_r10_i2i.append(r10)
        per_pair_ap_i2i.append(ap)

        scored_dict = {idx: s for s, idx in scored_items}
        ranked_pids = [idx_to_pid[idx] for idx in tied_ranked]
        ranked_scores = [round(float(scored_dict[idx]), 6) for idx in tied_ranked]

        saved_cir_i2i_records.append({
            "pair_id": item["id"],
            "changed_attr": item["changed_attr"],
            "dia_class": item["primary_dia"],
            "ranked_product_ids": ranked_pids,
            "scores": ranked_scores,
            "top10_ids": ranked_pids[:10],
            "r1": r1,
            "r5": r5,
            "r10": r10,
            "ap": ap,
        })

    replayed_cir_metrics["cir_i2i_baseline"] = {
        "recall@1": float(np.mean(per_pair_r1_i2i)),
        "recall@5": float(np.mean(per_pair_r5_i2i)),
        "recall@10": float(np.mean(per_pair_r10_i2i)),
        "mAP": float(np.mean(per_pair_ap_i2i)),
    }
    out_f = rankings_dir / "cir_i2i_baseline.json"
    out_f.write_text(json.dumps(saved_cir_i2i_records, indent=2, ensure_ascii=False), encoding="utf-8")
    h_f = sha256_file(out_f)
    saved_ranking_hashes["cir_i2i_baseline"] = h_f
    print(f"  [Saved CIR Baseline] cir_i2i_baseline -> {out_f.name} (SHA-256: {h_f[:16]}...)")

    # -------------------------------------------------------------------------
    # MANDATORY VERIFICATION AGAINST FROZEN final_eval_results.json
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("MANDATORY NUMERICAL VERIFICATION (TOLERANCE 1e-9)")
    print("================================================================================")
    frozen_json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    frozen_data = json.loads(frozen_json_path.read_text(encoding="utf-8"))

    mismatches: List[str] = []
    TOL = 1e-9

    print(f"{'Config ID':15s} | {'Metric':15s} | {'Replayed Value':18s} | {'Frozen JSON Value':18s} | {'Diff':12s} | {'Status'}")
    print("-" * 95)

    def _check(cfg_name: str, metric_name: str, val_rep: float, val_froz: float) -> None:
        diff = abs(val_rep - val_froz)
        status = "EXACT MATCH" if diff <= TOL else "MISMATCH"
        print(f"{cfg_name:15s} | {metric_name:15s} | {val_rep:18.9f} | {val_froz:18.9f} | {diff:12.2e} | {status}")
        if diff > TOL:
            mismatches.append(f"{cfg_name} {metric_name}: replayed {val_rep:.9f} vs frozen {val_froz:.9f} (diff {diff:.2e})")

    # Check T2I
    frozen_t2i = frozen_data["part_b1_t2i_quality"]["results_by_config"]
    for cid in ["model_only", "cfg_v5_01", "cfg_v5_06", "cfg_v5_09", "cfg_v5_08b"]:
        rep = replayed_t2i_metrics[cid]
        froz_s = frozen_t2i[cid]["summary"]
        _check(cid, "recall@1", rep["recall@1"], froz_s["recall@1"]["mean"])
        _check(cid, "recall@5", rep["recall@5"], froz_s["recall@5"]["mean"])
        _check(cid, "recall@10", rep["recall@10"], froz_s["recall@10"]["mean"])
        _check(cid, "true_recall@10", rep["true_recall@10"], froz_s["true_recall@10"]["mean"])
        _check(cid, "mAP", rep["mAP"], froz_s["mAP"]["mean"])

        froz_sec = frozen_t2i[cid].get("secondary_gt_non_circular", {})
        if froz_sec and "within_group_ap" in froz_sec:
            _check(cid, "sec_ap", rep["sec_ap"], froz_sec["within_group_ap"]["mean"])
            _check(cid, "sec_r1", rep["sec_r1"], froz_sec["within_group_r1"]["mean"])
            _check(cid, "sec_r5", rep["sec_r5"], froz_sec["within_group_r5"]["mean"])

    # Check CIR
    frozen_cir = frozen_data["part_b2_cir_quality"]["results_by_config"]
    for cid in ["cir_v1_01", "cir_v1_05"]:
        rep = replayed_cir_metrics[cid]
        froz_s = frozen_cir[cid]["primary_50_50"]["summary"]
        _check(cid, "recall@1", rep["recall@1"], froz_s["recall@1"]["mean"])
        _check(cid, "recall@5", rep["recall@5"], froz_s["recall@5"]["mean"])
        _check(cid, "recall@10", rep["recall@10"], froz_s["recall@10"]["mean"])
        _check(cid, "mAP", rep["mAP"], froz_s["mAP"]["mean"])

    # Save manifest of rankings
    manifest_rankings = {
        "status": "deterministic_replay_matched_frozen_json" if not mismatches else "mismatch_detected",
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED_LOCKED,
        "n_gallery": n_gallery,
        "tolerance": TOL,
        "total_checks": 5 * 5 + 5 * 3 + 2 * 4,
        "mismatches_count": len(mismatches),
        "ranking_file_hashes": saved_ranking_hashes,
        "replayed_t2i_metrics": replayed_t2i_metrics,
        "replayed_cir_metrics": replayed_cir_metrics,
    }
    manifest_f = rankings_dir / "manifest.json"
    manifest_f.write_text(json.dumps(manifest_rankings, indent=2), encoding="utf-8")
    print(f"\n[Saved Rankings Manifest] {manifest_f}")

    if mismatches:
        print("\n[CRITICAL ERROR] Mismatches detected with frozen JSON:")
        for m in mismatches:
            print(f"  - {m}")
        print("HALTING execution per Prompt 14h Step 1 rules.")
        return 1

    print("\n[SUCCESS] 100% EXACT MATCH (0 mismatches). Tái tạo xác định, khớp JSON đóng băng hoàn toàn.")
    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
