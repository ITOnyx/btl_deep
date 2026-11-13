"""
Hàm đánh giá Canonical duy nhất cho toàn bộ Pha 5b (Bước 1c, 2, 3, 4, 5).
Đảm bảo tính nhất quán tuyệt đối giữa mọi script đánh giá và báo cáo.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.metrics import (
    bootstrap_ci_diff,
    compute_average_precision,
    compute_hubness_stats,
    compute_recall_at_k,
    compute_retrieval_summary,
    is_item_relevant,
)
from src.serve.parser import compute_soft_attribute_match


def compute_true_recall_at_k(relevances: Sequence[int], k: int, total_relevant: int) -> float:
    """True-Recall@K (tỉ lệ ảnh đúng được thu hồi trong top-K, bổ sung bên cạnh hit-rate)."""
    if total_relevant <= 0:
        return 0.0
    return float(sum(relevances[:k]) / total_relevant)


def bootstrap_ci_absolute(
    scores: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap 95% CI cho chính giá trị tuyệt đối của một chỉ số."""
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(arr[idx]))
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def compute_weighted_attribute_match(
    parsed: Dict[str, Any],
    item_meta: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    gender_policy: str = "hybrid",
    strict_categories: Optional[Sequence[str]] = None,
) -> float:
    """
    Tính điểm khớp thuộc tính mềm có trọng số theo slot.
    - Nếu weights is None: dùng đúng compute_soft_attribute_match gốc (trọng số đều).
    - Nếu có weights (vd: {'articleType': 0.6, 'gender': 0.2, 'baseColour': 0.2}):
      chỉ tính trên các slot mà parser trả về (non-empty); chuẩn hóa tổng trọng số các slot có mặt về 1.0.
    """
    if weights is None:
        return compute_soft_attribute_match(parsed, item_meta, gender_policy=gender_policy)

    if not parsed:
        return 0.0

    from src.eval.metrics import is_gender_match

    total_w = 0.0
    matched_w = 0.0

    p_types = parsed.get("articleType") or []
    if p_types:
        w = float(weights.get("articleType", 0.5))
        total_w += w
        if str(item_meta.get("articleType", "")).strip() in p_types:
            matched_w += w

    p_gender = parsed.get("gender")
    if p_gender:
        w = float(weights.get("gender", 0.2))
        total_w += w
        if is_gender_match(
            p_gender,
            item_meta.get("gender"),
            item_meta.get("articleType"),
            policy=gender_policy,
            strict_categories=strict_categories,
        ):
            matched_w += w

    p_colours = parsed.get("baseColour") or []
    if p_colours:
        w = float(weights.get("baseColour", 0.3))
        total_w += w
        if str(item_meta.get("baseColour", "")).strip() in p_colours:
            matched_w += w

    if total_w <= 0.0:
        return 0.0
    return float(matched_w / total_w)


def apply_conservative_tie_break(
    scored_items: List[Tuple[float, int]],
    is_relevant_fn,
    epsilon: float = 1e-5,
) -> List[int]:
    """
    Áp dụng tie_rule='conservative' (config.yaml) với ngưỡng epsilon=1e-5:
    Sắp xếp theo điểm giảm dần; trong cùng nhóm các phần tử có hiệu điểm <= epsilon
    so với phần tử đầu nhóm, xếp ảnh KHÔNG liên quan (rel=0) lên trước ảnh liên quan (rel=1).
    """
    if not scored_items:
        return []
    # Sort giảm dần theo score, ổn định
    ordered = sorted(scored_items, key=lambda x: x[0], reverse=True)
    result_indices: List[int] = []
    n = len(ordered)
    i = 0
    while i < n:
        anchor_score = ordered[i][0]
        j = i + 1
        while j < n and (anchor_score - ordered[j][0]) <= epsilon:
            j += 1
        if j - i == 1:
            result_indices.append(ordered[i][1])
        else:
            # Trong nhóm tie [i, j): rel=0 xếp trước rel=1
            group = ordered[i:j]
            group_sorted = sorted(group, key=lambda x: (1 if is_relevant_fn(x[1]) else 0, -x[0]))
            result_indices.extend(x[1] for x in group_sorted)
        i = j
    return result_indices


def evaluate_t2i_canonical(
    engine,
    queries: List[Dict[str, Any]],
    query_vecs: np.ndarray,
    parsed_queries: List[Dict[str, Any]],
    gender_policy_cfg: Dict[str, Any],
    mode: str = "system",
    candidate_pool: int = 100,
    eval_top_k: int = 100,
    gamma: float = 0.3,
    slot_weights: Optional[Dict[str, float]] = None,
    hard_tier: bool = False,
    use_exact_when_full: bool = True,
    force_exact: bool = True,
    tie_rule: str = "conservative",
    tie_epsilon: float = 1e-5,
    zscore_cosine: bool = False,
    bm25_scores: Optional[np.ndarray] = None,
    bm25_weight: float = 0.2,
    compute_breakdowns: bool = False,
    policies: Sequence[str] = ("hybrid",),
    min_relevant_per_query: int = 1,
) -> Dict[str, Any]:
    """
    Hàm đánh giá Canonical duy nhất cho Text-to-Image Retrieval.
    - Nếu mode == 'model_only': xếp hạng thuần theo cosine.
      + Nếu candidate_pool >= n_gallery và use_exact_when_full=True: dùng exact matmul toàn gallery rồi lấy [:eval_top_k].
      + Ngược lại dùng engine.search_by_vector(..., mode='model_only', top_k=eval_top_k).
    - Nếu mode == 'system':
      + Nếu candidate_pool >= n_gallery và use_exact_when_full=True: lấy toàn bộ n_gallery bằng exact matmul để re-rank, sau đó cắt [:eval_top_k].
      + Ngược lại lấy top candidate_pool từ HNSW qua engine.hnsw.search, sau đó re-rank và cắt [:eval_top_k].
    """
    strict_cats = gender_policy_cfg["policies"]["hybrid"]["strict_categories"]
    id_map = engine.id_map
    n_gallery = len(id_map)
    eff_pool = min(candidate_pool, n_gallery)

    ranked_indices_list: List[List[int]] = []
    scored_items_per_query: List[List[Tuple[float, int]]] = []

    use_exact = force_exact or (use_exact_when_full and eff_pool >= n_gallery)
    full_cos_matrix = None
    if use_exact:
        q_mat = engine._apply_mean_centering(query_vecs)
        full_cos_matrix = q_mat @ engine.embeddings.T  # (n_queries, n_gallery)

    for i in range(len(queries)):
        q_vec = engine._apply_mean_centering(query_vecs[i : i + 1])

        if full_cos_matrix is not None:
            cos_row = full_cos_matrix[i].astype(np.float64)
            if zscore_cosine:
                mu = float(np.mean(cos_row))
                sd = float(np.std(cos_row))
                cos_row = (cos_row - mu) / max(sd, 1e-8)

            if eff_pool < n_gallery:
                pool_idxs = np.argsort(-cos_row, kind="mergesort")[:eff_pool]
            else:
                pool_idxs = np.arange(n_gallery)

            if mode == "model_only":
                scored = [(float(cos_row[int(idx)]), int(idx)) for idx in pool_idxs]
            else:
                parsed = parsed_queries[i]
                scored = []
                for idx_np in pool_idxs:
                    idx = int(idx_np)
                    base_s = float(cos_row[idx])
                    meta = id_map[idx]
                    mf = compute_weighted_attribute_match(
                        parsed, meta, weights=slot_weights, gender_policy="hybrid", strict_categories=strict_cats
                    )
                    if hard_tier:
                        s_val = mf * 100.0 + base_s
                    else:
                        s_val = base_s + gamma * mf
                    if bm25_scores is not None:
                        s_val += bm25_weight * float(bm25_scores[i, idx])
                    scored.append((s_val, idx))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored_items_per_query.append(scored)
            ranked_indices_list.append([x[1] for x in scored[:eval_top_k]])
        else:
            # Đường HNSW (chỉ dùng khi force_exact=False để tái hiện baseline Pha 5)
            fetch_k = eval_top_k if mode == "model_only" else max(eval_top_k, eff_pool)
            scores, indices = engine.hnsw.search(q_vec, top_k=min(fetch_k, n_gallery))
            parsed = parsed_queries[i]
            scored = []
            for idx, base_s in zip(indices[0], scores[0]):
                if idx < 0:
                    continue
                idx_int = int(idx)
                s_val = float(base_s)
                if mode == "model_only":
                    scored.append((s_val, idx_int))
                else:
                    meta = id_map[idx_int]
                    mf = compute_weighted_attribute_match(
                        parsed, meta, weights=slot_weights, gender_policy="hybrid", strict_categories=strict_cats
                    )
                    if hard_tier:
                        scored.append((mf * 100.0 + s_val, idx_int))
                    else:
                        scored.append((s_val + gamma * mf, idx_int))
            scored.sort(key=lambda x: x[0], reverse=True)
            scored_items_per_query.append(scored)
            ranked_indices_list.append([x[1] for x in scored[:eval_top_k]])

    hubness = compute_hubness_stats(
        [idxs[:10] for idxs in ranked_indices_list],
        gallery_size=n_gallery,
        top_k=10,
    )

    policy_outputs: Dict[str, Any] = {}
    for pol in policies:
        ranked_rels: List[List[int]] = []
        tot_rels: List[int] = []
        per_q_r1, per_q_r5, per_q_r10, per_q_ap, per_q_true_r10 = [], [], [], [], []
        valid_mask: List[bool] = []

        for i, q in enumerate(queries):
            spec = q["spec"]
            rel_flags = [is_item_relevant(spec, meta, pol, strict_cats) for meta in id_map]
            total_rel = sum(1 for x in rel_flags if x)
            tot_rels.append(total_rel)
            valid_mask.append(total_rel >= min_relevant_per_query)

            if tie_rule == "conservative":
                tied_indices = apply_conservative_tie_break(
                    scored_items_per_query[i],
                    is_relevant_fn=lambda idx: rel_flags[idx],
                    epsilon=tie_epsilon,
                )[:eval_top_k]
            else:
                tied_indices = ranked_indices_list[i]

            rels = [1 if rel_flags[idx] else 0 for idx in tied_indices]
            ranked_rels.append(rels)
            per_q_r1.append(compute_recall_at_k(rels, 1))
            per_q_r5.append(compute_recall_at_k(rels, 5))
            per_q_r10.append(compute_recall_at_k(rels, 10))
            per_q_ap.append(compute_average_precision(rels, total_relevant=total_rel) if total_rel > 0 else 0.0)
            per_q_true_r10.append(compute_true_recall_at_k(rels, 10, total_rel))

        valid_idxs = [i for i, ok in enumerate(valid_mask) if ok]
        summary_valid = compute_retrieval_summary(
            [ranked_rels[i] for i in valid_idxs],
            total_relevants=[tot_rels[i] for i in valid_idxs],
            topk=(1, 5, 10),
        )
        summary_valid["true_recall@10"] = float(np.mean([per_q_true_r10[i] for i in valid_idxs])) if valid_idxs else 0.0
        summary_valid["n_queries_total"] = len(queries)
        summary_valid["n_queries_valid"] = len(valid_idxs)

        policy_outputs[pol] = {
            "summary": summary_valid,
            "per_query_valid": {
                "indices": valid_idxs,
                "recall@1": [per_q_r1[i] for i in valid_idxs],
                "recall@5": [per_q_r5[i] for i in valid_idxs],
                "recall@10": [per_q_r10[i] for i in valid_idxs],
                "ap": [per_q_ap[i] for i in valid_idxs],
                "true_recall@10": [per_q_true_r10[i] for i in valid_idxs],
            },
        }

        if compute_breakdowns and pol == "hybrid":
            breakdowns: Dict[str, Dict[str, Any]] = {
                "by_thin_slot": {},
                "by_has_diacritics": {},
                "by_gender": {},
                "by_colour": {},
            }
            groups: Dict[str, Dict[str, List[int]]] = {
                "by_thin_slot": defaultdict(list),
                "by_has_diacritics": defaultdict(list),
                "by_gender": defaultdict(list),
                "by_colour": defaultdict(list),
            }
            for i in valid_idxs:
                q = queries[i]
                groups["by_thin_slot"][str(q.get("category", "unknown"))].append(i)
                groups["by_has_diacritics"][str(q.get("has_diacritics", True))].append(i)
                groups["by_gender"][str(q["spec"].get("gender") or "Unspecified")].append(i)
                colours = q["spec"].get("baseColour") or ["Unspecified"]
                for c in colours:
                    groups["by_colour"][str(c)].append(i)

            for b_name, g_map in groups.items():
                for g_key, idxs in sorted(g_map.items()):
                    breakdowns[b_name][g_key] = compute_retrieval_summary(
                        [ranked_rels[i] for i in idxs],
                        total_relevants=[tot_rels[i] for i in idxs],
                        topk=(1, 5, 10),
                    )
            policy_outputs[pol]["breakdowns"] = breakdowns

    return {
        "policies": policy_outputs,
        "hubness_top10": hubness,
        "ranked_indices": ranked_indices_list,
    }
