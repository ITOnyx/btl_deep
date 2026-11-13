"""
Chẩn đoán bổ sung Bước 1b (Pha 5b) trên DEV SET (eval/handwritten_queries.json) và index/test_gallery/.
Tuân thủ Luật 1, 3, 6, 7, C.5, C.10. Chỉ dùng để chẩn đoán, KHÔNG chọn tham số từ đây.
"""
import os
import sys
import json
import pathlib
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from src.serve.search_engine import ECommerceSearchEngine
from src.serve.parser import compute_soft_attribute_match
from src.eval.metrics import (
    load_gender_gt_policy,
    is_item_relevant,
    compute_recall_at_k,
    compute_average_precision,
    compute_retrieval_summary,
)


def _rank_all_items(
    cosine_scores: np.ndarray,
    id_map: list,
    parsed_query: dict,
    gamma: float = 0.3,
    mode: str = "system",
    hard_tier: bool = False,
) -> list:
    """
    Xếp hạng toàn bộ gallery (1.290 ảnh) theo cấu hình chỉ định.
    Trả về danh sách dict gồm index_pos, cosine, bonus, final_score, n_matched, n_spec.
    """
    n_spec = 0
    if parsed_query:
        if parsed_query.get("articleType"):
            n_spec += 1
        if parsed_query.get("gender"):
            n_spec += 1
        if parsed_query.get("baseColour"):
            n_spec += 1

    items = []
    for idx in range(len(id_map)):
        meta = id_map[idx]
        cos = float(cosine_scores[idx])
        bonus = 0.0
        match_frac = 0.0
        if mode in ("system", "oracle") and parsed_query is not None:
            match_frac = compute_soft_attribute_match(parsed_query, meta, gender_policy="hybrid")
            bonus = gamma * match_frac

        n_matched = round(match_frac * n_spec) if n_spec > 0 else 0
        items.append({
            "index_pos": idx,
            "item_id": meta.get("item_id"),
            "articleType": meta.get("articleType"),
            "gender": meta.get("gender"),
            "baseColour": meta.get("baseColour"),
            "cosine": cos,
            "match_frac": match_frac,
            "n_matched": n_matched,
            "bonus": bonus,
            "final_score": cos + bonus,
        })

    if hard_tier:
        # Xếp theo số thuộc tính khớp (hoặc match_frac) giảm dần trước, rồi mới tới cosine giảm dần
        items.sort(key=lambda x: (x["n_matched"], x["match_frac"], x["cosine"]), reverse=True)
    elif mode in ("system", "oracle"):
        items.sort(key=lambda x: x["final_score"], reverse=True)
    else:
        items.sort(key=lambda x: x["cosine"], reverse=True)

    return items


def run_1b():
    print("=" * 70)
    print("🔍 CHẠY CHẨN ĐOÁN BỔ SUNG BƯỚC 1b (DEV SET, TEST GALLERY)")
    print("=" * 70)

    engine = ECommerceSearchEngine(
        index_dir="index/test_gallery",
        config_path="config.yaml",
        device="cuda",
        use_mean_centering=False,
    )

    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    policy_cfg = load_gender_gt_policy("eval/gender_gt_policy.json")
    strict_cats = policy_cfg["policies"]["hybrid"]["strict_categories"]
    id_map = engine.id_map
    gallery_size = len(id_map)
    gallery_embs = engine.embeddings  # (1290, 1152), L2-normalized

    # Encode queries
    q_vecs = np.vstack([engine.encode_text_query(q["query"]) for q in queries])
    # Exact cosine matrix (100, 1290)
    cos_matrix = q_vecs @ gallery_embs.T

    # Ground truth mask & total relevant per query
    rel_masks = []
    tot_rels = []
    valid_idxs = []
    for i, q in enumerate(queries):
        spec = q["spec"]
        mask = [1 if is_item_relevant(spec, meta, "hybrid", strict_cats) else 0 for meta in id_map]
        rel_masks.append(mask)
        tot = sum(mask)
        tot_rels.append(tot)
        if tot > 0:
            valid_idxs.append(i)

    parsed_list = [engine.parser.parse(q["query"]) for q in queries]

    # ---------------------------------------------------------
    # 1b.2: Kiểm tra đối xứng định nghĩa (top_k=100 vs top_k=1290)
    # ---------------------------------------------------------
    print("\n--- 1b.2: KIỂM TRA ĐỐI XỨNG ĐỊNH NGHĨA (cut@100 vs full@1290) ---")
    def eval_ranking_list(ranked_indices_per_q, cut_k=None):
        r_list, t_list = [], []
        for i in valid_idxs:
            idxs = ranked_indices_per_q[i][:cut_k] if cut_k else ranked_indices_per_q[i]
            rels = [rel_masks[i][pos] for pos in idxs]
            r_list.append(rels)
            t_list.append(tot_rels[i])
        return compute_retrieval_summary(r_list, t_list, topk=(1, 5, 10))

    model_full_ranks = []
    sys_pool100_ranks = []
    sys_pool1290_ranks = []

    for i in range(len(queries)):
        # Model full
        m_items = _rank_all_items(cos_matrix[i], id_map, None, mode="model_only")
        model_full_ranks.append([x["index_pos"] for x in m_items])

        # System pool 100 (baseline exact)
        res_100 = engine.search_by_vector(
            q_vecs[i : i + 1], top_k=100, parsed_query=parsed_list[i], mode="system", candidate_pool=100
        )
        sys_pool100_ranks.append([r["index_pos"] for r in res_100])

        # System pool 1290
        s_items = _rank_all_items(cos_matrix[i], id_map, parsed_list[i], gamma=0.3, mode="system")
        sys_pool1290_ranks.append([x["index_pos"] for x in s_items])

    m_cut100 = eval_ranking_list(model_full_ranks, cut_k=100)
    m_full = eval_ranking_list(model_full_ranks, cut_k=1290)
    s100_cut100 = eval_ranking_list(sys_pool100_ranks, cut_k=100)
    s1290_cut100 = eval_ranking_list(sys_pool1290_ranks, cut_k=100)
    s1290_full = eval_ranking_list(sys_pool1290_ranks, cut_k=1290)

    print(f"Model thuần (cut@100)        : R@1={m_cut100['recall@1']:.4f}, R@10={m_cut100['recall@10']:.4f}, mAP={m_cut100['mAP']:.4f}")
    print(f"Model thuần (full@1290)      : R@1={m_full['recall@1']:.4f}, R@10={m_full['recall@10']:.4f}, mAP={m_full['mAP']:.4f}")
    print(f"Hệ thống pool=100 (cut@100)  : R@1={s100_cut100['recall@1']:.4f}, R@10={s100_cut100['recall@10']:.4f}, mAP={s100_cut100['mAP']:.4f}")
    print(f"Hệ thống pool=1290 (cut@100) : R@1={s1290_cut100['recall@1']:.4f}, R@10={s1290_cut100['recall@10']:.4f}, mAP={s1290_cut100['mAP']:.4f}")
    print(f"Hệ thống pool=1290 (full1290): R@1={s1290_full['recall@1']:.4f}, R@10={s1290_full['recall@10']:.4f}, mAP={s1290_full['mAP']:.4f}")

    # ---------------------------------------------------------
    # 1b.3: ORACLE PARSER (spec thay cho parsed_query, pool=1290)
    # ---------------------------------------------------------
    print("\n--- 1b.3: ORACLE PARSER (pool=1290, gamma=0.3) ---")
    oracle_ranks = []
    for i, q in enumerate(queries):
        spec_as_parsed = {
            "articleType": q["spec"].get("articleType") or [],
            "gender": q["spec"].get("gender"),
            "baseColour": q["spec"].get("baseColour") or [],
        }
        o_items = _rank_all_items(cos_matrix[i], id_map, spec_as_parsed, gamma=0.3, mode="oracle")
        oracle_ranks.append([x["index_pos"] for x in o_items])

    oracle_cut100 = eval_ranking_list(oracle_ranks, cut_k=100)
    oracle_full = eval_ranking_list(oracle_ranks, cut_k=1290)
    print(f"Oracle Parser (cut@100)  : R@1={oracle_cut100['recall@1']:.4f}, R@5={oracle_cut100['recall@5']:.4f}, R@10={oracle_cut100['recall@10']:.4f}, mAP={oracle_cut100['mAP']:.4f}")
    print(f"Oracle Parser (full@1290): R@1={oracle_full['recall@1']:.4f}, R@5={oracle_full['recall@5']:.4f}, R@10={oracle_full['recall@10']:.4f}, mAP={oracle_full['mAP']:.4f}")

    # Breakdown Oracle theo has_diacritics & thin_slot
    oracle_by_diac = defaultdict(list)
    oracle_by_slot = defaultdict(list)
    sys_by_diac = defaultdict(list)
    sys_by_slot = defaultdict(list)

    for i in valid_idxs:
        q = queries[i]
        d_key = "Có dấu" if q.get("has_diacritics", True) else "Không dấu"
        s_key = str(q.get("category", "unknown"))

        o_rels = [rel_masks[i][pos] for pos in oracle_ranks[i][:100]]
        s_rels = [rel_masks[i][pos] for pos in sys_pool1290_ranks[i][:100]]

        oracle_by_diac[d_key].append((o_rels, tot_rels[i]))
        oracle_by_slot[s_key].append((o_rels, tot_rels[i]))
        sys_by_diac[d_key].append((s_rels, tot_rels[i]))
        sys_by_slot[s_key].append((s_rels, tot_rels[i]))

    def summarize_group(g_dict):
        out = {}
        for k, pairs in sorted(g_dict.items()):
            out[k] = compute_retrieval_summary([p[0] for p in pairs], [p[1] for p in pairs], topk=(1, 5, 10))
        return out

    oracle_diac_sum = summarize_group(oracle_by_diac)
    oracle_slot_sum = summarize_group(oracle_by_slot)
    sys_diac_sum = summarize_group(sys_by_diac)
    sys_slot_sum = summarize_group(sys_by_slot)

    # ---------------------------------------------------------
    # 1b.4: Chi tiết 13 truy vấn thất bại ở baseline
    # ---------------------------------------------------------
    print("\n--- 1b.4: PHÂN TÍCH 13 TRUY VẤN THẤT BẠI (Baseline pool=100, gamma=0.3) ---")
    failed_13_details = []
    for i in valid_idxs:
        rels_base = [rel_masks[i][pos] for pos in sys_pool100_ranks[i]]
        rank_base = (rels_base.index(1) + 1) if any(rels_base) else None
        if rank_base is None or rank_base > 10:
            # Chạy trên pool toàn gallery (1290) để xem vị trí và điểm của ảnh đúng đầu tiên vs ảnh sai top 1
            s_items_full = _rank_all_items(cos_matrix[i], id_map, parsed_list[i], gamma=0.3, mode="system")
            first_rel_rank_full = None
            first_rel_item = None
            top_wrong_item = None
            for r_idx, it in enumerate(s_items_full, start=1):
                is_rel = rel_masks[i][it["index_pos"]] == 1
                if is_rel and first_rel_item is None:
                    first_rel_rank_full = r_idx
                    first_rel_item = it
                if not is_rel and top_wrong_item is None:
                    top_wrong_item = it

            # Cũng xem vị trí của ảnh đúng đầu tiên theo model thuần (cosine)
            m_items_full = _rank_all_items(cos_matrix[i], id_map, None, mode="model_only")
            first_rel_rank_model = next(
                (r_idx for r_idx, it in enumerate(m_items_full, start=1) if rel_masks[i][it["index_pos"]] == 1),
                None,
            )

            failed_13_details.append({
                "idx": i,
                "query": queries[i]["query"],
                "has_diacritics": queries[i].get("has_diacritics", True),
                "n_rel": tot_rels[i],
                "parsed": parsed_list[i],
                "spec": queries[i]["spec"],
                "rank_base_100": rank_base if rank_base else ">100",
                "rank_model_full": first_rel_rank_model,
                "rank_sys_full": first_rel_rank_full,
                "rel_cos": first_rel_item["cosine"],
                "rel_bonus": first_rel_item["bonus"],
                "rel_final": first_rel_item["final_score"],
                "rel_meta": f"{first_rel_item['articleType']} | {first_rel_item['gender']} | {first_rel_item['baseColour']}",
                "wrong_cos": top_wrong_item["cosine"],
                "wrong_bonus": top_wrong_item["bonus"],
                "wrong_final": top_wrong_item["final_score"],
                "wrong_meta": f"{top_wrong_item['articleType']} | {top_wrong_item['gender']} | {top_wrong_item['baseColour']}",
            })

    for d in failed_13_details:
        print(
            f"[{d['idx']}] '{d['query']}' | rel_rank(base={d['rank_base_100']}, model_full={d['rank_model_full']}, sys_full={d['rank_sys_full']}) "
            f"| Rel(cos={d['rel_cos']:.4f}, bon={d['rel_bonus']:.3f}, tot={d['rel_final']:.4f}) "
            f"vs TopWrong(cos={d['wrong_cos']:.4f}, bon={d['wrong_bonus']:.3f}, tot={d['wrong_final']:.4f}, {d['wrong_meta']})"
        )

    # ---------------------------------------------------------
    # 1b.5: Quét gamma (0.3, 0.5, 1.0, 2.0) & Hard-tier trên DEV SET (pool=1290)
    # ---------------------------------------------------------
    print("\n--- 1b.5: QUÉT GAMMA & HARD-TIER (CHỈ CHẨN ĐOÁN, DEV SET, pool=1290, cut@100) ---")
    sweep_results = []
    for g in [0.3, 0.5, 1.0, 2.0]:
        ranks_g = []
        for i in range(len(queries)):
            items = _rank_all_items(cos_matrix[i], id_map, parsed_list[i], gamma=g, mode="system", hard_tier=False)
            ranks_g.append([x["index_pos"] for x in items])
        res_cut100 = eval_ranking_list(ranks_g, cut_k=100)
        res_full = eval_ranking_list(ranks_g, cut_k=1290)
        sweep_results.append(("Parser hiện tại", f"gamma={g}", res_cut100, res_full))
        print(f"Parser hiện tại | gamma={g:<4} | cut@100: R@1={res_cut100['recall@1']:.4f}, R@10={res_cut100['recall@10']:.4f}, mAP={res_cut100['mAP']:.4f} | full: mAP={res_full['mAP']:.4f}")

    # Hard-tier với parser hiện tại
    ranks_ht = []
    for i in range(len(queries)):
        items = _rank_all_items(cos_matrix[i], id_map, parsed_list[i], gamma=0.3, mode="system", hard_tier=True)
        ranks_ht.append([x["index_pos"] for x in items])
    res_ht_100 = eval_ranking_list(ranks_ht, cut_k=100)
    res_ht_full = eval_ranking_list(ranks_ht, cut_k=1290)
    sweep_results.append(("Parser hiện tại", "hard-tier", res_ht_100, res_ht_full))
    print(f"Parser hiện tại | hard-tier  | cut@100: R@1={res_ht_100['recall@1']:.4f}, R@10={res_ht_100['recall@10']:.4f}, mAP={res_ht_100['mAP']:.4f} | full: mAP={res_ht_full['mAP']:.4f}")

    # Hard-tier với Oracle parser (để biết giới hạn lý thuyết tuyệt đối của metadata)
    ranks_o_ht = []
    for i, q in enumerate(queries):
        spec_as_parsed = {
            "articleType": q["spec"].get("articleType") or [],
            "gender": q["spec"].get("gender"),
            "baseColour": q["spec"].get("baseColour") or [],
        }
        items = _rank_all_items(cos_matrix[i], id_map, spec_as_parsed, gamma=0.3, mode="oracle", hard_tier=True)
        ranks_o_ht.append([x["index_pos"] for x in items])
    res_o_ht_100 = eval_ranking_list(ranks_o_ht, cut_k=100)
    res_o_ht_full = eval_ranking_list(ranks_o_ht, cut_k=1290)
    sweep_results.append(("Oracle Parser", "hard-tier", res_o_ht_100, res_o_ht_full))
    print(f"Oracle Parser   | hard-tier  | cut@100: R@1={res_o_ht_100['recall@1']:.4f}, R@10={res_o_ht_100['recall@10']:.4f}, mAP={res_o_ht_100['mAP']:.4f} | full: mAP={res_o_ht_full['mAP']:.4f}")

    # Lưu kết quả JSON chi tiết để phục vụ báo cáo
    out_json = {
        "1b_2_symmetry": {
            "model_cut100": m_cut100,
            "model_full1290": m_full,
            "system_pool100_cut100": s100_cut100,
            "system_pool1290_cut100": s1290_cut100,
            "system_pool1290_full1290": s1290_full,
        },
        "1b_3_oracle": {
            "oracle_cut100": oracle_cut100,
            "oracle_full1290": oracle_full,
            "by_diacritics": {"oracle": oracle_diac_sum, "system_pool1290": sys_diac_sum},
            "by_thin_slot": {"oracle": oracle_slot_sum, "system_pool1290": sys_slot_sum},
        },
        "1b_4_failed_13": failed_13_details,
        "1b_5_sweep": [
            {"parser": p, "method": m, "cut100": c100, "full1290": cf}
            for p, m, c100, cf in sweep_results
        ],
    }

    out_path = PROJECT_ROOT / "reports" / "phase5b" / "diagnostics_1b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu dữ liệu chẩn đoán 1b tại: {out_path}")


if __name__ == "__main__":
    run_1b()
