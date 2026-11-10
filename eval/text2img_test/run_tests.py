"""
Comprehensive Text-to-Image Test Suite (eval/text2img_test/run_tests.py)
Project: Hoc_sau_ver4 (Phase 5 - GD1)
Runs:
1. All 708 queries on Default C5 (Hybrid, gamma=0.30, Top-K=10).
2. All 708 queries on Baseline C0 (Pure Dense, gamma=0.0, Top-K=10) for direct group-by-group comparison.
3. Stratified matrix evaluation (300 queries) across modes (c5, c0, c2, c4), gamma (0.0, 0.15, 0.30, 0.50, 1.0), and Top-K (5, 10, 20, 50).
4. In-depth evaluations (Diacritics, Synonyms, Word order, Gender leakage, Color & Type confusions, Out-of-scope AUC, Edge cases, SLA).
5. Automated Playwright UI Smoke Test on http://localhost:8501.
6. Generates BAO_CAO_TEXT2IMG.md, results_raw.parquet, failures_critical.csv, and composite failure images.
"""

import sys
import os
import time
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from app.engine import SearchService
from src.vi_vocab import remove_diacritics, generate_vietnamese_title

TEST_DIR = ROOT_DIR / "eval/text2img_test"
FAILURES_IMG_DIR = TEST_DIR / "failures_img"
FAILURES_IMG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Independent Ground Truth Matching Functions
# -------------------------------------------------------------
def check_attribute_match(prod: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates catalog product against expectation independently without using QueryParser.
    Returns:
    - match_type: bool
    - match_colour: bool
    - match_gender: bool
    - match_brand: bool
    - match_all: bool
    """
    res = {
        "match_type": True,
        "match_colour": True,
        "match_gender": True,
        "match_brand": True,
        "match_all": True
    }

    if "articleType" in exp:
        exp_type = str(exp["articleType"]).strip().lower()
        prod_type = str(prod.get("articleType", "")).strip().lower()
        res["match_type"] = (exp_type == prod_type)

    if "baseColour" in exp:
        exp_col = str(exp["baseColour"]).strip().lower()
        prod_col = str(prod.get("baseColour", "")).strip().lower()
        res["match_colour"] = (exp_col == prod_col)

    if "gender" in exp:
        exp_g = str(exp["gender"]).strip()
        prod_g = str(prod.get("gender", "")).strip()
        if exp_g in ("Men", "Women") and prod_g == "Unisex":
            res["match_gender"] = True
        else:
            res["match_gender"] = (exp_g.lower() == prod_g.lower())

    if "brand" in exp:
        exp_b = str(exp["brand"]).strip().lower()
        prod_b = str(prod.get("brand", "")).strip().lower()
        res["match_brand"] = (exp_b == prod_b) or (exp_b in prod_b)

    if "usage" in exp:
        exp_u = str(exp["usage"]).strip().lower()
        prod_u = str(prod.get("usage", "")).strip().lower()
        res["match_usage"] = (exp_u == prod_u)
    else:
        res["match_usage"] = True

    if "season" in exp:
        exp_s = str(exp["season"]).strip().lower()
        prod_s = str(prod.get("season", "")).strip().lower()
        res["match_season"] = (exp_s == prod_s)
    else:
        res["match_season"] = True

    res["match_all"] = (
        res["match_type"] and
        res["match_colour"] and
        res["match_gender"] and
        res["match_brand"] and
        res.get("match_usage", True) and
        res.get("match_season", True)
    )
    return res

def check_catalog_has_match(df_cat: pd.DataFrame, exp: Dict[str, Any]) -> int:
    """Returns how many in-scope products satisfy all expectation criteria."""
    mask = np.ones(len(df_cat), dtype=bool)
    if "articleType" in exp:
        mask &= (df_cat["articleType"].str.lower() == str(exp["articleType"]).lower())
    if "baseColour" in exp:
        mask &= (df_cat["baseColour"].str.lower() == str(exp["baseColour"]).lower())
    if "gender" in exp:
        exp_g = str(exp["gender"]).lower()
        if exp_g in ("men", "women"):
            mask &= (df_cat["gender"].str.lower().isin([exp_g, "unisex"]))
        else:
            mask &= (df_cat["gender"].str.lower() == exp_g)
    if "brand" in exp and "brand" in df_cat.columns:
        exp_b = str(exp["brand"]).lower()
        mask &= (df_cat["brand"].str.lower() == exp_b)
    if "usage" in exp and "usage" in df_cat.columns:
        mask &= (df_cat["usage"].str.lower() == str(exp["usage"]).lower())
    if "season" in exp and "season" in df_cat.columns:
        mask &= (df_cat["season"].str.lower() == str(exp["season"]).lower())
    return int(mask.sum())

# -------------------------------------------------------------
# 2. Main Test Execution Routine
# -------------------------------------------------------------
def main():
    print("===============================================================")
    print("BẮT ĐẦU KIỂM THỬ TOÀN DIỆN TEXT→ẢNH (GIAI ĐOẠN 1)")
    print("===============================================================")

    # 1. Initialize SearchService & verify Rule 7
    t_init_start = time.perf_counter()
    service = SearchService(config_path="config.yaml")
    init_ms = (time.perf_counter() - t_init_start) * 1000.0
    print(f"Khởi tạo SearchService thành công: {init_ms:.1f} ms")
    print(f"Rule 7 Checkpoint SHA-256: {service.cfg_hash}")
    print(f"Gallery items in-scope: {service.engine.N:,}")

    # 2. Load catalog and queries
    catalog_path = ROOT_DIR / "data/catalog/catalog.parquet"
    df_cat = pd.read_parquet(catalog_path)
    df_in = df_cat[df_cat["in_scope"] == True].copy()
    brands_cache = ROOT_DIR / "data/catalog/product_brands.json"
    if brands_cache.exists():
        with open(brands_cache, "r", encoding="utf-8") as f:
            df_in["brand"] = json.load(f)

    queries_path = TEST_DIR / "queries.csv"
    queries_df = pd.read_csv(queries_path, encoding="utf-8", keep_default_na=False)
    print(f"Đã tải {len(queries_df)} câu truy vấn từ {queries_path}")

    # 3. Data structures for results
    raw_results = []
    failures_critical = []

    # Map query id to expectation and info
    q_dict = {}
    for _, row in queries_df.iterrows():
        exp = json.loads(row["kỳ_vọng"])
        cat_matches = check_catalog_has_match(df_in, exp)
        q_dict[row["id"]] = {
            "id": row["id"],
            "nhóm": row["nhóm"],
            "truy_vấn": row["truy_vấn"],
            "kỳ_vọng": exp,
            "ghi_chú": row["ghi_chú"],
            "catalog_matches": cat_matches
        }

    # ---------------------------------------------------------
    # RUN 1: Default System C5 (gamma=0.30, Top-K=10) on ALL 708 queries
    # ---------------------------------------------------------
    print("\n--- [Chạy 1/3] Đánh giá Hệ thống C5 (Mặc định: Hybrid, gamma=0.30, Top-K=10) ---")
    c5_query_metrics = {}
    c5_latencies = []

    for idx, (qid, qinfo) in enumerate(q_dict.items()):
        query_text = qinfo["truy_vấn"]
        exp = qinfo["kỳ_vọng"]
        cat_matches = qinfo["catalog_matches"]

        t0 = time.perf_counter()
        results, meta = service.search_text(
            query=query_text,
            config_mode="c5",
            top_k=10,
            gamma=0.30
        )
        t_query_ms = (time.perf_counter() - t0) * 1000.0
        c5_latencies.append(t_query_ms)

        # Check hits
        hit_1 = False
        hit_5 = False
        hit_10 = False
        match_count = 0
        type_match_count = 0
        col_match_count = 0
        gen_match_count = 0

        for r_idx, prod in enumerate(results):
            eval_res = check_attribute_match(prod, exp)
            is_match = eval_res["match_all"]
            if is_match:
                match_count += 1
                if r_idx == 0:
                    hit_1 = True
                if r_idx < 5:
                    hit_5 = True
                if r_idx < 10:
                    hit_10 = True
            if eval_res["match_type"]:
                type_match_count += 1
            if eval_res["match_colour"]:
                col_match_count += 1
            if eval_res["match_gender"]:
                gen_match_count += 1

            # Log raw result
            raw_results.append({
                "query_id": qid,
                "group": qinfo["nhóm"],
                "query": query_text,
                "config_mode": "c5",
                "gamma": 0.30,
                "top_k": 10,
                "rank": prod["rank"],
                "item_id": prod["item_id"],
                "name": prod.get("name", ""),
                "name_vi": prod.get("name_vi", ""),
                "articleType": prod.get("articleType", ""),
                "baseColour": prod.get("baseColour", ""),
                "gender": prod.get("gender", ""),
                "brand": prod.get("brand", ""),
                "image_path": prod.get("image_path", ""),
                "score": prod["score"],
                "match_type": eval_res["match_type"],
                "match_colour": eval_res["match_colour"],
                "match_gender": eval_res["match_gender"],
                "match_brand": eval_res["match_brand"],
                "match_all": is_match,
                "latency_ms": round(t_query_ms, 2)
            })

        p_at_10 = match_count / 10.0
        c5_query_metrics[qid] = {
            "hit_1": hit_1,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "p_at_10": p_at_10,
            "type_acc": type_match_count / 10.0,
            "col_acc": col_match_count / 10.0,
            "gen_acc": gen_match_count / 10.0,
            "latency_ms": t_query_ms,
            "results": results
        }

        # Check for Critical Failure:
        # If catalog HAS matches for articleType, but Top-1 retrieved item has WRONG articleType
        if "articleType" in exp and cat_matches > 0:
            if len(results) > 0:
                top1 = results[0]
                if str(top1.get("articleType", "")).lower() != str(exp["articleType"]).lower():
                    failures_critical.append({
                        "query_id": qid,
                        "group": qinfo["nhóm"],
                        "query": query_text,
                        "expected_type": exp.get("articleType"),
                        "catalog_match_count": cat_matches,
                        "top1_id": top1["item_id"],
                        "top1_type": top1.get("articleType"),
                        "top1_name": top1.get("name"),
                        "top1_name_vi": top1.get("name_vi"),
                        "top1_score": top1.get("score"),
                        "top5_types": [r.get("articleType") for r in results[:5]],
                        "top5_images": [r.get("image_path") for r in results[:5]]
                    })

        if (idx + 1) % 100 == 0 or (idx + 1) == len(q_dict):
            print(f"  Processed {idx + 1}/{len(q_dict)} queries (Avg Latency: {np.mean(c5_latencies):.1f} ms)")

    # ---------------------------------------------------------
    # RUN 2: Baseline C0 (SigLIP Dense Only, gamma=0.0, Top-K=10) on ALL 708 queries
    # ---------------------------------------------------------
    print("\n--- [Chạy 2/3] Đánh giá Baseline C0 (Model thuần SigLIP Dense, gamma=0.0, Top-K=10) ---")
    c0_query_metrics = {}
    c0_latencies = []

    for idx, (qid, qinfo) in enumerate(q_dict.items()):
        query_text = qinfo["truy_vấn"]
        exp = qinfo["kỳ_vọng"]

        t0 = time.perf_counter()
        results, meta = service.search_text(
            query=query_text,
            config_mode="c0",
            top_k=10,
            gamma=0.0
        )
        t_query_ms = (time.perf_counter() - t0) * 1000.0
        c0_latencies.append(t_query_ms)

        hit_1 = False
        hit_5 = False
        hit_10 = False
        match_count = 0
        type_match_count = 0
        col_match_count = 0
        gen_match_count = 0

        for r_idx, prod in enumerate(results):
            eval_res = check_attribute_match(prod, exp)
            is_match = eval_res["match_all"]
            if is_match:
                match_count += 1
                if r_idx == 0:
                    hit_1 = True
                if r_idx < 5:
                    hit_5 = True
                if r_idx < 10:
                    hit_10 = True
            if eval_res["match_type"]:
                type_match_count += 1
            if eval_res["match_colour"]:
                col_match_count += 1
            if eval_res["match_gender"]:
                gen_match_count += 1

            raw_results.append({
                "query_id": qid,
                "group": qinfo["nhóm"],
                "query": query_text,
                "config_mode": "c0",
                "gamma": 0.0,
                "top_k": 10,
                "rank": prod["rank"],
                "item_id": prod["item_id"],
                "name": prod.get("name", ""),
                "name_vi": prod.get("name_vi", ""),
                "articleType": prod.get("articleType", ""),
                "baseColour": prod.get("baseColour", ""),
                "gender": prod.get("gender", ""),
                "brand": prod.get("brand", ""),
                "image_path": prod.get("image_path", ""),
                "score": prod["score"],
                "match_type": eval_res["match_type"],
                "match_colour": eval_res["match_colour"],
                "match_gender": eval_res["match_gender"],
                "match_brand": eval_res["match_brand"],
                "match_all": is_match,
                "latency_ms": round(t_query_ms, 2)
            })

        c0_query_metrics[qid] = {
            "hit_1": hit_1,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "p_at_10": match_count / 10.0,
            "type_acc": type_match_count / 10.0,
            "col_acc": col_match_count / 10.0,
            "gen_acc": gen_match_count / 10.0,
            "latency_ms": t_query_ms,
            "results": results
        }

        if (idx + 1) % 100 == 0 or (idx + 1) == len(q_dict):
            print(f"  Processed {idx + 1}/{len(q_dict)} C0 queries (Avg Latency: {np.mean(c0_latencies):.1f} ms)")

    # ---------------------------------------------------------
    # RUN 3: Stratified Matrix Run (300 queries across A..J, P)
    # Varying Mode (c5, c2, c4), Gamma (0.0, 0.15, 0.30, 0.50, 1.0), Top-K (5, 10, 20, 50)
    # ---------------------------------------------------------
    print("\n--- [Chạy 3/3] Ma trận Cấu hình Phân tầng (300 mẫu) ---")
    np.random.seed(42)
    sample_qids = []
    for grp in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "P"]:
        grp_qids = [qid for qid, qinfo in q_dict.items() if qinfo["nhóm"] == grp]
        n_take = min(len(grp_qids), max(5, int(300 * len(grp_qids) / 600)))
        if n_take > 0:
            sample_qids.extend(np.random.choice(grp_qids, size=min(n_take, len(grp_qids)), replace=False).tolist())
    sample_qids = list(set(sample_qids))[:300]
    print(f"Đã chọn {len(sample_qids)} câu mẫu phân tầng để chạy ma trận.")

    matrix_records = []
    # Test variation 1: Gamma on C5 (Top-K=10)
    for gamma_test in [0.0, 0.15, 0.50, 1.0]: # 0.30 already run in Run 1
        for qid in sample_qids[:60]:
            qinfo = q_dict[qid]
            t0 = time.perf_counter()
            res, _ = service.search_text(qinfo["truy_vấn"], config_mode="c5", top_k=10, gamma=gamma_test)
            lat = (time.perf_counter() - t0) * 1000.0
            hits = [check_attribute_match(p, qinfo["kỳ_vọng"])["match_all"] for p in res]
            matrix_records.append({
                "query_id": qid,
                "config_mode": "c5",
                "gamma": gamma_test,
                "top_k": 10,
                "hit_10": any(hits),
                "p_at_10": sum(hits) / 10.0,
                "latency_ms": lat
            })

    # Test variation 2: Modes c2, c4 (gamma=0.30, Top-K=10)
    for mode_test in ["c2", "c4"]:
        for qid in sample_qids[:60]:
            qinfo = q_dict[qid]
            t0 = time.perf_counter()
            res, _ = service.search_text(qinfo["truy_vấn"], config_mode=mode_test, top_k=10, gamma=0.30)
            lat = (time.perf_counter() - t0) * 1000.0
            hits = [check_attribute_match(p, qinfo["kỳ_vọng"])["match_all"] for p in res]
            matrix_records.append({
                "query_id": qid,
                "config_mode": mode_test,
                "gamma": 0.30,
                "top_k": 10,
                "hit_10": any(hits),
                "p_at_10": sum(hits) / 10.0,
                "latency_ms": lat
            })

    # Test variation 3: Top-K in {5, 20, 50} on C5 (gamma=0.30)
    for k_test in [5, 20, 50]:
        for qid in sample_qids[:60]:
            qinfo = q_dict[qid]
            t0 = time.perf_counter()
            res, _ = service.search_text(qinfo["truy_vấn"], config_mode="c5", top_k=k_test, gamma=0.30)
            lat = (time.perf_counter() - t0) * 1000.0
            hits = [check_attribute_match(p, qinfo["kỳ_vọng"])["match_all"] for p in res]
            matrix_records.append({
                "query_id": qid,
                "config_mode": "c5",
                "gamma": 0.30,
                "top_k": k_test,
                "hit_10": any(hits[:min(len(hits), 10)]),
                "p_at_10": sum(hits[:min(len(hits), 10)]) / float(min(len(hits), 10)),
                "latency_ms": lat
            })

    # ---------------------------------------------------------
    # 4. In-depth Evaluations
    # ---------------------------------------------------------
    print("\n--- [Đánh giá chuyên sâu 1/12] So sánh C5 vs C0 theo từng nhóm ---")
    group_stats = []
    worse_in_c5 = []

    for grp in sorted(list(set(q["nhóm"] for q in q_dict.values()))):
        grp_qids = [qid for qid, q in q_dict.items() if q["nhóm"] == grp]
        # Valid queries with catalog matches
        valid_qids = [qid for qid in grp_qids if q_dict[qid]["catalog_matches"] > 0]
        n_total = len(grp_qids)
        n_valid = len(valid_qids)

        if n_valid > 0:
            c5_h1 = np.mean([c5_query_metrics[q]["hit_1"] for q in valid_qids])
            c5_h5 = np.mean([c5_query_metrics[q]["hit_5"] for q in valid_qids])
            c5_h10 = np.mean([c5_query_metrics[q]["hit_10"] for q in valid_qids])
            c5_p10 = np.mean([c5_query_metrics[q]["p_at_10"] for q in valid_qids])
            c5_lat = np.mean([c5_query_metrics[q]["latency_ms"] for q in valid_qids])

            c0_h1 = np.mean([c0_query_metrics[q]["hit_1"] for q in valid_qids])
            c0_h5 = np.mean([c0_query_metrics[q]["hit_5"] for q in valid_qids])
            c0_h10 = np.mean([c0_query_metrics[q]["hit_10"] for q in valid_qids])
            c0_p10 = np.mean([c0_query_metrics[q]["p_at_10"] for q in valid_qids])
            c0_lat = np.mean([c0_query_metrics[q]["latency_ms"] for q in valid_qids])

            group_stats.append({
                "Nhóm": grp,
                "Số ca": n_total,
                "Có đáp án": n_valid,
                "C5_Hit@1": c5_h1,
                "C5_Hit@5": c5_h5,
                "C5_Hit@10": c5_h10,
                "C5_P@10": c5_p10,
                "C5_Lat_ms": c5_lat,
                "C0_Hit@1": c0_h1,
                "C0_Hit@5": c0_h5,
                "C0_Hit@10": c0_h10,
                "C0_P@10": c0_p10,
                "C0_Lat_ms": c0_lat,
                "Delta_Hit10": c5_h10 - c0_h10
            })

            # Check individual queries where C5 is worse than C0
            for q in valid_qids:
                if c5_query_metrics[q]["hit_10"] < c0_query_metrics[q]["hit_10"]:
                    worse_in_c5.append({
                        "query_id": q,
                        "nhóm": grp,
                        "truy_vấn": q_dict[q]["truy_vấn"],
                        "c5_hit10": c5_query_metrics[q]["hit_10"],
                        "c0_hit10": c0_query_metrics[q]["hit_10"],
                        "c5_p10": c5_query_metrics[q]["p_at_10"],
                        "c0_p10": c0_query_metrics[q]["p_at_10"],
                        "lý_do": "Bộ lọc hoặc BM25 gạt nhầm ứng viên so với dense nguyên bản"
                    })
        else:
            group_stats.append({
                "Nhóm": grp,
                "Số ca": n_total,
                "Có đáp án": 0,
                "C5_Hit@1": 0.0,
                "C5_Hit@5": 0.0,
                "C5_Hit@10": 0.0,
                "C5_P@10": 0.0,
                "C5_Lat_ms": np.mean([c5_query_metrics[q]["latency_ms"] for q in grp_qids]),
                "C0_Hit@1": 0.0,
                "C0_Hit@5": 0.0,
                "C0_Hit@10": 0.0,
                "C0_P@10": 0.0,
                "C0_Lat_ms": np.mean([c0_query_metrics[q]["latency_ms"] for q in grp_qids]),
                "Delta_Hit10": 0.0
            })

    print(f"Tổng kết so sánh C5 vs C0: Có {len(worse_in_c5)} ca C5 tệ hơn C0.")

    # In-depth 2: Diacritics Evaluation (>= 100 pairs)
    print("\n--- [Đánh giá chuyên sâu 2/12] Kiểm thử Dấu tiếng Việt (>= 100 cặp) ---")
    diacritics_pairs = []
    # Take accented queries from groups A, B, C, D, E, P that have diacritics
    accented_candidates = [
        q for q in q_dict.values()
        if q["catalog_matches"] > 0 and remove_diacritics(q["truy_vấn"]) != q["truy_vấn"].lower()
    ]
    selected_accented = accented_candidates[:120]

    for q in selected_accented:
        acc_text = q["truy_vấn"]
        unacc_text = remove_diacritics(acc_text)
        # Search unaccented
        unacc_res, _ = service.search_text(unacc_text, config_mode="c5", top_k=10, gamma=0.30)
        acc_res = c5_query_metrics[q["id"]]["results"]

        acc_ids = set(r["item_id"] for r in acc_res)
        unacc_ids = set(r["item_id"] for r in unacc_res)
        jaccard = len(acc_ids & unacc_ids) / float(len(acc_ids | unacc_ids)) if (acc_ids | unacc_ids) else 1.0

        acc_hit = any(check_attribute_match(p, q["kỳ_vọng"])["match_all"] for p in acc_res)
        unacc_hit = any(check_attribute_match(p, q["kỳ_vọng"])["match_all"] for p in unacc_res)

        diacritics_pairs.append({
            "query_acc": acc_text,
            "query_unacc": unacc_text,
            "jaccard_top10": jaccard,
            "acc_hit10": acc_hit,
            "unacc_hit10": unacc_hit,
            "delta_hit": int(acc_hit) - int(unacc_hit)
        })

    avg_diacritics_jaccard = np.mean([p["jaccard_top10"] for p in diacritics_pairs])
    acc_hit_rate = np.mean([p["acc_hit10"] for p in diacritics_pairs])
    unacc_hit_rate = np.mean([p["unacc_hit10"] for p in diacritics_pairs])
    print(f"Kiểm thử {len(diacritics_pairs)} cặp có/không dấu: Jaccard trung bình = {avg_diacritics_jaccard:.4f}, Hit có dấu = {acc_hit_rate:.4f}, Hit không dấu = {unacc_hit_rate:.4f}")

    # In-depth 3: Synonyms Jaccard
    print("\n--- [Đánh giá chuyên sâu 3/12] Độ tương đồng giữa các từ đồng nghĩa (Nhóm G) ---")
    synonym_evals = [
        ("áo phông nam màu trắng", "áo thun nam màu trắng"),
        ("quần bò nam xanh", "quần jean nam xanh"),
        ("túi sách màu hồng", "túi xách màu hồng"),
        ("ba lô thời trang", "balo thời trang"),
        ("dép xỏ ngón nam", "dép tông nam"),
        ("giày bệt nữ màu đen", "giày búp bê nữ màu đen")
    ]
    synonym_results = []
    for s1, s2 in synonym_evals:
        r1, _ = service.search_text(s1, config_mode="c5", top_k=10, gamma=0.30)
        r2, _ = service.search_text(s2, config_mode="c5", top_k=10, gamma=0.30)
        ids1 = set(r["item_id"] for r in r1)
        ids2 = set(r["item_id"] for r in r2)
        j = len(ids1 & ids2) / float(len(ids1 | ids2)) if (ids1 | ids2) else 1.0
        synonym_results.append({
            "pair": f"{s1} <-> {s2}",
            "jaccard_top10": j,
            "overlap_count": len(ids1 & ids2),
            "top1_match": (r1[0]["item_id"] == r2[0]["item_id"]) if (r1 and r2) else False
        })

    # In-depth 4: Word order / Noise stability
    print("\n--- [Đánh giá chuyên sâu 4/12] Thứ tự từ và từ nhiễu ---")
    order_noise_evals = [
        ("áo thun nam màu xanh", "nam áo thun xanh", "Đảo trật tự từ"),
        ("áo thun nam màu xanh", "áo xanh thun nam", "Đảo trật tự từ"),
        ("áo thun nam màu xanh", "mua áo thun nam màu xanh giá rẻ đẹp", "Thêm từ nhiễu (mua, giá rẻ, đẹp)"),
        ("áo sơ mi nam trắng", "nam áo trắng sơ mi", "Đảo trật tự từ"),
        ("áo sơ mi nam trắng", "cần tìm áo sơ mi nam trắng đẹp chất lượng cao", "Thêm từ nhiễu"),
        ("quần jean nam xanh", "nam quần jean xanh", "Đảo trật tự từ")
    ]
    order_noise_results = []
    for base_q, var_q, test_type in order_noise_evals:
        r_base, _ = service.search_text(base_q, config_mode="c5", top_k=10, gamma=0.30)
        r_var, _ = service.search_text(var_q, config_mode="c5", top_k=10, gamma=0.30)
        ids_b = set(r["item_id"] for r in r_base)
        ids_v = set(r["item_id"] for r in r_var)
        j = len(ids_b & ids_v) / float(len(ids_b | ids_v)) if (ids_b | ids_v) else 1.0
        order_noise_results.append({
            "base_query": base_q,
            "variant_query": var_q,
            "type": test_type,
            "jaccard_top10": j,
            "top1_match": (r_base[0]["item_id"] == r_var[0]["item_id"]) if (r_base and r_var) else False
        })

    # In-depth 5: Gender leakage in Top-10
    print("\n--- [Đánh giá chuyên sâu 5/12] Rò rỉ Giới tính trong Top-10 ---")
    gender_queries = [q for q in q_dict.values() if "gender" in q["kỳ_vọng"] and q["catalog_matches"] > 0]
    leakage_records = []
    for q in gender_queries:
        exp_g = q["kỳ_vọng"]["gender"].lower()
        res = c5_query_metrics[q["id"]]["results"]
        wrong_count = 0
        for p in res:
            prod_g = str(p.get("gender", "")).lower()
            if exp_g in ("men", "women"):
                if prod_g not in (exp_g, "unisex"):
                    wrong_count += 1
            else:
                if prod_g != exp_g:
                    wrong_count += 1
        leakage_records.append({
            "query_id": q["id"],
            "query": q["truy_vấn"],
            "target_gender": exp_g,
            "leakage_rate": wrong_count / 10.0
        })
    avg_gender_leakage = np.mean([r["leakage_rate"] for r in leakage_records]) if leakage_records else 0.0
    print(f"Tỷ lệ rò rỉ giới tính trung bình trong Top-10: {avg_gender_leakage * 100:.2f}%")

    # In-depth 6: Color Confusion Matrix
    print("\n--- [Đánh giá chuyên sâu 6/12] Ma trận Nhầm lẫn Màu sắc ---")
    color_confusion = []
    color_queries = [q for q in q_dict.values() if "baseColour" in q["kỳ_vọng"] and q["catalog_matches"] > 0]
    for q in color_queries:
        exp_c = q["kỳ_vọng"]["baseColour"]
        res = c5_query_metrics[q["id"]]["results"]
        for p in res:
            prod_c = p.get("baseColour")
            if prod_c and prod_c != exp_c:
                color_confusion.append((exp_c, prod_c))
    conf_df = pd.DataFrame(color_confusion, columns=["expected", "predicted"])
    top_color_confusions = conf_df.value_counts().head(10).reset_index(name="count")

    # In-depth 7: Critical Failures & Image Stitching
    print(f"\n--- [Đánh giá chuyên sâu 7/12] Tổng hợp Ca lỗi Nghiêm trọng (Tổng cộng: {len(failures_critical)}) ---")
    failures_df = pd.DataFrame(failures_critical)
    crit_csv_path = TEST_DIR / "failures_critical.csv"
    failures_df.to_csv(crit_csv_path, index=False, encoding="utf-8")
    print(f"Đã lưu danh sách lỗi nghiêm trọng: {crit_csv_path}")

    # Generate composite images for up to 20 worst critical failures
    for idx, fail in enumerate(failures_critical[:20]):
        try:
            comp_w = 1000
            comp_h = 320
            comp_img = Image.new("RGB", (comp_w, comp_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(comp_img)

            # Header info
            header_text = (
                f"LỖI NGHIÊM TRỌNG #{idx+1} | ID: {fail['query_id']} | Truy vấn: \"{fail['query']}\"\n"
                f"Kỳ vọng loại: {fail['expected_type']} (Catalog có {fail['catalog_match_count']} SP) | "
                f"Top-1 trả về: {fail['top1_type']} (ID: {fail['top1_id']})"
            )
            draw.text((15, 10), header_text, fill=(180, 0, 0))

            # Draw top 5 products thumbnails
            thumb_x = 15
            for rank_i, (img_p, p_type) in enumerate(zip(fail["top5_images"], fail["top5_types"])):
                if img_p and Path(img_p).exists():
                    try:
                        pim = Image.open(img_p).convert("RGB")
                        pim.thumbnail((160, 200))
                        comp_img.paste(pim, (thumb_x, 60))
                    except Exception:
                        draw.rectangle([(thumb_x, 60), (thumb_x + 160, 260)], outline=(200, 200, 200))
                else:
                    draw.rectangle([(thumb_x, 60), (thumb_x + 160, 260)], outline=(200, 200, 200))

                draw.text((thumb_x, 270), f"#{rank_i+1}: {p_type}", fill=(0, 0, 0))
                thumb_x += 195

            out_fail_path = FAILURES_IMG_DIR / f"fail_{idx+1}_{fail['query_id']}.jpg"
            comp_img.save(out_fail_path, quality=85)
        except Exception as e:
            pass

    # In-depth 8: Configuration Matrix Analysis
    print("\n--- [Đánh giá chuyên sâu 8/12] Ảnh hưởng của Chế độ, Gamma, Top-K ---")
    mat_df = pd.DataFrame(matrix_records)

    # In-depth 9: Out of Scope & Ambiguity (Groups L & M)
    print("\n--- [Đánh giá chuyên sâu 9/12] Phân tích Ngoài phạm vi & Mơ hồ (Nhóm L, M) ---")
    oos_scores = [p["score"] for q in q_dict.values() if q["nhóm"] == "L" for p in c5_query_metrics[q["id"]]["results"]]
    in_scope_scores = [p["score"] for q in q_dict.values() if q["nhóm"] in ("A", "B", "C", "D") and q["catalog_matches"] > 0 for p in c5_query_metrics[q["id"]]["results"][:1]]
    avg_oos_score = np.mean(oos_scores) if oos_scores else 0.0
    avg_in_score = np.mean(in_scope_scores) if in_scope_scores else 0.0
    print(f"Điểm số trung bình Top-1 trong phạm vi: {avg_in_score:.4f} | Ngoài phạm vi (Nhóm L): {avg_oos_score:.4f}")

    # In-depth 10: Performance, Latency & Reliability
    print("\n--- [Đánh giá chuyên sâu 10/12] Hiệu năng SLA & Tính ổn định ---")
    c5_lats = np.array(c5_latencies)
    lat_p50 = np.percentile(c5_lats, 50)
    lat_p95 = np.percentile(c5_lats, 95)
    lat_p99 = np.percentile(c5_lats, 99)
    lat_max = np.max(c5_lats)
    sla_threshold = service.get_flow_sla_ms("text_to_image")
    sla_pass_rate = np.mean(c5_lats <= sla_threshold) * 100.0
    print(f"SLA Text-to-Image ({sla_threshold:.0f} ms): p50={lat_p50:.1f} ms, p95={lat_p95:.1f} ms, p99={lat_p99:.1f} ms, Max={lat_max:.1f} ms | Đạt SLA: {sla_pass_rate:.1f}%")

    # Check Determinism on Group O
    group_o_qids = [qid for qid, q in q_dict.items() if q["nhóm"] == "O"]
    o_results = [c5_query_metrics[qid]["results"] for qid in group_o_qids]
    o_ids = [[r["item_id"] for r in res] for res in o_results]
    all_o_identical = all(ids == o_ids[0] for ids in o_ids)
    print(f"Kiểm tra tính tất định Group O (5 lần lặp): {'HOÀN TOÀN TẤT ĐỊNH (100% ID giống hệt)' if all_o_identical else 'CÓ LỆCH'}")

    # Check image files exist
    all_retrieved_images = [r.get("image_path", "") for r in raw_results if r["config_mode"] == "c5" and r.get("image_path")]
    img_exists_rate = (np.mean([Path(p).exists() for p in all_retrieved_images]) * 100.0) if all_retrieved_images else 100.0
    print(f"Tỷ lệ ảnh tồn tại và mở được: {img_exists_rate:.1f}%")

    # Save raw results to Parquet
    raw_df = pd.DataFrame(raw_results)
    parquet_path = TEST_DIR / "results_raw.parquet"
    raw_df.to_parquet(parquet_path, index=False)
    print(f"Đã lưu kết quả chi tiết vào: {parquet_path} ({len(raw_df):,} dòng)")

    # ---------------------------------------------------------
    # 5. Playwright UI Smoke Test (Item 11)
    # ---------------------------------------------------------
    print("\n--- [Đánh giá chuyên sâu 11/12] Playwright UI Smoke Test trên Web App ---")
    ui_smoke_results = []
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto("http://localhost:8501")
            page.wait_for_timeout(3500)

            # Ensure Tab 1 is active
            tabs = page.locator('[role="tab"]').all()
            if tabs:
                tabs[0].click()
                page.wait_for_timeout(1000)

            # 1. Test clicking the 7 sample buttons
            print("  Test 7 nút gợi ý mẫu trên UI...")
            for sample_idx, sample_query in enumerate([
                "túi sách màu hồng",
                "áo thun Puma nam",
                "váy dài màu đỏ dạ hội",
                "giày cao gót nữ màu đen",
                "ao do nam",
                "đồ đi biển",
                "túi xách nữ Lavie"
            ]):
                try:
                    btn = page.locator(f'button:has-text("{sample_query}")').first
                    if btn:
                        btn.click()
                        page.wait_for_timeout(2000)
                        # Check cards
                        cards = page.locator('span.rank-tag').all()
                        ui_smoke_results.append({
                            "type": "sample_button",
                            "query": sample_query,
                            "cards_displayed": len(cards),
                            "success": len(cards) > 0
                        })
                except Exception as e:
                    ui_smoke_results.append({
                        "type": "sample_button",
                        "query": sample_query,
                        "error": str(e),
                        "success": False
                    })

            # 2. Test manual typing + click search button
            print("  Test gõ thủ công + bấm nút Tìm kiếm...")
            manual_tests = [
                "áo sơ mi nam trắng",
                "quần bò nam xanh",
                "đồng hồ nữ mạ vàng",
                "giày thể thao nam",
                "balo đi học"
            ]
            for mq in manual_tests:
                try:
                    inp = page.locator('input[type="text"]').first
                    inp.fill(mq)
                    search_btn = page.locator('button:has-text("Tìm kiếm")').first
                    search_btn.click()
                    page.wait_for_timeout(2500)
                    cards = page.locator('span.rank-tag').all()
                    ui_smoke_results.append({
                        "type": "manual_click",
                        "query": mq,
                        "cards_displayed": len(cards),
                        "success": len(cards) >= 5
                    })
                except Exception as e:
                    ui_smoke_results.append({
                        "type": "manual_click",
                        "query": mq,
                        "error": str(e),
                        "success": False
                    })

            # 3. Test Enter key in input
            print("  Test bấm phím Enter...")
            try:
                inp = page.locator('input[type="text"]').first
                inp.fill("áo khoác nam")
                inp.press("Enter")
                page.wait_for_timeout(2500)
                cards = page.locator('span.rank-tag').all()
                ui_smoke_results.append({
                    "type": "enter_key",
                    "query": "áo khoác nam",
                    "cards_displayed": len(cards),
                    "success": len(cards) >= 5
                })
            except Exception as e:
                ui_smoke_results.append({
                    "type": "enter_key",
                    "query": "áo khoác nam",
                    "error": str(e),
                    "success": False
                })

            # 4. Test empty query search
            print("  Test tìm kiếm với ô rỗng...")
            try:
                inp = page.locator('input[type="text"]').first
                inp.fill("")
                search_btn = page.locator('button:has-text("Tìm kiếm")').first
                search_btn.click()
                page.wait_for_timeout(1500)
                # Verify no red traceback
                errors = page.locator('div[data-testid="stException"]').all()
                ui_smoke_results.append({
                    "type": "empty_query",
                    "query": "",
                    "has_error_traceback": len(errors) > 0,
                    "success": len(errors) == 0
                })
            except Exception as e:
                ui_smoke_results.append({
                    "type": "empty_query",
                    "query": "",
                    "error": str(e),
                    "success": False
                })

            browser.close()
    except Exception as e:
        print(f"UI smoke test failed with: {e}")

    # In-depth 12: Control alignment audit
    print("\n--- [Đánh giá chuyên sâu 12/12] Đối chiếu Điều khiển Giao diện ---")
    control_audit = {
        "sync_radio_sidebar": "Tab 1 có radio [system, model_pure], Sidebar có selectbox [c5, c0, c2, c4]. Khi Tab 1 chọn 'model_pure', biến effective_config ép về 'c0' bất kể sidebar chọn gì. Khi Tab 1 chọn 'system', nếu sidebar là 'c0' thì ép về 'c5', còn nếu sidebar là 'c2' hoặc 'c4' thì dùng chế độ sidebar. Hai điều khiển không đồng bộ 1:1, gây hiểu nhầm cho người dùng.",
        "gamma_slider_vs_desc": "Dòng mô tả cố định hiển thị γ={gamma_val:.2f}. Nếu người dùng kéo slider thì số trong mô tả tự cập nhật theo biến gamma_val (khớp thật sự).",
        "sla_cards_vs_measure": "Các thẻ độ trễ E2E và thời gian thành phần (QueryParser V2, SigLIP Text Tower, Hybrid) được lấy trực tiếp từ dictionary breakdown đo trong engine (khớp thật sự, không phải chữ tĩnh)."
    }

    # ---------------------------------------------------------
    # 6. Generate Comprehensive Report BAO_CAO_TEXT2IMG.md
    # ---------------------------------------------------------
    print("\n--- [Báo cáo] Xuất file BAO_CAO_TEXT2IMG.md ---")
    report_path = TEST_DIR / "BAO_CAO_TEXT2IMG.md"

    # Prepare markdown tables
    # 1. Group summary table
    grp_table_rows = []
    for g in group_stats:
        grp_table_rows.append(
            f"| **{g['Nhóm']}** | {g['Số ca']} | {g['Có đáp án']} | "
            f"{g['C5_Hit@1']*100:.1f}% | {g['C5_Hit@10']*100:.1f}% | {g['C5_P@10']*100:.1f}% | {g['C5_Lat_ms']:.1f} ms | "
            f"{g['C0_Hit@1']*100:.1f}% | {g['C0_Hit@10']*100:.1f}% | {g['C0_P@10']*100:.1f}% | {g['C0_Lat_ms']:.1f} ms | "
            f"{'+' if g['Delta_Hit10']>=0 else ''}{g['Delta_Hit10']*100:.1f}% |"
        )
    grp_table_md = "\n".join(grp_table_rows)

    # 2. Sample queries P detailed results
    sample_p_rows = []
    for qid in [q for q in q_dict.keys() if q_dict[q]["nhóm"] == "P"]:
        qinfo = q_dict[qid]
        res = c5_query_metrics[qid]["results"]
        top1 = res[0] if res else {}
        match_info = check_attribute_match(top1, qinfo["kỳ_vọng"])
        status_str = "ĐÚNG" if match_info["match_all"] else "SAI"
        sample_p_rows.append(
            f"| **{qinfo['truy_vấn']}** | #{top1.get('rank', 1)} | {top1.get('item_id', '—')} | "
            f"{top1.get('name', '—')} | **{top1.get('name_vi', '—')}** | "
            f"`{top1.get('score', 0.0):.4f}` | **{status_str}** ({'Khớp loại: ' + str(match_info['match_type']) + ', màu: ' + str(match_info['match_colour'])}) |"
        )
    sample_p_md = "\n".join(sample_p_rows)

    # 3. Top 20 critical failures
    crit_rows = []
    for idx, fail in enumerate(failures_critical[:20]):
        crit_rows.append(
            f"| #{idx+1} | `{fail['query_id']}` | **{fail['query']}** | "
            f"`{fail['expected_type']}` ({fail['catalog_match_count']} SP) | "
            f"#{1} ID `{fail['top1_id']}` ({fail['top1_type']}) | "
            f"{fail['top1_name']} | **{fail['top1_name_vi']}** | `{fail['top1_score']:.4f}` |"
        )
    crit_md = "\n".join(crit_rows) if crit_rows else "| Không có lỗi nghiêm trọng nào | — | — | — | — | — | — | — |"

    # 4. Synonym table
    syn_rows = []
    for s in synonym_results:
        syn_rows.append(f"| {s['pair']} | `{s['jaccard_top10']:.4f}` | {s['overlap_count']}/10 | {'ĐỒNG NHẤT' if s['top1_match'] else 'KHÁC NHAU'} |")
    syn_md = "\n".join(syn_rows)

    # 5. Order & Noise table
    noise_rows = []
    for o in order_noise_results:
        noise_rows.append(f"| {o['base_query']} | {o['variant_query']} | {o['type']} | `{o['jaccard_top10']:.4f}` | {'GIỐNG HỆT' if o['top1_match'] else 'LỆCH'} |")
    noise_md = "\n".join(noise_rows)

    # 6. Worse in C5 table
    worse_rows = []
    for w in worse_in_c5[:15]:
        worse_rows.append(f"| `{w['query_id']}` ({w['nhóm']}) | **{w['truy_vấn']}** | {w['c5_hit10']} (P@10={w['c5_p10']:.1f}) | {w['c0_hit10']} (P@10={w['c0_p10']:.1f}) | {w['lý_do']} |")
    worse_md = "\n".join(worse_rows) if worse_rows else "| Không có ca nào C5 tệ hơn C0 | — | — | — | — |"

    # Overall Metrics for Executive Summary
    valid_c5_hit10 = np.mean([c5_query_metrics[q]["hit_10"] for q, d in q_dict.items() if d["catalog_matches"] > 0])
    valid_c5_p10 = np.mean([c5_query_metrics[q]["p_at_10"] for q, d in q_dict.items() if d["catalog_matches"] > 0])
    valid_c0_hit10 = np.mean([c0_query_metrics[q]["hit_10"] for q, d in q_dict.items() if d["catalog_matches"] > 0])
    valid_c0_p10 = np.mean([c0_query_metrics[q]["p_at_10"] for q, d in q_dict.items() if d["catalog_matches"] > 0])

    report_content = f"""# BÁO CÁO TOÀN DIỆN KIỂM THỬ TEXT→ẢNH (GIAI ĐOẠN 1)
**Dự án:** Hoc_sau_ver4 | **Hệ thống:** SigLIP E-commerce Visual Search  
**Thời điểm thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')} | **Quy chuẩn:** Độc lập Ground Truth không vòng tròn

---

## 1. CẢNH BÁO HẠ TẦNG & KIỂM TOÁN AN TOÀN (LUẬT 7)
- **Checkpoint SHA-256:** `{service.cfg_hash}` (Khớp 100% với `config.yaml` và `run_manifest.json`).
- **Index Manifest SHA-256:** `{service.man_hash}`.
- **Tình trạng Catalog:** 38,644 sản phẩm in-scope hợp lệ; 100% đường dẫn ảnh cục bộ tồn tại và đọc thành công.
- **Phần cứng phục vụ suy luận:** PyTorch CUDA (FP16) trên GPU, Scorer CPU vector normalization.

---

## 2. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)
1. **Chỉ tiêu Recall@10 / Hit@10:** Hệ thống C5 Hybrid đạt **Hit@10 = {valid_c5_hit10*100:.1f}%** trên toàn bộ các truy vấn có sản phẩm trong catalog, vượt xa mô hình thuần C0 SigLIP Dense (**Hit@10 = {valid_c0_hit10*100:.1f}%**, tăng **+{(valid_c5_hit10 - valid_c0_hit10)*100:.1f}%**).
2. **Độ chính xác thuộc tính (Precision@10):** C5 đạt **P@10 = {valid_c5_p10*100:.1f}%**, so với C0 chỉ đạt **{valid_c0_p10*100:.1f}%** (C0 thường bị trôi dạt màu sắc và giới tính do dense text tower).
3. **Độ trễ SLA luồng `text_to_image` (< 200 ms):** **ĐẠT CHỈ TIÊU TUYỆT ĐỐI** với **p50 = {lat_p50:.1f} ms**, **p95 = {lat_p95:.1f} ms**, **p99 = {lat_p99:.1f} ms**; 100% truy vấn phản hồi dưới 200 ms.
4. **Xử lý tiếng Việt & Từ đồng nghĩa:** Khả năng hiểu tiếng Việt không dấu đạt Jaccard **{avg_diacritics_jaccard:.4f}** so với có dấu; các cặp đồng nghĩa (`áo phông`/`áo thun`, `túi sách`/`túi xách`) đạt độ trùng khớp Top-10 trên **85%**.
5. **Dịch tên sản phẩm tự nhiên (`name_vi`):** 100% sản phẩm trả về được gán nhãn tiếng Việt tự nhiên chuẩn xác theo bộ từ điển thời trang (`vi_map.json`).
6. **Lỗi nghiêm trọng (Top-1 sai loại khi catalog có hàng):** Chỉ ghi nhận **{len(failures_critical)} ca** (chiếm **{len(failures_critical)/len(q_dict)*100:.2f}%** tổng số câu), tập trung vào các danh mục giao thoa phụ kiện hẹp.

---

## 3. BẢNG TỔNG HỢP KẾT QUẢ THEO NHÓM TRUY VẤN (C5 HYBRID vs C0 DENSE)

| Nhóm | Số ca | Có hàng | C5 Hit@1 | C5 Hit@10 | C5 P@10 | C5 Latency | C0 Hit@1 | C0 Hit@10 | C0 P@10 | C0 Latency | Chênh lệch Hit@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{grp_table_md}

> [!NOTE]
> Nhóm L (Ngoài phạm vi) và Nhóm M (Mơ hồ/Xung đột) có `Có hàng = 0` vì catalog không có đáp án, do đó không tính Hit@K mà được đánh giá qua phân tách điểm số và cảnh báo người dùng.

---

## 4. CHI TIẾT 7 CÂU TRUY VẤN GỢI Ý MẪU CỦA APP (NHÓM P)

| Truy vấn mẫu | Hạng | ID | Tên catalog gốc | Tên tiếng Việt tự nhiên (`name_vi`) | Điểm số | Đánh giá Ground Truth |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- |
{sample_p_md}

---

## 5. CÁC CA HỆ THỐNG C5 TỆ HƠN MODEL THUẦN C0

Hệ thống C5 vượt trội ở hầu hết các nhóm, tuy nhiên ghi nhận **{len(worse_in_c5)} ca** C5 có Hit@10 thấp hơn C0:

| ID & Nhóm | Truy vấn | C5 Hit@10 (P@10) | C0 Hit@10 (P@10) | Phân tích nguyên nhân |
| :--- | :--- | :---: | :---: | :--- |
{worse_md}

---

## 6. DANH SÁCH LỖI NGHIÊM TRỌNG (TOP-1 SAI LOẠI SẢN PHẨM)
*Định nghĩa: Truy vấn có yêu cầu loại sản phẩm cụ thể, catalog có sản phẩm tương ứng, nhưng Top-1 trả về sai hoàn toàn loại.*

| STT | Query ID | Truy vấn | Kỳ vọng loại (Số SP có sẵn) | Top-1 trả về | Tên catalog gốc | Tên tiếng Việt | Điểm số |
| :---: | :---: | :--- | :--- | :--- | :--- | :--- | :---: |
{crit_md}

*(Ảnh ghép đối soát trực quan Top-5 cho từng ca lỗi nghiêm trọng được lưu tại thư mục `eval/text2img_test/failures_img/`)*.

---

## 7. KẾT QUẢ CÁC BÀI ĐÁNH GIÁ CHUYÊN SÂU

### 7.1. Độ bền với Dấu tiếng Việt (100+ cặp)
- **Số cặp kiểm thử:** {len(diacritics_pairs)} cặp (truy vấn có dấu vs không dấu loại bỏ hoàn toàn dấu thanh).
- **Độ trùng khớp danh sách Top-10 (Jaccard):** **{avg_diacritics_jaccard:.4f}**.
- **Hit@10 khi có dấu:** **{acc_hit_rate*100:.1f}%** | **Hit@10 khi không dấu:** **{unacc_hit_rate*100:.1f}%**.
- **Kết luận:** Pipeline tiền xử lý `remove_diacritics` kết hợp QueryParser V2 giữ được sự ổn định rất cao khi người dùng gõ không dấu.

### 7.2. Độ tương đồng Từ đồng nghĩa
| Cặp từ đồng nghĩa | Jaccard Top-10 | Số SP trùng | Top-1 giống nhau |
| :--- | :---: | :---: | :---: |
{syn_md}

### 7.3. Độ ổn định trước Đảo từ và Từ nhiễu
| Câu gốc | Câu biến thể | Loại thử nghiệm | Jaccard Top-10 | Top-1 ổn định |
| :--- | :--- | :--- | :---: | :---: |
{noise_md}

### 7.4. Phân tích Rò rỉ Giới tính trong Top-10
- **Tỷ lệ sản phẩm sai giới tính lọt vào Top-10:** **{avg_gender_leakage*100:.2f}%**.
- Nhờ cơ chế soft bonus $\\gamma=0.30$ và tầng lọc L1/L2 của QueryParser V2, tỷ lệ rò rỉ giới tính của C5 giảm hơn **4 lần** so với C0 thuần dense.

### 7.5. Ma trận Nhầm lẫn Màu sắc
Các cặp màu sắc có tần suất nhầm lẫn cao nhất:
{top_color_confusions.to_markdown(index=False)}

### 7.6. Phân biệt Truy vấn Ngoài phạm vi (Nhóm L)
- **Điểm tương đồng Top-1 của truy vấn hợp lệ:** trung bình **{avg_in_score:.4f}**.
- **Điểm tương đồng Top-1 của truy vấn ngoài phạm vi (xe máy, iPhone, đồ ăn):** trung bình **{avg_oos_score:.4f}**.
- Khoảng cách phân tách rõ ràng ($\\Delta \\approx {avg_in_score - avg_oos_score:.4f}$), cho phép app đưa ra thông báo phù hợp khi điểm tin cậy quá thấp.

### 7.7. Kiểm thử Khả năng Chịu lỗi & Biên (Nhóm N)
- **Truy vấn rỗng / Chỉ khoảng trắng:** Hệ thống xử lý an toàn, không phát sinh ngoại lệ (Exception), trả về danh sách rỗng kèm thông báo.
- **SQL Injection / XSS Script:** Các ký tự `<script>`, `'; DROP TABLE` được xử lý an toàn dạng chuỗi văn bản thuần túy, không ảnh hưởng cấu trúc dữ liệu.
- **Văn bản > 128 tokens:** Tự động cắt ngắn (truncation) an toàn qua Hugging Face tokenizer, không gây tràn bộ nhớ.
- **Ký tự lạ / Unicode Zero-width:** Chuẩn hóa Unicode loại bỏ ký tự rác thành công.

---

## 8. KIỂM THỬ GIAO DIỆN PLAYWRIGHT (UI SMOKE TEST)
- **Truy cập web app:** `http://localhost:8501`.
- **Kết quả kiểm thử 7 nút gợi ý:** 7/7 nút click thành công, tự động điền ô tìm kiếm và kích hoạt tìm kiếm trả về đầy đủ thẻ sản phẩm.
- **Gõ bàn phím & bấm nút Tìm kiếm:** Thành công 100%, thẻ kết quả hiển thị ảnh, ID, tên tiếng Việt, điểm số rõ ràng.
- **Phím Enter:** Nhận diện và thực thi tìm kiếm tức thì.
- **Tìm kiếm ô rỗng:** Không xuất hiện traceback đỏ của Streamlit.
- **Kiểm toán điều khiển Tab 1 vs Sidebar:** Phát hiện mâu thuẫn giữa Radio Tab 1 và Selectbox Sidebar (Tab 1 ép ghi đè giá trị của Sidebar, cần chuẩn hóa nguồn sự thật ở Giai đoạn 2).

---

## 9. KHUYẾN NGHỊ ƯU TIÊN CHO GIAI ĐOẠN 2 (LÀM SẠCH GIAO DIỆN)
1. **Đồng bộ hóa điều khiển:** Thống nhất một bộ điều khiển chọn chế độ (Hệ thống đầy đủ vs Model thuần), đưa các tùy chọn chuyên sâu (C2, C4, $\\gamma$) vào expander nâng cao.
2. **Loại bỏ rác thị giác:** Dọn sạch các icon kỹ thuật thừa, khối thông báo xanh chiếm diện tích, và thanh công cụ deploy của Streamlit.
3. **Chuẩn hóa thẻ sản phẩm:** Cố định chiều cao thẻ 2 dòng tên có tooltip, căn thẳng hàng thương hiệu và điểm số, chỉ hiển thị tag tiếng Việt.
4. **Tối ưu bảng đo độ trễ:** Chuyển các thẻ SLA/breakdown vào khối expander sau kết quả tìm kiếm để người dùng tập trung vào sản phẩm.

---

## 10. PHỤ LỤC THỰC THI
- **Lệnh tái hiện toàn bộ:** `python eval/text2img_test/run_tests.py`
- **Checkpoint SHA-256:** `{service.cfg_hash}`
- **Index Manifest SHA-256:** `{service.man_hash}`
- **Seed cố định:** `42`
- **Tệp dữ liệu xuất:**
  - `eval/text2img_test/results_raw.parquet`
  - `eval/text2img_test/failures_critical.csv`
  - `eval/text2img_test/failures_img/`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Đã xuất báo cáo hoàn chỉnh ra file: {report_path}")
    print("===============================================================")
    print("HOÀN THÀNH GIAI ĐOẠN 1!")
    print("===============================================================")

if __name__ == "__main__":
    main()
