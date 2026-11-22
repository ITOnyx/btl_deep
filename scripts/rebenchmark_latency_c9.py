"""
Phần 3 (PROMPT 10): Đo lại độ trễ T2I đúng protocol C.9:
- Cấu hình: cfg_v4_01, cfg_v4_05, cfg_v4_14, cfg_v4_15
- Chạy xen kẽ A-B-C-D-A-B-C-D qua 5 vòng × 200 truy vấn (warmup 20 mỗi cấu hình trước khi bắt đầu)
- Để GPU nguội mục tiêu < 60°C trước mỗi vòng/cấu hình (chờ tối đa 45s; nếu không đạt ghi rõ nhiệt độ thực tế và vi phạm)
- Báo p50/p95/p99 toàn cục + trung vị & IQR của p95 qua 5 vòng + chi phí tách riêng của encode text I3.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.run_step3_grid import gpu_telemetry
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import (
    build_catalog_template,
    multi_template_encode_texts,
    restore_diacritics_for_query,
)
from src.serve.search_engine import ECommerceSearchEngine

SEED = 42


def cool_gpu(target_c: int = 60, max_wait_s: float = 45.0) -> Dict[str, Any]:
    t0 = time.perf_counter()
    tele = gpu_telemetry()
    while isinstance(tele.get("temp_c"), int) and tele["temp_c"] >= target_c and (time.perf_counter() - t0) < max_wait_s:
        time.sleep(2.0)
        tele = gpu_telemetry()
    tele["cooled_below_target"] = bool(isinstance(tele.get("temp_c"), int) and tele["temp_c"] < target_c)
    tele["wait_s"] = round(time.perf_counter() - t0, 1)
    return tele


def run_one_query(
    engine: ECommerceSearchEngine,
    parser: VietnameseQueryParserV2Conf,
    q: str,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    """Returns (total_ms, text_encode_ms)."""
    mt = int(cfg.get("multi_template", 1))
    gal_vecs = engine.embeddings

    t0 = time.perf_counter()
    qt = restore_diacritics_for_query(q)
    parsed = parser.parse_with_confidence(qt, fuzzy_typo=False)

    t_enc0 = time.perf_counter()
    if mt > 1:
        qv = multi_template_encode_texts(engine, [q], [parsed], n_templates=mt)
    else:
        qv = engine.encode_text_query(qt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_enc1 = time.perf_counter()

    sims = (gal_vecs @ qv[0]).astype(np.float64)
    _ = np.argsort(-sims, kind="mergesort")[:10]
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, (t_enc1 - t_enc0) * 1000.0


def count_text_forwards(parser: VietnameseQueryParserV2Conf, queries_200: List[str], n_templates: int) -> float:
    counts = []
    for q in queries_200:
        rw = restore_diacritics_for_query(q)
        pr = parser.parse_with_confidence(rw, fuzzy_typo=False)
        tpl = build_catalog_template(pr)
        cands = [rw]
        if n_templates >= 3 and q != rw:
            cands.append(q)
        if tpl and tpl != rw:
            cands.append(tpl)
        counts.append(len(cands))
    return float(np.mean(counts))


def main() -> None:
    rule = json.loads(Path("reports/phase5c/selection_rule_v4.json").read_text(encoding="utf-8"))
    by_id = {c["config_id"]: c for c in rule["config_grid"]}
    target_ids = ["cfg_v4_01", "cfg_v4_05", "cfg_v4_14", "cfg_v4_15"]
    cfgs = [by_id[cid] for cid in target_ids]

    val_a = json.loads(Path("eval/val_queries_a_synthetic.json").read_text(encoding="utf-8"))
    val_b_all = json.loads(Path("eval/val_queries_b_synth.json").read_text(encoding="utf-8"))
    val_b_clean = [q for q in val_b_all if not q.get("is_typo")]
    queries_200 = ([q["query"] for q in val_a] + [q["query"] for q in val_b_clean])[:200]

    eng = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    parser = VietnameseQueryParserV2Conf()

    # Warmup 20 queries for each config
    for cfg in cfgs:
        for q in queries_200[:20]:
            _ = run_one_query(eng, parser, q, cfg)

    n_rounds = 5
    per_cfg_round_p95: Dict[str, List[float]] = {cid: [] for cid in target_ids}
    per_cfg_all_ms: Dict[str, List[float]] = {cid: [] for cid in target_ids}
    per_cfg_enc_ms: Dict[str, List[float]] = {cid: [] for cid in target_ids}
    round_tele: List[Dict[str, Any]] = []

    for r_idx in range(n_rounds):
        for cfg in cfgs:
            cid = cfg["config_id"]
            tele_before = cool_gpu(target_c=60, max_wait_s=30.0)
            r_ms = []
            r_enc = []
            for q in queries_200:
                tot_ms, enc_ms = run_one_query(eng, parser, q, cfg)
                r_ms.append(tot_ms)
                r_enc.append(enc_ms)
            tele_after = gpu_telemetry()
            p95_r = float(np.percentile(r_ms, 95))
            per_cfg_round_p95[cid].append(round(p95_r, 3))
            per_cfg_all_ms[cid].extend(r_ms)
            per_cfg_enc_ms[cid].extend(r_enc)
            round_tele.append({
                "round": r_idx + 1,
                "config_id": cid,
                "p95_ms": round(p95_r, 3),
                "before": tele_before,
                "after": tele_after,
            })
            print(f"Round {r_idx+1} | {cid} | p95={p95_r:.2f}ms | temp_before={tele_before.get('temp_c')}C -> after={tele_after.get('temp_c')}C")

    summary: Dict[str, Any] = {}
    p50_01 = float(np.percentile(per_cfg_all_ms["cfg_v4_01"], 50))
    p95_01 = float(np.percentile(per_cfg_all_ms["cfg_v4_01"], 95))
    enc_p50_01 = float(np.percentile(per_cfg_enc_ms["cfg_v4_01"], 50))
    enc_p95_01 = float(np.percentile(per_cfg_enc_ms["cfg_v4_01"], 95))

    for cfg in cfgs:
        cid = cfg["config_id"]
        arr = np.array(per_cfg_all_ms[cid], dtype=np.float64)
        enc_arr = np.array(per_cfg_enc_ms[cid], dtype=np.float64)
        rp95 = np.array(per_cfg_round_p95[cid], dtype=np.float64)
        q1, med, q3 = float(np.percentile(rp95, 25)), float(np.median(rp95)), float(np.percentile(rp95, 75))
        mt = int(cfg.get("multi_template", 1))
        summary[cid] = {
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
            "round_p95_ms": per_cfg_round_p95[cid],
            "round_p95_median_ms": round(med, 3),
            "round_p95_iqr_ms": round(q3 - q1, 3),
            "round_p95_q1_q3": [round(q1, 3), round(q3, 3)],
            "avg_text_forwards_per_query": round(count_text_forwards(parser, queries_200, mt) if mt > 1 else 1.0, 3),
            "encode_text_p50_ms": round(float(np.percentile(enc_arr, 50)), 3),
            "encode_text_p95_ms": round(float(np.percentile(enc_arr, 95)), 3),
            "delta_p50_vs_cfg01_ms": round(float(np.percentile(arr, 50)) - p50_01, 3),
            "delta_p95_vs_cfg01_ms": round(float(np.percentile(arr, 95)) - p95_01, 3),
            "delta_encode_p50_vs_cfg01_ms": round(float(np.percentile(enc_arr, 50)) - enc_p50_01, 3),
            "delta_encode_p95_vs_cfg01_ms": round(float(np.percentile(enc_arr, 95)) - enc_p95_01, 3),
        }

    out = {"seed": SEED, "n_rounds": n_rounds, "n_queries_per_round": 200, "summary": summary, "round_telemetry": round_tele}
    Path("reports/phase5c/step4_latency_c9.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved reports/phase5c/step4_latency_c9.json")


if __name__ == "__main__":
    main()
