"""
Phase 3f: Latency Benchmark & Profiling (F4)
Measures latency breakdown and end-to-end latency across Configs 0..5 on >= 200 queries.
Enforces strict SLA: p95 < 200ms -> "ĐẠT", p95 >= 200ms -> "KHÔNG ĐẠT".
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.search_pipeline import VisualSearchEngine
from src.lexical import rrf_merge

def run_latency_benchmark():
    print("=====================================================================")
    print("Starting Phase 3f Latency Optimization & Profiling (F4)...")
    print("=====================================================================")
    
    seed = 42
    np.random.seed(seed)
    
    # 1. Collect >= 200 diverse queries
    query_pool = []
    
    # Q6 queries
    with open("eval/q6_queries.json", "r", encoding="utf-8") as f:
        q6 = json.load(f)
    for q in q6["val_queries"] + q6["test_queries"]:
        query_pool.append(q["query_text"])
        
    # Heldout queries
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        heldout = json.load(f)
    for q in heldout:
        query_pool.append(q["query"])
        
    # Handwritten queries if exists
    hw_path = Path("eval/handwritten_queries.json")
    if hw_path.exists():
        with open(hw_path, "r", encoding="utf-8") as f:
            hw = json.load(f)
        for q in hw:
            if isinstance(q, dict) and "query" in q:
                query_pool.append(q["query"])
            elif isinstance(q, str):
                query_pool.append(q)
                
    # Deduplicate and sample exactly 200 queries
    unique_queries = list(dict.fromkeys(query_pool))
    print(f"Total unique query candidates: {len(unique_queries)}")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_queries))
    selected_queries = [unique_queries[i] for i in perm[:200]]
    print(f"Sampled exactly {len(selected_queries)} queries for benchmarking.")
    
    # 2. Initialize VisualSearchEngine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing search engine on {device}...")
    engine = VisualSearchEngine(
        device=device,
        gamma=0.3,
        alpha=1.0,
        enable_text_fusion=False
    )
    
    # Warmup
    print("Warming up GPU and caches with 5 queries...")
    for q in selected_queries[:5]:
        _ = engine.search_text(q, top_k=10)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # 3. Component Breakdown Profiling on Config 5 (Full System)
    print("\n--- Profiling Component Breakdown on 200 queries (Config 5) ---")
    t_parse_list = []
    t_encode_list = []
    t_filter_list = []
    t_dense_list = []
    t_bm25_list = []
    t_rrf_list = []
    t_total_list = []
    
    for q in selected_queries:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        
        # 1. Parse
        tp0 = time.perf_counter()
        parsed = engine.parser.parse(q)
        if not engine.use_usage and "usage" in parsed:
            parsed["usage"] = None
        if not engine.use_season and "season" in parsed:
            parsed["season"] = None
        tp1 = time.perf_counter()
        
        # 2. Filter
        cand_idxs, tier = engine.attr_idx.filter_relaxed(parsed, min_candidates=engine.min_candidates)
        tp2 = time.perf_counter()
        
        # 3. Encode
        q_vec_c = engine.encode_text(q)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tp3 = time.perf_counter()
        
        # 4. Dense & top-100
        if engine.device == "cuda" and engine.t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=engine.device)
            q_t = torch.tensor(q_vec_c, dtype=torch.float16, device=engine.device)
            sims_img_t = torch.matmul(engine.t_emb_img_c[cand_t], q_t)
            bonus = engine.attr_idx.compute_attribute_soft_bonus(
                parsed, cand_idxs, gamma_type=engine.gamma, gamma_gender=engine.gamma,
                gamma_colour=engine.gamma, gamma_brand=engine.gamma
            )
            dense_scores_t = sims_img_t + torch.tensor(bonus, dtype=torch.float16, device=engine.device)
            k_top = min(100, len(cand_idxs))
            top_pos = torch.topk(dense_scores_t, k=k_top)[1].cpu().numpy()
            dense_order = cand_idxs[top_pos]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            sims_img = np.dot(engine.emb_img_c[cand_idxs], q_vec_c)
            bonus = engine.attr_idx.compute_attribute_soft_bonus(
                parsed, cand_idxs, gamma_type=engine.gamma, gamma_gender=engine.gamma,
                gamma_colour=engine.gamma, gamma_brand=engine.gamma
            )
            dense_scores = sims_img + bonus
            k_top = min(100, len(cand_idxs))
            dense_order = cand_idxs[np.argsort(-dense_scores)[:k_top]]
        tp4 = time.perf_counter()
        
        # 5. BM25
        cand_mask = np.zeros(engine.N, dtype=bool)
        cand_mask[cand_idxs] = True
        sparse_order, _ = engine.bm25.search(q, top_k=100, mask=cand_mask)
        tp5 = time.perf_counter()
        
        # 6. RRF Merge
        final_idxs, rrf_scores = rrf_merge(dense_order, sparse_order, k=engine.rrf_k, top_k=10)
        tp6 = time.perf_counter()
        
        t_parse_list.append((tp1 - tp0) * 1000.0)
        t_filter_list.append((tp2 - tp1) * 1000.0)
        t_encode_list.append((tp3 - tp2) * 1000.0)
        t_dense_list.append((tp4 - tp3) * 1000.0)
        t_bm25_list.append((tp5 - tp4) * 1000.0)
        t_rrf_list.append((tp6 - tp5) * 1000.0)
        t_total_list.append((tp6 - t_start) * 1000.0)
        
    breakdown = {
        "parse_query_ms": {"mean": round(float(np.mean(t_parse_list)), 2), "p95": round(float(np.percentile(t_parse_list, 95)), 2)},
        "filter_candidates_ms": {"mean": round(float(np.mean(t_filter_list)), 2), "p95": round(float(np.percentile(t_filter_list, 95)), 2)},
        "encode_text_ms": {"mean": round(float(np.mean(t_encode_list)), 2), "p95": round(float(np.percentile(t_encode_list, 95)), 2)},
        "dense_topk_gpu_ms": {"mean": round(float(np.mean(t_dense_list)), 2), "p95": round(float(np.percentile(t_dense_list, 95)), 2)},
        "sparse_bm25_ms": {"mean": round(float(np.mean(t_bm25_list)), 2), "p95": round(float(np.percentile(t_bm25_list, 95)), 2)},
        "rrf_merge_ms": {"mean": round(float(np.mean(t_rrf_list)), 2), "p95": round(float(np.percentile(t_rrf_list, 95)), 2)},
        "total_ms": {"mean": round(float(np.mean(t_total_list)), 2), "p95": round(float(np.percentile(t_total_list, 95)), 2)}
    }
    
    print("Component Breakdown (Mean / p95):")
    for comp, stats in breakdown.items():
        print(f"  {comp:<22}: {stats['mean']:>6.2f} ms (p95: {stats['p95']:>6.2f} ms)")
        
    # 4. Measure End-to-End Latency for Configs 0 to 5
    config_definitions = [
        {
            "id": 0, "name": "Config 0: Dense Model Only",
            "kwargs": {"enable_parser": False, "filter_mode": "none", "gamma": 0.0, "enable_bm25": False, "enable_text_fusion": False}
        },
        {
            "id": 1, "name": "Config 1: Candidate Filter + Dense",
            "kwargs": {"enable_parser": True, "filter_mode": "relaxed", "gamma": 0.0, "enable_bm25": False, "enable_text_fusion": False}
        },
        {
            "id": 2, "name": "Config 2: Candidate Filter + Soft Bonus + Dense",
            "kwargs": {"enable_parser": True, "filter_mode": "relaxed", "gamma": 0.3, "enable_bm25": False, "enable_text_fusion": False}
        },
        {
            "id": 3, "name": "Config 3: Candidate Filter + Soft Bonus + Text Fusion (alpha*=1.0)",
            "kwargs": {"enable_parser": True, "filter_mode": "relaxed", "gamma": 0.3, "alpha": 1.0, "enable_bm25": False, "enable_text_fusion": False}
        },
        {
            "id": 4, "name": "Config 4: BM25 Sparse Alone",
            "is_bm25_only": True
        },
        {
            "id": 5, "name": "Config 5: Full System (Hybrid Dense+BM25+RRF)",
            "kwargs": {"enable_parser": True, "filter_mode": "relaxed", "gamma": 0.3, "alpha": 1.0, "enable_bm25": True, "enable_text_fusion": False}
        }
    ]
    
    config_results = []
    print("\n--- Measuring End-to-End Latency across Configs 0..5 (200 queries each) ---")
    
    for cfg in config_definitions:
        latencies = []
        
        if cfg.get("is_bm25_only"):
            # BM25 alone
            for q in selected_queries:
                t0 = time.perf_counter()
                cand_mask = np.ones(engine.N, dtype=bool)
                _ = engine.bm25.search(q, top_k=10, mask=cand_mask)
                latencies.append((time.perf_counter() - t0) * 1000.0)
        else:
            # Configure engine
            for k, v in cfg["kwargs"].items():
                setattr(engine, k, v)
                
            for q in selected_queries:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = engine.search_text(q, top_k=10)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
                
        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        
        status = "DAT" if p95_lat < 200.0 else "KHONG DAT"
        status_vn = "ĐẠT" if p95_lat < 200.0 else "KHÔNG ĐẠT"
        
        res_entry = {
            "config_id": cfg["id"],
            "config_name": cfg["name"],
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "p99_ms": round(p99_lat, 2),
            "sla_p95_target_ms": 200.0,
            "status": status_vn
        }
        config_results.append(res_entry)
        
        print(f"[{cfg['id']}] {cfg['name']:<55} | Mean: {mean_lat:>6.2f}ms | p50: {p50_lat:>6.2f}ms | p95: {p95_lat:>6.2f}ms | p99: {p99_lat:>6.2f}ms -> {status}")
        
    out_payload = {
        "n_queries": len(selected_queries),
        "seed": seed,
        "device": device,
        "breakdown": breakdown,
        "configurations": config_results
    }
    
    out_file = Path("reports/phase3f_latency_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    run_latency_benchmark()
