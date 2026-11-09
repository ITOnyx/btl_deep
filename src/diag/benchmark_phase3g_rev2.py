"""
Benchmark Phase 3g Rev2: 5 Repeated Runs
Measures:
- Component breakdown repeated 5 times.
- Configs 0..5 (200 text queries, warmup 10, seed 42) repeated 5 times.
- Image-to-Image pipeline (200 images, PIL open, BILINEAR resize, GPU FP16 encode, topk) repeated 5 times.
Computes median, min, max for mean, p50, p95, p99 across 5 runs.
Saves to reports/phase3g_latency_multi_run.json.
"""

import sys
import json
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.search_pipeline import VisualSearchEngine
from src.lexical import rrf_merge

def get_stats(arr):
    return {
        "mean": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2)
    }

def aggregate_runs(runs_data):
    """Given a list of dicts with mean, p50, p95, p99 across runs, compute median, min, max."""
    keys = ["mean", "p50", "p95", "p99"]
    result = {}
    for k in keys:
        vals = [r[k] for r in runs_data]
        result[k] = {
            "median": round(float(np.median(vals)), 2),
            "min": round(float(np.min(vals)), 2),
            "max": round(float(np.max(vals)), 2),
            "all_runs": vals
        }
    return result

def get_gpu_environment():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,power.limit,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        gpu_info = {
            "gpu_name": parts[0],
            "driver_version": parts[1],
            "temperature_c": float(parts[2]),
            "power_draw_w": float(parts[3]),
            "power_limit_w": float(parts[4]),
            "memory_used_mb": float(parts[5]),
            "memory_total_mb": float(parts[6])
        }
    except Exception as e:
        gpu_info = {"error": str(e)}

    # Check background processes on GPU
    try:
        res_p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        gpu_info["background_processes"] = [line.strip() for line in res_p.stdout.strip().split("\n") if line.strip()]
    except Exception:
        gpu_info["background_processes"] = []
        
    return gpu_info

def main():
    print("=====================================================================")
    print("Phase 3g Rev2: Latency Benchmark (5 repeated runs)")
    print("=====================================================================")
    
    seed = 42
    np.random.seed(seed)
    
    # 1. Collect 200 text queries
    query_pool = []
    with open("eval/q6_queries.json", "r", encoding="utf-8") as f:
        q6 = json.load(f)
    for q in q6["val_queries"] + q6["test_queries"]:
        query_pool.append(q["query_text"])
        
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        heldout = json.load(f)
    for q in heldout:
        query_pool.append(q["query"])
        
    hw_path = Path("eval/handwritten_queries.json")
    if hw_path.exists():
        with open(hw_path, "r", encoding="utf-8") as f:
            hw = json.load(f)
        for q in hw:
            if isinstance(q, dict) and "query" in q:
                query_pool.append(q["query"])
            elif isinstance(q, str):
                query_pool.append(q)
                
    unique_queries = list(dict.fromkeys(query_pool))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_queries))
    selected_queries = [unique_queries[i] for i in perm[:200]]
    print(f"Loaded {len(selected_queries)} text queries.")
    
    # 2. Collect 200 valid image paths
    df_cat = pd.read_parquet("data/catalog/catalog.parquet")
    in_scope_paths = df_cat[df_cat["in_scope"] == True]["path"].dropna().tolist()
    
    valid_image_paths = []
    for p in in_scope_paths:
        pth = Path(p)
        if not pth.exists():
            pth = Path("data/fashion-dataset/images") / pth.name
        if pth.exists():
            valid_image_paths.append(str(pth))
        if len(valid_image_paths) >= 500:
            break
            
    print(f"Found {len(valid_image_paths)} valid images on disk.")
    rng_img = np.random.RandomState(seed + 1)
    sel_img_indices = rng_img.permutation(len(valid_image_paths))[:200]
    selected_images = [valid_image_paths[i] for i in sel_img_indices]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing search engine on {device}...")
    engine = VisualSearchEngine(device=device, gamma=0.3, alpha=1.0, enable_text_fusion=False)
    
    N_RUNS = 5
    components_runs = {
        "parse_query": [],
        "filter_candidates": [],
        "encode_text": [],
        "dense_topk_gpu": [],
        "sparse_bm25": [],
        "rrf_merge": [],
        "total_breakdown": []
    }
    
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
    
    configs_runs = {f"config_{i}": [] for i in range(6)}
    img2img_runs = []
    
    for run_idx in range(1, N_RUNS + 1):
        print(f"\n==================== RUN {run_idx}/{N_RUNS} ====================")
        
        # Warmup (10 queries & 10 images)
        print("Warming up...")
        for q in selected_queries[:10]:
            _ = engine.search_text(q, top_k=10)
        for ip in selected_images[:10]:
            _ = engine.encode_image(ip)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # 1. Component breakdown on Config 5
        t_parse_list, t_filter_list, t_encode_list, t_dense_list, t_bm25_list, t_rrf_list, t_total_list = [], [], [], [], [], [], []
        
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
            
        components_runs["parse_query"].append(get_stats(t_parse_list))
        components_runs["filter_candidates"].append(get_stats(t_filter_list))
        components_runs["encode_text"].append(get_stats(t_encode_list))
        components_runs["dense_topk_gpu"].append(get_stats(t_dense_list))
        components_runs["sparse_bm25"].append(get_stats(t_bm25_list))
        components_runs["rrf_merge"].append(get_stats(t_rrf_list))
        components_runs["total_breakdown"].append(get_stats(t_total_list))
        
        # 2. Configs 0..5 End-to-end
        print("Measuring Configs 0..5...")
        for cfg in config_definitions:
            cfg_id = cfg["id"]
            cfg_name = f"config_{cfg_id}"
            latencies = []
            
            if cfg.get("is_bm25_only"):
                for q in selected_queries:
                    t0 = time.perf_counter()
                    cand_mask = np.ones(engine.N, dtype=bool)
                    _ = engine.bm25.search(q, top_k=10, mask=cand_mask)
                    latencies.append((time.perf_counter() - t0) * 1000.0)
            else:
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
                    
            stats = get_stats(latencies)
            configs_runs[cfg_name].append(stats)
            print(f"  [{cfg_id}] {cfg['name']:<45}: mean={stats['mean']}ms, p50={stats['p50']}ms, p95={stats['p95']}ms, p99={stats['p99']}ms")
            
        # 3. Image-to-Image pipeline
        print("Measuring Image-to-Image...")
        image_latencies = []
        for ip in selected_images:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            img_vec = engine.encode_image(ip)
            if engine.device == "cuda" and engine.t_emb_img_c is not None:
                q_t = torch.tensor(img_vec, dtype=torch.float16, device=engine.device)
                sims_t = torch.matmul(engine.t_emb_img_c, q_t)
                topk_pos = torch.topk(sims_t, k=10)[1].cpu().numpy()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                sims = np.dot(engine.emb_img_c, img_vec)
                topk_pos = np.argsort(-sims)[:10]
                
            t1 = time.perf_counter()
            image_latencies.append((t1 - t0) * 1000.0)
            
        img_stats = get_stats(image_latencies)
        img2img_runs.append(img_stats)
        print(f"  Image-to-Image: mean={img_stats['mean']}ms, p50={img_stats['p50']}ms, p95={img_stats['p95']}ms, p99={img_stats['p99']}ms")

    # Aggregate across 5 runs
    components_agg = {k: aggregate_runs(v) for k, v in components_runs.items()}
    configs_agg = {k: aggregate_runs(v) for k, v in configs_runs.items()}
    img2img_agg = aggregate_runs(img2img_runs)
    
    gpu_env = get_gpu_environment()
    
    out_data = {
        "meta": {
            "num_runs": N_RUNS,
            "queries_per_run": len(selected_queries),
            "images_per_run": len(selected_images),
            "device": device,
            "gpu_env": gpu_env,
            "fp16": True,
            "batch_size": 1,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "components": components_agg,
        "configs": configs_agg,
        "image_to_image": img2img_agg
    }
    
    out_path = Path("reports/phase3g_latency_multi_run.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n=====================================================================")
    print(f"Saved latency multi-run results to {out_path}")
    print(f"=====================================================================")

if __name__ == "__main__":
    main()
