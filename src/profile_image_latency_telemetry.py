"""
Profile Image-to-Image latency telemetry across 5 runs.
Captures nvidia-smi per run and logs timestamps for all slow queries (> 500 ms).
"""
import sys
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from src.search_pipeline import VisualSearchEngine

def get_nvidia_smi():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "timestamp": parts[0],
            "temperature_c": float(parts[1]),
            "utilization_gpu_pct": float(parts[2]),
            "graphics_clock_mhz": float(parts[3]),
            "memory_clock_mhz": float(parts[4]),
            "power_draw_w": float(parts[5])
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

def main():
    print("=====================================================================")
    print("Profiling Image-to-Image Latency Telemetry & Slow Queries (>500 ms)")
    print("=====================================================================")
    
    seed = 42
    np.random.seed(seed)
    
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
            
    print(f"Found {len(valid_image_paths)} valid images.")
    rng_img = np.random.RandomState(seed + 1)
    sel_img_indices = rng_img.permutation(len(valid_image_paths))[:200]
    selected_images = [valid_image_paths[i] for i in sel_img_indices]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing VisualSearchEngine on {device}...")
    engine = VisualSearchEngine(device=device, gamma=0.3, alpha=1.0, enable_text_fusion=False)
    
    N_RUNS = 5
    run_telemetry = []
    all_slow_queries = []
    runs_stats = []
    
    # Warmup 10 queries
    print("Running initial 10 warmup queries...")
    for ip in selected_images[:10]:
        _ = engine.encode_image(ip)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    for run_idx in range(1, N_RUNS + 1):
        print(f"\n--- Starting Run {run_idx}/{N_RUNS} ---")
        smi_start = get_nvidia_smi()
        print(f"  GPU Start: Temp={smi_start.get('temperature_c')}C, Graphics Clock={smi_start.get('graphics_clock_mhz')}MHz")
        
        latencies = []
        run_slow_queries = []
        
        run_start_time = time.time()
        for idx, ip in enumerate(selected_images):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            q_timestamp = datetime.now().isoformat()
            
            # End-to-end image query
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
            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)
            
            if lat_ms > 500.0:
                slow_smi = get_nvidia_smi()
                slow_item = {
                    "run_idx": run_idx,
                    "query_idx": idx,
                    "image_path": ip,
                    "timestamp": q_timestamp,
                    "latency_ms": round(lat_ms, 2),
                    "gpu_smi": slow_smi
                }
                print(f"  [SLOW QUERY > 500ms] Run {run_idx} Query {idx}: {lat_ms:.2f} ms at {q_timestamp} (Temp: {slow_smi.get('temperature_c')}C)")
                run_slow_queries.append(slow_item)
                all_slow_queries.append(slow_item)
                
        run_end_time = time.time()
        smi_end = get_nvidia_smi()
        print(f"  GPU End: Temp={smi_end.get('temperature_c')}C, Graphics Clock={smi_end.get('graphics_clock_mhz')}MHz")
        
        stats = {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "max": round(float(np.max(latencies)), 2)
        }
        runs_stats.append(stats)
        print(f"  Run {run_idx} Stats: mean={stats['mean']}ms, p50={stats['p50']}ms, p95={stats['p95']}ms, p99={stats['p99']}ms, max={stats['max']}ms")
        
        run_telemetry.append({
            "run_idx": run_idx,
            "duration_s": round(run_end_time - run_start_time, 2),
            "gpu_smi_start": smi_start,
            "gpu_smi_end": smi_end,
            "stats": stats,
            "slow_queries_count": len(run_slow_queries)
        })
        
    out_data = {
        "metadata": {
            "task": "Image-to-Image latency telemetry and slow queries audit",
            "device": device,
            "n_runs": N_RUNS,
            "images_per_run": len(selected_images),
            "warmup_queries": 10,
            "timestamp": datetime.now().isoformat()
        },
        "runs_summary": runs_stats,
        "run_telemetry": run_telemetry,
        "all_slow_queries_over_500ms": all_slow_queries
    }
    
    out_path = Path("reports/phase3g_slow_queries_telemetry.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved telemetry report to {out_path}")
    print(f"Total slow queries (>500 ms) detected across 5 runs: {len(all_slow_queries)}")

if __name__ == "__main__":
    main()
