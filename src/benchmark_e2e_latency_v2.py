"""
End-to-End Latency Benchmark for Phase 4 Audit (Round 2)
Entrypoint: app/engine.py (SearchService)
Optimizations Integrated:
- Catalog Embedding Cache for CIR-catalog
- Batch-2 TTA for Image -> Image
- Text -> Image (C5 Serving)
- CIR-upload (External image upload with full ViT forward)

Protocol:
- Interleaved execution across 5 runs to prevent biased thermal buildup
- 10 warmups per pipeline
- 5 runs x 200 distinct queries per pipeline
- Cooldown pauses (5s) between runs
- Background thread nvidia-smi 1s telemetry
- Outputs saved to reports/phase4/phase4_latency_e2e_v2.json
"""

import sys
import os
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from app.engine import SearchService

class GpuMonitor:
    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.running = False
        self.thread = None
        self.samples = []
        self._lock = threading.Lock()

    def _monitor(self):
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,memory.used", "--format=csv,noheader,nounits"]
        while self.running:
            t0 = time.time()
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                line = res.stdout.strip()
                if line:
                    parts = [float(x.strip()) for x in line.split(",")]
                    with self._lock:
                        self.samples.append({
                            "timestamp": t0,
                            "temp_c": parts[0],
                            "clock_mhz": parts[1],
                            "mem_mb": parts[2]
                        })
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self) -> List[Dict[str, float]]:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        with self._lock:
            return list(self.samples)

def get_stats(latencies: List[float], gpu_samples: List[Dict[str, float]]) -> Dict[str, Any]:
    lat_arr = np.array(latencies, dtype=np.float64)
    temps = [s["temp_c"] for s in gpu_samples] if gpu_samples else [0.0]
    clocks = [s["clock_mhz"] for s in gpu_samples] if gpu_samples else [0.0]
    mems = [s["mem_mb"] for s in gpu_samples] if gpu_samples else [0.0]

    return {
        "queries_count": len(latencies),
        "p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "max_ms": round(float(np.max(lat_arr)), 2),
        "min_ms": round(float(np.min(lat_arr)), 2),
        "mean_ms": round(float(np.mean(lat_arr)), 2),
        "queries_over_200ms": int(np.sum(lat_arr > 200.0)),
        "queries_over_500ms": int(np.sum(lat_arr > 500.0)),
        "sla_p95_passed": bool(np.percentile(lat_arr, 95) < 200.0),
        "gpu_telemetry": {
            "samples_count": len(gpu_samples),
            "temp_min": round(float(np.min(temps)), 1) if temps else 0.0,
            "temp_max": round(float(np.max(temps)), 1) if temps else 0.0,
            "temp_mean": round(float(np.mean(temps)), 1) if temps else 0.0,
            "clock_min": round(float(np.min(clocks)), 0) if clocks else 0.0,
            "clock_max": round(float(np.max(clocks)), 0) if clocks else 0.0,
            "clock_mean": round(float(np.mean(clocks)), 0) if clocks else 0.0,
            "mem_mean": round(float(np.mean(mems)), 0) if mems else 0.0
        }
    }

def main():
    print("=== INITIALIZING END-TO-END SEARCH SERVICE (ROUND 2) ===")
    service = SearchService(config_path="config.yaml")
    df = service.engine.df
    
    # 1. Prepare 200 distinct text queries
    print("Preparing 200 distinct text queries...")
    text_queries = []
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        for q in json.load(f):
            t = q.get("query", "").strip()
            if t and t not in text_queries:
                text_queries.append(t)
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        for q in json.load(f).get("q2_attribute_queries", []):
            t = q.get("query_text", "").strip()
            if t and t not in text_queries:
                text_queries.append(t)
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        for q in json.load(f):
            t = q.get("query", "").strip()
            if t and t not in text_queries:
                text_queries.append(t)
    with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
        q5 = json.load(f)
        ql = q5.get("queries", []) if isinstance(q5, dict) else q5
        for q in ql:
            t = q.get("query_text", q.get("query", "")).strip() if isinstance(q, dict) else str(q).strip()
            if t and t not in text_queries:
                text_queries.append(t)
    text_queries = text_queries[:200]
    print(f"Loaded {len(text_queries)} text queries.")

    # 2. Prepare 200 distinct image paths for visual search
    print("Preparing 200 distinct image paths...")
    img_dir = Path("data/fashion-dataset/images")
    valid_img_paths = []
    rng = np.random.RandomState(20261109)
    shuffled_ids = rng.permutation(df["item_id"].astype(str).values)
    for sid in shuffled_ids:
        p = img_dir / f"{sid}.jpg"
        if p.exists():
            valid_img_paths.append(str(p))
            if len(valid_img_paths) == 200:
                break
    print(f"Loaded {len(valid_img_paths)} image paths.")

    # 3. Prepare 200 distinct CIR queries
    print("Preparing 200 distinct CIR queries...")
    modifiers = [
        "màu đen", "màu đỏ", "màu trắng", "màu xanh navy", "màu xám",
        "phong cách thể thao", "dáng rộng", "chất liệu da", "thời trang mùa hè",
        "phong cách công sở", "họa tiết kẻ sọc", "phom ôm năng động",
        "phong cách dạo phố", "cổ chữ v", "tay dài ấm áp", "đơn giản thanh lịch"
    ]
    cir_queries = []
    cir_img_paths = []
    for sid in shuffled_ids[::-1]:
        p = img_dir / f"{sid}.jpg"
        if p.exists():
            cir_img_paths.append(str(p))
            if len(cir_img_paths) == 200:
                break
    for i in range(200):
        mod = modifiers[i % len(modifiers)] + f" biến thể {i+1}"
        cir_queries.append((cir_img_paths[i], mod))
    print(f"Loaded {len(cir_queries)} CIR queries.")

    # Warmup all 4 pipelines
    print("\n--- RUNNING 10 WARM-UPS PER PIPELINE ---")
    for i in range(10):
        _ = service.search_text(text_queries[i], config_mode="c5", top_k=10)
        _ = service.search_image(valid_img_paths[i], top_k=10, use_flip_aug=True, use_knn_voting=True)
        img_p, mod = cir_queries[i]
        _ = service.search_cir(img_p, mod, top_k=10, force_upload=False)
        _ = service.search_cir(img_p, mod, top_k=10, force_upload=True)
    print("Warm-ups completed. Sleeping 5s before benchmarking...")
    time.sleep(5.0)

    # Define run schedules (interleaved order)
    run_schedules = [
        ["text", "image", "cir_cat", "cir_up"],
        ["cir_up", "cir_cat", "image", "text"],
        ["image", "cir_cat", "cir_up", "text"],
        ["cir_cat", "text", "cir_up", "image"],
        ["text", "cir_up", "image", "cir_cat"]
    ]

    pipeline_runs = {
        "text_to_image": [],
        "image_to_image": [],
        "cir_catalog": [],
        "cir_upload": []
    }

    for run_idx, schedule in enumerate(run_schedules, start=1):
        print(f"\n=======================================================")
        print(f"STARTING RUN {run_idx}/5 (Order: {' -> '.join(schedule)})")
        print(f"=======================================================")
        
        for p_name in schedule:
            print(f"  Executing pipeline: {p_name.upper()} ({run_idx}/5)...")
            monitor = GpuMonitor(interval_s=1.0)
            monitor.start()
            run_lats = []

            if p_name == "text":
                for q in text_queries:
                    t0 = time.perf_counter()
                    _ = service.search_text(q, config_mode="c5", top_k=10)
                    run_lats.append((time.perf_counter() - t0) * 1000.0)
                pipe_key = "text_to_image"

            elif p_name == "image":
                for img_p in valid_img_paths:
                    t0 = time.perf_counter()
                    _ = service.search_image(img_p, top_k=10, use_flip_aug=True, use_knn_voting=True)
                    run_lats.append((time.perf_counter() - t0) * 1000.0)
                pipe_key = "image_to_image"

            elif p_name == "cir_cat":
                for img_p, mod in cir_queries:
                    t0 = time.perf_counter()
                    _ = service.search_cir(img_p, mod, top_k=10, force_upload=False)
                    run_lats.append((time.perf_counter() - t0) * 1000.0)
                pipe_key = "cir_catalog"

            elif p_name == "cir_up":
                for img_p, mod in cir_queries:
                    t0 = time.perf_counter()
                    _ = service.search_cir(img_p, mod, top_k=10, force_upload=True)
                    run_lats.append((time.perf_counter() - t0) * 1000.0)
                pipe_key = "cir_upload"

            gpu_samples = monitor.stop()
            stats = get_stats(run_lats, gpu_samples)
            stats["run"] = run_idx
            pipeline_runs[pipe_key].append(stats)
            print(f"    [{pipe_key}] p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, p99={stats['p99_ms']}ms, max={stats['max_ms']}ms | GPU: {stats['gpu_telemetry']['temp_mean']}°C, {stats['gpu_telemetry']['clock_mean']}MHz")
            
            # Cooldown pause
            time.sleep(3.0)

        # Inter-run cooling
        print(f"  End of Run {run_idx}. Cooling down for 5 seconds...")
        time.sleep(5.0)

    # Structure final JSON
    benchmark_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": "NVIDIA GeForce RTX 2050 Mobile (4096 MiB)",
        "model_architecture": "SigLIP so400m-patch14-224 (hidden_dim 1152)",
        "protocol": "Interleaved 5 runs x 200 distinct queries/run with 10 warmups, cooldown pauses, and continuous nvidia-smi telemetry",
        "sla_definition": {
            "metric": "p95",
            "threshold_ms": 200.0
        },
        "pipelines": {
            "text_to_image": {
                "entrypoint": "app.engine.SearchService.search_text(config_mode='c5')",
                "sla_target": "p95 < 200 ms",
                "runs": pipeline_runs["text_to_image"]
            },
            "image_to_image": {
                "entrypoint": "app.engine.SearchService.search_image(use_flip_aug=True, use_knn_voting=True) [Batch-2 TTA]",
                "sla_target": "p95 < 200 ms",
                "runs": pipeline_runs["image_to_image"]
            },
            "cir_catalog": {
                "entrypoint": "app.engine.SearchService.search_cir(force_upload=False) [Catalog Cache]",
                "sla_target": "p95 < 200 ms",
                "runs": pipeline_runs["cir_catalog"]
            },
            "cir_upload": {
                "entrypoint": "app.engine.SearchService.search_cir(force_upload=True) [ViT Forward Pass]",
                "sla_target": "p95 < 200 ms",
                "runs": pipeline_runs["cir_upload"]
            }
        }
    }

    out_path = Path("reports/phase4/phase4_latency_e2e_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"BENCHMARK COMPLETED SUCCESSFULLY! Saved to {out_path}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
