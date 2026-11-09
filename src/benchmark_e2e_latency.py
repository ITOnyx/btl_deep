"""
End-to-End Latency Benchmark for Phase 4 Audit (T4.2)
Entrypoint: app/engine.py (SearchService)
Pipelines:
1. Text -> Image (search_text)
2. Image -> Image (search_image with flip aug & knn voting)
3. CIR (search_cir)

Protocol:
- 10 warm-up queries per pipeline
- 5 runs x 200 distinct queries per run
- Background thread samples nvidia-smi every 1s (recording min/max/mean temp & clock)
- Results saved to reports/phase4/phase4_latency_e2e.json
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
        "all_queries_under_200ms": bool(np.all(lat_arr < 200.0)),
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
    print("=== INITIALIZING END-TO-END SEARCH SERVICE ===")
    service = SearchService(config_path="config.yaml")
    df = service.engine.df
    
    # 1. Prepare 200 distinct text queries
    print("Preparing 200 distinct text queries...")
    text_queries = []
    # from handwritten (100)
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        hq = json.load(f)
        for q in hq:
            t = q.get("query", "").strip()
            if t and t not in text_queries:
                text_queries.append(t)
    # from testset_v1.json q2_attribute_queries (61)
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        ts = json.load(f)
        for q in ts.get("q2_attribute_queries", []):
            t = q.get("query_text", "").strip()
            if t and t not in text_queries:
                text_queries.append(t)
    # from heldout (30)
    with open("eval/heldout_queries.json", "r", encoding="utf-8") as f:
        ho = json.load(f)
        for q in ho:
            t = q.get("query", "").strip()
            if t and t not in text_queries:
                text_queries.append(t)
    # from q5 brand (150)
    with open("eval/q5_brand.json", "r", encoding="utf-8") as f:
        q5 = json.load(f)
        queries_list = q5.get("queries", []) if isinstance(q5, dict) else q5
        for q in queries_list:
            if isinstance(q, dict):
                t = q.get("query_text", q.get("query", "")).strip()
            else:
                t = str(q).strip()
            if t and t not in text_queries:
                text_queries.append(t)
    
    text_queries = text_queries[:200]
    print(f"Loaded {len(text_queries)} distinct text queries.")
    assert len(text_queries) == 200, f"Expected 200 queries, got {len(text_queries)}"

    # 2. Prepare 200 distinct image paths
    print("Preparing 200 distinct image paths from catalog...")
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
    print(f"Loaded {len(valid_img_paths)} distinct image paths.")
    assert len(valid_img_paths) == 200, f"Expected 200 image paths, got {len(valid_img_paths)}"

    # 3. Prepare 200 distinct CIR queries (image + distinct modifier)
    print("Preparing 200 distinct CIR queries...")
    modifiers = [
        "màu đen", "màu đỏ", "màu trắng", "màu xanh navy", "màu xám",
        "phong cách thể thao", "dáng rộng", "chất liệu da", "thời trang mùa hè",
        "phong cách công sở", "họa tiết kẻ sọc", "phom ôm năng động",
        "phong cách dạo phố", "cổ chữ v", "tay dài ấm áp", "đơn giản thanh lịch"
    ]
    cir_queries = []
    # Pick another 200 distinct images
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
    print(f"Loaded {len(cir_queries)} distinct CIR queries.")

    benchmark_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": "NVIDIA GeForce RTX 2050 Mobile",
        "protocol": "End-to-End via app/engine.py SearchService (10 warmups, 5 runs x 200 distinct queries/run, nvidia-smi 1s background thread sampling)",
        "pipelines": {}
    }

    # =========================================================================
    # PIPELINE 1: TEXT -> IMAGE
    # =========================================================================
    print("\n--- BENCHMARKING PIPELINE 1: TEXT -> IMAGE (search_text c5) ---")
    print("Running 10 warm-up text queries...")
    for i in range(10):
        _ = service.search_text(text_queries[i], config_mode="c5", top_k=10)

    pipe1_runs = []
    for run_idx in range(1, 6):
        print(f"  Starting Run {run_idx}/5 (200 distinct queries)...")
        monitor = GpuMonitor(interval_s=1.0)
        monitor.start()
        run_lats = []
        for q in text_queries:
            t0 = time.perf_counter()
            _ = service.search_text(q, config_mode="c5", top_k=10)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            run_lats.append(lat_ms)
        gpu_samples = monitor.stop()
        stats = get_stats(run_lats, gpu_samples)
        stats["run"] = run_idx
        pipe1_runs.append(stats)
        print(f"    Run {run_idx}: p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, p99={stats['p99_ms']}ms, max={stats['max_ms']}ms, GPU temp={stats['gpu_telemetry']['temp_mean']}°C, clock={stats['gpu_telemetry']['clock_mean']}MHz")

    # =========================================================================
    # PIPELINE 2: CIR (IMAGE + TEXT -> IMAGE)
    # =========================================================================
    print("\n--- BENCHMARKING PIPELINE 2: CIR (search_cir) ---")
    print("Running 10 warm-up CIR queries...")
    for i in range(10):
        img_p, mod = cir_queries[i]
        _ = service.search_cir(img_p, mod, top_k=10, use_soft_bonus=True)

    pipe2_runs = []
    for run_idx in range(1, 6):
        print(f"  Starting Run {run_idx}/5 (200 distinct queries)...")
        monitor = GpuMonitor(interval_s=1.0)
        monitor.start()
        run_lats = []
        for img_p, mod in cir_queries:
            t0 = time.perf_counter()
            _ = service.search_cir(img_p, mod, top_k=10, use_soft_bonus=True)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            run_lats.append(lat_ms)
        gpu_samples = monitor.stop()
        stats = get_stats(run_lats, gpu_samples)
        stats["run"] = run_idx
        pipe2_runs.append(stats)
        print(f"    Run {run_idx}: p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, p99={stats['p99_ms']}ms, max={stats['max_ms']}ms, GPU temp={stats['gpu_telemetry']['temp_mean']}°C, clock={stats['gpu_telemetry']['clock_mean']}MHz")

    # =========================================================================
    # PIPELINE 3: IMAGE -> IMAGE
    # =========================================================================
    print("\n--- BENCHMARKING PIPELINE 3: IMAGE -> IMAGE (search_image) ---")
    print("Running 10 warm-up image queries...")
    for i in range(10):
        _ = service.search_image(valid_img_paths[i], top_k=10, use_flip_aug=True, use_knn_voting=True)

    pipe3_runs = []
    for run_idx in range(1, 6):
        print(f"  Starting Run {run_idx}/5 (200 distinct queries)...")
        monitor = GpuMonitor(interval_s=1.0)
        monitor.start()
        run_lats = []
        for img_p in valid_img_paths:
            t0 = time.perf_counter()
            _ = service.search_image(img_p, top_k=10, use_flip_aug=True, use_knn_voting=True)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            run_lats.append(lat_ms)
        gpu_samples = monitor.stop()
        stats = get_stats(run_lats, gpu_samples)
        stats["run"] = run_idx
        pipe3_runs.append(stats)
        print(f"    Run {run_idx}: p50={stats['p50_ms']}ms, p95={stats['p95_ms']}ms, p99={stats['p99_ms']}ms, max={stats['max_ms']}ms, GPU temp={stats['gpu_telemetry']['temp_mean']}°C, clock={stats['gpu_telemetry']['clock_mean']}MHz")

    benchmark_output["pipelines"] = {
        "text_to_image": {
            "entrypoint": "app.engine.SearchService.search_text(config_mode='c5')",
            "sla_target_agents_md": "< 200 ms per query (all queries)",
            "sla_target_config_yaml": "p95 < 200 ms",
            "runs": pipe1_runs
        },
        "composed_image_retrieval_cir": {
            "entrypoint": "app.engine.SearchService.search_cir(use_soft_bonus=True)",
            "sla_target_agents_md": "< 200 ms per query (all queries)",
            "sla_target_config_yaml": "p95 < 200 ms",
            "runs": pipe2_runs
        },
        "image_to_image": {
            "entrypoint": "app.engine.SearchService.search_image(use_flip_aug=True, use_knn_voting=True)",
            "sla_target_agents_md": "< 200 ms per query (all queries)",
            "sla_target_config_yaml": "p95 < 200 ms",
            "runs": pipe3_runs
        }
    }

    out_path = Path("reports/phase4/phase4_latency_e2e.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2, ensure_ascii=False)

    print(f"\nAll 3 Pipelines successfully benchmarked and saved to {out_path}!")

if __name__ == "__main__":
    main()
