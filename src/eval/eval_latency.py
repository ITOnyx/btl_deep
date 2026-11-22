"""Phase 5: Latency SLA Benchmark (4 flows separately).

Compliance (AGENTS.md C.9 & config.yaml):
- Flows: text_to_image, image_to_image, cir_catalog, cir_upload
- >= 5 runs x 200 queries per flow, warmup 20
- torch.cuda.synchronize() around each query forward/search
- Logs GPU temperature via nvidia-smi
- Reports p50, p95, p99 (ms/query) vs SLA target p95 < 200 ms.
"""

from __future__ import annotations

import os

os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from PIL import Image

from src.serve.search_engine import ECommerceSearchEngine


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _query_gpu_telemetry() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        return {
            "gpu_name": parts[0],
            "temperature_c": float(parts[1]),
            "utilization_pct": float(parts[2]),
            "memory_used_mb": float(parts[3]),
            "memory_total_mb": float(parts[4]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark_flow(
    name: str,
    fn: Callable[[int], Any],
    device: str,
    warmup: int = 20,
    n_runs: int = 5,
    n_queries: int = 200,
    sla_p95_ms: float = 200.0,
) -> Dict[str, Any]:
    print(f"Benchmarking flow '{name}' (warmup={warmup}, {n_runs} runs x {n_queries} queries)...")
    for w in range(warmup):
        _sync_cuda(device)
        fn(w)
        _sync_cuda(device)

    all_latencies_ms: List[float] = []
    per_run_stats: List[Dict[str, float]] = []

    for r in range(n_runs):
        run_ms: List[float] = []
        for q_idx in range(n_queries):
            _sync_cuda(device)
            t0 = time.perf_counter()
            fn(q_idx)
            _sync_cuda(device)
            t1 = time.perf_counter()
            run_ms.append((t1 - t0) * 1000.0)

        arr = np.asarray(run_ms, dtype=np.float64)
        per_run_stats.append(
            {
                "run": r + 1,
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "mean_ms": float(np.mean(arr)),
            }
        )
        all_latencies_ms.extend(run_ms)

    total_arr = np.asarray(all_latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(total_arr, 50))
    p95 = float(np.percentile(total_arr, 95))
    p99 = float(np.percentile(total_arr, 99))

    return {
        "flow": name,
        "warmup": warmup,
        "n_runs": n_runs,
        "queries_per_run": n_queries,
        "total_measured_queries": int(len(total_arr)),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": float(np.mean(total_arr)),
        "min_ms": float(np.min(total_arr)),
        "max_ms": float(np.max(total_arr)),
        "sla_target_p95_ms": sla_p95_ms,
        "sla_pass": bool(p95 < sla_p95_ms),
        "per_run": per_run_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Latency SLA Benchmark")
    parser.add_argument("--index-dir", default="index/test_gallery")
    parser.add_argument("--queries", default="eval/handwritten_queries.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/phase5/latency_sla_results.json")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = ECommerceSearchEngine(index_dir=args.index_dir, config_path=args.config)

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)
    text_pool = [q["query"] for q in queries_raw]

    # Sample 200 valid catalog images in memory for image_to_image & cir_upload
    rng = np.random.default_rng(args.seed)
    records = engine.id_map
    sample_indices = rng.choice(len(records), size=min(args.n_queries, len(records)), replace=False)

    pil_pool: List[Image.Image] = []
    catalog_idx_pool: List[int] = []
    for idx in sample_indices:
        img_path = Path(records[int(idx)]["path"])
        if img_path.exists():
            with Image.open(img_path) as im:
                pil_pool.append(im.convert("RGB"))
            catalog_idx_pool.append(int(idx))
    if not pil_pool:
        raise RuntimeError("No valid images found in id_map for latency benchmark.")

    modifiers = [
        "đổi sang màu đen",
        "màu trắng cho nữ",
        "kiểu dáng thể thao màu xanh navy",
        "phiên bản màu đỏ",
    ]

    gpu_before = _query_gpu_telemetry()

    flows_results: Dict[str, Any] = {}
    flows_results["text_to_image"] = _benchmark_flow(
        name="text_to_image",
        fn=lambda i: engine.search_text(text_pool[i % len(text_pool)], top_k=10, mode="system"),
        device=engine.device,
        warmup=args.warmup,
        n_runs=args.n_runs,
        n_queries=args.n_queries,
    )
    flows_results["image_to_image"] = _benchmark_flow(
        name="image_to_image",
        fn=lambda i: engine.search_image(pil_pool[i % len(pil_pool)], top_k=10),
        device=engine.device,
        warmup=args.warmup,
        n_runs=args.n_runs,
        n_queries=args.n_queries,
    )
    flows_results["cir_catalog"] = _benchmark_flow(
        name="cir_catalog",
        fn=lambda i: engine.search_cir(
            catalog_idx_pool[i % len(catalog_idx_pool)],
            modifiers[i % len(modifiers)],
            top_k=10,
            mode="system",
            is_catalog_idx=True,
        ),
        device=engine.device,
        warmup=args.warmup,
        n_runs=args.n_runs,
        n_queries=args.n_queries,
    )
    flows_results["cir_upload"] = _benchmark_flow(
        name="cir_upload",
        fn=lambda i: engine.search_cir(
            pil_pool[i % len(pil_pool)],
            modifiers[i % len(modifiers)],
            top_k=10,
            mode="system",
            is_catalog_idx=False,
        ),
        device=engine.device,
        warmup=args.warmup,
        n_runs=args.n_runs,
        n_queries=args.n_queries,
    )

    gpu_after = _query_gpu_telemetry()

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "seed": args.seed,
            "device": engine.device,
            "dtype": str(next(engine.model.parameters()).dtype),
            "index_dir": args.index_dir,
            "gallery_size": int(len(engine.id_map)),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
        },
        "flows": flows_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved latency SLA results to {out_path}")
    for k, v in flows_results.items():
        print(
            f"  {k}: p50={v['p50_ms']:.2f} ms, p95={v['p95_ms']:.2f} ms, "
            f"p99={v['p99_ms']:.2f} ms -> PASS={v['sla_pass']}"
        )


if __name__ == "__main__":
    main()
