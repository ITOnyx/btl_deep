"""
Phase 5a/5b - Controlled E2E latency benchmark v3 (cuda, float16) via app.engine.SearchService.

Configurations (all measured, interleaved):
  text_to_image | image_to_image_fast | image_to_image_accurate | cir_catalog | cir_upload
Protocol: latency.min_runs independent runs x eval.n_latency_queries distinct queries,
          latency.warmup_queries warm-ups per config, torch.cuda.synchronize around every query,
          Latin-square interleaved order, 5 s cooldown between runs (3 s between configs),
          nvidia-smi sampled every 1 s per (config, run).
Output  : reports/phase4/phase4_latency_e2e_v3.json  -> top-level key phases[<tag>]; v2 is never touched.
          A tag that already exists is refused (no silent overwrite).
Usage   : python src/benchmark_e2e_latency_v3.py --tag 5a_baseline
"""
import sys, json, time, argparse, threading, subprocess, hashlib, platform
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from app.engine import SearchService

OUT = ROOT / "reports/phase4/phase4_latency_e2e_v3.json"
FLOW_OF = {"text_to_image": "text_to_image", "image_to_image_fast": "image_to_image",
           "image_to_image_accurate": "image_to_image", "cir_catalog": "cir_catalog", "cir_upload": "cir_upload"}
STAGES = ["decode_ms", "preprocess_ms", "vit_forward_ms", "gallery_search_ms", "rerank_ms"]


class GpuMonitor:
    def __init__(self, interval_s=1.0):
        self.interval_s, self.samples, self.running, self.th = interval_s, [], False, None

    def _loop(self):
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,memory.used,power.draw", "--format=csv,noheader,nounits"]
        while self.running:
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
                p = [float(x) if x.strip() not in ("[N/A]", "") else float("nan") for x in out.split(",")]
                self.samples.append({"temp_c": p[0], "clock_mhz": p[1], "mem_mb": p[2], "power_w": p[3]})
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.samples, self.running = [], True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    def stop(self):
        self.running = False
        if self.th: self.th.join(timeout=3)
        return list(self.samples)


def summarize(lat, bd, gpu, sla_ms):
    a = np.asarray(lat, np.float64)
    def g(k, f):
        v = [s[k] for s in gpu if s[k] == s[k]]
        return round(float(f(v)), 1) if v else None
    out = {"n": int(a.size), "p50_ms": round(float(np.percentile(a, 50)), 2),
           "p95_ms": round(float(np.percentile(a, 95)), 2), "p99_ms": round(float(np.percentile(a, 99)), 2),
           "mean_ms": round(float(a.mean()), 2), "max_ms": round(float(a.max()), 2),
           "n_over_sla": int((a > sla_ms).sum()), "n_over_500ms": int((a > 500).sum()),
           "sla_ms": sla_ms, "p95_pass": bool(np.percentile(a, 95) < sla_ms),
           "gpu": {"samples": len(gpu), "temp_mean": g("temp_c", np.mean), "temp_max": g("temp_c", np.max),
                   "clock_mean": g("clock_mhz", np.mean), "clock_min": g("clock_mhz", np.min),
                   "power_mean_w": g("power_w", np.mean), "mem_mean_mb": g("mem_mb", np.mean)}}
    if bd:
        out["breakdown_ms"] = {s: {"mean": round(float(np.mean([b[s] for b in bd])), 2),
                                   "p95": round(float(np.percentile([b[s] for b in bd], 95)), 2)} for s in STAGES}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu-preprocess", action="store_true", help="reserved for Phase 5b")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    n_q = int(cfg["eval"]["n_latency_queries"]); runs = int(cfg["latency"]["min_runs"])
    n_warm = int(cfg["latency"]["warmup_queries"]); seed = int(cfg["eval"]["seed"])
    flows = cfg["latency"]["flows"]
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"phases": {}}
    if args.tag in existing["phases"]:
        sys.exit(f"tag {args.tag} already in {OUT.name}; refusing to overwrite")

    svc = SearchService(config_path="config.yaml")
    assert svc.engine.device == "cuda", "latency benchmark requires cuda"
    df = svc.engine.df
    dev = torch.cuda.get_device_properties(0)

    texts = []
    def add(t):
        t = (t or "").strip()
        if t and t not in texts: texts.append(t)
    for q in json.load(open(ROOT / "eval/handwritten_queries.json", encoding="utf-8")): add(q.get("query"))
    for q in json.load(open(ROOT / "eval/testset_v1.json", encoding="utf-8")).get("q2_attribute_queries", []): add(q.get("query_text"))
    for q in json.load(open(ROOT / "eval/heldout_queries.json", encoding="utf-8")): add(q.get("query"))
    q5 = json.load(open(ROOT / "eval/q5_brand.json", encoding="utf-8"))
    for q in (q5.get("queries", []) if isinstance(q5, dict) else q5):
        add(q.get("query_text", q.get("query", "")) if isinstance(q, dict) else str(q))
    texts = texts[:n_q + n_warm]
    assert len(texts) >= n_q + n_warm, f"only {len(texts)} distinct text queries"

    img_dir = ROOT / "data/fashion-dataset/images"
    rng = np.random.RandomState(20261109)
    ids = rng.permutation(df["item_id"].astype(str).values)
    imgs = [str(img_dir / f"{s}.jpg") for s in ids if (img_dir / f"{s}.jpg").exists()]
    img_q, cir_img = imgs[: n_q + n_warm], imgs[::-1][: n_q + n_warm]
    mods = ["màu đen", "màu đỏ", "màu trắng", "màu xanh navy", "màu xám", "phong cách thể thao", "dáng rộng",
            "chất liệu da", "thời trang mùa hè", "phong cách công sở", "họa tiết kẻ sọc", "phom ôm năng động",
            "phong cách dạo phố", "cổ chữ v", "tay dài ấm áp", "đơn giản thanh lịch"]
    cir_q = [(cir_img[i], f"{mods[i % len(mods)]} biến thể {i+1}") for i in range(n_q + n_warm)]

    def call(name, i):
        if name == "text_to_image": return svc.search_text(texts[i], config_mode="c5", top_k=10)
        if name == "image_to_image_fast": return svc.search_image(img_q[i], top_k=10, mode="fast")
        if name == "image_to_image_accurate": return svc.search_image(img_q[i], top_k=10, mode="accurate")
        if name == "cir_catalog": return svc.search_cir(cir_q[i][0], cir_q[i][1], top_k=10, force_upload=False)
        if name == "cir_upload": return svc.search_cir(cir_q[i][0], cir_q[i][1], top_k=10, force_upload=True)

    names = list(FLOW_OF)
    print(f"warm-up {n_warm} per config ...")
    for nm in names:
        for i in range(n_warm):
            call(nm, i)
    torch.cuda.synchronize(); time.sleep(5)

    result = {nm: [] for nm in names}
    for r in range(runs):
        order = names[r % len(names):] + names[: r % len(names)]  # Latin-square rotation
        print(f"\nRUN {r+1}/{runs} order={order}", flush=True)
        for nm in order:
            mon = GpuMonitor(); mon.start()
            lat, bd = [], []
            for i in range(n_warm, n_warm + n_q):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                _, meta = call(nm, i)
                torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000.0)
                b = meta.get("breakdown") if isinstance(meta, dict) else None
                if b and all(s in b for s in STAGES): bd.append(b)
            st = summarize(lat, bd, mon.stop(), float(flows[FLOW_OF[nm]]["sla_ms"]))
            st["run"] = r + 1
            result[nm].append(st)
            print(f"  {nm:24s} p50={st['p50_ms']:7.2f} p95={st['p95_ms']:7.2f} p99={st['p99_ms']:7.2f} "
                  f"max={st['max_ms']:7.2f} | {st['gpu']['temp_mean']}C {st['gpu']['clock_mean']}MHz", flush=True)
            time.sleep(3)
        time.sleep(5)

    cfg_bytes = (ROOT / "config.yaml").read_bytes()
    phase = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tag": args.tag,
        "hardware": {"gpu": dev.name, "vram_mb": dev.total_memory // (1024 * 1024)},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "cuda": torch.version.cuda},
        "seed": seed, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "serving": cfg["serving"], "device": "cuda", "precision": "float16",
        "protocol": {"runs": runs, "queries_per_run": n_q, "warmup_per_config": n_warm,
                     "cooldown_between_runs_s": 5, "cooldown_between_configs_s": 3,
                     "order": "Latin-square rotation across runs", "sync": "torch.cuda.synchronize before/after each query",
                     "sla_metric": cfg["latency"]["sla_metric"]},
        "flows_sla_ms": {k: v["sla_ms"] for k, v in flows.items()},
        "results": result,
        "note": "FAISS is not used in the serving path: gallery search is an exact fp16 torch matmul on GPU "
                "(stage 'gallery_search_ms'); rerank_ms = argsort/kNN-voting on CPU.",
    }
    existing["phases"][args.tag] = phase
    OUT.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
