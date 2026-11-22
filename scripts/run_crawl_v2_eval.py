"""
Đánh giá ngoài miền MỘT lần trên index/crawl_v2_gallery với bộ truy vấn đã đóng băng
(eval/crawl_v2_t2i_queries.json, eval/crawl_v2_cir_pairs.json; kiểm SHA-256 trước khi chạy).

So sánh mọi cấu hình của app trên cùng truy vấn:
  Text→Ảnh: model_only, cfg_v5_01, cfg_v5_09, cfg_v5_10, cfg_v5_11
  CIR (catalog, loại ảnh tham chiếu): cir_v1_01, cir_v1_05, cir_v1_06, cir_v1_07
Chỉ số: R@1/5/10, mAP@10 (AP chia cho min(số liên quan, 10)), độ chính xác @10 theo loại / giới tính / màu,
tỉ lệ sai lĩnh vực; bootstrap 1.000 lần (seed 42) cho chênh lệch so với cấu hình chính thức (cfg_v5_09, cir_v1_05).
Ground truth: is_item_relevant, chính sách giới tính hybrid (như đánh giá chính thức). Có kiểm tra giới tính.

Kết quả: reports/crawl_v2_eval/{summary.json, per_query.csv, report.md}
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine_service import CIR_CONFIGS, T2I_CONFIGS, AppSearchService  # noqa: E402
from src.eval.metrics import bootstrap_ci_diff, compute_average_precision, is_item_relevant, load_gender_gt_policy  # noqa: E402
from src.serve.taxonomy import get_taxonomy  # noqa: E402

# RUN_TAG: thư mục con cho các lần chạy lại sau sửa lỗi (lần chạy đóng băng ghi ở thư mục gốc)
OUT = ROOT / "reports" / "crawl_v2_eval" / os.environ.get("RUN_TAG", "")
T2I_FILE = ROOT / "eval" / "crawl_v2_t2i_queries.json"
CIR_FILE = ROOT / "eval" / "crawl_v2_cir_pairs.json"
SHA_FILE = ROOT / "eval" / "crawl_v2_eval_set.sha256"
K = 10
T2I_ORDER = ["model_only", "cfg_v5_01", "cfg_v5_09", "cfg_v5_10", "cfg_v5_11"]
CIR_ORDER = ["cir_v1_01", "cir_v1_05", "cir_v1_06", "cir_v1_07"]
T2I_BASE, CIR_BASE = "cfg_v5_09", "cir_v1_05"


def verify_frozen() -> None:
    for line in SHA_FILE.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        actual = hashlib.sha256((ROOT / "eval" / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{name} đã bị sửa sau khi đóng băng (SHA-256 lệch) — dừng.")


def score(results, spec, svc, strict, exclude_id=None):
    rel = [int(is_item_relevant(spec, r, "hybrid", strict)) for r in results[:K]]
    tot = sum(is_item_relevant(spec, m, "hybrid", strict) for m in svc.id_map
              if m["item_id"] != exclude_id)
    ap = compute_average_precision(rel, total_relevant=min(tot, K)) if tot else 0.0
    t_dom = get_taxonomy({"articleType": spec["articleType"][0]})[0]
    top = results[:K] or [{}]
    out = {
        "r1": float(any(rel[:1])), "r5": float(any(rel[:5])), "r10": float(any(rel[:10])), "ap10": ap,
        "type_p10": np.mean([r.get("articleType") in spec["articleType"] for r in top]),
        "dom_err10": np.mean([r.get("domain") != t_dom for r in top]),
        "n_rel": tot,
    }
    if spec.get("gender"):
        out["gender_p10"] = np.mean([r.get("gender") in (spec["gender"], "Unisex") for r in top])
    if spec.get("baseColour"):
        out["colour_p10"] = np.mean([r.get("baseColour") in spec["baseColour"] for r in top])
    return out


def agg(rows):
    keys = ["r1", "r5", "r10", "ap10", "type_p10", "gender_p10", "colour_p10", "dom_err10"]
    return {k: round(float(np.mean([r[k] for r in rows if k in r])), 4) for k in keys if any(k in r for r in rows)} | {
        "n": len(rows)}


def main() -> None:
    verify_frozen()
    t2i = json.load(open(T2I_FILE, encoding="utf-8"))["queries"]
    cir = json.load(open(CIR_FILE, encoding="utf-8"))["pairs"]
    policy = load_gender_gt_policy(ROOT / "eval" / "gender_gt_policy.json")
    strict = set(policy["policies"]["hybrid"].get("strict_categories", []))
    svc = AppSearchService(index_dir="index/crawl_v2_gallery")
    assert set(T2I_ORDER) == set(T2I_CONFIGS) and set(CIR_ORDER) == set(CIR_CONFIGS)
    for _ in range(5):
        svc.search_text_to_image("áo thun nam", "cfg_v5_09", K)

    per = defaultdict(dict)  # (stream, cfg) -> qid -> metrics
    lat = defaultdict(list)
    rows_csv = []
    for cfg in T2I_ORDER:
        for q in t2i:
            res, _, ms = svc.search_text_to_image(q["query"], cfg, K)
            m = score(res, q["spec"], svc, strict)
            per[("t2i", cfg)][q["id"]] = m | {"kind": q["kind"], "unacc": not q["has_diacritics"]}
            lat[("t2i", cfg)].append(ms)
            rows_csv.append(["t2i", cfg, q["id"], q["kind"], q["query"], json.dumps(q["spec"], ensure_ascii=False),
                             m["r10"], round(m["ap10"], 4), round(m["type_p10"], 2), m.get("gender_p10", ""),
                             m.get("colour_p10", ""), round(m["dom_err10"], 2), round(ms, 1),
                             ";".join(f"{r['articleType']}/{r['gender']}/{r['baseColour']}" for r in res[:3])])
        print("t2i", cfg, agg(list(per[("t2i", cfg)].values())), flush=True)
    for cfg in CIR_ORDER:
        for p in cir:
            spec = {"articleType": [p["target_attrs"]["articleType"]], "gender": p["target_attrs"]["gender"],
                    "baseColour": [p["target_attrs"]["baseColour"]]}
            res, _, ms = svc.search_cir_catalog(p["ref_product_id"], p["modifier_vi"], cfg, K, exclude_reference=True)
            m = score(res, spec, svc, strict, exclude_id=p["ref_product_id"])
            per[("cir", cfg)][p["id"]] = m | {"kind": p["changed_attr"], "unacc": False}
            lat[("cir", cfg)].append(ms)
            rows_csv.append(["cir", cfg, p["id"], p["changed_attr"], f"{p['ref_product_id']} + {p['modifier_vi']}",
                             json.dumps(spec, ensure_ascii=False), m["r10"], round(m["ap10"], 4),
                             round(m["type_p10"], 2), m.get("gender_p10", ""), m.get("colour_p10", ""),
                             round(m["dom_err10"], 2), round(ms, 1),
                             ";".join(f"{r['articleType']}/{r['gender']}/{r['baseColour']}" for r in res[:3])])
        print("cir", cfg, agg(list(per[("cir", cfg)].values())), flush=True)

    summary = {"gallery": "crawl_v2_gallery", "n_gallery": len(svc.id_map), "k": K,
               "eval_set_sha256": SHA_FILE.read_text(encoding="utf-8"), "configs": {}, "vs_baseline": {}}
    for (stream, cfg), d in per.items():
        rows = list(d.values())
        slices = {"all": agg(rows)}
        for kind in sorted({r["kind"] for r in rows}):
            slices[kind] = agg([r for r in rows if r["kind"] == kind])
        if stream == "t2i":
            slices["unaccented"] = agg([r for r in rows if r["unacc"]])
        slices["latency_ms"] = {"p50": round(float(np.percentile(lat[(stream, cfg)], 50)), 1),
                                "p95": round(float(np.percentile(lat[(stream, cfg)], 95)), 1)}
        summary["configs"][f"{stream}:{cfg}"] = slices
        base = T2I_BASE if stream == "t2i" else CIR_BASE
        if cfg != base:
            b = per[(stream, base)]
            ids = sorted(d)
            summary["vs_baseline"][f"{stream}:{cfg} - {base}"] = {
                metric: {k: round(v, 4) for k, v in bootstrap_ci_diff([b[i][metric] for i in ids],
                                                                     [d[i][metric] for i in ids]).items()}
                for metric in ("r10", "ap10")}

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(OUT / "per_query.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "config", "id", "kind", "query", "spec", "R@10", "AP@10", "type_P@10", "gender_P@10",
                    "colour_P@10", "dom_err@10", "ms", "top3"])
        w.writerows(rows_csv)
    print("Xong ->", OUT, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f}s")
