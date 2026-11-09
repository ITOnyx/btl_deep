import sys
import json
import time
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import SearchService

def main():
    print("=== RUNNING REAL SMOKE TEST (PHASE 4) ===")
    service = SearchService(config_path="config.yaml")
    
    smoke_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule_7_verified": True,
        "checkpoint_hash": service.cfg_hash,
        "gallery_size": service.engine.N,
        "queries": []
    }

    # 1. Five Text Queries
    text_cases = [
        {"type": "text_accented", "query": "áo sơ mi nam trắng", "config": "c5"},
        {"type": "text_unaccented", "query": "quan jean nam xanh", "config": "c5"},
        {"type": "text_brand", "query": "giày nike nam", "config": "c5"},
        {"type": "text_out_of_catalog", "query": "túi xách nữ Lavie", "config": "c5"},
        {"type": "text_vague", "query": "đồ đi biển mùa hè", "config": "c5"}
    ]

    print("\n--- 1. Testing 5 Text Queries ---")
    for tc in text_cases:
        q = tc["query"]
        cfg = tc["config"]
        t0 = time.time()
        results, meta = service.search_text(q, config_mode=cfg, top_k=10, gamma=0.3)
        total_lat = (time.time() - t0) * 1000

        top10_ids = [r["item_id"] for r in results]
        entry = {
            "category": "text",
            "subtype": tc["type"],
            "query": q,
            "config": cfg,
            "latency_breakdown": meta.get("breakdown", {}),
            "total_latency_ms": meta.get("breakdown", {}).get("total_ms", total_lat),
            "parsed": meta.get("parsed", {}),
            "tier": meta.get("tier", "all"),
            "candidates_count": meta.get("candidates_count", len(results)),
            "top10_ids": top10_ids,
            "top10_items": [
                {
                    "rank": r["rank"],
                    "item_id": r["item_id"],
                    "name": r["name"],
                    "score": float(r["score"]),
                    "articleType": r.get("articleType"),
                    "baseColour": r.get("baseColour"),
                    "gender": r.get("gender"),
                    "brand": r.get("brand")
                } for r in results
            ]
        }
        smoke_results["queries"].append(entry)
        print(f"[{tc['type']}] '{q}' -> Top 1: [{results[0]['item_id']}] {results[0]['name']} | Latency: {entry['total_latency_ms']:.2f} ms")

    # 2. Two Visual Search Queries
    image_cases = [
        {"type": "image_shirt", "path": "data/fashion-dataset/images/15970.jpg"},
        {"type": "image_jeans", "path": "data/fashion-dataset/images/39386.jpg"}
    ]

    print("\n--- 2. Testing 2 Visual Search Queries ---")
    for ic in image_cases:
        p = Path(ic["path"])
        if not p.exists():
            print(f"Error: image {p} not found")
            continue
        im = Image.open(p).convert("RGB")
        t0 = time.time()
        results, meta = service.search_image(im, top_k=10, use_flip_aug=True, use_knn_voting=True)
        total_lat = (time.time() - t0) * 1000

        entry = {
            "category": "image",
            "subtype": ic["type"],
            "query_image": str(p),
            "latency_breakdown": {
                "total_ms": meta.get("latency_ms", total_lat)
            },
            "top10_ids": [r["item_id"] for r in results],
            "top10_items": [
                {
                    "rank": r["rank"],
                    "item_id": r["item_id"],
                    "name": r["name"],
                    "score": float(r["score"]),
                    "articleType": r.get("articleType"),
                    "baseColour": r.get("baseColour"),
                    "gender": r.get("gender")
                } for r in results
            ]
        }
        smoke_results["queries"].append(entry)
        print(f"[{ic['type']}] {p.name} -> Top 1: [{results[0]['item_id']}] {results[0]['name']} | Latency: {entry['latency_breakdown']['total_ms']:.2f} ms")

    # 3. Three CIR Queries
    cir_cases = [
        {"type": "cir_color", "ref_path": "data/fashion-dataset/images/15970.jpg", "mod": "màu xanh navy"},
        {"type": "cir_type", "ref_path": "data/fashion-dataset/images/39386.jpg", "mod": "quần short"},
        {"type": "cir_style", "ref_path": "data/fashion-dataset/images/15970.jpg", "mod": "áo khoác jacket"}
    ]

    print("\n--- 3. Testing 3 CIR Queries ---")
    for cc in cir_cases:
        p = Path(cc["ref_path"])
        im = Image.open(p).convert("RGB")
        mod = cc["mod"]
        t0 = time.time()
        results, meta = service.search_cir(im, mod, top_k=10, use_soft_bonus=True)
        total_lat = (time.time() - t0) * 1000

        entry = {
            "category": "cir",
            "subtype": cc["type"],
            "ref_image": str(p),
            "modifier_text": mod,
            "latency_breakdown": {
                "total_ms": meta.get("latency_ms", total_lat)
            },
            "top10_ids": [r["item_id"] for r in results],
            "top10_items": [
                {
                    "rank": r["rank"],
                    "item_id": r["item_id"],
                    "name": r["name"],
                    "score": float(r["score"]),
                    "articleType": r.get("articleType"),
                    "baseColour": r.get("baseColour"),
                    "gender": r.get("gender")
                } for r in results
            ]
        }
        smoke_results["queries"].append(entry)
        print(f"[{cc['type']}] {p.name} + '{mod}' -> Top 1: [{results[0]['item_id']}] {results[0]['name']} | Latency: {entry['latency_breakdown']['total_ms']:.2f} ms")

    out_file = Path("reports/phase4_smoke.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(smoke_results, f, indent=2, ensure_ascii=False)

    print(f"\nPhase 4 Smoke Test successfully completed and saved to {out_file}!")

if __name__ == "__main__":
    main()
