import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from PIL import Image

from app.engine import SearchService

def main():
    print("=== EVALUATING 20 CIR COMPOSITION QUERIES ===")
    service = SearchService(config_path="config.yaml")
    df = service.engine.df

    # 20 CIR Queries (10 color modifications, 10 category/style modifications)
    # We select real valid items from df as reference images
    cir_test_specs = [
        # --- 10 Color Modifications ---
        {"ref_id": 15970, "target_type": "Tshirts", "target_color": "Navy Blue", "mod": "màu xanh navy", "kind": "color"},
        {"ref_id": 15970, "target_type": "Tshirts", "target_color": "Black", "mod": "màu đen", "kind": "color"},
        {"ref_id": 15970, "target_type": "Tshirts", "target_color": "White", "mod": "màu trắng", "kind": "color"},
        {"ref_id": 15970, "target_type": "Tshirts", "target_color": "Red", "mod": "màu đỏ", "kind": "color"},
        {"ref_id": 39386, "target_type": "Casual Shoes", "target_color": "Navy Blue", "mod": "màu xanh navy", "kind": "color"},
        {"ref_id": 39386, "target_type": "Casual Shoes", "target_color": "Red", "mod": "màu đỏ", "kind": "color"},
        {"ref_id": 39386, "target_type": "Casual Shoes", "target_color": "White", "mod": "màu trắng", "kind": "color"},
        {"ref_id": 1855, "target_type": "Handbags", "target_color": "Black", "mod": "màu đen", "kind": "color"},
        {"ref_id": 1855, "target_type": "Handbags", "target_color": "Red", "mod": "màu đỏ", "kind": "color"},
        {"ref_id": 10866, "target_type": "Shirts", "target_color": "Blue", "mod": "màu xanh dương", "kind": "color"},

        # --- 10 Type / Style Modifications ---
        {"ref_id": 15970, "target_type": "Shirts", "target_color": None, "mod": "áo sơ mi có cổ", "kind": "type"},
        {"ref_id": 15970, "target_type": "Jackets", "target_color": None, "mod": "áo khoác jacket", "kind": "type"},
        {"ref_id": 15970, "target_type": "Sweaters", "target_color": None, "mod": "áo len sweater dài tay", "kind": "type"},
        {"ref_id": 39386, "target_type": "Sports Shoes", "target_color": None, "mod": "giày thể thao chạy bộ", "kind": "type"},
        {"ref_id": 39386, "target_type": "Sandals", "target_color": None, "mod": "dép sandal quai hậu", "kind": "type"},
        {"ref_id": 39386, "target_type": "Formal Shoes", "target_color": None, "mod": "giày tây công sở sang trọng", "kind": "type"},
        {"ref_id": 1855, "target_type": "Backpacks", "target_color": None, "mod": "balo đeo vai", "kind": "type"},
        {"ref_id": 1855, "target_type": "Clutches", "target_color": None, "mod": "ví cầm tay clutch dạ hội", "kind": "type"},
        {"ref_id": 10866, "target_type": "Tshirts", "target_color": None, "mod": "áo thun cộc tay cổ tròn", "kind": "type"},
        {"ref_id": 10866, "target_type": "Blazers", "target_color": None, "mod": "áo vest blazer sang trọng", "kind": "type"}
    ]

    records = []
    
    for i, spec in enumerate(cir_test_specs, 1):
        ref_id = str(spec["ref_id"])
        rows = df[df["item_id"].astype(str) == ref_id]
        if len(rows) == 0:
            print(f"Skipping {ref_id}, not found")
            continue
        ref_row = rows.iloc[0]
        ref_img_path = Path(f"data/fashion-dataset/images/{ref_id}.jpg")
        if not ref_img_path.exists():
            ref_img_path = Path(ref_row["path"])

        im = Image.open(ref_img_path).convert("RGB")
        mod_text = spec["mod"]
        
        # Run search_cir
        t0 = time.time()
        results, meta = service.search_cir(im, mod_text, top_k=10, use_soft_bonus=False)
        lat = (time.time() - t0) * 1000

        # Check metadata alignment
        tgt_col = spec["target_color"]
        tgt_type = spec["target_type"]
        kind = spec["kind"]

        matches_top1 = 0
        matches_top5 = 0
        matches_top10 = 0

        for r_idx, item in enumerate(results[:10]):
            is_match = False
            if kind == "color":
                is_match = (item.get("baseColour") == tgt_col)
            elif kind == "type":
                is_match = (item.get("articleType") == tgt_type)
            
            if is_match:
                if r_idx == 0:
                    matches_top1 += 1
                if r_idx < 5:
                    matches_top5 += 1
                if r_idx < 10:
                    matches_top10 += 1

        acc_top1 = 1.0 if matches_top1 > 0 else 0.0
        acc_top5 = matches_top5 / 5.0
        acc_top10 = matches_top10 / 10.0
        hit_top10 = 1.0 if matches_top10 > 0 else 0.0

        record = {
            "query_idx": i,
            "ref_id": ref_id,
            "ref_name": ref_row["name"],
            "ref_articleType": ref_row["articleType"],
            "ref_baseColour": ref_row["baseColour"],
            "modifier_text": mod_text,
            "kind": kind,
            "target_expected": tgt_col if kind == "color" else tgt_type,
            "latency_ms": lat,
            "hit_top1": acc_top1,
            "precision_top5": acc_top5,
            "precision_top10": acc_top10,
            "hit_top10": hit_top10,
            "top1_retrieved": f"[{results[0]['item_id']}] {results[0]['name']} ({results[0]['articleType']}, {results[0]['baseColour']})" if results else "None"
        }
        records.append(record)
        print(f"Q{i:02d} [{kind:5s}] Ref: {ref_id} ({ref_row['articleType']}, {ref_row['baseColour']}) + '{mod_text}' -> Top1: {record['top1_retrieved']} | P@10={acc_top10:.2f}, Hit@10={hit_top10:.0f} ({lat:.1f}ms)")

    # Aggregate metrics
    df_rec = pd.DataFrame(records)
    
    summary = {
        "total_queries": len(records),
        "overall": {
            "hit_at_1": float(df_rec["hit_top1"].mean()),
            "precision_at_5": float(df_rec["precision_top5"].mean()),
            "precision_at_10": float(df_rec["precision_top10"].mean()),
            "hit_at_10": float(df_rec["hit_top10"].mean()),
            "mean_latency_ms": float(df_rec["latency_ms"].mean())
        },
        "color_queries": {
            "count": int((df_rec["kind"] == "color").sum()),
            "hit_at_1": float(df_rec[df_rec["kind"] == "color"]["hit_top1"].mean()),
            "precision_at_10": float(df_rec[df_rec["kind"] == "color"]["precision_top10"].mean()),
            "hit_at_10": float(df_rec[df_rec["kind"] == "color"]["hit_top10"].mean())
        },
        "type_queries": {
            "count": int((df_rec["kind"] == "type").sum()),
            "hit_at_1": float(df_rec[df_rec["kind"] == "type"]["hit_top1"].mean()),
            "precision_at_10": float(df_rec[df_rec["kind"] == "type"]["precision_top10"].mean()),
            "hit_at_10": float(df_rec[df_rec["kind"] == "type"]["hit_top10"].mean())
        },
        "records": records
    }

    out_file = Path("reports/phase4_cir_20queries_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== CIR SUMMARY RESULTS ===")
    print(f"Overall Hit@1:  {summary['overall']['hit_at_1']:.4f}")
    print(f"Overall Hit@10: {summary['overall']['hit_at_10']:.4f}")
    print(f"Overall P@10:   {summary['overall']['precision_at_10']:.4f}")
    print(f"Color Hit@10:   {summary['color_queries']['hit_at_10']:.4f} (P@10={summary['color_queries']['precision_at_10']:.4f})")
    print(f"Type Hit@10:    {summary['type_queries']['hit_at_10']:.4f} (P@10={summary['type_queries']['precision_at_10']:.4f})")
    print(f"Mean Latency:   {summary['overall']['mean_latency_ms']:.2f} ms")

if __name__ == "__main__":
    main()
