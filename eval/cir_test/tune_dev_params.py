"""
Parameter Tuning & Ablation strictly on the 60-case Dev Suite (eval/cir_test/cir_dev_6groups.csv).
NEVER touches cir_heldout.csv!
"""
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image
from app.engine import SearchService

def run_dev_eval(
    service: SearchService,
    df_specs: pd.DataFrame,
    dynamic_category_anchor: bool,
    anchor_mode: str,
    modifier_weight: float,
    candidate_pool_k: Optional[int]
) -> Dict[str, Any]:
    engine = service.engine
    cir_dir = ROOT_DIR / "eval" / "cir_test"
    records = []

    for _, row in df_specs.iterrows():
        case_id = row["case_id"]
        cat = row["category"]
        ref_id = str(row["ref_id"])
        mod_text = str(row["modifier_text"]) if pd.notna(row["modifier_text"]) else ""
        flow = str(row["flow"])
        tgt_t = row["target_type"] if pd.notna(row["target_type"]) else None
        tgt_c = row["target_color"] if pd.notna(row["target_color"]) else None
        tgt_g = row["target_gender"] if pd.notna(row["target_gender"]) else None

        if flow == "cir_upload":
            img_path = cir_dir / ref_id
            if not img_path.exists():
                img_path = Path("data/fashion-dataset/images") / f"{ref_id}.jpg"
            ref_img_input = Image.open(img_path).convert("RGB")
        else:
            img_path = Path("data/fashion-dataset/images") / f"{ref_id}.jpg"
            ref_img_input = str(img_path) if img_path.exists() else str(engine.df[engine.df["item_id"].astype(str) == str(ref_id)].iloc[0]["path"])

        t0 = time.perf_counter()
        results, meta = service.search_cir(
            ref_image=ref_img_input,
            modifier_text=mod_text,
            top_k=10,
            use_soft_bonus=True,
            ref_id=ref_id if flow == "cir_catalog" else None,
            force_upload=(flow == "cir_upload"),
            dynamic_category_anchor=dynamic_category_anchor,
            anchor_mode=anchor_mode,
            modifier_weight=modifier_weight,
            candidate_pool_k=candidate_pool_k
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0

        type_matches = 0
        full_matches = 0
        top1_full = False
        is_crit = False
        if results:
            top1 = results[0]
            t1_ok = (top1.get("articleType") == tgt_t) if tgt_t else True
            a1_ok = True
            if tgt_c and top1.get("baseColour") != tgt_c:
                a1_ok = False
            if tgt_g and top1.get("gender") != tgt_g:
                a1_ok = False
            top1_full = t1_ok and a1_ok
            if (tgt_t and not t1_ok) or (tgt_c and top1.get("baseColour") != tgt_c):
                is_crit = True
            for it in results[:10]:
                t_ok = (it.get("articleType") == tgt_t) if tgt_t else True
                c_ok = (it.get("baseColour") == tgt_c) if tgt_c else True
                g_ok = (it.get("gender") == tgt_g) if tgt_g else True
                if t_ok:
                    type_matches += 1
                if t_ok and c_ok and g_ok:
                    full_matches += 1
        else:
            is_crit = True

        records.append({
            "case_id": case_id,
            "category": cat,
            "type_ret": type_matches / 10.0,
            "hit10": 1.0 if full_matches > 0 else 0.0,
            "hit1": 1.0 if top1_full else 0.0,
            "crit": 1.0 if is_crit else 0.0,
            "lat": lat_ms,
            "shortage": meta.get("anchor_shortage_notice"),
            "fallback_soft": meta.get("anchor_fallback_soft"),
            "exact_cnt": meta.get("exact_match_count"),
            "top1": f"{results[0]['articleType']}/{results[0]['baseColour']}" if results else "EMPTY",
            "tgt": f"{tgt_t}/{tgt_c}",
            "mod": mod_text,
            "intent": meta.get("intent_state")
        })

    df_r = pd.DataFrame(records)
    return {
        "hit10": float(df_r["hit10"].mean()),
        "hit1": float(df_r["hit1"].mean()),
        "type_ret": float(df_r["type_ret"].mean()),
        "crit": float(df_r["crit"].mean()),
        "p50": float(df_r["lat"].median()),
        "p95": float(df_r["lat"].quantile(0.95)),
        "shortage_cnt": int(df_r["shortage"].notna().sum()),
        "soft_fb_cnt": int(df_r["fallback_soft"].sum()),
        "by_cat": df_r.groupby("category")[["hit10", "hit1", "type_ret", "crit"]].mean().to_dict("index"),
        "failures": df_r[df_r["hit10"] == 0.0][["case_id", "category", "mod", "tgt", "top1", "intent", "exact_cnt"]].to_dict("records")
    }

def main():
    service = SearchService("config.yaml")
    df_dev = pd.read_csv(ROOT_DIR / "eval" / "cir_test" / "cir_dev_6groups.csv")

    configs = [
        ("baseline_no_anchor_lam1.0", False, "off", 1.0, None),
        ("anchor_auto_pool200_lam1.0", True, "auto", 1.0, 200),
        ("anchor_auto_pool500_lam1.0", True, "auto", 1.0, 500),
        ("anchor_auto_poolNone_lam1.0", True, "auto", 1.0, None),
        ("anchor_auto_pool200_lam0.5", True, "auto", 0.5, 200),
        ("anchor_auto_pool500_lam0.5", True, "auto", 0.5, 500),
        ("anchor_auto_poolNone_lam0.5", True, "auto", 0.5, None),
        ("anchor_auto_poolNone_lam0.8", True, "auto", 0.8, None),
        ("anchor_auto_poolNone_lam0.2", True, "auto", 0.2, None),
    ]

    print(f"\n=== TUNING ON 60-CASE DEV SUITE ===")
    print(f"{'Config':30s} | {'Hit@10':7s} | {'Hit@1':7s} | {'TypeRet':7s} | {'CritErr':7s} | {'p50':6s} | {'p95':6s} | {'Short':5s}")
    print("-" * 98)
    best_res = None
    all_out = {}
    for name, dca, am, lam, pool in configs:
        res = run_dev_eval(service, df_dev, dca, am, lam, pool)
        all_out[name] = res
        print(f"{name:30s} | {res['hit10']*100:6.1f}% | {res['hit1']*100:6.1f}% | {res['type_ret']*100:6.1f}% | {res['crit']*100:6.1f}% | {res['p50']:5.1f}  | {res['p95']:5.1f}  | {res['shortage_cnt']:5d}")

    print("\n--- Failed cases in anchor_auto_pool500_lam0.5 vs poolNone_lam0.8 ---")
    for cfg_name in ["anchor_auto_pool200_lam0.5", "anchor_auto_poolNone_lam0.5", "anchor_auto_poolNone_lam0.8"]:
        print(f"\n[{cfg_name}] Failures ({len(all_out[cfg_name]['failures'])}):")
        for f in all_out[cfg_name]["failures"]:
            print(" ", f)

    with open(ROOT_DIR / "eval" / "cir_test" / "results" / "dev_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
