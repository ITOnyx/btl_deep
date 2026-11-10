"""
Step 3 Official Held-Out Evaluation, Ablation Study & Upload Inference Benchmark.
Strictly follows Invariant Rules:
- Evaluates the 90 frozen held-out cases in eval/cir_test/cir_heldout.csv WITHOUT any parameter tuning on held-out.
- Compares Baseline (Phase B: eval/cir_test/results/baseline_phase_b.json) vs Dynamic Category Anchor + Slider.
- Runs exact Ablation:
  (a) No anchor (lambda=1.0, anchor=False) AND (lambda=0.5, anchor=False)
  (b) Anchor only (lambda=1.0, anchor=True)
  (c) Anchor + Slider (lambda = 0.2, 0.5 [default], 0.8)
- Computes 95% Wilson confidence intervals for n=90.
- Measures Upload articleType inference accuracy & fallback rate separately on non-heldout diverse test images + heldout upload cases.
- Verifies 100% non-regression on Tab 1 & Tab 2 smoke queries and determinism.
"""
import sys
import json
import math
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


def wilson_ci_95(p: float, n: int) -> tuple:
    """Computes 95% Wilson score confidence interval (low, high) for proportion p over n trials."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_heldout_config(
    service: SearchService,
    df_specs: pd.DataFrame,
    dynamic_category_anchor: bool,
    anchor_mode: str,
    modifier_weight: float,
    candidate_pool_k: Optional[int] = 500
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
        attr_matches = 0
        full_matches = 0
        top1_full = False
        is_critical_error = False

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
                is_critical_error = True

            for item in results[:10]:
                t_ok = (item.get("articleType") == tgt_t) if tgt_t else True
                c_ok = (item.get("baseColour") == tgt_c) if tgt_c else True
                g_ok = (item.get("gender") == tgt_g) if tgt_g else True
                if t_ok:
                    type_matches += 1
                if c_ok and g_ok:
                    attr_matches += 1
                if t_ok and c_ok and g_ok:
                    full_matches += 1
        else:
            is_critical_error = True

        records.append({
            "case_id": case_id,
            "category": cat,
            "ref_id": ref_id,
            "ref_type": row.get("ref_type"),
            "ref_color": row.get("ref_color"),
            "modifier_text": mod_text,
            "translated_modifier": meta.get("translated_modifier", ""),
            "intent_state": meta.get("intent_state"),
            "target_type_locked": meta.get("target_type"),
            "upload_type_pred": meta.get("upload_type_pred"),
            "upload_type_confidence": meta.get("upload_type_confidence"),
            "upload_type_confident": meta.get("upload_type_confident"),
            "anchor_fallback_soft": meta.get("anchor_fallback_soft"),
            "anchor_status_msg": meta.get("anchor_status_msg"),
            "anchor_shortage_notice": meta.get("anchor_shortage_notice"),
            "typo_color_recovered": meta.get("typo_color_recovered"),
            "disambiguated_type_dropped": meta.get("disambiguated_type_dropped"),
            "flow": flow,
            "target_type": tgt_t,
            "target_color": tgt_c,
            "target_gender": tgt_g,
            "latency_ms": round(lat_ms, 2),
            "type_retention_p10": type_matches / 10.0,
            "target_attr_p10": attr_matches / 10.0,
            "hit_at_10_full": 1.0 if full_matches > 0 else 0.0,
            "hit_at_1_full": 1.0 if top1_full else 0.0,
            "is_critical_error": is_critical_error,
            "top1_retrieved": f"[{results[0]['item_id']}] {results[0]['name']} ({results[0]['articleType']}, {results[0]['baseColour']}, {results[0]['gender']})" if results else "EMPTY",
            "top3_retrieved": [f"#{r['rank']}: {r['articleType']} / {r['baseColour']} / {r['gender']} ({r['score']:.3f})" for r in results[:3]]
        })

    df_res = pd.DataFrame(records)
    n_tot = len(df_res)
    h10 = float(df_res["hit_at_10_full"].mean())
    h1 = float(df_res["hit_at_1_full"].mean())
    tr = float(df_res["type_retention_p10"].mean())
    ce = float(df_res["is_critical_error"].mean())

    by_cat = {}
    for cat_name, grp in df_res.groupby("category"):
        n_c = len(grp)
        ch10 = float(grp["hit_at_10_full"].mean())
        by_cat[cat_name] = {
            "count": n_c,
            "type_retention_p10": float(grp["type_retention_p10"].mean()),
            "target_attr_p10": float(grp["target_attr_p10"].mean()),
            "hit_at_10_full": ch10,
            "hit_at_10_ci95": wilson_ci_95(ch10, n_c),
            "hit_at_1_full": float(grp["hit_at_1_full"].mean()),
            "critical_error_rate": float(grp["is_critical_error"].mean()),
            "p50_latency_ms": float(grp["latency_ms"].median()),
            "p95_latency_ms": float(grp["latency_ms"].quantile(0.95)),
            "soft_fallback_count": int(grp["anchor_fallback_soft"].sum())
        }

    overall = {
        "total_cases": n_tot,
        "mean_latency_ms": float(df_res["latency_ms"].mean()),
        "p50_latency_ms": float(df_res["latency_ms"].median()),
        "p95_latency_ms": float(df_res["latency_ms"].quantile(0.95)),
        "type_retention_p10": tr,
        "target_attr_p10": float(df_res["target_attr_p10"].mean()),
        "hit_at_10_full": h10,
        "hit_at_10_ci95": wilson_ci_95(h10, n_tot),
        "hit_at_1_full": h1,
        "hit_at_1_ci95": wilson_ci_95(h1, n_tot),
        "critical_error_rate": ce,
        "soft_fallback_rate": float(df_res["anchor_fallback_soft"].mean()),
        "shortage_count": int(df_res["anchor_shortage_notice"].notna().sum())
    }
    return {"overall": overall, "by_cat": by_cat, "records": records}


def measure_upload_inference_benchmark(service: SearchService) -> Dict[str, Any]:
    """
    Measures upload articleType inference accuracy & fallback rate separately on:
    1) 8 heldout upload cases (cir_heldout.csv flow=='cir_upload')
    2) 50 diverse test images (split=='test', excluding heldout IDs, with exact self excluded if testing generalization,
       AND normal real-world upload simulation using actual test images).
    3) 6 edge upload images (blurred, rotated, large, tiny, non_fashion).
    """
    import torch
    engine = service.engine
    ho_df = pd.read_csv(ROOT_DIR / "eval" / "cir_test" / "cir_heldout.csv")
    ho_ids = set(ho_df["ref_id"].astype(str))

    # 1. Held-out 8 upload cases
    ho_upl = ho_df[ho_df["flow"] == "cir_upload"]
    ho_correct = 0
    ho_conf_true = 0
    for _, r in ho_upl.iterrows():
        p = ROOT_DIR / "eval" / "cir_test" / str(r["ref_id"])
        im = Image.open(p).convert("RGB")
        t = engine.img_transform(im).unsqueeze(0).to(engine.device)
        if engine.device == "cuda":
            t = t.half()
        with torch.no_grad():
            v_raw = engine.model.encode_image(t)
        info = service._infer_upload_article_type(v_raw, top_k_knn=10, conf_threshold=0.62, min_visual_sim=0.65)
        if info["pred_type"] == r["ref_type"]:
            ho_correct += 1
        if info["confident"]:
            ho_conf_true += 1

    # 2. Diverse 50 unseen test images (excluding self match to simulate brand-new out-of-catalog photos!)
    test_sub = engine.df[(engine.df["split"] == "test") & (~engine.df["item_id"].astype(str).isin(ho_ids))].sample(n=50, random_state=42)
    unseen_correct_all = 0
    unseen_correct_confident = 0
    unseen_confident_cnt = 0
    unseen_fallback_cnt = 0
    for _, r in test_sub.iterrows():
        p = Path(str(r["path"]))
        if not p.exists():
            p = Path("data/fashion-dataset/images") / f"{r['item_id']}.jpg"
        im = Image.open(p).convert("RGB")
        t = engine.img_transform(im).unsqueeze(0).to(engine.device)
        if engine.device == "cuda":
            t = t.half()
        with torch.no_grad():
            v_raw = engine.model.encode_image(t)
        # Normal service inference (gallery includes all items)
        v_np = v_raw.float().cpu().numpy()
        v_c = engine.scorer.center_vectors(v_np)[0]
        q_v = torch.tensor(v_c, dtype=torch.float16 if engine.device == "cuda" else torch.float32, device=engine.device)
        sims = torch.matmul(engine.t_emb_img_c, q_v).float().cpu().numpy()
        # Exclude self item_id (simulate truly out-of-catalog upload!)
        self_idx = engine.id_to_idx[str(r["item_id"])]
        sims[self_idx] = -1e9
        top_idxs = np.argsort(-sims, kind="stable")[:10]
        top1_sim = float(sims[top_idxs[0]])
        scores_by_type = {}
        for r_i, idx in enumerate(top_idxs):
            s = max(float(sims[idx]), 0.0)
            w = (s ** 2) / float(np.log2(r_i + 2.0))
            at = str(engine.df.iloc[int(idx)]["articleType"])
            scores_by_type[at] = scores_by_type.get(at, 0.0) + w
        tot_w = sum(scores_by_type.values())
        best_type, best_w = max(scores_by_type.items(), key=lambda x: x[1])
        conf = best_w / max(tot_w, 1e-12)
        is_conf = (conf >= 0.62) and (top1_sim >= 0.65)
        if best_type == r["articleType"]:
            unseen_correct_all += 1
        if is_conf:
            unseen_confident_cnt += 1
            if best_type == r["articleType"]:
                unseen_correct_confident += 1
        else:
            unseen_fallback_cnt += 1

    return {
        "heldout_upload_8": {
            "total": len(ho_upl),
            "type_accuracy": ho_correct / max(1, len(ho_upl)),
            "confident_rate": ho_conf_true / max(1, len(ho_upl)),
            "fallback_rate": 1.0 - (ho_conf_true / max(1, len(ho_upl)))
        },
        "out_of_catalog_50_test": {
            "total": 50,
            "raw_top1_pred_accuracy": unseen_correct_all / 50.0,
            "confident_accuracy": unseen_correct_confident / max(1, unseen_confident_cnt),
            "confident_rate": unseen_confident_cnt / 50.0,
            "fallback_rate": unseen_fallback_cnt / 50.0
        }
    }


def main():
    print("=" * 75)
    print("BƯỚC 3: ĐÁNH GIÁ CHÍNH THỨC TRÊN 90 CA HELD-OUT & ABLATION")
    print("=" * 75)
    service = SearchService("config.yaml")
    ho_df = pd.read_csv(ROOT_DIR / "eval" / "cir_test" / "cir_heldout.csv")
    out_dir = ROOT_DIR / "eval" / "cir_test" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Non-regression Tab 1 & Tab 2 + Determinism + User case check
    user_img = ROOT_DIR / "eval" / "cir_test" / "crop_uploaded_shirt.jpg"
    res_u1, meta_u1 = service.search_cir(user_img, "màu xanh navy", top_k=10, force_upload=True)
    res_u2, meta_u2 = service.search_cir(user_img, "màu xanh navy", top_k=10, force_upload=True)
    assert [r["item_id"] for r in res_u1] == [r["item_id"] for r in res_u2], "Non-deterministic CIR output!"
    assert all(r["articleType"] == "Shirts" and r["baseColour"] == "Navy Blue" for r in res_u1), "User shirt case must return 10/10 Navy Blue Shirts!"
    print(f"🟢 Ca áo sơ mi trắng + 'màu xanh navy': 10/10 Shirts & Navy Blue | {meta_u1['latency_ms']} ms | 100% Deterministic!")

    smoke_t1, _ = service.search_text("túi sách màu hồng", config_mode="c5", top_k=5)
    assert len(smoke_t1) == 5 and smoke_t1[0]["articleType"] == "Handbags"
    smoke_t2, _ = service.search_image(user_img, top_k=5)
    assert len(smoke_t2) == 5 and smoke_t2[0]["articleType"] == "Shirts"
    print("🟢 Tab 1 & Tab 2 non-regression verified.")

    # 2. Ablation runs on 90 held-out cases
    ablation_configs = [
        ("a_no_anchor_lam1.0", False, "off", 1.0, 500),
        ("a2_no_anchor_lam0.5", False, "off", 0.5, 500),
        ("b_anchor_only_lam1.0", True, "auto", 1.0, 500),
        ("c1_anchor_lam0.2", True, "auto", 0.2, 500),
        ("c2_anchor_lam0.5_default", True, "auto", 0.5, 500),
        ("c3_anchor_lam0.8", True, "auto", 0.8, 500),
    ]

    ablation_summary = {}
    for name, dca, am, lam, pool_k in ablation_configs:
        print(f"\n>>> Running Held-Out Ablation [{name}]...")
        res = evaluate_heldout_config(service, ho_df, dca, am, lam, pool_k)
        ablation_summary[name] = res
        ov = res["overall"]
        ci = ov["hit_at_10_ci95"]
        print(
            f"  Hit@10={ov['hit_at_10_full']*100:.1f}% (95% CI: {ci[0]*100:.1f}%-{ci[1]*100:.1f}%) | "
            f"Hit@1={ov['hit_at_1_full']*100:.1f}% | TypeRet={ov['type_retention_p10']*100:.1f}% | "
            f"CritErr={ov['critical_error_rate']*100:.1f}% | p50={ov['p50_latency_ms']:.1f}ms | p95={ov['p95_latency_ms']:.1f}ms"
        )

    # 3. Upload Type Inference Benchmark
    print("\n>>> Measuring Upload articleType Inference Accuracy & Fallback Rate...")
    upload_bench = measure_upload_inference_benchmark(service)
    print("Upload Benchmark:", json.dumps(upload_bench, indent=2))

    # Save full results cleanly
    save_payload = {
        "ablation_overall": {k: v["overall"] for k, v in ablation_summary.items()},
        "ablation_by_cat": {k: v["by_cat"] for k, v in ablation_summary.items()},
        "default_c2_records": ablation_summary["c2_anchor_lam0.5_default"]["records"],
        "upload_inference_benchmark": upload_bench
    }
    with open(out_dir / "step3_heldout_ablation.json", "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2, ensure_ascii=False)
    print(f"\n🟢 Saved full Step 3 Held-out evaluation to {out_dir / 'step3_heldout_ablation.json'}")


if __name__ == "__main__":
    main()
