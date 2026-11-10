"""
Experiment to compare >= 4 English prompt patterns for CIR Combiner
on color-shift queries from dev set.
Patterns:
P1 (Minimal attribute): e.g. "navy blue"
P2 (Attribute + Target/Ref Type): e.g. "navy blue shirt"
P3 (Action instruction): e.g. "change to navy blue color"
P4 (Attribute + Type + Gender): e.g. "navy blue shirt for men"
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding='utf-8')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.engine import SearchService

def run_template_comparison():
    service = SearchService("config.yaml")
    engine = service.engine
    df = engine.df

    # Pick 20 diverse color shift test cases from dev set
    dev_specs = pd.read_csv("eval/cir_test/cir_queries.csv")
    color_cases = dev_specs[dev_specs["category"] == "color_shift"].sample(25, random_state=42)

    templates = {
        "P1_attribute_only": lambda c, t, g: f"{c}",
        "P2_attribute_and_type": lambda c, t, g: f"{c} {t}",
        "P3_instruction_action": lambda c, t, g: f"change to {c} color",
        "P4_full_attr_type_gender": lambda c, t, g: f"{c} {t} for {g}" if g else f"{c} {t}"
    }

    results_by_template = {t_name: [] for t_name in templates}

    for _, row in color_cases.iterrows():
        ref_id = str(row["ref_id"])
        tgt_t = row["target_type"]
        tgt_c = row["target_color"]
        ref_g = row["ref_gender"] or "men"
        
        # Prepare reference image path
        img_path = Path("data/fashion-dataset/images") / f"{ref_id}.jpg"
        if not img_path.exists():
            continue

        c_en = str(tgt_c).lower()
        t_en = str(tgt_t).lower()
        g_en = str(ref_g).lower()

        for t_name, t_fn in templates.items():
            en_query = t_fn(c_en, t_en, g_en)
            
            # Run CIR without soft bonus to evaluate pure combiner embedding power first,
            # and then with soft bonus to see final system performance
            res_pure, _ = service.search_cir(str(img_path), en_query, top_k=10, use_soft_bonus=False, ref_id=ref_id)
            res_system, _ = service.search_cir(str(img_path), en_query, top_k=10, use_soft_bonus=True, ref_id=ref_id)

            def score_res(res_list):
                full_matches = 0
                type_matches = 0
                for r in res_list[:10]:
                    t_ok = (r.get("articleType") == tgt_t)
                    c_ok = (r.get("baseColour") == tgt_c)
                    if t_ok:
                        type_matches += 1
                    if t_ok and c_ok:
                        full_matches += 1
                top1 = res_list[0] if res_list else {}
                top1_full = (top1.get("articleType") == tgt_t and top1.get("baseColour") == tgt_c)
                return {
                    "hit_at_10_full": 1.0 if full_matches > 0 else 0.0,
                    "hit_at_1_full": 1.0 if top1_full else 0.0,
                    "type_retention_p10": type_matches / 10.0,
                    "top1_score": float(top1.get("score", 0.0))
                }

            s_pure = score_res(res_pure)
            s_sys = score_res(res_system)
            results_by_template[t_name].append({
                "case_id": row["case_id"],
                "en_query": en_query,
                "pure_hit10": s_pure["hit_at_10_full"],
                "pure_type_p10": s_pure["type_retention_p10"],
                "sys_hit10": s_sys["hit_at_10_full"],
                "sys_hit1": s_sys["hit_at_1_full"],
                "sys_type_p10": s_sys["type_retention_p10"],
            })

    print("=" * 70)
    print("KẾT QUẢ SO SÁNH 4 DẠNG CÂU TIẾNG ANH CHO CIR COMBINER (25 CA DEV)")
    print("=" * 70)
    summary_rows = []
    for t_name, rows in results_by_template.items():
        df_t = pd.DataFrame(rows)
        row_dict = {
            "Template": t_name,
            "Sys Hit@10 Full (%)": df_t["sys_hit10"].mean() * 100,
            "Sys Hit@1 Full (%)": df_t["sys_hit1"].mean() * 100,
            "Type Retention P@10 (%)": df_t["sys_type_p10"].mean() * 100,
            "Pure Hit@10 (no bonus) (%)": df_t["pure_hit10"].mean() * 100,
            "Pure Type P@10 (%)": df_t["pure_type_p10"].mean() * 100
        }
        summary_rows.append(row_dict)
        print(f"[{t_name}]")
        print(f"  Sys Hit@10 (đầy đủ):   {row_dict['Sys Hit@10 Full (%)']:.1f}%")
        print(f"  Sys Hit@1 (đầy đủ):    {row_dict['Sys Hit@1 Full (%)']:.1f}%")
        print(f"  Type Retention P@10:   {row_dict['Type Retention P@10 (%)']:.1f}%")
        print(f"  Pure Hit@10 (no bonus):{row_dict['Pure Hit@10 (no bonus) (%)']:.1f}%\n")

    pd.DataFrame(summary_rows).to_csv("eval/cir_test/template_comparison.csv", index=False)
    print("Đã lưu kết quả so sánh vào eval/cir_test/template_comparison.csv")

if __name__ == "__main__":
    run_template_comparison()
