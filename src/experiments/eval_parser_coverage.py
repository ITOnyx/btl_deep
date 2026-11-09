"""
Evaluate Parser Coverage & Accuracy on Q2 and Q3 (src/eval_parser_coverage.py)
Phase 3e: Measures:
- Accuracy/Coverage for articleType, gender, baseColour, brand on Q2 (N=61) and Q3 (N=100)
- Count and percentage of queries where nothing was parsed
- Most frequent unparsed tokens
"""

import json
from pathlib import Path
from collections import Counter
import sys

# Ensure root directory in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.query_parser import QueryParser
from src.vi_vocab import remove_diacritics

def evaluate_parser_coverage():
    parser = QueryParser()

    # 1. Load Q2
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    q2_queries = testset.get("q2_attribute_queries", [])

    # 2. Load Q3
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        q3_queries = json.load(f)

    print(f"Loaded {len(q2_queries)} Q2 queries and {len(q3_queries)} Q3 queries.")

    # --- EVALUATE Q2 ---
    q2_total = len(q2_queries)
    q2_correct_type = 0
    q2_correct_gender = 0
    q2_correct_colour = 0
    q2_nothing_parsed = 0
    q2_unparsed_tokens = Counter()

    for q in q2_queries:
        q_text = q["query_text"]
        target = q["target_attributes"]
        parsed = parser.parse(q_text)

        # Type match
        exp_type = target.get("articleType")
        if parsed["articleType"] == exp_type:
            q2_correct_type += 1

        # Gender match
        exp_gen = target.get("gender")
        if parsed["gender"] == exp_gen:
            q2_correct_gender += 1

        # Colour match
        exp_col = target.get("baseColour")
        if parsed["baseColour"] == exp_col:
            q2_correct_colour += 1

        # Nothing parsed check
        if not parsed["articleType"] and not parsed["gender"] and not parsed["baseColour"] and not parsed["brand"]:
            q2_nothing_parsed += 1

        for tok in parsed["unparsed_tokens"]:
            q2_unparsed_tokens[tok] += 1

    # --- EVALUATE Q3 ---
    q3_total = len(q3_queries)
    q3_correct_type = 0
    q3_correct_gender = 0
    q3_correct_colour = 0
    q3_has_brand = 0
    q3_nothing_parsed = 0
    q3_unparsed_tokens = Counter()

    for q in q3_queries:
        q_text = q["query"]
        spec = q.get("spec", {})
        parsed = parser.parse(q_text)

        # Type match (spec articleType can be list or string)
        exp_types = spec.get("articleType", [])
        if isinstance(exp_types, str):
            exp_types = [exp_types]
        if exp_types is None:
            exp_types = []

        if parsed["articleType"] in exp_types or (not exp_types and parsed["articleType"] is None):
            q3_correct_type += 1

        # Gender match
        exp_gen = spec.get("gender")
        if parsed["gender"] == exp_gen:
            q3_correct_gender += 1

        # Colour match (spec baseColour can be list or string)
        exp_cols = spec.get("baseColour", [])
        if isinstance(exp_cols, str):
            exp_cols = [exp_cols]
        if exp_cols is None:
            exp_cols = []

        if parsed["baseColour"] in exp_cols or (not exp_cols and parsed["baseColour"] is None):
            q3_correct_colour += 1

        if parsed["brand"]:
            q3_has_brand += 1

        # Nothing parsed check
        if not parsed["articleType"] and not parsed["gender"] and not parsed["baseColour"] and not parsed["brand"]:
            q3_nothing_parsed += 1

        for tok in parsed["unparsed_tokens"]:
            q3_unparsed_tokens[tok] += 1

    # Format output
    report = {
        "Q2": {
            "total_queries": q2_total,
            "article_type_accuracy": round(q2_correct_type / q2_total * 100, 2),
            "gender_accuracy": round(q2_correct_gender / q2_total * 100, 2),
            "colour_accuracy": round(q2_correct_colour / q2_total * 100, 2),
            "all_three_correct": round(sum(
                1 for q in q2_queries
                if parser.parse(q["query_text"])["articleType"] == q["target_attributes"].get("articleType")
                and parser.parse(q["query_text"])["gender"] == q["target_attributes"].get("gender")
                and parser.parse(q["query_text"])["baseColour"] == q["target_attributes"].get("baseColour")
            ) / q2_total * 100, 2),
            "nothing_parsed_count": q2_nothing_parsed,
            "nothing_parsed_percent": round(q2_nothing_parsed / q2_total * 100, 2),
            "frequent_unparsed_tokens": q2_unparsed_tokens.most_common(10)
        },
        "Q3": {
            "total_queries": q3_total,
            "article_type_accuracy": round(q3_correct_type / q3_total * 100, 2),
            "gender_accuracy": round(q3_correct_gender / q3_total * 100, 2),
            "colour_accuracy": round(q3_correct_colour / q3_total * 100, 2),
            "brand_extracted_count": q3_has_brand,
            "nothing_parsed_count": q3_nothing_parsed,
            "nothing_parsed_percent": round(q3_nothing_parsed / q3_total * 100, 2),
            "frequent_unparsed_tokens": q3_unparsed_tokens.most_common(15)
        }
    }

    print("\n==========================================================")
    print("PARSER COVERAGE & ACCURACY REPORT (Q2 & Q3)")
    print("==========================================================")
    print(f"\n1. Bộ Truy Vấn Thuộc Tính Q2 (N={q2_total}):")
    print(f"   - Độ chính xác ArticleType: {report['Q2']['article_type_accuracy']}% ({q2_correct_type}/{q2_total})")
    print(f"   - Độ chính xác Gender     : {report['Q2']['gender_accuracy']}% ({q2_correct_gender}/{q2_total})")
    print(f"   - Độ chính xác BaseColour : {report['Q2']['colour_accuracy']}% ({q2_correct_colour}/{q2_total})")
    print(f"   - Khớp hoàn hảo cả 3      : {report['Q2']['all_three_correct']}%")
    print(f"   - Không nhận diện được gì : {report['Q2']['nothing_parsed_count']} câu ({report['Q2']['nothing_parsed_percent']}%)")
    print(f"   - Từ không nhận dạng (Q2) : {report['Q2']['frequent_unparsed_tokens']}")

    print(f"\n2. Bộ Truy Vấn Viết Tay Q3 (N={q3_total}):")
    print(f"   - Độ chính xác ArticleType: {report['Q3']['article_type_accuracy']}% ({q3_correct_type}/{q3_total})")
    print(f"   - Độ chính xác Gender     : {report['Q3']['gender_accuracy']}% ({q3_correct_gender}/{q3_total})")
    print(f"   - Độ chính xác BaseColour : {report['Q3']['colour_accuracy']}% ({q3_correct_colour}/{q3_total})")
    print(f"   - Không nhận diện được gì : {report['Q3']['nothing_parsed_count']} câu ({report['Q3']['nothing_parsed_percent']}%)")
    print(f"   - Từ không nhận dạng (Q3) : {report['Q3']['frequent_unparsed_tokens']}")

    # Save summary report
    out_json = Path("reports/parser_coverage_a6.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved coverage report to {out_json}")

    return report

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    evaluate_parser_coverage()
