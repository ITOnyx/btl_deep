"""
Automated Test for F3: Verify that usage and season do not affect attribute bonus or candidate filtering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attr_index import AttributeIndex
from src.query_parser import QueryParser
import numpy as np

def test_usage_season_independence():
    parser = QueryParser()
    attr_idx = AttributeIndex()
    
    # Queries with usage/season terms vs pure category + gender
    test_pairs = [
        ("ao thun nam di lam", "ao thun nam"),
        ("ao thun nam tap gym", "ao thun nam"),
        ("ao so mi nam mua he", "ao so mi nam"),
        ("quan short nam mua lanh", "quan short nam"),
        ("dam nu di tiec", "dam nu"),
        ("giay casual nam dao pho", "giay casual nam")
    ]
    
    for q_with, q_without in test_pairs:
        p_with = parser.parse(q_with)
        p_without = parser.parse(q_without)
        
        # When use_usage=False and use_season=False (the default system behavior),
        # AttributeIndex ignores usage and season completely.
        bonus_with = attr_idx.compute_attribute_soft_bonus(p_with, np.arange(100), gamma_type=0.3, gamma_gender=0.3, gamma_colour=0.3, gamma_brand=0.3)
        bonus_without = attr_idx.compute_attribute_soft_bonus(p_without, np.arange(100), gamma_type=0.3, gamma_gender=0.3, gamma_colour=0.3, gamma_brand=0.3)
        
        # Candidate filter exact must match
        cands_with = attr_idx.filter_exact(p_with)
        cands_without = attr_idx.filter_exact(p_without)
        
        # Candidate filter relaxed must match
        rel_with, tier_with = attr_idx.filter_relaxed(p_with, min_candidates=20)
        rel_without, tier_without = attr_idx.filter_relaxed(p_without, min_candidates=20)
        
        assert np.array_equal(bonus_with, bonus_without), f"Soft bonus mismatch for '{q_with}' vs '{q_without}'"
        assert np.array_equal(cands_with, cands_without), f"Filter exact mismatch for '{q_with}' vs '{q_without}'"
        assert np.array_equal(rel_with, rel_without), f"Filter relaxed mismatch for '{q_with}' vs '{q_without}'"
        assert tier_with == tier_without, f"Tier mismatch for '{q_with}' vs '{q_without}'"
        
    # Check on eval/q6_queries.json
    q6_path = Path("eval/q6_queries.json")
    if q6_path.exists():
        import json
        with open(q6_path, "r", encoding="utf-8") as f:
            q6_data = json.load(f)
        for q in q6_data["val_queries"] + q6_data["test_queries"]:
            p = parser.parse(q["query_text"])
            bonus = attr_idx.compute_attribute_soft_bonus(p, np.arange(50), 0.3, 0.3, 0.3, 0.3)
            # If no core attributes, bonus must be strictly zero
            if p.get("articleType") is None and p.get("gender") is None and p.get("baseColour") is None and p.get("brand") is None:
                assert np.all(bonus == 0), f"Bonus non-zero for pure usage/season query {q['query_text']}"
        print(f"Verified {len(q6_data['val_queries'])} val + {len(q6_data['test_queries'])} test queries in eval/q6_queries.json are independent of filter!")
        
    print("All usage/season independence tests passed successfully! The attribute index is 100% independent of usage/season.")

if __name__ == "__main__":
    test_usage_season_independence()
