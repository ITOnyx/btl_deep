"""
Comprehensive Unit Tests for Dynamic Category Anchor & Vector Interpolation (Step 3).
Covers:
1. State A (Attribute Shift) vs State B (Category Mutation) classification & disambiguation.
2. Hard category locking behavior on real SearchService.
3. Filtered results < 10 handling (family fallback + explicit shortage notice without silent wrong padding).
4. Upload image type inference fallback below confidence threshold (soft rerank + friendly status).
5. Vector interpolation at lambda = 0.0 and lambda = 1.0.
6. Determinism check.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.query_parser_v2 import QueryParserV2
from src.cir_normalizer import (
    sanitize_parsed_modifier,
    classify_cir_intent,
    normalize_cir_modifier,
    recover_color_from_typos,
)
from app.engine import SearchService

def test_normalizer_and_state_classification():
    print("[1/5] Testing State A/B Classification & Disambiguation...")
    parser = QueryParserV2()

    # 1a. False positive 'đậm' -> Dresses must be cleared and classified as State A
    mod_a1 = "tông màu xanh navy đậm"
    clean_a1, extra_a1, diag_a1 = sanitize_parsed_modifier(mod_a1, parser.parse(mod_a1))
    assert clean_a1.get("articleType") is None, f"Expected None, got {clean_a1.get('articleType')}"
    intent_a1 = classify_cir_intent(mod_a1, clean_a1, ref_type="Shirts")
    assert intent_a1["intent_state"] == "A_attribute_shift"
    assert intent_a1["target_type"] == "Shirts"

    # 1b. False positive 'có mũ' -> Caps must be cleared and become 'hooded'
    mod_a2 = "màu đen có mũ cho nam"
    clean_a2, extra_a2, diag_a2 = sanitize_parsed_modifier(mod_a2, parser.parse(mod_a2))
    assert clean_a2.get("articleType") is None, f"Expected None, got {clean_a2.get('articleType')}"
    assert "hooded" in extra_a2
    intent_a2 = classify_cir_intent(mod_a2, clean_a2, ref_type="Jackets")
    assert intent_a2["intent_state"] == "A_attribute_shift"
    assert intent_a2["target_type"] == "Jackets"

    # 1c. Repeating original category -> still State A
    mod_a3 = "áo sơ mi màu trắng dài tay"
    clean_a3, _, _ = sanitize_parsed_modifier(mod_a3, parser.parse(mod_a3))
    intent_a3 = classify_cir_intent(mod_a3, clean_a3, ref_type="Shirts")
    assert intent_a3["intent_state"] == "A_attribute_shift"
    assert intent_a3["target_type"] == "Shirts"

    # 1d. True State B (Category Mutation)
    mod_b1 = "chuyển thành chân váy màu đen"
    clean_b1, _, _ = sanitize_parsed_modifier(mod_b1, parser.parse(mod_b1))
    intent_b1 = classify_cir_intent(mod_b1, clean_b1, ref_type="Dresses")
    assert intent_b1["intent_state"] == "B_category_mutation"
    assert intent_b1["target_type"] == "Skirts"

    mod_b2 = "đổi sang áo khoác màu đen"
    clean_b2, _, _ = sanitize_parsed_modifier(mod_b2, parser.parse(mod_b2))
    intent_b2 = classify_cir_intent(mod_b2, clean_b2, ref_type="Shirts")
    assert intent_b2["intent_state"] == "B_category_mutation"
    assert intent_b2["target_type"] == "Jackets"

    # 1e. Color typo recovery & anchor shadowing
    assert recover_color_from_typos("màu đenn tuyền") == "Black"
    assert recover_color_from_typos("màu xanh navvy") == "Navy Blue"
    clean_shade, _, _ = sanitize_parsed_modifier("màu xanh than công sở", parser.parse("màu xanh than công sở"))
    assert clean_shade["baseColour"] == "Navy Blue"
    print("  🟢 Normalizer & State A/B tests passed.")


def test_vector_interpolation_math():
    print("[2/5] Testing lambda interpolation formula...")
    rng = np.random.default_rng(42)
    e_img = rng.normal(size=1152).astype(np.float32)
    e_comb = rng.normal(size=1152).astype(np.float32)

    q0 = SearchService.interpolate_cir_vector(e_img, e_comb, 0.0)
    assert np.isclose(np.dot(q0, e_img / np.linalg.norm(e_img)), 1.0, atol=1e-5)

    q1 = SearchService.interpolate_cir_vector(e_img, e_comb, 1.0)
    assert np.isclose(np.dot(q1, e_comb / np.linalg.norm(e_comb)), 1.0, atol=1e-5)

    q_half = SearchService.interpolate_cir_vector(e_img, e_comb, 0.5)
    assert np.isclose(np.linalg.norm(q_half), 1.0, atol=1e-5)
    expected = (e_img / np.linalg.norm(e_img)) + (e_comb / np.linalg.norm(e_comb))
    expected = expected / np.linalg.norm(expected)
    assert np.isclose(np.dot(q_half, expected), 1.0, atol=1e-5)
    print("  🟢 Interpolation formula tests passed.")


def test_engine_locking_shortage_fallback_lambda(service: SearchService):
    print("[3/5] Testing Dynamic Locking (State A & State B)...")
    shirt_id = "15970"
    shirt_img = ROOT_DIR / "data" / "fashion-dataset" / "images" / f"{shirt_id}.jpg"
    if not shirt_img.exists():
        shirt_img = str(service.engine.df[service.engine.df["item_id"].astype(str) == shirt_id].iloc[0]["path"])

    # State A locking on Shirts
    res_a, meta_a = service.search_cir(
        ref_image=shirt_img, modifier_text="màu xanh navy đậm", top_k=10,
        ref_id=shirt_id, dynamic_category_anchor=True, modifier_weight=0.5
    )
    assert meta_a["intent_state"] == "A_attribute_shift"
    assert meta_a["target_type"] == "Shirts"
    assert len(res_a) == 10
    assert all(r["articleType"] == "Shirts" for r in res_a), "State A lock failed!"

    # State B Category Mutation: Shirts -> Skirts
    res_b, meta_b = service.search_cir(
        ref_image=shirt_img, modifier_text="chuyển thành chân váy màu đen", top_k=10,
        ref_id=shirt_id, dynamic_category_anchor=True, modifier_weight=0.5
    )
    assert meta_b["intent_state"] == "B_category_mutation"
    assert meta_b["target_type"] == "Skirts"
    assert len(res_b) == 10
    assert all(r["articleType"] == "Skirts" for r in res_b), "State B mutation lock failed!"

    print("[4/5] Testing < 10 Filtered Results & Upload Low-Confidence Soft Fallback...")
    # Case 4a: Truly rare category WITH a related family in eval/families.json: Shrug (only 6 in catalog, family includes Tops/Tshirts/Tunics)
    shrug_row = service.engine.df[service.engine.df["articleType"] == "Shrug"].iloc[0]
    res_fam, meta_fam = service.search_cir(
        ref_image=str(shrug_row["path"]), modifier_text="màu đen", top_k=10,
        ref_id=str(shrug_row["item_id"]), dynamic_category_anchor=True
    )
    allowed_types = {"Shrug", "Tops", "Tshirts", "Tunics"}
    assert len(res_fam) == 10
    assert meta_fam["exact_match_count"] < 10 and meta_fam["family_fallback_count"] > 0
    assert all(r["articleType"] in allowed_types for r in res_fam), f"Unexpected type in family fallback: {[r['articleType'] for r in res_fam]}"
    assert "bổ sung" in (meta_fam.get("anchor_shortage_notice") or ""), f"Expected family supplement notice, got {meta_fam.get('anchor_shortage_notice')}"

    # Case 4b: Tight pool on Shirts (no other type in Shirts family) -> returns <= 5 items with explicit shortage notice
    res_short, meta_short = service.search_cir(
        ref_image=shirt_img, modifier_text="màu hồng", top_k=10,
        ref_id=shirt_id, dynamic_category_anchor=True, candidate_pool_k=5, expand_pool_if_short=False
    )
    assert len(res_short) <= 5, f"Expected <= 5 results when pool=5 & no family for Shirts, got {len(res_short)}"
    assert all(r["articleType"] == "Shirts" for r in res_short), "Must NEVER silently pad non-Shirts items!"
    assert "Chỉ có" in (meta_short.get("anchor_shortage_notice") or ""), f"Expected shortage notice, got {meta_short.get('anchor_shortage_notice')}"

    # Case 4c: Truly rare catalog category (< 10 total items in whole catalog): Umbrellas (6 items)
    umb_row = service.engine.df[service.engine.df["articleType"] == "Umbrellas"].iloc[0]
    res_umb, meta_umb = service.search_cir(
        ref_image=str(umb_row["path"]), modifier_text="màu đen", top_k=10,
        ref_id=str(umb_row["item_id"]), dynamic_category_anchor=True
    )
    assert len(res_umb) < 10, f"Expected < 10 Umbrellas, got {len(res_umb)}"
    assert all(r["articleType"] == "Umbrellas" for r in res_umb)
    assert "Chỉ có" in (meta_umb.get("anchor_shortage_notice") or "")

    # Case 4d: Low-confidence upload fallback (OOD image img_non_fashion.jpg)
    ood_img = ROOT_DIR / "eval" / "cir_test" / "img_non_fashion.jpg"
    res_ood, meta_ood = service.search_cir(
        ref_image=ood_img, modifier_text="màu xanh navy", top_k=10,
        force_upload=True, dynamic_category_anchor=True
    )
    assert meta_ood["upload_type_confident"] is False, "OOD image should be below confidence threshold!"
    assert meta_ood["anchor_fallback_soft"] is True
    assert "Không đủ chắc chắn" in meta_ood["anchor_status_msg"]
    assert len(res_ood) == 10

    print("[5/5] Testing lambda=0, lambda=1, and Determinism...")
    res_l0, meta_l0 = service.search_cir(
        ref_image=shirt_img, modifier_text="màu đỏ", top_k=10,
        use_soft_bonus=False, ref_id=shirt_id, dynamic_category_anchor=True, modifier_weight=0.0
    )
    assert np.isclose(meta_l0["query_vec_img_cosine"], 1.0, atol=1e-4)

    res_l1_a, meta_l1_a = service.search_cir(
        ref_image=shirt_img, modifier_text="màu đỏ", top_k=10,
        use_soft_bonus=True, ref_id=shirt_id, dynamic_category_anchor=True, modifier_weight=1.0
    )
    res_l1_b, meta_l1_b = service.search_cir(
        ref_image=shirt_img, modifier_text="màu đỏ", top_k=10,
        use_soft_bonus=True, ref_id=shirt_id, dynamic_category_anchor=True, modifier_weight=1.0
    )
    assert np.isclose(meta_l1_a["query_vec_comb_cosine"], 1.0, atol=1e-4)
    assert [r["item_id"] for r in res_l1_a] == [r["item_id"] for r in res_l1_b], "Results must be 100% deterministic!"
    assert [round(r["score"], 5) for r in res_l1_a] == [round(r["score"], 5) for r in res_l1_b]
    print("  🟢 Engine locking, shortage, upload fallback, lambda=0/1, and determinism tests passed 100%!")


if __name__ == "__main__":
    test_normalizer_and_state_classification()
    test_vector_interpolation_math()
    srv = SearchService("config.yaml")
    test_engine_locking_shortage_fallback_lambda(srv)
    print("\n🟢 TẤT CẢ UNIT TESTS ĐỀU PASS 100%!")
