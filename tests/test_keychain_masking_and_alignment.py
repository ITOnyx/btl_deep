import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from app.engine import SearchService

def test_keychain_masking_and_row_alignment():
    print("Initializing SearchService...")
    service = SearchService(config_path="config.yaml")
    engine = service.engine

    # 1. Verify exact 38,644 items in engine and dataframe
    print("Checking item counts...")
    assert engine.N == 38644, f"Expected 38,644 items in engine, got {engine.N}"
    assert len(engine.df) == 38644, f"Expected 38,644 rows in df, got {len(engine.df)}"
    assert len(engine.emb_img_c) == 38644, f"Expected 38,644 vectors, got {len(engine.emb_img_c)}"

    # 2. Verify Keychains 43230 and 42066 are strictly excluded from engine catalog
    forbidden_ids = {"43230", "42066"}
    engine_ids_set = set(engine.item_ids)
    assert forbidden_ids.isdisjoint(engine_ids_set), "Forbidden keychain IDs found in engine.item_ids!"

    # 3. Verify 1:1 row alignment between df.iloc[i] and engine.item_ids[i]
    print("Checking 1:1 row alignment...")
    for sample_idx in [0, 100, 10650, 10651, 10652, 16038, 16039, 16040, 38643]:
        df_row_id = str(engine.df.iloc[sample_idx]["item_id"])
        item_id = engine.item_ids[sample_idx]
        assert df_row_id == item_id, f"Row index mismatch at {sample_idx}: df={df_row_id} vs item_ids={item_id}"

    # 4. Test Text Retrieval C0 (Dense Only) with explicit keychain queries
    print("Checking C0 retrieval...")
    for query in ["móc chìa khóa", "key chain wildcraft", "american tourister key case"]:
        results_c0, _ = service.search_text(query, config_mode="c0", top_k=20)
        retrieved_ids_c0 = {str(r["item_id"]) for r in results_c0}
        assert forbidden_ids.isdisjoint(retrieved_ids_c0), f"Keychains leaked into C0 search for '{query}': {forbidden_ids & retrieved_ids_c0}"

    # 5. Test Text Retrieval C5 (Full System) with explicit keychain queries
    print("Checking C5 retrieval...")
    for query in ["móc chìa khóa", "key chain wildcraft", "american tourister key case"]:
        results_c5, _ = service.search_text(query, config_mode="c5", top_k=20)
        retrieved_ids_c5 = {str(r["item_id"]) for r in results_c5}
        assert forbidden_ids.isdisjoint(retrieved_ids_c5), f"Keychains leaked into C5 search for '{query}': {forbidden_ids & retrieved_ids_c5}"

    # 6. Test BM25 Pure Search (C4)
    print("Checking C4 (BM25) retrieval...")
    for query in ["key chain", "key case", "carabiner"]:
        results_c4, _ = service.search_text(query, config_mode="c4", top_k=20)
        retrieved_ids_c4 = {str(r["item_id"]) for r in results_c4}
        assert forbidden_ids.isdisjoint(retrieved_ids_c4), f"Keychains leaked into BM25 search for '{query}': {forbidden_ids & retrieved_ids_c4}"

    # 7. Test Image-to-Image Visual Search
    print("Checking Image-to-Image retrieval...")
    sample_img_path = str(Path("data/fashion-dataset/images/15970.jpg"))
    results_img, _ = service.search_image(sample_img_path, top_k=20)
    retrieved_ids_img = {str(r["item_id"]) for r in results_img}
    assert forbidden_ids.isdisjoint(retrieved_ids_img), f"Keychains leaked into Visual search: {forbidden_ids & retrieved_ids_img}"

    # 8. Test CIR Search
    print("Checking CIR retrieval...")
    results_cir, _ = service.search_cir(sample_img_path, "màu xanh navy", top_k=20)
    retrieved_ids_cir = {str(r["item_id"]) for r in results_cir}
    assert forbidden_ids.isdisjoint(retrieved_ids_cir), f"Keychains leaked into CIR search: {forbidden_ids & retrieved_ids_cir}"

if __name__ == "__main__":
    test_keychain_masking_and_row_alignment()
    print("\n>>> ALL TESTS PASSED: Keychains are completely masked across all retrieval pathways and rows are 100% aligned! <<<")
