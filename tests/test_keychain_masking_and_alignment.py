import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.engine import SearchService

def test_keychain_masking_and_row_alignment():
    print("\n--- 1. Initializing SearchService & Engine ---")
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

    # 3. DIRECT ASSERT: mask[idx] == False for 43230 and 42066 in raw index
    print("Direct assert on raw index mask...")
    with open("index_v2/id_map_v2.json", "r", encoding="utf-8") as f:
        raw_id_map = json.load(f)
    raw_ids = [str(x["item_id"] if isinstance(x, dict) else x) for x in raw_id_map]
    in_scope_set = set(engine.item_ids)
    raw_mask = np.array([iid in in_scope_set for iid in raw_ids], dtype=bool)

    idx_43230 = raw_ids.index("43230")
    idx_42066 = raw_ids.index("42066")
    assert idx_43230 == 10651, f"Expected 43230 at index 10651, got {idx_43230}"
    assert idx_42066 == 16039, f"Expected 42066 at index 16039, got {idx_42066}"
    assert raw_mask[idx_43230] == False, "raw_mask[10651] must be False!"
    assert raw_mask[idx_42066] == False, "raw_mask[16039] must be False!"
    print("  Direct assert passed: mask[10651]==False and mask[16039]==False.")

    # 4. Verify 1:1 row alignment between df.iloc[i] and engine.item_ids[i]
    print("Checking 1:1 row alignment...")
    for sample_idx in [0, 100, 10650, 10651, 10652, 16038, 16039, 16040, 38643]:
        df_row_id = str(engine.df.iloc[sample_idx]["item_id"])
        item_id = engine.item_ids[sample_idx]
        assert df_row_id == item_id, f"Row index mismatch at {sample_idx}: df={df_row_id} vs item_ids={item_id}"

    # 5. RE-ENCODING VERIFICATION: Check embedding <-> ID alignment
    print("Checking re-encoding consistency on sample images...")
    for sample_id in ["15970", "1855", "39386"]:
        idx = engine.id_to_idx[sample_id]
        stored_raw = engine.emb_img_raw[idx]
        img_p = f"data/fashion-dataset/images/{sample_id}.jpg"
        im = Image.open(img_p).convert("RGB")
        t = engine.img_transform(im).unsqueeze(0).to(engine.device)
        if engine.device == "cuda":
            t = t.half()
        with torch.no_grad():
            feat = engine.model.encode_image(t).cpu().numpy()[0]
        cos_sim = np.dot(stored_raw, feat) / (np.linalg.norm(stored_raw) * np.linalg.norm(feat))
        assert cos_sim > 0.99, f"Cosine similarity {cos_sim} too low for {sample_id}!"
        print(f"  Sample {sample_id} cosine similarity: {cos_sim:.6f} (> 0.99)")

    # 6. Test Text Retrieval C0, C5, C4 with explicit keychain queries
    print("Checking C0, C5, C4 text retrieval exclusion...")
    for query in ["móc chìa khóa", "key chain wildcraft", "american tourister key case"]:
        results_c0, _ = service.search_text(query, config_mode="c0", top_k=20)
        assert forbidden_ids.isdisjoint({str(r["item_id"]) for r in results_c0})
        results_c5, _ = service.search_text(query, config_mode="c5", top_k=20)
        assert forbidden_ids.isdisjoint({str(r["item_id"]) for r in results_c5})
        results_c4, _ = service.search_text(query, config_mode="c4", top_k=20)
        assert forbidden_ids.isdisjoint({str(r["item_id"]) for r in results_c4})

    # 7. Test Image-to-Image & CIR using KEYCHAIN IMAGE (43230.jpg) as query
    print("Testing Image->Image & CIR with actual Keychain image (43230.jpg)...")
    keychain_img_path = str(Path("data/fashion-dataset/images/43230.jpg"))
    assert Path(keychain_img_path).exists(), "43230.jpg must exist for this test!"

    # Image->Image query
    res_img, _ = service.search_image(keychain_img_path, top_k=20)
    retrieved_img = {str(r["item_id"]) for r in res_img}
    assert forbidden_ids.isdisjoint(retrieved_img), f"Keychains leaked into Visual search when querying keychain image: {forbidden_ids & retrieved_img}"

    # CIR query
    res_cir, _ = service.search_cir(keychain_img_path, "màu đen da bóng", top_k=20)
    retrieved_cir = {str(r["item_id"]) for r in res_cir}
    assert forbidden_ids.isdisjoint(retrieved_cir), f"Keychains leaked into CIR search when querying keychain image: {forbidden_ids & retrieved_cir}"

    print(">>> ALL TESTS PASSED: Keychains are completely masked, raw masks assert False, re-encoding > 0.99, and keychain query images return 0 leaks! <<<")

if __name__ == "__main__":
    test_keychain_masking_and_row_alignment()
