import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
from app.engine import SearchService, verify_rule_7_checkpoint_hash


def test_app_service_all_flows():
    print("1. Testing Rule 7 Checkpoint Hash Verification...")
    ok, cfg_h, man_h = verify_rule_7_checkpoint_hash()
    assert ok and cfg_h == man_h
    print(f"Rule 7 check passed: {ok} (Hash: {cfg_h[:16]}...)")

    print("\n2. Initializing SearchService...")
    service = SearchService()
    assert service.engine.N == 38644
    for flow in ("text_to_image", "image_to_image", "cir_catalog", "cir_upload"):
        assert service.get_flow_sla_ms(flow) == 200.0
    print(f"SearchService initialized. Gallery: {service.engine.N} items.")

    print("\n3. Testing service.search_text (C5, C0, C4, and Lavie)...")
    results_c5, meta_c5 = service.search_text("áo thun Puma nam", config_mode="c5", top_k=5)
    assert len(results_c5) == 5 and meta_c5["flow"] == "text_to_image"
    print(f"C5 Results count: {len(results_c5)}, top: {results_c5[0]['name']}, latency: {meta_c5['breakdown']['total_ms']} ms")

    results_c0, meta_c0 = service.search_text("áo thun Puma nam", config_mode="c0", top_k=3)
    assert len(results_c0) == 3 and meta_c0["flow"] == "text_to_image"

    results_c4, meta_c4 = service.search_text("áo thun Puma nam", config_mode="c4", top_k=3)
    assert len(results_c4) == 3 and meta_c4["flow"] == "text_to_image"

    results_lavie, meta_lavie = service.search_text("túi xách nữ Lavie", config_mode="c5", top_k=3)
    assert len(results_lavie) == 3 and "lavie" in [t.lower() for t in meta_lavie["parsed"].get("unparsed_tokens", [])]

    print("\n4. Testing service.search_image (fast default vs accurate Batch-2 TTA, self-exclusion)...")
    test_img_path = service.kaggle_images_dir / "15970.jpg"
    img_fast, meta_fast = service.search_image(test_img_path, top_k=5, exclude_self_id="15970", mode="fast")
    assert len(img_fast) == 5
    assert meta_fast["flow"] == "image_to_image" and meta_fast["i2i_mode"] == "fast" and not meta_fast["use_flip_aug"]
    assert "15970" not in [r["item_id"] for r in img_fast]
    assert all(k in meta_fast["breakdown"] for k in ("decode_ms", "preprocess_ms", "vit_forward_ms", "gallery_search_ms", "rerank_ms"))
    print(f"Image Fast latency: {meta_fast['latency_ms']} ms")

    img_acc, meta_acc = service.search_image(test_img_path, top_k=5, exclude_self_id="15970", mode="accurate")
    assert meta_acc["i2i_mode"] == "accurate" and meta_acc["use_flip_aug"]
    print(f"Image Accurate latency: {meta_acc['latency_ms']} ms")

    print("\n5. Testing service.search_cir (cir_catalog vs cir_upload)...")
    pil_img = Image.open(test_img_path).convert("RGB")
    cir_cat, meta_cat = service.search_cir(pil_img, "màu xanh navy", top_k=5, ref_id="15970", force_upload=False)
    assert len(cir_cat) == 5
    assert meta_cat["flow"] == "cir_catalog" and meta_cat["is_catalog_cached"] is True
    print(f"CIR Catalog latency: {meta_cat['latency_ms']} ms")

    cir_up, meta_up = service.search_cir(pil_img, "màu xanh navy", top_k=5, ref_id="15970", force_upload=True)
    assert len(cir_up) == 5
    assert meta_up["flow"] == "cir_upload" and meta_up["is_catalog_cached"] is False
    print(f"CIR Upload latency: {meta_up['latency_ms']} ms")

    print("\nALL SERVICE INTEGRATION TESTS PASSED!")


if __name__ == "__main__":
    test_app_service_all_flows()
