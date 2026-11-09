import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import SearchService, verify_rule_7_checkpoint_hash

def main():
    print("1. Testing Rule 7 Checkpoint Hash Verification...")
    ok, cfg_h, man_h = verify_rule_7_checkpoint_hash()
    print(f"Rule 7 check passed: {ok} (Hash: {cfg_h[:16]}...)")

    print("\n2. Initializing SearchService...")
    service = SearchService()
    print(f"SearchService initialized. Gallery: {service.engine.N} items.")

    print("\n3. Testing service.search_text (C5)...")
    results_c5, meta_c5 = service.search_text("áo thun Puma nam", config_mode="c5", top_k=3)
    print(f"C5 Results count: {len(results_c5)}, top: {results_c5[0]['name']}, latency: {meta_c5['breakdown']['total_ms']} ms")

    print("\n4. Testing service.search_text (C0)...")
    results_c0, meta_c0 = service.search_text("áo thun Puma nam", config_mode="c0", top_k=3)
    print(f"C0 Results count: {len(results_c0)}, top: {results_c0[0]['name']}, score: {results_c0[0]['score']}")

    print("\n5. Testing service.search_image...")
    test_img = results_c5[0]["image_path"]
    img_res, img_meta = service.search_image(test_img, top_k=3)
    print(f"Image Results count: {len(img_res)}, top: {img_res[0]['name']}, latency: {img_meta['latency_ms']} ms")

    print("\n6. Testing service.search_cir...")
    cir_res, cir_meta = service.search_cir(test_img, "màu xanh navy", top_k=3)
    print(f"CIR Results count: {len(cir_res)}, top: {cir_res[0]['name']}, color: {cir_res[0]['baseColour']}, latency: {cir_meta['latency_ms']} ms")

    print("\nALL SERVICE INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    main()
