import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search_pipeline import VisualSearchEngine

def main():
    print("Testing VisualSearchEngine initialization and searches...")
    engine = VisualSearchEngine(gamma=0.3, alpha=1.0, enable_text_fusion=False)

    print(f"Engine Gallery size: {engine.N}")
    assert engine.N == 38644, f"Expected 38644, got {engine.N}"
    assert len(engine.emb_img_c) == 38644, f"Expected emb_img_c 38644, got {len(engine.emb_img_c)}"

    # 1. Test search_text
    print("\n--- Test search_text ---")
    results = engine.search_text("áo thun Puma nam", top_k=5)
    print("Top result:", results[0]["name"], "|", results[0]["brand"], "| score:", results[0]["score"], "| latency:", results[0]["latency_ms"], "ms")
    assert len(results) == 5

    # 2. Test search_image
    print("\n--- Test search_image ---")
    ref_img = results[0]["image_path"]
    img_results = engine.search_image(ref_img, top_k=5)
    print("Top image result:", img_results[0]["name"], "| score:", img_results[0]["score"], "| latency:", img_results[0]["latency_ms"], "ms")
    assert len(img_results) == 5

    # 3. Test search_cir
    print("\n--- Test search_cir ---")
    cir_results = engine.search_cir(ref_img, "màu xanh navy", top_k=5)
    print("Top CIR result:", cir_results[0]["name"], "| color:", cir_results[0]["baseColour"], "| latency:", cir_results[0]["latency_ms"], "ms")
    assert len(cir_results) == 5

    print("\nALL SEARCH PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    main()
