import sys
import json
import numpy as np
import pandas as pd
import pathlib

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoTokenizer

def analyze_d2():
    json_path = pathlib.Path("data/checkpoints_siglip-vit/dense_captions_catalog.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    first_key = list(data.keys())[0]
    first_val = data[first_key]
    print(f"Structure of dense_captions_catalog.json:")
    print(f"  Key type: {type(first_key)} (e.g. '{first_key}')")
    print(f"  Value type: {type(first_val)} (e.g. '{first_val}')")
    print(f"  Does it have nested fields? No, it is a flat mapping: str(item_id) -> str(caption).")

    # Load testset Q1 queries
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)

    q1_queries = testset["q1_paired_queries"]
    captions = [q["query_text"] for q in q1_queries]

    tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-224")

    # Token length distribution
    token_lengths = []
    truncated_count = 0
    ten_goc_intact_count = 0

    for cap in captions:
        toks = tokenizer.encode(cap, add_special_tokens=True)
        length = len(toks)
        token_lengths.append(length)

        if length > 128:
            truncated_count += 1

        # Check if "Tên gốc:" is intact after truncation to 128
        trunc_toks = toks[:128]
        decoded_trunc = tokenizer.decode(trunc_toks)
        if "Tên gốc" in decoded_trunc:
            ten_goc_intact_count += 1

    p50 = float(np.percentile(token_lengths, 50))
    p95 = float(np.percentile(token_lengths, 95))
    max_len = int(np.max(token_lengths))

    print(f"\n--- D2 TOKEN LENGTH DISTRIBUTION (Q1 Captions, n={len(captions)}) ---")
    print(f"Median (p50): {p50:.1f} tokens")
    print(f"p95:          {p95:.1f} tokens")
    print(f"Max length:   {max_len} tokens")
    print(f"Truncated at 128 tokens: {truncated_count}/{len(captions)} ({truncated_count/len(captions)*100:.2f}%)")
    print(f"Captions where 'Tên gốc' is preserved after truncation: {ten_goc_intact_count}/{len(captions)} ({ten_goc_intact_count/len(captions)*100:.2f}%)")

    # 3 decoded token examples (head, middle, truncated/tail)
    print("\n--- 3 EXAMPLES OF DECODED TOKENS ---")
    indices_to_show = [0, len(captions)//2]
    # find a truncated one if any, else longest
    longest_idx = int(np.argmax(token_lengths))
    indices_to_show.append(longest_idx)

    for idx in indices_to_show:
        cap = captions[idx]
        toks = tokenizer.encode(cap, add_special_tokens=True)
        trunc_toks = toks[:128]
        decoded = tokenizer.decode(trunc_toks)
        print(f"\nExample idx {idx} (Original tokens: {len(toks)}, Kept: {len(trunc_toks)}):")
        print(f"  Raw:     {cap}")
        print(f"  Decoded: {decoded}")
        print(f"  Token IDs: {trunc_toks[:15]} ... {trunc_toks[-5:]}")

if __name__ == "__main__":
    analyze_d2()
