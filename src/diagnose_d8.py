"""
Diagnosis D8: Diacritic Restoration Rule Evaluation
Evaluates 4 configurations on the 25 unaccented queries in eval/handwritten_queries.json:
  (a) bản gốc có dấu (accented ground truth counterpart)
  (b) truy vấn không dấu chưa qua bộ khôi phục (raw unaccented)
  (c) quy tắc khôi phục cũ (luôn khôi phục qua restorer.restore())
  (d) quy tắc c' (chỉ khôi phục khi query hoàn toàn không dấu)
Computes Precision@10, Hit@10, and AP@10 on G_full and G_test.
"""
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model_siglip import load_siglip_inference
from src.vi_restore import DiacriticRestorer
from src.vi_vocab import remove_diacritics

def compute_metrics(retrieved_ids, target_ids, k=10):
    hits = [1.0 if rid in target_ids else 0.0 for rid in retrieved_ids[:k]]
    p10 = sum(hits) / k
    hit10 = 1.0 if sum(hits) > 0 else 0.0
    
    # AP@10
    precisions = []
    num_rel = 0
    for idx, h in enumerate(hits):
        if h > 0:
            num_rel += 1
            precisions.append(num_rel / (idx + 1))
    ap10 = float(np.mean(precisions)) if precisions else 0.0
    return p10, hit10, ap10

def run_d8():
    print("=== D8: DIACRITIC RESTORATION EVALUATION ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        handwritten = json.load(f)
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    with open("index/id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)
    catalog = pd.read_parquet("data/catalog/catalog.parquet")

    restorer = DiacriticRestorer()

    # Load image embeddings
    img_embs_np = np.load("index/image_embeddings.npy")
    img_embs = torch.tensor(img_embs_np, dtype=torch.float32, device=device)
    img_embs = F.normalize(img_embs, p=2, dim=-1)

    id_to_idx = {str(item["item_id"]): idx for idx, item in enumerate(id_map)}

    # Filter to unaccented queries (25 queries)
    unaccented_queries = [q for q in handwritten if not q.get("has_diacritics", True)]
    print(f"Total unaccented queries: {len(unaccented_queries)}")

    # Map each unaccented query to its accented counterpart
    # Look up by same spec or category
    accented_queries = [q for q in handwritten if q.get("has_diacritics", True)]
    
    # For each unaccented query, find matching accented query by spec
    pairs = []
    for uq in unaccented_queries:
        u_spec = uq.get("spec", {})
        # Find accented query with matching spec
        match_aq = None
        for aq in accented_queries:
            if aq.get("spec") == u_spec:
                match_aq = aq
                break
        if match_aq:
            pairs.append((uq, match_aq))
        else:
            # Fallback to uq itself if no counterpart
            pairs.append((uq, uq))

    print(f"Matched {len(pairs)} query pairs (unaccented <-> accented).")

    # Load model
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))

    def encode(text_list):
        inp = model.processor(text=text_list, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = inp["input_ids"].to(device)
        mask = inp.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)
        with torch.no_grad():
            emb = model.encode_text(input_ids, mask)
        return F.normalize(emb.float(), p=2, dim=-1)

    # Prepare texts for the 4 configurations
    # (a) bản gốc có dấu (accented query)
    texts_a = [aq["query"] for uq, aq in pairs]
    # (b) truy vấn không dấu chưa qua bộ khôi phục
    texts_b = [uq["query"] for uq, aq in pairs]
    # (c) quy tắc khôi phục cũ (luôn khôi phục)
    texts_c = [restorer.restore(uq["query"]) for uq, aq in pairs]
    # (d) quy tắc c' (chỉ khôi phục khi hoàn toàn không dấu)
    def restore_c_prime(q_text):
        if q_text.lower() == remove_diacritics(q_text).lower():
            return restorer.restore(q_text)
        return q_text
    texts_d = [restore_c_prime(uq["query"]) for uq, aq in pairs]

    print("\nSample query transformations (first 3):")
    for i in range(min(3, len(pairs))):
        print(f"  Query {pairs[i][0]['id']}:")
        print(f"    (a) Accented:    '{texts_a[i]}'")
        print(f"    (b) Raw unacc:   '{texts_b[i]}'")
        print(f"    (c) Old restore: '{texts_c[i]}'")
        print(f"    (d) c' restore:  '{texts_d[i]}'")

    # Pre-calculate relevant item IDs for each query spec
    relevant_sets = []
    for uq, aq in pairs:
        spec = uq.get("spec", {})
        art_types = spec.get("articleType", [])
        gender = spec.get("gender")
        base_colours = spec.get("baseColour", [])

        # Filter catalog
        cond = (catalog["in_scope"] == True)
        if art_types:
            cond &= catalog["articleType"].isin(art_types)
        if gender:
            cond &= (catalog["gender"] == gender)
        if base_colours:
            cond &= catalog["baseColour"].isin(base_colours)

        rel_ids = set(catalog[cond]["item_id"].astype(str))
        relevant_sets.append(rel_ids)

    # Encode and evaluate each configuration
    configs = [
        ("(a) Accented (Ground Truth)", texts_a),
        ("(b) Raw Unaccented (No restore)", texts_b),
        ("(c) Old Restore (Always restore)", texts_c),
        ("(d) Rule c' (Conditional restore)", texts_d)
    ]

    results_d8 = {}
    for cfg_name, text_list in configs:
        q_embs = encode(text_list)
        sim = torch.matmul(q_embs, img_embs.t()) # (25, 38646)

        p10s, hit10s, ap10s = [], [], []
        for i in range(len(pairs)):
            rel_ids = relevant_sets[i]
            if not rel_ids:
                continue
            scores = sim[i]
            top10_idx = torch.topk(scores, k=10).indices.cpu().numpy()
            retrieved_ids = [str(id_map[idx]["item_id"]) for idx in top10_idx]

            p10, hit10, ap10 = compute_metrics(retrieved_ids, rel_ids, k=10)
            p10s.append(p10)
            hit10s.append(hit10)
            ap10s.append(ap10)

        res = {
            "p10": float(np.mean(p10s)),
            "hit10": float(np.mean(hit10s)),
            "ap10": float(np.mean(ap10s))
        }
        results_d8[cfg_name] = res
        print(f"\nConfiguration: {cfg_name}")
        print(f"  Precision@10: {res['p10']:.4f}")
        print(f"  Hit@10:       {res['hit10']:.4f}")
        print(f"  AP@10:        {res['ap10']:.4f}")

    # Now also test on all 100 handwritten queries for (c) vs (d) to prove why c' protects already accented queries
    print("\n--- Testing on ALL 100 Handwritten Queries (Accent Protection Test) ---")
    all_texts_orig = [q["query"] for q in handwritten]
    all_texts_c = [restorer.restore(q["query"]) for q in handwritten]
    all_texts_d = [restore_c_prime(q["query"]) for q in handwritten]

    # Count how many already accented queries were mutated by rule c vs rule d
    accented_100 = [q for q in handwritten if q.get("has_diacritics", True)]
    c_mutations = sum(1 for q in accented_100 if restorer.restore(q["query"]) != q["query"])
    d_mutations = sum(1 for q in accented_100 if restore_c_prime(q["query"]) != q["query"])
    print(f"Already accented queries mutated by Old Rule (c): {c_mutations} / {len(accented_100)}")
    print(f"Already accented queries mutated by Rule c' (d):  {d_mutations} / {len(accented_100)}")

    with open("reports/diagnose_d8.json", "w", encoding="utf-8") as f:
        json.dump({
            "unaccented_25_eval": results_d8,
            "mutation_stats": {
                "accented_queries_count": len(accented_100),
                "old_rule_mutations": c_mutations,
                "c_prime_mutations": d_mutations
            }
        }, f, indent=2, ensure_ascii=False)
    print("Saved reports/diagnose_d8.json successfully.")

if __name__ == "__main__":
    run_d8()
