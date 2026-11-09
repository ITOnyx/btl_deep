"""
Diagnosis D0 and D5:
- D0: True rank calculation (1..N) without capping at 999999 for Q1 paired text->image on G_test (1,290) and G_full (38,646).
  Median rank, mean rank. P@min(R, 10) for attribute queries.
- D5: Hubness analysis. Top 20 gallery items appearing in Q1 top 10. Cosine distributions.
  Test mean-centering with empirical means and with vit_zero_train_stats.npz.
"""
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_d0_d5():
    print("--- Running D0 & D5 Diagnosis ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load catalog, testset, and stored embeddings
    catalog = pd.read_parquet("data/catalog/catalog.parquet")
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    with open("index/id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)

    # Map item_id (str) -> gallery index
    id_to_idx = {str(item["item_id"]): idx for idx, item in enumerate(id_map)}
    
    # Load all gallery embeddings (G_full: 38,646, dim: 1152)
    img_embs_np = np.load("index/image_embeddings.npy")
    img_embs = torch.tensor(img_embs_np, dtype=torch.float32, device=device)
    img_embs = F.normalize(img_embs, p=2, dim=-1) # shape: (38646, 1152)

    # G_test indices
    g_test_ids = [str(x) for x in testset["g_test_ids"]]
    g_test_indices = torch.tensor([id_to_idx[x] for x in g_test_ids], dtype=torch.long, device=device)
    img_embs_test = img_embs[g_test_indices] # shape: (1290, 1152)
    id_to_test_idx = {x: idx for idx, x in enumerate(g_test_ids)}

    # Load Q1 queries
    q1_queries = testset["q1_paired_queries"]
    print(f"Total Q1 queries: {len(q1_queries)}")

    # Load SigLIP model to encode Q1 queries (or check if already encoded)
    from src.model_siglip import load_siglip_inference
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))

    # Encode Q1 captions
    captions = [q["query_text"] for q in q1_queries]
    batch_size = 64
    txt_embs_list = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        inp = model.processor(text=batch, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = inp["input_ids"].to(device)
        attn_mask = inp.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        with torch.no_grad():
            emb = model.encode_text(input_ids, attn_mask)
            txt_embs_list.append(emb.float())
    q1_txt_embs = torch.cat(txt_embs_list, dim=0) # shape: (1290, 1152)
    q1_txt_embs = F.normalize(q1_txt_embs, p=2, dim=-1)

    # Compute similarity matrices
    # (1290, 1290) for G_test
    sim_test = torch.matmul(q1_txt_embs, img_embs_test.t()) # (1290, 1290)
    # (1290, 38646) for G_full
    sim_full = torch.matmul(q1_txt_embs, img_embs.t()) # (1290, 38646)

    # D0: Compute exact true rank for each query
    # For G_test: target is at id_to_test_idx[q["target_item_id"]]
    ranks_test = []
    top10_hits_test = 0
    top10_appearances = Counter()
    sim_targets = []
    sim_top1s = []

    for i, q in enumerate(q1_queries):
        tar_id = str(q["target_id"])
        tar_test_idx = id_to_test_idx[tar_id]
        scores = sim_test[i]
        tar_score = scores[tar_test_idx].item()
        sim_targets.append(tar_score)
        
        # Rank is count of items with score > tar_score + 1
        rank = (scores > tar_score).sum().item() + 1
        ranks_test.append(rank)
        if rank <= 10:
            top10_hits_test += 1

        # Track top 10 items for hubness
        top10_indices = torch.topk(scores, k=10).indices.cpu().numpy()
        for idx in top10_indices:
            top10_appearances[g_test_ids[idx]] += 1
        sim_top1s.append(scores[top10_indices[0]].item())

    # For G_full: target is at id_to_idx[q["target_id"]]
    ranks_full = []
    top10_hits_full = 0
    top10_appearances_full = Counter()

    for i, q in enumerate(q1_queries):
        tar_id = str(q["target_id"])
        tar_full_idx = id_to_idx[tar_id]
        scores = sim_full[i]
        tar_score = scores[tar_full_idx].item()
        rank = (scores > tar_score).sum().item() + 1
        ranks_full.append(rank)
        if rank <= 10:
            top10_hits_full += 1

        top10_indices_full = torch.topk(scores, k=10).indices.cpu().numpy()
        for idx in top10_indices_full:
            item_id = id_map[idx]["item_id"]
            top10_appearances_full[str(item_id)] += 1

    ranks_test = np.array(ranks_test)
    ranks_full = np.array(ranks_full)

    print("\n=== D0: EXACT RANK STATISTICS (PAIRED Q1) ===")
    print(f"G_test (N={len(ranks_test)}):")
    print(f"  Recall@1:  {(ranks_test == 1).mean():.4f}")
    print(f"  Recall@5:  {(ranks_test <= 5).mean():.4f}")
    print(f"  Recall@10: {(ranks_test <= 10).mean():.4f}")
    print(f"  Median Rank: {np.median(ranks_test):.1f} / {len(g_test_ids)}")
    print(f"  Mean Rank:   {np.mean(ranks_test):.2f} / {len(g_test_ids)}")
    print(f"  p90 Rank:    {np.percentile(ranks_test, 90):.1f}")
    print(f"  p95 Rank:    {np.percentile(ranks_test, 95):.1f}")
    print(f"  Max Rank:    {np.max(ranks_test)}")

    print(f"\nG_full (N={len(ranks_full)}):")
    print(f"  Recall@1:  {(ranks_full == 1).mean():.4f}")
    print(f"  Recall@5:  {(ranks_full <= 5).mean():.4f}")
    print(f"  Recall@10: {(ranks_full <= 10).mean():.4f}")
    print(f"  Median Rank: {np.median(ranks_full):.1f} / 38,646")
    print(f"  Mean Rank:   {np.mean(ranks_full):.2f} / 38,646")
    print(f"  p90 Rank:    {np.percentile(ranks_full, 90):.1f}")
    print(f"  Max Rank:    {np.max(ranks_full)}")

    # Attribute queries: P@min(R, 10)
    print("\n=== D0: ATTRIBUTE QUERIES P@min(R, 10) ===")
    q2_queries = testset["q2_attribute_queries"]
    cat_lookup = catalog.set_index(catalog["item_id"].astype(str))
    
    # We will compute P@min(R, 10) for Q2 across G_test and G_full
    # Encode Q2 queries
    q2_texts = [q["query_text"] for q in q2_queries]
    q2_embs_list = []
    for i in range(0, len(q2_texts), batch_size):
        batch = q2_texts[i:i+batch_size]
        inp = model.processor(text=batch, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        input_ids = inp["input_ids"].to(device)
        attn_mask = inp.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        with torch.no_grad():
            emb = model.encode_text(input_ids, attn_mask)
            q2_embs_list.append(emb.float())
    q2_txt_embs = torch.cat(q2_embs_list, dim=0)
    q2_txt_embs = F.normalize(q2_txt_embs, p=2, dim=-1)

    sim_q2_test = torch.matmul(q2_txt_embs, img_embs_test.t()) # (61, 1290)
    sim_q2_full = torch.matmul(q2_txt_embs, img_embs.t()) # (61, 38646)

    # For G_test
    p10_list_test = []
    p_min_r10_list_test = []
    for i, q in enumerate(q2_queries):
        rel_strict = set(q["target_ids_strict"]["g_test"])
        R = len(rel_strict)
        if R == 0:
            continue
        scores = sim_q2_test[i]
        top10_idx = torch.topk(scores, k=10).indices.cpu().numpy()
        retrieved_ids = [g_test_ids[idx] for idx in top10_idx]
        hits = sum(1 for rid in retrieved_ids if rid in rel_strict)
        p10 = hits / 10.0
        p_min_r10 = hits / min(R, 10)
        p10_list_test.append(p10)
        p_min_r10_list_test.append(p_min_r10)

    # For G_full
    p10_list_full = []
    p_min_r10_list_full = []
    for i, q in enumerate(q2_queries):
        rel_strict = set(q["target_ids_strict"]["g_full"])
        R = len(rel_strict)
        if R == 0:
            continue
        scores = sim_q2_full[i]
        top10_idx = torch.topk(scores, k=10).indices.cpu().numpy()
        retrieved_ids = [str(id_map[idx]["item_id"]) for idx in top10_idx]
        hits = sum(1 for rid in retrieved_ids if rid in rel_strict)
        p10 = hits / 10.0
        p_min_r10 = hits / min(R, 10)
        p10_list_full.append(p10)
        p_min_r10_list_full.append(p_min_r10)

    print(f"G_test Strict: P@10 = {np.mean(p10_list_test):.4f} | P@min(R, 10) = {np.mean(p_min_r10_list_test):.4f}")
    print(f"G_full Strict: P@10 = {np.mean(p10_list_full):.4f} | P@min(R, 10) = {np.mean(p_min_r10_list_full):.4f}")

    # D5: Hubness Analysis
    print("\n=== D5: HUBNESS ANALYSIS ===")
    print(f"Top 20 most frequent gallery items in Q1 top-10 on G_test (out of {len(q1_queries)} queries):")
    top20_hubs = top10_appearances.most_common(20)
    
    hub_details = []
    for item_id, count in top20_hubs:
        row = cat_lookup.loc[item_id] if item_id in cat_lookup.index else None
        art_type = row["articleType"] if row is not None else "Unknown"
        gender = row["gender"] if row is not None else "Unknown"
        colour = row["baseColour"] if row is not None else "Unknown"
        path_img = row["path"] if row is not None else "Unknown"
        freq = count / len(q1_queries) * 100.0
        print(f"  ID {item_id:>6s} | Count: {count:>4d} ({freq:5.1f}%) | {art_type:20s} | {gender:8s} | {colour:12s} | {path_img}")
        hub_details.append({
            "item_id": item_id,
            "count": count,
            "freq_pct": round(freq, 2),
            "articleType": art_type,
            "gender": gender,
            "baseColour": colour,
            "path": path_img
        })

    # Cosine distributions: Target vs Top-1 vs Hub
    top1_hub_id = top20_hubs[0][0]
    top1_hub_test_idx = id_to_test_idx[top1_hub_id]
    sim_hub = sim_test[:, top1_hub_test_idx].cpu().numpy()

    print("\nCosine similarity distributions across 1,290 Q1 queries:")
    print(f"  Target score:    Min={np.min(sim_targets):.4f}, Mean={np.mean(sim_targets):.4f}, Median={np.median(sim_targets):.4f}, Max={np.max(sim_targets):.4f}")
    print(f"  Top-1 score:     Min={np.min(sim_top1s):.4f}, Mean={np.mean(sim_top1s):.4f}, Median={np.median(sim_top1s):.4f}, Max={np.max(sim_top1s):.4f}")
    print(f"  Top Hub ({top1_hub_id}) score: Min={np.min(sim_hub):.4f}, Mean={np.mean(sim_hub):.4f}, Median={np.median(sim_hub):.4f}, Max={np.max(sim_hub):.4f}")

    # TEST MEAN CENTERING
    print("\n--- Testing Mean-Centering on G_test ---")
    # Empirical mean vectors
    mu_img_emp = img_embs_test.mean(dim=0, keepdim=True)
    mu_txt_emp = q1_txt_embs.mean(dim=0, keepdim=True)

    img_centered = F.normalize(img_embs_test - mu_img_emp, p=2, dim=-1)
    txt_centered = F.normalize(q1_txt_embs - mu_txt_emp, p=2, dim=-1)

    sim_mc = torch.matmul(txt_centered, img_centered.t())
    ranks_mc = []
    for i, q in enumerate(q1_queries):
        tar_id = str(q["target_id"])
        tar_test_idx = id_to_test_idx[tar_id]
        scores = sim_mc[i]
        tar_score = scores[tar_test_idx].item()
        rank = (scores > tar_score).sum().item() + 1
        ranks_mc.append(rank)
    ranks_mc = np.array(ranks_mc)

    print("Empirical Mean-Centering Results on G_test:")
    print(f"  Recall@1:  {(ranks_mc == 1).mean():.4f}")
    print(f"  Recall@5:  {(ranks_mc <= 5).mean():.4f}")
    print(f"  Recall@10: {(ranks_mc <= 10).mean():.4f}")
    print(f"  Median Rank: {np.median(ranks_mc):.1f}")
    print(f"  Mean Rank:   {np.mean(ranks_mc):.2f}")

    # Also test with checkpoint vit_zero_train_stats.npz
    stats = np.load("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
    mu_shared = torch.tensor(stats["mu_shared"], device=device, dtype=torch.float32)
    gamma = float(stats["cone_restore_gamma"])
    csls_beta = float(stats["csls_beta"])
    csls_k = int(stats["csls_k"])

    img_stats_mc = F.normalize(img_embs_test - gamma * mu_shared, p=2, dim=-1)
    txt_stats_mc = F.normalize(q1_txt_embs - gamma * mu_shared, p=2, dim=-1)

    sim_stats_mc = torch.matmul(txt_stats_mc, img_stats_mc.t())
    ranks_smc = []
    for i, q in enumerate(q1_queries):
        tar_id = str(q["target_id"])
        tar_test_idx = id_to_test_idx[tar_id]
        scores = sim_stats_mc[i]
        tar_score = scores[tar_test_idx].item()
        rank = (scores > tar_score).sum().item() + 1
        ranks_smc.append(rank)
    ranks_smc = np.array(ranks_smc)

    print("\nCheckpoint stats (mu_shared + gamma=0.75) Results on G_test:")
    print(f"  Recall@1:  {(ranks_smc == 1).mean():.4f}")
    print(f"  Recall@5:  {(ranks_smc <= 5).mean():.4f}")
    print(f"  Recall@10: {(ranks_smc <= 10).mean():.4f}")
    print(f"  Median Rank: {np.median(ranks_smc):.1f}")
    print(f"  Mean Rank:   {np.mean(ranks_smc):.2f}")

    # CSLS calculation
    print(f"\nTesting CSLS (k={csls_k}, beta={csls_beta}) on sim_test:")
    # CSLS: sim_csls(q, g) = 2 * cos(q, g) - beta * (knn_sim(q) + knn_sim(g))
    knn_q = torch.topk(sim_test, k=csls_k, dim=1).values.mean(dim=1, keepdim=True) # (1290, 1)
    knn_g = torch.topk(sim_test, k=csls_k, dim=0).values.mean(dim=0, keepdim=True) # (1, 1290)
    sim_csls = 2.0 * sim_test - csls_beta * (knn_q + knn_g)

    ranks_csls = []
    for i, q in enumerate(q1_queries):
        tar_id = str(q["target_id"])
        tar_test_idx = id_to_test_idx[tar_id]
        scores = sim_csls[i]
        tar_score = scores[tar_test_idx].item()
        rank = (scores > tar_score).sum().item() + 1
        ranks_csls.append(rank)
    ranks_csls = np.array(ranks_csls)

    print("CSLS Results on G_test:")
    print(f"  Recall@1:  {(ranks_csls == 1).mean():.4f}")
    print(f"  Recall@5:  {(ranks_csls <= 5).mean():.4f}")
    print(f"  Recall@10: {(ranks_csls <= 10).mean():.4f}")
    print(f"  Median Rank: {np.median(ranks_csls):.1f}")
    print(f"  Mean Rank:   {np.mean(ranks_csls):.2f}")

    # Save summary data
    res_d0_d5 = {
        "ranks_test": {
            "median": float(np.median(ranks_test)),
            "mean": float(np.mean(ranks_test)),
            "r1": float((ranks_test == 1).mean()),
            "r5": float((ranks_test <= 5).mean()),
            "r10": float((ranks_test <= 10).mean()),
            "p90": float(np.percentile(ranks_test, 90)),
            "max": int(np.max(ranks_test))
        },
        "ranks_full": {
            "median": float(np.median(ranks_full)),
            "mean": float(np.mean(ranks_full)),
            "r1": float((ranks_full == 1).mean()),
            "r5": float((ranks_full <= 5).mean()),
            "r10": float((ranks_full <= 10).mean()),
            "p90": float(np.percentile(ranks_full, 90)),
            "max": int(np.max(ranks_full))
        },
        "top20_hubs": hub_details,
        "cosine_dist": {
            "target": {"min": float(np.min(sim_targets)), "mean": float(np.mean(sim_targets)), "max": float(np.max(sim_targets))},
            "top1": {"min": float(np.min(sim_top1s)), "mean": float(np.mean(sim_top1s)), "max": float(np.max(sim_top1s))},
            "top_hub": {"min": float(np.min(sim_hub)), "mean": float(np.mean(sim_hub)), "max": float(np.max(sim_hub))}
        },
        "mean_centering": {
            "empirical": {"r10": float((ranks_mc <= 10).mean()), "median": float(np.median(ranks_mc)), "mean": float(np.mean(ranks_mc))},
            "checkpoint_stats": {"r10": float((ranks_smc <= 10).mean()), "median": float(np.median(ranks_smc)), "mean": float(np.mean(ranks_smc))},
            "csls": {"r10": float((ranks_csls <= 10).mean()), "median": float(np.median(ranks_csls)), "mean": float(np.mean(ranks_csls))}
        }
    }
    with open("reports/diagnose_d0_d5.json", "w", encoding="utf-8") as f:
        json.dump(res_d0_d5, f, indent=2, ensure_ascii=False)
    print("Saved reports/diagnose_d0_d5.json successfully.")

if __name__ == "__main__":
    run_d0_d5()
