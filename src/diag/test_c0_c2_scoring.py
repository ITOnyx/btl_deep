import numpy as np
import json
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

# 1. Load testset
with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
    testset = json.load(f)

g_test_ids = [str(x) for x in testset["g_test_ids"]]
id_to_idx = {pid: i for i, pid in enumerate(g_test_ids)}

# 2. Load existing test gallery embeddings (v1 BICUBIC, for baseline test)
with open("index/id_map.json", "r", encoding="utf-8") as f:
    v1_id_map = json.load(f)
v1_map = {str(item["item_id"]): i for i, item in enumerate(v1_id_map)}

img_embs_all = np.load("index/image_embeddings.npy")
g_test_indices = [v1_map[pid] for pid in g_test_ids]
g_test_embs = img_embs_all[g_test_indices] # (1290, 1152)
# Ensure L2 normalized
g_test_embs = g_test_embs / np.linalg.norm(g_test_embs, axis=-1, keepdims=True)

# 3. Load stats
stats = np.load("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
mu_shared = stats["mu_shared"]
gamma = float(stats["cone_restore_gamma"]) # 0.75
beta = float(stats["csls_beta"]) # 0.28
k = int(stats["csls_k"]) # 7
val_query_bank = stats["val_query_bank"] # (2000, 1152)

# 4. Load training format queries for test items from product_metadata.json
with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
meta_dict = {str(item["product_id"]): item for item in meta}

queries = [meta_dict[pid]["search_text"] for pid in g_test_ids]
print(f"Loaded {len(queries)} training-format queries for test items.")

# Encode queries with model
from src.model_siglip import load_siglip_inference
model = load_siglip_inference(
    checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    fp16=True
)
processor = model.processor

q_embs_list = []
batch_size = 64
with torch.no_grad():
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i:i+batch_size]
        inp = processor(text=batch_q, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        feats = model.encode_text(inp["input_ids"].to(model.device), None)
        q_embs_list.append(feats.cpu().numpy())

q_embs = np.concatenate(q_embs_list, axis=0) # (1290, 1152)
q_embs = q_embs / np.linalg.norm(q_embs, axis=-1, keepdims=True)

print("Computed query embeddings shape:", q_embs.shape)

# Mode 1: Raw Cosine
sims_raw = np.dot(q_embs, g_test_embs.T) # (1290, 1290)

# Mode 2: Mean-centering with mu_shared and gamma
q_centered = q_embs - gamma * mu_shared
q_centered = q_centered / np.linalg.norm(q_centered, axis=-1, keepdims=True)

g_centered = g_test_embs - gamma * mu_shared
g_centered = g_centered / np.linalg.norm(g_centered, axis=-1, keepdims=True)

sims_centered = np.dot(q_centered, g_centered.T)

# Mode 3: CSLS on sims_raw
topk_g_for_q = np.sort(sims_raw, axis=1)[:, -k:] # top-k largest
r_q = np.mean(topk_g_for_q, axis=1, keepdims=True) # (1290, 1)

sims_g_to_bank = np.dot(g_test_embs, val_query_bank.T) # (1290, 2000)
topk_bank_for_g = np.sort(sims_g_to_bank, axis=1)[:, -k:]
r_g = np.mean(topk_bank_for_g, axis=1, keepdims=True).T # (1, 1290)

sims_csls = 2 * sims_raw - beta * (r_q + r_g)

# Also Mode 4: CSLS on centered
topk_g_centered = np.sort(sims_centered, axis=1)[:, -k:]
r_q_c = np.mean(topk_g_centered, axis=1, keepdims=True)
bank_centered = val_query_bank - gamma * mu_shared
bank_centered = bank_centered / np.linalg.norm(bank_centered, axis=-1, keepdims=True)
sims_g_bank_c = np.dot(g_centered, bank_centered.T)
r_g_c = np.mean(np.sort(sims_g_bank_c, axis=1)[:, -k:], axis=1, keepdims=True).T
sims_csls_centered = 2 * sims_centered - beta * (r_q_c + r_g_c)

def eval_sims(sims_mat, name):
    ranks = []
    r1, r5, r10 = 0, 0, 0
    N = len(sims_mat)
    for i in range(N):
        row = sims_mat[i]
        target_score = row[i]
        rank = int(np.sum(row > target_score)) + 1
        ranks.append(rank)
        if rank <= 1: r1 += 1
        if rank <= 5: r5 += 1
        if rank <= 10: r10 += 1
    
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    print(f"\n[{name}]")
    print(f"  Recall@1 : {r1 / N * 100:.2f}% ({r1}/{N})")
    print(f"  Recall@5 : {r5 / N * 100:.2f}% ({r5}/{N})")
    print(f"  Recall@10: {r10 / N * 100:.2f}% ({r10}/{N})")
    print(f"  MRR      : {mrr:.4f}")
    print(f"  Median   : {np.median(ranks):.1f}")
    print(f"  Mean     : {np.mean(ranks):.2f}")

eval_sims(sims_raw, "Mode 1: Raw Cosine (Training Format Queries, v1 Images)")
eval_sims(sims_centered, "Mode 2: Mean-Centered (mu_shared, gamma=0.75)")
eval_sims(sims_csls, "Mode 3: CSLS on Raw (beta=0.28, k=7, val_query_bank)")
eval_sims(sims_csls_centered, "Mode 4: CSLS on Mean-Centered")
