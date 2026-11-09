import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_siglip import load_siglip_inference

def main():
    print("Precomputing product text embeddings for all 38,646 items...")
    t0 = time.time()

    # Load catalog and metadata
    df = pd.read_parquet("data/catalog/catalog.parquet")
    in_scope_df = df[df["in_scope"] == True].copy().reset_index(drop=True)
    N = len(in_scope_df)
    print(f"In-scope catalog items: {N}")

    with open("data/checkpoints_siglip-vit/product_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta_dict = {str(item["product_id"]): item["search_text"] for item in meta}

    texts = []
    for idx, row in in_scope_df.iterrows():
        item_id = str(row["item_id"])
        if item_id in meta_dict:
            texts.append(meta_dict[item_id])
        else:
            cat = f"{row.get('masterCategory', '')} > {row.get('subCategory', '')} > {row.get('articleType', '')}"
            title = str(row.get("name", ""))
            attrs = f"{row.get('gender', '')}, {row.get('baseColour', '')}"
            texts.append(f"[{cat}] {title} ({attrs})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP model on {device} (FP16)...")
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))

    print("Encoding product texts...")
    batch_size = 256
    embs_list = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            b_texts = texts[i:i+batch_size]
            inp = model.processor(text=b_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            feats = model.encode_text(inp["input_ids"].to(device), None)
            embs_list.append(feats.cpu().numpy())
            if (i // batch_size) % 30 == 0:
                print(f"  Processed {min(i+batch_size, N)}/{N}...")

    embs = np.vstack(embs_list)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embs = embs / norms

    out_path = Path("index_v2/text_embeddings_v2.npy")
    np.save(out_path, embs.astype(np.float32))
    print(f"Saved {embs.shape} embeddings to {out_path} in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
