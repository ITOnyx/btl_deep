import sys
import json
import random
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from src.model_siglip import load_siglip_inference

def main():
    print("=== TASK 1 (b): RE-EMBEDDING 100 IDs VERIFICATION ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP model on {device}...")
    model_wrapper = load_siglip_inference(
        checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
        device=device,
        fp16=(device == "cuda")
    )
    processor = model_wrapper.processor

    eval_image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
    ])

    # Load raw embeddings and id_map
    raw_emb = np.load("index_v2/image_embeddings_v2.npy", mmap_mode="r")
    with open("index_v2/id_map_v2.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)

    id_to_idx = {str(item["item_id"] if isinstance(item, dict) else item): idx for idx, item in enumerate(id_map)}

    # Select 100 IDs with seed 42
    # Must include 39386, 15970, IDs after 43230 (idx 10651) and 42066 (idx 16039)
    random.seed(42)
    all_ids = [str(item["item_id"] if isinstance(item, dict) else item) for item in id_map]
    
    # Exclude 43230 and 42066 themselves
    valid_ids = [iid for iid in all_ids if iid not in ["43230", "42066"]]
    
    # Specific required IDs
    required_ids = ["39386", "15970"]
    # Add IDs after idx 10651 (say idx 12000) and after idx 16039 (say idx 25000, 35000)
    id_after_10651 = all_ids[12000]
    id_after_16039 = all_ids[25000]
    id_late = all_ids[35000]
    
    selected_set = set(required_ids + [id_after_10651, id_after_16039, id_late])
    remaining_needed = 100 - len(selected_set)
    sampled = random.sample([iid for iid in valid_ids if iid not in selected_set], remaining_needed)
    test_100_ids = list(selected_set) + sampled

    print(f"Selected {len(test_100_ids)} IDs. Checking image paths and embeddings...")

    cos_sims = []
    cos_sims_catalog_iloc = [] # what if someone took row i from in_scope catalog?

    df = pd.read_parquet("data/catalog/catalog.parquet")
    in_scope_df = df[df["in_scope"]].reset_index(drop=True)
    in_scope_id_to_iloc = {str(iid): idx for idx, iid in enumerate(in_scope_df["item_id"])}

    below_999 = []

    for iid in test_100_ids:
        idx_in_id_map = id_to_idx[iid]
        indexed_vec = np.array(raw_emb[idx_in_id_map], dtype=np.float32)
        norm_indexed = indexed_vec / np.linalg.norm(indexed_vec)

        img_p = Path(f"data/fashion-dataset/images/{iid}.jpg")
        if not img_p.exists():
            print(f"Warning: {img_p} missing")
            continue

        with Image.open(img_p) as img:
            img_rgb = img.convert("RGB")
            img_t = eval_image_transform(img_rgb)

        inputs = processor(images=img_t, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        if device == "cuda":
            pixel_values = pixel_values.half()

        with torch.no_grad():
            feat = model_wrapper.encode_image(pixel_values).cpu().numpy()[0]
        norm_feat = feat / np.linalg.norm(feat)

        cos = float(np.dot(norm_indexed, norm_feat))
        cos_sims.append(cos)
        if cos < 0.999:
            below_999.append((iid, idx_in_id_map, cos))

        # Also check what happens if someone used in_scope_df index:
        iloc_in_catalog = in_scope_id_to_iloc[iid]
        vec_at_catalog_iloc = np.array(raw_emb[iloc_in_catalog], dtype=np.float32)
        norm_cat = vec_at_catalog_iloc / np.linalg.norm(vec_at_catalog_iloc)
        cos_cat = float(np.dot(norm_cat, norm_feat))
        cos_sims_catalog_iloc.append((iid, idx_in_id_map, iloc_in_catalog, cos, cos_cat))

    print(f"\n--- Results on 100 IDs using id_map_v2 mapping ---")
    print(f"Min cosine:  {min(cos_sims):.6f}")
    print(f"Mean cosine: {np.mean(cos_sims):.6f}")
    print(f"Max cosine:  {max(cos_sims):.6f}")
    print(f"Count >= 0.999: {sum(1 for c in cos_sims if c >= 0.999)} / {len(cos_sims)}")
    
    if below_999:
        print(f"Items below 0.999: {below_999}")
    else:
        print("ALL 100 IDs have cosine >= 0.999! (Index is bit-faithful to pipeline)")

    # Print specific IDs required by user:
    for specific_id in ["39386", "15970", id_after_10651, id_after_16039]:
        idx_m = id_to_idx[specific_id]
        iloc_c = in_scope_id_to_iloc[specific_id]
        for item in cos_sims_catalog_iloc:
            if item[0] == specific_id:
                print(f"Specific ID {specific_id}: id_map idx={idx_m}, catalog in_scope iloc={iloc_c} | cosine={item[3]:.6f} (raw iloc cosine={item[4]:.6f})")

if __name__ == "__main__":
    main()
