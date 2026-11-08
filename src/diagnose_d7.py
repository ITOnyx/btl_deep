import sys
import json
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model_siglip import load_siglip_inference

def compare_d7():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))
    processor = model.processor

    eval_image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
    ])

    # Load 30 test image paths from catalog
    import pandas as pd
    cat = pd.read_parquet("data/catalog/catalog.parquet")
    test_cat = cat[(cat["split"] == "test") & (cat["in_scope"] == True)]
    sample_30_paths = test_cat["path"].iloc[:30].tolist()

    cosines = []
    cosines_vs_stored = []

    # Also compare with our stored embeddings in image_embeddings.npy
    stored_embeddings = np.load("index/image_embeddings.npy")
    with open("index/id_map.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)
    id_to_stored_idx = {str(item["item_id"]): idx for idx, item in enumerate(id_map)}

    for i, p in enumerate(sample_30_paths):
        item_id = str(test_cat["item_id"].iloc[i])
        with Image.open(p) as raw_img:
            img_rgb = raw_img.convert("RGB")

            # Pipeline (i): our inference embedding (processor directly on PIL)
            inp_our = processor(images=[img_rgb], return_tensors="pt")
            pix_our = inp_our["pixel_values"].to(device)
            if device == "cuda":
                pix_our = pix_our.half()
            with torch.no_grad():
                emb_our = model.encode_image(pix_our)

            # Pipeline (ii): notebook training eval_image_transform + processor
            img_resized = eval_image_transform(img_rgb)
            inp_train = processor(images=[img_resized], return_tensors="pt")
            pix_train = inp_train["pixel_values"].to(device)
            if device == "cuda":
                pix_train = pix_train.half()
            with torch.no_grad():
                emb_train = model.encode_image(pix_train)

        sim = F.cosine_similarity(emb_our, emb_train).item()
        cosines.append(sim)

        # Compare with stored vector in image_embeddings.npy
        if item_id in id_to_stored_idx:
            stored_vec = torch.tensor(stored_embeddings[id_to_stored_idx[item_id]], device=device).unsqueeze(0)
            if device == "cuda":
                stored_vec = stored_vec.half()
            stored_sim = F.cosine_similarity(emb_our, stored_vec).item()
            cosines_vs_stored.append(stored_sim)

    print(f"\n--- D7 IMAGE PREPROCESSING COSINE SIMILARITY (n=30) ---")
    print(f"Cosine (Our pipeline vs Notebook eval_image_transform):")
    print(f"  Min:  {min(cosines):.6f}")
    print(f"  Mean: {np.mean(cosines):.6f}")
    print(f"  Max:  {max(cosines):.6f}")
    print(f"  All >= 0.999: {all(c >= 0.999 for c in cosines)}")

    if cosines_vs_stored:
        print(f"\nCosine (Our pipeline vs stored image_embeddings.npy):")
        print(f"  Min:  {min(cosines_vs_stored):.6f}")
        print(f"  Mean: {np.mean(cosines_vs_stored):.6f}")
        print(f"  Max:  {max(cosines_vs_stored):.6f}")
        print(f"  All >= 0.999: {all(c >= 0.999 for c in cosines_vs_stored)}")

if __name__ == "__main__":
    compare_d7()
