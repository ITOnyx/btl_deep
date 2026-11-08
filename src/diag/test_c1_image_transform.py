import numpy as np
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

# 1. Define eval_image_transform exactly matching training Cell 13
eval_image_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
])

# 2. Load model
from src.model_siglip import load_siglip_inference
model = load_siglip_inference(
    checkpoint_path="data/checkpoints_siglip-vit/best_model.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
    fp16=True
)
processor = model.processor

# 3. Load catalog to get 30 sample image paths
import pandas as pd
df = pd.read_parquet("data/catalog/catalog.parquet")
sample_30 = df[df["in_scope"]].head(30)

print(f"Testing image transform on {len(sample_30)} images...")

cosines_determinism = []
cosines_vs_bicubic = []

for idx, row in sample_30.iterrows():
    img_path = row["path"]
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        
        # Training Pipeline:
        img_bilinear = eval_image_transform(img_rgb)
        inp_train = processor(images=[img_bilinear], return_tensors="pt")
        with torch.no_grad():
            feat_train1 = model.encode_image(inp_train["pixel_values"].to(model.device, dtype=torch.float16))
            feat_train2 = model.encode_image(inp_train["pixel_values"].to(model.device, dtype=torch.float16))
        
        # Raw BICUBIC (v1):
        inp_bicubic = processor(images=[img_rgb], return_tensors="pt")
        with torch.no_grad():
            feat_bicubic = model.encode_image(inp_bicubic["pixel_values"].to(model.device, dtype=torch.float16))
        
        cos_det = F.cosine_similarity(feat_train1, feat_train2).item()
        cos_bic = F.cosine_similarity(feat_train1, feat_bicubic).item()
        cosines_determinism.append(cos_det)
        cosines_vs_bicubic.append(cos_bic)

print(f"Determinism min cosine: {min(cosines_determinism):.6f}, mean: {np.mean(cosines_determinism):.6f}")
print(f"Train (BILINEAR 224) vs v1 (BICUBIC) min cosine: {min(cosines_vs_bicubic):.6f}, mean: {np.mean(cosines_vs_bicubic):.6f}")
