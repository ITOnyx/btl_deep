import sys
import json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from src.model_siglip_cir import load_trained_siglip_cir
from src.scoring import Scorer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing CIR Retrieval on {device}...")
    model = load_trained_siglip_cir('data/checkpoints_siglip-vit/best_model.pt', device=device, fp16=(device=='cuda'))
    scorer = Scorer('data/checkpoints_siglip-vit/vit_zero_train_stats.npz')

    # Load id_map
    with open('index_v2/id_map_v2.json', 'r', encoding='utf-8') as f:
        id_map = json.load(f)
    id_map_ids = [str(x['item_id']) for x in id_map]

    df_full = pd.read_parquet('data/catalog/catalog.parquet')
    df = df_full[df_full['in_scope'] == True].reset_index(drop=True)
    in_scope_id_set = set(df['item_id'].astype(str))

    # Align embeddings to in_scope df (38,644 items)
    valid_indices = [i for i, iid in enumerate(id_map_ids) if iid in in_scope_id_set]
    gallery_raw = np.load('index_v2/image_embeddings_v2.npy')[valid_indices]
    gallery_c = scorer.center_vectors(gallery_raw)
    print(f"Aligned gallery shape: {gallery_c.shape}, df shape: {df.shape}")

    sample_row = df[(df['articleType'] == 'Tshirts') & (df['baseColour'] == 'Red')].iloc[0]
    ref_id = sample_row['item_id']
    ref_path = sample_row['path']
    if not Path(ref_path).exists():
        ref_path = f"data/fashion-dataset/images/{Path(ref_path).name}"

    print(f"\nReference: ID {ref_id} | {sample_row['name']} | {sample_row['articleType']} | {sample_row['baseColour']}")

    img_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    im = Image.open(ref_path).convert('RGB')
    img_tensor = img_transform(im).unsqueeze(0).to(device)
    if device == 'cuda':
        img_tensor = img_tensor.half()

    modifier = 'màu xanh navy'
    print(f"Modifier text: '{modifier}'")

    tok = model.processor(text=[modifier], padding='max_length', max_length=128, truncation=True, return_tensors='pt')
    ids = tok['input_ids'].to(device)

    with torch.no_grad():
        cir_feat = model.compose_query(img_tensor, ids).cpu().numpy()[0]
        cir_feat_c = scorer.center_vectors(cir_feat[np.newaxis, :])[0]

    sims = np.dot(gallery_c, cir_feat_c)
    top5 = np.argsort(-sims)[:5]

    print("\nTop 5 CIR results:")
    for r, idx in enumerate(top5, 1):
        row = df.iloc[idx]
        print(f"  {r}. [{row['item_id']}] {row['name']} | {row['articleType']} | {row['baseColour']} | sim: {sims[idx]:.4f}")

if __name__ == "__main__":
    main()
