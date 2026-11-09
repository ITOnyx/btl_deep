import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import pathlib

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from transformers import AutoModel, AutoProcessor
from src.model_siglip import load_siglip_inference

def compare_d3():
    # 1. Inspect training encode_text in notebook
    nb_path = pathlib.Path("notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Let's check how cell 34 and cell 29 encoded text during train
    print("Notebook tokenization pattern:")
    print("model.processor(text=[query_text], padding='max_length', truncation=True, max_length=128, return_tensors='pt')")

    # Load 30 test captions
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        testset = json.load(f)
    sample_30 = [q["query_text"] for q in testset["q1_paired_queries"][:30]]

    # Pipeline (i): our inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    our_model = load_siglip_inference("data/checkpoints_siglip-vit/best_model.pt", device=device, fp16=(device=="cuda"))

    # Pipeline (ii): exact notebook code definition
    # From cell 34:
    # def encode_text(self, input_ids, attention_mask=None):
    #     text_outputs = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
    #     txt_feats = text_outputs[1] if len(text_outputs) > 1 else text_outputs.pooler_output
    #     return torch.nn.functional.normalize(txt_feats, p=2, dim=-1)

    # Let's test both on sample_30
    processor = our_model.processor

    cosines = []
    for text in sample_30:
        # Our pipeline
        inp = processor(text=[text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        input_ids = inp["input_ids"].to(device)
        attention_mask = inp.get("attention_mask").to(device) if "attention_mask" in inp else None

        with torch.no_grad():
            # Our way
            emb_our = our_model.encode_text(input_ids, attention_mask)

            # Notebook way
            text_outputs = our_model.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            txt_feats = text_outputs[1] if len(text_outputs) > 1 else text_outputs.pooler_output
            emb_train = F.normalize(txt_feats, p=2, dim=-1)

        sim = F.cosine_similarity(emb_our, emb_train).item()
        cosines.append(sim)

    print(f"\n--- D3 TEXT ENCODING COSINE SIMILARITY (n=30) ---")
    print(f"Min cosine:  {min(cosines):.6f}")
    print(f"Mean cosine: {np.mean(cosines):.6f}")
    print(f"Max cosine:  {max(cosines):.6f}")
    print(f"All >= 0.999: {all(c >= 0.999 for c in cosines)}")

if __name__ == "__main__":
    compare_d3()
