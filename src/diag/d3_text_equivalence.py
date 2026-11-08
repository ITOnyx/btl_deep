"""
Script: src/diag/d3_text_equivalence.py
Chẩn đoán chi tiết theo yêu cầu:
1. In type(...) và is None của mọi biến được gọi .to() trước khi gọi.
2. Kiểm tra tokenizer, model, processor, các khóa đầu ra.
3. Đối chiếu và kiểm tra attention_mask và encode_text.
"""
import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model_siglip import load_siglip_inference

def run_diagnosis():
    print("=== BƯỚC 1: KIỂM TRA MÔ HÌNH VÀ HÀM NẠP ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")

    checkpoint_path = "data/checkpoints_siglip-vit/best_model.pt"
    model = load_siglip_inference(checkpoint_path, device=device, fp16=(device=="cuda"))

    print(f"type(model): {type(model)} | is None: {model is None}")
    print("\n--- named_children() mức đầu của model ---")
    for name, child in model.named_children():
        print(f"  Child '{name}': {type(child)}")

    print("\n=== BƯỚC 2: KIỂM TRA TOKENIZER / PROCESSOR ===")
    processor = model.processor
    tokenizer = getattr(model, "tokenizer", None)
    print(f"type(processor): {type(processor)} | is None: {processor is None}")
    print(f"type(tokenizer): {type(tokenizer)} | is None: {tokenizer is None}")

    sample_texts = [
        "áo thun nam màu xám",
        "đầm hoa nữ mùa hè dáng xòe"
    ]
    print(f"\nSample input texts: {sample_texts}")

    inputs = processor(text=sample_texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    print("\n--- Kiểm tra từng khóa trong dict đầu ra của processor ---")
    print(f"type(inputs): {type(inputs)}")
    print(f"Keys in inputs: {list(inputs.keys())}")
    for k, v in inputs.items():
        print(f"  Key '{k}': type={type(v)}, is None={v is None}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

    # Kiểm tra riêng attention_mask
    mask_in_dict = "attention_mask" in inputs
    mask_val = inputs.get("attention_mask")
    print(f"\n'attention_mask' in inputs: {mask_in_dict}")
    print(f"inputs.get('attention_mask'): type={type(mask_val)}, is None={mask_val is None}")

    print("\n=== BƯỚC 3: KIỂM TRA GỌI .to() TRÊN TỪNG BIẾN ===")
    # 1. model
    print(f"1. model: type={type(model)}, is None={model is None}")
    # 2. input_ids
    input_ids = inputs["input_ids"]
    print(f"2. input_ids: type={type(input_ids)}, is None={input_ids is None}")
    input_ids_cuda = input_ids.to(device)
    print(f"   -> input_ids.to(device) OK: shape={input_ids_cuda.shape}, device={input_ids_cuda.device}")

    # 3. attention_mask (nếu gọi .to())
    print(f"3. attention_mask: type={type(mask_val)}, is None={mask_val is None}")
    if mask_val is None:
        print("   -> CẢNH BÁO: inputs.get('attention_mask') là None!")
        print("   -> Code train: mask = inputs.get('attention_mask').to(...) IF mask is not None ELSE None.")
        print("   -> Với SigLIP so400m, processor KHÔNG trả về attention_mask. Gọi .to() trực tiếp sẽ gây AttributeError!")
    else:
        mask_cuda = mask_val.to(device)
        print(f"   -> mask_cuda OK: shape={mask_cuda.shape}")

    print("\n=== BƯỚC 4: THỰC THI ENCODE_TEXT THEO ĐÚNG CODE TRAIN ===")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = None

    with torch.no_grad():
        emb = model.encode_text(input_ids_cuda, attention_mask)
    print(f"Embedding shape: {emb.shape}, norm: {torch.norm(emb, dim=-1)}")
    print("✅ Encode text thành công mà không có lỗi!")

if __name__ == "__main__":
    run_diagnosis()
