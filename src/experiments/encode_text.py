"""
Text Encoding module for Vietnamese E-commerce Visual Search.
Uses fine-tuned SigLIP text tower:
- padding="max_length", max_length=128 tokens
- L2-normalized output vectors (dim=1152)
- Native Vietnamese understanding (no VI->EN translation layer)
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, ".")
from src.model_siglip import load_siglip_inference, SiglipVisionTextInference


class TextEncoder:
    def __init__(
        self,
        checkpoint_path: str = "data/checkpoints_siglip-vit/best_model.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fp16: bool = True
    ):
        self.device = device
        self.fp16 = fp16 and (device != "cpu")
        self.model = load_siglip_inference(checkpoint_path=checkpoint_path, device=device, fp16=self.fp16)
        self.processor = self.model.processor
        self.max_length = 128

    def encode(self, texts, return_numpy: bool = True) -> np.ndarray:
        """
        Encode single text string or list of text strings into L2-normalized embeddings.
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        inp = self.processor(
            text=texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inp["input_ids"].to(self.device)
        attention_mask = inp.get("attention_mask").to(self.device) if "attention_mask" in inp else None

        with torch.no_grad():
            embs = self.model.encode_text(input_ids, attention_mask)

        if return_numpy:
            arr = embs.float().cpu().numpy()
            return arr[0] if is_single else arr
        return embs[0] if is_single else embs


def encode_query_text(text: str, encoder: TextEncoder = None) -> np.ndarray:
    """Convenience helper to encode a single query string."""
    if encoder is None:
        encoder = TextEncoder()
    return encoder.encode(text, return_numpy=True)
