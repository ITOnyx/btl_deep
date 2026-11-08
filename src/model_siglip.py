"""
SigLIP Vision and Text Tower for Phase 2 Inference and Indexing.
Loads strictly vision and text towers from checkpoint, omitting CIR combiner.
"""
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer

class SiglipVisionTextInference(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-224",
        max_length: int = 128,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.max_text_len = max_length

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.siglip = AutoModel.from_pretrained(model_name)

        # Expand position embeddings to max_length (128)
        if hasattr(self.siglip, "text_model") and hasattr(self.siglip.text_model, "embeddings"):
            pos_emb = self.siglip.text_model.embeddings.position_embedding
            old_max_len, dim = pos_emb.weight.shape
            if old_max_len < self.max_text_len:
                new_weight = torch.zeros(
                    (self.max_text_len, dim),
                    device=pos_emb.weight.device,
                    dtype=pos_emb.weight.dtype
                )
                new_weight[:old_max_len, :] = pos_emb.weight.data
                new_weight[old_max_len:, :] = pos_emb.weight.data[-1:]
                self.siglip.text_model.embeddings.position_embedding = nn.Embedding(
                    self.max_text_len, dim, _weight=new_weight
                )
                self.siglip.text_model.embeddings.position_ids = torch.arange(
                    self.max_text_len, device=pos_emb.weight.device
                ).expand((1, -1))
                if hasattr(self.siglip.config, "text_config"):
                    self.siglip.config.text_config.max_position_embeddings = self.max_text_len

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.siglip.vision_model(pixel_values=pixel_values)
        img_feats = vision_outputs[1] if len(vision_outputs) > 1 else vision_outputs.pooler_output
        return F.normalize(img_feats, p=2, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        text_outputs = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = text_outputs[1] if len(text_outputs) > 1 else text_outputs.pooler_output
        return F.normalize(txt_feats, p=2, dim=-1)


def load_siglip_inference(
    checkpoint_path: str = "data/checkpoints_siglip-vit/best_model.pt",
    device: str = "cuda",
    fp16: bool = True
) -> SiglipVisionTextInference:
    """
    Loads checkpoint weights for vision and text towers only.
    Frees CPU memory immediately after transfer to GPU.
    """
    model = SiglipVisionTextInference(
        model_name="google/siglip-so400m-patch14-224",
        max_length=128,
        device=device
    )

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Extract only siglip.* keys
    siglip_sd = {k[len("siglip."):]: v for k, v in sd.items() if k.startswith("siglip.")}
    del sd
    gc.collect()

    missing, unexpected = model.siglip.load_state_dict(siglip_sd, strict=True)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    del siglip_sd
    gc.collect()

    if fp16 and device != "cpu":
        model = model.half()
    model = model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    return model
