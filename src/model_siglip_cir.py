"""
Multimodal SigLIP with Upgraded Attention CIR Combiner
Architecture implementation matching checkpoints_siglip-vit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class UpgradedAttentionCIRCombiner(nn.Module):
    """
    Cross-Attention CIR Combiner module.
    Blends reference image embedding and modifier text embedding into a target query embedding.
    """
    def __init__(self, feature_dim: int = 1152, projection_dim: int = 1024, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim

        self.text_proj = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_att = nn.LayerNorm(projection_dim)

        self.fusion = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, feature_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, 1),
            nn.Sigmoid()
        )
        self.norm_final = nn.LayerNorm(feature_dim)

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        img_feat_norm = F.normalize(img_feat, p=2, dim=-1)
        text_feat_norm = F.normalize(text_feat, p=2, dim=-1)

        p_img = self.image_proj(img_feat_norm)
        p_txt = self.text_proj(text_feat_norm)

        q = p_img.unsqueeze(1)
        k = p_txt.unsqueeze(1)
        v = p_txt.unsqueeze(1)
        att_out, _ = self.cross_attention(q, k, v)
        att_out = att_out.squeeze(1)

        p_combined = self.norm_att(p_img + att_out)
        fused = self.fusion(p_combined)

        concat_gate = torch.cat([p_img, p_txt], dim=-1)
        alpha = self.gate(concat_gate)

        out = fused + alpha * text_feat_norm + (1.0 - alpha) * img_feat_norm
        out = self.norm_final(out)
        return F.normalize(out, p=2, dim=-1)


class MultimodalSigLIPWithCIR(nn.Module):
    """
    SigLIP Backbone (google/siglip-so400m-patch14-224) + UpgradedAttentionCIRCombiner
    Embedding dimension: 1152. Max text position embeddings: 128 tokens.
    """
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-224",
        max_length: int = 128,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.max_text_len = max_length

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.siglip = AutoModel.from_pretrained(model_name)

        if hasattr(self.siglip.config, "text_config"):
            self.embed_dim = self.siglip.config.text_config.hidden_size
        else:
            self.embed_dim = 1152

        # Expand position embeddings if needed
        if hasattr(self.siglip, "text_model") and hasattr(self.siglip.text_model, "embeddings"):
            pos_emb = self.siglip.text_model.embeddings.position_embedding
            old_max_len, dim = pos_emb.weight.shape
            if old_max_len < self.max_text_len:
                new_weight = torch.zeros((self.max_text_len, dim), device=pos_emb.weight.device, dtype=pos_emb.weight.dtype)
                new_weight[:old_max_len, :] = pos_emb.weight.data
                new_weight[old_max_len:, :] = pos_emb.weight.data[-1:]
                self.siglip.text_model.embeddings.position_embedding = nn.Embedding(self.max_text_len, dim, _weight=new_weight)
                self.siglip.text_model.embeddings.position_ids = torch.arange(self.max_text_len, device=pos_emb.weight.device).expand((1, -1))
                if hasattr(self.siglip.config, "text_config"):
                    self.siglip.config.text_config.max_position_embeddings = self.max_text_len

        self.cir_combiner = UpgradedAttentionCIRCombiner(feature_dim=self.embed_dim, projection_dim=1024)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.siglip.vision_model(pixel_values=pixel_values)
        img_feats = vision_outputs[1] if len(vision_outputs) > 1 else vision_outputs.pooler_output
        return F.normalize(img_feats, p=2, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        text_outputs = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = text_outputs[1] if len(text_outputs) > 1 else text_outputs.pooler_output
        return F.normalize(txt_feats, p=2, dim=-1)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        img_emb = self.encode_image(pixel_values)
        txt_emb = self.encode_text(input_ids, attention_mask)
        return img_emb, txt_emb

    def compose_query(self, ref_pixels: torch.Tensor, mod_ids: torch.Tensor, mod_mask: torch.Tensor = None):
        ref_img_emb = self.encode_image(ref_pixels)
        mod_txt_emb = self.encode_text(mod_ids, mod_mask)
        return self.cir_combiner(ref_img_emb, mod_txt_emb)


def load_trained_siglip_cir(checkpoint_path: str, device: str = "cpu", fp16: bool = False):
    """Loads weights from checkpoint into MultimodalSigLIPWithCIR."""
    model = MultimodalSigLIPWithCIR(
        model_name="google/siglip-so400m-patch14-224",
        max_length=128,
        device=device
    )
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Filter or load directly
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    if fp16 and device != "cpu":
        model = model.half()
    model = model.to(device)
    model.eval()
    return model
