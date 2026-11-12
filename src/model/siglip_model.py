"""
Module: src.model.siglip_model
Kiến trúc SigLIP SO400M và bộ kết hợp CIR (UpgradedAttentionCIRCombiner).
Tuân thủ Luật 1 (Không sửa checkpoint, không huấn luyện) và Luật 6 (Hash Guard).
"""
import os
import gc
import hashlib
import pathlib
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Xóa biến môi trường bị lỗi đường dẫn TLS trên Windows nếu có
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)


def compute_file_sha256(filepath: str | pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    """Tính toán SHA-256 checksum của một tệp."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_checkpoint_hash(checkpoint_path: str | pathlib.Path, expected_hash: str) -> bool:
    """
    Xác minh Hash Guard (Luật 6):
    So sánh SHA-256 thực tế của checkpoint với chuỗi hash kỳ vọng từ config.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy tệp checkpoint tại: {checkpoint_path}")
    
    actual_hash = compute_file_sha256(checkpoint_path)
    expected_clean = expected_hash.strip().lower()
    if actual_hash != expected_clean:
        raise ValueError(
            f"❌ HASH GUARD VIOLATION (Luật 6)! "
            f"Checkpoint SHA-256 không khớp: "
            f"Thực tế = '{actual_hash}' != Kỳ vọng = '{expected_clean}'"
        )
    return True


def expand_siglip_position_embeddings(model: nn.Module, target_max_length: int = 128) -> None:
    """
    Mở rộng ma trận nhúng vị trí (position embeddings) của SigLIP Text Model từ 64/16 lên 128 tokens
    để tương thích tuyệt đối với tham số huấn luyện max_length = 128.
    (Trích xuất từ Cell 27 của notebook huấn luyện gốc).
    """
    if not hasattr(model, "siglip") or not hasattr(model.siglip, "text_model"):
        # Trường hợp gọi trực tiếp trên text_model
        text_model = getattr(model, "text_model", model)
    else:
        text_model = model.siglip.text_model

    if not hasattr(text_model, "embeddings") or not hasattr(text_model.embeddings, "position_embedding"):
        return

    pos_emb = text_model.embeddings.position_embedding
    old_max_len, dim = pos_emb.weight.shape

    if old_max_len >= target_max_length:
        return

    new_weight = torch.zeros((target_max_length, dim), device=pos_emb.weight.device, dtype=pos_emb.weight.dtype)
    new_weight[:old_max_len, :] = pos_emb.weight.data
    new_weight[old_max_len:, :] = pos_emb.weight.data[-1:]

    text_model.embeddings.position_embedding = nn.Embedding(target_max_length, dim, _weight=new_weight)
    text_model.embeddings.position_ids = torch.arange(target_max_length, device=pos_emb.weight.device).expand((1, -1))

    if hasattr(model, "siglip") and hasattr(model.siglip, "config") and hasattr(model.siglip.config, "text_config"):
        model.siglip.config.text_config.max_position_embeddings = target_max_length
    if hasattr(model, "max_text_len"):
        model.max_text_len = target_max_length


class UpgradedAttentionCIRCombiner(nn.Module):
    """
    Mô-đun kết hợp CIR sử dụng Multi-head Cross-Attention (Cell 19 notebook).
    Hòa trộn đặc trưng ảnh tham chiếu (ref image) và chuỗi text chỉnh sửa (modifier text).
    """
    def __init__(
        self,
        feature_dim: int = 1152,
        projection_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
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


class HighlyEffectiveSigLIP(nn.Module):
    """
    Kiến trúc mô hình hoàn chỉnh phục vụ suy luận SigLIP E-commerce.
    Bao gồm SigLIP Vision & Text Towers và CIR Combiner.
    """
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-224",
        max_length: int = 128,
        projection_dim: int = 1024
    ):
        super().__init__()
        self.model_name = model_name
        self.max_text_len = max_length

        self.siglip = AutoModel.from_pretrained(model_name)
        if hasattr(self.siglip.config, "text_config"):
            self.embed_dim = self.siglip.config.text_config.hidden_size
        else:
            self.embed_dim = 1152

        self.cir_combiner = UpgradedAttentionCIRCombiner(
            feature_dim=self.embed_dim,
            projection_dim=projection_dim
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Trích xuất embedding ảnh, chuẩn hóa L2."""
        vision_outputs = self.siglip.vision_model(pixel_values=pixel_values)
        img_feats = vision_outputs[1] if len(vision_outputs) > 1 else vision_outputs.pooler_output
        return F.normalize(img_feats, p=2, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Trích xuất embedding văn bản, chuẩn hóa L2."""
        text_outputs = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = text_outputs[1] if len(text_outputs) > 1 else text_outputs.pooler_output
        return F.normalize(txt_feats, p=2, dim=-1)

    def compose_query(
        self,
        ref_pixel_values: torch.Tensor,
        mod_input_ids: torch.Tensor,
        mod_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Kết hợp ảnh tham chiếu và text modifier qua CIR Combiner."""
        ref_img_emb = self.encode_image(ref_pixel_values)
        mod_txt_emb = self.encode_text(mod_input_ids, mod_attention_mask)
        return self.cir_combiner(ref_img_emb, mod_txt_emb)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        img_emb = self.encode_image(pixel_values) if pixel_values is not None else None
        txt_emb = self.encode_text(input_ids, attention_mask) if input_ids is not None else None
        return img_emb, txt_emb


def load_siglip_model(
    checkpoint_path: str = "data/checkpoints_siglip-vit/best_model.pt",
    expected_hash: Optional[str] = "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d",
    device: str = "cuda",
    fp16: bool = True,
    model_name: str = "google/siglip-so400m-patch14-224"
) -> HighlyEffectiveSigLIP:
    """
    Nạp checkpoint SigLIP tuân thủ Hash Guard.
    """
    if expected_hash:
        verify_checkpoint_hash(checkpoint_path, expected_hash)

    model = HighlyEffectiveSigLIP(model_name=model_name, max_length=128)
    expand_siglip_position_embeddings(model, target_max_length=128)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(f"Lỗi nạp weights: missing={len(missing)}, unexpected={len(unexpected)}")

    del state_dict
    gc.collect()

    if fp16 and device == "cuda":
        model = model.half()
    
    model = model.to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    return model
