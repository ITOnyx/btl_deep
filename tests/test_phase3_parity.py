"""
Unit tests cho Pha 3:
- Hash Guard (Luật 6)
- Position Embedding Expansion
- Preprocessor Pipeline (Luật 4)
- Parity Check (Cosine >= 0.999)
"""
import os
import json
import pytest
import torch
import numpy as np
from PIL import Image

# Dọn dẹp biến môi trường TLS Windows nếu có
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from src.model.siglip_model import (
    verify_checkpoint_hash,
    compute_file_sha256,
    expand_siglip_position_embeddings,
    HighlyEffectiveSigLIP,
    load_siglip_model
)
from src.preprocess.pipeline import SigLIPPreprocessor

CHECKPOINT_PATH = "data/checkpoints_siglip-vit/best_model.pt"
EXPECTED_HASH = "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d"


def test_hash_guard_success():
    """Kiểm tra Hash Guard vượt qua khi cung cấp đúng SHA-256."""
    assert os.path.exists(CHECKPOINT_PATH), "Checkpoint không tồn tại"
    assert verify_checkpoint_hash(CHECKPOINT_PATH, EXPECTED_HASH) is True


def test_hash_guard_tamper_detection():
    """Kiểm tra Hash Guard phát hiện sai lệch hash và ném lỗi ValueError."""
    fake_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    with pytest.raises(ValueError, match="HASH GUARD VIOLATION"):
        verify_checkpoint_hash(CHECKPOINT_PATH, fake_hash)


def test_expand_siglip_position_embeddings():
    """Kiểm tra logic mở rộng ma trận vị trí từ độ dài ban đầu lên 128 tokens."""
    dummy_model = torch.nn.Module()
    dummy_model.siglip = torch.nn.Module()
    dummy_model.siglip.text_model = torch.nn.Module()
    dummy_model.siglip.text_model.embeddings = torch.nn.Module()
    
    # Giả lập ma trận ban đầu 16x1152
    dummy_model.siglip.text_model.embeddings.position_embedding = torch.nn.Embedding(16, 1152)
    dummy_model.siglip.config = type("Config", (), {"text_config": type("TextConfig", (), {})()})()
    
    expand_siglip_position_embeddings(dummy_model, target_max_length=128)
    
    pos_emb = dummy_model.siglip.text_model.embeddings.position_embedding
    assert pos_emb.weight.shape == (128, 1152)
    assert dummy_model.siglip.text_model.embeddings.position_ids.shape == (1, 128)
    assert dummy_model.siglip.config.text_config.max_position_embeddings == 128


def test_preprocessor_image_and_text_shapes():
    """Kiểm tra SigLIPPreprocessor tạo đúng tensor shape cho ảnh và văn bản."""
    preprocessor = SigLIPPreprocessor(device="cpu")
    
    # Test Text
    text_inp = preprocessor.preprocess_texts(["áo thun nam màu trắng cotton"], device="cpu")
    assert "input_ids" in text_inp
    assert text_inp["input_ids"].shape == (1, 128)
    if "attention_mask" in text_inp:
        assert text_inp["attention_mask"].shape == (1, 128)
    
    # Test Image
    dummy_pil = Image.new("RGB", (300, 400), color=(120, 150, 180))
    img_tensor = preprocessor.preprocess_images([dummy_pil], device="cpu", half_precision=False)
    assert img_tensor.shape == (1, 3, 224, 224)


def test_parity_cosine_on_sample_images():
    """
    Kiểm tra Parity Check trên 5 ảnh mẫu:
    Cosine similarity so với embedding đối chứng phải >= 0.999 (Luật 4).
    """
    id_map_path = "data/_quarantine/legacy_rebuild_20261111/untouchable/index/id_map.json"
    gt_embs_path = "data/_quarantine/legacy_rebuild_20261111/untouchable/index/image_embeddings.npy"
    
    if not os.path.exists(id_map_path) or not os.path.exists(gt_embs_path):
        pytest.skip("Thiếu dữ liệu đối chứng index cũ để chạy parity test.")
        
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
        
    gt_embs = np.load(gt_embs_path, mmap_mode="r")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = SigLIPPreprocessor(device=device)
    model = load_siglip_model(
        checkpoint_path=CHECKPOINT_PATH,
        expected_hash=EXPECTED_HASH,
        device=device,
        fp16=(device == "cuda")
    )
    
    n_test = 5
    sample_paths = [id_map[i]["path"] for i in range(n_test)]
    pixel_values = preprocessor.preprocess_images(sample_paths, device=device, half_precision=(device == "cuda"))
    
    with torch.no_grad():
        new_embs = model.encode_image(pixel_values).float().cpu().numpy()
        
    new_norm = new_embs / np.linalg.norm(new_embs, axis=1, keepdims=True)
    gt_sample = np.array(gt_embs[:n_test], dtype=np.float32)
    gt_norm = gt_sample / np.linalg.norm(gt_sample, axis=1, keepdims=True)
    
    cosines = (new_norm * gt_norm).sum(axis=1)
    
    assert len(cosines) == n_test
    assert cosines.min() >= 0.999, f"Cosine similarity thấp hơn 0.999: min={cosines.min():.6f}"
