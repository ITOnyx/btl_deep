"""
Module: src.preprocess.pipeline
Đường code tiền xử lý DUY NHẤT (Luật 4) dùng chung cho:
- Tạo Index Gallery (Pha 4)
- Truy vấn Text->Image, Image->Image, CIR (Pha 5, 6)
- Đánh giá benchmark (Pha 5)
"""
import os
import pathlib
from typing import List, Union, Dict, Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

# Dọn dẹp biến môi trường TLS Windows nếu có
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)


class SigLIPPreprocessor:
    """
    Tiền xử lý thống nhất đa phương thức cho SigLIP SO400M.
    Đảm bảo 100% khớp quy chuẩn huấn luyện và tạo index (Luật 4).
    """
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-224",
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)

    def load_image(self, image_input: Union[str, pathlib.Path, Image.Image]) -> Image.Image:
        """Nạp và chuẩn hóa ảnh về định dạng PIL RGB."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        
        path_str = str(image_input)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Không tìm thấy tệp ảnh tại: {path_str}")
        
        with Image.open(path_str) as img:
            return img.convert("RGB")

    def preprocess_images(
        self,
        images: Union[List[Union[str, pathlib.Path, Image.Image]], Union[str, pathlib.Path, Image.Image]],
        device: Optional[str] = None,
        half_precision: bool = True
    ) -> torch.Tensor:
        """
        Tiền xử lý một danh sách ảnh hoặc một ảnh đơn.
        Trả về tensor pixel_values chuẩn bị cho vision tower.
        """
        target_device = device or self.device
        if not isinstance(images, list):
            images = [images]

        pil_images = [self.load_image(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(target_device)

        if half_precision and target_device == "cuda":
            pixel_values = pixel_values.half()

        return pixel_values

    def preprocess_texts(
        self,
        texts: Union[str, List[str]],
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tiền xử lý chuỗi văn bản (query hoặc modifier).
        Quy định: max_length=128, padding='max_length', truncation=True.
        """
        target_device = device or self.device
        if isinstance(texts, str):
            texts = [texts]

        cleaned_texts = [t.strip() for t in texts]
        inputs = self.processor(
            text=cleaned_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        res = {"input_ids": inputs["input_ids"].to(target_device)}
        if "attention_mask" in inputs:
            res["attention_mask"] = inputs["attention_mask"].to(target_device)
        return res

    def preprocess_single_image(
        self,
        image_input: Union[str, pathlib.Path, Image.Image],
        device: Optional[str] = None,
        half_precision: bool = True
    ) -> torch.Tensor:
        """Tiện ích tiền xử lý một ảnh đơn (trả về tensor có batch_dim = 1)."""
        return self.preprocess_images([image_input], device=device, half_precision=half_precision)

    def preprocess_single_text(
        self,
        text: str,
        device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Tiện ích tiền xử lý một câu truy vấn text đơn (batch_dim = 1)."""
        return self.preprocess_texts([text], device=device)
