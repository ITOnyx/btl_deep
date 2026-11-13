"""
Module: src.serve.search_engine
Engine tìm kiếm đa phương thức thống nhất cho Pha 5 (Đánh giá) và Pha 6 (Ứng dụng Streamlit).
Hỗ trợ:
- Text -> Image (Model thuần vs Hệ thống + Soft Bonus)
- Image -> Image
- CIR (Composed Image Retrieval - Catalog & Upload)
- Mean-centering chống Hubness (Mục C.8)
- Hash Guard (Luật 6)
"""
import os
import json
import pathlib
from typing import List, Dict, Any, Optional, Union, Tuple

import yaml
import torch
import numpy as np
from PIL import Image

from src.model.siglip_model import load_siglip_model, verify_checkpoint_hash
from src.preprocess.pipeline import SigLIPPreprocessor
from src.index.faiss_index import FaissHNSWIndex
from src.serve.parser import VietnameseQueryParser, compute_soft_attribute_match


class ECommerceSearchEngine:
    """
    Search Engine cho Hệ thống tra cứu sản phẩm thời trang đa phương thức.
    """
    def __init__(
        self,
        index_dir: str = "index/test_gallery",
        config_path: str = "config.yaml",
        device: Optional[str] = None,
        use_mean_centering: bool = False,
        gamma: float = 0.3,
        ef_search: int = 128
    ):
        self.index_dir = pathlib.Path(index_dir)
        self.config_path = pathlib.Path(config_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_mean_centering = use_mean_centering
        self.gamma = gamma
        self.ef_search = ef_search

        # 1. Đọc config.yaml
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 2. Nạp manifest và kiểm tra Hash Guard (Luật 6)
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Không tìm thấy manifest.json tại {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        expected_hash = self.cfg["checkpoint"]["checkpoint_hash"]
        if self.manifest["checkpoint_hash"].lower() != expected_hash.lower():
            raise ValueError(
                f"HASH GUARD LỆCH (Luật 6): manifest={self.manifest['checkpoint_hash']} != config={expected_hash}"
            )

        # 3. Nạp id_map và embeddings
        with open(self.index_dir / "id_map.json", "r", encoding="utf-8") as f:
            self.id_map: List[Dict[str, Any]] = json.load(f)

        self.item_id_to_idx = {str(item["item_id"]): i for i, item in enumerate(self.id_map)}

        self.embeddings = np.load(self.index_dir / "image_embeddings.npy").astype(np.float32)
        # Tính gallery mean vector cho Mean-Centering (chống hubness)
        self.gallery_mean = np.mean(self.embeddings, axis=0, keepdims=True)

        if self.use_mean_centering:
            centered = self.embeddings - self.gallery_mean
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            centered_norm = centered / np.maximum(norms, 1e-12)
            self.hnsw = FaissHNSWIndex(dim=1152, M=32, ef_construction=200, ef_search=self.ef_search)
            self.hnsw.add(centered_norm)
        else:
            self.hnsw = FaissHNSWIndex.load(self.index_dir / "faiss_hnsw.bin", ef_search=self.ef_search)

        # 4. Nạp Model và Preprocessor
        ckpt_path = self.cfg["paths"]["checkpoint_file"]
        self.preprocessor = SigLIPPreprocessor(device=self.device)
        self.model = load_siglip_model(
            checkpoint_path=ckpt_path,
            expected_hash=expected_hash,
            device=self.device,
            fp16=(self.device == "cuda")
        )

        # 5. Parser tiếng Việt
        self.parser = VietnameseQueryParser()

    def _apply_mean_centering(self, vec: np.ndarray) -> np.ndarray:
        """Áp dụng mean-centering cho query vector nếu đang bật chế độ chống hubness."""
        if not self.use_mean_centering:
            return vec
        centered = vec - self.gallery_mean
        norms = np.linalg.norm(centered, axis=1, keepdims=True)
        return (centered / np.maximum(norms, 1e-12)).astype(np.float32)

    def encode_text_query(self, query_text: str) -> np.ndarray:
        """Mã hóa câu truy vấn văn bản sang vector 1152-d L2-normalized."""
        inputs = self.preprocessor.preprocess_single_text(query_text, device=self.device)
        with torch.no_grad():
            emb = self.model.encode_text(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask")
            )
            vec = emb.float().cpu().numpy()
        return self._apply_mean_centering(vec)

    def encode_image_query(self, image_input: Union[str, pathlib.Path, Image.Image]) -> np.ndarray:
        """Mã hóa ảnh tải lên sang vector 1152-d L2-normalized."""
        pixel_values = self.preprocessor.preprocess_single_image(
            image_input, device=self.device, half_precision=(self.device == "cuda")
        )
        with torch.no_grad():
            emb = self.model.encode_image(pixel_values)
            vec = emb.float().cpu().numpy()
        return self._apply_mean_centering(vec)

    def encode_cir_query(
        self,
        ref_image: Union[str, pathlib.Path, Image.Image, int],
        modifier_text: str,
        is_catalog_idx: bool = False
    ) -> np.ndarray:
        """
        Mã hóa truy vấn CIR (Ảnh tham chiếu + Văn bản chỉnh sửa).
        Nếu is_catalog_idx=True, ref_image là chỉ số trong catalog/index, lấy vector trực tiếp.
        """
        mod_inputs = self.preprocessor.preprocess_single_text(modifier_text, device=self.device)
        with torch.no_grad():
            mod_txt_emb = self.model.encode_text(
                input_ids=mod_inputs["input_ids"],
                attention_mask=mod_inputs.get("attention_mask")
            )
            if is_catalog_idx:
                ref_vec = self.embeddings[int(ref_image) : int(ref_image) + 1]
                ref_img_emb = torch.from_numpy(ref_vec).to(self.device)
                if self.device == "cuda":
                    ref_img_emb = ref_img_emb.half()
            else:
                pixel_values = self.preprocessor.preprocess_single_image(
                    ref_image, device=self.device, half_precision=(self.device == "cuda")
                )
                ref_img_emb = self.model.encode_image(pixel_values)

            cir_emb = self.model.cir_combiner(ref_img_emb, mod_txt_emb)
            vec = cir_emb.float().cpu().numpy()
        return self._apply_mean_centering(vec)

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        parsed_query: Optional[Dict[str, Any]] = None,
        mode: str = "model_only",
        candidate_pool: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm từ vector query.
        - mode == "model_only": xếp hạng thuần theo cosine similarity.
        - mode == "system": lấy top candidate_pool rồi re-rank bằng Soft Bonus từ parsed_query.
        """
        fetch_k = top_k if mode == "model_only" else max(top_k, candidate_pool)
        fetch_k = min(fetch_k, len(self.id_map))

        scores, indices = self.hnsw.search(query_vec, top_k=fetch_k)
        cand_scores = scores[0]
        cand_indices = indices[0]

        results = []
        for rank_idx, (idx, base_score) in enumerate(zip(cand_indices, cand_scores)):
            if idx < 0:
                continue
            meta = dict(self.id_map[int(idx)])
            final_score = float(base_score)
            bonus = 0.0

            if mode == "system" and parsed_query is not None and self.gamma > 0:
                match_frac = compute_soft_attribute_match(parsed_query, meta, gender_policy="hybrid")
                bonus = self.gamma * match_frac
                final_score += bonus

            meta["cosine_score"] = float(base_score)
            meta["attribute_bonus"] = float(bonus)
            meta["final_score"] = float(final_score)
            meta["index_pos"] = int(idx)
            results.append(meta)

        if mode == "system":
            results.sort(key=lambda x: x["final_score"], reverse=True)

        return results[:top_k]

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        mode: str = "system",
        candidate_pool: int = 100
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Thực hiện luồng Text -> Image."""
        q_vec = self.encode_text_query(query_text)
        parsed = self.parser.parse(query_text) if mode == "system" else {}
        results = self.search_by_vector(
            query_vec=q_vec,
            top_k=top_k,
            parsed_query=parsed,
            mode=mode,
            candidate_pool=candidate_pool
        )
        return results, parsed

    def search_image(
        self,
        image_input: Union[str, pathlib.Path, Image.Image],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Thực hiện luồng Image -> Image."""
        q_vec = self.encode_image_query(image_input)
        return self.search_by_vector(query_vec=q_vec, top_k=top_k, mode="model_only")

    def search_cir(
        self,
        ref_image: Union[str, pathlib.Path, Image.Image, int],
        modifier_text: str,
        top_k: int = 10,
        mode: str = "system",
        is_catalog_idx: bool = False,
        candidate_pool: int = 100
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Thực hiện luồng CIR (Composed Image Retrieval)."""
        q_vec = self.encode_cir_query(ref_image, modifier_text, is_catalog_idx=is_catalog_idx)
        parsed = self.parser.parse(modifier_text) if mode == "system" else {}
        results = self.search_by_vector(
            query_vec=q_vec,
            top_k=top_k,
            parsed_query=parsed,
            mode=mode,
            candidate_pool=candidate_pool
        )
        return results, parsed
