"""
Search service for SigLIP Visual Search.
Enforces checkpoint hash verification against index manifest.
Supports:
  1. Text-to-Image Search (Vietnamese queries)
  2. Image-to-Image Search
  3. Optional post-filtering (Default is NO filtering)
"""
import os
import sys
import json
import time
import hashlib
import pathlib
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import faiss
from PIL import Image

sys.path.insert(0, ".")
from src.model_siglip import load_siglip_inference, SiglipVisionTextInference


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
    return h.hexdigest()


class VisualSearchEngine:
    def __init__(
        self,
        index_dir: str = "index",
        checkpoint_path: str = "data/checkpoints_siglip-vit/best_model.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ef_search: int = 128
    ):
        self.index_dir = pathlib.Path(index_dir)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.device = device
        self.ef_search = ef_search

        # 1. Verify Manifest and Checkpoint Hash Integrity
        self._verify_manifest()

        # 2. Load FAISS HNSW Index
        index_path = self.index_dir / "faiss_hnsw_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run index construction first.")
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        self.index.hnsw.efSearch = self.ef_search

        # 3. Load ID Map
        id_map_path = self.index_dir / "id_map.json"
        with open(id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        print(f"Loaded ID map with {len(self.id_map):,} items.")

        # 4. Load Model
        print(f"Loading SigLIP model on {self.device}...")
        self.model = load_siglip_inference(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            fp16=(self.device == "cuda")
        )
        self.processor = self.model.processor

    def _verify_manifest(self):
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Index manifest not found at {manifest_path}!")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        expected_hash = manifest.get("checkpoint_hash")
        print(f"Verifying checkpoint hash against manifest ({expected_hash[:12]}...)...")
        actual_hash = compute_file_sha256(str(self.checkpoint_path))

        if actual_hash != expected_hash:
            raise ValueError(
                f"CHECKPOINT HASH MISMATCH! Manifest expected {expected_hash}, but {self.checkpoint_path} has {actual_hash}. "
                "The index was built with a different checkpoint. Refusing to load!"
            )
        print("Checkpoint hash verified successfully.")

    def encode_text(self, text: str) -> np.ndarray:
        inp = self.processor(
            text=[text],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inp["input_ids"].to(self.device)
        attention_mask = inp.get("attention_mask").to(self.device) if "attention_mask" in inp else None

        with torch.no_grad():
            emb = self.model.encode_text(input_ids, attention_mask)
        return emb.float().cpu().numpy()  # [1, 1152]

    def encode_image(self, image_or_path: Union[str, Image.Image]) -> np.ndarray:
        if isinstance(image_or_path, str):
            with Image.open(image_or_path) as img:
                img = img.convert("RGB")
        else:
            img = image_or_path.convert("RGB")

        inp = self.processor(images=[img], return_tensors="pt")
        pix = inp["pixel_values"].to(self.device)
        if self.device == "cuda":
            pix = pix.half()

        with torch.no_grad():
            emb = self.model.encode_image(pix)
        return emb.float().cpu().numpy()  # [1, 1152]

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search nearest neighbors using query vector.
        Default is NO attribute filtering.
        If filters are provided, oversamples candidates (e.g. top 200) and filters.
        """
        # Ensure query is float32 and normalized
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        faiss.normalize_L2(query_vec)

        if filters:
            # Candidate pool for post-filtering
            candidate_k = min(len(self.id_map), max(200, top_k * 10))
            scores, indices = self.index.search(query_vec, candidate_k)
            scores = scores[0]
            indices = indices[0]

            results = []
            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.id_map):
                    continue
                item = self.id_map[idx]
                # Check filters
                match = True
                for k, v in filters.items():
                    if item.get(k) != v:
                        match = False
                        break
                if match:
                    res = dict(item)
                    res["score"] = float(score)
                    results.append(res)
                    if len(results) >= top_k:
                        break
            return results
        else:
            # Pure model retrieval (Rule 21: default is NO filtering)
            scores, indices = self.index.search(query_vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.id_map):
                    continue
                res = dict(self.id_map[idx])
                res["score"] = float(score)
                results.append(res)
            return results

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query_vec = self.encode_text(query_text)
        return self.search_by_vector(query_vec, top_k=top_k, filters=filters)

    def search_image(
        self,
        query_image_or_path: Union[str, Image.Image],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        query_vec = self.encode_image(query_image_or_path)
        return self.search_by_vector(query_vec, top_k=top_k, filters=filters)
