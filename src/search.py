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


def verify_manifest(
    index_dir: Union[str, pathlib.Path],
    checkpoint_path: Optional[Union[str, pathlib.Path]] = None,
    catalog_path: Optional[Union[str, pathlib.Path]] = None
) -> Dict[str, Any]:
    """
    Verifies that the manifest.json inside index_dir matches the sha256 hashes
    of the checkpoint and catalog files.
    Raises FileNotFoundError if manifest or target files do not exist.
    Raises ValueError if any hash does not match.
    """
    index_dir = pathlib.Path(index_dir)
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Index manifest not found at {manifest_path}!")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify Checkpoint Hash
    expected_ckpt_hash = manifest.get("checkpoint_hash")
    if expected_ckpt_hash:
        ckpt_to_check = pathlib.Path(checkpoint_path) if checkpoint_path else pathlib.Path(manifest.get("checkpoint_path", ""))
        if not ckpt_to_check.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_to_check} for hash verification!")
        actual_ckpt_hash = compute_file_sha256(str(ckpt_to_check))
        if actual_ckpt_hash != expected_ckpt_hash:
            raise ValueError(
                f"CHECKPOINT HASH MISMATCH! Manifest expected {expected_ckpt_hash}, but {ckpt_to_check} has {actual_ckpt_hash}. "
                "The index was built with a different checkpoint. Refusing to load!"
            )

    # 2. Verify Catalog Hash
    expected_cat_hash = manifest.get("catalog_hash")
    if expected_cat_hash:
        cat_to_check = pathlib.Path(catalog_path) if catalog_path else pathlib.Path(manifest.get("catalog_path", ""))
        if not cat_to_check.exists():
            raise FileNotFoundError(f"Catalog file not found at {cat_to_check} for hash verification!")
        actual_cat_hash = compute_file_sha256(str(cat_to_check))
        if actual_cat_hash != expected_cat_hash:
            raise ValueError(
                f"CATALOG HASH MISMATCH! Manifest expected {expected_cat_hash}, but {cat_to_check} has {actual_cat_hash}. "
                "The index was built with a different catalog version. Refusing to load!"
            )

    return manifest


def load_index(
    index_dir: Union[str, pathlib.Path] = "index",
    checkpoint_path: Optional[Union[str, pathlib.Path]] = None,
    catalog_path: Optional[Union[str, pathlib.Path]] = None,
    ef_search: int = 128
):
    """
    Loads FAISS index, ID map, and manifest after verifying checkpoint and catalog hashes.
    Raises ValueError if hashes do not match.
    """
    index_dir = pathlib.Path(index_dir)
    manifest = verify_manifest(index_dir, checkpoint_path=checkpoint_path, catalog_path=catalog_path)

    hnsw_filename = manifest.get("files", {}).get("hnsw_index", "faiss_hnsw_index.bin")
    index_path = index_dir / hnsw_filename
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}!")

    index = faiss.read_index(str(index_path))
    index.hnsw.efSearch = ef_search

    id_map_filename = manifest.get("files", {}).get("id_map", "id_map.json")
    id_map_path = index_dir / id_map_filename
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    return index, id_map, manifest


class VisualSearchEngine:
    def __init__(
        self,
        index_dir: str = "index",
        checkpoint_path: str = "data/checkpoints_siglip-vit/best_model.pt",
        catalog_path: Optional[str] = "data/catalog/catalog.parquet",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ef_search: int = 128
    ):
        self.index_dir = pathlib.Path(index_dir)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.catalog_path = pathlib.Path(catalog_path) if catalog_path else None
        self.device = device
        self.ef_search = ef_search

        # 1 & 2 & 3. Verify Manifest & Load Index and ID map
        self.index, self.id_map, self.manifest = load_index(
            index_dir=self.index_dir,
            checkpoint_path=self.checkpoint_path,
            catalog_path=self.catalog_path,
            ef_search=self.ef_search
        )
        print(f"Loaded FAISS index and ID map with {len(self.id_map):,} items from {self.index_dir}.")

        # 4. Load Model
        print(f"Loading SigLIP model on {self.device}...")
        self.model = load_siglip_inference(
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
            fp16=(self.device == "cuda")
        )
        self.processor = self.model.processor

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
