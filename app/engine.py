"""
Search Engine Service for Streamlit Web App (app/engine.py)
Implements:
- Rule 7 checkpoint hash verification on startup.
- Cached search engine loading with @st.cache_resource.
- Unified wrappers for Text Search, Visual Search, and CIR.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import yaml
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.search_pipeline import VisualSearchEngine

def verify_rule_7_checkpoint_hash(
    config_path: str = "config.yaml",
    manifest_path: str = "index_v2/manifest.json"
) -> Tuple[bool, str, str]:
    """
    Enforces Invariant Rule 7:
    Index must match checkpoint. Manifest records checkpoint hash; app refuses to load if mismatched.
    """
    cfg_p = ROOT_DIR / config_path
    man_p = ROOT_DIR / manifest_path

    if not cfg_p.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_p}")
    if not man_p.exists():
        raise FileNotFoundError(f"Index manifest not found: {man_p}")

    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(man_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cfg_hash = str(cfg.get("checkpoint", {}).get("checkpoint_hash", "")).strip().lower()
    man_hash = str(manifest.get("checkpoint_hash", "")).strip().lower()

    if not cfg_hash or not man_hash or cfg_hash != man_hash:
        raise RuntimeError(
            f"❌ VI PHẠM LUẬT 7: Checkpoint hash không khớp giữa cấu hình và index manifest!\n"
            f"- Config hash: {cfg_hash}\n"
            f"- Manifest hash: {man_hash}\n"
            f"Ứng dụng từ chối nạp hệ thống để bảo vệ tính toàn vẹn dữ liệu."
        )

    return True, cfg_hash, man_hash


class SearchService:
    """
    Singleton service wrapper around VisualSearchEngine for Streamlit app.
    """
    def __init__(self, config_path: str = "config.yaml"):
        # 1. Enforce Rule 7
        self.verified, self.cfg_hash, self.man_hash = verify_rule_7_checkpoint_hash(config_path)

        with open(ROOT_DIR / config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        inf_cfg = self.config.get("inference", {})
        self.engine = VisualSearchEngine(
            checkpoint_path=str(ROOT_DIR / "data/checkpoints_siglip-vit/best_model.pt"),
            stats_path=str(ROOT_DIR / "data/checkpoints_siglip-vit/vit_zero_train_stats.npz"),
            image_index_path=str(ROOT_DIR / "index_v2/image_embeddings_v2.npy"),
            text_index_path=str(ROOT_DIR / "index_v2/text_embeddings_v2.npy"),
            catalog_path=str(ROOT_DIR / "data/catalog/catalog.parquet"),
            gamma=inf_cfg.get("gamma", 0.3),
            alpha=inf_cfg.get("alpha", 1.0),
            rrf_k=inf_cfg.get("rrf_k", 60),
            min_candidates=inf_cfg.get("min_candidates", 20),
            enable_parser=inf_cfg.get("enable_parser", True),
            enable_bm25=inf_cfg.get("enable_bm25", True),
            enable_text_fusion=inf_cfg.get("enable_text_fusion", False),
            filter_mode=inf_cfg.get("filter_mode", "relaxed")
        )

    def search_text(
        self,
        query: str,
        config_mode: str = "c5",
        top_k: int = 10,
        gamma: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes text retrieval under specified configuration.
        """
        # Save previous state
        old_enable_parser = self.engine.enable_parser
        old_filter_mode = self.engine.filter_mode
        old_enable_bm25 = self.engine.enable_bm25
        old_gamma = self.engine.gamma

        try:
            self.engine.gamma = gamma
            if config_mode == "c5": # Full System: Parser + Filter + Soft Bonus + Dense + BM25
                self.engine.enable_parser = True
                self.engine.filter_mode = "relaxed"
                self.engine.enable_bm25 = True
            elif config_mode == "c0": # Dense Only (Model Thuan)
                self.engine.enable_parser = False
                self.engine.filter_mode = "none"
                self.engine.enable_bm25 = False
                self.engine.gamma = 0.0
            elif config_mode == "c2": # Filter + Soft Bonus + Dense
                self.engine.enable_parser = True
                self.engine.filter_mode = "relaxed"
                self.engine.enable_bm25 = False
            elif config_mode == "c4": # BM25 Sparse Only
                # Emulate BM25 pure search
                t0 = time.perf_counter()
                cand_mask = np.ones(self.engine.N, dtype=bool)
                sparse_order, sparse_scores = self.engine.bm25.search(query, top_k=top_k, mask=cand_mask)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                results = []
                for rank, (idx, score) in enumerate(zip(sparse_order[:top_k], sparse_scores[:top_k]), start=1):
                    row = self.engine.df.iloc[idx]
                    img_p = str(row["path"])
                    if not Path(img_p).exists():
                        img_p = str(Path("data/fashion-dataset/images") / Path(img_p).name)
                    results.append({
                        "rank": rank,
                        "item_id": str(row["item_id"]),
                        "name": str(row.get("name", "")),
                        "articleType": row["articleType"],
                        "gender": row["gender"],
                        "baseColour": row["baseColour"],
                        "brand": str(row.get("brand", "")),
                        "image_path": img_p,
                        "score": float(score),
                        "tier": "bm25_only",
                        "latency_ms": round(lat_ms, 2)
                    })
                meta = {
                    "parsed": self.engine.parser.parse(query),
                    "tier": "bm25_only",
                    "candidates_count": self.engine.N,
                    "breakdown": {"total_ms": round(lat_ms, 2)}
                }
                return results, meta

            results = self.engine.search_text(query, top_k=top_k)
            meta = getattr(self.engine, "last_query_meta", {})
            return results, meta
        finally:
            self.engine.enable_parser = old_enable_parser
            self.engine.filter_mode = old_filter_mode
            self.engine.enable_bm25 = old_enable_bm25
            self.engine.gamma = old_gamma

    def search_image(
        self,
        image_input: Any,
        top_k: int = 10,
        exclude_self_id: Optional[str] = None,
        use_flip_aug: bool = True,
        use_knn_voting: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Visual Search (Image -> Image) with Batch-2 TTA on GPU.
        """
        t0 = time.perf_counter()
        im = Image.open(image_input).convert("RGB") if isinstance(image_input, (str, Path)) else image_input.convert("RGB")
        
        if use_flip_aug:
            im_flip = ImageOps.mirror(im)
            t1 = self.engine.img_transform(im)
            t2 = self.engine.img_transform(im_flip)
            batch = torch.stack([t1, t2]).to(self.engine.device)
            if self.engine.device == "cuda":
                batch = batch.half()
            with torch.no_grad():
                feats = self.engine.model.encode_image(batch).cpu().numpy()
            v1 = feats[0] / max(float(np.linalg.norm(feats[0])), 1e-12)
            v2 = feats[1] / max(float(np.linalg.norm(feats[1])), 1e-12)
            aug_vec = v1 + v2
            aug_vec = aug_vec / max(float(np.linalg.norm(aug_vec)), 1e-12)
            query_vec_c = self.engine.scorer.center_vectors(aug_vec[np.newaxis, :])[0]
        else:
            t1 = self.engine.img_transform(im).unsqueeze(0).to(self.engine.device)
            if self.engine.device == "cuda":
                t1 = t1.half()
            with torch.no_grad():
                feat = self.engine.model.encode_image(t1).cpu().numpy()[0]
            v1 = feat / max(float(np.linalg.norm(feat)), 1e-12)
            query_vec_c = self.engine.scorer.center_vectors(v1[np.newaxis, :])[0]

        cand_idxs = np.arange(self.engine.N)
        if self.engine.device == "cuda" and self.engine.t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=self.engine.device)
            q_t = torch.tensor(query_vec_c, dtype=torch.float16, device=self.engine.device)
            sims_t = torch.matmul(self.engine.t_emb_img_c[cand_t], q_t)
            sims = sims_t.float().cpu().numpy()
        else:
            sims = np.dot(self.engine.emb_img_c, query_vec_c)

        if exclude_self_id and str(exclude_self_id) in self.engine.id_to_idx:
            self_idx = self.engine.id_to_idx[str(exclude_self_id)]
            sims[self_idx] = -1e9

        if use_knn_voting:
            k_cand = min(50, len(cand_idxs))
            top_cand_pos = np.argsort(-sims)[:k_cand]
            voted_scores = sims[top_cand_pos].copy()
            final_idxs = top_cand_pos[np.argsort(-voted_scores)[:top_k]]
            final_scores = voted_scores[np.argsort(-voted_scores)[:top_k]]
        else:
            top_pos = np.argsort(-sims)[:top_k]
            final_idxs = top_pos
            final_scores = sims[top_pos]

        lat_ms = (time.perf_counter() - t0) * 1000.0
        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), start=1):
            row = self.engine.df.iloc[idx]
            img_p = str(row["path"])
            if not Path(img_p).exists():
                img_p = str(Path("data/fashion-dataset/images") / Path(img_p).name)
            results.append({
                "rank": rank,
                "item_id": str(row["item_id"]),
                "name": str(row.get("name", "")),
                "articleType": row["articleType"],
                "gender": row["gender"],
                "baseColour": row["baseColour"],
                "brand": str(row.get("brand", "")),
                "image_path": img_p,
                "score": float(score),
                "latency_ms": round(lat_ms, 2)
            })
        meta = {
            "tier": "visual_search_batch2_tta",
            "latency_ms": round(lat_ms, 2),
            "use_flip_aug": use_flip_aug
        }
        return results, meta

    def search_cir(
        self,
        ref_image: Any,
        modifier_text: str,
        top_k: int = 10,
        use_soft_bonus: bool = True,
        ref_id: Optional[str] = None,
        force_upload: bool = False
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        CIR with Catalog Embedding Cache if reference image is in catalog,
        otherwise falls back to ViT Image Tower encoding (CIR-upload).
        """
        t0 = time.perf_counter()
        
        cand_id = None
        if not force_upload:
            if ref_id and str(ref_id) in self.engine.id_to_idx:
                cand_id = str(ref_id)
            elif isinstance(ref_image, (int, str)):
                s_ref = str(ref_image).strip()
                if s_ref in self.engine.id_to_idx:
                    cand_id = s_ref
                else:
                    stem = Path(s_ref).stem
                    if stem in self.engine.id_to_idx:
                        cand_id = stem
            elif hasattr(ref_image, "item_id") and str(ref_image.item_id) in self.engine.id_to_idx:
                cand_id = str(ref_image.item_id)

        is_catalog_cached = (cand_id is not None)

        if is_catalog_cached:
            idx = self.engine.id_to_idx[cand_id]
            raw_emb = self.engine.emb_img_raw[idx]
            t_ref_emb = torch.tensor(raw_emb, dtype=torch.float16, device=self.engine.device).unsqueeze(0)
            
            tok = self.engine.processor(text=[modifier_text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            input_ids = tok["input_ids"].to(self.engine.device)
            with torch.no_grad():
                mod_txt_emb = self.engine.model.encode_text(input_ids)
                cir_feat = self.engine.model.cir_combiner(t_ref_emb, mod_txt_emb)
                cir_vec = cir_feat.cpu().numpy()[0]
        else:
            im = Image.open(ref_image).convert("RGB") if isinstance(ref_image, (str, Path)) else ref_image.convert("RGB")
            img_tensor = self.engine.img_transform(im).unsqueeze(0).to(self.engine.device)
            if self.engine.device == "cuda":
                img_tensor = img_tensor.half()
            tok = self.engine.processor(text=[modifier_text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            input_ids = tok["input_ids"].to(self.engine.device)
            with torch.no_grad():
                cir_feat = self.engine.model.compose_query(img_tensor, input_ids)
                cir_vec = cir_feat.cpu().numpy()[0]

        norm = max(float(np.linalg.norm(cir_vec)), 1e-12)
        cir_vec = cir_vec / norm
        cir_vec_c = self.engine.scorer.center_vectors(cir_vec[np.newaxis, :])[0]

        cand_idxs = np.arange(self.engine.N)
        parsed_mod = self.engine.parser.parse(modifier_text)

        if self.engine.device == "cuda" and self.engine.t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=self.engine.device)
            q_t = torch.tensor(cir_vec_c, dtype=torch.float16, device=self.engine.device)
            sims_t = torch.matmul(self.engine.t_emb_img_c[cand_t], q_t)

            if use_soft_bonus and self.engine.gamma > 0:
                bonus = self.engine.attr_idx.compute_attribute_soft_bonus(
                    parsed_mod, cand_idxs,
                    gamma_type=self.engine.gamma, gamma_gender=self.engine.gamma, gamma_colour=self.engine.gamma, gamma_brand=self.engine.gamma
                )
                sims_t = sims_t + torch.tensor(bonus, dtype=torch.float16, device=self.engine.device)

            k_top = min(top_k, len(cand_idxs))
            topk_vals, top_pos = torch.topk(sims_t, k=k_top)
            final_idxs = cand_idxs[top_pos.cpu().numpy()]
            final_scores = topk_vals.float().cpu().numpy()
        else:
            sims = np.dot(self.engine.emb_img_c, cir_vec_c)
            if use_soft_bonus and self.engine.gamma > 0:
                bonus = self.engine.attr_idx.compute_attribute_soft_bonus(
                    parsed_mod, cand_idxs,
                    gamma_type=self.engine.gamma, gamma_gender=self.engine.gamma, gamma_colour=self.engine.gamma, gamma_brand=self.engine.gamma
                )
                sims = sims + bonus
            k_top = min(top_k, len(cand_idxs))
            top_cand_pos = np.argsort(-sims)[:k_top]
            final_idxs = cand_idxs[top_cand_pos]
            final_scores = sims[top_cand_pos]

        lat_ms = (time.perf_counter() - t0) * 1000.0
        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), start=1):
            row = self.engine.df.iloc[idx]
            img_p = str(row["path"])
            if not Path(img_p).exists():
                img_p = str(Path("data/fashion-dataset/images") / Path(img_p).name)
            results.append({
                "rank": rank,
                "item_id": str(row["item_id"]),
                "name": str(row.get("name", "")),
                "articleType": row["articleType"],
                "gender": row["gender"],
                "baseColour": row["baseColour"],
                "brand": str(row.get("brand", "")),
                "image_path": img_p,
                "score": float(score),
                "latency_ms": round(lat_ms, 2)
            })
        meta = {
            "tier": "cir_cached" if is_catalog_cached else "cir_upload",
            "is_catalog_cached": is_catalog_cached,
            "latency_ms": round(lat_ms, 2)
        }
        return results, meta
