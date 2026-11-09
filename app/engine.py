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
import pandas as pd
from PIL import Image

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
        use_flip_aug: bool = True,
        use_knn_voting: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results = self.engine.search_image(
            image_input,
            top_k=top_k,
            use_flip_aug=use_flip_aug,
            use_knn_voting=use_knn_voting
        )
        meta = getattr(self.engine, "last_query_meta", {})
        return results, meta

    def search_cir(
        self,
        ref_image: Any,
        modifier_text: str,
        top_k: int = 10,
        use_soft_bonus: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results = self.engine.search_cir(
            ref_image,
            modifier_text,
            top_k=top_k,
            use_soft_bonus=use_soft_bonus
        )
        meta = getattr(self.engine, "last_query_meta", {})
        return results, meta
