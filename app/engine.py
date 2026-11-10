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
from src.vi_vocab import generate_vietnamese_title
from src.cir_normalizer import normalize_cir_modifier

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
        paths_cfg = self.config.get("paths", {}) or {}
        self.serving_cfg = self.config.get("serving", {}) or {}
        self.latency_cfg = self.config.get("latency", {}) or {}
        default_sla = float(self.latency_cfg.get("sla_ms", 200.0))
        flows_cfg = self.latency_cfg.get("flows", {}) or {}
        self.flows_sla: Dict[str, float] = {
            "text_to_image": float(flows_cfg.get("text_to_image", {}).get("sla_ms", default_sla)),
            "image_to_image": float(flows_cfg.get("image_to_image", {}).get("sla_ms", default_sla)),
            "cir_catalog": float(flows_cfg.get("cir_catalog", {}).get("sla_ms", default_sla)),
            "cir_upload": float(flows_cfg.get("cir_upload", {}).get("sla_ms", default_sla)),
        }

        ckpt_dir = ROOT_DIR / paths_cfg.get("checkpoint_dir", "data/checkpoints_siglip-vit")
        catalog_p = ROOT_DIR / paths_cfg.get("catalog", "data/catalog/catalog.parquet")
        kaggle_dir = ROOT_DIR / paths_cfg.get("kaggle_dir", "data/fashion-dataset")
        self.kaggle_images_dir = kaggle_dir / "images"

        self.engine = VisualSearchEngine(
            checkpoint_path=str(ckpt_dir / "best_model.pt"),
            stats_path=str(ckpt_dir / "vit_zero_train_stats.npz"),
            image_index_path=str(ROOT_DIR / "index_v2/image_embeddings_v2.npy"),
            text_index_path=str(ROOT_DIR / "index_v2/text_embeddings_v2.npy"),
            catalog_path=str(catalog_p),
            gamma=inf_cfg.get("gamma", 0.3),
            alpha=inf_cfg.get("alpha", 1.0),
            rrf_k=inf_cfg.get("rrf_k", 60),
            min_candidates=inf_cfg.get("min_candidates", 20),
            enable_parser=inf_cfg.get("enable_parser", True),
            enable_bm25=inf_cfg.get("enable_bm25", True),
            enable_text_fusion=inf_cfg.get("enable_text_fusion", False),
            filter_mode=inf_cfg.get("filter_mode", "relaxed")
        )
        self._warmup()

    def _warmup(self) -> None:
        """Warm up CPU/GPU pipelines on startup so the first interactive query runs at steady-state latency."""
        try:
            _ = self.engine.search_text("áo thun Puma nam", top_k=5)
            dummy_pil = Image.new("RGB", (224, 224), color=(128, 128, 128))
            t1 = self.engine.img_transform(dummy_pil).unsqueeze(0).to(self.engine.device)
            if self.engine.device == "cuda":
                t1 = t1.half()
            tok = self.engine.processor(
                text=["màu xanh navy"], padding="max_length", max_length=128, truncation=True, return_tensors="pt"
            )
            input_ids = tok["input_ids"].to(self.engine.device)
            with torch.no_grad():
                img_f = self.engine.model.encode_image(t1)
                txt_f = self.engine.model.encode_text(input_ids)
                _ = self.engine.model.cir_combiner(img_f, txt_f)
                _ = self.engine.model.compose_query(t1, input_ids)
            self._sync()
        except Exception:
            pass

    def get_flow_sla_ms(self, flow_name: str) -> float:
        """Returns the SLA threshold (ms) configured in config.yaml for a specific flow."""
        return float(self.flows_sla.get(flow_name, self.latency_cfg.get("sla_ms", 200.0)))

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
                        img_p = str(self.kaggle_images_dir / Path(img_p).name)
                    results.append({
                        "rank": rank,
                        "item_id": str(row["item_id"]),
                        "name": str(row.get("name", "")),
                        "name_vi": generate_vietnamese_title(
                            name=str(row.get("name", "")),
                            article_type=row.get("articleType"),
                            gender=row.get("gender"),
                            base_colour=row.get("baseColour"),
                            brand=row.get("brand")
                        ),
                        "articleType": row["articleType"],
                        "gender": row["gender"],
                        "baseColour": row["baseColour"],
                        "brand": str(row.get("brand", "")),
                        "image_path": img_p,
                        "score": float(score),
                        "tier": "bm25_only",
                        "latency_ms": round(lat_ms, 2)
                    })
                sla_ms = self.get_flow_sla_ms("text_to_image")
                meta = {
                    "flow": "text_to_image",
                    "sla_ms": sla_ms,
                    "sla_pass": lat_ms <= sla_ms,
                    "parsed": self.engine.parser.parse(query),
                    "tier": "bm25_only",
                    "candidates_count": self.engine.N,
                    "breakdown": {"total_ms": round(lat_ms, 2)}
                }
                return results, meta

            results = self.engine.search_text(query, top_k=top_k)
            meta = dict(getattr(self.engine, "last_query_meta", {}) or {})
            sla_ms = self.get_flow_sla_ms("text_to_image")
            total_ms = float(meta.get("breakdown", {}).get("total_ms", results[0]["latency_ms"] if results else 0.0))
            meta["flow"] = "text_to_image"
            meta["sla_ms"] = sla_ms
            meta["sla_pass"] = total_ms <= sla_ms
            return results, meta
        finally:
            self.engine.enable_parser = old_enable_parser
            self.engine.filter_mode = old_filter_mode
            self.engine.enable_bm25 = old_enable_bm25
            self.engine.gamma = old_gamma

    def _sync(self):
        if self.engine.device == "cuda":
            torch.cuda.synchronize()

    def resolve_i2i_flip(self, use_flip_aug: Optional[bool] = None, mode: Optional[str] = None) -> bool:
        """i2i_mode="fast" (1 view) is default; "accurate" = Batch-2 TTA flip. Explicit use_flip_aug wins."""
        if use_flip_aug is not None:
            return bool(use_flip_aug)
        mode = mode or self.serving_cfg.get("i2i_mode", "fast")
        return mode == "accurate"

    def search_image(
        self,
        image_input: Any,
        top_k: int = 10,
        exclude_self_id: Optional[str] = None,
        use_flip_aug: Optional[bool] = None,
        use_knn_voting: bool = True,
        mode: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Visual Search (Image -> Image). Mode comes from config serving.i2i_mode
        ("fast": single view, default | "accurate": Batch-2 TTA flip).
        Per-stage timings (ms) are returned in meta["breakdown"].
        """
        use_flip_aug = self.resolve_i2i_flip(use_flip_aug, mode)
        self._sync()
        t0 = time.perf_counter()
        im = Image.open(image_input).convert("RGB") if isinstance(image_input, (str, Path)) else image_input.convert("RGB")
        t_dec = time.perf_counter()

        if use_flip_aug:
            im_flip = ImageOps.mirror(im)
            t1 = self.engine.img_transform(im)
            t2 = self.engine.img_transform(im_flip)
            batch = torch.stack([t1, t2]).to(self.engine.device)
            if self.engine.device == "cuda":
                batch = batch.half()
            t_pre = time.perf_counter()
            with torch.no_grad():
                feats = self.engine.model.encode_image(batch)
                self._sync()
                feats = feats.float().cpu().numpy()
            t_fwd = time.perf_counter()
            v1 = feats[0] / max(float(np.linalg.norm(feats[0])), 1e-12)
            v2 = feats[1] / max(float(np.linalg.norm(feats[1])), 1e-12)
            aug_vec = v1 + v2
            aug_vec = aug_vec / max(float(np.linalg.norm(aug_vec)), 1e-12)
            query_vec_c = self.engine.scorer.center_vectors(aug_vec[np.newaxis, :])[0]
        else:
            t1 = self.engine.img_transform(im).unsqueeze(0).to(self.engine.device)
            if self.engine.device == "cuda":
                t1 = t1.half()
            t_pre = time.perf_counter()
            with torch.no_grad():
                feat = self.engine.model.encode_image(t1)
                self._sync()
                feat = feat.float().cpu().numpy()[0]
            t_fwd = time.perf_counter()
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
        t_srch = time.perf_counter()

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
        t_rr = time.perf_counter()

        lat_ms = (t_rr - t0) * 1000.0
        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), start=1):
            row = self.engine.df.iloc[idx]
            img_p = str(row["path"])
            if not Path(img_p).exists():
                img_p = str(self.kaggle_images_dir / Path(img_p).name)
            results.append({
                "rank": rank,
                "item_id": str(row["item_id"]),
                "name": str(row.get("name", "")),
                "name_vi": generate_vietnamese_title(
                    name=str(row.get("name", "")),
                    article_type=row.get("articleType"),
                    gender=row.get("gender"),
                    base_colour=row.get("baseColour"),
                    brand=row.get("brand")
                ),
                "articleType": row["articleType"],
                "gender": row["gender"],
                "baseColour": row["baseColour"],
                "brand": str(row.get("brand", "")),
                "image_path": img_p,
                "score": float(score),
                "latency_ms": round(lat_ms, 2)
            })
        sla_ms = self.get_flow_sla_ms("image_to_image")
        meta = {
            "flow": "image_to_image",
            "i2i_mode": "accurate" if use_flip_aug else "fast",
            "sla_ms": sla_ms,
            "sla_pass": lat_ms <= sla_ms,
            "tier": "visual_search_batch2_tta" if use_flip_aug else "visual_search_fast",
            "latency_ms": round(lat_ms, 2),
            "use_flip_aug": use_flip_aug,
            "breakdown": {
                "decode_ms": (t_dec - t0) * 1000.0,
                "preprocess_ms": (t_pre - t_dec) * 1000.0,
                "vit_forward_ms": (t_fwd - t_pre) * 1000.0,
                "gallery_search_ms": (t_srch - t_fwd) * 1000.0,
                "rerank_ms": (t_rr - t_srch) * 1000.0,
            }
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
        self._sync()
        t0 = time.perf_counter()
        t_dec = t_pre = t0
        
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
        cand_idxs = np.arange(self.engine.N)
        parsed_mod = self.engine.parser.parse(modifier_text) if modifier_text.strip() else {}

        # Determine reference type and gender for prompt anchoring
        ref_type = None
        ref_gender = None
        if is_catalog_cached:
            row_ref = self.engine.df.iloc[self.engine.id_to_idx[cand_id]]
            ref_type = row_ref.get("articleType")
            ref_gender = row_ref.get("gender")
        elif ref_id and str(ref_id) in self.engine.id_to_idx:
            row_ref = self.engine.df.iloc[self.engine.id_to_idx[str(ref_id)]]
            ref_type = row_ref.get("articleType")
            ref_gender = row_ref.get("gender")

        # 1. Normalize modifier text for CIR Combiner
        en_prompt, is_fallback, untranslated = normalize_cir_modifier(
            modifier_text=modifier_text,
            parsed_spec=parsed_mod,
            ref_type=ref_type,
            ref_gender=ref_gender
        )

        fallback_msg = None
        if not modifier_text.strip():
            is_fallback = True
        elif is_fallback or not en_prompt.strip():
            fallback_msg = "Chưa hiểu yêu cầu chỉnh sửa, đang hiển thị kết quả tương tự ảnh gốc"

        if is_catalog_cached:
            idx = self.engine.id_to_idx[cand_id]
            raw_emb = self.engine.emb_img_raw[idx]
            t_ref_emb = torch.tensor(raw_emb, dtype=torch.float16, device=self.engine.device).unsqueeze(0)
            
            if is_fallback:
                cir_vec = raw_emb
            else:
                tok = self.engine.processor(text=[en_prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
                input_ids = tok["input_ids"].to(self.engine.device)
                t_pre = time.perf_counter()
                with torch.no_grad():
                    mod_txt_emb = self.engine.model.encode_text(input_ids)
                    cir_feat = self.engine.model.cir_combiner(t_ref_emb, mod_txt_emb)
                    self._sync()
                    cir_vec = cir_feat.cpu().numpy()[0]
        else:
            im = Image.open(ref_image).convert("RGB") if isinstance(ref_image, (str, Path)) else ref_image.convert("RGB")
            t_dec = time.perf_counter()
            img_tensor = self.engine.img_transform(im).unsqueeze(0).to(self.engine.device)
            if self.engine.device == "cuda":
                img_tensor = img_tensor.half()

            # Predict visual category if ref_type is not yet known
            if not ref_type and not parsed_mod.get("articleType") and not is_fallback and en_prompt:
                with torch.no_grad():
                    v_raw = self.engine.model.encode_image(img_tensor)
                    v_c = self.engine.scorer.center_vectors(v_raw.cpu().numpy())[0]
                    q_v = torch.tensor(v_c, dtype=torch.float16, device=self.engine.device)
                    top1_vis = torch.argmax(torch.matmul(self.engine.t_emb_img_c, q_v)).item()
                    pred_type = self.engine.df.iloc[top1_vis]["articleType"]
                    en_prompt, _, _ = normalize_cir_modifier(modifier_text, parsed_mod, ref_type=pred_type)

            if is_fallback:
                with torch.no_grad():
                    cir_feat = self.engine.model.encode_image(img_tensor)
                    self._sync()
                    cir_vec = cir_feat.cpu().numpy()[0]
            else:
                tok = self.engine.processor(text=[en_prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
                input_ids = tok["input_ids"].to(self.engine.device)
                t_pre = time.perf_counter()
                with torch.no_grad():
                    cir_feat = self.engine.model.compose_query(img_tensor, input_ids)
                    self._sync()
                    cir_vec = cir_feat.cpu().numpy()[0]

        t_fwd = time.perf_counter()
        norm = max(float(np.linalg.norm(cir_vec)), 1e-12)
        cir_vec = cir_vec / norm
        cir_vec_c = self.engine.scorer.center_vectors(cir_vec[np.newaxis, :])[0]

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

        t_end = time.perf_counter()
        lat_ms = (t_end - t0) * 1000.0
        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), start=1):
            row = self.engine.df.iloc[idx]
            img_p = str(row["path"])
            if not Path(img_p).exists():
                img_p = str(self.kaggle_images_dir / Path(img_p).name)
            results.append({
                "rank": rank,
                "item_id": str(row["item_id"]),
                "name": str(row.get("name", "")),
                "name_vi": generate_vietnamese_title(
                    name=str(row.get("name", "")),
                    article_type=row.get("articleType"),
                    gender=row.get("gender"),
                    base_colour=row.get("baseColour"),
                    brand=row.get("brand")
                ),
                "articleType": row["articleType"],
                "gender": row["gender"],
                "baseColour": row["baseColour"],
                "brand": str(row.get("brand", "")),
                "image_path": img_p,
                "score": float(score),
                "latency_ms": round(lat_ms, 2)
            })
        flow_name = "cir_catalog" if is_catalog_cached else "cir_upload"
        sla_ms = self.get_flow_sla_ms(flow_name)
        meta = {
            "flow": flow_name,
            "sla_ms": sla_ms,
            "sla_pass": lat_ms <= sla_ms,
            "tier": "cir_cached" if is_catalog_cached else "cir_upload",
            "is_catalog_cached": is_catalog_cached,
            "catalog_ref_id": cand_id,
            "latency_ms": round(lat_ms, 2),
            "translated_modifier": en_prompt,
            "fallback_message": fallback_msg,
            "untranslated_tokens": untranslated,
            "breakdown": {
                "decode_ms": (t_dec - t0) * 1000.0,
                "preprocess_ms": (t_pre - t_dec) * 1000.0,
                "vit_forward_ms": (t_fwd - t_pre) * 1000.0,
                "gallery_search_ms": (t_end - t_fwd) * 1000.0,
                "rerank_ms": 0.0,
            }
        }
        return results, meta
