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
from src.cir_normalizer import (
    normalize_cir_modifier,
    sanitize_parsed_modifier,
    classify_cir_intent,
)

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

    def _get_related_article_types(self, target_type: str) -> List[str]:
        """
        Returns symmetric related articleTypes from eval/families.json for graceful fallback
        when hard category filtering yields fewer than top_k items.
        """
        if not target_type:
            return []
        art_fams = getattr(self.engine, "art_families", {}) or {}
        related = []
        # Direct family list
        if target_type in art_fams:
            for t in art_fams[target_type]:
                if t != target_type and t not in related:
                    related.append(t)
        # Symmetric reverse lookup
        for fam_key, fam_list in art_fams.items():
            if target_type in fam_list:
                if fam_key != target_type and fam_key not in related:
                    related.append(fam_key)
                for t in fam_list:
                    if t != target_type and t not in related:
                        related.append(t)
        return related

    def _infer_upload_article_type(
        self,
        v_raw: torch.Tensor,
        top_k_knn: int = 10,
        conf_threshold: float = 0.62,
        min_visual_sim: float = 0.65
    ) -> Dict[str, Any]:
        """
        Infers articleType for an uploaded image using visual-only weighted k-NN voting
        in the centered image embedding gallery.
        Weight for rank r (0-based): max(sim, 0)^2 / log2(r + 2).
        Both vote ratio >= conf_threshold and top1_sim >= min_visual_sim are required for hard locking.
        """
        v_np = v_raw.float().cpu().numpy()
        v_c = self.engine.scorer.center_vectors(v_np)[0]

        if self.engine.device == "cuda" and self.engine.t_emb_img_c is not None:
            q_v = torch.tensor(v_c, dtype=torch.float16, device=self.engine.device)
            sims_t = torch.matmul(self.engine.t_emb_img_c, q_v)
            k_nn = min(top_k_knn, self.engine.N)
            top_vals_t, top_idxs_t = torch.topk(sims_t, k=k_nn)
            top_vals = top_vals_t.float().cpu().numpy()
            top_idxs = top_idxs_t.cpu().numpy()
        else:
            sims = np.dot(self.engine.emb_img_c, v_c)
            k_nn = min(top_k_knn, self.engine.N)
            top_idxs = np.argsort(-sims, kind="stable")[:k_nn]
            top_vals = sims[top_idxs]

        top1_sim = float(top_vals[0]) if len(top_vals) > 0 else 0.0
        scores_by_type: Dict[str, float] = {}
        for r_i, (idx, sim_val) in enumerate(zip(top_idxs, top_vals)):
            s = max(float(sim_val), 0.0)
            w = (s ** 2) / float(np.log2(r_i + 2.0))
            at = str(self.engine.df.iloc[int(idx)]["articleType"])
            scores_by_type[at] = scores_by_type.get(at, 0.0) + w

        total_w = sum(scores_by_type.values())
        if total_w <= 1e-12:
            return {
                "pred_type": None,
                "confidence": 0.0,
                "top1_sim": top1_sim,
                "confident": False,
                "type_probs": {}
            }

        type_probs = {k: v / total_w for k, v in scores_by_type.items()}
        best_type, best_prob = max(type_probs.items(), key=lambda x: x[1])
        confident = (best_prob >= conf_threshold) and (top1_sim >= min_visual_sim)

        return {
            "pred_type": best_type,
            "confidence": float(best_prob),
            "top1_sim": top1_sim,
            "confident": confident,
            "type_probs": type_probs
        }

    @staticmethod
    def interpolate_cir_vector(e_img_raw: np.ndarray, e_comb_raw: np.ndarray, modifier_weight: float) -> np.ndarray:
        """
        Computes normalized linear interpolation:
            e_query = normalize((1 - lambda) * normalize(e_img) + lambda * normalize(e_combiner))
        """
        lam = max(0.0, min(1.0, float(modifier_weight)))
        e_img_n = e_img_raw / max(float(np.linalg.norm(e_img_raw)), 1e-12)
        if lam <= 0.0:
            return e_img_n
        e_comb_n = e_comb_raw / max(float(np.linalg.norm(e_comb_raw)), 1e-12)
        if lam >= 1.0:
            return e_comb_n
        e_mix = (1.0 - lam) * e_img_n + lam * e_comb_n
        return e_mix / max(float(np.linalg.norm(e_mix)), 1e-12)

    def _select_with_category_anchor(
        self,
        sims_all: np.ndarray,
        top_k: int,
        target_type: Optional[str],
        anchor_active: bool,
        anchor_mode: str,
        anchor_fallback_soft: bool,
        upload_type_probs: Dict[str, float],
        candidate_pool_k: Optional[int],
        expand_pool_if_short: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str], int, int, Optional[str]]:
        """
        Applies dynamic category anchor filtering/reranking cleanly and deterministically.

        When `anchor_active` and not `anchor_fallback_soft`:
        - Fetches candidate pool (top `candidate_pool_k` sorted by score descending, or full gallery if None).
        - Filters candidates matching `articleType == target_type`.
        - If fewer than `top_k` results remain in pool and `expand_pool_if_short=True`, searches remaining gallery items matching `target_type`.
        - If still fewer than `top_k` results remain:
          1) Supplements from related family types (eval/families.json) by descending score.
          2) If still < `top_k`, returns only the N valid matching items + shortage message (never silently pads wrong types).

        Returns:
            (final_idxs, final_scores, match_tiers, exact_count, family_count, shortage_notice)
        """
        n_total = len(sims_all)
        sorted_order = np.argsort(-sims_all, kind="stable")

        # Soft rerank fallback (below upload confidence threshold or explicit soft mode)
        if anchor_fallback_soft or anchor_mode == "soft":
            if upload_type_probs:
                soft_bonus = np.zeros(n_total, dtype=np.float32)
                for art_t, prob in upload_type_probs.items():
                    if art_t in self.engine.attr_idx.art2id:
                        tid = self.engine.attr_idx.art2id[art_t]
                        soft_bonus += np.where(self.engine.attr_idx.article_type_ids == tid, 0.20 * prob, 0.0).astype(np.float32)
                adj_sims = sims_all + soft_bonus
                adj_order = np.argsort(-adj_sims, kind="stable")[:top_k]
                return adj_order, adj_sims[adj_order], ["soft_rerank"] * len(adj_order), 0, 0, None
            elif target_type and target_type in self.engine.attr_idx.art2id:
                tid = self.engine.attr_idx.art2id[target_type]
                soft_bonus = np.where(self.engine.attr_idx.article_type_ids == tid, 0.25, 0.0).astype(np.float32)
                adj_sims = sims_all + soft_bonus
                adj_order = np.argsort(-adj_sims, kind="stable")[:top_k]
                return adj_order, adj_sims[adj_order], ["soft_rerank"] * len(adj_order), 0, 0, None

        # Anchor disabled or unknown target_type
        if not anchor_active or not target_type or target_type not in self.engine.attr_idx.art2id:
            sel = sorted_order[:top_k]
            return sel, sims_all[sel], ["unconstrained"] * len(sel), len(sel), 0, None

        # Hard/Auto Category Lock on candidate pool
        pool_limit = min(int(candidate_pool_k), n_total) if candidate_pool_k is not None else n_total
        pool_idxs = sorted_order[:pool_limit]
        art_ids = self.engine.attr_idx.article_type_ids
        target_tid = self.engine.attr_idx.art2id[target_type]

        exact_mask = (art_ids[pool_idxs] == target_tid)
        exact_idxs = pool_idxs[exact_mask]
        selected_idxs = [int(x) for x in exact_idxs[:top_k]]

        # Expand to full gallery exact matches if pool yielded < top_k (crucial for Category Mutation or rare colors)
        if len(selected_idxs) < top_k and expand_pool_if_short and pool_limit < n_total:
            all_exact_mask = (art_ids[sorted_order] == target_tid)
            selected_idxs = [int(x) for x in sorted_order[all_exact_mask][:top_k]]

        match_tiers = ["exact_type"] * len(selected_idxs)
        exact_count = len(selected_idxs)
        family_count = 0
        shortage_notice = None

        # If exact matches still < top_k, expand to related articleType families,
        # or report exact N items found without silently padding unrelated categories
        if len(selected_idxs) < top_k:
            related_types = self._get_related_article_types(target_type)
            related_tids = [self.engine.attr_idx.art2id[t] for t in related_types if t in self.engine.attr_idx.art2id]
            search_scope = sorted_order if expand_pool_if_short else pool_idxs
            if related_tids:
                already_set = set(selected_idxs)
                fam_mask = np.isin(art_ids[search_scope], related_tids)
                for idx in search_scope[fam_mask]:
                    i_int = int(idx)
                    if i_int not in already_set:
                        selected_idxs.append(i_int)
                        match_tiers.append("family_fallback")
                        already_set.add(i_int)
                        family_count += 1
                        if len(selected_idxs) >= top_k:
                            break

            if len(selected_idxs) < top_k:
                shortage_notice = f"Chỉ có {len(selected_idxs)} sản phẩm phù hợp (trong nhóm loại {target_type})"
            elif family_count > 0:
                shortage_notice = (
                    f"Chỉ có {exact_count} sản phẩm đúng loại {target_type}; "
                    f"đã bổ sung {family_count} sản phẩm cùng nhóm loại gần nhất"
                )

        final_idxs = np.array(selected_idxs, dtype=np.int64)
        final_scores = sims_all[final_idxs] if len(final_idxs) > 0 else np.array([], dtype=np.float32)
        return final_idxs, final_scores, match_tiers, exact_count, family_count, shortage_notice

    def search_cir(
        self,
        ref_image: Any,
        modifier_text: str,
        top_k: int = 10,
        use_soft_bonus: bool = True,
        ref_id: Optional[str] = None,
        force_upload: bool = False,
        dynamic_category_anchor: bool = True,
        anchor_mode: str = "auto",
        modifier_weight: float = 0.5,
        candidate_pool_k: Optional[int] = 500,
        expand_pool_if_short: bool = True,
        upload_knn_k: int = 10,
        upload_conf_threshold: float = 0.62,
        upload_min_sim: float = 0.65
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Composed Image Retrieval (CIR) with Dynamic Category Locking and Vector Interpolation Slider.

        Parameters:
        - dynamic_category_anchor: Enable two-state Category Locking (State A attribute shift vs State B category mutation).
        - anchor_mode: "auto" (lock when confident, soft rerank when uncertain) | "hard" | "soft" | "off".
        - modifier_weight (lambda): Blend in [0.0, 1.0]: e_query = normalize((1 - lambda)*e_img + lambda*e_combiner).
        - candidate_pool_k: Top-K candidate pool retrieved before category locking (None = full gallery).
        """
        self._sync()
        t0 = time.perf_counter()
        t_dec = t_pre = t0
        lam = max(0.0, min(1.0, float(modifier_weight)))

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
        raw_parsed_mod = self.engine.parser.parse(modifier_text) if modifier_text.strip() else {}

        # Sanitize parser false positives ('đậm'->Dresses, 'có mũ'->Caps) and recover color typos
        clean_spec, extra_style_terms, diag_info = sanitize_parsed_modifier(modifier_text, raw_parsed_mod)

        # Determine reference type/gender and image raw vector
        ref_type = None
        ref_gender = None
        ref_type_source = "unknown"
        upload_knn_info = {
            "pred_type": None, "confidence": 0.0, "top1_sim": 0.0, "confident": False, "type_probs": {}
        }

        if is_catalog_cached:
            idx = self.engine.id_to_idx[cand_id]
            row_ref = self.engine.df.iloc[idx]
            ref_type = row_ref.get("articleType")
            ref_gender = row_ref.get("gender")
            ref_type_source = "catalog_metadata"
            e_img_raw = self.engine.emb_img_raw[idx]
            e_img_raw = e_img_raw / max(float(np.linalg.norm(e_img_raw)), 1e-12)
            t_ref_emb = torch.tensor(self.engine.emb_img_raw[idx], dtype=torch.float16 if self.engine.device == "cuda" else torch.float32, device=self.engine.device).unsqueeze(0)
        else:
            im = Image.open(ref_image).convert("RGB") if isinstance(ref_image, (str, Path)) else ref_image.convert("RGB")
            t_dec = time.perf_counter()
            img_tensor = self.engine.img_transform(im).unsqueeze(0).to(self.engine.device)
            if self.engine.device == "cuda":
                img_tensor = img_tensor.half()

            with torch.no_grad():
                v_raw = self.engine.model.encode_image(img_tensor)
            t_ref_emb = v_raw
            e_img_raw = v_raw.float().cpu().numpy()[0]
            e_img_raw = e_img_raw / max(float(np.linalg.norm(e_img_raw)), 1e-12)

            if ref_id and str(ref_id) in self.engine.id_to_idx:
                row_ref = self.engine.df.iloc[self.engine.id_to_idx[str(ref_id)]]
                ref_type = row_ref.get("articleType")
                ref_gender = row_ref.get("gender")
                ref_type_source = "catalog_metadata"
            else:
                upload_knn_info = self._infer_upload_article_type(
                    v_raw,
                    top_k_knn=upload_knn_k,
                    conf_threshold=upload_conf_threshold,
                    min_visual_sim=upload_min_sim
                )
                ref_type = upload_knn_info["pred_type"]
                ref_type_source = "upload_knn"

        # 1. Intent classification (State A: Attribute Shift vs State B: Category Mutation)
        intent_info = classify_cir_intent(modifier_text, clean_spec, ref_type=ref_type)
        intent_state = intent_info["intent_state"]
        target_type = intent_info["target_type"]

        # 2. Translate normalized modifier to optimal P2 English prompt
        prompt_ref_type = target_type if intent_state == "B_category_mutation" else ref_type
        en_prompt, is_fallback, untranslated = normalize_cir_modifier(
            modifier_text=modifier_text,
            parsed_spec=clean_spec,
            ref_type=prompt_ref_type,
            ref_gender=ref_gender,
            extra_style_terms=extra_style_terms
        )

        fallback_msg = None
        if not modifier_text.strip():
            is_fallback = True
        elif is_fallback or not en_prompt.strip():
            fallback_msg = "Chưa hiểu yêu cầu chỉnh sửa, đang hiển thị kết quả tương tự ảnh gốc"

        # 3. Compose CIR query vector and interpolate with lambda
        e_comb_raw = None
        if is_fallback or lam <= 0.0:
            cir_vec = e_img_raw
            t_pre = time.perf_counter()
        else:
            tok = self.engine.processor(text=[en_prompt], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            input_ids = tok["input_ids"].to(self.engine.device)
            t_pre = time.perf_counter()
            with torch.no_grad():
                mod_txt_emb = self.engine.model.encode_text(input_ids)
                cir_feat = self.engine.model.cir_combiner(t_ref_emb, mod_txt_emb)
                self._sync()
                e_comb_raw = cir_feat.float().cpu().numpy()[0]
            cir_vec = self.interpolate_cir_vector(e_img_raw, e_comb_raw, lam)

        t_fwd = time.perf_counter()
        cir_vec_c = self.engine.scorer.center_vectors(cir_vec[np.newaxis, :])[0]

        # 4. Compute similarity + soft bonus across gallery
        if self.engine.device == "cuda" and self.engine.t_emb_img_c is not None:
            q_t = torch.tensor(cir_vec_c, dtype=torch.float16, device=self.engine.device)
            sims_t = torch.matmul(self.engine.t_emb_img_c, q_t)
            sims_all = sims_t.float().cpu().numpy()
        else:
            sims_all = np.dot(self.engine.emb_img_c, cir_vec_c)

        # Build aligned soft-bonus spec (inherit target_type if locking so bonus agrees with anchor)
        bonus_spec = dict(clean_spec)
        if use_soft_bonus and self.engine.gamma > 0:
            if not bonus_spec.get("articleType") and target_type:
                if ref_type_source == "catalog_metadata" or upload_knn_info["confident"] or intent_state == "B_category_mutation":
                    bonus_spec["articleType"] = target_type
            bonus = self.engine.attr_idx.compute_attribute_soft_bonus(
                bonus_spec, cand_idxs,
                gamma_type=self.engine.gamma, gamma_gender=self.engine.gamma,
                gamma_colour=self.engine.gamma, gamma_brand=self.engine.gamma
            )
            sims_all = sims_all + bonus

        # 5. Determine Dynamic Category Anchor status & apply locking / soft fallback
        eff_mode = "off" if (not dynamic_category_anchor or anchor_mode == "off") else anchor_mode
        anchor_active = (eff_mode != "off")
        anchor_fallback_soft = False
        anchor_status_msg = ""

        if not anchor_active:
            anchor_status_msg = "Không khóa loại (Chế độ tự do)"
        elif eff_mode == "soft":
            anchor_fallback_soft = True
            anchor_status_msg = f"Xếp hạng mềm theo loại: {target_type or 'chưa xác định'}"
        elif intent_state == "B_category_mutation" and target_type:
            anchor_status_msg = f"Đang chuyển sang loại mới: {target_type}"
        elif ref_type_source == "catalog_metadata" and target_type:
            anchor_status_msg = f"Đang giữ loại: {target_type}"
        elif ref_type_source == "upload_knn":
            if upload_knn_info["confident"] and target_type:
                anchor_status_msg = f"Đang giữ loại (Upload ~{upload_knn_info['confidence']*100:.0f}%): {target_type}"
            else:
                anchor_fallback_soft = True
                anchor_status_msg = "Không đủ chắc chắn về loại gốc, dùng xếp hạng mềm"
        else:
            anchor_active = False
            anchor_status_msg = "Chưa xác định được loại gốc, dùng xếp hạng tự do"

        final_idxs, final_scores, match_tiers, exact_count, family_count, shortage_notice = self._select_with_category_anchor(
            sims_all=sims_all,
            top_k=top_k,
            target_type=target_type,
            anchor_active=anchor_active,
            anchor_mode=eff_mode,
            anchor_fallback_soft=anchor_fallback_soft,
            upload_type_probs=upload_knn_info["type_probs"] if anchor_fallback_soft else {},
            candidate_pool_k=candidate_pool_k,
            expand_pool_if_short=expand_pool_if_short
        )

        t_end = time.perf_counter()
        lat_ms = (t_end - t0) * 1000.0
        results = []
        for rank, (idx, score, m_tier) in enumerate(zip(final_idxs, final_scores, match_tiers), start=1):
            row = self.engine.df.iloc[int(idx)]
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
                "anchor_match_type": m_tier,
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
            # Dynamic Category Anchor & Slider metadata
            "intent_state": intent_state,
            "target_type": target_type,
            "ref_type": ref_type,
            "ref_type_source": ref_type_source,
            "upload_type_pred": upload_knn_info["pred_type"],
            "upload_type_confidence": upload_knn_info["confidence"],
            "upload_top1_sim": upload_knn_info["top1_sim"],
            "upload_type_confident": upload_knn_info["confident"],
            "anchor_active": anchor_active,
            "anchor_mode": eff_mode,
            "anchor_fallback_soft": anchor_fallback_soft,
            "anchor_status_msg": anchor_status_msg,
            "anchor_shortage_notice": shortage_notice,
            "exact_match_count": exact_count,
            "family_fallback_count": family_count,
            "modifier_weight": lam,
            "candidate_pool_k": candidate_pool_k,
            "typo_color_recovered": diag_info.get("typo_color_recovered"),
            "disambiguated_type_dropped": diag_info.get("disambiguated_type_dropped"),
            "query_vec_img_cosine": float(np.dot(cir_vec, e_img_raw)),
            "query_vec_comb_cosine": float(np.dot(cir_vec, e_comb_raw / max(float(np.linalg.norm(e_comb_raw)), 1e-12))) if e_comb_raw is not None else None,
            "breakdown": {
                "decode_ms": (t_dec - t0) * 1000.0,
                "preprocess_ms": (t_pre - t_dec) * 1000.0,
                "vit_forward_ms": (t_fwd - t_pre) * 1000.0,
                "gallery_search_ms": (t_end - t_fwd) * 1000.0,
                "rerank_ms": 0.0,
            }
        }
        return results, meta
