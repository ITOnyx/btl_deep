"""
Unified Visual & Multimodal Search Pipeline (src/search_pipeline.py)
Inference-only pipeline for SigLIP E-commerce Visual Search.
Features:
- Query parser V2 with disambiguation and unaccented input rules.
- Multi-level attribute relaxation (L0..L6) and exact filtering.
- Soft attribute scoring with gamma bonus.
- Product text embedding fusion (alpha).
- Dense + Sparse (BM25 CSR) hybrid search with RRF (k=60).
- Visual search with flip augmentation and k-NN attribute voting.
- Composed Image Retrieval (CIR: Image + Modifier Text).
- Strictly aligned to 38,644 in-scope catalog items.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_siglip_cir import load_trained_siglip_cir
from src.scoring import Scorer
from src.attr_index import AttributeIndex
from src.query_parser_v2 import QueryParserV2
from src.lexical import BM25Index, rrf_merge

class VisualSearchEngine:
    """
    Production-ready search engine supporting text, visual, and multimodal hybrid search.
    """
    def __init__(
        self,
        checkpoint_path: str = "data/checkpoints_siglip-vit/best_model.pt",
        stats_path: str = "data/checkpoints_siglip-vit/vit_zero_train_stats.npz",
        image_index_path: str = "index_v2/image_embeddings_v2.npy",
        text_index_path: str = "index_v2/text_embeddings_v2.npy",
        catalog_path: str = "data/catalog/catalog.parquet",
        device: Optional[str] = None,
        # Default hyper-parameters frozen from validation
        gamma: float = 0.3,
        alpha: float = 1.0,
        rrf_k: int = 60,
        enable_parser: bool = True,
        filter_mode: str = "relaxed", # "relaxed" | "soft" | "hard" | "none"
        enable_bm25: bool = True,
        enable_text_fusion: bool = False,
        min_candidates: int = 20,
        use_usage: bool = False,
        use_season: bool = False
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisualSearchEngine] Initializing on {self.device}...")

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.enable_parser = enable_parser
        self.filter_mode = filter_mode
        self.enable_bm25 = enable_bm25
        self.enable_text_fusion = enable_text_fusion
        self.min_candidates = min_candidates
        self.use_usage = use_usage
        self.use_season = use_season

        # 1. Attribute index & catalog (in_scope only: 38,644 items)
        self.attr_idx = AttributeIndex(catalog_path=catalog_path)
        self.df = self.attr_idx.df
        self.N = self.attr_idx.N
        self.item_ids = [str(x) for x in self.attr_idx.item_ids]
        self.id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        # 2. Query Parser V2 & BM25
        self.parser = QueryParserV2(brands_path="src/brands_v2.json")
        self.bm25 = BM25Index(catalog_path=catalog_path)

        # 3. Scorer & Centering
        self.scorer = Scorer(stats_path)

        # 4. Embeddings & Gallery Alignment
        print("[VisualSearchEngine] Loading gallery embeddings...")
        raw_img_emb = np.load(image_index_path)
        raw_txt_emb = np.load(text_index_path)

        # Align with in_scope catalog if id_map has 38,646 items
        id_map_path = Path("index_v2/id_map_v2.json")
        if id_map_path.exists() and len(raw_img_emb) > self.N:
            with open(id_map_path, "r", encoding="utf-8") as f:
                id_map = json.load(f)
            id_map_ids = [str(x["item_id"]) for x in id_map]
            in_scope_set = set(self.item_ids)
            valid_indices = [i for i, iid in enumerate(id_map_ids) if iid in in_scope_set]
            self.emb_img_raw = raw_img_emb[valid_indices]
            self.emb_txt_raw = raw_txt_emb[valid_indices]
        else:
            self.emb_img_raw = raw_img_emb
            self.emb_txt_raw = raw_txt_emb

        self.emb_img_c = self.scorer.center_vectors(self.emb_img_raw)
        self.emb_txt_c = self.scorer.center_vectors(self.emb_txt_raw)

        # Move to GPU FP16 tensors if CUDA available
        if self.device == "cuda":
            print("[VisualSearchEngine] Caching embeddings on GPU (FP16)...")
            self.t_emb_img_c = torch.tensor(self.emb_img_c, dtype=torch.float16, device=self.device)
            self.t_emb_txt_c = torch.tensor(self.emb_txt_c, dtype=torch.float16, device=self.device)
        else:
            self.t_emb_img_c = None
            self.t_emb_txt_c = None

        # 5. Model (Unified Vision, Text, and CIR Combiner)
        print(f"[VisualSearchEngine] Loading Multimodal SigLIP + CIR model on {self.device}...")
        self.model = load_trained_siglip_cir(checkpoint_path, device=self.device, fp16=(self.device == "cuda"))
        self.processor = self.model.processor

        # 6. Image Transform
        self.img_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 7. Families
        with open("eval/families.json", "r", encoding="utf-8") as f:
            self.families = json.load(f)
        self.gender_compat = self.families.get("gender_compatibility", {})
        self.art_families = self.families.get("article_type_families", {})
        self.col_families = self.families.get("colour_families", {})

        self.last_query_meta = {}
        print(f"[VisualSearchEngine] Engine ready. Gallery: {self.N} items.\n")

    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inp = self.processor(text=[text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            feat = self.model.encode_text(inp["input_ids"].to(self.device), None)
            vec = feat.cpu().numpy()[0]
        norm = max(float(np.linalg.norm(vec)), 1e-12)
        vec = vec / norm
        return self.scorer.center_vectors(vec[np.newaxis, :])[0]

    def encode_image(self, image_input: Union[str, Image.Image], flip: bool = False) -> np.ndarray:
        if isinstance(image_input, str):
            im = Image.open(image_input).convert("RGB")
        else:
            im = image_input.convert("RGB")
        if flip:
            im = ImageOps.mirror(im)
        tensor = self.img_transform(im).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            vec = feat.cpu().numpy()[0]
        norm = max(float(np.linalg.norm(vec)), 1e-12)
        vec = vec / norm
        return self.scorer.center_vectors(vec[np.newaxis, :])[0]

    def search_text(
        self,
        query: str,
        top_k: int = 10,
        gallery_mask: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes text query retrieval with configured inference pipeline.
        """
        t0 = time.perf_counter()
        if gallery_mask is None:
            gallery_mask = np.ones(self.N, dtype=bool)

        # 1. Parse Query
        t_parse_0 = time.perf_counter()
        if self.enable_parser:
            parsed = self.parser.parse(query)
            if not self.use_usage and "usage" in parsed:
                parsed["usage"] = None
            if not self.use_season and "season" in parsed:
                parsed["season"] = None
        else:
            parsed = {
                "raw_query": query, "articleType": None, "gender": None, "baseColour": None, "brand": None,
                "usage": None, "season": None, "confidence": 0.0
            }
        t_parse_ms = (time.perf_counter() - t_parse_0) * 1000.0

        # 2. Candidate Filtering
        t_filter_0 = time.perf_counter()
        tier = "all"
        if self.filter_mode == "hard":
            cand_idxs = self.attr_idx.filter_exact(parsed, mask=gallery_mask)
            if len(cand_idxs) == 0:
                cand_idxs = np.where(gallery_mask)[0]
                tier = "fallback_all"
            else:
                tier = "exact"
        elif self.filter_mode == "relaxed":
            cand_idxs, tier = self.attr_idx.filter_relaxed(parsed, min_candidates=self.min_candidates, mask=gallery_mask)
        else: # "soft" or "none"
            cand_idxs = np.where(gallery_mask)[0]
            tier = "all"
        t_filter_ms = (time.perf_counter() - t_filter_0) * 1000.0

        # 3. Encode Query
        t_enc_0 = time.perf_counter()
        q_vec_c = self.encode_text(query)
        t_enc_ms = (time.perf_counter() - t_enc_0) * 1000.0

        # 4. Dense Scoring
        t_search_0 = time.perf_counter()
        if self.device == "cuda" and self.t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=self.device)
            q_t = torch.tensor(q_vec_c, dtype=torch.float16, device=self.device)
            sims_img_t = torch.matmul(self.t_emb_img_c[cand_t], q_t)
            if self.enable_text_fusion and self.t_emb_txt_c is not None:
                sims_txt_t = torch.matmul(self.t_emb_txt_c[cand_t], q_t)
                dense_scores_t = self.alpha * sims_img_t + (1.0 - self.alpha) * sims_txt_t
            else:
                dense_scores_t = sims_img_t

            # Soft attribute bonus
            if self.filter_mode in ["relaxed", "soft"] and self.gamma > 0:
                bonus = self.attr_idx.compute_attribute_soft_bonus(
                    parsed, cand_idxs,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                dense_scores_t = dense_scores_t + torch.tensor(bonus, dtype=torch.float16, device=self.device)

            if self.enable_bm25:
                k_top = min(100, len(cand_idxs))
                top_pos = torch.topk(dense_scores_t, k=k_top)[1].cpu().numpy()
                dense_order = cand_idxs[top_pos]
                cand_mask = np.zeros(self.N, dtype=bool)
                cand_mask[cand_idxs] = True
                sparse_order, _ = self.bm25.search(query, top_k=100, mask=cand_mask)
                final_idxs, rrf_scores = rrf_merge(dense_order, sparse_order, k=self.rrf_k, top_k=top_k)
            else:
                k_top = min(top_k, len(cand_idxs))
                topk_vals, top_pos = torch.topk(dense_scores_t, k=k_top)
                final_idxs = cand_idxs[top_pos.cpu().numpy()]
                rrf_scores = topk_vals.float().cpu().numpy()
        else:
            sims_img = np.dot(self.emb_img_c[cand_idxs], q_vec_c)
            if self.enable_text_fusion:
                sims_txt = np.dot(self.emb_txt_c[cand_idxs], q_vec_c)
                dense_scores = self.alpha * sims_img + (1.0 - self.alpha) * sims_txt
            else:
                dense_scores = sims_img

            # Soft attribute bonus
            if self.filter_mode in ["relaxed", "soft"] and self.gamma > 0:
                bonus = self.attr_idx.compute_attribute_soft_bonus(
                    parsed, cand_idxs,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                dense_scores = dense_scores + bonus

            # Hybrid Search with BM25
            if self.enable_bm25:
                k_dense = min(100, len(cand_idxs))
                if len(cand_idxs) > k_dense:
                    part_pos = np.argpartition(-dense_scores, k_dense)[:k_dense]
                    dense_order = cand_idxs[part_pos[np.argsort(-dense_scores[part_pos])]]
                else:
                    dense_order = cand_idxs[np.argsort(-dense_scores)]
                cand_mask = np.zeros(self.N, dtype=bool)
                cand_mask[cand_idxs] = True
                sparse_order, _ = self.bm25.search(query, top_k=100, mask=cand_mask)
                final_idxs, rrf_scores = rrf_merge(dense_order, sparse_order, k=self.rrf_k, top_k=top_k)
            else:
                k_top = min(top_k, len(cand_idxs))
                if len(cand_idxs) > k_top:
                    part_pos = np.argpartition(-dense_scores, k_top)[:k_top]
                    sort_order = part_pos[np.argsort(-dense_scores[part_pos])]
                else:
                    sort_order = np.argsort(-dense_scores)[:k_top]
                final_idxs = cand_idxs[sort_order]
                rrf_scores = dense_scores[sort_order]

        t_search_ms = (time.perf_counter() - t_search_0) * 1000.0
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Format Results
        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, rrf_scores), start=1):
            row = self.df.iloc[idx]
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
                "tier": tier,
                "latency_ms": round(total_latency_ms, 2)
            })

        self.last_query_meta = {
            "parsed": parsed,
            "tier": tier,
            "candidates_count": len(cand_idxs),
            "breakdown": {
                "parse_ms": round(t_parse_ms, 2),
                "filter_ms": round(t_filter_ms, 2),
                "encode_ms": round(t_enc_ms, 2),
                "search_ms": round(t_search_ms, 2),
                "total_ms": round(total_latency_ms, 2)
            }
        }

        return results

    def search_image(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 10,
        gallery_mask: Optional[np.ndarray] = None,
        exclude_self_id: Optional[str] = None,
        use_flip_aug: bool = True,
        use_knn_voting: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes visual retrieval (image->image) with optional flip-aug and k-NN voting.
        """
        t0 = time.perf_counter()
        if gallery_mask is None:
            gallery_mask = np.ones(self.N, dtype=bool)

        cand_idxs = np.where(gallery_mask)[0]

        # 1. Encode Image
        raw_vec_c = self.encode_image(image_input, flip=False)
        if use_flip_aug:
            flip_vec_c = self.encode_image(image_input, flip=True)
            aug_vec = raw_vec_c + flip_vec_c
            aug_vec = aug_vec / max(float(np.linalg.norm(aug_vec)), 1e-12)
            query_vec_c = self.scorer.center_vectors(aug_vec[np.newaxis, :])[0]
        else:
            query_vec_c = raw_vec_c

        # 2. Dense Similarity
        if self.device == "cuda" and self.t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=self.device)
            q_t = torch.tensor(query_vec_c, dtype=torch.float16, device=self.device)
            sims_t = torch.matmul(self.t_emb_img_c[cand_t], q_t)
            sims = sims_t.float().cpu().numpy()
        else:
            sims = np.dot(self.emb_img_c[cand_idxs], query_vec_c)

        # Exclude self if provided
        if exclude_self_id and str(exclude_self_id) in self.id_to_idx:
            self_full_idx = self.id_to_idx[str(exclude_self_id)]
            pos = np.where(cand_idxs == self_full_idx)[0]
            if len(pos) > 0:
                sims[pos[0]] = -999.0

        # 3. k-NN Voting Rescore
        if use_knn_voting:
            top20_cand_pos = np.argsort(-sims)[:20]
            top20_rows = [self.df.iloc[cand_idxs[p]] for p in top20_cand_pos]
            voted_art = Counter([r["articleType"] for r in top20_rows]).most_common(1)[0][0]
            voted_gen = Counter([r["gender"] for r in top20_rows]).most_common(1)[0][0]
            voted_col = Counter([r["baseColour"] for r in top20_rows]).most_common(1)[0][0]

            voted_parsed = {"articleType": voted_art, "gender": voted_gen, "baseColour": voted_col, "brand": None}
            bonus = self.attr_idx.compute_attribute_soft_bonus(
                voted_parsed, cand_idxs,
                gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
            )
            sims = sims + bonus

        # 4. Top-K
        top_cand_pos = np.argsort(-sims)[:top_k]
        final_idxs = cand_idxs[top_cand_pos]
        final_scores = sims[top_cand_pos]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), start=1):
            row = self.df.iloc[idx]
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
                "latency_ms": round(latency_ms, 2)
            })

        self.last_query_meta = {
            "latency_ms": round(latency_ms, 2),
            "use_flip_aug": use_flip_aug,
            "use_knn_voting": use_knn_voting
        }

        return results

    def search_cir(
        self,
        ref_image: Union[str, Image.Image],
        modifier_text: str,
        top_k: int = 10,
        gallery_mask: Optional[np.ndarray] = None,
        use_soft_bonus: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes Composed Image Retrieval (CIR): Reference Image + Modifier Text -> Target Products.
        """
        t0 = time.perf_counter()
        if gallery_mask is None:
            gallery_mask = np.ones(self.N, dtype=bool)

        cand_idxs = np.where(gallery_mask)[0]

        # 1. Transform Image
        if isinstance(ref_image, str):
            im = Image.open(ref_image).convert("RGB")
        else:
            im = ref_image.convert("RGB")
        img_tensor = self.img_transform(im).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            img_tensor = img_tensor.half()

        # 2. Tokenize Modifier Text
        tok = self.processor(text=[modifier_text], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        input_ids = tok["input_ids"].to(self.device)

        # 3. Compose Query Feature
        with torch.no_grad():
            cir_feat = self.model.compose_query(img_tensor, input_ids)
            cir_vec = cir_feat.cpu().numpy()[0]

        norm = max(float(np.linalg.norm(cir_vec)), 1e-12)
        cir_vec = cir_vec / norm
        cir_vec_c = self.scorer.center_vectors(cir_vec[np.newaxis, :])[0]

        # 4. Dense Scoring
        parsed_mod = self.parser.parse(modifier_text)
        if self.device == "cuda" and self.t_emb_img_c is not None:
            cand_t = torch.tensor(cand_idxs, dtype=torch.long, device=self.device)
            q_t = torch.tensor(cir_vec_c, dtype=torch.float16, device=self.device)
            sims_t = torch.matmul(self.t_emb_img_c[cand_t], q_t)

            if use_soft_bonus and self.gamma > 0:
                bonus = self.attr_idx.compute_attribute_soft_bonus(
                    parsed_mod, cand_idxs,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                sims_t = sims_t + torch.tensor(bonus, dtype=torch.float16, device=self.device)

            k_top = min(top_k, len(cand_idxs))
            topk_vals, top_pos = torch.topk(sims_t, k=k_top)
            final_idxs = cand_idxs[top_pos.cpu().numpy()]
            final_scores = topk_vals.float().cpu().numpy()
        else:
            sims = np.dot(self.emb_img_c[cand_idxs], cir_vec_c)
            if use_soft_bonus and self.gamma > 0:
                bonus = self.attr_idx.compute_attribute_soft_bonus(
                    parsed_mod, cand_idxs,
                    gamma_type=self.gamma, gamma_gender=self.gamma, gamma_colour=self.gamma, gamma_brand=self.gamma
                )
                sims = sims + bonus
            k_top = min(top_k, len(cand_idxs))
            top_cand_pos = np.argsort(-sims)[:k_top]
            final_idxs = cand_idxs[top_cand_pos]
            final_scores = sims[top_cand_pos]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, (idx, score) in enumerate(zip(final_idxs, final_scores), start=1):
            row = self.df.iloc[idx]
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
                "latency_ms": round(latency_ms, 2)
            })

        self.last_query_meta = {
            "parsed_modifier": parsed_mod,
            "latency_ms": round(latency_ms, 2)
        }

        return results
