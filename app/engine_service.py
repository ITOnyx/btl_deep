"""
Service layer for Streamlit Application (Phase 6).
Handles:
- Hash Guard verification (Luật 6)
- Model & Faiss HNSW index loading (st.cache_resource)
- 4 Search Streams: Text->Image, Image->Image, CIR Catalog, CIR Upload
- Configuration mapping (cfg_v5_10, cfg_v5_09, cfg_v5_01, model_only, cir_v1_06, cir_v1_05, cir_v1_01)
- 3-tier taxonomy constraints & cross-domain guard (cfg_v5_10, cir_v1_06)
- In-memory image processing and validation (Luật 4, no disk writes for user uploads)
- Precise timing with torch.cuda.synchronize()
"""
from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import yaml

from src.model.siglip_model import load_siglip_model, compute_file_sha256
from src.preprocess.pipeline import SigLIPPreprocessor
from src.index.faiss_index import FaissHNSWIndex
from src.serve.parser import VietnameseQueryParser, compute_soft_attribute_match
from src.serve.parser_v3 import VietnameseQueryParserV3
from src.serve.query_rewriter import restore_diacritics_for_query
from src.serve.taxonomy import get_taxonomy, match_query_taxonomy


class HashGuardError(Exception):
    """Raised when checkpoint hash does not match index manifest (Rule 6)."""
    pass


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_and_load_image(file_data: Union[bytes, io.BytesIO], filename: str = "upload.jpg") -> Image.Image:
    """
    Validates uploaded image strictly in-memory:
    - Checks file extension
    - Checks byte size (<= 10MB)
    - Verifies valid image stream via Pillow
    - Converts to RGB
    - NEVER writes user image to disk
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Định dạng tệp '{ext}' không được hỗ trợ. Chỉ chấp nhận các định dạng: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    if isinstance(file_data, io.BytesIO):
        raw_bytes = file_data.getvalue()
    elif isinstance(file_data, bytes):
        raw_bytes = file_data
    else:
        raise ImageValidationError("Dữ liệu ảnh không hợp lệ (phải là bytes hoặc BytesIO).")

    if len(raw_bytes) == 0:
        raise ImageValidationError("Tệp tải lên rỗng (0 bytes).")

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        mb_size = len(raw_bytes) / (1024 * 1024)
        raise ImageValidationError(
            f"Dung lượng tệp ({mb_size:.2f} MB) vượt quá giới hạn cho phép (tối đa 10 MB)."
        )

    try:
        bio = io.BytesIO(raw_bytes)
        with Image.open(bio) as img_check:
            img_check.verify()

        bio_read = io.BytesIO(raw_bytes)
        img = Image.open(bio_read).convert("RGB")
        w, h = img.size
        if w < 10 or h < 10:
            raise ImageValidationError(f"Kích thước ảnh quá nhỏ ({w}x{h} px). Yêu cầu tối thiểu 10x10 px.")
        if w > 8000 or h > 8000:
            raise ImageValidationError(f"Kích thước ảnh quá lớn ({w}x{h} px). Yêu cầu tối đa 8000x8000 px.")
        return img
    except ImageValidationError:
        raise
    except Exception as e:
        raise ImageValidationError(f"Không thể giải mã hình ảnh. Tệp có thể bị lỗi hoặc hỏng định dạng ({e}).")


# Ảnh upload trùng (gần như) một sản phẩm gallery: cosine ảnh-ảnh >= ngưỡng này thì loại khỏi kết quả CIR Upload.
# Hiệu chỉnh: ảnh upload đúng file catalog cho cosine ~1.0; hai sản phẩm KHÁC nhau đạt >= 0.995 ở 0.47% sản phẩm
# của reference_gallery (đa số là ảnh trùng thật, cos >= 0.999), còn ở mức 0.99 là 1.73% -> chọn 0.995.
NEAR_DUPLICATE_COSINE = 0.995
_NEAR_DUP_SLACK = 3  # lấy dư vài kết quả để vẫn đủ top_k sau khi loại ảnh trùng

# Ảnh upload có cosine top-1 tới gallery thấp hơn ngưỡng này -> coi là không nhận diện được lĩnh vực (ràng buộc mềm + cảnh báo).
# Ngưỡng phụ thuộc mật độ gallery (gallery càng dày, ảnh lạ càng dễ có láng giềng gần):
# - test_gallery (1,290): 0.50 -> 7/31 ảnh không phải thời trang lọt qua, 9/300 ảnh thời trang thật bị chuyển sang ràng buộc mềm.
# - reference_gallery (38,646): 0.70 -> 6/32 ảnh lạ lọt qua, 0/300 ảnh thật bị chặn (0.50 để lọt 18/32).
FASHION_MIN_TOP1_COSINE_BY_GALLERY = {"reference_gallery": 0.70}
FASHION_MIN_TOP1_COSINE = 0.50  # mặc định cho gallery khác (vd. test_gallery)


class AppSearchService:
    """
    Unified search service for Streamlit application.
    Enforces Rule 4 (Single Preprocessing Pipeline) and Rule 6 (Hash Guard).
    """

    def __init__(
        self,
        index_dir: str = "index/test_gallery",
        config_path: str = "config.yaml",
        device: Optional[str] = None,
        ef_search: int = 128
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        p_idx = Path(index_dir)
        self.index_dir = p_idx if p_idx.is_absolute() else (self.project_root / index_dir)
        p_cfg = Path(config_path)
        self.config_path = p_cfg if p_cfg.is_absolute() else (self.project_root / config_path)
        self.ef_search = ef_search
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load config
        if not self.config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 2. Hash Guard (Rule 6)
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Không tìm thấy manifest.json tại {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        expected_hash = self.cfg["checkpoint"]["checkpoint_hash"].strip().lower()
        actual_manifest_hash = str(self.manifest.get("checkpoint_hash", "")).strip().lower()

        if actual_manifest_hash != expected_hash:
            raise HashGuardError(
                f"HASH GUARD TỪ CHỐI NẠP (Luật 6): Lệch mã băm SHA-256!\n"
                f"- Checkpoint trong config.yaml: {expected_hash}\n"
                f"- Manifest trong {self.index_dir.name}: {actual_manifest_hash}"
            )
        self.hash_guard_ok = True

        # 3. Load id_map and gallery embeddings
        with open(self.index_dir / "id_map.json", "r", encoding="utf-8") as f:
            self.id_map: List[Dict[str, Any]] = json.load(f)

        self.item_id_to_idx = {str(item["item_id"]): i for i, item in enumerate(self.id_map)}
        self.embeddings = np.load(self.index_dir / "image_embeddings.npy").astype(np.float32)

        # Load FAISS HNSW Index
        self.hnsw = FaissHNSWIndex.load(self.index_dir / "faiss_hnsw.bin", ef_search=self.ef_search)

        # 4. Load SigLIP Model & Preprocessor (Rule 4: identical preprocessing)
        ckpt_path = self.project_root / self.cfg["paths"]["checkpoint_file"]
        self.preprocessor = SigLIPPreprocessor(device=self.device)
        self.model = load_siglip_model(
            checkpoint_path=str(ckpt_path),
            expected_hash=expected_hash,
            device=self.device,
            fp16=(self.device == "cuda")
        )

        # 5. Vietnamese Parser
        self.parser = VietnameseQueryParser()
        self.parser_v3 = VietnameseQueryParserV3()  # dùng cho cfg_v5_11 và cir_v1_07 (DEV SET)

        # 6. Validate domain_ranking configuration weights
        dr_cfg = self.cfg.get("domain_ranking", {})
        w = dr_cfg.get("weights", {})
        if w:
            w_grp = abs(w.get("group_mismatch_penalty", 0.8))
            w_col = w.get("color_match_bonus", 0.4)
            w_typ = w.get("type_match_bonus", 0.5)
            w_typ_mis = abs(w.get("type_mismatch_penalty", 0.1))
            assert w_grp > w_col, "Cấu hình vi phạm: |group_mismatch_penalty| phải > color_match_bonus"
            assert w_typ > w_col, "Cấu hình vi phạm: type_match_bonus phải > color_match_bonus"
            assert w_typ_mis < w_grp, "Cấu hình vi phạm: |type_mismatch_penalty| phải < |group_mismatch_penalty|"

    def get_system_metadata(self) -> Dict[str, Any]:
        """Returns metadata for sidebar display."""
        vram_mb = 0.0
        vram_total_gb = 0.0
        device_name = "CPU"
        if torch.cuda.is_available() and "cuda" in self.device:
            device_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        return {
            "device": self.device,
            "device_name": device_name,
            "vram_allocated_mb": vram_mb,
            "vram_total_gb": vram_total_gb,
            "checkpoint_hash": self.cfg["checkpoint"]["checkpoint_hash"],
            "index_name": self.index_dir.name,
            "total_items": len(self.id_map),
            "hash_guard_ok": getattr(self, "hash_guard_ok", False),
            "ef_search": self.ef_search,
            "seed": self.cfg.get("seed", 42),
            "is_reference_gallery": "reference" in self.index_dir.name,
        }

    def _execute_ranking(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        parsed_query: Optional[Dict[str, Any]] = None,
        config_id: str = "cfg_v5_09",
        candidate_pool: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Executes search ranking and scores computation for legacy configurations:
        - cfg_v5_09: gamma=1.0, tier_soft (mf + 0.5 if mf >= 0.999)
        - cfg_v5_01: gamma=1.5, tier_soft (mf + 0.5 if mf >= 0.999)
        - model_only: gamma=0.0 (pure cosine)
        - cir_v1_05: gamma=1.0, tier_soft
        - cir_v1_01: gamma=0.0 (pure cosine)
        """
        if config_id in ("cfg_v5_09", "cir_v1_05"):
            gamma = 1.0
            use_bonus = True
        elif config_id == "cfg_v5_01":
            gamma = 1.5
            use_bonus = True
        else:
            gamma = 0.0
            use_bonus = False

        fetch_k = top_k if not use_bonus else min(len(self.id_map), max(top_k, candidate_pool))
        scores, indices = self.hnsw.search(query_vec, top_k=fetch_k)
        cand_scores = scores[0]
        cand_indices = indices[0]

        results = []
        for rank_idx, (idx, base_score) in enumerate(zip(cand_indices, cand_scores)):
            if idx < 0:
                continue
            meta = dict(self.id_map[int(idx)])
            base_cos = float(base_score)
            bonus = 0.0

            if use_bonus and parsed_query and gamma > 0:
                mf = compute_soft_attribute_match(parsed_query, meta, gender_policy="hybrid")
                if mf > 0.0:
                    tier_b = 0.5 if mf >= 0.999 else 0.0
                    bonus = gamma * (mf + tier_b)

            final_score = base_cos + bonus
            raw_path = meta.get("path", "")
            img_full_path = self.project_root / raw_path

            tax = get_taxonomy(meta)

            results.append({
                "rank": rank_idx + 1,
                "item_id": str(meta.get("item_id", meta.get("product_id", "N/A"))),
                "product_id": str(meta.get("product_id", meta.get("item_id", "N/A"))),
                "title": f"{meta.get('gender', '')} {meta.get('baseColour', '')} {meta.get('articleType', '')}".strip() or f"Sản phẩm #{meta.get('item_id')}",
                "articleType": meta.get("articleType", "N/A"),
                "gender": meta.get("gender", "N/A"),
                "baseColour": meta.get("baseColour", "N/A"),
                "domain": tax[0],
                "group": tax[1],
                "image_path": str(img_full_path),
                "image_exists": img_full_path.exists(),
                "cosine_score": base_cos,
                "attribute_bonus": bonus,
                "final_score": final_score,
                "index_pos": int(idx),
            })

        if use_bonus:
            results.sort(key=lambda x: x["final_score"], reverse=True)
            for new_rank, r in enumerate(results[:top_k], 1):
                r["rank"] = new_rank

        return results[:top_k]

    def _exact_search(self, query_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cosine chính xác tới toàn bộ gallery, sắp giảm dần (cùng định dạng [1, N] với hnsw.search)."""
        emb_n = getattr(self, "_emb_normed", None)
        if emb_n is None:
            emb_n = self.embeddings / np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12)
            self._emb_normed = emb_n
        qv = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        sims = emb_n @ qv
        order = np.argsort(-sims)
        return sims[order][None, :].astype(np.float32), order[None, :].astype(np.int64)

    def _execute_ranking_3tier(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        target_domain: Optional[str] = None,
        target_group: Optional[str] = None,
        target_type: Optional[str] = None,
        target_gender: Optional[str] = None,
        target_color: Optional[str] = None,
        domain_hard_filter: bool = True,
        unrecognized_domain_warning: bool = False,
        gender_scale: float = 1.0,
        inject_attr_match: bool = False,
        inject_type: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes 3-tier taxonomy ranking & cross-domain guard:
        - Domain hard filter: completely excludes candidates from different domains.
        - Soft ranking within domain: type match bonus, group mismatch penalty, color/gender bonuses.
        """
        dr_cfg = self.cfg.get("domain_ranking", {})
        pool_k = dr_cfg.get("candidate_pool_k", 200)
        weights = dr_cfg.get("weights", {})

        w_dom_mis = weights.get("domain_mismatch_penalty", 2.0)
        w_grp_mis = weights.get("group_mismatch_penalty", 0.8)
        w_typ_mis = weights.get("type_mismatch_penalty", 0.1)
        w_typ_match = weights.get("type_match_bonus", 0.5)
        w_col_match = weights.get("color_match_bonus", 0.4)
        w_gen_match = weights.get("gender_match_bonus", 0.2) * gender_scale
        w_gen_mis = weights.get("gender_mismatch_penalty", 0.2) * gender_scale

        total_items = len(self.id_map)
        k_steps = [pool_k, min(400, total_items), min(800, total_items), total_items]
        k_steps = sorted(list(dict.fromkeys(k_steps)))

        results = []
        for fetch_k in k_steps:
            fetch_k = min(total_items, max(top_k, fetch_k))
            if fetch_k >= total_items:
                # Bước cuối: HNSW với k = toàn bộ gallery chỉ trả được phần đồ thị duyệt tới (vd. 916/38,646, còn lại -1),
                # nên lĩnh vực mục tiêu nằm xa vector truy vấn có thể bị bỏ sót hoàn toàn -> tính cosine chính xác.
                scores, indices = self._exact_search(query_vec)
            else:
                scores, indices = self.hnsw.search(query_vec, top_k=fetch_k)
            cand_scores = scores[0]
            cand_indices = indices[0]

            if inject_type and target_type:
                # DEV SET: loại sản phẩm hiếm có cosine thấp có thể nằm ngoài pool -> bổ sung top-N cùng loại theo cosine
                tmap = getattr(self, "_type_index_map", None)
                if tmap is None:
                    tmap = {}
                    for ii, mm in enumerate(self.id_map):
                        tmap.setdefault(get_taxonomy(mm)[2], []).append(ii)
                    self._type_index_map = {k: np.asarray(v, dtype=np.int64) for k, v in tmap.items()}
                    tmap = self._type_index_map
                tid = tmap.get(target_type)
                if tid is not None and len(tid):
                    qv = np.asarray(query_vec, dtype=np.float32).reshape(-1)
                    qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
                    emb = np.asarray(self.embeddings, dtype=np.float32)[tid]
                    en = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
                    cs = en @ qv
                    top = list(np.argsort(-cs)[:30])
                    if inject_attr_match and (target_gender or target_color):
                        # CIR: ứng viên cùng loại khớp giới tính/màu mục tiêu có thể nằm xa vector tham chiếu
                        # (vd. áo thun nam xám -> "cho nữ màu hồng") -> bổ sung top-30 cùng loại khớp từng tổ hợp thuộc tính.
                        gens = np.asarray([self.id_map[int(i)].get("gender") for i in tid])
                        cols = np.asarray([self.id_map[int(i)].get("baseColour") for i in tid])
                        g_ok = np.isin(gens, [target_gender, "Unisex"]) if target_gender else np.ones(len(tid), bool)
                        c_ok = (cols == target_color) if target_color else np.ones(len(tid), bool)
                        for mask in (g_ok, g_ok & c_ok):
                            sub = np.flatnonzero(mask)
                            top.extend(sub[np.argsort(-cs[sub])[:30]].tolist())
                    have = set(int(x) for x in cand_indices if x >= 0)
                    add_i, add_s = [], []
                    for j in top:
                        ii = int(tid[j])
                        if ii not in have:
                            have.add(ii)
                            add_i.append(ii)
                            add_s.append(float(cs[j]))
                    cand_indices = np.concatenate([np.asarray(cand_indices), np.asarray(add_i, dtype=np.int64)])
                    cand_scores = np.concatenate([np.asarray(cand_scores, dtype=np.float32), np.asarray(add_s, dtype=np.float32)])

            results = []
            for idx, base_score in zip(cand_indices, cand_scores):
                if idx < 0:
                    continue
                meta = dict(self.id_map[int(idx)])
                base_cos = float(base_score)

                d_c, g_c, t_c = get_taxonomy(meta)

                # Rule 1: Hard filter on domain (if enabled and target_domain is known)
                if domain_hard_filter and target_domain and d_c != target_domain:
                    continue

                tax_bonus = 0.0
                tax_penalty = 0.0
                col_bonus = 0.0
                gen_bonus = 0.0

                if target_domain:
                    if d_c == target_domain:
                        if target_group:
                            if g_c == target_group:
                                if target_type:
                                    if t_c == target_type:
                                        tax_bonus += w_typ_match
                                    else:
                                        tax_penalty += w_typ_mis
                            else:
                                tax_penalty += w_grp_mis
                    else:
                        tax_penalty += w_dom_mis

                if target_color:
                    if meta.get("baseColour") == target_color:
                        col_bonus += w_col_match

                if target_gender:
                    cand_gen = meta.get("gender")
                    if cand_gen == target_gender:
                        gen_bonus += w_gen_match
                    elif cand_gen == "Unisex":
                        gen_bonus += w_gen_match * 0.5
                    else:
                        gen_bonus -= w_gen_mis

                bonus_net = tax_bonus - tax_penalty + col_bonus + gen_bonus
                final_score = base_cos + bonus_net

                raw_path = meta.get("path", "")
                img_full_path = self.project_root / raw_path

                results.append({
                    "rank": 0,
                    "item_id": str(meta.get("item_id", meta.get("product_id", "N/A"))),
                    "product_id": str(meta.get("product_id", meta.get("item_id", "N/A"))),
                    "title": f"{meta.get('gender', '')} {meta.get('baseColour', '')} {meta.get('articleType', '')}".strip() or f"Sản phẩm #{meta.get('item_id')}",
                    "articleType": meta.get("articleType", "N/A"),
                    "gender": meta.get("gender", "N/A"),
                    "baseColour": meta.get("baseColour", "N/A"),
                    "domain": d_c,
                    "group": g_c,
                    "image_path": str(img_full_path),
                    "image_exists": img_full_path.exists(),
                    "cosine_score": base_cos,
                    "tax_bonus": tax_bonus,
                    "tax_penalty": tax_penalty,
                    "color_bonus": col_bonus,
                    "gender_bonus": gen_bonus,
                    "attribute_bonus": bonus_net,
                    "final_score": final_score,
                    "constraint_mode": "Cứng (loại bỏ khác lĩnh vực)" if (domain_hard_filter and target_domain) else "Mềm",
                    "unrecognized_domain": unrecognized_domain_warning,
                    "index_pos": int(idx),
                })

            if len(results) >= top_k or not domain_hard_filter or not target_domain or fetch_k >= total_items:
                break

        results.sort(key=lambda x: x["final_score"], reverse=True)
        final_top = results[:top_k]
        for r_idx, r in enumerate(final_top, start=1):
            r["rank"] = r_idx

        return final_top

    def search_text_to_image(
        self,
        query_text: str,
        config_id: str = "cfg_v5_09",
        top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """
        Stream 1: Text-to-Image.
        Encodes query_text with SigLIP text encoder, applies parser, and re-ranks.
        Supports cfg_v5_10 (3-tier taxonomy guard) and legacy cfg_v5_09 / cfg_v5_01 / model_only.
        Returns: (results, parsed_attributes, elapsed_ms)
        """
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # cfg_v5_11 (DEV SET): khôi phục dấu trước khi encode (đúng mô tả docs bước 1) + parser v3
        is_v11 = (config_id == "cfg_v5_11")
        encode_text = restore_diacritics_for_query(query_text) if is_v11 else query_text

        # Rule 4: single preprocessing path
        inputs = self.preprocessor.preprocess_single_text(encode_text, device=self.device)
        with torch.no_grad():
            emb = self.model.encode_text(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask")
            )
            q_vec = emb.float().cpu().numpy()

        if config_id == "model_only":
            parsed = {}
        elif is_v11:
            parsed = self.parser_v3.parse(query_text)
        else:
            parsed = self.parser.parse(query_text)

        if config_id in ("cfg_v5_10", "cfg_v5_11"):
            tax_match = match_query_taxonomy(query_text, parsed)
            t_dom = tax_match["domain"]
            t_grp = tax_match["group"]
            t_typ = tax_match["articleType"]
            t_col = parsed.get("baseColour", [None])[0] if parsed.get("baseColour") else None
            t_gen = parsed.get("gender")

            results = self._execute_ranking_3tier(
                query_vec=q_vec,
                top_k=top_k,
                target_domain=t_dom,
                target_group=t_grp,
                target_type=t_typ,
                target_gender=t_gen,
                target_color=t_col,
                domain_hard_filter=(t_dom is not None),
                unrecognized_domain_warning=False,
                inject_type=is_v11,
                gender_scale=(3.0 if is_v11 else 1.0),  # DEV SET: giới tính nêu tường minh phải thắng thưởng màu
            )
        else:
            results = self._execute_ranking(
                query_vec=q_vec,
                top_k=top_k,
                parsed_query=parsed,
                config_id=config_id
            )

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return results, parsed, elapsed_ms

    def search_image_to_image(
        self,
        image_input: Union[Image.Image, Path, str],
        top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Stream 2: Image-to-Image.
        Pure vision similarity (always model_only, no text parser).
        Returns: (results, elapsed_ms)
        """
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pixel_values = self.preprocessor.preprocess_single_image(
            image_input, device=self.device, half_precision=(self.device == "cuda")
        )
        with torch.no_grad():
            img_emb = self.model.encode_image(pixel_values)
            q_vec = img_emb.float().cpu().numpy()

        results = self._execute_ranking(
            query_vec=q_vec,
            top_k=top_k,
            parsed_query=None,
            config_id="model_only"
        )

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return results, elapsed_ms

    def search_cir_catalog(
        self,
        product_id: str,
        modifier_text: str,
        config_id: str = "cir_v1_05",
        top_k: int = 10,
        exclude_reference: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """
        Stream 3: CIR Catalog.
        exclude_reference: loại chính sản phẩm tham chiếu khỏi kết quả (mặc định). Truyền False để tái lập
        số liệu eval cũ của cir_v1_05/cir_v1_06 (trước đây sản phẩm gốc có thể nằm trong top-k).
        Takes a catalog product_id, uses pre-indexed vector, fuses with modifier_text.
        Supports cir_v1_06 (3-tier taxonomy preservation) and legacy cir_v1_05 / cir_v1_01.
        Returns: (results, parsed_attributes, elapsed_ms)
        """
        pid_clean = str(product_id).strip()
        if pid_clean not in self.item_id_to_idx:
            raise ValueError(
                f"Không tìm thấy mã sản phẩm '{pid_clean}' trong gallery hiện tại ({self.index_dir.name})."
            )

        cat_idx = self.item_id_to_idx[pid_clean]

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        mod_inputs = self.preprocessor.preprocess_single_text(modifier_text, device=self.device)
        with torch.no_grad():
            mod_txt_emb = self.model.encode_text(
                input_ids=mod_inputs["input_ids"],
                attention_mask=mod_inputs.get("attention_mask")
            )
            ref_vec = self.embeddings[cat_idx : cat_idx + 1]
            ref_img_emb = torch.from_numpy(ref_vec).to(self.device)
            if self.device == "cuda":
                ref_img_emb = ref_img_emb.half()

            if config_id == "cir_v1_07":
                # combiner chưa huấn luyện hiệu quả: dùng vector ảnh tham chiếu + ràng buộc thuộc tính (DEV SET)
                q_vec = ref_img_emb.float().cpu().numpy()
            else:
                cir_emb = self.model.cir_combiner(ref_img_emb, mod_txt_emb)
                q_vec = cir_emb.float().cpu().numpy()

        if config_id == "cir_v1_07":
            parsed = self.parser_v3.parse(modifier_text)
        else:
            parsed = self.parser.parse(modifier_text) if config_id != "cir_v1_01" else {}

        if config_id in ("cir_v1_06", "cir_v1_07"):
            ref_meta = self.id_map[cat_idx]
            ref_d, ref_g, ref_t = get_taxonomy(ref_meta)
            ref_gen = ref_meta.get("gender")

            # Check if modifier requested a change of domain/type
            mod_tax = match_query_taxonomy(modifier_text, parsed)
            if mod_tax["domain"] is not None:
                target_d = mod_tax["domain"]
                target_g = mod_tax["group"]
                target_t = mod_tax["articleType"]
            else:
                # PRESERVE reference product's domain, group, and type!
                target_d = ref_d
                target_g = ref_g
                target_t = ref_t

            target_gen = parsed.get("gender") or ref_gen
            target_col = parsed.get("baseColour", [None])[0] if parsed.get("baseColour") else None
            if config_id == "cir_v1_07" and target_col is None:
                target_col = ref_meta.get("baseColour")  # giữ màu gốc khi modifier không đổi màu

            results = self._execute_ranking_3tier(
                query_vec=q_vec,
                top_k=top_k + (1 if (exclude_reference or config_id == "cir_v1_07") else 0),
                target_domain=target_d,
                target_group=target_g,
                target_type=target_t,
                target_gender=target_gen,
                target_color=target_col,
                domain_hard_filter=True,
                unrecognized_domain_warning=False,
                # cir_v1_07: khi modifier nêu giới tính tường minh, giới tính phải thắng màu gốc được giữ lại
                gender_scale=(5.0 if (config_id == "cir_v1_07" and parsed.get("gender")) else 1.0),
                inject_type=(config_id == "cir_v1_07"),
                inject_attr_match=(config_id == "cir_v1_07"),
            )
        else:
            results = self._execute_ranking(
                query_vec=q_vec,
                top_k=top_k + (1 if exclude_reference else 0),
                parsed_query=parsed,
                config_id=config_id
            )

        if exclude_reference or config_id == "cir_v1_07":
            results = [r for r in results if str(r.get("item_id", "")) != pid_clean][:top_k]
            for r_i, r in enumerate(results, start=1):
                r["rank"] = r_i

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return results, parsed, elapsed_ms

    def _predict_image_taxonomy(
        self, img_emb: torch.Tensor
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], bool]:
        """
        Infers taxonomy of an uploaded image via top-5 nearest neighbors in gallery.
        Runs in < 1 ms without external dependencies.
        Returns: (domain, group, type, gender, is_confident)
        """
        q_vec = img_emb.float().cpu().numpy()
        scores, indices = self.hnsw.search(q_vec, top_k=5)
        cand_scores = scores[0]
        cand_indices = indices[0]

        min_top1 = FASHION_MIN_TOP1_COSINE_BY_GALLERY.get(self.index_dir.name, FASHION_MIN_TOP1_COSINE)
        if len(cand_scores) == 0 or cand_scores[0] < min_top1:
            return (None, None, None, None, False)

        tax_votes = []
        gen_votes = []
        for idx in cand_indices:
            if idx >= 0:
                meta = self.id_map[int(idx)]
                tax_votes.append(get_taxonomy(meta))
                gen_votes.append(meta.get("gender"))

        if not tax_votes:
            return (None, None, None, None, False)

        top_dom = Counter([t[0] for t in tax_votes]).most_common(1)[0]
        top_grp = Counter([t[1] for t in tax_votes]).most_common(1)[0]
        top_typ = Counter([t[2] for t in tax_votes]).most_common(1)[0]
        top_gen = Counter([g for g in gen_votes if g]).most_common(1)[0][0] if any(gen_votes) else None

        is_confident = (top_dom[1] >= 3)
        return (top_dom[0], top_grp[0], top_typ[0], top_gen, is_confident)

    def search_cir_upload(
        self,
        uploaded_image: Image.Image,
        modifier_text: str,
        config_id: str = "cir_v1_05",
        top_k: int = 10,
        exclude_near_duplicates: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
        """
        Stream 4: CIR Upload.
        exclude_near_duplicates: loại các sản phẩm gallery gần như trùng ảnh upload
        (cosine ảnh >= NEAR_DUPLICATE_COSINE), vd. khi người dùng tải lên chính ảnh có trong catalog.
        Takes an in-memory PIL image, encodes via SigLIP vision, fuses with modifier_text.
        Supports cir_v1_06 (3-tier taxonomy preservation) and legacy cir_v1_05 / cir_v1_01.
        Returns: (results, parsed_attributes, elapsed_ms)
        """
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pixel_values = self.preprocessor.preprocess_single_image(
            uploaded_image, device=self.device, half_precision=(self.device == "cuda")
        )
        mod_inputs = self.preprocessor.preprocess_single_text(modifier_text, device=self.device)

        with torch.no_grad():
            ref_img_emb = self.model.encode_image(pixel_values)
            mod_txt_emb = self.model.encode_text(
                input_ids=mod_inputs["input_ids"],
                attention_mask=mod_inputs.get("attention_mask")
            )
            if config_id == "cir_v1_07":
                q_vec = ref_img_emb.float().cpu().numpy()
            else:
                cir_emb = self.model.cir_combiner(ref_img_emb, mod_txt_emb)
                q_vec = cir_emb.float().cpu().numpy()

        if config_id == "cir_v1_07":
            parsed = self.parser_v3.parse(modifier_text)
        else:
            parsed = self.parser.parse(modifier_text) if config_id != "cir_v1_01" else {}

        if config_id in ("cir_v1_06", "cir_v1_07"):
            inf_d, inf_g, inf_t, inf_gen, is_confident = self._predict_image_taxonomy(ref_img_emb)

            # Check if modifier requested a change of domain/type
            mod_tax = match_query_taxonomy(modifier_text, parsed)
            if mod_tax["domain"] is not None:
                target_d = mod_tax["domain"]
                target_g = mod_tax["group"]
                target_t = mod_tax["articleType"]
                hard_filter = True
                unrec_warn = False
            elif is_confident:
                target_d = inf_d
                target_g = inf_g
                target_t = inf_t
                hard_filter = True
                unrec_warn = False
            else:
                # Unrecognized domain or out-of-catalog image: fallback to soft constraint
                target_d = None
                target_g = None
                target_t = None
                hard_filter = False
                unrec_warn = True

            target_gen = parsed.get("gender") or inf_gen
            target_col = parsed.get("baseColour", [None])[0] if parsed.get("baseColour") else None

            results = self._execute_ranking_3tier(
                query_vec=q_vec,
                top_k=top_k + (_NEAR_DUP_SLACK if exclude_near_duplicates else 0),
                target_domain=target_d,
                target_group=target_g,
                target_type=target_t,
                target_gender=target_gen,
                target_color=target_col,
                domain_hard_filter=hard_filter,
                unrecognized_domain_warning=unrec_warn,
                gender_scale=(5.0 if (config_id == "cir_v1_07" and parsed.get("gender")) else 1.0),
                inject_type=(config_id == "cir_v1_07"),
                inject_attr_match=(config_id == "cir_v1_07"),
            )
        else:
            results = self._execute_ranking(
                query_vec=q_vec,
                top_k=top_k + (_NEAR_DUP_SLACK if exclude_near_duplicates else 0),
                parsed_query=parsed,
                config_id=config_id
            )

        if exclude_near_duplicates and results:
            img_vec = ref_img_emb.float().cpu().numpy().reshape(-1)
            img_vec = img_vec / max(float(np.linalg.norm(img_vec)), 1e-12)
            pos = np.asarray([r["index_pos"] for r in results], dtype=np.int64)
            emb = np.asarray(self.embeddings[pos], dtype=np.float32)
            emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            sims = emb @ img_vec
            results = [r for r, sim in zip(results, sims) if sim < NEAR_DUPLICATE_COSINE]
        results = results[:top_k]
        for r_i, r in enumerate(results, start=1):
            r["rank"] = r_i

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return results, parsed, elapsed_ms
