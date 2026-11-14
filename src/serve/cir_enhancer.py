"""
Cải tiến suy luận cho Composed Image Retrieval (CIR — Prompt 13):
- Luật 1 & AGENTS.md F.2: Không huấn luyện lại UpgradedAttentionCIRCombiner; mọi cải tiến nằm ở tầng suy luận.
- C1: Dịch modifier tiếng Việt -> tiếng Anh bằng bảng thuộc tính miền thời trang sinh từ catalog
      (màu, giới, kiểu dáng); token lạ giữ nguyên.
- C2: Hợp nhất text-only: score = beta * cos(combiner, img) + (1 - beta) * cos(text_query_attr, img).
- C3: Bonus thuộc tính từ parser_v2.1 trên phần modifier (đổi màu/giới), không gating slot không chắc.
- C4: PRF Rocchio nhẹ cho CIR (chỉ giữ nếu Delta mAP có CI loại trừ 0 và đạt 5(iv)).
"""
from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.eval.metrics import compute_weighted_attribute_match
from src.serve.query_rewriter import restore_diacritics_for_query
from src.serve.template_cache import TemplateCacheV5


# Bảng cụm từ nối thông dụng trong truy vấn chỉnh sửa CIR tiếng Việt -> tiếng Anh
CIR_CONNECTOR_PHRASES: List[Tuple[str, str]] = [
    ("kiểu dáng này nhưng đổi sang màu", "this style but change to color"),
    ("kiểu này nhưng đổi sang màu", "this style but change to color"),
    ("giống mẫu này nhưng chuyển sang màu", "similar style but change to color"),
    ("đổi sang màu", "change to color"),
    ("chuyển sang màu", "change to color"),
    ("thay bằng màu", "replace with color"),
    ("chuyển thành màu", "change to color"),
    ("mẫu tương tự màu", "similar style in color"),
    ("phiên bản dành cho", "version for"),
    ("mẫu dành cho", "style for"),
    ("đổi sang cho", "change for"),
    ("chuyển sang cho", "change for"),
    ("dành cho", "for"),
    ("đổi sang", "change to"),
    ("chuyển sang", "change to"),
    ("thay bằng", "replace with"),
    ("chuyển thành", "change to"),
    ("kiểu này nhưng là", "this style but"),
    ("phiên bản tương tự", "similar version"),
    ("mẫu tương tự", "similar style"),
    ("phiên bản", "version"),
    ("kiểu dáng", "style"),
    ("màu", "color"),
]


class CatalogModifierTranslator:
    """
    C1: Dịch modifier tiếng Việt sang tiếng Anh bằng bảng thuộc tính miền thời trang
    sinh trực tiếp từ data/catalog/catalog_rebuild.parquet (màu, giới, kiểu dáng).
    Token lạ không nằm trong bảng được giữ nguyên.
    """

    def __init__(self, catalog_path: str | pathlib.Path = "data/catalog/catalog_rebuild.parquet") -> None:
        df = pd.read_parquet(catalog_path)
        if "in_scope" in df.columns:
            df = df[df["in_scope"] == True].copy()

        phrase_map: Dict[str, str] = {}
        # 1. Thuộc tính từ catalog: baseColour, gender, articleType
        for col_en, col_vi in (
            ("baseColour", "baseColour_vi"),
            ("gender", "gender_vi"),
            ("articleType", "articleType_vi"),
        ):
            if col_en in df.columns and col_vi in df.columns:
                sub = df[[col_en, col_vi]].dropna().drop_duplicates()
                for _, row in sub.iterrows():
                    en_val = str(row[col_en]).strip().lower()
                    vi_val = str(row[col_vi]).strip().lower()
                    if vi_val and en_val:
                        phrase_map[vi_val] = en_val

        # Bổ sung một số cách gọi ngắn phổ biến của màu/giới trong catalog
        extra_catalog_aliases = {
            "xanh dương": "blue",
            "xanh lam": "blue",
            "xanh lá": "green",
            "xanh lục": "green",
            "xanh đen": "navy blue",
            "xanh tím than": "navy blue",
            "trắng kem": "off white",
            "đỏ đô": "maroon",
            "hồng đậm": "magenta",
            "vàng đồng": "gold",
            "bạc": "silver",
            "thép": "steel",
            "be": "beige",
            "nâu": "brown",
            "đen": "black",
            "trắng": "white",
            "đỏ": "red",
            "vàng": "yellow",
            "hồng": "pink",
            "tím": "purple",
            "xám": "grey",
            "cam": "orange",
            "nam": "men",
            "nữ": "women",
            "bé trai": "boys",
            "bé gái": "girls",
            "unisex": "unisex",
        }
        for k, v in extra_catalog_aliases.items():
            phrase_map.setdefault(k, v)

        for vi_conn, en_conn in CIR_CONNECTOR_PHRASES:
            phrase_map[vi_conn] = en_conn

        # Sắp xếp cụm từ dài trước để khớp greedy không cắt ngang từ ghép
        self.sorted_phrases: List[Tuple[str, str]] = sorted(
            phrase_map.items(), key=lambda kv: (-len(kv[0].split()), -len(kv[0]), kv[0])
        )
        self.phrase_map = phrase_map

    def translate(self, modifier_text: str) -> str:
        """
        Khôi phục dấu câu modifier tiếng Việt (nếu mất dấu), sau đó thay thế các cụm
        thuộc tính/từ nối sang tiếng Anh; các token lạ giữ nguyên.
        """
        restored = restore_diacritics_for_query(modifier_text).strip().lower()
        restored = re.sub(r"\s+", " ", restored)
        if not restored:
            return modifier_text

        tokens = restored.split(" ")
        n = len(tokens)
        out_tokens: List[str] = []
        i = 0
        while i < n:
            matched = False
            for max_span in range(min(6, n - i), 0, -1):
                candidate = " ".join(tokens[i : i + max_span])
                if candidate in self.phrase_map:
                    out_tokens.append(self.phrase_map[candidate])
                    i += max_span
                    matched = True
                    break
            if not matched:
                # Token lạ giữ nguyên
                out_tokens.append(tokens[i])
                i += 1

        return re.sub(r"\s+", " ", " ".join(out_tokens)).strip()


def score_cir_single_query(
    engine: Any,
    ref_img_emb_gpu: torch.Tensor,
    mod_txt_emb_gpu: torch.Tensor,
    parsed_mod: Dict[str, Any],
    tpl_cache: TemplateCacheV5,
    strict_cats: List[str],
    beta: float = 1.0,
    use_c3_bonus: bool = False,
    gamma: float = 1.0,
    rocchio_k: int = 0,
    rocchio_beta: float = 0.0,
) -> Tuple[np.ndarray, List[int]]:
    """
    Tính điểm xếp hạng CIR trên toàn bộ gallery cho 1 truy vấn:
    - Bước 1: Tính vector combiner từ (ref_img_emb_gpu, mod_txt_emb_gpu).
    - Bước 2 (C4 tùy chọn): PRF Rocchio nhẹ trên vector combiner.
    - Bước 3 (C2): Hợp nhất điểm cos(combiner, img) và cos(text_query_attr, img) theo hệ số beta.
    - Bước 4 (C3): Cộng bonus thuộc tính tier_soft từ parser_v2.1 trên phần modifier (không gating).
    """
    with torch.no_grad():
        cir_emb = engine.model.cir_combiner(ref_img_emb_gpu, mod_txt_emb_gpu)
        q_vec = cir_emb.float().cpu().numpy()
    q_vec = engine._apply_mean_centering(q_vec)[0].astype(np.float64)

    gal_embs = engine.embeddings
    comb_cos = (gal_embs @ q_vec).astype(np.float64)

    if rocchio_k > 0 and rocchio_beta > 0.0:
        top_k_idx = np.argsort(-comb_cos, kind="mergesort")[:rocchio_k]
        prf_vec = np.mean(gal_embs[top_k_idx].astype(np.float64), axis=0)
        q_new = q_vec + float(rocchio_beta) * prf_vec
        q_norm = float(np.linalg.norm(q_new))
        if q_norm > 1e-8:
            q_new = q_new / q_norm
        comb_cos = (gal_embs @ q_new).astype(np.float64)

    if beta < 1.0:
        tpl_vec = tpl_cache.lookup(parsed_mod)
        if tpl_vec is None:
            with torch.no_grad():
                t_vec = mod_txt_emb_gpu.float().cpu().numpy()
            tpl_vec = engine._apply_mean_centering(t_vec)[0]
        txt_cos = (gal_embs @ tpl_vec.astype(np.float64)).astype(np.float64)
        base_scores = float(beta) * comb_cos + (1.0 - float(beta)) * txt_cos
    else:
        base_scores = comb_cos

    if not use_c3_bonus:
        ranked = np.argsort(-base_scores, kind="mergesort")
        return base_scores, [int(x) for x in ranked]

    # C3: Bonus thuộc tính từ parser_v2.1 trên phần modifier (đổi màu/giới), không gating
    has_active_slot = bool(
        parsed_mod.get("articleType") or parsed_mod.get("gender") or parsed_mod.get("baseColour")
    )
    if not has_active_slot:
        ranked = np.argsort(-base_scores, kind="mergesort")
        return base_scores, [int(x) for x in ranked]

    final_scores = base_scores.copy()
    for idx, meta in enumerate(engine.id_map):
        mf = compute_weighted_attribute_match(
            parsed_mod, meta, weights=None, gender_policy="hybrid", strict_categories=strict_cats
        )
        if mf > 0.0:
            tier_b = 0.5 if mf >= 0.999 else 0.0
            final_scores[idx] += float(gamma) * (mf + tier_b)

    ranked = np.argsort(-final_scores, kind="mergesort")
    return final_scores, [int(x) for x in ranked]
