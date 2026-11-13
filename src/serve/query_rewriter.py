"""
Module: src.serve.query_rewriter
Khôi phục dấu tiếng Việt nhẹ theo từ điển catalog + cụm cố định (Bước 3c)
và hỗ trợ Dual-Encode / BM25 cho lưới cấu hình selection_rule_v3.json (cfg_11..cfg_19).
Không dùng dữ liệu từ DEV/val/B-human.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.serve.parser_v1 import remove_diacritics
from src.serve.parser_v2 import (
    ARTICLE_TYPE_LEXICON_V2,
    COLOUR_LEXICON_V2,
    GENDER_LEXICON_V2,
    PROTECTED_NON_COLOUR_PHRASES,
)


def _build_restoration_table() -> List[Tuple[re.Pattern, str, int]]:
    """
    Xây bảng ánh xạ không-dấu -> có-dấu từ lexicon v2 (catalog) và cụm cố định (homograph 2c.1c).
    Sắp xếp longest-match-first.
    """
    pairs: Dict[str, str] = {}
    for ph in PROTECTED_NON_COLOUR_PHRASES:
        ph_norm = remove_diacritics(ph)
        if ph_norm != ph.lower() and ph_norm not in pairs:
            pairs[ph_norm] = ph.lower()

    # Thêm các từ nối/giới từ chuẩn tiếng Việt hay gặp trong cụm tìm kiếm
    pairs.setdefault("mau", "màu")
    pairs.setdefault("cho nu", "cho nữ")
    pairs.setdefault("cua nu", "của nữ")
    pairs.setdefault("di lam", "đi làm")
    pairs.setdefault("di choi", "đi chơi")
    pairs.setdefault("di hoc", "đi học")
    pairs.setdefault("di tiec", "đi tiệc")
    pairs.setdefault("hang ngay", "hằng ngày")

    for lex in (ARTICLE_TYPE_LEXICON_V2, GENDER_LEXICON_V2, COLOUR_LEXICON_V2):
        for kw_list in lex.values():
            for kw in kw_list:
                kw_low = re.sub(r"\s+", " ", kw.lower()).strip()
                kw_norm = remove_diacritics(kw_low)
                if kw_norm != kw_low and kw_norm not in pairs:
                    pairs[kw_norm] = kw_low

    compiled: List[Tuple[re.Pattern, str, int]] = []
    for k_norm, k_acc in pairs.items():
        pat = re.compile(rf"(?<!\w){re.escape(k_norm)}(?!\w)")
        compiled.append((pat, k_acc, len(k_norm)))
    compiled.sort(key=lambda x: x[2], reverse=True)
    return compiled


_RESTORATION_PATTERNS = _build_restoration_table()


def restore_diacritics_for_query(query: str) -> str:
    """
    Khôi phục dấu tiếng Việt nhẹ cho câu truy vấn (Bước 3c).
    Nếu token/cụm đã có dấu thì giữ nguyên; chỉ thay thế các cụm thực sự không dấu.
    """
    text = re.sub(r"\s+", " ", (query or "").strip())
    if not text:
        return ""
    if remove_diacritics(text) != text.lower():
        # Câu đã có dấu (toàn phần hoặc một phần): không ghi đè toàn bộ
        return text

    out = text.lower()
    for pat, replacement, _ in _RESTORATION_PATTERNS:
        out = pat.sub(replacement, out)
    return out


def dual_encode_texts(engine, queries: List[str]) -> np.ndarray:
    """
    Mã hóa kép: trung bình L2-normalized giữa câu gốc và câu đã khôi phục dấu.
    """
    orig_vecs = np.vstack([engine.encode_text_query(q) for q in queries])
    rewritten = [restore_diacritics_for_query(q) for q in queries]
    rew_vecs = np.vstack([engine.encode_text_query(q) for q in rewritten])
    blended = 0.5 * orig_vecs + 0.5 * rew_vecs
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    return (blended / np.maximum(norms, 1e-12)).astype(np.float32)


def _encode_batch_fast(engine, texts: List[str]) -> np.ndarray:
    """Batch encode texts bằng preprocessor + model của engine (fallback sang từng câu nếu mock)."""
    if hasattr(engine, "preprocessor") and hasattr(engine, "model"):
        import torch

        inputs = engine.preprocessor.preprocess_texts(texts, device=engine.device)
        with torch.no_grad():
            embs = engine.model.encode_text(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            vecs = embs.float().cpu().numpy()
        return engine._apply_mean_centering(vecs)
    return np.vstack([engine.encode_text_query(t) for t in texts])


def build_catalog_template(parsed: Dict[str, Any]) -> str:
    """Sinh mẫu chuẩn hóa tiếng Việt thuần từ nhãn catalog đã parse (I3)."""
    parts: List[str] = []
    arts = parsed.get("articleType") or []
    if arts and arts[0] in ARTICLE_TYPE_LEXICON_V2:
        parts.append(ARTICLE_TYPE_LEXICON_V2[arts[0]][0])
    gen = parsed.get("gender")
    if gen and gen in GENDER_LEXICON_V2:
        # Ưu tiên từ ngắn chuẩn ('nam', 'nữ', 'bé trai', ...)
        g_kws = GENDER_LEXICON_V2[gen]
        parts.append(g_kws[-1] if gen in ("Men", "Women") else g_kws[0])
    cols = parsed.get("baseColour") or []
    if cols and cols[0] in COLOUR_LEXICON_V2:
        parts.append("màu " + COLOUR_LEXICON_V2[cols[0]][0])
    return " ".join(parts).strip()


def multi_template_encode_texts(
    engine,
    queries: List[str],
    parsed_list: List[Dict[str, Any]],
    n_templates: int = 2,
) -> np.ndarray:
    """
    I3: Mở rộng truy vấn phía text (đa mẫu) dùng batch forward:
    - n_templates=2: trung bình [câu khôi phục dấu, mẫu chuẩn hóa catalog (nếu có)].
    - n_templates>=3: trung bình [câu gốc, câu khôi phục dấu, mẫu chuẩn hóa catalog].
    """
    all_vecs: List[np.ndarray] = []
    for q, pr in zip(queries, parsed_list):
        rw = restore_diacritics_for_query(q)
        tpl = build_catalog_template(pr)
        cands = [rw]
        if n_templates >= 3 and q != rw:
            cands.append(q)
        if tpl and tpl != rw:
            cands.append(tpl)
        vecs = _encode_batch_fast(engine, cands)
        avg = np.mean(vecs, axis=0, keepdims=True)
        norm = np.linalg.norm(avg, axis=1, keepdims=True)
        all_vecs.append((avg / np.maximum(norm, 1e-12)).astype(np.float32))
    return np.vstack(all_vecs)


class BM25GalleryScorer:
    """BM25 gọn nhẹ trên metadata/caption của gallery để phục vụ cfg_17..cfg_19."""

    def __init__(self, id_map: List[Dict[str, Any]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(id_map)
        self.doc_tfs: List[Counter] = []
        self.doc_lens: List[int] = []
        df: Counter = Counter()

        for m in id_map:
            parts = [
                str(m.get("articleType", "")),
                str(m.get("articleType_vi", "")),
                str(m.get("gender", "")),
                str(m.get("baseColour", "")),
                str(m.get("baseColour_vi", "")),
                str(m.get("productDisplayName", "")),
                str(m.get("caption", "")),
            ]
            toks = re.findall(r"\w+", remove_diacritics(" ".join(parts)))
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(toks))
            for t in tf.keys():
                df[t] += 1

        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs > 0 else 1.0
        self.idf: Dict[str, float] = {
            t: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for t, freq in df.items()
        }

    def score_query(self, query: str) -> np.ndarray:
        q_toks = re.findall(r"\w+", remove_diacritics(query))
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for i in range(self.n_docs):
            tf = self.doc_tfs[i]
            dl = self.doc_lens[i]
            denom_base = self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-8)))
            s = 0.0
            for t in q_toks:
                f = tf.get(t, 0)
                if f > 0:
                    idf = self.idf.get(t, 0.0)
                    s += idf * ((f * (self.k1 + 1.0)) / (f + denom_base))
            scores[i] = s
        sd = float(np.std(scores))
        if sd > 1e-8:
            scores = (scores - float(np.mean(scores))) / sd
        return scores
