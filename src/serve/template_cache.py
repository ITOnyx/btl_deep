"""
I3_cached: Nạp và tra cứu cache embedding mẫu chuẩn hóa (index/template_cache_v5/)
kèm Hash Guard (Luật 6) và cơ chế FALLBACK về 1 forward khi khóa không có trong map (Bước 12.1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.serve.query_rewriter import _encode_batch_fast, restore_diacritics_for_query


def make_template_cache_key(parsed: Dict[str, Any]) -> str:
    arts = "|".join(sorted(str(x) for x in (parsed.get("articleType") or [])))
    gd = str(parsed.get("gender") or "")
    cols = "|".join(sorted(str(x) for x in (parsed.get("baseColour") or [])))
    return f"{arts}__{gd}__{cols}"


def compute_cache_hash(emb_path: Path, map_path: Path) -> str:
    h = hashlib.sha256()
    h.update(emb_path.read_bytes())
    h.update(map_path.read_bytes())
    return h.hexdigest()


class TemplateCacheV5:
    def __init__(
        self,
        embeddings: np.ndarray,
        key_map: Dict[str, int],
        manifest: Dict[str, Any],
    ) -> None:
        self.embeddings = embeddings
        self.key_map = key_map
        self.manifest = manifest

    @classmethod
    def load(
        cls,
        cache_dir: str | Path = "index/template_cache_v5",
        expected_checkpoint_hash: Optional[str] = None,
        expected_catalog_hash: Optional[str] = None,
    ) -> "TemplateCacheV5":
        cdir = Path(cache_dir)
        manifest_path = cdir / "manifest.json"
        emb_path = cdir / "template_embeddings.npy"
        map_path = cdir / "template_key_map.json"
        if not (manifest_path.exists() and emb_path.exists() and map_path.exists()):
            raise FileNotFoundError(f"Thiếu file trong {cdir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_checkpoint_hash and manifest.get("checkpoint_hash") != expected_checkpoint_hash:
            raise ValueError(
                f"Hash Guard (template_cache_v5): checkpoint_hash lệch "
                f"({manifest.get('checkpoint_hash')} != {expected_checkpoint_hash})"
            )
        if expected_catalog_hash and manifest.get("catalog_hash") != expected_catalog_hash:
            raise ValueError(
                f"Hash Guard (template_cache_v5): catalog_hash lệch "
                f"({manifest.get('catalog_hash')} != {expected_catalog_hash})"
            )
        actual_cache_hash = compute_cache_hash(emb_path, map_path)
        if manifest.get("cache_hash") != actual_cache_hash:
            raise ValueError(
                f"Hash Guard (template_cache_v5): cache_hash lệch "
                f"({manifest.get('cache_hash')} != {actual_cache_hash})"
            )

        embs = np.load(emb_path).astype(np.float32)
        key_map = json.loads(map_path.read_text(encoding="utf-8"))
        if embs.shape[0] != len(key_map):
            raise ValueError(f"Số vector ({embs.shape[0]}) != len(key_map) ({len(key_map)})")
        return cls(embeddings=embs, key_map=key_map, manifest=manifest)

    def lookup(self, parsed: Dict[str, Any]) -> Optional[np.ndarray]:
        key = make_template_cache_key(parsed)
        idx = self.key_map.get(key)
        if idx is None:
            return None
        return self.embeddings[idx]


def encode_texts_with_template_cache(
    engine: Any,
    queries: List[str],
    parsed_list: List[Dict[str, Any]],
    cache: TemplateCacheV5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    1 forward GPU cho câu khôi phục dấu, sau đó tra cache mẫu chuẩn hóa từ RAM:
    - Nếu khớp khóa: v_final = normalize(0.5 * v_q + 0.5 * v_tpl).
    - Nếu KHÔNG có khóa trong template_key_map: FALLBACK về 1 forward (v_q), không báo lỗi.
    """
    rw_queries = [restore_diacritics_for_query(q) for q in queries]
    q_vecs = _encode_batch_fast(engine, rw_queries)

    out_vecs: List[np.ndarray] = []
    n_fallback = 0
    for qv, pr in zip(q_vecs, parsed_list):
        tpl_vec = cache.lookup(pr)
        if tpl_vec is None:
            n_fallback += 1
            out_vecs.append(qv.astype(np.float32))
        else:
            avg = 0.5 * qv.astype(np.float64) + 0.5 * tpl_vec.astype(np.float64)
            norm = float(np.linalg.norm(avg))
            out_vecs.append((avg / max(norm, 1e-12)).astype(np.float32))

    n_total = len(queries)
    stats = {
        "n_total": n_total,
        "n_cache_hit": n_total - n_fallback,
        "n_fallback": n_fallback,
        "fallback_rate": round(n_fallback / n_total, 4) if n_total else 0.0,
    }
    return np.vstack(out_vecs), stats
