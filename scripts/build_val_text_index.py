"""
Dựng index văn bản phụ trợ cho val_gallery ở thư mục riêng index/val_gallery_text_v4/ (Luật 3, 4, 5, 6).
Lưu ý trung thực: val_gallery không có caption InternVL2-2B trong data/catalog/captions/;
chỉ có title_vi (cho cfg_v4_11/12/18) và metadata_aux (title_vi + 3 slot, cho cfg_v4_13).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from src.serve.query_rewriter import _encode_batch_fast
from src.serve.search_engine import ECommerceSearchEngine


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    val_dir = Path("index/val_gallery")
    out_dir = Path("index/val_gallery_text_v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_manifest = json.loads((val_dir / "manifest.json").read_text(encoding="utf-8"))
    engine = ECommerceSearchEngine(index_dir=str(val_dir), use_mean_centering=False)

    title_texts = []
    meta_aux_texts = []
    for meta in engine.id_map:
        title = str(meta.get("title_vi") or meta.get("productDisplayName") or "").strip()
        if not title:
            title = str(meta.get("articleType") or "sản phẩm")
        title_texts.append(title)

        parts = [
            title,
            str(meta.get("articleType_vi") or meta.get("articleType") or "").strip(),
            str(meta.get("gender_vi") or meta.get("gender") or "").strip(),
            str(meta.get("baseColour_vi") or meta.get("baseColour") or "").strip(),
        ]
        aux = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
        meta_aux_texts.append(aux)

    t0 = time.perf_counter()
    def _encode_chunks(texts: list[str], bs: int = 64) -> np.ndarray:
        chunks = [_encode_batch_fast(engine, texts[i : i + bs]) for i in range(0, len(texts), bs)]
        return np.vstack(chunks).astype(np.float32)

    title_vecs = _encode_chunks(title_texts, 64)
    aux_vecs = _encode_chunks(meta_aux_texts, 64)
    elapsed = time.perf_counter() - t0

    np.save(out_dir / "title_vi_embeddings.npy", title_vecs)
    np.save(out_dir / "metadata_aux_embeddings.npy", aux_vecs)

    manifest = {
        "gallery_type": "val_tuning_split_text_v4",
        "checkpoint_hash": val_manifest["checkpoint_hash"],
        "catalog_hash": val_manifest["catalog_hash"],
        "n_vectors": int(len(engine.id_map)),
        "embedding_dim": int(title_vecs.shape[1]),
        "seed": 42,
        "commit_hash": _get_git_commit(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "internvl2_captions_present": False,
        "note": (
            "val_gallery has no InternVL2-2B captions in data/catalog/captions/; "
            "title_vi_embeddings encodes title_vi/productDisplayName only, "
            "metadata_aux_embeddings encodes title_vi + articleType + gender + baseColour (auxiliary per Law 5)."
        ),
        "sources": {
            "title_vi_embeddings": "title_vi (fallback productDisplayName)",
            "metadata_aux_embeddings": "title_vi + articleType + gender + baseColour",
        },
        "files": {
            "title_vi_embeddings": "title_vi_embeddings.npy",
            "metadata_aux_embeddings": "metadata_aux_embeddings.npy",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Built {out_dir} in {elapsed:.2f}s: n_vectors={manifest['n_vectors']}, "
        f"title_vi={title_vecs.shape}, metadata_aux={aux_vecs.shape}"
    )


if __name__ == "__main__":
    main()
