"""
Dựng index/val_gallery_text_v5/ (Bước 11.1 — Luật 3, 4, 5, 6):
- Mã hóa bằng text tower của best_model.pt (qua _encode_batch_fast / SiglipProcessor max_length=128):
  1) internvl2_only_embeddings.npy (Config A: caption_A)
  2) internvl2_plus_aux_embeddings.npy (Config B: caption_B = merge_v1)
  3) metadata_aux_embeddings.npy (metadata thuần: title_vi + 3 slot, tách riêng theo Luật 5)
- Lưu manifest.json có checkpoint_hash, catalog_hash, n_vectors, commit_hash, seed=42, SHA-256 các file.
- Ghi chú Luật D.5: caption mới không so trực tiếp với số liệu cũ.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serve.query_rewriter import _encode_batch_fast
from src.serve.search_engine import ECommerceSearchEngine


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    val_dir = PROJECT_ROOT / "index/val_gallery"
    out_dir = PROJECT_ROOT / "index/val_gallery_text_v5"
    captions_dir = PROJECT_ROOT / "data/catalog/captions"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(captions_dir.glob("shard_val_[0-9]*.parquet"))
    if not shard_files:
        raise FileNotFoundError(
            "Không tìm thấy shard_val_[0-9]*.parquet trong data/catalog/captions/. "
            "Hãy chạy scripts/caption_val_gallery.py --mode full trước."
        )

    df_caps = pd.concat([pd.read_parquet(p) for p in shard_files], ignore_index=True)
    df_caps["pid_str"] = df_caps["product_id"].astype(str)
    cap_map: Dict[str, Dict] = df_caps.set_index("pid_str").to_dict(orient="index")

    val_manifest = json.loads((val_dir / "manifest.json").read_text(encoding="utf-8"))
    engine = ECommerceSearchEngine(index_dir=str(val_dir), use_mean_centering=False)

    missing_pids = [
        str(m.get("product_id") or m.get("item_id"))
        for m in engine.id_map
        if str(m.get("product_id") or m.get("item_id")) not in cap_map
    ]
    if missing_pids:
        raise RuntimeError(f"Thiếu {len(missing_pids)}/{len(engine.id_map)} caption val_gallery (ví dụ: {missing_pids[:5]}).")

    texts_a: List[str] = []
    texts_b: List[str] = []
    texts_meta: List[str] = []

    for m in engine.id_map:
        pid = str(m.get("product_id") or m.get("item_id"))
        c_row = cap_map[pid]
        texts_a.append(str(c_row.get("caption_A") or "").strip())
        texts_b.append(str(c_row.get("caption_B") or "").strip())

        title = str(m.get("title_vi") or m.get("productDisplayName") or "").strip()
        if not title:
            title = str(m.get("articleType") or "sản phẩm")
        parts = [
            title,
            str(m.get("articleType_vi") or m.get("articleType") or "").strip(),
            str(m.get("gender_vi") or m.get("gender") or "").strip(),
            str(m.get("baseColour_vi") or m.get("baseColour") or "").strip(),
        ]
        texts_meta.append(re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip())

    def _encode_chunks(texts: List[str], bs: int = 64) -> np.ndarray:
        chunks = [_encode_batch_fast(engine, texts[i : i + bs]) for i in range(0, len(texts), bs)]
        return np.vstack(chunks).astype(np.float32)

    t0 = time.perf_counter()
    vecs_a = _encode_chunks(texts_a, 64)
    vecs_b = _encode_chunks(texts_b, 64)
    vecs_meta = _encode_chunks(texts_meta, 64)
    elapsed = time.perf_counter() - t0

    f_a = out_dir / "internvl2_only_embeddings.npy"
    f_b = out_dir / "internvl2_plus_aux_embeddings.npy"
    f_m = out_dir / "metadata_aux_embeddings.npy"
    np.save(f_a, vecs_a)
    np.save(f_b, vecs_b)
    np.save(f_m, vecs_meta)

    manifest = {
        "gallery_type": "val_tuning_split_text_v5",
        "checkpoint_hash": val_manifest["checkpoint_hash"],
        "catalog_hash": val_manifest["catalog_hash"],
        "n_vectors": int(len(engine.id_map)),
        "embedding_dim": int(vecs_a.shape[1]),
        "seed": 42,
        "commit_hash": _get_git_commit(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "internvl2_captions_present": True,
        "law_d5_note": "Caption mới từ InternVL2-2B (và merge_v1) không so trực tiếp với số liệu cũ của checkpoint (Luật D.5).",
        "files": {
            "internvl2_only_embeddings": {"file": f_a.name, "sha256": _sha256_file(f_a)},
            "internvl2_plus_aux_embeddings": {"file": f_b.name, "sha256": _sha256_file(f_b)},
            "metadata_aux_embeddings": {"file": f_m.name, "sha256": _sha256_file(f_m)},
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {out_dir} in {elapsed:.2f}s: n_vectors={manifest['n_vectors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
