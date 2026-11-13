"""
Dựng index/template_cache_v5/ (Bước 12.2):
- Nguồn sinh mẫu: CHỈ từ bảng nhãn hợp lệ trong data/catalog/catalog_rebuild.parquet
  (1-slot, 2-slot, 3-slot), tuyệt đối KHÔNG đọc từ bất kỳ tập truy vấn nào.
- Encode bằng best_model.pt qua cùng SiglipProcessor (max_length=128, Luật 4).
- Ghi template_embeddings.npy, template_key_map.json, manifest.json (Hash Guard).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serve.query_rewriter import _encode_batch_fast, build_catalog_template
from src.serve.search_engine import ECommerceSearchEngine
from src.serve.template_cache import compute_cache_hash, make_template_cache_key


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
    cat_path = PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet"
    val_manifest = json.loads((PROJECT_ROOT / "index/val_gallery/manifest.json").read_text(encoding="utf-8"))
    out_dir = PROJECT_ROOT / "index/template_cache_v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cat_path)
    if "in_scope" in df.columns:
        df = df[df["in_scope"] == True].copy()

    specs: Dict[str, Dict[str, Any]] = {}

    def _add_spec(at: str | None, gd: str | None, cl: str | None) -> None:
        parsed = {
            "articleType": [at] if at else [],
            "gender": gd if gd else None,
            "baseColour": [cl] if cl else [],
        }
        tpl = build_catalog_template(parsed)
        if not tpl:
            return
        key = make_template_cache_key(parsed)
        if key not in specs:
            specs[key] = {"parsed": parsed, "template": tpl}

    # Enumerate 1-slot, 2-slot, 3-slot combinations present in catalog_rebuild.parquet
    for at in sorted(df["articleType"].dropna().unique()):
        _add_spec(str(at), None, None)
    for gd in sorted(df["gender"].dropna().unique()):
        _add_spec(None, str(gd), None)
    for cl in sorted(df["baseColour"].dropna().unique()):
        _add_spec(None, None, str(cl))
    for (at, gd), _ in df.groupby(["articleType", "gender"]):
        _add_spec(str(at), str(gd), None)
    for (at, cl), _ in df.groupby(["articleType", "baseColour"]):
        _add_spec(str(at), None, str(cl))
    for (gd, cl), _ in df.groupby(["gender", "baseColour"]):
        _add_spec(None, str(gd), str(cl))
    for (at, gd, cl), _ in df.groupby(["articleType", "gender", "baseColour"]):
        _add_spec(str(at), str(gd), str(cl))

    keys = sorted(specs.keys())
    texts = [specs[k]["template"] for k in keys]
    key_map = {k: i for i, k in enumerate(keys)}

    engine = ECommerceSearchEngine(index_dir="index/val_gallery", use_mean_centering=False)
    t0 = time.perf_counter()
    chunks = [_encode_batch_fast(engine, texts[i : i + 64]) for i in range(0, len(texts), 64)]
    embs = np.vstack(chunks).astype(np.float32)
    elapsed = time.perf_counter() - t0

    emb_path = out_dir / "template_embeddings.npy"
    map_path = out_dir / "template_key_map.json"
    np.save(emb_path, embs)
    map_path.write_text(json.dumps(key_map, ensure_ascii=False, indent=2), encoding="utf-8")

    cache_hash = compute_cache_hash(emb_path, map_path)
    manifest = {
        "cache_type": "template_cache_v5",
        "source": "data/catalog/catalog_rebuild.parquet (in_scope=True)",
        "checkpoint_hash": val_manifest["checkpoint_hash"],
        "catalog_hash": _sha256_file(cat_path),
        "cache_hash": cache_hash,
        "n_templates": len(keys),
        "embedding_dim": int(embs.shape[1]),
        "seed": 42,
        "commit_hash": _get_git_commit(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {out_dir} in {elapsed:.2f}s: n_templates={len(keys)}, shape={embs.shape}, cache_hash={cache_hash[:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
