"""
Dựng gallery ngoài miền index/crawl_v2_gallery/ từ data/crawl_v2 (ảnh sản phẩm 11 thương hiệu Việt Nam).

- Chỉ lấy ảnh filter_flag == pass và label_status == ok (bỏ 372 ca needs_review).
- Dùng đúng đường tiền xử lý + checkpoint của app (Luật 4, Luật 6: kiểm tra SHA-256 checkpoint).
- Ghi: image_embeddings.npy, id_map.json, faiss_hnsw.bin, manifest.json (cùng định dạng index/test_gallery).
- Không sửa dữ liệu nguồn. Seed 42.

Chạy: python scripts/build_crawl_v2_gallery.py [--batch 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index.faiss_index import FaissHNSWIndex  # noqa: E402
from src.model.siglip_model import load_siglip_model  # noqa: E402
from src.preprocess.pipeline import SigLIPPreprocessor  # noqa: E402

META = ROOT / "reports" / "crawl_v2" / "metadata.jsonl"
OUT = ROOT / "index" / "crawl_v2_gallery"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    ckpt_hash = cfg["checkpoint"]["checkpoint_hash"]

    rows = [json.loads(line) for line in open(META, encoding="utf-8")]
    rows = [r for r in rows if r.get("filter_flag") == "pass" and r.get("label_status") == "ok"]
    rows.sort(key=lambda r: r["crawl_id"])
    print(f"{len(rows)} ảnh (pass + ok)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre = SigLIPPreprocessor(device=device)
    model = load_siglip_model(str(ROOT / cfg["paths"]["checkpoint_file"]), expected_hash=ckpt_hash,
                              device=device, fp16=(device == "cuda"))

    embs, id_map, bad = [], [], []
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        ok_rows, imgs = [], []
        for r in chunk:
            try:
                imgs.append(pre.load_image(ROOT / r["local_path"]))
                ok_rows.append(r)
            except Exception as e:  # ảnh hỏng: bỏ qua, ghi lại
                bad.append((r["crawl_id"], str(e)))
        if not imgs:
            continue
        with torch.no_grad():
            pv = pre.preprocess_images(imgs, device=device, half_precision=(device == "cuda"))
            e = model.encode_image(pv).float().cpu().numpy()
        embs.append(e)
        for r in ok_rows:
            id_map.append({
                "idx": len(id_map), "item_id": r["crawl_id"], "product_id": r["crawl_id"],
                "path": r["local_path"], "articleType": r["articleType"], "gender": r["gender"],
                "baseColour": r["baseColour"], "brand": r["brand"], "productDisplayName": r["product_name"],
                "split": "external_crawl_v2", "in_scope": True,
            })
        if (i // args.batch) % 50 == 0:
            done = i + len(chunk)
            print(f"{done}/{len(rows)}  {done / max(time.time() - t0, 1e-6):.1f} ảnh/s", flush=True)

    E = np.vstack(embs).astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "image_embeddings.npy", E)
    json.dump(id_map, open(OUT / "id_map.json", "w", encoding="utf-8"), ensure_ascii=False)
    idx = FaissHNSWIndex(dim=E.shape[1], M=32, ef_construction=200, ef_search=128)
    idx.add(E.copy())
    idx.save(OUT / "faiss_hnsw.bin")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "gallery_type": "external_out_of_domain_crawl_v2",
        "checkpoint_hash": ckpt_hash,
        "source_metadata": str(META.relative_to(ROOT)).replace("\\", "/"),
        "source_metadata_sha256": hashlib.sha256(META.read_bytes()).hexdigest(),
        "filter": "filter_flag == pass and label_status == ok",
        "n_vectors": int(E.shape[0]), "embedding_dim": int(E.shape[1]),
        "hnsw_params": {"type": "IndexHNSWFlat", "M": 32, "ef_construction": 200, "ef_search": 128,
                        "metric": "inner_product"},
        "commit_hash": commit, "seed": 42, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skipped_bad_images": bad,
        "files": {"embeddings": "image_embeddings.npy", "id_map": "id_map.json", "index_bin": "faiss_hnsw.bin"},
    }
    json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Xong: {E.shape[0]} vector, {len(bad)} ảnh lỗi, {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
