"""
Xây dựng index/val_gallery/ cho tập validation (1.884 sản phẩm sạch in-scope từ eval/val_split.json).
Tuân thủ Luật 3 (thư mục mới), Luật 4 (đường code tiền xử lý duy nhất), Luật 6 (Hash Guard).
Lưu ý trung thực: Các ảnh val đã được checkpoint thấy khi train => chỉ dùng so sánh TƯƠNG ĐỐI.
"""
import os
import sys
import time
import json
import pathlib
import subprocess
import yaml
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from src.model.siglip_model import load_siglip_model, verify_checkpoint_hash, compute_file_sha256
from src.preprocess.pipeline import SigLIPPreprocessor
from src.index.faiss_index import FaissHNSWIndex


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def build_val_gallery():
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / config["paths"]["checkpoint_file"]
    expected_hash = config["checkpoint"]["checkpoint_hash"]
    verify_checkpoint_hash(ckpt_path, expected_hash)

    model = load_siglip_model(ckpt_path, expected_hash=expected_hash, device=device)
    model.eval()
    preprocessor = SigLIPPreprocessor(device=device)

    with open(PROJECT_ROOT / "eval/val_split.json", "r", encoding="utf-8") as f:
        val_ids = set(str(x) for x in json.load(f))

    catalog_path = PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet"
    df = pd.read_parquet(catalog_path)
    df["pid_str"] = df["product_id"].astype(str)
    val_df = df[(df["pid_str"].isin(val_ids)) & (df["filter_flag"] == "pass") & (df["in_scope"] == True)].copy()
    val_df = val_df.sort_values("pid_str").reset_index(drop=True)
    print(f"• Số lượng sản phẩm val sạch in_scope: {len(val_df)}")

    out_dir = PROJECT_ROOT / "index" / "val_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Encode images theo batches dùng đúng SigLIPPreprocessor
    batch_size = 64
    embs_list = []
    image_paths = val_df["image_path"].tolist()

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = [PROJECT_ROOT / p for p in image_paths[i : i + batch_size]]
            pixel_values = preprocessor.preprocess_images(
                batch_paths, device=device, half_precision=(device == "cuda")
            )
            emb = model.encode_image(pixel_values)
            embs_list.append(emb.float().cpu().numpy())

    val_embs = np.vstack(embs_list).astype(np.float32)
    # Bảo đảm L2-normalized
    norms = np.linalg.norm(val_embs, axis=1, keepdims=True)
    val_embs = (val_embs / np.maximum(norms, 1e-12)).astype(np.float32)
    print(f"✅ Đã encode {val_embs.shape} trong {time.time() - t0:.2f}s")

    np.save(out_dir / "image_embeddings.npy", val_embs)

    id_map = []
    for idx, row in val_df.iterrows():
        id_map.append({
            "item_id": str(row["product_id"]),
            "path": str(row["image_path"]),
            "articleType": str(row["articleType"]),
            "articleType_vi": str(row.get("articleType_vi", "")),
            "gender": str(row["gender"]),
            "gender_vi": str(row.get("gender_vi", "")),
            "baseColour": str(row["baseColour"]),
            "baseColour_vi": str(row.get("baseColour_vi", "")),
            "title_raw": str(row.get("title_raw", "")),
            "split": str(row.get("split", "val")),
            "idx": int(idx),
        })

    with open(out_dir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2, ensure_ascii=False)

    faiss_cfg = config["faiss"]
    M = faiss_cfg.get("M", 32)
    ef_const = faiss_cfg.get("ef_construction", 200)
    ef_search = faiss_cfg.get("ef_search", 128)

    hnsw = FaissHNSWIndex(dim=1152, M=M, ef_construction=ef_const, ef_search=ef_search)
    hnsw.add(val_embs)
    hnsw.save(out_dir / "faiss_hnsw.bin")

    manifest = {
        "gallery_type": "val_tuning_split",
        "note": "Ảnh val đã được checkpoint thấy khi train; CHỈ dùng để so sánh TƯƠNG ĐỐI giữa các cấu hình suy luận.",
        "checkpoint_hash": expected_hash,
        "catalog_hash": compute_file_sha256(catalog_path),
        "n_vectors": int(hnsw.ntotal),
        "embedding_dim": 1152,
        "hnsw_params": {
            "type": "IndexHNSWFlat",
            "M": M,
            "ef_construction": ef_const,
            "ef_search": ef_search,
            "metric": "inner_product",
        },
        "commit_hash": get_git_commit_hash(),
        "seed": config.get("seed", 42),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {
            "embeddings": "image_embeddings.npy",
            "id_map": "id_map.json",
            "index_bin": "faiss_hnsw.bin",
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã dựng hoàn chỉnh index/val_gallery/ với {hnsw.ntotal} vectors!")


if __name__ == "__main__":
    build_val_gallery()
