"""Build text gallery embeddings v5 for test split (Option 1B, Prompt 14c).

Reuses `_encode_batch_fast`, `_sha256_file`, `_get_git_commit`, and the exact text
construction + `SiglipProcessor(max_length=128, padding="max_length")` pipeline from
`scripts/build_val_text_index_v5.py` without modifying `scripts/build_val_text_index_v5.py`.

Also supports:
1. Verification run on `val` into `index/_verify_val_text_v5/` (never overwriting
   `index/val_gallery_text_v5/`) and row-by-row cosine comparison against
   `index/val_gallery_text_v5/` (requiring min cosine >= 0.9999 across all 3 `.npy` files).
2. Token length distribution & truncation rate (`> 128` tokens) comparison between
   `val` (`max_new_tokens=48`) and `test` (`max_new_tokens=96`) using the exact
   SigLIP tokenizer.
3. Law 6 Hash Guard & sanity check on `index/test_gallery_text_v5/`.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Direct import of shared helpers and encoding pipeline from scripts/build_val_text_index_v5.py
from scripts.build_val_text_index_v5 import (  # noqa: E402
    _encode_batch_fast,
    _get_git_commit,
    _sha256_file,
)
from src.serve.search_engine import ECommerceSearchEngine  # noqa: E402

LOCKED_BUILD_VAL_SCRIPT_SHA256 = (
    "da87e18c0027c769a072d99d59fe47e7c5e2fc9a71f1a4116c9280d232d2e867"
)


def _sha256_json_list(items: List[str]) -> str:
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_texts_for_gallery(
    id_map: List[Dict[str, Any]],
    shard_paths: List[Path],
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Any]]:
    """Construct `(texts_a, texts_b, texts_meta, id_order, provenance)` using the exact
    same logic as `scripts/build_val_text_index_v5.py` lines 53-89."""
    if not shard_paths:
        raise FileNotFoundError("No caption shards found for gallery text index build.")

    dfs = [pd.read_parquet(p) for p in sorted(shard_paths)]
    cap_df = pd.concat(dfs, ignore_index=True)
    cap_df["product_id"] = cap_df["product_id"].astype(str)
    cap_map = {str(r["product_id"]): r for _, r in cap_df.iterrows()}

    texts_a: List[str] = []
    texts_b: List[str] = []
    texts_meta: List[str] = []
    id_order: List[str] = []
    missing_ids: List[str] = []

    for m in id_map:
        pid = str(m.get("product_id") or m.get("item_id"))
        id_order.append(pid)
        title = str(m.get("title_vi") or m.get("productDisplayName") or "").strip()
        if not title:
            title = str(m.get("articleType") or "sản phẩm")
        parts = [
            title,
            str(m.get("articleType_vi") or m.get("articleType") or "").strip(),
            str(m.get("gender_vi") or m.get("gender") or "").strip(),
            str(m.get("baseColour_vi") or m.get("baseColour") or "").strip(),
        ]
        meta_txt = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
        texts_meta.append(meta_txt)

        c_row = cap_map.get(pid)
        if c_row is not None:
            ca = str(c_row.get("caption_A") or "").strip()
            cb = str(c_row.get("caption_B") or "").strip()
            texts_a.append(ca if ca else meta_txt)
            texts_b.append(cb if cb else meta_txt)
            if not ca or not cb:
                missing_ids.append(pid)
        else:
            missing_ids.append(pid)
            texts_a.append(meta_txt)
            texts_b.append(meta_txt)

    if missing_ids:
        raise RuntimeError(
            f"Missing captions for {len(missing_ids)} / {len(id_map)} items: {missing_ids[:5]}"
        )

    # Extract provenance metadata from the matched rows
    matched_rows = [cap_map[pid] for pid in id_order]
    caption_models = sorted({str(r.get("caption_model") or "") for r in matched_rows})
    model_revisions = sorted({str(r.get("model_revision") or "") for r in matched_rows})
    comb_methods = sorted({str(r.get("combination_method") or "") for r in matched_rows})
    seeds = sorted({int(r.get("seed", 42)) for r in matched_rows})
    gen_params_raw = sorted({str(r.get("gen_params") or "{}") for r in matched_rows})
    parsed_gen_params = [json.loads(gp) for gp in gen_params_raw]

    provenance = {
        "caption_model": caption_models[0] if len(caption_models) == 1 else caption_models,
        "model_revision": model_revisions[0] if len(model_revisions) == 1 else model_revisions,
        "combination_method": comb_methods[0] if len(comb_methods) == 1 else comb_methods,
        "seed": seeds[0] if len(seeds) == 1 else seeds,
        "generation_params": (
            parsed_gen_params[0] if len(parsed_gen_params) == 1 else parsed_gen_params
        ),
        "shard_hashes": {p.name: _sha256_file(p) for p in sorted(shard_paths)},
    }
    return texts_a, texts_b, texts_meta, id_order, provenance


def encode_text_gallery(
    engine: ECommerceSearchEngine,
    texts_a: List[str],
    texts_b: List[str],
    texts_meta: List[str],
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode `(texts_a, texts_b, texts_meta)` using `_encode_batch_fast` from
    `scripts/build_val_text_index_v5.py`."""

    def _encode_chunks(texts: List[str], bs: int = batch_size) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), bs):
            chunks.append(_encode_batch_fast(engine, texts[i : i + bs]))
        return np.vstack(chunks).astype(np.float32)

    emb_a = _encode_chunks(texts_a, batch_size)
    emb_b = _encode_chunks(texts_b, batch_size)
    emb_meta = _encode_chunks(texts_meta, batch_size)
    return emb_a, emb_b, emb_meta


def compute_token_length_stats(
    tokenizer: Any,
    texts_a: List[str],
    texts_b: List[str],
    texts_meta: List[str],
    max_length: int = 128,
) -> Dict[str, Any]:
    """Compute exact un-truncated token length distribution using the SigLIP tokenizer."""

    def _stats_for_list(texts: List[str]) -> Dict[str, Any]:
        lengths: List[int] = []
        for t in texts:
            ids = tokenizer.encode(t, truncation=False)
            lengths.append(len(ids))
        arr = np.array(lengths, dtype=np.int32)
        n_over = int(np.sum(arr > max_length))
        n_at_or_over = int(np.sum(arr >= max_length))
        return {
            "n_items": int(len(arr)),
            "min": int(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(np.max(arr)),
            "count_gt_128_truncated": n_over,
            "rate_gt_128_truncated": float(n_over / max(1, len(arr))),
            "count_ge_128": n_at_or_over,
            "rate_ge_128": float(n_at_or_over / max(1, len(arr))),
        }

    return {
        "max_length": max_length,
        "caption_A_internvl2_only": _stats_for_list(texts_a),
        "caption_B_internvl2_plus_aux": _stats_for_list(texts_b),
        "metadata_aux": _stats_for_list(texts_meta),
    }


def verify_text_gallery_index(
    out_dir: Path,
    gallery_dir: Path,
    config_path: Path,
    expected_n_vectors: int,
) -> Dict[str, Any]:
    """Run Law 6 Hash Guard and sanity checks on a built text gallery index."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    expected_ckpt_hash = cfg["checkpoint"]["checkpoint_hash"]

    with open(gallery_dir / "manifest.json", "r", encoding="utf-8") as f:
        gallery_manifest = json.load(f)
    with open(gallery_dir / "id_map.json", "r", encoding="utf-8") as f:
        gallery_id_map = json.load(f)
    expected_ids = [str(m.get("product_id") or m.get("item_id")) for m in gallery_id_map]

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Law 6 Hash Guard
    ckpt_hash_match = (
        manifest.get("checkpoint_hash") == expected_ckpt_hash
        and manifest.get("checkpoint_hash") == gallery_manifest.get("checkpoint_hash")
    )
    catalog_hash_match = manifest.get("catalog_hash") == gallery_manifest.get("catalog_hash")
    n_vectors_match = int(manifest.get("n_vectors", -1)) == expected_n_vectors
    id_order = manifest.get("id_order", [])
    id_order_match = id_order == expected_ids
    n_duplicate_ids = len(id_order) - len(set(id_order))

    file_checks: Dict[str, Any] = {}
    all_files_valid = True
    for key in [
        "internvl2_only_embeddings",
        "internvl2_plus_aux_embeddings",
        "metadata_aux_embeddings",
    ]:
        finfo = manifest["files"][key]
        fpath = out_dir / finfo["file"]
        actual_sha = _sha256_file(fpath)
        sha_ok = actual_sha == finfo["sha256"]
        arr = np.load(fpath)
        shape_ok = arr.shape == (expected_n_vectors, 1152) and arr.dtype == np.float32
        nan_count = int(np.isnan(arr).sum() + np.isinf(arr).sum())
        norms = np.linalg.norm(arr, axis=1)
        min_norm = float(np.min(norms))
        max_norm = float(np.max(norms))
        mean_norm = float(np.mean(norms))
        norm_ok = bool(np.allclose(norms, 1.0, atol=1e-3))
        file_ok = sha_ok and shape_ok and (nan_count == 0) and norm_ok
        if not file_ok:
            all_files_valid = False
        file_checks[key] = {
            "file": finfo["file"],
            "sha256": actual_sha,
            "sha256_matches_manifest": sha_ok,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nan_or_inf_count": nan_count,
            "l2_norm_min": min_norm,
            "l2_norm_mean": mean_norm,
            "l2_norm_max": max_norm,
            "l2_norm_ok": norm_ok,
            "passed": file_ok,
        }

    passed = (
        ckpt_hash_match
        and catalog_hash_match
        and n_vectors_match
        and id_order_match
        and (n_duplicate_ids == 0)
        and all_files_valid
    )
    return {
        "passed": passed,
        "checkpoint_hash_match": ckpt_hash_match,
        "catalog_hash_match": catalog_hash_match,
        "n_vectors": int(manifest.get("n_vectors", -1)),
        "expected_n_vectors": expected_n_vectors,
        "n_vectors_match": n_vectors_match,
        "n_duplicate_ids": n_duplicate_ids,
        "id_order_matches_gallery_id_map": id_order_match,
        "id_order_sha256": manifest.get("id_order_sha256"),
        "files": file_checks,
    }


def write_text_gallery_index(
    out_dir: Path,
    gallery_type: str,
    gallery_manifest: Dict[str, Any],
    id_order: List[str],
    provenance: Dict[str, Any],
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    emb_meta: np.ndarray,
) -> Dict[str, Any]:
    """Write `.npy` files and `manifest.json` into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path_a = out_dir / "internvl2_only_embeddings.npy"
    path_b = out_dir / "internvl2_plus_aux_embeddings.npy"
    path_meta = out_dir / "metadata_aux_embeddings.npy"

    np.save(path_a, emb_a)
    np.save(path_b, emb_b)
    np.save(path_meta, emb_meta)

    manifest = {
        "gallery_type": gallery_type,
        "checkpoint_hash": gallery_manifest.get("checkpoint_hash"),
        "catalog_hash": gallery_manifest.get("catalog_hash"),
        "shard_hashes": provenance["shard_hashes"],
        "n_vectors": len(id_order),
        "embedding_dim": 1152,
        "id_order_sha256": _sha256_json_list(id_order),
        "id_order": id_order,
        "caption_model": provenance["caption_model"],
        "model_revision": provenance["model_revision"],
        "combination_method": provenance["combination_method"],
        "generation_params": provenance["generation_params"],
        "preprocessing": {
            "processor": "SiglipProcessor",
            "max_length": 128,
            "padding": "max_length",
            "truncation": True,
            "shared_encode_fn": "scripts.build_val_text_index_v5._encode_batch_fast",
        },
        "seed": 42,
        "commit_hash": _get_git_commit(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "internvl2_captions_present": True,
        "law_d5_note": (
            "Caption mới từ InternVL2-2B (và merge_v1) không so trực tiếp với số liệu cũ "
            "của checkpoint (Luật D.5)."
        ),
        "files": {
            "internvl2_only_embeddings": {
                "file": path_a.name,
                "sha256": _sha256_file(path_a),
            },
            "internvl2_plus_aux_embeddings": {
                "file": path_b.name,
                "sha256": _sha256_file(path_b),
            },
            "metadata_aux_embeddings": {
                "file": path_meta.name,
                "sha256": _sha256_file(path_meta),
            },
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def compare_text_gallery_dirs(dir_ref: Path, dir_cand: Path) -> Dict[str, Any]:
    """Compare row-by-row cosine similarity between `dir_ref` and `dir_cand` across all 3 `.npy` files."""
    results: Dict[str, Any] = {}
    all_passed = True
    for fname in [
        "internvl2_only_embeddings.npy",
        "internvl2_plus_aux_embeddings.npy",
        "metadata_aux_embeddings.npy",
    ]:
        ref = np.load(dir_ref / fname)
        cand = np.load(dir_cand / fname)
        if ref.shape != cand.shape:
            raise ValueError(f"Shape mismatch on {fname}: {ref.shape} vs {cand.shape}")
        # Row-by-row cosine (both are L2-normalized, but normalize defensively)
        ref_n = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
        cand_n = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        cos = np.sum(ref_n * cand_n, axis=1)
        min_cos = float(np.min(cos))
        mean_cos = float(np.mean(cos))
        max_abs_diff = float(np.max(np.abs(ref - cand)))
        n_below_09999 = int(np.sum(cos < 0.9999))
        passed = min_cos >= 0.9999 and n_below_09999 == 0
        if not passed:
            all_passed = False
        results[fname] = {
            "n_rows": int(ref.shape[0]),
            "min_cosine": min_cos,
            "mean_cosine": mean_cos,
            "max_abs_diff": max_abs_diff,
            "count_cosine_lt_0_9999": n_below_09999,
            "sha256_ref": _sha256_file(dir_ref / fname),
            "sha256_cand": _sha256_file(dir_cand / fname),
            "exact_byte_match": _sha256_file(dir_ref / fname) == _sha256_file(dir_cand / fname),
            "passed_0_9999": passed,
        }
    return {
        "passed_all_0_9999": all_passed,
        "files": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build index/test_gallery_text_v5/ and verify parity on val (Option 1B)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--test-gallery-dir",
        type=Path,
        default=PROJECT_ROOT / "index" / "test_gallery",
        help="Path to index/test_gallery",
    )
    parser.add_argument(
        "--test-shard-glob",
        type=str,
        default="data/catalog/captions/shard_test_[0-9]*.parquet",
        help="Glob pattern for test caption shards",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "index" / "test_gallery_text_v5",
        help="Output directory for test_gallery_text_v5",
    )
    parser.add_argument(
        "--val-gallery-dir",
        type=Path,
        default=PROJECT_ROOT / "index" / "val_gallery",
        help="Path to index/val_gallery (for verification & token comparison)",
    )
    parser.add_argument(
        "--val-shard-glob",
        type=str,
        default="data/catalog/captions/shard_val_[0-9]*.parquet",
        help="Glob pattern for val caption shards",
    )
    parser.add_argument(
        "--val-ref-text-dir",
        type=Path,
        default=PROJECT_ROOT / "index" / "val_gallery_text_v5",
        help="Existing index/val_gallery_text_v5 directory (read-only reference)",
    )
    parser.add_argument(
        "--verify-val-out-dir",
        type=Path,
        default=PROJECT_ROOT / "index" / "_verify_val_text_v5",
        help="Temporary output directory for val parity verification",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase5c" / "step1_test_text_index_v5_audit.json",
        help="Output JSON path for Step 1 verification & token length audit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for text encoding",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Verify scripts/build_val_text_index_v5.py was not modified
    val_script_path = PROJECT_ROOT / "scripts" / "build_val_text_index_v5.py"
    val_script_sha = _sha256_file(val_script_path)
    if val_script_sha != LOCKED_BUILD_VAL_SCRIPT_SHA256:
        raise RuntimeError(
            f"LOCKED FILE VIOLATION: scripts/build_val_text_index_v5.py SHA-256 changed! "
            f"Expected {LOCKED_BUILD_VAL_SCRIPT_SHA256}, got {val_script_sha}"
        )

    # Guard against accidentally writing to index/val_gallery_text_v5
    if args.verify_val_out_dir.resolve() == args.val_ref_text_dir.resolve():
        raise ValueError("verify_val_out_dir MUST NOT overwrite val_ref_text_dir!")
    if args.out_dir.resolve() == args.val_ref_text_dir.resolve():
        raise ValueError("out_dir MUST NOT overwrite val_ref_text_dir!")

    print("[Step 1] Loading ECommerceSearchEngine on index/test_gallery...")
    engine = ECommerceSearchEngine(
        index_dir=args.test_gallery_dir,
        config_path=args.config,
        device="cuda",
    )
    tokenizer = engine.preprocessor.processor.tokenizer
    assert engine.preprocessor.max_length == 128, (
        f"Expected max_length=128, got {engine.preprocessor.max_length}"
    )

    # 1. Verify parity on val -> index/_verify_val_text_v5/
    print("[Step 1.1] Running val verification build into index/_verify_val_text_v5/...")
    with open(args.val_gallery_dir / "id_map.json", "r", encoding="utf-8") as f:
        val_id_map = json.load(f)
    with open(args.val_gallery_dir / "manifest.json", "r", encoding="utf-8") as f:
        val_gallery_manifest = json.load(f)
    val_shards = [Path(p) for p in sorted(glob.glob(str(PROJECT_ROOT / args.val_shard_glob)))]
    val_texts_a, val_texts_b, val_texts_meta, val_id_order, val_prov = build_texts_for_gallery(
        val_id_map, val_shards
    )
    val_emb_a, val_emb_b, val_emb_meta = encode_text_gallery(
        engine, val_texts_a, val_texts_b, val_texts_meta, batch_size=args.batch_size
    )
    write_text_gallery_index(
        out_dir=args.verify_val_out_dir,
        gallery_type="verify_val_tuning_split_text_v5",
        gallery_manifest=val_gallery_manifest,
        id_order=val_id_order,
        provenance=val_prov,
        emb_a=val_emb_a,
        emb_b=val_emb_b,
        emb_meta=val_emb_meta,
    )
    val_parity = compare_text_gallery_dirs(args.val_ref_text_dir, args.verify_val_out_dir)
    for fname, stats in val_parity["files"].items():
        print(
            f"  [Val Parity] {fname}: min_cos={stats['min_cosine']:.8f}, "
            f"mean_cos={stats['mean_cosine']:.8f}, lt_0.9999={stats['count_cosine_lt_0_9999']}"
        )
    if not val_parity["passed_all_0_9999"]:
        raise RuntimeError(f"Val parity check failed (< 0.9999): {val_parity}")

    # 2. Build index/test_gallery_text_v5/
    print("[Step 1.2] Building index/test_gallery_text_v5/ from shard_test_0000..0002.parquet...")
    with open(args.test_gallery_dir / "id_map.json", "r", encoding="utf-8") as f:
        test_id_map = json.load(f)
    with open(args.test_gallery_dir / "manifest.json", "r", encoding="utf-8") as f:
        test_gallery_manifest = json.load(f)
    test_shards = [Path(p) for p in sorted(glob.glob(str(PROJECT_ROOT / args.test_shard_glob)))]
    test_texts_a, test_texts_b, test_texts_meta, test_id_order, test_prov = build_texts_for_gallery(
        test_id_map, test_shards
    )
    test_emb_a, test_emb_b, test_emb_meta = encode_text_gallery(
        engine, test_texts_a, test_texts_b, test_texts_meta, batch_size=args.batch_size
    )
    test_manifest = write_text_gallery_index(
        out_dir=args.out_dir,
        gallery_type="test_split_text_v5",
        gallery_manifest=test_gallery_manifest,
        id_order=test_id_order,
        provenance=test_prov,
        emb_a=test_emb_a,
        emb_b=test_emb_b,
        emb_meta=test_emb_meta,
    )

    # 3. Hash Guard (Law 6) and sanity check on index/test_gallery_text_v5/
    print("[Step 1.3] Running Law 6 Hash Guard & Sanity Check on index/test_gallery_text_v5/...")
    test_sanity = verify_text_gallery_index(
        out_dir=args.out_dir,
        gallery_dir=args.test_gallery_dir,
        config_path=args.config,
        expected_n_vectors=1290,
    )
    if not test_sanity["passed"]:
        raise RuntimeError(f"Sanity check failed on index/test_gallery_text_v5/: {test_sanity}")
    print(
        f"  [Sanity OK] n_vectors={test_sanity['n_vectors']}, "
        f"dup_ids={test_sanity['n_duplicate_ids']}, "
        f"id_order_match={test_sanity['id_order_matches_gallery_id_map']}"
    )

    # 4. Token length distribution & truncation rate comparison (val vs test)
    print("[Step 1.4] Computing token length distribution & truncation rate (val vs test)...")
    val_token_stats = compute_token_length_stats(
        tokenizer, val_texts_a, val_texts_b, val_texts_meta, max_length=128
    )
    test_token_stats = compute_token_length_stats(
        tokenizer, test_texts_a, test_texts_b, test_texts_meta, max_length=128
    )

    audit_payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit_hash": _get_git_commit(),
        "seed": 42,
        "build_val_text_index_v5_sha256": val_script_sha,
        "build_val_text_index_v5_untouched": val_script_sha == LOCKED_BUILD_VAL_SCRIPT_SHA256,
        "val_parity_verification": val_parity,
        "test_gallery_text_v5_sanity": test_sanity,
        "test_gallery_text_v5_manifest_summary": {
            "gallery_type": test_manifest["gallery_type"],
            "checkpoint_hash": test_manifest["checkpoint_hash"],
            "catalog_hash": test_manifest["catalog_hash"],
            "shard_hashes": test_manifest["shard_hashes"],
            "n_vectors": test_manifest["n_vectors"],
            "id_order_sha256": test_manifest["id_order_sha256"],
            "caption_model": test_manifest["caption_model"],
            "model_revision": test_manifest["model_revision"],
            "combination_method": test_manifest["combination_method"],
            "generation_params": test_manifest["generation_params"],
            "files": test_manifest["files"],
        },
        "token_length_comparison_max_length_128": {
            "val_max_new_tokens_48": {
                "generation_params": val_prov["generation_params"],
                "stats": val_token_stats,
            },
            "test_max_new_tokens_96": {
                "generation_params": test_prov["generation_params"],
                "stats": test_token_stats,
            },
        },
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2, ensure_ascii=False)
    print(f"[Step 1] Wrote audit report -> {args.audit_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
