"""Phase 3 Independent Verification Script (Items 12-16).
Must be executed against the clean worktree checkout of commit 0c8e49e (reports/verification/_clean_head)
with read-only junctions to data/ and index/.
Writes output ONLY to reports/verification/phase_3/ and reports/verification/phase3/.
"""
import argparse
import gc
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

import faiss
import numpy as np
import pandas as pd
import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worktree-root",
        type=str,
        required=True,
        help="Path to clean checkout worktree of 0c8e49e",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Path to reports/verification/phase_3",
    )
    args = parser.parse_args()

    wt_root = pathlib.Path(args.worktree_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_alt = out_dir.parent / "phase3"
    out_dir_alt.mkdir(parents=True, exist_ok=True)

    # Ensure imports come strictly from the clean worktree 0c8e49e
    sys.path.insert(0, str(wt_root))

    from src.model.siglip_model import (
        HighlyEffectiveSigLIP,
        expand_siglip_position_embeddings,
        verify_checkpoint_hash,
    )
    from src.preprocess.pipeline import SigLIPPreprocessor
    from src.index.faiss_index import FaissFlatIndex, FaissHNSWIndex

    # Verify worktree git HEAD and module paths
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(wt_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(wt_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()

    import src.model.siglip_model as m_mod
    import src.preprocess.pipeline as p_mod
    import src.index.faiss_index as f_mod

    cfg = yaml.safe_load((wt_root / "config.yaml").read_text(encoding="utf-8"))
    ckpt_path = wt_root / cfg["paths"]["checkpoint_file"]
    expected_hash = cfg["checkpoint"]["checkpoint_hash"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res: dict = {
        "phase": 3,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "worktree_verification": {
            "worktree_root": str(wt_root),
            "git_head": git_head,
            "git_status_porcelain": git_status,
            "siglip_model_module_path": str(pathlib.Path(m_mod.__file__).resolve()),
            "preprocessor_module_path": str(pathlib.Path(p_mod.__file__).resolve()),
            "faiss_index_module_path": str(pathlib.Path(f_mod.__file__).resolve()),
            "all_modules_from_clean_worktree": all(
                str(pathlib.Path(mod.__file__).resolve()).startswith(str(wt_root))
                for mod in (m_mod, p_mod, f_mod)
            ),
        },
    }

    # =========================================================================
    # ITEM 12: Checkpoint loading (908/908 keys strict=True, pos_emb 128, dim 1152, L2 norm ~ 1)
    # =========================================================================
    verify_checkpoint_hash(ckpt_path, expected_hash)
    model = HighlyEffectiveSigLIP(model_name="google/siglip-so400m-patch14-224", max_length=128)
    expand_siglip_position_embeddings(model, target_max_length=128)
    model_keys = list(model.state_dict().keys())

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_keys = list(state_dict.keys())
    inc_keys = model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()

    if device == "cuda":
        model = model.half()
    model = model.to(device)
    model.eval()

    pos_emb_weight = model.siglip.text_model.embeddings.position_embedding.weight
    pos_ids = model.siglip.text_model.embeddings.position_ids

    preprocessor = SigLIPPreprocessor(device=device)
    # Test text encoding shape and L2 norm via SigLIPPreprocessor
    sample_texts = ["áo sơ mi nam màu trắng", "ao thun nu mau den", "giày thể thao"]
    txt_inputs = preprocessor.preprocess_texts(sample_texts, device=device)
    with torch.no_grad():
        txt_embs = model.encode_text(txt_inputs["input_ids"], txt_inputs.get("attention_mask")).float().cpu().numpy()
    txt_norms = np.linalg.norm(txt_embs, axis=1)

    res["12_checkpoint_and_architecture"] = {
        "checkpoint_sha256_verified": True,
        "checkpoint_tensor_key_count": len(ckpt_keys),
        "model_tensor_key_count": len(model_keys),
        "strict_load_missing_keys": len(inc_keys.missing_keys),
        "strict_load_unexpected_keys": len(inc_keys.unexpected_keys),
        "matched_keys_ratio": f"{len(ckpt_keys)}/{len(model_keys)}",
        "position_embedding_shape": list(pos_emb_weight.shape),
        "position_ids_shape": list(pos_ids.shape),
        "max_position_embeddings_config": int(model.siglip.config.text_config.max_position_embeddings),
        "embedding_dim": int(model.embed_dim),
        "text_input_ids_shape": list(txt_inputs["input_ids"].shape),
        "text_embedding_output_shape": list(txt_embs.shape),
        "text_embedding_l2_norms": [float(x) for x in txt_norms],
    }

    # =========================================================================
    # ITEM 13: Preprocessing Parity on 200 reference sample images (Rule K4)
    # =========================================================================
    ref_id_map_path = wt_root / "data/_quarantine/legacy_rebuild_20261111/untouchable/index/id_map.json"
    ref_gt_embs_path = wt_root / "data/_quarantine/legacy_rebuild_20261111/untouchable/index/image_embeddings.npy"
    ref_id_map = json.loads(ref_id_map_path.read_text(encoding="utf-8"))
    ref_gt_embs = np.load(ref_gt_embs_path, mmap_mode="r")

    n_samples = 200
    batch_size = 32
    gt_subset = np.array(ref_gt_embs[:n_samples], dtype=np.float32)
    gt_norms = np.linalg.norm(gt_subset, axis=1, keepdims=True)
    gt_subset_norm = gt_subset / np.maximum(gt_norms, 1e-12)

    sample_records = ref_id_map[:n_samples]
    new_embs_list = []
    for b_start in range(0, n_samples, batch_size):
        b_end = min(b_start + batch_size, n_samples)
        batch_paths = [wt_root / item["path"] for item in sample_records[b_start:b_end]]
        pix = preprocessor.preprocess_images(
            images=batch_paths,
            device=device,
            half_precision=(device == "cuda"),
        )
        with torch.no_grad():
            embs = model.encode_image(pix)
            new_embs_list.append(embs.float().cpu().numpy())

    new_embs = np.vstack(new_embs_list)
    new_norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
    new_embs_norm = new_embs / np.maximum(new_norms, 1e-12)

    cosines = (new_embs_norm * gt_subset_norm).sum(axis=1)
    sorted_idx = np.argsort(cosines)
    bottom_10 = [
        {
            "index": int(idx),
            "item_id": str(sample_records[idx].get("item_id")),
            "path": str(sample_records[idx].get("path")),
            "articleType": str(sample_records[idx].get("articleType")),
            "cosine": float(cosines[idx]),
        }
        for idx in sorted_idx[:10]
    ]
    top_5 = [
        {
            "index": int(idx),
            "item_id": str(sample_records[idx].get("item_id")),
            "path": str(sample_records[idx].get("path")),
            "articleType": str(sample_records[idx].get("articleType")),
            "cosine": float(cosines[idx]),
        }
        for idx in sorted_idx[-5:][::-1]
    ]

    rep_min_cos = 0.99966007
    rep_mean_cos = 0.99998093
    actual_min_cos = float(cosines.min())
    actual_mean_cos = float(cosines.mean())

    res["13_parity_check_200"] = {
        "n_samples": n_samples,
        "passed_ge_0999_count": int((cosines >= 0.999).sum()),
        "reported_min_cosine": rep_min_cos,
        "actual_min_cosine": actual_min_cos,
        "min_cosine_abs_diff": abs(actual_min_cos - rep_min_cos),
        "reported_mean_cosine": rep_mean_cos,
        "actual_mean_cosine": actual_mean_cos,
        "mean_cosine_abs_diff": abs(actual_mean_cos - rep_mean_cos),
        "actual_median_cosine": float(np.median(cosines)),
        "actual_max_cosine": float(cosines.max()),
        "actual_std_cosine": float(cosines.std()),
        "image_embedding_l2_norm_min": float(new_norms.min()),
        "image_embedding_l2_norm_max": float(new_norms.max()),
        "image_embedding_l2_norm_mean": float(new_norms.mean()),
        "lowest_10_samples": bottom_10,
        "highest_5_samples": top_5,
    }

    # =========================================================================
    # ITEMS 14, 15, 16: FAISS HNSW parameters, ANN Recall@10, Self-Query, ID Mapping, 4 Sanity Checks
    # =========================================================================
    df_cat = pd.read_parquet(wt_root / "data/catalog/catalog_rebuild.parquet")
    df_cat["pid_str"] = df_cat["product_id"].astype(str)
    train_ids_set = set(df_cat[df_cat["split"] == "train"]["pid_str"])
    test_split_set = {str(x) for x in json.loads((wt_root / "eval/test_split.json").read_text(encoding="utf-8"))}

    def audit_gallery(gallery_rel: str, n_ann_exhaustive: int) -> dict:
        g_dir = wt_root / gallery_rel
        embs = np.load(g_dir / "image_embeddings.npy").astype(np.float32)
        id_map = json.loads((g_dir / "id_map.json").read_text(encoding="utf-8"))
        manifest = json.loads((g_dir / "manifest.json").read_text(encoding="utf-8"))
        hnsw_wrapper = FaissHNSWIndex.load(g_dir / "faiss_hnsw.bin", ef_search=128)
        raw_index = hnsw_wrapper.index

        # Item 14: FAISS HNSW structure
        class_name = type(raw_index).__name__
        d = int(raw_index.d)
        ntotal = int(raw_index.ntotal)
        metric_type = int(raw_index.metric_type)
        metric_is_ip = metric_type == faiss.METRIC_INNER_PRODUCT
        ef_construction = int(raw_index.hnsw.efConstruction)
        ef_search = int(raw_index.hnsw.efSearch)
        # In FAISS HNSW, level 0 has 2*M neighbors, level >=1 has M neighbors
        nb_neighbors_L0 = int(raw_index.hnsw.nb_neighbors(0))
        nb_neighbors_L1 = int(raw_index.hnsw.nb_neighbors(1))
        inferred_M = nb_neighbors_L1

        # Reconstruct vectors from FAISS HNSW to check exact vector order alignment with image_embeddings.npy
        reconstructed = raw_index.reconstruct_n(0, ntotal)
        max_reconstruct_abs_diff = float(np.max(np.abs(reconstructed - embs)))
        emb_norms = np.linalg.norm(embs, axis=1)

        # Verify id_map order and live re-encode of 5 random items (seed 42)
        rng_enc = np.random.default_rng(42)
        check_idxs = rng_enc.choice(ntotal, size=min(5, ntotal), replace=False)
        check_paths = [
            wt_root / (id_map[int(i)].get("image_path") or id_map[int(i)].get("path"))
            for i in check_idxs
        ]
        pix_chk = preprocessor.preprocess_images(check_paths, device=device, half_precision=(device == "cuda"))
        with torch.no_grad():
            live_embs = model.encode_image(pix_chk).float().cpu().numpy()
        live_cosines = [
            float(np.dot(live_embs[k], embs[int(i)]))
            for k, i in enumerate(check_idxs)
        ]

        # Item 15a: ANN Recall@10 vs IndexFlatIP (seed=42, n=100 as in build_index.py, AND larger/exhaustive)
        flat = FaissFlatIndex(dim=1152)
        flat.add(embs)

        rng_ann = np.random.default_rng(42)
        q_idx_100 = rng_ann.choice(ntotal, size=min(100, ntotal), replace=False)
        q_vecs_100 = embs[q_idx_100]
        _, gt_top10_100 = flat.search(q_vecs_100, top_k=10)

        ann_by_ef_100 = {}
        for ef in [32, 64, 128, 256]:
            hnsw_wrapper.set_ef_search(ef)
            _ = hnsw_wrapper.search(q_vecs_100[:5], top_k=10)
            t0 = time.perf_counter()
            _, ann_top10 = hnsw_wrapper.search(q_vecs_100, top_k=10)
            dt_ms = (time.perf_counter() - t0) * 1000.0 / len(q_vecs_100)
            recs = [len(set(gt_top10_100[i]) & set(ann_top10[i])) / 10.0 for i in range(len(q_vecs_100))]
            ann_by_ef_100[str(ef)] = {
                "recall_at_10": float(np.mean(recs)),
                "latency_ms_per_query": round(dt_ms, 4),
            }
        hnsw_wrapper.set_ef_search(128)

        # Exhaustive (all 1290 for test_gallery, or 1000 for reference_gallery) at efSearch=128
        if n_ann_exhaustive >= ntotal:
            q_vecs_ex = embs
        else:
            rng_ex = np.random.default_rng(42)
            q_vecs_ex = embs[rng_ex.choice(ntotal, size=n_ann_exhaustive, replace=False)]
        _, gt_top10_ex = flat.search(q_vecs_ex, top_k=10)
        _, ann_top10_ex = hnsw_wrapper.search(q_vecs_ex, top_k=10)
        recs_ex = [len(set(gt_top10_ex[i]) & set(ann_top10_ex[i])) / 10.0 for i in range(len(q_vecs_ex))]
        ann_exhaustive_ef128 = {
            "n_queries_tested": len(q_vecs_ex),
            "recall_at_10": float(np.mean(recs_ex)),
            "perfect_10_of_10_queries": int(sum(1 for r in recs_ex if r == 1.0)),
        }

        # Item 15b & 16 (Criteria 1-4): Sanity Checks
        all_pids = [str(item["item_id"]) for item in id_map]
        num_dup_ids = len(all_pids) - len(set(all_pids))
        num_test_leak_into_train = len(test_split_set & train_ids_set)

        # Self-query 50 random samples (seed 42) as in sanity_check.py
        rng_sq = np.random.default_rng(42)
        sq_idx_50 = rng_sq.choice(ntotal, size=min(50, ntotal), replace=False)
        sq_scores_50, sq_preds_50 = hnsw_wrapper.search(embs[sq_idx_50], top_k=5)
        sq_rank1_50 = sum(
            1 for i, tid in enumerate(sq_idx_50)
            if int(sq_preds_50[i][0]) == int(tid) and float(sq_scores_50[i][0]) >= 0.9999
        )
        sq_cos_50 = [float(sq_scores_50[i][0]) for i in range(len(sq_idx_50))]

        # Also check all vectors self-query for test_gallery
        sq_all_summary = None
        if ntotal <= 2000:
            sq_scores_all, sq_preds_all = hnsw_wrapper.search(embs, top_k=5)
            # Note: if two items have identical embedding (cosine=1.0), top-1 or top-2 could tie
            exact_idx_rank1 = sum(1 for i in range(ntotal) if int(sq_preds_all[i][0]) == i)
            in_top2_cos1 = sum(
                1 for i in range(ntotal)
                if i in [int(x) for x in sq_preds_all[i][:2]] and float(sq_scores_all[i][0]) >= 0.9999
            )
            sq_all_summary = {
                "total_vectors": ntotal,
                "exact_self_at_rank1": exact_idx_rank1,
                "self_in_top2_with_cos_ge_09999": in_top2_cos1,
                "min_top1_cosine": float(sq_scores_all[:, 0].min()),
                "mean_top1_cosine": float(sq_scores_all[:, 0].mean()),
            }

        c1_pass = num_dup_ids == 0
        c2_pass = num_test_leak_into_train == 0
        c3_pass = sq_rank1_50 == len(sq_idx_50)
        c4_pass = len(embs) == len(id_map) == ntotal == int(manifest["n_vectors"])
        hash_guard_pass = manifest.get("checkpoint_hash", "").lower() == expected_hash.lower()

        return {
            "14_faiss_hnsw_params": {
                "index_class": class_name,
                "d": d,
                "M": inferred_M,
                "nb_neighbors_level0": nb_neighbors_L0,
                "efConstruction": ef_construction,
                "efSearch": ef_search,
                "metric_type_code": metric_type,
                "metric_is_inner_product": metric_is_ip,
                "ntotal": ntotal,
                "embeddings_shape": list(embs.shape),
                "embeddings_l2_norm_min": float(emb_norms.min()),
                "embeddings_l2_norm_max": float(emb_norms.max()),
                "embeddings_l2_norm_mean": float(emb_norms.mean()),
            },
            "15_ann_recall_self_query_and_mapping": {
                "ann_recall_100_random_seed42": ann_by_ef_100,
                "ann_recall_large_ef128": ann_exhaustive_ef128,
                "faiss_reconstruct_vs_npy_max_abs_diff": max_reconstruct_abs_diff,
                "live_reencode_5_samples_cosines": live_cosines,
                "live_reencode_min_cosine": min(live_cosines),
                "self_query_50_seed42": {
                    "tested": len(sq_idx_50),
                    "rank1_matches": sq_rank1_50,
                    "min_cosine": min(sq_cos_50),
                    "mean_cosine": float(np.mean(sq_cos_50)),
                },
                "self_query_all_vectors": sq_all_summary,
            },
            "16_sanity_checks_phase4": {
                "c1_zero_duplicate_ids": {"passed": c1_pass, "duplicate_count": num_dup_ids, "total_ids": len(all_pids)},
                "c2_zero_test_leak_into_train_by_id": {"passed": c2_pass, "leak_count": num_test_leak_into_train},
                "c3_self_query_rank1_50_seed42": {"passed": c3_pass, "rank1_matches": f"{sq_rank1_50}/{len(sq_idx_50)}", "min_cosine": min(sq_cos_50)},
                "c4_vector_count_consistency": {
                    "passed": c4_pass,
                    "n_embeddings": len(embs),
                    "n_id_map": len(id_map),
                    "n_hnsw": ntotal,
                    "n_manifest": int(manifest["n_vectors"]),
                },
                "hash_guard": {"passed": hash_guard_pass, "manifest_checkpoint_hash": manifest.get("checkpoint_hash")},
                "all_4_sanity_checks_passed": bool(c1_pass and c2_pass and c3_pass and c4_pass and hash_guard_pass),
            },
        }

    res["test_gallery"] = audit_gallery("index/test_gallery", n_ann_exhaustive=1290)
    res["reference_gallery"] = audit_gallery("index/reference_gallery", n_ann_exhaustive=1000)

    # Save results
    for d_out in (out_dir, out_dir_alt):
        (d_out / "phase3_results.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
