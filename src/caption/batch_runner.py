"""Batch Dense Caption Runner for Phase 2.
Executes Option 1: Configuration B (InternVL2-2B Core + Vintern-1B-v3_5 Aux + Metadata Anchor).
Operates in resumable shards (default: 500 items/shard) and logs provenance strictly.
"""

import os
import sys
import gc
import re
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Sanitize environment SSL certificates
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import pandas as pd
import numpy as np

from src.caption.generator import (
    VLMCaptionPipeline,
    load_image_tensor,
    check_chinese_characters,
    check_english_contamination,
    check_metadata_match,
)
from src.caption.prompts import INTERNVL2_PROMPT_TEMPLATE, VINTERN_PROMPT_TEMPLATE
from src.caption.merger import merge_caption_v1, clean_introductory_phrases, get_siglip_tokenizer


CORE_MODEL_ID = "OpenGVLab/InternVL2-2B"
CORE_REVISION = "e4f6747bd20f139e637642c6a058c6bd00b36919"
AUX_MODEL_ID = "5CD-AI/Vintern-1B-v3_5"
AUX_REVISION = "b98f263eab246eb5269ade64edbdca8a887dc44d"
SEED = 42


def run_batch_captioning(
    catalog_path: str = "data/catalog/catalog_rebuild.parquet",
    output_dir: str = "data/catalog/captions",
    split_filter: str = "test",
    shard_size: int = 500,
    max_items: int = None,
    quantization: str = "8bit",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = Path("reports/caption_errors.csv")
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{time.strftime('%X')}] Reading catalog from {catalog_path}...")
    df_cat = pd.read_parquet(catalog_path)
    
    if split_filter and split_filter.lower() != "all":
        df_target = df_cat[df_cat["split"] == split_filter].copy()
    else:
        df_target = df_cat.copy()

    if max_items:
        df_target = df_target.head(max_items).copy()

    total_items = len(df_target)
    print(f"[{time.strftime('%X')}] Target split: '{split_filter}' | Total items to caption: {total_items}")

    # Check already processed product IDs across existing shards
    existing_pids = set()
    for shard_file in out_dir.glob("shard_*.parquet"):
        try:
            sdf = pd.read_parquet(shard_file, columns=["product_id"])
            existing_pids.update(sdf["product_id"].astype(str).tolist())
        except Exception as e:
            print(f"Warning reading existing shard {shard_file}: {e}")

    df_pending = df_target[~df_target["product_id"].astype(str).isin(existing_pids)].copy()
    print(f"[{time.strftime('%X')}] Already completed: {len(existing_pids)} | Pending: {len(df_pending)}")

    if len(df_pending) == 0:
        print(f"[{time.strftime('%X')}] All items in split '{split_filter}' have already been captioned!")
        return

    # Shard preparation
    pending_records = df_pending.to_dict(orient="records")
    n_shards = (len(pending_records) + shard_size - 1) // shard_size
    print(f"[{time.strftime('%X')}] Processing in {n_shards} shard(s) of size <= {shard_size}...")

    pipeline = VLMCaptionPipeline(seed=SEED, quantization=quantization)
    siglip_tokenizer = get_siglip_tokenizer()

    for shard_idx in range(n_shards):
        shard_records = pending_records[shard_idx * shard_size : (shard_idx + 1) * shard_size]
        shard_tag = f"{split_filter}_{shard_idx:04d}"
        shard_out_path = out_dir / f"shard_{shard_tag}.parquet"

        print(f"\n[{time.strftime('%X')}] >>> Starting Shard {shard_idx + 1}/{n_shards} ({len(shard_records)} items) -> {shard_out_path}")
        
        shard_cache_core_path = out_dir / f"cache_core_{shard_tag}.json"
        shard_cache_aux_path = out_dir / f"cache_aux_{shard_tag}.json"

        # 1. Core Generation (InternVL2-2B)
        print(f"[{time.strftime('%X')}] Stage 1: Core VLM ({CORE_MODEL_ID})...")
        core_captions = pipeline.run_core_generation(
            samples=shard_records,
            model_id=CORE_MODEL_ID,
            revision=CORE_REVISION,
            cache_path=str(shard_cache_core_path),
        )

        # 2. Aux Generation (Vintern-1B-v3_5)
        print(f"[{time.strftime('%X')}] Stage 2: Auxiliary VLM ({AUX_MODEL_ID})...")
        aux_captions = pipeline.run_aux_generation(
            samples=shard_records,
            model_id=AUX_MODEL_ID,
            revision=AUX_REVISION,
            cache_path=str(shard_cache_aux_path),
        )

        # 3. Deterministic Merger (merge_v1)
        print(f"[{time.strftime('%X')}] Stage 3: Merging & Provenance Recording...")
        shard_rows = []
        errors = []

        for rec in shard_records:
            pid = str(rec["product_id"])
            c_entry = core_captions.get(pid, {})
            a_entry = aux_captions.get(pid, {})

            cap_core = c_entry.get("cleaned_caption", "")
            cap_aux = a_entry.get("cleaned_caption", "")

            # Log any VLM generation error
            if c_entry.get("error") or a_entry.get("error"):
                errors.append({
                    "product_id": pid,
                    "image_path": rec.get("image_path", ""),
                    "core_error": c_entry.get("error", ""),
                    "aux_error": a_entry.get("error", ""),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            # Merge
            m_res = merge_caption_v1(
                core_caption=cap_core,
                metadata=rec,
                aux_caption=cap_aux,
                max_tokens=128,
            )

            row = dict(rec)
            # Standardized Provenance Columns (Luật 5, Luật D.1)
            row["caption_model"] = f"{CORE_MODEL_ID} + {AUX_MODEL_ID}"
            row["model_revision"] = f"{CORE_REVISION} + {AUX_REVISION}"
            row["seed"] = SEED
            row["combination_method"] = "merge_v1"
            row["gen_params"] = json.dumps({"max_new_tokens": 96, "do_sample": False, "num_beams": 1, "quantization": quantization})
            
            # Configurations A, B, C outputs
            row["caption_A"] = cap_core
            row["caption_B"] = m_res["caption"]
            row["caption_C"] = cap_aux
            
            # Official caption for downstream indexing is Configuration B
            row["caption_vi"] = m_res["caption"]
            row["token_len"] = m_res["token_len"]
            row["merge_conflicts"] = json.dumps(m_res["conflicts"], ensure_ascii=False)

            shard_rows.append(row)

        df_shard = pd.DataFrame(shard_rows)
        df_shard.to_parquet(shard_out_path, index=False)
        print(f"[{time.strftime('%X')}] Shard saved: {shard_out_path} ({len(df_shard)} rows)")

        # Record errors if any
        if errors:
            df_err = pd.DataFrame(errors)
            if error_log_path.exists():
                df_err.to_csv(error_log_path, mode="a", header=False, index=False)
            else:
                df_err.to_csv(error_log_path, index=False)
            print(f"[{time.strftime('%X')}] Logged {len(errors)} error(s) to {error_log_path}")

        # Clean up cache files for completed shard
        if shard_cache_core_path.exists():
            shard_cache_core_path.unlink()
        if shard_cache_aux_path.exists():
            shard_cache_aux_path.unlink()

    print(f"\n[{time.strftime('%X')}] All shards completed successfully for split '{split_filter}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Dense Captioning Runner")
    parser.add_argument("--catalog", type=str, default="data/catalog/catalog_rebuild.parquet")
    parser.add_argument("--output-dir", type=str, default="data/catalog/captions")
    parser.add_argument("--split", type=str, default="test", help="test | val | train | all")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--quantization", type=str, default="8bit")

    args = parser.parse_args()
    run_batch_captioning(
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        split_filter=args.split,
        shard_size=args.shard_size,
        max_items=args.max_items,
        quantization=args.quantization,
    )
