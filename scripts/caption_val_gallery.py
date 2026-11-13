"""
Caption val_gallery (1.884 sản phẩm) bằng pipeline Pha 2 (Bước 11.1):
- Dùng đúng VLMCaptionPipeline (InternVL2-2B 8-bit + Vintern-1B-v3_5 8-bit) & merge_caption_v1.
- Hỗ trợ:
  1) --pilot-only: Chạy thử 200 ảnh val (phân tầng seed=42), in 20 caption để người dùng đọc,
     kiểm tra 3 điều kiện dừng: >128 token SigLIP > 0%, chữ Hán > 0%, khớp metadata < 100%.
  2) --full: Chạy toàn bộ 1.884 ảnh val theo shard (mặc định 500 ảnh/shard) có resume và log lỗi,
     lưu vào data/catalog/captions/shard_val_0000..0003.parquet với đầy đủ provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.caption.batch_runner import (
    AUX_MODEL_ID,
    AUX_REVISION,
    CORE_MODEL_ID,
    CORE_REVISION,
    SEED,
)
from src.caption.generator import (
    VLMCaptionPipeline,
    check_chinese_characters,
    check_english_contamination,
    check_metadata_match,
)
from src.caption.merger import get_siglip_tokenizer, merge_caption_v1


def load_val_catalog_records(
    val_id_map_path: pathlib.Path,
    catalog_path: pathlib.Path,
) -> List[Dict[str, Any]]:
    val_id_map = json.loads(val_id_map_path.read_text(encoding="utf-8"))
    val_pids = [str(m.get("product_id") or m.get("item_id")) for m in val_id_map]

    df_cat = pd.read_parquet(catalog_path)
    df_cat["pid_str"] = df_cat["product_id"].astype(str)
    cat_by_pid = df_cat.set_index("pid_str").to_dict(orient="index")

    records: List[Dict[str, Any]] = []
    for pid, m in zip(val_pids, val_id_map):
        rec = dict(cat_by_pid.get(pid, {}))
        rec["product_id"] = pid
        if not rec.get("image_path"):
            rec["image_path"] = m.get("path", "")
        for col in ("articleType", "articleType_vi", "gender", "gender_vi", "baseColour", "baseColour_vi", "title_raw"):
            if not rec.get(col) and m.get(col):
                rec[col] = m.get(col)
        records.append(rec)
    return records


def sample_val_pilot(records: List[Dict[str, Any]], n_samples: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=min(n_samples, len(records)), replace=False)
    indices = sorted(indices.tolist())
    return [records[i] for i in indices]


def run_val_pilot(
    records: List[Dict[str, Any]],
    n_samples: int = 200,
    quantization: str = "8bit",
) -> Dict[str, Any]:
    pilot_records = sample_val_pilot(records, n_samples=n_samples, seed=SEED)
    out_dir = PROJECT_ROOT / "reports/phase5c"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = PROJECT_ROOT / "eval"

    pipeline = VLMCaptionPipeline(seed=SEED, quantization=quantization)
    core_cache = str(eval_dir / "cache_internvl2_val_pilot.json")
    aux_cache = str(eval_dir / "cache_vintern_val_pilot.json")

    core_captions = pipeline.run_core_generation(
        samples=pilot_records,
        model_id=CORE_MODEL_ID,
        revision=CORE_REVISION,
        cache_path=core_cache,
    )
    aux_captions = pipeline.run_aux_generation(
        samples=pilot_records,
        model_id=AUX_MODEL_ID,
        revision=AUX_REVISION,
        cache_path=aux_cache,
    )

    siglip_tok = get_siglip_tokenizer()
    rows: List[Dict[str, Any]] = []
    for rec in pilot_records:
        pid = str(rec["product_id"])
        cap_a = core_captions.get(pid, {}).get("cleaned_caption", "")
        cap_c = aux_captions.get(pid, {}).get("cleaned_caption", "")
        m_res = merge_caption_v1(core_caption=cap_a, metadata=rec, aux_caption=cap_c, max_tokens=128)
        cap_b = m_res["caption"]

        tok_a = len(siglip_tok.encode(cap_a, add_special_tokens=True)) if cap_a else 0
        tok_b = m_res["token_len"]
        zh_a = check_chinese_characters(cap_a)
        zh_b = check_chinese_characters(cap_b)
        en_a = check_english_contamination(cap_a)
        en_b = check_english_contamination(cap_b)
        mm_a = check_metadata_match(cap_a, rec)["all_match"]
        mm_b = check_metadata_match(cap_b, rec)["all_match"]

        rows.append({
            "product_id": pid,
            "articleType_vi": rec.get("articleType_vi", ""),
            "gender_vi": rec.get("gender_vi", ""),
            "baseColour_vi": rec.get("baseColour_vi", ""),
            "caption_A": cap_a,
            "tokens_A": tok_a,
            "zh_A": zh_a,
            "en_A": en_a,
            "meta_match_A": mm_a,
            "caption_B": cap_b,
            "tokens_B": tok_b,
            "zh_B": zh_b,
            "en_B": en_b,
            "meta_match_B": mm_b,
        })

    n = len(rows)
    summary = {
        "n_samples": n,
        "seed": SEED,
        "config_A": {
            "gt_128_rate": round(sum(1 for r in rows if r["tokens_A"] > 128) / n * 100, 2),
            "zh_rate": round(sum(1 for r in rows if r["zh_A"]) / n * 100, 2),
            "en_rate": round(sum(1 for r in rows if r["en_A"]) / n * 100, 2),
            "meta_match_rate": round(sum(1 for r in rows if r["meta_match_A"]) / n * 100, 2),
        },
        "config_B": {
            "gt_128_rate": round(sum(1 for r in rows if r["tokens_B"] > 128) / n * 100, 2),
            "zh_rate": round(sum(1 for r in rows if r["zh_B"]) / n * 100, 2),
            "en_rate": round(sum(1 for r in rows if r["en_B"]) / n * 100, 2),
            "meta_match_rate": round(sum(1 for r in rows if r["meta_match_B"]) / n * 100, 2),
        },
    }
    summary["stop_gate_config_B_triggered"] = (
        summary["config_B"]["gt_128_rate"] > 0.0
        or summary["config_B"]["zh_rate"] > 0.0
        or summary["config_B"]["meta_match_rate"] < 100.0
    )

    (eval_dir / "val_pilot_200_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (eval_dir / "val_pilot_200_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== VAL PILOT 200 SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== 20 SAMPLE CAPTIONS FOR MANUAL REVIEW ===")
    for i, r in enumerate(rows[:20], 1):
        print(f"[{i:02d}] pid={r['product_id']} ({r['articleType_vi']} | {r['gender_vi']} | {r['baseColour_vi']})")
        print(f"     A (tok={r['tokens_A']}): {r['caption_A'][:120]}")
        print(f"     B (tok={r['tokens_B']}): {r['caption_B'][:120]}")

    return summary


def run_val_full(
    records: List[Dict[str, Any]],
    output_dir: pathlib.Path,
    shard_size: int = 500,
    quantization: str = "8bit",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    error_log_path = PROJECT_ROOT / "reports/caption_errors.csv"
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    existing_pids = set()
    for shard_file in output_dir.glob("shard_val_*.parquet"):
        try:
            sdf = pd.read_parquet(shard_file, columns=["product_id"])
            existing_pids.update(sdf["product_id"].astype(str).tolist())
        except Exception as e:
            print(f"Warning reading {shard_file}: {e}")

    pending = [r for r in records if str(r["product_id"]) not in existing_pids]
    print(f"Val total: {len(records)} | Already captioned: {len(existing_pids)} | Pending: {len(pending)}")
    if not pending:
        print("All val_gallery items already captioned!")
        return

    pipeline = VLMCaptionPipeline(seed=SEED, quantization=quantization)
    n_shards = (len(pending) + shard_size - 1) // shard_size
    existing_shards = sorted(output_dir.glob("shard_val_*.parquet"))
    next_idx = len(existing_shards)

    for i in range(n_shards):
        shard_recs = pending[i * shard_size : (i + 1) * shard_size]
        shard_tag = f"val_{next_idx + i:04d}"
        shard_path = output_dir / f"shard_{shard_tag}.parquet"
        core_cache = str(output_dir / f"cache_core_{shard_tag}.json")
        aux_cache = str(output_dir / f"cache_aux_{shard_tag}.json")

        core_caps = pipeline.run_core_generation(
            samples=shard_recs,
            model_id=CORE_MODEL_ID,
            revision=CORE_REVISION,
            cache_path=core_cache,
        )
        aux_caps = pipeline.run_aux_generation(
            samples=shard_recs,
            model_id=AUX_MODEL_ID,
            revision=AUX_REVISION,
            cache_path=aux_cache,
        )

        shard_rows = []
        errors = []
        for rec in shard_recs:
            pid = str(rec["product_id"])
            c_ent = core_caps.get(pid, {})
            a_ent = aux_caps.get(pid, {})
            if c_ent.get("error") or a_ent.get("error"):
                errors.append({
                    "product_id": pid,
                    "image_path": rec.get("image_path", ""),
                    "core_error": c_ent.get("error", ""),
                    "aux_error": a_ent.get("error", ""),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })

            cap_a = c_ent.get("cleaned_caption", "")
            cap_c = a_ent.get("cleaned_caption", "")
            m_res = merge_caption_v1(core_caption=cap_a, metadata=rec, aux_caption=cap_c, max_tokens=128)

            row = dict(rec)
            row["caption_model"] = f"{CORE_MODEL_ID} + {AUX_MODEL_ID}"
            row["model_revision"] = f"{CORE_REVISION} + {AUX_REVISION}"
            row["seed"] = SEED
            row["combination_method"] = "merge_v1"
            row["gen_params"] = json.dumps(
                {"max_new_tokens": 48, "do_sample": False, "num_beams": 1, "quantization": quantization}
            )
            row["caption_A"] = cap_a
            row["caption_B"] = m_res["caption"]
            row["caption_C"] = cap_c
            row["caption_vi"] = m_res["caption"]
            row["token_len"] = m_res["token_len"]
            row["merge_conflicts"] = json.dumps(m_res["conflicts"], ensure_ascii=False)
            shard_rows.append(row)

        pd.DataFrame(shard_rows).to_parquet(shard_path, index=False)
        print(f"Saved {shard_path} ({len(shard_rows)} rows)")

        if errors:
            df_err = pd.DataFrame(errors)
            df_err.to_csv(error_log_path, mode="a" if error_log_path.exists() else "w", header=not error_log_path.exists(), index=False)

        for cp in (pathlib.Path(core_cache), pathlib.Path(aux_cache)):
            if cp.exists():
                cp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Caption val_gallery bằng InternVL2-2B + merge_v1.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--pilot-n", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--quantization", default="8bit")
    args = parser.parse_args()

    records = load_val_catalog_records(
        PROJECT_ROOT / "index/val_gallery/id_map.json",
        PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet",
    )
    if args.mode == "pilot":
        summary = run_val_pilot(records, n_samples=args.pilot_n, quantization=args.quantization)
        return 1 if summary["stop_gate_config_B_triggered"] else 0
    else:
        run_val_full(
            records,
            output_dir=PROJECT_ROOT / "data/catalog/captions",
            shard_size=args.shard_size,
            quantization=args.quantization,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
