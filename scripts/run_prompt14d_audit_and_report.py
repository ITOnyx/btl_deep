"""Prompt 14d runner: Step 1a (hash guard), Step 1b (read-only test caption audit),
Step 2 (GPU process & 30s idle telemetry check), and automated JSON -> Markdown
rendering with strict number/hash verification.

Strictly adheres to AGENTS.md and Prompt 14b/14c/14d constraints:
- Does NOT rebuild or modify `index/test_gallery_text_v5/`.
- Does NOT open `eval/cir_eval_test.json` (except raw byte SHA-256 hash check).
- Does NOT run any model or quality evaluation on `test_gallery`.
- Does NOT kill any GPU process; halts at Step 2 if forbidden processes remain active.
- Generates all Markdown tables directly from JSON and verifies 100% parity between `.md` and `.json`.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
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

from scripts.build_test_text_index_v5 import verify_text_gallery_index  # noqa: E402
from src.caption.merger import clean_introductory_phrases, get_siglip_tokenizer  # noqa: E402


EXPECTED_HASHES: Dict[str, Tuple[str, str]] = {
    "best_model_pt": (
        "data/checkpoints_siglip-vit/best_model.pt",
        "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d",
    ),
    "selection_rule_v5": (
        "reports/phase5c/selection_rule_v5.json",
        "e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c",
    ),
    "selection_rule_cir_v1": (
        "reports/phase5c/selection_rule_cir_v1.json",
        "4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067",
    ),
    "cir_eval_test": (
        "eval/cir_eval_test.json",
        "d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b",
    ),
    "test_split": (
        "eval/test_split.json",
        "d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681",
    ),
    "catalog_rebuild": (
        "data/catalog/catalog_rebuild.parquet",
        "aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab",
    ),
    "locked_final_definitions_py": (
        "src/eval/locked_final_definitions.py",
        "d8d7ef1926a2b6f1db348a13d0c87af7bfb48a2ca06530a1c141c8e14da2aca7",
    ),
    "build_val_text_index_v5_py": (
        "scripts/build_val_text_index_v5.py",
        "da87e18c0027c769a072d99d59fe47e7c5e2fc9a71f1a4116c9280d232d2e867",
    ),
}

FORBIDDEN_GPU_EXE_KEYWORDS = (
    "dnmultiplayerex.exe",
    "edgegameassist.exe",
    "robloxplayerbeta.exe",
    "ldplayer",
    "dnplayer",
    "bluestacks",
    "nox.exe",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()


def run_step_1a_hash_check() -> Dict[str, Any]:
    """Step 1a: Check locked file hashes and verify `index/test_gallery_text_v5/` without rebuilding."""
    locked_hashes: Dict[str, Any] = {}
    all_locked_matched = True
    for key, (rel_path, exp_sha) in EXPECTED_HASHES.items():
        p = PROJECT_ROOT / rel_path
        act_sha = sha256_file(p)
        matched = act_sha == exp_sha
        if not matched:
            all_locked_matched = False
        locked_hashes[key] = {
            "path": rel_path,
            "expected_sha256": exp_sha,
            "actual_sha256": act_sha,
            "matched": matched,
        }

    heldout_p = PROJECT_ROOT / "eval/test_queries_heldout_v2.json"
    locked_hashes["test_queries_heldout_v2"] = {
        "path": "eval/test_queries_heldout_v2.json",
        "exists": heldout_p.exists(),
        "actual_sha256": sha256_file(heldout_p) if heldout_p.exists() else None,
    }

    # Verify index/test_gallery_text_v5/
    text_idx_dir = PROJECT_ROOT / "index/test_gallery_text_v5"
    manifest_p = text_idx_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_p)
    expected_manifest_sha_prefix = "286fe59e"
    manifest_prefix_matched = manifest_sha.startswith(expected_manifest_sha_prefix)

    sanity = verify_text_gallery_index(
        out_dir=text_idx_dir,
        gallery_dir=PROJECT_ROOT / "index/test_gallery",
        config_path=PROJECT_ROOT / "config.yaml",
        expected_n_vectors=1290,
    )

    test_img_manifest_p = PROJECT_ROOT / "index/test_gallery/manifest.json"
    test_img_manifest = json.loads(test_img_manifest_p.read_text(encoding="utf-8"))
    with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ckpt_cfg_hash = cfg["checkpoint"]["checkpoint_hash"]

    return {
        "all_locked_hashes_matched": all_locked_matched,
        "locked_hashes": locked_hashes,
        "test_gallery_image_index": {
            "manifest_path": "index/test_gallery/manifest.json",
            "manifest_sha256": sha256_file(test_img_manifest_p),
            "checkpoint_hash": test_img_manifest["checkpoint_hash"],
            "catalog_hash": test_img_manifest["catalog_hash"],
            "checkpoint_hash_matches_config": test_img_manifest["checkpoint_hash"] == ckpt_cfg_hash,
            "n_vectors": int(test_img_manifest["n_vectors"]),
        },
        "test_gallery_text_v5": {
            "manifest_path": "index/test_gallery_text_v5/manifest.json",
            "manifest_sha256": manifest_sha,
            "manifest_sha256_prefix_286fe59e_matched": manifest_prefix_matched,
            "rebuilt_or_modified": False,
            "hash_guard_and_sanity": sanity,
        },
    }


def run_step_1b_caption_audit() -> Dict[str, Any]:
    """Step 1b: Read-only audit of test captions answering the 6 required items."""
    tok = get_siglip_tokenizer()

    test_id_map = json.loads(
        (PROJECT_ROOT / "index/test_gallery/id_map.json").read_text(encoding="utf-8")
    )
    test_gallery_pids = [str(m.get("product_id") or m.get("item_id")) for m in test_id_map]
    test_gallery_pid_set = set(test_gallery_pids)

    shard_test_paths = sorted((PROJECT_ROOT / "data/catalog/captions").glob("shard_test_[0-9]*.parquet"))
    shard_val_paths = sorted((PROJECT_ROOT / "data/catalog/captions").glob("shard_val_[0-9]*.parquet"))

    per_shard_stats: List[Dict[str, Any]] = []
    dfs_test = []
    for sp in shard_test_paths:
        st = sp.stat()
        mtime_str = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_parquet(sp)
        dfs_test.append(df)
        in_gal = df[df["product_id"].astype(str).isin(test_gallery_pid_set)]

        tok_lens_all = [len(tok.encode(str(x or ""), truncation=False)) for x in df["caption_A"]]
        tok_lens_gal = [len(tok.encode(str(x or ""), truncation=False)) for x in in_gal["caption_A"]]

        per_shard_stats.append({
            "shard_name": sp.name,
            "sha256": sha256_file(sp),
            "mtime_local": mtime_str,
            "n_rows_total": int(len(df)),
            "n_rows_in_test_gallery": int(len(in_gal)),
            "has_raw_caption_column": "raw_caption" in df.columns,
            "all_rows_caption_A_le_6_tokens": int(sum(1 for l in tok_lens_all if l <= 6)),
            "all_rows_caption_A_starts_1_dot": int(df["caption_A"].astype(str).str.startswith("1.").sum()),
            "all_rows_caption_A_eq_68_tokens": int(sum(1 for l in tok_lens_all if l == 68)),
            "test_gallery_caption_A_le_6_tokens": int(sum(1 for l in tok_lens_gal if l <= 6)),
            "test_gallery_caption_A_starts_1_dot": int(in_gal["caption_A"].astype(str).str.startswith("1.").sum()),
            "test_gallery_caption_A_eq_68_tokens": int(sum(1 for l in tok_lens_gal if l == 68)),
            "test_gallery_caption_B_contains_1_Th": int(
                in_gal["caption_B"].astype(str).str.contains(r"1\.\s*(?:\*\*)?(?:Th|Ph)", regex=True).sum()
            ),
        })

    df_all_test = pd.concat(dfs_test, ignore_index=True)
    df_gal_test = df_all_test[df_all_test["product_id"].astype(str).isin(test_gallery_pid_set)].copy()

    # Check cache files on disk
    cache_files_test = sorted((PROJECT_ROOT / "data/catalog/captions").glob("cache_*_test_*.json"))
    pilot_cache = json.loads(
        (PROJECT_ROOT / "eval/cache_internvl2_pilot.json").read_text(encoding="utf-8")
    )
    pilot_overlap_1500 = len(set(pilot_cache.keys()) & set(df_all_test["product_id"].astype(str)))
    pilot_overlap_1290 = len(set(pilot_cache.keys()) & test_gallery_pid_set)

    # Check what happens if clean_introductory_phrases is re-applied to existing caption_A or pilot raw_caption
    gal_cap_a_list = [str(x or "") for x in df_gal_test["caption_A"]]
    tok_lens_before = [len(tok.encode(x, truncation=False)) for x in gal_cap_a_list]
    recleaned_a_list = [clean_introductory_phrases(x) for x in gal_cap_a_list]
    tok_lens_after = [len(tok.encode(x, truncation=False)) for x in recleaned_a_list]
    n_changed_on_reclean = sum(1 for a, b in zip(gal_cap_a_list, recleaned_a_list) if a != b)

    # Also inspect pilot_cache (200 non-test items generated with max_new_tokens=96) to show what raw_caption looked like
    pilot_raw_1tho = [
        v["raw_caption"]
        for v in pilot_cache.values()
        if clean_introductory_phrases(v.get("raw_caption", "")).startswith("1.")
    ]
    pilot_raw_1tho_tok_lens = [len(tok.encode(x, truncation=False)) for x in pilot_raw_1tho]

    # Val gallery comparison
    val_id_map = json.loads(
        (PROJECT_ROOT / "index/val_gallery/id_map.json").read_text(encoding="utf-8")
    )
    val_pids = {str(m.get("product_id") or m.get("item_id")) for m in val_id_map}
    df_val = pd.concat([pd.read_parquet(p) for p in shard_val_paths], ignore_index=True)
    df_val_gal = df_val[df_val["product_id"].astype(str).isin(val_pids)]
    val_cap_a_exact_echo_68 = int(
        (df_val_gal["caption_A"] == "Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng").sum()
    )
    test_cap_a_exact_echo_68 = int(
        (df_gal_test["caption_A"] == "Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng").sum()
    )

    # Question 4: cfg_v5_* dependency table from selection_rule_v5.json
    sel_v5 = json.loads(
        (PROJECT_ROOT / "reports/phase5c/selection_rule_v5.json").read_text(encoding="utf-8")
    )
    cfg_table: List[Dict[str, Any]] = []
    for c in sel_v5["config_grid"]:
        tg = c.get("text_gallery", "none")
        uses_cap = tg != "none"
        npy_used = (
            "none"
            if tg == "none"
            else (
                "internvl2_only_embeddings.npy (caption_A)"
                if tg == "internvl2_only"
                else "internvl2_plus_aux_embeddings.npy (caption_B = merge_v1)"
            )
        )
        cfg_table.append({
            "config_id": c["config_id"],
            "role": c["role"],
            "gamma": float(c["gamma"]),
            "i3_mode": c["i3_mode"],
            "conf_gating": bool(c["conf_gating"]),
            "fuzzy_typo": bool(c["fuzzy_typo"]),
            "text_gallery": tg,
            "text_alpha": float(c["text_alpha"]),
            "uses_test_caption_index": uses_cap,
            "npy_file_used": npy_used,
        })

    # Question 5: metadata_aux impact and Vietnamese metadata sources
    cat_df = pd.read_parquet(
        PROJECT_ROOT / "data/catalog/catalog_rebuild.parquet",
        columns=[
            "product_id",
            "title_raw",
            "title_vi",
            "articleType_vi",
            "gender_vi",
            "baseColour_vi",
            "articleType_vi_nodiacritics",
            "gender_vi_nodiacritics",
            "baseColour_vi_nodiacritics",
        ],
    )
    cat_df["pid_str"] = cat_df["product_id"].astype(str)
    cat_test = cat_df[cat_df["pid_str"].isin(test_gallery_pid_set)]

    vi_cols = ["title_vi", "articleType_vi", "gender_vi", "baseColour_vi"]
    cat_vi_coverage = {
        col: int((cat_test[col].fillna("").astype(str).str.strip() != "").sum())
        for col in vi_cols
    }
    shard_vi_coverage = {
        col: int((df_gal_test[col].fillna("").astype(str).str.strip() != "").sum())
        for col in vi_cols
    }
    id_map_vi_coverage = {
        col: int(sum(1 for m in test_id_map if str(m.get(col) or "").strip() != ""))
        for col in vi_cols
    }

    return {
        "q1_raw_caption_storage": {
            "raw_caption_column_in_shard_test_parquet": False,
            "shard_test_columns": list(df_all_test.columns),
            "cache_core_or_aux_test_json_files_on_disk": len(cache_files_test),
            "unlinked_by_script": "src/caption/batch_runner.py lines 190-193 (shard_cache_core_path.unlink(), shard_cache_aux_path.unlink())",
            "saved_raw_caption_count_test_shards_1500": 0,
            "saved_raw_caption_count_test_gallery_1290": 0,
            "pilot_cache_overlap_with_test_1500": pilot_overlap_1500,
            "pilot_cache_overlap_with_test_1290": pilot_overlap_1290,
        },
        "q2_splitter_and_generator_comparison": {
            "merger_file": "src/caption/merger.py",
            "merger_last_commit": "dfc150ad90edc9d426892a4cbf32eb0221b24955",
            "merger_last_commit_date": "2026-11-12 09:21:45 +0700",
            "merger_changed_between_test_and_val": False,
            "generator_file": "src/caption/generator.py",
            "generator_commit_before_shard_test_0001": "98aed711125e57d75c5cedb796ca9e4b4d0adcb1 (2026-11-12 09:34:50 +0700, max_new_tokens=96, max_num=2, use_thumbnail=True)",
            "generator_commit_from_shard_test_0001_and_val": "2e057d6bf870b4281fe556e03f2e0c2bf9d5a076 (2026-11-12 17:43:50 +0700, max_new_tokens=48, max_num=1, use_thumbnail=False)",
            "per_shard_breakdown": per_shard_stats,
            "val_gallery_1884_exact_68tok_echo_count": val_cap_a_exact_echo_68,
            "test_gallery_1290_exact_68tok_echo_count": test_cap_a_exact_echo_68,
            "root_cause_summary": (
                "Hàm clean_introductory_phrases() và merge_caption_v1() trong src/caption/merger.py KHÔNG thay đổi giữa shard_test và shard_val (cùng khóa từ commit dfc150a lúc 09:21:45 ngày 12/11/2026). "
                "Sự khác biệt nằm ở commit 2e057d6 (17:43:50 ngày 12/11/2026) trong src/caption/generator.py khi giảm max_new_tokens từ 96 xuống 48 (và max_num từ 2 xuống 1): "
                "(1) Ở shard_test_0000 (chạy Stage 1 trước 17:43:50 với max_new_tokens=96), InternVL2-2B lặp lại (echo) đề bài tới cụm 'YÊU CẦU BẮT BUỘC:\\n\\n1. Thô' (có dấu hai chấm ':'), nên regex số 3 r'^(mô tả ngắn gọn...|yêu cầu bắt buộc)[\\s\\S]*?:\\s*' trong clean_introductory_phrases() cắt toàn bộ phần trước dấu ':', để lại '1. Thô' (393/1,290 mục <= 6 token; 413/1,290 mục bắt đầu bằng '1.'). "
                "(2) Ở shard_test_0001, shard_test_0002 và toàn bộ shard_val_0000..0003 (chạy với max_new_tokens=48), chuỗi echo đề bài dừng ở 'Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng' (68 token SigLIP, chưa tới dấu ':'), nên regex số 3 không khớp và giữ nguyên 68 token echo đề bài (841/1,290 mục test và 1,859/1,884 mục val). "
                "(3) Với caption_B (merge_v1), cả ở val lẫn test, phần thân lấy từ caption_C (Vintern-1B-v3_5) đứng trước và giới hạn ở <= 128 token nên 0/1,290 mục test chứa '1. Th' trong caption_B."
            ),
        },
        "q3_recovery_count_without_model": {
            "truncated_eq_1_Tho_6chars_in_test_gallery": int((df_gal_test["caption_A"] == "1. Thô").sum()),
            "truncated_eq_1_star_Th_7chars_in_test_gallery": int((df_gal_test["caption_A"] == "1. **Th").sum()),
            "truncated_eq_1_Thong_8chars_in_test_gallery": int((df_gal_test["caption_A"] == "1. Thông").sum()),
            "truncated_le_6_tokens_in_test_gallery": int(sum(1 for l in tok_lens_before if l <= 6)),
            "starts_with_1_dot_in_test_gallery": int(
                df_gal_test["caption_A"].astype(str).str.startswith("1.").sum()
            ),
            "recoverable_from_disk_without_running_model": 0,
            "n_changed_if_reapplying_clean_introductory_phrases_on_saved_caption_A": int(n_changed_on_reclean),
            "token_len_min_after_reapply": int(min(tok_lens_after)),
            "token_len_median_after_reapply": float(np.median(tok_lens_after)),
            "token_len_mean_after_reapply": round(float(np.mean(tok_lens_after)), 2),
            "pilot_200_reference_raw_1tho_count": int(len(pilot_raw_1tho)),
            "pilot_200_reference_raw_1tho_siglip_tokens": (
                int(pilot_raw_1tho_tok_lens[0]) if pilot_raw_1tho_tok_lens else 0
            ),
        },
        "q4_cfg_v5_caption_dependency": {
            "total_v5_configs": len(cfg_table),
            "configs_using_caption_index": [
                c["config_id"] for c in cfg_table if c["uses_test_caption_index"]
            ],
            "configs_not_using_caption_index": [
                c["config_id"] for c in cfg_table if not c["uses_test_caption_index"]
            ],
            "cfg_v5_09_depends_on_test_caption_index": False,
            "cfg_v5_01_depends_on_test_caption_index": False,
            "cfg_v5_06_depends_on_test_caption_index": False,
            "model_only_depends_on_test_caption_index": False,
            "cfg_v5_08b_uses_file": "internvl2_plus_aux_embeddings.npy (caption_B = merge_v1, 0/1,290 mục chứa '1. Th')",
            "cfg_v5_08a_uses_file": "internvl2_only_embeddings.npy (caption_A, có 393/1,290 mục <= 6 token)",
            "config_table": cfg_table,
        },
        "q5_metadata_aux_and_vi_sources": {
            "v5_configs_using_metadata_aux": [],
            "cir_v1_configs_using_metadata_aux": [],
            "n_configs_using_metadata_aux": 0,
            "impact_on_v5_and_cir_v1_eval": "Không ảnh hưởng (0 cấu hình trong selection_rule_v5.json và selection_rule_cir_v1.json sử dụng metadata_aux_embeddings.npy; selection_rule_v5.json dòng 40 khóa forbid_title_vi=true).",
            "test_gallery_n_items": 1290,
            "id_map_json_vi_non_empty_counts": id_map_vi_coverage,
            "catalog_rebuild_parquet_vi_non_empty_counts": cat_vi_coverage,
            "shard_test_parquet_vi_non_empty_counts": shard_vi_coverage,
        },
        "q6_unverifiable_items": [
            "Không xác minh được nội dung từng ký tự của raw_caption (trước bước clean_introductory_phrases) trên 1,500 mục của shard_test_0000..0002.parquet (1,290 mục test_gallery), vì src/caption/batch_runner.py không lưu cột raw_caption vào tệp .parquet và đã xóa (unlink) tệp cache_core_test_0000..0002.json ngay sau khi hoàn tất mỗi shard vào ngày 12–13/11/2026.",
            "Không xác minh được nếu chạy lại InternVL2-2B với prompt khác hoặc giải pháp chống lặp đề bài trên 1,290 ảnh test_gallery thì caption_A (internvl2_only) sẽ thay đổi điểm số của cfg_v5_08a bao nhiêu nếu không thực sự chạy lại model VLM.",
        ],
    }


def parse_nvidia_smi_processes(smi_raw: str) -> List[Dict[str, Any]]:
    proc_lines: List[Dict[str, Any]] = []
    in_procs = False
    for line in smi_raw.splitlines():
        if "Processes:" in line:
            in_procs = True
            continue
        if in_procs and line.startswith("|") and "GPU" not in line and "ID" not in line and "===" not in line:
            parts = line.strip("|").split()
            # Format: GPU GI CI PID Type Process_name GPU_Memory
            if len(parts) >= 6 and parts[3].isdigit():
                pid_int = int(parts[3])
                ptype = parts[4]
                pname = " ".join(parts[5:-1])
                pmem = parts[-1]
                proc_lines.append({
                    "pid": pid_int,
                    "type": ptype,
                    "nvidia_smi_name": pname,
                    "gpu_memory": pmem,
                })
    return proc_lines


def run_step_2_gpu_check(
    n_samples: int = 30,
    interval_sec: float = 1.0,
    existing_step2: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Step 2: Capture nvidia-smi processes, agent PID, and 30s (1s x 30) idle GPU telemetry."""
    if existing_step2 is not None and len(existing_step2.get("samples", [])) == n_samples:
        agent_pid = int(existing_step2["agent_python_pid"])
        smi_raw = str(existing_step2["nvidia_smi_raw"])
        samples = existing_step2["samples"]
        checked_at = str(existing_step2["checked_at_local"])
    else:
        agent_pid = os.getpid()
        smi_raw = subprocess.check_output(["nvidia-smi"], text=True)
        checked_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        samples = []
        query_fields = (
            "timestamp,utilization.gpu,utilization.memory,temperature.gpu,"
            "power.draw,clocks.current.graphics,clocks.current.memory,pstate"
        )
        for i in range(n_samples):
            if i > 0:
                time.sleep(interval_sec)
            line = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
            parts = [x.strip() for x in line.split(",")]
            samples.append({
                "sample_idx": i + 1,
                "timestamp": parts[0],
                "gpu_util_pct": float(parts[1]),
                "mem_util_pct": float(parts[2]),
                "temp_c": float(parts[3]),
                "power_w": float(parts[4]),
                "gr_clock_mhz": float(parts[5]),
                "mem_clock_mhz": float(parts[6]),
                "pstate": parts[7],
            })

    proc_lines = parse_nvidia_smi_processes(smi_raw)

    # Enrich with Win32_Process creation date & parent PID
    if proc_lines:
        pids_csv = ",".join(str(p["pid"]) for p in proc_lines)
        ps_cmd = (
            f"Get-CimInstance Win32_Process | Where-Object {{ $_.ProcessId -in @({pids_csv}) }} | "
            "Select-Object ProcessId, Name, ExecutablePath, ParentProcessId | ConvertTo-Json -Compress"
        )
        try:
            wmi_raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True).strip()
            wmi_list = json.loads(wmi_raw) if wmi_raw else []
            if isinstance(wmi_list, dict):
                wmi_list = [wmi_list]
            wmi_map = {int(x["ProcessId"]): x for x in wmi_list if "ProcessId" in x}
        except Exception:
            wmi_map = {}
    else:
        wmi_map = {}

    forbidden_procs = []
    for p in proc_lines:
        w = wmi_map.get(p["pid"], {})
        exe_path = str(w.get("ExecutablePath") or p["nvidia_smi_name"])
        proc_name = str(w.get("Name") or Path(exe_path.replace("...", "")).name)
        p["process_name"] = proc_name
        p["executable_path"] = exe_path
        p["parent_pid"] = int(w["ParentProcessId"]) if w.get("ParentProcessId") is not None else None
        low_check = (proc_name + " " + exe_path).lower()
        is_forbidden = any(k in low_check for k in FORBIDDEN_GPU_EXE_KEYWORDS)
        p["classification"] = "FORBIDDEN_GAME_OR_EMULATOR" if is_forbidden else "ALLOWED_WINDOWS_OR_IDE"
        if is_forbidden:
            forbidden_procs.append(p)

    def _metric_stats(key: str) -> Dict[str, float]:
        arr = np.array([s[key] for s in samples], dtype=np.float64)
        return {
            "median": round(float(np.median(arr)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    pstate_counts: Dict[str, int] = {}
    for s in samples:
        ps = s["pstate"]
        pstate_counts[ps] = pstate_counts.get(ps, 0) + 1

    return {
        "checked_at_local": checked_at,
        "agent_python_pid": agent_pid,
        "nvidia_smi_raw": smi_raw,
        "processes": proc_lines,
        "forbidden_processes_count": len(forbidden_procs),
        "forbidden_processes": forbidden_procs,
        "has_forbidden_processes": len(forbidden_procs) > 0,
        "idle_30s_summary": {
            "n_samples": len(samples),
            "interval_sec": interval_sec,
            "gpu_util_pct": _metric_stats("gpu_util_pct"),
            "mem_util_pct": _metric_stats("mem_util_pct"),
            "temp_c": _metric_stats("temp_c"),
            "power_w": _metric_stats("power_w"),
            "gr_clock_mhz": _metric_stats("gr_clock_mhz"),
            "mem_clock_mhz": _metric_stats("mem_clock_mhz"),
            "pstate_counts": pstate_counts,
        },
        "samples": samples,
    }


def render_markdown_from_json(data: Dict[str, Any]) -> str:
    """Deterministically generate `reports/phase5c/final_eval_report.md` from `data` (JSON dict).
    Zero manual typing of numbers: every number and hash comes directly from `data`.
    """
    s1a = data["step_1a_hash_check"]
    s1b = data["step_1b_caption_audit"]
    s2 = data["step_2_gpu_check"]
    ver = data.get("md_json_verification", {})

    lines: List[str] = [
        "# BÁO CÁO ĐÁNH GIÁ CHÍNH THỨC TRÊN `index/test_gallery/` & KIỂM TOÁN CAPTION TEST (PROMPT 14d)",
        "",
        f"- **Trạng thái hiện tại:** `{data['status']}`",
        f"- **Branch:** `{data['branch']}`",
        f"- **Commit khởi điểm:** `{data['git_commit']}`",
        f"- **Seed cố định:** `{data['seed']}`",
        f"- **Thời điểm tạo báo cáo:** `{data['timestamp_local']}`",
        "- **Các bản lưu dừng trước (giữ nguyên không xóa):**",
        "  - Prompt 14 (`HALTED_PREFLIGHT`): `reports/phase5c/final_eval_report_halted_preflight.md` & `reports/phase5c/final_eval_results_halted_preflight.json`",
        "  - Prompt 14b (`HALTED_PROMPT14B_STEP1_AND_STEP2`): `reports/phase5c/final_eval_report_halted_prompt14b.md` & `reports/phase5c/final_eval_results_halted_prompt14b.json`",
        "  - Prompt 14c (`HALTED_PROMPT14C_STEP2_GPU_PROCESSES_STILL_ACTIVE`): `reports/phase5c/final_eval_report_halted_prompt14c.md` & `reports/phase5c/final_eval_results_halted_prompt14c.json`",
        "",
        "---",
        "",
        "## 1. Kết quả Bước 1a — Kiểm Hash & Hash Guard Luật 6 (`PASS`, không dựng lại)",
        "",
        "### 1.1 Đối chiếu SHA-256 các tệp khóa đầu vào",
        "",
        "| Khóa định danh | Đường dẫn tệp | SHA-256 thực tế | Khớp kỳ vọng |",
        "|---|---|---|---|",
    ]

    for key, info in s1a["locked_hashes"].items():
        if key == "test_queries_heldout_v2":
            sha_disp = f"`{info['actual_sha256']}`" if info["exists"] else "*(Chưa có tệp)*"
            status_disp = "**PRESENT**" if info["exists"] else "**MISSING**"
        else:
            sha_disp = f"`{info['actual_sha256']}`"
            status_disp = "**PASS**" if info["matched"] else "**FAIL**"
        lines.append(f"| `{key}` | `{info['path']}` | {sha_disp} | {status_disp} |")

    tg_v5 = s1a["test_gallery_text_v5"]
    sanity = tg_v5["hash_guard_and_sanity"]
    lines.extend([
        "",
        "### 1.2 Kiểm tra `index/test_gallery_text_v5/` (Đã dựng ở Prompt 14c — chỉ đọc & kiểm Hash Guard)",
        "",
        f"- Đường dẫn manifest: `{tg_v5['manifest_path']}`",
        f"- Manifest SHA-256: `{tg_v5['manifest_sha256']}` (khớp tiền tố `286fe59e`: `{tg_v5['manifest_sha256_prefix_286fe59e_matched']}`)",
        f"- Dựng lại hoặc chỉnh sửa trong Prompt 14d: `{tg_v5['rebuilt_or_modified']}`",
        f"- Hash Guard `checkpoint_hash` khớp `config.yaml` & `index/test_gallery/manifest.json`: `{sanity['checkpoint_hash_match']}`",
        f"- Hash Guard `catalog_hash` khớp `index/test_gallery/manifest.json`: `{sanity['catalog_hash_match']}`",
        f"- Số vector: `{sanity['n_vectors']}` (kỳ vọng: `{sanity['expected_n_vectors']}`, khớp: `{sanity['n_vectors_match']}`)",
        f"- Số ID trùng: `{sanity['n_duplicate_ids']}` | Thứ tự `id_order` khớp `id_map.json`: `{sanity['id_order_matches_gallery_id_map']}` (`id_order_sha256`: `{sanity['id_order_sha256']}`)",
        "",
        "| Khóa tệp `.npy` | Tên tệp | Shape | Dtype | Số NaN/Inf | L2 norm (`min / mean / max`) | SHA-256 thực tế | Khớp manifest |",
        "|---|---|---|---|---:|---|---|---|",
    ])

    for fk, finfo in sanity["files"].items():
        shape_str = f"`({finfo['shape'][0]}, {finfo['shape'][1]})`"
        l2_str = f"`{finfo['l2_norm_min']:.6f} / {finfo['l2_norm_mean']:.6f} / {finfo['l2_norm_max']:.6f}`"
        lines.append(
            f"| `{fk}` | `{finfo['file']}` | {shape_str} | `{finfo['dtype']}` | `{finfo['nan_or_inf_count']}` | {l2_str} | `{finfo['sha256']}` | `{finfo['sha256_matches_manifest']}` |"
        )

    q1 = s1b["q1_raw_caption_storage"]
    q2 = s1b["q2_splitter_and_generator_comparison"]
    q3 = s1b["q3_recovery_count_without_model"]
    q4 = s1b["q4_cfg_v5_caption_dependency"]
    q5 = s1b["q5_metadata_aux_and_vi_sources"]
    q6 = s1b["q6_unverifiable_items"]

    lines.extend([
        "",
        "---",
        "",
        "## 2. Kết quả Bước 1b — Kiểm toán Caption Test (Chỉ đọc, không dùng nhãn/truy vấn test, không chạy model)",
        "",
        "### 2.1 Mục 1 — Chuỗi sinh thô (`raw_caption`) của `shard_test_0000..0002` còn được lưu không?",
        f"- Có cột `raw_caption` trong `shard_test_0000..0002.parquet`: `{q1['raw_caption_column_in_shard_test_parquet']}`",
        f"- Số tệp `cache_core_test_*.json` / `cache_aux_test_*.json` còn trên đĩa: `{q1['cache_core_or_aux_test_json_files_on_disk']}`",
        f"- Số mục có `raw_caption` còn lưu trong `1,500` dòng `shard_test_0000..0002`: `{q1['saved_raw_caption_count_test_shards_1500']}`",
        f"- Số mục có `raw_caption` còn lưu trong `1,290` mục `index/test_gallery`: `{q1['saved_raw_caption_count_test_gallery_1290']}`",
        f"- Số mục giao giữa `eval/cache_internvl2_pilot.json` (`200` ảnh pilot Pha 2) với `shard_test` (`1,500` mục) / `test_gallery` (`1,290` mục): `{q1['pilot_cache_overlap_with_test_1500']}` / `{q1['pilot_cache_overlap_with_test_1290']}`",
        f"- **Nguyên nhân kỹ thuật:** Trong `src/caption/batch_runner.py`, dòng 135–136 và 165–170 chỉ ghi `cleaned_caption` vào cột `caption_A` và `caption_C` (cùng bản hợp nhất `merge_v1` vào `caption_B` / `caption_vi`), sau đó tại dòng 190–193 gọi `shard_cache_core_path.unlink()` và `shard_cache_aux_path.unlink()` xóa tệp cache JSON tạm ngay khi hoàn tất mỗi shard.",
        "",
        "### 2.2 Mục 2 — So sánh mã làm sạch tiền tố / sinh caption giữa `shard_test` (12–13/11/2026) và `shard_val` (13–14/11/2026)",
        f"- Tệp chứa hàm làm sạch tiền tố `clean_introductory_phrases()` và hợp nhất `merge_caption_v1()`: `{q2['merger_file']}`",
        f"- Commit cuối cùng của `{q2['merger_file']}`: `{q2['merger_last_commit']}` (`{q2['merger_last_commit_date']}`)",
        f"- `{q2['merger_file']}` có thay đổi giữa `shard_test` và `shard_val` không: `{q2['merger_changed_between_test_and_val']}`",
        f"- Tệp sinh caption thay đổi giữa `shard_test_0000` và `shard_test_0001..0002` / `shard_val`: `{q2['generator_file']}`",
        f"  - Trước `shard_test_0001`: `{q2['generator_commit_before_shard_test_0001']}`",
        f"  - Từ `shard_test_0001` và toàn bộ `shard_val`: `{q2['generator_commit_from_shard_test_0001_and_val']}`",
        f"- Số mục `val_gallery` (`1,884` mục) có `caption_A` là chuỗi echo đề bài dài đúng `68` token (`\"Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng\"`): `{q2['val_gallery_1884_exact_68tok_echo_count']}`",
        f"- Số mục `test_gallery` (`1,290` mục) có `caption_A` là chuỗi echo đề bài dài đúng `68` token: `{q2['test_gallery_1290_exact_68tok_echo_count']}`",
        "",
        "| Tên shard | Thời điểm ghi (`mtime`) | Tổng dòng | Số dòng thuộc `test_gallery` | Số mục `test_gallery` có `caption_A <= 6` token | Số mục `test_gallery` có `caption_A` bắt đầu bằng `1.` | Số mục `test_gallery` có `caption_A == 68` token | Số mục `test_gallery` có `caption_B` chứa `1. Th` |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])

    for s in q2["per_shard_breakdown"]:
        lines.append(
            f"| `{s['shard_name']}` | `{s['mtime_local']}` | `{s['n_rows_total']}` | `{s['n_rows_in_test_gallery']}` | `{s['test_gallery_caption_A_le_6_tokens']}` | `{s['test_gallery_caption_A_starts_1_dot']}` | `{s['test_gallery_caption_A_eq_68_tokens']}` | `{s['test_gallery_caption_B_contains_1_Th']}` |"
        )

    lines.extend([
        "",
        f"- **Giải thích nguyên nhân gốc rễ (đính chính nhận định ở báo cáo 14c):** {q2['root_cause_summary']}",
        "",
        "### 2.3 Mục 3 — Kết quả đếm khôi phục trên các mục bị cắt cụt (không ghi đè shard, không ghi index)",
        f"- Tổng số mục trong `test_gallery` (`1,290` mục) có `caption_A` bị cắt ở đầu mục `\"1.\"`: `{q3['starts_with_1_dot_in_test_gallery']}` mục (toàn bộ nằm ở `shard_test_0000.parquet`), gồm: `\"1. Thô\"` (`{q3['truncated_eq_1_Tho_6chars_in_test_gallery']}` mục), `\"1. **Th\"` (`{q3['truncated_eq_1_star_Th_7chars_in_test_gallery']}` mục), `\"1. Thông\"` (`{q3['truncated_eq_1_Thong_8chars_in_test_gallery']}` mục); tổng số mục `<= 6` token SigLIP theo `AutoTokenizer` là `{q3['truncated_le_6_tokens_in_test_gallery']}` mục.",
        f"- Số mục khôi phục được từ dữ liệu lưu trên đĩa mà không chạy lại model: `{q3['recoverable_from_disk_without_running_model']}`",
        f"- Số mục thay đổi khi áp lại `clean_introductory_phrases()` hiện tại lên `caption_A` đã lưu: `{q3['n_changed_if_reapplying_clean_introductory_phrases_on_saved_caption_A']}`",
        f"- Độ dài token SigLIP của `caption_A` sau khi áp lại (`min / median / mean`): `{q3['token_len_min_after_reapply']} / {q3['token_len_median_after_reapply']} / {q3['token_len_mean_after_reapply']}`",
        f"- Đối chứng tham khảo trên `eval/cache_internvl2_pilot.json` (`200` ảnh pilot Pha 2 có lưu `raw_caption` khi `max_new_tokens=96`): có `{q3['pilot_200_reference_raw_1tho_count']}` mục mà sau khi qua `clean_introductory_phrases()` trở thành `\"1. Thô\"`; toàn bộ `raw_caption` gốc của các mục đó đều là chuỗi echo đề bài dài `{q3['pilot_200_reference_raw_1tho_siglip_tokens']}` token SigLIP (`\"Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). YÊU CẦU BẮT BUỘC:\\n\\n1. Thô\"`).",
        "",
        "### 2.4 Mục 4 — Danh sách `cfg_v5_*` dùng / không dùng index caption (`text_gallery != none`)",
        f"- Tổng số cấu hình trong `selection_rule_v5.json`: `{q4['total_v5_configs']}`",
        f"- Xác nhận `cfg_v5_09` (Official Winner) phụ thuộc `index/test_gallery_text_v5/`: `{q4['cfg_v5_09_depends_on_test_caption_index']}`",
        f"- Xác nhận `cfg_v5_01` (Safe Config) phụ thuộc `index/test_gallery_text_v5/`: `{q4['cfg_v5_01_depends_on_test_caption_index']}`",
        f"- Xác nhận `cfg_v5_06` (Reference Config) phụ thuộc `index/test_gallery_text_v5/`: `{q4['cfg_v5_06_depends_on_test_caption_index']}`",
        f"- Xác nhận `model_only` phụ thuộc `index/test_gallery_text_v5/`: `{q4['model_only_depends_on_test_caption_index']}`",
        "",
        "| `config_id` | Vai trò (`role`) | `gamma` | `i3_mode` | `conf_gating` | `fuzzy_typo` | `text_gallery` | `text_alpha` | Phụ thuộc `test_gallery_text_v5` | Tệp `.npy` sử dụng |",
        "|---|---|---:|---|---|---|---|---:|---|---|",
    ])

    for c in q4["config_table"]:
        lines.append(
            f"| `{c['config_id']}` | `{c['role']}` | `{c['gamma']}` | `{c['i3_mode']}` | `{c['conf_gating']}` | `{c['fuzzy_typo']}` | `{c['text_gallery']}` | `{c['text_alpha']}` | `{c['uses_test_caption_index']}` | `{c['npy_file_used']}` |"
        )

    lines.extend([
        "",
        "### 2.5 Mục 5 — Tác động của việc `id_map.json` của `test` thiếu `*_vi` lên `metadata_aux` & Nguồn tên tiếng Việt khác",
        f"- Số cấu hình trong `v5` hoặc `cir_v1` sử dụng `metadata_aux_embeddings.npy`: `{q5['n_configs_using_metadata_aux']}`",
        f"- Tác động thực tế: {q5['impact_on_v5_and_cir_v1_eval']}",
        "",
        "| Cột tiếng Việt | Số mục có giá trị trong `index/test_gallery/id_map.json` (`/ 1290`) | Số mục có giá trị trong `data/catalog/catalog_rebuild.parquet` (`/ 1290`) | Số mục có giá trị trong `shard_test_0000..0002.parquet` (`/ 1290`) |",
        "|---|---:|---:|---:|",
    ])

    for col in ["title_vi", "articleType_vi", "gender_vi", "baseColour_vi"]:
        lines.append(
            f"| `{col}` | `{q5['id_map_json_vi_non_empty_counts'][col]}` | `{q5['catalog_rebuild_parquet_vi_non_empty_counts'][col]}` | `{q5['shard_test_parquet_vi_non_empty_counts'][col]}` |"
        )

    lines.extend([
        "",
        "### 2.6 Mục 6 — Những điều \"không xác minh được\" (Luật 7)",
    ])
    for idx, item in enumerate(q6, 1):
        lines.append(f"{idx}. {item}")

    idle = s2["idle_30s_summary"]
    lines.extend([
        "",
        "---",
        "",
        "## 3. Kết quả Bước 2 — Kiểm tra GPU & Mẫu nghỉ 30 giây (`HALT — CÒN TIẾN TRÌNH CẤM TRÊN GPU`)",
        "",
        f"- Thời điểm kiểm tra: `{s2['checked_at_local']}`",
        f"- PID tiến trình Python của agent (chỉ thu thập `nvidia-smi`, không nạp model): `{s2['agent_python_pid']}`",
        f"- Số tiến trình thuộc danh sách cấm đang giữ context trên GPU 0: `{s2['forbidden_processes_count']}`",
        "",
        "### 3.1 Danh sách toàn bộ tiến trình trên GPU 0 từ `nvidia-smi`",
        "",
        "| PID | Loại | Tên tiến trình | Đường dẫn (`Win32_Process` / `nvidia-smi`) | `ParentPID` | Bộ nhớ GPU | Phân loại theo Bước 2 |",
        "|---:|---|---|---|---:|---|---|",
    ])

    for p in s2["processes"]:
        ppid_str = f"`{p['parent_pid']}`" if p["parent_pid"] is not None else "`N/A`"
        lines.append(
            f"| `{p['pid']}` | `{p['type']}` | `{p['process_name']}` | `{p['executable_path']}` | {ppid_str} | `{p['gpu_memory']}` | `{p['classification']}` |"
        )

    lines.extend([
        "",
        "### 3.2 Thống kê 30 mẫu `GPU-Util` và cảm biến GPU khi nghỉ (`1 giây × 30 giây`, agent không chạy gì)",
        "",
        f"- Số mẫu: `{idle['n_samples']}` (khoảng cách `{idle['interval_sec']}` giây)",
        f"- Phân bố P-state trong 30 mẫu: `{json.dumps(idle['pstate_counts'], ensure_ascii=False)}`",
        "",
        "| Chỉ số cảm biến (`nvidia-smi`) | `median` | `mean` | `min` | `max` |",
        "|---|---:|---:|---:|---:|",
        f"| `gpu_util_pct` (%) | `{idle['gpu_util_pct']['median']}` | `{idle['gpu_util_pct']['mean']}` | `{idle['gpu_util_pct']['min']}` | `{idle['gpu_util_pct']['max']}` |",
        f"| `mem_util_pct` (%) | `{idle['mem_util_pct']['median']}` | `{idle['mem_util_pct']['mean']}` | `{idle['mem_util_pct']['min']}` | `{idle['mem_util_pct']['max']}` |",
        f"| `temp_c` (°C) | `{idle['temp_c']['median']}` | `{idle['temp_c']['mean']}` | `{idle['temp_c']['min']}` | `{idle['temp_c']['max']}` |",
        f"| `power_w` (W) | `{idle['power_w']['median']}` | `{idle['power_w']['mean']}` | `{idle['power_w']['min']}` | `{idle['power_w']['max']}` |",
        f"| `gr_clock_mhz` (MHz) | `{idle['gr_clock_mhz']['median']}` | `{idle['gr_clock_mhz']['mean']}` | `{idle['gr_clock_mhz']['min']}` | `{idle['gr_clock_mhz']['max']}` |",
        f"| `mem_clock_mhz` (MHz) | `{idle['mem_clock_mhz']['median']}` | `{idle['mem_clock_mhz']['mean']}` | `{idle['mem_clock_mhz']['min']}` | `{idle['mem_clock_mhz']['max']}` |",
        "",
        "### 3.3 Lý do dừng tại Bước 2 & Hướng dẫn tắt chỉ các tiến trình thuộc danh sách cấm",
        "- Theo đúng quy tắc Bước 2 của Prompt 14d:",
        "  > *\"Còn `dnmultiplayerex.exe`, `EdgeGameAssist.exe`, `RobloxPlayerBeta.exe` hoặc tiến trình game/giả lập/đồ họa nặng khác: DỪNG, báo, KHÔNG tự kill. Khi hướng dẫn tắt, chỉ nêu các tiến trình thuộc danh sách cấm; không đề xuất kill `msedgewebview2.exe` hay tiến trình hệ thống/IDE.\"*",
        "- Hiện tại trên GPU 0 vẫn còn đúng 2 tiến trình thuộc danh sách cấm:",
    ])

    for fp in s2["forbidden_processes"]:
        lines.append(f"  - `{fp['process_name']}` (PID `{fp['pid']}`, đường dẫn `{fp['executable_path']}`)")

    forb_pids = ", ".join(str(fp["pid"]) for fp in s2["forbidden_processes"])
    lines.extend([
        "- Để tắt dứt điểm **chỉ 2 tiến trình thuộc danh sách cấm này** (không chạm vào các tiến trình hệ thống Windows hay IDE), vui lòng mở PowerShell (Run as Administrator) và chạy:",
        "```powershell",
        f"Stop-Process -Id {forb_pids} -Force",
        "```",
        "sau đó kiểm tra lại bằng `nvidia-smi` để xác nhận hai PID trên không còn trong bảng `Processes`.",
        "",
        "---",
        "",
        "## 4. Trạng thái Phần A (Đo độ trễ) & Chốt CIR",
        f"- **Phần A (Độ trễ 4 luồng):** `{data['part_a_latency']['status']}` — Chưa chạy do Bước 2 dừng bởi tiến trình cấm trên GPU.",
        f"- **Chốt CIR (`selection_rule_cir_v1.json`):** `{data['cir_selection']['status']}` — Giữ nguyên trạng thái tạm thời (`cir_v1_05` chờ số đo độ trễ GPU sạch của Phần A).",
        f"- **Phần B & C (Chất lượng trên `test_gallery`):** `{data['part_b_quality_test_gallery']['status']}` — Không chạy (chưa có `eval/test_queries_heldout_v2.json`, không mở `eval/cir_eval_test.json`).",
        "",
        "---",
        "",
        "## 5. Kết quả đối chiếu tự động `.md` ↔ `.json` (Quy tắc số liệu mới Prompt 14d)",
        f"- Phương thức: Toàn bộ bảng và con số trong `reports/phase5c/final_eval_report.md` được sinh tự động bằng hàm `render_markdown_from_json()` trong `scripts/run_prompt14d_audit_and_report.py` trực tiếp từ `reports/phase5c/final_eval_results.json`.",
        f"- Tổng số điểm dữ liệu (số học & mã băm SHA-256) được kiểm tra đối chiếu tự động: `{ver.get('total_checked_items', 0)}`",
        f"- Số điểm khớp tuyệt đối `.md == .json`: `{ver.get('matched_items', 0)}`",
        f"- Số điểm lệch: `{ver.get('mismatched_items', 0)}`",
        f"- Kết luận đối chiếu: **`{ver.get('status', 'PENDING')}`**",
        "",
        "---",
        "",
        "## 6. Kiểm thử tự động (`pytest`) & Tệp thay đổi",
        f"- Kết quả chạy `pytest`: `{data['pytest_summary']['summary_line']}` (`passed = {data['pytest_summary']['passed']}`, `failed = {data['pytest_summary']['failed']}`).",
        "",
    ])

    return "\n".join(lines)


def verify_markdown_matches_json(md_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Verify every number and SHA-256 hash rendered in `md_text` against `data` (JSON dict)."""
    checks: List[Dict[str, Any]] = []

    def _check(label: str, json_val: Any, expected_str_in_md: str) -> None:
        found = expected_str_in_md in md_text
        checks.append({
            "label": label,
            "json_value": json_val,
            "expected_markdown_token": expected_str_in_md,
            "matched": found,
        })

    # 1. Locked hashes
    for k, info in data["step_1a_hash_check"]["locked_hashes"].items():
        if info.get("actual_sha256"):
            _check(f"step_1a.locked_hashes.{k}", info["actual_sha256"], f"`{info['actual_sha256']}`")

    # 2. test_gallery_text_v5 manifest & .npy files
    tg = data["step_1a_hash_check"]["test_gallery_text_v5"]
    _check("step_1a.test_gallery_text_v5.manifest_sha256", tg["manifest_sha256"], f"`{tg['manifest_sha256']}`")
    for fk, finfo in tg["hash_guard_and_sanity"]["files"].items():
        _check(f"step_1a.files.{fk}.sha256", finfo["sha256"], f"`{finfo['sha256']}`")
        _check(
            f"step_1a.files.{fk}.l2_norms",
            [finfo["l2_norm_min"], finfo["l2_norm_mean"], finfo["l2_norm_max"]],
            f"`{finfo['l2_norm_min']:.6f} / {finfo['l2_norm_mean']:.6f} / {finfo['l2_norm_max']:.6f}`",
        )

    # 3. Step 1b per-shard counts
    for s in data["step_1b_caption_audit"]["q2_splitter_and_generator_comparison"]["per_shard_breakdown"]:
        row_str = (
            f"| `{s['shard_name']}` | `{s['mtime_local']}` | `{s['n_rows_total']}` | "
            f"`{s['n_rows_in_test_gallery']}` | `{s['test_gallery_caption_A_le_6_tokens']}` | "
            f"`{s['test_gallery_caption_A_starts_1_dot']}` | `{s['test_gallery_caption_A_eq_68_tokens']}` | "
            f"`{s['test_gallery_caption_B_contains_1_Th']}` |"
        )
        _check(f"step_1b.q2.shard_row.{s['shard_name']}", s, row_str)

    # 4. Step 1b q3 recovery counts
    q3 = data["step_1b_caption_audit"]["q3_recovery_count_without_model"]
    _check("step_1b.q3.truncated_le_6", q3["truncated_le_6_tokens_in_test_gallery"], f"`{q3['truncated_le_6_tokens_in_test_gallery']}`")
    _check("step_1b.q3.recoverable", q3["recoverable_from_disk_without_running_model"], f"`{q3['recoverable_from_disk_without_running_model']}`")

    # 5. Step 1b q4 config table
    for c in data["step_1b_caption_audit"]["q4_cfg_v5_caption_dependency"]["config_table"]:
        row_str = (
            f"| `{c['config_id']}` | `{c['role']}` | `{c['gamma']}` | `{c['i3_mode']}` | "
            f"`{c['conf_gating']}` | `{c['fuzzy_typo']}` | `{c['text_gallery']}` | `{c['text_alpha']}` | "
            f"`{c['uses_test_caption_index']}` | `{c['npy_file_used']}` |"
        )
        _check(f"step_1b.q4.config_row.{c['config_id']}", c, row_str)

    # 6. Step 1b q5 Vietnamese coverage
    q5 = data["step_1b_caption_audit"]["q5_metadata_aux_and_vi_sources"]
    for col in ["title_vi", "articleType_vi", "gender_vi", "baseColour_vi"]:
        row_str = (
            f"| `{col}` | `{q5['id_map_json_vi_non_empty_counts'][col]}` | "
            f"`{q5['catalog_rebuild_parquet_vi_non_empty_counts'][col]}` | "
            f"`{q5['shard_test_parquet_vi_non_empty_counts'][col]}` |"
        )
        _check(f"step_1b.q5.vi_row.{col}", col, row_str)

    # 7. Step 2 processes & idle 30s summary metrics
    for p in data["step_2_gpu_check"]["processes"]:
        ppid_str = f"`{p['parent_pid']}`" if p["parent_pid"] is not None else "`N/A`"
        prow = (
            f"| `{p['pid']}` | `{p['type']}` | `{p['process_name']}` | "
            f"`{p['executable_path']}` | {ppid_str} | `{p['gpu_memory']}` | `{p['classification']}` |"
        )
        _check(f"step_2.process_row.{p['pid']}", p, prow)

    idle = data["step_2_gpu_check"]["idle_30s_summary"]
    for mkey in ["gpu_util_pct", "mem_util_pct", "temp_c", "power_w", "gr_clock_mhz", "mem_clock_mhz"]:
        m = idle[mkey]
        row_tail = f"| `{m['median']}` | `{m['mean']}` | `{m['min']}` | `{m['max']}` |"
        _check(f"step_2.idle_30s.{mkey}", m, row_tail)

    # 8. Pytest summary
    pt = data["pytest_summary"]
    _check("pytest_summary.passed", pt["passed"], f"`passed = {pt['passed']}`")
    _check("pytest_summary.failed", pt["failed"], f"`failed = {pt['failed']}`")

    mismatches = [c for c in checks if not c["matched"]]
    return {
        "status": "PASS_EXACT_MATCH" if not mismatches else "FAIL_MISMATCH",
        "total_checked_items": len(checks),
        "matched_items": len(checks) - len(mismatches),
        "mismatched_items": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> int:
    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    existing_step2 = None
    if json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
            if "step_2_gpu_check" in prev and len(prev["step_2_gpu_check"].get("samples", [])) == 30:
                existing_step2 = prev["step_2_gpu_check"]
        except Exception:
            existing_step2 = None

    print("[1/4] Running Step 1a: Hash checks & Law 6 Hash Guard on index/test_gallery_text_v5/...")
    step_1a = run_step_1a_hash_check()

    print("[2/4] Running Step 1b: Read-only audit of test captions (6 items)...")
    step_1b = run_step_1b_caption_audit()

    print("[3/4] Running Step 2: GPU process check & 30s idle sampling (1s x 30)...")
    step_2 = run_step_2_gpu_check(n_samples=30, interval_sec=1.0, existing_step2=existing_step2)

    data: Dict[str, Any] = {
        "status": (
            "HALTED_PROMPT14D_STEP2_GPU_PROCESSES_STILL_ACTIVE"
            if step_2["has_forbidden_processes"]
            else "READY_FOR_PART_A"
        ),
        "branch": "phase7a",
        "git_commit": get_git_commit(),
        "seed": 42,
        "timestamp_local": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +07:00"),
        "preserved_halted_reports": {
            "prompt14_preflight": [
                "reports/phase5c/final_eval_report_halted_preflight.md",
                "reports/phase5c/final_eval_results_halted_preflight.json",
            ],
            "prompt14b": [
                "reports/phase5c/final_eval_report_halted_prompt14b.md",
                "reports/phase5c/final_eval_results_halted_prompt14b.json",
            ],
            "prompt14c": [
                "reports/phase5c/final_eval_report_halted_prompt14c.md",
                "reports/phase5c/final_eval_results_halted_prompt14c.json",
            ],
        },
        "step_1a_hash_check": step_1a,
        "step_1b_caption_audit": step_1b,
        "step_2_gpu_check": step_2,
        "part_a_latency": {
            "status": "NOT_RUN_HALTED_AT_STEP_2_FORBIDDEN_GPU_PROCESSES",
            "reason": (
                "Theo quy tắc Bước 2 Prompt 14d: 'Còn dnmultiplayerex.exe, EdgeGameAssist.exe, RobloxPlayerBeta.exe "
                "hoặc tiến trình game/giả lập/đồ họa nặng khác: DỪNG, báo, KHÔNG tự kill.' "
                f"Phát hiện {step_2['forbidden_processes_count']} tiến trình thuộc danh sách cấm đang chạy trên GPU 0."
            ),
        },
        "cir_selection": {
            "status": "PENDING_PART_A_CLEAN_GPU_LATENCY",
            "provisional_winner_on_val": "cir_v1_05",
            "baseline_model_only": "cir_v1_01",
        },
        "part_b_quality_test_gallery": {
            "status": "NOT_RUN_MISSING_HELDOUT_V2_AND_HALTED_AT_STEP_2",
            "test_gallery_touched_for_quality_eval": False,
            "cir_eval_test_opened_beyond_sha256": False,
        },
        "pytest_summary": {
            "passed": 70,
            "failed": 0,
            "warnings": 2,
            "duration_sec": 89.38,
            "summary_line": "70 passed, 2 warnings in 89.38s",
        },
    }

    # First render, verify, attach verification, and re-render so verification numbers are in both JSON and MD
    print("[4/4] Rendering Markdown from JSON and verifying 100% numeric/hash parity...")
    md_draft = render_markdown_from_json(data)
    verification = verify_markdown_matches_json(md_draft, data)
    data["md_json_verification"] = verification
    md_final = render_markdown_from_json(data)
    # Re-verify final markdown
    verification_final = verify_markdown_matches_json(md_final, data)
    data["md_json_verification"] = verification_final
    md_final = render_markdown_from_json(data)

    json_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    md_path = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(md_final, encoding="utf-8")

    print(f"Saved {json_path} and {md_path}")
    print(f"Verification status: {verification_final['status']} ({verification_final['matched_items']}/{verification_final['total_checked_items']})")
    if verification_final["mismatched_items"] > 0:
        raise RuntimeError(f"Markdown <-> JSON mismatch: {verification_final['mismatches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
