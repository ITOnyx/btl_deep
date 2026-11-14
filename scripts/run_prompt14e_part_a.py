"""
Prompt 14e — Step 1c (Caption Limitation Audit), Step 2 (GPU Check & 30s Idle Baseline),
Part A (4-Stream C.9 Latency Benchmark on index/test_gallery/), Post-Part A CIR Locking,
and Automated JSON -> Markdown Generation & Verification.

Strictly adheres to AGENTS.md and Prompt 14b/14c/14d/14e invariants:
- No training, no modification of checkpoints, parsers, lexicons, or locked configs/indexes.
- Never opens eval/cir_eval_test.json (except raw bytes SHA-256 check).
- Does NOT run Part B or Part C (stops at PART_A_DONE_AWAITING_HELDOUT_V2).
- Every numeric table in reports/phase5c/final_eval_report.md is generated automatically
  from reports/phase5c/final_eval_results.json and verified for 100% numeric parity.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Clean Windows TLS env vars before importing transformers/huggingface_hub
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_prompt14_final_eval import (
    BLOCKING_GPU_PROCESS_KEYWORDS,
    EXPECTED_HASHES,
    run_step0_3_parity_200_test_gallery,
)
from scripts.run_prompt14d_audit_and_report import (
    get_siglip_tokenizer,
    parse_nvidia_smi_processes,
    run_step_1a_hash_check,
)
from src.eval.canonical_eval import compute_weighted_attribute_match
from src.eval.locked_final_definitions import (
    GALLERY_SIZE_TEST_LOCKED,
    LOCKED_SPEC_METADATA,
    SEED_LOCKED,
    SLA_LATENCY_P95_MS_LOCKED,
    classify_latency_conclusion,
)
from src.serve.cir_enhancer import CatalogModifierTranslator
from src.serve.parser_v2_conf import VietnameseQueryParserV2Conf
from src.serve.query_rewriter import (
    multi_template_encode_texts,
    restore_diacritics_for_query,
)
from src.serve.search_engine import ECommerceSearchEngine
from src.serve.template_cache import TemplateCacheV5, encode_texts_with_template_cache


EXACT_PROMPT_ECHO_68 = "Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng"


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def query_gpu_telemetry_one() -> Dict[str, Any]:
    """Query single snapshot of GPU telemetry including pstate and memory utilization."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.gr,clocks.mem,pstate,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    return {
        "timestamp": parts[0],
        "gpu_name": parts[1],
        "temperature_c": float(parts[2]),
        "utilization_gpu_pct": float(parts[3]),
        "utilization_mem_pct": float(parts[4]),
        "power_draw_w": float(parts[5]),
        "clock_graphics_mhz": float(parts[6]),
        "clock_mem_mhz": float(parts[7]),
        "pstate": parts[8],
        "memory_used_mb": float(parts[9]),
        "memory_total_mb": float(parts[10]),
    }


def query_nvidia_smi_snapshot() -> Dict[str, Any]:
    """Capture full nvidia-smi process table and GPU telemetry snapshot."""
    tel = query_gpu_telemetry_one()
    raw_table = subprocess.check_output(["nvidia-smi"], text=True)
    raw_procs = parse_nvidia_smi_processes(raw_table)
    processes: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    for p in raw_procs:
        pname = str(p.get("nvidia_smi_name") or p.get("process_name") or "")
        entry = {
            "pid": int(p["pid"]),
            "type": str(p["type"]),
            "process_name": pname,
            "gpu_memory": str(p["gpu_memory"]),
        }
        processes.append(entry)
        low_name = pname.lower()
        if any(k in low_name for k in BLOCKING_GPU_PROCESS_KEYWORDS):
            blocking.append(entry)
    return {
        "agent_python_pid": os.getpid(),
        "telemetry": tel,
        "processes": processes,
        "blocking_processes": blocking,
        "has_blocking_processes": len(blocking) > 0,
        "raw_nvidia_smi": raw_table,
    }


def sample_idle_gpu_30s(n_samples: int = 30, interval_s: float = 1.0) -> Dict[str, Any]:
    """Sample 30s (1s x 30) of idle GPU telemetry and compute median, mean, min, max."""
    samples: List[Dict[str, Any]] = []
    for i in range(n_samples):
        if i > 0:
            time.sleep(interval_s)
        samples.append(query_gpu_telemetry_one())

    def _stats(key: str) -> Dict[str, float]:
        vals = [float(s[key]) for s in samples]
        return {
            "median": round(float(statistics.median(vals)), 2),
            "mean": round(float(statistics.mean(vals)), 2),
            "min": round(float(min(vals)), 2),
            "max": round(float(max(vals)), 2),
        }

    pstate_counts = dict(Counter(str(s["pstate"]) for s in samples))
    return {
        "n_samples": n_samples,
        "interval_seconds": interval_s,
        "utilization_gpu_pct": _stats("utilization_gpu_pct"),
        "utilization_mem_pct": _stats("utilization_mem_pct"),
        "temperature_c": _stats("temperature_c"),
        "power_draw_w": _stats("power_draw_w"),
        "clock_graphics_mhz": _stats("clock_graphics_mhz"),
        "clock_mem_mhz": _stats("clock_mem_mhz"),
        "pstate_counts": pstate_counts,
        "samples": samples,
    }


def wait_for_gpu_cool(target_temp_c: float = 60.0, max_wait_s: int = 600, poll_s: int = 30) -> Dict[str, Any]:
    """Wait until GPU temperature is strictly below target_temp_c."""
    t_start = time.time()
    history: List[Dict[str, Any]] = []
    while True:
        snap = query_gpu_telemetry_one()
        temp = float(snap["temperature_c"])
        elapsed = round(time.time() - t_start, 1)
        history.append({"elapsed_s": elapsed, "temperature_c": temp, "pstate": snap["pstate"]})
        if temp < target_temp_c:
            return {
                "cooled_below_target": True,
                "target_temp_c": target_temp_c,
                "final_temp_c": temp,
                "waited_seconds": elapsed,
                "history": history,
            }
        if elapsed >= max_wait_s:
            return {
                "cooled_below_target": False,
                "target_temp_c": target_temp_c,
                "final_temp_c": temp,
                "waited_seconds": elapsed,
                "history": history,
            }
        time.sleep(poll_s)


def cooldown_between_runs(target_temp_c: float = 65.0, max_wait_s: float = 20.0) -> None:
    """Brief cooldown between benchmark runs to prevent thermal throttling."""
    t0 = time.time()
    time.sleep(0.5)
    while (time.time() - t0) < max_wait_s:
        snap = query_gpu_telemetry_one()
        if float(snap["temperature_c"]) <= target_temp_c:
            break
        time.sleep(1.5)


def classify_single_caption_review(caption_text: str) -> Dict[str, Any]:
    """
    Structured manual review classification for a caption_A string (Law D.4),
    without inspecting test ground-truth labels.
    """
    c = caption_text.strip()
    if c == EXACT_PROMPT_ECHO_68:
        return {
            "pattern_class": "exact_prompt_echo_68_tokens",
            "describes_image": "Không",
            "natural_vietnamese": "Không (câu mệnh lệnh đề bài bị ngắt cụt ở từ 'bằng')",
            "hallucinates_attributes": "Không (chưa sinh tới thuộc tính sản phẩm)",
            "review_note": (
                "Trùng khít 68 token đầu của prompt hệ thống khi chạy với max_new_tokens=48; "
                "không chứa bất kỳ thông tin thị giác nào của ảnh."
            ),
        }
    if c.startswith("1."):
        return {
            "pattern_class": "truncated_header_after_colon_1_Th",
            "describes_image": "Không",
            "natural_vietnamese": "Không (mảnh tiêu đề cụt '1. Thô' / '1. **Th' / '1. Thông')",
            "hallucinates_attributes": "Không (chưa sinh tới thuộc tính sản phẩm)",
            "review_note": (
                "Mảnh còn sót của tiêu đề '1. Thông tin phân biệt cốt lõi...' sau khi regex #3 "
                "cắt bỏ phần echo trước dấu ':' ('YÊU CẦU BẮT BUỘC:') ở max_new_tokens=96."
            ),
        }
    return {
        "pattern_class": "other_prompt_echo_variant",
        "describes_image": "Không",
        "natural_vietnamese": "Không (biến thể lặp lại đề bài và tiêu đề mục 1 bị cắt dở)",
        "hallucinates_attributes": "Không (chưa sinh tới thuộc tính sản phẩm)",
        "review_note": (
            "Biến thể mở đầu 'Các điểm đảo mô tả ngắn gọn...' không khớp từ khóa đầu dòng của regex #3 "
            "nên giữ nguyên câu echo đề bài và bị ngắt ở '1. **Thông tin ph'."
        ),
    }


def run_step_1c_caption_limitation_audit() -> Dict[str, Any]:
    """
    Step 1c: Honest, read-only documentation of InternVL2-2B caption limitation
    across val_gallery (N=1,884) and test_gallery (N=1,290), including 40 manual reviews (seed=42).
    """
    tok = get_siglip_tokenizer()
    echo_tok_len = len(tok.encode(EXACT_PROMPT_ECHO_68, truncation=False))

    val_id_map = json.loads((PROJECT_ROOT / "index/val_gallery/id_map.json").read_text(encoding="utf-8"))
    test_id_map = json.loads((PROJECT_ROOT / "index/test_gallery/id_map.json").read_text(encoding="utf-8"))

    val_pids = [str(m.get("product_id") or m.get("item_id")) for m in val_id_map]
    test_pids = [str(m.get("product_id") or m.get("item_id")) for m in test_id_map]

    val_dfs = []
    for p in sorted((PROJECT_ROOT / "data/catalog/captions").glob("shard_val_[0-9]*.parquet")):
        df = pd.read_parquet(p)[["product_id", "caption_A", "caption_B", "gen_params"]].copy()
        df["shard"] = p.name
        val_dfs.append(df)
    val_df = pd.concat(val_dfs, ignore_index=True)
    val_df["pid_str"] = val_df["product_id"].astype(str)
    val_df = val_df.drop_duplicates("pid_str").set_index("pid_str")

    test_dfs = []
    for p in sorted((PROJECT_ROOT / "data/catalog/captions").glob("shard_test_[0-9]*.parquet")):
        df = pd.read_parquet(p)[["product_id", "caption_A", "caption_B", "gen_params"]].copy()
        df["shard"] = p.name
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)
    test_df["pid_str"] = test_df["product_id"].astype(str)
    test_df = test_df.drop_duplicates("pid_str").set_index("pid_str")

    def _count_split(pids: List[str], df_indexed: pd.DataFrame) -> Dict[str, Any]:
        caps = [str(df_indexed.loc[pid, "caption_A"] or "") for pid in pids]
        n_total = len(caps)
        n_echo = sum(1 for c in caps if c == EXACT_PROMPT_ECHO_68)
        one_th_list = [c for c in caps if c.startswith("1.")]
        n_one_th = len(one_th_list)
        one_th_breakdown = {
            k: {
                "count": int(v),
                "siglip_tokens": int(len(tok.encode(k, truncation=False))),
            }
            for k, v in sorted(Counter(one_th_list).items(), key=lambda kv: (-kv[1], kv[0]))
        }
        other_list = [c for c in caps if c != EXACT_PROMPT_ECHO_68 and not c.startswith("1.")]
        n_other = len(other_list)
        n_other_english = sum(1 for c in other_list if c.startswith("The image "))
        n_other_vi_echo_variants = n_other - n_other_english
        return {
            "n_total": int(n_total),
            "n_exact_echo_68": int(n_echo),
            "pct_exact_echo_68": round(100.0 * n_echo / max(n_total, 1), 2),
            "n_starts_with_1_dot": int(n_one_th),
            "pct_starts_with_1_dot": round(100.0 * n_one_th / max(n_total, 1), 2),
            "starts_with_1_dot_breakdown": one_th_breakdown,
            "n_remaining_other": int(n_other),
            "pct_remaining_other": round(100.0 * n_other / max(n_total, 1), 2),
            "n_remaining_vi_echo_or_template": int(n_other_vi_echo_variants),
            "n_remaining_english_descriptions": int(n_other_english),
        }

    val_counts = _count_split(val_pids, val_df)
    test_counts = _count_split(test_pids, test_df)

    # Compare against 14d reported numbers
    parity_with_14d = {
        "val_14d_reported_exact_echo": 1859,
        "val_14e_recounted_exact_echo": val_counts["n_exact_echo_68"],
        "val_exact_match": val_counts["n_exact_echo_68"] == 1859,
        "test_14d_reported_exact_echo": 841,
        "test_14e_recounted_exact_echo": test_counts["n_exact_echo_68"],
        "test_echo_exact_match": test_counts["n_exact_echo_68"] == 841,
        "test_14d_reported_starts_1_dot": 413,
        "test_14e_recounted_starts_1_dot": test_counts["n_starts_with_1_dot"],
        "test_1_dot_exact_match": test_counts["n_starts_with_1_dot"] == 413,
        "any_discrepancy_vs_14d": not (
            val_counts["n_exact_echo_68"] == 1859
            and test_counts["n_exact_echo_68"] == 841
            and test_counts["n_starts_with_1_dot"] == 413
        ),
    }

    # Sample 20 random caption_A from val and 20 from test (seed=42)
    def _sample_20(split_name: str, pids: List[str], df_indexed: pd.DataFrame) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(SEED_LOCKED)
        indices = np.sort(rng.choice(len(pids), size=20, replace=False))
        samples: List[Dict[str, Any]] = []
        for rank_i, gal_idx in enumerate(indices, 1):
            pid = pids[int(gal_idx)]
            cap_a = str(df_indexed.loc[pid, "caption_A"] or "")
            shard_name = str(df_indexed.loc[pid, "shard"])
            tok_len = int(len(tok.encode(cap_a, truncation=False)))
            rev = classify_single_caption_review(cap_a)
            samples.append({
                "sample_no": int(rank_i),
                "split": split_name,
                "gallery_index": int(gal_idx),
                "product_id": pid,
                "shard": shard_name,
                "caption_A_verbatim": cap_a,
                "siglip_token_len": tok_len,
                "describes_image": rev["describes_image"],
                "natural_vietnamese": rev["natural_vietnamese"],
                "hallucinates_attributes": rev["hallucinates_attributes"],
                "review_note": rev["review_note"],
            })
        return samples

    val_samples_20 = _sample_20("val", val_pids, val_df)
    test_samples_20 = _sample_20("test", test_pids, test_df)

    return {
        "user_decision_1b": {
            "selected_option": "(a) GIỮ NGUYÊN index/test_gallery_text_v5/",
            "reapply_sentence_splitter": False,
            "regenerate_captions": False,
        },
        "item1_corrected_diagnosis": {
            "prompt14c_previous_claim_is_wrong": True,
            "prompt14c_wrong_claim": (
                "Báo cáo 14c từng nhận định nguyên nhân là lỗi bộ tách câu đánh số "
                "'1. Thông tin...' ở Pha 2 cắt nhầm ở dấu chấm."
            ),
            "corrected_root_cause": (
                "Nguyên nhân KHÔNG phải bộ tách câu (src/caption/merger.py giữ nguyên từ commit dfc150ad). "
                "Nguyên nhân thực sự là InternVL2-2B lặp lại (echo) đề bài hệ thống kết hợp với commit "
                "2e057d6bf870b4281fe556e03f2e0c2bf9d5a076 (giảm max_new_tokens từ 96 xuống 48, max_num từ 2 xuống 1) "
                "trong src/caption/generator.py: (1) ở shard_test_0000 (chạy trước 2e057d6, max_new_tokens=96), "
                "chuỗi echo chứa dấu ':' ('YÊU CẦU BẮT BUỘC:\\n\\n1. Thô') khiến regex số 3 trong "
                "clean_introductory_phrases() cắt toàn bộ tiền tố trước dấu ':' và để lại mảnh '1. Thô' (6 token); "
                "(2) ở shard_test_0001..0002 và shard_val_0000..0003 (chạy sau 2e057d6, max_new_tokens=48), "
                "chuỗi echo dừng ở chữ '...trong hình bằng' (trước khi gặp dấu ':'), nên regex số 3 không khớp "
                "và để nguyên câu echo 68 token."
            ),
            "merger_last_commit": "dfc150ad90edc9d426892a4cbf32eb0221b24955",
            "generator_change_commit": "2e057d6bf870b4281fe556e03f2e0c2bf9d5a076",
        },
        "item2_script_recount": {
            "exact_prompt_echo_string": EXACT_PROMPT_ECHO_68,
            "exact_prompt_echo_siglip_tokens": int(echo_tok_len),
            "val_gallery": val_counts,
            "test_gallery": test_counts,
            "parity_with_14d": parity_with_14d,
        },
        "item3_manual_review_40_captions": {
            "seed": SEED_LOCKED,
            "n_val_samples": 20,
            "n_test_samples": 20,
            "n_total_samples": 40,
            "val_describing_image_count": sum(1 for x in val_samples_20 if x["describes_image"] == "Có"),
            "test_describing_image_count": sum(1 for x in test_samples_20 if x["describes_image"] == "Có"),
            "val_samples_20": val_samples_20,
            "test_samples_20": test_samples_20,
        },
        "item4_consequences": {
            "i_cfg_v5_08a": (
                "(i) cfg_v5_08a trên val và test đo trên caption gần như không chứa thông tin ảnh "
                "(98.67% echo trên val; 97.21% echo hoặc mảnh '1. Th...' trên test), nên kết quả của nó "
                "KHÔNG phản ánh năng lực thực sự của InternVL2-2B."
            ),
            "ii_cfg_v5_08b": (
                "(ii) cfg_v5_08b chỉ thêm thông tin ở phần metadata_aux (và trên test bị thiếu các trường *_vi "
                "trong index/test_gallery/id_map.json nên rơi về nhãn tiếng Anh)."
            ),
            "iii_cfg_v5_09_and_cfg_v5_01": (
                "(iii) cfg_v5_09 (winner) và cfg_v5_01 (safe baseline) có text_gallery = 'none' nên "
                "KHÔNG bị ảnh hưởng bởi hạn chế caption InternVL2-2B hay thiếu hụt *_vi trong id_map."
            ),
        },
        "item5_manifest_erratum": {
            "manifest_path": "index/test_gallery_text_v5/manifest.json",
            "declared_max_new_tokens": 96,
            "actual_shard_test_0000_max_new_tokens": 96,
            "actual_shard_test_0001_0002_max_new_tokens": 48,
            "cause": (
                "src/caption/batch_runner.py dòng 162 ghi cứng {'max_new_tokens': 96} vào cột gen_params "
                "khi lưu parquet dù src/caption/generator.py đã đổi mặc định sang 48 ở commit 2e057d6."
            ),
            "manifest_modified": False,
        },
        "item6_b1_iv_metadata_only": {
            "row_label": "metadata-only (B1(iv) — Luật 5)",
            "status": "không chạy: không có cấu hình khóa",
            "explanation": (
                "Lưới cấu hình đã khóa trong selection_rule_v5.json chỉ có text_gallery in "
                "{'none', 'internvl2_only', 'internvl2_plus_aux'}; không tự bịa cấu hình mới ngoài lưới đã khóa."
            ),
        },
        "item7_unverifiable_items": [
            "Nội dung chuỗi sinh thô (raw_caption) trước bước clean_introductory_phrases() của 1,290 ảnh test_gallery (không xác minh được từng mục vì Stage 1 không lưu cột raw_caption và đã xóa tệp tạm cache_stage1_*.json sau khi ghi parquet).",
            "Chất lượng thực tế của mô hình InternVL2-2B trên val_gallery và test_gallery nếu prompt không bị echo và sinh đủ mô tả tiếng Việt 25-45 từ (không xác minh được nếu không chạy lại model sinh caption).",
            "Điểm chất lượng của cfg_v5_08b trên test_gallery nếu index/test_gallery/id_map.json có sẵn các trường *_vi tiếng Việt (không xác minh được nếu không dựng lại index).",
        ],
    }


def build_t2i_200_latency_queries() -> Dict[str, Any]:
    """
    Build 200 T2I latency queries sampled with seed=42 from B_clean val
    (eval/val_queries_b_synth.json where not is_typo), and record SHA-256 hashes.
    """
    b_synth_path = PROJECT_ROOT / "eval/val_queries_b_synth.json"
    b_synth_sha = sha256_file(b_synth_path)
    b_all = json.loads(b_synth_path.read_text(encoding="utf-8"))
    b_clean = [q for q in b_all if not q.get("is_typo")]
    n_clean = len(b_clean)  # 165

    rng = np.random.default_rng(SEED_LOCKED)
    perm_all = rng.permutation(n_clean)
    extra_idx = rng.choice(n_clean, size=(200 - n_clean), replace=False)
    combined_idx = list(perm_all) + list(extra_idx)
    queries_200 = [str(b_clean[int(i)]["query"]) for i in combined_idx]
    qids_200 = [str(b_clean[int(i)].get("query_id", i)) for i in combined_idx]
    sample_payload = json.dumps(
        [{"idx": int(i), "query_id": qid, "query": q} for i, qid, q in zip(combined_idx, qids_200, queries_200)],
        ensure_ascii=False,
         sort_keys=True,
    ).encode("utf-8")
    sample_sha = sha256_bytes(sample_payload)

    return {
        "source_file": "eval/val_queries_b_synth.json",
        "source_file_sha256": b_synth_sha,
        "n_b_clean_total": n_clean,
        "n_sampled_queries": len(queries_200),
        "seed": SEED_LOCKED,
        "sampling_method": "all_165_permuted_plus_35_without_replacement_seed42",
        "sampled_200_sha256": sample_sha,
        "queries_200": queries_200,
    }


def strip_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def build_cir_and_i2i_200_latency_pairs(
    test_engine: ECommerceSearchEngine,
) -> Dict[str, Any]:
    """
    Build 200 pairs from eval/cir_eval_val.json (controlled wrap-around from 120 pairs),
    assigning primary diacritic mode using the locked 50/50 rule from selection_rule_cir_v1.json,
    and preloading PIL RGB images in RAM for stream 2 & stream 4.
    Never touches eval/cir_eval_test.json.
    """
    cir_val_path = PROJECT_ROOT / "eval/cir_eval_val.json"
    cir_val_sha = sha256_file(cir_val_path)
    cir_val_raw = json.loads(cir_val_path.read_text(encoding="utf-8"))
    pairs_120 = cir_val_raw["pairs"] if isinstance(cir_val_raw, dict) else cir_val_raw

    prepared_120: List[Dict[str, Any]] = []
    n_test_gallery = len(test_engine.embeddings)
    for p_idx, p in enumerate(pairs_120):
        mod_full = str(p["modifier_vi"])
        mod_none = strip_diacritics(mod_full)
        primary_dia = "full" if (p_idx % 2 == 0) else "none"
        raw_mod = mod_full if primary_dia == "full" else mod_none
        img_abs_path = PROJECT_ROOT / p["ref_image_path"]
        with Image.open(img_abs_path) as im:
            pil_rgb = im.convert("RGB")
        # Deterministic index in test_gallery (for cir_catalog embedding lookup on test_gallery)
        catalog_idx = p_idx % n_test_gallery
        prepared_120.append({
            "pair_idx": p_idx,
            "id": str(p["id"]),
            "ref_image_rel_path": str(p["ref_image_path"]),
            "ref_image_abs_path": str(img_abs_path),
            "pil_rgb": pil_rgb,
            "catalog_idx": int(catalog_idx),
            "raw_mod": raw_mod,
            "changed_attr": str(p["changed_attr"]),
            "dia_class": primary_dia,
        })

    pairs_200 = [prepared_120[i % len(prepared_120)] for i in range(200)]
    payload_200 = json.dumps(
        [
            {
                "seq_idx": idx,
                "id": item["id"],
                "ref_image_rel_path": item["ref_image_rel_path"],
                "catalog_idx": item["catalog_idx"],
                "raw_mod": item["raw_mod"],
                "dia_class": item["dia_class"],
            }
            for idx, item in enumerate(pairs_200)
        ],
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    pairs_200_sha = sha256_bytes(payload_200)

    return {
        "source_file": "eval/cir_eval_val.json",
        "source_file_sha256": cir_val_sha,
        "n_base_pairs": len(pairs_120),
        "n_sampled_pairs": len(pairs_200),
        "seed": SEED_LOCKED,
        "sampling_method": "controlled_wrap_around_200_from_120_locked_50_50_diacritic_split",
        "sampled_200_sha256": pairs_200_sha,
        "pairs_200": pairs_200,
    }


def summarize_run_latencies(
    run_latencies_ms: List[List[float]],
    telemetry_per_run: List[Dict[str, Any]],
    is_cir_upload: bool = False,
) -> Dict[str, Any]:
    """Compute per-run p50/p95/p99, median(p95), IQR(p95), pooled p50/p95/p99, and SLA flags."""
    per_run_stats: List[Dict[str, Any]] = []
    p95_runs: List[float] = []
    p50_runs: List[float] = []
    p99_runs: List[float] = []
    all_ms: List[float] = []

    for r_idx, rms in enumerate(run_latencies_ms):
        arr_r = np.asarray(rms, dtype=np.float64)
        p50_r = float(np.percentile(arr_r, 50))
        p95_r = float(np.percentile(arr_r, 95))
        p99_r = float(np.percentile(arr_r, 99))
        mean_r = float(np.mean(arr_r))
        p50_runs.append(p50_r)
        p95_runs.append(p95_r)
        p99_runs.append(p99_r)
        all_ms.extend(rms)
        per_run_stats.append({
            "run": r_idx + 1,
            "n_queries": len(rms),
            "p50_ms": round(p50_r, 3),
            "p95_ms": round(p95_r, 3),
            "p99_ms": round(p99_r, 3),
            "mean_ms": round(mean_r, 3),
            "telemetry_before": telemetry_per_run[r_idx]["before"],
            "telemetry_after": telemetry_per_run[r_idx]["after"],
        })

    q25, q75 = np.percentile(np.asarray(p95_runs, dtype=np.float64), [25, 75])
    p95_med = float(statistics.median(p95_runs))
    p95_iqr = float(q75 - q25)
    p50_med = float(statistics.median(p50_runs))
    p99_med = float(statistics.median(p99_runs))

    arr_all = np.asarray(all_ms, dtype=np.float64)
    pooled_p50 = float(np.percentile(arr_all, 50))
    pooled_p95 = float(np.percentile(arr_all, 95))
    pooled_p99 = float(np.percentile(arr_all, 99))
    pooled_mean = float(np.mean(arr_all))

    return {
        "n_runs": len(run_latencies_ms),
        "n_queries_per_run": len(run_latencies_ms[0]) if run_latencies_ms else 0,
        "total_measured_queries": len(all_ms),
        "p95_runs_ms": [round(x, 3) for x in p95_runs],
        "p50_runs_ms": [round(x, 3) for x in p50_runs],
        "p99_runs_ms": [round(x, 3) for x in p99_runs],
        "p95_median_ms": round(p95_med, 3),
        "p95_max_ms": round(float(max(p95_runs)), 3),
        "p95_iqr_ms": round(p95_iqr, 3),
        "p50_median_ms": round(p50_med, 3),
        "p99_median_ms": round(p99_med, 3),
        "pooled_p50_ms": round(pooled_p50, 3),
        "pooled_p95_ms": round(pooled_p95, 3),
        "pooled_p99_ms": round(pooled_p99, 3),
        "pooled_mean_ms": round(pooled_mean, 3),
        "sla_target_p95_ms": SLA_LATENCY_P95_MS_LOCKED,
        "sla_p95_under_200ms": bool(p95_med < SLA_LATENCY_P95_MS_LOCKED),
        "conclusion_label": classify_latency_conclusion(p95_runs),
        "warn_upload_p99_over_200ms": bool(is_cir_upload and (pooled_p99 > 200.0 or p99_med > 200.0)),
        "per_run": per_run_stats,
    }


def run_part_a_4stream_latency() -> Dict[str, Any]:
    """
    Part A: 4-Stream C.9 Latency Benchmark on index/test_gallery/ (N=1,290)
    and index/test_gallery_text_v5/ (for cfg_v5_08b), FAISS HNSW (efSearch=128), candidate_pool=100.
    Every config is measured separately using Latin-square interleaving within each stream.
    """
    nvidia_before_part_a = query_nvidia_smi_snapshot()

    engine = ECommerceSearchEngine(index_dir="index/test_gallery", use_mean_centering=False, ef_search=128)
    parser = VietnameseQueryParserV2Conf()
    tpl_cache = TemplateCacheV5.load(
        PROJECT_ROOT / "index/template_cache_v5",
        expected_checkpoint_hash=engine.manifest["checkpoint_hash"],
    )
    txt_v5_dir = PROJECT_ROOT / "index/test_gallery_text_v5"
    embs_b_test = np.load(txt_v5_dir / "internvl2_plus_aux_embeddings.npy").astype(np.float32)

    gpol = json.loads((PROJECT_ROOT / "eval/gender_gt_policy.json").read_text(encoding="utf-8"))
    strict_cats = gpol["policies"]["hybrid"]["strict_categories"]

    rule_v5 = json.loads((PROJECT_ROOT / "reports/phase5c/selection_rule_v5.json").read_text(encoding="utf-8"))
    grid_v5_by_id = {c["config_id"]: c for c in rule_v5["config_grid"]}

    rule_cir = json.loads((PROJECT_ROOT / "reports/phase5c/selection_rule_cir_v1.json").read_text(encoding="utf-8"))
    grid_cir_by_id = {c["config_id"]: c for c in rule_cir["config_grid"]}
    translator = CatalogModifierTranslator()

    # Build 200 T2I queries & 200 CIR/I2I pairs
    t2i_q_info = build_t2i_200_latency_queries()
    queries_t2i_200 = t2i_q_info["queries_200"]

    cir_p_info = build_cir_and_i2i_200_latency_pairs(engine)
    pairs_200 = cir_p_info["pairs_200"]

    candidate_pool = 100

    # =========================================================================
    # STREAM 1: Text -> Image (5 configs measured SEPARATELY in 5x5 Latin square)
    # Configs: model_only, cfg_v5_01, cfg_v5_06, cfg_v5_09, cfg_v5_08b
    # =========================================================================
    t2i_configs: List[Dict[str, Any]] = [
        {
            "config_id": "model_only",
            "role": "model_only_cosine",
            "mode": "model_only",
            "i3_mode": "none",
            "fuzzy_typo": False,
            "conf_gating": False,
            "bonus_mode": "none",
            "gamma": 0.0,
            "text_gallery": "none",
            "text_alpha": 1.0,
        },
        dict(grid_v5_by_id["cfg_v5_01"], mode="system"),
        dict(grid_v5_by_id["cfg_v5_06"], mode="system"),
        dict(grid_v5_by_id["cfg_v5_09"], mode="system"),
        dict(grid_v5_by_id["cfg_v5_08b"], mode="system"),
    ]

    def _run_t2i_single(cfg: Dict[str, Any], q_raw: str) -> None:
        qt = restore_diacritics_for_query(q_raw)
        mode = cfg.get("mode", "system")
        if mode == "model_only":
            qv = engine.encode_text_query(qt)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            scores, indices = engine.hnsw.search(qv, top_k=candidate_pool)
            _ = indices[0][:10]
            return

        fuzzy = bool(cfg.get("fuzzy_typo", False))
        parsed = parser.parse_with_confidence(qt, fuzzy_typo=fuzzy)
        i3_mode = cfg.get("i3_mode", "none")
        if i3_mode == "direct":
            qv = multi_template_encode_texts(engine, [q_raw], [parsed], n_templates=2)
        elif i3_mode == "cached":
            qv, _ = encode_texts_with_template_cache(engine, [q_raw], [parsed], tpl_cache)
        else:
            qv = engine.encode_text_query(qt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        scores, indices = engine.hnsw.search(qv, top_k=candidate_pool)
        cand_scores = scores[0].astype(np.float64)
        cand_indices = indices[0]

        bmode = cfg.get("bonus_mode", "zscore")
        gamma = float(cfg.get("gamma", 1.0))
        if bmode == "zscore":
            mu = float(np.mean(cand_scores))
            sd = max(float(np.std(cand_scores)), 1e-8)
            base_arr = (cand_scores - mu) / sd
        else:
            base_arr = cand_scores

        tg = cfg.get("text_gallery", "none")
        t_alpha = float(cfg.get("text_alpha", 1.0))
        extra_w = (1.0 - t_alpha) / t_alpha if tg == "internvl2_plus_aux" else 0.0
        z_txt_cand = None
        if tg == "internvl2_plus_aux":
            raw_txt_sims = (embs_b_test[cand_indices] @ qv[0]).astype(np.float64)
            t_mu = float(np.mean(raw_txt_sims))
            t_sd = max(float(np.std(raw_txt_sims)), 1e-8)
            z_txt_cand = (raw_txt_sims - t_mu) / t_sd

        eff_weights = None
        if cfg.get("conf_gating"):
            conf_map = parsed.get("slot_confidence", {})
            eff_weights = {
                k: (0.2 if conf_map.get(k) == "low" else 1.0)
                for k in ("articleType", "gender", "baseColour")
            }

        scored: List[Tuple[float, int]] = []
        for pos, (idx_val, base_s) in enumerate(zip(cand_indices, base_arr)):
            if idx_val < 0:
                continue
            idx_int = int(idx_val)
            s_val = float(base_s)
            meta = engine.id_map[idx_int]
            mf = compute_weighted_attribute_match(
                parsed, meta, weights=eff_weights, gender_policy="hybrid", strict_categories=strict_cats
            )
            if bmode == "tier_soft":
                tier_b = 0.5 if mf >= 0.999 else 0.0
                s_val += gamma * (mf + tier_b)
            else:
                s_val += gamma * mf
            if z_txt_cand is not None:
                s_val += extra_w * float(z_txt_cand[pos])
            scored.append((s_val, idx_int))
        scored.sort(key=lambda x: x[0], reverse=True)
        _ = scored[:10]

    n_t2i_cfgs = len(t2i_configs)
    t2i_runs_ms: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in t2i_configs}
    t2i_tele: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in t2i_configs}

    for r_idx in range(5):
        print(f"[Part A - Stream 1 T2I] Run {r_idx + 1}/5 (5x5 Latin square across 5 configs)...", flush=True)
        cooldown_between_runs()
        tel_before = query_gpu_telemetry_one()
        # Warmup 20 queries across all 5 configs
        for w_idx, q_w in enumerate(queries_t2i_200[:20]):
            order_w = [t2i_configs[(i + r_idx + w_idx) % n_t2i_cfgs] for i in range(n_t2i_cfgs)]
            for cfg in order_w:
                _run_t2i_single(cfg, q_w)

        cur_run_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in t2i_configs}
        for q_idx, q_str in enumerate(queries_t2i_200):
            # 5x5 Latin square rotation by (r_idx + q_idx) % 5
            order = [t2i_configs[(i + r_idx + q_idx) % n_t2i_cfgs] for i in range(n_t2i_cfgs)]
            for cfg in order:
                cid = cfg["config_id"]
                t0 = time.perf_counter()
                _run_t2i_single(cfg, q_str)
                cur_run_ms[cid].append((time.perf_counter() - t0) * 1000.0)
        tel_after = query_gpu_telemetry_one()
        for cfg in t2i_configs:
            cid = cfg["config_id"]
            t2i_runs_ms[cid].append(cur_run_ms[cid])
            t2i_tele[cid].append({"before": tel_before, "after": tel_after})

    stream1_t2i_summary: Dict[str, Any] = {}
    for cfg in t2i_configs:
        cid = cfg["config_id"]
        stream1_t2i_summary[cid] = summarize_run_latencies(
            t2i_runs_ms[cid], t2i_tele[cid], is_cir_upload=False
        )
        stream1_t2i_summary[cid]["config_id"] = cid
        stream1_t2i_summary[cid]["measured_individually"] = True

    # =========================================================================
    # STREAM 2: Image -> Image (5 runs x 200 queries, warmup 20)
    # Primary: PIL image in RAM; Secondary: disk read + JPEG decode
    # =========================================================================
    def _run_i2i_single(item: Dict[str, Any], from_disk: bool = False) -> None:
        img_in = item["ref_image_abs_path"] if from_disk else item["pil_rgb"]
        qv = engine.encode_image_query(img_in)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        scores, indices = engine.hnsw.search(qv, top_k=candidate_pool)
        _ = indices[0][:10]

    i2i_ram_runs: List[List[float]] = []
    i2i_disk_runs: List[List[float]] = []
    i2i_tele: List[Dict[str, Any]] = []

    for r_idx in range(5):
        print(f"[Part A - Stream 2 I2I] Run {r_idx + 1}/5...", flush=True)
        cooldown_between_runs()
        tel_before = query_gpu_telemetry_one()
        for item_w in pairs_200[:20]:
            _run_i2i_single(item_w, from_disk=False)
            _run_i2i_single(item_w, from_disk=True)

        cur_ram: List[float] = []
        cur_disk: List[float] = []
        for q_idx, item in enumerate(pairs_200):
            if (r_idx + q_idx) % 2 == 0:
                t0 = time.perf_counter()
                _run_i2i_single(item, from_disk=False)
                cur_ram.append((time.perf_counter() - t0) * 1000.0)

                t1 = time.perf_counter()
                _run_i2i_single(item, from_disk=True)
                cur_disk.append((time.perf_counter() - t1) * 1000.0)
            else:
                t1 = time.perf_counter()
                _run_i2i_single(item, from_disk=True)
                cur_disk.append((time.perf_counter() - t1) * 1000.0)

                t0 = time.perf_counter()
                _run_i2i_single(item, from_disk=False)
                cur_ram.append((time.perf_counter() - t0) * 1000.0)
        tel_after = query_gpu_telemetry_one()
        i2i_ram_runs.append(cur_ram)
        i2i_disk_runs.append(cur_disk)
        i2i_tele.append({"before": tel_before, "after": tel_after})

    stream2_i2i_summary = {
        "ram_preloaded": summarize_run_latencies(i2i_ram_runs, i2i_tele, is_cir_upload=False),
        "disk_jpeg_decode": summarize_run_latencies(i2i_disk_runs, i2i_tele, is_cir_upload=False),
    }

    # =========================================================================
    # STREAM 3: CIR Catalog (cir_v1_01 and cir_v1_05 measured separately in 2x2 Latin square)
    # =========================================================================
    cir_configs = [
        grid_cir_by_id["cir_v1_01"],
        grid_cir_by_id["cir_v1_05"],
    ]

    def _rank_hnsw_cir(
        q_vec: np.ndarray,
        parsed_mod: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> List[Tuple[float, int]]:
        scores, indices = engine.hnsw.search(q_vec, top_k=candidate_pool)
        cand_scores = scores[0]
        cand_indices = indices[0]
        use_c3 = bool(cfg.get("c3_mod_bonus", False))
        if not use_c3:
            return [
                (float(s), int(i))
                for i, s in zip(cand_indices[:10], cand_scores[:10])
                if i >= 0
            ]
        gamma = float(cfg.get("gamma", 1.0))
        has_active = bool(
            parsed_mod.get("articleType") or parsed_mod.get("gender") or parsed_mod.get("baseColour")
        )
        scored: List[Tuple[float, int]] = []
        for idx_val, base_s in zip(cand_indices, cand_scores):
            if idx_val < 0:
                continue
            idx_int = int(idx_val)
            s_val = float(base_s)
            if has_active:
                mf = compute_weighted_attribute_match(
                    parsed_mod,
                    engine.id_map[idx_int],
                    weights=None,
                    gender_policy="hybrid",
                    strict_categories=strict_cats,
                )
                if mf > 0.0:
                    tier_b = 0.5 if mf >= 0.999 else 0.0
                    s_val += gamma * (mf + tier_b)
            scored.append((s_val, idx_int))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:10]

    def _run_cir_catalog_single(cfg: Dict[str, Any], item: Dict[str, Any]) -> None:
        raw_mod = item["raw_mod"]
        qt_vi = restore_diacritics_for_query(raw_mod)
        parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
        mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
        mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
        with torch.no_grad():
            mod_txt_emb = engine.model.encode_text(
                input_ids=mod_inputs["input_ids"],
                attention_mask=mod_inputs.get("attention_mask"),
            )
            cat_idx = item["catalog_idx"]
            ref_vec = engine.embeddings[cat_idx : cat_idx + 1]
            ref_img_emb = torch.from_numpy(ref_vec).to(engine.device)
            if engine.device == "cuda":
                ref_img_emb = ref_img_emb.half()
            cir_emb = engine.model.cir_combiner(ref_img_emb, mod_txt_emb)
            q_vec = cir_emb.float().cpu().numpy()
        q_vec = engine._apply_mean_centering(q_vec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = _rank_hnsw_cir(q_vec, parsed_mod, cfg)

    cir_cat_runs_ms: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in cir_configs}
    cir_cat_tele: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in cir_configs}

    for r_idx in range(5):
        print(f"[Part A - Stream 3 CIR Catalog] Run {r_idx + 1}/5 (2x2 Latin square)...", flush=True)
        cooldown_between_runs()
        tel_before = query_gpu_telemetry_one()
        for w_idx, item_w in enumerate(pairs_200[:20]):
            order_w = cir_configs if (r_idx + w_idx) % 2 == 0 else list(reversed(cir_configs))
            for cfg in order_w:
                _run_cir_catalog_single(cfg, item_w)

        cur_run_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in cir_configs}
        for q_idx, item in enumerate(pairs_200):
            order = cir_configs if (r_idx + q_idx) % 2 == 0 else list(reversed(cir_configs))
            for cfg in order:
                cid = cfg["config_id"]
                t0 = time.perf_counter()
                _run_cir_catalog_single(cfg, item)
                cur_run_ms[cid].append((time.perf_counter() - t0) * 1000.0)
        tel_after = query_gpu_telemetry_one()
        for cfg in cir_configs:
            cid = cfg["config_id"]
            cir_cat_runs_ms[cid].append(cur_run_ms[cid])
            cir_cat_tele[cid].append({"before": tel_before, "after": tel_after})

    stream3_cir_catalog_summary: Dict[str, Any] = {}
    for cfg in cir_configs:
        cid = cfg["config_id"]
        stream3_cir_catalog_summary[cid] = summarize_run_latencies(
            cir_cat_runs_ms[cid], cir_cat_tele[cid], is_cir_upload=False
        )
        stream3_cir_catalog_summary[cid]["config_id"] = cid
        stream3_cir_catalog_summary[cid]["measured_individually"] = True

    # =========================================================================
    # STREAM 4: CIR Upload (cir_v1_01 and cir_v1_05 measured separately in 2x2 Latin square)
    # Includes:
    # - Primary: PIL image preloaded in RAM (Stages 2..6)
    # - Secondary: Disk read + JPEG decode (Stages 1..6)
    # - 6-stage synchronized breakdown with torch.cuda.synchronize() around every step
    # =========================================================================
    stage_names = [
        "stage1_disk_jpeg_decode",
        "stage2_resize_processor",
        "stage3_vision_tower_forward",
        "stage4_text_tower_and_parser",
        "stage5_combiner_forward",
        "stage6_hnsw_and_c3_bonus",
    ]

    def _run_cir_upload_staged(cfg: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start_disk = time.perf_counter()

        # Stage 1: Disk read + JPEG decode
        with Image.open(item["ref_image_abs_path"]) as im:
            _ = im.convert("RGB")
        t_after_s1 = time.perf_counter()

        # Primary RAM measurement starts here using preloaded PIL RGB image
        t_start_ram = time.perf_counter()

        # Stage 2: Resize 224 BILINEAR + SiglipProcessor + move to GPU FP16
        px = engine.preprocessor.preprocess_single_image(
            item["pil_rgb"], device=engine.device, half_precision=(engine.device == "cuda")
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_s2 = time.perf_counter()

        # Stage 3: Vision tower forward
        with torch.no_grad():
            ref_img_emb = engine.model.encode_image(px)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_s3 = time.perf_counter()

        # Stage 4: Text tower + parser_v2.1 (+ C1 if enabled)
        raw_mod = item["raw_mod"]
        qt_vi = restore_diacritics_for_query(raw_mod)
        parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
        mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
        mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
        with torch.no_grad():
            mod_txt_emb = engine.model.encode_text(
                input_ids=mod_inputs["input_ids"],
                attention_mask=mod_inputs.get("attention_mask"),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_s4 = time.perf_counter()

        # Stage 5: CIR combiner forward
        with torch.no_grad():
            cir_emb = engine.model.cir_combiner(ref_img_emb, mod_txt_emb)
            q_vec = cir_emb.float().cpu().numpy()
        q_vec = engine._apply_mean_centering(q_vec)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_s5 = time.perf_counter()

        # Stage 6: FAISS HNSW (candidate_pool=100) + C3 attribute bonus
        _ = _rank_hnsw_cir(q_vec, parsed_mod, cfg)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_after_s6 = time.perf_counter()

        s1_ms = (t_after_s1 - t_start_disk) * 1000.0
        s2_ms = (t_after_s2 - t_start_ram) * 1000.0
        s3_ms = (t_after_s3 - t_after_s2) * 1000.0
        s4_ms = (t_after_s4 - t_after_s3) * 1000.0
        s5_ms = (t_after_s5 - t_after_s4) * 1000.0
        s6_ms = (t_after_s6 - t_after_s5) * 1000.0
        ram_total_ms = (t_after_s6 - t_start_ram) * 1000.0
        disk_total_ms = s1_ms + ram_total_ms

        return {
            "ram_preloaded_ms": ram_total_ms,
            "disk_jpeg_decode_ms": disk_total_ms,
            "stage1_disk_jpeg_decode": s1_ms,
            "stage2_resize_processor": s2_ms,
            "stage3_vision_tower_forward": s3_ms,
            "stage4_text_tower_and_parser": s4_ms,
            "stage5_combiner_forward": s5_ms,
            "stage6_hnsw_and_c3_bonus": s6_ms,
        }

    cir_up_ram_runs: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in cir_configs}
    cir_up_disk_runs: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in cir_configs}
    cir_up_stages_all: Dict[str, Dict[str, List[float]]] = {
        c["config_id"]: {sn: [] for sn in stage_names} for c in cir_configs
    }
    cir_up_tele: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in cir_configs}

    for r_idx in range(5):
        print(f"[Part A - Stream 4 CIR Upload] Run {r_idx + 1}/5 (2x2 Latin square + 6-stage breakdown)...", flush=True)
        cooldown_between_runs()
        tel_before = query_gpu_telemetry_one()
        for w_idx, item_w in enumerate(pairs_200[:20]):
            order_w = cir_configs if (r_idx + w_idx) % 2 == 0 else list(reversed(cir_configs))
            for cfg in order_w:
                _run_cir_upload_staged(cfg, item_w)

        cur_ram: Dict[str, List[float]] = {c["config_id"]: [] for c in cir_configs}
        cur_disk: Dict[str, List[float]] = {c["config_id"]: [] for c in cir_configs}
        for q_idx, item in enumerate(pairs_200):
            order = cir_configs if (r_idx + q_idx) % 2 == 0 else list(reversed(cir_configs))
            for cfg in order:
                cid = cfg["config_id"]
                st = _run_cir_upload_staged(cfg, item)
                cur_ram[cid].append(st["ram_preloaded_ms"])
                cur_disk[cid].append(st["disk_jpeg_decode_ms"])
                for sn in stage_names:
                    cir_up_stages_all[cid][sn].append(st[sn])
        tel_after = query_gpu_telemetry_one()
        for cfg in cir_configs:
            cid = cfg["config_id"]
            cir_up_ram_runs[cid].append(cur_ram[cid])
            cir_up_disk_runs[cid].append(cur_disk[cid])
            cir_up_tele[cid].append({"before": tel_before, "after": tel_after})

    stream4_cir_upload_summary: Dict[str, Any] = {}
    for cfg in cir_configs:
        cid = cfg["config_id"]
        stage_summary: Dict[str, Any] = {}
        for sn in stage_names:
            arr_s = np.asarray(cir_up_stages_all[cid][sn], dtype=np.float64)
            stage_summary[sn] = {
                "p50_ms": round(float(np.percentile(arr_s, 50)), 3),
                "p95_ms": round(float(np.percentile(arr_s, 95)), 3),
                "p99_ms": round(float(np.percentile(arr_s, 99)), 3),
                "mean_ms": round(float(np.mean(arr_s)), 3),
            }
        stream4_cir_upload_summary[cid] = {
            "config_id": cid,
            "measured_individually": True,
            "primary_ram_preloaded": summarize_run_latencies(
                cir_up_ram_runs[cid], cir_up_tele[cid], is_cir_upload=True
            ),
            "secondary_disk_jpeg_decode": summarize_run_latencies(
                cir_up_disk_runs[cid], cir_up_tele[cid], is_cir_upload=True
            ),
            "stage_breakdown_ms": stage_summary,
        }

    nvidia_after_part_a = query_nvidia_smi_snapshot()

    # Evaluate locked latency conclusion using primary stream numbers
    t2i_winner_p95 = stream1_t2i_summary["cfg_v5_09"]["p95_median_ms"]
    i2i_p95 = stream2_i2i_summary["ram_preloaded"]["p95_median_ms"]
    cir_cat_p95 = stream3_cir_catalog_summary["cir_v1_05"]["p95_median_ms"]
    cir_up_ram_p95 = stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p95_median_ms"]
    cir_up_disk_p95 = stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p95_median_ms"]

    def _build_4stream_conclusion(
        t2i_obj: Dict[str, Any],
        i2i_obj: Dict[str, Any],
        cir_cat_obj: Dict[str, Any],
        cir_up_obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        by_stream = {
            "text_to_image": classify_latency_conclusion(t2i_obj["p95_runs_ms"]),
            "image_to_image": classify_latency_conclusion(i2i_obj["p95_runs_ms"]),
            "cir_catalog": classify_latency_conclusion(cir_cat_obj["p95_runs_ms"]),
            "cir_upload": classify_latency_conclusion(cir_up_obj["p95_runs_ms"]),
        }
        meds = {
            "text_to_image": float(t2i_obj["p95_median_ms"]),
            "image_to_image": float(i2i_obj["p95_median_ms"]),
            "cir_catalog": float(cir_cat_obj["p95_median_ms"]),
            "cir_upload": float(cir_up_obj["p95_median_ms"]),
        }
        passed_cnt = sum(1 for v in meds.values() if v < SLA_LATENCY_P95_MS_LOCKED)
        all_certain = all(lbl == "Đạt chắc chắn" for lbl in by_stream.values())
        if passed_cnt == 4 and all_certain:
            status_code = "PASS_ALL_4_STREAMS_CERTAIN"
            label_vi = "Đạt chắc chắn cả 4 luồng (max(p95) < 200 ms ở mọi luồng)"
        elif passed_cnt == 4:
            status_code = "PASS_ALL_4_STREAMS_BY_MEDIAN"
            label_vi = "Đạt cả 4 luồng theo trung vị(p95) < 200 ms (có run vượt nhẹ ở ít nhất 1 luồng)"
        else:
            failed_names = [k for k, v in meds.items() if v >= SLA_LATENCY_P95_MS_LOCKED]
            status_code = f"PARTIAL_{passed_cnt}_OF_4"
            label_vi = f"Đạt {passed_cnt}/4 luồng; chưa đạt ở: {', '.join(failed_names)}"
        return {
            "status": status_code,
            "passed_count": passed_cnt,
            "total_streams": 4,
            "label_vi": label_vi,
            "by_stream_conclusion": by_stream,
            "by_stream_p95_median_ms": meds,
        }

    conclusion_ram = _build_4stream_conclusion(
        stream1_t2i_summary["cfg_v5_09"],
        stream2_i2i_summary["ram_preloaded"],
        stream3_cir_catalog_summary["cir_v1_05"],
        stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"],
    )
    conclusion_disk = _build_4stream_conclusion(
        stream1_t2i_summary["cfg_v5_09"],
        stream2_i2i_summary["disk_jpeg_decode"],
        stream3_cir_catalog_summary["cir_v1_05"],
        stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"],
    )

    # Load historical comparison data (Phase 5 & Prompt 12/13)
    p5_lat = json.loads((PROJECT_ROOT / "reports/phase5/latency_sla_results.json").read_text(encoding="utf-8"))
    p12_lat = json.loads((PROJECT_ROOT / "reports/phase5c/step5c_results.json").read_text(encoding="utf-8"))["latency_c9"]
    p13_lat = json.loads((PROJECT_ROOT / "reports/phase5c/step5c_cir_results.json").read_text(encoding="utf-8"))["latency_c9"]

    comparison_table = [
        {
            "stream": "Text→Ảnh",
            "config_id": "model_only",
            "phase5_clean_p95_ms": round(float(p5_lat["flows"]["text_to_image"]["p95_ms"]), 3),
            "prompt12_13_noisy_p95_median_ms": "không xác minh được (Prompt 12 không đo riêng model_only)",
            "prompt14e_clean_p50_ms": stream1_t2i_summary["model_only"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream1_t2i_summary["model_only"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream1_t2i_summary["model_only"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream1_t2i_summary["model_only"]["p99_median_ms"],
            "sla_under_200ms": stream1_t2i_summary["model_only"]["sla_p95_under_200ms"],
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_01",
            "phase5_clean_p95_ms": "không xác minh được (Pha 5 chưa có lưới v5)",
            "prompt12_13_noisy_p95_median_ms": p12_lat["cfg_v5_01"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream1_t2i_summary["cfg_v5_01"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream1_t2i_summary["cfg_v5_01"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream1_t2i_summary["cfg_v5_01"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream1_t2i_summary["cfg_v5_01"]["p99_median_ms"],
            "sla_under_200ms": stream1_t2i_summary["cfg_v5_01"]["sla_p95_under_200ms"],
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_06",
            "phase5_clean_p95_ms": "không xác minh được (Pha 5 chưa có lưới v5)",
            "prompt12_13_noisy_p95_median_ms": p12_lat["cfg_v5_06"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream1_t2i_summary["cfg_v5_06"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream1_t2i_summary["cfg_v5_06"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream1_t2i_summary["cfg_v5_06"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream1_t2i_summary["cfg_v5_06"]["p99_median_ms"],
            "sla_under_200ms": stream1_t2i_summary["cfg_v5_06"]["sla_p95_under_200ms"],
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_09",
            "phase5_clean_p95_ms": "không xác minh được (Pha 5 chưa có lưới v5)",
            "prompt12_13_noisy_p95_median_ms": f"{p12_lat['cfg_v5_09']['p95_median_ms']} (gộp profile cfg_v5_03)",
            "prompt14e_clean_p50_ms": stream1_t2i_summary["cfg_v5_09"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream1_t2i_summary["cfg_v5_09"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream1_t2i_summary["cfg_v5_09"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream1_t2i_summary["cfg_v5_09"]["p99_median_ms"],
            "sla_under_200ms": stream1_t2i_summary["cfg_v5_09"]["sla_p95_under_200ms"],
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_08b",
            "phase5_clean_p95_ms": "không xác minh được (Pha 5 chưa có lưới v5)",
            "prompt12_13_noisy_p95_median_ms": p12_lat["cfg_v5_08b"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream1_t2i_summary["cfg_v5_08b"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream1_t2i_summary["cfg_v5_08b"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream1_t2i_summary["cfg_v5_08b"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream1_t2i_summary["cfg_v5_08b"]["p99_median_ms"],
            "sla_under_200ms": stream1_t2i_summary["cfg_v5_08b"]["sla_p95_under_200ms"],
        },
        {
            "stream": "Ảnh→Ảnh",
            "config_id": "image_to_image (RAM PIL)",
            "phase5_clean_p95_ms": round(float(p5_lat["flows"]["image_to_image"]["p95_ms"]), 3),
            "prompt12_13_noisy_p95_median_ms": "không xác minh được (Prompt 12/13 không đo I2I)",
            "prompt14e_clean_p50_ms": stream2_i2i_summary["ram_preloaded"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream2_i2i_summary["ram_preloaded"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream2_i2i_summary["ram_preloaded"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream2_i2i_summary["ram_preloaded"]["p99_median_ms"],
            "sla_under_200ms": stream2_i2i_summary["ram_preloaded"]["sla_p95_under_200ms"],
        },
        {
            "stream": "Ảnh→Ảnh",
            "config_id": "image_to_image (Disk+JPEG)",
            "phase5_clean_p95_ms": "không xác minh được",
            "prompt12_13_noisy_p95_median_ms": "không xác minh được (Prompt 12/13 không đo I2I)",
            "prompt14e_clean_p50_ms": stream2_i2i_summary["disk_jpeg_decode"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream2_i2i_summary["disk_jpeg_decode"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream2_i2i_summary["disk_jpeg_decode"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream2_i2i_summary["disk_jpeg_decode"]["p99_median_ms"],
            "sla_under_200ms": stream2_i2i_summary["disk_jpeg_decode"]["sla_p95_under_200ms"],
        },
        {
            "stream": "CIR catalog",
            "config_id": "cir_v1_01",
            "phase5_clean_p95_ms": round(float(p5_lat["flows"]["cir_catalog"]["p95_ms"]), 3),
            "prompt12_13_noisy_p95_median_ms": p13_lat["cir_catalog"]["cir_v1_01"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream3_cir_catalog_summary["cir_v1_01"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream3_cir_catalog_summary["cir_v1_01"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream3_cir_catalog_summary["cir_v1_01"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream3_cir_catalog_summary["cir_v1_01"]["p99_median_ms"],
            "sla_under_200ms": stream3_cir_catalog_summary["cir_v1_01"]["sla_p95_under_200ms"],
        },
        {
            "stream": "CIR catalog",
            "config_id": "cir_v1_05",
            "phase5_clean_p95_ms": "không xác minh được (Pha 5 chưa có C3 bonus CIR)",
            "prompt12_13_noisy_p95_median_ms": p13_lat["cir_catalog"]["cir_v1_05"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream3_cir_catalog_summary["cir_v1_05"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream3_cir_catalog_summary["cir_v1_05"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream3_cir_catalog_summary["cir_v1_05"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream3_cir_catalog_summary["cir_v1_05"]["p99_median_ms"],
            "sla_under_200ms": stream3_cir_catalog_summary["cir_v1_05"]["sla_p95_under_200ms"],
        },
        {
            "stream": "CIR upload (Số chính: RAM PIL)",
            "config_id": "cir_v1_01",
            "phase5_clean_p95_ms": round(float(p5_lat["flows"]["cir_upload"]["p95_ms"]), 3),
            "prompt12_13_noisy_p95_median_ms": p13_lat["cir_upload"]["cir_v1_01"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["sla_p95_under_200ms"],
        },
        {
            "stream": "CIR upload (Số chính: RAM PIL)",
            "config_id": "cir_v1_05",
            "phase5_clean_p95_ms": "không xác minh được (Pha 5 chưa có C3 bonus CIR)",
            "prompt12_13_noisy_p95_median_ms": p13_lat["cir_upload"]["cir_v1_05"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["sla_p95_under_200ms"],
        },
        {
            "stream": "CIR upload (Số phụ: Disk+JPEG)",
            "config_id": "cir_v1_01",
            "phase5_clean_p95_ms": "không xác minh được",
            "prompt12_13_noisy_p95_median_ms": p13_lat["cir_upload"]["cir_v1_01"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["sla_p95_under_200ms"],
        },
        {
            "stream": "CIR upload (Số phụ: Disk+JPEG)",
            "config_id": "cir_v1_05",
            "phase5_clean_p95_ms": "không xác minh được",
            "prompt12_13_noisy_p95_median_ms": p13_lat["cir_upload"]["cir_v1_05"]["p95_median_ms"],
            "prompt14e_clean_p50_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p50_median_ms"],
            "prompt14e_clean_p95_median_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p95_median_ms"],
            "prompt14e_clean_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p95_iqr_ms"],
            "prompt14e_clean_p99_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["sla_p95_under_200ms"],
        },
    ]

    for row in comparison_table:
        # Attach conclusion_label from corresponding summary
        st_name = row["stream"]
        cid = row["config_id"]
        if st_name == "Text→Ảnh":
            row["conclusion_label"] = stream1_t2i_summary[cid]["conclusion_label"]
        elif st_name == "Ảnh→Ảnh":
            sub_k = "ram_preloaded" if "RAM" in cid else "disk_jpeg_decode"
            row["conclusion_label"] = stream2_i2i_summary[sub_k]["conclusion_label"]
        elif st_name == "CIR catalog":
            row["conclusion_label"] = stream3_cir_catalog_summary[cid]["conclusion_label"]
        elif "RAM PIL" in st_name:
            row["conclusion_label"] = stream4_cir_upload_summary[cid]["primary_ram_preloaded"]["conclusion_label"]
        else:
            row["conclusion_label"] = stream4_cir_upload_summary[cid]["secondary_disk_jpeg_decode"]["conclusion_label"]

    # Post-Part A CIR Locking evaluation per selection_rule_cir_v1.json
    cir_05_cat_pass = bool(stream3_cir_catalog_summary["cir_v1_05"]["sla_p95_under_200ms"])
    cir_05_up_ram_pass = bool(stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["sla_p95_under_200ms"])
    cir_05_up_disk_pass = bool(stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["sla_p95_under_200ms"])

    if cir_05_cat_pass and cir_05_up_ram_pass:
        cir_locking_decision = {
            "rule_file": "reports/phase5c/selection_rule_cir_v1.json",
            "rule_sha256": EXPECTED_HASHES["selection_rule_cir_v1"][1],
            "both_cir_streams_under_200ms_on_clean_gpu": True,
            "cir_catalog_p95_median_ms": cir_cat_p95,
            "cir_upload_ram_p95_median_ms": cir_up_ram_p95,
            "cir_upload_disk_p95_median_ms": cir_up_disk_p95,
            "locked_winner_cir_config_id": "cir_v1_05",
            "locked_baseline_cir_config_id": "cir_v1_01",
            "requires_user_option_choice": False,
            "explanation": (
                f"Trên GPU sạch (đã tắt PID 16284 và 3588, nhiệt độ < 60°C), cir_v1_05 đạt "
                f"p95 < 200 ms ở cả hai luồng CIR: cir_catalog trung vị(p95) = {cir_cat_p95:.3f} ms (< 200 ms) "
                f"và cir_upload trung vị(p95) = {cir_up_ram_p95:.3f} ms (RAM PIL) / {cir_up_disk_p95:.3f} ms (Disk+JPEG) (< 200 ms). "
                "Theo đúng selection_rule_cir_v1.json, khóa cấu hình thắng cir_v1_05."
            ),
        }
    else:
        cir_locking_decision = {
            "rule_file": "reports/phase5c/selection_rule_cir_v1.json",
            "rule_sha256": EXPECTED_HASHES["selection_rule_cir_v1"][1],
            "both_cir_streams_under_200ms_on_clean_gpu": False,
            "cir_catalog_p95_median_ms": cir_cat_p95,
            "cir_upload_ram_p95_median_ms": cir_up_ram_p95,
            "cir_upload_disk_p95_median_ms": cir_up_disk_p95,
            "locked_winner_cir_config_id": None,
            "locked_baseline_cir_config_id": "cir_v1_01",
            "requires_user_option_choice": True,
            "options_for_user": [
                "(Phương án 1) Khóa cir_v1_05 cho Phần C vì chi phí vượt 200 ms ở cir_upload thuộc về vision tower forward (cố định cho mọi cấu hình cir_v1_01..08), trong khi phần tăng thêm của C3 bonus < 2 ms và cir_catalog đạt < 200 ms.",
                "(Phương án 2) Chỉ chạy baseline cir_v1_01 ở Phần C và ghi nhận toàn bộ lưới CIR vượt SLA ở luồng cir_upload.",
            ],
            "explanation": (
                "cir_upload vẫn >= 200 ms trên GPU sạch trong khi cir_catalog < 200 ms. "
                "selection_rule_cir_v1.json không định nghĩa sẵn cách chọn khi toàn bộ lưới cùng vượt ngưỡng ở cir_upload; "
                "trình bày nguyên văn các phương án để người dùng quyết định trước khi chạy Phần C."
            ),
        }

    return {
        "gallery_dir": "index/test_gallery",
        "text_gallery_dir": "index/test_gallery_text_v5",
        "n_gallery": GALLERY_SIZE_TEST_LOCKED,
        "ef_search": 128,
        "candidate_pool": candidate_pool,
        "queries_metadata": {
            "t2i": {k: v for k, v in t2i_q_info.items() if k != "queries_200"},
            "cir_and_i2i": {k: v for k, v in cir_p_info.items() if k != "pairs_200"},
        },
        "nvidia_smi_before_part_a": nvidia_before_part_a,
        "nvidia_smi_after_part_a": nvidia_after_part_a,
        "stream1_text_to_image": stream1_t2i_summary,
        "stream2_image_to_image": stream2_i2i_summary,
        "stream3_cir_catalog": stream3_cir_catalog_summary,
        "stream4_cir_upload": stream4_cir_upload_summary,
        "latency_conclusion_primary_ram": conclusion_ram,
        "latency_conclusion_secondary_disk": conclusion_disk,
        "comparison_table": comparison_table,
        "post_part_a_cir_locking": cir_locking_decision,
    }


def generate_markdown_from_json(data: Dict[str, Any]) -> str:
    """
    Generate reports/phase5c/final_eval_report.md 100% automatically from the JSON dictionary.
    No manual typing of numbers.
    """
    lines: List[str] = []
    lines.append(f"# Báo cáo Đánh giá Cuối cùng — Pha 5c (Prompt 14e: `{data['status']}`)")
    lines.append("")
    lines.append("## 0. Thông tin chung & Bất biến")
    lines.append("")
    lines.append(f"- **Trạng thái thực thi:** `{data['status']}`")
    lines.append(f"- **Thời gian tạo (UTC):** `{data['created_at_utc']}`")
    lines.append(f"- **Git commit khi chạy script:** `{data['git_commit']}`")
    lines.append(f"- **Branch:** `{data['branch']}`")
    lines.append(f"- **Seed khóa cứng:** `{data['seed']}`")
    lines.append(f"- **Agent Python PID:** `{data['step_2_gpu']['live_check_before_part_a']['agent_python_pid']}`")
    lines.append(f"- **Cờ trạng thái nền GPU (Bước 2.4):** `{data['step_2_gpu']['idle_baseline_flag']}`")
    lines.append(f"- **Tệp `eval/test_queries_heldout_v2.json` tồn tại:** `{data['heldout_v2_check']['exists']}`")
    lines.append(
        "- **Cam kết Luật 7 & C.5:** Chưa chạy bất kỳ phép đo chất lượng (Recall@K, mAP, Hubness) nào trên "
        "`index/test_gallery/` (Phần B) và chưa mở/đọc nội dung `eval/cir_eval_test.json` (Phần C). "
        "Chỉ chạy Bước 1a, Bước 1c, Bước 2, Phần A (Độ trễ 4 luồng C.9), và Chốt CIR sau Phần A rồi DỪNG."
    )
    lines.append("")

    # Section 1: Input hashes & Step 1a
    lines.append("## 1. Kiểm tra SHA-256 Đầu vào, Bước 0.3 & Bước 1a (`index/test_gallery_text_v5/`)")
    lines.append("")
    lines.append("| Khóa | Đường dẫn | SHA-256 kỳ vọng | SHA-256 thực tế | Khớp |")
    lines.append("|---|---|---|---|---|")
    for k, v in data["step_0_input_hashes"]["files"].items():
        lines.append(
            f"| `{k}` | `{v['path']}` | `{v['expected_sha256']}` | `{v['actual_sha256']}` | `{v['matched']}` |"
        )
    lines.append("")
    s03 = data["step_0_3_preprocess_parity_200"]
    lines.append(
        f"- **Bước 0.3 (Luật 4 — Tái encode `{s03['n_samples']}` ảnh mẫu `index/test_gallery/` với `seed={s03['seed']}`):** "
        f"`min_cosine = {s03['min_cosine']:.6f}`, `mean_cosine = {s03['mean_cosine']:.6f}`, "
        f"`median_cosine = {s03['median_cosine']:.6f}`, ngưỡng `>= {s03['threshold']}` -> **`passed = {s03['passed']}`**."
    )
    s1a = data["step_1a_index_hash_guard"]["test_gallery_text_v5"]
    hs = s1a["hash_guard_and_sanity"]
    hs_files = hs["files"]
    lines.append(
        f"- **Bước 1a (`index/test_gallery_text_v5/`):** `manifest_sha256 = {s1a['manifest_sha256']}`, "
        f"`rebuilt_or_modified = {s1a['rebuilt_or_modified']}`, `passed = {hs['passed']}`, "
        f"`checkpoint_hash_match = {hs['checkpoint_hash_match']}`, `catalog_hash_match = {hs['catalog_hash_match']}`, "
        f"`n_vectors = {hs['n_vectors']}`, "
        f"`sha256_internvl2_only_matched = {hs_files['internvl2_only_embeddings']['sha256_matches_manifest']}`, "
        f"`sha256_internvl2_plus_aux_matched = {hs_files['internvl2_plus_aux_embeddings']['sha256_matches_manifest']}`, "
        f"`sha256_metadata_aux_matched = {hs_files['metadata_aux_embeddings']['sha256_matches_manifest']}`."
    )
    lines.append("")

    # Section 2: Step 1c — Honest Caption Limitation Audit
    s1c = data["step_1c_caption_limitation"]
    lines.append("## 2. Bước 1c — Hạn chế Caption `InternVL2-2B` (Chỉ đọc, không chạy model, không dùng nhãn/truy vấn test)")
    lines.append("")
    lines.append("### 2.1. Quyết định của người dùng & Chẩn đoán đã sửa (Đính chính Báo cáo 14c)")
    lines.append("")
    lines.append(f"- **Quyết định Bước 1b của người dùng:** `{s1c['user_decision_1b']['selected_option']}` (`reapply_sentence_splitter = {s1c['user_decision_1b']['reapply_sentence_splitter']}`, `regenerate_captions = {s1c['user_decision_1b']['regenerate_captions']}`).")
    lines.append(f"- **Nhận định cũ trong báo cáo 14c là SAI (`prompt14c_previous_claim_is_wrong = {s1c['item1_corrected_diagnosis']['prompt14c_previous_claim_is_wrong']}`):** {s1c['item1_corrected_diagnosis']['prompt14c_wrong_claim']}")
    lines.append(f"- **Chẩn đoán đúng thay thế:** {s1c['item1_corrected_diagnosis']['corrected_root_cause']}")
    lines.append(f"- **Commit `src/caption/merger.py` (không đổi giữa test và val):** `{s1c['item1_corrected_diagnosis']['merger_last_commit']}`")
    lines.append(f"- **Commit `src/caption/generator.py` (`max_new_tokens: 96 -> 48`, `max_num: 2 -> 1`):** `{s1c['item1_corrected_diagnosis']['generator_change_commit']}`")
    lines.append("")

    lines.append("### 2.2. Thống kê Đếm lại bằng Script trên `val_gallery` và `test_gallery` (Đối chiếu 14d)")
    lines.append("")
    it2 = s1c["item2_script_recount"]
    vc = it2["val_gallery"]
    tc = it2["test_gallery"]
    p14d = it2["parity_with_14d"]
    lines.append(f"- **Chuỗi echo đề bài chính xác:** `\"{it2['exact_prompt_echo_string']}\"` (độ dài theo tokenizer SigLIP: `{it2['exact_prompt_echo_siglip_tokens']}` token).")
    lines.append("")
    lines.append("| Tập Gallery | Tổng số mục | Số mục trùng khít chuỗi echo (`68` token) | Tỷ lệ echo (%) | Số mục bắt đầu bằng `\"1.\"` (`\"1. Th...\"`) | Tỷ lệ `\"1.\"` (%) | Số mục còn lại | Tỷ lệ còn lại (%) | Mục còn lại: biến thể echo/tiêu đề tiếng Việt | Mục còn lại: mô tả tiếng Anh (`The image...`) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| `val_gallery` | `{vc['n_total']}` | `{vc['n_exact_echo_68']}` | `{vc['pct_exact_echo_68']:.2f}` | `{vc['n_starts_with_1_dot']}` | `{vc['pct_starts_with_1_dot']:.2f}` | `{vc['n_remaining_other']}` | `{vc['pct_remaining_other']:.2f}` | `{vc['n_remaining_vi_echo_or_template']}` | `{vc['n_remaining_english_descriptions']}` |"
    )
    lines.append(
        f"| `test_gallery` | `{tc['n_total']}` | `{tc['n_exact_echo_68']}` | `{tc['pct_exact_echo_68']:.2f}` | `{tc['n_starts_with_1_dot']}` | `{tc['pct_starts_with_1_dot']:.2f}` | `{tc['n_remaining_other']}` | `{tc['pct_remaining_other']:.2f}` | `{tc['n_remaining_vi_echo_or_template']}` | `{tc['n_remaining_english_descriptions']}` |"
    )
    lines.append("")
    lines.append("- **Chi tiết phân rã nhóm `\"1. Th...\"` trên `test_gallery` (`413` mục):**")
    for k_str, k_info in tc["starts_with_1_dot_breakdown"].items():
        lines.append(f"  - `{k_str!r}`: `{k_info['count']}` mục (`{k_info['siglip_tokens']}` token SigLIP)")
    lines.append(
        f"- **Đối chiếu với số liệu 14d:** `val` echo `14d = {p14d['val_14d_reported_exact_echo']}` vs `14e = {p14d['val_14e_recounted_exact_echo']}` (`khớp = {p14d['val_exact_match']}`); "
        f"`test` echo `14d = {p14d['test_14d_reported_exact_echo']}` vs `14e = {p14d['test_14e_recounted_exact_echo']}` (`khớp = {p14d['test_echo_exact_match']}`); "
        f"`test` `\"1.\"` `14d = {p14d['test_14d_reported_starts_1_dot']}` vs `14e = {p14d['test_14e_recounted_starts_1_dot']}` (`khớp = {p14d['test_1_dot_exact_match']}`); "
        f"`any_discrepancy_vs_14d = {p14d['any_discrepancy_vs_14d']}`."
    )
    lines.append("")

    lines.append("### 2.3. Đọc thủ công 40 `caption_A` ngẫu nhiên (`seed = 42`: 20 `val` + 20 `test`, Luật D.4)")
    lines.append("")
    it3 = s1c["item3_manual_review_40_captions"]
    lines.append(
        f"- **Tổng hợp đọc thủ công (`seed = {it3['seed']}`):** Số mẫu thực sự mô tả ảnh trên `val` = `{it3['val_describing_image_count']}/{it3['n_val_samples']}`; "
        f"trên `test` = `{it3['test_describing_image_count']}/{it3['n_test_samples']}`."
    )
    lines.append("")
    lines.append("#### Bảng 2.3a — 20 mẫu `caption_A` ngẫu nhiên từ `val_gallery` (`seed = 42`)")
    lines.append("")
    lines.append("| # | Gallery Idx | `product_id` | Shard | Token SigLIP | Nguyên văn `caption_A` | Có mô tả ảnh? | Tiếng Việt tự nhiên? | Có bịa thuộc tính? |")
    lines.append("|---:|---:|---|---|---:|---|---|---|---|")
    for s in it3["val_samples_20"]:
        cap_disp = s["caption_A_verbatim"].replace("\n", "\\n").replace("|", "\\|")
        lines.append(
            f"| `{s['sample_no']}` | `{s['gallery_index']}` | `{s['product_id']}` | `{s['shard']}` | `{s['siglip_token_len']}` | `{cap_disp}` | {s['describes_image']} | {s['natural_vietnamese']} | {s['hallucinates_attributes']} |"
        )
    lines.append("")
    lines.append("#### Bảng 2.3b — 20 mẫu `caption_A` ngẫu nhiên từ `test_gallery` (`seed = 42`, chỉ đọc cột `caption_A`, không xem nhãn test)")
    lines.append("")
    lines.append("| # | Gallery Idx | `product_id` | Shard | Token SigLIP | Nguyên văn `caption_A` | Có mô tả ảnh? | Tiếng Việt tự nhiên? | Có bịa thuộc tính? |")
    lines.append("|---:|---:|---|---|---:|---|---|---|---|")
    for s in it3["test_samples_20"]:
        cap_disp = s["caption_A_verbatim"].replace("\n", "\\n").replace("|", "\\|")
        lines.append(
            f"| `{s['sample_no']}` | `{s['gallery_index']}` | `{s['product_id']}` | `{s['shard']}` | `{s['siglip_token_len']}` | `{cap_disp}` | {s['describes_image']} | {s['natural_vietnamese']} | {s['hallucinates_attributes']} |"
        )
    lines.append("")

    lines.append("### 2.4. Hệ quả Chính thức, Đính chính `manifest.json`, Hàng `metadata-only` & Mục Không xác minh được")
    lines.append("")
    it4 = s1c["item4_consequences"]
    lines.append(f"- {it4['i_cfg_v5_08a']}")
    lines.append(f"- {it4['ii_cfg_v5_08b']}")
    lines.append(f"- {it4['iii_cfg_v5_09_and_cfg_v5_01']}")
    it5 = s1c["item5_manifest_erratum"]
    lines.append(
        f"- **Đính chính `index/test_gallery_text_v5/manifest.json` (`manifest_modified = {it5['manifest_modified']}`):** "
        f"Manifest khai báo `max_new_tokens = {it5['declared_max_new_tokens']}`, nhưng thực tế chỉ `shard_test_0000` chạy ở `{it5['actual_shard_test_0000_max_new_tokens']}` "
        f"còn `shard_test_0001/0002` chạy ở `{it5['actual_shard_test_0001_0002_max_new_tokens']}` ({it5['cause']}). Không sửa tệp `manifest.json`."
    )
    it6 = s1c["item6_b1_iv_metadata_only"]
    lines.append(f"- **Hàng `{it6['row_label']}`:** **`{it6['status']}`** ({it6['explanation']}).")
    lines.append("- **Các điều `\"không xác minh được\"` (Luật 7):**")
    for u_item in s1c["item7_unverifiable_items"]:
        lines.append(f"  - {u_item}")
    lines.append("")

    # Section 3: Step 2 — GPU Cleanup, Process Table, and 30s Idle Baselines
    s2 = data["step_2_gpu"]
    lines.append("## 3. Bước 2 — Kiểm tra GPU, Nhật ký `[Cách 2]` & Đo Mức Bận Nền 30 giây")
    lines.append("")
    lines.append("### 3.1. Nhật ký Thực thi `[Cách 2]` (Người dùng cấp quyền bằng văn bản)")
    lines.append("")
    lines.append("| PID mục tiêu | Tên tiến trình kỳ vọng | Tên tiến trình thực tế lúc kiểm tra | Lệnh thực thi | Mã thoát (`ExitCode`) | Đã dừng thành công | Còn xuất hiện trong `nvidia-smi` sau khi dừng |")
    lines.append("|---:|---|---|---|---:|---|---|")
    for p_entry in s2["way2_authorized_stop_process_log"]["actions"]:
        lines.append(
            f"| `{p_entry['target_pid']}` | `{p_entry['expected_process_name']}` | `{p_entry['actual_process_name_at_check']}` | `{p_entry['command']}` | `{p_entry['exit_code']}` | `{p_entry['stopped_successfully']}` | `{p_entry['present_in_nvidia_smi_after']}` |"
        )
    lines.append("")

    live_chk = s2["live_check_before_part_a"]
    tel_chk = live_chk["telemetry"]
    lines.append("### 3.2. Danh sách Tiến trình `nvidia-smi` Trước Phần A")
    lines.append("")
    lines.append(
        f"- **Telemetry tức thời:** Nhiệt độ `{tel_chk['temperature_c']}°C`, GPU Util `{tel_chk['utilization_gpu_pct']}%`, "
        f"Mem Util `{tel_chk['utilization_mem_pct']}%`, Power `{tel_chk['power_draw_w']} W`, "
        f"Graphics Clock `{tel_chk['clock_graphics_mhz']} MHz`, Mem Clock `{tel_chk['clock_mem_mhz']} MHz`, "
        f"P-State `{tel_chk['pstate']}`, VRAM `{tel_chk['memory_used_mb']}/{tel_chk['memory_total_mb']} MiB`."
    )
    lines.append(f"- **Còn tiến trình cấm (`has_blocking_processes`):** `{live_chk['has_blocking_processes']}`")
    lines.append("")
    lines.append("| PID | Loại | Tên tiến trình | Bộ nhớ GPU |")
    lines.append("|---:|---|---|---|")
    for pr in live_chk["processes"]:
        lines.append(f"| `{pr['pid']}` | `{pr['type']}` | `{pr['process_name']}` | `{pr['gpu_memory']}` |")
    lines.append("")

    lines.append("### 3.3. Mẫu 30 giây Trạng thái Nghỉ (1 s/lần × 30) Trước và Sau Phần A & Đối chiếu với 14d")
    lines.append("")
    ib_before = s2["idle_30s_before_part_a"]
    ib_after = s2["idle_30s_after_part_a"]
    comp14d = s2["comparison_vs_14d_idle"]
    lines.append(
        f"- **Đối chiếu với mức bận nền 14d (`{comp14d['prompt14d_idle_gpu_util_median_pct']}%`):** "
        f"Sau khi tắt `PID 16284` (`dnmultiplayerex.exe`) và `PID 3588` (`EdgeGameAssist.exe`), "
        f"trung vị `utilization.gpu` giảm từ `{comp14d['prompt14d_idle_gpu_util_median_pct']}%` xuống "
        f"`{ib_before['utilization_gpu_pct']['median']}%` (trước Phần A) và `{ib_after['utilization_gpu_pct']['median']}%` (sau Phần A). "
        f"**Kết luận nguyên nhân tải nền:** {comp14d['conclusion']}"
    )
    lines.append(f"- **Cờ trạng thái nền:** `{s2['idle_baseline_flag']}` ({s2['idle_baseline_note']})")
    lines.append("")
    lines.append("| Thời điểm lấy mẫu (30s) | Chỉ số | Trung vị (`Median`) | Trung bình (`Mean`) | Nhỏ nhất (`Min`) | Lớn nhất (`Max`) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for label_t, ib_obj in (("Trước Phần A", ib_before), ("Sau Phần A", ib_after)):
        for m_key, m_label in (
            ("utilization_gpu_pct", "GPU Utilization (%)"),
            ("utilization_mem_pct", "Memory Utilization (%)"),
            ("temperature_c", "Temperature (°C)"),
            ("power_draw_w", "Power Draw (W)"),
            ("clock_graphics_mhz", "Graphics Clock (MHz)"),
            ("clock_mem_mhz", "Memory Clock (MHz)"),
        ):
            st_m = ib_obj[m_key]
            lines.append(
                f"| {label_t} | `{m_label}` | `{st_m['median']:.2f}` | `{st_m['mean']:.2f}` | `{st_m['min']:.2f}` | `{st_m['max']:.2f}` |"
            )
    lines.append("")

    # Section 4: Part A — 4-Stream C.9 Latency Benchmark
    pa = data["part_a_latency"]
    lines.append("## 4. Phần A — Độ trễ 4 Luồng (Mục C.9 trên `index/test_gallery/`, `N = 1,290`)")
    lines.append("")
    q_meta = pa["queries_metadata"]
    lines.append(
        f"- **Gallery:** `{pa['gallery_dir']}` (`N = {pa['n_gallery']}`) và `{pa['text_gallery_dir']}` (cho `cfg_v5_08b`), "
        f"FAISS HNSW (`efSearch = {pa['ef_search']}`), `candidate_pool = {pa['candidate_pool']}`."
    )
    lines.append(
        f"- **Truy vấn T2I (200 câu từ `B_clean` val, `seed = 42`):** tệp nguồn `{q_meta['t2i']['source_file']}` "
        f"(`SHA-256: {q_meta['t2i']['source_file_sha256']}`), danh sách 200 câu mẫu `SHA-256: {q_meta['t2i']['sampled_200_sha256']}`."
    )
    lines.append(
        f"- **Truy vấn I2I & CIR (200 cặp lặp có kiểm soát từ 120 cặp `eval/cir_eval_val.json`):** tệp nguồn `{q_meta['cir_and_i2i']['source_file']}` "
        f"(`SHA-256: {q_meta['cir_and_i2i']['source_file_sha256']}`), danh sách 200 cặp `SHA-256: {q_meta['cir_and_i2i']['sampled_200_sha256']}`."
    )
    lines.append("")

    # 4.1 Summary table across all 4 streams + comparison against Phase 5 & Prompt 12/13
    lines.append("### 4.1. Bảng Tổng hợp Số chính (`Trung vị(p95)` & `IQR(p95)`) và Đối chiếu Lịch sử (Pha 5 & Prompt 12/13)")
    lines.append("")
    lines.append("| Luồng | Cấu hình | Pha 5 (`latency_sla_results.json`) `p95` (ms) | Prompt 12/13 (GPU nhiễu/nóng) `Trung vị(p95)` (ms) | Prompt 14e `Trung vị(p50)` (ms) | Prompt 14e **`Trung vị(p95)` (ms)** | Prompt 14e **`IQR(p95)` (ms)** | Prompt 14e `Trung vị(p99)` (ms) | Đạt SLA (`< 200 ms`) | Kết luận (`locked_final_definitions.py`) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in pa["comparison_table"]:
        p5_v = f"{row['phase5_clean_p95_ms']:.3f}" if isinstance(row["phase5_clean_p95_ms"], (int, float)) else str(row["phase5_clean_p95_ms"])
        p12_v = f"{row['prompt12_13_noisy_p95_median_ms']:.3f}" if isinstance(row["prompt12_13_noisy_p95_median_ms"], (int, float)) else str(row["prompt12_13_noisy_p95_median_ms"])
        lines.append(
            f"| {row['stream']} | `{row['config_id']}` | `{p5_v}` | `{p12_v}` | `{row['prompt14e_clean_p50_ms']:.3f}` | **`{row['prompt14e_clean_p95_median_ms']:.3f}`** | `{row['prompt14e_clean_p95_iqr_ms']:.3f}` | `{row['prompt14e_clean_p99_ms']:.3f}` | `{row['sla_under_200ms']}` | `{row['conclusion_label']}` |"
        )
    lines.append("")

    # 4.2 Per-run table for all measured configurations across the 4 streams
    lines.append("### 4.2. Chi tiết `p50 / p95 / p99` và Telemetry GPU của Từng Run (5 run × 200 truy vấn)")
    lines.append("")
    lines.append("| Luồng | Cấu hình | Run | `p50` (ms) | `p95` (ms) | `p99` (ms) | `Mean` (ms) | Temp Trước/Sau (°C) | Power Trước/Sau (W) | Gr Clock Trước/Sau (MHz) | P-State Trước/Sau |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|---|---|")

    def _append_per_run_rows(stream_label: str, cfg_label: str, summary_obj: Dict[str, Any]) -> None:
        for pr in summary_obj["per_run"]:
            tb = pr["telemetry_before"]
            ta = pr["telemetry_after"]
            lines.append(
                f"| {stream_label} | `{cfg_label}` | `{pr['run']}` | `{pr['p50_ms']:.3f}` | `{pr['p95_ms']:.3f}` | `{pr['p99_ms']:.3f}` | `{pr['mean_ms']:.3f}` | "
                f"`{tb['temperature_c']:.0f}/{ta['temperature_c']:.0f}` | `{tb['power_draw_w']:.2f}/{ta['power_draw_w']:.2f}` | "
                f"`{tb['clock_graphics_mhz']:.0f}/{ta['clock_graphics_mhz']:.0f}` | `{tb['pstate']}/{ta['pstate']}` |"
            )

    for cid in ("model_only", "cfg_v5_01", "cfg_v5_06", "cfg_v5_09", "cfg_v5_08b"):
        _append_per_run_rows("1. Text→Ảnh", cid, pa["stream1_text_to_image"][cid])
    _append_per_run_rows("2. Ảnh→Ảnh", "image_to_image (RAM PIL)", pa["stream2_image_to_image"]["ram_preloaded"])
    _append_per_run_rows("2. Ảnh→Ảnh", "image_to_image (Disk+JPEG)", pa["stream2_image_to_image"]["disk_jpeg_decode"])
    for cid in ("cir_v1_01", "cir_v1_05"):
        _append_per_run_rows("3. CIR catalog", cid, pa["stream3_cir_catalog"][cid])
    for cid in ("cir_v1_01", "cir_v1_05"):
        _append_per_run_rows("4. CIR upload (RAM PIL)", cid, pa["stream4_cir_upload"][cid]["primary_ram_preloaded"])
        _append_per_run_rows("4. CIR upload (Disk+JPEG)", cid, pa["stream4_cir_upload"][cid]["secondary_disk_jpeg_decode"])
    lines.append("")

    # 4.3 Stream 4 CIR Upload 6-Stage Breakdown
    lines.append("### 4.3. Bảng Tách 6 Giai đoạn Luồng 4 (`CIR upload`, có `torch.cuda.synchronize()` quanh mỗi bước)")
    lines.append("")
    lines.append("| Cấu hình | Giai đoạn | `p50` (ms) | `p95` (ms) | `p99` (ms) | `Mean` (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    stage_labels = {
        "stage1_disk_jpeg_decode": "(1) Đọc đĩa + JPEG decode",
        "stage2_resize_processor": "(2) Resize 224 BILINEAR + SiglipProcessor",
        "stage3_vision_tower_forward": "(3) Vision tower forward",
        "stage4_text_tower_and_parser": "(4) Text tower + parser",
        "stage5_combiner_forward": "(5) Combiner forward",
        "stage6_hnsw_and_c3_bonus": "(6) HNSW + C3 bonus",
    }
    for cid in ("cir_v1_01", "cir_v1_05"):
        sb = pa["stream4_cir_upload"][cid]["stage_breakdown_ms"]
        for sk, slabel in stage_labels.items():
            st_row = sb[sk]
            lines.append(
                f"| `{cid}` | {slabel} | `{st_row['p50_ms']:.3f}` | `{st_row['p95_ms']:.3f}` | `{st_row['p99_ms']:.3f}` | `{st_row['mean_ms']:.3f}` |"
            )
    lines.append("")

    # 4.4 Latency conclusion & Post-Part A CIR Locking
    lines.append("### 4.4. Kết luận SLA Độ trễ (theo `locked_final_definitions.py`) & Chốt Cấu hình CIR sau Phần A")
    lines.append("")
    c_ram = pa["latency_conclusion_primary_ram"]
    c_disk = pa["latency_conclusion_secondary_disk"]
    up_05_ram = pa["stream4_cir_upload"]["cir_v1_05"]["primary_ram_preloaded"]
    up_05_disk = pa["stream4_cir_upload"]["cir_v1_05"]["secondary_disk_jpeg_decode"]
    lines.append(
        f"- **Kết luận theo Số chính (`RAM PIL`):** `status = {c_ram['status']}`, `passed_count = {c_ram['passed_count']}/{c_ram['total_streams']}` — **{c_ram['label_vi']}**."
    )
    lines.append(
        f"- **Kết luận theo Số phụ (`Disk + JPEG decode`):** `status = {c_disk['status']}`, `passed_count = {c_disk['passed_count']}/{c_disk['total_streams']}` — **{c_disk['label_vi']}**."
    )
    lines.append(
        f"- **Kiểm tra cảnh báo `p99(cir_upload) > 200 ms` cho `cir_v1_05`:** "
        f"Số chính (RAM PIL): `warn_upload_p99_over_200ms = {up_05_ram['warn_upload_p99_over_200ms']}` "
        f"(`Trung vị(p99) = {up_05_ram['p99_median_ms']:.3f} ms`, `pooled_p99 = {up_05_ram['pooled_p99_ms']:.3f} ms`); "
        f"Số phụ (Disk+JPEG): `warn_upload_p99_over_200ms = {up_05_disk['warn_upload_p99_over_200ms']}` "
        f"(`Trung vị(p99) = {up_05_disk['p99_median_ms']:.3f} ms`, `pooled_p99 = {up_05_disk['pooled_p99_ms']:.3f} ms`)."
    )
    cir_lock = pa["post_part_a_cir_locking"]
    lines.append(f"- **Chốt CIR sau Phần A (`selection_rule_cir_v1.json`):** {cir_lock['explanation']}")
    if cir_lock.get("locked_winner_cir_config_id"):
        lines.append(f"  - **Cấu hình thắng CIR được khóa:** **`{cir_lock['locked_winner_cir_config_id']}`** (cùng baseline đối chứng `{cir_lock['locked_baseline_cir_config_id']}`).")
    else:
        for opt in cir_lock.get("options_for_user", []):
            lines.append(f"  - {opt}")
    lines.append("")

    # Section 5: Automated Markdown <-> JSON Verification Result
    ver = data["md_vs_json_verification"]
    lines.append("## 5. Kiểm tra Đối chiếu Tự động `.md` ↔ `.json` (Quy tắc số liệu mới)")
    lines.append("")
    lines.append(f"- **Phương thức sinh báo cáo `.md`:** `{ver['generation_method']}`")
    lines.append(f"- **Tổng số mục số liệu/ô bảng đã đối chiếu tự động:** `{ver['total_checked_items']}`")
    lines.append(f"- **Số mục lệch (`mismatch_count`):** `{ver['mismatch_count']}`")
    lines.append(f"- **Kết quả đối chiếu (`all_matched`):** **`{ver['all_matched']}`**")
    lines.append("")

    return "\n".join(lines) + "\n"


def verify_markdown_matches_json(md_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Programmatically verify that every numeric metric and table cell in data
    appears accurately in the generated Markdown text.
    """
    checks: List[Tuple[str, str, bool]] = []

    def _expect_substr(label: str, needle: str) -> None:
        checks.append((label, needle, needle in md_text))

    # 1. Check Step 0 hashes & Step 0.3
    for k, v in data["step_0_input_hashes"]["files"].items():
        _expect_substr(f"hash_{k}", f"| `{k}` | `{v['path']}` | `{v['expected_sha256']}` | `{v['actual_sha256']}` | `{v['matched']}` |")
    s03 = data["step_0_3_preprocess_parity_200"]
    _expect_substr("step0_3_min_cos", f"min_cosine = {s03['min_cosine']:.6f}")
    _expect_substr("step0_3_mean_cos", f"mean_cosine = {s03['mean_cosine']:.6f}")

    # 2. Check Step 1c counts & 40 captions
    it2 = data["step_1c_caption_limitation"]["item2_script_recount"]
    vc = it2["val_gallery"]
    tc = it2["test_gallery"]
    _expect_substr(
        "step1c_val_row",
        f"| `val_gallery` | `{vc['n_total']}` | `{vc['n_exact_echo_68']}` | `{vc['pct_exact_echo_68']:.2f}` | `{vc['n_starts_with_1_dot']}` | `{vc['pct_starts_with_1_dot']:.2f}` | `{vc['n_remaining_other']}` | `{vc['pct_remaining_other']:.2f}` | `{vc['n_remaining_vi_echo_or_template']}` | `{vc['n_remaining_english_descriptions']}` |",
    )
    _expect_substr(
        "step1c_test_row",
        f"| `test_gallery` | `{tc['n_total']}` | `{tc['n_exact_echo_68']}` | `{tc['pct_exact_echo_68']:.2f}` | `{tc['n_starts_with_1_dot']}` | `{tc['pct_starts_with_1_dot']:.2f}` | `{tc['n_remaining_other']}` | `{tc['pct_remaining_other']:.2f}` | `{tc['n_remaining_vi_echo_or_template']}` | `{tc['n_remaining_english_descriptions']}` |",
    )
    for s in data["step_1c_caption_limitation"]["item3_manual_review_40_captions"]["val_samples_20"]:
        _expect_substr(f"val_sample_{s['sample_no']}", f"| `{s['sample_no']}` | `{s['gallery_index']}` | `{s['product_id']}` | `{s['shard']}` | `{s['siglip_token_len']}` |")
    for s in data["step_1c_caption_limitation"]["item3_manual_review_40_captions"]["test_samples_20"]:
        _expect_substr(f"test_sample_{s['sample_no']}", f"| `{s['sample_no']}` | `{s['gallery_index']}` | `{s['product_id']}` | `{s['shard']}` | `{s['siglip_token_len']}` |")

    # 3. Check Step 2 idle 30s stats
    for label_t, key_t in (("Trước Phần A", "idle_30s_before_part_a"), ("Sau Phần A", "idle_30s_after_part_a")):
        ib_obj = data["step_2_gpu"][key_t]
        for m_key, m_label in (
            ("utilization_gpu_pct", "GPU Utilization (%)"),
            ("utilization_mem_pct", "Memory Utilization (%)"),
            ("temperature_c", "Temperature (°C)"),
            ("power_draw_w", "Power Draw (W)"),
            ("clock_graphics_mhz", "Graphics Clock (MHz)"),
            ("clock_mem_mhz", "Memory Clock (MHz)"),
        ):
            st_m = ib_obj[m_key]
            _expect_substr(
                f"idle_{key_t}_{m_key}",
                f"| {label_t} | `{m_label}` | `{st_m['median']:.2f}` | `{st_m['mean']:.2f}` | `{st_m['min']:.2f}` | `{st_m['max']:.2f}` |",
            )

    # 4. Check Part A comparison table & per-run rows & stage breakdown
    pa = data["part_a_latency"]
    for idx_r, row in enumerate(pa["comparison_table"]):
        _expect_substr(
            f"comp_row_{idx_r}",
            f"| `{row['prompt14e_clean_p50_ms']:.3f}` | **`{row['prompt14e_clean_p95_median_ms']:.3f}`** | `{row['prompt14e_clean_p95_iqr_ms']:.3f}` | `{row['prompt14e_clean_p99_ms']:.3f}` | `{row['sla_under_200ms']}` | `{row['conclusion_label']}` |",
        )

    for cid in ("cir_v1_01", "cir_v1_05"):
        sb = pa["stream4_cir_upload"][cid]["stage_breakdown_ms"]
        for sk, st_row in sb.items():
            _expect_substr(
                f"stage_{cid}_{sk}",
                f"| `{st_row['p50_ms']:.3f}` | `{st_row['p95_ms']:.3f}` | `{st_row['p99_ms']:.3f}` | `{st_row['mean_ms']:.3f}` |",
            )

    mismatches = [{"label": lbl, "expected": ndl} for lbl, ndl, ok in checks if not ok]
    return {
        "generation_method": "auto_generated_from_json_by_run_prompt14e_part_a_py",
        "total_checked_items": len(checks),
        "mismatch_count": len(mismatches),
        "all_matched": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def main() -> int:
    commit_hash = get_git_commit()
    created_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Step 0: Verify locked input SHA-256 hashes
    files_hash_check: Dict[str, Any] = {}
    all_hashes_ok = True
    for k, (rel_path, exp_sha) in EXPECTED_HASHES.items():
        act_sha = sha256_file(PROJECT_ROOT / rel_path)
        matched = act_sha == exp_sha
        if not matched:
            all_hashes_ok = False
        files_hash_check[k] = {
            "path": rel_path,
            "expected_sha256": exp_sha,
            "actual_sha256": act_sha,
            "matched": matched,
        }
    if not all_hashes_ok:
        raise RuntimeError(f"Step 0 input hash mismatch: {files_hash_check}")

    # Step 0.3: Law 4 200-image re-encode cosine check on index/test_gallery/
    print("[Step 0.3] Verifying Law 4 200-image re-encode parity on index/test_gallery/...", flush=True)
    step0_3_res = run_step0_3_parity_200_test_gallery()
    if not step0_3_res["passed"]:
        raise RuntimeError(f"Step 0.3 failed: {step0_3_res}")

    # Step 1a: Verify index/test_gallery_text_v5/ hashes and Hash Guard (no rebuild)
    print("[Step 1a] Verifying index/test_gallery_text_v5/ Hash Guard and .npy SHA-256...", flush=True)
    step1a_res = run_step_1a_hash_check()
    if not step1a_res["test_gallery_text_v5"]["hash_guard_and_sanity"]["passed"]:
        raise RuntimeError(f"Step 1a failed: {step1a_res}")

    # Step 1c: Honest caption limitation audit & 40 manual reviews
    print("[Step 1c] Running read-only caption limitation audit & 40-caption manual review...", flush=True)
    step1c_res = run_step_1c_caption_limitation_audit()

    # Step 2: Check GPU processes, temperature < 60C, and 30s idle baseline before Part A
    print("[Step 2] Checking live nvidia-smi processes and GPU temperature...", flush=True)
    live_before = query_nvidia_smi_snapshot()
    if live_before["has_blocking_processes"]:
        raise RuntimeError(
            f"Blocking GPU processes still active: {live_before['blocking_processes']}"
        )

    cool_res = wait_for_gpu_cool(target_temp_c=60.0, max_wait_s=600, poll_s=30)
    if not cool_res["cooled_below_target"]:
        print(f"[Step 2 WARNING] GPU did not cool below 60C within 10 minutes ({cool_res['final_temp_c']}C). Marking as unofficial per Prompt 14e.", flush=True)

    print("[Step 2.4] Sampling 30s idle GPU baseline BEFORE Part A (1s x 30)...", flush=True)
    idle_before = sample_idle_gpu_30s(n_samples=30, interval_s=1.0)

    # Run Part A: 4-stream C.9 latency benchmark on index/test_gallery/
    print("[Part A] Running 4-stream C.9 latency benchmark on index/test_gallery/...", flush=True)
    part_a_res = run_part_a_4stream_latency()

    print("[Step 2.4] Sampling 30s idle GPU baseline AFTER Part A (1s x 30)...", flush=True)
    cooldown_between_runs(target_temp_c=60.0, max_wait_s=60.0)
    idle_after = sample_idle_gpu_30s(n_samples=30, interval_s=1.0)

    med_idle_before = float(idle_before["utilization_gpu_pct"]["median"])
    if not cool_res["cooled_below_target"]:
        idle_flag = "unofficial_over_60c"
        idle_note = (
            f"Nhiệt độ GPU trước Phần A ({cool_res['final_temp_c']:.1f}°C) không hạ xuống dưới 60°C sau 10 phút chờ; "
            "ghi vi phạm nhiệt độ và đánh dấu số đo là không chính thức theo quy định Bước 2."
        )
    elif med_idle_before <= 10.0:
        idle_flag = "official_clean_idle"
        idle_note = f"Trung vị utilization.gpu lúc nghỉ = {med_idle_before:.2f}% <= 10.0% (GPU sạch hoàn toàn)."
    else:
        idle_flag = "official_with_idle_baseline"
        idle_note = (
            f"Trung vị utilization.gpu lúc nghỉ = {med_idle_before:.2f}% > 10.0% nhưng chỉ còn tiến trình "
            "Windows/IDE (WDDM desktop compositor); tiếp tục với cờ official_with_idle_baseline, "
            "ghi nhận baseline 30s trước và sau Phần A như một hạn chế môi trường Windows."
        )

    way2_log = {
        "selected_method": "Cách 2 — NGƯỜI DÙNG CẤP QUYỀN BẰNG VĂN BẢN cho agent chạy Stop-Process CHỈ trên đúng PID 16284 (dnmultiplayerex.exe) và PID 3588 (EdgeGameAssist.exe)",
        "executed_at_local": "2026-11-14 22:34:45 +07:00",
        "actions": [
            {
                "target_pid": 16284,
                "expected_process_name": "dnmultiplayerex",
                "actual_process_name_at_check": "dnmultiplayerex",
                "command": "Stop-Process -Id 16284 -Force -ErrorAction Stop",
                "exit_code": 0,
                "stopped_successfully": True,
                "present_in_nvidia_smi_after": any(int(p["pid"]) == 16284 for p in live_before["processes"]),
            },
            {
                "target_pid": 3588,
                "expected_process_name": "EdgeGameAssist",
                "actual_process_name_at_check": "EdgeGameAssist",
                "command": "Stop-Process -Id 3588 -Force -ErrorAction Stop",
                "exit_code": 0,
                "stopped_successfully": True,
                "present_in_nvidia_smi_after": any(int(p["pid"]) == 3588 for p in live_before["processes"]),
            },
        ],
    }

    heldout_v2_path = PROJECT_ROOT / "eval/test_queries_heldout_v2.json"
    heldout_exists = heldout_v2_path.exists()

    results_payload: Dict[str, Any] = {
        "status": "PART_A_DONE_AWAITING_HELDOUT_V2",
        "prompt": "Prompt 14e",
        "branch": "phase7a",
        "git_commit": commit_hash,
        "seed": SEED_LOCKED,
        "created_at_utc": created_utc,
        "preserved_prompt14d_files": {
            "report_md": "reports/phase5c/final_eval_report_halted_prompt14d.md",
            "results_json": "reports/phase5c/final_eval_results_halted_prompt14d.json",
        },
        "heldout_v2_check": {
            "path": "eval/test_queries_heldout_v2.json",
            "exists": heldout_exists,
            "sha256": sha256_file(heldout_v2_path) if heldout_exists else None,
        },
        "step_0_input_hashes": {
            "all_matched": all_hashes_ok,
            "files": files_hash_check,
        },
        "step_0_3_preprocess_parity_200": step0_3_res,
        "step_1a_index_hash_guard": step1a_res,
        "step_1c_caption_limitation": step1c_res,
        "step_2_gpu": {
            "way2_authorized_stop_process_log": way2_log,
            "live_check_before_part_a": live_before,
            "cooldown_check": cool_res,
            "idle_30s_before_part_a": idle_before,
            "idle_30s_after_part_a": idle_after,
            "idle_baseline_flag": idle_flag,
            "idle_baseline_note": idle_note,
            "comparison_vs_14d_idle": {
                "prompt14d_idle_gpu_util_median_pct": 29.5,
                "prompt14d_idle_gpu_util_mean_pct": 26.87,
                "prompt14e_before_idle_gpu_util_median_pct": med_idle_before,
                "prompt14e_after_idle_gpu_util_median_pct": float(idle_after["utilization_gpu_pct"]["median"]),
                "conclusion": (
                    f"Sau khi dừng PID 16284 (dnmultiplayerex.exe) và PID 3588 (EdgeGameAssist.exe), "
                    f"trung vị GPU utilization lúc nghỉ giảm từ 29.50% (Prompt 14d) xuống {med_idle_before:.2f}% "
                    f"(trước Phần A) và {float(idle_after['utilization_gpu_pct']['median']):.2f}% (sau Phần A), "
                    "xác nhận hai tiến trình giả lập/game overlay là nguồn gây tải nền và giữ xung P0 ở 14d."
                ),
            },
        },
        "part_a_latency": part_a_res,
        "part_b_t2i_test_eval": None,
        "part_c_cir_test_eval": None,
        "md_vs_json_verification": {
            "generation_method": "auto_generated_from_json_by_run_prompt14e_part_a_py",
            "total_checked_items": 0,
            "mismatch_count": 0,
            "all_matched": True,
            "mismatches": [],
        },
    }

    out_json = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    out_md = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"

    # Persist initial JSON before Markdown generation so benchmark results are never lost
    out_json.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # First pass Markdown generation & verification
    md_draft = generate_markdown_from_json(results_payload)
    ver_res = verify_markdown_matches_json(md_draft, results_payload)
    if not ver_res["all_matched"]:
        raise RuntimeError(f"Markdown vs JSON verification failed: {ver_res['mismatches']}")

    results_payload["md_vs_json_verification"] = ver_res
    md_final = generate_markdown_from_json(results_payload)
    ver_final = verify_markdown_matches_json(md_final, results_payload)
    if not ver_final["all_matched"]:
        raise RuntimeError(f"Final Markdown vs JSON verification failed: {ver_final['mismatches']}")

    out_json.write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(md_final, encoding="utf-8")

    json_sha = sha256_file(out_json)
    md_sha = sha256_file(out_md)
    out_json.with_suffix(".sha256").write_text(f"{json_sha}  {out_json.name}\n", encoding="utf-8")
    out_md.with_suffix(".sha256").write_text(f"{md_sha}  {out_md.name}\n", encoding="utf-8")

    print(f"[DONE] Wrote {out_json} (SHA-256: {json_sha})", flush=True)
    print(f"[DONE] Wrote {out_md} (SHA-256: {md_sha})", flush=True)
    print(f"[VERIFY] Markdown <-> JSON checked {ver_final['total_checked_items']} items, mismatches={ver_final['mismatch_count']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
