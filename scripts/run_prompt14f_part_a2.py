"""
Prompt 14f — Part A2: Strict C.9 Re-benchmark, Step 1 CIR User Decision Locking,
Step 2 Erratum 14e, Step 3 Environment Telemetry, Step 5 Phase 5 Diagnostic Parity,
and Automated Verification.

Invariants:
- Branch phase7a, seed 42.
- No training, no modifications to checkpoints, parsers, lexicons, configs, or locked indices.
- Never opens eval/cir_eval_test.json beyond raw hash check.
- Does NOT run Part B or Part C (held-out v2 not locked yet; stops at PART_A2_DONE_AWAITING_HELDOUT_V2).
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
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Clean Windows TLS env vars before importing transformers/huggingface_hub
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import numpy as np
import psutil
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


def query_gpu_temp() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return 999.0


def query_gpu_telemetry_one() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.gr,clocks.mem,pstate,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        return {
            "timestamp": parts[0],
            "gpu_name": parts[1],
            "temperature_c": float(parts[2]),
            "utilization_gpu_pct": float(parts[3]),
            "utilization_mem_pct": float(parts[4]),
            "power_draw_w": float(parts[5]) if parts[5] != "[N/A]" else None,
            "clock_graphics_mhz": float(parts[6]),
            "clock_mem_mhz": float(parts[7]),
            "pstate": parts[8],
            "memory_used_mb": float(parts[9]),
            "memory_total_mb": float(parts[10]),
        }
    except Exception as exc:
        return {"error": str(exc)}


class TelemetrySampler(threading.Thread):
    """
    1 Hz background thread sampling GPU and CPU metrics during a latency run.
    """
    def __init__(self, interval_s: float = 1.0):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.samples: List[Dict[str, Any]] = []

    def run(self):
        while not self.stop_event.is_set():
            t_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:
                smi_out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,utilization.gpu,utilization.memory,clocks.current.graphics,clocks.current.memory,power.draw,pstate,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=3,
                ).strip()
                first = smi_out.splitlines()[0]
                p = [x.strip() for x in first.split(",")]
                gpu_temp = float(p[0])
                gpu_util = float(p[1])
                mem_util = float(p[2])
                clk_gfx = float(p[3])
                clk_mem = float(p[4])
                pwr = float(p[5]) if p[5] != "[N/A]" else 0.0
                pstate = p[6]
                th_active = p[7]
                th_sw_therm = p[9]
                th_hw_therm = p[10]
                th_hw_brake = p[12]

                is_throttled = False
                try:
                    val = int(th_active, 16)
                    if val > 1:
                        is_throttled = True
                except Exception:
                    pass
                if any(x == "Active" for x in (th_sw_therm, th_hw_therm, th_hw_brake)):
                    is_throttled = True

                cpu_u = float(psutil.cpu_percent(interval=None))
                cfreq = psutil.cpu_freq()
                cpu_f = float(cfreq.current) if cfreq else 0.0

                self.samples.append({
                    "timestamp_utc": t_now,
                    "temperature_c": gpu_temp,
                    "gpu_util_pct": gpu_util,
                    "mem_util_pct": mem_util,
                    "clock_graphics_mhz": clk_gfx,
                    "clock_mem_mhz": clk_mem,
                    "power_draw_w": pwr,
                    "pstate": pstate,
                    "throttle_active_hex": th_active,
                    "is_throttled": is_throttled,
                    "cpu_util_pct": cpu_u,
                    "cpu_freq_mhz": cpu_f,
                })
            except Exception:
                pass
            self.stop_event.wait(self.interval_s)

    def stop(self) -> List[Dict[str, Any]]:
        self.stop_event.set()
        self.join(timeout=3)
        return self.samples


def summarize_telemetry_samples(
    samples: List[Dict[str, Any]],
    start_temp: float,
    end_temp: float,
    is_official: bool,
    wait_s: float,
) -> Dict[str, Any]:
    if not samples:
        return {
            "n_samples": 0,
            "is_official": is_official,
            "starting_temp_c": start_temp,
            "ending_temp_c": end_temp,
            "cooldown_wait_s": round(wait_s, 1),
            "pct_throttled": 0.0,
            "temp_min": start_temp,
            "temp_median": start_temp,
            "temp_max": end_temp,
        }

    def _stats(key: str) -> Tuple[float, float, float]:
        vals = [float(s[key]) for s in samples if s.get(key) is not None]
        if not vals:
            return 0.0, 0.0, 0.0
        return float(min(vals)), float(statistics.median(vals)), float(max(vals))

    t_min, t_med, t_max = _stats("temperature_c")
    gu_min, gu_med, gu_max = _stats("gpu_util_pct")
    mu_min, mu_med, mu_max = _stats("mem_util_pct")
    cg_min, cg_med, cg_max = _stats("clock_graphics_mhz")
    cm_min, cm_med, cm_max = _stats("clock_mem_mhz")
    pw_min, pw_med, pw_max = _stats("power_draw_w")
    cu_min, cu_med, cu_max = _stats("cpu_util_pct")
    cf_min, cf_med, cf_max = _stats("cpu_freq_mhz")

    n_throttled = sum(1 for s in samples if s.get("is_throttled"))
    pct_throttled = round(n_throttled / len(samples) * 100.0, 2)

    pstates = [s.get("pstate") for s in samples if s.get("pstate")]
    pstate_mode = Counter(pstates).most_common(1)[0][0] if pstates else "unknown"

    return {
        "n_samples": len(samples),
        "is_official": is_official,
        "cooldown_wait_s": round(wait_s, 1),
        "starting_temp_c": round(start_temp, 1),
        "ending_temp_c": round(end_temp, 1),
        "pct_throttled": pct_throttled,
        "pstate_mode": pstate_mode,
        "temperature_c": {"min": round(t_min, 1), "median": round(t_med, 1), "max": round(t_max, 1)},
        "gpu_util_pct": {"min": round(gu_min, 1), "median": round(gu_med, 1), "max": round(gu_max, 1)},
        "mem_util_pct": {"min": round(mu_min, 1), "median": round(mu_med, 1), "max": round(mu_max, 1)},
        "clock_graphics_mhz": {"min": round(cg_min, 1), "median": round(cg_med, 1), "max": round(cg_max, 1)},
        "clock_mem_mhz": {"min": round(cm_min, 1), "median": round(cm_med, 1), "max": round(cm_max, 1)},
        "power_draw_w": {"min": round(pw_min, 2), "median": round(pw_med, 2), "max": round(pw_max, 2)},
        "cpu_util_pct": {"min": round(cu_min, 1), "median": round(cu_med, 1), "max": round(cu_max, 1)},
        "cpu_freq_mhz": {"min": round(cf_min, 1), "median": round(cf_med, 1), "max": round(cf_max, 1)},
    }


def wait_for_gpu_cooldown(
    target_temp_c: float = 60.0,
    check_interval_s: float = 5.0,
    max_wait_s: float = 600.0,
) -> Tuple[bool, float, float]:
    """
    Polls GPU temp every check_interval_s up to max_wait_s.
    Returns (is_official, start_temp, elapsed_wait_s).
    """
    t_start = time.perf_counter()
    while True:
        cur_t = query_gpu_temp()
        if cur_t < target_temp_c:
            elapsed = time.perf_counter() - t_start
            return True, cur_t, elapsed
        elapsed = time.perf_counter() - t_start
        if elapsed >= max_wait_s:
            print(f"  [Cooldown Timeout] GPU still at {cur_t:.1f}°C >= {target_temp_c}°C after {elapsed:.1f}s. Proceeding as UNOFFICIAL.", flush=True)
            return False, cur_t, elapsed
        time.sleep(check_interval_s)


def build_t2i_200_latency_queries() -> Dict[str, Any]:
    b_synth_path = PROJECT_ROOT / "eval/val_queries_b_synth.json"
    b_synth_sha = sha256_file(b_synth_path)
    b_all = json.loads(b_synth_path.read_text(encoding="utf-8"))
    b_clean = [q for q in b_all if not q.get("is_typo")]
    n_clean = len(b_clean)

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


def build_cir_and_i2i_200_latency_pairs(test_engine: ECommerceSearchEngine) -> Dict[str, Any]:
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


def summarize_stream_runs(
    run_latencies_ms: List[List[float]],
    telemetry_summaries: List[Dict[str, Any]],
    is_cir_upload: bool = False,
) -> Dict[str, Any]:
    """
    Summarize latencies per run, separating official from unofficial runs.
    Conclusion rule is evaluated strictly on official runs.
    """
    per_run_stats: List[Dict[str, Any]] = []
    p95_runs: List[float] = []
    p50_runs: List[float] = []
    p99_runs: List[float] = []
    all_ms: List[float] = []

    p95_official: List[float] = []
    p50_official: List[float] = []
    p99_official: List[float] = []
    all_ms_official: List[float] = []

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

        tele = telemetry_summaries[r_idx]
        is_off = tele["is_official"]
        if is_off:
            p50_official.append(p50_r)
            p95_official.append(p95_r)
            p99_official.append(p99_r)
            all_ms_official.extend(rms)

        per_run_stats.append({
            "run": r_idx + 1,
            "n_queries": len(rms),
            "is_official": is_off,
            "label": "CHÍNH THỨC" if is_off else "KHÔNG CHÍNH THỨC (nhiệt độ)",
            "p50_ms": round(p50_r, 3),
            "p95_ms": round(p95_r, 3),
            "p99_ms": round(p99_r, 3),
            "mean_ms": round(mean_r, 3),
            "telemetry": tele,
        })

    # Stats on official runs (primary for conclusion)
    target_p95_list = p95_official if p95_official else p95_runs
    target_p50_list = p50_official if p50_official else p50_runs
    target_p99_list = p99_official if p99_official else p99_runs
    target_all_ms = all_ms_official if all_ms_official else all_ms

    q25, q75 = np.percentile(np.asarray(target_p95_list, dtype=np.float64), [25, 75])
    p95_med = float(statistics.median(target_p95_list))
    p95_iqr = float(q75 - q25)
    p50_med = float(statistics.median(target_p50_list))
    p99_med = float(statistics.median(target_p99_list))

    arr_all = np.asarray(target_all_ms, dtype=np.float64)
    pooled_p50 = float(np.percentile(arr_all, 50))
    pooled_p95 = float(np.percentile(arr_all, 95))
    pooled_p99 = float(np.percentile(arr_all, 99))
    pooled_mean = float(np.mean(arr_all))

    n_official = len(p95_official)
    n_unofficial = len(p95_runs) - n_official

    conclusion = classify_latency_conclusion(p95_official) if p95_official else "KHÔNG CHÍNH THỨC (nhiệt độ)"

    return {
        "n_runs_total": len(run_latencies_ms),
        "n_runs_official": n_official,
        "n_runs_unofficial": n_unofficial,
        "all_runs_official": (n_unofficial == 0),
        "n_queries_per_run": len(run_latencies_ms[0]) if run_latencies_ms else 0,
        "total_measured_queries": len(all_ms),
        "p95_runs_ms": [round(x, 3) for x in p95_runs],
        "p50_runs_ms": [round(x, 3) for x in p50_runs],
        "p99_runs_ms": [round(x, 3) for x in p99_runs],
        "p95_median_ms": round(p95_med, 3),
        "p95_max_ms": round(float(max(target_p95_list)), 3),
        "p95_iqr_ms": round(p95_iqr, 3),
        "p50_median_ms": round(p50_med, 3),
        "p99_median_ms": round(p99_med, 3),
        "pooled_p50_ms": round(pooled_p50, 3),
        "pooled_p95_ms": round(pooled_p95, 3),
        "pooled_p99_ms": round(pooled_p99, 3),
        "pooled_mean_ms": round(pooled_mean, 3),
        "sla_target_p95_ms": SLA_LATENCY_P95_MS_LOCKED,
        "sla_p95_under_200ms": bool(p95_med < SLA_LATENCY_P95_MS_LOCKED and n_official > 0),
        "conclusion_label": conclusion,
        "warn_upload_p99_over_200ms": bool(is_cir_upload and (pooled_p99 > 200.0 or p99_med > 200.0)),
        "per_run": per_run_stats,
    }


def record_step3_environment(agent_pid: int) -> Dict[str, Any]:
    def _run_cmd(cmd: str) -> str:
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=10).strip()
        except Exception as e:
            return f"ERROR: {e}"

    battery = psutil.sensors_battery()
    bat_info = {
        "power_plugged": battery.power_plugged if battery else True,
        "percent": battery.percent if battery else 100,
        "secsleft": battery.secsleft if battery else -2,
    } if battery else "Desktop / No battery"

    cfreq = psutil.cpu_freq()
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "utilization_pct": psutil.cpu_percent(interval=1),
        "frequency_mhz": cfreq._asdict() if cfreq else None,
    }

    smi_gpu = _run_cmd(
        "nvidia-smi --query-gpu=name,power.limit,power.draw,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown,temperature.gpu,utilization.gpu,utilization.memory,clocks.current.graphics,clocks.current.memory,pstate --format=csv,noheader,nounits"
    )

    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x.get("cpu_percent") or 0.0, reverse=True)
    top_15_cpu = procs[:15]

    gpu_procs = _run_cmd("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "power_scheme": _run_cmd("powercfg /getactivescheme"),
        "power_status_ac_battery": bat_info,
        "cpu_info": cpu_info,
        "gpu_power_and_throttles": smi_gpu,
        "top_15_cpu_processes": top_15_cpu,
        "gpu_processes": gpu_procs,
        "agent_python_pid": agent_pid,
        "agent_parent_pid": os.getppid(),
        "execution_context": "Antigravity IDE Agent Environment (Prompt 14f Branch phase7a)",
    }


def main() -> None:
    print("=" * 80)
    print("PROMPT 14f — HIỆU CHỈNH PHẦN A (PROMPT 14e) VÀ CHỐT QUYẾT ĐỊNH CIR")
    print(f"Timestamp (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Git commit: {get_git_commit()}")
    print("=" * 80, flush=True)

    agent_pid = os.getpid()

    # =========================================================================
    # STEP 1: Verify CIR User Decision file
    # =========================================================================
    cir_dec_path = PROJECT_ROOT / "reports/phase5c/cir_user_decision_prompt14f.json"
    if not cir_dec_path.exists():
        raise RuntimeError("Missing reports/phase5c/cir_user_decision_prompt14f.json!")
    cir_decision_data = json.loads(cir_dec_path.read_text(encoding="utf-8"))
    cir_decision_sha = sha256_file(cir_dec_path)
    print(f"[Step 1] CIR User Decision SHA-256: {cir_decision_sha}")
    print(f"         Selection: {cir_decision_data.get('decision_id')}")

    # =========================================================================
    # STEP 2: Erratum 14e evidence assembly
    # =========================================================================
    p14e_results_path = PROJECT_ROOT / "reports/phase5c/final_eval_results_prompt14e.json"
    if not p14e_results_path.exists():
        raise RuntimeError("Missing renamed 14e results: reports/phase5c/final_eval_results_prompt14e.json!")
    p14e_res = json.loads(p14e_results_path.read_text(encoding="utf-8"))

    # Extract 14e temperatures
    p14e_pal = p14e_res.get("part_a_latency", {})
    temps_14e: List[float] = []
    def _extract_temps(obj):
        if isinstance(obj, dict):
            if "telemetry_before" in obj and "temperature_c" in obj["telemetry_before"]:
                temps_14e.append(float(obj["telemetry_before"]["temperature_c"]))
            for v in obj.values():
                _extract_temps(v)
        elif isinstance(obj, list):
            for it in obj:
                _extract_temps(it)
    _extract_temps(p14e_pal)

    erratum_14e = {
        "item_a_temperature": {
            "total_runs_measured": len(temps_14e),
            "runs_starting_ge_60c": sum(1 for t in temps_14e if t >= 60.0),
            "temp_min": float(min(temps_14e)),
            "temp_median": float(statistics.median(temps_14e)),
            "temp_max": float(max(temps_14e)),
            "label_applied_in_14e": "official_with_idle_baseline (bị gán sai theo quy tắc nhiệt độ)",
            "correct_classification": "KHÔNG CHÍNH THỨC (nhiệt độ)",
            "explanation": "Theo luật 14d, nhãn official_with_idle_baseline chỉ được phép cho GPU-Util nghỉ, không cho nhiệt độ. 65/65 run bắt đầu ở >= 60°C nên toàn bộ số đo độ trễ 14e phải gắn nhãn 'KHÔNG CHÍNH THỨC (nhiệt độ)'."
        },
        "item_b_background_load": {
            "idle_util_14d_median_pct": 29.5,
            "idle_util_14e_before_pct": 23.0,
            "idle_util_14e_after_pct": 28.5,
            "dismissed_claim": "Bỏ khẳng định hai tiến trình LDPlayer/EdgeGameAssist là nguyên nhân tải nền.",
            "remaining_background_load_source": "nguồn tải nền còn lại: không xác minh được",
            "evidence": "Tải GPU nền vẫn duy trì ở mức 23% - 28.5% ngay cả khi LDPlayer/EdgeGameAssist đã tắt hoàn toàn."
        },
        "item_c_telemetry_duplication": {
            "script_file": "scripts/run_prompt14e_part_a.py",
            "code_line_evidence": "cir_up_tele[cid] được truyền cho cả primary_ram_preloaded và secondary_disk_jpeg_decode tại vòng lặp Stream 4.",
            "is_copied": True,
            "validity": "Telemetry của biến thể đọc đĩa trong 14e không hợp lệ do dùng chung dữ liệu với biến thể RAM."
        },
        "item_d_map_numbers": {
            "value_0_4077": {
                "source_file": "reports/phase5b/step3_results.json",
                "exact_metric": "map_a_cluster cho cấu hình Text→Ảnh cfg_22 trên tập val cluster A",
                "is_cir_metric": False,
                "note": "Hoàn toàn không phải là mAP của CIR."
            },
            "value_0_5471": {
                "source_file": "không xác minh được",
                "exact_metric": "không tồn tại trong bất kỳ báo cáo hoặc kết quả CIR nào (chỉ xuất hiện trong timestamp của tệp kết quả cũ)",
                "is_cir_metric": False,
                "status": "KHÔNG DÙNG"
            },
            "verified_val_cir_map": {
                "cir_v1_05": 0.2902,
                "cir_v1_01": 0.0910,
                "source_report": "reports/phase5c/step5c_cir_report.md"
            }
        },
        "item_e_phase5_gap": {
            "t2i_model_only_p95_gap": "29.857 ms (Pha 5) -> 109.851 ms (14e)",
            "cir_catalog_cir_v1_01_p95_gap": "36.976 ms (Pha 5) -> 120.800 ms (14e)",
            "i2i_ram_p95_gap": "126.513 ms (Pha 5) -> 289.102 ms (14e)",
            "cir_upload_ram_p95_gap": "165.489 ms (Pha 5) -> 368.505 ms (14e)",
            "status": "Hiện CHƯA có nguyên nhân xác minh được. Giải thích ở Prompt 13 (Roblox, ghìm nhiệt) không còn đủ vì các tiến trình đó đã tắt mà chênh lệch vẫn còn."
        },
        "item_f_p99_warning": {
            "cfg_v5_01_p99_median_ms": 202.342,
            "rule_scope": "Luật cảnh báo p99 chỉ áp cho luồng cir_upload theo locked_final_definitions.py, nên việc cfg_v5_01 vượt 200 ms được ghi nhận tham khảo, không vi phạm SLA luồng."
        }
    }
    print("[Step 2] Erratum 14e assembled successfully.")

    # =========================================================================
    # STEP 3: Record Environment (read-only)
    # =========================================================================
    env_snapshot = record_step3_environment(agent_pid)
    print(f"[Step 3] Environment snapshot recorded. Agent PID={agent_pid}, Power Scheme={env_snapshot['power_scheme']}")

    # =========================================================================
    # Check Inputs, Preprocess Parity 0.3, and Step 1a Hash Guard
    # =========================================================================
    print("[Pre-flight] Checking input hashes...")
    step0_hashes = {
        "selection_rule_cir_v1": (
            "reports/phase5c/selection_rule_cir_v1.json",
            EXPECTED_HASHES["selection_rule_cir_v1"][1],
            sha256_file(PROJECT_ROOT / "reports/phase5c/selection_rule_cir_v1.json"),
        ),
        "selection_rule_v5": (
            "reports/phase5c/selection_rule_v5.json",
            EXPECTED_HASHES["selection_rule_v5"][1],
            sha256_file(PROJECT_ROOT / "reports/phase5c/selection_rule_v5.json"),
        ),
        "cir_user_decision_prompt14f": (
            "reports/phase5c/cir_user_decision_prompt14f.json",
            cir_decision_sha,
            sha256_file(cir_dec_path),
        ),
        "val_queries_b_synth": (
            "eval/val_queries_b_synth.json",
            "41739174392fd1fe4cb14c76bddefc33ff23cf7dd7e885d9e999987db3bea6a1",
            sha256_file(PROJECT_ROOT / "eval/val_queries_b_synth.json"),
        ),
        "cir_eval_val": (
            "eval/cir_eval_val.json",
            "c735c7742c5277cffc4113c68dd8ee59fd3475c742284c8fe9e098af0819062a",
            sha256_file(PROJECT_ROOT / "eval/cir_eval_val.json"),
        ),
        "cir_eval_test_raw_hash_only": (
            "eval/cir_eval_test.json",
            EXPECTED_HASHES["cir_eval_test"][1],
            sha256_file(PROJECT_ROOT / "eval/cir_eval_test.json"),
        ),
    }
    for k, (p, exp, act) in step0_hashes.items():
        if exp != act:
            raise RuntimeError(f"Hash mismatch for {k} ({p}): expected {exp}, got {act}")
        print(f"  OK: {k} -> {act}")

    # Check heldout_v2
    heldout_v2_path = PROJECT_ROOT / "eval/test_queries_heldout_v2.json"
    heldout_v2_exists = heldout_v2_path.exists()
    heldout_v2_info = {
        "exists": heldout_v2_exists,
        "path": "eval/test_queries_heldout_v2.json",
        "sha256": sha256_file(heldout_v2_path) if heldout_v2_exists else None,
        "note": "eval/test_queries_heldout_v2.json chưa khóa. Dừng sau Phần A2." if not heldout_v2_exists else "Đã khóa nhưng không chạy B/C trong prompt này."
    }
    print(f"[Pre-flight] Heldout v2 check: exists={heldout_v2_exists}")

    # Step 0.3 Parity
    print("[Step 0.3] Running 200-sample preprocess parity check on index/test_gallery/...")
    step0_3_res = run_step0_3_parity_200_test_gallery()
    if not step0_3_res["passed"]:
        raise RuntimeError(f"Step 0.3 parity failed! min_cosine={step0_3_res['min_cosine']}")
    print(f"  OK: min_cosine={step0_3_res['min_cosine']:.6f} >= 0.999")

    # Step 1a Hash guard
    print("[Step 1a] Running Hash Guard on index/test_gallery_text_v5/...")
    step1a_res = run_step_1a_hash_check()
    if not step1a_res["test_gallery_text_v5"]["hash_guard_and_sanity"]["passed"]:
        raise RuntimeError("Step 1a hash guard failed on test_gallery_text_v5!")
    print("  OK: Step 1a passed.")

    # Step 1c Reuse from 14e
    step1c_res = p14e_res["step_1c_caption_limitation"]

    # =========================================================================
    # STEP 4: PART A2 — 4-STREAM C.9 RE-BENCHMARK
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4 — PHẦN A2: ĐO LẠI ĐÚNG C.9 (GIAO THỨC LÀM NGUỘI < 60°C & TELEMETRY ĐỘC LẬP)")
    print("=" * 80, flush=True)

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

    t2i_q_info = build_t2i_200_latency_queries()
    queries_t2i_200 = t2i_q_info["queries_200"]

    cir_p_info = build_cir_and_i2i_200_latency_pairs(engine)
    pairs_200 = cir_p_info["pairs_200"]

    candidate_pool = 100
    all_telemetry_raw: Dict[str, Any] = {}

    def _rank_hnsw_cir(q_vec: np.ndarray, parsed_mod: Dict[str, Any], cfg: Dict[str, Any]) -> List[Tuple[float, int]]:
        scores, indices = engine.hnsw.search(q_vec, top_k=candidate_pool)
        cand_scores = scores[0]
        cand_indices = indices[0]
        use_c3 = bool(cfg.get("c3_mod_bonus", False))
        if not use_c3:
            return [(float(s), int(i)) for i, s in zip(cand_indices[:10], cand_scores[:10]) if i >= 0]
        gamma = float(cfg.get("gamma", 1.0))
        has_active = bool(parsed_mod.get("articleType") or parsed_mod.get("gender") or parsed_mod.get("baseColour"))
        scored: List[Tuple[float, int]] = []
        for idx_val, base_s in zip(cand_indices, cand_scores):
            if idx_val < 0:
                continue
            idx_int = int(idx_val)
            s_val = float(base_s)
            if has_active:
                mf = compute_weighted_attribute_match(
                    parsed_mod, engine.id_map[idx_int], weights=None, gender_policy="hybrid", strict_categories=strict_cats
                )
                if mf > 0.0:
                    tier_b = 0.5 if mf >= 0.999 else 0.0
                    s_val += gamma * (mf + tier_b)
            scored.append((s_val, idx_int))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:10]

    # Helper to check stream halt condition (> 50% unofficial runs)
    halted_stream_name: Optional[str] = None

    # =========================================================================
    # STREAM 4 (PRIORITY 1): CIR Upload
    # RAM and Disk are measured SEPARATELY in independent 5-run sequences!
    # =========================================================================
    print("\n--- STREAM (1/4): CIR Upload (Priority 1) ---")
    cir_configs = [grid_cir_by_id["cir_v1_01"], grid_cir_by_id["cir_v1_05"]]
    stage_names = [
        "stage1_disk_jpeg_decode",
        "stage2_resize_processor",
        "stage3_vision_tower_forward",
        "stage4_text_tower_and_parser",
        "stage5_combiner_forward",
        "stage6_hnsw_and_c3_bonus",
    ]

    # Part A2 - 1.1 CIR Upload (RAM PIL)
    print("\n[Stream 1/4 - Variant A: CIR Upload - Primary RAM Preloaded]")
    cir_up_ram_runs: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in cir_configs}
    cir_up_ram_tele: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in cir_configs}
    cir_up_ram_stages: Dict[str, Dict[str, List[float]]] = {c["config_id"]: {sn: [] for sn in stage_names} for c in cir_configs}

    for r_idx in range(5):
        print(f"  Run {r_idx + 1}/5 (RAM)... Waiting for GPU < 60°C...", flush=True)
        is_official, start_t, wait_s = wait_for_gpu_cooldown()
        print(f"  Start temp={start_t:.1f}°C, wait={wait_s:.1f}s, official={is_official}")

        # Warmup 20
        for w_idx, item_w in enumerate(pairs_200[:20]):
            cfg = cir_configs[(r_idx + w_idx) % 2]
            px = engine.preprocessor.preprocess_single_image(item_w["pil_rgb"], device=engine.device, half_precision=(engine.device == "cuda"))
            with torch.no_grad():
                ref_img_emb = engine.model.encode_image(px)
            raw_mod = item_w["raw_mod"]
            qt_vi = restore_diacritics_for_query(raw_mod)
            parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
            mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
            mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
            with torch.no_grad():
                mod_txt_emb = engine.model.encode_text(input_ids=mod_inputs["input_ids"], attention_mask=mod_inputs.get("attention_mask"))
                cir_emb = engine.model.cir_combiner(ref_img_emb, mod_txt_emb)
                q_vec = cir_emb.float().cpu().numpy()
            q_vec = engine._apply_mean_centering(q_vec)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = _rank_hnsw_cir(q_vec, parsed_mod, cfg)

        sampler = TelemetrySampler(interval_s=1.0)
        sampler.start()

        cur_ram_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in cir_configs}
        for q_idx, item in enumerate(pairs_200):
            order = cir_configs if (r_idx + q_idx) % 2 == 0 else list(reversed(cir_configs))
            for cfg in order:
                cid = cfg["config_id"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_ram_start = time.perf_counter()

                px = engine.preprocessor.preprocess_single_image(item["pil_rgb"], device=engine.device, half_precision=(engine.device == "cuda"))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s2 = time.perf_counter()

                with torch.no_grad():
                    ref_img_emb = engine.model.encode_image(px)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s3 = time.perf_counter()

                raw_mod = item["raw_mod"]
                qt_vi = restore_diacritics_for_query(raw_mod)
                parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
                mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
                mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
                with torch.no_grad():
                    mod_txt_emb = engine.model.encode_text(input_ids=mod_inputs["input_ids"], attention_mask=mod_inputs.get("attention_mask"))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s4 = time.perf_counter()

                with torch.no_grad():
                    cir_emb = engine.model.cir_combiner(ref_img_emb, mod_txt_emb)
                    q_vec = cir_emb.float().cpu().numpy()
                q_vec = engine._apply_mean_centering(q_vec)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s5 = time.perf_counter()

                _ = _rank_hnsw_cir(q_vec, parsed_mod, cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s6 = time.perf_counter()

                ram_total = (t_s6 - t_ram_start) * 1000.0
                cur_ram_ms[cid].append(ram_total)
                cir_up_ram_stages[cid]["stage1_disk_jpeg_decode"].append(0.0)
                cir_up_ram_stages[cid]["stage2_resize_processor"].append((t_s2 - t_ram_start) * 1000.0)
                cir_up_ram_stages[cid]["stage3_vision_tower_forward"].append((t_s3 - t_s2) * 1000.0)
                cir_up_ram_stages[cid]["stage4_text_tower_and_parser"].append((t_s4 - t_s3) * 1000.0)
                cir_up_ram_stages[cid]["stage5_combiner_forward"].append((t_s5 - t_s4) * 1000.0)
                cir_up_ram_stages[cid]["stage6_hnsw_and_c3_bonus"].append((t_s6 - t_s5) * 1000.0)

        raw_samples = sampler.stop()
        end_t = query_gpu_temp()
        tele_summary = summarize_telemetry_samples(raw_samples, start_t, end_t, is_official, wait_s)
        all_telemetry_raw[f"cir_upload_ram_run_{r_idx + 1}"] = raw_samples

        for cfg in cir_configs:
            cid = cfg["config_id"]
            cir_up_ram_runs[cid].append(cur_ram_ms[cid])
            cir_up_ram_tele[cid].append(tele_summary)

    # Part A2 - 1.2 CIR Upload (Disk JPEG decode)
    print("\n[Stream 1/4 - Variant B: CIR Upload - Secondary Disk JPEG Decode]")
    cir_up_disk_runs: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in cir_configs}
    cir_up_disk_tele: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in cir_configs}
    cir_up_disk_stages: Dict[str, Dict[str, List[float]]] = {c["config_id"]: {sn: [] for sn in stage_names} for c in cir_configs}

    for r_idx in range(5):
        print(f"  Run {r_idx + 1}/5 (Disk)... Waiting for GPU < 60°C...", flush=True)
        is_official, start_t, wait_s = wait_for_gpu_cooldown()
        print(f"  Start temp={start_t:.1f}°C, wait={wait_s:.1f}s, official={is_official}")

        # Warmup 20
        for w_idx, item_w in enumerate(pairs_200[:20]):
            cfg = cir_configs[(r_idx + w_idx) % 2]
            with Image.open(item_w["ref_image_abs_path"]) as im:
                im_rgb = im.convert("RGB")
            px = engine.preprocessor.preprocess_single_image(im_rgb, device=engine.device, half_precision=(engine.device == "cuda"))
            with torch.no_grad():
                ref_img_emb = engine.model.encode_image(px)
            raw_mod = item_w["raw_mod"]
            qt_vi = restore_diacritics_for_query(raw_mod)
            parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
            mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
            mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
            with torch.no_grad():
                mod_txt_emb = engine.model.encode_text(input_ids=mod_inputs["input_ids"], attention_mask=mod_inputs.get("attention_mask"))
                cir_emb = engine.model.cir_combiner(ref_img_emb, mod_txt_emb)
                q_vec = cir_emb.float().cpu().numpy()
            q_vec = engine._apply_mean_centering(q_vec)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _ = _rank_hnsw_cir(q_vec, parsed_mod, cfg)

        sampler = TelemetrySampler(interval_s=1.0)
        sampler.start()

        cur_disk_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in cir_configs}
        for q_idx, item in enumerate(pairs_200):
            order = cir_configs if (r_idx + q_idx) % 2 == 0 else list(reversed(cir_configs))
            for cfg in order:
                cid = cfg["config_id"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_disk_start = time.perf_counter()

                with Image.open(item["ref_image_abs_path"]) as im:
                    im_rgb = im.convert("RGB")
                t_s1 = time.perf_counter()

                px = engine.preprocessor.preprocess_single_image(im_rgb, device=engine.device, half_precision=(engine.device == "cuda"))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s2 = time.perf_counter()

                with torch.no_grad():
                    ref_img_emb = engine.model.encode_image(px)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s3 = time.perf_counter()

                raw_mod = item["raw_mod"]
                qt_vi = restore_diacritics_for_query(raw_mod)
                parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
                mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
                mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
                with torch.no_grad():
                    mod_txt_emb = engine.model.encode_text(input_ids=mod_inputs["input_ids"], attention_mask=mod_inputs.get("attention_mask"))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s4 = time.perf_counter()

                with torch.no_grad():
                    cir_emb = engine.model.cir_combiner(ref_img_emb, mod_txt_emb)
                    q_vec = cir_emb.float().cpu().numpy()
                q_vec = engine._apply_mean_centering(q_vec)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s5 = time.perf_counter()

                _ = _rank_hnsw_cir(q_vec, parsed_mod, cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_s6 = time.perf_counter()

                disk_total = (t_s6 - t_disk_start) * 1000.0
                cur_disk_ms[cid].append(disk_total)
                cir_up_disk_stages[cid]["stage1_disk_jpeg_decode"].append((t_s1 - t_disk_start) * 1000.0)
                cir_up_disk_stages[cid]["stage2_resize_processor"].append((t_s2 - t_s1) * 1000.0)
                cir_up_disk_stages[cid]["stage3_vision_tower_forward"].append((t_s3 - t_s2) * 1000.0)
                cir_up_disk_stages[cid]["stage4_text_tower_and_parser"].append((t_s4 - t_s3) * 1000.0)
                cir_up_disk_stages[cid]["stage5_combiner_forward"].append((t_s5 - t_s4) * 1000.0)
                cir_up_disk_stages[cid]["stage6_hnsw_and_c3_bonus"].append((t_s6 - t_s5) * 1000.0)

        raw_samples = sampler.stop()
        end_t = query_gpu_temp()
        tele_summary = summarize_telemetry_samples(raw_samples, start_t, end_t, is_official, wait_s)
        all_telemetry_raw[f"cir_upload_disk_run_{r_idx + 1}"] = raw_samples

        for cfg in cir_configs:
            cid = cfg["config_id"]
            cir_up_disk_runs[cid].append(cur_disk_ms[cid])
            cir_up_disk_tele[cid].append(tele_summary)

    stream4_cir_upload_summary: Dict[str, Any] = {}
    for cfg in cir_configs:
        cid = cfg["config_id"]
        stage_summary: Dict[str, Any] = {}
        for sn in stage_names:
            arr_s = np.asarray(cir_up_disk_stages[cid][sn], dtype=np.float64)
            stage_summary[sn] = {
                "p50_ms": round(float(np.percentile(arr_s, 50)), 3),
                "p95_ms": round(float(np.percentile(arr_s, 95)), 3),
                "p99_ms": round(float(np.percentile(arr_s, 99)), 3),
                "mean_ms": round(float(np.mean(arr_s)), 3),
            }
        stream4_cir_upload_summary[cid] = {
            "config_id": cid,
            "measured_individually": True,
            "primary_ram_preloaded": summarize_stream_runs(
                cir_up_ram_runs[cid], cir_up_ram_tele[cid], is_cir_upload=True
            ),
            "secondary_disk_jpeg_decode": summarize_stream_runs(
                cir_up_disk_runs[cid], cir_up_disk_tele[cid], is_cir_upload=True
            ),
            "stage_breakdown_ms": stage_summary,
        }

    # Check unofficial halt for CIR upload
    n_unofficial_cu = stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["n_runs_unofficial"]
    if n_unofficial_cu > 2:
        halted_stream_name = "cir_upload"
        print(f"!!! HALTING AFTER CIR UPLOAD: {n_unofficial_cu}/5 runs exceeded 60°C cooldown timeout.")

    # =========================================================================
    # STREAM 2 (PRIORITY 2): Image -> Image
    # =========================================================================
    stream2_i2i_summary: Dict[str, Any] = {}
    if halted_stream_name is None:
        print("\n--- STREAM (2/4): Image -> Image (Priority 2) ---")
        def _run_i2i_single(item: Dict[str, Any], from_disk: bool = False) -> None:
            img_in = item["ref_image_abs_path"] if from_disk else item["pil_rgb"]
            qv = engine.encode_image_query(img_in)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            scores, indices = engine.hnsw.search(qv, top_k=candidate_pool)
            _ = indices[0][:10]

        # I2I RAM
        print("  Measuring I2I - Primary RAM Preloaded...")
        i2i_ram_runs: List[List[float]] = []
        i2i_ram_tele: List[Dict[str, Any]] = []
        for r_idx in range(5):
            print(f"  Run {r_idx + 1}/5 (I2I RAM)... Waiting for GPU < 60°C...", flush=True)
            is_official, start_t, wait_s = wait_for_gpu_cooldown()
            for item_w in pairs_200[:20]:
                _run_i2i_single(item_w, from_disk=False)
            sampler = TelemetrySampler(interval_s=1.0)
            sampler.start()
            cur_ram = []
            for item in pairs_200:
                t0 = time.perf_counter()
                _run_i2i_single(item, from_disk=False)
                cur_ram.append((time.perf_counter() - t0) * 1000.0)
            raw_samples = sampler.stop()
            end_t = query_gpu_temp()
            tele_summary = summarize_telemetry_samples(raw_samples, start_t, end_t, is_official, wait_s)
            all_telemetry_raw[f"i2i_ram_run_{r_idx + 1}"] = raw_samples
            i2i_ram_runs.append(cur_ram)
            i2i_ram_tele.append(tele_summary)

        # I2I Disk
        print("  Measuring I2I - Secondary Disk JPEG Decode...")
        i2i_disk_runs: List[List[float]] = []
        i2i_disk_tele: List[Dict[str, Any]] = []
        for r_idx in range(5):
            print(f"  Run {r_idx + 1}/5 (I2I Disk)... Waiting for GPU < 60°C...", flush=True)
            is_official, start_t, wait_s = wait_for_gpu_cooldown()
            for item_w in pairs_200[:20]:
                _run_i2i_single(item_w, from_disk=True)
            sampler = TelemetrySampler(interval_s=1.0)
            sampler.start()
            cur_disk = []
            for item in pairs_200:
                t0 = time.perf_counter()
                _run_i2i_single(item, from_disk=True)
                cur_disk.append((time.perf_counter() - t0) * 1000.0)
            raw_samples = sampler.stop()
            end_t = query_gpu_temp()
            tele_summary = summarize_telemetry_samples(raw_samples, start_t, end_t, is_official, wait_s)
            all_telemetry_raw[f"i2i_disk_run_{r_idx + 1}"] = raw_samples
            i2i_disk_runs.append(cur_disk)
            i2i_disk_tele.append(tele_summary)

        stream2_i2i_summary = {
            "ram_preloaded": summarize_stream_runs(i2i_ram_runs, i2i_ram_tele, is_cir_upload=False),
            "disk_jpeg_decode": summarize_stream_runs(i2i_disk_runs, i2i_disk_tele, is_cir_upload=False),
        }
        if stream2_i2i_summary["ram_preloaded"]["n_runs_unofficial"] > 2:
            halted_stream_name = "image_to_image"
            print("!!! HALTING AFTER I2I: > 50% runs unofficial.")

    # =========================================================================
    # STREAM 3 (PRIORITY 3): CIR Catalog
    # =========================================================================
    stream3_cir_catalog_summary: Dict[str, Any] = {}
    if halted_stream_name is None:
        print("\n--- STREAM (3/4): CIR Catalog (Priority 3) ---")
        def _run_cir_catalog_single(cfg: Dict[str, Any], item: Dict[str, Any]) -> None:
            raw_mod = item["raw_mod"]
            qt_vi = restore_diacritics_for_query(raw_mod)
            parsed_mod = parser.parse_with_confidence(qt_vi, fuzzy_typo=False)
            mod_str = translator.translate(raw_mod) if cfg.get("c1_vi_to_en") else qt_vi
            mod_inputs = engine.preprocessor.preprocess_single_text(mod_str, device=engine.device)
            with torch.no_grad():
                mod_txt_emb = engine.model.encode_text(input_ids=mod_inputs["input_ids"], attention_mask=mod_inputs.get("attention_mask"))
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
            print(f"  Run {r_idx + 1}/5 (CIR Catalog)... Waiting for GPU < 60°C...", flush=True)
            is_official, start_t, wait_s = wait_for_gpu_cooldown()
            for w_idx, item_w in enumerate(pairs_200[:20]):
                order_w = cir_configs if (r_idx + w_idx) % 2 == 0 else list(reversed(cir_configs))
                for cfg in order_w:
                    _run_cir_catalog_single(cfg, item_w)

            sampler = TelemetrySampler(interval_s=1.0)
            sampler.start()

            cur_run_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in cir_configs}
            for q_idx, item in enumerate(pairs_200):
                order = cir_configs if (r_idx + q_idx) % 2 == 0 else list(reversed(cir_configs))
                for cfg in order:
                    cid = cfg["config_id"]
                    t0 = time.perf_counter()
                    _run_cir_catalog_single(cfg, item)
                    cur_run_ms[cid].append((time.perf_counter() - t0) * 1000.0)

            raw_samples = sampler.stop()
            end_t = query_gpu_temp()
            tele_summary = summarize_telemetry_samples(raw_samples, start_t, end_t, is_official, wait_s)
            all_telemetry_raw[f"cir_catalog_run_{r_idx + 1}"] = raw_samples

            for cfg in cir_configs:
                cid = cfg["config_id"]
                cir_cat_runs_ms[cid].append(cur_run_ms[cid])
                cir_cat_tele[cid].append(tele_summary)

        for cfg in cir_configs:
            cid = cfg["config_id"]
            stream3_cir_catalog_summary[cid] = summarize_stream_runs(
                cir_cat_runs_ms[cid], cir_cat_tele[cid], is_cir_upload=False
            )
            stream3_cir_catalog_summary[cid]["config_id"] = cid
            stream3_cir_catalog_summary[cid]["measured_individually"] = True

        if stream3_cir_catalog_summary["cir_v1_05"]["n_runs_unofficial"] > 2:
            halted_stream_name = "cir_catalog"
            print("!!! HALTING AFTER CIR CATALOG: > 50% runs unofficial.")

    # =========================================================================
    # STREAM 1 (PRIORITY 4): Text -> Image
    # =========================================================================
    stream1_t2i_summary: Dict[str, Any] = {}
    if halted_stream_name is None:
        print("\n--- STREAM (4/4): Text -> Image (Priority 4) ---")
        t2i_configs: List[Dict[str, Any]] = [
            {"config_id": "model_only", "role": "model_only_cosine", "mode": "model_only", "i3_mode": "none", "fuzzy_typo": False, "conf_gating": False, "bonus_mode": "none", "gamma": 0.0, "text_gallery": "none", "text_alpha": 1.0},
            dict(grid_v5_by_id["cfg_v5_01"], mode="system"),
            dict(grid_v5_by_id["cfg_v5_06"], mode="system"),
            dict(grid_v5_by_id["cfg_v5_09"], mode="system"),
            dict(grid_v5_by_id["cfg_v5_08b"], mode="system"),
        ]
        n_t2i_cfgs = len(t2i_configs)

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
                eff_weights = {k: (0.2 if conf_map.get(k) == "low" else 1.0) for k in ("articleType", "gender", "baseColour")}

            scored: List[Tuple[float, int]] = []
            for pos, (idx_val, base_s) in enumerate(zip(cand_indices, base_arr)):
                if idx_val < 0:
                    continue
                idx_int = int(idx_val)
                s_val = float(base_s)
                meta = engine.id_map[idx_int]
                mf = compute_weighted_attribute_match(parsed, meta, weights=eff_weights, gender_policy="hybrid", strict_categories=strict_cats)
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

        t2i_runs_ms: Dict[str, List[List[float]]] = {c["config_id"]: [] for c in t2i_configs}
        t2i_tele: Dict[str, List[Dict[str, Any]]] = {c["config_id"]: [] for c in t2i_configs}

        for r_idx in range(5):
            print(f"  Run {r_idx + 1}/5 (T2I 5x5 Latin square)... Waiting for GPU < 60°C...", flush=True)
            is_official, start_t, wait_s = wait_for_gpu_cooldown()

            for w_idx, q_w in enumerate(queries_t2i_200[:20]):
                order_w = [t2i_configs[(i + r_idx + w_idx) % n_t2i_cfgs] for i in range(n_t2i_cfgs)]
                for cfg in order_w:
                    _run_t2i_single(cfg, q_w)

            sampler = TelemetrySampler(interval_s=1.0)
            sampler.start()

            cur_run_ms: Dict[str, List[float]] = {c["config_id"]: [] for c in t2i_configs}
            for q_idx, q_str in enumerate(queries_t2i_200):
                order = [t2i_configs[(i + r_idx + q_idx) % n_t2i_cfgs] for i in range(n_t2i_cfgs)]
                for cfg in order:
                    cid = cfg["config_id"]
                    t0 = time.perf_counter()
                    _run_t2i_single(cfg, q_str)
                    cur_run_ms[cid].append((time.perf_counter() - t0) * 1000.0)

            raw_samples = sampler.stop()
            end_t = query_gpu_temp()
            tele_summary = summarize_telemetry_samples(raw_samples, start_t, end_t, is_official, wait_s)
            all_telemetry_raw[f"t2i_run_{r_idx + 1}"] = raw_samples

            for cfg in t2i_configs:
                cid = cfg["config_id"]
                t2i_runs_ms[cid].append(cur_run_ms[cid])
                t2i_tele[cid].append(tele_summary)

        for cfg in t2i_configs:
            cid = cfg["config_id"]
            stream1_t2i_summary[cid] = summarize_stream_runs(t2i_runs_ms[cid], t2i_tele[cid], is_cir_upload=False)
            stream1_t2i_summary[cid]["config_id"] = cid
            stream1_t2i_summary[cid]["measured_individually"] = True

    # Save raw telemetry
    telem_raw_path = PROJECT_ROOT / "reports/phase5c/part_a2_telemetry_raw.json"
    telem_raw_path.write_text(json.dumps(all_telemetry_raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Part A2] Saved raw 1 Hz telemetry to {telem_raw_path}")

    # Build locked conclusion
    def _build_conclusion(t2i_o, i2i_o, cir_cat_o, cir_up_o) -> Dict[str, Any]:
        by_stream = {
            "text_to_image": t2i_o.get("conclusion_label", "chưa đo") if t2i_o else "chưa đo",
            "image_to_image": i2i_o.get("conclusion_label", "chưa đo") if i2i_o else "chưa đo",
            "cir_catalog": cir_cat_o.get("conclusion_label", "chưa đo") if cir_cat_o else "chưa đo",
            "cir_upload": cir_up_o.get("conclusion_label", "chưa đo") if cir_up_o else "chưa đo",
        }
        passed_cnt = sum(
            1 for o in (t2i_o, i2i_o, cir_cat_o, cir_up_o)
            if o and o.get("sla_p95_under_200ms") and o.get("conclusion_label") in ("Đạt chắc chắn", "Đạt theo trung vị nhưng có run vượt")
        )
        if passed_cnt == 4:
            st = "PASS_FULL"
            lbl = "Đạt SLA toàn bộ 4 luồng"
        elif passed_cnt >= 3:
            st = "PASS_CIR_UPLOAD_BORDERLINE_OR_EXCEPTION"
            lbl = "Đạt SLA phần lớn luồng (ngoại lệ cir_upload)"
        else:
            st = "SLA_FAILED"
            lbl = f"Chưa đạt SLA ({passed_cnt}/4 luồng đạt)"
        return {"status": st, "passed_count": passed_cnt, "total_streams": 4, "label_vi": lbl, "by_stream": by_stream}

    conclusion_ram = _build_conclusion(
        stream1_t2i_summary.get("cfg_v5_09"),
        stream2_i2i_summary.get("ram_preloaded"),
        stream3_cir_catalog_summary.get("cir_v1_05"),
        stream4_cir_upload_summary.get("cir_v1_05", {}).get("primary_ram_preloaded"),
    )
    conclusion_disk = _build_conclusion(
        stream1_t2i_summary.get("cfg_v5_09"),
        stream2_i2i_summary.get("disk_jpeg_decode"),
        stream3_cir_catalog_summary.get("cir_v1_05"),
        stream4_cir_upload_summary.get("cir_v1_05", {}).get("secondary_disk_jpeg_decode"),
    )

    # =========================================================================
    # STEP 5: Phase 5 Diagnostic Benchmark Parity
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5 — ĐỐI CHIẾU PHA 5: CHẠY NGUYÊN VẸN eval_latency.py CỦA PHA 5")
    print("=" * 80, flush=True)

    p5_today_out = PROJECT_ROOT / "reports/phase5c/eval_latency_phase5_today.json"
    p5_cmd = [
        sys.executable,
        "-m",
        "src.eval.eval_latency",
        "--index-dir", "index/test_gallery",
        "--queries", "eval/handwritten_queries.json",
        "--config", "config.yaml",
        "--output", str(p5_today_out),
        "--warmup", "20",
        "--n-runs", "5",
        "--n-queries", "200",
        "--seed", "42",
    ]
    print(f"Running Phase 5 eval_latency: {' '.join(p5_cmd)}", flush=True)
    is_official_p5, start_t_p5, wait_s_p5 = wait_for_gpu_cooldown()
    print(f"  GPU Cooldown before Phase 5 run: temp={start_t_p5:.1f}°C, wait={wait_s_p5:.1f}s, official={is_official_p5}")

    p5_proc = subprocess.run(p5_cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if p5_proc.returncode != 0:
        print(f"ERROR running eval_latency.py: {p5_proc.stderr}")
        p5_today_data: Optional[Dict[str, Any]] = None
    else:
        p5_today_data = json.loads(p5_today_out.read_text(encoding="utf-8"))
        print(f"Phase 5 eval_latency run successfully completed. Saved to {p5_today_out}")

    # Load Phase 5 historical numbers
    p5_hist = json.loads((PROJECT_ROOT / "reports/phase5/latency_sla_results.json").read_text(encoding="utf-8"))

    step5_comparison: List[Dict[str, Any]] = []
    flows_map = [
        ("text_to_image", "Text→Ảnh", "model_only", stream1_t2i_summary.get("model_only", {}).get("p95_median_ms", "N/A")),
        ("image_to_image", "Ảnh→Ảnh", "RAM PIL", stream2_i2i_summary.get("ram_preloaded", {}).get("p95_median_ms", "N/A")),
        ("cir_catalog", "CIR catalog", "cir_v1_01", stream3_cir_catalog_summary.get("cir_v1_01", {}).get("p95_median_ms", "N/A")),
        ("cir_upload", "CIR upload", "cir_v1_01 (RAM)", stream4_cir_upload_summary.get("cir_v1_01", {}).get("primary_ram_preloaded", {}).get("p95_median_ms", "N/A")),
    ]
    for fl_key, fl_name, fl_cfg, h14f_num in flows_map:
        hist_fl = p5_hist["flows"].get(fl_key, {})
        today_fl = p5_today_data["flows"].get(fl_key, {}) if p5_today_data else {}
        step5_comparison.append({
            "flow_key": fl_key,
            "flow_name": fl_name,
            "config_note": fl_cfg,
            "phase5_historical_p95_ms": round(float(hist_fl.get("p95_ms", 0.0)), 3),
            "eval_latency_today_p95_ms": round(float(today_fl.get("p95_ms", 0.0)), 3) if today_fl else "ERROR",
            "eval_latency_today_p50_ms": round(float(today_fl.get("p50_ms", 0.0)), 3) if today_fl else "ERROR",
            "harness_14f_p95_median_ms": h14f_num,
        })

    # Evidence-based diagnostic interpretation
    diag_lines: List[str] = []
    if p5_today_data:
        p5_t2i_today = float(p5_today_data["flows"]["text_to_image"]["p95_ms"])
        p5_t2i_hist = float(p5_hist["flows"]["text_to_image"]["p95_ms"])
        ratio_t2i = p5_t2i_today / p5_t2i_hist

        p5_i2i_today = float(p5_today_data["flows"]["image_to_image"]["p95_ms"])
        p5_i2i_hist = float(p5_hist["flows"]["image_to_image"]["p95_ms"])
        ratio_i2i = p5_i2i_today / p5_i2i_hist

        if abs(ratio_t2i - 1.0) < 0.25 and abs(ratio_i2i - 1.0) < 0.25:
            diag_lines.append(
                f"- **Kết luận chẩn đoán:** `eval_latency.py` hôm nay tương đương số Pha 5 cũ (T2I: {p5_t2i_today:.2f} vs {p5_t2i_hist:.2f} ms; I2I: {p5_i2i_today:.2f} vs {p5_i2i_hist:.2f} ms). "
                "Điều này chứng minh phần cứng GPU/CPU không bị suy giảm hiệu năng cố hữu; chênh lệch trong harness 14e/14f bắt nguồn từ phạm vi đo rộng hơn: "
                "khôi phục dấu tiếng Việt, phân tích parser_v2.1 + slot confidence, logic tính attribute soft bonus, và phân bổ bộ nhớ Python."
            )
        else:
            diag_lines.append(
                f"- **Kết luận chẩn đoán:** `eval_latency.py` hôm nay cũng chậm hơn lịch sử Pha 5 ({ratio_t2i:.2f}x ở T2I, {ratio_i2i:.2f}x ở I2I). "
                "Điều này cho thấy trạng thái hệ thống nền (nền tảng OS, xung nhịp đồ họa/CPU, hoặc tải đồ họa của Desktop Window Manager) "
                "ảnh hưởng trực tiếp lên cả script đo Pha 5 gốc."
            )

    # Comparison Table for Part A2 (Prompt 14f vs 14e vs Phase 5)
    comparison_table = [
        {
            "stream": "Text→Ảnh",
            "config_id": "model_only",
            "phase5_clean_p95_ms": round(float(p5_hist["flows"]["text_to_image"]["p95_ms"]), 3),
            "prompt14e_p95_ms": p14e_pal["stream1_text_to_image"]["model_only"]["p95_median_ms"],
            "prompt14f_p50_ms": stream1_t2i_summary.get("model_only", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream1_t2i_summary.get("model_only", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream1_t2i_summary.get("model_only", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream1_t2i_summary.get("model_only", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream1_t2i_summary.get("model_only", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream1_t2i_summary.get("model_only", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_01",
            "phase5_clean_p95_ms": "không xác minh được (chưa có lưới v5)",
            "prompt14e_p95_ms": p14e_pal["stream1_text_to_image"]["cfg_v5_01"]["p95_median_ms"],
            "prompt14f_p50_ms": stream1_t2i_summary.get("cfg_v5_01", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream1_t2i_summary.get("cfg_v5_01", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream1_t2i_summary.get("cfg_v5_01", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream1_t2i_summary.get("cfg_v5_01", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream1_t2i_summary.get("cfg_v5_01", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream1_t2i_summary.get("cfg_v5_01", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_09",
            "phase5_clean_p95_ms": "không xác minh được (chưa có lưới v5)",
            "prompt14e_p95_ms": p14e_pal["stream1_text_to_image"]["cfg_v5_09"]["p95_median_ms"],
            "prompt14f_p50_ms": stream1_t2i_summary.get("cfg_v5_09", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream1_t2i_summary.get("cfg_v5_09", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream1_t2i_summary.get("cfg_v5_09", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream1_t2i_summary.get("cfg_v5_09", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream1_t2i_summary.get("cfg_v5_09", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream1_t2i_summary.get("cfg_v5_09", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "Text→Ảnh",
            "config_id": "cfg_v5_08b",
            "phase5_clean_p95_ms": "không xác minh được (chưa có lưới v5)",
            "prompt14e_p95_ms": p14e_pal["stream1_text_to_image"]["cfg_v5_08b"]["p95_median_ms"],
            "prompt14f_p50_ms": stream1_t2i_summary.get("cfg_v5_08b", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream1_t2i_summary.get("cfg_v5_08b", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream1_t2i_summary.get("cfg_v5_08b", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream1_t2i_summary.get("cfg_v5_08b", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream1_t2i_summary.get("cfg_v5_08b", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream1_t2i_summary.get("cfg_v5_08b", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "Ảnh→Ảnh",
            "config_id": "image_to_image (RAM PIL)",
            "phase5_clean_p95_ms": round(float(p5_hist["flows"]["image_to_image"]["p95_ms"]), 3),
            "prompt14e_p95_ms": p14e_pal["stream2_image_to_image"]["ram_preloaded"]["p95_median_ms"],
            "prompt14f_p50_ms": stream2_i2i_summary.get("ram_preloaded", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream2_i2i_summary.get("ram_preloaded", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream2_i2i_summary.get("ram_preloaded", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream2_i2i_summary.get("ram_preloaded", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream2_i2i_summary.get("ram_preloaded", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream2_i2i_summary.get("ram_preloaded", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "Ảnh→Ảnh",
            "config_id": "image_to_image (Disk+JPEG)",
            "phase5_clean_p95_ms": "không xác minh được",
            "prompt14e_p95_ms": p14e_pal["stream2_image_to_image"]["disk_jpeg_decode"]["p95_median_ms"],
            "prompt14f_p50_ms": stream2_i2i_summary.get("disk_jpeg_decode", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream2_i2i_summary.get("disk_jpeg_decode", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream2_i2i_summary.get("disk_jpeg_decode", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream2_i2i_summary.get("disk_jpeg_decode", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream2_i2i_summary.get("disk_jpeg_decode", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream2_i2i_summary.get("disk_jpeg_decode", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "CIR catalog",
            "config_id": "cir_v1_01",
            "phase5_clean_p95_ms": round(float(p5_hist["flows"]["cir_catalog"]["p95_ms"]), 3),
            "prompt14e_p95_ms": p14e_pal["stream3_cir_catalog"]["cir_v1_01"]["p95_median_ms"],
            "prompt14f_p50_ms": stream3_cir_catalog_summary.get("cir_v1_01", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream3_cir_catalog_summary.get("cir_v1_01", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream3_cir_catalog_summary.get("cir_v1_01", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream3_cir_catalog_summary.get("cir_v1_01", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream3_cir_catalog_summary.get("cir_v1_01", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream3_cir_catalog_summary.get("cir_v1_01", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "CIR catalog",
            "config_id": "cir_v1_05",
            "phase5_clean_p95_ms": "không xác minh được (chưa có C3 bonus)",
            "prompt14e_p95_ms": p14e_pal["stream3_cir_catalog"]["cir_v1_05"]["p95_median_ms"],
            "prompt14f_p50_ms": stream3_cir_catalog_summary.get("cir_v1_05", {}).get("p50_median_ms", "N/A"),
            "prompt14f_p95_median_ms": stream3_cir_catalog_summary.get("cir_v1_05", {}).get("p95_median_ms", "N/A"),
            "prompt14f_p95_iqr_ms": stream3_cir_catalog_summary.get("cir_v1_05", {}).get("p95_iqr_ms", "N/A"),
            "prompt14f_p99_ms": stream3_cir_catalog_summary.get("cir_v1_05", {}).get("p99_median_ms", "N/A"),
            "sla_under_200ms": stream3_cir_catalog_summary.get("cir_v1_05", {}).get("sla_p95_under_200ms", False),
            "conclusion_label": stream3_cir_catalog_summary.get("cir_v1_05", {}).get("conclusion_label", "N/A"),
        },
        {
            "stream": "CIR upload (Số chính: RAM PIL)",
            "config_id": "cir_v1_01",
            "phase5_clean_p95_ms": round(float(p5_hist["flows"]["cir_upload"]["p95_ms"]), 3),
            "prompt14e_p95_ms": p14e_pal["stream4_cir_upload"]["cir_v1_01"]["primary_ram_preloaded"]["p95_median_ms"],
            "prompt14f_p50_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p50_median_ms"],
            "prompt14f_p95_median_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p95_median_ms"],
            "prompt14f_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p95_iqr_ms"],
            "prompt14f_p99_ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["sla_p95_under_200ms"],
            "conclusion_label": stream4_cir_upload_summary["cir_v1_01"]["primary_ram_preloaded"]["conclusion_label"],
        },
        {
            "stream": "CIR upload (Số chính: RAM PIL)",
            "config_id": "cir_v1_05",
            "phase5_clean_p95_ms": "không xác minh được (chưa có C3 bonus)",
            "prompt14e_p95_ms": p14e_pal["stream4_cir_upload"]["cir_v1_05"]["primary_ram_preloaded"]["p95_median_ms"],
            "prompt14f_p50_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p50_median_ms"],
            "prompt14f_p95_median_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p95_median_ms"],
            "prompt14f_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p95_iqr_ms"],
            "prompt14f_p99_ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["sla_p95_under_200ms"],
            "conclusion_label": stream4_cir_upload_summary["cir_v1_05"]["primary_ram_preloaded"]["conclusion_label"],
        },
        {
            "stream": "CIR upload (Số phụ: Disk+JPEG)",
            "config_id": "cir_v1_01",
            "phase5_clean_p95_ms": "không xác minh được",
            "prompt14e_p95_ms": p14e_pal["stream4_cir_upload"]["cir_v1_01"]["secondary_disk_jpeg_decode"]["p95_median_ms"],
            "prompt14f_p50_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p50_median_ms"],
            "prompt14f_p95_median_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p95_median_ms"],
            "prompt14f_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p95_iqr_ms"],
            "prompt14f_p99_ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["sla_p95_under_200ms"],
            "conclusion_label": stream4_cir_upload_summary["cir_v1_01"]["secondary_disk_jpeg_decode"]["conclusion_label"],
        },
        {
            "stream": "CIR upload (Số phụ: Disk+JPEG)",
            "config_id": "cir_v1_05",
            "phase5_clean_p95_ms": "không xác minh được",
            "prompt14e_p95_ms": p14e_pal["stream4_cir_upload"]["cir_v1_05"]["secondary_disk_jpeg_decode"]["p95_median_ms"],
            "prompt14f_p50_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p50_median_ms"],
            "prompt14f_p95_median_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p95_median_ms"],
            "prompt14f_p95_iqr_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p95_iqr_ms"],
            "prompt14f_p99_ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["p99_median_ms"],
            "sla_under_200ms": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["sla_p95_under_200ms"],
            "conclusion_label": stream4_cir_upload_summary["cir_v1_05"]["secondary_disk_jpeg_decode"]["conclusion_label"],
        },
    ]

    # Optimization candidates for user decision
    opt_candidates = [
        "1. Tránh chuyển đổi tensor lặp đi lặp lại giữa CPU và GPU trong parser và combiner.",
        "2. Batch hóa tiền xử lý SigLIP hoặc dùng pipeline xử lý bất đồng bộ cho Disk JPEG decode.",
        "3. Cache kết quả phân tích cú pháp query (diacritics + parser slots) cho các truy vấn phổ biến.",
        "Mọi tối ưu hóa ở pha riêng phải tuân thủ Luật 4: cosine >= 0.999 so với embedding checkpoint và Top-10 không đổi trên val.",
    ]

    # Build final results dict
    results_payload: Dict[str, Any] = {
        "status": "PART_A2_DONE_AWAITING_HELDOUT_V2" if halted_stream_name is None else f"HALTED_AFTER_{halted_stream_name.upper()}",
        "prompt": "14f",
        "branch": "phase7a",
        "git_commit": get_git_commit(),
        "seed": SEED_LOCKED,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "step_1_cir_user_decision": cir_decision_data,
        "step_1_cir_user_decision_sha256": cir_decision_sha,
        "step_2_erratum_14e": erratum_14e,
        "step_3_environment": env_snapshot,
        "heldout_v2_check": heldout_v2_info,
        "step_0_input_hashes": {
            "files": {k: {"path": p, "expected_sha256": exp, "actual_sha256": act, "matched": (exp == act)} for k, (p, exp, act) in step0_hashes.items()}
        },
        "step_0_3_preprocess_parity_200": step0_3_res,
        "step_1a_index_hash_guard": step1a_res,
        "step_1c_caption_limitation": step1c_res,
        "part_a2_latency": {
            "gallery_dir": "index/test_gallery",
            "n_gallery": len(engine.embeddings),
            "ef_search": 128,
            "candidate_pool": candidate_pool,
            "queries_metadata": {
                "t2i": {k: v for k, v in t2i_q_info.items() if k != "queries_200"},
                "cir_and_i2i": {k: v for k, v in cir_p_info.items() if k != "pairs_200"},
            },
            "stream4_cir_upload": stream4_cir_upload_summary,
            "stream2_image_to_image": stream2_i2i_summary,
            "stream3_cir_catalog": stream3_cir_catalog_summary,
            "stream1_text_to_image": stream1_t2i_summary,
            "latency_conclusion_primary_ram": conclusion_ram,
            "latency_conclusion_secondary_disk": conclusion_disk,
            "comparison_table": comparison_table,
            "halted_stream": halted_stream_name,
        },
        "step_5_phase5_comparison": {
            "comparison_table": step5_comparison,
            "diagnostic_conclusions": diag_lines,
            "optimization_candidates": opt_candidates,
            "phase5_eval_latency_today": p5_today_data,
        },
    }

    # Generate Markdown Report automatically from JSON dictionary
    report_md = generate_markdown_report_14f(results_payload)

    # Save outputs
    res_path = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    rep_path = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"

    res_content = json.dumps(results_payload, indent=2, ensure_ascii=False)
    res_path.write_text(res_content, encoding="utf-8")
    rep_path.write_text(report_md, encoding="utf-8")

    # Hash outputs
    res_sha = hashlib.sha256(res_content.encode("utf-8")).hexdigest()
    rep_sha = hashlib.sha256(report_md.encode("utf-8")).hexdigest()
    (PROJECT_ROOT / "reports/phase5c/final_eval_results.sha256").write_text(f"{res_sha}  final_eval_results.json\n", encoding="utf-8")
    (PROJECT_ROOT / "reports/phase5c/final_eval_report.sha256").write_text(f"{rep_sha}  final_eval_report.md\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("SUCCESS: Prompt 14f Part A2 completed.")
    print(f"Results JSON: {res_path} (SHA-256: {res_sha})")
    print(f"Report Markdown: {rep_path} (SHA-256: {rep_sha})")
    print("=" * 80, flush=True)


def generate_markdown_report_14f(d: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Báo cáo Đánh giá Cuối cùng — Pha 5c (Prompt 14f: `{d['status']}`)")
    lines.append("")
    lines.append("## 0. Thông tin chung & Bất biến")
    lines.append("")
    lines.append(f"- **Trạng thái thực thi:** `{d['status']}`")
    lines.append(f"- **Thời gian tạo (UTC):** `{d['created_at_utc']}`")
    lines.append(f"- **Git commit khi chạy script:** `{d['git_commit']}`")
    lines.append(f"- **Branch:** `{d['branch']}`")
    lines.append(f"- **Seed khóa cứng:** `{d['seed']}`")
    lines.append(f"- **Agent Python PID:** `{d['step_3_environment']['agent_python_pid']}`")
    lines.append(f"- **Tệp `eval/test_queries_heldout_v2.json` tồn tại:** `{d['heldout_v2_check']['exists']}` ({d['heldout_v2_check']['note']})")
    lines.append(
        "- **Cam kết Luật 7 & C.5:** Không chạy bất kỳ phép đo chất lượng (Recall@K, mAP, Hubness) nào trên "
        "`index/test_gallery/` (Phần B) và không mở/đọc nội dung `eval/cir_eval_test.json` (Phần C). "
        "Chỉ chạy Bước 1 (Chốt CIR), Bước 2 (Erratum 14e), Bước 3 (Môi trường), Bước 4 (Phần A2 C.9), và Bước 5 (Đối chiếu Pha 5) rồi DỪNG."
    )
    lines.append("")

    # Section 1: CIR User Decision
    dec = d["step_1_cir_user_decision"]
    lines.append("## 1. Bước 1 — Ghi Quyết định CIR của Người dùng")
    lines.append("")
    lines.append(f"- **Tệp quyết định:** `reports/phase5c/cir_user_decision_prompt14f.json`")
    lines.append(f"- **SHA-256 của tệp quyết định:** `{d['step_1_cir_user_decision_sha256']}`")
    lines.append(f"- **Thời gian ghi (UTC):** `{dec['timestamp_iso']}`")
    lines.append(f"- **Git commit:** `{dec['git_commit']}`")
    lines.append(f"- **Quyết định (1):** `{dec['decision_content']}`")
    lines.append(f"- **Cấu hình Hệ thống CIR:** `{dec['system_cir_config']}`")
    lines.append(f"- **Cấu hình Model thuần CIR:** `{dec['model_only_cir_config']}`")
    lines.append(f"- **Xác nhận:** *\"{dec['confirmation']}\"*")
    lines.append("")

    # Section 2: Erratum 14e
    e14 = d["step_2_erratum_14e"]
    lines.append("## 2. Bước 2 — Erratum Phần A của Prompt 14e")
    lines.append("")
    ea = e14["item_a_temperature"]
    lines.append(f"### 2.1. Nhiệt độ bắt đầu run trong 14e (`{ea['runs_starting_ge_60c']}/{ea['total_runs_measured']}` run >= 60°C)")
    lines.append(f"- **Bằng chứng từ JSON 14e:** Nhiệt độ min = `{ea['temp_min']}°C`, trung vị = `{ea['temp_median']}°C`, max = `{ea['temp_max']}°C`.")
    lines.append(f"- **Kết luận đính chính:** {ea['explanation']}")
    lines.append("")

    eb = e14["item_b_background_load"]
    lines.append("### 2.2. Nguồn tải GPU nền")
    lines.append(f"- **Bằng chứng từ JSON 14e:** {eb['evidence']} (14d median `{eb['idle_util_14d_median_pct']}%` -> trước Phần A `{eb['idle_util_14e_before_pct']}%` -> sau Phần A `{eb['idle_util_14e_after_pct']}%`).")
    lines.append(f"- **Kết luận đính chính:** {eb['dismissed_claim']} Ghi nhận chính thức: **\"{eb['remaining_background_load_source']}\"**.")
    lines.append("")

    ec = e14["item_c_telemetry_duplication"]
    lines.append("### 2.3. Dữ liệu Telemetry trùng lặp giữa biến thể RAM và Disk trong 14e")
    lines.append(f"- **Kiểm tra mã nguồn (`{ec['script_file']}`):** {ec['code_line_evidence']}.")
    lines.append(f"- **Kết luận đính chính:** {ec['validity']}")
    lines.append("")

    ed = e14["item_d_map_numbers"]
    lines.append("### 2.4. Xác minh hai số mAP `0.5471` và `0.4077` xuất hiện trong chat 14e")
    lines.append(f"- **Số `0.4077`:** {ed['value_0_4077']['exact_metric']} trong `{ed['value_0_4077']['source_file']}`. {ed['value_0_4077']['note']}")
    lines.append(f"- **Số `0.5471`:** Nguồn `{ed['value_0_5471']['source_file']}`. {ed['value_0_5471']['exact_metric']} -> Trạng thái: **`{ed['value_0_5471']['status']}`**.")
    lines.append(f"- **Số mAP đã xác minh của CIR trên tập val:** `cir_v1_05` = **`{ed['verified_val_cir_map']['cir_v1_05']}`** vs `cir_v1_01` = **`{ed['verified_val_cir_map']['cir_v1_01']}`** (`{ed['verified_val_cir_map']['source_report']}`).")
    lines.append("")

    ee = e14["item_e_phase5_gap"]
    lines.append("### 2.5. Chênh lệch độ trễ giữa 14e và Pha 5")
    lines.append(f"- **Độ lệch p95:** T2I model_only (`{ee['t2i_model_only_p95_gap']}`), CIR catalog cir_v1_01 (`{ee['cir_catalog_cir_v1_01_p95_gap']}`), I2I (`{ee['i2i_ram_p95_gap']}`), CIR upload (`{ee['cir_upload_ram_p95_gap']}`).")
    lines.append(f"- **Kết luận đính chính:** {ee['status']}")
    lines.append("")

    ef = e14["item_f_p99_warning"]
    lines.append(f"### 2.6. Cảnh báo p99 cho `cfg_v5_01` (`p99 = {ef['cfg_v5_01_p99_median_ms']:.3f} ms`)")
    lines.append(f"- **Ghi nhận:** {ef['rule_scope']}")
    lines.append("")

    # Section 3: Environment
    env = d["step_3_environment"]
    lines.append("## 3. Bước 3 — Môi trường Đo đạc (Chỉ đọc)")
    lines.append("")
    lines.append(f"- **Windows Power Scheme:** `{env['power_scheme']}`")
    lines.append(f"- **Nguồn điện:** `{env['power_status_ac_battery']}`")
    lines.append(f"- **CPU:** `{env['cpu_info']['physical_cores']}` cores vật lý, `{env['cpu_info']['logical_cores']}` luồng; xung nhịp `{env['cpu_info']['frequency_mhz']}`; sử dụng `{env['cpu_info']['utilization_pct']}%`.")
    lines.append(f"- **GPU Query & Throttle Snapshot:** `{env['gpu_power_and_throttles']}`")
    lines.append(f"- **Ngữ cảnh thực thi:** `{env['execution_context']}` (Agent PID: `{env['agent_python_pid']}`)")
    lines.append("")

    # Section 4: Part A2 Latency Benchmark
    pa2 = d["part_a2_latency"]
    lines.append("## 4. Bước 4 — Kết quả Độ trễ Phần A2 (Giao thức C.9 Nghiêm ngặt: Làm nguội < 60°C & Đo riêng)")
    lines.append("")
    lines.append(f"- **Gallery thử nghiệm:** `{pa2['gallery_dir']}` (`N = {pa2['n_gallery']}`), `efSearch = {pa2['ef_search']}`, `candidate_pool = {pa2['candidate_pool']}`.")
    lines.append(f"- **Thứ tự ưu tiên thực thi:** (1) CIR upload [RAM trước, rồi Disk], (2) I2I [RAM trước, rồi Disk], (3) CIR catalog, (4) Text→Ảnh.")
    lines.append(f"- **Trạng thái dừng:** `{pa2['halted_stream'] or 'Không có (Hoàn thành cả 4 luồng)'}`")
    lines.append("")

    lines.append("### 4.1. Bảng Tổng hợp Độ trễ Đối chiếu (Pha 5 vs 14e vs 14f)")
    lines.append("")
    lines.append("| Luồng | Cấu hình | Pha 5 `p95` (ms) | 14e `p95` (ms) | 14f `p50` (ms) | 14f `p95` Median (ms) | 14f `p95` IQR (ms) | 14f `p99` (ms) | SLA < 200ms | Kết luận Khóa |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|:---:|:---|")
    for r in pa2["comparison_table"]:
        p5_val = f"{r['phase5_clean_p95_ms']:.3f}" if isinstance(r["phase5_clean_p95_ms"], (int, float)) else str(r["phase5_clean_p95_ms"])
        p14e_val = f"{r['prompt14e_p95_ms']:.3f}" if isinstance(r["prompt14e_p95_ms"], (int, float)) else str(r["prompt14e_p95_ms"])
        p50_val = f"{r['prompt14f_p50_ms']:.3f}" if isinstance(r["prompt14f_p50_ms"], (int, float)) else str(r["prompt14f_p50_ms"])
        p95_val = f"{r['prompt14f_p95_median_ms']:.3f}" if isinstance(r["prompt14f_p95_median_ms"], (int, float)) else str(r["prompt14f_p95_median_ms"])
        iqr_val = f"{r['prompt14f_p95_iqr_ms']:.3f}" if isinstance(r["prompt14f_p95_iqr_ms"], (int, float)) else str(r["prompt14f_p95_iqr_ms"])
        p99_val = f"{r['prompt14f_p99_ms']:.3f}" if isinstance(r["prompt14f_p99_ms"], (int, float)) else str(r["prompt14f_p99_ms"])
        sla_box = "PASS" if r["sla_under_200ms"] else "FAIL"
        lines.append(
            f"| {r['stream']} | `{r['config_id']}` | {p5_val} | {p14e_val} | {p50_val} | {p95_val} | {iqr_val} | {p99_val} | {sla_box} | {r['conclusion_label']} |"
        )
    lines.append("")

    # Section 4.2: Stream 4 Stage breakdown
    s4 = pa2["stream4_cir_upload"]
    lines.append("### 4.2. Bảng Tách 6 Giai đoạn Luồng 4 (`CIR upload`, có `torch.cuda.synchronize()` quanh mỗi bước)")
    lines.append("")
    lines.append("| Cấu hình | Giai đoạn | `p50` (ms) | `p95` (ms) | `p99` (ms) | `Mean` (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    stage_labels = {
        "stage1_disk_jpeg_decode": "1. Đọc đĩa + Giải mã JPEG (chỉ có ở Disk)",
        "stage2_resize_processor": "2. Resize 224 BILINEAR + SiglipProcessor",
        "stage3_vision_tower_forward": "3. Forward Vision Tower (SigLIP ViT-SO400M)",
        "stage4_text_tower_and_parser": "4. Forward Text Tower + Parser v2.1",
        "stage5_combiner_forward": "5. Forward CIR Combiner",
        "stage6_hnsw_and_c3_bonus": "6. Tìm kiếm FAISS HNSW + C3 Attribute Bonus",
    }
    for cid in ("cir_v1_01", "cir_v1_05"):
        sb = s4[cid]["stage_breakdown_ms"]
        for sn in [
            "stage1_disk_jpeg_decode",
            "stage2_resize_processor",
            "stage3_vision_tower_forward",
            "stage4_text_tower_and_parser",
            "stage5_combiner_forward",
            "stage6_hnsw_and_c3_bonus",
        ]:
            st = sb[sn]
            lines.append(
                f"| `{cid}` | {stage_labels[sn]} | {st['p50_ms']:.3f} | {st['p95_ms']:.3f} | {st['p99_ms']:.3f} | {st['mean_ms']:.3f} |"
            )
    lines.append("")

    # Section 4.3: Per-run breakdown and Official/Unofficial run counts
    lines.append("### 4.3. Thống kê Chi tiết Từng Run & Số run Chính thức / Không chính thức")
    lines.append("")
    lines.append("| Luồng & Cấu hình | Run # | Trạng thái | Nhiệt độ Bắt đầu (°C) | Chờ Cooldown (s) | `p50` (ms) | `p95` (ms) | `p99` (ms) | % Throttled |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    def _add_run_rows(stream_lbl: str, summary_obj: Dict[str, Any]):
        if not summary_obj:
            return
        for r_stat in summary_obj.get("per_run", []):
            tele = r_stat["telemetry"]
            lines.append(
                f"| {stream_lbl} | {r_stat['run']} | {r_stat['label']} | {tele['starting_temp_c']} | {tele['cooldown_wait_s']} | {r_stat['p50_ms']:.3f} | {r_stat['p95_ms']:.3f} | {r_stat['p99_ms']:.3f} | {tele['pct_throttled']}% |"
            )

    _add_run_rows("CIR upload (RAM)", s4["cir_v1_05"]["primary_ram_preloaded"])
    _add_run_rows("CIR upload (Disk)", s4["cir_v1_05"]["secondary_disk_jpeg_decode"])
    if pa2["stream2_image_to_image"]:
        _add_run_rows("I2I (RAM)", pa2["stream2_image_to_image"]["ram_preloaded"])
        _add_run_rows("I2I (Disk)", pa2["stream2_image_to_image"]["disk_jpeg_decode"])
    if pa2["stream3_cir_catalog"]:
        _add_run_rows("CIR catalog (cir_v1_05)", pa2["stream3_cir_catalog"]["cir_v1_05"])
    if pa2["stream1_text_to_image"]:
        _add_run_rows("T2I (cfg_v5_09)", pa2["stream1_text_to_image"]["cfg_v5_09"])
    lines.append("")

    # Section 5: Step 5 Phase 5 Diagnostic Benchmark Parity
    s5 = d["step_5_phase5_comparison"]
    lines.append("## 5. Bước 5 — Đối chiếu Chẩn đoán Pha 5 (Chạy Nguyên vẹn `src/eval/eval_latency.py` trên `index/test_gallery/`)")
    lines.append("")
    lines.append("| Luồng | Cấu hình | Số Pha 5 Cũ (`p95` ms) | `eval_latency.py` Hôm nay (`p95` ms) | Harness 14f (`p95` median ms) |")
    lines.append("|---|---|---:|---:|---:|")
    for row in s5["comparison_table"]:
        p5_c = f"{row['phase5_historical_p95_ms']:.3f}"
        p5_t = f"{row['eval_latency_today_p95_ms']:.3f}" if isinstance(row['eval_latency_today_p95_ms'], (int, float)) else str(row['eval_latency_today_p95_ms'])
        h14f = f"{row['harness_14f_p95_median_ms']:.3f}" if isinstance(row['harness_14f_p95_median_ms'], (int, float)) else str(row['harness_14f_p95_median_ms'])
        lines.append(f"| {row['flow_name']} | `{row['config_note']}` | {p5_c} | {p5_t} | {h14f} |")
    lines.append("")

    lines.append("### 5.1. Diễn giải Kết quả Chẩn đoán Trong Giới hạn Bằng chứng")
    lines.append("")
    for dl in s5["diagnostic_conclusions"]:
        lines.append(dl)
    lines.append("")

    lines.append("### 5.2. Các Hướng Tối ưu Khả thi (Ứng viên để Người dùng Quyết định ở Pha Riêng)")
    lines.append("")
    for oc in s5["optimization_candidates"]:
        lines.append(f"- {oc}")
    lines.append("")

    # Section 6: Conclusions
    c_ram = pa2["latency_conclusion_primary_ram"]
    c_disk = pa2["latency_conclusion_secondary_disk"]
    lines.append("## 6. Kết luận Độ trễ Cuối cùng")
    lines.append("")
    lines.append(
        f"- **Theo Số chính (`RAM PIL`):** Trạng thái = `{c_ram['status']}`, `{c_ram['passed_count']}/{c_ram['total_streams']}` luồng đạt — **{c_ram['label_vi']}**."
    )
    lines.append(
        f"- **Theo Số phụ (`Disk + JPEG decode`):** Trạng thái = `{c_disk['status']}`, `{c_disk['passed_count']}/{c_disk['total_streams']}` luồng đạt — **{c_disk['label_vi']}**."
    )
    lines.append(
        f"- **Cảnh báo `p99` Luồng CIR upload:** `{s4['cir_v1_05']['primary_ram_preloaded']['warn_upload_p99_over_200ms']}` "
        f"(Pooled p99 RAM = `{s4['cir_v1_05']['primary_ram_preloaded']['pooled_p99_ms']:.3f} ms`, "
        f"Disk = `{s4['cir_v1_05']['secondary_disk_jpeg_decode']['pooled_p99_ms']:.3f} ms`)."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
