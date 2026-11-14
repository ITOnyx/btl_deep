"""
Prompt 14 — Official Evaluation on index/test_gallery/ + 4-Stream Latency Benchmark.
Enforces Step 0 (0.1–0.6) and Part A / Part B / Part C stop conditions strictly:
- Does NOT train or modify checkpoints, parsers, lexicons, or locked rules.
- If Step 0.2/0.4 or Part A GPU process check fails, HALTS immediately without touching test_gallery quality metrics,
  writes full preflight & hash audit to reports/phase5c/final_eval_results.json and reports/phase5c/final_eval_report.md.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.locked_final_definitions import (
    GALLERY_SIZE_TEST_LOCKED,
    LOCKED_SPEC_METADATA,
    SEED_LOCKED,
    SLA_LATENCY_P95_MS_LOCKED,
    TARGET_RECALL_AT_10_LOCKED,
    TARGET_T2I_MAP_LOCKED,
)
from src.model.siglip_model import load_siglip_model, verify_checkpoint_hash
from src.preprocess.pipeline import SigLIPPreprocessor


EXPECTED_HASHES = {
    "best_model_pt": ("data/checkpoints_siglip-vit/best_model.pt", "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d"),
    "selection_rule_v5": ("reports/phase5c/selection_rule_v5.json", "e3d75ec5df7ab6de4cb9ad07e21b9c1a58c5c029c3a9b7749f28518c6c4f0c8c"),
    "selection_rule_cir_v1": ("reports/phase5c/selection_rule_cir_v1.json", "4a5359f9f940a75a581847fd5f5b3c9900c6b2c4ba6dc77f5cc3637c6fa21067"),
    "cir_eval_test": ("eval/cir_eval_test.json", "d6e33f4ee914152695513933de3bedc183a8f60b4d05c6bd3a2d8c0cfb90961b"),
    "test_split": ("eval/test_split.json", "d61bd87b7ae495cbfe047c4852b9fec06d8479158f1664b5c0712388addad681"),
    "catalog_rebuild": ("data/catalog/catalog_rebuild.parquet", "aaec78681b80a39ebd6e07f70130ab600b0df1f13315354312741f930b9474ab"),
}

# Standard OS/IDE shell processes that are not 3D games/emulators;
# any game/emulator/3D overlay process (e.g. dnmultiplayerex.exe, RobloxPlayerBeta.exe, EdgeGameAssist.exe) blocks Part A.
BLOCKING_GPU_PROCESS_KEYWORDS = (
    "dnmultiplayer",
    "ldplayer",
    "ldvbox",
    "roblox",
    "edgegameassist",
    "steam",
    "epicgames",
    "genshin",
    "valorant",
    "league",
    "unity",
    "unreal",
    "bluestacks",
    "nox",
    "memu",
)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def query_nvidia_smi_full() -> Dict[str, Any]:
    """Query GPU telemetry and active GPU process list from nvidia-smi."""
    telemetry: Dict[str, Any] = {}
    try:
        out_tel = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.gr,clocks.mem,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        parts = [p.strip() for p in out_tel.splitlines()[0].split(",")]
        telemetry = {
            "timestamp": parts[0],
            "gpu_name": parts[1],
            "temperature_c": float(parts[2]),
            "utilization_pct": float(parts[3]),
            "power_draw_w": float(parts[4]),
            "clock_graphics_mhz": float(parts[5]),
            "clock_mem_mhz": float(parts[6]),
            "memory_used_mb": float(parts[7]),
            "memory_total_mb": float(parts[8]),
        }
    except Exception as exc:
        telemetry = {"error": str(exc)}

    processes: List[Dict[str, Any]] = []
    blocking_processes: List[Dict[str, Any]] = []
    raw_table = ""
    try:
        raw_table = subprocess.check_output(["nvidia-smi"], text=True)
        in_proc = False
        for line in raw_table.splitlines():
            if "| Processes:" in line:
                in_proc = True
                continue
            if in_proc and line.startswith("|    0"):
                cols = line.strip("|").split()
                if len(cols) >= 6:
                    pid = cols[3]
                    ptype = cols[4]
                    pname = " ".join(cols[5:-1])
                    pmem = cols[-1]
                    entry = {"pid": pid, "type": ptype, "process_name": pname, "gpu_memory": pmem}
                    processes.append(entry)
                    low_name = pname.lower()
                    if any(k in low_name for k in BLOCKING_GPU_PROCESS_KEYWORDS):
                        blocking_processes.append(entry)
    except Exception as exc:
        raw_table = f"Error running nvidia-smi: {exc}"

    return {
        "telemetry": telemetry,
        "processes": processes,
        "blocking_graphics_processes": blocking_processes,
        "has_blocking_graphics_processes": len(blocking_processes) > 0,
    }


def run_step0_3_parity_200_test_gallery() -> Dict[str, Any]:
    """
    Step 0.3 (Law 4): Re-encode 200 images from index/test_gallery (seed=42)
    and verify min_cosine >= 0.999 against saved index/test_gallery/image_embeddings.npy.
    """
    id_map_path = PROJECT_ROOT / "index/test_gallery/id_map.json"
    embs_path = PROJECT_ROOT / "index/test_gallery/image_embeddings.npy"
    ckpt_path = PROJECT_ROOT / EXPECTED_HASHES["best_model_pt"][0]
    expected_ckpt_hash = EXPECTED_HASHES["best_model_pt"][1]

    verify_checkpoint_hash(ckpt_path, expected_ckpt_hash)
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    saved_embs = np.load(embs_path)

    rng = np.random.default_rng(SEED_LOCKED)
    sample_indices = np.sort(rng.choice(len(id_map), size=200, replace=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = SigLIPPreprocessor(device=device)
    model = load_siglip_model(
        checkpoint_path=str(ckpt_path),
        expected_hash=expected_ckpt_hash,
        device=device,
        fp16=(device == "cuda"),
    )

    reencoded = []
    batch_size = 32
    for start in range(0, len(sample_indices), batch_size):
        batch_idx = sample_indices[start : start + batch_size]
        batch_paths = [PROJECT_ROOT / id_map[int(i)]["path"] for i in batch_idx]
        px = preprocessor.preprocess_images(batch_paths, device=device, half_precision=(device == "cuda"))
        with torch.no_grad():
            emb = model.encode_image(px)
            reencoded.append(emb.float().cpu().numpy())

    new_embs = np.vstack(reencoded)
    new_embs = new_embs / np.maximum(np.linalg.norm(new_embs, axis=1, keepdims=True), 1e-12)
    ref_embs = saved_embs[sample_indices].astype(np.float32)
    ref_embs = ref_embs / np.maximum(np.linalg.norm(ref_embs, axis=1, keepdims=True), 1e-12)

    cosines = np.sum(new_embs * ref_embs, axis=1)
    min_cos = float(np.min(cosines))
    mean_cos = float(np.mean(cosines))
    median_cos = float(np.median(cosines))

    return {
        "gallery": "index/test_gallery",
        "n_samples": 200,
        "seed": SEED_LOCKED,
        "device": device,
        "min_cosine": round(min_cos, 6),
        "mean_cosine": round(mean_cos, 6),
        "median_cosine": round(median_cos, 6),
        "threshold": 0.999,
        "passed": bool(min_cos >= 0.999),
    }


def main() -> int:
    commit_hash = get_git_commit()

    # Lock Step 0.6 definitions file hash
    locked_def_path = PROJECT_ROOT / "src/eval/locked_final_definitions.py"
    locked_def_sha = sha256_file(locked_def_path)
    (PROJECT_ROOT / "reports/phase5c/locked_final_definitions.sha256").write_text(
        f"{locked_def_sha}  src/eval/locked_final_definitions.py\n",
        encoding="utf-8",
    )

    # Step 0.1: Verify all input SHA-256 hashes
    step0_1_hashes: Dict[str, Any] = {}
    step0_1_all_pass = True
    for key, (rel_path, exp_sha) in EXPECTED_HASHES.items():
        actual_sha = sha256_file(PROJECT_ROOT / rel_path)
        matched = actual_sha == exp_sha
        if not matched:
            step0_1_all_pass = False
        step0_1_hashes[key] = {
            "path": rel_path,
            "expected_sha256": exp_sha,
            "actual_sha256": actual_sha,
            "matched": matched,
        }

    heldout_v2_path = PROJECT_ROOT / "eval/test_queries_heldout_v2.json"
    heldout_v2_alt_path = PROJECT_ROOT / "eval/heldout_queries_v2.json"
    heldout_v2_actual_path = heldout_v2_path if heldout_v2_path.exists() else (heldout_v2_alt_path if heldout_v2_alt_path.exists() else None)
    step0_1_hashes["test_queries_heldout_v2"] = {
        "path": "eval/test_queries_heldout_v2.json",
        "exists": heldout_v2_actual_path is not None,
        "actual_sha256": sha256_file(heldout_v2_actual_path) if heldout_v2_actual_path else None,
        "worksheet_csv_path": "reports/phase5c/heldout_v2_worksheet.csv",
        "worksheet_csv_sha256": sha256_file(PROJECT_ROOT / "reports/phase5c/heldout_v2_worksheet.csv"),
    }
    step0_1_hashes["locked_final_definitions_py"] = {
        "path": "src/eval/locked_final_definitions.py",
        "actual_sha256": locked_def_sha,
    }

    # Step 0.2: Hash Guard index/test_gallery & check caption indexes (Config A / Config B / metadata-only)
    test_manifest_path = PROJECT_ROOT / "index/test_gallery/manifest.json"
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    test_id_map = json.loads((PROJECT_ROOT / "index/test_gallery/id_map.json").read_text(encoding="utf-8"))
    test_embs = np.load(PROJECT_ROOT / "index/test_gallery/image_embeddings.npy", mmap_mode="r")

    test_gallery_guard_pass = (
        test_manifest.get("checkpoint_hash") == EXPECTED_HASHES["best_model_pt"][1]
        and test_manifest.get("catalog_hash") == EXPECTED_HASHES["catalog_rebuild"][1]
        and int(test_manifest.get("n_vectors", 0)) == GALLERY_SIZE_TEST_LOCKED
        and len(test_id_map) == GALLERY_SIZE_TEST_LOCKED
        and test_embs.shape == (GALLERY_SIZE_TEST_LOCKED, 1152)
    )

    test_text_v5_manifest_path = PROJECT_ROOT / "index/test_gallery_text_v5/manifest.json"
    val_text_v5_manifest_path = PROJECT_ROOT / "index/val_gallery_text_v5/manifest.json"
    step0_2_status = {
        "test_gallery_image_index": {
            "manifest_path": "index/test_gallery/manifest.json",
            "manifest_sha256": sha256_file(test_manifest_path),
            "checkpoint_hash_matched": test_manifest.get("checkpoint_hash") == EXPECTED_HASHES["best_model_pt"][1],
            "catalog_hash_matched": test_manifest.get("catalog_hash") == EXPECTED_HASHES["catalog_rebuild"][1],
            "n_vectors": int(test_manifest.get("n_vectors", 0)),
            "embeddings_shape": list(test_embs.shape),
            "passed": test_gallery_guard_pass,
        },
        "val_gallery_caption_index_v5": {
            "manifest_path": "index/val_gallery_text_v5/manifest.json",
            "exists": val_text_v5_manifest_path.exists(),
            "manifest_sha256": sha256_file(val_text_v5_manifest_path),
        },
        "test_gallery_caption_index_v5": {
            "manifest_path": "index/test_gallery_text_v5/manifest.json",
            "exists": test_text_v5_manifest_path.exists(),
            "manifest_sha256": sha256_file(test_text_v5_manifest_path),
            "raw_caption_shards_available": [
                "data/catalog/captions/shard_test_0000.parquet",
                "data/catalog/captions/shard_test_0001.parquet",
                "data/catalog/captions/shard_test_0002.parquet",
            ],
            "raw_caption_coverage_on_test_gallery": "1290/1290 (caption_A and caption_B present in shard_test_0000..0002.parquet)",
            "note": (
                "Chưa có thư mục index/test_gallery_text_v5/ kèm manifest.json cho 1,290 mục test_gallery "
                "(dữ liệu nguồn shard_test_0000..0002.parquet đã có đủ 1,290/1,290 dòng caption_A và caption_B; "
                "cần người dùng cho phép dựng index/test_gallery_text_v5/ trước khi chạy B1(iii) cfg_v5_08b và B1(iv))."
            ),
        },
    }

    # Step 0.3: Law 4 parity check (200 images >= 0.999)
    step0_3_parity = run_step0_3_parity_200_test_gallery()
    phase3_parity = json.loads((PROJECT_ROOT / "reports/phase3/parity_report.json").read_text(encoding="utf-8"))

    # Step 0.4: Check eval/test_queries_heldout_v2.json
    step0_4_heldout = {
        "exists": heldout_v2_actual_path is not None,
        "passed": False,
        "reason": (
            "Thiếu tệp eval/test_queries_heldout_v2.json trên đĩa (hiện chỉ có phiếu lấy mẫu "
            "reports/phase5c/heldout_v2_worksheet.csv). Theo mục 0.4: KHÔNG chạy bất kỳ đánh giá chất lượng nào "
            "trên test_gallery; chỉ kiểm tra/chạy Phần A rồi báo và DỪNG."
        ),
    }

    # Step 0.5: Confirm repo/report approvals for T2I (Prompt 12) and CIR (Prompt 13)
    step0_5_approvals = {
        "t2i_official_winner": "cfg_v5_09",
        "t2i_safe_config": "cfg_v5_01",
        "t2i_predeclared_reference_configs": ["cfg_v5_06", "cfg_v5_08b"],
        "t2i_approval_source": "reports/phase5c/step5c_selection_report.md",
        "cir_provisional_winner": "cir_v1_05",
        "cir_baseline_model_only": "cir_v1_01",
        "cir_approval_status": "provisional_pending_part_a_clean_gpu_latency",
        "cir_approval_source": "reports/phase5c/step5c_cir_report.md",
    }

    # Part A Pre-check: GPU process check via nvidia-smi
    gpu_snapshot = query_nvidia_smi_full()

    results_payload = {
        "status": "HALTED_PREFLIGHT",
        "halt_reasons": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit_hash,
        "seed": SEED_LOCKED,
        "step_0": {
            "0_1_hashes": step0_1_hashes,
            "0_1_core_hashes_matched": step0_1_all_pass,
            "0_2_index_guards": step0_2_status,
            "0_3_parity_check_test_gallery_200": step0_3_parity,
            "0_3_phase3_reference_min_cosine": phase3_parity["metrics"]["min_cosine"],
            "0_4_heldout_v2_check": step0_4_heldout,
            "0_5_approvals": step0_5_approvals,
            "0_6_locked_definitions": LOCKED_SPEC_METADATA,
        },
        "part_a_latency": {
            "status": "BLOCKED_BY_BACKGROUND_GPU_PROCESSES" if gpu_snapshot["has_blocking_graphics_processes"] else "READY",
            "gpu_precheck": gpu_snapshot,
            "policy_note": (
                "Theo mệnh lệnh Phần A: 'Trước/sau: ghi tiến trình GPU (nvidia-smi). Còn tiến trình đồ họa nền thì DỪNG, "
                "báo người dùng, không tự kill.' Phát hiện tiến trình đồ họa nền đang chạy trên GPU nên DỪNG ngay, "
                "không tự kill và không đo độ trễ trên GPU đang bị chiếm tải."
            ),
        },
        "part_b_quality_test_gallery": {
            "status": "NOT_RUN_PER_STEP_0_4",
            "test_gallery_touched_for_quality_eval": False,
        },
    }

    if not step0_2_status["test_gallery_caption_index_v5"]["exists"]:
        results_payload["halt_reasons"].append(
            "Step 0.2: Chưa có index/test_gallery_text_v5/manifest.json (index nhúng caption Config A / Config B / metadata-only cho 1,290 mục test_gallery)."
        )
    if not step0_4_heldout["exists"]:
        results_payload["halt_reasons"].append(
            "Step 0.4: Thiếu tệp eval/test_queries_heldout_v2.json -> cấm chạy Phần B (đánh giá chất lượng trên test_gallery)."
        )
    if gpu_snapshot["has_blocking_graphics_processes"]:
        procs_str = ", ".join(f"{p['process_name']} (PID {p['pid']})" for p in gpu_snapshot["blocking_graphics_processes"])
        results_payload["halt_reasons"].append(
            f"Phần A Pre-check: Phát hiện tiến trình đồ họa nền trên GPU ({procs_str}, GPU-Util={gpu_snapshot['telemetry'].get('utilization_pct')}%) -> DỪNG, báo người dùng, không tự kill."
        )

    out_json = PROJECT_ROOT / "reports/phase5c/final_eval_results.json"
    out_json.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
