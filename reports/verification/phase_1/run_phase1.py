"""Phase 1 Independent Verification Script (read-only on repo; writes only to reports/verification/)."""
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "reports" / "verification" / "phase_1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = PROJECT_ROOT / "reports" / "verification" / "run_manifest.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def get_pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:
        return f"ERROR: {e}"


def main() -> None:
    results: dict = {
        "phase": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "seed": 42,
    }

    # 1. Git HEAD, branch, status
    git_head = run_cmd(["git", "rev-parse", "HEAD"])
    git_branch = run_cmd(["git", "branch", "--show-current"])
    git_status = run_cmd(["git", "status", "--short"])
    expected_head = "0c8e49e2afe14328bcd272fa2a50e184389702ea"
    wip_lines = [line for line in git_status["stdout"].splitlines() if line.strip()]
    results["1_git"] = {
        "head": git_head["stdout"],
        "expected_head": expected_head,
        "head_match": git_head["stdout"] == expected_head,
        "branch": git_branch["stdout"],
        "wip_count": len(wip_lines),
        "wip_files": wip_lines,
    }

    # 2. Checkpoint SHA-256
    ckpt_path = PROJECT_ROOT / "data" / "checkpoints_siglip-vit" / "best_model.pt"
    expected_ckpt = "e2f82fee26c786c2ea9e521d8341e7bbfc29c0980cebac46688c5522ff64215d"
    actual_ckpt = sha256_file(ckpt_path) if ckpt_path.exists() else None
    results["2_checkpoint"] = {
        "path": str(ckpt_path),
        "exists": ckpt_path.exists(),
        "size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else None,
        "expected_sha256": expected_ckpt,
        "actual_sha256": actual_ckpt,
        "match": actual_ckpt == expected_ckpt,
    }

    # 3. Eval SHA-256
    eval_checks = [
        ("eval/test_split.json", "eval/test_split.sha256", "d61bd87b"),
        ("eval/test_queries_heldout_v2.json", "eval/test_queries_heldout_v2.json.sha256", "96c75433"),
        ("eval/cir_eval_test.json", "eval/cir_eval_test.sha256", "d6e33f4e"),
    ]
    eval_res = {}
    for rel_file, rel_sha, expected_prefix in eval_checks:
        fp = PROJECT_ROOT / rel_file
        sp = PROJECT_ROOT / rel_sha
        actual = sha256_file(fp) if fp.exists() else None
        recorded = None
        if sp.exists():
            recorded = sp.read_text(encoding="utf-8").strip().split()[0]
        eval_res[rel_file] = {
            "exists": fp.exists(),
            "actual_sha256": actual,
            "recorded_sha256_file": rel_sha if sp.exists() else None,
            "recorded_sha256": recorded,
            "prefix_match": bool(actual and actual.startswith(expected_prefix)),
            "full_match_with_repo_sha256": bool(actual and recorded and actual.lower() == recorded.lower()),
        }
    results["3_eval_hashes"] = eval_res

    # 4. index/*/manifest.json + Hash Guard rejection test on a COPY
    manifests = {}
    for mp in sorted((PROJECT_ROOT / "index").glob("*/manifest.json")):
        rel = str(mp.relative_to(PROJECT_ROOT)).replace("\\", "/")
        data = json.loads(mp.read_text(encoding="utf-8"))
        m_hash = data.get("checkpoint_hash")
        manifests[rel] = {
            "checkpoint_hash": m_hash,
            "matches_expected_ckpt": bool(m_hash and m_hash.lower() == expected_ckpt.lower()),
            "keys": sorted(list(data.keys())),
        }

    # Test Hash Guard rejection on a copy in reports/verification/phase_1/tampered_index/
    tampered_dir = OUT_DIR / "tampered_index"
    tampered_dir.mkdir(parents=True, exist_ok=True)
    orig_manifest = json.loads((PROJECT_ROOT / "index" / "test_gallery" / "manifest.json").read_text(encoding="utf-8"))
    tampered_manifest = dict(orig_manifest)
    tampered_manifest["checkpoint_hash"] = "0" * 64
    (tampered_dir / "manifest.json").write_text(
        json.dumps(tampered_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from src.serve.search_engine import ECommerceSearchEngine

    hash_guard_test = {"rejected": False, "exception_type": None, "exception_message": None}
    try:
        ECommerceSearchEngine(
            index_dir=str(tampered_dir),
            config_path=str(PROJECT_ROOT / "config.yaml"),
            device="cpu",
        )
    except ValueError as e:
        hash_guard_test["rejected"] = True
        hash_guard_test["exception_type"] = type(e).__name__
        hash_guard_test["exception_message"] = str(e)
    except Exception as e:
        hash_guard_test["exception_type"] = type(e).__name__
        hash_guard_test["exception_message"] = str(e)

    results["4_manifests_and_hash_guard"] = {
        "manifests": manifests,
        "all_manifests_match_ckpt": all(v["matches_expected_ckpt"] for v in manifests.values()),
        "hash_guard_rejection_test": hash_guard_test,
    }

    # 5. Environment & GPU
    import torch

    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1)
        if torch.cuda.is_available()
        else None,
    }
    nvidia_smi = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,temperature.gpu,utilization.gpu",
        "--format=csv",
    ])
    env_info = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "torch": get_pkg_version("torch"),
        "faiss": get_pkg_version("faiss"),
        "bitsandbytes": get_pkg_version("bitsandbytes"),
        "transformers": get_pkg_version("transformers"),
        "numpy": get_pkg_version("numpy"),
        "pandas": get_pkg_version("pandas"),
        "streamlit": get_pkg_version("streamlit"),
        "pytest": get_pkg_version("pytest"),
        "gpu_torch": gpu_info,
        "nvidia_smi": nvidia_smi["stdout"],
    }
    results["5_environment"] = env_info

    # Write phase1_results.json
    (OUT_DIR / "phase1_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write run_manifest.json (Rule K5)
    run_manifest = {
        "created_utc": results["timestamp_utc"],
        "seed": 42,
        "git_head": results["1_git"]["head"],
        "git_branch": results["1_git"]["branch"],
        "git_status_short": results["1_git"]["wip_files"],
        "environment": env_info,
    }
    MANIFEST_OUT.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
