"""Resumable Downloader for Large HuggingFace Safetensors Blobs.
Avoids timeouts and socket stalls by streaming with HTTP Range headers and chunked writes.
"""

import os
import sys
import time
import shutil
from pathlib import Path
import requests

# Fix SSL environment
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)


def download_blob_with_resume(
    url: str,
    target_blob_path: Path,
    expected_size: int,
    incomplete_path: Path = None,
    chunk_size: int = 8 * 1024 * 1024, # 8 MB chunks
):
    target_blob_path.parent.mkdir(parents=True, exist_ok=True)
    
    if target_blob_path.exists() and target_blob_path.stat().st_size == expected_size:
        print(f"[{time.strftime('%X')}] ✅ Target blob already complete: {target_blob_path} ({expected_size} bytes)")
        return target_blob_path

    if incomplete_path is None:
        incomplete_path = target_blob_path.with_name(f"{target_blob_path.name}.download.incomplete")

    current_size = 0
    if incomplete_path.exists():
        current_size = incomplete_path.stat().st_size
        print(f"[{time.strftime('%X')}] Found existing incomplete file: {incomplete_path.name} with {current_size / 1024**2:.2f} MB")

    if current_size >= expected_size:
        print(f"[{time.strftime('%X')}] Incomplete file already reached expected size. Finalizing...")
        shutil.move(str(incomplete_path), str(target_blob_path))
        return target_blob_path

    headers = {}
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"[{time.strftime('%X')}] Resuming from byte {current_size} ({current_size / 1024**2:.2f} MB / {expected_size / 1024**2:.2f} MB)...")
    else:
        print(f"[{time.strftime('%X')}] Starting fresh download for {target_blob_path.name} ({expected_size / 1024**2:.2f} MB)...")

    mode = "ab" if current_size > 0 else "wb"
    t_start = time.time()
    downloaded_this_session = 0

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP request failed: status {resp.status_code}")

        with open(incomplete_path, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    downloaded_this_session += len(chunk)
                    total_downloaded = current_size + downloaded_this_session
                    pct = (total_downloaded / expected_size) * 100
                    elapsed = time.time() - t_start
                    speed_mb = (downloaded_this_session / (1024 * 1024)) / (elapsed + 1e-6)
                    eta_sec = (expected_size - total_downloaded) / ((speed_mb * 1024 * 1024) + 1e-6)
                    sys.stdout.write(
                        f"\r[{time.strftime('%X')}] {pct:5.1f}% | {total_downloaded / 1024**2:7.1f} / {expected_size / 1024**2:.1f} MB "
                        f"| Speed: {speed_mb:5.2f} MB/s | ETA: {int(eta_sec)}s"
                    )
                    sys.stdout.flush()

    print(f"\n[{time.strftime('%X')}] Download loop finished. Validating size...")
    final_size = incomplete_path.stat().st_size
    if final_size != expected_size:
        raise ValueError(f"Size mismatch: expected {expected_size}, got {final_size}")

    print(f"[{time.strftime('%X')}] Moving incomplete file to final blob destination: {target_blob_path}")
    shutil.move(str(incomplete_path), str(target_blob_path))
    return target_blob_path


def setup_internvl2():
    blob_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--OpenGVLab--InternVL2-2B/blobs"))
    snap_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--OpenGVLab--InternVL2-2B/snapshots/e4f6747bd20f139e637642c6a058c6bd00b36919"))
    
    blob_name = "c57c830f03c9141b77b70f84735a0473458a0ebf99250515b0962f20cd9fa3dc"
    target_blob = blob_dir / blob_name
    expected_size = 4411571040
    
    # Check for existing partial file from earlier attempts
    partial_files = list(blob_dir.glob(f"{blob_name}.*incomplete"))
    incomplete = partial_files[0] if partial_files else None
    
    url = "https://huggingface.co/OpenGVLab/InternVL2-2B/resolve/e4f6747bd20f139e637642c6a058c6bd00b36919/model.safetensors"
    
    final_blob = download_blob_with_resume(
        url=url,
        target_blob_path=target_blob,
        expected_size=expected_size,
        incomplete_path=incomplete,
    )
    
    # Place into snapshot if not there
    snap_target = snap_dir / "model.safetensors"
    if not snap_target.exists() or snap_target.stat().st_size != expected_size:
        print(f"Copying/linking blob to snapshot: {snap_target}")
        shutil.copyfile(str(final_blob), str(snap_target))
    print(f"✅ InternVL2-2B model.safetensors ready at {snap_target}!")


def setup_vintern():
    blob_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--5CD-AI--Vintern-1B-v3_5/blobs"))
    snap_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--5CD-AI--Vintern-1B-v3_5/snapshots/b98f263eab246eb5269ade64edbdca8a887dc44d"))
    
    snap_target = snap_dir / "model.safetensors"
    expected_size = 3752849256
    
    # First make sure config/code files are downloaded
    from huggingface_hub import snapshot_download
    print(f"Downloading Vintern metadata and code files...")
    snapshot_download(
        repo_id="5CD-AI/Vintern-1B-v3_5",
        revision="b98f263eab246eb5269ade64edbdca8a887dc44d",
        ignore_patterns=["*.safetensors", "*.bin"],
    )
    
    url = "https://huggingface.co/5CD-AI/Vintern-1B-v3_5/resolve/b98f263eab246eb5269ade64edbdca8a887dc44d/model.safetensors"
    target_blob = blob_dir / "vintern_model.safetensors"
    
    final_blob = download_blob_with_resume(
        url=url,
        target_blob_path=target_blob,
        expected_size=expected_size,
    )
    
    if not snap_target.exists() or snap_target.stat().st_size != expected_size:
        print(f"Copying/linking blob to Vintern snapshot: {snap_target}")
        shutil.copyfile(str(final_blob), str(snap_target))
    print(f"✅ Vintern-1B-v3_5 model.safetensors ready at {snap_target}!")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "internvl"
    if mode in ("internvl", "both"):
        setup_internvl2()
    if mode in ("vintern", "both"):
        setup_vintern()
