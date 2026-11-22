"""Model Downloader with Resume and Clean SSL Environment.
Downloads OpenGVLab/InternVL2-2B and 5CD-AI/Vintern-1B-v3_5 safely.
"""

import os
import sys
import time

# Clean faulty system SSL cert variables (e.g. from PostgreSQL)
os.environ.pop("CURL_CA_BUNDLE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("SSL_CERT_FILE", None)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download


def download_model(repo_id: str, revision: str = None):
    print(f"[{time.strftime('%X')}] Starting download/verification for: {repo_id} (revision: {revision})...")
    t0 = time.time()
    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        resume_download=True,
        max_workers=4,
    )
    elapsed = time.time() - t0
    print(f"[{time.strftime('%X')}] ✅ Finished {repo_id} in {elapsed:.1f}s -> {local_dir}")
    return local_dir


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"
    
    if target in ("internvl", "both"):
        download_model("OpenGVLab/InternVL2-2B", revision="e4f6747bd20f139e637642c6a058c6bd00b36919")
    if target in ("vintern", "both"):
        download_model("5CD-AI/Vintern-1B-v3_5", revision="b98f263eab246eb5269ade64edbdca8a887dc44d")
