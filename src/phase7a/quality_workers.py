"""Image quality & perceptual hash workers matching Phase 1 criteria."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image


def init_quality_worker() -> None:
    """Disable OpenCV internal multi-threading per worker to avoid N*M thread thrashing."""
    cv2.setNumThreads(0)


def compute_quality_metrics_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Compute short_side, Laplacian blur variance, Pha-1 banner heuristic, and pHash."""
    uid, abs_path_str = args
    p = Path(abs_path_str)
    empty = {
        "uid": uid,
        "qual_status": "error",
        "cv_width": 0,
        "cv_height": 0,
        "short_side": 0,
        "blur_score": 0.0,
        "phash": "",
        "banner_flag": False,
        "is_top_solid": False,
        "is_bot_solid": False,
        "is_top_text": False,
        "is_side_frame": False,
        "top_mean": 0.0,
        "top_std": 0.0,
        "bot_mean": 0.0,
        "bot_std": 0.0,
        "top_edge_density": 0.0,
        "bot_edge_density": 0.0,
        "side_edge_density": 0.0,
    }
    try:
        if not p.exists():
            empty["qual_status"] = "not_found"
            return empty
        raw = p.read_bytes()
        if not raw:
            empty["qual_status"] = "zero_byte"
            return empty

        gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            with Image.open(io.BytesIO(raw)) as im:
                gray = np.asarray(im.convert("L"))
        if gray is None or gray.size == 0:
            empty["qual_status"] = "corrupt"
            return empty

        h, w = gray.shape[:2]
        short_side = int(min(w, h))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        phash_str = str(imagehash.phash(Image.fromarray(gray)))

        # Banner & border heuristic (identical thresholds to src/diag/phase1_quality.py)
        top_h = max(1, int(h * 0.08))
        bot_h = max(1, int(h * 0.08))
        side_w = max(1, int(w * 0.05))

        top_strip = gray[:top_h, :]
        bot_strip = gray[-bot_h:, :]
        top_mean = float(np.mean(top_strip))
        top_std = float(np.std(top_strip))
        bot_mean = float(np.mean(bot_strip))
        bot_std = float(np.std(bot_strip))

        edges = cv2.Canny(gray, 100, 200)
        top_edge_density = float(np.mean(edges[:top_h, :] > 0))
        bot_edge_density = float(np.mean(edges[-bot_h:, :] > 0))
        sides_combined = np.concatenate([edges[:, :side_w], edges[:, -side_w:]], axis=1)
        side_edge_density = float(np.mean(sides_combined > 0))

        is_top_solid = bool(top_mean < 230.0 and top_std < 10.0)
        is_bot_solid = bool(bot_mean < 230.0 and bot_std < 10.0)
        is_top_text = bool(top_edge_density > 0.025)
        is_side_frame = bool(side_edge_density > 0.030)
        banner_flag = bool(is_top_solid or is_bot_solid or is_top_text or is_side_frame)

        return {
            "uid": uid,
            "qual_status": "ok",
            "cv_width": int(w),
            "cv_height": int(h),
            "short_side": short_side,
            "blur_score": blur_score,
            "phash": phash_str,
            "banner_flag": banner_flag,
            "is_top_solid": is_top_solid,
            "is_bot_solid": is_bot_solid,
            "is_top_text": is_top_text,
            "is_side_frame": is_side_frame,
            "top_mean": top_mean,
            "top_std": top_std,
            "bot_mean": bot_mean,
            "bot_std": bot_std,
            "top_edge_density": top_edge_density,
            "bot_edge_density": bot_edge_density,
            "side_edge_density": side_edge_density,
        }
    except Exception as exc:
        empty["qual_status"] = f"error:{type(exc).__name__}"
        return empty
