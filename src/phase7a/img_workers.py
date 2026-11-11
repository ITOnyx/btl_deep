"""Lightweight image inspection workers for GĐ1 (PIL metadata + full bitstream verify)."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile

# Fail loudly on truncated files when PIL full decode is invoked.
ImageFile.LOAD_TRUNCATED_IMAGES = False

EXIF_ORIENTATION_TAG = 274


def inspect_merged_image_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Read image once -> SHA-1, PIL header/mode/EXIF metadata, and full pixel decode verification."""
    uid, abs_path_str = args
    p = Path(abs_path_str)
    base_res = {
        "uid": uid,
        "abs_path": abs_path_str,
        "file_exists": False,
        "file_size": 0,
        "sha1": "",
        "readable": False,
        "read_error": "",
        "pil_format": "",
        "pil_mode": "",
        "pil_width": 0,
        "pil_height": 0,
        "exif_orientation": 1,
        "has_exif_rotation": False,
        "has_alpha": False,
        "is_cmyk": False,
        "is_grayscale": False,
    }
    try:
        if not p.exists():
            base_res["read_error"] = "file_not_found"
            return base_res
        raw = p.read_bytes()
        base_res["file_exists"] = True
        base_res["file_size"] = len(raw)
        if not raw:
            base_res["read_error"] = "zero_byte"
            return base_res
        base_res["sha1"] = hashlib.sha1(raw).hexdigest()

        with Image.open(io.BytesIO(raw)) as im:
            mode = str(im.mode or "")
            fmt = str(im.format or "")
            w, h = im.size
            exif = im.getexif()
            orient = int(exif.get(EXIF_ORIENTATION_TAG, 1) or 1) if exif is not None else 1
            has_alpha = ("A" in mode) or (mode == "P" and "transparency" in im.info)
            is_cmyk = mode == "CMYK"
            is_gray = mode in ("L", "1", "I", "F", "LA")

            # Fast C++ full pixel decode via OpenCV when standard JPEG/PNG and EOI intact;
            # fall back to strict PIL im.load() for CMYK, exotic formats, or suspicious EOF.
            need_pil_load = (
                is_cmyk
                or fmt not in ("JPEG", "PNG")
                or (fmt == "JPEG" and b"\xff\xd9" not in raw[-64:])
            )
            if not need_pil_load:
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                if arr is None or arr.shape[1] != w or arr.shape[0] != h:
                    need_pil_load = True
            if need_pil_load:
                im.load()

        base_res.update(
            {
                "readable": True,
                "read_error": "",
                "pil_format": fmt,
                "pil_mode": mode,
                "pil_width": int(w),
                "pil_height": int(h),
                "exif_orientation": orient,
                "has_exif_rotation": orient != 1,
                "has_alpha": bool(has_alpha),
                "is_cmyk": bool(is_cmyk),
                "is_grayscale": bool(is_gray),
            }
        )
        return base_res
    except Exception as exc:
        base_res["read_error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        return base_res


def hash_catalog_image_worker(args: Tuple[str, str]) -> Dict[str, Any]:
    """Compute SHA-1 and existence for a catalog image."""
    item_id, abs_path_str = args
    p = Path(abs_path_str)
    try:
        if not p.exists():
            return {"item_id": item_id, "file_exists": False, "file_size": 0, "sha1": ""}
        raw = p.read_bytes()
        return {
            "item_id": item_id,
            "file_exists": True,
            "file_size": len(raw),
            "sha1": hashlib.sha1(raw).hexdigest() if raw else "",
        }
    except Exception:
        return {"item_id": item_id, "file_exists": False, "file_size": 0, "sha1": ""}
