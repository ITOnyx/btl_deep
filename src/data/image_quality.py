"""Image quality metrics and contact sheet generator."""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import hashlib
import imagehash
from pathlib import Path


def read_image_safely(file_path):
    """Read image bytes safely on Windows with Unicode path support."""
    try:
        p = Path(file_path)
        if not p.exists():
            return None, "not_found"
        size = p.stat().st_size
        if size == 0:
            return None, "zero_byte"
        with open(p, "rb") as f:
            buf = f.read()
        gray = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, "corrupt"
        return gray, "ok"
    except Exception as e:
        return None, f"error_{str(e)}"


def compute_sha1_and_phash(file_path):
    """Compute exact SHA-1 and perceptual hash for an image."""
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return "", ""
    with open(p, "rb") as f:
        buf = f.read()
    sha1 = hashlib.sha1(buf).hexdigest()
    try:
        with Image.open(p) as img:
            ph = str(imagehash.phash(img.convert("L")))
    except Exception:
        ph = ""
    return sha1, ph


def compute_image_metrics(file_path):
    """Compute resolution, blur Laplacian variance, and banner flag."""
    gray, status = read_image_safely(file_path)
    if status != "ok":
        return {
            "status": status,
            "width": 0,
            "height": 0,
            "short_side": 0,
            "blur_score": 0.0,
            "banner_flag": False
        }

    h, w = gray.shape[:2]
    short_side = min(w, h)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Banner & border detection heuristic
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
    sides_combined = np.concatenate([edges[:, :side_w], edges[:, -side_w:]], axis=1)
    side_edge_density = float(np.mean(sides_combined > 0))

    is_top_solid = (top_mean < 230.0 and top_std < 10.0)
    is_bot_solid = (bot_mean < 230.0 and bot_std < 10.0)
    is_top_text = (top_edge_density > 0.025)
    is_side_frame = (side_edge_density > 0.030)

    banner_flag = bool(is_top_solid or is_bot_solid or is_top_text or is_side_frame)

    return {
        "status": "ok",
        "width": int(w),
        "height": int(h),
        "short_side": int(short_side),
        "blur_score": blur_score,
        "banner_flag": banner_flag
    }


def create_contact_sheet(image_items, output_path, cols=8, thumb_size=(150, 200)):
    """
    Generate a tiled contact sheet from image_items.
    image_items: list of dicts with 'path', 'title', 'metric_text'
    """
    n = len(image_items)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    cell_w, cell_h = thumb_size[0], thumb_size[1] + 30
    canvas_w = cols * cell_w
    canvas_h = rows * cell_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for idx, item in enumerate(image_items):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * cell_h

        img_p = Path(item["path"])
        if img_p.exists() and img_p.stat().st_size > 0:
            try:
                with Image.open(img_p) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((cell_w - 4, thumb_size[1] - 4))
                    # Center in thumb box
                    offset_x = x0 + (cell_w - im_rgb.width) // 2
                    offset_y = y0 + (thumb_size[1] - im_rgb.height) // 2
                    canvas.paste(im_rgb, (offset_x, offset_y))
            except Exception:
                draw.text((x0 + 10, y0 + 50), "Error opening", fill=(255, 0, 0))
        else:
            draw.text((x0 + 10, y0 + 50), "Missing", fill=(200, 0, 0))

        # Text label
        label = item.get("metric_text", "")
        draw.text((x0 + 5, y0 + thumb_size[1] + 5), str(label)[:25], fill=(30, 30, 30))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=85)
    print(f"Saved contact sheet: {output_path} ({n} images)")
