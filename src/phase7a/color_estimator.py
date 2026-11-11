"""Independent pixel-based dominant garment color estimator for Phase 7a GD4.

Used strictly for CROSS-CHECKING attribute color labels (and recording estimated
pixel colors for color=-1 items), NEVER overwriting source attribute labels.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Clean garment-focused subset of catalog 46 baseColour LAB centroids.
# Excludes accessory/metallic/skin/taupe/bronze prototypes that steal Beige/Grey apparel.
GARMENT_CENTROID_COLOURS: Tuple[str, ...] = (
    "Black",
    "Charcoal",
    "Grey",
    "White",
    "Beige",
    "Khaki",
    "Brown",
    "Coffee Brown",
    "Tan",
    "Navy Blue",
    "Blue",
    "Teal",
    "Turquoise Blue",
    "Red",
    "Maroon",
    "Burgundy",
    "Rust",
    "Pink",
    "Magenta",
    "Peach",
    "Purple",
    "Lavender",
    "Green",
    "Olive",
    "Sea Green",
    "Lime Green",
    "Yellow",
    "Mustard",
    "Orange",
)

# Fallbacks if catalog_color_centroids_lab.json is unavailable in tests.
_DEFAULT_LAB_CENTROIDS: Dict[str, List[float]] = {
    "Beige": [176.62, 132.95, 141.98],
    "Black": [51.62, 128.34, 128.01],
    "Blue": [150.90, 128.31, 120.94],
    "Brown": [91.79, 132.46, 133.98],
    "Burgundy": [63.96, 143.80, 130.39],
    "Charcoal": [83.92, 128.51, 127.39],
    "Coffee Brown": [62.89, 132.10, 131.21],
    "Cream": [194.24, 129.87, 137.27],
    "Green": [177.35, 113.78, 134.04],
    "Grey": [153.32, 128.88, 128.84],
    "Khaki": [136.70, 130.50, 144.79],
    "Lavender": [177.45, 139.69, 126.81],
    "Lime Green": [216.43, 121.31, 159.98],
    "Magenta": [113.16, 169.77, 129.52],
    "Maroon": [83.13, 150.16, 137.44],
    "Mustard": [162.76, 138.69, 169.77],
    "Navy Blue": [65.74, 129.84, 120.87],
    "Off White": [200.21, 129.29, 137.36],
    "Olive": [101.81, 126.54, 137.65],
    "Orange": [169.35, 157.44, 165.06],
    "Peach": [186.54, 149.69, 146.09],
    "Pink": [172.22, 144.03, 132.19],
    "Purple": [107.23, 141.85, 120.01],
    "Red": [105.11, 172.35, 153.82],
    "Rust": [107.84, 151.09, 151.00],
    "Sea Green": [202.30, 107.88, 127.63],
    "Tan": [127.80, 138.74, 150.10],
    "Teal": [107.33, 117.55, 123.86],
    "Turquoise Blue": [186.08, 109.98, 121.54],
    "White": [203.63, 128.79, 129.34],
    "Yellow": [204.59, 129.77, 162.46],
}


@lru_cache(maxsize=8)
def load_colour_family_map(families_path: str) -> Dict[str, str]:
    """Load baseColour -> colour family mapping from eval/families.json."""
    data = json.loads(Path(families_path).read_text(encoding="utf-8"))
    col_to_fam: Dict[str, str] = {}
    for fam_name, cols in data["colour_families"].items():
        for c in cols:
            col_to_fam[str(c)] = str(fam_name)
    col_to_fam.setdefault("Unknown", "Unknown")
    return col_to_fam


@lru_cache(maxsize=8)
def load_garment_centroids(centroids_path: str) -> Tuple[List[str], np.ndarray]:
    """Load curated garment CIELAB centroids as (names, float32[N, 3])."""
    raw = dict(_DEFAULT_LAB_CENTROIDS)
    p = Path(centroids_path)
    if p.exists():
        loaded = json.loads(p.read_text(encoding="utf-8"))
        raw.update({k: v for k, v in loaded.items() if k in GARMENT_CENTROID_COLOURS})
    names = [c for c in GARMENT_CENTROID_COLOURS if c in raw]
    mat = np.array([raw[c] for c in names], dtype=np.float32)
    return names, mat


def _border_bg_floodfill(rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask of bright neutral studio background reachable from borders."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    is_bg_cand = ((hsv[:, :, 2] >= 224) & (hsv[:, :, 1] <= 18)).astype(np.uint8)
    _, labels = cv2.connectedComponents(is_bg_cand, connectivity=4)
    border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    border_labels = border_labels[border_labels > 0]
    if len(border_labels) == 0:
        return np.zeros(is_bg_cand.shape, dtype=bool)
    return np.isin(labels, border_labels)


def _hue_bucket_from_hsv(h_val: float, s_val: float, v_val: float, b_lab: float) -> Optional[str]:
    """Map dominant chromatic HSV/Lab tuple to canonical baseColour when chroma is clear."""
    if v_val < 42.0:
        # Dark pixels: only classify as Navy Blue when blue chroma + saturation are clear.
        if 88.0 <= h_val <= 126.0 and b_lab <= 123.5 and s_val >= 45.0:
            return "Navy Blue"
        return "Black"
    # Strong blue/navy guard against Lavender/Purple drift on pale or washed blues.
    if 88.0 <= h_val <= 125.0 and b_lab < 126.5 and s_val >= 16.0:
        return "Navy Blue" if v_val < 95.0 else "Blue"
    if s_val < 35.0:
        return None
    if h_val <= 6.0 or h_val >= 170.0:
        if v_val < 115.0:
            return "Maroon" if h_val >= 165.0 else "Burgundy"
        if 1.0 <= h_val <= 6.0 and b_lab >= 140.0 and v_val < 195.0 and s_val < 130.0:
            return "Brown"
        if h_val >= 2.0 and v_val >= 245.0 and b_lab >= 175.0:
            return "Orange"
        if (h_val >= 166.0 or b_lab <= 135.0) and v_val >= 172.0 and s_val < 105.0:
            return "Pink"
        return "Red"
    if 6.0 < h_val <= 17.0:
        if v_val < 125.0:
            return "Brown"
        if s_val < 115.0 and v_val >= 165.0:
            if s_val < 52.0 and v_val >= 190.0:
                return "Beige"
            return "Peach" if h_val < 12.0 else "Beige"
        return "Orange"
    if 17.0 < h_val <= 34.0:
        if v_val < 130.0:
            return "Olive" if h_val > 25.0 else "Brown"
        if s_val < 100.0 and h_val < 24.0:
            return "Beige"
        if s_val < 145.0 and v_val < 185.0:
            return "Mustard"
        return "Yellow"
    if 34.0 < h_val <= 84.0:
        if v_val < 120.0 and h_val <= 52.0:
            return "Olive"
        return "Green"
    if 84.0 < h_val < 88.0:
        return "Teal" if v_val < 155.0 else "Turquoise Blue"
    if 125.0 < h_val <= 150.0:
        return "Lavender" if v_val > 165.0 and s_val < 115.0 else "Purple"
    if 150.0 < h_val < 173.0:
        return "Magenta" if (s_val >= 120.0 and v_val < 175.0) else "Pink"
    return None


def predict_color_pixel(
    abs_path: str | Path,
    centroid_names: Tuple[str, ...] | List[str],
    centroid_lab: np.ndarray,
) -> Tuple[str, Dict[str, Any]]:
    """Estimate dominant garment color independently from image pixels.

    Returns (color_name, debug_info). Raises nothing if image fails; returns ("Unknown", err).
    """
    try:
        with Image.open(abs_path) as im:
            rgb_full = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        return "Unknown", {"error": str(exc)}

    # Downsample to 96x96 for fast, reproducible foreground extraction.
    rgb = cv2.resize(rgb_full, (96, 96), interpolation=cv2.INTER_AREA)
    bg_mask = _border_bg_floodfill(rgb)
    fg_mask = ~bg_mask

    # Fallback center crop if near-all pixels became border background (e.g. bright white tee).
    if int(fg_mask.sum()) < 96 * 96 * 0.04:
        cy0, cy1, cx0, cx1 = 19, 77, 24, 72
        fg_mask = np.zeros((96, 96), dtype=bool)
        fg_mask[cy0:cy1, cx0:cx1] = True

    # Center weight bias + inner white hole rejection (studio wall visible inside V-neck/straps/legs).
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    fg_no_white_hole = fg_mask & ~((hsv[:, :, 2] >= 248) & (hsv[:, :, 1] <= 8))
    use_mask = fg_no_white_hole if int(fg_no_white_hole.sum()) >= 120 else fg_mask

    # Focus on central 70% bounding box of foreground to suppress hanger/floor shadows.
    ys, xs = np.where(use_mask)
    if len(ys) >= 100:
        y_lo, y_hi = np.percentile(ys, [12, 88])
        x_lo, x_hi = np.percentile(xs, [15, 85])
        core_mask = use_mask.copy()
        yy, xx = np.ogrid[:96, :96]
        core_mask &= (yy >= y_lo) & (yy <= y_hi) & (xx >= x_lo) & (xx <= x_hi)
        if int(core_mask.sum()) >= 80:
            use_mask = core_mask

    pix_lab = lab[use_mask]
    pix_hsv = hsv[use_mask].astype(np.float32)
    pix_rgb = rgb[use_mask].astype(np.int32)
    rgb_span = pix_rgb.max(axis=1) - pix_rgb.min(axis=1)

    # Check chromatic diversity for genuine Multi-colour garments (suppressing dark-shadow JPEG chroma noise).
    sat_mask = (pix_hsv[:, 1] >= 50.0) & (pix_hsv[:, 2] >= 55.0) & (rgb_span >= 28)
    sat_ratio = float(sat_mask.mean())
    if sat_ratio >= 0.35 and int(sat_mask.sum()) >= 75:
        sat_h = pix_hsv[sat_mask, 0]
        # 6 coarse perceptual hue bins: Red, Orange/Yellow, Green, Teal/Blue, Purple, Pink/Magenta.
        bins = np.zeros(6, dtype=np.int32)
        bins[0] = int(((sat_h <= 8) | (sat_h >= 172)).sum())
        bins[1] = int(((sat_h > 8) & (sat_h <= 34)).sum())
        bins[2] = int(((sat_h > 34) & (sat_h <= 84)).sum())
        bins[3] = int(((sat_h > 84) & (sat_h <= 125)).sum())
        bins[4] = int(((sat_h > 125) & (sat_h <= 150)).sum())
        bins[5] = int(((sat_h > 150) & (sat_h < 172)).sum())
        fracs = np.sort(bins / max(1, int(sat_mask.sum())))[::-1]
        if fracs[0] >= 0.24 and fracs[1] >= 0.24 and (bins > 0.20 * len(sat_h)).sum() >= 2:
            # Ensure Red and Pink don't falsely split a solid red/coral knit into Multi.
            major_idx = set(int(i) for i in np.where(bins >= 0.20 * len(sat_h))[0])
            if major_idx not in ({0, 5}, {0, 1}, {2, 3}):
                return "Multi", {"reason": "multi_hue_bins", "fracs": fracs[:3].round(3).tolist()}

    # Quantize LAB into 12-unit histogram cells to pick the dominant garment mode (avoids print/skin dilution).
    q_lab = np.round(pix_lab / 12.0).astype(np.int32)
    packed = (q_lab[:, 0] << 16) + (q_lab[:, 1] << 8) + q_lab[:, 2]
    uniq, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    dom_idx = int(np.argmax(counts))
    dom_pixels = inv == dom_idx

    med_lab = np.median(pix_lab[dom_pixels], axis=0)
    med_hsv = np.median(pix_hsv[dom_pixels], axis=0)
    l_val, a_val, b_val = float(med_lab[0]), float(med_lab[1]), float(med_lab[2])
    h_val, s_val, v_val = float(med_hsv[0]), float(med_hsv[1]), float(med_hsv[2])
    chroma = float(np.hypot(a_val - 128.0, b_val - 128.0))

    # 1. Clear Blue / Navy guard before achromatic check so pale sky-blue (b<=123.5) is not collapsed to White.
    if 88.0 <= h_val <= 125.0 and b_val <= 123.5 and v_val >= 42.0 and s_val >= 14.0:
        col = "Navy Blue" if v_val < 95.0 else "Blue"
        return col, {"L": round(l_val, 1), "a": round(a_val, 1), "b": round(b_val, 1), "H": round(h_val, 1), "S": round(s_val, 1), "V": round(v_val, 1), "mode": "blue_guard"}

    # 2. Achromatic branch (chroma < 6.0 catches dark blacks/charcoals where small RGB noise inflates HSV S).
    if chroma < 6.0 or (s_val < 22.0 and chroma < 7.5):
        if v_val >= 188.0:
            col = "White"
        elif v_val >= 95.0:
            col = "Grey"
        elif v_val >= 60.0:
            col = "Charcoal"
        else:
            col = "Black"
        return col, {"L": round(l_val, 1), "a": round(a_val, 1), "b": round(b_val, 1), "H": round(h_val, 1), "S": round(s_val, 1), "V": round(v_val, 1), "mode": "achromatic"}

    # 3. Clear chromatic / Dark Black branch.
    hue_guess = _hue_bucket_from_hsv(h_val, s_val, v_val, b_val)
    if hue_guess is not None:
        return hue_guess, {"L": round(l_val, 1), "a": round(a_val, 1), "b": round(b_val, 1), "H": round(h_val, 1), "S": round(s_val, 1), "V": round(v_val, 1), "mode": "hsv_rule"}

    # 4. Soft-chromatic fallback (e.g. Beige, Khaki, Charcoal, faded tones) -> weighted CIELAB centroid.
    weights = np.array([0.45, 1.35, 1.35], dtype=np.float32)
    dists = np.linalg.norm((centroid_lab - med_lab.astype(np.float32)) * weights, axis=1)
    best_name = str(centroid_names[int(np.argmin(dists))])
    # Prevent pale blue from hitting Lavender/Purple when Lab-b is negative.
    if best_name in ("Lavender", "Purple") and b_val < 126.0 and a_val <= 132.0:
        best_name = "Blue"
    # Warm pale ivory/sand with clear yellow bias belongs to Beige, not Peach or White.
    if best_name == "White" and b_val >= 134.0 and s_val >= 18.0:
        best_name = "Beige"
    return best_name, {"L": round(l_val, 1), "a": round(a_val, 1), "b": round(b_val, 1), "H": round(h_val, 1), "S": round(s_val, 1), "V": round(v_val, 1), "mode": "lab_centroid"}


def predict_color_pixel_worker(task: Tuple[str, str, str]) -> Dict[str, Any]:
    """Thread/process worker wrapper returning {uid, color_pixel, px_mode}."""
    uid, abs_path_str, centroids_path = task
    names, mat = load_garment_centroids(centroids_path)
    col, info = predict_color_pixel(abs_path_str, names, mat)
    return {
        "uid": uid,
        "color_pixel": col,
        "px_mode": str(info.get("mode", info.get("reason", info.get("error", "")))),
    }
