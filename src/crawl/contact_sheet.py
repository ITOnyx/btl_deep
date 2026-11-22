"""
Script tạo Contact Sheet (lưới ảnh kèm nhãn crawl_id, brand, articleType, gender, baseColour)
phục vụ duyệt thủ công ở Pha 2 (50 ảnh), Pha 4 (nếu có ca giáp ngưỡng) và Pha 5 (300 ảnh phân tầng).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 13) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Ưu tiên font hỗ trợ tiếng Việt trên Windows
    for fp in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ):
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_contact_sheet(
    records: List[Dict[str, Any]],
    out_path: str | Path,
    cols: int = 5,
    cell_w: int = 240,
    img_h: int = 220,
    caption_h: int = 72,
    title: str = "Contact Sheet",
) -> Path:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        img = Image.new("RGB", (600, 120), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 40), f"{title}: 0 records", fill=(0, 0, 0), font=_get_font(16))
        img.save(out_p)
        return out_p

    n = len(records)
    rows = (n + cols - 1) // cols
    header_h = 44
    cell_h = img_h + caption_h
    canvas_w = cols * cell_w + 20
    canvas_h = header_h + rows * cell_h + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    font_title = _get_font(16)
    font_cap = _get_font(12)

    draw.text((14, 12), f"{title} (n={n})", fill=(20, 20, 20), font=font_title)

    for idx, rec in enumerate(records):
        r = idx // cols
        c = idx % cols
        x0 = 10 + c * cell_w
        y0 = header_h + r * cell_h

        # Khung ô
        draw.rectangle([x0 + 3, y0 + 3, x0 + cell_w - 4, y0 + cell_h - 4], fill=(255, 255, 255), outline=(210, 215, 220))

        img_path = Path(rec.get("local_path", ""))
        if img_path.exists():
            try:
                with Image.open(img_path) as im:
                    rgb = im.convert("RGB")
                    rgb.thumbnail((cell_w - 16, img_h - 12), Image.Resampling.LANCZOS)
                    tw, th = rgb.size
                    px = x0 + (cell_w - tw) // 2
                    py = y0 + 6 + (img_h - 12 - th) // 2
                    canvas.paste(rgb, (px, py))
            except Exception:
                draw.text((x0 + 16, y0 + 80), "[Decode Error]", fill=(200, 0, 0), font=font_cap)
        else:
            draw.text((x0 + 16, y0 + 80), "[Missing File]", fill=(200, 0, 0), font=font_cap)

        cid = str(rec.get("crawl_id", ""))
        brand = str(rec.get("brand", ""))[:16]
        art = str(rec.get("articleType", ""))[:18]
        gen = str(rec.get("gender", ""))[:12]
        col = str(rec.get("baseColour", ""))[:14] or "-"
        status = str(rec.get("label_status", ""))
        pname = str(rec.get("product_name", ""))
        if len(pname) > 28:
            pname = pname[:26] + ".."

        cy = y0 + img_h + 2
        status_color = (0, 128, 60) if status == "ok" else (190, 95, 0)
        draw.text((x0 + 8, cy), f"{cid} | {brand} [{status}]", fill=status_color, font=font_cap)
        draw.text((x0 + 8, cy + 16), f"{art} | {gen} | {col}", fill=(30, 30, 30), font=font_cap)
        draw.text((x0 + 8, cy + 32), pname, fill=(90, 90, 90), font=font_cap)
        draw.text((x0 + 8, cy + 48), f"src:{rec.get('label_source','')} {rec.get('width',0)}x{rec.get('height',0)}", fill=(120, 120, 120), font=font_cap)

    canvas.save(out_p, format="PNG", optimize=True)
    return out_p
