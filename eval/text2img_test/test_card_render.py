import sys
import html
import re
import base64
from pathlib import Path
from functools import lru_cache

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.vi_vocab import ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP

@lru_cache(maxsize=4096)
def get_image_base64_data_uri(path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_file():
        return ""
    try:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        ext = p.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""

def render_card_html(prod: dict, show_tech: bool = False) -> str:
    """
    Renders a complete, standalone, pure HTML product card string.
    Rules:
    - Exactly 1 root element: <div class="product-card">...</div>
    - Balanced <div and </div> tags (checked via unit test).
    - No leading whitespace >= 4 spaces on any line.
    - No blank lines.
    - All dynamic values escaped via html.escape.
    """
    rank = int(prod.get("rank", 0))
    rank_str = f"#{rank}"
    
    brand_val = str(prod.get("brand", "") or "").strip()
    has_brand = bool(brand_val and brand_val.lower() not in ("none", "nan", "unknown", ""))
    brand_esc = html.escape(brand_val, quote=True)
    brand_html = f'<span class="brand-tag" title="{brand_esc}">{brand_esc}</span>' if has_brand else '<span class="brand-tag-empty"></span>'

    raw_path = str(prod.get("image_path") or prod.get("path") or "")
    img_uri = get_image_base64_data_uri(raw_path)

    title_vi = str(prod.get("name_vi") or prod.get("name") or "")
    raw_name = str(prod.get("name") or "")
    title_esc = html.escape(title_vi, quote=True)
    raw_name_esc = html.escape(raw_name, quote=True)

    raw_type = str(prod.get("articleType") or "")
    raw_col = str(prod.get("baseColour") or "")
    raw_gen = str(prod.get("gender") or "")

    vn_type = ARTICLE_TYPE_MAP.get(raw_type, raw_type)
    vn_col = COLOUR_MAP.get(raw_col, raw_col)
    vn_gen = GENDER_MAP.get(raw_gen, raw_gen)

    tag_parts = [p for p in (vn_type, vn_col, vn_gen) if p]
    tags_str = " • ".join(tag_parts)
    tags_esc = html.escape(tags_str, quote=True)

    score_val = float(prod.get("score", 0.0))
    score_pct = max(0, min(100, int(score_val * 2500)))

    item_id_esc = html.escape(str(prod.get("item_id", "")), quote=True)
    id_html = f'<span class="id-subtle">ID: {item_id_esc}</span>' if show_tech else '<span class="id-subtle"></span>'

    if img_uri:
        img_block = f'<div class="card-img-wrap"><img class="card-img" src="{img_uri}" alt="{title_esc}" loading="lazy"/></div>'
    else:
        img_block = '<div class="card-img-wrap"><div class="card-img-fallback">(Không tìm thấy ảnh)</div></div>'

    lines = [
        '<div class="product-card">',
        f'<div class="card-top"><span class="rank-tag">{rank_str}</span>{brand_html}</div>',
        img_block,
        f'<div class="card-title-vi" title="{title_esc} (Catalog: {raw_name_esc})">{title_esc}</div>',
        f'<div class="card-tags-vi" title="{tags_esc}">{tags_esc}</div>',
        '<div class="card-meta-row">',
        f'<div class="card-score-box"><span class="score-badge">{score_val:.3f}</span><div class="progress-bar-container"><div class="progress-bar-fill" style="width: {score_pct}%;"></div></div></div>',
        id_html,
        '</div>',
        '</div>'
    ]

    # Join without any blank lines or indentation
    result = "".join(lines)
    return result

# Verification test
sample_prod = {
    "rank": 1,
    "item_id": 15970,
    "name": 'Turtle Check <Men> Navy "Blue" Shirt & More',
    "name_vi": 'Áo Sơ Mi Nam Kẻ Ca-Rô Xanh Navy & Đen',
    "brand": 'Turtle',
    "articleType": 'Shirts',
    "baseColour": 'Navy Blue',
    "gender": 'Men',
    "score": 0.0314,
    "image_path": 'data/fashion-dataset/images/15970.jpg'
}

html_out = render_card_html(sample_prod, show_tech=True)
print("HTML Output Length:", len(html_out))
div_opens = len(re.findall(r'<div\b', html_out))
div_closes = len(re.findall(r'</div>', html_out))
print(f"<div> count: {div_opens}, </div> count: {div_closes}")
assert div_opens == div_closes, f"Mismatch: {div_opens} vs {div_closes}"

# Check for any leading whitespace >= 4 spaces
for i, line in enumerate(html_out.splitlines()):
    assert not line.startswith("    "), f"Line {i} has >= 4 leading spaces: {line[:20]}"
    assert line.strip() != "", f"Line {i} is blank"

# Check escaping of special chars
assert "&lt;Men&gt;" in html_out
assert "&amp;" in html_out
assert '"Blue"' not in html_out # should be &quot;Blue&quot;

print("ALL ASSERTIONS PASSED!")
