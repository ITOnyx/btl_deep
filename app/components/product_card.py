"""
Product Card UI component for Streamlit application.
Renders minimalist, clean, perfectly aligned 5-column product grid with Vietnamese localization.
Uses st.popover for details to prevent column displacement.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

from app.i18n_vi import (
    product_name_vi,
    translate_article_type,
    translate_color,
    translate_gender,
)


def _get_image_base64(image_path: str) -> str:
    """Reads image file and returns base64 string for clean HTML rendering."""
    p = Path(image_path)
    if not p.exists():
        return ""
    try:
        data = p.read_bytes()
        ext = p.suffix.lower().lstrip(".") or "jpeg"
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"
    except Exception:
        return ""


def _e(value: Any) -> str:
    """HTML-escape a metadata value before interpolating it into markup."""
    return html.escape(str(value), quote=True)


def render_result_card(item: Dict[str, Any]) -> None:
    """Renders a single minimalist product card with uniform height and alignment."""
    rank = item.get("rank", 1)
    score = item.get("final_score", 0.0)
    cos_score = item.get("cosine_score", 0.0)
    bonus = item.get("attribute_bonus", 0.0)
    pid = str(item.get("product_id", item.get("item_id", "N/A")))
    orig_title = str(item.get("title", f"Sản phẩm {pid}")).strip()

    # Vietnamese localization
    title_vi = product_name_vi(item)

    cat_raw = str(item.get("articleType", "N/A"))
    color_raw = str(item.get("baseColour", "N/A"))
    gender_raw = str(item.get("gender", "N/A"))

    cat_vi = translate_article_type(cat_raw) if cat_raw != "N/A" else "Không xác định"
    color_vi = translate_color(color_raw) if color_raw != "N/A" else "Không xác định"
    gender_vi = translate_gender(gender_raw) if gender_raw != "N/A" else "Không xác định"

    # Taxonomy 3-tier fields
    domain_val = item.get("domain")
    group_val = item.get("group")
    tax_b = item.get("tax_bonus")
    tax_p = item.get("tax_penalty")
    col_b = item.get("color_bonus")
    constraint_str = item.get("constraint_mode")

    img_path = item.get("image_path", "")
    b64_img = _get_image_base64(img_path)

    # Clean HTML image container with white background (seamless with fashion images)
    if b64_img:
        img_html = f"""
        <div class="product-img-container">
            <img src="{b64_img}" alt="{_e(title_vi)}" class="product-img" loading="lazy" />
        </div>
        """
    else:
        img_html = """
        <div class="product-img-container product-img-placeholder">
            <span>Không có ảnh</span>
        </div>
        """

    st.markdown(img_html, unsafe_allow_html=True)

    # Card body: Rank + Score line, Vietnamese title line (fixed height), ID caption
    card_body_html = f"""
    <div class="product-info-container">
        <div class="product-meta-row">
            <span class="product-rank">#{rank}</span>
            <span class="product-score">Điểm {score:.4f}</span>
        </div>
        <div class="product-title" title="{_e(orig_title)}">{_e(title_vi)}</div>
        <div class="product-id">ID {_e(pid)}</div>
    </div>
    """
    st.markdown(card_body_html, unsafe_allow_html=True)

    # Use st.popover to show details in a floating popover overlay
    with st.popover("Chi tiết", use_container_width=True):
        detail_lines = [
            '<div class="detail-content">',
            f'<div>Cosine gốc: <code>{cos_score:.4f}</code></div>',
            f'<div>Điểm tổng: <code>{score:.4f}</code></div>',
        ]

        if tax_b is not None and col_b is not None:
            detail_lines.append(f'<div>Thưởng giữ loại: <code>{tax_b:+.4f}</code></div>')
            detail_lines.append(f'<div>Phạt lệch nhóm: <code>{-tax_p:+.4f}</code></div>')
            detail_lines.append(f'<div>Thưởng màu mới: <code>{col_b:+.4f}</code></div>')
            if constraint_str:
                detail_lines.append(f'<div>Ràng buộc: <b>{_e(constraint_str)}</b></div>')
        else:
            detail_lines.append(f'<div>Điểm thưởng: <code>{bonus:+.4f}</code></div>')

        detail_lines.append('<hr style="margin: 6px 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.1);" />')

        if domain_val:
            detail_lines.append(f'<div>Lĩnh vực: <b>{_e(domain_val)}</b></div>')
        if group_val:
            detail_lines.append(f'<div>Nhóm: <b>{_e(group_val)}</b></div>')

        detail_lines.append(f'<div>Loại: <b>{_e(cat_vi)}</b></div>')
        detail_lines.append(f'<div>Màu: <b>{_e(color_vi.capitalize())}</b></div>')
        detail_lines.append(f'<div>Giới: <b>{_e(gender_vi.capitalize())}</b></div>')
        detail_lines.append('</div>')

        st.markdown("\n".join(detail_lines), unsafe_allow_html=True)


def render_result_grid(results: List[Dict[str, Any]], num_cols: int = 5) -> None:
    """Renders a responsive 5-column grid of products."""
    if not results:
        st.caption("Không tìm thấy sản phẩm phù hợp.")
        return

    for row_start in range(0, len(results), num_cols):
        row_items = results[row_start : row_start + num_cols]
        cols = st.columns(num_cols)
        for col_idx, item in enumerate(row_items):
            with cols[col_idx]:
                render_result_card(item)
