"""
Product Card UI component.
Renders product image, title, attributes, and similarity score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st


def render_product_grid(results: list[Dict[str, Any]], num_cols: int = 5) -> None:
    if not results:
        st.info("Không tìm thấy sản phẩm phù hợp.")
        return

    cols = st.columns(num_cols)
    for idx, item in enumerate(results):
        col = cols[idx % num_cols]
        with col:
            img_path = item.get("image_path")
            if img_path and Path(img_path).exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.write("📷 *(Không có ảnh)*")

            rank = idx + 1
            score = item.get("score", 0.0)
            pid = item.get("product_id", "N/A")
            title = item.get("title", f"Sản phẩm #{pid}")
            category = item.get("articleType", "Thời trang")
            color = item.get("baseColour", "—")
            gender = item.get("gender", "—")

            st.markdown(f"**#{rank}** ({score:.3f})")
            st.caption(f"**{title}**")
            st.caption(f"Loại: {category} | Màu: {color} | Giới: {gender}")
