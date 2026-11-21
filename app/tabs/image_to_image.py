"""
Tab 2: Image-to-Image Search (Hình ảnh).
Minimalist layout for image-based product discovery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from PIL import Image
import streamlit as st

from app.components.product_card import render_result_grid
from app.engine_service import validate_and_load_image, ImageValidationError


def render_tab_image_to_image(service: Any, config: Dict[str, Any]) -> None:
    st.caption("Tìm kiếm sản phẩm tương đồng từ hình ảnh.")

    if "i2i_results" not in st.session_state:
        st.session_state["i2i_results"] = None
    if "i2i_elapsed_s" not in st.session_state:
        st.session_state["i2i_elapsed_s"] = 0.0

    col_up, col_preview = st.columns([3, 2])
    img_to_search: Image.Image | None = None

    with col_up:
        uploaded_file = st.file_uploader(
            "Tải ảnh sản phẩm (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="i2i_file_uploader",
        )
        use_sample = st.checkbox("Dùng ảnh mẫu từ gallery", value=False, key="i2i_use_sample")
        if use_sample:
            sample_ids = ["15970", "1855", "7990", "25947"]
            chosen_pid = st.selectbox("Chọn ảnh mẫu", sample_ids, index=0, key="i2i_sample_select")
            sample_path = service.sample_image_path(chosen_pid)
            if sample_path.exists():
                img_to_search = Image.open(sample_path).convert("RGB")
            else:
                st.warning(f"Không tìm thấy ảnh mẫu: {sample_path}")
        elif uploaded_file is not None:
            try:
                img_to_search = validate_and_load_image(
                    file_data=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )
            except ImageValidationError as ve:
                st.error(str(ve))
                img_to_search = None

        btn_search = st.button("Tìm kiếm", type="primary", key="btn_i2i_submit", use_container_width=True)

    with col_preview:
        if img_to_search is not None:
            st.image(img_to_search, caption=f"Ảnh truy vấn ({img_to_search.size[0]}x{img_to_search.size[1]} px)", width=200)

    if btn_search and img_to_search is not None:
        with st.spinner("Đang tìm kiếm..."):
            try:
                results, elapsed_ms = service.search_image_to_image(
                    image_input=img_to_search,
                    top_k=config["top_k"],
                )
                st.session_state["i2i_results"] = results
                st.session_state["i2i_elapsed_s"] = elapsed_ms / 1000.0
            except Exception as e:
                st.error(f"Lỗi tìm kiếm: {e}")
                st.session_state["i2i_results"] = None

    saved_results = st.session_state.get("i2i_results")
    if saved_results is not None:
        elapsed_s = st.session_state.get("i2i_elapsed_s", 0.0)
        st.caption(f"Tìm thấy {len(saved_results)} kết quả trong {elapsed_s:.2f} giây")
        render_result_grid(saved_results, num_cols=5)
