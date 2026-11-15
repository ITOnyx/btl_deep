"""
Tab 2: Image-to-Image Search.
Finds similar products based on a query image.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from PIL import Image
import streamlit as st
from app.components.product_card import render_product_grid


def render_tab_image_to_image(engine: Any, config: Dict[str, Any]) -> None:
    st.subheader("🖼️ Tra cứu Sản phẩm Tương đồng từ Hình ảnh")
    st.markdown("Tải lên hình ảnh sản phẩm để tìm các sản phẩm có kiểu dáng, màu sắc tương tự trong catalog.")

    uploaded_file = st.file_uploader(
        "Tải ảnh sản phẩm (JPEG, PNG):", type=["jpg", "jpeg", "png"], key="uploader_i2i"
    )

    if uploaded_file is not None:
        col_img, col_info = st.columns([1, 3])
        with col_img:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Ảnh truy vấn", use_container_width=True)
        with col_info:
            st.write(f"Kích thước ảnh: {img.size[0]} x {img.size[1]} px")
            if st.button("Tìm sản phẩm tương tự", key="btn_i2i"):
                with st.spinner("Đang trích xuất đặc trưng hình ảnh và tìm kiếm..."):
                    t0 = time.perf_counter()
                    # Pipeline call will be integrated here in Phase 6 demo
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if config.get("show_latency"):
                        st.success(f"Thời gian phản hồi: **{elapsed_ms:.2f} ms** (SLA < 200 ms)")
                    # render_product_grid(results, num_cols=5)
