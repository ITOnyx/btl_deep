"""
Tab 4: CIR from Uploaded Image.
Composes a user-uploaded image with a text modifier.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from PIL import Image
import streamlit as st
from app.components.product_card import render_product_grid


def render_tab_cir_upload(engine: Any, config: Dict[str, Any]) -> None:
    st.subheader("📤 Tra cứu Kết hợp (CIR) từ Ảnh Tải lên")
    st.markdown("Tải lên ảnh bất kỳ bên ngoài và nhập yêu cầu thay đổi thuộc tính (màu sắc, giới tính, kiểu dáng).")

    uploaded_file = st.file_uploader(
        "Tải ảnh sản phẩm gốc:", type=["jpg", "jpeg", "png"], key="uploader_cir_upload"
    )

    modifier_text = st.text_input(
        "Nhập yêu cầu biến đổi (Modifier):",
        placeholder="nhưng màu xanh dương đậm, kiểu thể thao...",
        key="mod_cir_upload",
    )

    if uploaded_file is not None and modifier_text.strip():
        col_img, col_act = st.columns([1, 3])
        with col_img:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Ảnh gốc tham chiếu", use_container_width=True)
        with col_act:
            if st.button("Tìm kiếm biến thể sản phẩm", key="btn_cir_upload"):
                with st.spinner("Đang xử lý ảnh tải lên và kết hợp vector..."):
                    t0 = time.perf_counter()
                    # Pipeline call will be integrated here in Phase 6 demo
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    if config.get("show_latency"):
                        st.success(f"Thời gian phản hồi: **{elapsed_ms:.2f} ms** (SLA < 200 ms)")
                    # render_product_grid(results, num_cols=5)
