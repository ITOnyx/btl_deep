"""
Tab 3: CIR from Catalog Reference.
Composes a catalog reference product with a text modifier (e.g. 'nhưng màu đỏ', 'dành cho nữ').
"""
from __future__ import annotations

import time
from typing import Any, Dict

import streamlit as st
from app.components.product_card import render_product_grid


def render_tab_cir_catalog(engine: Any, config: Dict[str, Any]) -> None:
    st.subheader("🔄 Tra cứu Kết hợp (CIR) từ Sản phẩm Catalog")
    st.markdown("Chọn một sản phẩm tham chiếu trong catalog và nhập câu chỉnh sửa mong muốn.")

    col_ref, col_mod = st.columns([2, 3])
    with col_ref:
        product_id = st.text_input("Mã sản phẩm tham chiếu (Product ID):", value="15970")
    with col_mod:
        modifier_text = st.text_input("Yêu cầu chỉnh sửa (Modifier):", value="nhưng màu đen")

    if st.button("Tìm kiếm CIR Catalog", key="btn_cir_catalog"):
        with st.spinner("Đang điều hướng không gian đặc trưng đa phương thức..."):
            t0 = time.perf_counter()
            # Pipeline call will be integrated here in Phase 6 demo
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if config.get("show_latency"):
                st.success(f"Thời gian phản hồi: **{elapsed_ms:.2f} ms** (SLA < 200 ms)")
            # render_product_grid(results, num_cols=5)
