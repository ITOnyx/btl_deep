"""
Tab 1: Text-to-Image Search.
Supports Vietnamese queries (with and without diacritics).
"""
from __future__ import annotations

import time
from typing import Any, Dict

import streamlit as st
from app.components.product_card import render_product_grid


def render_tab_text_to_image(engine: Any, config: Dict[str, Any]) -> None:
    st.subheader("🔍 Tra cứu Sản phẩm từ Văn bản Tiếng Việt")
    st.markdown("Hỗ trợ truy vấn tự nhiên tiếng Việt, cả **có dấu** và **không dấu**.")

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        query_text = st.text_input(
            "Nhập mô tả sản phẩm (ví dụ: 'áo sơ mi nam màu xanh', 'vay hoa nu', 'giay the thao'):",
            placeholder="áo sơ mi nam ngắn tay màu xanh navy...",
        )
    with col_btn:
        st.write("")
        st.write("")
        btn_search = st.button("Tìm kiếm", key="btn_t2i", use_container_width=True)

    if btn_search and query_text.strip():
        with st.spinner("Đang tìm kiếm..."):
            t0 = time.perf_counter()
            # Pipeline call will be integrated here in Phase 6 demo
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if config.get("show_latency"):
                st.success(f"Thời gian phản hồi: **{elapsed_ms:.2f} ms** (SLA < 200 ms)")

            # Sample display placeholder
            st.markdown(f"**Kết quả Top-{config['top_k']} cho:** *'{query_text}'*")
            # render_product_grid(results, num_cols=5)
