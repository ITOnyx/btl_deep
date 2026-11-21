"""
Tab 1: Text-to-Image Search (Văn bản).
Minimalist layout with inline search form, quick chips, and session state persistence.
"""
from __future__ import annotations

from typing import Any, Dict
import streamlit as st

from app.components.product_card import render_result_grid


def _apply_chip(text: str) -> None:
    """on_click của chip gợi ý: chạy TRƯỚC khi widget được vẽ lại, nên ô nhập hiển thị đúng câu của chip
    và câu người dùng gõ sau đó không bị mất (widget có key cố định, không bị tạo lại)."""
    st.session_state["t2i_query_input"] = text
    st.session_state["t2i_pending_chip"] = text


def render_tab_text_to_image(service: Any, config: Dict[str, Any]) -> None:
    st.caption("Nhập mô tả bằng tiếng Việt, có dấu hoặc không dấu.")

    # Initialize session state keys for persistence
    if "t2i_query" not in st.session_state:
        st.session_state["t2i_query"] = "túi xách màu đỏ"
    if "t2i_query_input" not in st.session_state:
        st.session_state["t2i_query_input"] = st.session_state["t2i_query"]
    if "t2i_results" not in st.session_state:
        st.session_state["t2i_results"] = None
    if "t2i_elapsed_s" not in st.session_state:
        st.session_state["t2i_elapsed_s"] = 0.0

    # Inline search form (5:1 ratio, vertically aligned, Enter-to-submit)
    with st.form("form_text_search", border=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query_val = st.text_input(
                "Mô tả sản phẩm",
                key="t2i_query_input",
                placeholder="Ví dụ: áo sơ mi nam màu xanh",
                label_visibility="collapsed",
            )
        with col_btn:
            submit_search = st.form_submit_button("Tìm kiếm", type="primary", use_container_width=True)

    # 4 Quick suggestion chips (starting from left margin, aligned with label)
    chip_cols = st.columns([0.8, 2.5, 1.8, 2.2, 2.5, 2.2])
    with chip_cols[0]:
        st.markdown(
            '<div style="line-height: 38px; font-size: 13px; color: #8B949E; font-weight: 500;">Gợi ý:</div>',
            unsafe_allow_html=True,
        )
    suggestions = [
        "áo sơ mi nam màu xanh",
        "váy hoa nữ",
        "giày thể thao nam",
        "đồng hồ nữ màu đen",
    ]
    for i, s_text in enumerate(suggestions):
        with chip_cols[i + 1]:
            st.button(s_text, key=f"t2i_chip_{i}", use_container_width=True, on_click=_apply_chip, args=(s_text,))
    chip_clicked = st.session_state.pop("t2i_pending_chip", None)

    # Determine if search should run
    should_run = False
    active_query = query_val.strip()

    if chip_clicked:
        active_query = chip_clicked
        st.session_state["t2i_query"] = chip_clicked
        should_run = True
    elif submit_search and active_query:
        st.session_state["t2i_query"] = active_query
        should_run = True

    if should_run:
        with st.spinner("Đang tìm kiếm..."):
            try:
                results, _, elapsed_ms = service.search_text_to_image(
                    query_text=active_query,
                    config_id=config["t2i_config_id"],
                    top_k=config["top_k"],
                )
                st.session_state["t2i_results"] = results
                st.session_state["t2i_elapsed_s"] = elapsed_ms / 1000.0
            except Exception as e:
                st.error(f"Lỗi tìm kiếm: {e}")
                st.session_state["t2i_results"] = None

    # Render results from session_state
    saved_results = st.session_state.get("t2i_results")
    if saved_results is not None:
        elapsed_s = st.session_state.get("t2i_elapsed_s", 0.0)
        st.caption(f"Tìm thấy {len(saved_results)} kết quả trong {elapsed_s:.2f} giây")
        if len(saved_results) < config.get("top_k", 10):
            st.caption(f"Chỉ có {len(saved_results)} sản phẩm cùng lĩnh vực phù hợp.")
        render_result_grid(saved_results, num_cols=5)
