"""
Tab 3: CIR from Catalog Reference (Composed Image Retrieval).
Fuses a known catalog reference product vector with a text modifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import streamlit as st

from app.components.product_card import render_result_grid


def _apply_chip(pid: str, mod: str) -> None:
    """on_click của chip gợi ý: đặt giá trị cho 2 ô nhập (có key) TRƯỚC khi chúng được vẽ lại,
    để ô nhập luôn khớp với truy vấn vừa chạy và lần bấm "Tìm kiếm" sau đó dùng đúng giá trị."""
    st.session_state["cir_cat_pid_input"] = pid
    st.session_state["cir_cat_mod_input"] = mod
    st.session_state["cir_cat_pending_chip"] = (pid, mod)


def render_tab_cir_catalog(service: Any, config: Dict[str, Any]) -> None:
    st.caption("Tính năng thử nghiệm")
    st.caption("Tìm kiếm sản phẩm từ mã sản phẩm tham chiếu và yêu cầu chỉnh sửa.")

    # Initialize session state keys for persistence
    if "cir_cat_pid" not in st.session_state:
        st.session_state["cir_cat_pid"] = "15970"
    if "cir_cat_mod" not in st.session_state:
        st.session_state["cir_cat_mod"] = "nhưng màu đỏ"
    if "cir_cat_pid_input" not in st.session_state:
        st.session_state["cir_cat_pid_input"] = st.session_state["cir_cat_pid"]
    if "cir_cat_mod_input" not in st.session_state:
        st.session_state["cir_cat_mod_input"] = st.session_state["cir_cat_mod"]
    if "cir_cat_results" not in st.session_state:
        st.session_state["cir_cat_results"] = None
    if "cir_cat_elapsed_s" not in st.session_state:
        st.session_state["cir_cat_elapsed_s"] = 0.0

    # Quick sample buttons
    st.caption("Gợi ý tham chiếu:")
    sample_cols = st.columns(4)
    sample_refs = [
        ("15970", "Áo sơ mi 15970 (nhưng màu đỏ)", "15970", "nhưng màu đỏ"),
        ("1855", "Áo thun 1855 (nhưng màu hồng)", "1855", "nhưng cho nữ màu hồng"),
        ("7990", "Áo thun 7990 (nhưng màu xanh lá)", "7990", "nhưng màu xanh lá"),
        ("8485", "Đầm 8485 (nhưng màu hồng)", "8485", "nhưng màu hồng"),
    ]
    for idx, (_, label, s_pid, s_mod) in enumerate(sample_refs):
        with sample_cols[idx]:
            st.button(label, key=f"cir_cat_chip_{idx}", use_container_width=True,
                      on_click=_apply_chip, args=(s_pid, s_mod))
    sample_selected = st.session_state.pop("cir_cat_pending_chip", None)

    if sample_selected:
        st.session_state["cir_cat_pid"] = sample_selected[0]
        st.session_state["cir_cat_mod"] = sample_selected[1]

    # Input Form
    with st.form("form_cir_catalog", border=False):
        col_pid, col_mod, col_btn = st.columns([2, 3, 1])
        with col_pid:
            pid_val = st.text_input(
                "Mã sản phẩm",
                placeholder="Ví dụ: 15970",
                key="cir_cat_pid_input",
            )
        with col_mod:
            mod_val = st.text_input(
                "Yêu cầu chỉnh sửa",
                placeholder="Ví dụ: nhưng màu đỏ, cho nữ...",
                key="cir_cat_mod_input",
            )
        with col_btn:
            st.markdown('<div style="height: 28px;"></div>', unsafe_allow_html=True)
            submit_search = st.form_submit_button("Tìm kiếm", type="primary", use_container_width=True)

    # Reference Image Preview
    active_pid = pid_val.strip()
    if active_pid:
        sample_img_path = Path("data/fashion-dataset/images") / f"{active_pid}.jpg"
        if sample_img_path.exists():
            st.caption(f"Ảnh tham chiếu ID {active_pid}")
            st.image(str(sample_img_path), width=120)

    # Search Trigger
    should_run = False
    if sample_selected:
        should_run = True
    elif submit_search and active_pid and mod_val.strip():
        st.session_state["cir_cat_pid"] = active_pid
        st.session_state["cir_cat_mod"] = mod_val.strip()
        should_run = True

    if should_run:
        with st.spinner("Đang tìm kiếm..."):
            try:
                results, _, elapsed_ms = service.search_cir_catalog(
                    product_id=st.session_state["cir_cat_pid"],
                    modifier_text=st.session_state["cir_cat_mod"],
                    config_id=config["cir_config_id"],
                    top_k=config["top_k"],
                )
                st.session_state["cir_cat_results"] = results
                st.session_state["cir_cat_elapsed_s"] = elapsed_ms / 1000.0
            except ValueError as ve:
                st.error(f"Lỗi: {ve}")
                st.session_state["cir_cat_results"] = None
            except Exception as e:
                st.error(f"Lỗi tìm kiếm: {e}")
                st.session_state["cir_cat_results"] = None

    # Render saved results
    saved_results = st.session_state.get("cir_cat_results")
    if saved_results is not None:
        elapsed_s = st.session_state.get("cir_cat_elapsed_s", 0.0)
        st.caption(f"Tìm thấy {len(saved_results)} kết quả trong {elapsed_s:.2f} giây")
        if len(saved_results) < config.get("top_k", 10):
            st.caption(f"Chỉ có {len(saved_results)} sản phẩm cùng lĩnh vực phù hợp.")
        render_result_grid(saved_results, num_cols=5)
