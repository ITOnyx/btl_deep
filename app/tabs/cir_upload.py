"""
Tab 4: CIR from Uploaded Image (Composed Image Retrieval).
Fuses a user-uploaded image vector with a natural language modifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from PIL import Image
import streamlit as st

from app.components.errors import load_sample_image, show_search_error
from app.components.product_card import render_result_grid
from app.engine_service import validate_and_load_image, ImageValidationError


def render_tab_cir_upload(service: Any, config: Dict[str, Any]) -> None:
    st.caption("Tính năng thử nghiệm")
    st.caption("Tìm kiếm sản phẩm bằng cách tải ảnh lên và nhập yêu cầu chỉnh sửa.")

    # Initialize session state keys for persistence
    if "cir_upl_results" not in st.session_state:
        st.session_state["cir_upl_results"] = None
    if "cir_upl_elapsed_s" not in st.session_state:
        st.session_state["cir_upl_elapsed_s"] = 0.0

    col_input, col_preview = st.columns([3, 2])
    img_to_search: Image.Image | None = None
    img_source = None

    with col_input:
        uploaded_file = st.file_uploader(
            "Tải ảnh sản phẩm (JPG, PNG, WebP — tối đa 10 MB)",
            type=["jpg", "jpeg", "png", "webp"],
            key="cir_upload_file_uploader",
        )
        use_sample = st.checkbox("Dùng ảnh mẫu từ gallery", value=False, key="cir_upl_use_sample")
        if use_sample:
            sample_ids = ["15970", "1855", "7990", "6908"]
            chosen_pid = st.selectbox("Chọn ảnh mẫu", sample_ids, index=0, key="cir_upl_sample_select")
            img_source = f"sample:{chosen_pid}"
            sample_path = service.sample_image_path(chosen_pid)
            img_to_search = load_sample_image(sample_path)
        elif uploaded_file is not None:
            img_source = f"upload:{uploaded_file.file_id}"
            try:
                # In-memory validation (never written to disk)
                img_to_search = validate_and_load_image(
                    file_data=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                )
            except ImageValidationError as ve:
                st.error(str(ve))
                img_to_search = None
            except Exception as e:  # đọc file upload lỗi (mất kết nối giữa chừng...) không được làm vỡ trang
                show_search_error(e)
                img_to_search = None

        mod_text = st.text_input(
            "Yêu cầu chỉnh sửa",
            value="nhưng màu vàng",
            key="cir_upl_mod_input",
            placeholder="Ví dụ: nhưng màu vàng, kiểu thể thao, cho nữ...",
        )

        btn_search = st.button("Tìm kiếm", type="primary", key="btn_cir_upl_submit", use_container_width=True)

    # Ảnh truy vấn đổi (tải ảnh khác / chọn mẫu khác / bỏ ảnh) -> bỏ kết quả của ảnh cũ
    if st.session_state.get("cir_upl_img_source") != img_source:
        st.session_state["cir_upl_results"] = None
        st.session_state["cir_upl_img_source"] = img_source

    with col_preview:
        if img_to_search is not None:
            st.image(img_to_search, caption=f"Ảnh tham chiếu ({img_to_search.size[0]}x{img_to_search.size[1]} px)", width=160)

    if btn_search and img_to_search is None:
        st.warning("Vui lòng tải ảnh lên hoặc chọn ảnh mẫu trước khi tìm kiếm.")
    elif btn_search and not mod_text.strip():
        st.warning("Vui lòng nhập yêu cầu chỉnh sửa.")
    elif btn_search:
        with st.spinner("Đang tìm kiếm..."):
            try:
                results, _, elapsed_ms = service.search_cir_upload(
                    uploaded_image=img_to_search,
                    modifier_text=mod_text.strip(),
                    config_id=config["cir_config_id"],
                    top_k=config["top_k"],
                )
                st.session_state["cir_upl_results"] = results
                st.session_state["cir_upl_elapsed_s"] = elapsed_ms / 1000.0
            except Exception as e:
                show_search_error(e)
                st.session_state["cir_upl_results"] = None

    saved_results = st.session_state.get("cir_upl_results")
    if saved_results is not None:
        elapsed_s = st.session_state.get("cir_upl_elapsed_s", 0.0)
        st.caption(f"Tìm thấy {len(saved_results)} kết quả trong {elapsed_s:.2f} giây")
        if saved_results and saved_results[0].get("unrecognized_domain"):
            st.warning(
                "Không nhận diện được loại sản phẩm trong ảnh (có thể không phải ảnh thời trang). "
                "Kết quả chỉ dựa trên độ tương đồng, không lọc theo lĩnh vực."
            )
        if (len(saved_results) < config.get("top_k", 10) and config["cir_config_id"] in ("cir_v1_06", "cir_v1_07")
                and not (saved_results and saved_results[0].get("unrecognized_domain"))):
            st.caption(f"Chỉ có {len(saved_results)} sản phẩm cùng lĩnh vực phù hợp.")
        render_result_grid(saved_results, num_cols=5)
