"""
Main entrypoint for the Fashion Product Multimodal Search Streamlit Application.
Implements Phase 6 requirements according to AGENTS.md (Rules A, B, E).
"""
from __future__ import annotations

import streamlit as st
from app.components.sidebar import render_sidebar
from app.tabs.cir_catalog import render_tab_cir_catalog
from app.tabs.cir_upload import render_tab_cir_upload
from app.tabs.image_to_image import render_tab_image_to_image
from app.tabs.text_to_image import render_tab_text_to_image

st.set_page_config(
    page_title="Hệ thống Tra cứu Sản phẩm Đa phương thức (SigLIP + InternVL2)",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("🛍️ Hệ thống Tra cứu Sản phẩm Thời trang Đa phương thức")
    st.caption("Kiến trúc SigLIP ViT-SO400M + Dense Captioning InternVL2-2B | FAISS HNSW Index | Tiếng Việt")

    config = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔤 1. Text → Ảnh",
            "🖼️ 2. Ảnh → Ảnh",
            "🔄 3. CIR (Catalog)",
            "📤 4. CIR (Tải lên)",
        ]
    )

    engine = None  # Engine initialization placeholder for live service

    with tab1:
        render_tab_text_to_image(engine, config)

    with tab2:
        render_tab_image_to_image(engine, config)

    with tab3:
        render_tab_cir_catalog(engine, config)

    with tab4:
        render_tab_cir_upload(engine, config)


if __name__ == "__main__":
    main()
