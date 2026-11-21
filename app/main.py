"""
Main entrypoint for the Fashion Product Multimodal Search Streamlit Application.
Implements Phase 6 requirements with a minimalist, clean, and professional UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so that `app` and `src` packages can be resolved
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.engine_service import AppSearchService, HashGuardError
from app.components.sidebar import render_sidebar
from app.tabs.text_to_image import render_tab_text_to_image
from app.tabs.image_to_image import render_tab_image_to_image
from app.tabs.cir_catalog import render_tab_cir_catalog
from app.tabs.cir_upload import render_tab_cir_upload

# Configure page without icons
st.set_page_config(
    page_title="Tra cứu sản phẩm thời trang",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    """Injects minimalist dark theme styling matching .streamlit/config.toml."""
    custom_css = """
    <style>
    /* Hide Streamlit default header decorations and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container max-width & padding */
    .main .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    /* Sidebar width control */
    section[data-testid="stSidebar"] {
        min-width: 280px;
        max-width: 300px;
    }

    /* Uniform heights for text inputs and buttons in search forms */
    .stTextInput > div > div > input {
        height: 40px;
        font-size: 14px;
        border-radius: 6px;
    }
    .stButton > button {
        height: 40px;
        font-size: 14px;
        border-radius: 6px;
        transition: all 0.15s ease-in-out;
    }

    /* Product Grid Card Styling */
    .product-img-container {
        width: 100%;
        aspect-ratio: 1 / 1;
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 6px;
    }
    .product-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }
    .product-img-placeholder {
        color: #6E7681;
        font-size: 12px;
        background-color: #12151C;
    }
    .product-info-container {
        padding: 2px 0 4px 0;
    }
    .product-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 2px;
    }
    .product-rank {
        color: #4C8BF5;
        font-weight: 600;
    }
    .product-score {
        color: #8B949E;
    }
    .product-title {
        font-size: 13px;
        font-weight: 500;
        line-height: 1.3;
        height: 34px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        color: #E6E8EB;
        margin-bottom: 2px;
    }
    .product-id {
        font-size: 11px;
        color: #6E7681;
    }

    /* Popover button for details */
    .stPopover {
        margin-top: 4px;
    }
    .stPopover > button {
        height: 28px !important;
        min-height: 28px !important;
        font-size: 11px !important;
        padding: 0 8px !important;
        border-radius: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: #181B22 !important;
        color: #8B949E !important;
        transition: all 0.15s ease-in-out;
    }
    .stPopover > button:hover {
        border-color: #4C8BF5 !important;
        color: #E6E8EB !important;
    }

    /* Detail content inside popover */
    .detail-content {
        font-size: 12px;
        color: #E6E8EB;
        line-height: 1.6;
    }
    .detail-content code {
        color: #4C8BF5;
        font-size: 11px;
    }

    /* Chips button styling — secondary buttons via Streamlit data-testid */
    [data-testid="stBaseButton-secondary"],
    [data-testid="baseButton-secondary"] {
        height: 32px !important;
        min-height: 32px !important;
        font-size: 12px !important;
        border-radius: 16px !important;
        padding: 0 10px !important;
        background-color: #181B22 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #C9D1D9 !important;
    }
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="baseButton-secondary"]:hover {
        border-color: #4C8BF5 !important;
        color: #FFFFFF !important;
    }

    /* Sidebar Title */
    .sidebar-title {
        font-size: 16px;
        font-weight: 600;
        color: #E6E8EB;
        margin-bottom: 1rem;
    }
    .system-info-box {
        font-size: 12px;
        color: #8B949E;
        line-height: 1.6;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang khởi tạo mô hình SigLIP ViT-SO400M...")
def get_cached_search_service(gallery_name: str) -> AppSearchService:
    """Loads and caches the search service instance for the selected gallery."""
    index_path = f"index/{gallery_name}"
    return AppSearchService(index_dir=index_path, config_path="config.yaml")


def main() -> None:
    inject_custom_css()

    st.title("Tra cứu sản phẩm thời trang")
    st.caption("Tìm kiếm sản phẩm bằng văn bản hoặc hình ảnh.")

    # Temporary gallery selection state before service load
    active_gallery = st.session_state.get("active_gallery", "test_gallery")

    # Load service with Hash Guard
    try:
        service = get_cached_search_service(active_gallery)
        service_meta = service.get_system_metadata()
    except HashGuardError as hge:
        st.error(f"Lỗi bảo mật Hash Guard: {hge}")
        st.stop()
    except Exception as e:
        st.error(f"Không thể khởi động hệ thống: {e}")
        st.stop()

    # Render sidebar with controls and metadata
    config = render_sidebar(service_meta)

    # Drop saved results when the settings that produced them change, so stale
    # results (and the "Chỉ có N sản phẩm..." caption) never outlive their config.
    settings_sig = (
        config["gallery_name"],
        config["t2i_config_id"],
        config["cir_config_id"],
        config["top_k"],
    )
    if st.session_state.get("results_settings_sig") != settings_sig:
        for key in ("t2i_results", "i2i_results", "cir_cat_results", "cir_upl_results"):
            st.session_state[key] = None
        st.session_state["results_settings_sig"] = settings_sig

    # Handle gallery change
    if config["gallery_name"] != active_gallery:
        st.session_state["active_gallery"] = config["gallery_name"]
        st.rerun()

    # 4 Stream Tabs with clean names and no emojis
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Văn bản",
            "Hình ảnh",
            "Kết hợp (Catalog)",
            "Kết hợp (Tải lên)",
        ]
    )

    with tab1:
        render_tab_text_to_image(service, config)

    with tab2:
        render_tab_image_to_image(service, config)

    with tab3:
        render_tab_cir_catalog(service, config)

    with tab4:
        render_tab_cir_upload(service, config)


if __name__ == "__main__":
    main()
