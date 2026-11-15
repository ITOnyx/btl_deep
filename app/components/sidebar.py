"""
Sidebar component for Streamlit Application.
Displays system status, model configurations, and query parameter controls.
"""
from __future__ import annotations

import streamlit as st


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Cấu hình Hệ thống")

        st.markdown(
            """
            **Mô hình Nền tảng:**
            - **Vision-Language:** SigLIP ViT-SO400M (1152-d)
            - **Dense Captioning:** InternVL2-2B
            - **Chỉ mục:** FAISS HNSW (`IndexHNSWFlat`)
            - **Cấu hình Thắng:** `cfg_v5_09` (T2I), `cir_v1_05` (CIR)
            """
        )

        st.divider()

        top_k = st.slider("Số lượng kết quả (Top-K):", min_value=5, max_value=20, value=10, step=1)
        use_c3_bonus = st.toggle("Kích hoạt C3 Attribute Soft Bonus", value=True)
        show_latency = st.toggle("Hiển thị thời gian phản hồi (ms)", value=True)

        st.divider()
        st.caption("Khóa bảo vệ Hash Guard: Checkpoint SHA-256 e2f82fee... | Seed: 42")

        return {
            "top_k": top_k,
            "use_c3_bonus": use_c3,
            "show_latency": show_latency,
        }
