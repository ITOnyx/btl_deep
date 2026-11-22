"""
Sidebar component for Streamlit application.
Minimalist layout with clean selectboxes and consolidated system details.
"""
from __future__ import annotations

from typing import Any, Dict
import streamlit as st


def render_sidebar(service_metadata: Dict[str, Any]) -> Dict[str, Any]:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Cài đặt</div>', unsafe_allow_html=True)

        # 1. Gallery Selection
        gallery_options = {
            "test_gallery": "test_gallery",
            "reference_gallery": "reference_gallery",
        }
        selected_gallery = st.selectbox(
            "Bộ sưu tập",
            options=list(gallery_options.keys()),
            format_func=lambda k: gallery_options[k],
            index=0,
            key="sidebar_gallery",
            help=(
                "test_gallery: 1,290 sản phẩm test split dùng để đánh giá chuẩn.\n"
                "reference_gallery: 38,646 sản phẩm catalog lớn (ước lượng lạc quan do chứa ảnh train)."
            ),
        )

        # 2. Text to Image Configurations
        t2i_configs = {
            "cfg_v5_11": "cfg_v5_11",
            "cfg_v5_10": "cfg_v5_10",
            "cfg_v5_09": "cfg_v5_09",
            "cfg_v5_01": "cfg_v5_01",
            "model_only": "model_only",
        }
        selected_t2i_cfg = st.selectbox(
            "Cấu hình Text - Ảnh",
            options=list(t2i_configs.keys()),
            format_func=lambda k: t2i_configs[k],
            index=0,
            help=(
                "cfg_v5_11 (mặc định, DEV SET): cfg_v5_10 + khôi phục dấu trước khi encode + parser v3; "
                "xử lý tốt truy vấn không dấu (chưa chạy trên held-out).\n"
                "cfg_v5_10: Phân loại 3 tầng + Chặn lệch lĩnh vực sản phẩm (có số liệu held-out; "
                "kém với truy vấn không dấu).\n"
                "cfg_v5_09: Cấu hình baseline Pha 5c (gamma 1.0, tier_soft bonus).\n"
                "cfg_v5_01: Cấu hình an toàn baseline (gamma 1.5).\n"
                "model_only: Cosine thô của mô hình SigLIP, không dùng parser."
            ),
        )

        # 3. CIR Configurations
        cir_configs = {
            "cir_v1_07": "cir_v1_07",
            "cir_v1_06": "cir_v1_06",
            "cir_v1_05": "cir_v1_05",
            "cir_v1_01": "cir_v1_01",
        }
        selected_cir_cfg = st.selectbox(
            "Cấu hình CIR",
            options=list(cir_configs.keys()),
            format_func=lambda k: cir_configs[k],
            index=0,
            help=(
                "cir_v1_07 (mặc định, DEV SET): dùng vector ảnh tham chiếu (không qua combiner) + giữ loại/màu/giới tính, "
                "đổi theo modifier; xử lý tốt đổi giới tính và câu không dấu (chưa chạy trên held-out).\n"
                "cir_v1_06: Phân loại 3 tầng + Giữ nguyên lĩnh vực/loại ảnh tham chiếu (có số liệu held-out; "
                "kém khi đổi giới tính).\n"
                "cir_v1_05: Cấu hình baseline Pha 5c (bộ nối F.2 + C3 soft attribute bonus).\n"
                "cir_v1_01: Bộ nối fused combiner thuần tiếng Anh."
            ),
        )

        # 4. Top-K Slider
        top_k = st.slider("Số lượng kết quả", min_value=1, max_value=20, value=10, step=1)

        # 5. Consolidated System Information Expander (collapsed by default)
        with st.expander("Thông tin hệ thống", expanded=False):
            device_str = service_metadata.get("device", "cpu")
            dev_name = service_metadata.get("device_name", "")
            vram_mb = service_metadata.get("vram_allocated_mb", 0.0)
            vram_gb = service_metadata.get("vram_total_gb", 0.0)
            vram_display = f"{vram_mb:.0f} MB / {vram_gb:.1f} GB" if vram_gb > 0 else "N/A"
            ckpt_hash = service_metadata.get("checkpoint_hash", "")
            short_hash = f"{ckpt_hash[:16]}..." if ckpt_hash else "N/A"
            total_items = service_metadata.get("total_items")
            total_items_str = f"{total_items:,} vector" if isinstance(total_items, int) else "N/A"
            guard_ok = service_metadata.get("hash_guard_ok")
            guard_str = "Hợp lệ" if guard_ok is True else ("Không hợp lệ" if guard_ok is False else "N/A")
            seed = service_metadata.get("seed", "N/A")

            st.markdown(
                f"""
                <div class="system-info-box">
                    <div>Thiết bị: {device_str} ({dev_name})</div>
                    <div>VRAM: {vram_display}</div>
                    <div>Checkpoint: {short_hash}</div>
                    <div>Hash Guard: {guard_str}</div>
                    <div>Quy mô gallery: {total_items_str}</div>
                    <div>Seed: {seed}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        return {
            "gallery_name": selected_gallery,
            "t2i_config_id": selected_t2i_cfg,
            "cir_config_id": selected_cir_cfg,
            "top_k": top_k,
        }
