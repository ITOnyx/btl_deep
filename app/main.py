"""
Streamlit Web Application for SigLIP E-commerce Visual Search (app/main.py)
Project: Hoc_sau_ver4 (Phase 5 - GD2 Clean UI)
Clean, modern, accessible eCommerce user interface with WCAG AA compliance.
"""

import os
import re
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import streamlit as st
from PIL import Image

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.engine import SearchService, verify_rule_7_checkpoint_hash
from src.vi_vocab import ARTICLE_TYPE_MAP, COLOUR_MAP, GENDER_MAP

# Streamlit Page Configuration
st.set_page_config(
    page_title="SigLIP E-commerce Visual Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern Dark eCommerce styling (WCAG AA compliant)
st.markdown("""
<style>
    /* Global Typography & Theme Polish */
    body, [class*="ViewTable"], .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .app-header {
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #334155;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        line-height: 1.4;
    }

    /* Product Card Styling */
    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .rank-tag {
        font-weight: 700;
        color: #38BDF8;
        font-size: 0.88rem;
    }
    .brand-tag {
        background: #334155;
        color: #E2E8F0;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        display: inline-block;
    }
    .card-title-vi {
        font-size: 0.92rem;
        font-weight: 600;
        color: #F8FAFC;
        line-height: 1.35;
        height: 2.5rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        text-overflow: ellipsis;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .card-tags-vi {
        font-size: 0.76rem;
        color: #94A3B8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 6px;
    }
    .card-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 6px;
        padding-top: 4px;
        border-top: 1px solid #334155;
    }
    .score-badge {
        font-weight: 700;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-family: monospace;
    }
    .id-subtle {
        font-size: 0.72rem;
        color: #64748B;
    }

    /* Metric Cards in Dark Mode */
    .metric-card-dark {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }
    .metric-label-dark {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-bottom: 2px;
    }
    .metric-val-dark {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .sla-pass-dark {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .sla-fail-dark {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Progress bar mini */
    .progress-bar-container {
        width: 100%;
        background-color: #334155;
        border-radius: 3px;
        height: 4px;
        margin-top: 3px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background-color: #38BDF8;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang khởi tạo VisualSearchEngine & kiểm toán an toàn (Luật 7)...")
def get_service() -> SearchService:
    """Cached singleton search service loaded from config.yaml."""
    return SearchService(config_path="config.yaml")


def load_json_safe(rel_path: str) -> Optional[Dict[str, Any]]:
    p = ROOT_DIR / rel_path
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -------------------------------------------------------------
# 1. Initialize and Verify Service
# -------------------------------------------------------------
try:
    service = get_service()
    rule7_ok = True
    rule7_err = None
except Exception as e:
    service = None
    rule7_ok = False
    rule7_err = str(e)

if not rule7_ok or service is None:
    st.error(f"❌ Vi phạm Luật 7!\n{rule7_err}")
    st.stop()

cfg = service.config
inf_cfg = cfg.get("inference", {}) or {}
serving_cfg = cfg.get("serving", {}) or {}
eval_cfg = cfg.get("eval", {}) or {}
gender_gt_cfg = eval_cfg.get("gender_gt", {}) or {}
default_gamma = float(inf_cfg.get("gamma", cfg.get("scoring", {}).get("mu_gamma", 0.3)))
default_i2i_mode = str(serving_cfg.get("i2i_mode", "fast"))


# -------------------------------------------------------------
# 2. Sidebar Controls
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### Cấu hình Hệ thống")
    
    # Toggle Chế độ kỹ thuật (mặc định tắt cho người dùng cuối)
    tech_mode = st.toggle(
        "Chế độ kỹ thuật",
        value=False,
        help="Bật để xem thông số chi tiết mô hình, phân rã thời gian từng bước và kiểm toán hạ tầng."
    )

    if tech_mode:
        st.markdown(
            f"""<div style="background:#1E293B; border:1px solid #3B82F6; border-radius:6px; padding:0.6rem; margin-bottom:0.8rem;">
            <div style="color:#60A5FA; font-weight:700; font-size:0.85rem; margin-bottom:4px;">Kiểm toán an toàn (Luật 7)</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Checkpoint SHA-256:<br><code style="color:#38BDF8;">{service.cfg_hash[:16]}...</code></div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:3px;">Index Manifest:<br><code style="color:#38BDF8;">{service.man_hash[:16]}...</code></div>
            </div>""",
            unsafe_allow_html=True
        )
        st.caption(f"Catalog in-scope: **{service.engine.N:,}** sản phẩm")
        st.caption(f"Phần cứng: **{service.engine.device.upper()}** (FP16)")
        st.caption(
            f"GT Giới tính: **{gender_gt_cfg.get('official', 'hybrid').upper()}** "
            f"(`{gender_gt_cfg.get('policy_file', 'eval/gender_gt_policy.json')}`)"
        )
    else:
        # Trong chế độ người dùng: Khối kiểm toán Luật 7 luôn có thể truy cập qua 1 lần click
        with st.expander("Thông tin kiểm toán an toàn (Luật 7)", expanded=False):
            st.markdown(f"**Trạng thái:** Đạt kiểm toán an toàn")
            st.caption(f"Checkpoint SHA-256: `{service.cfg_hash[:16]}...`")
            st.caption(f"Index Manifest: `{service.man_hash[:16]}...`")
            st.caption(f"Số lượng catalog: {service.engine.N:,} sản phẩm")

    st.markdown("---")
    st.markdown("#### Định mức SLA theo luồng")
    st.markdown(
        f"""
        - Text → Ảnh: **< {service.get_flow_sla_ms('text_to_image'):.0f} ms**
        - Ảnh → Ảnh: **< {service.get_flow_sla_ms('image_to_image'):.0f} ms**
        - CIR Catalog: **< {service.get_flow_sla_ms('cir_catalog'):.0f} ms**
        - CIR Upload: **< {service.get_flow_sla_ms('cir_upload'):.0f} ms**
        """
    )


# -------------------------------------------------------------
# 3. Main Header & App Tabs
# -------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="main-title">SigLIP E-commerce Visual Search</div>
    <div class="subtitle">Hệ sinh thái tra cứu thời trang đa phương thức: Văn bản tiếng Việt, Ảnh tương đồng và Tìm kiếm kết hợp (CIR)</div>
</div>
""", unsafe_allow_html=True)

tab_text, tab_image, tab_cir, tab_benchmark = st.tabs([
    "Text tiếng Việt → Ảnh",
    "Ảnh → Ảnh (Visual Search)",
    "Ảnh + Text bổ trợ (CIR)",
    "Đối soát SLA, GT & Báo cáo"
])


# -------------------------------------------------------------
# Helper: Render Product Card Grid (Standardized Height & Clean Typography)
# -------------------------------------------------------------
def render_product_grid(results: List[Dict[str, Any]], cols_per_row: int = 5, show_tech: bool = False):
    if not results:
        st.info("Không tìm thấy sản phẩm phù hợp với mô tả.")
        return

    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, item_idx in enumerate(range(i, min(i + cols_per_row, len(results)))):
            prod = results[item_idx]
            with cols[col_idx]:
                # 1. Top row: Rank & Brand
                rank_str = f"#{prod['rank']}"
                brand_val = str(prod.get("brand", "")).strip()
                has_brand = bool(brand_val and brand_val not in ("None", "nan", "Unknown", ""))

                brand_html = f"<span class='brand-tag' title='{brand_val}'>{brand_val}</span>" if has_brand else ""
                st.markdown(
                    f"""<div class="card-top">
                        <span class="rank-tag">{rank_str}</span>
                        {brand_html}
                    </div>""",
                    unsafe_allow_html=True
                )

                # 2. Product Image
                img_path = prod.get("image_path", "")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    # Clean fallback placeholder
                    st.markdown(
                        """<div style="height:190px; background:#0B132B; border-radius:6px; display:flex; align-items:center; justify-content:center; color:#64748B; font-size:0.8rem;">
                        (Không tìm thấy ảnh)
                        </div>""",
                        unsafe_allow_html=True
                    )

                # 3. Product Title (Vietnamese primary, English catalog in tooltip)
                title_vi = prod.get("name_vi") or prod.get("name", "")
                raw_name = prod.get("name", "")
                st.markdown(
                    f"""<div class="card-title-vi" title="{title_vi} (Catalog: {raw_name})">
                        {title_vi}
                    </div>""",
                    unsafe_allow_html=True
                )

                # 4. Standardized Vietnamese Tags from vi_map
                raw_type = prod.get("articleType", "")
                raw_col = prod.get("baseColour", "")
                raw_gen = prod.get("gender", "")

                vn_type = ARTICLE_TYPE_MAP.get(raw_type, raw_type)
                vn_col = COLOUR_MAP.get(raw_col, raw_col)
                vn_gen = GENDER_MAP.get(raw_gen, raw_gen)

                tag_parts = []
                if vn_type:
                    tag_parts.append(vn_type)
                if vn_col:
                    tag_parts.append(vn_col)
                if vn_gen:
                    tag_parts.append(vn_gen)
                tags_str = " • ".join(tag_parts)

                st.markdown(
                    f"""<div class="card-tags-vi" title="{tags_str}">
                        {tags_str}
                    </div>""",
                    unsafe_allow_html=True
                )

                # 5. Score & ID Row
                score_val = float(prod.get("score", 0.0))
                score_pct = max(0, min(100, int(score_val * 2500))) # normalized visual progress
                id_html = f"<span class='id-subtle'>ID: {prod['item_id']}</span>" if show_tech else ""

                st.markdown(
                    f"""<div class="card-meta-row">
                        <div>
                            <span class="score-badge">{score_val:.3f}</span>
                            <div class="progress-bar-container">
                                <div class="progress-bar-fill" style="width: {score_pct}%;"></div>
                            </div>
                        </div>
                        {id_html}
                    </div>""",
                    unsafe_allow_html=True
                )
                st.write("")


# -------------------------------------------------------------
# TAB 1: Text-to-Image Search
# -------------------------------------------------------------
with tab_text:
    # 1. Clean Mode Selector
    search_mode_ui = st.radio(
        "Chế độ tìm kiếm:",
        options=["system", "model_pure"],
        format_func=lambda x: "Hệ thống đầy đủ" if x == "system" else "Model thuần",
        horizontal=True,
        index=0,
        help="Hệ thống đầy đủ: Kết hợp SigLIP Dense, QueryParser V2 bóc tách thuộc tính và BM25 RRF. Model thuần: Chỉ dùng vector SigLIP Dense nguyên bản không lọc thuộc tính."
    )

    # Advanced Algorithm Options in Expander
    with st.expander("Tùy chọn nâng cao", expanded=False):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            adv_config = st.selectbox(
                "Cấu hình thuật toán",
                options=["c5", "c2", "c4"],
                format_func=lambda x: {
                    "c5": "C5: Hybrid Đầy đủ (Parser + Soft Bonus + BM25)",
                    "c2": "C2: Lọc nới lỏng + Soft Bonus + Dense (không BM25)",
                    "c4": "C4: BM25 Sparse thuần từ vựng"
                }[x],
                index=0
            )
        with adv_col2:
            adv_gamma = st.slider(
                "Trọng số cộng điểm mềm (Soft bonus)",
                min_value=0.0,
                max_value=1.0,
                value=default_gamma,
                step=0.05,
                help="Thưởng điểm khi sản phẩm khớp các thuộc tính bóc tách (loại SP, giới tính, màu sắc, brand)."
            )

    # Determine effective config
    if search_mode_ui == "model_pure":
        effective_config = "c0"
        effective_gamma = 0.0
        st.caption("ℹ️ Đang chọn **Model thuần (C0)** — Tìm kiếm trực tiếp qua SigLIP Dense Tower, không lọc thuộc tính.")
    else:
        effective_config = adv_config
        effective_gamma = adv_gamma
        st.caption(f"Đang chọn **Hệ thống ({effective_config.upper()})** — Bóc tách thuộc tính, nới lỏng ứng viên đa tầng và cộng điểm mềm (γ={effective_gamma:.2f}).")

    st.write("")

    # 2. Sample Suggestions (Even Grid)
    st.markdown("<div style='font-size:0.86rem; font-weight:600; color:#94A3B8; margin-bottom:0.35rem;'>Thử ngay:</div>", unsafe_allow_html=True)
    sample_cols = st.columns(7)
    sample_queries = [
        "túi sách màu hồng",
        "áo thun Puma nam",
        "váy dài màu đỏ dạ hội",
        "giày cao gót nữ màu đen",
        "ao do nam",
        "đồ đi biển",
        "túi xách nữ Lavie"
    ]

    selected_sample = None
    for idx, (col, sq) in enumerate(zip(sample_cols, sample_queries)):
        with col:
            if st.button(sq, key=f"sq_btn_{idx}", use_container_width=True):
                selected_sample = sq

    # 3. Search Bar + Top-K + Button in One Unified Row
    search_col_inp, search_col_k, search_col_btn = st.columns([5.2, 1.3, 1.5])
    
    with search_col_inp:
        # Use session state or sample selection
        default_val = selected_sample if selected_sample else "áo thun Puma nam"
        query_input = st.text_input(
            "Mô tả sản phẩm:",
            value=default_val,
            placeholder="Ví dụ: áo sơ mi nam trắng, đầm dạ hội đỏ, túi xách...",
            label_visibility="collapsed"
        )
    with search_col_k:
        top_k = st.selectbox(
            "Số lượng",
            options=[5, 10, 20, 50],
            index=1,
            label_visibility="collapsed"
        )
    with search_col_btn:
        btn_search = st.button("Tìm kiếm", type="primary", use_container_width=True)

    # 4. Search Execution Logic
    if btn_search or selected_sample or query_input:
        cleaned_query = query_input.strip()
        if not cleaned_query:
            st.info("Vui lòng nhập mô tả sản phẩm để bắt đầu tra cứu.")
        else:
            with st.spinner("Đang tìm kiếm sản phẩm phù hợp..."):
                results, meta = service.search_text(
                    query=cleaned_query,
                    config_mode=effective_config,
                    top_k=top_k,
                    gamma=effective_gamma
                )

            total_lat = float(meta.get("breakdown", {}).get("total_ms", results[0]["latency_ms"] if results else 0.0))
            sla_ms = float(meta.get("sla_ms", service.get_flow_sla_ms("text_to_image")))
            is_pass_sla = total_lat <= sla_ms

            # Brand out-of-catalog notice
            parsed = meta.get("parsed", {}) or {}
            is_lavie = (
                (parsed.get("brand") and str(parsed.get("brand")).lower() == "lavie") or
                "lavie" in [str(t).lower() for t in parsed.get("unparsed_tokens", [])] or
                re.search(r"\blavie\b", cleaned_query, re.IGNORECASE) is not None
            )
            if is_lavie:
                st.warning("Thông báo: Thương hiệu 'Lavie' không có trong catalog; hệ thống đang tự động hiển thị các sản phẩm cùng danh mục có kiểu dáng tương tự.")

            # Header with results count and query time
            st.markdown(f"#### Kết quả tìm kiếm ({len(results)} sản phẩm • <span style='color:#38BDF8; font-weight:normal; font-size:0.95rem;'>{total_lat:.1f} ms</span>)", unsafe_allow_html=True)
            render_product_grid(results, cols_per_row=5, show_tech=tech_mode)

            # Performance & Breakdown block placed AFTER results in clean expander
            with st.expander("Thông tin hiệu năng & SLA chi tiết", expanded=False):
                bd = meta.get("breakdown", {}) or {}
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                with m1:
                    st.markdown(
                        f"""<div class="metric-card-dark">
                            <div class="metric-label-dark">SLA (&lt; {sla_ms:.0f} ms)</div>
                            <div class="{'sla-pass-dark' if is_pass_sla else 'sla-fail-dark'}">
                                {'ĐẠT SLA' if is_pass_sla else 'KHÔNG ĐẠT'}
                            </div>
                        </div>""", unsafe_allow_html=True
                    )
                with m2:
                    st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Tổng độ trễ</div><div class="metric-val-dark">{total_lat:.1f} ms</div></div>""", unsafe_allow_html=True)
                with m3:
                    parse_ms = float(bd.get("parse_ms", 0.0))
                    st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">QueryParser V2</div><div class="metric-val-dark">{parse_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
                with m4:
                    filter_ms = float(bd.get("filter_ms", 0.0))
                    st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Lọc ứng viên</div><div class="metric-val-dark">{filter_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
                with m5:
                    enc_ms = float(bd.get("encode_ms", 0.0))
                    st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">SigLIP Text Tower</div><div class="metric-val-dark">{enc_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
                with m6:
                    search_ms = float(bd.get("search_ms", 0.0))
                    st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Hybrid/BM25</div><div class="metric-val-dark">{search_ms:.1f} ms</div></div>""", unsafe_allow_html=True)

                if tech_mode:
                    st.write("")
                    st.caption(f"Tầng lọc: `{meta.get('tier', 'all')}` | Số ứng viên xét duyệt: `{meta.get('candidates_count', 0):,}`")
                    st.caption(f"Thuộc tính bóc tách: Loại=`{parsed.get('articleType')}` | Màu=`{parsed.get('baseColour')}` | Giới tính=`{parsed.get('gender')}` | Brand=`{parsed.get('brand')}`")


# -------------------------------------------------------------
# TAB 2: Image-to-Image Visual Search (flow: image_to_image)
# -------------------------------------------------------------
with tab_image:
    st.markdown("### Tra cứu Sản phẩm Tương đồng bằng Ảnh")

    img_col1, img_col2 = st.columns([1, 2])

    with img_col1:
        st.markdown("**1. Tải ảnh lên hoặc chọn ảnh mẫu:**")
        uploaded_file = st.file_uploader("Chọn tệp ảnh sản phẩm (PNG, JPG):", type=["png", "jpg", "jpeg"])

        sample_ids = ["15970", "39386", "59263", "1855"]
        sample_img_paths = [str(service.kaggle_images_dir / f"{sid}.jpg") for sid in sample_ids]
        available_samples = [p for p in sample_img_paths if Path(p).exists()]

        selected_sample_img = None
        if available_samples:
            sample_choice = st.selectbox(
                "Hoặc chọn nhanh từ ảnh mẫu catalog:",
                options=["(Không chọn)"] + available_samples,
                index=1 if uploaded_file is None else 0,
                format_func=lambda x: x if x == "(Không chọn)" else f"ID {Path(x).stem} ({Path(x).name})"
            )
            if sample_choice != "(Không chọn)":
                selected_sample_img = sample_choice

        st.markdown("**2. Chế độ suy luận:**")
        i2i_mode_choice = st.radio(
            "Chế độ `i2i_mode`:",
            options=["fast", "accurate"],
            index=0 if default_i2i_mode == "fast" else 1,
            format_func=lambda m: (
                "⚡ fast (1 ảnh gốc - Mặc định, ĐẠT SLA < 200ms)"
                if m == "fast"
                else "🔬 accurate (Batch-2 TTA Flip)"
            )
        )
        use_knn = st.checkbox("Bật k-NN Attribute Voting Rescoring", value=True)

        query_image = None
        exclude_self_id = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Ảnh truy vấn tải lên", use_container_width=True)
        elif selected_sample_img is not None:
            query_image = Image.open(selected_sample_img).convert("RGB")
            exclude_self_id = Path(selected_sample_img).stem
            sample_row = service.engine.df[service.engine.df["item_id"].astype(str) == str(exclude_self_id)]
            sample_name = str(sample_row.iloc[0]["name"]) if len(sample_row) > 0 else f"ID {exclude_self_id}"
            st.image(query_image, caption=f"Ảnh mẫu catalog: {sample_name}", use_container_width=True)
            st.caption(f"ℹ️ Đã tự động loại bỏ sản phẩm ID {exclude_self_id} khỏi gallery kết quả.")

    with img_col2:
        if query_image is not None:
            if st.button("Tìm kiếm ảnh tương đồng", type="primary", use_container_width=True):
                with st.spinner("Đang trích xuất đặc trưng ảnh SigLIP và so khớp..."):
                    img_results, img_meta = service.search_image(
                        query_image,
                        top_k=top_k,
                        exclude_self_id=exclude_self_id,
                        mode=i2i_mode_choice,
                        use_knn_voting=use_knn
                    )

                lat = float(img_meta.get("latency_ms", img_results[0]["latency_ms"] if img_results else 0.0))
                sla_ms = float(img_meta.get("sla_ms", service.get_flow_sla_ms("image_to_image")))
                is_pass = lat <= sla_ms

                st.markdown(f"#### Sản phẩm tương đồng ({len(img_results)} sp • <span style='color:#38BDF8;'>{lat:.1f} ms</span>):", unsafe_allow_html=True)
                render_product_grid(img_results, cols_per_row=4, show_tech=tech_mode)

                with st.expander("Thông tin hiệu năng Ảnh → Ảnh", expanded=False):
                    bd = img_meta.get("breakdown", {}) or {}
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    with ic1:
                        st.markdown(
                            f"""<div class="metric-card-dark">
                                <div class="metric-label-dark">SLA (&lt; {sla_ms:.0f} ms)</div>
                                <div class="{'sla-pass-dark' if is_pass else 'sla-fail-dark'}">
                                    {'ĐẠT SLA' if is_pass else 'KHÔNG ĐẠT'}
                                </div>
                            </div>""", unsafe_allow_html=True
                        )
                    with ic2:
                        st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Tổng độ trễ</div><div class="metric-val-dark">{lat:.1f} ms</div></div>""", unsafe_allow_html=True)
                    with ic3:
                        st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">ViT Forward</div><div class="metric-val-dark">{bd.get('vit_forward_ms', 0.0):.1f} ms</div></div>""", unsafe_allow_html=True)
                    with ic4:
                        srch_rr = float(bd.get("gallery_search_ms", 0.0)) + float(bd.get("rerank_ms", 0.0))
                        st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Matmul + Rerank</div><div class="metric-val-dark">{srch_rr:.1f} ms</div></div>""", unsafe_allow_html=True)
        else:
            st.info("Vui lòng tải lên hoặc chọn một ảnh sản phẩm để bắt đầu tra cứu thị giác.")


# -------------------------------------------------------------
# TAB 3: CIR (Composed Image Retrieval)
# -------------------------------------------------------------
with tab_cir:
    st.markdown("### Tra cứu Đa phương thức (CIR: Ảnh + Text điều chỉnh)")

    cir_c1, cir_c2 = st.columns([1, 2])

    with cir_c1:
        st.markdown("**1. Nguồn ảnh tham chiếu:**")
        cir_upload = st.file_uploader("Tải ảnh mới từ thiết bị:", type=["png", "jpg", "jpeg"], key="cir_up")

        cir_sample_ids = ["15970", "39386", "10866"]
        cir_sample_choices = [str(service.kaggle_images_dir / f"{sid}.jpg") for sid in cir_sample_ids]
        avail_cir = [p for p in cir_sample_choices if Path(p).exists()]
        cir_sample = None
        if avail_cir:
            cir_sc = st.selectbox(
                "Hoặc chọn ảnh có sẵn trong catalog:",
                ["(Chọn ảnh)"] + avail_cir,
                index=1 if cir_upload is None else 0,
                format_func=lambda x: x if x == "(Chọn ảnh)" else f"ID {Path(x).stem} ({Path(x).name})"
            )
            if cir_sc != "(Chọn ảnh)":
                cir_sample = cir_sc

        force_upload_mode = False
        ref_img_pil = None
        cir_ref_id = None

        if cir_upload is not None:
            ref_img_pil = Image.open(cir_upload).convert("RGB")
            force_upload_mode = True
            st.image(ref_img_pil, caption="Ảnh tải lên từ thiết bị (luồng cir_upload)", use_container_width=True)
        elif cir_sample is not None:
            ref_img_pil = Image.open(cir_sample).convert("RGB")
            cir_ref_id = Path(cir_sample).stem
            st.image(ref_img_pil, caption=f"Ảnh mẫu catalog ID {cir_ref_id}", use_container_width=True)
            force_upload_mode = st.checkbox(
                "Bỏ qua Catalog Cache (`force_upload=True`)",
                value=False,
                help="Khi tắt, ảnh catalog dùng vector trong GPU cache. Khi bật, hệ thống chạy lại ViT Image Tower."
            )

        st.markdown("**2. Văn bản điều chỉnh:**")
        modifier_input = st.text_input(
            "Yêu cầu biến đổi kiểu dáng / màu sắc:",
            value="màu xanh navy",
            placeholder="Ví dụ: màu đỏ, quần short, áo khoác..."
        )
        use_cir_bonus = st.checkbox("Cộng điểm thưởng thuộc tính mềm cho từ khóa điều chỉnh", value=True)

    with cir_c2:
        if ref_img_pil is not None and modifier_input.strip():
            if st.button("Tổng hợp Truy vấn & Tìm kiếm", type="primary", use_container_width=True):
                with st.spinner("Đang tổng hợp đặc trưng qua mô-đun CIRCombiner..."):
                    cir_results, cir_meta = service.search_cir(
                        ref_image=ref_img_pil if force_upload_mode or cir_ref_id is None else cir_sample,
                        modifier_text=modifier_input.strip(),
                        top_k=top_k,
                        use_soft_bonus=use_cir_bonus,
                        ref_id=cir_ref_id,
                        force_upload=force_upload_mode
                    )

                flow_executed = str(cir_meta.get("flow", "cir_upload"))
                sla_ms = float(cir_meta.get("sla_ms", service.get_flow_sla_ms(flow_executed)))
                lat = float(cir_meta.get("latency_ms", cir_results[0]["latency_ms"] if cir_results else 0.0))
                is_pass = lat <= sla_ms

                st.markdown(f"#### Kết quả sau khi biến đổi theo: *\"{modifier_input}\"* ({len(cir_results)} sp • <span style='color:#38BDF8;'>{lat:.1f} ms</span>):", unsafe_allow_html=True)
                render_product_grid(cir_results, cols_per_row=4, show_tech=tech_mode)

                with st.expander("Thông tin hiệu năng CIR", expanded=False):
                    bd = cir_meta.get("breakdown", {}) or {}
                    cc1, cc2, cc3 = st.columns(3)
                    with cc1:
                        st.markdown(
                            f"""<div class="metric-card-dark">
                                <div class="metric-label-dark">SLA ({flow_executed} &lt; {sla_ms:.0f} ms)</div>
                                <div class="{'sla-pass-dark' if is_pass else 'sla-fail-dark'}">
                                    {'ĐẠT SLA' if is_pass else 'KHÔNG ĐẠT'}
                                </div>
                            </div>""", unsafe_allow_html=True
                        )
                    with cc2:
                        st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Tổng độ trễ</div><div class="metric-val-dark">{lat:.1f} ms</div></div>""", unsafe_allow_html=True)
                    with cc3:
                        st.markdown(f"""<div class="metric-card-dark"><div class="metric-label-dark">Luồng thực thi</div><div class="metric-val-dark" style="font-size:0.95rem;">{flow_executed}</div></div>""", unsafe_allow_html=True)
        else:
            st.info("Vui lòng cung cấp ảnh tham chiếu và câu mô tả điều chỉnh để thực hiện tìm kiếm đa phương thức CIR.")


# -------------------------------------------------------------
# TAB 4: Benchmarks, SLA & GT Policy Report
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("### Báo cáo Đối soát Chất lượng, SLA 4 Luồng & Chính sách GT Giới tính")

    lat_v3_data = load_json_safe("reports/phase4/phase4_latency_e2e_v3.json")
    i2i_qual_data = load_json_safe("reports/phase4/phase5a_i2i_quality.json")

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("#### 1. Paired Retrieval (Q1b trên $G_{test}$, $n = 1,290$ ảnh test)")
        st.markdown(
            """
            | Cấu hình | Recall@10 | mAP (MRR) | Recall@1 | Recall@5 | Kết luận Chỉ tiêu |
            | :--- | :---: | :---: | :---: | :---: | :---: |
            | **Model thuần (C0 Dense)** | 0.5225 | 0.2362 | 0.1116 | 0.3543 | 🔴 Không đạt |
            | **Hệ thống (C2 Parser + Soft Bonus)** | **0.9008** | **0.5205** | **0.3442** | **0.7465** | 🟢 Đạt R@10 (≥ 0.85) |
            """
        )

        st.markdown("#### 2. Truy vấn Thuộc tính & Chính sách GT Giới tính")
        st.markdown(
            """
            - **Chính sách GT Giới tính (`eval/gender_gt_policy.json`):**
              - **Hybrid (Chính thức):** **P@10 = 0.9180**
              - **Strict (Chặt chẽ):** **P@10 = 0.9010**
              - **Relaxed (Nới lỏng):** **P@10 = 0.9180**
            - **Q2 (61 câu thuộc tính tổ hợp):** P@10 = **0.9984**, AP@10 = **0.9982**
            - **Unseen Testset v2 ($n=32$ câu mới đóng băng):** Mean AP@10 = **0.7864**, Mean P@10 = **0.8065**
            """
        )

    with b_col2:
        st.markdown("#### 3. Đối soát Độ trễ & SLA Riêng Từng Luồng (`phase4_latency_e2e_v3.json`)")
        if lat_v3_data and "5a_baseline" in lat_v3_data.get("phases", {}):
            phase_5a = lat_v3_data["phases"]["5a_baseline"]
            res_map = phase_5a.get("results", {})
            flow_labels = [
                ("text_to_image", "1. Text → Ảnh (C5 Hybrid)", "text_to_image"),
                ("image_to_image_fast", "2a. Ảnh → Ảnh (mode=fast, Mặc định)", "image_to_image"),
                ("image_to_image_accurate", "2b. Ảnh → Ảnh (mode=accurate)", "image_to_image"),
                ("cir_catalog", "3a. CIR Catalog (GPU Cache)", "cir_catalog"),
                ("cir_upload", "3b. CIR Upload (ViT Forward)", "cir_upload"),
            ]
            rows_md = [
                "| Luồng / Cấu hình | SLA | Mean p50 | Mean p95 | Dải p95 (5 runs) | Kết luận SLA |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |"
            ]
            for cfg_key, label, flow_key in flow_labels:
                runs_list = res_map.get(cfg_key, [])
                if not runs_list:
                    continue
                sla_t = runs_list[0].get("sla_ms", 200.0)
                p50s = [r["p50_ms"] for r in runs_list]
                p95s = [r["p95_ms"] for r in runs_list]
                passes = sum(1 for r in runs_list if r.get("p95_pass", r["p95_ms"] < sla_t))
                status_str = "🟢 **ĐẠT SLA**" if passes == len(runs_list) else f"🔴 **KHÔNG ĐẠT** ({passes}/{len(runs_list)})"
                rows_md.append(
                    f"| **{label}** | < {sla_t:.0f} ms | {np.mean(p50s):.1f} ms | **{np.mean(p95s):.1f} ms** | "
                    f"`{min(p95s):.1f} – {max(p95s):.1f}` | {status_str} |"
                )
            st.markdown("\n".join(rows_md))
