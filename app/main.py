"""
Streamlit Web Application for SigLIP E-commerce Visual Search (app/main.py)
Project: Hoc_sau_ver4 (Phase 5)
Supports:
1. Text-to-Image Search (Văn bản tiếng Việt -> Ảnh sản phẩm) with QueryParserV2 & Config C5/C0/C2/C4
2. Image-to-Image Visual Search with configurable serving.i2i_mode ("fast" single-view vs "accurate" Batch-2 TTA)
3. Composed Image Retrieval (CIR) separated into cir_catalog (GPU Cache) and cir_upload (ViT Forward)
4. Invariant Rule 7 Checkpoint Hash Verification & Frozen Gender GT Policy (Hybrid / Strict / Relaxed)
5. Per-Flow Real-Time Latency Breakdown & Strict SLA Reporting (< 200 ms per flow)
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

# Streamlit Page Configuration
st.set_page_config(
    page_title="SigLIP E-commerce Visual Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished eCommerce styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .sla-pass {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .sla-fail {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-rule7 {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.35rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748B;
    }
    .rank-tag {
        font-weight: 700;
        color: #2563EB;
        font-size: 0.9rem;
    }
    .brand-tag {
        background: #F1F5F9;
        color: #334155;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
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
default_top_k_list = [5, 10, 20, 30]


# -------------------------------------------------------------
# 2. Sidebar Controls
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Cấu hình Hệ thống")

    # Rule 7 Checkpoint Badge
    st.markdown(
        f"""<div class="badge-rule7">
        🔒 <b>Luật 7: Đạt kiểm toán</b><br>
        <small>Checkpoint SHA-256: <code>{service.cfg_hash[:12]}...</code></small><br>
        <small>Index Manifest: <code>{service.man_hash[:12]}...</code></small>
        </div>""",
        unsafe_allow_html=True
    )
    st.caption(f"📦 Catalog in-scope: **{service.engine.N:,}** sản phẩm")
    st.caption(f"⚡ Phần cứng: **{service.engine.device.upper()}** (FP16)")
    st.caption(
        f"📏 GT Giới tính đóng băng: **{gender_gt_cfg.get('official', 'hybrid').upper()}** "
        f"(`{gender_gt_cfg.get('policy_file', 'eval/gender_gt_policy.json')}`)"
    )

    st.markdown("---")
    st.subheader("Cấu hình Tra cứu Văn bản")

    config_choice = st.selectbox(
        "Chế độ thuật toán (Text Search)",
        options=["c5", "c0", "c2", "c4"],
        format_func=lambda x: {
            "c5": "C5: Hệ thống Đầy đủ (Hybrid + Soft Bonus) [Chuẩn]",
            "c0": "C0: Model thuần (SigLIP Dense Only) [Đối chiếu]",
            "c2": "C2: Lọc nới lỏng + Soft Bonus + Dense",
            "c4": "C4: BM25 Sparse từ vựng thuần"
        }[x],
        index=0
    )

    gamma_val = st.slider(
        r"Trọng số cộng điểm mềm (Soft bonus $\gamma$)",
        min_value=0.0,
        max_value=1.0,
        value=default_gamma,
        step=0.05,
        help="Thưởng điểm khi sản phẩm khớp các thuộc tính bóc tách (loại SP, giới tính, màu sắc, brand) từ config.yaml"
    )

    top_k = st.selectbox("Số lượng kết quả (Top-K)", options=default_top_k_list, index=1)

    st.markdown("---")
    st.markdown("### ⏱️ Định mức SLA theo Luồng (`config.yaml`)")
    st.markdown(
        f"""
        - `text_to_image`: **< {service.get_flow_sla_ms('text_to_image'):.0f} ms** (p95)
        - `image_to_image`: **< {service.get_flow_sla_ms('image_to_image'):.0f} ms** (p95, mode `{default_i2i_mode}`)
        - `cir_catalog`: **< {service.get_flow_sla_ms('cir_catalog'):.0f} ms** (p95, GPU Cache)
        - `cir_upload`: **< {service.get_flow_sla_ms('cir_upload'):.0f} ms** (p95, ViT Forward)
        """
    )


# -------------------------------------------------------------
# 3. Main Header & App Tabs
# -------------------------------------------------------------
st.markdown('<div class="main-title">🛍️ SigLIP E-commerce Visual Search</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hệ sinh thái tra cứu thời trang đa phương thức: Văn bản tiếng Việt, Ảnh tương đồng và Tìm kiếm kết hợp (CIR)</div>', unsafe_allow_html=True)

tab_text, tab_image, tab_cir, tab_benchmark = st.tabs([
    "🔍 1. Text tiếng Việt → Ảnh",
    "🖼️ 2. Ảnh → Ảnh (Visual Search)",
    "🎨 3. Ảnh + Text bổ trợ (CIR)",
    "📊 4. Đối soát SLA, GT & Báo cáo Hệ thống"
])


# -------------------------------------------------------------
# Helper: Render Product Card
# -------------------------------------------------------------
def render_product_grid(results: List[Dict[str, Any]], cols_per_row: int = 5):
    if not results:
        st.info("Không tìm thấy sản phẩm phù hợp.")
        return

    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, item_idx in enumerate(range(i, min(i + cols_per_row, len(results)))):
            prod = results[item_idx]
            with cols[col_idx]:
                st.markdown(f"""
                <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.5rem; background: white; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span class="rank-tag">#{prod['rank']}</span>
                        <span style="font-size: 0.75rem; color: #64748B;">ID: {prod['item_id']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                img_path = prod.get("image_path", "")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    st.warning(f"Không tìm thấy ảnh: {img_path}")

                st.markdown(f"**{prod['name'][:42]}...**" if len(prod['name']) > 42 else f"**{prod['name']}**")

                tags = []
                if prod.get("brand") and str(prod.get("brand")) not in ("None", "nan", ""):
                    tags.append(f"<span class='brand-tag'>{prod['brand']}</span>")
                if prod.get("articleType"):
                    tags.append(f"<span style='font-size:0.75rem; color:#475569;'>{prod['articleType']}</span>")
                if prod.get("baseColour"):
                    tags.append(f"<span style='font-size:0.75rem; color:#475569;'>• {prod['baseColour']}</span>")
                if prod.get("gender"):
                    tags.append(f"<span style='font-size:0.75rem; color:#475569;'>• {prod['gender']}</span>")

                st.markdown(" ".join(tags), unsafe_allow_html=True)
                st.caption(f"Điểm: `{prod['score']:.4f}`")


# -------------------------------------------------------------
# TAB 1: Text-to-Image Search (flow: text_to_image)
# -------------------------------------------------------------
with tab_text:
    st.markdown("### Tra cứu Sản phẩm bằng Văn bản tiếng Việt (`flow: text_to_image`)")

    search_mode_ui = st.radio(
        "🎯 **Chế độ Tìm kiếm:**",
        options=["system", "model_pure"],
        format_func=lambda x: {
            "system": "🛡️ Hệ thống (Parser + Metadata + BM25) [Cấu hình C5 Hybrid]",
            "model_pure": "🤖 Model thuần (SigLIP Dense) [Cấu hình C0 Không can thiệp]"
        }[x],
        horizontal=True,
        index=0 if config_choice != "c0" else 1,
        help="Phân định minh bạch giữa Hệ thống hoàn chỉnh (Parser + Metadata + BM25 RRF + Soft Bonus) và Mô hình SigLIP thuần."
    )
    effective_config = "c0" if search_mode_ui == "model_pure" else ("c5" if config_choice == "c0" else config_choice)

    if effective_config == "c0":
        st.info("ℹ️ **Đang chọn: Model thuần (SigLIP Dense - C0)** — Truy vấn văn bản được mã hóa trực tiếp qua SigLIP Text Tower (padding='max_length'=128) và tính cosine similarity đã trừ tâm. Không qua QueryParser, không lọc thuộc tính, không dùng BM25.")
    else:
        st.success(f"✨ **Đang chọn: Hệ thống ({effective_config.upper()})** — Sử dụng QueryParser V2 trích xuất thuộc tính, nới lỏng ứng viên đa tầng L0..L6, cộng thưởng điểm mềm (γ={gamma_val:.2f}) và kết hợp BM25 qua RRF.")

    st.markdown("**Gợi ý truy vấn mẫu:**")
    sample_cols = st.columns(6)
    sample_queries = [
        "áo thun Puma nam",
        "váy dài màu đỏ dạ hội",
        "giày cao gót nữ màu đen",
        "ao do nam",
        "đồ đi biển",
        "túi xách nữ Lavie"
    ]

    selected_sample = None
    for idx, (c, sq) in enumerate(zip(sample_cols, sample_queries)):
        with c:
            if st.button(sq, key=f"sq_{idx}", use_container_width=True):
                selected_sample = sq

    query_input = st.text_input(
        "Nhập mô tả sản phẩm (hỗ trợ có dấu và không dấu):",
        value=selected_sample if selected_sample else "áo thun Puma nam",
        placeholder="Ví dụ: áo sơ mi nam dài tay màu xanh navy, đầm nữ trắng..."
    )

    if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True) or query_input:
        cleaned_query = query_input.strip()
        if cleaned_query:
            with st.spinner("Đang xử lý truy vấn..."):
                results, meta = service.search_text(
                    query=cleaned_query,
                    config_mode=effective_config,
                    top_k=top_k,
                    gamma=gamma_val
                )

            # 1. Latency & SLA Indicator (flow: text_to_image)
            sla_ms = float(meta.get("sla_ms", service.get_flow_sla_ms("text_to_image")))
            total_lat = float(meta.get("breakdown", {}).get("total_ms", results[0]["latency_ms"] if results else 0.0))
            is_pass_sla = total_lat <= sla_ms

            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            with m_col1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">SLA (text_to_image &lt; {sla_ms:.0f}ms)</div>
                        <div class="{'sla-pass' if is_pass_sla else 'sla-fail'}">
                            {'🟢 ĐẠT SLA' if is_pass_sla else '🔴 KHÔNG ĐẠT SLA'}
                        </div>
                    </div>""", unsafe_allow_html=True
                )
            with m_col2:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Tổng độ trễ E2E</div><div class="metric-val">{total_lat:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col3:
                parse_ms = float(meta.get("breakdown", {}).get("parse_ms", 0.0))
                st.markdown(f"""<div class="metric-card"><div class="metric-label">1. QueryParser V2</div><div class="metric-val">{parse_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col4:
                filter_ms = float(meta.get("breakdown", {}).get("filter_ms", 0.0))
                st.markdown(f"""<div class="metric-card"><div class="metric-label">2. Lọc ứng viên</div><div class="metric-val">{filter_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col5:
                enc_ms = float(meta.get("breakdown", {}).get("encode_ms", 0.0))
                st.markdown(f"""<div class="metric-card"><div class="metric-label">3. SigLIP Text Tower</div><div class="metric-val">{enc_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col6:
                search_ms = float(meta.get("breakdown", {}).get("search_ms", 0.0))
                st.markdown(f"""<div class="metric-card"><div class="metric-label">4. Hybrid/BM25/TopK</div><div class="metric-val">{search_ms:.1f} ms</div></div>""", unsafe_allow_html=True)

            st.write("")

            # 2. QueryParserV2 Info Expansion
            parsed = meta.get("parsed", {}) or {}
            with st.expander("🧠 Phân tích Ngữ nghĩa (QueryParser V2)", expanded=True):
                p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
                p_col1.metric("Loại sản phẩm", str(parsed.get("articleType") or "—"))
                p_col2.metric("Giới tính", str(parsed.get("gender") or "—"))
                p_col3.metric("Màu sắc", str(parsed.get("baseColour") or "—"))
                p_col4.metric("Thương hiệu", str(parsed.get("brand") or "—"))
                p_col5.metric("Tầng lọc (Tier)", str(meta.get("tier", "all")))
                p_col6.metric("Ứng viên xét duyệt", f"{meta.get('candidates_count', 0):,}")

            # 3. Lavie Out-of-Catalog Brand Notice Rule
            is_lavie = (
                (parsed.get("brand") and str(parsed.get("brand")).lower() == "lavie") or
                "lavie" in [str(t).lower() for t in parsed.get("unparsed_tokens", [])] or
                re.search(r"\blavie\b", cleaned_query, re.IGNORECASE) is not None
            )
            if is_lavie:
                st.warning("ℹ️ **Thông báo thương hiệu:** Thương hiệu 'Lavie' hiện không có trong catalog; hệ thống đang tự động nới lỏng và hiển thị các sản phẩm cùng danh mục có kiểu dáng tương tự.")

            # 4. Results Grid
            st.markdown(f"#### Kết quả tìm kiếm ({len(results)} sản phẩm):")
            render_product_grid(results, cols_per_row=5)


# -------------------------------------------------------------
# TAB 2: Image-to-Image Visual Search (flow: image_to_image)
# -------------------------------------------------------------
with tab_image:
    st.markdown("### Tra cứu Sản phẩm Tương đồng bằng Ảnh (`flow: image_to_image`)")

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

        st.markdown("**2. Chế độ suy luận Ảnh → Ảnh (`serving.i2i_mode`):**")
        i2i_mode_choice = st.radio(
            "Chế độ `i2i_mode`:",
            options=["fast", "accurate"],
            index=0 if default_i2i_mode == "fast" else 1,
            format_func=lambda m: (
                "⚡ fast (1 ảnh gốc - Mặc định config.yaml, p95 ~143 ms ĐẠT SLA < 200ms)"
                if m == "fast"
                else "🔬 accurate (Batch-2 TTA Flip - p95 ~211-225 ms KHÔNG ĐẠT SLA 200ms trên RTX 2050)"
            ),
            help="Pha 5a chứng minh chế độ 'fast' có P@10=0.6998 tương đương 'accurate' (P@10=0.6997) nhưng giảm độ trễ p95 từ ~211 ms xuống ~143 ms."
        )
        use_knn = st.checkbox("Bật k-NN Attribute Voting Rescoring", value=True)

        query_image = None
        exclude_self_id = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Ảnh truy vấn tải lên (Không loại trừ khỏi catalog)", use_container_width=True)
            st.info("ℹ️ **Ảnh tải lên từ thiết bị:** Không loại trừ sản phẩm nào khỏi gallery kết quả.")
        elif selected_sample_img is not None:
            query_image = Image.open(selected_sample_img).convert("RGB")
            exclude_self_id = Path(selected_sample_img).stem
            sample_row = service.engine.df[service.engine.df["item_id"].astype(str) == str(exclude_self_id)]
            sample_name = str(sample_row.iloc[0]["name"]) if len(sample_row) > 0 else f"ID {exclude_self_id}"
            st.image(query_image, caption=f"Ảnh mẫu catalog: {sample_name} (ID: {exclude_self_id})", use_container_width=True)
            st.warning(f"🛡️ **Ảnh chọn từ catalog ({sample_name} - ID: {exclude_self_id}):** Đã kích hoạt cơ chế tự động **loại bỏ chính sản phẩm này** khỏi danh sách kết quả.")

    with img_col2:
        if query_image is not None:
            if st.button("🖼️ Tìm kiếm ảnh tương đồng", type="primary", use_container_width=True):
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
                bd = img_meta.get("breakdown", {}) or {}

                ic1, ic2, ic3, ic4, ic5, ic6 = st.columns(6)
                with ic1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">SLA (i2i_{i2i_mode_choice} &lt; {sla_ms:.0f}ms)</div>
                            <div class="{'sla-pass' if is_pass else 'sla-fail'}">
                                {'🟢 ĐẠT SLA' if is_pass else '🔴 KHÔNG ĐẠT SLA'}
                            </div>
                        </div>""", unsafe_allow_html=True
                    )
                with ic2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">Tổng độ trễ E2E</div><div class="metric-val">{lat:.1f} ms</div></div>""", unsafe_allow_html=True)
                with ic3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">1. Giải mã ảnh</div><div class="metric-val">{bd.get('decode_ms', 0.0):.1f} ms</div></div>""", unsafe_allow_html=True)
                with ic4:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">2. Tiền xử lý</div><div class="metric-val">{bd.get('preprocess_ms', 0.0):.1f} ms</div></div>""", unsafe_allow_html=True)
                with ic5:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">3. ViT Forward</div><div class="metric-val">{bd.get('vit_forward_ms', 0.0):.1f} ms</div></div>""", unsafe_allow_html=True)
                with ic6:
                    srch_rr = float(bd.get("gallery_search_ms", 0.0)) + float(bd.get("rerank_ms", 0.0))
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">4. Matmul + Rerank</div><div class="metric-val">{srch_rr:.1f} ms</div></div>""", unsafe_allow_html=True)

                st.write("")
                st.markdown(f"#### Sản phẩm có thị giác tương đồng nhất ({len(img_results)} sp, chế độ `{i2i_mode_choice}`):")
                render_product_grid(img_results, cols_per_row=4)
        else:
            st.info("Vui lòng tải lên hoặc chọn một ảnh sản phẩm từ bên trái để bắt đầu tra cứu thị giác.")


# -------------------------------------------------------------
# TAB 3: CIR (Composed Image Retrieval: cir_catalog vs cir_upload)
# -------------------------------------------------------------
with tab_cir:
    st.markdown("### 🎨 Tra cứu Đa phương thức (CIR: Ảnh + Text điều chỉnh) — `cir_catalog` & `cir_upload`")
    st.warning("⚠️ **Ghi chú thực nghiệm:** Tính năng Composed Image Retrieval (CIR) đạt Hit@10 = 0.3500 trên bộ 20 truy vấn kiểm thử (`phase4_cir_20queries_eval.json`). Độ trễ SLA được tách riêng thành 2 luồng độc lập theo `AGENTS.md`: **`cir_catalog`** (dùng GPU Embedding Cache cho ảnh có sẵn trong catalog, p95 ~66.5 ms — ĐẠT SLA) và **`cir_upload`** (chạy qua ViT Image Tower cho ảnh tải mới từ thiết bị, p95 ~176.1–205.4 ms).")

    cir_c1, cir_c2 = st.columns([1, 2])

    with cir_c1:
        st.markdown("**1. Nguồn ảnh tham chiếu (Reference Image):**")
        cir_upload = st.file_uploader("Tải ảnh mới từ thiết bị (Luồng `cir_upload`):", type=["png", "jpg", "jpeg"], key="cir_up")

        cir_sample_ids = ["15970", "39386", "10866"]
        cir_sample_choices = [str(service.kaggle_images_dir / f"{sid}.jpg") for sid in cir_sample_ids]
        avail_cir = [p for p in cir_sample_choices if Path(p).exists()]
        cir_sample = None
        if avail_cir:
            cir_sc = st.selectbox(
                "Hoặc chọn ảnh có sẵn trong catalog (Luồng `cir_catalog`):",
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
            st.image(ref_img_pil, caption="Ảnh tải lên từ thiết bị -> Kích hoạt luồng `cir_upload` (ViT Forward)", use_container_width=True)
        elif cir_sample is not None:
            ref_img_pil = Image.open(cir_sample).convert("RGB")
            cir_ref_id = Path(cir_sample).stem
            st.image(ref_img_pil, caption=f"Ảnh mẫu catalog ID {cir_ref_id} ({Path(cir_sample).name})", use_container_width=True)
            force_upload_mode = st.checkbox(
                "Bỏ qua Catalog Cache (`force_upload=True` để đo luồng `cir_upload`)",
                value=False,
                help="Khi tắt (mặc định), ảnh thuộc catalog dùng thẳng vector trong GPU FP16 cache (`cir_catalog`). Khi bật, hệ thống giải mã và chạy lại qua ViT Image Tower (`cir_upload`)."
            )

        st.markdown("**2. Văn bản điều chỉnh (Modifier Text):**")
        modifier_input = st.text_input(
            "Yêu cầu biến đổi kiểu dáng / màu sắc:",
            value="màu xanh navy",
            placeholder="Ví dụ: màu đỏ, quần short, áo khoác jacket..."
        )

        use_cir_bonus = st.checkbox("Cộng điểm thưởng thuộc tính mềm (γ) cho từ khóa điều chỉnh", value=True)

    with cir_c2:
        if ref_img_pil is not None and modifier_input.strip():
            if st.button("🎨 Tổng hợp Truy vấn Đa phương thức & Tìm kiếm", type="primary", use_container_width=True):
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
                bd = cir_meta.get("breakdown", {}) or {}

                cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
                with cc1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-label">SLA ({flow_executed} &lt; {sla_ms:.0f}ms)</div>
                            <div class="{'sla-pass' if is_pass else 'sla-fail'}">
                                {'🟢 ĐẠT SLA' if is_pass else '🔴 KHÔNG ĐẠT SLA'}
                            </div>
                        </div>""", unsafe_allow_html=True
                    )
                with cc2:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">Tổng độ trễ E2E</div><div class="metric-val">{lat:.1f} ms</div></div>""", unsafe_allow_html=True)
                with cc3:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">Luồng thực thi</div><div class="metric-val" style="font-size:1.0rem;">{flow_executed}</div></div>""", unsafe_allow_html=True)
                with cc4:
                    dec_pre = float(bd.get("decode_ms", 0.0)) + float(bd.get("preprocess_ms", 0.0))
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">1. Decode + Preproc</div><div class="metric-val">{dec_pre:.1f} ms</div></div>""", unsafe_allow_html=True)
                with cc5:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">2. Forward + Combiner</div><div class="metric-val">{bd.get('vit_forward_ms', 0.0):.1f} ms</div></div>""", unsafe_allow_html=True)
                with cc6:
                    st.markdown(f"""<div class="metric-card"><div class="metric-label">3. Gallery Search</div><div class="metric-val">{bd.get('gallery_search_ms', 0.0):.1f} ms</div></div>""", unsafe_allow_html=True)

                st.write("")
                st.markdown(f"#### Kết quả sau khi biến đổi theo: *\"{modifier_input}\"* ({len(cir_results)} sp — luồng `{flow_executed}`):")
                render_product_grid(cir_results, cols_per_row=4)
        else:
            st.info("Vui lòng cung cấp ảnh tham chiếu và câu mô tả điều chỉnh để thực hiện tìm kiếm đa phương thức CIR.")


# -------------------------------------------------------------
# TAB 4: Benchmarks, 4-Flow SLA & GT Policy Report
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("### 📊 Báo cáo Đối soát Chất lượng, SLA 4 Luồng & Chính sách GT Giới tính")

    lat_v3_data = load_json_safe("reports/phase4/phase4_latency_e2e_v3.json")
    lat_v2_data = load_json_safe("reports/phase4/phase4_latency_e2e_v2.json")
    i2i_qual_data = load_json_safe("reports/phase4/phase5a_i2i_quality.json")
    unseen_v2_data = load_json_safe("reports/phase4/unseen_testset_v2_eval.json")
    gender_policy_data = load_json_safe(gender_gt_cfg.get("policy_file", "eval/gender_gt_policy.json"))

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("#### 1. Paired Retrieval (Q1b trên $G_{test}$, $n = 1,290$ ảnh test)")
        st.error("⚠️ **QUY TẮC ĐÁNH GIÁ (AGENTS.md):** Chỉ tiêu dự án (Recall@10 ≥ 0.85, mAP ≥ 0.75) CHỈ đối chiếu trên tập test ($G_{test} = 1,290$ ảnh). Tuyệt đối tách bạch kết quả Model thuần (C0) và Hệ thống (C2).")

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.markdown(
                """<div style="background:#FFF1F2; border:1px solid #FECDD3; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem;">
                <h5 style="color:#BE123C; margin-top:0;">🤖 Cột 1: Model thuần (SigLIP Dense C0)</h5>
                <p style="font-size:0.85rem; color:#475569; margin-bottom:0.4rem;">Chỉ trích xuất vector SigLIP + trừ tâm, không lọc thuộc tính</p>
                <div style="font-size:1.35rem; font-weight:700; color:#BE123C;">Recall@10: 0.5225</div>
                <div style="font-size:0.95rem; color:#334155; margin-top:0.2rem;">mAP (MRR): <b>0.2362</b></div>
                <div style="font-size:0.85rem; color:#64748B;">Recall@1: 0.1116 | Recall@5: 0.3543</div>
                <div style="font-size:0.85rem; color:#64748B;">Tham khảo lạc quan $G_{full}$ R@10: <b>0.0628</b></div>
                <hr style="margin:0.5rem 0;">
                <span style="background:#FEE2E2; color:#991B1B; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:700;">
                    ❌ KHÔNG ĐẠT CHỈ TIÊU (R@10 &lt; 0.85, mAP &lt; 0.75)
                </span>
                </div>""", unsafe_allow_html=True
            )
        with sub_c2:
            st.markdown(
                """<div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem;">
                <h5 style="color:#047857; margin-top:0;">🛡️ Cột 2: Hệ thống (Parser + Soft Bonus C2)</h5>
                <p style="font-size:0.85rem; color:#475569; margin-bottom:0.4rem;">QueryParser V2 + Soft Bonus &gamma;=0.3 (không dùng BM25)</p>
                <div style="margin-bottom:0.4rem;">
                    <div style="font-size:1.3rem; font-weight:700; color:#047857;">Recall@10: 0.9008</div>
                    <span style="background:#D1FAE5; color:#065F46; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:700;">
                        🟢 ĐẠT CHỈ TIÊU R@10 (&ge; 0.85)
                    </span>
                </div>
                <div style="margin-bottom:0.4rem;">
                    <div style="font-size:1.05rem; font-weight:700; color:#B45309;">mAP (MRR): 0.5205</div>
                    <span style="background:#FEE2E2; color:#991B1B; padding:0.15rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:700;">
                        ❌ KHÔNG ĐẠT CHỈ TIÊU mAP (&lt; 0.75)
                    </span>
                </div>
                <div style="font-size:0.85rem; color:#64748B;">Recall@1: 0.3442 | Recall@5: 0.7465 | $G_{full}$ R@10: <b>0.2682</b></div>
                </div>""", unsafe_allow_html=True
            )

        st.markdown("#### 2. Truy vấn Tự nhiên theo Thuộc tính & Chính sách GT Giới tính Đóng băng")
        st.markdown(
            """
            - **Chính sách GT Giới tính (`eval/gender_gt_policy.json`, SHA-256 `9c0d2cb5...`):** Đóng băng trước khi chấm điểm trên 50 câu truy vấn viết tay Q3 (Cấu hình phục vụ C5):
              - **Hybrid (Chính thức — Apparel dùng Strict; Accessories/Footwear cho phép Unisex):** **P@10 = 0.9180**
              - **Strict (Chặt chẽ — Không tính Unisex cho truy vấn Nam/Nữ):** **P@10 = 0.9010**
              - **Relaxed (Nới lỏng — Cho phép Unisex toàn bộ danh mục):** **P@10 = 0.9180**
            - **Các bộ truy vấn thuộc tính khác ($G_{full}$ tham khảo phục vụ C5):**
              - **Q2 (61 câu thuộc tính tổ hợp):** P@10 = **0.9984**, AP@10 = **0.9982** ($G_{test}$ P@10 = **0.7230**)
              - **Heldout Spec ($n=25$ hợp lệ trên $G_{full}$, $n=24$ trên $G_{test}$):** $G_{full}$ AP@10 = **0.9646** (P@10 = **0.9680**); $G_{test}$ AP@10 = **0.9418** (P@10 = **0.5958**)
              - **Unseen Testset v2 (`eval/unseen_testset_v2.json`, $n=32$ câu mới đóng băng SHA-256 `90e09202...`, đo 1 lần duy nhất):** Mean AP@10 = **0.7864**, Mean P@10 = **0.8065**
            """
        )

    with b_col2:
        st.markdown("#### 3. Đối soát Độ trễ & SLA Riêng Từng Luồng (`phase4_latency_e2e_v3.json`)")
        st.caption("Giao thức: 5 runs độc lập × 200 truy vấn/run, 20 truy vấn warm-up/cấu hình, `torch.cuda.synchronize()` quanh mỗi truy vấn, thứ tự Latin-square, nghỉ 5s giữa các run, giám sát `nvidia-smi` mỗi 1 giây trên GPU NVIDIA GeForce RTX 2050 (4096 MiB, FP16).")

        if lat_v3_data and "5a_baseline" in lat_v3_data.get("phases", {}):
            phase_5a = lat_v3_data["phases"]["5a_baseline"]
            res_map = phase_5a.get("results", {})
            flow_labels = [
                ("text_to_image", "1. Text → Ảnh (C5 Hybrid)", "text_to_image"),
                ("image_to_image_fast", "2a. Ảnh → Ảnh (mode=fast, Mặc định)", "image_to_image"),
                ("image_to_image_accurate", "2b. Ảnh → Ảnh (mode=accurate, Batch-2 TTA)", "image_to_image"),
                ("cir_catalog", "3a. CIR Catalog (GPU Embedding Cache)", "cir_catalog"),
                ("cir_upload", "3b. CIR Upload (ViT Forward Pass)", "cir_upload"),
            ]
            rows_md = [
                "| Luồng / Cấu hình | SLA (ms) | Mean p50 (ms) | Mean p95 (ms) | Dải p95 (5 runs) | Mean p99 (ms) | Run đạt SLA | Kết luận SLA |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            ]
            for cfg_key, label, flow_key in flow_labels:
                runs_list = res_map.get(cfg_key, [])
                if not runs_list:
                    continue
                sla_t = runs_list[0].get("sla_ms", 200.0)
                p50s = [r["p50_ms"] for r in runs_list]
                p95s = [r["p95_ms"] for r in runs_list]
                p99s = [r["p99_ms"] for r in runs_list]
                passes = sum(1 for r in runs_list if r.get("p95_pass", r["p95_ms"] < sla_t))
                status_str = "🟢 **ĐẠT SLA**" if passes == len(runs_list) else f"🔴 **KHÔNG ĐẠT** ({passes}/{len(runs_list)} run đạt)"
                rows_md.append(
                    f"| **{label}** | < {sla_t:.0f} | {np.mean(p50s):.2f} | **{np.mean(p95s):.2f}** | "
                    f"`{min(p95s):.2f} – {max(p95s):.2f}` | {np.mean(p99s):.2f} | {passes}/{len(runs_list)} | {status_str} |"
                )
            st.markdown("\n".join(rows_md))
            st.caption("*Lưu ý đối chiếu với Round 2 (`phase4_latency_e2e_v2.json`):* Ở lần đo Round 2 trước đó, `cir_upload` có p95 trung bình = **205.44 ms** (2/5 run < 200 ms, 3/5 run 204–234 ms -> KHÔNG ĐẠT 5/5 run) và `image_to_image` (Batch-2 TTA) có p95 trung bình = **225.25 ms** (0/5 run đạt). Ở lần đo `5a_baseline` (`v3`), `image_to_image_fast` (1 ảnh gốc) đạt p95 = **130.67 – 155.49 ms** (ĐẠT 5/5 run), trong khi `image_to_image_accurate` (Batch-2 TTA) có p95 = **195.52 – 222.57 ms** (chỉ 1/5 run < 200 ms -> KHÔNG ĐẠT SLA).")

            with st.expander("🔎 Chi tiết từng Run (Run 1..5), Phân rã Giai đoạn & Nhiệt độ GPU (`phase4_latency_e2e_v3.json`)"):
                sel_cfg = st.selectbox(
                    "Chọn luồng để xem chi tiết 5 runs:",
                    options=[k for k, _, _ in flow_labels],
                    format_func=lambda k: dict((a, b) for a, b, _ in flow_labels)[k]
                )
                sel_runs = res_map.get(sel_cfg, [])
                if sel_runs:
                    detail_md = [
                        "| Run | n | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | # > 200ms | GPU Temp Mean/Max (°C) | GPU Clock Mean (MHz) | p95 < 200ms |",
                        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
                    ]
                    for r in sel_runs:
                        g = r.get("gpu", {}) or {}
                        pass_icon = "🟢 Đạt" if r.get("p95_pass") else "🔴 Không đạt"
                        detail_md.append(
                            f"| Run {r['run']} | {r['n']} | {r['p50_ms']:.2f} | **{r['p95_ms']:.2f}** | {r['p99_ms']:.2f} | "
                            f"{r['max_ms']:.2f} | {r['n_over_sla']} | {g.get('temp_mean', '—')} / {g.get('temp_max', '—')} | "
                            f"{g.get('clock_mean', '—')} | {pass_icon} |"
                        )
                    st.markdown("\n".join(detail_md))

        st.markdown("#### 4. Thực nghiệm Chất lượng Ảnh → Ảnh: `fast` vs `accurate` (`phase5a_i2i_quality.json`)")
        if i2i_qual_data:
            f_q = i2i_qual_data["fast"]
            a_q = i2i_qual_data["accurate"]
            ov_q = i2i_qual_data["C_fast_vs_accurate"]
            st.markdown(
                f"""
                Đánh giá trên toàn bộ **$n = {i2i_qual_data['n_queries']:,}$ ảnh test** (`split == "test"`, gallery $G_{{test}} = 1,290$ sp, CPU float32):
                | Tiêu chí đánh giá | `i2i_mode = fast` (1 ảnh) | `i2i_mode = accurate` (Batch-2 TTA) | Chênh lệch (`accurate - fast`) |
                | :--- | :---: | :---: | :---: |
                | **A. Paired Self-Match Recall@1** | **{f_q['A_paired_self_match']['recall@1']:.4f}** | {a_q['A_paired_self_match']['recall@1']:.4f} | `{a_q['A_paired_self_match']['recall@1'] - f_q['A_paired_self_match']['recall@1']:+.4f}` |
                | **A. Paired Self-Match Recall@10** | **{f_q['A_paired_self_match']['recall@10']:.4f}** | **{a_q['A_paired_self_match']['recall@10']:.4f}** | `0.0000` |
                | **B. Attribute NN (loại trừ chính nó) P@10** | **{f_q['B_attribute_nn_excl_self']['P@10']:.4f}** | {a_q['B_attribute_nn_excl_self']['P@10']:.4f} | `{a_q['B_attribute_nn_excl_self']['P@10'] - f_q['B_attribute_nn_excl_self']['P@10']:+.4f}` |
                | **B. Attribute NN (loại trừ chính nó) Hit@10** | {f_q['B_attribute_nn_excl_self']['Hit@10']:.4f} | **{a_q['B_attribute_nn_excl_self']['Hit@10']:.4f}** | `{a_q['B_attribute_nn_excl_self']['Hit@10'] - f_q['B_attribute_nn_excl_self']['Hit@10']:+.4f}` |
                | **B. Attribute NN (loại trừ chính nó) AP@10** | **{f_q['B_attribute_nn_excl_self']['AP@10']:.4f}** | {a_q['B_attribute_nn_excl_self']['AP@10']:.4f} | `{a_q['B_attribute_nn_excl_self']['AP@10'] - f_q['B_attribute_nn_excl_self']['AP@10']:+.4f}` |
                | **C. Độ trùng lặp Top-10 & Cosine Query** | Top-10 Overlap: **{ov_q['top10_overlap_mean']*100:.2f}%** | Top-1 Agreement: **{ov_q['top1_agreement']*100:.2f}%** | Cosine mean: `{ov_q['query_vec_cosine_mean']:.4f}` |
                """
            )
        st.caption("*Kết luận kỹ thuật:* Chế độ `fast` giữ nguyên chất lượng P@10 (`0.6998` vs `0.6997`) và AP@10 (`0.6830` vs `0.6822`) trên tập test $G_{test}$ trong khi giúp luồng Ảnh → Ảnh đạt chuẩn SLA p95 < 200 ms cả 5/5 run.")
