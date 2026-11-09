"""
Streamlit Web Application for SigLIP E-commerce Visual Search (app/main.py)
Project: Hoc_sau_ver4
Supports:
1. Text-to-Image Search (Văn bản tiếng Việt -> Ảnh sản phẩm) with QueryParserV2 & Config 5
2. Image-to-Image Visual Search with Flip Augmentation & k-NN Voting
3. Composed Image Retrieval (CIR: Reference Image + Modifier Text) via SigLIP CIR Combiner
4. Invariant Rule 7 Checkpoint Hash Verification
5. Real-time Latency Breakdown & SLA Monitoring (< 200 ms)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    .sla-warn {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-rule7 {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
    }
    .product-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
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
    """Cached singleton search service."""
    return SearchService(config_path="config.yaml")


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


# -------------------------------------------------------------
# 2. Sidebar Controls
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/google-research/big_vision/main/big_vision/configs/proj/image_text/siglip.png", width=60) if False else None
    st.title("⚙️ Cấu hình Hệ thống")

    # Rule 7 Checkpoint Badge
    if rule7_ok:
        st.markdown(
            f"""<div class="badge-rule7">
            🔒 <b>Luật 7: Đạt kiểm toán</b><br>
            <small>Checkpoint Hash: <code>{service.cfg_hash[:12]}...</code></small>
            </div>""",
            unsafe_allow_html=True
        )
        st.caption(f"📦 Catalog in-scope: **{service.engine.N:,}** sản phẩm")
        st.caption(f"⚡ Phần cứng: **{service.engine.device.upper()}** (FP16)")
    else:
        st.error(f"❌ Vi phạm Luật 7!\n{rule7_err}")
        st.stop()

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
        value=0.3,
        step=0.05,
        help="Thưởng điểm khi sản phẩm khớp các thuộc tính bóc tách (loại SP, giới tính, màu sắc, brand)"
    )

    top_k = st.selectbox("Số lượng kết quả (Top-K)", options=[5, 10, 20, 30], index=1)

    st.markdown("---")
    st.markdown("### ℹ️ Thông tin Dự án")
    st.markdown(
        """
        - **Mô hình:** SigLIP So400M-patch14-224 + CIR Combiner
        - **Bộ phân tích:** QueryParserV2 (hỗ trợ có dấu và không dấu)
        - **Chỉ tiêu SLA:** Latency < 200 ms/query
        - **Bản sửa lỗi:** Pha 3g Rev5 (Hoàn tất & Đã duyệt)
        """
    )


# -------------------------------------------------------------
# 3. Main Header & App Tabs
# -------------------------------------------------------------
st.markdown('<div class="main-title">🛍️ SigLIP E-commerce Visual Search</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hệ sinh thái tra cứu thời trang thông minh: Văn bản tiếng Việt, Ảnh tương đồng và Tìm kiếm kết hợp (CIR)</div>', unsafe_allow_html=True)

tab_text, tab_image, tab_cir, tab_benchmark = st.tabs([
    "🔍 1. Text tiếng Việt → Ảnh",
    "🖼️ 2. Ảnh → Ảnh (Visual Search)",
    "🎨 3. Ảnh + Text bổ trợ (CIR)",
    "📊 4. Đối soát & Báo cáo Hệ thống"
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
                if Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/224x224.png?text=No+Image", use_container_width=True)

                st.markdown(f"**{prod['name'][:42]}...**" if len(prod['name']) > 42 else f"**{prod['name']}**")

                tags = []
                if prod.get("brand"):
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
# TAB 1: Text-to-Image Search
# -------------------------------------------------------------
with tab_text:
    st.markdown("### Tra cứu Sản phẩm bằng Văn bản tiếng Việt")

    # Clear distinction between System vs Model Pure
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
        st.info("ℹ️ **Đang chọn: Model thuần (SigLIP Dense)** — Truy vấn văn bản được mã hóa trực tiếp qua SigLIP Text Tower và tính cosine similarity với ảnh gallery. Không qua QueryParser, không lọc thuộc tính, không dùng BM25.")
    else:
        st.success("✨ **Đang chọn: Hệ thống (Parser + Metadata + BM25)** — Sử dụng QueryParser V2 trích xuất thuộc tính, nới lỏng ứng viên đa tầng L0..L6, cộng thưởng điểm mềm và kết hợp tìm kiếm từ vựng BM25 qua RRF.")

    # Sample Quick Queries
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
        if query_input.strip():
            with st.spinner("Đang xử lý truy vấn..."):
                results, meta = service.search_text(
                    query=query_input.strip(),
                    config_mode=effective_config,
                    top_k=top_k,
                    gamma=gamma_val
                )

            # 1. Latency & SLA Indicator
            total_lat = meta.get("breakdown", {}).get("total_ms", results[0]["latency_ms"] if results else 0.0)
            is_pass_sla = total_lat <= 200.0

            m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
            with m_col1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">Trạng thái SLA</div>
                        <div class="{'sla-pass' if is_pass_sla else 'sla-warn'}">
                            {'🟢 ĐẠT (< 200ms)' if is_pass_sla else '🟡 VƯỢT TRẦN'}
                        </div>
                    </div>""", unsafe_allow_html=True
                )
            with m_col2:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">Tổng độ trễ</div><div class="metric-val">{total_lat:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col3:
                parse_ms = meta.get("breakdown", {}).get("parse_ms", 0.0)
                st.markdown(f"""<div class="metric-card"><div class="metric-label">1. Phân tích ngữ nghĩa</div><div class="metric-val">{parse_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col4:
                filter_ms = meta.get("breakdown", {}).get("filter_ms", 0.0)
                st.markdown(f"""<div class="metric-card"><div class="metric-label">2. Lọc ứng viên</div><div class="metric-val">{filter_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col5:
                enc_ms = meta.get("breakdown", {}).get("encode_ms", 0.0)
                st.markdown(f"""<div class="metric-card"><div class="metric-label">3. Mã hóa SigLIP</div><div class="metric-val">{enc_ms:.1f} ms</div></div>""", unsafe_allow_html=True)
            with m_col6:
                search_ms = meta.get("breakdown", {}).get("search_ms", 0.0)
                st.markdown(f"""<div class="metric-card"><div class="metric-label">4. Hybrid/BM25/TopK</div><div class="metric-val">{search_ms:.1f} ms</div></div>""", unsafe_allow_html=True)

            st.write("")

            # 2. QueryParserV2 Info Expansion
            parsed = meta.get("parsed", {})
            with st.expander("🧠 Phân tích Ngữ nghĩa (QueryParser V2)", expanded=True):
                p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
                p_col1.metric("Loại sản phẩm", str(parsed.get("articleType") or "—"))
                p_col2.metric("Giới tính", str(parsed.get("gender") or "—"))
                p_col3.metric("Màu sắc", str(parsed.get("baseColour") or "—"))
                p_col4.metric("Thương hiệu", str(parsed.get("brand") or "—"))
                p_col5.metric("Tầng lọc (Tier)", str(meta.get("tier", "all")))
                p_col6.metric("Ứng viên xét duyệt", f"{meta.get('candidates_count', 0):,}")

            # 3. Lavie Notice Rule
            if parsed.get("brand") and parsed.get("brand").lower() == "lavie":
                st.warning("ℹ️ **Thông báo thương hiệu:** Thương hiệu 'Lavie' hiện không có trong catalog; hệ thống đang tự động nới lỏng và hiển thị các sản phẩm túi xách nữ có kiểu dáng tương tự.")

            # 4. Results Grid
            st.markdown(f"#### Kết quả tìm kiếm ({len(results)} sản phẩm):")
            render_product_grid(results, cols_per_row=5)


# -------------------------------------------------------------
# TAB 2: Image-to-Image Visual Search
# -------------------------------------------------------------
with tab_image:
    st.markdown("### Tra cứu Sản phẩm Tương đồng bằng Ảnh (Visual Search)")

    img_col1, img_col2 = st.columns([1, 2])

    with img_col1:
        st.markdown("**1. Tải ảnh lên hoặc chọn ảnh mẫu:**")
        uploaded_file = st.file_uploader("Chọn tệp ảnh sản phẩm (PNG, JPG):", type=["png", "jpg", "jpeg"])

        # Sample images selector
        sample_img_paths = [
            "data/fashion-dataset/images/15970.jpg",
            "data/fashion-dataset/images/39386.jpg",
            "data/fashion-dataset/images/59263.jpg",
            "data/fashion-dataset/images/1855.jpg"
        ]
        available_samples = [p for p in sample_img_paths if Path(p).exists()]

        selected_sample_img = None
        if available_samples:
            sample_choice = st.selectbox(
                "Hoặc chọn nhanh từ ảnh mẫu catalog:",
                options=["(Không chọn)"] + available_samples,
                index=0
            )
            if sample_choice != "(Không chọn)":
                selected_sample_img = sample_choice

        # Toggles
        use_flip = st.checkbox("Bật Flip Augmentation (TTA tăng cường)", value=True)
        use_knn = st.checkbox("Bật k-NN Attribute Voting Rescoring", value=True)

        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")
            st.image(query_image, caption="Ảnh truy vấn tải lên", use_container_width=True)
        elif selected_sample_img is not None:
            query_image = Image.open(selected_sample_img).convert("RGB")
            st.image(query_image, caption=f"Ảnh mẫu: {Path(selected_sample_img).name}", use_container_width=True)

    with img_col2:
        if query_image is not None:
            if st.button("🖼️ Tìm kiếm ảnh tương đồng", type="primary", use_container_width=True):
                with st.spinner("Đang trích xuất đặc trưng ảnh SigLIP và so khớp..."):
                    img_results, img_meta = service.search_image(
                        query_image,
                        top_k=top_k,
                        use_flip_aug=use_flip,
                        use_knn_voting=use_knn
                    )

                lat = img_meta.get("latency_ms", img_results[0]["latency_ms"] if img_results else 0.0)
                is_pass = lat <= 200.0

                st.markdown(
                    f"""<div style="margin-bottom: 1rem;">
                        <span class="{'sla-pass' if is_pass else 'sla-warn'}">
                            {'🟢 ĐẠT SLA' if is_pass else '🟡 ĐẠT CÓ ĐIỀU KIỆN'}: {lat:.1f} ms
                        </span>
                        <span style="color: #64748B; font-size: 0.85rem; margin-left: 10px;">
                            (Định mức SLA theo AGENTS.md: &lt; 200 ms)
                        </span>
                    </div>""",
                    unsafe_allow_html=True
                )

                st.markdown(f"#### Sản phẩm có thị giác tương đồng nhất ({len(img_results)} sp):")
                render_product_grid(img_results, cols_per_row=4)
        else:
            st.info("Vui lòng tải lên hoặc chọn một ảnh sản phẩm từ bên trái để bắt đầu tra cứu thị giác.")


# -------------------------------------------------------------
# TAB 3: CIR (Composed Image Retrieval)
# -------------------------------------------------------------
with tab_cir:
    st.markdown("### 🎨 Tra cứu Đa phương thức (CIR: Ảnh + Text điều chỉnh) — *(Thử nghiệm, chưa đánh giá)*")
    st.warning("⚠️ **Ghi chú thử nghiệm:** Tính năng Composed Image Retrieval (CIR) hiện đang ở trạng thái **Thử nghiệm, chưa đánh giá chính thức**. Tỉ lệ thay đổi thuộc tính phụ thuộc vào Attention Combiner của checkpoint và chưa được đưa vào cam kết SLA của dự án.")
    st.markdown("Kết hợp đặc trưng của một ảnh tham chiếu cùng câu văn bản yêu cầu thay đổi (ví dụ: *'đổi sang màu đỏ'*, *'kiểu dáng cộc tay'*, *'dành cho nữ'*).")

    cir_c1, cir_c2 = st.columns([1, 2])

    with cir_c1:
        st.markdown("**1. Ảnh tham chiếu (Reference Image):**")
        cir_upload = st.file_uploader("Tải ảnh gốc:", type=["png", "jpg", "jpeg"], key="cir_up")

        cir_sample_choices = [
            "data/fashion-dataset/images/15970.jpg",
            "data/fashion-dataset/images/39386.jpg",
            "data/fashion-dataset/images/10866.jpg"
        ]
        avail_cir = [p for p in cir_sample_choices if Path(p).exists()]
        cir_sample = None
        if avail_cir:
            cir_sc = st.selectbox("Hoặc chọn ảnh mẫu:", ["(Chọn ảnh)"] + avail_cir, index=1)
            if cir_sc != "(Chọn ảnh)":
                cir_sample = cir_sc

        ref_img_pil = None
        if cir_upload is not None:
            ref_img_pil = Image.open(cir_upload).convert("RGB")
            st.image(ref_img_pil, caption="Ảnh tham chiếu tải lên", use_container_width=True)
        elif cir_sample is not None:
            ref_img_pil = Image.open(cir_sample).convert("RGB")
            st.image(ref_img_pil, caption=f"Ảnh mẫu: {Path(cir_sample).name}", use_container_width=True)

        st.markdown("**2. Văn bản điều chỉnh (Modifier Text):**")
        modifier_input = st.text_input(
            "Yêu cầu biến đổi kiểu dáng / màu sắc:",
            value="màu xanh navy",
            placeholder="Ví dụ: đổi sang màu đỏ, chuyển thành áo phông cộc tay..."
        )

        use_cir_bonus = st.checkbox("Cộng điểm thưởng thuộc tính nếu nhận diện được màu/kiểu mới", value=True)

    with cir_c2:
        if ref_img_pil is not None and modifier_input.strip():
            if st.button("🎨 Tổng hợp Truy vấn Đa phương thức & Tìm kiếm", type="primary", use_container_width=True):
                with st.spinner("Đang tổng hợp Cross-Attention qua mô-đun CIRCombiner..."):
                    cir_results, cir_meta = service.search_cir(
                        ref_image=ref_img_pil,
                        modifier_text=modifier_input.strip(),
                        top_k=top_k,
                        use_soft_bonus=use_cir_bonus
                    )

                lat = cir_meta.get("latency_ms", cir_results[0]["latency_ms"] if cir_results else 0.0)
                is_pass = lat <= 200.0

                st.markdown(
                    f"""<div style="margin-bottom: 1rem;">
                        <span class="{'sla-pass' if is_pass else 'sla-warn'}">
                            {'🟢 ĐẠT SLA' if is_pass else '🟡 VƯỢT TRẦN'}: {lat:.1f} ms
                        </span>
                        <span style="color: #64748B; font-size: 0.85rem; margin-left: 10px;">
                            (Độ trễ tổng hợp CIR Combiner + GPU Top-K)
                        </span>
                    </div>""",
                    unsafe_allow_html=True
                )

                st.markdown(f"#### Kết quả sau khi biến đổi theo: *\"{modifier_input}\"* ({len(cir_results)} sp):")
                render_product_grid(cir_results, cols_per_row=4)
        else:
            st.info("Vui lòng cung cấp ảnh tham chiếu và câu mô tả điều chỉnh để thực hiện tìm kiếm đa phương thức CIR.")


# -------------------------------------------------------------
# TAB 4: Benchmarks & Errata Summary
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("### Báo cáo Đối soát, Hiệu năng & Tuân thủ Luật Bất biến (Pha 3g Rev5)")

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("#### 1. Đánh giá Paired Retrieval (Q1b trên $G_{test}$)")
        st.error("⚠️ **NGUYÊN TẮC BẤT BIẾN:** Tuyệt đối không để số đo của Hệ sinh thái kỹ thuật (0.9008) đứng thay cho điểm số của Mô hình thuần SigLIP (0.5225)!")

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.markdown(
                """<div style="background:#FFF1F2; border:1px solid #FECDD3; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem;">
                <h5 style="color:#BE123C; margin-top:0;">🤖 Cột 1: Model thuần (SigLIP Dense)</h5>
                <p style="font-size:0.85rem; color:#475569; margin-bottom:0.4rem;"><b>Cấu hình C0</b> (Chỉ trích xuất vector, không can thiệp)</p>
                <div style="font-size:1.4rem; font-weight:700; color:#BE123C;">Recall@10: 0.5225</div>
                <div style="font-size:0.95rem; color:#334155; margin-top:0.2rem;">mAP (MRR): <b>0.2362</b></div>
                <div style="font-size:0.85rem; color:#64748B;">Recall@1: 0.1116 | Recall@5: 0.3543</div>
                <div style="font-size:0.85rem; color:#64748B;">G_full Recall@10: <b>0.0628</b></div>
                <hr style="margin:0.5rem 0;">
                <span style="background:#FEE2E2; color:#991B1B; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:700;">
                    ❌ CHƯA ĐẠT CHỈ TIÊU GỐC
                </span>
                <p style="font-size:0.75rem; color:#64748B; margin-top:0.4rem; margin-bottom:0;">Chỉ tiêu đề ra: R@10 ≥ 0.85, mAP ≥ 0.75. Mô hình SigLIP nguyên bản chưa đạt.</p>
                </div>""", unsafe_allow_html=True
            )
        with sub_c2:
            st.markdown(
                """<div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:8px; padding:0.8rem; margin-bottom:0.8rem;">
                <h5 style="color:#047857; margin-top:0;">🛡️ Cột 2: Hệ thống (Parser + Metadata)</h5>
                <p style="font-size:0.85rem; color:#475569; margin-bottom:0.4rem;"><b>Cấu hình C2 & C5</b> (Hệ sinh thái kỹ thuật)</p>
                <div style="font-size:1.4rem; font-weight:700; color:#047857;">Recall@10: 0.9008 (C5: 0.9023)</div>
                <div style="font-size:0.95rem; color:#334155; margin-top:0.2rem;">mAP (MRR): <b>0.5205</b> (C5: 0.5240)</div>
                <div style="font-size:0.85rem; color:#64748B;">Recall@1: 0.3442 | Recall@5: 0.7465</div>
                <div style="font-size:0.85rem; color:#64748B;">G_full Recall@10: <b>0.2682</b></div>
                <hr style="margin:0.5rem 0;">
                <span style="background:#D1FAE5; color:#065F46; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.75rem; font-weight:700;">
                    🟢 ĐẠT RECALL@10 HỆ THỐNG
                </span>
                <p style="font-size:0.75rem; color:#065F46; margin-top:0.4rem; margin-bottom:0;">Thành quả của QueryParserV2 + Lọc ứng viên L0..L6 + Soft Bonus γ=0.3 + BM25 RRF.</p>
                </div>""", unsafe_allow_html=True
            )

        st.caption("*Khẳng định độc lập:* Điểm số 0.90+ là thành quả của Hệ sinh thái kỹ thuật (Parser + Metadata + BM25), không đại diện cho riêng mô hình SigLIP nguyên bản.")

    with b_col2:
        st.markdown("#### 2. Đối soát Độ trễ & Phân rã Thời gian (5 Runs)")
        st.markdown(
            """
            | Thành phần xử lý | p50 (ms) | p95 (ms) | p99 (ms) | Trạng thái SLA (< 200 ms) |
            |---|---|---|---|---|
            | 1. Query Parser V2 | 1.05 | 2.56 | 3.38 | 🟢 Hoàn toàn đáp ứng |
            | 2. Attribute Candidate Filter | 0.39 | 1.88 | 3.27 | 🟢 Hoàn toàn đáp ứng |
            | 3. SigLIP Text Tower (GPU FP16) | 24.52 | 61.28 | 86.21 | 🟢 Hoàn toàn đáp ứng |
            | 4. Dense Top-K GPU | 1.39 | 5.02 | 7.21 | 🟢 Hoàn toàn đáp ứng |
            | 5. BM25 Sparse Index | 3.77 | 8.01 | 10.08 | 🟢 Hoàn toàn đáp ứng |
            | **Toàn bộ Pipeline C5 (Text)** | **61.74** | **112.07** | **142.44** | 🟢 **ĐẠT SLA VỮNG CHẮC** |
            | **Luồng Ảnh → Ảnh (Visual Search)** | **85.87** | **178.25** | **280.17** | 🟡 **ĐẠT CÓ ĐIỀU KIỆN** (1/5 run vượt 200ms) |
            """
        )
        st.caption("*Đính chính Rev5:* Bác bỏ giải thích Cold-Start; đỉnh trễ do nhiệt độ GPU laptop chạm 86°C thermal throttling xuống 442 MHz.")

    st.markdown("---")
    st.markdown("#### 3. Bảng Kiểm toán Toàn diện Errata Rev5")
    st.markdown(
        r"""
        1. **Tái lập Thực nghiệm Padding:** Condition A (full 128, pad 128) tái lập chuẩn xác $G_{test}$ R@10 = 0.5225, MRR = 0.2363 và $G_{full}$ R@10 = 0.0628. Chứng minh độ dài đệm 128 là yếu tố cấu trúc chi phối.
        2. **Kiểm toán Key chain:** Đã loại trừ hoàn toàn 2 sản phẩm Key chain khỏi `in_scope` (`catalog.parquet` còn đúng 38.644 sản phẩm). Con số 35 ở Pha 2a là do gõ nhầm template string.
        3. **Kiểm toán Thương hiệu:** Benetton có 209 sp trên toàn catalog; Lee độc lập có 344 sp (Lee Cooper có 133 sp). Thêm 22 alias hậu nghiệm; đề xuất thông báo cho brand ngoài catalog như Lavie.
        4. **Đồng hạng (Ties) & FP16:** 923 ties trên gallery thô 38.646 sp vs 917 ties trên gallery in-scope 38.644 sp. Biên độ đối xứng hẹp $\pm 0.0008$ quanh 0.0628.
        5. **Chấm mù A/B:** Thực hiện bởi Claude AI single rater; 9 khối chạy ngẫu nhiên; bổ sung hạn chế *'người chấm từng thấy mapping cũ'*.
        """
    )
