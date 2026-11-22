"""
Package app: Giao diện người dùng Streamlit cho Hệ thống Tra cứu Sản phẩm Đa phương thức.
Tuân thủ AGENTS.md Phần A & E (Pha 6).
Hỗ trợ 4 luồng nghiệp vụ:
1. Text-to-Image (Text tiếng Việt có dấu/không dấu -> Top-10 sản phẩm)
2. Image-to-Image (Ảnh truy vấn -> Top-10 sản phẩm tương đồng)
3. CIR Catalog (Ảnh sản phẩm catalog + Modifier tiếng Việt -> Top-10 sản phẩm biến đổi)
4. CIR Upload (Ảnh người dùng tải lên + Modifier tiếng Việt -> Top-10 sản phẩm biến đổi)
"""
