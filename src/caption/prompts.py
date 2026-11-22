"""Versioned Prompts for Multimodal Dense Captioning.
SigLIP + Dense Captioning E-commerce Visual Search (Hoc_sau_ver4).
"""

PROMPT_VERSION = "v1.0"

# Prompt tiếng Việt cố định cho InternVL2-2B (Cấu hình A & B)
# Thông tin phân biệt (loại, giới, màu, kiểu dáng) bắt buộc đứng đầu câu.
INTERNVL2_PROMPT_TEMPLATE = (
    "<image>\n"
    "Mô tả ngắn gọn, súc tích sản phẩm thời trang trong hình bằng tiếng Việt (khoảng 25-45 từ). "
    "YÊU CẦU BẮT BUỘC:\n"
    "1. Thông tin phân biệt cốt lõi bao gồm: [loại sản phẩm], [giới tính người mặc], [màu sắc chủ đạo], [kiểu dáng đặc trưng] PHẢI đứng ngay đầu câu.\n"
    "2. Tiếp theo mô tả chi tiết hoa văn, chất liệu vải, cổ áo/tay áo (nếu có).\n"
    "3. Tuyệt đối không bịa đặt phụ kiện hoặc chi tiết không nhìn thấy trong ảnh.\n"
    "4. Không dùng từ ngữ quảng cáo, tâng bốc."
)

# Prompt bổ trợ tiếng Việt cho Vintern-1B-v3_5 (Cấu hình B & C)
VINTERN_PROMPT_TEMPLATE = (
    "<image>\n"
    "Hãy mô tả chi tiết sản phẩm thời trang trong ảnh bằng tiếng Việt tự nhiên: "
    "nêu rõ loại trang phục, màu sắc, đối tượng mặc (nam/nữ/trẻ em), kiểu dáng và đặc điểm nổi bật."
)
