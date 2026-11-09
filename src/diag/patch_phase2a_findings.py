import re
from pathlib import Path

p = Path("reports/phase2/phase2a/phase2a_findings.md")
text = p.read_text(encoding="utf-8")

old_sec = """### Danh sách các `articleType` mơ hồ / nhạy cảm đề xuất duyệt:
| articleType | Số lượng | Đề xuất | Lý do |
|---|---|---|---|
| Mobile Pouch | 33 | In-Scope | Túi đựng điện thoại đeo chéo mini (phụ kiện thời trang) |
| Tablet Sleeve | 31 | In-Scope | Túi chống sốc máy tính bảng thiết kế thời trang |
| Wristbands | 12 | In-Scope | Băng đeo tay thể thao thời trang |
| Shoe Laces | 12 | In-Scope | Dây giày phụ kiện thời trang |
| Shoe Accessories | 24 | In-Scope | Phụ kiện giày |
| Umbrellas | 6 | In-Scope | Ô dù cầm tay |
| Key chain | 35 | Out-of-Scope | Móc chìa khóa (quà tặng lưu niệm) |
| Free Gifts | 8 | Out-of-Scope | Quà tặng kèm không rõ danh mục |
| Fragrance Gift Set | 34 | Out-of-Scope | Bộ quà nước hoa (mỹ phẩm) |
| Accessory Gift Set | 358 | In-Scope | Set quà cà vạt/ví/thắt lưng thời trang nam |"""

new_sec = """> [!CAUTION]
> **CẢNH BÁO KIỂM TOÁN NGUỒN GỐC DỮ LIỆU (ERRATA AUDIT):**
> Bảng dưới đây trong báo cáo gốc Pha 2a được đánh dấu chính thức là **"KHÔNG ĐÁNG TIN (HARDCODED TEMPLATE STRING MISMATCH)"**.
> Bảng này xuất phát từ chuỗi Markdown tĩnh hardcode trong script khảo sát chứ **không phải được tổng hợp động từ `scope_proposal.csv`**. Khi đối soát 1:1 với dữ liệu gốc `reports/phase1/phase1b/scope_proposal.csv`, **có tới 9/10 dòng bị lệch số lượng** (chỉ có duy nhất dòng `Umbrellas = 6` là khớp).

### Bảng Đối chiếu Chi tiết Từng Dòng: Bảng gốc Pha 2a vs Dữ liệu Thực tế `scope_proposal.csv`:

| articleType | Số lượng ghi trong Pha 2a (Bản cũ) | Số lượng Thực tế trong `scope_proposal.csv` | Trạng thái Đối soát | Phân rã Thực tế theo MasterCategory / Giới tính |
|---|:---:|:---:|:---:|---|
| **Mobile Pouch** | 33 | **47** | **LỆCH (-14)** | 6 Unisex + 41 Women (Accessories) |
| **Tablet Sleeve** | 31 | **3** | **LỆCH (+28)** | 2 Unisex + 1 Women (Accessories) |
| **Wristbands** | 12 | **4** | **LỆCH (+8)** | 2 Accessories + 2 Sporting Goods (Unisex) |
| **Shoe Laces** | 12 | **1** | **LỆCH (+11)** | 1 Unisex (Accessories) |
| **Shoe Accessories** | 24 | **22** | **LỆCH (+2)** | 2 Men + 20 Unisex (Accessories) |
| **Umbrellas** | 6 | **6** | **KHỚP** | 6 Women (Accessories) |
| **Key chain** | 35 | **2** | **LỆCH (+33)** | 2 Unisex (Wildcraft 43230, American Tourister 42066) |
| **Free Gifts** | 8 | **86** | **LỆCH (-78)** | 43 Men + 8 Unisex + 35 Women (Bản cũ chỉ đếm nhóm Unisex = 8) |
| **Fragrance Gift Set** | 34 | **57** | **LỆCH (-23)** | 31 Men + 1 Unisex + 25 Women (Personal Care) |
| **Accessory Gift Set** | 358 | **92** | **LỆCH (+266)** | 92 Men (Accessories) |

### Bảng Đề xuất Chuẩn hóa đã Được Duyệt (Căn cứ trên `scope_proposal.csv`):
| articleType | Số lượng Thực tế | Đề xuất | Lý do Thực tế |
|---|:---:|---|---|
| Mobile Pouch | 47 | In-Scope | Túi đựng điện thoại đeo chéo mini (phụ kiện thời trang) |
| Tablet Sleeve | 3 | In-Scope | Túi chống sốc máy tính bảng thiết kế thời trang |
| Wristbands | 4 | In-Scope | Băng đeo tay thể thao thời trang |
| Shoe Laces | 1 | In-Scope | Dây giày phụ kiện thời trang |
| Shoe Accessories | 22 | In-Scope | Phụ kiện giày |
| Umbrellas | 6 | In-Scope | Ô dù cầm tay |
| Key chain | 2 | Out-of-Scope | Móc chìa khóa (Wildcraft 43230, American Tourister 42066) |
| Free Gifts | 86 | Out-of-Scope | Quà tặng kèm không rõ danh mục |
| Fragrance Gift Set | 57 | Out-of-Scope | Bộ quà nước hoa (mỹ phẩm) |
| Accessory Gift Set | 92 | In-Scope | Set quà cà vạt/ví/thắt lưng thời trang nam |"""

# Normalize newlines for replacement
text_norm = text.replace("\r\n", "\n")
old_sec_norm = old_sec.replace("\r\n", "\n")
new_sec_norm = new_sec.replace("\r\n", "\n")

if old_sec_norm in text_norm:
    text_norm = text_norm.replace(old_sec_norm, new_sec_norm)
    # Write back with CRLF
    p.write_text(text_norm.replace("\n", "\r\n"), encoding="utf-8")
    print("SUCCESSFULLY REPLACED TABLE 5 IN phase2a_findings.md!")
else:
    print("ERROR: old_sec_norm not found in text_norm!")
