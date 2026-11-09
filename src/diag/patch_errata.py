import re
from pathlib import Path

errata_path = Path("reports/phase3/phase3f/phase3f_errata.md")
content = errata_path.read_text(encoding="utf-8")

# 1. Update B.3
pat1 = r"(\s+-\s+\*\*Quy tắc nghiệm thu SLA:\*\*\s+\*\*Tuyệt đối không kết luận đạt/vượt SLA từ chỉ 3 truy vấn ảnh khói \(smoke test\)\*\*.*)"
m1 = re.search(pat1, content)
assert m1, "pat1 not found"

insert_b3 = """
     - **Phân định rõ hai phép đo độ trễ:**
       + Phép đo ~2 ms/truy vấn trong `reports/phase4_verifications.json:task4_telemetry` là **chỉ phần tìm kiếm vector / có cache embedding** (`torch.matmul` + topk trên embedding đã tiền tính toán, không gồm forward pass SigLIP ViT/Text Tower). Phép đo này **bị loại bỏ hoàn toàn khỏi mọi kết luận SLA hệ thống**.
       + Phép đo SLA chính thức của hệ thống là **đo lường toàn diện End-to-End** qua đúng entrypoint `app/engine.py:SearchService` (gồm tokenize văn bản, nạp và tiền xử lý ảnh PIL, forward pass SigLIP FP16, centering, FAISS/matmul, BM25 RRF, attribute scoring và filtering), được ghi nhận tại [`reports/phase4/phase4_latency_e2e.json`](file:///d:/Kì 7/Hoc_sau_ver4/reports/phase4/phase4_latency_e2e.json).
"""
content = content[:m1.start()] + insert_b3 + content[m1.start():]

# 2. Update D.2
pat2 = r"(\s+-\s+\*\*Kết quả đo trên tập Validation \(7\.470 sản phẩm, seed 42, 28 truy vấn val tại `reports/phase4_verifications\.json`\):\*\*)"
m2 = re.search(pat2, content)
assert m2, "pat2 not found"

rep2 = """   - **Kết quả đo trên tập Validation (7.470 sản phẩm sạch, seed 42, 28 truy vấn val tại `reports/phase4_verifications.json:task5_val_truncation`):**
     *(Lưu ý về số lượng val: Bản 3e ghi 7.471 sp do tính trên gallery thô 38.646 sp [37.356 non-test * 0.20 = 7.471]; sau khi loại 2 Key chain ở Phase 4, gallery sạch là 38.644 sp nên val sạch có chính xác 7.470 sp [37.354 * 0.20 = 7.470]. 28 truy vấn val được sinh động từ top bộ 3 thuộc tính >= 5 sp qua `src/diag/run_phase4_all_verifications.py:478-503`)*:"""
content = content[:m2.start()] + rep2 + content[m2.end():]

# 3. Update L.1 table
pat3 = r"(\| \*\*Heldout \(26 Spec Queries\)\*\* \| \$G_\{test\}\$ \| 0\.8816 \| \*\*0\.8863\*\* \| 0\.7680 \| Cải thiện nhẹ \(\+0\.0047, \$n=25\$\) \|\n\| \*\*Heldout \(26 Spec Queries\)\*\* \| \$G_\{full\}\$ \| 0\.9688 \| \*\*0\.9701\*\* \| 0\.9840 \| Cải thiện nhẹ \(\+0\.0013, \$n=25\$\) \|)"
m3 = re.search(pat3, content)
assert m3, "pat3 not found"

rep3 = """| **Heldout (26 Spec Queries)** | $G_{test}$ | **0.9393** | **0.9418** | **0.5958** | Đánh giá chuẩn với $n=24$ hợp lệ (loại trừ `x010` R=0 và `x017` R=0) |
| **Heldout (26 Spec Queries)** | $G_{full}$ | **0.9631** | **0.9646** | **0.9680** | Đánh giá chuẩn với $n=25$ hợp lệ (loại trừ `x017` R=0; `x018` có R=209) |"""
content = content[:m3.start()] + rep3 + content[m3.end():]

# 4. Update Section G table
pat4 = r"(\| 3 \| Thông báo thương hiệu Lavie \(x017\) \| `ĐÃ DUYỆT \(2026-11-09\)` \| Đã duyệt giữ nguyên thông báo nới lỏng loại SP/giới tính và cảnh báo hiển thị trên giao diện người dùng \(Tab 1\)\. \|)"
m4 = re.search(pat4, content)
assert m4, "pat4 not found"

rep4 = """| 3 | Thông báo thương hiệu Lavie (x017) | `ĐÃ DUYỆT (2026-11-09)` | Đã duyệt giữ nguyên thông báo nới lỏng loại SP/giới tính và cảnh báo hiển thị trên giao diện người dùng (Tab 1). |
| 4 | Xung đột Phạm vi Danh mục (T5: 38.646 vs 38.644 và Sporting Goods) | `CHƯA XÁC ĐỊNH (Chờ duyệt)` | `config.yaml` loại trừ `Sporting Goods`, nhưng catalog có 2 sản phẩm Wristbands (ID `42830`, `42828`) thuộc `Sporting Goods`. Do Phase 2a xếp Wristbands là in-scope nên hiện 2 sản phẩm này có `in_scope=True` trong `catalog.parquet` và nằm trong `index_v2`. Chờ Người dùng chọn: (Phương án A) Giữ Wristbands in-scope và đồng bộ `config.yaml`, hoặc (Phương án B) Loại bỏ 2 sản phẩm này và rebuild index ở pha sau. |"""
content = content[:m4.start()] + rep4 + content[m4.end():]

# 5. Add T8 Notes at end of Section L
pat5 = r"(\s+- `x018` \(Benetton, \$R=209\$\): C0 P@10 = `0\.0000`, AP@10 = `0\.0000`)"
m5 = re.search(pat5, content)
assert m5, "pat5 not found"

rep5 = """    - `x018` (Benetton, $R=209$): C0 P@10 = `0.0000`, AP@10 = `0.0000`

### L.4 Các Lưu ý Kỹ thuật Bắt buộc (T8)
1. **Phân kỳ Toán học giữa AP@10 và P@10 (T8.1):**
   - Chỉ số $\\text{AP@10}$ được chuẩn hóa theo $\\min(R, 10)$: $\\text{AP@10} = \\frac{1}{\\min(R, 10)} \\sum_{k=1}^{10} P@k \\cdot \\text{rel}_k$.
   - Khi $R < 10$ (rất phổ biến trên $G_{test}$ hẹp gồm 1.290 sp, ví dụ Q5 brand queries có trung bình $\\sim 2.5$ sp liên quan/câu), nếu hệ thống xếp đúng toàn bộ $R$ sp ở các vị trí đầu (rank 1..R), thì $\\text{AP@10} = 1.0$ nhưng $\\text{P@10} = R/10 \\le 0.30$. Đây là lý do toán học dẫn đến sự chênh lệch giữa AP@10 = 0.9491 và P@10 = 0.2543 ở Q5 $G_{test}$.
2. **Cảnh báo Quy tắc 24 trên $G_{full}$ (T8.2):**
   - Theo `AGENTS.md`, số liệu so với chỉ tiêu CHỈ được tính trên tập test thuần túy ($G_{test}$, 1.290 sp test).
   - Số liệu trên $G_{full}$ (chứa 37.354 sp train) **chỉ mang tính chất tham khảo và là ước lượng lạc quan (optimistic reference)**.
3. **Nhắc nhở Cỡ Mẫu Nhỏ trên $G_{test}$ (T8.4):**
   - Trên $G_{test}$, câu `x018` (Benetton) chỉ có $R = 7$ sản phẩm liên quan; nhiều câu thuộc tính hẹp chỉ có $R \\le 3$. Các kết quả trên tập test nhỏ này có phương sai lớn và chịu trần toán học của P@10 ($P@10 \\le R/10$)."""
content = content[:m5.start()] + rep5 + content[m5.end():]

errata_path.write_text(content, encoding="utf-8")
print("SUCCESSFULLY PATCHED phase3f_errata.md via regex")
