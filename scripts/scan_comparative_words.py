"""
Audit and verification of all comparative words and subjective adjectives
in reports/phase5c/final_eval_report.md and reports/phase5c/final_summary_for_chat.md
(Prompt 14j Step 4)
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
report_path = PROJECT_ROOT / "reports/phase5c/final_eval_report.md"
chat_path = PROJECT_ROOT / "reports/phase5c/final_summary_for_chat.md"

report_text = report_path.read_text(encoding="utf-8")
chat_text = chat_path.read_text(encoding="utf-8")
report_lines = report_text.splitlines()

# 1. Prohibited Subjective Adjectives Audit
SUBJECTIVE_ADJECTIVES = ["khổng lồ", "vượt bậc", "vượt trội", "xuất sắc", "hoàn hảo"]

print("================================================================================")
print("1. PROHIBITED SUBJECTIVE ADJECTIVES AUDIT (Prompt 14j Step 2 & 4)")
print("================================================================================")

found_subjective_report = {}
found_subjective_chat = {}

for adj in SUBJECTIVE_ADJECTIVES:
    m_rep = re.findall(r'\b' + re.escape(adj) + r'\b', report_text, re.IGNORECASE)
    m_chat = re.findall(r'\b' + re.escape(adj) + r'\b', chat_text, re.IGNORECASE)
    found_subjective_report[adj] = len(m_rep)
    found_subjective_chat[adj] = len(m_chat)
    print(f"  - '{adj}': report = {len(m_rep)}, chat_summary = {len(m_chat)}")

total_subjective = sum(found_subjective_report.values()) + sum(found_subjective_chat.values())
print(f"Tổng số tính từ chủ quan phát hiện: {total_subjective}")
assert total_subjective == 0, f"Found subjective adjectives: report={found_subjective_report}, chat={found_subjective_chat}"
print("[PASS] Hoàn toàn không có tính từ chủ quan (0/0).")

# 2. Categorization of Comparative Statements
print("\n================================================================================")
print("2. COMPARATIVE STATEMENTS CATEGORIZATION & AUDIT (Prompt 14j Step 4)")
print("================================================================================")

KEYWORDS = ["vượt", "hơn", "kém", "tương đương", "khác biệt có ý nghĩa", "chưa kết luận được"]

cat_a_mech_ci = []      # (a) So sánh sinh cơ học từ CI
cat_b_pure_num = []     # (b) Mô tả thuần số (delta, SLA, skewness)
cat_c_hypo_labeled = [] # (c) Giải thích viết tay có gắn nhãn 'giả thuyết' / 'giải thích'
cat_c_ungrounded = []   # (c) Viết tay suy diễn không có căn cứ (phải = 0)

for idx, line in enumerate(report_lines, 1):
    matched = [kw for kw in KEYWORDS if re.search(r'\b' + re.escape(kw) + r'\b', line, re.IGNORECASE)]
    if not matched:
        continue

    # Classify into categories:
    # Cat A: Mechanical CI comparisons
    is_cat_a = any(token in line for token in [
        "hơn", "kém", "tương đương", "chưa kết luận được", "khác biệt có ý nghĩa"
    ]) and ("[" in line and "]" in line and "CI" in line)

    # Cat B: Purely numerical / benchmark / SLA thresholds
    is_cat_b = any(token in line for token in [
        "ms", "SLA", "p95", "p99", "skewness", "ngưỡng", "≥", "<", "mean-centering", "bác bỏ", "erratum"
    ]) and not is_cat_a

    # Cat C: Hypotheses or explanations
    is_cat_c_labeled = any(token in line.lower() for token in [
        "giả thuyết", "giải thích", "quan sát", "cơ chế", "do ", "nguyên nhân", "bằng chứng", "vòng tròn"
    ])

    if is_cat_a:
        cat_a_mech_ci.append((idx, matched, line))
    elif is_cat_b:
        cat_b_pure_num.append((idx, matched, line))
    elif is_cat_c_labeled:
        cat_c_hypo_labeled.append((idx, matched, line))
    else:
        # Check if grounded by any numbers/stats
        has_stats = any(char.isdigit() for char in line)
        if has_stats:
            cat_b_pure_num.append((idx, matched, line))
        else:
            cat_c_ungrounded.append((idx, matched, line))

print(f"  (a) So sánh sinh cơ học từ bootstrap CI:                      {len(cat_a_mech_ci):3d} dòng")
print(f"  (b) Mô tả thuần số / chuẩn đối chiếu / ngưỡng SLA:            {len(cat_b_pure_num):3d} dòng")
print(f"  (c) Giải thích viết tay có gắn nhãn (giả thuyết/giải thích):  {len(cat_c_hypo_labeled):3d} dòng")
print(f"  (c_ungrounded) Nhận định suy diễn KHÔNG có căn cứ:            {len(cat_c_ungrounded):3d} dòng")

if cat_c_ungrounded:
    print("\n[FAIL] Found ungrounded comparative statements:")
    for idx, matched, line in cat_c_ungrounded:
        print(f"  Line {idx}: {line.strip()[:100]}")
    raise AssertionError(f"Found {len(cat_c_ungrounded)} ungrounded comparative statements!")

print("[PASS] Số nhận định viết tay suy diễn không có căn cứ = 0.")

# 3. Check un-interpolated placeholders
print("\n================================================================================")
print("3. UN-INTERPOLATED PLACEHOLDER CHECK")
print("================================================================================")
# Match un-interpolated template placeholders like {name}, {var_123}, [ĐIỀN...], [TODO...]
# Note: do not match LaTeX math subscripts like _{10}
placeholders_report = re.findall(r'\[(?:ĐIỀN|TODO|TBD|CHỌN)[^\]]*\]|\{[a-zA-Z_][a-zA-Z0-9_]*\}', report_text)
placeholders_chat = re.findall(r'\[(?:ĐIỀN|TODO|TBD|CHỌN)[^\]]*\]|\{[a-zA-Z_][a-zA-Z0-9_]*\}', chat_text)
total_placeholders = len(placeholders_report) + len(placeholders_chat)
print(f"Placeholders in report: {len(placeholders_report)}")
print(f"Placeholders in chat:   {len(placeholders_chat)}")
assert total_placeholders == 0, f"Found placeholders: report={placeholders_report}, chat={placeholders_chat}"
print("[PASS] Hoàn toàn không có placeholder chưa điền (0/0).")
print("\n[ALL AUDIT CHECKS PASSED SUCCESSFULLY]")
