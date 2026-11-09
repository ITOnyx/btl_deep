import json
import hashlib
from pathlib import Path

def get_sha256(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return "FILE_NOT_FOUND"
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

manifest_path = Path("reports/run_manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

# 1. Annotate old superseded/rejected blocks
if "phase3e_eval" in manifest:
    manifest["phase3e_eval"]["status"] = "SUPERSEDED_PARTIALLY_REJECTED"
    manifest["phase3e_eval"]["audit_note"] = (
        "Mục 6 cũ chứa 5 ca thất bại không truy được nguồn (text_x002 áo sơ mi, x007 giày gót be, "
        "x015 bomber, x021 hoodie, visual_1855); đã chính thức đánh dấu 'KHÔNG TRUY ĐƯỢC NGUỒN, KHÔNG ĐÁNG TIN' "
        "và phục hồi danh sách 20 ca thật từ eval/results_3e.json."
    )

if "phase3f_reconciliation" in manifest:
    manifest["phase3f_reconciliation"]["status"] = "SUPERSEDED_BY_ERRATA_SERIES"
    manifest["phase3f_reconciliation"]["audit_note"] = (
        "Báo cáo 3f reconciliation chứa các sai sót: Benetton ghi R=0, gán ghép SLA p95 sai văn bản AGENTS.md, "
        "và tính toán số câu heldout n=24/23 do thiếu alias; toàn bộ đã bị bác bỏ/đính chính ở phase3f_errata (Rev1..Rev5)."
    )

if "phase3g_errata" in manifest:
    manifest["phase3g_errata"]["status"] = "SUPERSEDED_BY_REV2_TO_REV5"
    manifest["phase3g_errata"]["audit_note"] = "Bản nháp Errata vòng 1; đã được kế thừa và hoàn thiện ở các bản sau."

if "phase3g_errata_rev2" in manifest:
    manifest["phase3g_errata_rev2"]["status"] = "SUPERSEDED_BY_REV3_TO_REV5"
    manifest["phase3g_errata_rev2"]["audit_note"] = "Bản nháp Errata vòng 2; đã được nâng cấp khi thêm kiểm toán Lee độc lập."

if "phase3g_errata_rev3" in manifest:
    manifest["phase3g_errata_rev3"]["status"] = "SUPERSEDED_BY_REV4_AND_REV5"
    manifest["phase3g_errata_rev3"]["audit_note"] = "Rev3 ghi nhầm dải p95 latency thành [171.2-192.4]; đã bị bác bỏ ở Rev4/Rev5."

if "phase3g_errata_rev4" in manifest:
    manifest["phase3g_errata_rev4"]["status"] = "SUPERSEDED_BY_REV5"
    manifest["phase3g_errata_rev4"]["audit_note"] = (
        "Rev4 giải thích sai đỉnh latency do 'Cold-Start JIT' (đã bị bác bỏ ở Rev5 do có 10 warmup queries trước run 1); "
        "đồng thời thực nghiệm padding chạy nhầm caption cũ; đã sửa hoàn chỉnh ở Rev5."
    )

# 2. Update phase4 block
if "phase4" in manifest:
    manifest["phase4"]["status"] = "COMPLETED_REVISED_FINAL"
    manifest["phase4"]["audit_reconciliations"] = {
        "task1_failure_cases_audited": "5 ca không truy được nguồn đã đánh dấu 'KHÔNG TRUY ĐƯỢC NGUỒN, KHÔNG ĐÁNG TIN' và thay bằng 20 ca thật từ results_3e.json.",
        "task2_image_label_bug_fixed": "Khắc phục bug gán nhãn 1855: catalog xác nhận 1855 là Inkfruit Mens Chain Reaction T-shirt. Đã kiểm 200 ID ngẫu nhiên khớp 100% catalog.parquet theo id.",
        "task3_serving_config_evaluated": "Chạy lại Q2, Q3, Q5, heldout trên cấu hình phục vụ (ParserV2 + brands_v2 + Lavie notice), ghi hậu nghiệm; kiểm định 'quan jean nam xanh' và 'do tap gym' không bắn thông báo.",
        "task4_telemetry_5runs_audited": "Hạ quá nhiệt B.3 thành 'giả thuyết chưa tái lập' (0/1000 truy vấn > 500 ms); không kết luận SLA từ 3 truy vấn ảnh.",
        "task5_val_truncation_tuned": "Đánh giá Condition A vs B vs C trên split val (7470 items, seed 42); xóa/đánh dấu 'giả thuyết chưa đo' cho lý do loại bỏ nhiễu.",
        "task6_table5_scope_reconciled": "Đối chiếu 10 dòng bảng mục 5 findings 2a với scope_proposal.csv: 9/10 dòng lệch, đã đánh dấu không đáng tin và sửa chuẩn.",
        "task7_ab_permutation_verified": "Khớp 100% RandomState(20261109) với ab_mapping.json; sửa chuỗi 30 phần tử trong errata.",
        "task8_brand_c0_new_gt": "C0 P@10 trên GT mới (x018=209) đạt 0.0714 so với C5 đạt 0.8000; ghi rõ GT Condition A cũ (n=24/23)."
    }

# 3. Recalculate all relevant file hashes
files_to_hash = {
    "reports/phase3/phase3f/phase3f_errata.md": get_sha256("reports/phase3/phase3f/phase3f_errata.md"),
    "data/catalog/catalog.parquet": get_sha256("data/catalog/catalog.parquet"),
    "config.yaml": get_sha256("config.yaml"),
    "reports/phase2/phase2a/phase2a_findings.md": get_sha256("reports/phase2/phase2a/phase2a_findings.md"),
    "reports/phase3/phase3e/phase3e_inference_only.md": get_sha256("reports/phase3/phase3e/phase3e_inference_only.md"),
    "reports/phase1/phase1b/scope_proposal.csv": get_sha256("reports/phase1/phase1b/scope_proposal.csv"),
    "src/run_phase4_smoke_test.py": get_sha256("src/run_phase4_smoke_test.py"),
    "reports/phase4/phase4b/phase4_smoke.json": get_sha256("reports/phase4/phase4b/phase4_smoke.json"),
    "reports/phase4_verifications.json": get_sha256("reports/phase4_verifications.json"),
    "app/main.py": get_sha256("app/main.py"),
    "app/engine.py": get_sha256("app/engine.py"),
    "src/search_pipeline.py": get_sha256("src/search_pipeline.py"),
    "src/brands_v2.json": get_sha256("src/brands_v2.json"),
    "src/query_parser_v2.py": get_sha256("src/query_parser_v2.py"),
    "eval/_sealed/ab_mapping.json": get_sha256("eval/_sealed/ab_mapping.json")
}

# Update in phase3g_rev5 and phase4
if "phase3g_rev5" in manifest:
    manifest["phase3g_rev5"]["errata_rev5_sha256"] = files_to_hash["reports/phase3/phase3f/phase3f_errata.md"]
    manifest["phase3g_rev5"]["files"]["reports/phase3/phase3f/phase3f_errata.md"] = files_to_hash["reports/phase3/phase3f/phase3f_errata.md"]

if "phase4" in manifest:
    for k, v in files_to_hash.items():
        manifest["phase4"]["files"][k] = v

manifest["root_hashes"] = {
    "config_yaml": files_to_hash["config.yaml"],
    "catalog_parquet": files_to_hash["data/catalog/catalog.parquet"],
    "errata_rev5": files_to_hash["reports/phase3/phase3f/phase3f_errata.md"],
    "findings_2a": files_to_hash["reports/phase2/phase2a/phase2a_findings.md"],
    "inference_3e": files_to_hash["reports/phase3/phase3e/phase3e_inference_only.md"],
    "verifications_phase4": files_to_hash["reports/phase4_verifications.json"]
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("SUCCESSFULLY UPDATED reports/run_manifest.json!")
print("Recalculated hashes:")
for k, v in manifest["root_hashes"].items():
    print(f"  {k}: {v}")
