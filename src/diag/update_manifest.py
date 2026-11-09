import json, hashlib
from pathlib import Path

manifest_path = Path("reports/run_manifest.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

# Compute fresh hashes
def get_hash(rel_path):
    p = Path(rel_path)
    return hashlib.sha256(p.read_bytes()).hexdigest()

audit_final_hash = get_hash("reports/phase4/phase4_audit_final.md")
latency_e2e_hash = get_hash("reports/phase4/phase4_latency_e2e.json")
verifications_hash = get_hash("reports/phase4_verifications.json")
errata_hash = get_hash("reports/phase3/phase3f/phase3f_errata.md")
inference_3e_hash = get_hash("reports/phase3/phase3e/phase3e_inference_only.md")
findings_2a_hash = get_hash("reports/phase2/phase2a/phase2a_findings.md")
catalog_hash = get_hash("data/catalog/catalog.parquet")
config_hash = get_hash("config.yaml")

# 1. Update deprecated blocks status
if "phase3c_eval_v2" in manifest:
    manifest["phase3c_eval_v2"]["status"] = "DEPRECATED_BASELINE"
    manifest["phase3c_eval_v2"]["audit_note"] = "Bị thay thế bởi phương pháp suy luận thuần Pha 3e/3f và kiểm toán Pha 4."

if "phase3d_addendum" in manifest:
    manifest["phase3d_addendum"]["status"] = "DEPRECATED_BASELINE"
    manifest["phase3d_addendum"]["audit_note"] = "Bị thay thế bởi phương pháp suy luận thuần Pha 3e/3f và kiểm toán Pha 4."

# 2. Update phase4 files
manifest["phase4"]["files"]["reports/phase3/phase3f/phase3f_errata.md"] = errata_hash
manifest["phase4"]["files"]["reports/phase2/phase2a/phase2a_findings.md"] = findings_2a_hash
manifest["phase4"]["files"]["reports/phase3/phase3e/phase3e_inference_only.md"] = inference_3e_hash
manifest["phase4"]["files"]["reports/phase4_verifications.json"] = verifications_hash
manifest["phase4"]["files"]["reports/phase4/phase4_latency_e2e.json"] = latency_e2e_hash
manifest["phase4"]["files"]["reports/phase4/phase4_audit_final.md"] = audit_final_hash

# 3. Add phase4_audit_final block
manifest["phase4_audit_final"] = {
    "timestamp": "2026-11-09T21:05:00",
    "status": "COMPLETED_AUDIT_FINAL",
    "description": "Final Phase 4 Audit and Data Reconciliation covering Tasks T1 through T8",
    "tasks_summary": {
        "t1_p10_serving_normalized": "XÁC NHẬN LỖI (Repo chuẩn: Q2=0.9984, Q3=0.8663; 0.9980 và 0.7936 là số ảo trong chat)",
        "t2_val_size_and_queries": "ĐÃ XỬ LÝ (Val size 7,470 sp sạch trên gallery 38,644 sp; 28 truy vấn val sinh động)",
        "t3_heldout_valid_n": "XÁC NHẬN LỖI & ĐÃ XỬ LÝ (G_test n=24 do x010, x017 rỗng; G_full n=25 do x018=209. Sửa bug colour field)",
        "t4_latency_e2e_benchmark": "ĐÃ XỬ LÝ (Tách phép đo vector ~2ms có cache khỏi SLA; đo e2e 1000 query/luồng. Text đạt SLA; CIR p99 224-275 ms, max 306 ms không đạt SLA tuyệt đối)",
        "t5_index_scope_and_sporting_goods": "CHƯA XÁC ĐỊNH (Xung đột 2 Wristbands Sporting Goods, chờ duyệt Phương án A hoặc B)",
        "t6_catalog_audit": "ĐÃ XỬ LÝ (200/200 ID khớp 100%; ID 1855 là Inkfruit Mens Chain Reaction T-shirt)",
        "t7_manifest_hashes_synced": "ĐÃ XỬ LÝ",
        "t8_evaluation_rules_and_limitations": "ĐÃ XỬ LÝ (AP@10 chuẩn hóa min(R,10); Rule 24 cho G_full; chuỗi A/B seed 20261109 khớp 100%; mẫu nhỏ G_test)"
    },
    "files": {
        "reports/phase4/phase4_audit_final.md": audit_final_hash,
        "reports/phase4/phase4_latency_e2e.json": latency_e2e_hash,
        "reports/phase4_verifications.json": verifications_hash,
        "reports/phase3/phase3f/phase3f_errata.md": errata_hash,
        "reports/phase3/phase3e/phase3e_inference_only.md": inference_3e_hash,
        "reports/phase2/phase2a/phase2a_findings.md": findings_2a_hash
    }
}

# 4. Update root_hashes
manifest["root_hashes"] = {
    "config_yaml": config_hash,
    "catalog_parquet": catalog_hash,
    "errata_rev5": errata_hash,
    "findings_2a": findings_2a_hash,
    "inference_3e": inference_3e_hash,
    "verifications_phase4": verifications_hash,
    "latency_e2e_phase4": latency_e2e_hash,
    "audit_final_phase4": audit_final_hash
}

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("Successfully updated reports/run_manifest.json with all fresh hashes!")
