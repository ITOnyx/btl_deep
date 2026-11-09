import json

with open("reports/run_manifest.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for k in ['phase3f_reconciliation', 'phase3g_errata', 'phase3g_errata_rev2', 'phase3g_errata_rev3', 'phase3g_errata_rev4', 'phase3g_rev5', 'phase4']:
    block = d.get(k, {})
    print(f"Key: {k}")
    print(f"  status: {block.get('status')}")
    print(f"  note/title: {block.get('note', block.get('title'))}")
