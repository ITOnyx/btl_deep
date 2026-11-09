import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open("reports/phase4_verifications.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("TASK 3:")
print(json.dumps(d["task3_serving_re_eval"], ensure_ascii=False, indent=2))

print("\nTASK 8:")
print(json.dumps(d["task8_brand_c0"]["heldout_brand_c0_eval"]["comparison_with_c5"], ensure_ascii=False, indent=2))
for item in d["task8_brand_c0"]["heldout_brand_c0_eval"]["details"]:
    print(f"  [{item['id']}] {item['query']} (R={item['R_new_gt']}) -> C0 P@10: {item['c0_p10']}, AP@10: {item['c0_ap10']}")

print("\nTASK 5:")
print(json.dumps(d["task5_val_truncation"], ensure_ascii=False, indent=2))

print("\nTASK 4:")
print(json.dumps(d["task4_telemetry"], ensure_ascii=False, indent=2))
