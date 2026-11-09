import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

csv_path = Path("reports/phase1/phase1b/scope_proposal.csv")
if not csv_path.exists():
    csv_path = Path("reports/scope_proposal.csv")

df = pd.read_csv(csv_path)
print(f"Loaded {csv_path}, shape: {df.shape}")
print(df.head())

# Filter for the articleTypes in table 5 of phase2a_findings.md
table_types = [
    "Mobile Pouch", "Tablet Sleeve", "Wristbands", "Shoe Laces",
    "Shoe Accessories", "Umbrellas", "Key chain", "Free Gifts",
    "Fragrance Gift Set", "Accessory Gift Set"
]

print("\n--- Comparing table 5 types against scope_proposal.csv ---")
for t in table_types:
    m = df[df['articleType'] == t]
    if len(m) == 0:
        print(f"Type: '{t}' -> NOT FOUND in scope_proposal.csv!")
    else:
        print(f"Type: '{t}' -> CSV rows: {len(m)}")
        for idx, r in m.iterrows():
            print(f"   CSV: count={r.get('count', r.get('total_count', 'N/A'))}, proposal={r.get('proposal', r.get('scope', 'N/A'))}, reason={r.get('reason', 'N/A')}")
