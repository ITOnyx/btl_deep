import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

df = pd.read_csv("reports/phase1/phase1b/scope_proposal.csv")
print("Columns:", df.columns.tolist())

table_types = [
    "Mobile Pouch", "Tablet Sleeve", "Wristbands", "Shoe Laces",
    "Shoe Accessories", "Umbrellas", "Key chain", "Free Gifts",
    "Fragrance Gift Set", "Accessory Gift Set"
]

sub = df[df['articleType'].isin(table_types)]
print(sub.to_string())

# Also aggregate by articleType to see the total counts across splits/genders
agg = sub.groupby(['articleType', 'in_scope']).agg({'count': 'sum'}).reset_index()
print("\n--- Aggregated by articleType and in_scope ---")
print(agg.to_string())
