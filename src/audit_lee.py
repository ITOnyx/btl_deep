import re
import pandas as pd

df = pd.read_parquet('data/catalog/catalog.parquet')
df_in = df[df['in_scope'] == True]

name_lower = df_in['name'].str.lower()
has_lee = name_lower.str.contains('lee', na=False)
has_lee_cooper = name_lower.str.contains('lee cooper', na=False)
has_lee_only = has_lee & (~has_lee_cooper)

print(f"Total in_scope: {len(df_in)}")
print(f"Substring 'lee' (all): {has_lee.sum()}")
print(f"Substring 'lee cooper': {has_lee_cooper.sum()}")
print(f"Substring 'lee' excluding 'lee cooper': {has_lee_only.sum()}")

has_lee_word = df_in['name'].apply(lambda s: bool(re.search(r'\blee\b', str(s), re.IGNORECASE)))
has_lee_cooper_word = df_in['name'].apply(lambda s: bool(re.search(r'\blee cooper\b', str(s), re.IGNORECASE)))
has_lee_word_only = has_lee_word & (~has_lee_cooper_word)

print(f"Word boundary '\\blee\\b' (all): {has_lee_word.sum()}")
print(f"Word boundary '\\blee cooper\\b': {has_lee_cooper_word.sum()}")
print(f"Word boundary '\\blee\\b' only (excluding Lee Cooper): {has_lee_word_only.sum()}")

# Check where 477 vs 344 came from
print(f"Difference: 477 - 133 = {477 - 133}")
