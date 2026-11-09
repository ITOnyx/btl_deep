import pandas as pd

styles = pd.read_csv("data/fashion-dataset/styles.csv", on_bad_lines='skip')
print(f"styles.csv shape: {styles.shape}")
print("\nstyles.csv at iloc[1855]:")
print(styles.iloc[1855][['id', 'productDisplayName', 'articleType', 'gender', 'baseColour', 'masterCategory']])

print("\nstyles.csv where id == 1855:")
print(styles[styles['id'] == 1855][['id', 'productDisplayName', 'articleType', 'gender', 'baseColour', 'masterCategory']])
