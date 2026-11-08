import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8")

data = np.load("data/checkpoints_siglip-vit/vit_zero_train_stats.npz")
print("Keys in vit_zero_train_stats.npz:")
for k in data.files:
    arr = data[k]
    print(f"  {k:20s}: shape={arr.shape}, dtype={arr.dtype}, sample={arr.flatten()[:5] if arr.ndim > 0 else arr}")
