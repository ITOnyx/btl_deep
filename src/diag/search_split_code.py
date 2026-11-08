import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

kws = ["train_test_split", "GroupKFold", "train_clean", "val_clean", "test_clean", "max_eval_per_source"]
dirs = [".", "notebooks", "src", "docs", "data/checkpoints_siglip-vit"]

for d in dirs:
    for fname in os.listdir(d):
        p = os.path.join(d, fname)
        if os.path.isfile(p) and p.endswith((".py", ".ipynb", ".md", ".json")):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    c = f.read()
                found = [k for k in kws if k in c]
                if found:
                    print(f"{p}: {found}")
            except Exception:
                pass
