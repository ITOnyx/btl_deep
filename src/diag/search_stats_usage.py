import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

kws = ["cone_restore_gamma", "mu_shared", "val_query_bank", "gallery_hub_scores", "csls_beta", "rrf_w_dense"]

dirs_to_check = ["notebooks", "src", "docs"]

for d in dirs_to_check:
    for root, dirs, files in os.walk(d):
        for fname in files:
            if fname.endswith((".py", ".ipynb", ".md")):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    found = [k for k in kws if k in content]
                    if found:
                        print(f"{fpath}: {found}")
                except Exception as e:
                    pass
