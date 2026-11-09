import sys
import json
import pathlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    nb_path = pathlib.Path("notebooks/siglip_vietnamese_ecommerce_colab_ver2.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    for i, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type", "")
        src = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])
        out_text = ""
        for out in outputs:
            if "text" in out:
                out_text += "".join(out["text"])
            elif "data" in out and "text/plain" in out["data"]:
                out_text += "".join(out["data"]["text/plain"])

        # Check interesting keywords in code
        lower_src = src.lower()
        if any(w in lower_src for w in ["class ", "def encode", "def forward", "class ecommerce", "collate_fn", "eval_loop", "evaluate", "r@10", "recall@10"]):
            print(f"\n================ CELL {i} (Code) ================")
            print(src[:1500])

        if any(w in out_text.lower() for w in ["recall", "r@1", "r@5", "r@10", "val_loss", "epoch ", "evaluation"]):
            print(f"\n**************** CELL {i} (Output) ****************")
            out_lines = [l for l in out_text.split("\n") if any(k in l.lower() for k in ["recall", "r@", "loss", "epoch", "val", "test", "step"])]
            print("\n".join(out_lines[:15]))

    # Also check files in data/checkpoints_siglip-vit/
    ckpt_dir = pathlib.Path("data/checkpoints_siglip-vit")
    print(f"\nFiles in {ckpt_dir}:")
    for p in ckpt_dir.iterdir():
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
