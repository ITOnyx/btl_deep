import sys
import json
sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.vi_restore import restore_diacritics

def main():
    with open("eval/handwritten_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    unaccented = [q for q in queries if not q["has_diacritics"]]
    print(f"Testing restore on {len(unaccented)} unaccented handwritten queries:")
    for q in unaccented:
        orig = q["query"]
        restored = restore_diacritics(orig)
        print(f"  {orig:<32} -> {restored}")

if __name__ == "__main__":
    main()
