import json
import sys

def main():
    with open("eval/testset_v1.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    print("Keys in testset_v1.json:", list(d.keys()))
    print("q1 count:", len(d.get("q1_paired_queries", [])))
    print("q2 count:", len(d.get("q2_attribute_queries", [])))
    print("q4 count:", len(d.get("q4_image_queries", [])))
    print("Sample q2 item:", d.get("q2_attribute_queries", [])[0])
    print("Sample q4 item:", d.get("q4_image_queries", [])[0])

if __name__ == "__main__":
    main()
