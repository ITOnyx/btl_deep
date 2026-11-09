import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query_parser_v2 import QueryParserV2

def main():
    parser = QueryParserV2()
    test_queries = [
        "do tap gym",
        "do di bien",
        "ao thun do"
    ]
    print("=== Testing QueryParserV2 on unaccented inputs ===")
    for q in test_queries:
        res = parser.parse(q)
        print(f"\nQuery: '{q}'")
        print(f"  articleType: {res['articleType']}")
        print(f"  gender:      {res['gender']}")
        print(f"  baseColour:  {res['baseColour']}")
        print(f"  brand:       {res['brand']}")
        print(f"  usage:       {res['usage']}")
        print(f"  raw:         {res}")

if __name__ == "__main__":
    main()
