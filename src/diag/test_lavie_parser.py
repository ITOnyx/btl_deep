import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.query_parser_v2 import QueryParserV2

parser = QueryParserV2(brands_path="src/brands_v2.json")

test_qs = [
    "túi xách Lavie nữ",
    "túi xách nữ Lavie",
    "quan jean nam xanh",
    "do tap gym",
    "áo thun Benetton nam",
    "dong ho Fossil nam"
]

for q in test_qs:
    res = parser.parse(q)
    print(f"\nQuery: '{q}'")
    print(f"  Parsed: articleType={res.get('articleType')}, gender={res.get('gender')}, baseColour={res.get('baseColour')}, brand={res.get('brand')}, unparsed={res.get('unparsed_tokens')}")
