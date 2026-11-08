import sys, os
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding="utf-8")
from src.query_builder import build_structured_query, preprocess_user_query

samples = [
    'ao thun nam mau den',
    'áo sơ mi nữ công sở màu trắng',
    'giay chay bo the thao nu mau tim',
    'dong ho nam mau bac'
]
for s in samples:
    print('Raw       :', s)
    print('Preprocess:', preprocess_user_query(s))
    print('Structured:', build_structured_query(s))
    print('-'*50)
