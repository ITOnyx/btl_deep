import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Searching for colour / baseColour accesses across repository:")
    patterns = ['baseColour', 'colour']
    for py_file in sorted(Path('.').rglob('*.py')):
        if any(p in py_file.parts for p in ['.git', '.system_generated', '__pycache__', 'env', 'venv']):
            continue
        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception:
            continue
        for i, line in enumerate(content.splitlines(), start=1):
            if any(p in line for p in patterns):
                if any(k in line for k in ['get(', 'spec', '["colour"]', "['colour']", '["baseColour"]', "['baseColour']"]):
                    print(f"{py_file}:{i}: {line.strip()[:110]}")

if __name__ == '__main__':
    main()
