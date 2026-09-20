import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('logs/collect_2h_20260908_103451.log', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if '12:12:01' in l:
        print(f"Linha {i+1}:")
        for c in lines[max(0, i-10):min(len(lines), i+25)]:
            print(" ", c.strip())
        break
