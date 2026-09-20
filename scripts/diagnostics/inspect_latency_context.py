import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('logs/collect_2h_20260908_103451.log', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

target_times = ['11:05:11', '11:05:16', '11:49:04', '12:01:58']
for i, l in enumerate(lines):
    for t in target_times:
        if t in l and "LATÊNCIA CRÍTICA" in l:
            print(f"\n=== Contexto para {t} (linha {i+1}) ===")
            for c in lines[max(0, i-8):min(len(lines), i+9)]:
                print(" ", c.strip())
            break
