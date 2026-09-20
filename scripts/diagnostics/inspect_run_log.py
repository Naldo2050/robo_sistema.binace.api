import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open("logs/run.log", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print(f"Total lines in run.log: {len(lines)}")

starts = []
for i, l in enumerate(lines):
    if "Iniciando bot para BTCUSDT" in l or "Enhanced Market Bot" in l:
        starts.append((i, l.strip()))

print(f"Total bot startups found: {len(starts)}")
for idx, s in starts[-10:]:
    print(f"Line {idx}: {s}")

print("\nFirst 3 lines:")
for l in lines[:3]:
    print(" ", l.strip())

print("\nLast 10 lines:")
for l in lines[-10:]:
    print(" ", l.strip())
