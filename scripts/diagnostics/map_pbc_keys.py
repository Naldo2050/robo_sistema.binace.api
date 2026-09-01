"""ETAPA 5B - mapear chaves do payload_builder_compact (helper)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

txt = open("market_orchestrator/ai/payload_builder_compact.py", encoding="utf-8").read()
lines = txt.splitlines()
out = []
for i, l in enumerate(lines):
    s = l.strip()
    if s.startswith("def ") or '"' in s or "'" in s:
        if any(k in s for k in ("price", "sr", "ctx", "def ", "_build", "def_bias")):
            out.append(f"{i+1}: {s[:110]}")
with open("C:/Users/Micro/AppData/Local/Temp/pbc_keys.txt", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out))
print("OK", len(out))
