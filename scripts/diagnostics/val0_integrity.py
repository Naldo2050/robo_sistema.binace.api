import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log_path = "logs/collect_2h_20260908_103451.log"
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

p1 = re.compile(r"error|exception|traceback|crash|fatal", re.IGNORECASE)
p2 = re.compile(r"reconnect|conexão perdida|websocket closed", re.IGNORECASE)

m1 = [l.strip() for l in lines if p1.search(l)]
m2 = [l.strip() for l in lines if p2.search(l)]

print("=" * 80)
print("VALIDAÇÃO 0 — INTEGRIDADE DA SESSÃO DE 2 HORAS")
print("=" * 80)
print(f"Log avaliado: {log_path}")
print(f"Total de linhas no log: {len(lines)}")
print(f"Count (error|exception|traceback|crash|fatal): {len(m1)}")
print(f"Count (reconnect|conexão perdida|websocket closed): {len(m2)}")

print("\n--- Linhas com 'error|exception|traceback|crash|fatal' ---")
for l in m1:
    print(" ", l)

print("\n--- Linhas com 'reconnect|conexão perdida|websocket closed' ---")
for l in m2:
    print(" ", l)
