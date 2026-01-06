#!/usr/bin/env python3
"""Mostra linhas problemáticas do main.py"""

with open("main.py", 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("="*70)
print("📍 LINHA 424 (Close sem reconnect):")
print("="*70)
for i in range(max(0, 424-10), min(len(lines), 424+10)):
    marker = "👉" if i == 423 else "  "
    print(f"{marker} {i+1:4d} | {lines[i].rstrip()}")

print("\n" + "="*70)
print("📍 LINHA 1446 (sys.exit sem retry):")
print("="*70)
for i in range(max(0, 1446-10), min(len(lines), 1446+10)):
    marker = "👉" if i == 1445 else "  "
    print(f"{marker} {i+1:4d} | {lines[i].rstrip()}")

print("\n" + "="*70)
print("🔍 Buscando todas as ocorrências de sys.exit e websocket close...")
print("="*70)

problems = []
for i, line in enumerate(lines, 1):
    if 'sys.exit' in line.lower() or 'systemexit' in line.lower():
        problems.append((i, "sys.exit", line.strip()))
    if 'ws.close' in line.lower() or 'websocket.close' in line.lower():
        problems.append((i, "ws.close", line.strip()))
    if 'ping' in line.lower() and 'pong' in line.lower() and 'timeout' in line.lower():
        problems.append((i, "ping/pong timeout", line.strip()))

for line_num, prob_type, code in problems:
    print(f"\n{prob_type} na linha {line_num}:")
    print(f"  {code}")