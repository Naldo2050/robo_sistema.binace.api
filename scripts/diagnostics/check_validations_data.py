import sqlite3
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=== CHECK VALIDAÇÃO 2 ===")
run_log = Path("logs/run.log")
if run_log.exists():
    content = run_log.read_text(encoding="utf-8", errors="replace")
    print("Contagem 'ML neutralizado' em logs/run.log:", content.count("ML neutralizado"))
    print("Contagem 'ml_stale' em logs/run.log:", content.count("ml_stale"))
    print("Contagem 'valid_for_futures' em logs/run.log:", content.count("valid_for_futures"))
    # procurar linhas com ml_stale
    for line in content.splitlines():
        if "ml_stale" in line or "valid_for_futures" in line or "ML neutralizado" in line:
            print("  Linha encontrada:", line.strip())
else:
    print("logs/run.log nao existe")

db_path = Path("dados/trading_bot.db")
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("Tabelas no DB:", tables)
    conn.close()

print("\n=== CHECK VALIDAÇÃO 3 ===")
# Checar se existe tabela trades
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT count(*) FROM trades")
        print("Total trades:", cur.fetchone()[0])
    except Exception as e:
        print("Erro ao consultar trades:", e)
    conn.close()

print("\n=== CHECK VALIDAÇÃO 4 ===")
if run_log.exists():
    spikes = [l.strip() for l in content.splitlines() if "VOLUME_SPIKE" in l]
    print("Contagem 'VOLUME_SPIKE' em logs/run.log:", len(spikes))
    for s in spikes[:10]:
        print("  Spike:", s)

print("\n=== CHECK VALIDAÇÃO 5 ===")
if db_path.exists():
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT count(*) FROM events WHERE is_signal = 1")
        print("Sinais em events:", cur.fetchone()[0])
    except Exception as e:
        print("Erro ao consultar events para sinais:", e)
    conn.close()

print("\n=== CHECK VALIDAÇÃO 6 ===")
if run_log.exists():
    errors = [l.strip() for l in content.splitlines() if any(k in l.lower() for k in ["error", "exception", "traceback"])]
    print("Total linhas com error/exception/traceback em logs/run.log:", len(errors))
    reconnects = [l.strip() for l in content.splitlines() if any(k in l.lower() for k in ["reconnect", "reconnecting", "conexão perdida"])]
    print("Total reconexoes em logs/run.log:", len(reconnects))
