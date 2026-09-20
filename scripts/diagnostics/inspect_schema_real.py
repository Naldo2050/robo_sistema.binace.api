import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

conn = sqlite3.connect('dados/trading_bot.db')
cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tabelas = [r[0] for r in cur.fetchall()]
print('Tabelas existentes:', tabelas)
for t in tabelas:
    cur2 = conn.execute(f'PRAGMA table_info({t})')
    print(f'--- {t} ---')
    for col in cur2.fetchall():
        print(' ', col)
    cur3 = conn.execute(f'SELECT COUNT(*) FROM {t}')
    print(' linhas:', cur3.fetchone()[0])
conn.close()
