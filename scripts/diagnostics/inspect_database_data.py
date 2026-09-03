# scripts/diagnostics/inspect_database_data.py
import sqlite3
import os
import glob

db = 'dados/trading_bot.db'
print("Directory listing of dados/:", glob.glob('dados/*'))
if os.path.exists(db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print("Tables in trading_bot.db:")
    for (t_name,) in tables:
        cur.execute(f"SELECT count(*) FROM {t_name}")
        count = cur.fetchone()[0]
        print(f"  - {t_name}: {count} records")
        if count > 0:
            cur.execute(f"SELECT * FROM {t_name} LIMIT 1")
            cols = [desc[0] for desc in cur.description]
            print(f"    Columns ({len(cols)}): {cols}")
    conn.close()
