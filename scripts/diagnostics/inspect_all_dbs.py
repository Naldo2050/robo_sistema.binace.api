import sqlite3
import os

dbs = [
    'dados/trading_bot.db',
    'backups/shadow_dataset_preflight_test.db',
    'dados/archive_spot_2026-09-03/trading_bot.db',
    'database/trading_bot.db'
]

for db in dbs:
    print("=" * 60)
    print(f"DB: {db} (exists: {os.path.exists(db)}, size: {os.path.getsize(db) if os.path.exists(db) else 0} bytes)")
    if os.path.exists(db):
        try:
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            for (t_name,) in tables:
                cur.execute(f"SELECT count(*) FROM [{t_name}]")
                cnt = cur.fetchone()[0]
                print(f"  Table '{t_name}': {cnt} rows")
                if cnt > 0:
                    cur.execute(f"SELECT * FROM [{t_name}] LIMIT 1")
                    cols = [d[0] for d in cur.description]
                    print(f"     cols: {cols[:6]}...")
            conn.close()
        except Exception as e:
            print("  Error:", e)
