import sqlite3
from pathlib import Path

db_path = Path("dados/trading_bot.db")
print("DB:", db_path.resolve(), "exists:", db_path.exists(), "size:", db_path.stat().st_size if db_path.exists() else 0)

con = sqlite3.connect(db_path)
cur = con.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
print("Tables:", tables)

for t in tables:
    try:
        n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"{t}: {n}")
    except Exception as e:
        print(f"{t}: ERROR -> {e}")