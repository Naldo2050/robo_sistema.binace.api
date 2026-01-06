import sqlite3
from pathlib import Path

db = Path("dados/trading_bot.db")
con = sqlite3.connect(db)
cur = con.cursor()

sql = cur.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='events'"
).fetchone()

print("CREATE TABLE events:")
print(sql[0] if sql else "NOT FOUND")

print("\nPRAGMA table_info(events):")
for row in cur.execute("PRAGMA table_info(events)"):
    print(row)