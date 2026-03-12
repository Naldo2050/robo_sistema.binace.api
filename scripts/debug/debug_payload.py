import sqlite3
import json
from pathlib import Path
import pandas as pd

DB_PATH = Path("dados/trading_bot.db")  # ajuste se seu .db estiver em outro lugar

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT timestamp_ms, event_type, symbol, is_signal, payload
    FROM events
    ORDER BY timestamp_ms DESC
    LIMIT 5
""", conn)
conn.close()

print("=== Cabeçalho básico ===")
print(df[["timestamp_ms", "event_type", "symbol", "is_signal"]])

print("\n=== Payload bruto do evento mais recente ===")
print(df["payload"].iloc[0])

print("\n=== Payload em JSON formatado ===")
print(json.dumps(json.loads(df["payload"].iloc[0]), indent=2, ensure_ascii=False))