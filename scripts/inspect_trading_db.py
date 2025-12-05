# scripts/inspect_trading_db.py
import os
import sys
from datetime import datetime

# Garante que a pasta raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.event_store import EventStore

DB_PATH = os.path.join("dados", "trading_bot.db")


def main():
    if not os.path.exists(DB_PATH):
        print(f"❌ Banco não encontrado: {DB_PATH}")
        return

    db = EventStore(DB_PATH)
    events = db.get_recent_events(limit=20)

    print(f"\n📦 Lidos {len(events)} eventos de {DB_PATH}\n")

    if not events:
        print("⚠️ Nenhum evento encontrado no banco.\n")
        return

    for i, evt in enumerate(events, 1):
        ts = evt.get("epoch_ms") or evt.get("timestamp_ms")
        if ts:
            try:
                dt = datetime.fromtimestamp(int(ts) / 1000)
                ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                ts_str = f"epoch_ms={ts}"
        else:
            ts_str = "sem timestamp"

        tipo = evt.get("tipo_evento") or evt.get("type", "N/A")
        ativo = evt.get("ativo") or evt.get("symbol", "N/A")

        print(f"{i:02d}. [{ts_str}] tipo={tipo} | ativo={ativo}")

    print()


if __name__ == "__main__":
    main()