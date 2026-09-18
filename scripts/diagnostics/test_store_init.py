import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import sqlite3
from database.event_store import EventStore
from common.backfill_guard import enforce_no_live_bot

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# AÇÃO 2 (auditoria 18/09): este script instancia o store DEFAULT, que aponta
# para o DB de produção (dados/trading_bot.db). Padrão copiado por backfills:
# abortar se o bot ao vivo estiver rodando para evitar contaminação.
enforce_no_live_bot()

store = EventStore()
print("EventStore initialized at:", store.db_path)
conn = sqlite3.connect(store.db_path)
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables in DB after EventStore init:", tables)
conn.close()
