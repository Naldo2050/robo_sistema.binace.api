import sqlite3
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log_path = "logs/collect_2h_20260908_103451.log"
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

ml_neut_count = text.count("ML neutralizado")
print(f'grep -c "ML neutralizado" {log_path}: {ml_neut_count}')

conn = sqlite3.connect("dados/trading_bot.db")
cur = conn.cursor()
cur.execute("SELECT id, event_type, payload FROM events")
rows = cur.fetchall()

violations_vff = 0
violations_mls = 0
ml_events_checked = 0

for eid, etype, p_raw in rows:
    p = json.loads(p_raw) if isinstance(p_raw, str) else (p_raw or {})
    
    def check_dict(d):
        global violations_vff, violations_mls, ml_events_checked
        if not isinstance(d, dict):
            return
        if "valid_for_futures" in d:
            ml_events_checked += 1
            if d["valid_for_futures"] is True or str(d["valid_for_futures"]).lower() == "true":
                violations_vff += 1
        if "ml_stale" in d:
            if d["ml_stale"] is False or str(d["ml_stale"]).lower() == "false":
                violations_mls += 1
        for k, v in d.items():
            if isinstance(v, dict):
                check_dict(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        check_dict(item)

    check_dict(p)

print(f"Total de eventos checados no SQLite: {len(rows)}")
print(f"Campos ML avaliados recursivamente: {ml_events_checked}")
print(f"Violações valid_for_futures=True no DB: {violations_vff}")
print(f"Violações ml_stale=False no DB: {violations_mls}")
