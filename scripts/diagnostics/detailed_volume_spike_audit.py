import sqlite3
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trading.alert_engine import _get_volume_baseline_p95

log_path = "logs/collect_2h_20260908_103451.log"
print("=" * 80)
print("AUDITORIA DETALHADA: VALIDAÇÃO 4 — GATE DUPLO DE VOLUME_SPIKE")
print("=" * 80)

# 1. Checar matches no log
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    log_lines = f.readlines()

log_matches = [l.strip() for l in log_lines if "VOLUME_SPIKE" in l or "VOLUME SPIKE" in l]
print(f'grep "VOLUME_SPIKE" {log_path}: {len(log_matches)} disparos')
for m in log_matches:
    print("  ", m)

# 2. Avaliar janelas no SQLite
conn = sqlite3.connect("dados/trading_bot.db")
cur = conn.cursor()
cur.execute("SELECT id, timestamp_ms, event_type, payload FROM events WHERE event_type = 'ANALYSIS_TRIGGER' ORDER BY timestamp_ms ASC")
rows = cur.fetchall()

volumes = []
window_records = []
for eid, ts, etype, raw in rows:
    p = json.loads(raw) if isinstance(raw, str) else (raw or {})
    vol = p.get("volume_total", 0.0)
    dt_utc = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    hour_utc = dt_utc.hour
    p95_thresh = _get_volume_baseline_p95(hour_utc)
    window_records.append({
        "id": eid,
        "ts": dt_utc.strftime("%H:%M:%S UTC"),
        "hour_utc": hour_utc,
        "vol": float(vol),
        "p95": p95_thresh
    })
    volumes.append(float(vol))

print(f"\nTotal de janelas de 1 minuto avaliadas: {len(window_records)}")
print(f"Volume mínimo: {min(volumes):.2f} BTC | Volume máximo: {max(volumes):.2f} BTC | Volume mediano: {np.median(volumes):.2f} BTC")
print(f"Baseline p95 horário de referência: {window_records[0]['p95']:.2f} BTC (h={window_records[0]['hour_utc']})")

# Avaliação com média móvel (20 períodos) e média da sessão
quase_disparos_ratio = []
quase_disparos_p95 = []
disparos_reais = []

for i, w in enumerate(window_records):
    # MA dos últimos até 20 períodos
    start_idx = max(0, i - 20)
    past_vols = [window_records[j]["vol"] for j in range(start_idx, i)] if i > 0 else [w["vol"]]
    ma = np.mean(past_vols) if past_vols else w["vol"]
    ratio = (w["vol"] / ma) if ma > 0 else 1.0
    
    cond1_ratio = ratio >= 3.0
    cond2_p95 = w["vol"] >= w["p95"]

    if cond1_ratio and cond2_p95:
        disparos_reais.append({**w, "ma": ma, "ratio": ratio})
    elif cond1_ratio and not cond2_p95:
        quase_disparos_ratio.append({**w, "ma": ma, "ratio": ratio})
    elif not cond1_ratio and cond2_p95:
        quase_disparos_p95.append({**w, "ma": ma, "ratio": ratio})

print(f"\nDisparos confirmados com AMBAS as condições satisfeitas: {len(disparos_reais)}")
for d in disparos_reais:
    print(f"  ✅ Disparo Janela {d['id']} às {d['ts']}: Vol={d['vol']:.2f} BTC >= p95({d['p95']:.2f}) E Ratio={d['ratio']:.2f}x >= 3.0x")

print(f"\nQuase-disparos — Condição 1 satisfeita (Ratio >= 3.0x) mas Bloqueado por p95 (Vol < {window_records[0]['p95']:.2f} BTC): {len(quase_disparos_ratio)}")
for q in quase_disparos_ratio:
    print(f"  ⚠️ Quase-disparo Janela {q['id']} às {q['ts']}: Vol={q['vol']:.2f} BTC (Ratio={q['ratio']:.2f}x >= 3.0x, mas BLOQUEADO por p95={q['p95']:.2f} BTC)")

print(f"\nQuase-disparos — Condição 2 satisfeita (Vol >= p95) mas Bloqueado por Ratio (< 3.0x): {len(quase_disparos_p95)}")
for q in quase_disparos_p95:
    print(f"  ⚠️ Quase-disparo Janela {q['id']} às {q['ts']}: Vol={q['vol']:.2f} BTC >= p95={q['p95']:.2f}, mas Ratio={q['ratio']:.2f}x < 3.0x")

print("=" * 80)
