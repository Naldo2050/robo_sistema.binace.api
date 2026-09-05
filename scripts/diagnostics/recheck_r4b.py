#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/recheck_r4b.py
Executa com travas estritas a Etapa B corrigida da Auditoria R4b.
"""

import os
import json
import math
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# -------------------------------------------------------------
# PASSO 1: Leitura do CSV e verificação de t0, t1 e ASSERT 1
# -------------------------------------------------------------
print("=== PASSO 1: LEITURA E ASSERTS DE TIMESTAMP ===")
csv_path = "dados/audit/windows_flat.csv"
df = pd.read_csv(csv_path)

# Filtrar sessão 1
# Nota: na auditoria R1 e R3, a sessão 1 corresponde às janelas ANALYSIS_TRIGGER da sessão 1
s1 = df[df["meta_session"] == 1].copy()
epochs = s1["meta_epoch_ms"].astype("int64")
t0 = int(epochs.min())
t1 = int(epochs.max())

dt0 = datetime.fromtimestamp(t0 / 1000, tz=timezone.utc)
dt1 = datetime.fromtimestamp(t1 / 1000, tz=timezone.utc)

print(f"t0: {t0} ({dt0.strftime('%Y-%m-%dT%H:%M:%SZ')})")
print(f"t1: {t1} ({dt1.strftime('%Y-%m-%dT%H:%M:%SZ')})")

t0_bound_low = int(datetime(2026, 9, 1, 23, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
t0_bound_high = int(datetime(2026, 9, 1, 23, 30, 0, tzinfo=timezone.utc).timestamp() * 1000)

assert t0_bound_low <= t0 <= t0_bound_high, (
    f"ASSERT 1 FALHOU: t0={t0} ({dt0.isoformat()}) fora de [2026-09-01T23:00Z, 2026-09-01T23:30Z]"
)
print("[OK] ASSERT 1: t0 está estritamente entre 2026-09-01T23:00Z e 2026-09-01T23:30Z.")

# -------------------------------------------------------------
# PASSO 2: Buscar klines 1m SPOT e FUTURES via REST
# -------------------------------------------------------------
print("\n=== PASSO 2: BUSCA DE KLINES 1M (SPOT & FUTURES) ===")
# startTime = floor(t0/60000)*60000 - 60000
start_time = math.floor(t0 / 60000) * 60000 - 60000
end_time = t1 + 60000
limit = 200

print(f"start_time: {start_time} ({datetime.fromtimestamp(start_time/1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})")
print(f"end_time:   {end_time} ({datetime.fromtimestamp(end_time/1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})")

url_spot = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime={start_time}&endTime={end_time}&limit={limit}"  # SPOT intencional porque compara spot vs fut na auditoria R4b
url_fut = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime={start_time}&endTime={end_time}&limit={limit}"

def fetch_and_save(url, path):
    print(f"Buscando {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read().decode("utf-8")
        data = json.loads(content)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Salvo em {path} ({len(data)} candles)")
    return data

spot_file = "dados/audit/klines_spot_s1.json"
fut_file = "dados/audit/klines_fut_s1.json"

spot_klines = fetch_and_save(url_spot, spot_file)
fut_klines = fetch_and_save(url_fut, fut_file)

# Imprimir primeiro e último candle cru
first_spot = spot_klines[0]
last_spot = spot_klines[-1]
first_fut = fut_klines[0]
last_fut = fut_klines[-1]

first_spot_ot_dt = datetime.fromtimestamp(first_spot[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
last_spot_ot_dt = datetime.fromtimestamp(last_spot[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
first_fut_ot_dt = datetime.fromtimestamp(first_fut[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
last_fut_ot_dt = datetime.fromtimestamp(last_fut[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

print(f"Primeiro candle SPOT: open_time={first_spot[0]} ({first_spot_ot_dt}), close={first_spot[4]}, vol={first_spot[5]}")
print(f"Último candle SPOT:   open_time={last_spot[0]} ({last_spot_ot_dt}), close={last_spot[4]}, vol={last_spot[5]}")
print(f"Primeiro candle FUT:  open_time={first_fut[0]} ({first_fut_ot_dt}), close={first_fut[4]}, vol={first_fut[5]}")
print(f"Último candle FUT:    open_time={last_fut[0]} ({last_fut_ot_dt}), close={last_fut[4]}, vol={last_fut[5]}")

# ASSERT: open_time do primeiro candle está dentro de 2 min de t0
diff_spot_ms = abs(first_spot[0] - t0)
diff_fut_ms = abs(first_fut[0] - t0)
print(f"Diferença first candle SPOT vs t0: {diff_spot_ms/1000:.1f}s")
print(f"Diferença first candle FUT vs t0:  {diff_fut_ms/1000:.1f}s")

assert diff_spot_ms <= 120_000, f"ASSERT 2 FALHOU: open_time SPOT ({first_spot[0]}) a mais de 2 min de t0 ({t0})"
assert diff_fut_ms <= 120_000, f"ASSERT 2 FALHOU: open_time FUT ({first_fut[0]}) a mais de 2 min de t0 ({t0})"
print("[OK] ASSERT 2: open_time do primeiro candle de ambos está dentro de 2 min de t0.")
