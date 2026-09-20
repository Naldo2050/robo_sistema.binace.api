import json
import re
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log_path = "logs/collect_2h_20260908_103451.log"
trades_path = "dados/trades_collect_2h.jsonl"

print("=" * 80)
print("INVESTIGAÇÃO DE CAUSA RAIZ: PONTO 1 — 11 OCORRÊNCIAS DE LATÊNCIA CRÍTICA")
print("=" * 80)

# 1. Extrair os 11 eventos do log
pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ERROR - 🚨 LATÊNCIA CRÍTICA: process_trade took ([\d\.]+)ms")

occurrences = []
with open(log_path, "r", encoding="utf-8", errors="replace") as f:
    for line_idx, line in enumerate(f, 1):
        m = pattern.search(line)
        if m:
            dt_str, dur_str = m.groups()
            # Log local time (America/Sao_Paulo: UTC-3)
            dt_local = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S,%f")
            # Converter para UTC (adicionar 3 horas)
            ts_epoch = int(dt_local.timestamp()) # local timestamp on Windows
            occurrences.append({
                "line": line_idx,
                "dt_str": dt_str,
                "dt_local": dt_local,
                "duration_ms": float(dur_str)
            })

print(f"Total de ocorrências identificadas: {len(occurrences)}")

# 2. Carregar trades_por_segundo de dados/trades_collect_2h.jsonl
print(f"Lendo trades de {trades_path}...")
trades_por_segundo = defaultdict(int)
total_trades = 0
min_trade_ts = float("inf")
max_trade_ts = 0

with open(trades_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        t = json.loads(line)
        ts_ms = t.get("timestamp") or t.get("T") or t.get("ts_ms") or t.get("ts")
        if ts_ms:
            ts_sec = ts_ms // 1000
            trades_por_segundo[ts_sec] += 1
            total_trades += 1
            if ts_ms < min_trade_ts: min_trade_ts = ts_ms
            if ts_ms > max_trade_ts: max_trade_ts = ts_ms

all_rates = list(trades_por_segundo.values())
mean_rate = np.mean(all_rates)
p50_rate = np.percentile(all_rates, 50)
p90_rate = np.percentile(all_rates, 90)
p95_rate = np.percentile(all_rates, 95)
p99_rate = np.percentile(all_rates, 99)
max_rate = np.max(all_rates)

print(f"Estatísticas globais da sessão (N={total_trades} trades):")
print(f"  • Média: {mean_rate:.1f} trades/s | p50: {p50_rate:.1f} | p90: {p90_rate:.1f} | p95: {p95_rate:.1f} | p99: {p99_rate:.1f} | Max: {max_rate} trades/s")

# 3. Cruzar cada ocorrência com trades_por_segundo
# Como o log tem timestamp local do sistema operacional, vamos alinhar pelo timestamp do primeiro trade
# min_trade_ts em UTC vs início da sessão no log
# Vamos encontrar a correspondência exata de epoch
print("\n" + "=" * 80)
print("CRUZAMENTO DAS 11 OCORRÊNCIAS COM TAXA DE TRADES/S")
print("=" * 80)

# Para cada ocorrência, calcular epoch_sec em UTC:
# dt_local = YYYY-MM-DD HH:MM:SS,fff local (-03:00) -> UTC = +3h
for idx, occ in enumerate(occurrences, 1):
    # hora local para UTC: adicionar 3 horas
    utc_dt = occ["dt_local"].replace(tzinfo=timezone.utc).timestamp() + (3 * 3600)
    epoch_sec = int(utc_dt)
    
    # Pegar janela [-5s, ..., +1s]
    window_rates = [trades_por_segundo.get(epoch_sec + offset, 0) for offset in range(-5, 2)]
    rate_at_t = trades_por_segundo.get(epoch_sec, 0)
    rate_prev5_avg = np.mean(window_rates[:5])
    max_in_window = max(window_rates)
    
    print(f"[{idx:02d}] {occ['dt_str']} (Local) / {datetime.fromtimestamp(epoch_sec, tz=timezone.utc).strftime('%H:%M:%S')} UTC | Duração: {occ['duration_ms']:7.2f} ms")
    print(f"     Taxa no segundo T: {rate_at_t:3d} trades/s | Janela T-5..T: {window_rates[:-1]} (Máx janela: {max_in_window} trades/s)")
    if max_in_window >= p95_rate:
        print(f"     🔥 PICO DE VOLUME CONFIRMADO: Taxa na janela ({max_in_window} t/s) >= p95 global ({p95_rate:.1f} t/s)")
    else:
        print(f"     ℹ️ Volume moderado/normal ({max_in_window} t/s)")
