#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amostragem estratificada de trades aggTrade (Binance Futures BTCUSDT)
Cobre 3 sessões de 1h:
1. Sessão Asiática (02:00 - 03:00 UTC)
2. Sessão de Londres (09:00 - 10:00 UTC)
3. Sessão de Nova York (14:00 - 15:00 UTC)
Calcula p50, p90, p99, p99.5 e max de trade size (BTC).
"""
import urllib.request
import json
import time
from datetime import datetime, timezone
import numpy as np

def fetch_continuous_hour(start_hour: int, date_day: int = 5):
    start_dt = datetime(2026, 9, date_day, start_hour, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, date_day, start_hour + 1, 0, 0, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    all_quantities = []
    current_ms = start_ms
    
    # Amostramos intervalos de 1 minuto a cada 3 minutos para cobrir a hora inteira de forma homogênea (20 janelas por hora)
    for minute in range(0, 60, 3):
        win_start = start_ms + minute * 60 * 1000
        win_end = win_start + 60 * 1000
        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&startTime={win_start}&endTime={win_end}&limit=1000"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TradeAudit/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                trades = json.loads(resp.read().decode("utf-8"))
                for t in trades:
                    all_quantities.append(float(t["q"]))
        except Exception as e:
            print(f"Erro na janela {minute}m: {e}")
        time.sleep(0.05)
        
    arr = np.array(all_quantities)
    if len(arr) == 0:
        return {"count": 0, "p50": 0, "p90": 0, "p99": 0, "p99_5": 0, "max": 0}
        
    return {
        "count": len(arr),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "p99_5": float(np.percentile(arr, 99.5)),
        "max": float(np.max(arr))
    }

if __name__ == "__main__":
    sessions = [
        ("Sessão Asiática (02:00 - 03:00 UTC)", 2),
        ("Sessão de Londres (09:00 - 10:00 UTC)", 9),
        ("Sessão de Nova York (14:00 - 15:00 UTC)", 14),
    ]
    
    print("=" * 75)
    print("AMOSTRAGEM ESTRATIFICADA DE TRADE SIZES (BTCUSDT Binance Futures) - 2026-09-05")
    print("=" * 75)
    
    results = {}
    for name, hour in sessions:
        stats = fetch_continuous_hour(hour, date_day=5)
        results[name] = stats
        print(f"\n--- {name} ---")
        print(f"  Trades analisados: {stats['count']:,}")
        print(f"  p50 (Mediana):     {stats['p50']:.4f} BTC")
        print(f"  p90:               {stats['p90']:.4f} BTC")
        print(f"  p99:               {stats['p99']:.4f} BTC")
        print(f"  p99.5:             {stats['p99_5']:.4f} BTC")
        print(f"  Máximo:            {stats['max']:.4f} BTC")

    print("\n" + "=" * 75)
    print("COMPARAÇÃO ENTRE SESSÕES (p99 e Variação):")
    print("=" * 75)
    for name, stats in results.items():
        print(f"{name:40s} | p99: {stats['p99']:.4f} BTC | max: {stats['max']:.4f} BTC")
