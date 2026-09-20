#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/decompose_london_ny_trades.py
Baixa aggTrades das sessões de Londres (09-10h UTC) e NY (14-15h UTC) do dia 2026-09-05,
salva em JSON e investiga todos os trades > 10 BTC:
- Timestamp, Preço, Quantidade, Lado, First ID, Last ID, Num Fills
- Verifica se há clusters temporais de alta concentração (ex: múltiplos trades no mesmo segundo/sub-segundo)
"""
import urllib.request
import json
import time
from datetime import datetime, timezone

def fetch_session_trades(start_hour: int, output_file: str, date_day: int = 5):
    start_dt = datetime(2026, 9, date_day, start_hour, 0, 0, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    
    all_trades = []
    for minute in range(0, 60, 3):
        win_start = start_ms + minute * 60 * 1000
        win_end = win_start + 60 * 1000
        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&startTime={win_start}&endTime={win_end}&limit=1000"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Audit/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                trades = json.loads(resp.read().decode("utf-8"))
                all_trades.extend(trades)
        except Exception as e:
            print(f"Erro na janela {minute}m: {e}")
        time.sleep(0.05)
        
    with open(output_file, "w") as f:
        json.dump(all_trades, f)
        
    print(f"[{output_file}] Total trades salvos: {len(all_trades)}")
    return all_trades

def analyze_session(name: str, trades: list):
    print("\n" + "=" * 80)
    print(f"ANÁLISE DE TRADES > 10 BTC: {name}")
    print("=" * 80)
    
    grandes = [t for t in trades if float(t["q"]) > 10.0]
    print(f"Total de trades > 10 BTC encontrados: {len(grandes)} (de {len(trades)} trades)")
    
    if not grandes:
        print("Nenhum trade > 10 BTC nesta sessão.")
        return
        
    for t in grandes:
        dt = datetime.fromtimestamp(t["T"] / 1000, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        side = "SELL" if t["m"] else "BUY"
        fills = int(t.get("l", 0)) - int(t.get("f", 0)) + 1
        usd = float(t["q"]) * float(t["p"])
        print(f"{dt} UTC | Preço: {t['p']} | Qtd: {t['q']} BTC (~${usd:,.0f}) | Lado: {side:4s} | Fills: {fills:2d} | IDs: {t['f']}..{t['l']}")

if __name__ == "__main__":
    t_london = fetch_session_trades(9, "dados/audit/aggtrades_london_sample.json")
    analyze_session("Sessão de Londres (09:00 - 10:00 UTC)", t_london)
    
    t_ny = fetch_session_trades(14, "dados/audit/aggtrades_ny_sample.json")
    analyze_session("Sessão de Nova York (14:00 - 15:00 UTC)", t_ny)
