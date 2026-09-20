#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import urllib.request
import json
from datetime import datetime, timezone

def investigate():
    start_ms = int(datetime(2026, 9, 5, 1, 55, 0, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(2026, 9, 5, 2, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)

    url_kline = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime={start_ms}&endTime={end_ms}"
    req = urllib.request.Request(url_kline, headers={"User-Agent": "Audit/1.0"})
    with urllib.request.urlopen(req) as r:
        klines = json.loads(r.read())

    print("=== KLINES 1m (01:55 - 02:10 UTC em 2026-09-05) ===")
    for k in klines:
        t_dt = datetime.fromtimestamp(k[0]/1000, tz=timezone.utc).strftime("%H:%M")
        o, h, l, c, vol, n_trades = k[1], k[2], k[3], k[4], k[5], k[8]
        print(f"{t_dt} UTC | Open: {o} | High: {h} | Low: {l} | Close: {c} | Vol: {float(vol):.2f} BTC | Trades: {n_trades}")

    url_oi = f"https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&startTime={start_ms - 15*60*1000}&endTime={end_ms + 15*60*1000}&limit=15"
    try:
        req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "Audit/1.0"})
        with urllib.request.urlopen(req_oi) as r:
            oi_data = json.loads(r.read())
        print("\n=== OPEN INTEREST HIST (5m) ===")
        for row in oi_data:
            t_dt = datetime.fromtimestamp(row["timestamp"]/1000, tz=timezone.utc).strftime("%H:%M")
            oi_btc = float(row["sumOpenInterest"])
            oi_val = float(row["sumOpenInterestValue"])
            print(f"{t_dt} UTC | Sum Open Interest: {oi_btc:,.2f} BTC | Sum OI Value: ${oi_val:,.0f}")
    except Exception as e:
        print("Erro OI:", e)

if __name__ == "__main__":
    investigate()
