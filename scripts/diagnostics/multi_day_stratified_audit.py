#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/multi_day_stratified_audit.py

Auditoria multi-dia estratificada (3 dias completos: 03/09, 04/09, 05/09 de 2026)
cobrindo as 3 sessões:
- Ásia (02:00 - 03:00 UTC)
- Londres (09:00 - 10:00 UTC)
- NY (14:00 - 15:00 UTC)

Fonte de dados:
- 2026-09-03: Arquivo diário oficial da Binance Data Vision (100% dos trades)
- 2026-09-04: Arquivo diário oficial da Binance Data Vision (100% dos trades)
- 2026-09-05: Coleta estratificada via Binance Futures REST API (fapi/v1/aggTrades)
"""
import urllib.request
import zipfile
import io
import csv
import json
from datetime import datetime, timezone
import numpy as np

SESSIONS = [
    ("Ásia", 2, 3),
    ("Londres", 9, 10),
    ("NY", 14, 15),
]

def load_day_from_datavision(date_str: str):
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{date_str}.zip"
    print(f"Baixando arquivo oficial da Binance Data Vision para {date_str}...")
    req = urllib.request.Request(url, headers={"User-Agent": "MultiDayAudit/1.0"})
    with urllib.request.urlopen(req) as r:
        content = r.read()
    print(f"Download concluído ({len(content)/(1024*1024):.1f} MB). Descompactando em memória...")
    z = zipfile.ZipFile(io.BytesIO(content))
    csv_filename = z.namelist()[0]
    
    # Mapear timestamps das 3 sessões
    session_windows = {}
    for s_name, s_start, s_end in SESSIONS:
        dt_start = datetime.strptime(f"{date_str} {s_start:02d}:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dt_end = datetime.strptime(f"{date_str} {s_end:02d}:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        session_windows[s_name] = (int(dt_start.timestamp() * 1000), int(dt_end.timestamp() * 1000))
        
    session_trades = {s_name: [] for s_name, _, _ in SESSIONS}
    
    with z.open(csv_filename) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
        header = next(reader)
        # header: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
        for row in reader:
            t_ms = int(row[5])
            for s_name, (w_start, w_end) in session_windows.items():
                if w_start <= t_ms < w_end:
                    session_trades[s_name].append({
                        "a": int(row[0]),
                        "p": row[1],
                        "q": row[2],
                        "f": int(row[3]),
                        "l": int(row[4]),
                        "T": t_ms,
                        "m": row[6].lower() == "true",
                    })
                    break
    print(f"Extração concluída: {', '.join(f'{k}: {len(v)} trades' for k, v in session_trades.items())}")
    return session_trades

def analyze_trades_group(trades):
    if not trades:
        return {"n": 0, "p50": 0, "p90": 0, "p99": 0, "p99_5": 0, "max": 0, "grandes_n": 0, "clusters_n": 0, "grandes": []}
    
    qs = np.array([float(t["q"]) for t in trades])
    grandes = [t for t in trades if float(t["q"]) > 10.0]
    
    # Identificar clusters temporais nos grandes trades (delta <= 2000 ms)
    clusters = []
    if len(grandes) > 1:
        grandes_sorted = sorted(grandes, key=lambda x: x["T"])
        current_cluster = [grandes_sorted[0]]
        for t in grandes_sorted[1:]:
            if (t["T"] - current_cluster[-1]["T"]) <= 2000:
                current_cluster.append(t)
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [t]
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            
    return {
        "n": len(qs),
        "p50": float(np.percentile(qs, 50)),
        "p90": float(np.percentile(qs, 90)),
        "p99": float(np.percentile(qs, 99)),
        "p99_5": float(np.percentile(qs, 99.5)),
        "max": float(np.max(qs)),
        "grandes_n": len(grandes),
        "clusters_n": len(clusters),
        "grandes": grandes,
    }

def main():
    print("=" * 90)
    print("AUDITORIA MULTI-DIA ESTRATIFICADA DE SESSÕES (03/09, 04/09 e 05/09 de 2026)")
    print("=" * 90)
    
    all_data = {}
    
    # 1. Carregar 03/09 e 04/09 da Binance Data Vision
    all_data["2026-09-03"] = load_day_from_datavision("2026-09-03")
    all_data["2026-09-04"] = load_day_from_datavision("2026-09-04")
    
    # 2. Carregar 05/09 dos arquivos JSON locais salvos
    all_data["2026-09-05"] = {
        "Ásia": json.load(open("dados/audit/aggtrades_asia_sample.json")),
        "Londres": json.load(open("dados/audit/aggtrades_london_sample.json")),
        "NY": json.load(open("dados/audit/aggtrades_ny_sample.json")),
    }
    
    summary = {}
    for date_str, sessions_dict in all_data.items():
        summary[date_str] = {}
        print(f"\n>>> DIA: {date_str} <<<")
        for s_name, _, _ in SESSIONS:
            t_list = sessions_dict[s_name]
            stats = analyze_trades_group(t_list)
            summary[date_str][s_name] = stats
            print(f"  [{s_name:7s}] N: {stats['n']:6,d} | p50: {stats['p50']:.4f} | p90: {stats['p90']:.4f} | p99: {stats['p99']:.4f} | max: {stats['max']:6.2f} BTC | >10BTC: {stats['grandes_n']} | clusters: {stats['clusters_n']}")
            if stats["grandes"]:
                for g in stats["grandes"]:
                    dt_str = datetime.fromtimestamp(g["T"]/1000, tz=timezone.utc).strftime("%H:%M:%S.%f")[:-3]
                    fills = int(g.get("l", 0)) - int(g.get("f", 0)) + 1
                    side = "SELL" if g["m"] else "BUY"
                    print(f"       -> {dt_str} UTC | {g['q']} BTC | {side} | fills: {fills}")

    print("\n" + "=" * 90)
    print("TABELA COMPARATIVA DE PERCENTIL 99 (p99) ENTRE SESSÕES E DIAS (BTC)")
    print("=" * 90)
    print(f"{'Sessão':10s} | {'2026-09-03':12s} | {'2026-09-04':12s} | {'2026-09-05':12s} | {'Média p99':10s} | {'Mín p99':10s} | {'Máx p99':10s}")
    print("-" * 90)
    for s_name, _, _ in SESSIONS:
        vals = [summary[d][s_name]["p99"] for d in ["2026-09-03", "2026-09-04", "2026-09-05"]]
        print(f"{s_name:10s} | {vals[0]:12.4f} | {vals[1]:12.4f} | {vals[2]:12.4f} | {np.mean(vals):10.4f} | {np.min(vals):10.4f} | {np.max(vals):10.4f}")

    print("\n" + "=" * 90)
    print("OCORRÊNCIA DE CLUSTERS TEMPORAIS (Trades > 10 BTC em <= 2 segundos)")
    print("=" * 90)
    for date_str in ["2026-09-03", "2026-09-04", "2026-09-05"]:
        for s_name, _, _ in SESSIONS:
            c_cnt = summary[date_str][s_name]["clusters_n"]
            g_cnt = summary[date_str][s_name]["grandes_n"]
            print(f"  • {date_str} {s_name:7s}: {g_cnt} trades > 10 BTC, {c_cnt} clusters temporais")

    with open("dados/audit/multi_day_stratified_summary.json", "w") as f:
        # Remover grandes details para json conciso
        clean_summary = {}
        for d, s_dict in summary.items():
            clean_summary[d] = {}
            for s, st in s_dict.items():
                st_copy = {k: v for k, v in st.items() if k != "grandes"}
                clean_summary[d][s] = st_copy
        json.dump(clean_summary, f, indent=2)

if __name__ == "__main__":
    main()
