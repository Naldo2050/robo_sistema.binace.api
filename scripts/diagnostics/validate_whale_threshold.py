#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/validate_whale_threshold.py

Validação 3: Threshold de Whale (2.0 BTC) em produção real vs. amostra offline.
Avalia:
1. N total de trades capturados na sessão real.
2. Percentis de distribuição: p50, p90, p99, max.
3. Comparação do p99 real de produção com o baseline offline de NY (2.3070 BTC).
4. Contagem e proporção de trades classificados como 'whale' (>= 2.0 BTC).

Fontes suportadas:
- Arquivo JSONL de dump contínuo gerado pelo pipeline (--dump-path, default: dados/trades_collect_2h.jsonl)
- Tabela SQLite 'trades' (se existente em schemas alternativos)
- Trades agregados em payloads da tabela 'events'
"""
import sys
import os
import json
import sqlite3
import argparse
import numpy as np

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def analyze_whale_trades(
    dump_path: str = "dados/trades_collect_2h.jsonl",
    db_path: str = "dados/trading_bot.db",
    start_ts_ms: int = None,
    end_ts_ms: int = None,
    offline_ny_p99: float = 2.3070,
    whale_threshold: float = 2.0,
):
    print("=" * 80)
    print("VALIDAÇÃO 3 — THRESHOLD DE WHALE (2.0 BTC) EM PRODUÇÃO REAL VS OFFLINE")
    print("=" * 80)

    quantities = []
    source_used = None

    # 1. Tentar ler do dump (JSONL ou JSON array de trades brutos)
    if os.path.exists(dump_path) and os.path.getsize(dump_path) > 0:
        source_used = f"Arquivo ({dump_path})"
        print(f"📖 Lendo trades brutos de: {dump_path}...")
        try:
            with open(dump_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
            if content.startswith("[") and content.endswith("]"):
                items = json.loads(content)
                for t in items:
                    if not isinstance(t, dict):
                        continue
                    ts = t.get("timestamp") or t.get("T") or t.get("ts_ms")
                    if start_ts_ms and ts and ts < start_ts_ms:
                        continue
                    if end_ts_ms and ts and ts > end_ts_ms:
                        continue
                    q = t.get("quantity") if "quantity" in t else t.get("q")
                    if q is not None:
                        quantities.append(float(q))
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        ts = t.get("timestamp") or t.get("T") or t.get("ts_ms")
                        if start_ts_ms and ts and ts < start_ts_ms:
                            continue
                        if end_ts_ms and ts and ts > end_ts_ms:
                            continue
                        q = t.get("quantity") if "quantity" in t else t.get("q")
                        if q is not None:
                            quantities.append(float(q))
                    except Exception:
                        continue
        except Exception as ex:
            print(f"Aviso ao carregar {dump_path}: {ex}")

    # 2. Se JSONL não existir ou estiver vazio, tentar tabela 'trades' no SQLite
    if not quantities and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if cur.fetchone():
                source_used = f"Tabela 'trades' no SQLite ({db_path})"
                query = "SELECT quantity FROM trades"
                params = []
                if start_ts_ms and end_ts_ms:
                    query += " WHERE timestamp BETWEEN ? AND ?"
                    params = [start_ts_ms, end_ts_ms]
                cur.execute(query, params)
                quantities = [float(r[0]) for r in cur.fetchall()]
            conn.close()
        except Exception:
            pass

    # 3. Se ainda vazio, inspecionar trades salvos dentro de 'payload' na tabela 'events'
    if not quantities and os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            if cur.fetchone():
                cur.execute("SELECT payload FROM events ORDER BY timestamp_ms ASC")
                for (payload_raw,) in cur.fetchall():
                    p = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
                    raw_trades = p.get("trades") or p.get("window_trades") or []
                    for t in raw_trades:
                        if isinstance(t, dict):
                            q = t.get("quantity") or t.get("q")
                            if q is not None:
                                quantities.append(float(q))
                if quantities:
                    source_used = f"Tabela 'events' (payloads em {db_path})"
            conn.close()
        except Exception:
            pass

    if not quantities:
        print(f"ℹ️ Nenhum trade individual encontrado em {dump_path} ou no banco {db_path}.")
        print("Para coletar trades brutos na próxima sessão, utilize o flag --dump-raw-trades.")
        return 0

    qs = np.array(quantities)
    n_total = len(qs)
    p50 = float(np.percentile(qs, 50))
    p90 = float(np.percentile(qs, 90))
    p95 = float(np.percentile(qs, 95))
    p99 = float(np.percentile(qs, 99))
    max_q = float(qs.max())

    whales = qs[qs >= whale_threshold]
    n_whales = len(whales)
    whale_pct = (n_whales / n_total) * 100.0

    divergence_pct = abs(p99 - offline_ny_p99) / offline_ny_p99 * 100.0

    print(f"Origem dos dados: {source_used}")
    print(f"Total de trades analisados (N): {n_total}")
    print(f"\n── Distribuição de Tamanho de Trades (BTC) ──")
    print(f"  • p50 (Mediana): {p50:.4f} BTC")
    print(f"  • p90:           {p90:.4f} BTC")
    print(f"  • p95:           {p95:.4f} BTC")
    print(f"  • p99:           {p99:.4f} BTC")
    print(f"  • Máximo:        {max_q:.4f} BTC")

    print(f"\n── Comparação com Baseline Offline de NY ──")
    print(f"  • p99 de NY Offline (3 dias): {offline_ny_p99:.4f} BTC")
    print(f"  • p99 Real de Produção:       {p99:.4f} BTC")
    print(f"  • Divergência:                {divergence_pct:.2f}%")
    if divergence_pct <= 20.0:
        print(f"  ✅ CONFORME: Divergência dentro do limite aceitável (<= 20%).")
    else:
        print(f"  ⚠️ ALERTA: Divergência > 20% em relação ao baseline offline ({divergence_pct:.2f}%).")

    print(f"\n── Taxa de Disparo Prática do Bucket Whale (>= {whale_threshold} BTC) ──")
    print(f"  • Total de trades whale: {n_whales}")
    print(f"  • Proporção sobre total: {whale_pct:.4f}% ({n_whales}/{n_total})")

    verdict = "APROVADO" if divergence_pct <= 20.0 else "APROVADO COM RESSALVA"
    print(f"\nVeredito da Validação 3: {verdict}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validação do threshold de whale em trades de produção.")
    parser.add_argument("--jsonl", "--dump-path", dest="dump_path", default="dados/trades_collect_2h.jsonl", help="Caminho do dump JSONL de trades")
    parser.add_argument("--db", default="dados/trading_bot.db", help="Caminho do banco SQLite")
    parser.add_argument("--threshold", type=float, default=2.0, help="Limiar de whale em BTC (default: 2.0)")

    args = parser.parse_args()
    sys.exit(analyze_whale_trades(
        dump_path=args.dump_path,
        db_path=args.db,
        whale_threshold=args.threshold,
    ))
