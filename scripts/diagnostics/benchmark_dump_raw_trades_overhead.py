#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/benchmark_dump_raw_trades_overhead.py

Benchmark de estresse para avaliar overhead de latência no processamento
de trades WebSocket com e sem a persistência de trades brutos (--dump-raw-trades).
Simula fluxo de alta frequência (1.000 trades/segundo durante 10 segundos = 10.000 trades).
"""
import sys
import os
import time
import json
import tempfile
from pathlib import Path
import numpy as np

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def simulate_trade_processing(dump_mode="none", total_trades=10000, target_rate=1000.0):
    """
    Simula o processamento de trades no caminho crítico do WebSocket.
    dump_mode:
      - 'none': Sem persistência
      - 'sync_flush': flush() síncrono a cada trade
      - 'buffered_flush': buffer em memória com flush a cada 200 trades ou 1s
    """
    tmp_file = None
    f = None
    if dump_mode != "none":
        tmp_dir = tempfile.mkdtemp()
        tmp_file = Path(tmp_dir) / f"bench_{dump_mode}.jsonl"
        f = open(tmp_file, "w", encoding="utf-8")

    latencies_us = []
    unflushed = 0
    last_flush = time.perf_counter()
    flush_interval_sec = 1.0
    flush_batch_size = 200

    base_ts = int(time.time() * 1000)

    try:
        for i in range(total_trades):
            trade_id = 1000000 + i
            timestamp = base_ts + i
            price = 65000.0 + (i % 50) * 0.5
            quantity = 0.05 + (i % 10) * 0.1
            is_buyer_maker = (i % 2 == 0)
            stream_source = "fut_agg"

            # Medição do caminho de gravação
            t0 = time.perf_counter()

            if dump_mode == "none":
                # Apenas processamento mínimo simulado em memória
                _ = (price, quantity)
            elif dump_mode == "sync_flush":
                raw_record = {
                    "trade_id": trade_id,
                    "timestamp": timestamp,
                    "price": price,
                    "quantity": quantity,
                    "is_buyer_maker": is_buyer_maker,
                    "source": stream_source,
                }
                f.write(json.dumps(raw_record) + "\n")
                f.flush()
            elif dump_mode == "buffered_flush":
                raw_record = {
                    "trade_id": trade_id,
                    "timestamp": timestamp,
                    "price": price,
                    "quantity": quantity,
                    "is_buyer_maker": is_buyer_maker,
                    "source": stream_source,
                }
                f.write(json.dumps(raw_record) + "\n")
                unflushed += 1
                now = time.perf_counter()
                if unflushed >= flush_batch_size or (now - last_flush) >= flush_interval_sec:
                    f.flush()
                    unflushed = 0
                    last_flush = now

            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000.0)  # microssegundos
    finally:
        if f:
            f.flush()
            f.close()
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
                tmp_file.parent.rmdir()
            except Exception:
                pass

    arr = np.array(latencies_us)
    return {
        "count": len(arr),
        "mean_us": float(np.mean(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "p90_us": float(np.percentile(arr, 90)),
        "p99_us": float(np.percentile(arr, 99)),
        "max_us": float(np.max(arr)),
        "mean_ms": float(np.mean(arr) / 1000.0),
        "p99_ms": float(np.percentile(arr, 99) / 1000.0),
    }


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK DE OVERHEAD DE LATÊNCIA: DUMP DE TRADES BRUTOS")
    print("Simulação: 10.000 trades a ~1.000 trades/segundo")
    print("=" * 80)

    # 1. SEM persistência
    res_none = simulate_trade_processing("none")
    print(f"1. Sem dump (baseline):")
    print(f"   Média: {res_none['mean_us']:.2f} µs ({res_none['mean_ms']:.4f} ms) | p99: {res_none['p99_us']:.2f} µs ({res_none['p99_ms']:.4f} ms)")

    # 2. COM dump síncrono (flush por trade)
    res_sync = simulate_trade_processing("sync_flush")
    print(f"\n2. Com dump síncrono (flush a cada trade):")
    print(f"   Média: {res_sync['mean_us']:.2f} µs ({res_sync['mean_ms']:.4f} ms) | p99: {res_sync['p99_us']:.2f} µs ({res_sync['p99_ms']:.4f} ms)")

    # 3. COM dump bufferizado (flush periódico / em lote)
    res_buf = simulate_trade_processing("buffered_flush")
    print(f"\n3. Com dump bufferizado (flush a cada 200 trades ou 1s):")
    print(f"   Média: {res_buf['mean_us']:.2f} µs ({res_buf['mean_ms']:.4f} ms) | p99: {res_buf['p99_us']:.2f} µs ({res_buf['p99_ms']:.4f} ms)")

    print("\n" + "=" * 80)
    diff_sync_ms = res_sync['mean_ms'] - res_none['mean_ms']
    diff_buf_ms = res_buf['mean_ms'] - res_none['mean_ms']
    print(f"Diferença síncrona vs baseline: +{diff_sync_ms:.4f} ms por trade")
    print(f"Diferença bufferizada vs baseline: +{diff_buf_ms:.4f} ms por trade")
    print(f"Ganho de performance da bufferização: {res_sync['mean_us'] / max(res_buf['mean_us'], 0.001):.1f}x mais rápida")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
