# tests/benchmark/test_forced_liquidation_benchmark.py
# -*- coding: utf-8 -*-
"""
Benchmark isolado de latência end-to-end do pipeline de liquidação forçada:
dispatch + parse + dedup + aggregate (p50, p95, p99).
"""

import json
import time
import numpy as np

from fetchers.binance_liquidation_stream import BinanceLiquidationListener


# ── P2-D2: BENCHMARK DISPATCH + PARSE + AGGREGATE (p50, p95, p99) ─────────────

def test_benchmark_dispatch_parse_aggregate_percentiles():
    """
    Mede a latência end-to-end de dispatch + parse + dedup + aggregate
    em lote de 2.000 mensagens e reporta p50, p95 e p99 em microssegundos.
    """
    listener = BinanceLiquidationListener(symbol="BTCUSDT")
    latencies_us = []

    for i in range(2000):
        raw_msg = json.dumps({
            "e": "forceOrder",
            "E": 1720000000000 + i * 50,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL" if i % 2 == 0 else "BUY",
                "o": "LIMIT",
                "f": "IOC",
                "q": "0.250",
                "p": "61000.0",
                "ap": "61000.0",
                "X": "FILLED",
                "l": "0.250",
                "z": "0.250",
                "T": 1720000000000 + i * 50,
            }
        })
        t0 = time.perf_counter_ns()
        ev = listener.handle_raw_message(raw_msg)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)
        assert ev is not None

    arr = np.array(latencies_us)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    print("\n[BENCHMARK END-TO-END] 2000 events dispatch+parse+aggregate:")
    print(f"  p50: {p50:.2f} µs")
    print(f"  p95: {p95:.2f} µs")
    print(f"  p99: {p99:.2f} µs")

    assert p50 < 100.0
    assert p99 < 500.0
    assert len(listener.aggregator._events) == 2000
