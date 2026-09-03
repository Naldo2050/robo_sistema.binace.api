# scripts/diagnostics/measure_session_vwap_performance.py
# -*- coding: utf-8 -*-
"""
Benchmark e Diagnóstico de Performance para Session VWAP (Fase P1.2).
Mede:
1. Warm-up / Rebuild time via Binance API (klines 1m desde 00:00 UTC).
2. Steady-state O(1) update latency (p50, p95, max).
3. Payload token impact.
"""

import asyncio
import os
import sys
import time
import numpy as np

sys.path.insert(0, ".")

from institutional.session_vwap import SessionVWAPTracker, get_utc_session_start_ms
from market_orchestrator.analysis.institutional_analytics import InstitutionalAnalyticsEngine
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


async def benchmark_session_vwap():
    print("=" * 70)
    print("BENCHMARK DE PERFORMANCE — CANONICAL SESSION VWAP (FASE P1.2)")
    print("=" * 70)

    tracker = SessionVWAPTracker(symbol="BTCUSDT")

    # 1. Warm-up / Rebuild Time
    t0 = time.perf_counter()
    success = await tracker.rebuild_from_binance()
    rebuild_time_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n1. RECONSTRUÇÃO PÓS-RESTART (Binance REST 1m klines desde 00:00 UTC):")
    print(f"   - Sucesso: {success}")
    print(f"   - Barras recuperadas: {tracker._bars_count}")
    print(f"   - Session VWAP atual: ${tracker.current_vwap:,.2f}" if tracker.current_vwap else "   - Session VWAP: N/A")
    print(f"   - Tempo total de rebuild: {rebuild_time_ms:.2f} ms")

    # 2. Steady-State O(1) Update Latency
    latencies_us = []
    now_ms = int(time.time() * 1000)
    for i in range(1000):
        t_start = time.perf_counter()
        tracker.update_candle(
            timestamp_ms=now_ms + i * 60000,
            high=77500.0,
            low=77300.0,
            close=77400.0,
            volume=5.0,
        )
        t_end = time.perf_counter()
        latencies_us.append((t_end - t_start) * 1_000_000.0)

    p50 = np.percentile(latencies_us, 50)
    p95 = np.percentile(latencies_us, 95)
    p99 = np.percentile(latencies_us, 99)
    max_lat = max(latencies_us)

    print(f"\n2. LATÊNCIA DE ATUALIZAÇÃO INCREMENTAL O(1) (1.000 iterações):")
    print(f"   - p50: {p50:.2f} µs ({p50/1000.0:.4f} ms)")
    print(f"   - p95: {p95:.2f} µs ({p95/1000.0:.4f} ms)")
    print(f"   - p99: {p99:.2f} µs ({p99/1000.0:.4f} ms)")
    print(f"   - max: {max_lat:.2f} µs ({max_lat/1000.0:.4f} ms)")

    # 3. Payload Impact
    engine = InstitutionalAnalyticsEngine(symbol="BTCUSDT")
    engine.session_vwap_tracker = tracker
    res = engine.compute_all(current_price=77500.0)

    event = {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 77500.0,
        "institutional_analytics": res,
    }
    compact = build_compact_payload(event)
    groq_summary = AIAnalyzer._build_groq_payload_summary(compact)

    vwap_section = groq_summary.get("vwap", {})
    import json
    vwap_json = json.dumps(vwap_section)
    print(f"\n3. IMPACTO NO TOKEN BUDGET DO LLM:")
    print(f"   - Seção compactada 'vwap': {vwap_json}")
    print(f"   - Tamanho em caracteres: {len(vwap_json)} bytes")
    print(f"   - Custo estimado em tokens: ~{max(1, len(vwap_json)//4)} tokens")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(benchmark_session_vwap())
