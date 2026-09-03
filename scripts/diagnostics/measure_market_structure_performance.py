# scripts/diagnostics/measure_market_structure_performance.py
# -*- coding: utf-8 -*-
"""
Benchmark de Performance e Diagnóstico Shadow para Market Structure (BOS & Sweep).
Fase P1.3.

Mede:
1. Latência por execução em janela de candles (p50, p95, max).
2. Impacto no payload compactado do LLM (bytes e tokens).
3. Shadow observation de eventos estruturais.
"""

import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from institutional.market_structure import MarketStructureDetector
from market_orchestrator.analysis.institutional_analytics import InstitutionalAnalyticsEngine
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer


def benchmark_market_structure():
    print("=" * 70)
    print("BENCHMARK DE PERFORMANCE — MARKET STRUCTURE (BOS & SWEEP) P1.3")
    print("=" * 70)

    detector = MarketStructureDetector(left_bars=2, right_bars=2, timeframe="5m")

    # 1. Gera série sintética de 100 candles
    base_ts = 1788300000000
    candles = []
    for i in range(100):
        h = 75000.0 + (math.sin(i * 0.2) * 1500.0) + 50.0
        l = 75000.0 + (math.sin(i * 0.2) * 1500.0) - 50.0
        c = 75000.0 + (math.sin(i * 0.2) * 1500.0)
        o = c - 10.0
        candles.append({"t": base_ts + i * 300000, "o": o, "h": h, "l": l, "c": c, "v": 20.0})

    # 2. Benchmark de Latência (1.000 iterações)
    latencies_us = []
    for _ in range(1000):
        t0 = time.perf_counter()
        res = detector.analyze_candles(candles)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000.0)

    p50 = np.percentile(latencies_us, 50)
    p95 = np.percentile(latencies_us, 95)
    p99 = np.percentile(latencies_us, 99)
    max_lat = max(latencies_us)

    print(f"\n1. LATÊNCIA DE ANÁLISE ESTRUTURAL (100 candles, 1.000 iterações):")
    print(f"   - p50: {p50:.2f} µs ({p50/1000.0:.4f} ms)")
    print(f"   - p95: {p95:.2f} µs ({p95/1000.0:.4f} ms)")
    print(f"   - p99: {p99:.2f} µs ({p99/1000.0:.4f} ms)")
    print(f"   - max: {max_lat:.2f} µs ({max_lat/1000.0:.4f} ms)")
    print(f"   - Swings confirmados encontrados: {res.confirmed_swings_count}")
    print(f"   - Último BOS ativo: {res.active_bos.type.value if res.active_bos else 'Nenhum'}")
    print(f"   - Último Sweep ativo: {res.active_sweep.type.value if res.active_sweep else 'Nenhum'}")

    # 3. Payload & Token Budget Impact
    df = pd.DataFrame(candles)
    engine = InstitutionalAnalyticsEngine(symbol="BTCUSDT")
    inst_res = engine.compute_all(current_price=75000.0, candles_df=df)

    event = {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 75000.0,
        "institutional_analytics": inst_res,
    }
    compact = build_compact_payload(event)
    groq_summary = AIAnalyzer._build_groq_payload_summary(compact)

    ms_section = groq_summary.get("ms", {})
    ms_json = json.dumps(ms_section)
    print(f"\n2. IMPACTO NO TOKEN BUDGET DO LLM:")
    print(f"   - Seção compactada 'ms': {ms_json}")
    print(f"   - Tamanho em caracteres: {len(ms_json)} bytes")
    print(f"   - Custo estimado em tokens: ~{max(1, len(ms_json)//4)} tokens")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    benchmark_market_structure()
