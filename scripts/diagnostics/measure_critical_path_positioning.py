# scripts/diagnostics/measure_critical_path_positioning.py
# -*- coding: utf-8 -*-
"""
Benchmark de Critical Path e Latência para Coleta de Posicionamento Binance.
Fase P1.1B.
Mede:
1. Latência individual dos 4 endpoints da Binance.
2. Context collection SEM positioning vs COM positioning (cold & warm cache).
3. Tempo de cada corrotina no gather para identificar a corrotina limitante do critical path.
4. Estatísticas p50, p95, p99 e Max.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

import asyncio
import logging
import statistics
import time
import urllib.request
import json
from typing import Dict, List

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


async def benchmark_individual_endpoints(n_samples=5):
    """Mede a latência individual dos 4 endpoints REST da Binance Futures."""
    endpoints = {
        "global_account_ratio": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=5",
        "top_account_ratio": "https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=5",
        "top_position_ratio": "https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=5m&limit=5",
        "open_interest_hist": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=5",
    }
    
    latencies: Dict[str, List[float]] = {k: [] for k in endpoints}
    
    timeout = aiohttp.ClientTimeout(total=5.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(n_samples):
            for name, url in endpoints.items():
                t0 = time.perf_counter()
                try:
                    async with session.get(url) as resp:
                        await resp.read()
                        ms = (time.perf_counter() - t0) * 1000
                        latencies[name].append(ms)
                except Exception as e:
                    logger.warning(f"Erro em {name}: {e}")
            await asyncio.sleep(0.2)
            
    print("\n" + "="*70)
    print("1. LATÊNCIA INDIVIDUAL DOS ENDPOINTS BINANCE (amostras = %d)" % n_samples)
    print("="*70)
    for name, vals in latencies.items():
        if vals:
            p50 = statistics.median(vals)
            p95 = statistics.quantiles(vals, n=20)[18] if len(vals) >= 20 else max(vals)
            print(f"  - {name:<25}: min={min(vals):.1f}ms | p50={p50:.1f}ms | max={max(vals):.1f}ms (n={len(vals)})")


async def benchmark_fetcher_parallel():
    """Mede o tempo do fetcher assíncrono consolidando as 4 requisições em paralelo."""
    from fetchers.binance_positioning_fetcher import BinancePositioningFetcher
    fetcher = BinancePositioningFetcher()
    
    # Cold cache
    t0 = time.perf_counter()
    snap_cold = await fetcher.fetch_positioning("BTCUSDT", force_refresh=True)
    cold_ms = (time.perf_counter() - t0) * 1000
    
    # Warm cache (100 iterações)
    warm_samples = []
    for _ in range(100):
        t0 = time.perf_counter()
        snap_warm = await fetcher.fetch_positioning("BTCUSDT", force_refresh=False)
        warm_samples.append((time.perf_counter() - t0) * 1000)
        
    print("\n" + "="*70)
    print("2. BINANCE POSITIONING FETCHER (PARALELO 4 ENDPOINTS)")
    print("="*70)
    print(f"  - Cold Cache Fetch (4 requests paralelos): {cold_ms:.2f}ms")
    print(f"  - Warm Cache Fetch (p50): {statistics.median(warm_samples)*1000:.2f} µs | max: {max(warm_samples)*1000:.2f} µs")


async def benchmark_gather_critical_path():
    """Mede a duração de cada corrotina no gather do ContextCollector."""
    from fetchers.context_collector import ContextCollector
    collector = ContextCollector(symbol="BTCUSDT")
    
    # Cria uma sessão mock para testar a duração das subtarefas
    async with aiohttp.ClientSession() as session:
        timings: Dict[str, float] = {}
        
        async def timed_task(name, coro):
            t0 = time.perf_counter()
            try:
                res = await coro
            except Exception as e:
                res = e
            timings[name] = (time.perf_counter() - t0) * 1000
            return res

        tasks = {
            "mtf": timed_task("mtf", collector._analyze_mtf_trends(session)),
            "intermarket": timed_task("intermarket", collector._fetch_intermarket_data(session)),
            "external": timed_task("external", collector._fetch_external_markets(session)),
            "derivatives": timed_task("derivatives", collector._fetch_derivatives_data(session)),
            "sentiment": timed_task("sentiment", collector._fetch_onchain_sentiment(session)),
            "positioning": timed_task("positioning", collector._fetch_positioning_context(session)),
            "market_env": timed_task("market_env", collector._calculate_market_environment(session)),
            "pivots": timed_task("pivots", collector._calculate_pivots(session)),
        }
        
        t_gather_start = time.perf_counter()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        total_gather_ms = (time.perf_counter() - t_gather_start) * 1000
        
        print("\n" + "="*70)
        print("3. TEMPO DE CADA CORROTINA NO GATHER & CRITICAL PATH")
        print("="*70)
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        for name, dur in sorted_timings:
            is_bottleneck = (dur == sorted_timings[0][1])
            tag = "  <-- CRITICAL PATH BOTTLENECK" if is_bottleneck else ""
            print(f"  - {name:<20}: {dur:7.1f} ms{tag}")
            
        print(f"\n  Total asyncio.gather time: {total_gather_ms:.1f} ms")


async def main():
    await benchmark_individual_endpoints(n_samples=5)
    await benchmark_fetcher_parallel()
    await benchmark_gather_critical_path()

if __name__ == "__main__":
    asyncio.run(main())
