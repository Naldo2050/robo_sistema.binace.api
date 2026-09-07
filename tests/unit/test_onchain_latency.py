# tests/unit/test_onchain_latency.py
"""
FASE D: latência determinística do hot path + shutdown limpo + benchmark.

  - Fetch HTTP de 30s no updater NÃO aumenta a latência da janela: a janela
    lê snapshot em tempo pequeno e determinístico (limite generoso contra
    regressão grosseira; distribuição real sai no benchmark abaixo).
  - Erro/timeout de rede não bloqueia a janela.
  - Ausência nunca vira 0 no caminho da janela.
  - Shutdown não deixa thread/session viva.
  - Benchmark: N leituras de snapshot com distribuição (média/p99).
"""

import statistics
import threading
import time

import pytest

from data_processing.data_enricher import DataEnricher
from fetchers.onchain_updater import OnchainUpdater

# Limite generoso anti-regressão grosseira (NÃO é SLO; ver benchmark).
WINDOW_ONCHAIN_BUDGET_S = 5.0


def _raw_event():
    return {
        "preco_fechamento": 79421.2,
        "volume_total": 3.268,
        "symbol": "BTCUSDT",
        "timestamp": "2026-09-07T12:41:05+00:00",
        "ohlc": {"close": 79421.2},
        "multi_tf": {},
    }


def test_slow_fetch_does_not_block_window():
    """Updater travado em fetch de 30s; janela responde em tempo limitado."""
    import fetchers.onchain_fetcher as fetcher_mod

    async def _slow_fetch(self, session=None):
        await __import__("asyncio").sleep(30.0)
        return {"mempool_size": 1}

    orig = fetcher_mod.OnchainFetcher.fetch_all
    fetcher_mod.OnchainFetcher.fetch_all = _slow_fetch
    updater = OnchainUpdater()
    try:
        updater.policy.refresh_interval_s = 3600.0
        updater.start()
        time.sleep(0.5)  # updater entrou no fetch de 30s
        enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
        start = time.perf_counter()
        out = enricher.enrich_from_raw_event(_raw_event())
        elapsed = time.perf_counter() - start
        assert elapsed < WINDOW_ONCHAIN_BUDGET_S, elapsed
        # Sem snapshot ainda: warming_up, sem valores, sem zeros fabricados.
        assert out["onchain_metrics"] == {}
        assert enricher.last_onchain_view["fast"]["status"] == "warming_up"
    finally:
        fetcher_mod.OnchainFetcher.fetch_all = orig
        updater.stop(timeout=35.0)


def test_network_error_does_not_block_window(monkeypatch):
    import fetchers.onchain_fetcher as fetcher_mod

    async def _boom(self, session=None):
        raise TimeoutError("rede onchain fora do ar")

    monkeypatch.setattr(fetcher_mod.OnchainFetcher, "fetch_all", _boom)
    updater = OnchainUpdater()
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    start = time.perf_counter()
    assert enricher._build_onchain_metrics() == {}
    assert time.perf_counter() - start < WINDOW_ONCHAIN_BUDGET_S
    assert updater._refresh_once() is False
    assert updater.last_error is not None
    # Snapshot anterior (inexistente) preservado: segue sem valores.
    assert enricher._build_onchain_metrics() == {}


def test_shutdown_leaves_no_thread_alive():
    updater = OnchainUpdater()
    updater.policy.refresh_interval_s = 3600.0
    before = threading.active_count()
    updater.start()
    time.sleep(0.5)
    assert updater._thread is not None and updater._thread.is_alive()
    updater.stop(timeout=10.0)
    assert updater._thread is None
    time.sleep(0.2)
    assert threading.active_count() <= before


def test_snapshot_read_benchmark(capsys):
    """Benchmark: distribuição de N leituras (NÃO é gate rígido)."""
    updater = OnchainUpdater()
    updater._store_snapshot(
        fast={"mempool_size": 25000}, slow={"difficulty": 145.04}
    )
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    samples = []
    for _ in range(50):
        start = time.perf_counter()
        enricher.enrich_from_raw_event(_raw_event())
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    mean_ms = statistics.mean(samples)
    p99_ms = samples[int(len(samples) * 0.99) - 1]
    print(f"\nBENCH onchain-window: n={len(samples)} "
          f"mean={mean_ms:.2f}ms p50={samples[24]:.2f}ms p99={p99_ms:.2f}ms "
          f"max={samples[-1]:.2f}ms")
    assert max(samples) < WINDOW_ONCHAIN_BUDGET_S * 1000.0
