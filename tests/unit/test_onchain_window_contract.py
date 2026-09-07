# tests/unit/test_onchain_window_contract.py
"""
Contrato FASE B (hot path sem rede onchain).

Exige:
  1. Caminho da janela executa ZERO HTTP onchain (fetch_all nunca chamado),
     com snapshot presente, ausente (cold) ou falhando.
  2. Nenhum ThreadPoolExecutor é criado por chamada/janela no DataEnricher.
  3. Cold start => warming_up + {} (sem bloquear, sem HTTP).
  4. Falha no refresh => snapshot completo anterior preservado (all-or-nothing).
  5. NEVER-EVIDENCE (exchange_netflow/whale_transactions/exchange_reserves/sopr)
     nunca aparece como zero numérico (fonte inexistente != valor zero).
  6. Freshness usa monotonic: wall-clock alterado não muda age/status.
  7. Múltiplos DataEnricher compartilham 1 updater (sem threads/sessões extras).
"""

import threading
import time

import pytest

from data_processing.data_enricher import DataEnricher
from fetchers.onchain_updater import (
    NEVER_EVIDENCE_FIELDS,
    OnchainUpdater,
)


def _raise_http(*args, **kwargs):
    raise AssertionError("HTTP onchain executado no hot path da janela")


@pytest.fixture
def no_http(monkeypatch):
    import fetchers.onchain_fetcher as fetcher_mod

    async def _boom(self, session=None):
        raise AssertionError("HTTP onchain executado no hot path da janela")

    monkeypatch.setattr(fetcher_mod.OnchainFetcher, "fetch_all", _boom)


def _seeded_updater():
    updater = OnchainUpdater()
    updater._store_snapshot(
        fast={"mempool_size": 25000, "fees_fastest_sat_vb": 3},
        slow={"difficulty": 145.04, "hash_rate": 1000.0},
    )
    return updater


def _raw_event():
    return {
        "preco_fechamento": 79421.2,
        "volume_total": 3.268,
        "symbol": "BTCUSDT",
        "timestamp": "2026-09-07T12:41:05+00:00",
        "ohlc": {"close": 79421.2},
        "multi_tf": {},
    }


def test_window_path_makes_zero_http(no_http):
    updater = _seeded_updater()
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    out = enricher.enrich_from_raw_event(_raw_event())
    assert out["onchain_metrics"]["mempool_size"] == 25000
    event = {"raw_event": dict(_raw_event())}
    enricher.enrich_event_with_advanced_analysis(event)


def test_no_executor_per_call(no_http, monkeypatch):
    import concurrent.futures

    def _boom(*args, **kwargs):
        raise AssertionError("ThreadPoolExecutor criado no hot path")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _boom)
    updater = _seeded_updater()
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    enricher.enrich_from_raw_event(_raw_event())


def test_cold_start_warming_up_without_http(no_http):
    updater = OnchainUpdater()  # nunca iniciado: sem snapshot
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    assert enricher._build_onchain_metrics() == {}
    assert enricher.last_onchain_view["fast"]["status"] == "warming_up"
    assert enricher.last_onchain_view["slow"]["status"] == "warming_up"


def test_no_updater_means_unavailable_without_http(no_http):
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"})
    assert enricher._build_onchain_metrics() == {}


def test_failed_refresh_preserves_last_complete(no_http, monkeypatch):
    import fetchers.onchain_fetcher as fetcher_mod

    updater = _seeded_updater()
    before = updater.read_view()
    monkeypatch.setattr(
        fetcher_mod.OnchainFetcher, "fetch_all", _raise_http
    )
    assert updater._refresh_once() is False
    after = updater.read_view()
    assert after["fast"]["values"] == before["fast"]["values"]
    assert after["slow"]["values"] == before["slow"]["values"]
    assert updater.last_error is not None


def test_never_evidence_is_not_zero(no_http, monkeypatch):
    import fetchers.onchain_fetcher as fetcher_mod

    async def _fake_fetch_all(self, session=None):
        return {
            "exchange_netflow": 0.0,
            "whale_transactions": 0,
            "exchange_reserves": 0.0,
            "sopr": 0.0,
            "mempool_size": 25000,
            "difficulty": 145.04,
            "is_real_data": True,
        }

    monkeypatch.setattr(
        fetcher_mod.OnchainFetcher, "fetch_all", _fake_fetch_all
    )
    updater = OnchainUpdater()
    assert updater._refresh_once() is True
    view = updater.read_view()
    flat = {**view["fast"]["values"], **view["slow"]["values"]}
    for field in NEVER_EVIDENCE_FIELDS:
        assert flat.get(field) is None, f"{field} presente como valor"
    assert set(view["capabilities"]) >= set(NEVER_EVIDENCE_FIELDS)


def test_wall_clock_shift_does_not_change_freshness(no_http, monkeypatch):
    updater = _seeded_updater()
    age_before = updater.read_view()["fast"]["age_seconds"]
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 86400.0)
    age_after = updater.read_view()["fast"]["age_seconds"]
    assert age_after == pytest.approx(age_before, abs=5.0)
    assert updater.read_view()["fast"]["status"] == "fresh"


def test_multiple_enrichers_share_single_updater(no_http):
    updater = _seeded_updater()
    threads_before = threading.active_count()
    enrichers = [
        DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
        for _ in range(5)
    ]
    assert all(e._onchain_updater is updater for e in enrichers)
    for e in enrichers:
        e.enrich_from_raw_event(_raw_event())
    assert threading.active_count() <= threads_before + 1
    assert updater._thread is None  # updater não iniciado: zero threads
