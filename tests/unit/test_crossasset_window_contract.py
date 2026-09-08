# tests/unit/test_crossasset_window_contract.py
"""
Contrato E3-B (cross-asset fora do hot path).

Exige:
  1. Janela cold faz ZERO chamadas externas cross-asset.
  2. Updater lento (10s) não atrasa a janela.
  3. Cold => warming_up/null, nunca 0 fabricado.
  4. Fresh chega aos consumidores com age/status.
  5. TTL expirado => stale explícito.
  6. Refresh parcial/falho preserva exatamente o último completo.
  7. Leitor nunca vê snapshot parcial.
  8. Múltiplos consumidores não criam múltiplos updaters.
  9. Wall-clock/NTP não afeta age.
 10. Shutdown não deixa recursos vivos.
 11. FeatureStore/dataset preservam freshness suficiente.
 12. Payload final da IA recebe freshness/age.
 13. model_metadata_latest prova que o modelo atual não exige cross-asset.
"""

import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.pipeline import DataPipeline
from market_analysis.cross_asset_updater import (
    CROSS_ASSET_TEMPORAL_AUDIT,
    CrossAssetSnapshot,
    CrossAssetUpdater,
)

REPO = Path(__file__).resolve().parents[2]
METADATA = REPO / "ml" / "models" / "model_metadata_latest.json"

# Limite generoso anti-regressão grosseira (benchmark real reporta ms).
WINDOW_BUDGET_S = 5.0


def _trades(n=60, base=1700000000000):
    return [
        {"p": 79000.0 + (i % 7), "q": 0.5, "T": base + i * 1000,
         "m": i % 2 == 0}
        for i in range(n)
    ]


def _boom(*args, **kwargs):
    raise AssertionError("HTTP/rede cross-asset executado no hot path")


@pytest.fixture
def no_net(monkeypatch):
    import market_analysis.cross_asset_correlations as ca

    monkeypatch.setattr(ca, "_fetch_binance_klines", _boom)
    monkeypatch.setattr(ca, "_fetch_yfinance_data_with_fallbacks", _boom)
    monkeypatch.setattr(ca, "_fetch_yfinance_data", _boom)
    monkeypatch.setattr(ca, "_run_async_safely", _boom)


def _seeded_view(fast_age_s=10.0, slow_age_s=100.0):
    now_mono = time.monotonic()
    wall_ms = int(time.time() * 1000)
    snap = CrossAssetSnapshot(
        result={"status": "ok", "btc_eth_corr_7d": 0.92,
                "btc_dxy_corr_30d": 0.16},
        fetched_at_ms=wall_ms - int(max(fast_age_s, slow_age_s) * 1000),
        fetched_monotonic=now_mono - max(fast_age_s, slow_age_s),
        last_error=None,
    )
    updater = CrossAssetUpdater(monotonic_fn=lambda: now_mono)
    updater._snapshot = snap
    return updater


def test_1_window_cold_zero_external_calls(no_net):
    updater = CrossAssetUpdater()  # nunca iniciado: sem snapshot
    pipe = DataPipeline(_trades(), "BTCUSDT",
                        cross_asset_snapshot=updater.read_view())
    feats = pipe.get_final_features()
    assert feats["ml_features"]["cross_asset"] == {}
    assert feats["ml_features"]["data_quality"]["has_cross_asset"] is False


def test_2_slow_updater_does_not_block_window(no_net):
    import market_analysis.cross_asset_correlations as ca

    orig = ca.get_enhanced_cross_asset_correlations

    def _slow(now_utc=None, stop_event=None):
        time.sleep(10.0)
        return {"status": "ok"}

    ca.get_enhanced_cross_asset_correlations = _slow
    updater = CrossAssetUpdater()
    updater.policy.refresh_interval_s = 3600.0
    try:
        updater.start()
        time.sleep(0.5)  # updater preso no fetch de 10s
        pipe = DataPipeline(_trades(), "BTCUSDT",
                            cross_asset_snapshot=updater.read_view())
        start = time.perf_counter()
        feats = pipe.get_final_features()
        elapsed = time.perf_counter() - start
        assert elapsed < WINDOW_BUDGET_S, elapsed
        assert feats["ml_features"]["cross_asset"] == {}
    finally:
        ca.get_enhanced_cross_asset_correlations = orig
        updater.stop(timeout=15.0)


def test_3_cold_is_warming_up_never_zero(no_net):
    updater = CrossAssetUpdater()
    view = updater.read_view()
    assert view["status"] == "warming_up"
    assert view["values"] == {}
    assert view["age_seconds"] is None
    pipe = DataPipeline(_trades(), "BTCUSDT", cross_asset_snapshot=view)
    ml = pipe.get_final_features()["ml_features"]
    assert ml["cross_asset"] == {}
    assert ml["cross_asset_status"] == "warming_up"
    assert ml["cross_asset_age_seconds"] is None
    assert ml["data_quality"]["has_cross_asset"] is False


def test_4_fresh_reaches_consumers_with_age_status(no_net):
    updater = _seeded_view(10.0, 100.0)
    pipe = DataPipeline(_trades(), "BTCUSDT",
                        cross_asset_snapshot=updater.read_view())
    ml = pipe.get_final_features()["ml_features"]
    assert ml["cross_asset"]["btc_eth_corr_7d"] == 0.92
    assert ml["cross_asset_status"] == "fresh"
    assert ml["cross_asset_age_seconds"] is not None
    assert ml["data_quality"]["has_cross_asset"] is True


def test_5_ttl_expired_is_explicit_stale(no_net):
    # 1000s: além de fresh (300s), dentro de usable (3600s) => stale.
    updater = _seeded_view(1000.0, 1000.0)
    view = updater.read_view()
    assert view["status"] == "stale"
    assert view["age_seconds"] is not None
    pipe = DataPipeline(_trades(), "BTCUSDT", cross_asset_snapshot=view)
    ml = pipe.get_final_features()["ml_features"]
    assert ml["cross_asset_status"] == "stale"
    # Stale_usable: valor pode aparecer, mas marcado.
    assert ml["cross_asset"]["btc_eth_corr_7d"] == 0.92


def test_5b_beyond_usable_is_unavailable_without_values(no_net):
    updater = _seeded_view(100000.0, 100000.0)
    view = updater.read_view()
    assert view["status"] == "unavailable"
    pipe = DataPipeline(_trades(), "BTCUSDT", cross_asset_snapshot=view)
    ml = pipe.get_final_features()["ml_features"]
    assert ml["cross_asset_status"] == "unavailable"
    assert ml["cross_asset"].get("btc_eth_corr_7d") is None


def test_6_partial_failed_preserves_last_complete(no_net, monkeypatch):
    import market_analysis.cross_asset_correlations as ca

    updater = _seeded_view(10.0, 10.0)
    before = updater.read_view()
    monkeypatch.setattr(
        ca, "get_enhanced_cross_asset_correlations",
        lambda now_utc=None, stop_event=None: {"status": "partial", "btc_eth_corr_7d": 0.5},
    )
    assert updater._refresh_once() is False
    after = updater.read_view()
    assert after["values"] == before["values"]
    assert after["fetched_at_ms"] == before["fetched_at_ms"]
    assert updater.last_error is None  # partial não é exceção: só observabilidade
    assert updater.partial_total >= 1


def test_7_reader_never_sees_partial(no_net, monkeypatch):
    import market_analysis.cross_asset_correlations as ca

    calls = {"n": 0}

    def _flip(now_utc=None, stop_event=None):
        calls["n"] += 1
        time.sleep(0.02)  # alarga a janela de corrida p/ os leitores
        if calls["n"] % 2 == 0:
            return {"status": "partial", "btc_eth_corr_7d": 0.1}
        return {"status": "ok", "btc_eth_corr_7d": 0.9,
                "n": calls["n"]}

    monkeypatch.setattr(ca, "get_enhanced_cross_asset_correlations", _flip)
    updater = CrossAssetUpdater()
    seen = []
    stop = threading.Event()

    def _reader():
        while not stop.is_set():
            v = updater.read_view()["values"]
            if v:
                seen.append((v.get("btc_eth_corr_7d"), v.get("n")))

    threads = [threading.Thread(target=_reader) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(6):
        updater._refresh_once()
    stop.set()
    for t in threads:
        t.join()
    assert seen, "leitores deveriam observar snapshots publicados"
    # Cada visão: ou vazia (warming) ou um snapshot completo com n ímpar.
    for corr, n in seen:
        assert corr == 0.9
        assert n is not None and n % 2 == 1


def test_8_multiple_consumers_share_single_updater(no_net):
    updater = _seeded_view()
    threads_before = threading.active_count()
    pipes = [DataPipeline(_trades(), "BTCUSDT",
                          cross_asset_snapshot=updater.read_view())
             for _ in range(3)]
    assert updater._thread is None  # não iniciado: zero threads
    for p in pipes:
        p.get_final_features()
    assert threading.active_count() <= threads_before + 1


def test_9_wall_clock_shift_does_not_change_age(no_net, monkeypatch):
    updater = _seeded_view(10.0, 10.0)
    age_before = updater.read_view()["age_seconds"]
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 86400.0)
    view = updater.read_view()
    assert view["age_seconds"] == pytest.approx(age_before, abs=5.0)
    assert view["status"] == "fresh"


def test_10_shutdown_leaves_nothing_alive(no_net):
    updater = CrossAssetUpdater()
    updater.policy.refresh_interval_s = 3600.0
    before = threading.active_count()
    updater.start()
    time.sleep(0.5)
    assert updater._thread is not None and updater._thread.is_alive()
    updater.stop(timeout=10.0)
    assert updater._thread is None
    time.sleep(0.2)
    assert threading.active_count() <= before
    updater.stop()  # duplo stop não quebra


def test_11_featurestore_dataset_preserve_freshness(no_net, tmp_path):
    from data_processing.feature_store import FeatureStore

    updater = _seeded_view(10.0, 100.0)
    pipe = DataPipeline(_trades(), "BTCUSDT",
                        cross_asset_snapshot=updater.read_view())
    ml = pipe.get_final_features()["ml_features"]
    store = FeatureStore(base_dir=str(tmp_path))
    store.save_features("W1", {**ml["cross_asset"],
                               "cross_asset_status": ml["cross_asset_status"],
                               "cross_asset_age_seconds": ml["cross_asset_age_seconds"]})
    store._flush()
    parts = list(tmp_path.rglob("*.parquet"))
    assert parts, "nenhum parquet gravado"
    df = pd.read_parquet(parts[0])
    assert "cross_asset_age_seconds" in df.columns
    assert "cross_asset_status" in df.columns
    # Dataset collector: numerics passam (age), strings caem (status) por design.
    flat = {}
    for k, v in {**ml["cross_asset"],
                 "cross_asset_status": ml["cross_asset_status"],
                 "cross_asset_age_seconds": ml["cross_asset_age_seconds"]}.items():
        if isinstance(v, (int, float)):
            flat[k] = v
    assert "cross_asset_age_seconds" in flat
    assert "cross_asset_status" not in flat


def test_12_final_payload_receives_freshness(no_net):
    from market_orchestrator.ai.analyzer_qwen import AIAnalyzer
    from market_orchestrator.ai.llm_payload_guardrail import guardrail_rewrap
    from market_orchestrator.ai.payload_builder_compact import (
        build_compact_payload,
    )

    updater = _seeded_view(10.0, 100.0)
    pipe = DataPipeline(_trades(), "BTCUSDT",
                        cross_asset_snapshot=updater.read_view())
    ml = pipe.get_final_features()["ml_features"]
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 79000.0, "epoch_ms": 1700000060000,
             "ml_features": ml}
    compact = build_compact_payload(event)
    assert compact["cross"]["st"] == "fresh"
    assert compact["cross"]["age"] is not None
    rewrapped = guardrail_rewrap(compact)
    assert rewrapped["ai_payload"]["cross"]["st"] == "fresh"
    final = AIAnalyzer._build_groq_payload_summary(rewrapped["ai_payload"])
    assert final["cross"]["st"] == "fresh"


def test_13_model_does_not_require_cross_asset():
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    model_inputs = set(meta["feature_names"])
    cross_keys = {"btc_eth_corr_7d", "btc_eth_corr_30d", "btc_dxy_corr_30d",
                  "btc_dxy_corr_90d", "btc_ndx_corr_30d", "dxy_return_5d",
                  "dxy_return_20d", "vix_current", "us10y_yield", "gold_price",
                  "oil_price", "btc_dominance", "macro_regime"}
    assert model_inputs & cross_keys == set()
    assert CROSS_ASSET_TEMPORAL_AUDIT == "pending"
