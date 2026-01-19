import json

import pytest

from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.ai_payload_builder import build_ai_input
from market_orchestrator.ai.payload_section_cache import canonical_ref

pytestmark = pytest.mark.payload


def _dummy_payload():
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000000000,
        "signal_metadata": {
            "type": "TEST",
            "battle_result": "OK",
            "severity": "INFO",
            "window_id": 1,
            "description": "dummy",
        },
        "price_context": {
            "current_price": 50000,
            "ohlc": {"open": 1, "high": 2, "low": 1, "close": 2},
        },
        "flow_context": {"order_flow": {}, "liquidity_heatmap": {"clusters": list(range(10))}},
        "orderbook_context": {"imbalance": 0.1},
        "technical_indicators": {"rsi": 50},
        "cross_asset_context": {"btc_eth_correlations": {}},
        "macro_context": {"session": "us"},
        "historical_stats": {"long_prob": 0.3},
        "raw_event": {"should_remove": True},
        "observability": {"debug": True},
        "historical_vp": {"daily": {}},
    }


def test_compress_payload_limits_size_and_sets_version():
    payload = _dummy_payload()
    compressed = compress_payload(payload, max_bytes=512)

    assert compressed["_v"] == 2
    assert "raw_event" not in compressed
    assert "observability" not in compressed
    assert "historical_vp" not in compressed
    size_bytes = len(json.dumps(compressed, ensure_ascii=False).encode("utf-8"))
    assert size_bytes <= 512
    assert len(compressed["flow_context"]["liquidity_heatmap"]["clusters"]) <= 3


def test_compress_payload_keeps_signal_fields():
    payload = _dummy_payload()
    compressed = compress_payload(payload, max_bytes=4096)
    assert compressed["symbol"] == "BTCUSDT"
    assert compressed["signal_metadata"]["type"] == "TEST"
    assert compressed["price_context"]["current_price"] == 50000
    assert "epoch_ms" in compressed


def test_builder_fallback_when_missing_price():
    symbol = "BTCUSDT"
    signal = {"tipo_evento": "X", "descricao": "y"}  # sem preco_fechamento
    payload = build_ai_input(
        symbol=symbol,
        signal=signal,
        enriched={},
        flow_metrics={},
        historical_profile={},
        macro_context={},
        market_environment={},
        orderbook_data={},
        ml_features={},
    )
    assert "_v" not in payload  # fallback para v1


def test_builder_generates_v2_with_decision_hash():
    symbol = "BTCUSDT"
    signal = {"tipo_evento": "X", "descricao": "y", "preco_fechamento": 1.0}
    payload = build_ai_input(
        symbol=symbol,
        signal=signal,
        enriched={"ohlc": {"open": 1, "high": 2, "low": 1, "close": 2}},
        flow_metrics={},
        historical_profile={},
        macro_context={},
        market_environment={},
        orderbook_data={},
        ml_features={},
    )
    assert payload.get("_v") == 2
    assert "decision_features_hash" in payload


def test_cluster_time_normalization_from_timestamps():
    payload = _dummy_payload()
    payload["epoch_ms"] = 2500
    payload["flow_context"]["liquidity_heatmap"] = {
        "clusters": [
            {
                "first_seen_ms": 1000,
                "last_seen_ms": 2000,
                "age_ms": 123,  # will be recomputed
                "cluster_duration_ms": 0,
            }
        ]
    }
    compressed = compress_payload(payload, max_bytes=2048)
    cluster = compressed["flow_context"]["liquidity_heatmap"]["clusters"][0]
    assert cluster["age_ms"] == 500
    assert cluster["cluster_duration_ms"] == 1000
    for redundant in ["first_seen_ms", "last_seen_ms", "recent_ts_ms", "recent_timestamp"]:
        assert redundant not in cluster


def test_cluster_time_preserves_existing_when_no_timestamps():
    payload = _dummy_payload()
    payload["flow_context"]["liquidity_heatmap"] = {
        "clusters": [
            {
                "age_ms": 289,
                "cluster_duration_ms": 77,
            }
        ]
    }
    compressed = compress_payload(payload, max_bytes=2048)
    cluster = compressed["flow_context"]["liquidity_heatmap"]["clusters"][0]
    assert cluster["age_ms"] == 289
    assert cluster["cluster_duration_ms"] == 77


def test_budget_enforcement_reduces_sections():
    payload = _dummy_payload()
    # Deixa heatmap e orderbook grandes para forçar reduções
    payload["flow_context"]["liquidity_heatmap"]["clusters"] = [
        {"price": 100 + i, "liquidity": 1000 + i, "volume": 5, "age_ms": 10, "note": "x" * 150}
        for i in range(5)
    ]
    payload["orderbook_context"] = {
        "imbalance": 0.5,
        "depth_metrics": {
            "bid_liquidity_top5": list(range(200)),
            "ask_liquidity_top5": list(range(200)),
            "depth_imbalance": list(range(100)),
        },
        "market_impact_score": 10,
        "walls_detected": True,
    }
    payload["macro_context"]["correlations"] = {"sp500": "x" * 200, "dxy": -0.2}
    payload["macro_context"]["multi_timeframe_trends"] = {"1h": {"rsi": 70, "extra": "y" * 200}}

    compressed = compress_payload(payload, max_bytes=1200)
    size_bytes = len(json.dumps(compressed, ensure_ascii=False).encode("utf-8"))
    assert size_bytes <= 1200

    clusters = compressed["flow_context"]["liquidity_heatmap"]["clusters"]
    assert len(clusters) <= 1  # Orçamento força poda agressiva

    depth_metrics = compressed["orderbook_context"]["depth_metrics"]
    assert isinstance(depth_metrics, dict)
    # Seções devem respeitar o orçamento estimado
    scale = 1200 / 6144
    budgets = {
        "liquidity_heatmap": max(256, int(1800 * scale)),
        "orderbook_context": max(256, int(1400 * scale)),
    }
    heatmap_bytes = len(json.dumps(compressed["flow_context"]["liquidity_heatmap"]).encode("utf-8"))
    orderbook_bytes = len(json.dumps(compressed["orderbook_context"]).encode("utf-8"))
    assert heatmap_bytes <= budgets["liquidity_heatmap"]
    assert orderbook_bytes <= budgets["orderbook_context"]

    assert "correlations" not in compressed.get("macro_context", {})


def test_cache_hit_reduces_payload(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setenv("PAYLOAD_SECTION_CACHE_PATH", str(cache_path))

    signal = {"tipo_evento": "X", "descricao": "y", "preco_fechamento": 1.0, "epoch_ms": 1000}
    payload1 = build_ai_input(
        symbol="BTCUSDT",
        signal=signal,
        enriched={"ohlc": {"open": 1, "high": 2, "low": 1, "close": 2}},
        flow_metrics={},
        historical_profile={},
        macro_context={"session": "us", "correlations": {"sp500": 0.1}},
        market_environment={},
        orderbook_data={},
        ml_features={},
    )
    size1 = len(json.dumps(payload1, ensure_ascii=False).encode("utf-8"))

    signal2 = dict(signal)
    signal2["epoch_ms"] = signal["epoch_ms"] + 1000
    payload2 = build_ai_input(
        symbol="BTCUSDT",
        signal=signal2,
        enriched={"ohlc": {"open": 1, "high": 2, "low": 1, "close": 2}},
        flow_metrics={},
        historical_profile={},
        macro_context={"session": "us", "correlations": {"sp500": 0.1}},
        market_environment={},
        orderbook_data={},
        ml_features={},
    )
    size2 = len(json.dumps(payload2, ensure_ascii=False).encode("utf-8"))

    macro_section = payload2.get("macro_context", {})
    assert macro_section.get("present") is False
    assert "data" not in macro_section
    assert size2 < size1


def test_cache_miss_on_change(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setenv("PAYLOAD_SECTION_CACHE_PATH", str(cache_path))

    base_macro = {"trading_session": "us", "correlations": {"sp500": 0.1}}
    signal = {"tipo_evento": "X", "descricao": "y", "preco_fechamento": 1.0, "epoch_ms": 1000}
    payload1 = build_ai_input(
        symbol="BTCUSDT",
        signal=signal,
        enriched={"ohlc": {"open": 1, "high": 2, "low": 1, "close": 2}},
        flow_metrics={},
        historical_profile={},
        macro_context=base_macro,
        market_environment={},
        orderbook_data={},
        ml_features={},
    )
    ref1 = payload1["macro_context"]["ref"]

    updated_macro = {"trading_session": "eu", "correlations": {"sp500": 0.2}}
    signal2 = dict(signal)
    signal2["epoch_ms"] = signal["epoch_ms"] + 2000
    payload2 = build_ai_input(
        symbol="BTCUSDT",
        signal=signal2,
        enriched={"ohlc": {"open": 1, "high": 2, "low": 1, "close": 2}},
        flow_metrics={},
        historical_profile={},
        macro_context=updated_macro,
        market_environment={},
        orderbook_data={},
        ml_features={},
    )

    macro_section = payload2.get("macro_context", {})
    assert macro_section.get("present") is True
    assert "data" in macro_section
    assert macro_section["ref"] != ref1
