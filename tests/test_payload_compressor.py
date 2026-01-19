import json

import pytest

from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.ai_payload_builder import build_ai_input


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
