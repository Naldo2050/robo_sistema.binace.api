import pytest

from build_compact_payload import build_compact_payload
from common.ai_payload_types import (
    COMPACT_AI_ALLOWED_ROOT_KEYS,
    compact_primary_section_count,
    is_compact_ai_payload,
    is_wrapped_compact_ai_payload,
)


pytestmark = pytest.mark.payload


def _make_event() -> dict:
    return {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "epoch_ms": 1775300000000,
        "preco_fechamento": 73250.0,
        "contextual_snapshot": {
            "ohlc": {
                "open": 73100.0,
                "high": 73400.0,
                "low": 72900.0,
                "close": 73250.0,
                "vwap": 73175.0,
            },
        },
        "market_environment": {
            "volatility_regime": "MEDIUM",
            "trend_direction": "UP",
        },
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 280000,
                "net_flow_5m": 316000,
                "net_flow_15m": 450000,
                "flow_imbalance": 0.55,
                "aggressive_buy_pct": 78,
                "buy_sell_ratio": {"buy_sell_ratio": 3.45},
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 1599170,
            "ask_depth_usd": 1200000,
            "imbalance": 0.14,
        },
        "order_book_depth": {
            "L5": {"imbalance": 0.22},
        },
        "multi_tf": {
            "15m": {
                "tendencia": "Alta",
                "rsi_short": 62,
                "macd": 15,
                "macd_signal": 10,
                "adx": 28,
                "atr": 150,
                "regime": "Tendência",
            },
        },
    }


def test_compact_payload_helper_accepts_build_output():
    payload = build_compact_payload(_make_event())

    assert compact_primary_section_count(payload) >= 5
    assert is_compact_ai_payload(
        payload,
        require_identity=True,
        minimum_primary_sections=3,
    )


def test_compact_payload_helper_rejects_minimal_emergency_shape():
    payload = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1,
        "trigger": "EMERGENCY",
        "price": {"c": 100.0},
    }

    assert compact_primary_section_count(payload) == 1
    assert not is_compact_ai_payload(
        payload,
        require_identity=True,
        minimum_primary_sections=3,
    )


def test_compact_payload_wrapper_detection():
    payload = {
        "ai_payload": {
            "symbol": "BTCUSDT",
            "epoch_ms": 1,
            "price": {"c": 100.0},
            "flow": {"d1": "+10K"},
            "ob": {"imb": 0.1},
            "tf": {"1m": {"rsi": 50}},
        }
    }

    assert is_wrapped_compact_ai_payload(payload) is True


def test_allowed_root_keys_include_gap_and_summary_sections():
    for key in ("summary", "ofi", "vwap", "liq", "sm", "cvd_div", "mr", "iceberg"):
        assert key in COMPACT_AI_ALLOWED_ROOT_KEYS
