import copy
import pytest

import build_compact_payload as bcp


@pytest.fixture(autouse=True)
def reset_static_cache():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0


def make_event() -> dict:
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "preco_fechamento": 66875,
        "contextual_snapshot": {
            "ohlc": {
                "open": 66884,
                "high": 66884,
                "low": 66856,
                "close": 66875,
                "vwap": 66879,
            }
        },
        "market_context": {
            "trading_session": "NY",
            "session_phase": "ACTIVE",
        },
        "market_environment": {
            "volatility_regime": "HIGH",
            "market_structure": "RANGE_BOUND",
        },
        "external_markets": {
            "VIX": {"preco_atual": 23.9},
            "FEAR_GREED": {"preco_atual": 50},
        },
        "multi_tf": {
            "15m": {"tendencia": "Baixa", "rsi_short": 45, "regime": "Range"},
            "1h": {"tendencia": "Baixa", "rsi_short": 50, "regime": "Range"},
            "4h": {"tendencia": "Baixa", "rsi_short": 43, "regime": "Manipulação"},
            "1d": {"tendencia": "Baixa", "rsi_short": 40, "regime": "Acumulação"},
        },
        "fluxo_continuo": {
            "order_flow": {
                "net_flow_1m": 16000,
                "flow_imbalance": 0.18,
            }
        },
        "orderbook_data": {
            "bid_depth_usd": 303000,
            "ask_depth_usd": 1700000,
            "imbalance": -0.7,
        },
        "institutional_analytics": {
            "profile_analysis": {
                "profile_shape": {"shape": "b"},
                "poor_extremes": {
                    "action_bias": "expect_retest_both",
                    "poor_high": {"detected": 1},
                    "poor_low": {
                        "detected": 1,
                        "volume_ratio": 5.145,
                    },
                },
                "va_volume_pct": {
                    "breakout_risk": "VERY_HIGH",
                    "compression_signal": 1,
                },
            },
            "quality": {
                "calendar": {
                    "expected_liquidity": "REDUCED",
                },
                "latency": {
                    "latency_ms": 3622,
                    "latency_category": "DEGRADED",
                },
            },
        },
        "technical_indicators_extended": {
            "cci_signal": "NEUTRAL",
            "stochastic": {"k": 51},
            "williams_r": {"value": -58},
        },
        "pattern_recognition": {
            "fibonacci_levels": {
                "high": 67100,
                "low": 66731.08,
                "38.2": 66872.0,
                "61.8": 66959.07,
            },
            "smart_money": {
                "fair_value_gaps": [
                    {"type": "BULLISH"},
                    {"type": "BEARISH"},
                ],
                "market_structure": {
                    "structure": "BEARISH",
                    "bos_detected": 0,
                },
            },
        },
        "historical_vp": {
            "daily": {
                "status": "success",
                "poc": 66444,
                "val": 66388,
                "vah": 67771,
            }
        },
        "derivatives": {
            "BTCUSDT": {
                "long_short_ratio": 1.83,
                "open_interest": 90101.906,
                "funding_rate_percent": -0.01,
            }
        },
        "ml_features": {
            "cross_asset": {
                "btc_dxy_corr_30d": -0.41,
            }
        },
    }


def test_quality_should_be_built_from_institutional_analytics_quality():
    payload = bcp.build_compact_payload(make_event())

    assert payload["qual"] == {
        "lat": "DEGR",
        "liq": "REDUCED",
        "ms": 3622,
    }


def test_poor_low_detected_should_set_pl_flag():
    payload = bcp.build_compact_payload(make_event())

    assert payload["price"]["pl"] == 1


def test_fvg_last_should_distinguish_bullish_from_bearish():
    payload = bcp.build_compact_payload(make_event())

    assert payload["ext"]["smc"]["fvg_last"] == "BE"


def test_fib_should_be_built_from_pattern_recognition_fibonacci_levels():
    event = make_event()
    event.pop("technical_indicators_extended", None)

    payload = bcp.build_compact_payload(event)

    assert payload["ext"]["fib"] == {
        "hi": 67100,
        "lo": 66731,
        "382": 66872,
        "618": 66959,
    }


def test_regime_should_expose_market_mode_or_structure():
    payload = bcp.build_compact_payload(make_event())

    assert payload["regime"]["mode"] in {"MR", "TRD", "BRK", "RB"}
