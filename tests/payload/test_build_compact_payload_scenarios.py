import json
import copy
import pytest
import build_compact_payload as bcp

HARD_LIMIT = 6144
SOFT_LIMIT = 3000
WARN_LIMIT = 2800

@pytest.fixture(autouse=True)
def reset_static_cache():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0

def make_base_event() -> dict:
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 66875,
        "delta": 0.123,
        "volume_total": 1.5,
        "volume_compra": 0.9,
        "contextual_snapshot": {
            "ohlc": { "open": 66884, "high": 66890, "low": 66856, "close": 66875, "vwap": 66879 }
        },
        "market_context": { "trading_session": "NY", "session_phase": "ACTIVE" },
        "market_environment": { "volatility_regime": "HIGH", "market_structure": "RANGE_BOUND" },
        "historical_vp": { "daily": { "status": "success", "poc": 66444, "val": 66388, "vah": 67771 } },
        "external_markets": { "VIX": {"preco_atual": 23.9}, "FEAR_GREED": {"preco_atual": 50}, "DXY": {"preco_atual": 104.2}, "SP500": {"preco_atual": 669.1} },
        "derivatives": {
            "BTCUSDT": { "long_short_ratio": 1.83, "open_interest": 90101.906, "funding_rate_percent": -0.01, "longs_usd": 3893635557.19, "shorts_usd": 2129881055.29 },
            "ETHUSDT": { "long_short_ratio": 1.66 }
        },
        "ml_features": {
            "cross_asset": { "btc_eth_corr_7d": 0.92, "btc_dxy_corr_30d": -0.41 },
            "microstructure": { "trade_intensity_v2": 12.3, "tick_rule_sum": 10, "order_book_slope": -0.25 }
        },
        "multi_tf": {
            "1h": { "tendencia": "Baixa", "rsi_short": 50, "macd": -151, "adx": 21, "atr": 384, "regime": "Range" },
            "4h": { "tendencia": "Baixa", "rsi_short": 43, "macd": -178, "adx": 27, "atr": 968, "regime": "Manipulação" }
        },
        "fluxo_continuo": {
            "cvd": 0.2,
            "sector_flow": { "whale": {"delta": 0.4}, "retail": {"delta": -0.2} },
            "order_flow": { "net_flow_1m": 16000, "net_flow_5m": 12000, "flow_imbalance": 0.18, "aggressive_buy_pct": 59, "buy_sell_ratio": {"buy_sell_ratio": 1.44} },
            "absorption_analysis": { "current_absorption": { "buyer_strength": 5.9, "seller_exhaustion": 1.8, "continuation_probability": 0.22 } }
        },
        "institutional_analytics": {
            "profile_analysis": { "poor_extremes": { "action_bias": "expect_retest_both", "poor_high": {"detected": 1}, "poor_low": {"detected": 1} } },
            "quality": { "calendar": {"expected_liquidity": "NORMAL"}, "latency": { "latency_ms": 820, "latency_category": "OK" } },
            "flow_analysis": { "passive_aggressive": { "composite": { "signal": "buy_absorption", "conviction": "MEDIUM" } } },
            "sr_analysis": { "defense_zones": { "status": "success", "sell_defense": [{"center": 66931, "strength": 66}], "buy_defense": [{"center": 66839, "strength": 58}] } }
        },
        "technical_indicators_extended": { "cci_signal": "NEUTRAL", "stochastic": {"k": 51} },
        "pattern_recognition": { "smart_money": { "fair_value_gaps": [{"type": "BULLISH"}, {"type": "BEARISH"}], "market_structure": {"structure": "BEARISH"} } },
        "alerts": { "active_alerts": [{"type": "SUPPORT_TEST", "severity": "HIGH", "level": 66839}] },
        "data_reliability": { "latency_acceptable": 1, "onchain_coverage": "full" }
    }

def test_scenario_mean_reversion_range():
    event = make_base_event()
    payload = bcp.build_compact_payload(event)
    assert payload["regime"]["mode"] in {"MR", "RB"}
    assert payload["flow"]["pa"] == "buy_absorp"

def test_scenario_breakout_event():
    event = make_base_event()
    event["tipo_evento"] = "Breakout"
    payload = bcp.build_compact_payload(event)
    assert payload["trigger"] == "BRK"
    assert payload["ctx"].get("cached") is not True

def test_scenario_absorption():
    event = make_base_event()
    event["tipo_evento"] = "Absorção"
    event["institutional_analytics"]["flow_analysis"]["passive_aggressive"]["composite"] = {"signal": "buy_absorption", "conviction": "HIGH"}
    payload = bcp.build_compact_payload(event)
    assert payload["trigger"] == "ABS"
    assert payload["flow"]["pa"] == "buy_absorp"
    assert len(json.dumps(payload)) < SOFT_LIMIT

def test_scenario_derivatives():
    event = make_base_event()
    event["derivatives"]["BTCUSDT"]["long_short_ratio"] = 2.35
    bcp.build_compact_payload(event) # Warmup
    second = bcp.build_compact_payload(event)
    assert second["ctx"]["cached"] is True
    assert second["ctx"]["ses"] is not None
