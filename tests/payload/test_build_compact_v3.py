"""
Tests for build_compact_payload.py v3.

Verifica as 6 correções da PRIMEIRA ETAPA:
1. Mercados externos (DXY, SP500, GOLD, etc.) presentes no ctx
2. compact_number() com sinal explícito (+280K, -34K)
3. Flow keys renomeadas: n1→d1, n5→d5, n15→d15
4. ctx forçado para eventos importantes (Absorção, Exaustão)
5. TFs nulos filtrados
6. Whale = apenas score numérico
"""

import sys
import os
import time

# Adicionar raiz do projeto ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from build_compact_payload import (
    build_compact_payload,
    compact_number,
    _safe_price,
    _safe_int,
    _build_static_context,
    _build_flow,
    _build_price,
    _build_regime,
    _build_orderbook,
    _build_whale,
    _build_timeframes,
    IMPORTANT_EVENTS,
    _should_send_static,
    _last_static_ts,
)

# Reset static cache between tests
import build_compact_payload as bcp


def _reset_static_cache():
    """Reset static context cache."""
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0


# ============================================================
# FIXTURE: event_data realista
# ============================================================

def _make_event_data(**overrides):
    """Cria event_data completo para testes."""
    data = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "janela_numero": 5,
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
            "risk_sentiment": "BULLISH",
        },
        "fluxo_continuo": {
            "cvd": 2.5,
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
        "institutional_analytics": {
            "flow_analysis": {
                "whale_accumulation": {"score": 35},
            },
            "profile_analysis": {
                "profile_shape": {"shape": "P"},
                "poor_extremes": {
                    "action_bias": "buy_side",
                    "poor_high": {"detected": True},
                    "poor_low": {"detected": False, "volume_ratio": 0.8},
                },
            },
        },
        "external_markets": {
            "DXY": {"preco_atual": 119.49},
            "TNX": {"preco_atual": 4.21},
            "SP500": {"preco_atual": 662.97},
            "NASDAQ": {"preco_atual": 593.33},
            "GOLD": {"preco_atual": 5051.30},
            "WTI": {"preco_atual": 98.07},
            "VIX": {"preco_atual": 27.41},
            "FEAR_GREED": {"preco_atual": 15},
        },
        "market_context": {
            "trading_session": "NY_OVERLAP",
            "session_phase": "ACTIVE",
        },
        "historical_vp": {
            "daily": {"poc": 73025, "val": 71569, "vah": 73895},
        },
        "derivatives": {
            "BTCUSDT": {"long_short_ratio": 0.99, "open_interest": 85777},
            "ETHUSDT": {"long_short_ratio": 1.36},
        },
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": 0.9,
                "btc_dxy_corr_30d": 0.23,
            },
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
            "1h": {
                "tendencia": "Alta",
                "rsi_short": 58,
                "macd": 45,
                "macd_signal": 30,
                "adx": 32,
                "atr": 300,
                "regime": "Acumulação",
            },
            "4h": {
                "tendencia": "NE",
                "rsi_short": None,
            },
        },
    }
    data.update(overrides)
    return data


# ============================================================
# TEST 1: compact_number() com sinal explícito
# ============================================================

class TestCompactNumber:
    def test_positive_large(self):
        assert compact_number(280000) == "+280K"

    def test_negative_large(self):
        assert compact_number(-34000) == "-34K"

    def test_positive_millions(self):
        assert compact_number(1599170) == "+1.6M"

    def test_negative_millions(self):
        assert compact_number(-4593927) == "-4.6M"

    def test_positive_small(self):
        assert compact_number(0.55) == "+0.55"

    def test_zero(self):
        assert compact_number(0) == "0"

    def test_none(self):
        assert compact_number(None) == "0"

    def test_negative_small(self):
        assert compact_number(-0.33) == "-0.33"

    def test_no_force_sign_positive(self):
        result = compact_number(280000, force_sign=False)
        assert result == "280K"

    def test_no_force_sign_negative(self):
        result = compact_number(-34000, force_sign=False)
        assert result == "-34K"

    def test_integer_range(self):
        assert compact_number(500) == "+500"

    def test_negative_integer_range(self):
        assert compact_number(-42) == "-42"


# ============================================================
# TEST 2: Flow keys renomeadas d1/d5/d15 (não n1/n5/n15)
# ============================================================

class TestFlowKeys:
    def test_flow_uses_d_keys(self):
        event = _make_event_data()
        flow = _build_flow(event)
        assert "d1" in flow, f"Missing 'd1' in flow: {flow}"
        assert "d5" in flow, f"Missing 'd5' in flow: {flow}"
        assert "d15" in flow, f"Missing 'd15' in flow: {flow}"
        assert "n1" not in flow, f"Old key 'n1' still in flow: {flow}"
        assert "n5" not in flow, f"Old key 'n5' still in flow: {flow}"

    def test_flow_d1_has_sign(self):
        event = _make_event_data()
        flow = _build_flow(event)
        assert flow["d1"].startswith("+") or flow["d1"].startswith("-"), \
            f"d1 should have explicit sign: {flow['d1']}"

    def test_flow_d1_value(self):
        event = _make_event_data()
        flow = _build_flow(event)
        assert flow["d1"] == "+280K"

    def test_flow_negative_delta(self):
        event = _make_event_data()
        event["fluxo_continuo"]["order_flow"]["net_flow_1m"] = -34000
        flow = _build_flow(event)
        assert flow["d1"] == "-34K"


# ============================================================
# TEST 3: Mercados externos no ctx
# ============================================================

class TestExternalMarkets:
    def test_ctx_has_external_prices(self):
        event = _make_event_data()
        ctx = _build_static_context(event)

        assert "dxy" in ctx, f"Missing DXY price in ctx: {ctx}"
        assert "tnx" in ctx, f"Missing TNX price in ctx: {ctx}"
        assert "spx" in ctx, f"Missing SP500 price in ctx: {ctx}"
        assert "ndx" in ctx, f"Missing NASDAQ price in ctx: {ctx}"
        assert "gold" in ctx, f"Missing GOLD price in ctx: {ctx}"
        assert "wti" in ctx, f"Missing WTI price in ctx: {ctx}"
        assert "vix" in ctx, f"Missing VIX price in ctx: {ctx}"
        assert "fg" in ctx, f"Missing Fear&Greed in ctx: {ctx}"

    def test_ctx_prices_correct(self):
        event = _make_event_data()
        ctx = _build_static_context(event)

        assert ctx["dxy"] == 119.49
        assert ctx["tnx"] == 4.21
        assert ctx["spx"] == 662.97
        assert ctx["ndx"] == 593.33
        assert ctx["gold"] == 5051
        assert ctx["wti"] == 98.07
        assert ctx["vix"] == 27.4  # 1 decimal
        assert ctx["fg"] == 15

    def test_ctx_has_derivatives(self):
        event = _make_event_data()
        ctx = _build_static_context(event)

        assert "lsr" in ctx, f"Missing BTC LSR in ctx: {ctx}"
        assert ctx["lsr"] == 0.99
        assert "eth_lsr" in ctx, f"Missing ETH LSR in ctx: {ctx}"
        assert ctx["eth_lsr"] == 1.36
        assert "oi" in ctx, f"Missing OI in ctx: {ctx}"
        assert ctx["oi"] == 86  # 85777 / 1000 rounded

    def test_ctx_has_correlations(self):
        event = _make_event_data()
        ctx = _build_static_context(event)

        assert "eth7" in ctx
        assert ctx["eth7"] == 0.9
        assert "dxy30" in ctx
        assert ctx["dxy30"] == 0.23

    def test_ctx_has_volume_profile(self):
        event = _make_event_data()
        ctx = _build_static_context(event)

        assert ctx["poc"] == 73025
        assert ctx["val"] == 71569
        assert ctx["vah"] == 73895

    def test_ctx_has_session(self):
        event = _make_event_data()
        ctx = _build_static_context(event)
        assert ctx["ses"] == "NY_OVL"

    def test_ctx_missing_external_markets(self):
        """Se external_markets vazio, ctx não deve ter preços mas não crashar."""
        event = _make_event_data(external_markets={})
        ctx = _build_static_context(event)
        assert "dxy" not in ctx
        assert "spx" not in ctx
        # Sessão ainda deve estar presente
        assert "ses" in ctx


# ============================================================
# TEST 4: Forçar ctx para eventos importantes
# ============================================================

class TestForceCtx:
    def test_important_event_forces_ctx(self):
        _reset_static_cache()
        event = _make_event_data(tipo_evento="ANALYSIS_TRIGGER")
        # Primeira chamada — sempre envia ctx
        result1 = build_compact_payload(event)
        assert "ctx" in result1

        # Segunda chamada imediata — ctx cacheado
        result2 = build_compact_payload(event)
        assert "ctx" not in result2, "ctx should be CACHED on second AT call"

        # Evento importante — ctx forçado
        event_abs = _make_event_data(tipo_evento="Absorção")
        result3 = build_compact_payload(event_abs)
        assert "ctx" in result3, "ctx should be FORCED for Absorção"
        assert result3["t"] == "ABS"

    def test_important_events_set(self):
        assert "Absorção" in IMPORTANT_EVENTS
        assert "Exaustão" in IMPORTANT_EVENTS
        assert "Breakout" in IMPORTANT_EVENTS
        assert "ABS" in IMPORTANT_EVENTS
        assert "EXH" in IMPORTANT_EVENTS
        assert "BRK" in IMPORTANT_EVENTS


# ============================================================
# TEST 5: TFs nulos filtrados
# ============================================================

class TestTimeframes:
    def test_null_tf_filtered(self):
        event = _make_event_data()
        tf = _build_timeframes(event)
        # 4h has tendencia=NE, rsi=None — should be filtered
        assert "4h" not in tf, f"4h should be filtered: {tf}"
        assert "15m" in tf
        assert "1h" in tf

    def test_tf_has_correct_data(self):
        event = _make_event_data()
        tf = _build_timeframes(event)
        assert tf["15m"]["t"] == "UP"
        assert tf["15m"]["rsi"] == 62
        assert tf["15m"]["macd"] == [15, 10]
        assert tf["15m"]["adx"] == 28


# ============================================================
# TEST 6: Whale = apenas score numérico
# ============================================================

class TestWhale:
    def test_whale_is_int(self):
        event = _make_event_data()
        whale = _build_whale(event)
        assert isinstance(whale, int)
        assert whale == 35

    def test_whale_zero_not_in_payload(self):
        _reset_static_cache()
        event = _make_event_data()
        event["institutional_analytics"]["flow_analysis"]["whale_accumulation"]["score"] = 0
        result = build_compact_payload(event)
        assert "w" not in result


# ============================================================
# TEST 7: Payload completo — integração
# ============================================================

class TestFullPayload:
    def test_full_payload_structure(self):
        _reset_static_cache()
        event = _make_event_data()
        result = build_compact_payload(event)

        # Chaves v3
        assert "t" in result  # trigger
        assert "p" in result  # price
        assert "r" in result  # regime
        assert "f" in result  # flow
        assert "ob" in result  # orderbook
        assert "w" in result  # whale
        assert "tf" in result  # timeframes
        assert "ctx" in result  # context (first call)

        # Chaves v2 NÃO devem existir
        assert "symbol" not in result
        assert "window" not in result
        assert "epoch_ms" not in result
        assert "trigger" not in result
        assert "price" not in result
        assert "regime" not in result

    def test_payload_flow_keys_are_v3(self):
        _reset_static_cache()
        event = _make_event_data()
        result = build_compact_payload(event)
        flow = result["f"]
        assert "d1" in flow
        assert "n1" not in flow

    def test_payload_ctx_has_market_prices(self):
        _reset_static_cache()
        event = _make_event_data()
        result = build_compact_payload(event)
        ctx = result["ctx"]
        assert "dxy" in ctx
        assert "spx" in ctx
        assert "gold" in ctx


# ============================================================
# TEST 8: _safe_price e _safe_int
# ============================================================

class TestSafeHelpers:
    def test_safe_price_dict(self):
        ext = {"DXY": {"preco_atual": 119.49}}
        assert _safe_price(ext, "DXY") == 119.49

    def test_safe_price_numeric(self):
        ext = {"DXY": 119.49}
        assert _safe_price(ext, "DXY") == 119.49

    def test_safe_price_missing(self):
        ext = {}
        assert _safe_price(ext, "DXY") is None

    def test_safe_price_zero(self):
        ext = {"DXY": {"preco_atual": 0}}
        assert _safe_price(ext, "DXY") is None

    def test_safe_int_dict(self):
        ext = {"FEAR_GREED": {"preco_atual": 15}}
        assert _safe_int(ext, "FEAR_GREED") == 15

    def test_safe_int_none(self):
        ext = {}
        assert _safe_int(ext, "FEAR_GREED") is None


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
