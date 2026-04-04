import copy
import pytest

import build_compact_payload as bcp


@pytest.fixture(autouse=True)
def reset_static_cache():
    """Reseta o mini-cache global entre testes."""
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
        },
        "external_markets": {
            "VIX": {"preco_atual": 23.9},
            "FEAR_GREED": {"preco_atual": 50},
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
                "longs_usd": 3893635557.19,
                "shorts_usd": 2129881055.29,
            },
            "ETHUSDT": {
                "long_short_ratio": 1.66,
            },
        },
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": 0.9,
                "btc_dxy_corr_30d": -0.41,
            },
            "microstructure": {
                "trade_intensity_v2": 12.3,
                "tick_rule_sum": 10,
                "order_book_slope": -0.25,
            },
        },
        "multi_tf": {
            "15m": {
                "tendencia": "Baixa",
                "rsi_short": 45,
                "macd": 27,
                "macd_signal": 41,
                "adx": 11,
                "atr": 127,
                "regime": "Range",
            },
            "1h": {
                "tendencia": "Baixa",
                "rsi_short": 50,
                "macd": -151,
                "macd_signal": -241,
                "adx": 21,
                "atr": 384,
                "regime": "Range",
            },
            "4h": {
                "tendencia": "Baixa",
                "rsi_short": 43,
                "macd": -178,
                "macd_signal": -64,
                "adx": 27,
                "atr": 968,
                "regime": "Manipulação",
            },
            "1d": {
                "tendencia": "Baixa",
                "rsi_short": 40,
                "macd": -829,
                "macd_signal": -584,
                "adx": 21,
                "atr": 2525,
                "regime": "Acumulação",
            },
        },
        "fluxo_continuo": {
            "cvd": 0.2,
            "order_flow": {
                "net_flow_1m": 16000,
                "net_flow_5m": 16000,
                "net_flow_15m": 16000,
                "flow_imbalance": 0.18,
                "aggressive_buy_pct": 59,
                "buy_sell_ratio": {
                    "buy_sell_ratio": 1.44,
                },
            },
            "absorption_analysis": {
                "current_absorption": {
                    "buyer_strength": 5.9,
                    "seller_exhaustion": 1.8,
                }
            },
        },
        "delta": 0.123,
        "volume_total": 1.5,
        "volume_compra": 0.9,
        "orderbook_data": {
            "bid_depth_usd": 303000,
            "ask_depth_usd": 1700000,
            "imbalance": -0.7,
        },
        "order_book_depth": {
            "L5": {"imbalance": -0.96},
        },
        "market_impact": {
            "slippage_matrix": {
                "100k_usd": {
                    "buy": 0.05,
                    "sell": 0.06,
                }
            }
        },
        "institutional_analytics": {
            "profile_analysis": {
                "profile_shape": {
                    "shape": "b",
                },
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
            "flow_analysis": {
                "passive_aggressive": {
                    "composite": {
                        "signal": "buy_absorption",
                        "conviction": "MEDIUM",
                    }
                },
                "whale_accumulation": {
                    "score": 12,
                    "classification": "NEUTRAL",
                },
            },
            "sr_analysis": {
                "defense_zones": {
                    "status": "success",
                    "sell_defense": [
                        {"center": 66931, "strength": 66, "source_count": 5},
                        {"center": 67771, "strength": 52, "source_count": 3},
                    ],
                    "buy_defense": [
                        {"center": 66410, "strength": 62, "source_count": 4},
                        {"center": 66839, "strength": 42, "source_count": 2},
                    ],
                    "defense_asymmetry": {
                        "bias": "neutral",
                    },
                }
            },
        },
        "technical_indicators_extended": {
            "cci_signal": "NEUTRAL",
            "stochastic": {"k": 51},
            "williams_r": {"value": -58},
            "garch_forecast_1h": 0.0055,
            "hurst_exponent": 0.341,
            "shannon_entropy": 3.461,
            "fractal_dimension": 0.619,
            "kalman_filter": {
                "kalman_price": 66938.24,
                "deviation_pct": -0.0815,
                "trend_direction": "DOWN",
            },
            "regression_channel": {
                "slope_per_bar": -2.8103,
                "position_in_channel": 0.4995,
                "deviation_from_trend": -0.06,
            },
            "monte_carlo": {
                "prob_up": 0.426,
                "p10": 66756.3,
                "p90": 66976.79,
            },
            "dominant_cycles": {
                "dominant_cycles": [66.7, 100.0, 40.0],
            },
        },
        "pattern_recognition": {
            "smart_money": {
                "fair_value_gaps": [
                    {"type": "BULLISH"},
                    {"type": "BEARISH"},
                ],
                "market_structure": {
                    "structure": "TRANSITION",
                    "bos_detected": 0,
                },
            }
        },
    }


def test_build_compact_payload_has_required_sections():
    payload = bcp.build_compact_payload(make_event())

    assert payload["symbol"] == "BTCUSDT"
    assert payload["epoch_ms"] == 1775173800000
    assert payload["trigger"] == "AT"

    for section in ("price", "regime", "flow", "ob", "tf", "sr", "ctx"):
        assert section in payload, f"Seção obrigatória ausente: {section}"


def test_build_compact_payload_uses_long_keys_not_short_keys():
    payload = bcp.build_compact_payload(make_event())

    assert "price" in payload
    assert "flow" in payload
    assert "regime" in payload
    assert "ob" in payload

    assert "p" not in payload
    assert "r" not in payload
    assert "t" not in payload


def test_breakout_risk_very_high_maps_to_v_hi():
    payload = bcp.build_compact_payload(make_event())

    assert payload["price"]["brk_risk"] == "V_HI"


def test_flow_and_orderbook_include_execution_signals():
    payload = bcp.build_compact_payload(make_event())

    assert payload["flow"]["d1"] == "+16K"
    assert payload["flow"]["d5"] == "+16K"
    assert payload["flow"]["d15"] == "+16K"
    assert payload["flow"]["imb"] == 0.18
    assert payload["flow"]["ab"] == 59
    assert payload["flow"]["bsr"] == 1.44
    assert payload["flow"]["pa"] == "buy_absorp"
    assert payload["flow"]["conv"] == "M"
    assert payload["flow"]["abs_buy_str"] == 5.9
    assert payload["flow"]["abs_sell_exh"] == 1.8

    assert payload["ob"]["bias"] == "SELL"
    assert payload["ob"]["imb"] == -0.7
    assert payload["ob"]["t5"] == -0.96
    assert payload["ob"]["slip_b"] == 5.0
    assert payload["ob"]["slip_s"] == 6.0


def test_sr_includes_distance_fields_and_confluence():
    payload = bcp.build_compact_payload(make_event())
    sr = payload["sr"]

    assert sr["r1"] == [66931, 66]
    assert sr["r1_conf"] == 5
    assert sr["r1_dist"] == 56

    assert sr["r2"] == [67771, 52]
    assert sr["r2_conf"] == 3
    assert sr["r2_dist"] == 896

    assert sr["s1"] == [66410, 62]
    assert sr["s1_conf"] == 4
    assert sr["s1_dist"] == 465

    assert sr["s2"] == [66839, 42]
    assert sr["s2_dist"] == 36

    assert sr["def_bias"] == "neutral"


def test_first_call_sends_full_ctx_and_second_call_sends_cached_min_ctx():
    event = make_event()

    first = bcp.build_compact_payload(copy.deepcopy(event))
    second = bcp.build_compact_payload(copy.deepcopy(event))

    assert "cached" not in first["ctx"]
    assert first["ctx"]["ses"] == "NY"
    assert first["ctx"]["lsr"] == 1.83
    assert first["ctx"]["oi"] == 90
    assert first["ctx"]["dxy30"] == -0.41

    assert second["ctx"]["cached"] is True
    assert second["ctx"]["ses"] == "NY"
    assert second["ctx"]["lsr"] == 1.83
    assert second["ctx"]["oi"] == 90
    assert second["ctx"]["dxy30"] == -0.41


def test_important_event_forces_full_ctx_even_when_cache_exists():
    normal_event = make_event()
    breakout_event = make_event()
    breakout_event["tipo_evento"] = "Breakout"

    _ = bcp.build_compact_payload(copy.deepcopy(normal_event))
    payload = bcp.build_compact_payload(copy.deepcopy(breakout_event))

    assert "cached" not in payload["ctx"]
    assert payload["ctx"]["ses"] == "NY"
    assert payload["ctx"]["lsr"] == 1.83
    assert payload["ctx"]["fr"] == -0.01


def test_wrapper_preserves_sections_and_metadata():
    wrapped = bcp.build_compact_payload_for_llm(
        make_event(),
        symbol="BTCUSDT",
        window=3,
        epoch_ms=1775173800000,
    )

    assert wrapped["symbol"] == "BTCUSDT"
    assert wrapped["window"] == 3
    assert wrapped["epoch_ms"] == 1775173800000
    assert wrapped["trigger"] == "AT"
    assert wrapped["price"]["c"] == 66875
    assert wrapped["flow"]["pa"] == "buy_absorp"
    assert wrapped["tipo_evento"] == "ANALYSIS_TRIGGER"
    assert wrapped["descricao"] == "Evento automático para análise da IA"
