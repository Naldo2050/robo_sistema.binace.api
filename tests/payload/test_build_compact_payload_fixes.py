"""
Testes de regressão para os fixes aplicados em produção.

Cobre:
  - Fix 1: guardrail whitelist
  - Fix 2: quality VERY_LOW e feriado
  - Fix 3a: qual preserva liq completo e feriado
  - Fix 3b: stoch_sig e candlestick
  - Fix 3c: regime.mode usa regime_analysis
  - Fix 3d: mr threshold reduzido
  - Fix 3e: ofi usa order_flow como fonte secundária
"""

import pytest
import build_compact_payload as bcp
from market_orchestrator.ai.payload_sections.quality_summary import (
    build_quality_summary,
    _resolve_liquidity,
)


@pytest.fixture(autouse=True)
def reset_cache():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0


def make_production_event() -> dict:
    """Evento baseado nos dados reais de produção (Janela 3, 2026-04-03)."""
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775239800000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "preco_fechamento": 66810,
        "delta": -0.386,
        "volume_total": 2.47,
        "volume_compra": 1.042,
        "contextual_snapshot": {
            "ohlc": {
                "open": 66820.8,
                "high": 66820.9,
                "low": 66809.9,
                "close": 66810,
                "vwap": 66813.8,
            }
        },
        "market_context": {
            "trading_session": "NY_OVERLAP",
            "session_phase": "ACTIVE",
        },
        "market_environment": {
            "volatility_regime": "HIGH",
            "market_structure": "RANGE_BOUND",
        },
        "regime_analysis": {
            "current_regime": "MEAN_REVERTING",
            "regime_probabilities": {
                "trending": 0,
                "mean_reverting": 1.0,
                "breakout": 0,
            },
        },
        "historical_vp": {
            "daily": {
                "status": "success",
                "poc": 67015,
                "val": 66635,
                "vah": 67277,
            }
        },
        "external_markets": {
            "VIX": {"preco_atual": 23.87},
            "FEAR_GREED": {"preco_atual": 9},
        },
        "derivatives": {
            "BTCUSDT": {
                "long_short_ratio": 1.72,
                "open_interest": 90413.354,
                "funding_rate_percent": 0.0,
                "longs_usd": 3823259495.96,
                "shorts_usd": 2216639318.16,
            },
            "ETHUSDT": {"long_short_ratio": 1.8},
        },
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": 0.923,
                "btc_dxy_corr_30d": -0.4248,
            },
            "microstructure": {
                "trade_intensity_v2": 10.4167,
                "tick_rule_sum": -64,
                "order_book_slope": -0.251838,
                "flow_imbalance": -0.1563,
            },
        },
        "multi_tf": {
            "15m": {
                "tendencia": "Baixa",
                "rsi_short": 42.9,
                "macd": 12.9855,
                "macd_signal": 19.7341,
                "adx": 28.65,
                "atr": 110.81,
                "regime": "Range",
            },
            "1h": {
                "tendencia": "Baixa",
                "rsi_short": 49.97,
                "macd": -19.7546,
                "macd_signal": -45.5299,
                "adx": 30.78,
                "atr": 281.59,
                "regime": "Range",
            },
            "4h": {
                "tendencia": "Baixa",
                "rsi_short": 44.52,
                "macd": -208.8916,
                "macd_signal": -160.3746,
                "adx": 29.32,
                "atr": 802.81,
                "regime": "Manipulação",
            },
            "1d": {
                "tendencia": "Baixa",
                "rsi_short": 40.17,
                "macd": -867.713,
                "macd_signal": -641.9388,
                "adx": 22.05,
                "atr": 2333.71,
                "regime": "Manipulação",
            },
        },
        "fluxo_continuo": {
            "cvd": -2.3271,
            "sector_flow": {
                "whale":  {"delta": 0.0},
                "retail": {"delta": -2.327},
            },
            "order_flow": {
                "net_flow_1m": -26271.9421,
                "net_flow_5m": -155510.1608,
                "net_flow_15m": -155510.1608,
                "flow_imbalance": -0.1563,
                "aggressive_buy_pct": 42.19,
                "buy_sell_ratio": {"buy_sell_ratio": 0.73},
            },
            "absorption_analysis": {
                "current_absorption": {
                    "buyer_strength": 4.2,
                    "seller_exhaustion": 1.6,
                    "continuation_probability": 0.02,
                }
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 613388.13,
            "ask_depth_usd": 476394.66,
            "imbalance": 0.126,
        },
        "order_book_depth": {
            "L5": {"imbalance": 0.13},
        },
        "market_impact": {
            "slippage_matrix": {
                "100k_usd": {"buy": 0.05, "sell": 0.05},
            }
        },
        "volatility_metrics": {
            "realized_vol_24h": 0.0322,
            "realized_vol_7d": 0.0852,
            "volatility_regime": "HIGH",
        },
        "institutional_analytics": {
            "status": "ok",
            "profile_analysis": {
                "profile_shape": {"shape": "b"},
                "poor_extremes": {
                    "action_bias": "expect_retest_both",
                    "poor_high": {"detected": 1, "volume_ratio": 0.548},
                    "poor_low":  {"detected": 1, "volume_ratio": 4.869},
                },
                "va_volume_pct": {
                    "breakout_risk": "VERY_HIGH",
                    "compression_signal": 1,
                },
            },
            "quality": {
                "calendar": {
                    "day_of_week": "Friday",
                    "is_us_holiday": 1,
                    "holiday_name": "Good Friday",
                    "expected_liquidity": "VERY_LOW",
                    "liquidity_warning": 1,
                },
                "latency": {
                    "latency_ms": 3265,
                    "latency_category": "DEGRADED",
                },
            },
            "flow_analysis": {
                "passive_aggressive": {
                    "composite": {
                        "signal": "sell_absorption",
                        "conviction": "MEDIUM",
                    }
                },
                "whale_accumulation": {
                    "score": 10,
                    "classification": "NEUTRAL",
                },
            },
            "sr_analysis": {
                "defense_zones": {
                    "status": "success",
                    "sell_defense": [
                        {"center": 67014.39, "strength": 60, "source_count": 3},
                        {"center": 66879.41, "strength": 54, "source_count": 4},
                    ],
                    "buy_defense": [
                        {"center": 66657.82, "strength": 50, "source_count": 3},
                        {"center": 66780.65, "strength": 49, "source_count": 3},
                    ],
                    "defense_asymmetry": {
                        "bias": "strong_sell_defense",
                    },
                }
            },
            "candlestick_patterns": {
                "patterns_detected": 2,
                "patterns": [
                    {
                        "name": "pin_bar",
                        "type": "bearish",
                        "confidence": 0.72,
                    },
                    {
                        "name": "gravestone_doji",
                        "type": "bearish",
                        "confidence": 0.65,
                    },
                ],
                "dominant_signal": "bearish",
                "max_confidence": 0.72,
            },
        },
        "technical_indicators_extended": {
            "cci_signal": "NEUTRAL",
            "stochastic": {
                "k": 15.54,
                "d": 36.16,
                "signal": "OVERSOLD",
            },
            "williams_r": {"value": -100},
            "garch_forecast_1h": 0.0051,
            "hurst_exponent": 0.378,
            "shannon_entropy": 3.8349,
            "fractal_dimension": 0.5387,
            "kalman_filter": {
                "kalman_price": 66880.63,
                "deviation_pct": -0.09,
                "trend_direction": "DOWN",
            },
            "regression_channel": {
                "slope_per_bar": -0.9694,
                "position_in_channel": 0.2717,
                "deviation_from_trend": -21.87,
            },
            "monte_carlo": {
                "prob_up": 0.43,
                "p10": 66709.01,
                "p90": 66899.38,
            },
            "dominant_cycles": {
                "dominant_cycles": [100, 40, 20],
            },
        },
        "pattern_recognition": {
            "smart_money": {
                "fair_value_gaps": [
                    {"type": "BEARISH"},
                    {"type": "BEARISH"},
                    {"type": "BULLISH"},
                ],
                "market_structure": {
                    "structure": "TRANSITIONING",
                    "bos_detected": 0,
                },
            }
        },
        "alerts": {
            "active_alerts": [
                {
                    "type": "SUPPORT_TEST",
                    "severity": "HIGH",
                    "level": 66780.65,
                },
            ]
        },
        "data_reliability": {
            "latency_acceptable": 1,
            "onchain_coverage": "full",
        },
        "whale_activity": {
            "iceberg_activity": 0,
            "hidden_orders_detected": 0,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# FIX 1 — Guardrail whitelist
# ═══════════════════════════════════════════════════════════════════

class TestGuardrailWhitelist:

    def test_guardrail_allows_summary_key(self):
        try:
            from market_orchestrator.ai.llm_payload_guardrail import (
                _ALLOWED_FLAT_KEYS,
                _is_allowed_key,
            )
        except ImportError:
            pytest.skip("llm_payload_guardrail não encontrado")

        assert "summary" in _ALLOWED_FLAT_KEYS or _is_allowed_key("summary"), (
            "summary não está na whitelist do guardrail"
        )

    def test_guardrail_allows_gap_keys(self):
        try:
            from market_orchestrator.ai.llm_payload_guardrail import _is_allowed_key
        except ImportError:
            pytest.skip("llm_payload_guardrail não encontrado")

        for key in ("ofi", "vwap", "liq", "sm", "cvd_div", "mr", "iceberg", "w", "ctx", "ext"):
            assert _is_allowed_key(key), (
                f"Key '{key}' não permitida no guardrail"
            )

    def test_guardrail_rewrap_preserves_new_keys(self):
        try:
            from market_orchestrator.ai.llm_payload_guardrail import guardrail_rewrap
        except ImportError:
            pytest.skip("llm_payload_guardrail não encontrado")

        payload = {
            "symbol": "BTCUSDT",
            "price": {"c": 66810},
            "flow": {"d1": "-26K"},
            "summary": {"flow": {"bias": "SELL"}},
            "ofi": {"score": -0.156, "dir": "SELL"},
            "vwap": {"dev": -0.006, "side": "below"},
            "w": {"s": 10, "c": "N"},
            "ctx": {"ses": "NY"},
        }

        result = guardrail_rewrap(payload)
        inner = result.get("ai_payload", result)

        for key in ("summary", "ofi", "vwap", "w", "ctx"):
            assert key in inner, f"Guardrail removeu '{key}' indevidamente"


# ═══════════════════════════════════════════════════════════════════
# FIX 2 — Quality VERY_LOW e feriado
# ═══════════════════════════════════════════════════════════════════

class TestQualityVeryLow:

    def test_resolve_liquidity_very_low(self):
        cap, label = _resolve_liquidity("VERY_LOW")
        assert cap == 0.5
        assert "baixa" in label.lower()

    def test_resolve_liquidity_truncated_ver(self):
        cap, label = _resolve_liquidity("VER")
        assert cap == 0.5

    def test_resolve_liquidity_truncated_very(self):
        cap, label = _resolve_liquidity("VERY")
        assert cap == 0.5

    def test_quality_summary_very_low_liquidity(self):
        payload = {
            "qual": {"lat": "OK", "liq": "VERY_LOW"},
            "ctx": {},
        }
        result = build_quality_summary(payload)
        assert result["confidence_cap"] == 0.5
        assert result["reliable"] is False
        assert any("baixa" in i.lower() for i in result["issues"])

    def test_quality_summary_holiday_reduces_confidence(self):
        payload = {
            "qual": {
                "lat": "DEGR",
                "liq": "VERY_LOW",
                "holiday": "Good Friday",
                "ms": 3265,
            },
            "ctx": {},
        }
        result = build_quality_summary(payload)
        assert result["reliable"] is False
        assert result["confidence_cap"] <= 0.6
        assert any("good friday" in i.lower() or "feriado" in i.lower()
                   for i in result["issues"])

    def test_quality_summary_note_includes_confidence_pct(self):
        payload = {
            "qual": {"lat": "DEGR", "liq": "VERY_LOW", "holiday": "Good Friday"},
            "ctx": {},
        }
        result = build_quality_summary(payload)
        assert "%" in result["note"]


# ═══════════════════════════════════════════════════════════════════
# FIX 3a — qual preserva liq completo e feriado
# ═══════════════════════════════════════════════════════════════════

class TestQualBuilderPreservesLiquidity:

    def test_qual_preserves_very_low_without_truncation(self):
        payload = bcp.build_compact_payload(make_production_event())
        qual = payload.get("qual", {})

        liq = qual.get("liq", "")
        assert liq == "VERY_LOW", (
            f"liq foi truncado: esperado 'VERY_LOW', recebido '{liq}'"
        )

    def test_qual_includes_holiday_on_us_holiday(self):
        payload = bcp.build_compact_payload(make_production_event())
        qual = payload.get("qual", {})

        assert "holiday" in qual, "holiday não está no qual"
        assert "Good Friday" in qual["holiday"] or "Good" in qual["holiday"]

    def test_qual_includes_degraded_latency(self):
        payload = bcp.build_compact_payload(make_production_event())
        qual = payload.get("qual", {})

        assert qual.get("lat") == "DEGR"
        assert qual.get("ms") == 3265


# ═══════════════════════════════════════════════════════════════════
# FIX 3b — stoch_sig e candlestick
# ═══════════════════════════════════════════════════════════════════

class TestExtIndicatorsEnrichment:

    def test_stoch_signal_oversold_is_extracted(self):
        payload = bcp.build_compact_payload(make_production_event())
        ext = payload.get("ext", {})

        assert "stoch" in ext, "stoch ausente"
        assert ext["stoch"] == 16, f"stoch incorreto: {ext['stoch']}"

        assert "stoch_sig" in ext, (
            "stoch_sig ausente — sinal OVERSOLD não foi extraído"
        )
        assert ext["stoch_sig"] == "OS", (
            f"stoch_sig incorreto: {ext['stoch_sig']}"
        )

    def test_candlestick_bearish_pattern_extracted(self):
        payload = bcp.build_compact_payload(make_production_event())
        ext = payload.get("ext", {})

        assert "candle" in ext, (
            "candle ausente — pin_bar bearish não foi extraído"
        )
        assert ext["candle"]["sig"] == "bear"
        assert ext["candle"]["conf"] >= 0.65
        assert "pin_bar" in ext["candle"].get("name", "")

    def test_candlestick_absent_when_confidence_low(self):
        event = make_production_event()
        event["institutional_analytics"]["candlestick_patterns"]["max_confidence"] = 0.50
        payload = bcp.build_compact_payload(event)
        ext = payload.get("ext", {})

        assert "candle" not in ext, (
            "candle presente com confiança < 0.65"
        )

    def test_candlestick_absent_when_neutral(self):
        event = make_production_event()
        event["institutional_analytics"]["candlestick_patterns"]["dominant_signal"] = "neutral"
        payload = bcp.build_compact_payload(event)
        ext = payload.get("ext", {})

        assert "candle" not in ext


# ═══════════════════════════════════════════════════════════════════
# FIX 3c — regime.mode usa regime_analysis
# ═══════════════════════════════════════════════════════════════════

class TestRegimeModeFromRegimeAnalysis:

    def test_mean_reverting_sets_mode_mr(self):
        payload = bcp.build_compact_payload(make_production_event())
        assert payload["regime"]["mode"] == "MR", (
            f"regime.mode incorreto: {payload['regime']['mode']} "
            f"(esperado MR para MEAN_REVERTING)"
        )

    def test_breakout_regime_sets_mode_brk(self):
        event = make_production_event()
        event["regime_analysis"]["current_regime"] = "BREAKOUT"
        payload = bcp.build_compact_payload(event)
        assert payload["regime"]["mode"] == "BRK"

    def test_trending_regime_sets_mode_trd(self):
        event = make_production_event()
        event["regime_analysis"]["current_regime"] = "TRENDING"
        payload = bcp.build_compact_payload(event)
        assert payload["regime"]["mode"] == "TRD"

    def test_fallback_to_market_structure_when_no_regime_analysis(self):
        event = make_production_event()
        event.pop("regime_analysis", None)
        payload = bcp.build_compact_payload(event)
        assert payload["regime"]["mode"] in {"MR", "RB", "TRD", "BRK"}


# ═══════════════════════════════════════════════════════════════════
# FIX 3d — mr threshold reduzido
# ═══════════════════════════════════════════════════════════════════

class TestMrThreshold:

    def test_mr_appears_with_hurst_0378_and_pos_0272(self):
        """
        Dados reais de produção: hurst=0.378, pos=0.2717
        abs(0.2717 - 0.5) = 0.2283 > 0.2 (novo threshold)
        """
        payload = bcp.build_compact_payload(make_production_event())

        assert "mr" in payload, (
            "mr ausente — threshold ainda muito alto para dados de produção "
            f"(hurst=0.378, pos=0.2717)"
        )
        assert payload["mr"]["sig"] == "stretched_bear"
        assert payload["mr"]["src"] == "inferred"

    def test_mr_absent_when_hurst_above_048(self):
        event = make_production_event()
        event["technical_indicators_extended"]["hurst_exponent"] = 0.60
        payload = bcp.build_compact_payload(event)
        assert "mr" not in payload

    def test_mr_absent_when_position_too_centered(self):
        event = make_production_event()
        event["technical_indicators_extended"]["regression_channel"][
            "position_in_channel"
        ] = 0.49
        payload = bcp.build_compact_payload(event)
        assert "mr" not in payload


# ═══════════════════════════════════════════════════════════════════
# FIX 3e — ofi usa order_flow como fonte secundária
# ═══════════════════════════════════════════════════════════════════

class TestOfiOrderFlowSource:

    def test_ofi_uses_order_flow_flow_imbalance(self):
        payload = bcp.build_compact_payload(make_production_event())

        assert "ofi" in payload, "ofi ausente"
        ofi = payload["ofi"]

        assert ofi["score"] == round(-0.1563, 3)
        assert ofi["dir"] == "SELL"
        assert ofi.get("src") == "order_flow"

    def test_ofi_direction_sell_when_negative(self):
        event = make_production_event()
        event["fluxo_continuo"]["order_flow"]["flow_imbalance"] = -0.45
        payload = bcp.build_compact_payload(event)
        assert payload["ofi"]["dir"] == "SELL"

    def test_ofi_direction_buy_when_positive(self):
        event = make_production_event()
        event["fluxo_continuo"]["order_flow"]["flow_imbalance"] = 0.35
        payload = bcp.build_compact_payload(event)
        assert payload["ofi"]["dir"] == "BUY"

    def test_ofi_direction_neu_when_near_zero(self):
        event = make_production_event()
        event["fluxo_continuo"]["order_flow"]["flow_imbalance"] = 0.02
        payload = bcp.build_compact_payload(event)
        assert payload["ofi"]["dir"] == "NEU"


# ═══════════════════════════════════════════════════════════════════
# INTEGRAÇÃO — evento de produção completo
# ═══════════════════════════════════════════════════════════════════

class TestProductionEventIntegration:

    def test_production_event_generates_valid_payload(self):
        payload = bcp.build_compact_payload(make_production_event())
        assert payload["symbol"] == "BTCUSDT"
        assert payload["price"]["c"] == 66810

    def test_production_event_has_all_expected_gaps(self):
        payload = bcp.build_compact_payload(make_production_event())
        assert "ofi" in payload
        assert "vwap" in payload
        assert "mr" in payload

    def test_production_event_regime_mode_is_mr(self):
        payload = bcp.build_compact_payload(make_production_event())
        assert payload["regime"]["mode"] == "MR"

    def test_production_event_quality_is_unreliable(self):
        payload = bcp.build_compact_payload(make_production_event())

        if "summary" in payload:
            qual_s = payload["summary"]["quality"]
            assert qual_s["reliable"] is False
            assert qual_s["confidence_cap"] <= 0.6

    def test_production_event_summary_note_mentions_holiday(self):
        payload = bcp.build_compact_payload(make_production_event())

        if "summary" in payload:
            note = payload["summary"]["quality"]["note"]
            assert (
                "feriado" in note.lower()
                or "good friday" in note.lower()
                or "holiday" in note.lower()
            ), f"Nota não menciona feriado: '{note}'"

    def test_production_event_is_serializable(self):
        import json
        payload = bcp.build_compact_payload(make_production_event())
        raw = json.dumps(payload, ensure_ascii=False)
        reparsed = json.loads(raw)
        assert reparsed["symbol"] == "BTCUSDT"
