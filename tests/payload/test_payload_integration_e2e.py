"""
Testes de integração end-to-end do pipeline de payload.

Valida que:
  1. build_compact_payload gera payload válido
  2. summary builders são chamados corretamente
  3. summary é coerente com os dados do payload
  4. budget guard protege o pipeline em caso de crescimento
  5. fallback gracioso quando builders falham
  6. payload final é serializável e dentro do contrato
"""

import json
import copy
import pytest
from unittest.mock import patch

import build_compact_payload as bcp
from market_orchestrator.ai.payload_sections import (
    build_flow_summary,
    build_sr_summary,
    build_regime_summary,
    build_institutional_summary,
    build_quality_summary,
)


@pytest.fixture(autouse=True)
def reset_static_cache():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0


def make_full_event() -> dict:
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "descricao": "Evento automático para análise da IA",
        "preco_fechamento": 66875,
        "delta": 0.123,
        "volume_total": 1.5,
        "volume_compra": 0.9,
        "contextual_snapshot": {
            "ohlc": {
                "open": 66884,
                "high": 66890,
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
        "historical_vp": {
            "daily": {
                "status": "success",
                "poc": 66444,
                "val": 66388,
                "vah": 67771,
            }
        },
        "external_markets": {
            "VIX": {"preco_atual": 23.9},
            "FEAR_GREED": {"preco_atual": 50},
            "DXY": {"preco_atual": 104.2},
            "SP500": {"preco_atual": 669.1},
        },
        "derivatives": {
            "BTCUSDT": {
                "long_short_ratio": 1.83,
                "open_interest": 90101.906,
                "funding_rate_percent": -0.01,
                "longs_usd": 3893635557.19,
                "shorts_usd": 2129881055.29,
            },
            "ETHUSDT": {"long_short_ratio": 1.66},
        },
        "ml_features": {
            "cross_asset": {
                "btc_eth_corr_7d": 0.92,
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
        "volatility_metrics": {
            "bbw": 0.18,
            "atr_pct": 0.57,
        },
        "fluxo_continuo": {
            "cvd": 0.2,
            "sector_flow": {
                "whale": {"delta": 0.4},
                "retail": {"delta": -0.2},
            },
            "order_flow": {
                "net_flow_1m": 16000,
                "net_flow_5m": 12000,
                "net_flow_15m": 8000,
                "flow_imbalance": 0.18,
                "aggressive_buy_pct": 59,
                "buy_sell_ratio": {"buy_sell_ratio": 1.44},
            },
            "absorption_analysis": {
                "current_absorption": {
                    "buyer_strength": 5.9,
                    "seller_exhaustion": 1.8,
                    "continuation_probability": 0.22,
                }
            },
        },
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
                "100k_usd": {"buy": 0.05, "sell": 0.06},
            }
        },
        "institutional_analytics": {
            "profile_analysis": {
                "profile_shape": {"shape": "b"},
                "poor_extremes": {
                    "action_bias": "expect_retest_low",
                    "poor_high": {"detected": 0},
                    "poor_low": {"detected": 1, "volume_ratio": 5.145},
                },
                "va_volume_pct": {
                    "breakout_risk": "VERY_HIGH",
                    "compression_signal": 1,
                },
            },
            "quality": {
                "calendar": {"expected_liquidity": "NORMAL"},
                "latency": {
                    "latency_ms": 820,
                    "latency_category": "OK",
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
                    "score": 18,
                    "classification": "MILD_ACCUMULATION",
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
                        {"center": 66839, "strength": 58, "source_count": 4},
                        {"center": 66410, "strength": 62, "source_count": 4},
                    ],
                    "defense_asymmetry": {"bias": "neutral"},
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
        "alerts": {
            "active_alerts": [
                {"type": "SUPPORT_TEST",    "severity": "HIGH", "level": 66839},
                {"type": "RESISTANCE_TEST", "severity": "HIGH", "level": 66931},
            ]
        },
        "data_reliability": {
            "latency_acceptable": 1,
            "onchain_coverage": "full",
        },
    }


# ═══════════════════════════════════════════════════════════════════
# TESTES DE INTEGRAÇÃO
# ═══════════════════════════════════════════════════════════════════

class TestE2EPipelineIntegration:

    def test_full_pipeline_produces_valid_payload(self):
        payload = bcp.build_compact_payload(make_full_event())

        assert "symbol" in payload
        assert "price" in payload
        assert "flow" in payload
        assert "sr" in payload
        assert payload["symbol"] == "BTCUSDT"
        assert payload["price"]["c"] == 66875

    def test_summary_section_present_in_full_pipeline(self):
        payload = bcp.build_compact_payload(make_full_event())

        assert "summary" in payload, (
            "summary ausente no payload final — verificar integração"
        )

    def test_summary_has_all_five_builders(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        for key in ("flow", "sr", "regime", "institutional", "quality"):
            assert key in payload["summary"], (
                f"Builder ausente no summary: {key}"
            )

    def test_summary_flow_bias_matches_pa_signal(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        pa = str(payload["flow"].get("pa", "")).lower()
        bias = payload["summary"]["flow"]["bias"]

        assert bias in ("BUY", "SELL", "NEUTRAL")
        if "buy" in pa:
            assert bias in ("BUY", "NEUTRAL")

    def test_summary_quality_reliable_when_data_ok(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        qual = payload["summary"]["quality"]
        assert qual["reliable"] is True
        assert qual["confidence_cap"] == 1.0

    def test_summary_quality_unreliable_when_latency_degraded(self):
        event = make_full_event()
        event["institutional_analytics"]["quality"]["latency"]["latency_category"] = "DEGRADED"
        event["institutional_analytics"]["quality"]["latency"]["latency_ms"] = 3622

        payload = bcp.build_compact_payload(event)

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        qual = payload["summary"]["quality"]
        assert qual["reliable"] is False
        assert qual["confidence_cap"] == 0.7

    def test_summary_regime_label_matches_mode(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        mode = payload["regime"].get("mode", "")
        label = payload["summary"]["regime"]["label"]

        expected = {
            "MR": "Mean Reversion",
            "RB": "Range Bound",
            "TRD": "Trending",
            "BRK": "Breakout",
        }
        if mode in expected:
            assert label == expected[mode]

    def test_summary_regime_strategies_not_empty(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        strategies = payload["summary"]["regime"]["strategies"]
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_summary_institutional_unfinished_has_low(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        unfinished = payload["summary"]["institutional"]["unfinished"]
        assert "low" in unfinished

    def test_summary_sr_nearest_is_valid_value(self):
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        nearest = payload["summary"]["sr"]["nearest"]
        assert nearest in {"support", "resistance", "equidistant", "unknown"}

    def test_full_payload_is_json_serializable(self):
        payload = bcp.build_compact_payload(make_full_event())

        try:
            raw = json.dumps(payload, ensure_ascii=False)
            reparsed = json.loads(raw)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"Payload final não serializável: {exc}")

        assert reparsed["symbol"] == "BTCUSDT"

    def test_full_payload_below_hard_limit(self):
        payload = bcp.build_compact_payload(make_full_event())
        size = len(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ))
        assert size < 6144, f"Payload ultrapassou hard limit: {size} bytes"

    def test_full_payload_below_soft_limit(self):
        payload = bcp.build_compact_payload(make_full_event())
        size = len(json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ))
        # Limite auditado de qualidade (Prompt E)
        SOFT_LIMIT = 3500
        assert size < SOFT_LIMIT, (
            f"Payload ultrapassou soft limit: {size} bytes (limite={SOFT_LIMIT})"
        )


class TestE2EBudgetGuard:

    def test_budget_guard_removes_summary_when_payload_exceeds_hard_limit(self):
        """
        Simula cenário onde summary faz o payload ultrapassar o hard limit.
        Budget guard deve remover o summary para proteger o pipeline.
        """
        def giant_summary(_payload):
            return {"flow": {"note": "x" * 10000}}

        with patch.object(bcp, "_build_summary_section", side_effect=giant_summary):
            payload = bcp.build_compact_payload(make_full_event())

        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

        assert size < 6144, f"Budget guard falhou: {size} bytes"
        assert "summary" not in payload or len(
            json.dumps(payload["summary"])
        ) < 500

    def test_budget_guard_compacts_summary_when_over_soft_limit(self, caplog):
        """
        Simula cenário onde summary causa estouro do soft limit.
        Budget guard deve compactar (manter só note/bias/reliable).
        """
        import logging

        def large_summary(_payload):
            return {
                "flow": {
                    "bias": "BUY",
                    "type": "absorption",
                    "actor": "whale",
                    "conf": "H",
                    "note": "Absorção compradora ativa.",
                    "extra_large_field": "y" * 3000,
                },
                "sr": {"nearest": "resistance", "note": "S/R nota.", "compressed": False, "conf_bias": "NEUTRAL"},
                "regime": {"label": "MR", "strategies": [], "avoid": [], "duration": "1h", "note": "Regime nota."},
                "institutional": {"auction_state": "ok", "whale_bias": "ACCUMULATING", "profile_bias": "BULLISH", "unfinished": [], "note": "Inst nota."},
                "quality": {"reliable": True, "confidence_cap": 1.0, "issues": [], "note": "Quality nota."},
            }

        with caplog.at_level(logging.WARNING, logger="build_compact_payload"):
            with patch.object(bcp, "_build_summary_section", side_effect=large_summary):
                payload = bcp.build_compact_payload(make_full_event())

        if "summary" in payload:
            flow_s = payload["summary"].get("flow", {})
            assert "extra_large_field" not in flow_s, (
                "Budget guard deveria ter removido extra_large_field"
            )

    def test_builder_exception_does_not_crash_pipeline(self):
        """
        Se um builder lançar exceção, o pipeline continua.
        Os outros builders devem funcionar normalmente.
        """
        original = build_flow_summary

        def failing_flow_summary(_payload):
            raise RuntimeError("Simulando falha no builder")

        with patch(
            "market_orchestrator.ai.payload_sections.flow_summary.build_flow_summary",
            side_effect=failing_flow_summary,
        ):
            try:
                payload = bcp.build_compact_payload(make_full_event())
                assert "symbol" in payload
                assert "price" in payload
            except Exception as exc:
                pytest.fail(
                    f"Pipeline quebrou quando builder falhou: {exc}"
                )

    def test_all_builders_exception_produces_empty_summary(self):
        """
        Se todos os builders falharem, summary deve estar ausente
        ou vazio — nunca deve crashar o pipeline.
        """
        def empty_summary(_payload):
            return {}

        with patch.object(bcp, "_build_summary_section", side_effect=empty_summary):
            payload = bcp.build_compact_payload(make_full_event())

        assert "symbol" in payload
        assert "price" in payload
        if "summary" in payload:
            assert payload["summary"] == {} or len(payload["summary"]) == 0


class TestE2ECoherence:

    def test_summary_notes_reference_real_values(self):
        """
        As notas do summary devem referenciar valores reais do payload,
        não valores hardcoded de fallback.
        """
        payload = bcp.build_compact_payload(make_full_event())

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        r1_price = payload["sr"].get("r1", [None])[0]
        sr_note = payload["summary"]["sr"]["note"]

        if r1_price and sr_note != "Sem dados de S/R disponíveis.":
            assert str(r1_price) in sr_note, (
                f"Nota de S/R não menciona o preço real {r1_price}: '{sr_note}'"
            )

    def test_summary_regime_avoids_opposite_direction_strategies(self):
        """
        Em regime BEAR, estratégias de compra não devem aparecer
        nas strategies recomendadas.
        """
        event = make_full_event()
        payload = bcp.build_compact_payload(event)

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        cs = payload["regime"].get("cs", "")
        mode = payload["regime"].get("mode", "")
        strategies = payload["summary"]["regime"].get("strategies", [])

        if cs == "BEAR" and mode in ("MR", "TRD"):
            for s in strategies:
                assert "comprar" not in s.lower(), (
                    f"Estratégia de compra em BEAR {mode}: '{s}'"
                )

    def test_summary_quality_issues_match_qual_fields(self):
        """
        Os issues do quality_summary devem corresponder ao que
        está em payload.qual.
        """
        event = make_full_event()
        event["institutional_analytics"]["quality"]["latency"]["latency_category"] = "DEGRADED"
        event["institutional_analytics"]["quality"]["latency"]["latency_ms"] = 3622
        event["institutional_analytics"]["quality"]["calendar"]["expected_liquidity"] = "REDUCED"

        payload = bcp.build_compact_payload(event)

        if "summary" not in payload:
            pytest.skip("summary não disponível")

        issues = payload["summary"]["quality"]["issues"]
        assert any("degradada" in i.lower() or "latên" in i.lower() for i in issues)
        assert any("reduzida" in i.lower() or "liquid" in i.lower() for i in issues)

    def test_wrapper_payload_also_contains_summary(self):
        """
        O wrapper build_compact_payload_for_llm deve preservar
        o summary no payload final.
        """
        wrapped = bcp.build_compact_payload_for_llm(
            make_full_event(),
            symbol="BTCUSDT",
            window=3,
            epoch_ms=1775173800000,
        )

        assert "price" in wrapped
        assert "flow" in wrapped

        if "summary" in wrapped:
            assert isinstance(wrapped["summary"], dict)
