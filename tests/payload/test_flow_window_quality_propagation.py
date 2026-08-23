"""
Propagação de integridade temporal do fluxo (flow_window_integrity -> flow.q).

Contrato:
    flow.q = {
        "1m"/"5m"/"15m": {"s": "full|warm|trunc", "c": <coverage 0..100>},
    }

Política fail-closed ESTRITA:
    - metadata ausente (eventos legados) => q NÃO é emitido (nunca inventar FULL);
    - janela só entra em q se status reconhecido E coverage válida;
    - status desconhecido OU coverage ausente/não numérica/NaN/±Inf/bool/str
      => janela omitida POR INTEIRO (nunca emitir {"s": ...} sem "c");
    - coverage finita válida => clamp [0, 100], 1 decimal;
    - d1/d5/d15 NUNCA são nulificados ou removidos.
"""

import copy
import json
from types import SimpleNamespace

import pytest

from market_orchestrator.ai import payload_builder_compact as bcp
from market_orchestrator.ai.analyzer_qwen import (
    AIAnalyzer,
    _LARGE_GROQ_MODELS,
    _MODELS_WITHOUT_JSON_MODE,
)
from market_orchestrator.ai.llm_payload_guardrail import (
    ensure_safe_llm_payload,
)
from market_orchestrator.ai.payload_sections.flow_summary import (
    build_flow_summary,
)

pytestmark = pytest.mark.payload


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

def _window(status="FULL", coverage=100.0):
    return {
        "status": status,
        "effective_coverage_pct": coverage,
        "is_temporal_coverage_valid": status == "FULL",
    }


def _integrity(w1=None, w5=None, w15=None):
    out = {}
    if w1 is not None:
        out["1m"] = w1
    if w5 is not None:
        out["5m"] = w5
    if w15 is not None:
        out["15m"] = w15
    return out


FULL_ALL = _integrity(_window(), _window(), _window())
TRUNC_15M = _integrity(
    _window(),
    _window(),
    _window(status="CAPACITY_TRUNCATED", coverage=55.6),
)
WARM_15M = _integrity(
    _window(),
    _window(),
    _window(status="WARMING_UP", coverage=37.2),
)


def make_event(integrity=None, net_flow_15m=-3400.0):
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "tipo_evento": "Absorcao",
        "descricao": "Absorcao compradora na janela",
        "preco_fechamento": 66875,
        "delta": 0.123,
        "volume_total": 1.5,
        "volume_compra": 0.9,
        "volume_venda": 0.6,
        "contextual_snapshot": {
            "ohlc": {
                "open": 66884, "high": 66884,
                "low": 66856, "close": 66875, "vwap": 66879,
            },
        },
        "market_context": {"trading_session": "NY", "session_phase": "ACTIVE"},
        "multi_tf": {
            "15m": {
                "tendencia": "Baixa", "rsi_short": 45,
                "macd": 27, "adx": 11, "regime": "Range",
            },
        },
        "fluxo_continuo": {
            "cvd": 0.2,
            "order_flow": {
                "net_flow_1m": 16000.0,
                "net_flow_5m": -4200.0,
                "net_flow_15m": net_flow_15m,
                "flow_imbalance": 0.18,
                "aggressive_buy_pct": 59,
                "buy_sell_ratio": {"buy_sell_ratio": 1.44},
            },
            "absorption_analysis": {
                "current_absorption": {
                    "buyer_strength": 5.9, "seller_exhaustion": 1.8,
                },
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 303000,
            "ask_depth_usd": 1700000,
            "imbalance": -0.7,
            "data_source": "live",
        },
    }
    if integrity is not None:
        event["fluxo_continuo"]["flow_window_integrity"] = integrity
    return event


# ---------------------------------------------------------------------------
# 6. Teste A/B obrigatório
# ---------------------------------------------------------------------------

class TestFullVsTruncated:
    def test_a_vs_b(self):
        event_a = make_event(integrity=FULL_ALL)
        event_b = make_event(integrity=TRUNC_15M)

        payload_a = bcp.build_compact_payload(copy.deepcopy(event_a))
        payload_b = bcp.build_compact_payload(copy.deepcopy(event_b))

        assert payload_a != payload_b
        assert payload_a["flow"]["d15"] == payload_b["flow"]["d15"]
        assert payload_a["flow"]["q"]["15m"] == {"s": "full", "c": 100.0}
        assert payload_b["flow"]["q"]["15m"] == {"s": "trunc", "c": 55.6}

    def test_full_all_windows(self):
        payload = bcp.build_compact_payload(make_event(integrity=FULL_ALL))
        assert payload["flow"]["q"] == {
            "1m": {"s": "full", "c": 100.0},
            "5m": {"s": "full", "c": 100.0},
            "15m": {"s": "full", "c": 100.0},
        }

    def test_deltas_never_removed_or_nulled(self):
        payload = bcp.build_compact_payload(make_event(integrity=TRUNC_15M))
        flow = payload["flow"]
        assert flow["d1"] == "+16K"
        assert flow["d5"] == "-4K"
        assert flow["d15"] == "-3K"


# ---------------------------------------------------------------------------
# 7. Warm-up
# ---------------------------------------------------------------------------

class TestWarmUp:
    def test_warming_up_maps_to_warm_and_keeps_d15(self):
        payload = bcp.build_compact_payload(make_event(integrity=WARM_15M))
        assert payload["flow"]["q"]["15m"] == {"s": "warm", "c": 37.2}
        assert payload["flow"]["d15"] == "-3K"

    def test_warmup_without_eviction_partial_window(self):
        integrity = _integrity(
            _window(status="WARMING_UP", coverage=12.5),
            _window(),
            _window(),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert payload["flow"]["q"]["1m"] == {"s": "warm", "c": 12.5}


# ---------------------------------------------------------------------------
# 8. Legado (sem flow_window_integrity)
# ---------------------------------------------------------------------------

class TestLegacyEvent:
    def test_no_integrity_no_q_and_deltas_preserved(self):
        payload = bcp.build_compact_payload(make_event(integrity=None))
        assert "q" not in payload["flow"]
        assert payload["flow"]["d1"] == "+16K"
        assert payload["flow"]["d5"] == "-4K"
        assert payload["flow"]["d15"] == "-3K"

    def test_empty_integrity_dict_does_not_invent_quality(self):
        payload = bcp.build_compact_payload(make_event(integrity={}))
        assert "q" not in payload["flow"]

    def test_partial_metadata_only_valid_windows(self):
        integrity = _integrity(
            _window(status="CAPACITY_TRUNCATED", coverage=40.0),
            None,  # 5m ausente
            _window(status="NOT_A_KNOWN_STATUS", coverage=99.9),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert payload["flow"]["q"] == {
            "1m": {"s": "trunc", "c": 40.0},
        }


# ---------------------------------------------------------------------------
# 9. Sanitização
# ---------------------------------------------------------------------------

class TestStrictFailClosed:
    """Contrato estrito: janela só entra em q com status válido E coverage válida."""

    @pytest.mark.parametrize("bad_coverage", [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "100",
        {"pct": 100},
    ])
    def test_full_with_invalid_coverage_omits_window(self, bad_coverage):
        integrity = _integrity(
            _window(status="FULL", coverage=bad_coverage),
            None,
            None,
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert "q" not in payload["flow"]

    def test_unknown_status_with_valid_coverage_omits_window(self):
        integrity = _integrity(
            _window(status="WEIRD_STATUS", coverage=100),
            None,
            None,
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert "q" not in payload["flow"]

    def test_valid_entries_emit_status_and_coverage(self):
        integrity = _integrity(
            _window(status="FULL", coverage=100.0),
            _window(status="WARMING_UP", coverage=37.2),
            _window(status="CAPACITY_TRUNCATED", coverage=55.6),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert payload["flow"]["q"] == {
            "1m": {"s": "full", "c": 100.0},
            "5m": {"s": "warm", "c": 37.2},
            "15m": {"s": "trunc", "c": 55.6},
        }

    def test_out_of_range_finite_coverage_still_clamped(self):
        integrity = _integrity(
            _window(status="FULL", coverage=-10),
            None,
            _window(status="CAPACITY_TRUNCATED", coverage=150),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert payload["flow"]["q"]["1m"] == {"s": "full", "c": 0.0}
        assert payload["flow"]["q"]["15m"] == {"s": "trunc", "c": 100.0}

    def test_mixed_valid_and_invalid_windows(self):
        integrity = {
            "1m": {"status": "FULL"},                              # sem coverage -> omitir
            "5m": _window(),                                        # válida
            "15m": {"status": "CAPACITY_TRUNCATED", "effective_coverage_pct": "55.6"},
        }
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert payload["flow"]["q"] == {"5m": {"s": "full", "c": 100.0}}

    def test_every_entry_has_exactly_s_and_c(self):
        integrity = _integrity(
            W_FULL := _window(),
            _window(status="WARMING_UP", coverage=12.5),
            _window(status="CAPACITY_TRUNCATED", coverage=80.05),
        )
        del W_FULL
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        for entry in payload["flow"]["q"].values():
            assert set(entry.keys()) == {"s", "c"}
            assert entry["s"] in ("full", "warm", "trunc")
            assert isinstance(entry["c"], float)
            assert 0.0 <= entry["c"] <= 100.0


class TestSanitization:
    @pytest.mark.parametrize("bad_coverage", [
        float("nan"),
        float("inf"),
        float("-inf"),
        "55.6",
        None,
        True,
        {"pct": 50},
    ])
    def test_invalid_coverage_omits_window_entirely(self, bad_coverage):
        integrity = _integrity(
            _window(status="CAPACITY_TRUNCATED", coverage=bad_coverage),
            _window(),
            _window(),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert "1m" not in payload["flow"]["q"]
        assert payload["flow"]["q"]["5m"] == {"s": "full", "c": 100.0}

    @pytest.mark.parametrize("raw,expected", [
        (-10, 0.0),
        (-0.01, 0.0),
        (150, 100.0),
        (100.02, 100.0),
        (55.6, 55.6),
        (0, 0.0),
        (37.24, 37.2),
    ])
    def test_valid_coverage_clamped_to_0_100(self, raw, expected):
        integrity = _integrity(
            _window(status="WARMING_UP", coverage=raw),
            _window(),
            _window(),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert payload["flow"]["q"]["1m"] == {"s": "warm", "c": expected}

    def test_unknown_status_window_omitted_not_full(self):
        integrity = _integrity(
            _window(status="SOMETHING_ELSE", coverage=100.0),
            _window(),
            _window(),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert "1m" not in payload["flow"]["q"]
        assert payload["flow"]["q"]["5m"] == {"s": "full", "c": 100.0}

    def test_all_unknown_status_means_no_q_at_all(self):
        integrity = _integrity(
            _window(status="X", coverage=10),
            _window(status="Y", coverage=10),
            _window(status="Z", coverage=10),
        )
        payload = bcp.build_compact_payload(make_event(integrity=integrity))
        assert "q" not in payload["flow"]

    def test_non_integrity_dict_ignored(self):
        payload = bcp.build_compact_payload(make_event(integrity="garbage"))
        assert "q" not in payload["flow"]

    def test_final_json_never_contains_non_finite(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            integrity = _integrity(
                _window(status="CAPACITY_TRUNCATED", coverage=bad),
                _window(),
                _window(),
            )
            payload = bcp.build_compact_payload(make_event(integrity=integrity))
            serialized = json.dumps(payload, allow_nan=False)
            assert "NaN" not in serialized
            assert "Infinity" not in serialized
            assert "1m" not in payload["flow"]["q"]


# ---------------------------------------------------------------------------
# 11. Path real: builder -> guardrail -> prompt final
# ---------------------------------------------------------------------------

class TestGuardrailPathPreservesQ:
    def test_ensure_safe_llm_payload_preserves_flow_q(self):
        payload = bcp.build_compact_payload(make_event(integrity=TRUNC_15M))

        wrapped_event = {
            "ai_payload": payload,
            "raw_event": {"qualquer": "dado_bruto"},
            "tipo_evento": "Absorcao",
            "symbol": "BTCUSDT",
        }
        safe = ensure_safe_llm_payload(copy.deepcopy(wrapped_event))

        assert isinstance(safe, dict)
        assert safe["flow"]["q"]["15m"] == {"s": "trunc", "c": 55.6}

    def test_final_prompt_json_contains_q_truncated_marker(self):
        payload = bcp.build_compact_payload(make_event(integrity=TRUNC_15M))
        wrapped_event = {
            "ai_payload": payload,
            "raw_event": {"qualquer": "dado_bruto"},
            "tipo_evento": "Absorcao",
            "symbol": "BTCUSDT",
        }
        safe = ensure_safe_llm_payload(copy.deepcopy(wrapped_event))
        prompt_json = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))

        reparsed = json.loads(prompt_json)
        assert reparsed["flow"]["q"]["15m"] == {"s": "trunc", "c": 55.6}

    def test_guardrail_clean_path_also_preserves_q(self):
        payload = bcp.build_compact_payload(make_event(integrity=WARM_15M))
        safe = ensure_safe_llm_payload(copy.deepcopy(payload))
        assert safe["flow"]["q"]["15m"] == {"s": "warm", "c": 37.2}


# ---------------------------------------------------------------------------
# 5. flow_summary — semântica degradada
# ---------------------------------------------------------------------------

class TestFlowSummaryDegradedTemporal:
    @staticmethod
    def _summary_for(flow_extra):
        flow = {
            "pa": "neutral",
            "imb": 0.3,
            "d1": "+16K",
            "d5": "-8K",
        }
        flow.update(flow_extra)
        return build_flow_summary({"flow": flow})

    def test_divergence_still_signaled_when_degraded_window_is_15m_only(self):
        result = self._summary_for({
            "q": {
                "1m": {"s": "full", "c": 100.0},
                "5m": {"s": "full", "c": 100.0},
                "15m": {"s": "trunc", "c": 55.6},
            },
        })
        assert result.get("reversal_signal") is True
        assert "15m parcial (55.6% cobertura)" in result["note"]

    def test_divergence_suppressed_when_comparison_window_degraded(self):
        result = self._summary_for({
            "q": {
                "1m": {"s": "full", "c": 100.0},
                "5m": {"s": "warm", "c": 42.0},
            },
        })
        assert "reversal_signal" not in result
        assert "5m em aquecimento (42.0% cobertura)" in result["note"]

    def test_legacy_without_q_keeps_old_behavior(self):
        result = self._summary_for({})
        assert result.get("reversal_signal") is True
        assert "Cobertura temporal parcial" not in result["note"]

    def test_full_windows_add_no_note_noise(self):
        result = self._summary_for({
            "q": {
                "1m": {"s": "full", "c": 100.0},
                "5m": {"s": "full", "c": 100.0},
                "15m": {"s": "full", "c": 100.0},
            },
        })
        assert "Cobertura temporal parcial" not in result["note"]
        assert result.get("reversal_signal") is True


# ---------------------------------------------------------------------------
# 4. Legenda no system prompt (injeção real, sem rede)
# ---------------------------------------------------------------------------

class TestLegendContract:
    @staticmethod
    def _system_prompt(mode, model_name):
        fake_self = SimpleNamespace(config={}, mode=mode, model_name=model_name)
        return AIAnalyzer._get_system_prompt(fake_self)

    def test_field_legend_documents_q_semantics(self):
        from common.ai_field_legend import FIELD_LEGEND

        assert "q=window_temporal_quality" in FIELD_LEGEND
        assert "full=" in FIELD_LEGEND
        assert "warm=" in FIELD_LEGEND
        assert "trunc=" in FIELD_LEGEND
        assert "PARTIAL" in FIELD_LEGEND

    def test_groq_path_injects_legend_into_system_prompt(self):
        for model_name in ("llama-3.3-70b-versatile", next(iter(_MODELS_WITHOUT_JSON_MODE))):
            prompt = self._system_prompt("groq", model_name)
            assert "q=window_temporal_quality" in prompt
            assert "PARTIAL" in prompt

    def test_large_groq_model_is_known(self):
        assert "llama-3.3-70b-versatile" in _LARGE_GROQ_MODELS


# ---------------------------------------------------------------------------
# 12. Budget
# ---------------------------------------------------------------------------

class TestPayloadBudgetImpact:
    def test_q_overhead_is_small(self):
        legacy = bcp.build_compact_payload(make_event(integrity=None))
        full_q = bcp.build_compact_payload(make_event(integrity=FULL_ALL))
        trunc_q = bcp.build_compact_payload(make_event(integrity=TRUNC_15M))

        size = lambda p: len(json.dumps(p, ensure_ascii=False, separators=(",", ":")))  # noqa: E731

        assert size(full_q) - size(legacy) < 250
        assert size(trunc_q) - size(legacy) < 250

    def test_payload_with_q_below_hard_limit(self):
        payload = bcp.build_compact_payload(make_event(integrity=TRUNC_15M))
        size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        assert size < 6144
