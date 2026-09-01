"""
test_funding_rate_fallback.py
-----------------------------
Testes conceituais e de integração do contrato canônico de Funding Rate:
- _build_btc_funding(event) retorna SEMPRE em PERCENTUAL finito (nunca fração, nunca NaN/Inf).
- Precedência explícita: funding_rate_percent [%] > funding_rate_pct [%] > funding_rate [fraction] * 100.
- Rejeição estrita de dados não-finitos (NaN, +Inf, -Inf) e tipos incompatíveis (bool, str inválida).
- A Onda 1 do enricher (enrich_signal) não duplica conversão (x100).
- Casos A até G, conflito de fontes, robustez de fallback e integração com payload_builder_compact.
"""

import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from institutional.enricher import _build_btc_funding, enrich_signal
import market_orchestrator.ai.payload_builder_compact as pbc
from market_orchestrator.ai.payload_builder_compact import build_compact_payload


@pytest.fixture(autouse=True)
def reset_payload_compact_cache():
    """Garante isolamento dos testes resetando o cache de seção estática."""
    pbc._last_static_ctx = None
    pbc._last_static_ts = 0
    yield
    pbc._last_static_ctx = None
    pbc._last_static_ts = 0


def _base_event(derivatives_btc: dict) -> dict:
    return {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "timestamp_ms": 1786141980000,
        "epoch_ms": 1786141980000,
        "preco_fechamento": 65000.0,
        "volume_total": 100.0,
        "derivatives": {
            "BTCUSDT": derivatives_btc
        },
        "market_context": {
            "orderbook": {"bids": [[64999.0, 1.0]], "asks": [[65001.0, 1.0]]}
        }
    }


class TestCanonicalFundingRateContract:
    """Testes dos Casos A até G e de conflito de precedência."""

    def test_caso_a_percent_normal(self):
        """Caso A: funding_rate_percent=0.01 -> final 0.01."""
        ev = _base_event({"funding_rate_percent": 0.01})
        val = _build_btc_funding(ev)
        assert val == 0.01

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == 0.01

    def test_caso_b_pct_normal(self):
        """Caso B: funding_rate_pct=0.01 -> final 0.01."""
        ev = _base_event({"funding_rate_pct": 0.01})
        val = _build_btc_funding(ev)
        assert val == 0.01

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == 0.01

    def test_caso_c_fraction_raw_converts_once(self):
        """Caso C: funding_rate=0.0001 -> final 0.01 (NUNCA 1.0)."""
        ev = _base_event({"funding_rate": 0.0001})
        val = _build_btc_funding(ev)
        assert val == 0.01

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == 0.01, f"BUG: Dupla conversão detectada! Esperado 0.01, obtido {final_val}"

    def test_caso_d_all_present_simultaneously(self):
        """Caso D: todos presentes simultaneamente -> final 0.01."""
        ev = _base_event({
            "funding_rate_percent": 0.01,
            "funding_rate_pct": 0.01,
            "funding_rate": 0.0001
        })
        val = _build_btc_funding(ev)
        assert val == 0.01

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == 0.01

    def test_caso_e_negative_raw(self):
        """Caso E: raw=-0.0001 -> final -0.01."""
        ev = _base_event({"funding_rate": -0.0001})
        val = _build_btc_funding(ev)
        assert val == -0.01

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == -0.01

    def test_caso_f_non_round_raw(self):
        """Caso F: raw=0.000085 -> final 0.0085."""
        ev = _base_event({"funding_rate": 0.000085})
        val = _build_btc_funding(ev)
        assert round(val, 6) == 0.0085

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert round(final_val, 6) == 0.0085

    def test_caso_g_small_legitimate_percent_never_multiplied(self):
        """Caso G: percentual legítimo muito pequeno (0.0005) NÃO é multiplicado x100."""
        ev = _base_event({"funding_rate_percent": 0.0005})
        val = _build_btc_funding(ev)
        assert val == 0.0005

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == 0.0005, f"BUG: Magnitude causou conversão indevida! Obtido {final_val}"

    def test_conflict_sources_precedence_determinism(self):
        """Precedência determinística: funding_rate_percent (0.01) tem precedência sobre funding_rate_pct (0.02)."""
        ev = _base_event({
            "funding_rate_percent": 0.01,
            "funding_rate_pct": 0.02
        })
        val = _build_btc_funding(ev)
        assert val == 0.01

        enriched = enrich_signal(ev)
        final_val = enriched["derivatives"]["BTCUSDT"]["funding_rate_percent"]
        assert final_val == 0.01


class TestNonFiniteAndTypeFallbacks:
    """Testes de robustez com NaN, Inf, -Inf, booleanos, strings e fallbacks em cascata."""

    def test_fallback_a_nan_percent_falls_back_to_valid_pct(self, caplog):
        """A: funding_rate_percent = NaN, funding_rate_pct = 0.01 -> retorna 0.01 sem falso warning."""
        ev = _base_event({
            "funding_rate_percent": float("nan"),
            "funding_rate_pct": 0.01
        })
        val = _build_btc_funding(ev)
        assert val == 0.01
        assert "Inconsistência em derivativos BTC" not in caplog.text

    def test_fallback_b_inf_percent_falls_back_to_valid_pct(self):
        """B: funding_rate_percent = +Inf, funding_rate_pct = 0.01 -> retorna 0.01."""
        ev = _base_event({
            "funding_rate_percent": float("inf"),
            "funding_rate_pct": 0.01
        })
        val = _build_btc_funding(ev)
        assert val == 0.01

        # Teste com -Inf também
        ev_neg = _base_event({
            "funding_rate_percent": float("-inf"),
            "funding_rate_pct": 0.01
        })
        assert _build_btc_funding(ev_neg) == 0.01

    def test_fallback_c_invalid_percent_and_pct_falls_back_to_raw(self):
        """C: funding_rate_percent inválido, funding_rate_pct inválido, funding_rate = 0.0001 -> retorna 0.01."""
        ev = _base_event({
            "funding_rate_percent": "invalido",
            "funding_rate_pct": float("nan"),
            "funding_rate": 0.0001
        })
        val = _build_btc_funding(ev)
        assert val == 0.01

    def test_fallback_d_all_invalid_returns_none(self):
        """D: todos inválidos/non-finite -> retorna None."""
        ev = _base_event({
            "funding_rate_percent": float("nan"),
            "funding_rate_pct": float("inf"),
            "funding_rate": float("-inf")
        })
        assert _build_btc_funding(ev) is None

        ev_empty = _base_event({
            "funding_rate_percent": "",
            "funding_rate_pct": None,
            "funding_rate": "abc"
        })
        assert _build_btc_funding(ev_empty) is None

    def test_fallback_e_raw_non_finite_returns_none(self):
        """E: funding_rate raw = NaN/Inf -> retorna None (nunca converte NaN * 100)."""
        ev_nan = _base_event({"funding_rate": float("nan")})
        assert _build_btc_funding(ev_nan) is None

        ev_inf = _base_event({"funding_rate": float("inf")})
        assert _build_btc_funding(ev_inf) is None

    def test_types_boolean_rejected_explicitly(self):
        """Booleanos (True/False) não são aceitos como funding rate numérico."""
        ev_true = _base_event({"funding_rate_percent": True, "funding_rate_pct": False, "funding_rate": 0.0001})
        val = _build_btc_funding(ev_true)
        # True e False devem ser ignorados, caindo no fallback do funding_rate
        assert val == 0.01

        ev_bool_only = _base_event({"funding_rate_percent": True, "funding_rate_pct": False, "funding_rate": True})
        assert _build_btc_funding(ev_bool_only) is None


class TestPayloadIntegrationFunding:
    """Valida o valor que chega no payload_builder_compact para a IA."""

    def test_compact_payload_receives_exact_percent_from_raw_fallback(self):
        """Garante que fallback raw-only (0.0001) produz ctx.fr == 0.01 e NUNCA 1.0."""
        ev = _base_event({"funding_rate": 0.0001})
        enriched = enrich_signal(ev)
        payload = build_compact_payload(enriched)

        ctx = payload.get("ctx", {})
        assert "fr" in ctx
        assert ctx["fr"] == 0.01, f"BUG no payload IA: esperado ctx['fr'] == 0.01, obtido {ctx['fr']}"

    def test_compact_payload_receives_exact_percent_from_percent_field(self):
        """Garante que funding_rate_percent=0.01 produz ctx.fr == 0.01."""
        ev = _base_event({"funding_rate_percent": 0.01})
        enriched = enrich_signal(ev)
        payload = build_compact_payload(enriched)

        ctx = payload.get("ctx", {})
        assert "fr" in ctx
        assert ctx["fr"] == 0.01
