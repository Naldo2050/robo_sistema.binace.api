# tests/unit/test_latency_reliability_regression.py
# -*- coding: utf-8 -*-
"""
ETAPA 2 — Auditoria de qualidade/latência/reliability (2026-08-10).

Cobre as contradições observadas em JANELA 1 e JANELA 4:
  - quality.latency.is_acceptable = false
  - data_reliability.latency_acceptable = true  (mesmo evento)
  - data_quality_score = 10.0 / reliability_score = 10.0
  - payload compacto da IA: reliable=true, confidence_cap=1.0,
    issues=[], note="Dados em tempo real sem anomalias... confiança plena"

Casos:
  A — latência aceitável  -> latency_acceptable=True
  B — latência inaceitável -> latency_acceptable=False
  C — latência ausente     -> FAIL-CLOSED (False), nunca true por default
  D — propagação até a IA  -> payload compacto não pode afirmar
      reliable=true/1.0/[] quando latência POOR/DELAYED ou desconhecida
  E — bool/int             -> normalização segura (True/1 e False/0 equivalentes)
  F — invariante           -> quality.latency.is_acceptable e
      data_reliability.latency_acceptable nunca discordam no mesmo evento
  G — reliability_score    -> latência inaceitável não pode render 10.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from institutional.enricher import enrich_signal
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from market_orchestrator.ai.payload_sections.quality_summary import (
    build_quality_summary,
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _base_event() -> dict:
    return {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "epoch_ms": 1786142460000,
        "preco_fechamento": 64856.34,
        "fluxo_continuo": {},
        "orderbook_data": {},
        "ml_features": {},
        "multi_tf": {},
        "historical_vp": {},
        "derivatives": {},
        "market_context": {},
        "market_environment": {},
    }


def _event_with_latency(
    latency_ms: int,
    category: str,
    is_acceptable,
    is_stale=False,
    freshness: str = "DELAYED",
) -> dict:
    ev = _base_event()
    ev["institutional_analytics"] = {
        "quality": {
            "latency": {
                "latency_ms": latency_ms,
                "latency_category": category,
                "data_freshness": freshness,
                "is_acceptable": is_acceptable,
                "is_stale": is_stale,
            },
            # FIX (ETAPA 3): calendar sempre emitido pelo produtor
            # (institutional_analytics.py:668-672). Fixture espelha o
            # contrato para não dar falso "liquidez desconhecida".
            "calendar": {
                "expected_liquidity": "NORMAL",
            },
        }
    }
    return ev


def _summary_for_qual(qual: dict) -> dict:
    # FIX (ETAPA 3): liq é sempre emitido pelo builder quando há calendário
    # (contrato do produtor). Helper espelha o contrato para isolar latência.
    qual = dict(qual)
    qual.setdefault("liq", "NORMAL")
    return build_quality_summary({"qual": qual, "ctx": {}})


# ---------------------------------------------------------------------------
# CASO A — LATÊNCIA ACEITÁVEL
# ---------------------------------------------------------------------------


def test_caso_a_acceptable_latency_is_acceptable_true():
    ev = _event_with_latency(
        3000, "ACCEPTABLE", True, freshness="NEAR_REAL_TIME"
    )
    enrich_signal(ev)
    assert ev["data_reliability"]["latency_acceptable"] is True


# ---------------------------------------------------------------------------
# CASO B — LATÊNCIA INACEITÁVEL (POOR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "latency_ms,category,freshness",
    [
        (5212, "POOR", "DELAYED"),   # JANELA 4
        (7312, "POOR", "DELAYED"),   # JANELA 1
        (11450, "POOR", "DELAYED"),
    ],
)
def test_caso_b_unacceptable_latency_is_false(latency_ms, category, freshness):
    ev = _event_with_latency(latency_ms, category, False, freshness=freshness)
    enrich_signal(ev)
    assert ev["data_reliability"]["latency_acceptable"] is False


# ---------------------------------------------------------------------------
# CASO C — LATÊNCIA AUSENTE -> FAIL-CLOSED
# ---------------------------------------------------------------------------
# Contrato escolhido: para dado crítico de trading, ausência de informação
# de latência NÃO é promovida a "saudável". O default fail-open
# `_latency.get("is_acceptable", 1)` transformava dado ausente em
# latency_acceptable=True em 100% dos eventos. Novo contrato: sem medição
# de latência -> False (fail-closed).


def test_caso_c_missing_latency_is_fail_closed():
    ev = _base_event()
    enrich_signal(ev)
    assert ev["data_reliability"]["latency_acceptable"] is False


def test_caso_c_institutional_error_blocks_latency_is_fail_closed():
    ev = _base_event()
    ev["institutional_analytics"] = {"status": "error", "error": "boom"}
    enrich_signal(ev)
    assert ev["data_reliability"]["latency_acceptable"] is False


# ---------------------------------------------------------------------------
# CASO D — PROPAGAÇÃO ATÉ O PAYLOAD DA IA
# ---------------------------------------------------------------------------


def test_caso_d_poor_latency_payload_is_not_fully_reliable():
    ev = _event_with_latency(7312, "POOR", False, freshness="DELAYED")
    payload = build_compact_payload(ev)
    quality = payload["summary"]["quality"]
    assert quality["reliable"] is False
    assert quality["confidence_cap"] < 1.0


def test_caso_d_missing_latency_payload_cannot_claim_full_confidence():
    ev = _base_event()
    payload = build_compact_payload(ev)
    quality = payload["summary"]["quality"]
    assert quality["reliable"] is False
    assert quality["confidence_cap"] < 1.0
    assert quality["issues"], "payload sem latência não pode ter issues=[]"
    assert "plena" not in quality["note"].lower()
    assert "sem anomalias" not in quality["note"].lower()


def test_caso_d_summary_without_qual_is_fail_closed():
    summary = build_quality_summary({})
    assert summary["reliable"] is False
    assert summary["confidence_cap"] < 1.0
    assert summary["issues"]
    assert "plena" not in summary["note"].lower()


def test_caso_d_healthy_latency_keeps_full_confidence():
    import market_orchestrator.ai.payload_builder_compact as bcp
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0
    ev = _event_with_latency(50, "EXCELLENT", True, freshness="REAL_TIME")
    payload = build_compact_payload(ev)
    quality = payload["summary"]["quality"]
    assert quality["reliable"] is True
    assert quality["confidence_cap"] == 1.0


# ---------------------------------------------------------------------------
# CASO E — BOOL/INT COMPATIBILITY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_acceptable", [True, 1])
def test_caso_e_true_and_1_are_acceptable(is_acceptable):
    ev = _event_with_latency(3000, "ACCEPTABLE", is_acceptable)
    enrich_signal(ev)
    assert ev["data_reliability"]["latency_acceptable"] is True


@pytest.mark.parametrize("is_acceptable", [False, 0])
def test_caso_e_false_and_0_are_unacceptable(is_acceptable):
    ev = _event_with_latency(5212, "POOR", is_acceptable)
    enrich_signal(ev)
    assert ev["data_reliability"]["latency_acceptable"] is False


# ---------------------------------------------------------------------------
# FASE 9 — INVARIANTE: AS DUAS LATÊNCIAS NUNCA DISCORDAM
# ---------------------------------------------------------------------------


def test_invariant_latency_signals_never_disagree():
    for latency_ms, cat, acc in [
        (3000, "ACCEPTABLE", True),
        (4500, "ACCEPTABLE", True),
        (5212, "POOR", False),
        (7312, "POOR", False),
        (16000, "CRITICAL", False),
    ]:
        ev = _event_with_latency(latency_ms, cat, acc)
        enrich_signal(ev)
        src = ev["institutional_analytics"]["quality"]["latency"]
        assert bool(src["is_acceptable"]) == ev["data_reliability"]["latency_acceptable"], (
            f"latency_ms={latency_ms}: quality.latency.is_acceptable={src['is_acceptable']} "
            f"mas data_reliability.latency_acceptable={ev['data_reliability']['latency_acceptable']}"
        )


# ---------------------------------------------------------------------------
# FASE 4/3 — RELIABILITY SCORE vs LATÊNCIA
# ---------------------------------------------------------------------------
# Contrato derivado da fonte canônica (time_manager.track_data_latency):
# is_acceptable = latency_ms < 5000; is_stale = latency_ms > 15000.
# Latência inaceitável (>= 5000ms) deve reduzir reliability_score — a
# situação "unacceptable mas reliability_score = 10.0" (J1) é contradição.


def test_reliability_score_penalizes_unacceptable_latency():
    ev = _event_with_latency(7312, "POOR", False)
    enrich_signal(ev)
    assert ev["reliability_score"] <= 9.5
    assert ev["data_quality_score"] < 10.0


def test_reliability_score_penalizes_stale_latency():
    ev = _event_with_latency(16000, "CRITICAL", False, is_stale=True, freshness="STALE")
    enrich_signal(ev)
    assert ev["reliability_score"] <= 9.0


def test_reliability_score_keeps_full_score_for_acceptable_latency():
    ev = _event_with_latency(3000, "ACCEPTABLE", True, freshness="NEAR_REAL_TIME")
    # FIX (ETAPA 3): evento saudável tem orderbook_quality="live" injetado
    # pelo market_orchestrator._enrich_signal (fonte "live" do analyzer).
    # Ausência de orderbook_quality agora é fail-closed (degradado).
    ev["orderbook_quality"] = "live"
    enrich_signal(ev)
    assert ev["reliability_score"] == 10.0


# ---------------------------------------------------------------------------
# MAPEAMENTO DE CATEGORIAS CANÔNICAS (time_manager) -> CAPS DO SUMMARY
# ---------------------------------------------------------------------------
# time_manager produz EXCELLENT/GOOD/ACCEPTABLE/DEGRADED/POOR/CRITICAL.
# O resumo truncava para EXCE/GOOD/ACCE e caía no fallback 0.3
# (latência boa penalizada como desconhecida — falso negativo).


@pytest.mark.parametrize(
    "category,expected_cap",
    [
        ("EXCELLENT", 1.0),
        ("EXCE", 1.0),
        ("GOOD", 1.0),
        ("ACCEPTABLE", 0.9),
        ("ACCE", 0.9),
        ("DEGRADED", 0.7),
        ("POOR", 0.4),
        ("CRITICAL", 0.4),
    ],
)
def test_canonical_latency_categories_map_to_caps(category, expected_cap):
    summary = _summary_for_qual({"lat": category, "ms": 1500})
    assert summary["confidence_cap"] == expected_cap


def test_excellent_latency_is_reliable():
    summary = _summary_for_qual({"lat": "EXCE", "ms": 50})
    assert summary["reliable"] is True
    assert summary["confidence_cap"] == 1.0
