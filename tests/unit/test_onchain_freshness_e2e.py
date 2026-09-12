# tests/unit/test_onchain_freshness_e2e.py
"""
E2E de contrato FASE C (freshness até o payload final).

Cadeia: snapshot (OnchainUpdater) -> DataEnricher -> evento
(advanced_analysis) -> build_compact_payload -> guardrail_rewrap ->
groq summary (payload FINAL ao modelo).

Exige:
  - onchain_status/onchain_age_seconds/source presentes até o final;
  - stale_usable: valor aparece MAS marcado stale + idade;
  - unavailable/além do usable: nenhum valor como evidência atual;
  - last_error / fetched_monotonic NUNCA no evento/payload;
  - PFIX-LOW: coverage deriva de field_status (P04), nunca de freshness
    (fresh/stale com mesma field_status => mesma coverage).
"""

import time

from data_processing.data_enricher import DataEnricher
from fetchers.onchain_updater import OnchainSnapshot, OnchainUpdater
from institutional.enricher import enrich_signal
from market_orchestrator.ai.analyzer_qwen import AIAnalyzer
from market_orchestrator.ai.llm_payload_guardrail import guardrail_rewrap
from market_orchestrator.ai.payload_builder_compact import build_compact_payload


def _updater_with_ages(fast_age_s, slow_age_s, field_status=None):
    """Updater com snapshot envelhecido por grupo, sem rede."""
    base_mono = time.monotonic()
    wall_ms = int(time.time() * 1000)
    updater = OnchainUpdater(monotonic_fn=lambda: base_mono)
    updater._snapshot = OnchainSnapshot(
        fast={"mempool_size": 25000, "fees_fastest_sat_vb": 3},
        slow={"difficulty": 145.04, "hash_rate": 1000.0},
        fast_fetched_at_ms=wall_ms - int(fast_age_s * 1000),
        slow_fetched_at_ms=wall_ms - int(slow_age_s * 1000),
        fast_fetched_monotonic=base_mono - fast_age_s,
        slow_fetched_monotonic=base_mono - slow_age_s,
        last_error=None,
        field_status=dict(field_status) if field_status else {},
    )
    return updater


def _event():
    return {
        "symbol": "BTCUSDT",
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 79421.2,
        "epoch_ms": 1700000060000,
        "raw_event": {
            "preco_fechamento": 79421.2,
            "volume_total": 3.268,
            "symbol": "BTCUSDT",
            "multi_tf": {},
            "timestamp_utc": "2026-09-07T12:41:05+00:00",
        },
    }


def _enrich(updater):
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"}, onchain_updater=updater)
    event = _event()
    enricher.enrich_event_with_advanced_analysis(event)
    return event


def _to_final(event):
    compact = build_compact_payload(event)
    rewrapped = guardrail_rewrap(compact)
    return AIAnalyzer._build_groq_payload_summary(rewrapped["ai_payload"])


def test_fresh_reaches_final_payload():
    event = _enrich(_updater_with_ages(10.0, 100.0))
    adv = event["raw_event"]["advanced_analysis"]
    assert adv["onchain_status"] == "fresh"
    assert adv["onchain_age_seconds"] is not None
    assert "last_error" not in adv
    assert "fetched_monotonic" not in adv

    compact = build_compact_payload(event)
    assert compact["onchain"]["st"] == "fresh"
    assert compact["onchain"]["age"] is not None
    assert compact["onchain"]["mempool_sz"] == 25000

    final = _to_final(event)
    assert final["onchain"]["st"] == "fresh"
    assert final["onchain"]["mempool_sz"] == 25000


def test_stale_values_marked_with_age():
    event = _enrich(_updater_with_ages(400.0, 100.0))
    adv = event["raw_event"]["advanced_analysis"]
    assert adv["onchain_status"] == "stale"
    # Valor stale_usable pode aparecer, mas marcado + com idade.
    assert adv["onchain_metrics"]["mempool_size"] == 25000
    compact = build_compact_payload(event)
    assert compact["onchain"]["st"] == "stale"
    assert compact["onchain"]["age"] is not None
    final = _to_final(event)
    assert final["onchain"]["st"] == "stale"


def test_unavailable_has_no_values_as_evidence():
    event = _enrich(_updater_with_ages(100000.0, 100000.0))
    adv = event["raw_event"]["advanced_analysis"]
    assert adv["onchain_status"] == "unavailable"
    assert adv["onchain_metrics"].get("mempool_size") is None
    assert adv["onchain_metrics"].get("difficulty") is None
    compact = build_compact_payload(event)
    assert compact["onchain"]["st"] == "unavailable"
    assert "mempool_sz" not in compact["onchain"]
    final = _to_final(event)
    assert "mempool_sz" not in final.get("onchain", {})


def test_coverage_derives_from_field_status_not_freshness():
    # PFIX-LOW: mesma field_status parcial => mesma coverage, fresh ou stale.
    # field_status cobre só os campos do snapshot; demais contam como ausentes.
    fs = {"mempool_size": "VALID", "fees_fastest_sat_vb": "API_ERROR",
          "difficulty": "VALID", "hash_rate": "REAL_ZERO"}
    event = _enrich(_updater_with_ages(400.0, 100.0, field_status=fs))
    out = enrich_signal(event)
    assert out["data_reliability"]["onchain_coverage"] == "partial"

    event = _enrich(_updater_with_ages(10.0, 100.0, field_status=fs))
    out = enrich_signal(event)
    assert out["data_reliability"]["onchain_coverage"] == "partial"
    assert out["data_reliability"]["onchain_coverage_pct"] == round(
        3 / 15 * 100, 1)

    # Sem field_status (legado): UNKNOWN, nunca "full" silencioso.
    event = _enrich(_updater_with_ages(10.0, 100.0))
    out = enrich_signal(event)
    assert out["data_reliability"]["onchain_coverage"] == "unknown"
