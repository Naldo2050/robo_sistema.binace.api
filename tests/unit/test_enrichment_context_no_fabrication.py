# tests/unit/test_enrichment_context_no_fabrication.py
"""
Regressão FASE A (P0 integridade): ausência de dado não pode gerar conclusão.

Bug em market_orchestrator/ai/ai_enrichment_context.py:
  - exchange_netflow ausente -> "distribution" (fabricado; o produtor fixa 0.0
    sem API paga e o filtro remove a chave, então SEMPRE "distribution").
  - put_call_ratio ausente -> "bullish" (fabricado; stub estático).
  - current_volatility ausente -> "low_vol" (fabricado).
  - Valor explícito None -> TypeError (sem try no caminho).

Contrato exigido:
  - ausente/None/NaN/±Inf -> "unknown" (nunca conclusão direcional);
  - zero legítimo presente continua zero (netflow 0.0 -> "neutral", valor 0.0
    preservado nos campos, nunca null);
  - presente válido mantém a classificação anterior.
"""

import pytest

from market_orchestrator.ai.ai_enrichment_context import (
    build_enriched_ai_context,
)


def _raw(**advanced):
    return {"advanced_analysis": advanced}


def test_netflow_missing_is_not_distribution():
    ctx = build_enriched_ai_context(_raw(onchain_metrics={"hash_rate": 1.0}))
    assert ctx["onchain_context"]["sentiment"] == "unknown"


def test_netflow_none_is_not_distribution_nor_crash():
    ctx = build_enriched_ai_context(
        _raw(onchain_metrics={"exchange_netflow": None})
    )
    assert ctx["onchain_context"]["sentiment"] == "unknown"


def test_pcr_missing_or_none_is_not_bullish():
    # Seção vazia => seção ausente (sem fabricação); chave None => "unknown".
    ctx = build_enriched_ai_context(_raw(options_metrics={}))
    assert ctx.get("options_context", {}).get("sentiment", "unknown") == "unknown"
    ctx = build_enriched_ai_context(
        _raw(options_metrics={"put_call_ratio": None})
    )
    assert ctx["options_context"]["sentiment"] == "unknown"


def test_volatility_missing_or_none_is_not_low_vol():
    # Seção vazia => seção ausente (sem fabricação); chave None => "unknown".
    ctx = build_enriched_ai_context(_raw(adaptive_thresholds={}))
    assert ctx.get("risk_context", {}).get("market_regime", "unknown") == "unknown"
    ctx = build_enriched_ai_context(
        _raw(adaptive_thresholds={"current_volatility": None})
    )
    assert ctx["risk_context"]["market_regime"] == "unknown"


def test_legit_zero_stays_zero_and_neutral():
    ctx = build_enriched_ai_context(
        _raw(onchain_metrics={"exchange_netflow": 0.0})
    )
    assert ctx["onchain_context"]["exchange_netflow"] == 0.0
    assert ctx["onchain_context"]["sentiment"] == "neutral"


def test_valid_values_keep_previous_classification():
    ctx = build_enriched_ai_context(
        _raw(onchain_metrics={"exchange_netflow": -5.0})
    )
    assert ctx["onchain_context"]["sentiment"] == "accumulation"
    ctx = build_enriched_ai_context(
        _raw(onchain_metrics={"exchange_netflow": 5.0})
    )
    assert ctx["onchain_context"]["sentiment"] == "distribution"

    ctx = build_enriched_ai_context(
        _raw(options_metrics={"put_call_ratio": 1.5})
    )
    assert ctx["options_context"]["sentiment"] == "bearish"
    ctx = build_enriched_ai_context(
        _raw(options_metrics={"put_call_ratio": 0.5})
    )
    assert ctx["options_context"]["sentiment"] == "bullish"

    ctx = build_enriched_ai_context(
        _raw(adaptive_thresholds={"current_volatility": 0.05})
    )
    assert ctx["risk_context"]["market_regime"] == "high_vol"
    ctx = build_enriched_ai_context(
        _raw(adaptive_thresholds={"current_volatility": 0.0})
    )
    assert ctx["risk_context"]["market_regime"] == "low_vol"


def test_confluence_none_does_not_crash():
    ctx = build_enriched_ai_context(
        _raw(price_targets=[{"confidence": None, "weight": None,
                             "source": "x"}])
    )
    assert ctx["targets_context"]["total_targets"] == 1
