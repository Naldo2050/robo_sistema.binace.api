# tests/unit/test_llm_prompt_v3_contract.py
# -*- coding: utf-8 -*-
"""
Suíte de testes para P2-F2: Semantic LLM Prompt Contract v3.

Cobre:
- Ausência rigorosa de linguagem proibida (double counting, pseudo-probabilidades, trades forçados)
- Ausência de campos legados perigosos no schema v3
- Aceitação expressa de respostas de espera (WAIT, INSUFFICIENT_DATA)
- Ausência de obrigatoriedade de confiança contínua arbitrária
- Verificação do estado padrão da feature flag LLM_SEMANTIC_PAYLOAD_V3_ENABLED (default OFF).
"""

import pytest

from config import settings
from market_orchestrator.ai.semantic_payload_builder_v3 import (
    build_semantic_payload_v3,
)
from market_orchestrator.ai.semantic_prompt_v3 import (
    SEMANTIC_PROMPT_VERSION,
    SYSTEM_PROMPT_SEMANTIC_V3,
)


# ==============================================================================
# 1. FORBIDDEN PROMPT LANGUAGE AUDIT
# ==============================================================================

def test_semantic_prompt_v3_forbids_double_counting_and_false_probabilities():
    """
    Garante que o System Prompt v3 não contém frases ou instruções que induzam:
    - contagem de confluências ingênua ("4+ confluências", "conte fatores")
    - probabilidade matemática ilusória ("probabilidade matemática", "probabilidades matemáticas")
    - obrigação de dar opinião em ruído ("sempre dê uma opinião", "sempre dê opiniao")
    - votação por maioria ("maioria dos sinais", "maioria dos votos").
    """
    forbidden_phrases = [
        "4+ confluências",
        "4+ confluencias",
        "conte fatores",
        "probabilidade matemática",
        "probabilidades matemáticas",
        "probabilidade matematica",
        "probabilidades matematicas",
        "sempre dê uma opinião",
        "sempre de uma opiniao",
        "maioria dos sinais",
        "maioria dos votos",
    ]

    prompt_lower = SYSTEM_PROMPT_SEMANTIC_V3.lower()

    for phrase in forbidden_phrases:
        assert phrase not in prompt_lower, f"Frase proibida '{phrase}' encontrada no SYSTEM_PROMPT_SEMANTIC_V3!"


# ==============================================================================
# 2. SCHEMA V3 FORBIDDEN FIELDS AUDIT
# ==============================================================================

def test_semantic_payload_v3_has_no_dangerous_legacy_fields():
    """
    Testa que o payload v3 não exporta campos legados perigosos:
    - slip_b / slip_s (slippage x100)
    - prob_trend / prob_rev / prob_break (falsas probabilidades de regime)
    - bsr como evidência direcional independente.
    """
    dummy_event = {
        "symbol": "BTCUSDT",
        "fluxo_continuo": {
            "order_flow": {
                "flow_imbalance": 0.5,
                "buy_sell_ratio": {"buy_sell_ratio": 3.0},
            }
        },
        "orderbook_data": {
            "imbalance": 0.2,
        },
        "regime_analysis": {
            "current_regime": "TRENDING",
            "regime_probabilities": {
                "trending": 0.8,
                "mean_reverting": 0.1,
                "breakout": 0.1,
            },
        },
        "market_impact": {
            "slippage_matrix": {
                "100k_usd": {"buy": 2.0, "sell": 2.0}
            }
        },
    }

    payload = build_semantic_payload_v3(dummy_event, symbol="BTCUSDT")

    # 1. Banimento de slip_b / slip_s
    assert "slip_b" not in payload["execution_context"]
    assert "slip_s" not in payload["execution_context"]

    # 2. Banimento de prob_* no regime
    reg = payload["non_voting_context"]["regime"]
    assert "prob_trend" not in reg
    assert "prob_rev" not in reg
    assert "prob_break" not in reg

    # 3. Banimento de bsr na lista de evidências direcionais ativas
    dir_items = [e["field_id"] for e in payload["directional_evidence"]["items"]]
    assert "flow.buy_sell_ratio" not in dir_items


# ==============================================================================
# 3. WAIT & INSUFFICIENT_DATA ARE EXPLICITLY PERMITTED
# ==============================================================================

def test_prompt_v3_permits_wait_and_insufficient_data():
    """Valida que o prompt v3 instrui e permite explicitamente WAIT e INSUFFICIENT_DATA."""
    prompt = SYSTEM_PROMPT_SEMANTIC_V3

    assert "INSUFFICIENT_DATA" in prompt
    assert "MIXED_DIRECTIONS" in prompt
    assert "WAIT" in prompt
    assert "OBSERVE" in prompt
    assert 'action="WAIT"' in prompt or 'action="wait"' in prompt.lower()


# ==============================================================================
# 4. NO MANDATORY CONFIDENCE REQUIREMENT
# ==============================================================================

def test_prompt_v3_response_schema_has_no_arbitrary_numeric_confidence():
    """
    Garante que o schema de resposta JSON v3 não exige campo 'confidence: 0.0-1.0'
    que forçava o LLM a inventar números aleatórios.
    """
    prompt = SYSTEM_PROMPT_SEMANTIC_V3

    # O schema do formato da resposta não deve pedir confidence
    # Deve pedir assessment, action, rationale, execution_feasibility, data_sufficiency
    assert '"confidence":' not in prompt
    assert '"assessment":' in prompt
    assert '"action":' in prompt
    assert '"execution_feasibility":' in prompt
    assert '"data_sufficiency":' in prompt


# ==============================================================================
# 5. FEATURE FLAG DEFAULT STATE
# ==============================================================================

def test_semantic_payload_v3_feature_flag_is_off_by_default():
    """
    Garante que a feature flag LLM_SEMANTIC_PAYLOAD_V3_ENABLED está
    desligada (False) por padrão neste commit, preservando a produção intacta.
    """
    assert settings.LLM_SEMANTIC_PAYLOAD_V3_ENABLED is False
