"""
Testes de Neutralização de Claims de Microestrutura (Fase Pré-Captura Final).

Garante formalmente que:
A) REST-only -> no iceberg claim no payload.
B) REST-only -> no spoofing claim.
C) REST-only -> no hidden-order claim.
D) Trade fragmentation não alimenta payload["iceberg"].
E) large_orders_1h documentado como executed aggressive trades (aggTrades).
F) Prompt não contém frases categóricas sobre institucionais comprando/vendendo de varejo.
G) Absorção usa linguagem inferencial observável ("compatível com absorção passiva").
H) SMC/BOS/FVG é contextualizado como price-action, não entidade observada.
"""
import pytest
from unittest.mock import patch
from market_orchestrator.capabilities import (
    CONTINUOUS_TRADES_WS,
    POINT_IN_TIME_L2_SNAPSHOT,
    CONTINUOUS_L2,
    ORDER_CANCEL_TRACKING,
    QUEUE_REPLENISHMENT_TRACKING,
    ICEBERG_DETECTION_SUPPORTED,
    SPOOFING_DETECTION_SUPPORTED,
    HIDDEN_ORDERS_SUPPORTED,
)
from market_orchestrator.ai.payload_builder_compact import (
    _build_iceberg,
    build_compact_payload,
)
from market_orchestrator.ai.analyzer_qwen import SYSTEM_PROMPT, SYSTEM_PROMPT_LEGACY
from institutional.enricher import _register_large_trades, _build_whale_activity


def test_capability_contract_defaults():
    """Garante que o Capability Contract reflete a arquitetura REST snapshot-only."""
    assert CONTINUOUS_TRADES_WS is True
    assert POINT_IN_TIME_L2_SNAPSHOT is True
    assert CONTINUOUS_L2 is False
    assert ORDER_CANCEL_TRACKING is False
    assert QUEUE_REPLENISHMENT_TRACKING is False
    assert ICEBERG_DETECTION_SUPPORTED is False
    assert SPOOFING_DETECTION_SUPPORTED is False
    assert HIDDEN_ORDERS_SUPPORTED is False


def test_rest_only_no_iceberg_claim_in_payload():
    """A) Com REST-only (CONTINUOUS_L2=False), payload['iceberg'] NUNCA é emitido."""
    mock_event = {
        "symbol": "BTCUSDT",
        "institutional_analytics": {
            "iceberg_detector": {
                "detected": True,
                "side": "BUY",
                "estimated_size": 25.0,
            }
        },
        "whale_activity": {
            "iceberg_activity": True,
        }
    }
    # _build_iceberg direto deve retornar {}
    result = _build_iceberg(mock_event)
    assert result == {}, "Deveria ser omitido ({}), não claim positivo ou falso negativo"

    # No payload compacto, a chave 'iceberg' deve ser estritamente omitida (não iceberg=0)
    payload = build_compact_payload(mock_event)
    assert "iceberg" not in payload, "Chave 'iceberg' não deve estar presente no payload compacto"


def test_rest_only_no_spoofing_claim():
    """B) Payload não deve emitir chave de spoofing ou manipulação não comprovada."""
    mock_event = {"symbol": "BTCUSDT"}
    payload = build_compact_payload(mock_event)
    assert "spoofing" not in payload
    assert "is_spoofing" not in payload
    assert "manipulation" not in payload


def test_rest_only_no_hidden_order_claim():
    """C) Heurística de hidden orders não deve alimentar o payload destinado à IA."""
    mock_event = {
        "symbol": "BTCUSDT",
        "whale_activity": {
            "hidden_orders_detected": 1,
        }
    }
    payload = build_compact_payload(mock_event)
    assert "hidden_orders" not in payload
    assert "hidden_orders_detected" not in payload
    assert "hidden_order" not in payload


def test_trade_fragmentation_does_not_become_iceberg():
    """D) Heurística de clusters de trades fragmentados NÃO alimenta payload['iceberg']."""
    mock_event = {
        "symbol": "BTCUSDT",
        "whale_activity": {
            "iceberg_activity": True,  # cluster com 500+ microtrades
        }
    }
    res = _build_iceberg(mock_event)
    assert res == {}


def test_large_orders_documented_as_executed_trade():
    """E) Documentação de large_orders_1h atesta proveniência como executed trades (aggTrades)."""
    doc_register = _register_large_trades.__doc__ or ""
    assert "executed aggressive trades" in doc_register
    assert "aggTrades" in doc_register
    assert "resting limit orders" in doc_register

    doc_activity = _build_whale_activity.__doc__ or ""
    assert "executed aggressive trades" in doc_activity


def test_prompt_no_categorical_institutional_counterparty():
    """F) Prompt não contém frases categóricas sobre institucionais identificados comprando/vendendo de varejo."""
    forbidden = [
        "institucionais comprando tudo que varejo vende",
        "institucionais vendendo para varejo",
        "institucionais comprando de varejo",
        "whales vendendo para varejo",
        "whales comprando do varejo",
        "institucionais vendendo em cada alta",
    ]
    prompt_lower = SYSTEM_PROMPT.lower()
    for phrase in forbidden:
        assert phrase not in prompt_lower, f"Frase proibida encontrada no prompt: {phrase}"


def test_absorption_uses_inferential_language():
    """G) Absorção no prompt usa linguagem observável e inferencial sem identificar contraparte."""
    assert "compatível com absorção passiva" in SYSTEM_PROMPT
    assert "sem identificação de contraparte" in SYSTEM_PROMPT


def test_smc_not_presented_as_observed_entity():
    """H) SMC/BOS/Sweep no prompt é contextualizado como price action, não rastreamento de entidade."""
    assert "não representam rastreamento de entidade institucional" in SYSTEM_PROMPT


def test_offline_no_llm_enforcement():
    """Garante que os testes rodam 100% offline sem chamadas a APIs de LLM."""
    with patch("openai.resources.chat.completions.Completions.create", side_effect=AssertionError("No external LLM calls allowed")):
        # Mock event build is pure Python
        payload = build_compact_payload({"symbol": "BTCUSDT"})
        assert isinstance(payload, dict)


def test_all_offline_prompt_modes_neutralized():
    """Testa todos os modos de prompt ativos contra afirmações institucionais categóricas."""
    from tests.unit.test_sr_etapa5b_contract import _offline_prompt, GROQ_MODES
    prompts = {name: _offline_prompt(*args) for name, args in GROQ_MODES.items()} | {
        "nongroq-compressed": _offline_prompt("openai", "x", True, {}),
        "nongroq-default": _offline_prompt("openai", "x", False, {}),
    }
    forbidden = [
        "institucionais comprando tudo que varejo vende",
        "institucionais vendendo para varejo",
        "institucionais comprando de varejo",
        "whales vendendo para varejo",
        "whales comprando do varejo",
        "institucionais vendendo em cada alta",
    ]
    for name, ptext in prompts.items():
        low = ptext.lower()
        for phrase in forbidden:
            assert phrase not in low, f"Modo {name} contém frase proibida: {phrase}"
