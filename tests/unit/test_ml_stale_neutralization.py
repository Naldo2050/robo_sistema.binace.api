# tests/unit/test_ml_stale_neutralization.py
# -*- coding: utf-8 -*-

import json
import os
import pytest
from pathlib import Path

from market_orchestrator.ai.payload_builder_compact import _build_quant, build_compact_payload
from ml.hybrid_decision import HybridDecisionMaker


def test_build_quant_includes_ml_stale():
    """Testa se o bloco quant/ML do compact payload inclui ml_stale: True."""
    # Com ml_prediction preenchido
    event_with_ml = {
        "ml_prediction": {
            "status": "ok",
            "prob_up": 0.75,
            "confidence": 0.80,
        }
    }
    quant = _build_quant(event_with_ml)
    assert quant.get("ml_stale") is True
    assert quant.get("pu") == 0.75
    assert quant.get("c") == 0.80

    # Sem ml_prediction (stub default)
    event_empty = {}
    quant_empty = _build_quant(event_empty)
    assert quant_empty.get("ml_stale") is True
    assert quant_empty.get("pu") == 0.5


def test_hybrid_decision_ignores_ml_when_ml_stale():
    """Testa se hybrid_decision ignora a predição do modelo quando ml_stale: True."""
    maker = HybridDecisionMaker()

    # Se ml_stale é True, mesmo com forte predição buy do ML (prob_up=0.95),
    # o ML deve ser ignorado e a decisão deve seguir apenas o LLM.
    ml_stale_pred = {
        "status": "ok",
        "prob_up": 0.95,
        "confidence": 0.90,
        "ml_stale": True,
    }
    ai_result = {
        "action": "sell",
        "confidence": 0.85,
        "sentiment": "bearish",
        "rationale": "LLM bearish signal",
    }

    decision = maker.fuse_decisions(ml_stale_pred, ai_result)
    # Como o ML é ignorado, a ação permanece a do LLM sem penalidade de conflito direcional
    assert decision.model_prob_up is None
    assert decision.action == "sell"
    assert decision.confidence == 0.85
    assert decision.source == "llm_only"


def test_hybrid_decision_ignores_ml_when_config_ml_stale(monkeypatch):
    """Testa se flag global de config ignora o ML mesmo sem flag explícita no payload."""
    import config
    monkeypatch.setattr(config, "ML_STALE", True, raising=False)

    maker = HybridDecisionMaker()
    ml_pred = {
        "status": "ok",
        "prob_up": 0.95,
        "confidence": 0.90,
    }
    ai_result = {
        "action": "sell",
        "confidence": 0.85,
        "sentiment": "bearish",
        "rationale": "LLM bearish signal",
    }

    decision = maker.fuse_decisions(ml_pred, ai_result)
    assert decision.model_prob_up is None
    assert decision.action == "sell"


def test_model_metadata_spot_invalidation():
    """Valida se model_metadata_latest.json registra trained_on: spot e valid_for_futures: False."""
    meta_path = Path("ml/models/model_metadata_latest.json")
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta.get("trained_on") == "spot"
    assert meta.get("valid_for_futures") is False
