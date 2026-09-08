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
    # P1-C3: ok SEM metadata de elegibilidade => inelegível (sem pu/c).
    event_with_ml = {
        "ml_prediction": {
            "status": "ok",
            "prob_up": 0.75,
            "confidence": 0.80,
        }
    }
    quant = _build_quant(event_with_ml)
    assert quant.get("ml_stale") is True
    assert "pu" not in quant
    assert quant.get("reason") == "valid_for_futures_not_true"

    # Elegível emite pu/c normalmente.
    event_ok = {"ml_prediction": {"status": "ok", "prob_up": 0.75,
                                  "confidence": 0.80,
                                  "valid_for_futures": True,
                                  "ml_stale": False}}
    quant_ok = _build_quant(event_ok)
    assert quant_ok.get("ml_stale") is False
    assert quant_ok.get("pu") == 0.75
    assert quant_ok.get("c") == 0.80

    # Sem ml_prediction (stub default: sem pu artificial)
    event_empty = {}
    quant_empty = _build_quant(event_empty)
    assert quant_empty.get("ml_stale") is True
    assert "pu" not in quant_empty


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


def test_model_metadata_training_eligibility_block():
    """P1-C2: bloco estruturado de elegibilidade; leitores antigos toleram."""
    meta_path = Path("ml/models/model_metadata_latest.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    block = meta.get("training_eligibility")
    assert isinstance(block, dict)
    assert block.get("futures") is False
    assert block.get("requires_retraining") is True
    assert isinstance(block.get("reason"), str) and block["reason"]
    # chaves antigas intactas (compat)
    assert meta.get("trained_on") == "spot"
    assert meta.get("valid_for_futures") is False
    assert "rsi" in meta.get("feature_names", [])

    from ml.inference_engine import MLInferenceEngine
    engine = MLInferenceEngine.__new__(MLInferenceEngine)
    engine.metadata_path = meta_path
    engine.valid_for_futures = False
    engine.ml_stale = True
    engine.metadata = meta
    assert engine.valid_for_futures is False
    assert engine.ml_stale is True


def test_sanitize_dmatrix_values_inf_never_reaches_model():
    """P1-C2 trava: ±Inf vira NaN (ausente nativo); resto intacto."""
    from ml.inference_engine import sanitize_dmatrix_values

    out = sanitize_dmatrix_values(
        [70000.0, 0.0, 50.0, float("inf"), float("-inf"), float("nan"), 1.0])
    assert out[0] == 70000.0 and out[1] == 0.0 and out[2] == 50.0
    assert out[3] != out[3] and out[4] != out[4] and out[5] != out[5]
    assert out[6] == 1.0
    for v in out:
        assert v != float("inf") and v != float("-inf")
