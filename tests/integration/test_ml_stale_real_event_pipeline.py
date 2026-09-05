# tests/integration/test_ml_stale_real_event_pipeline.py
# -*- coding: utf-8 -*-
"""
Teste de integração ponta a ponta para neutralização de modelo ML stale.
Fluxo real:
  Evento real do DB (dados/trading_bot.db)
  -> MLInferenceEngine.predict() (lê model_metadata_latest.json)
  -> HybridDecisionMaker.fuse_decisions()
  -> assert model_ok is False e log de neutralização presente.
"""

import json
import logging
import sqlite3
from pathlib import Path
import pytest

from ml.inference_engine import MLInferenceEngine
from ml.hybrid_decision import HybridDecisionMaker
import ml.hybrid_decision as hybrid_mod
import config.settings as settings_mod


def test_ml_stale_real_event_neutralization(caplog, monkeypatch):
    """
    Testa pipeline completo sem injeção manual de flags:
    1. Lê evento real gravado na coleta em dados/trading_bot.db
    2. Com HYBRID_ENABLED=True, roda o modelo XGBoost real sobre as features reais
    3. Engine embute valid_for_futures=False e ml_stale=True a partir da metadata real
    4. HybridDecisionMaker detecta ml_stale=True na predição 'ok' e neutraliza com log
    """
    monkeypatch.setattr(settings_mod, "HYBRID_ENABLED", True)
    monkeypatch.setattr(hybrid_mod, "HYBRID_ENABLED", True)

    db_path = Path("dados/trading_bot.db")
    assert db_path.exists(), f"Banco de dados da coleta não encontrado em {db_path}"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT payload FROM events WHERE event_type = 'ANALYSIS_TRIGGER' LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Nenhum evento ANALYSIS_TRIGGER encontrado no DB de coleta"
    event_payload = json.loads(row[0])

    # 1. Inferência com engine real
    engine = MLInferenceEngine()
    assert engine.valid_for_futures is False, "Metadata deveria ter valid_for_futures=False"
    assert engine.ml_stale is True, "Metadata deveria resultar em ml_stale=True"

    # Previsão direta do payload real do banco de dados (executa XGBoost real)
    ml_prediction = engine.predict(event_payload)
    assert ml_prediction.get("status") == "ok", f"Inference engine falhou: {ml_prediction}"
    assert "ml_stale" in ml_prediction, "predict() deve retornar ml_stale"
    assert ml_prediction["ml_stale"] is True
    assert ml_prediction.get("valid_for_futures") is False

    # 2. Tomada de decisão híbrida com HybridDecisionMaker real
    decision_maker = HybridDecisionMaker()
    ai_result = {
        "action": "buy",
        "confidence": 0.85,
        "sentiment": "bullish",
        "rationale": "Teste de integração com sinal de alta",
        "key_factors": ["orderbook_imbalance", "cvd_absorption"],
    }

    with caplog.at_level(logging.WARNING):
        result = decision_maker.fuse_decisions(
            ml_prediction=ml_prediction,
            ai_result=ai_result,
        )

    # 3. Asserts
    # Log de neutralização DEVE estar presente no HybridDecisionMaker
    stale_logs = [
        r.message for r in caplog.records
        if "ML neutralizado" in r.message
    ]
    assert len(stale_logs) > 0, f"Log de neutralização do ML deve ser emitido. Logs capturados: {[r.message for r in caplog.records]}"

    # O resultado deve manter ação da LLM pura sem que o ML interfira
    assert result is not None
    assert result.action == "buy"
    # A confiança final deve ser a da LLM pura (85%), sem fusão com modelo ML neutralizado
    assert result.confidence == pytest.approx(0.85, abs=0.01)
