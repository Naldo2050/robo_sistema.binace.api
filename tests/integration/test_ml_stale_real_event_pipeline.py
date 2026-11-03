# tests/integration/test_ml_stale_real_event_pipeline.py
# -*- coding: utf-8 -*-
"""
Teste de integração ponta a ponta para neutralização de modelo ML stale.

Fluxo:
  Evento realista em SQLite temporário
  -> MLInferenceEngine.predict() (lê model_metadata_latest.json)
  -> HybridDecisionMaker.fuse_decisions()
  -> assert ml_stale=True e log de neutralização presente.

Não acessa dados/trading_bot.db nem dados/audit/.
"""

import json
import logging
import sqlite3

import pytest

from ml.inference_engine import MLInferenceEngine
from ml.hybrid_decision import HybridDecisionMaker
import ml.hybrid_decision as hybrid_mod
import config.settings as settings_mod


def test_ml_stale_real_event_neutralization(tmp_path, caplog, monkeypatch):
    """
    Testa o pipeline completo sem depender do banco operacional:
    1. Cria um evento ANALYSIS_TRIGGER realista em SQLite temporário
    2. Com HYBRID_ENABLED=True, roda o modelo XGBoost real sobre as features
    3. Engine embute valid_for_futures=False e ml_stale=True da metadata real
    4. HybridDecisionMaker detecta ml_stale=True na predição 'ok' e neutraliza
    """
    monkeypatch.setattr(settings_mod, "HYBRID_ENABLED", True)
    monkeypatch.setattr(hybrid_mod, "HYBRID_ENABLED", True)

    db_path = tmp_path / "ml_stale_events.db"
    seed_payload = {
        "epoch_ms": 1700000000000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "preco_fechamento": 65000.0,
        "volume_total": 150.0,
        "rsi": 45.0,
    }

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "event_type TEXT, "
            "payload TEXT"
            ")"
        )
        conn.execute(
            "INSERT INTO events (event_type, payload) VALUES (?, ?)",
            ("ANALYSIS_TRIGGER", json.dumps(seed_payload)),
        )
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT payload FROM events WHERE event_type = 'ANALYSIS_TRIGGER' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "ANALYSIS_TRIGGER não encontrado no SQLite temporário"
    event_payload = json.loads(row[0])

    # 1. Inferência com engine real
    engine = MLInferenceEngine()
    assert engine.valid_for_futures is False, "Metadata deveria ter valid_for_futures=False"
    assert engine.ml_stale is True, "Metadata deveria resultar em ml_stale=True"

    # Previsão direta do payload carregado do SQLite temporário (XGBoost real)
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
