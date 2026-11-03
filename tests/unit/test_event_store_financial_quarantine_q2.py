# -*- coding: utf-8 -*-
"""
tests/unit/test_event_store_financial_quarantine_q2.py

Suíte de Testes Herméticos Dedicada ao Gate H1-B2-Q2:
Integração do Validador Financeiro e Quarentena Forense no EventStore.
Fase 5.11 / Bloco 4 — Etapa 2 (INC5-B4-I2).

Garante:
1. Rejeição fail-closed e quarentena durável de NaN, Infinity e -Infinity em valores de preço.
2. Rejeição de tipos inválidos e booleanos disfarçados em campos monetários.
3. Canonicalização e persistência de zero negativo (-0.0 -> 0.0) sem colisão de representação.
4. Validação estrita e fail-closed de last_touched em eventos de Zona.
5. Preservação imutável de event_id e deepcopy do payload bruto em falhas para quarentena.
6. Idempotência transacional estrita em reenvios idênticos.
7. Detecção de colisões divergentes (no lote e com histórico) com registro em quarentena (COLLISION).
8. Cobertura completa da matriz de produtores (Absorção, Exaustão, Zona, ANALYSIS_TRIGGER).
"""

import copy
import json
import math
import os
import sqlite3
from pathlib import Path
import pytest

from database.event_store import (
    EventStore,
    BatchSaveResult,
    IntegrityCollisionError,
)
from common.financial_quarantine import FinancialQuarantine, decode_typed_payload


@pytest.fixture
def hermetic_env(tmp_path):
    """Cria um ambiente hermético isolado com EventStore e FinancialQuarantine em tmp_path."""
    db_file = tmp_path / "test_trading_bot.db"
    quar_dir = tmp_path / "quarantine"
    quarantine = FinancialQuarantine(quarantine_dir=quar_dir)
    store = EventStore(db_path=str(db_file), quarantine=quarantine)
    return {
        "store": store,
        "quarantine": quarantine,
        "db_file": db_file,
        "quar_dir": quar_dir,
    }


# ==============================================================================
# TESTE Q2-01: Rejeição e Quarentena de NaN em Preço (Absorção)
# ==============================================================================
def test_q2_01_nan_in_preco_fechamento_absorcao_quarantined(hermetic_env):
    store = hermetic_env["store"]
    quar_dir = hermetic_env["quar_dir"]

    original_event = {
        "event_id": "abs_nan_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": float("nan"),
        "volume_total": 150.0,
        "delta": 25.0,
        "epoch_ms": 1700000000000,
    }
    input_copy = copy.deepcopy(original_event)

    res = store.save_batch([original_event])

    # Asserções de Resultado do Batch
    assert res.commit_status == "FAILED"
    assert res.total_received == 1
    assert res.persisted == 0
    assert res.rejected_pre_commit == 1
    assert res.rows_inserted == 0

    # Imutabilidade do dicionário de entrada
    assert math.isnan(original_event["preco_fechamento"])
    assert original_event["event_id"] == input_copy["event_id"]

    # Verificação no SQLite: zero linhas inseridas
    with sqlite3.connect(str(store.db_path)) as conn:
        cnt = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        assert cnt == 0

    # Verificação na Quarentena: arquivo em invalid/
    invalid_dir = quar_dir / "invalid"
    assert invalid_dir.exists()
    q_files = list(invalid_dir.glob("invalid_events_*.jsonl"))
    assert len(q_files) == 1
    with open(q_files[0], "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    rec = records[0]
    assert rec["category"] == "INVALID"
    assert rec["rejected_field"] == "preco_fechamento"
    assert "NAN" in rec["rejection_reason"]
    decoded_payload = decode_typed_payload(rec["original_payload"])
    assert math.isnan(decoded_payload["preco_fechamento"])
    assert decoded_payload["symbol"] == "BTCUSDT"


# ==============================================================================
# TESTE Q2-02: Rejeição e Quarentena de Infinity e -Infinity (Exaustão)
# ==============================================================================
def test_q2_02_infinity_in_preco_fechamento_exaustao_quarantined(hermetic_env):
    store = hermetic_env["store"]
    quar_dir = hermetic_env["quar_dir"]

    ev_inf = {
        "event_id": "ex_inf_01",
        "tipo_evento": "Exaustão",
        "symbol": "BTCUSDT",
        "preco_fechamento": float("inf"),
        "delta": -10.5,
        "epoch_ms": 1700000001000,
    }
    ev_neginf = {
        "event_id": "ex_neginf_01",
        "tipo_evento": "Exaustão",
        "symbol": "BTCUSDT",
        "preco_fechamento": float("-inf"),
        "delta": 10.5,
        "epoch_ms": 1700000002000,
    }

    res = store.save_batch([ev_inf, ev_neginf])
    assert res.commit_status == "FAILED"
    assert res.total_received == 2
    assert res.persisted == 0
    assert res.rejected_pre_commit == 2

    # Banco de dados deve ter 0 linhas
    with sqlite3.connect(str(store.db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0

    # Quarentena deve conter 2 registros
    invalid_files = list((quar_dir / "invalid").glob("*.jsonl"))
    assert len(invalid_files) == 1
    with open(invalid_files[0], "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    assert records[0]["rejected_field"] == "preco_fechamento"
    assert records[1]["rejected_field"] == "preco_fechamento"


# ==============================================================================
# TESTE Q2-03: Rejeição de Booleano em Campo Monetário
# ==============================================================================
def test_q2_03_boolean_in_price_field_rejected(hermetic_env):
    store = hermetic_env["store"]
    ev_bool = {
        "event_id": "bool_price_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": True,
        "epoch_ms": 1700000003000,
    }
    res = store.save_batch([ev_bool])
    assert res.rejected_pre_commit == 1
    assert res.persisted == 0


# ==============================================================================
# TESTE Q2-04: Rejeição de Preço Negativo ou Zero
# ==============================================================================
@pytest.mark.parametrize("invalid_price", [-100.0, 0.0, -0.0000001])
def test_q2_04_negative_or_zero_price_rejected(hermetic_env, invalid_price):
    store = hermetic_env["store"]
    ev = {
        "event_id": f"neg_price_{invalid_price}",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": invalid_price,
        "epoch_ms": 1700000004000,
    }
    res = store.save_batch([ev])
    assert res.rejected_pre_commit == 1
    assert res.persisted == 0


# ==============================================================================
# TESTE Q2-05: Canonicalização de Zero Negativo (-0.0) em Delta (Exaustão)
# ==============================================================================
def test_q2_05_canonicalizacao_zero_negativo_delta(hermetic_env):
    store = hermetic_env["store"]

    # Delta -0.0 em Exaustão é válido (delta zero é admitido), mas deve ser canonicalizado para 0.0
    ev = {
        "event_id": "ex_negzero_01",
        "tipo_evento": "Exaustão",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50000.0,
        "delta": -0.0,
        "epoch_ms": 1700000005000,
    }
    res = store.save_batch([ev])
    assert res.commit_status == "COMMITTED"
    assert res.persisted == 1
    assert res.rows_inserted == 1

    # Verificar no SQLite que o payload armazenado tem delta 0.0 positivo
    with sqlite3.connect(str(store.db_path)) as conn:
        row = conn.execute("SELECT payload FROM events WHERE window_id = ''").fetchone()
        payload = json.loads(row[0])
        assert payload["delta"] == 0.0
        assert f"{payload['delta']:.8f}" == "0.00000000"


# ==============================================================================
# TESTE Q2-06: Validação Fail-Closed de last_touched em Zona
# ==============================================================================
@pytest.mark.parametrize("invalid_touch", [True, False, -100, float("nan"), "data-invalida", "   "])
def test_q2_06_last_touched_invalido_zona_rejeitado(hermetic_env, invalid_touch):
    store = hermetic_env["store"]
    ev = {
        "event_id": "zone_inv_touch_01",
        "tipo_evento": "Zona",
        "symbol": "BTCUSDT",
        "top": 51000.0,
        "bottom": 50000.0,
        "last_touched": invalid_touch,
        "epoch_ms": 1700000006000,
    }
    res = store.save_batch([ev])
    assert res.rejected_pre_commit == 1
    assert res.persisted == 0


# ==============================================================================
# TESTE Q2-07: Idempotência com Reenvio Idêntico
# ==============================================================================
def test_q2_07_idempotencia_reenvio_identico(hermetic_env):
    store = hermetic_env["store"]
    ev = {
        "event_id": "idem_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50500.0,
        "volume_total": 100.0,
        "epoch_ms": 1700000007000,
    }

    # Primeiro envio
    res1 = store.save_batch([ev])
    assert res1.commit_status == "COMMITTED"
    assert res1.rows_inserted == 1
    assert res1.duplicates_skipped == 0
    assert res1.persisted == 1

    # Segundo envio idêntico
    res2 = store.save_batch([ev])
    assert res2.commit_status == "COMMITTED"
    assert res2.rows_inserted == 0
    assert res2.duplicates_skipped == 1
    assert res2.persisted == 1

    # Banco deve conter exatamente 1 registro
    with sqlite3.connect(str(store.db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1


# ==============================================================================
# TESTE Q2-08: Colisão Divergente no Mesmo Batch Envia para Quarentena
# ==============================================================================
def test_q2_08_colisao_divergente_mesmo_batch_quarentena(hermetic_env):
    store = hermetic_env["store"]
    quar_dir = hermetic_env["quar_dir"]

    ev1 = {
        "event_id": "col_batch_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50000.0,
        "epoch_ms": 1700000008000,
    }
    ev2 = {
        "event_id": "col_batch_01",  # Mesmo ID
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50100.0,  # Preço divergente!
        "epoch_ms": 1700000008000,
    }

    res = store.save_batch([ev1, ev2])
    assert res.commit_status == "FAILED"
    assert res.error_type == "IntegrityCollisionError"
    assert res.persisted == 0

    # Quarentena deve registrar a colisão
    col_files = list((quar_dir / "collision").glob("*.jsonl"))
    assert len(col_files) == 1
    with open(col_files[0], "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    assert len(recs) == 1
    assert recs[0]["category"] == "COLLISION"
    assert recs[0]["rejected_field"] == "event_id"


# ==============================================================================
# TESTE Q2-09: Colisão Divergente com Registro Histórico Envia para Quarentena
# ==============================================================================
def test_q2_09_colisao_divergente_historico_sqlite_quarentena(hermetic_env):
    store = hermetic_env["store"]
    quar_dir = hermetic_env["quar_dir"]

    ev_hist = {
        "event_id": "col_hist_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50000.0,
        "epoch_ms": 1700000009000,
    }
    res1 = store.save_batch([ev_hist])
    assert res1.commit_status == "COMMITTED"

    # Segundo envio com mesmo ID e payload divergente
    ev_div = {
        "event_id": "col_hist_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50999.0,
        "epoch_ms": 1700000009000,
    }
    res2 = store.save_batch([ev_div])
    assert res2.commit_status == "FAILED"
    assert res2.error_type == "IntegrityCollisionError"
    assert res2.persisted == 0

    col_files = list((quar_dir / "collision").glob("*.jsonl"))
    assert len(col_files) == 1
    with open(col_files[0], "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    assert len(recs) == 1
    assert recs[0]["category"] == "COLLISION"
    assert recs[0]["rejected_field"] == "event_id"


# ==============================================================================
# TESTE Q2-10: Matriz dos 4 Produtores Registrados (Sucesso e Rejeição)
# ==============================================================================
def test_q2_10_matriz_produtores_sucesso_e_rejeicao(hermetic_env):
    store = hermetic_env["store"]

    # 4 eventos válidos cobrindo Absorção, Exaustão, Zona e ANALYSIS_TRIGGER
    valid_batch = [
        {
            "event_id": "prod_abs_01",
            "tipo_evento": "Absorção",
            "symbol": "BTCUSDT",
            "preco_fechamento": 51000.0,
            "epoch_ms": 1700000010000,
        },
        {
            "event_id": "prod_ex_01",
            "tipo_evento": "Exaustão",
            "symbol": "BTCUSDT",
            "preco_fechamento": 51100.0,
            "delta": 15.0,
            "epoch_ms": 1700000011000,
        },
        {
            "event_id": "prod_zone_01",
            "tipo_evento": "Zona",
            "symbol": "BTCUSDT",
            "top": 51500.0,
            "bottom": 51200.0,
            "last_touched": 1700000012000,
            "epoch_ms": 1700000012000,
        },
        {
            "event_id": "prod_trig_01",
            "tipo_evento": "ANALYSIS_TRIGGER",
            "symbol": "BTCUSDT",
            "anchor_price": 51300.0,
            "epoch_ms": 1700000013000,
        },
    ]

    res = store.save_batch(valid_batch)
    assert res.commit_status == "COMMITTED"
    assert res.total_received == 4
    assert res.persisted == 4
    assert res.rows_inserted == 4
    assert res.rejected_pre_commit == 0

    # Lote com os 4 produtores, mas cada um com uma falha de preço/nan
    invalid_batch = [
        {"event_id": "bad_abs", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": float("nan")},
        {"event_id": "bad_ex", "tipo_evento": "Exaustão", "symbol": "BTCUSDT", "preco_fechamento": -50.0},
        {"event_id": "bad_zone", "tipo_evento": "Zona", "symbol": "BTCUSDT", "top": float("inf"), "bottom": 50000.0},
        {"event_id": "bad_trig", "tipo_evento": "ANALYSIS_TRIGGER", "symbol": "BTCUSDT", "anchor_price": True},
    ]

    res_inv = store.save_batch(invalid_batch)
    assert res_inv.commit_status == "FAILED"
    assert res_inv.rejected_pre_commit == 4
    assert res_inv.persisted == 0

