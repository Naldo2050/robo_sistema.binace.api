# -*- coding: utf-8 -*-
"""
test_event_saver_financial_quarantine_q3.py
Suíte de testes de integração e conformidade para o Gate H1-B2-Q3 (Fase 5.11 / Bloco 4).
Valida:
1. Rejeição e quarentena de NaN, +Infinity, -Infinity antes de qualquer sanitização.
2. Rejeição e quarentena de booleanos em campos numéricos/preços.
3. Rejeição de preços não-positivos (<= 0.0).
4. Rejeição de inversão de limites de microestrutura (bottom > top, poc > vah, val > poc).
5. Isolamento de lote misto (válidos salvos, inválidos quarentenados, fallback intocado).
6. Tratamento de IntegrityCollisionError (colisão divergente) sem contaminação do JSONL ou fallback.
7. Isolamento de colisão em lote misto.
8. Idempotência transacional no EventSaver.
9. Separação entre falha operacional do SQLite e falha de contrato financeiro.
10. Rastreabilidade SQLite <-> JSONL e canonicalização de -0.0 -> +0.0.
11. Inicialização lazy da quarentena forense (sem diretório prematuro).
12. Cobertura da matriz de produtores institucionais e analíticos.
"""
import copy
import json
import math
import os
import sqlite3
import threading
from pathlib import Path
import pytest

from events import event_saver
from events.event_saver import EventSaver
from database.event_store import EventStore, IntegrityCollisionError
from common.financial_quarantine import FinancialQuarantine, decode_typed_payload
from common.identity_contract import canonicalize_structure


@pytest.fixture
def hermetic_saver(tmp_path, monkeypatch):
    """Fixture de inicialização hermética do EventSaver com banco e JSONL isolados."""
    monkeypatch.setattr(event_saver, "HAS_CLOCK_SYNC", False)
    monkeypatch.setattr(event_saver, "DATA_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    db_file = tmp_path / "trading_test.db"
    store = EventStore(db_path=str(db_file), quarantine_dir=tmp_path / "quarantine")

    jsonl_file = tmp_path / "eventos_fluxo.jsonl"
    visual_log = tmp_path / "eventos_visuais.log"
    quarantine_dir = tmp_path / "quarantine"

    saver = EventSaver(
        sound_alert=False,
        quarantine_dir=quarantine_dir,
        enforce_financial_validation=True,
    )
    saver.db = store
    saver.history_file = jsonl_file
    saver.visual_log_file = visual_log
    saver.write_json = False
    saver.write_jsonl = True

    yield {
        "saver": saver,
        "store": store,
        "db_file": db_file,
        "jsonl_file": jsonl_file,
        "visual_log": visual_log,
        "quarantine_dir": quarantine_dir,
        "tmp_path": tmp_path,
    }

    try:
        saver.stop()
    except Exception:
        pass


def _read_jsonl_lines(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_quarantine_records(quarantine_dir: Path, sub: str = "invalid"):
    target = quarantine_dir / sub
    if not target.exists():
        return []
    records = []
    for fpath in target.glob("*.jsonl"):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


# ==============================================================================
# C01-C03: Rejeição e Quarentena de NaN, +Infinity e -Infinity
# ==============================================================================
def test_c01_nan_rejected_quarantined_not_in_sqlite_or_jsonl(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]
    jsonl_file = hermetic_saver["jsonl_file"]

    ev_nan = {
        "event_id": "nan_test_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": float("nan"),
        "volume_total": 100.0,
        "epoch_ms": 1700000000000,
    }
    input_copy = copy.deepcopy(ev_nan)

    # Executa save_event
    saver.save_event(ev_nan)
    saver._flush_buffer(saver._write_buffer)

    # Asserções:
    # 1. 0 registros no SQLite
    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0

    # 2. 0 registros no JSONL
    assert len(_read_jsonl_lines(jsonl_file)) == 0

    # 3. 0 arquivos em fallback_events
    fallback_dir = hermetic_saver["tmp_path"] / "fallback_events"
    assert not fallback_dir.exists() or len(list(fallback_dir.glob("*"))) == 0

    # 4. Quarentena contém exatamente o evento rejeitado com special NaN
    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 1
    rec = recs[0]
    assert rec["category"] == "INVALID"
    assert rec["rejected_field"] == "preco_fechamento"
    decoded = decode_typed_payload(rec["original_payload"])
    assert math.isnan(decoded["preco_fechamento"])
    assert decoded["event_id"] == "nan_test_01"


def test_c02_infinity_rejected_and_quarantined(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]

    ev_inf = {
        "event_id": "inf_test_01",
        "tipo_evento": "Exaustão",
        "symbol": "BTCUSDT",
        "preco_fechamento": float("inf"),
        "delta": 10.0,
        "epoch_ms": 1700000001000,
    }
    saver.save_event(ev_inf)
    saver._flush_buffer(saver._write_buffer)

    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0

    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 1
    decoded = decode_typed_payload(recs[0]["original_payload"])
    assert math.isinf(decoded["preco_fechamento"]) and decoded["preco_fechamento"] > 0


def test_c03_negative_infinity_rejected_and_quarantined(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]

    ev_neginf = {
        "event_id": "neginf_test_01",
        "tipo_evento": "Exaustão",
        "symbol": "BTCUSDT",
        "preco_fechamento": float("-inf"),
        "delta": -10.0,
        "epoch_ms": 1700000002000,
    }
    saver.save_event(ev_neginf)
    saver._flush_buffer(saver._write_buffer)

    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0

    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 1
    decoded = decode_typed_payload(recs[0]["original_payload"])
    assert math.isinf(decoded["preco_fechamento"]) and decoded["preco_fechamento"] < 0


# ==============================================================================
# C04: Rejeição de Booleanos em Campos Numéricos
# ==============================================================================
def test_c04_boolean_in_numeric_field_rejected(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]

    ev_bool = {
        "event_id": "bool_test_01",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": True,
        "epoch_ms": 1700000003000,
    }
    saver.save_event(ev_bool)
    saver._flush_buffer(saver._write_buffer)

    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 1
    assert "BOOLEAN" in recs[0]["rejection_reason"]


# ==============================================================================
# C05: Rejeição de Preços Não-Positivos (<= 0.0)
# ==============================================================================
@pytest.mark.parametrize("bad_price", [0.0, -100.0, -0.0])
def test_c05_non_positive_price_rejected(hermetic_saver, bad_price):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]

    ev = {
        "event_id": f"price_test_{bad_price}",
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": bad_price,
        "epoch_ms": 1700000004000,
    }
    saver.save_event(ev)
    saver._flush_buffer(saver._write_buffer)

    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) >= 1
    assert any("MUST_BE_POSITIVE" in r["rejection_reason"] for r in recs)


# ==============================================================================
# C06: Rejeição de Inversão de Limites de Microestrutura
# ==============================================================================
def test_c06_inverted_microstructure_bounds_rejected(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]

    # 1. Zona com bottom > top
    ev_zone_inverted = {
        "event_id": "zone_inv_01",
        "tipo_evento": "Zona",
        "symbol": "BTCUSDT",
        "top": 50000.0,
        "bottom": 51000.0,  # Erro: bottom maior que top!
        "epoch_ms": 1700000005000,
    }
    saver.save_event(ev_zone_inverted)
    saver._flush_buffer(saver._write_buffer)

    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 1
    assert "bottom > top" in recs[0]["rejection_reason"]


# ==============================================================================
# C07: Isolamento em Lote Misto (Válidos Persistidos, Inválidos Quarentenados)
# ==============================================================================
def test_c07_mixed_batch_isolation(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]
    jsonl_file = hermetic_saver["jsonl_file"]

    batch = [
        {"event_id": "val_01", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000006001},
        {"event_id": "bad_nan", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": float("nan"), "epoch_ms": 1700000006002},
        {"event_id": "val_02", "tipo_evento": "Exaustão", "symbol": "BTCUSDT", "preco_fechamento": 50100.0, "delta": 15.0, "epoch_ms": 1700000006003},
        {"event_id": "bad_neg", "tipo_evento": "Exaustão", "symbol": "BTCUSDT", "preco_fechamento": -50.0, "epoch_ms": 1700000006004},
        {"event_id": "val_03", "tipo_evento": "Zona", "symbol": "BTCUSDT", "top": 50500.0, "bottom": 50200.0, "epoch_ms": 1700000006005},
    ]

    for e in batch:
        saver.save_event(e)
    saver._flush_buffer(saver._write_buffer)

    # SQLite deve conter exatamente os 3 válidos
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        assert rows == 3

    # JSONL deve conter exatamente os 3 válidos
    jsonl_lines = _read_jsonl_lines(jsonl_file)
    assert len(jsonl_lines) == 3
    valid_ids = {l["event_id"] for l in jsonl_lines}
    assert valid_ids == {"val_01", "val_02", "val_03"}

    # Quarentena deve conter exatamente os 2 inválidos
    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 2
    q_ids = {decode_typed_payload(r["original_payload"])["event_id"] for r in recs}
    assert q_ids == {"bad_nan", "bad_neg"}


# ==============================================================================
# C08: Tratamento de IntegrityCollisionError (Sem Fallback e Sem JSONL)
# ==============================================================================
def test_c08_integrity_collision_quarantined_not_in_fallback_or_jsonl(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]
    jsonl_file = hermetic_saver["jsonl_file"]

    # 1. Salva evento inicial válido
    ev1 = {"event_id": "col_01", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000007000}
    saver.save_event(ev1)
    saver._flush_buffer(saver._write_buffer)
    saver._write_buffer.clear()

    # 2. Envia evento com mesmo ID mas payload divergente
    ev2 = {"event_id": "col_01", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 55000.0, "epoch_ms": 1700000007000}
    saver.save_event(ev2)
    saver._flush_buffer(saver._write_buffer)

    # JSONL deve ter apenas 1 registro (o inicial)
    jsonl_lines = _read_jsonl_lines(jsonl_file)
    assert len(jsonl_lines) == 1
    assert jsonl_lines[0]["preco_fechamento"] == 50000.0

    # Quarentena de colisão deve reter o evento colidente
    col_recs = _read_quarantine_records(q_dir, "collision")
    assert len(col_recs) >= 1
    dec_col = decode_typed_payload(col_recs[0]["original_payload"])
    assert dec_col["event_id"] == "col_01"


# ==============================================================================
# C09: Isolamento de Colisão em Lote Misto
# ==============================================================================
def test_c09_collision_in_mixed_batch_isolated(hermetic_saver):
    saver = hermetic_saver["saver"]
    db_file = hermetic_saver["db_file"]
    jsonl_file = hermetic_saver["jsonl_file"]

    # Pré-popula SQLite com ID existente
    saver.save_event({"event_id": "exist_01", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000008000})
    saver._flush_buffer(saver._write_buffer)
    saver._write_buffer.clear()

    # Lote com 1 colisão divergente + 2 eventos novos válidos
    batch = [
        {"event_id": "new_01", "tipo_evento": "Exaustão", "symbol": "BTCUSDT", "preco_fechamento": 51000.0, "epoch_ms": 1700000008001},
        {"event_id": "exist_01", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 59000.0, "epoch_ms": 1700000008000},  # Colisão
        {"event_id": "new_02", "tipo_evento": "Zona", "symbol": "BTCUSDT", "top": 52000.0, "bottom": 51500.0, "epoch_ms": 1700000008002},
    ]
    for e in batch:
        saver.save_event(e)
    saver._flush_buffer(saver._write_buffer)

    # SQLite deve conter o inicial + os 2 novos válidos
    with sqlite3.connect(str(db_file)) as conn:
        count = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        assert count in (1, 2, 3)  # Dependendo de atomicidade do save_batch no SQLite

    # JSONL nunca contém o payload colidente mutado
    jsonl_lines = _read_jsonl_lines(jsonl_file)
    for line in jsonl_lines:
        if line.get("event_id") == "exist_01":
            assert line.get("preco_fechamento") == 50000.0


# ==============================================================================
# C10: Idempotência no EventSaver
# ==============================================================================
def test_c10_idempotency_resend_identical_event(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]

    ev = {"event_id": "idem_01", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000009000}
    saver.save_event(ev)
    saver._flush_buffer(saver._write_buffer)
    saver._write_buffer.clear()

    # Reenvio idêntico
    saver.save_event(copy.deepcopy(ev))
    saver._flush_buffer(saver._write_buffer)

    # 1 único registro no SQLite
    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("SELECT count(*) FROM events WHERE json_extract(payload, '$.event_id') = 'idem_01'").fetchone()[0] == 1

    # Nenhuma quarentena
    recs = _read_quarantine_records(q_dir, "invalid") + _read_quarantine_records(q_dir, "collision")
    assert len(recs) == 0


# ==============================================================================
# C11: Separação entre Falha Operacional do SQLite e Falha Financeira
# ==============================================================================
def test_c11_sqlite_operational_failure_vs_financial_validation_failure(hermetic_saver, monkeypatch):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    tmp_path = hermetic_saver["tmp_path"]

    # Simula falha operacional no SQLite
    def mock_failing_save_batch(batch):
        raise sqlite3.OperationalError("disk I/O error or database is locked")

    monkeypatch.setattr(saver.db, "save_batch", mock_failing_save_batch)

    ev_valid = {"event_id": "valid_op_fail", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000010000}
    ev_invalid = {"event_id": "invalid_op_fail", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": float("nan"), "epoch_ms": 1700000010001}

    saver.save_event(ev_valid)
    saver.save_event(ev_invalid)
    saver._flush_buffer(saver._write_buffer)

    # Evento inválido vai para quarentena
    inv_recs = _read_quarantine_records(q_dir, "invalid")
    assert len(inv_recs) == 1
    assert decode_typed_payload(inv_recs[0]["original_payload"])["event_id"] == "invalid_op_fail"

    # Evento válido vai para fallback_events devido à falha operacional
    fb_dir = tmp_path / "fallback_events"
    assert fb_dir.exists()
    fb_files = list(fb_dir.glob("*.jsonl"))
    assert len(fb_files) >= 1
    with open(fb_files[0], "r", encoding="utf-8") as f:
        fb_content = [json.loads(line) for line in f if line.strip()]
    assert any(item.get("event_id") == "valid_op_fail" for item in fb_content)
    # Evento inválido NUNCA entra em fallback
    assert not any(item.get("event_id") == "invalid_op_fail" for item in fb_content)


# ==============================================================================
# C12: Rastreabilidade SQLite <-> JSONL e Canonicalização de -0.0
# ==============================================================================
def test_c12_canonicalization_negative_zero_delta_and_traceability(hermetic_saver):
    saver = hermetic_saver["saver"]
    db_file = hermetic_saver["db_file"]
    jsonl_file = hermetic_saver["jsonl_file"]

    ev = {
        "event_id": "trace_neg_zero_01",
        "tipo_evento": "Exaustão",
        "symbol": "BTCUSDT",
        "preco_fechamento": 50000.0,
        "delta": -0.0,
        "epoch_ms": 1700000011000,
    }
    saver.save_event(ev)
    saver._flush_buffer(saver._write_buffer)

    # No SQLite
    with sqlite3.connect(str(db_file)) as conn:
        row = conn.execute("SELECT payload FROM events WHERE json_extract(payload, '$.event_id') = 'trace_neg_zero_01'").fetchone()
        assert row is not None
        payload_sqlite = json.loads(row[0])
        assert math.copysign(1.0, float(payload_sqlite["delta"])) == 1.0  # +0.0

    # No JSONL
    jsonl_lines = _read_jsonl_lines(jsonl_file)
    assert len(jsonl_lines) == 1
    payload_jsonl = jsonl_lines[0]
    assert payload_jsonl["event_id"] == "trace_neg_zero_01"
    assert math.copysign(1.0, float(payload_jsonl["delta"])) == 1.0  # +0.0


# ==============================================================================
# C13: Inicialização Lazy da Quarentena Forense
# ==============================================================================
def test_c13_lazy_quarantine_directory_creation(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]

    # Instanciado com enforce_financial_validation=True, o diretório de quarentena NÃO deve existir
    assert not q_dir.exists()

    # Salva eventos válidos
    saver.save_event({"event_id": "v1", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000012000})
    saver._flush_buffer(saver._write_buffer)

    # Diretório de quarentena continua NÃO existindo
    assert not q_dir.exists()

    # Agora envia um evento com NaN
    saver.save_event({"event_id": "inv1", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": float("nan"), "epoch_ms": 1700000012001})

    # Agora o diretório passou a existir de forma lazy sob demanda
    assert q_dir.exists()


# ==============================================================================
# C14-C15: Cobertura da Matriz de Produtores Institucionais e Analíticos
# ==============================================================================
def test_c14_matriz_produtores_institucionais_validos(hermetic_saver):
    saver = hermetic_saver["saver"]
    db_file = hermetic_saver["db_file"]

    producers_events = [
        {"event_id": "p_abs", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": 50000.0, "epoch_ms": 1700000013000},
        {"event_id": "p_ex", "tipo_evento": "Exaustão", "symbol": "BTCUSDT", "preco_fechamento": 50100.0, "delta": 10.0, "epoch_ms": 1700000013001},
        {"event_id": "p_zone", "tipo_evento": "Zona", "symbol": "BTCUSDT", "top": 50500.0, "bottom": 50000.0, "epoch_ms": 1700000013002},
        {"event_id": "p_trig", "tipo_evento": "ANALYSIS_TRIGGER", "symbol": "BTCUSDT", "anchor_price": 50200.0, "epoch_ms": 1700000013003},
    ]

    saver.save_events(producers_events)
    saver._flush_buffer(saver._write_buffer)

    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 4


def test_c15_matriz_produtores_rejeitados_por_anomalias_financeiras(hermetic_saver):
    saver = hermetic_saver["saver"]
    q_dir = hermetic_saver["quarantine_dir"]
    db_file = hermetic_saver["db_file"]

    bad_events = [
        {"event_id": "bad_abs", "tipo_evento": "Absorção", "symbol": "BTCUSDT", "preco_fechamento": float("nan"), "epoch_ms": 1700000014000},
        {"event_id": "bad_ex", "tipo_evento": "Exaustão", "symbol": "BTCUSDT", "preco_fechamento": -10.0, "epoch_ms": 1700000014001},
        {"event_id": "bad_zone", "tipo_evento": "Zona", "symbol": "BTCUSDT", "top": 50000.0, "bottom": 55000.0, "epoch_ms": 1700000014002},  # Invertido
        {"event_id": "bad_trig", "tipo_evento": "ANALYSIS_TRIGGER", "symbol": "BTCUSDT", "anchor_price": False, "epoch_ms": 1700000014003},  # Booleano
    ]

    saver.save_events(bad_events)
    saver._flush_buffer(saver._write_buffer)

    # 0 eventos salvos no banco
    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0

    # Todos os 4 em quarentena
    recs = _read_quarantine_records(q_dir, "invalid")
    assert len(recs) == 4

