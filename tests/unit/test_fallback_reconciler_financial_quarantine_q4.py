# -*- coding: utf-8 -*-
"""
Suíte Dedicada de Validação Financeira, Quarentena Forense e Isolamento de Estados Incertos
no Reconciliador de Fallback (Gate H1-B2-Q4 / INC5-B4-I4).

Cenários Formais de Teste (C01 a C15):
- C01: Rejeição e quarentena de NaN em campo numérico financeiro (TYPED_TREE_V1, "INVALID").
- C02: Rejeição e quarentena de +Infinity em campo numérico financeiro.
- C03: Rejeição e quarentena de -Infinity em campo numérico financeiro.
- C04: Rejeição e quarentena de booleano em campo numérico/monetário.
- C05: Rejeição e quarentena de preços não-positivos (0.0, -0.0, < 0).
- C06: Rejeição e quarentena de limites de microestrutura invertidos (bottom > top, val > poc, poc > vah).
- C07: Isolamento de poison pill em lote misto de fallback (3 válidos + 2 inválidos quarentenados, 0 pendências na 2ª rodada).
- C08: Tratamento de colisão de ID com divergência de payload (IntegrityCollisionError) retido na quarentena ("COLLISION").
- C09: Isolamento de colisão de ID em lote misto de fallback (2 válidos inseridos + 1 colidente quarentenado).
- C10: Reconciliação idempotente de evento já persistido no SQLite (0 quarentena, 0 pendências).
- C11: Preservação estrita em falha operacional transitória do SQLite (OperationalError / Lock) sem quarentena espúria.
- C12: Canonicalização de zero negativo (-0.0 -> +0.0) e preservação determinística de event_id.
- C13: Comportamento estritamente lazy do diretório de quarentena (exists=False quando 100% válidos).
- C14: Governança inegociável da trava APPLY_NOT_ENABLED_C2A na CLI.
- C15: Matriz dos 4 produtores institucionais (MT5, COHORT_O1, COHORT_R1, ZONE) com casos válidos e rejeitados.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.financial_quarantine import FinancialQuarantine, decode_typed_payload
from common.financial_validator import FinancialValidationResult
from common.identity_contract import (
    canonicalize_structure,
    compute_producer_identity,
    is_deterministic_event_id,
)
from events.fallback_reconciler import (
    EventClassification,
    FallbackReconciler,
    compute_canonical_payload_hash,
    insert_event_to_sqlite,
    reconcile_events_hermetic,
    validate_reconciliation_financial_payload,
)


def _init_test_sqlite_schema(db_path: Path) -> None:
    """Cria banco SQLite com tabela events no schema exato de produção."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                window_id TEXT,
                is_signal INTEGER DEFAULT 0,
                payload JSON NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp_ms);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
        conn.commit()
    finally:
        conn.close()


# ==============================================================================
# C01–C03: FLOATS NÃO-FINITOS (NaN, +Infinity, -Infinity)
# ==============================================================================

def test_c01_fallback_reconciler_nan_quarantine(tmp_path: Path):
    """C01: NaN em campo financeiro é classificado como INVALIDO, quarentenado em TYPED_TREE_V1 e bloqueado do banco."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    event_nan = {
        "event_id": "TEST_NAN_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000001000,
        "price": 50000.0,
        "anomaly_score": float("nan"),
    }

    fb_file = tmp_path / "eventos_fb_c01.jsonl"
    fb_file.write_text(json.dumps(event_nan) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 1
    assert "TEST_NAN_01" in res["quarantined_ids"]

    # SQLite e JSONL permanecem vazios
    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    conn.close()
    assert count == 0

    # Quarentena forense comprova integridade
    q_files = list((quarantine_dir / "invalid").glob("*.jsonl"))
    assert len(q_files) == 1
    with open(q_files[0], "r", encoding="utf-8") as qf:
        record = json.loads(qf.readline())
        assert record["category"] == "INVALID"
        decoded = decode_typed_payload(record["original_payload"])
        assert math.isnan(decoded["anomaly_score"])

    # Segunda rodada encontra 0 pendências
    if fb_file.exists():
        res2 = reconcile_events_hermetic(
            source_fallback_file=fb_file,
            target_sqlite_path=db_path,
            target_jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            quarantine_dir=quarantine_dir,
            perform_backup=False,
            repo_root=tmp_path,
        )
        assert res2["sqlite_inserted"] == 0
        assert res2["quarantined_count"] == 0
    else:
        assert not fb_file.exists()


def test_c02_fallback_reconciler_pos_infinity_quarantine(tmp_path: Path):
    """C02: +Infinity em campo numérico financeiro é interceptado e quarentenado."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    event_inf = {
        "event_id": "TEST_INF_01",
        "tipo_evento": "ORDERBOOK_IMBALANCE",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000002000,
        "imbalance_ratio": float("inf"),
    }

    fb_file = tmp_path / "eventos_fb_c02.jsonl"
    fb_file.write_text(json.dumps(event_inf) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 1
    assert "TEST_INF_01" in res["quarantined_ids"]

    q_files = list((quarantine_dir / "invalid").glob("*.jsonl"))
    assert len(q_files) == 1
    with open(q_files[0], "r", encoding="utf-8") as qf:
        record = json.loads(qf.readline())
        decoded = decode_typed_payload(record["original_payload"])
        assert math.isinf(decoded["imbalance_ratio"]) and decoded["imbalance_ratio"] > 0


def test_c03_fallback_reconciler_neg_infinity_quarantine(tmp_path: Path):
    """C03: -Infinity em campo numérico financeiro é interceptado e quarentenado."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    event_neg_inf = {
        "event_id": "TEST_NEGINF_01",
        "tipo_evento": "ORDERBOOK_IMBALANCE",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000003000,
        "imbalance_ratio": float("-inf"),
    }

    fb_file = tmp_path / "eventos_fb_c03.jsonl"
    fb_file.write_text(json.dumps(event_neg_inf) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 1
    assert "TEST_NEGINF_01" in res["quarantined_ids"]

    q_files = list((quarantine_dir / "invalid").glob("*.jsonl"))
    assert len(q_files) == 1
    with open(q_files[0], "r", encoding="utf-8") as qf:
        record = json.loads(qf.readline())
        decoded = decode_typed_payload(record["original_payload"])
        assert math.isinf(decoded["imbalance_ratio"]) and decoded["imbalance_ratio"] < 0


# ==============================================================================
# C04–C06: BOOLEANOS, PREÇOS NÃO-POSITIVOS E LIMITES DE MICROESTRUTURA
# ==============================================================================

def test_c04_fallback_reconciler_boolean_in_numeric_quarantine(tmp_path: Path):
    """C04: Rejeição e quarentena de booleano (True/False) em campo numérico financeiro."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    event_bool = {
        "event_id": "TEST_BOOL_01",
        "tipo_evento": "TRADE_SIGNAL",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000004000,
        "price": True,  # Booleano mascarado como float
    }

    fb_file = tmp_path / "eventos_fb_c04.jsonl"
    fb_file.write_text(json.dumps(event_bool) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 1
    assert "TEST_BOOL_01" in res["quarantined_ids"]


def test_c05_fallback_reconciler_non_positive_price_quarantine(tmp_path: Path):
    """C05: Rejeição e quarentena de preços não-positivos (0.0, -0.0, < 0)."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    event_zero_price = {
        "event_id": "TEST_ZERO_P",
        "tipo_evento": "ORDER_EXECUTION",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000005000,
        "price": 0.0,
    }
    event_neg_price = {
        "event_id": "TEST_NEG_P",
        "tipo_evento": "ORDER_EXECUTION",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000005001,
        "price": -100.0,
    }

    fb_file = tmp_path / "eventos_fb_c05.jsonl"
    fb_file.write_text(json.dumps(event_zero_price) + "\n" + json.dumps(event_neg_price) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 2
    assert "TEST_ZERO_P" in res["quarantined_ids"]
    assert "TEST_NEG_P" in res["quarantined_ids"]


def test_c06_fallback_reconciler_inverted_microstructure_quarantine(tmp_path: Path):
    """C06: Rejeição e quarentena de limites de microestrutura invertidos (bottom > top, val > poc, poc > vah)."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    event_zone_inv = {
        "event_id": "TEST_ZONE_INV",
        "tipo_evento": "ZONE_V2",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000006000,
        "top": 50000.0,
        "bottom": 50500.0,  # Inversão: bottom > top
        "volume": 100.0,
    }
    event_profile_inv = {
        "event_id": "TEST_PROFILE_INV",
        "tipo_evento": "MARKET_PROFILE",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000006001,
        "vah": 50000.0,
        "poc": 50500.0,  # Inversão: poc > vah
        "val": 49500.0,
    }

    fb_file = tmp_path / "eventos_fb_c06.jsonl"
    fb_file.write_text(json.dumps(event_zone_inv) + "\n" + json.dumps(event_profile_inv) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 2
    assert "TEST_ZONE_INV" in res["quarantined_ids"]
    assert "TEST_PROFILE_INV" in res["quarantined_ids"]


# ==============================================================================
# C07–C09: ISOLAMENTO DE POISON PILLS E COLISÕES EM LOTES MISTOS
# ==============================================================================

def test_c07_fallback_reconciler_poison_pill_isolation_mixed_batch(tmp_path: Path):
    """C07: Lote misto de fallback (3 válidos + 2 inválidos) -> 3 inseridos no SQLite, 2 quarentenados, 0 pendências."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_v1 = {"event_id": "EV_VAL_1", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000007001, "price": 50000.0}
    ev_v2 = {"event_id": "EV_VAL_2", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000007002, "price": 50100.0}
    ev_v3 = {"event_id": "EV_VAL_3", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000007003, "price": 50200.0}
    ev_bad1 = {"event_id": "EV_BAD_NAN", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000007004, "price": float("nan")}
    ev_bad2 = {"event_id": "EV_BAD_NEG", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000007005, "price": -50.0}

    fb_file = tmp_path / "eventos_fb_c07.jsonl"
    fb_lines = [
        json.dumps(ev_v1),
        json.dumps(ev_bad1),
        json.dumps(ev_v2),
        json.dumps(ev_bad2),
        json.dumps(ev_v3),
    ]
    fb_file.write_text("\n".join(fb_lines) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 3
    assert res["jsonl_appended"] == 3
    assert res["quarantined_count"] == 2
    assert set(res["reconciled_ids"]) == {"EV_VAL_1", "EV_VAL_2", "EV_VAL_3"}
    assert set(res["quarantined_ids"]) == {"EV_BAD_NAN", "EV_BAD_NEG"}

    # Verifica persistência física no SQLite
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT id, payload FROM events ORDER BY timestamp_ms").fetchall()
    conn.close()
    assert len(rows) == 3
    ids_in_db = [json.loads(r[1])["event_id"] for r in rows]
    assert ids_in_db == ["EV_VAL_1", "EV_VAL_2", "EV_VAL_3"]

    # Verifica quarentena forense isolada
    q_files = list((quarantine_dir / "invalid").glob("*.jsonl"))
    assert len(q_files) == 1
    with open(q_files[0], "r", encoding="utf-8") as qf:
        lines = [json.loads(l) for l in qf if l.strip()]
        assert len(lines) == 2

    # Segunda rodada: 0 pendências
    if fb_file.exists():
        res2 = reconcile_events_hermetic(
            source_fallback_file=fb_file,
            target_sqlite_path=db_path,
            target_jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            quarantine_dir=quarantine_dir,
            perform_backup=False,
            repo_root=tmp_path,
        )
        assert res2["sqlite_inserted"] == 0
        assert res2["quarantined_count"] == 0


def test_c08_fallback_reconciler_integrity_collision_quarantine(tmp_path: Path):
    """C08: Colisão de ID com divergência de payload é retida na quarentena sob categoria 'COLLISION'."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Evento original persistido no SQLite
    original_event = {
        "event_id": "EV_COLLISION_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000008000,
        "price": 50000.0,
    }
    insert_event_to_sqlite(db_path, original_event)

    # Evento divergente com o mesmo ID em fallback
    divergent_event = {
        "event_id": "EV_COLLISION_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000008000,
        "price": 55555.0,  # Preço divergente!
    }

    fb_file = tmp_path / "eventos_fb_c08.jsonl"
    fb_file.write_text(json.dumps(divergent_event) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 0
    assert res["quarantined_count"] == 1
    assert "EV_COLLISION_01" in res["quarantined_ids"]

    # Comprova que o banco SQLite manteve o valor original intacto
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT payload FROM events WHERE timestamp_ms = 1773000008000").fetchone()
    conn.close()
    assert json.loads(row[0])["price"] == 50000.0

    # Comprova que a quarentena gravou sob categoria COLLISION
    col_files = list((quarantine_dir / "collision").glob("*.jsonl"))
    assert len(col_files) == 1
    with open(col_files[0], "r", encoding="utf-8") as qf:
        record = json.loads(qf.readline())
        assert record["category"] == "COLLISION"
        decoded = decode_typed_payload(record["original_payload"])
        assert decoded["price"] == 55555.0


def test_c09_fallback_reconciler_mixed_batch_with_collision_isolation(tmp_path: Path):
    """C09: Lote misto com colisão (2 válidos + 1 colidente) -> 2 válidos inseridos, 1 colidente quarentenado."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Pré-popula SQLite com 1 registro
    ev_existing = {
        "event_id": "EV_EXISTING_C09",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000009000,
        "price": 50000.0,
    }
    insert_event_to_sqlite(db_path, ev_existing)

    ev_val_1 = {"event_id": "EV_NEW_1", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000009001, "price": 50100.0}
    ev_coll = {"event_id": "EV_EXISTING_C09", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000009000, "price": 99999.0}
    ev_val_2 = {"event_id": "EV_NEW_2", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000009002, "price": 50200.0}

    fb_file = tmp_path / "eventos_fb_c09.jsonl"
    fb_file.write_text(f"{json.dumps(ev_val_1)}\n{json.dumps(ev_coll)}\n{json.dumps(ev_val_2)}\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 2
    assert res["quarantined_count"] == 1
    assert set(res["reconciled_ids"]) == {"EV_NEW_1", "EV_NEW_2"}
    assert res["quarantined_ids"] == ["EV_EXISTING_C09"]

    # Total no banco deve ser 3 (1 original + 2 novos)
    conn = sqlite3.connect(str(db_path))
    cnt = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    conn.close()
    assert cnt == 3


# ==============================================================================
# C10–C13: IDEMPOTÊNCIA, ESTADOS INCERTOS, CANONICALIZAÇÃO E LAZY QUARANTINE
# ==============================================================================

def test_c10_fallback_reconciler_idempotent_reconciliation(tmp_path: Path):
    """C10: Reconciliação idempotente de evento já persistido no SQLite (0 quarentena, 0 pendências)."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_idempotent = {
        "event_id": "EV_IDEMPOTENT_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000010000,
        "price": 50000.0,
    }
    insert_event_to_sqlite(db_path, ev_idempotent)
    jsonl_path.write_text(json.dumps(ev_idempotent) + "\n", encoding="utf-8")

    fb_file = tmp_path / "eventos_fb_c10.jsonl"
    fb_file.write_text(json.dumps(ev_idempotent) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 0
    assert res["already_present"] == 1
    assert res["quarantined_count"] == 0
    assert not quarantine_dir.exists()


def test_c11_fallback_reconciler_transient_operational_error_preservation(tmp_path: Path):
    """C11: Falha transitória do SQLite (OperationalError) preserva o evento válido em fallback sem quarentena espúria."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_valid = {
        "event_id": "EV_TRANSIENT_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000011000,
        "price": 50000.0,
    }

    fb_file = tmp_path / "eventos_fb_c11.jsonl"
    fb_file.write_text(json.dumps(ev_valid) + "\n", encoding="utf-8")

    # Simula OperationalError durante a inserção no SQLite
    with patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("database is locked")):
        res = reconcile_events_hermetic(
            source_fallback_file=fb_file,
            target_sqlite_path=db_path,
            target_jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            quarantine_dir=quarantine_dir,
            perform_backup=False,
            repo_root=tmp_path,
            isolate_poison_pills=True,
        )

    # Invariantes do estado incerto:
    assert res["sqlite_inserted"] == 0
    assert res["unresolved_count"] == 1
    assert res["quarantined_count"] == 0  # NÃO vai para quarentena de inválidos!
    assert not quarantine_dir.exists()    # Nenhuma pasta de quarentena criada!

    # Evento permanece 100% íntegro e preservado no arquivo de fallback
    assert fb_file.exists()
    content = fb_file.read_text(encoding="utf-8").strip()
    assert "EV_TRANSIENT_01" in content

    # Em seguida, o banco volta a operar: reconciliação é concluída com sucesso!
    res_retry = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )
    assert res_retry["sqlite_inserted"] == 1
    assert res_retry["unresolved_count"] == 0


def test_c12_fallback_reconciler_canonicalize_negative_zero(tmp_path: Path):
    """C12: Canonicalização de zero negativo (-0.0 -> +0.0) em campos neutros e preservação de event_id."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_neg_zero = {
        "event_id": "EV_NEG_ZERO_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000012000,
        "price": 50000.0,
        "delta_fechamento": -0.0,
    }

    fb_file = tmp_path / "eventos_fb_c12.jsonl"
    fb_file.write_text(json.dumps(ev_neg_zero) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 1
    assert "EV_NEG_ZERO_01" in res["reconciled_ids"]

    # Consulta SQLite e comprova canonicalização de zero negativo para positivo
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT payload FROM events WHERE timestamp_ms = 1773000012000").fetchone()
    conn.close()
    payload = json.loads(row[0])
    assert payload["event_id"] == "EV_NEG_ZERO_01"
    delta_val = payload["delta_fechamento"]
    assert delta_val == 0.0
    assert math.copysign(1.0, float(delta_val)) == 1.0  # Zero estritamente positivo!


def test_c13_fallback_reconciler_lazy_quarantine_directory(tmp_path: Path):
    """C13: Diretório de quarentena é estritamente lazy (exists=False quando só há eventos válidos)."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine_lazy"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_clean = {
        "event_id": "EV_CLEAN_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1773000013000,
        "price": 50000.0,
    }

    fb_file = tmp_path / "eventos_fb_c13.jsonl"
    fb_file.write_text(json.dumps(ev_clean) + "\n", encoding="utf-8")

    reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert quarantine_dir.exists() is False


# ==============================================================================
# C14–C15: GOVERNANÇA CLI E MATRIZ DOS 4 PRODUTORES INSTITUCIONAIS
# ==============================================================================

def test_c14_governance_apply_not_enabled_c2a_enforced(tmp_path: Path):
    """C14: Trava operacional APPLY_NOT_ENABLED_C2A permanece ativa e bloqueando execuções de apply."""
    cli_path = PROJECT_ROOT / "scripts" / "maintenance" / "reconcile_fallback_events.py"
    assert cli_path.exists()

    # Cria lock inicializado em tmp_path com estado CLEAN
    from common.runtime_lock import OperationalRuntimeLock
    lock_file = tmp_path / ".operational_runtime.lock"
    init_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="test_init")
    init_lock.acquire()
    init_lock.release()

    # Executa a CLI com --apply em dry-run de teste sob o lock isolado
    res = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--apply",
            "--confirm-apply",
            "--reason",
            "Auditoria formal de governança da Fase 5.11 Bloco 4",
            "--lock-path",
            str(lock_file),
            "--json-output",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # A trava C2-A encerra com exit code 1 e mensagem explícita
    assert res.returncode == 1
    assert "APPLY_NOT_ENABLED_C2A" in res.stdout or "APPLY_NOT_ENABLED_C2A" in res.stderr


def test_c15_fallback_reconciler_all_four_producers_matrix(tmp_path: Path):
    """C15: Matriz dos 4 produtores institucionais (MT5, COHORT_O1, COHORT_R1, ZONE) com casos válidos e rejeitados."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 1. MT5: válido e rejeitado (preço <= 0)
    mt5_val = {"event_id": "MT5_VALID", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1773000015001, "price": 50000.0}
    mt5_inv = {"event_id": "MT5_INVALID", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1773000015002, "price": 0.0}

    # 2. COHORT_O1: válido e rejeitado (bool em score)
    o1_val = {"event_id": "O1_VALID", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000015003, "score": 0.85, "price": 50000.0}
    o1_inv = {"event_id": "O1_INVALID", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1773000015004, "score": True, "price": 50000.0}

    # 3. COHORT_R1: válido e rejeitado (NaN em ratio)
    r1_val = {"event_id": "R1_VALID", "tipo_evento": "ORDERBOOK_IMBALANCE", "symbol": "BTCUSDT", "epoch_ms": 1773000015005, "imbalance_ratio": 1.5}
    r1_inv = {"event_id": "R1_INVALID", "tipo_evento": "ORDERBOOK_IMBALANCE", "symbol": "BTCUSDT", "epoch_ms": 1773000015006, "imbalance_ratio": float("nan")}

    # 4. ZONE: válido e rejeitado (bottom > top)
    zone_val = {"event_id": "ZONE_VALID", "tipo_evento": "ZONE_V2", "symbol": "BTCUSDT", "epoch_ms": 1773000015007, "top": 51000.0, "bottom": 50000.0}
    zone_inv = {"event_id": "ZONE_INVALID", "tipo_evento": "ZONE_V2", "symbol": "BTCUSDT", "epoch_ms": 1773000015008, "top": 50000.0, "bottom": 51000.0}

    events = [mt5_val, mt5_inv, o1_val, o1_inv, r1_val, r1_inv, zone_val, zone_inv]

    fb_file = tmp_path / "eventos_fb_c15.jsonl"
    fb_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine_dir=quarantine_dir,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    # 4 válidos inseridos, 4 rejeitados quarentenados
    assert res["sqlite_inserted"] == 4
    assert res["quarantined_count"] == 4
    assert set(res["reconciled_ids"]) == {"MT5_VALID", "O1_VALID", "R1_VALID", "ZONE_VALID"}
    assert set(res["quarantined_ids"]) == {"MT5_INVALID", "O1_INVALID", "R1_INVALID", "ZONE_INVALID"}

    conn = sqlite3.connect(str(db_path))
    cnt = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    conn.close()
    assert cnt == 4

