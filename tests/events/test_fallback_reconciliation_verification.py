"""
Suíte hermética de testes para o motor de pós-verificação financeira.
FASE 5.11 — G4-D B3-B2-C2-B2.1

Cobre rigorosamente os requisitos T01 a T25:
- T01 — Evento presente em ambos os destinos.
- T02 — Evento presente somente em SQLite quando ambos são obrigatórios: rejeitar.
- T03 — Evento presente somente em JSONL quando ambos são obrigatórios: rejeitar.
- T04 — Evento ausente em ambos: rejeitar.
- T05 — Hash SQLite divergente: rejeitar.
- T06 — Hash JSONL divergente: rejeitar.
- T07 — ID SQLite divergente: rejeitar.
- T08 — ID JSONL divergente: rejeitar.
- T09 — Duplicata SQLite: rejeitar.
- T10 — Duplicata JSONL: rejeitar.
- T11 — Mesmo ID com payload divergente: rejeitar.
- T12 — Timestamp coincidente com identidade diferente: não considerar presente.
- T13 — JSONL truncado: rejeitar.
- T14 — JSONL inválido: rejeitar.
- T15 — SQLite corrompido: rejeitar.
- T16 — SQLite ausente: rejeitar.
- T17 — Archive contendo evento legítimo: reconhecer conforme contrato existente.
- T18 — Archive contendo conflito: rejeitar.
- T19 — Symlink indevido: rejeitar.
- T20 — Erro de leitura: fail-closed.
- T21 — Verificação não altera SQLite.
- T22 — Verificação não altera JSONL.
- T23 — Verificação não altera Archive.
- T24 — Resultado estruturado contém contadores e diagnósticos corretos.
- T25 — CLI apply permanece bloqueada.
- Adicional: Higiene do lock (P3-LOCK-CLEANUP-01).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from common.runtime_lock import OperationalRuntimeLock
from events.fallback_reconciler import (
    TargetVerificationCode,
    TargetVerificationResult,
    compute_canonical_payload_hash,
    verify_reconciliation_targets,
)


def _init_sqlite_db(db_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Cria tabela events idêntica ao EventStore e insere registros de teste."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            window_id TEXT,
            is_signal BOOLEAN DEFAULT 0,
            payload JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    for r in rows:
        payload_json = json.dumps(r.get("payload", {}))
        cur.execute(
            """
            INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("timestamp_ms", 0),
                r.get("event_type", "ORDER"),
                r.get("symbol", "BTCUSDT"),
                r.get("window_id", "w1"),
                1 if r.get("is_signal") else 0,
                payload_json,
            ),
        )
    conn.commit()
    conn.close()


def _write_jsonl(file_path: Path, events: List[Dict[str, Any]]) -> None:
    """Grava eventos linha a linha em formato JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _calc_file_sha256(file_path: Path) -> str:
    """Calcula hash SHA256 exato de um arquivo em disco."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# T01 — Evento presente em ambos os destinos
# ----------------------------------------------------------------------
def test_t01_event_present_in_both_destinations(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-001",
        "epoch_ms": 1700000001000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
        "price": 50000.0,
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": ev["symbol"],
        "payload": ev,
    }])
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is True
    assert res.code == TargetVerificationCode.VERIFICATION_SUCCESS
    assert res.total_expected == 1
    assert res.verified_count == 1
    assert res.missing_count == 0
    assert res.duplicate_count == 0
    assert res.divergent_hash_count == 0
    assert res.details_by_event["EVT-2026-001"]["in_sqlite"] is True
    assert res.details_by_event["EVT-2026-001"]["in_jsonl"] is True


# ----------------------------------------------------------------------
# T02 — Evento presente somente em SQLite quando ambos são obrigatórios: rejeitar
# ----------------------------------------------------------------------
def test_t02_event_present_only_in_sqlite(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-002",
        "epoch_ms": 1700000002000,
        "tipo_evento": "ORDER_PLACED",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": ev["symbol"],
        "payload": ev,
    }])
    _write_jsonl(jl_p, [])  # JSONL vazio

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.VERIFICATION_FAILED
    assert res.verified_count == 0
    assert res.missing_count == 1
    assert "EVT-2026-002" in res.problematic_event_ids


# ----------------------------------------------------------------------
# T03 — Evento presente somente em JSONL quando ambos são obrigatórios: rejeitar
# ----------------------------------------------------------------------
def test_t03_event_present_only_in_jsonl(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-003",
        "epoch_ms": 1700000003000,
        "tipo_evento": "ORDER_FILLED",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [])  # SQLite vazio
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.VERIFICATION_FAILED
    assert res.verified_count == 0
    assert res.missing_count == 1
    assert "EVT-2026-003" in res.problematic_event_ids


# ----------------------------------------------------------------------
# T04 — Evento ausente em ambos: rejeitar
# ----------------------------------------------------------------------
def test_t04_event_absent_in_both(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-004",
        "epoch_ms": 1700000004000,
        "tipo_evento": "POSITION_CLOSED",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [])
    _write_jsonl(jl_p, [])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.VERIFICATION_FAILED
    assert res.missing_count == 1
    assert res.verified_count == 0


# ----------------------------------------------------------------------
# T05 — Hash SQLite divergente: rejeitar
# ----------------------------------------------------------------------
def test_t05_hash_sqlite_divergent(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev_expected = {
        "event_id": "EVT-2026-005",
        "epoch_ms": 1700000005000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
        "price": 50000.0,
    }

    ev_sqlite_divergent = dict(ev_expected)
    ev_sqlite_divergent["price"] = 51000.0  # Preço alterado gera divergência de hash

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_sqlite_divergent["epoch_ms"],
        "event_type": ev_sqlite_divergent["tipo_evento"],
        "symbol": ev_sqlite_divergent["symbol"],
        "payload": ev_sqlite_divergent,
    }])
    _write_jsonl(jl_p, [ev_expected])

    res = verify_reconciliation_targets(
        expected_events=[ev_expected],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.divergent_hash_count == 1
    assert res.conflict_count == 1
    assert res.details_by_event["EVT-2026-005"]["sqlite_hash_match"] is False


# ----------------------------------------------------------------------
# T06 — Hash JSONL divergente: rejeitar
# ----------------------------------------------------------------------
def test_t06_hash_jsonl_divergent(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev_expected = {
        "event_id": "EVT-2026-006",
        "epoch_ms": 1700000006000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
        "volume": 1.5,
    }

    ev_jsonl_divergent = dict(ev_expected)
    ev_jsonl_divergent["volume"] = 2.0

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_expected["epoch_ms"],
        "event_type": ev_expected["tipo_evento"],
        "symbol": ev_expected["symbol"],
        "payload": ev_expected,
    }])
    _write_jsonl(jl_p, [ev_jsonl_divergent])

    res = verify_reconciliation_targets(
        expected_events=[ev_expected],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.divergent_hash_count == 1
    assert res.details_by_event["EVT-2026-006"]["jsonl_hash_match"] is False


# ----------------------------------------------------------------------
# T07 — ID SQLite divergente: rejeitar
# ----------------------------------------------------------------------
def test_t07_id_sqlite_divergent(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev_expected = {
        "event_id": "EVT-2026-007",
        "epoch_ms": 1700000007000,
        "tipo_evento": "ORDER_PLACED",
        "symbol": "BTCUSDT",
    }

    ev_wrong_id = dict(ev_expected)
    ev_wrong_id["event_id"] = "EVT-OUTRO-999"

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_wrong_id["epoch_ms"],
        "event_type": ev_wrong_id["tipo_evento"],
        "symbol": ev_wrong_id["symbol"],
        "payload": ev_wrong_id,
    }])
    _write_jsonl(jl_p, [ev_expected])

    res = verify_reconciliation_targets(
        expected_events=[ev_expected],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.missing_count == 1
    assert "EVT-2026-007" in res.problematic_event_ids


# ----------------------------------------------------------------------
# T08 — ID JSONL divergente: rejeitar
# ----------------------------------------------------------------------
def test_t08_id_jsonl_divergent(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev_expected = {
        "event_id": "EVT-2026-008",
        "epoch_ms": 1700000008000,
        "tipo_evento": "ORDER_PLACED",
        "symbol": "BTCUSDT",
    }

    ev_wrong_id = dict(ev_expected)
    ev_wrong_id["event_id"] = "EVT-OUTRO-888"

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_expected["epoch_ms"],
        "event_type": ev_expected["tipo_evento"],
        "symbol": ev_expected["symbol"],
        "payload": ev_expected,
    }])
    _write_jsonl(jl_p, [ev_wrong_id])

    res = verify_reconciliation_targets(
        expected_events=[ev_expected],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.missing_count == 1


# ----------------------------------------------------------------------
# T09 — Duplicata SQLite: rejeitar
# ----------------------------------------------------------------------
def test_t09_duplicate_sqlite(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-009",
        "epoch_ms": 1700000009000,
        "tipo_evento": "ORDER_PLACED",
        "symbol": "BTCUSDT",
    }

    # Duas linhas com o mesmo event_id persistidas no SQLite
    _init_sqlite_db(db_p, [
        {"timestamp_ms": ev["epoch_ms"], "event_type": ev["tipo_evento"], "symbol": ev["symbol"], "payload": ev},
        {"timestamp_ms": ev["epoch_ms"], "event_type": ev["tipo_evento"], "symbol": ev["symbol"], "payload": ev},
    ])
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.duplicate_count == 1
    assert "EVT-2026-009" in res.problematic_event_ids


# ----------------------------------------------------------------------
# T10 — Duplicata JSONL: rejeitar
# ----------------------------------------------------------------------
def test_t10_duplicate_jsonl(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-010",
        "epoch_ms": 1700000010000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": ev["symbol"],
        "payload": ev,
    }])
    # Duas linhas com mesmo ID no JSONL
    _write_jsonl(jl_p, [ev, ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.duplicate_count == 1


# ----------------------------------------------------------------------
# T11 — Mesmo ID com payload divergente: rejeitar
# ----------------------------------------------------------------------
def test_t11_same_id_with_divergent_payload(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev_expected = {
        "event_id": "EVT-2026-011",
        "epoch_ms": 1700000011000,
        "tipo_evento": "ORDER_PLACED",
        "symbol": "BTCUSDT",
        "strategy": "MOMENTUM_A",
    }
    ev_persisted = dict(ev_expected)
    ev_persisted["strategy"] = "MOMENTUM_B"

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_persisted["epoch_ms"],
        "event_type": ev_persisted["tipo_evento"],
        "symbol": ev_persisted["symbol"],
        "payload": ev_persisted,
    }])
    _write_jsonl(jl_p, [ev_persisted])

    res = verify_reconciliation_targets(
        expected_events=[ev_expected],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.divergent_hash_count >= 1
    assert res.conflict_count >= 1


# ----------------------------------------------------------------------
# T12 — Timestamp coincidente com identidade diferente: não considerar presente
# ----------------------------------------------------------------------
def test_t12_timestamp_coincident_different_id_not_present(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev_target = {
        "event_id": "EVT-TARGET-12",
        "epoch_ms": 1700000012000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
    }

    ev_other = {
        "event_id": "EVT-OTHER-99",
        "epoch_ms": 1700000012000,  # Mesmo timestamp
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_other["epoch_ms"],
        "event_type": ev_other["tipo_evento"],
        "symbol": ev_other["symbol"],
        "payload": ev_other,
    }])
    _write_jsonl(jl_p, [ev_other])

    res = verify_reconciliation_targets(
        expected_events=[ev_target],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.missing_count == 1
    assert res.verified_count == 0
    # Verifica que diagnóstico registrou correlação temporal não aceita
    diag = res.details_by_event["EVT-TARGET-12"]["errors"]
    assert any("Coincidência temporal" in msg for msg in diag)


# ----------------------------------------------------------------------
# T13 — JSONL truncado: rejeitar
# ----------------------------------------------------------------------
def test_t13_jsonl_truncated(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-013",
        "epoch_ms": 1700000013000,
        "tipo_evento": "SIGNAL",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": ev["symbol"],
        "payload": ev,
    }])

    # Linha truncada no JSONL
    jl_p.write_text('{"event_id": "EVT-2026-013", "epoch_ms": 1700000013000, "ti\n', encoding="utf-8")

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.CORRUPTED_TARGET


# ----------------------------------------------------------------------
# T14 — JSONL inválido: rejeitar
# ----------------------------------------------------------------------
def test_t14_jsonl_invalid(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-014",
        "epoch_ms": 1700000014000,
        "tipo_evento": "SIGNAL",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": ev["symbol"],
        "payload": ev,
    }])

    jl_p.write_text("ESTO_NAO_E_JSON_DE_MANEIRA_ALGUMA\n", encoding="utf-8")

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.CORRUPTED_TARGET


# ----------------------------------------------------------------------
# T15 — SQLite corrompido: rejeitar
# ----------------------------------------------------------------------
def test_t15_sqlite_corrupted(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-015",
        "epoch_ms": 1700000015000,
        "tipo_evento": "SIGNAL",
        "symbol": "BTCUSDT",
    }

    # Arquivo com bytes inválidos simulando corrupção
    db_p.write_bytes(b"LIXO_CORROMPIDO_NAO_E_SQLITE_VALIDO_000011112222")
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.CORRUPTED_TARGET


# ----------------------------------------------------------------------
# T16 — SQLite ausente: rejeitar
# ----------------------------------------------------------------------
def test_t16_sqlite_missing(tmp_path: Path):
    db_p = tmp_path / "non_existent_db.sqlite"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {
        "event_id": "EVT-2026-016",
        "epoch_ms": 1700000016000,
        "tipo_evento": "SIGNAL",
        "symbol": "BTCUSDT",
    }
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.TARGET_UNAVAILABLE


# ----------------------------------------------------------------------
# T17 — Archive contendo evento legítimo: reconhecer conforme contrato
# ----------------------------------------------------------------------
def test_t17_archive_containing_legitimate_event(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()

    ev = {
        "event_id": "EVT-2026-017",
        "epoch_ms": 1700000017000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
    }

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": ev["symbol"],
        "payload": ev,
    }])
    _write_jsonl(jl_p, [])  # JSONL ativo vazio
    # Gravado no arquivo de archive segmentado
    _write_jsonl(arch_dir / "eventos_fluxo_20260301_001.jsonl", [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        archive_dir=arch_dir,
    )

    assert res.approved is True
    assert res.code == TargetVerificationCode.VERIFICATION_SUCCESS
    assert res.verified_count == 1
    assert res.details_by_event["EVT-2026-017"]["in_archive"] is True


# ----------------------------------------------------------------------
# T18 — Archive contendo conflito: rejeitar
# ----------------------------------------------------------------------
def test_t18_archive_containing_conflict(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()

    ev_expected = {
        "event_id": "EVT-2026-018",
        "epoch_ms": 1700000018000,
        "tipo_evento": "SIGNAL_GENERATED",
        "symbol": "BTCUSDT",
        "val": 100,
    }

    ev_conflicting = dict(ev_expected)
    ev_conflicting["val"] = 999  # Payload conflitante no archive

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev_expected["epoch_ms"],
        "event_type": ev_expected["tipo_evento"],
        "symbol": ev_expected["symbol"],
        "payload": ev_expected,
    }])
    _write_jsonl(jl_p, [])
    _write_jsonl(arch_dir / "eventos_fluxo_20260301_001.jsonl", [ev_conflicting])

    res = verify_reconciliation_targets(
        expected_events=[ev_expected],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        archive_dir=arch_dir,
    )

    assert res.approved is False
    assert res.divergent_hash_count >= 1


# ----------------------------------------------------------------------
# T19 — Symlink indevido: rejeitar
# ----------------------------------------------------------------------
def test_t19_symlink_forbidden(tmp_path: Path):
    db_p = tmp_path / "real.db"
    jl_p = tmp_path / "real.jsonl"
    _init_sqlite_db(db_p, [])
    _write_jsonl(jl_p, [])

    link_db = tmp_path / "link_trading_bot.db"
    try:
        link_db.symlink_to(db_p)
    except (OSError, NotImplementedError):
        pytest.skip("Sistema operacional não autorizou criação de symlink sem privilégios")

    ev = {"event_id": "EVT-19", "epoch_ms": 1700000019000, "tipo_evento": "SIG"}
    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=link_db,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.SECURITY_VIOLATION


# ----------------------------------------------------------------------
# T20 — Erro de leitura: fail-closed
# ----------------------------------------------------------------------
def test_t20_read_error_fail_closed(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "unreadable_dir"  # Passa um diretório no lugar de arquivo regular
    jl_p.mkdir()

    ev = {"event_id": "EVT-20", "epoch_ms": 1700000020000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.TARGET_UNAVAILABLE


# ----------------------------------------------------------------------
# T21 — Verificação não altera SQLite
# ----------------------------------------------------------------------
def test_t21_verification_does_not_modify_sqlite(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {"event_id": "EVT-21", "epoch_ms": 1700000021000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [ev])

    sha_before = _calc_file_sha256(db_p)
    size_before = db_p.stat().st_size

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is True
    sha_after = _calc_file_sha256(db_p)
    size_after = db_p.stat().st_size

    assert sha_before == sha_after
    assert size_before == size_after


# ----------------------------------------------------------------------
# T22 — Verificação não altera JSONL
# ----------------------------------------------------------------------
def test_t22_verification_does_not_modify_jsonl(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev = {"event_id": "EVT-22", "epoch_ms": 1700000022000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [ev])

    sha_before = _calc_file_sha256(jl_p)

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is True
    assert _calc_file_sha256(jl_p) == sha_before


# ----------------------------------------------------------------------
# T23 — Verificação não altera Archive
# ----------------------------------------------------------------------
def test_t23_verification_does_not_modify_archive(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()
    arch_file = arch_dir / "eventos_fluxo_20260301_001.jsonl"

    ev = {"event_id": "EVT-23", "epoch_ms": 1700000023000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [])
    _write_jsonl(arch_file, [ev])

    sha_before = _calc_file_sha256(arch_file)

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        archive_dir=arch_dir,
    )

    assert res.approved is True
    assert _calc_file_sha256(arch_file) == sha_before


# ----------------------------------------------------------------------
# T24 — Resultado estruturado contém contadores e diagnósticos corretos
# ----------------------------------------------------------------------
def test_t24_structured_result_counters_and_diagnostics(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    ev1 = {"event_id": "EVT-OK", "epoch_ms": 1700000024001, "tipo_evento": "SIG", "val": 1}
    ev2_missing = {"event_id": "EVT-MISSING", "epoch_ms": 1700000024002, "tipo_evento": "SIG", "val": 2}
    ev3_dup = {"event_id": "EVT-DUP", "epoch_ms": 1700000024003, "tipo_evento": "SIG", "val": 3}
    ev4_div = {"event_id": "EVT-DIV", "epoch_ms": 1700000024004, "tipo_evento": "SIG", "val": 4}

    # SQLite
    _init_sqlite_db(db_p, [
        {"timestamp_ms": ev1["epoch_ms"], "event_type": ev1["tipo_evento"], "symbol": "BTCUSDT", "payload": ev1},
        # ev2_missing ausente
        {"timestamp_ms": ev3_dup["epoch_ms"], "event_type": ev3_dup["tipo_evento"], "symbol": "BTCUSDT", "payload": ev3_dup},
        {"timestamp_ms": ev3_dup["epoch_ms"], "event_type": ev3_dup["tipo_evento"], "symbol": "BTCUSDT", "payload": ev3_dup},  # duplicata
        {"timestamp_ms": ev4_div["epoch_ms"], "event_type": ev4_div["tipo_evento"], "symbol": "BTCUSDT", "payload": dict(ev4_div, val=99)},  # hash div
    ])

    # JSONL
    _write_jsonl(jl_p, [
        ev1,
        # ev2_missing ausente
        ev3_dup,
        ev4_div,
    ])

    res = verify_reconciliation_targets(
        expected_events=[ev1, ev2_missing, ev3_dup, ev4_div],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.VERIFICATION_FAILED
    assert res.total_expected == 4
    assert res.verified_count == 1
    assert res.missing_count == 1
    assert res.duplicate_count == 1
    assert res.divergent_hash_count == 1
    assert set(res.problematic_event_ids) == {"EVT-MISSING", "EVT-DUP", "EVT-DIV"}
    assert "sqlite" in res.targets_checked
    assert "jsonl" in res.targets_checked

    # Valida dicionário serializável
    d = res.to_dict()
    assert d["approved"] is False
    assert d["code"] == "VERIFICATION_FAILED"
    assert d["total_expected"] == 4


# ----------------------------------------------------------------------
# T25 — CLI apply permanece bloqueada
# ----------------------------------------------------------------------
def test_t25_cli_apply_remains_blocked(tmp_path: Path):
    lock_file = tmp_path / ".runtime.lock"
    lock = OperationalRuntimeLock(lock_file)
    lock.acquire()
    lock.release()

    cmd = [
        sys.executable,
        "scripts/maintenance/reconcile_fallback_events.py",
        "--apply",
        "--confirm-apply",
        "--reason",
        "Validacao de seguranca da fase C2-B2.1",
        "--lock-path",
        str(lock_file),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # A CLI deve falhar com código != 0 e reportar explicitamente APPLY_NOT_ENABLED_C2A
    assert proc.returncode != 0
    assert "APPLY_NOT_ENABLED_C2A" in (proc.stdout + proc.stderr)


# ----------------------------------------------------------------------
# Higiene do Lock (P3-LOCK-CLEANUP-01)
# ----------------------------------------------------------------------
def test_lock_release_cleans_maintenance_session_flag(tmp_path: Path):
    lock_file = tmp_path / ".runtime.lock"
    lock = OperationalRuntimeLock(lock_file)
    lock.acquire()
    lock.release()

    lock.acquire_maintenance()
    assert lock.is_locked is True
    assert getattr(lock, "_is_maintenance_session", False) is True

    lock.release()

    assert lock.is_locked is False
    assert getattr(lock, "_is_maintenance_session", False) is False
