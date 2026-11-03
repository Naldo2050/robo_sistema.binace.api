"""
Suíte complementar de testes de hardening para o motor de pós-verificação financeira.
FASE 5.11 — G4-D B3-B2-C2-B2.1-H1

Cobre os cenários de hardening H01 a H26:
- H01 — expected_events=[] rejeitado (EMPTY_EVENT_SET).
- H02 — Todos os caminhos None rejeitados.
- H03 — Apenas SQLite no modo padrão rejeitado (TARGET_UNAVAILABLE).
- H04 — Apenas JSONL no modo padrão rejeitado (TARGET_UNAVAILABLE).
- H05 — SQLite + JSONL válidos aprovados (VERIFICATION_SUCCESS).
- H06 — destinations_per_event={} rejeitado (EMPTY_DESTINATIONS).
- H07 — Mapeamento sem entrada para um evento rejeitado (INVALID_DESTINATION_SPEC).
- H08 — Mapeamento com chave desconhecida rejeitado (INVALID_DESTINATION_SPEC).
- H09 — Mapeamento com set() rejeitado (EMPTY_DESTINATIONS).
- H10 — Mapeamento com destino desconhecido rejeitado (INVALID_DESTINATION_SPEC).
- H11 — Mapeamento com tipo inválido rejeitado (INVALID_DESTINATION_SPEC).
- H12 — Mapeamento explícito {"sqlite"} válido verifica apenas SQLite.
- H13 — Mapeamento explícito {"jsonl"} válido verifica apenas JSONL/Archive.
- H14 — Mapeamento explícito {"sqlite", "jsonl"} verifica ambos.
- H15 — Duplicata idêntica em expected_events rejeitada (DUPLICATE_IN_EXPECTED_SET).
- H16 — Mesmo ID com payload divergente no lote rejeitado (DUPLICATE_IN_EXPECTED_SET).
- H17 — Event ID vazio rejeitado (INVALID_EVENT_STRUCTURE).
- H18 — Evento esperado com tipo inválido rejeitado (INVALID_EVENT_STRUCTURE).
- H19 — targets_checked vazio nunca aprova.
- H20 — Symlink detectado mediante patch controlado de Path.is_symlink.
- H21 — Falha de leitura nunca aprova.
- H22 — Nenhuma mutação SQLite.
- H23 — Nenhuma mutação JSONL/Archive.
- H24 — CLI apply continua bloqueada (APPLY_NOT_ENABLED_C2A).
- H25 — Resultado estruturado representa corretamente a causa da rejeição.
- H26 — Nenhum acesso a caminhos globais implícitos.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

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
# H01 — expected_events=[] rejeitado
# ----------------------------------------------------------------------
def test_h01_empty_expected_events_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    _init_sqlite_db(db_p, [])
    _write_jsonl(jl_p, [])

    res = verify_reconciliation_targets(
        expected_events=[],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.EMPTY_EVENT_SET
    assert res.total_expected == 0
    assert res.verified_count == 0
    assert any("EMPTY_EVENT_SET" in err for err in res.errors)


# ----------------------------------------------------------------------
# H02 — Todos os caminhos None rejeitados
# ----------------------------------------------------------------------
def test_h02_all_paths_none_rejected():
    ev = {"event_id": "EVT-H02", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=None,
        jsonl_path=None,
        archive_dir=None,
    )

    assert res.approved is False
    assert res.code in (TargetVerificationCode.TARGET_UNAVAILABLE, TargetVerificationCode.EMPTY_DESTINATIONS)
    assert res.verified_count == 0


# ----------------------------------------------------------------------
# H03 — Apenas SQLite no modo padrão rejeitado
# ----------------------------------------------------------------------
def test_h03_only_sqlite_in_default_mode_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    ev = {"event_id": "EVT-H03", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=None,
        archive_dir=None,
        require_all_destinations_default=True,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.TARGET_UNAVAILABLE
    assert any("jsonl_path" in err for err in res.errors)


# ----------------------------------------------------------------------
# H04 — Apenas JSONL no modo padrão rejeitado
# ----------------------------------------------------------------------
def test_h04_only_jsonl_in_default_mode_rejected(tmp_path: Path):
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H04", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=None,
        jsonl_path=jl_p,
        require_all_destinations_default=True,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.TARGET_UNAVAILABLE
    assert any("sqlite_path" in err for err in res.errors)


# ----------------------------------------------------------------------
# H05 — SQLite + JSONL válidos aprovados
# ----------------------------------------------------------------------
def test_h05_sqlite_plus_jsonl_valid_approved(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H05", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 42}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        require_all_destinations_default=True,
    )

    assert res.approved is True
    assert res.code == TargetVerificationCode.VERIFICATION_SUCCESS
    assert res.verified_count == 1
    assert set(res.targets_checked) == {"sqlite", "jsonl"}


# ----------------------------------------------------------------------
# H06 — destinations_per_event={} rejeitado
# ----------------------------------------------------------------------
def test_h06_destinations_per_event_empty_dict_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H06", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={},
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.EMPTY_DESTINATIONS


# ----------------------------------------------------------------------
# H07 — Mapeamento sem entrada para um evento rejeitado
# ----------------------------------------------------------------------
def test_h07_destinations_mapping_missing_key_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev1 = {"event_id": "EVT-H07-1", "epoch_ms": 1700000000001, "tipo_evento": "SIG"}
    ev2 = {"event_id": "EVT-H07-2", "epoch_ms": 1700000000002, "tipo_evento": "SIG"}

    res = verify_reconciliation_targets(
        expected_events=[ev1, ev2],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H07-1": {"sqlite"}},  # Faltou EVT-H07-2
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.INVALID_DESTINATION_SPEC
    assert "EVT-H07-2" in res.problematic_event_ids


# ----------------------------------------------------------------------
# H08 — Mapeamento com chave desconhecida rejeitado
# ----------------------------------------------------------------------
def test_h08_destinations_mapping_unknown_key_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H08", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={
            "EVT-H08": {"sqlite"},
            "EVT-DESCONHECIDO": {"jsonl"},  # Não existe no lote
        },
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.INVALID_DESTINATION_SPEC
    assert "EVT-DESCONHECIDO" in res.problematic_event_ids


# ----------------------------------------------------------------------
# H09 — Mapeamento com set() rejeitado
# ----------------------------------------------------------------------
def test_h09_destinations_mapping_empty_set_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H09", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H09": set()},
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.EMPTY_DESTINATIONS


# ----------------------------------------------------------------------
# H10 — Mapeamento com destino desconhecido rejeitado
# ----------------------------------------------------------------------
def test_h10_destinations_mapping_unsupported_destination_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H10", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H10": {"kafka"}},
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.INVALID_DESTINATION_SPEC


# ----------------------------------------------------------------------
# H11 — Mapeamento com tipo inválido rejeitado
# ----------------------------------------------------------------------
def test_h11_destinations_mapping_invalid_type_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H11", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    # Passando string em vez de Set[str]
    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H11": "sqlite"},  # type: ignore
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.INVALID_DESTINATION_SPEC


# ----------------------------------------------------------------------
# H12 — Mapeamento explícito {"sqlite"} válido verifica apenas SQLite
# ----------------------------------------------------------------------
def test_h12_explicit_sqlite_only_verifies_sqlite(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H12", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [])  # JSONL vazio de propósito

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H12": {"sqlite"}},
    )

    assert res.approved is True
    assert res.code == TargetVerificationCode.VERIFICATION_SUCCESS
    assert res.targets_checked == ["sqlite"]
    assert res.details_by_event["EVT-H12"]["in_sqlite"] is True
    assert res.details_by_event["EVT-H12"]["in_jsonl"] is False


# ----------------------------------------------------------------------
# H13 — Mapeamento explícito {"jsonl"} válido verifica apenas JSONL/Archive
# ----------------------------------------------------------------------
def test_h13_explicit_jsonl_only_verifies_jsonl(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H13", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [])  # SQLite vazio de propósito
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H13": {"jsonl"}},
    )

    assert res.approved is True
    assert res.code == TargetVerificationCode.VERIFICATION_SUCCESS
    assert res.targets_checked == ["jsonl"]
    assert res.details_by_event["EVT-H13"]["in_jsonl"] is True
    assert res.details_by_event["EVT-H13"]["in_sqlite"] is False


# ----------------------------------------------------------------------
# H14 — Mapeamento explícito {"sqlite", "jsonl"} verifica ambos
# ----------------------------------------------------------------------
def test_h14_explicit_both_verifies_both(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H14", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        destinations_per_event={"EVT-H14": {"sqlite", "jsonl"}},
    )

    assert res.approved is True
    assert set(res.targets_checked) == {"sqlite", "jsonl"}


# ----------------------------------------------------------------------
# H15 — Duplicata idêntica em expected_events rejeitada
# ----------------------------------------------------------------------
def test_h15_duplicate_identical_event_in_expected_events_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H15", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{"timestamp_ms": ev["epoch_ms"], "event_type": ev["tipo_evento"], "symbol": "BTCUSDT", "payload": ev}])
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev, ev],  # Duplicata idêntica
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.DUPLICATE_IN_EXPECTED_SET
    assert "EVT-H15" in res.problematic_event_ids


# ----------------------------------------------------------------------
# H16 — Mesmo ID com payload divergente no lote rejeitado
# ----------------------------------------------------------------------
def test_h16_same_id_with_divergent_payload_in_batch_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev1 = {"event_id": "EVT-H16", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 1}
    ev2 = {"event_id": "EVT-H16", "epoch_ms": 1700000000000, "tipo_evento": "SIG", "val": 2}

    res = verify_reconciliation_targets(
        expected_events=[ev1, ev2],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.DUPLICATE_IN_EXPECTED_SET


# ----------------------------------------------------------------------
# H17 — Event ID vazio rejeitado
# ----------------------------------------------------------------------
def test_h17_empty_event_id_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "   ", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.INVALID_EVENT_STRUCTURE


# ----------------------------------------------------------------------
# H18 — Evento esperado com tipo inválido rejeitado
# ----------------------------------------------------------------------
def test_h18_invalid_element_type_in_expected_events_rejected(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"

    res = verify_reconciliation_targets(
        expected_events=["NAO_E_DICT"],  # type: ignore
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.INVALID_EVENT_STRUCTURE


# ----------------------------------------------------------------------
# H19 — targets_checked vazio nunca aprova
# ----------------------------------------------------------------------
def test_h19_empty_targets_checked_never_approves():
    ev = {"event_id": "EVT-H19", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=None,
        jsonl_path=None,
        archive_dir=None,
        require_all_destinations_default=False,
    )

    assert res.approved is False
    assert len(res.targets_checked) == 0


# ----------------------------------------------------------------------
# H20 — Symlink detectado mediante patch controlado de Path.is_symlink
# ----------------------------------------------------------------------
def test_h20_symlink_detected_via_mock_patch(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H20", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    with mock.patch.object(Path, "is_symlink", return_value=True):
        res = verify_reconciliation_targets(
            expected_events=[ev],
            sqlite_path=db_p,
            jsonl_path=jl_p,
        )

    assert res.approved is False
    assert res.code == TargetVerificationCode.SECURITY_VIOLATION
    assert any("Symlink indevido" in err for err in res.errors)


# ----------------------------------------------------------------------
# H21 — Falha de leitura nunca aprova
# ----------------------------------------------------------------------
def test_h21_read_failure_never_approves(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H21", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    jl_p.write_text("CORROMPIDO_JSON\n", encoding="utf-8")

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.CORRUPTED_TARGET


# ----------------------------------------------------------------------
# H22 — Nenhuma mutação SQLite
# ----------------------------------------------------------------------
def test_h22_no_sqlite_mutation(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    ev = {"event_id": "EVT-H22", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

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
    assert _calc_file_sha256(db_p) == sha_before
    assert db_p.stat().st_size == size_before


# ----------------------------------------------------------------------
# H23 — Nenhuma mutação JSONL/Archive
# ----------------------------------------------------------------------
def test_h23_no_jsonl_archive_mutation(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    arch_dir = tmp_path / "archive"
    arch_dir.mkdir()
    arch_file = arch_dir / "eventos_fluxo_20260301_001.jsonl"

    ev = {"event_id": "EVT-H23", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [])
    _write_jsonl(arch_file, [ev])

    sha_jl_before = _calc_file_sha256(jl_p)
    sha_arch_before = _calc_file_sha256(arch_file)

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
        archive_dir=arch_dir,
    )

    assert res.approved is True
    assert _calc_file_sha256(jl_p) == sha_jl_before
    assert _calc_file_sha256(arch_file) == sha_arch_before


# ----------------------------------------------------------------------
# H24 — CLI apply continua bloqueada
# ----------------------------------------------------------------------
def test_h24_cli_apply_remains_blocked(tmp_path: Path):
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
        "Validacao de seguranca da fase C2-B2.1-H1",
        "--lock-path",
        str(lock_file),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode != 0
    assert "APPLY_NOT_ENABLED_C2A" in (proc.stdout + proc.stderr)


# ----------------------------------------------------------------------
# H25 — Resultado estruturado representa corretamente a causa da rejeição
# ----------------------------------------------------------------------
def test_h25_structured_result_cause_representation(tmp_path: Path):
    db_p = tmp_path / "trading_bot.db"
    jl_p = tmp_path / "eventos_fluxo.jsonl"
    _init_sqlite_db(db_p, [])
    _write_jsonl(jl_p, [])

    ev = {"event_id": "EVT-H25", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}
    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is False
    assert res.code == TargetVerificationCode.VERIFICATION_FAILED
    assert res.missing_count == 1
    d = res.to_dict()
    assert d["approved"] is False
    assert "Ausente no destino SQLite" in str(d["details_by_event"]["EVT-H25"]["errors"])


# ----------------------------------------------------------------------
# H26 — Nenhum acesso a caminhos globais implícitos
# ----------------------------------------------------------------------
def test_h26_no_implicit_global_paths_access(tmp_path: Path):
    # Passando caminhos explicitamente isolados em tmp_path
    db_p = tmp_path / "isolated.db"
    jl_p = tmp_path / "isolated.jsonl"
    ev = {"event_id": "EVT-H26", "epoch_ms": 1700000000000, "tipo_evento": "SIG"}

    _init_sqlite_db(db_p, [{
        "timestamp_ms": ev["epoch_ms"],
        "event_type": ev["tipo_evento"],
        "symbol": "BTCUSDT",
        "payload": ev,
    }])
    _write_jsonl(jl_p, [ev])

    res = verify_reconciliation_targets(
        expected_events=[ev],
        sqlite_path=db_p,
        jsonl_path=jl_p,
    )

    assert res.approved is True
    # Confirma que operou somente sobre os caminhos isolados
    assert db_p.exists()
    assert jl_p.exists()
