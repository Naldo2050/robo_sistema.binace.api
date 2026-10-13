# -*- coding: utf-8 -*-
"""
Suíte de Testes para a Ferramenta Offline de Diagnóstico de Recovery (Gate G4-D B3-B2-B1).

Cobre exaustivamente:
1. Diagnóstico de estado DIRTY.
2. Diagnóstico de estado UNKNOWN.
3. Diagnóstico de RUNNING residual (crash detectado).
4. Diagnóstico de estado CLEAN operacional.
5. Diagnóstico de Lock ausente (violação anti-bypass).
6. Diagnóstico de Manifesto ausente (violação anti-bypass).
7. Detecção de SQLite inválido/corrompido.
8. Detecção de JSONL inválido ou com última linha incompleta.
9. Detecção de Archive com temporários .tmp residuais.
10. Classificação de Fallback fixtures Git isoladas.
11. Detecção de eventos com presença parcial (SQLite-only ou JSONL-only).
12. Detecção de colisões de payload entre destinos.
13. Garantia de Dry-Run: zero escritas ou mutações em disco.
14. Rejeição estrita de flags proibidas na CLI (--confirm-recovery, etc.).
15. Emissão de relatório JSON estruturado válido.
16. Detecção de arquivos WAL/SHM e garantia de zero checkpoint ou modificação.

Todos os testes rodam estritamente em tmp_path. Zero escrita ou acesso a dados de produção.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeState,
    MANIFEST_FILENAME,
    write_runtime_manifest,
    bootstrap_runtime_environment,
)
from scripts.maintenance.recover_runtime_state import diagnose_runtime_recovery


def _create_test_sqlite_with_events(db_path: Path, events: list[dict]) -> None:
    """Cria banco SQLite com o schema real da tabela 'events'."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            window_id TEXT,
            is_signal BOOLEAN DEFAULT 0,
            payload JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    for ev in events:
        cur.execute(
            """
            INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ev.get("timestamp_ms", 1710000000000),
                ev.get("event_type", "trade"),
                ev.get("symbol", "BTCUSDT"),
                ev.get("window_id", "w1"),
                ev.get("is_signal", 0),
                json.dumps(ev.get("payload", {})),
            ),
        )
    conn.commit()
    conn.close()


def _create_test_jsonl(jsonl_path: Path, records: list[dict], trailing_newline: bool = True) -> None:
    """Cria arquivo JSONL com os registros especificados."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            line = json.dumps(rec)
            if idx == len(records) - 1 and not trailing_newline:
                f.write(line)
            else:
                f.write(line + "\n")


def test_01_diagnose_dirty_state(tmp_path: Path):
    """1. Diagnostica estado DIRTY indicando shutdown incompleto e recomendação adequada."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)

    # Simula gravação de estado DIRTY
    dirty_meta = {
        "status": "DIRTY",
        "holder": "market_bot",
        "pid": 1111,
        "reason": "Writers ativos: EventSaverFlushWorker",
    }
    with open(lock_path, "wb") as f:
        f.write(b"#" + json.dumps(dirty_meta).encode("utf-8"))

    report = diagnose_runtime_recovery(data_dir=data_dir, lock_path=lock_path)
    assert report["mode"] == "DRY_RUN_READ_ONLY"
    assert report["recovery_classification"] == "DIRTY_INCOMPLETE_SHUTDOWN"
    assert report["lock"]["persisted_status"] == "RuntimeState.DIRTY"
    assert "Writers ativos" in report["lock"]["reason"]
    assert any("Lock retido com status DIRTY" in r for r in report["recommendations"])


def test_02_diagnose_unknown_corrupted_metadata(tmp_path: Path):
    """2. Diagnostica estado UNKNOWN por metadados truncados ou corrompidos."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)

    with open(lock_path, "wb") as f:
        f.write(b"#{\"status\": \"CORRUPTED_JSON_TRUNCATED")

    report = diagnose_runtime_recovery(data_dir=data_dir, lock_path=lock_path)
    assert report["recovery_classification"] == "LOCK_METADATA_CORRUPTED"
    assert report["lock"]["persisted_status"] == "RuntimeState.UNKNOWN"


def test_03_diagnose_residual_running_crash(tmp_path: Path):
    """3. Diagnostica RUNNING residual sem kernel lock como CRASH_RESIDUAL_RUNNING."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)

    running_meta = {
        "status": "RUNNING",
        "holder": "crashed_bot",
        "pid": 2222,
    }
    with open(lock_path, "wb") as f:
        f.write(b"#" + json.dumps(running_meta).encode("utf-8"))

    report = diagnose_runtime_recovery(data_dir=data_dir, lock_path=lock_path)
    assert report["recovery_classification"] == "CRASH_RESIDUAL_RUNNING"
    assert report["lock"]["is_kernel_locked"] is False
    assert any("Crash abrupto detectado" in r for r in report["recommendations"])


def test_04_diagnose_clean_state(tmp_path: Path):
    """4. Diagnostica estado CLEAN operacional como apto e sem ação necessária."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)

    report = diagnose_runtime_recovery(data_dir=data_dir, lock_path=lock_path)
    assert report["recovery_classification"] == "CLEAN_OPERATIONAL"
    assert report["lock"]["persisted_status"] == "RuntimeState.CLEAN"
    assert any("Ambiente já em estado CLEAN" in r for r in report["recommendations"])


def test_05_diagnose_missing_lock_with_data(tmp_path: Path):
    """5. Diagnostica lock ausente com dados existentes como violação anti-bypass."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])

    report = diagnose_runtime_recovery(data_dir=data_dir, lock_path=lock_path)
    assert report["recovery_classification"] == "LOCK_MISSING_ANTI_BYPASS_VIOLATION"
    assert report["lock"]["anti_bypass_ok"] is False
    assert len(report["conflicts"]) > 0


def test_06_diagnose_missing_manifest_with_lock(tmp_path: Path):
    """6. Diagnostica manifesto ausente com lock existente como violação anti-bypass."""
    data_dir = tmp_path / "dados"
    lock_path = data_dir / ".operational_runtime.lock"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir, lock_path=lock_path)

    # Deleta manifesto
    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_path.unlink()

    report = diagnose_runtime_recovery(data_dir=data_dir, lock_path=lock_path)
    assert report["recovery_classification"] == "MANIFEST_MISSING_ANTI_BYPASS_VIOLATION"
    assert report["lock"]["anti_bypass_ok"] is False


def test_07_diagnose_corrupted_sqlite(tmp_path: Path):
    """7. Diagnostica SQLite corrompido reportando falha no integrity_check."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    _create_test_sqlite_with_events(sqlite_path, [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir)

    # Corrompe SQLite
    with open(sqlite_path, "r+b") as f:
        f.seek(16)
        f.write(b"CORRUPTED_PAGE_DATA_HEADER_LIXO_123456789")

    report = diagnose_runtime_recovery(data_dir=data_dir)
    assert report["sqlite"]["integrity_check_ok"] is False
    assert any("Falha estrutural PRAGMA integrity_check" in c for c in report["conflicts"])


def test_08_diagnose_corrupted_jsonl(tmp_path: Path):
    """8. Diagnostica JSONL corrompido e última linha truncada sem quebra de linha."""
    data_dir = tmp_path / "dados"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    bootstrap_runtime_environment(data_dir=data_dir)

    # Cria JSONL com linha corrompida e sem newline final
    records = [{"event_id": "ev_1", "type": "trade"}]
    _create_test_jsonl(jsonl_path, records, trailing_newline=False)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write("{\"corrupted_line_incomplete: true")

    report = diagnose_runtime_recovery(data_dir=data_dir)
    assert report["jsonl"]["last_line_clean"] is False
    assert report["jsonl"]["corrupt_lines_count"] > 0
    assert any("Arquivo JSONL ativo contém" in c for c in report["conflicts"])


def test_09_diagnose_archive_with_tmp_files(tmp_path: Path):
    """9. Diagnostica presença de arquivos temporários .tmp residuais no archive."""
    data_dir = tmp_path / "dados"
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir)

    # Cria arquivo residual .tmp
    (archive_dir / "eventos_fluxo_20260301.jsonl.tmp").write_text("{\"test\": 1}\n", encoding="utf-8")

    report = diagnose_runtime_recovery(data_dir=data_dir)
    assert report["archive"]["has_residual_tmp"] is True
    assert "eventos_fluxo_20260301.jsonl.tmp" in report["archive"]["tmp_files_detected"]
    assert any("Diretório de archive contém arquivos temporários" in c for c in report["conflicts"])


def test_10_diagnose_fallback_fixtures_classified(tmp_path: Path):
    """10. Classifica arquivos de fixtures Git no fallback e não os conta como pendentes."""
    data_dir = tmp_path / "dados"
    fallback_dir = tmp_path / "fallback_events"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir)

    # Cria arquivo fixture conhecido
    fixture_file = fallback_dir / "eventos_20260307.json"
    fixture_file.write_text("[{\"event_id\": \"fix_1\", \"type\": \"trade\"}]\n", encoding="utf-8")

    report = diagnose_runtime_recovery(data_dir=data_dir, fallback_dir=fallback_dir)
    assert report["fallback"]["fixtures_count"] == 1
    # Fixtures de teste não devem ser contadas como pendentes não reconciliadas
    assert report["consistency"]["fallback_pending_unreconciled_count"] == 0


def test_11_diagnose_partial_presence(tmp_path: Path):
    """11. Identifica eventos presentes somente no SQLite ou somente no JSONL."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"

    sq_events = [
        {"timestamp_ms": 1000, "event_type": "trade", "symbol": "BTC", "payload": {"event_id": "both_1", "p": 10}},
        {"timestamp_ms": 2000, "event_type": "trade", "symbol": "BTC", "payload": {"event_id": "sq_only_1", "p": 20}},
    ]
    jl_records = [
        {"event_id": "both_1", "epoch_ms": 1000, "tipo_evento": "trade", "symbol": "BTC", "p": 10},
        {"event_id": "jl_only_1", "epoch_ms": 3000, "tipo_evento": "trade", "symbol": "BTC", "p": 30},
    ]

    _create_test_sqlite_with_events(sqlite_path, sq_events)
    _create_test_jsonl(jsonl_path, jl_records)
    bootstrap_runtime_environment(data_dir=data_dir)

    report = diagnose_runtime_recovery(data_dir=data_dir)
    assert report["consistency"]["events_in_both_sqlite_and_jsonl"] == 1
    assert report["consistency"]["events_sqlite_only"] == 1
    assert report["consistency"]["events_jsonl_only"] == 1


def test_12_diagnose_payload_collisions(tmp_path: Path):
    """12. Detecta colisão de hash de payload quando o mesmo event_id tem dados divergentes."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"

    sq_events = [
        {"timestamp_ms": 1000, "event_type": "trade", "symbol": "BTC", "payload": {"event_id": "ev_col_1", "price": 50000}},
    ]
    jl_records = [
        {"event_id": "ev_col_1", "epoch_ms": 1000, "tipo_evento": "trade", "symbol": "BTC", "price": 99999},
    ]

    _create_test_sqlite_with_events(sqlite_path, sq_events)
    _create_test_jsonl(jsonl_path, jl_records)
    bootstrap_runtime_environment(data_dir=data_dir)

    report = diagnose_runtime_recovery(data_dir=data_dir)
    assert report["consistency"]["payload_hash_collisions_count"] == 1
    assert any("Detectadas 1 colisões de hash" in c for c in report["conflicts"])


def test_13_guaranteed_zero_writes(tmp_path: Path):
    """13. Comprova que a execução do diagnóstico mantém hashes SHA256 de todos os arquivos 100% inalterados."""
    data_dir = tmp_path / "dados"
    sqlite_path = data_dir / "trading_bot.db"
    jsonl_path = data_dir / "eventos_fluxo.jsonl"
    _create_test_sqlite_with_events(sqlite_path, [{"payload": {"event_id": "e1"}}])
    _create_test_jsonl(jsonl_path, [{"event_id": "e1"}])
    bootstrap_runtime_environment(data_dir=data_dir)

    lock_path = data_dir / ".operational_runtime.lock"
    manifest_path = data_dir / MANIFEST_FILENAME

    def _hash_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    h_sq_before = _hash_file(sqlite_path)
    h_jl_before = _hash_file(jsonl_path)
    h_lk_before = _hash_file(lock_path)
    h_mf_before = _hash_file(manifest_path)

    # Executa diagnóstico
    report = diagnose_runtime_recovery(data_dir=data_dir)
    assert report["mode"] == "DRY_RUN_READ_ONLY"

    # Confirma hashes estritamente idênticos
    assert _hash_file(sqlite_path) == h_sq_before
    assert _hash_file(jsonl_path) == h_jl_before
    assert _hash_file(lock_path) == h_lk_before
    assert _hash_file(manifest_path) == h_mf_before


def test_14_cli_rejection_of_mutation_flags():
    """14. A CLI do recover_runtime_state rejeita sumariamente qualquer flag de mutação com código 2."""
    script_path = PROJECT_ROOT / "scripts" / "maintenance" / "recover_runtime_state.py"

    forbidden_to_test = [
        "--confirm-recovery",
        "--clear-dirty-state",
        "--force",
        "--repair",
        "--delete",
        "-f",
    ]

    for flag in forbidden_to_test:
        res = subprocess.run(
            [sys.executable, str(script_path), flag],
            capture_output=True,
            text=True,
            errors="replace",
        )
        assert res.returncode == 2
        assert res.stderr is not None
        assert "PROIBIDO" in res.stderr
        assert "READ-ONLY" in res.stderr


def test_15_cli_json_output_schema(tmp_path: Path):
    """15. A CLI emite JSON estruturado com schema completo quando executada com --json-output."""
    data_dir = tmp_path / "dados"
    _create_test_sqlite_with_events(data_dir / "trading_bot.db", [])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [])
    bootstrap_runtime_environment(data_dir=data_dir)

    script_path = PROJECT_ROOT / "scripts" / "maintenance" / "recover_runtime_state.py"

    res = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir", str(data_dir),
            "--json-output",
        ],
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["mode"] == "DRY_RUN_READ_ONLY"
    assert "recovery_classification" in data
    assert "lock" in data
    assert "sqlite" in data
    assert "jsonl" in data
    assert "archive" in data
    assert "fallback" in data
    assert "consistency" in data
    assert "conflicts" in data
    assert "recommendations" in data
    assert any("Nenhuma autorização automática" in r for r in data["recommendations"])


def test_16_diagnose_wal_shm_detection(tmp_path: Path):
    """16. Detecta fielmente arquivos WAL e SHM, reporta tamanhos e garante zero modificação ou checkpoint."""
    data_dir = tmp_path / "dados"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "trading_bot.db"
    wal_path = data_dir / "trading_bot.db-wal"
    shm_path = data_dir / "trading_bot.db-shm"

    _create_test_sqlite_with_events(sqlite_path, [{"payload": {"event_id": "wal_ev_1"}}])
    _create_test_jsonl(data_dir / "eventos_fluxo.jsonl", [{"event_id": "wal_ev_1"}])
    bootstrap_runtime_environment(data_dir=data_dir)

    # Cria arquivos auxiliares SQLite de teste (WAL e SHM) com tamanhos conhecidos
    dummy_wal_bytes = b"SQLITE_WAL_SIMULATED_TEST_FRAME_" * 32  # 1024 bytes
    dummy_shm_bytes = b"SQLITE_SHM_SIMULATED_INDEX_PAGES" * 1024  # 32768 bytes
    wal_path.write_bytes(dummy_wal_bytes)
    shm_path.write_bytes(dummy_shm_bytes)

    wal_size_before = wal_path.stat().st_size
    shm_size_before = shm_path.stat().st_size
    sqlite_size_before = sqlite_path.stat().st_size

    wal_hash_before = hashlib.sha256(dummy_wal_bytes).hexdigest()
    sqlite_hash_before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()

    report = diagnose_runtime_recovery(data_dir=data_dir)

    # 1. Confirmação de detecção de existência
    assert report["sqlite"]["wal_exists"] is True
    assert report["sqlite"]["shm_exists"] is True

    # 2. Confirmação de tamanho informado
    assert report["sqlite"]["wal_size_bytes"] == wal_size_before
    assert report["sqlite"]["shm_size_bytes"] == shm_size_before

    # 3. Nenhuma modificação e nenhum checkpoint executado (zero escrita no banco ou WAL)
    assert hashlib.sha256(wal_path.read_bytes()).hexdigest() == wal_hash_before
    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == sqlite_hash_before
    assert wal_path.stat().st_size == wal_size_before
    assert shm_path.stat().st_size == shm_size_before
    assert sqlite_path.stat().st_size == sqlite_size_before
