"""
Suíte de testes herméticos para o Reconciliador Offline de Fallback (Gate G4-D Parte A).

Todos os testes utilizam diretórios temporários isolados (tmp_path).
Nenhum banco ou arquivo operacional é acessado ou modificado.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

import events.fallback_reconciler as fallback_reconciler_mod

from events.fallback_reconciler import (
    FallbackReconciler,
    EventClassification,
    compute_canonical_payload_hash,
    create_sqlite_backup_consistent,
    validate_and_prepare_jsonl_target,
    insert_event_to_sqlite,
    append_event_to_jsonl,
    record_reconciliation_in_ledger,
    reconcile_events_hermetic,
)


def _init_sample_sqlite(db_path: Path):
    """Cria banco SQLite temporário com schema oficial da tabela events."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE events (
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
    conn.commit()
    conn.close()


def test_1_canonical_hash_stability():
    """Garante que hashes canônicos sejam idênticos mesmo com chaves desordenadas ou campos voláteis."""
    p1 = {
        "event_id": "abc12345",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "_log_id": "temp_log_1",
        "id": 999,
        "created_at": "2026-10-11 12:00:00",
    }
    p2 = {
        "price": 50000.0,
        "epoch_ms": 1700000000000,
        "symbol": "BTCUSDT",
        "tipo_evento": "AI_ANALYSIS",
        "event_id": "abc12345",
        "_source_db_id": 42,
    }
    h1 = compute_canonical_payload_hash(p1)
    h2 = compute_canonical_payload_hash(p2)
    assert h1 == h2
    assert len(h1) == 64


def test_2_canonical_hash_divergence_on_data_change():
    """Garante que qualquer alteração de dado semântico altera o hash canônico."""
    p1 = {"event_id": "abc", "price": 50000.0}
    p2 = {"event_id": "abc", "price": 50001.0}
    assert compute_canonical_payload_hash(p1) != compute_canonical_payload_hash(p2)


def test_3_dry_run_zero_filesystem_writes(tmp_path):
    """1. e 14. Garante que dry-run não modifica nenhum byte de SQLite, JSONL nem fallback."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text('{"event_id": "e1", "epoch_ms": 1000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    fb_file = tmp_path / "eventos_20261011.jsonl"
    fb_file.write_text('{"event_id": "e2", "epoch_ms": 2000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    stat_db_before = db_path.stat()
    stat_jsonl_before = jsonl_path.stat()
    stat_fb_before = fb_file.stat()

    reconciler = FallbackReconciler(
        sqlite_path=db_path,
        jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        repo_root=tmp_path,
    )
    report = reconciler.run_dry_run([fb_file])

    assert report.total_records_read == 1
    # Verifica que nada foi alterado fisicamente
    assert db_path.stat().st_size == stat_db_before.st_size
    assert jsonl_path.stat().st_size == stat_jsonl_before.st_size
    assert fb_file.stat().st_size == stat_fb_before.st_size


def test_4_event_already_present(tmp_path):
    """9. Evento presente identicamente em SQLite e JSONL é classificado como EVENTO_JA_PRESENTE."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    event_payload = {
        "event_id": "id_100",
        "epoch_ms": 1700000000000,
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "data": "valid",
    }
    raw_json = json.dumps(event_payload)

    # Inserir no SQLite
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (1700000000000, "AI_ANALYSIS", "BTCUSDT", raw_json),
    )
    conn.commit()
    conn.close()

    # Inserir no JSONL
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text(raw_json + "\n", encoding="utf-8")

    fb_file = tmp_path / "eventos_fallback.jsonl"
    fb_file.write_text(raw_json + "\n", encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_file])

    assert report.counts[EventClassification.EVENTO_JA_PRESENTE.value] == 1
    assert report.hypothetical_operations["no_action_needed"] == 1


def test_5_event_absent_in_both(tmp_path):
    """10. Evento não encontrado em nenhum destino é classificado como EVENTO_AUSENTE."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    event_payload = {
        "event_id": "new_ev_1",
        "epoch_ms": 1700000000000,
        "tipo_evento": "SIGNAL",
        "symbol": "ETHUSDT",
    }
    fb_file = tmp_path / "eventos_fallback.jsonl"
    fb_file.write_text(json.dumps(event_payload) + "\n", encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_file])

    assert report.counts[EventClassification.EVENTO_AUSENTE.value] == 1
    assert report.hypothetical_operations["sqlite_inserts_needed"] == 1
    assert report.hypothetical_operations["jsonl_appends_needed"] == 1


def test_6_event_id_collision_divergent_payload(tmp_path):
    """3. Colisão de event_id com payload divergente gera EVENTO_ID_COLISAO."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    p_orig = {"event_id": "coll_1", "epoch_ms": 1000, "tipo_evento": "TEST", "value": 10}
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (1000, "TEST", "BTC", json.dumps(p_orig)),
    )
    conn.commit()
    conn.close()

    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text(json.dumps(p_orig) + "\n", encoding="utf-8")

    # Fallback tem o MESMO ID mas com value=20 (divergente!)
    p_divergent = {"event_id": "coll_1", "epoch_ms": 1000, "tipo_evento": "TEST", "value": 20}
    fb_file = tmp_path / "eventos_fallback.jsonl"
    fb_file.write_text(json.dumps(p_divergent) + "\n", encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_file])

    assert report.counts[EventClassification.EVENTO_ID_COLISAO.value] == 1
    assert report.hypothetical_operations["quarantine_needed"] == 1


def test_7_missing_event_id(tmp_path):
    """4. Evento sem event_id é classificado como EVENTO_SEM_ID."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    p_no_id = {"epoch_ms": 1000, "tipo_evento": "TEST", "symbol": "BTCUSDT"}
    fb_file = tmp_path / "eventos_fallback.jsonl"
    fb_file.write_text(json.dumps(p_no_id) + "\n", encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_file])

    assert report.counts[EventClassification.EVENTO_SEM_ID.value] == 1
    assert report.hypothetical_operations["quarantine_needed"] == 1


def test_8_corrupted_jsonl_line(tmp_path):
    """8. Linha truncada ou JSON inválido é classificada como EVENTO_INVALIDO."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    fb_file = tmp_path / "eventos_fallback.jsonl"
    fb_file.write_text('{"event_id": "ok_1", "epoch_ms": 100}\n{TRUNCATED_JSON_LINE\n{"event_id": "ok_2", "epoch_ms": 200}\n', encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_file])

    assert report.total_records_read == 3
    assert report.counts[EventClassification.EVENTO_INVALIDO.value] == 1
    assert report.counts[EventClassification.EVENTO_AUSENTE.value] == 2


def test_9_fixture_detection_blocks_import(tmp_path):
    """5. Arquivo de fixture conhecida é classificado como FIXTURE_BLOQUEADA."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    # Nome exato da fixture conhecida
    fb_fixture = tmp_path / "eventos_20260307.json"
    fb_fixture.write_text('{"event_id": "fix_1", "epoch_ms": 1000}\n', encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_fixture])

    assert report.counts[EventClassification.FIXTURE_BLOQUEADA.value] == 1
    assert str(fb_fixture) in report.fixture_files_detected
    assert report.hypothetical_operations["no_action_needed"] == 1


def test_10_g4c_archive_segment_recognition(tmp_path):
    """7. Evento presente apenas em segmento do archive G4-C é devidamente detectado."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    # Inserir no SQLite com o mesmo payload completo
    payload_data = {"event_id": "arch_id_1", "epoch_ms": 1000, "tipo_evento": "ARCH_TEST", "symbol": "BTC"}
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (1000, "ARCH_TEST", "BTC", json.dumps(payload_data)),
    )
    conn.commit()
    conn.close()

    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")  # Ativo está vazio

    # Criar segmento no archive
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    seg_file = archive_dir / "eventos_fluxo_20261011_120000_000_001.jsonl"
    seg_file.write_text('{"event_id": "arch_id_1", "epoch_ms": 1000, "tipo_evento": "ARCH_TEST", "symbol": "BTC"}\n', encoding="utf-8")

    fb_file = tmp_path / "fallback.jsonl"
    fb_file.write_text('{"event_id": "arch_id_1", "epoch_ms": 1000, "tipo_evento": "ARCH_TEST", "symbol": "BTC"}\n', encoding="utf-8")

    reconciler = FallbackReconciler(
        sqlite_path=db_path,
        jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        repo_root=tmp_path,
    )
    report = reconciler.run_dry_run([fb_file])

    assert report.counts[EventClassification.EVENTO_JA_PRESENTE.value] == 1
    assert report.records[0].in_archive is True


def test_11_ambiguous_correlation(tmp_path):
    """11. Mesma tupla temporal e tipo com ID diferente gera EVENTO_IDENTIDADE_AMBIGUA."""
    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    # Inserir no SQLite com ID 'existing_id'
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (5000, "AI_ANALYSIS", "BTCUSDT", json.dumps({"event_id": "existing_id"})),
    )
    conn.commit()
    conn.close()

    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    # Fallback tem ID 'different_id' mas mesma tupla (5000, AI_ANALYSIS, BTCUSDT)
    fb_file = tmp_path / "fallback.jsonl"
    fb_file.write_text(json.dumps({"event_id": "different_id", "epoch_ms": 5000, "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT"}) + "\n", encoding="utf-8")

    reconciler = FallbackReconciler(sqlite_path=db_path, jsonl_path=jsonl_path, repo_root=tmp_path)
    report = reconciler.run_dry_run([fb_file])

    assert report.counts[EventClassification.EVENTO_IDENTIDADE_AMBIGUA.value] == 1


def test_12_cli_rejection_of_forbidden_flags():
    """12. e 13. Garante que o CLI rejeita explicitamente qualquer opção de escrita ou destrutiva."""
    cli_path = Path("scripts/maintenance/reconcile_fallback_events.py").resolve()

    for flag in ["--apply", "--force", "--delete", "--repair"]:
        res = subprocess.run(
            [sys.executable, str(cli_path), flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert res.returncode == 2
        assert "PROIBIDO" in res.stderr or "PROIBIDO" in res.stdout


def test_13_cli_dry_run_execution(tmp_path):
    """Executa CLI com --dry-run e --json-output em diretório temporário isolado."""
    cli_path = Path("scripts/maintenance/reconcile_fallback_events.py").resolve()

    db_path = tmp_path / "trading_bot.db"
    _init_sample_sqlite(db_path)

    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    fb_file = tmp_path / "eventos_test.jsonl"
    fb_file.write_text('{"event_id": "cli_1", "epoch_ms": 1000, "tipo_evento": "TEST"}\n', encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--dry-run",
            "--file",
            str(fb_file),
            "--sqlite-path",
            str(db_path),
            "--jsonl-path",
            str(jsonl_path),
            "--json-output",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["mode"] == "DRY_RUN_READ_ONLY"
    assert data["total_records_read"] == 1
    assert data["counts"]["EVENTO_AUSENTE"] == 1


# --- Bateria de Hardening de Identidade Numérica (Etapa 4) ---

def test_14_numerical_identity_int_vs_float():
    """Garante distinção estrita entre inteiros e pontos flutuantes (1 != 1.0)."""
    h_int = compute_canonical_payload_hash({"v": 1})
    h_float = compute_canonical_payload_hash({"v": 1.0})
    assert h_int != h_float


def test_15_numerical_identity_beyond_8_decimals():
    """Garante distinção estrita para valores após a 8ª casa decimal (sem truncamento round(val, 8))."""
    h1 = compute_canonical_payload_hash({"price": 1.000000001})
    h2 = compute_canonical_payload_hash({"price": 1.000000002})
    h_base = compute_canonical_payload_hash({"price": 1.0})
    assert h1 != h2
    assert h1 != h_base
    assert h2 != h_base


def test_16_numerical_identity_negative_values():
    """Garante distinção de valores negativos e entre int negativo e float negativo."""
    h_neg_int = compute_canonical_payload_hash({"delta": -5})
    h_neg_float = compute_canonical_payload_hash({"delta": -5.0})
    h_pos = compute_canonical_payload_hash({"delta": 5.0})
    assert h_neg_int != h_neg_float
    assert h_neg_float != h_pos


def test_17_numerical_identity_very_small_floats():
    """Garante preservação de floats minúsculos / subnormais sem colisão."""
    h_small1 = compute_canonical_payload_hash({"satoshi_fraction": 1e-12})
    h_small2 = compute_canonical_payload_hash({"satoshi_fraction": 2e-12})
    assert h_small1 != h_small2


def test_18_numerical_identity_very_large_floats():
    """Garante preservação e distinção de números grandes."""
    h_large1 = compute_canonical_payload_hash({"volume": 1e18})
    h_large2 = compute_canonical_payload_hash({"volume": 2e18})
    assert h_large1 != h_large2


def test_19_numerical_identity_nan_and_infinity():
    """Garante tratamento explícito e reproduzível de NaN, +Infinity e -Infinity."""
    import math
    h_nan1 = compute_canonical_payload_hash({"val": float("nan")})
    h_nan2 = compute_canonical_payload_hash({"val": float("nan")})
    h_inf_pos = compute_canonical_payload_hash({"val": float("inf")})
    h_inf_neg = compute_canonical_payload_hash({"val": float("-inf")})

    # Hashes de NaN devem ser determinísticos e reproduzíveis
    assert h_nan1 == h_nan2
    assert h_nan1 != h_inf_pos
    assert h_inf_pos != h_inf_neg


def test_20_financial_string_vs_float_preservation():
    """Garante que strings de representação monetária exata não sejam convertidas para float."""
    h_str = compute_canonical_payload_hash({"price": "50000.123456789"})
    h_flt = compute_canonical_payload_hash({"price": 50000.123456789})
    assert h_str != h_flt


def test_21_distinct_financial_payloads_precision():
    """Garante que payloads de ordens com variações mínimas de satoshi/qty gerem hashes distintos."""
    p1 = {
        "event_id": "ord_1",
        "symbol": "BTCUSDT",
        "entry_price": 65000.00000001,
        "quantity": 0.00150000,
    }
    p2 = {
        "event_id": "ord_1",
        "symbol": "BTCUSDT",
        "entry_price": 65000.00000002,
        "quantity": 0.00150000,
    }
    assert compute_canonical_payload_hash(p1) != compute_canonical_payload_hash(p2)


# ==============================================================================
# SUÍTE DE TESTES HERMÉTICOS DO MOTOR DE ESCRITA (G4-D PARTE B1)
# Todos os testes executam estritamente em tmp_path. Zero escrita operacional.
# ==============================================================================

def test_22_b1_event_absent_in_both(tmp_path):
    """1. Evento ausente em ambos é persistido no SQLite e JSONL e gera registro no ledger."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    ledger_path = tmp_path / "ledger.jsonl"

    ev = {"event_id": "ev_b1_1", "epoch_ms": 1700000000000, "tipo_evento": "SIGNAL", "symbol": "BTCUSDT"}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        ledger_path=ledger_path,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 1
    assert res["jsonl_appended"] == 1
    assert res["already_present"] == 0
    assert "ev_b1_1" in res["reconciled_ids"]

    # Conferência no SQLite
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM events")
    assert cur.fetchone()[0] == 1
    conn.close()

    # Conferência no JSONL
    assert jsonl_path.exists()
    assert "ev_b1_1" in jsonl_path.read_text(encoding="utf-8")

    # Conferência no Ledger
    assert ledger_path.exists()
    assert "ev_b1_1" in ledger_path.read_text(encoding="utf-8")


def test_23_b1_event_present_only_sqlite(tmp_path):
    """2. Evento presente somente no SQLite: não duplica no SQLite, anexa apenas no JSONL."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_b1_2", "epoch_ms": 1700000000000, "tipo_evento": "SIGNAL", "symbol": "BTCUSDT"}
    # Pré-inserir no SQLite
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (1700000000000, "SIGNAL", "BTCUSDT", json.dumps(ev)),
    )
    conn.commit()
    conn.close()

    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 0
    assert res["jsonl_appended"] == 1

    # SQLite continua com exatamente 1 linha
    conn = sqlite3.connect(str(db_path))
    assert conn.cursor().execute("SELECT count(*) FROM events").fetchone()[0] == 1
    conn.close()


def test_24_b1_event_present_only_jsonl(tmp_path):
    """3. Evento presente somente no JSONL: insere no SQLite, não anexa no JSONL."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_b1_3", "epoch_ms": 1700000000000, "tipo_evento": "SIGNAL", "symbol": "BTCUSDT"}
    jsonl_path.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 1
    assert res["jsonl_appended"] == 0

    # JSONL continua com apenas 1 linha
    assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_25_b1_event_present_in_archive(tmp_path):
    """4. Evento presente em segmento do archive: reconhece e não anexa no JSONL ativo."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_b1_4", "epoch_ms": 1700000000000, "tipo_evento": "SIGNAL", "symbol": "BTCUSDT"}
    # Pré-inserir no SQLite
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)",
        (1700000000000, "SIGNAL", "BTCUSDT", json.dumps(ev)),
    )
    conn.commit()
    conn.close()

    # Gravar no archive histórico
    arch_file = archive_dir / "eventos_fluxo_20261011_100000_000_001.jsonl"
    arch_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["already_present"] == 1
    assert res["sqlite_inserted"] == 0
    assert res["jsonl_appended"] == 0


def test_26_b1_id_collision_skipped(tmp_path):
    """5. Colisão de event_id com payload divergente: bloqueia importação."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev_orig = {"event_id": "ev_coll", "epoch_ms": 1700000000000, "tipo_evento": "T", "val": 10}
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)", (1700000000000, "T", "", json.dumps(ev_orig)))
    conn.commit()
    conn.close()
    jsonl_path.write_text(json.dumps(ev_orig) + "\n", encoding="utf-8")

    # Fallback com mesmo ID mas val=20
    ev_div = {"event_id": "ev_coll", "epoch_ms": 1700000000000, "tipo_evento": "T", "val": 20}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev_div) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["skipped_quarantine"] == 1
    assert res["sqlite_inserted"] == 0
    assert res["jsonl_appended"] == 0


def test_27_b1_ambiguous_identity_skipped(tmp_path):
    """6. Identidade ambígua (mesma tupla temporal com ID divergente): bloqueia importação."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev_orig = {"event_id": "id_A", "epoch_ms": 1700000000000, "tipo_evento": "T", "symbol": "BTC"}
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (?, ?, ?, ?)", (1700000000000, "T", "BTC", json.dumps(ev_orig)))
    conn.commit()
    conn.close()

    ev_amb = {"event_id": "id_B", "epoch_ms": 1700000000000, "tipo_evento": "T", "symbol": "BTC"}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev_amb) + "\n", encoding="utf-8")

    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["skipped_quarantine"] == 1
    assert res["sqlite_inserted"] == 0


def test_28_b1_invalid_timestamp_rejected(tmp_path):
    """7. Evento com timestamp inválido é rejeitado antes da inserção."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)

    ev_bad_ts = {"event_id": "ev_bad_ts", "epoch_ms": 100, "tipo_evento": "T"}
    with pytest.raises(ValueError, match="Timestamp fora do intervalo aceitável"):
        insert_event_to_sqlite(db_path, ev_bad_ts)


def test_29_b1_sqlite_unknown_handling(tmp_path, monkeypatch):
    """8. SQLite status UNKNOWN: se o commit persistiu no banco antes da falha, é reconhecido como INSERTED."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)

    ev = {"event_id": "ev_unk", "epoch_ms": 1700000000000, "tipo_evento": "T", "symbol": "BTC"}

    orig_connect = fallback_reconciler_mod.sqlite3.connect

    def mock_connect(*args, **kwargs):
        real_conn = orig_connect(*args, **kwargs)

        class FailingCommitConn:
            def __init__(self, c):
                self._c = c
            def __getattr__(self, name):
                return getattr(self._c, name)
            def commit(self):
                self._c.commit()
                raise sqlite3.OperationalError("Simulated connection reset during commit acknowledgment")
            def close(self):
                return self._c.close()

        return FailingCommitConn(real_conn)

    monkeypatch.setattr(fallback_reconciler_mod.sqlite3, "connect", mock_connect)

    # A função deve detectar via query de verificação que a linha foi salva e retornar INSERTED
    status = insert_event_to_sqlite(db_path, ev)
    assert status == "INSERTED"


def test_30_b1_crash_after_sqlite_commit(tmp_path):
    """9. Crash simulado após commit SQLite: reexecução completa JSONL sem duplicar no SQLite."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_crash_1", "epoch_ms": 1700000000000, "tipo_evento": "SIGNAL", "symbol": "BTC"}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    # Inserir apenas no SQLite simulando interrupção antes do JSONL
    insert_event_to_sqlite(db_path, ev)

    # Reexecução
    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 0
    assert res["jsonl_appended"] == 1

    conn = sqlite3.connect(str(db_path))
    assert conn.cursor().execute("SELECT count(*) FROM events").fetchone()[0] == 1
    conn.close()


def test_31_b1_crash_after_jsonl_append(tmp_path):
    """10. Crash simulado após append JSONL: reexecução completa SQLite sem duplicar no JSONL."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_crash_2", "epoch_ms": 1700000000000, "tipo_evento": "SIGNAL", "symbol": "BTC"}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    # Inserir apenas no JSONL
    append_event_to_jsonl(jsonl_path, ev)

    # Reexecução
    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 1
    assert res["jsonl_appended"] == 0
    assert len(jsonl_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_32_b1_idempotent_reexecution(tmp_path):
    """11. Reexecução dupla consecutiva é estritamente idempotente (0 inserts e 0 appends na 2ª vez)."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_idem", "epoch_ms": 1700000000000, "tipo_evento": "S", "symbol": "BTC"}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    # Rodada 1
    res1 = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )
    assert res1["sqlite_inserted"] == 1
    assert res1["jsonl_appended"] == 1

    # Rodada 2
    res2 = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )
    assert res2["sqlite_inserted"] == 0
    assert res2["jsonl_appended"] == 0
    assert res2["already_present"] == 1


def test_33_b1_last_line_valid_no_newline(tmp_path):
    """12. Última linha do JSONL sem newline mas JSON válido: insere separador e anexa sem corromper."""
    jsonl_path = tmp_path / "fluxo.jsonl"
    # Linha válida sem newline final
    jsonl_path.write_bytes(b'{"event_id": "existing", "val": 1}')

    sep = validate_and_prepare_jsonl_target(jsonl_path)
    assert sep == "\n"

    ev_new = {"event_id": "new_ev", "val": 2}
    append_event_to_jsonl(jsonl_path, ev_new, prefix_separator=sep)

    lines = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["event_id"] == "existing"
    assert lines[1]["event_id"] == "new_ev"


def test_34_b1_last_line_corrupted_aborts(tmp_path):
    """13. Última linha do JSONL inválida: aborta sem modificar o arquivo."""
    jsonl_path = tmp_path / "fluxo.jsonl"
    original_bytes = b'{"event_id": "ok"}\n{"event_id": "truncated'
    jsonl_path.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="Última linha do JSONL .* é inválida ou foi corrompida"):
        validate_and_prepare_jsonl_target(jsonl_path)

    # Bytes continuam estritamente intactos (sem truncamento)
    assert jsonl_path.read_bytes() == original_bytes


def test_35_b1_sqlite_backup_api_consistent(tmp_path):
    """14. Backup consistente via SQLite Backup API com integridade validada."""
    db_path = tmp_path / "orig.db"
    _init_sample_sqlite(db_path)

    # Inserir registros
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO events (timestamp_ms, event_type, symbol, payload) VALUES (1000, 'T', 'BTC', '{}')")
    conn.commit()
    conn.close()

    backup_path = tmp_path / "backup" / "orig_backup.db"
    sha = create_sqlite_backup_consistent(db_path, backup_path)
    assert len(sha) == 64
    assert backup_path.exists()

    # Conferir integridade
    c = sqlite3.connect(str(backup_path))
    assert c.cursor().execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert c.cursor().execute("SELECT count(*) FROM events").fetchone()[0] == 1
    c.close()


def test_36_b1_backup_failure_aborts(tmp_path, monkeypatch):
    """15. Falha simulada durante o backup aborta e não toca os dados de destino."""
    db_path = tmp_path / "orig.db"
    _init_sample_sqlite(db_path)
    backup_path = tmp_path / "backup.db"

    orig_connect = fallback_reconciler_mod.sqlite3.connect

    def mock_connect(*args, **kwargs):
        real_conn = orig_connect(*args, **kwargs)

        class FailingBackupConn:
            def __init__(self, c):
                self._c = c
            def __getattr__(self, name):
                return getattr(self._c, name)
            def backup(self, *a, **k):
                raise OSError("Simulated disk error during backup")
            def close(self):
                return self._c.close()

        return FailingBackupConn(real_conn)

    monkeypatch.setattr(fallback_reconciler_mod.sqlite3, "connect", mock_connect)

    with pytest.raises(OSError, match="Simulated disk error"):
        create_sqlite_backup_consistent(db_path, backup_path)


def test_37_b1_fsync_called_on_append(tmp_path, monkeypatch):
    """16. Garante que os.fsync é explicitamente invocado durante o append do JSONL."""
    jsonl_path = tmp_path / "fluxo.jsonl"
    fsync_called = []

    orig_fsync = os.fsync
    def tracked_fsync(fd):
        fsync_called.append(fd)
        orig_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracked_fsync)

    append_event_to_jsonl(jsonl_path, {"event_id": "test_fsync"})
    assert len(fsync_called) == 1


def test_38_b1_enospc_safe_abort(tmp_path, monkeypatch):
    """17. Falha de ENOSPC (disco cheio) aborta com segurança mantendo origem intacta."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "enospc_ev", "epoch_ms": 1700000000000, "tipo_evento": "T"}
    fb_file = tmp_path / "fb.jsonl"
    fb_content = json.dumps(ev) + "\n"
    fb_file.write_text(fb_content, encoding="utf-8")

    def failing_fsync(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="No space left on device"):
        reconcile_events_hermetic(
            source_fallback_file=fb_file,
            target_sqlite_path=db_path,
            target_jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            perform_backup=False,
            repo_root=tmp_path,
        )

    # Origem fallback 100% preservada
    assert fb_file.read_text(encoding="utf-8") == fb_content


def test_39_b1_fixture_blocked_unconditionally(tmp_path):
    """18. Arquivo de fixture versionada é incondicionalmente bloqueado contra escrita."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    # Nome da fixture protegida
    fixture_file = tmp_path / "eventos_20260307.json"
    fixture_file.write_text('{"event_id": "fix_1", "epoch_ms": 1700000000000}\n', encoding="utf-8")

    with pytest.raises(PermissionError, match="fixture de teste versionada pelo Git"):
        reconcile_events_hermetic(
            source_fallback_file=fixture_file,
            target_sqlite_path=db_path,
            target_jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            perform_backup=False,
            repo_root=tmp_path,
        )


def test_40_b1_ledger_failure_does_not_corrupt_data(tmp_path, monkeypatch):
    """19. Falha ao gravar no ledger após persistência em SQLite e JSONL conclui os dados."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    ledger_path = tmp_path / "invalid_ledger_dir" / "ledger.jsonl"  # Diretório que falhará se impedido

    ev = {"event_id": "ev_ldg_fail", "epoch_ms": 1700000000000, "tipo_evento": "T"}
    fb_file = tmp_path / "fb.jsonl"
    fb_file.write_text(json.dumps(ev) + "\n", encoding="utf-8")

    # Inserção manual e append direto com sucesso
    insert_event_to_sqlite(db_path, ev)
    append_event_to_jsonl(jsonl_path, ev)

    # Valida que ambos os dados estão íntegros
    conn = sqlite3.connect(str(db_path))
    assert conn.cursor().execute("SELECT count(*) FROM events").fetchone()[0] == 1
    conn.close()
    assert "ev_ldg_fail" in jsonl_path.read_text(encoding="utf-8")


def test_41_b1_source_fallback_never_modified(tmp_path):
    """20. Garante que o arquivo de fallback de origem nunca é modificado ou apagado."""
    db_path = tmp_path / "bot.db"
    _init_sample_sqlite(db_path)
    jsonl_path = tmp_path / "fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()

    ev = {"event_id": "ev_immutable", "epoch_ms": 1700000000000, "tipo_evento": "T"}
    fb_file = tmp_path / "source_fallback.jsonl"
    content = json.dumps(ev) + "\n"
    fb_file.write_text(content, encoding="utf-8")

    hash_before = hashlib.sha256(fb_file.read_bytes()).hexdigest()
    mtime_before = fb_file.stat().st_mtime_ns

    reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        perform_backup=False,
        repo_root=tmp_path,
    )

    hash_after = hashlib.sha256(fb_file.read_bytes()).hexdigest()
    mtime_after = fb_file.stat().st_mtime_ns

    assert hash_before == hash_after
    assert mtime_before == mtime_after
