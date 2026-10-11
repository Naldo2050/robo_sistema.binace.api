"""
Suíte de testes herméticos para o Reconciliador Offline de Fallback (Gate G4-D Parte A).

Todos os testes utilizam diretórios temporários isolados (tmp_path).
Nenhum banco ou arquivo operacional é acessado ou modificado.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
import pytest

from events.fallback_reconciler import (
    FallbackReconciler,
    EventClassification,
    compute_canonical_payload_hash,
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
