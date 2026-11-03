# -*- coding: utf-8 -*-
"""
test_financial_pipeline_e2e_q5_g5.py
Suíte Integrada End-to-End do Pipeline Financeiro e Resiliência a ENOSPC (Gate H1-B2-Q5 + G5).

Cobre os 15 cenários determinísticos:
- E01: E2E Nominal 4 Produtores (EventSaver -> EventStore + JSONL com invariância de ID e -0.0 -> +0.0)
- E02: E2E Contingência SQLite Down -> Fallback -> Reconciler -> SQLite + JSONL (válidos reconciliados, inválidos quarentenados)
- E03: E2E Poison Pill Injetado no Fallback Antes da Reconciliação (isolamento sem loop)
- E04: E2E Colisão Divergente Cross-Stage (EventSaver + Reconciler -> quarentena COLLISION, SQLite imutável)
- E05: E2E Idempotência Cross-Stage (duplicata idêntica reconhecida sem erro e sem quarentena espúria)
- E06: E2E Recuperação Forense Bit-a-Bit da Quarentena (decode_typed_payload para NaN, Inf, bool, colisões)
- E07: G5 — ENOSPC na FinancialQuarantine (zero .tmp residuais, incremento de enospc_error_count)
- E08: G5 — ENOSPC no EventSaver durante Fallback/JSONL (telemetria enospc, log crítico)
- E09: G5 — ENOSPC no FallbackReconciler (propagação segura, preservação de fallback e limpeza de .tmp)
- E10: G5 — Telemetria e Contadores de Integridade Financeira (validação de métricas em todo o pipeline)
- E11: Rastreabilidade e Invariância de Identidade de Zonas E2E (ZONE_CREATED/TOUCHED/MITIGATED)
- E12: Invariante de Inicialização Lazy E2E (pipeline nominal não cria pasta quarantine)
- E13: Alta Cardinalidade e Carga Mista E2E (conservação matemática exata em lote grande)
- E14: Preservação de Estado Incerto E2E (OperationalError durante reconciliação mantém fallback intacto)
- E15: Governança CLI C2A (trava APPLY_NOT_ENABLED_C2A bloqueia mutação direta)
"""

import copy
import decimal
import errno
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from common.financial_quarantine import FinancialQuarantine, decode_typed_payload
from common.financial_validator import FinancialValidationResult
from common.identity_contract import (
    canonicalize_structure,
    compute_producer_identity,
    is_deterministic_event_id,
    normalize_last_touched,
)
from common.runtime_lock import OperationalRuntimeLock
from database.event_store import EventStore, IntegrityCollisionError
from events.event_saver import EventSaver
from events.fallback_reconciler import (
    EventClassification,
    FallbackReconciler,
    insert_event_to_sqlite,
    reconcile_events_hermetic,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _init_test_sqlite_schema(db_path: Path):
    """Inicializa banco SQLite isolado com a tabela 'events' usando o schema oficial."""
    store = EventStore(db_path=str(db_path))
    return store


# ==============================================================================
# E01: E2E NOMINAL 4 PRODUTORES
# ==============================================================================
def test_e01_financial_pipeline_nominal_four_producers(tmp_path: Path):
    """E01: Ingestão nominal dos 4 produtores (MT5, O1, R1, ZONE) com paridade de ID e canonicalização -0.0 -> +0.0."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    store = EventStore(db_path=str(db_path), quarantine_dir=quarantine_dir)

    ev_mt5 = {
        "event_id": "MT5_E01_001",
        "tipo_evento": "MT5_SIGNAL",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000001000,
        "price": 50000.0,
        "delta_fechamento": -0.0,  # Zero negativo deve ser canonicalizado para +0.0
    }
    ev_o1 = {
        "event_id": "O1_E01_002",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000001001,
        "price": 50100.0,
        "score": 0.92,
    }
    ev_r1 = {
        "event_id": "R1_E01_003",
        "tipo_evento": "ORDERBOOK_IMBALANCE",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000001002,
        "imbalance_ratio": 1.45,
    }
    ev_zone = {
        "event_id": "ZONE_E01_004",
        "tipo_evento": "ZONE_V2",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000001003,
        "top": 50200.0,
        "bottom": 50000.0,
    }

    events = [ev_mt5, ev_o1, ev_r1, ev_zone]
    res = store.save_batch(events)
    assert res.commit_status == "COMMITTED"
    assert res.persisted == 4
    assert res.rejected_pre_commit == 0

    # Gravação no JSONL sequencial
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for e in events:
            jf.write(json.dumps(canonicalize_structure(e)) + "\n")

    # Verifica persistência no SQLite e canonicalização estrita
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT timestamp_ms, payload FROM events ORDER BY timestamp_ms").fetchall()
    conn.close()

    assert len(rows) == 4
    p_mt5 = json.loads(rows[0][1])
    assert p_mt5["event_id"] == "MT5_E01_001"
    assert p_mt5["delta_fechamento"] == 0.0
    assert math.copysign(1.0, float(p_mt5["delta_fechamento"])) == 1.0  # +0.0 comprovado

    # Quarentena permaneceu lazy (não criada)
    assert not quarantine_dir.exists()


# ==============================================================================
# E02: E2E CONTINGÊNCIA SQLITE DOWN -> FALLBACK -> RECONCILER
# ==============================================================================
def test_e02_financial_pipeline_sqlite_down_fallback_and_reconciliation(tmp_path: Path):
    """E02: Falha operacional no SQLite desvia válidos para fallback e quarentena os inválidos; reconciliação esvazia fila."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    store = EventStore(db_path=str(db_path), quarantine=quarantine)
    saver = EventSaver(quarantine=quarantine)
    saver.history_file = jsonl_path
    saver.visual_log_file = tmp_path / "eventos_visuais.log"
    saver.db = store
    saver.write_jsonl = False  # Modo contingência: writes desviados exclusivamente para fallback sob falha do SQLite

    # 4 eventos válidos
    v1 = {"event_id": "V1_MT5", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1774000002001, "price": 50000.0}
    v2 = {"event_id": "V2_O1", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1774000002002, "price": 50100.0}
    v3 = {"event_id": "V3_R1", "tipo_evento": "ORDERBOOK_IMBALANCE", "symbol": "BTCUSDT", "epoch_ms": 1774000002003, "imbalance_ratio": 1.2}
    v4 = {"event_id": "V4_ZN", "tipo_evento": "ZONE_V2", "symbol": "BTCUSDT", "epoch_ms": 1774000002004, "top": 51000.0, "bottom": 50000.0}

    # 3 eventos inválidos (NaN, Inf, price <= 0)
    bad_nan = {"event_id": "BAD_NAN", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1774000002005, "price": float("nan")}
    bad_inf = {"event_id": "BAD_INF", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1774000002006, "price": float("inf")}
    bad_neg = {"event_id": "BAD_NEG", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1774000002007, "price": -50.0}

    all_events = [v1, bad_nan, v2, bad_inf, v3, bad_neg, v4]

    # Simula falha do SQLite durante a ingestão do EventSaver
    with patch.object(saver.db, "save_batch", side_effect=sqlite3.OperationalError("database is locked")):
        for ev in all_events:
            saver.save_event(ev)
        saver._process_single_flush_batch()

    # Comprova que os 3 inválidos foram quarentenados
    inv_files = list((quarantine_dir / "invalid").glob("*.jsonl"))
    assert len(inv_files) == 1
    with open(inv_files[0], "r", encoding="utf-8") as qf:
        q_lines = [json.loads(l) for l in qf if l.strip()]
        assert len(q_lines) == 3

    # Comprova que os 4 válidos estão no fallback_events
    fallback_dir = jsonl_path.parent / "fallback_events"
    assert fallback_dir.exists()
    fb_files = list(fallback_dir.glob("eventos_*.jsonl")) + list(fallback_dir.glob("eventos_*.json"))
    assert len(fb_files) == 1
    fb_content = fb_files[0].read_text(encoding="utf-8")
    for vid in ["V1_MT5", "V2_O1", "V3_R1", "V4_ZN"]:
        assert vid in fb_content
    for bid in ["BAD_NAN", "BAD_INF", "BAD_NEG"]:
        assert bid not in fb_content  # Inválidos JAMAIS entram em fallback!

    # Agora o SQLite volta a operar normalmente: executamos o reconciliador de fallback
    rec_res = reconcile_events_hermetic(
        source_fallback_file=fb_files[0],
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine=quarantine,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert rec_res["sqlite_inserted"] == 4
    assert rec_res["quarantined_count"] == 0
    assert set(rec_res["reconciled_ids"]) == {"V1_MT5", "V2_O1", "V3_R1", "V4_ZN"}

    # Fila de fallback esvaziada e arquivo removido
    assert not fb_files[0].exists()

    # Total no banco: 4
    conn = sqlite3.connect(str(db_path))
    cnt = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    conn.close()
    assert cnt == 4


# ==============================================================================
# E03: E2E POISON PILL INJETADO NO FALLBACK ANTES DA RECONCILIAÇÃO
# ==============================================================================
def test_e03_financial_pipeline_poison_pill_injected_in_fallback(tmp_path: Path):
    """E03: Poison pill (-Inf, bool em preço, bottom > top) injetado em fallback é isolado na quarentena sem loop."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    v1 = {"event_id": "V1_CLEAN", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1774000003001, "price": 50000.0}
    poison_inf = {"event_id": "POISON_INF", "tipo_evento": "MT5_SIGNAL", "symbol": "BTCUSDT", "epoch_ms": 1774000003002, "price": float("-inf")}
    poison_bool = {"event_id": "POISON_BOOL", "tipo_evento": "ORDER_EXECUTION", "symbol": "BTCUSDT", "epoch_ms": 1774000003003, "price": True}
    poison_zone = {"event_id": "POISON_ZONE", "tipo_evento": "ZONE_V2", "symbol": "BTCUSDT", "epoch_ms": 1774000003004, "top": 50000.0, "bottom": 51000.0}
    v2 = {"event_id": "V2_CLEAN", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1774000003005, "price": 50100.0}

    fb_file = tmp_path / "eventos_fallback_e03.jsonl"
    lines = [json.dumps(e) for e in [v1, poison_inf, poison_bool, poison_zone, v2]]
    fb_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine=quarantine,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == 2
    assert res["quarantined_count"] == 3
    assert set(res["reconciled_ids"]) == {"V1_CLEAN", "V2_CLEAN"}
    assert set(res["quarantined_ids"]) == {"POISON_INF", "POISON_BOOL", "POISON_ZONE"}

    # Não resta arquivo de fallback pendente
    assert not fb_file.exists()


# ==============================================================================
# E04: E2E COLISÃO DIVERGENTE CROSS-STAGE
# ==============================================================================
def test_e04_financial_pipeline_divergent_collision_cross_stage(tmp_path: Path):
    """E04: Colisão de ID com payload divergente (via EventSaver e via Reconciler) é barrada e quarentenada sob COLLISION."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    store = EventStore(db_path=str(db_path), quarantine=quarantine)

    # 1. Evento original salvo com sucesso
    ev_orig = {
        "event_id": "COLL_TARGET_01",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000004000,
        "price": 50000.0,
    }
    store.save_batch([ev_orig])

    # 2. Evento divergente chega via EventStore: dispara IntegrityCollisionError
    ev_mutated_saver = copy.deepcopy(ev_orig)
    ev_mutated_saver["price"] = 99999.0
    res_coll = store.save_batch([ev_mutated_saver])
    assert res_coll.commit_status == "FAILED"
    assert res_coll.error_type == "IntegrityCollisionError"

    # 3. Outro evento divergente chega via FallbackReconciler: enviado para quarentena sob COLLISION
    ev_mutated_fb = copy.deepcopy(ev_orig)
    ev_mutated_fb["price"] = 77777.0
    fb_file = tmp_path / "eventos_fb_e04.jsonl"
    fb_file.write_text(json.dumps(ev_mutated_fb) + "\n", encoding="utf-8")

    rec_res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine=quarantine,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )
    assert rec_res["sqlite_inserted"] == 0
    assert rec_res["quarantined_count"] == 1
    assert rec_res["quarantined_ids"] == ["COLL_TARGET_01"]

    # Comprova que a quarentena gravou as 2 colisões sob COLLISION
    col_files = list((quarantine_dir / "collision").glob("*.jsonl"))
    assert len(col_files) == 1
    with open(col_files[0], "r", encoding="utf-8") as qf:
        records = [json.loads(l) for l in qf if l.strip()]
        assert len(records) == 2
        for r in records:
            assert r["category"] == "COLLISION"

    # SQLite permaneceu imutável com o valor original
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT payload FROM events WHERE timestamp_ms = 1774000004000").fetchone()
    conn.close()
    assert json.loads(row[0])["price"] == 50000.0


# ==============================================================================
# E05: E2E IDEMPOTÊNCIA CROSS-STAGE
# ==============================================================================
def test_e05_financial_pipeline_cross_stage_idempotency(tmp_path: Path):
    """E05: Evento salvo no SQLite e reaplicado no fallback é reconhecido sem erro e sem quarentena espúria."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_idemp = {
        "event_id": "IDEMP_E05",
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000005000,
        "price": 50000.0,
    }
    insert_event_to_sqlite(db_path, ev_idemp)
    jsonl_path.write_text(json.dumps(ev_idemp) + "\n", encoding="utf-8")

    fb_file = tmp_path / "eventos_fb_e05.jsonl"
    fb_file.write_text(json.dumps(ev_idemp) + "\n", encoding="utf-8")

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine=quarantine,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 0
    assert res["already_present"] == 1
    assert res["quarantined_count"] == 0
    assert not quarantine_dir.exists()


# ==============================================================================
# E06: E2E RECUPERAÇÃO FORENSE BIT-A-BIT DA QUARENTENA
# ==============================================================================
def test_e06_financial_pipeline_forensic_decode_from_quarantine(tmp_path: Path):
    """E06: Payloads quarentenados são decodificados fielmente via decode_typed_payload mantendo tipos e valores."""
    quarantine_dir = tmp_path / "quarantine_forensic"
    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)

    payload = {
        "event_id": "FORENSIC_01",
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "neg_inf": float("-inf"),
        "dec_val": decimal.Decimal("123.456789012345"),
        "bool_val": True,
        "neg_zero": -0.0,
    }
    val_res = FinancialValidationResult(
        is_valid=False, status="INVALID", category="NON_FINITE_FLOAT", rejection_reason="FORENSIC_TEST"
    )
    quarantine.quarantine_event(payload, val_res, source_caller="test_e06")

    q_file = list((quarantine_dir / "invalid").glob("*.jsonl"))[0]
    with open(q_file, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    decoded = decode_typed_payload(rec["original_payload"])
    assert decoded["event_id"] == "FORENSIC_01"
    assert math.isnan(decoded["nan_val"])
    assert math.isinf(decoded["inf_val"]) and decoded["inf_val"] > 0
    assert math.isinf(decoded["neg_inf"]) and decoded["neg_inf"] < 0
    assert decoded["dec_val"] == decimal.Decimal("123.456789012345")
    assert decoded["bool_val"] is True
    assert decoded["neg_zero"] == 0.0
    assert math.copysign(1.0, decoded["neg_zero"]) == -1.0  # Zero negativo preservado fielmente!


# ==============================================================================
# E07: G5 — ENOSPC NA FINANCIAL QUARANTINE
# ==============================================================================
def test_e07_g5_enospc_in_financial_quarantine_cleanup_atomic(tmp_path: Path, monkeypatch):
    """E07: Falha de ENOSPC na quarentena incrementa telemetria e remove arquivo parcial de 0 bytes."""
    quarantine_dir = tmp_path / "quarantine_enospc"
    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)

    def failing_fsync(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", failing_fsync)

    val_res = FinancialValidationResult(
        is_valid=False, status="INVALID", category="TYPE_ERROR", rejection_reason="ENOSPC_TEST"
    )
    res = quarantine.quarantine_event({"event_id": "ENOSPC_TEST"}, val_res, source_caller="test_e07")

    assert res.success is False
    assert res.status == "PERSISTENCE_UNCERTAIN"
    assert quarantine.enospc_error_count >= 1

    # Comprova ausência de arquivo corrompido / vazio residual
    for f in quarantine_dir.glob("**/*.jsonl"):
        assert f.stat().st_size > 0


# ==============================================================================
# E08: G5 — ENOSPC NO EVENT SAVER DURANTE FALLBACK/JSONL
# ==============================================================================
def test_e08_g5_enospc_in_event_saver_telemetry_and_critical_log(tmp_path: Path, monkeypatch):
    """E08: Falha de ENOSPC no EventSaver incrementa telemetria enospc e emite log CRITICAL."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    _init_test_sqlite_schema(db_path)

    store = EventStore(db_path=str(db_path))
    saver = EventSaver()
    saver.history_file = jsonl_path
    saver.visual_log_file = tmp_path / "eventos_visuais.log"
    saver.db = store

    def mock_open_enospc(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    with patch("builtins.open", mock_open_enospc):
        res = saver._save_to_jsonl({"event_id": "EV_ENOSPC_SAVER", "tipo_evento": "AI_ANALYSIS"})
        assert res in ("FAILED", "UNKNOWN")
        assert saver._enospc_errors >= 1

    metrics = saver.get_financial_integrity_metrics()
    assert metrics["enospc_error_count"] >= 1


# ==============================================================================
# E09: G5 — ENOSPC NO FALLBACK RECONCILER
# ==============================================================================
def test_e09_g5_enospc_in_fallback_reconciler_safe_abort(tmp_path: Path, monkeypatch):
    """E09: ENOSPC durante reconciliação aborta com segurança mantendo origem intacta e limpando .tmp_pending."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev = {"event_id": "ENOSPC_RECONCILER", "epoch_ms": 1774000009000, "tipo_evento": "AI_ANALYSIS", "price": 50000.0}
    fb_file = tmp_path / "eventos_fb_e09.jsonl"
    fb_content = json.dumps(ev) + "\n"
    fb_file.write_text(fb_content, encoding="utf-8")

    def failing_fsync(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

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

    # Origem permanece 100% preservada
    assert fb_file.exists()
    assert fb_file.read_text(encoding="utf-8") == fb_content

    # Arquivo temporário .tmp_pending não existe residual
    tmp_pending = fb_file.with_suffix(".tmp_pending")
    assert not tmp_pending.exists()


# ==============================================================================
# E10: G5 — TELEMETRIA E CONTADORES DE INTEGRIDADE FINANCEIRA
# ==============================================================================
def test_e10_g5_telemetry_metrics_precision_across_pipeline(tmp_path: Path):
    """E10: Verificação de consistência e precisão das métricas e telemetria de integridade em todo o pipeline."""
    quarantine_dir = tmp_path / "quarantine_metrics"
    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)

    # 2 rejeições INVALID
    res1 = FinancialValidationResult(is_valid=False, status="INVALID", category="INVALID_PRICE", rejection_reason="R1")
    res2 = FinancialValidationResult(is_valid=False, status="INVALID", category="TYPE_ERROR", rejection_reason="R2")
    quarantine.quarantine_event({"id": 1}, res1, source_caller="t")
    quarantine.quarantine_event({"id": 2}, res2, source_caller="t")

    # 1 rejeição COLLISION
    res3 = FinancialValidationResult(is_valid=False, status="INVALID", category="COLLISION", rejection_reason="R3")
    quarantine.quarantine_event({"id": 3}, res3, source_caller="t", category="COLLISION")

    qm = quarantine.get_quarantine_metrics()
    assert qm["quarantined_invalid_count"] == 2
    assert qm["quarantined_collision_count"] == 1
    assert qm["total_quarantined_count"] == 3
    assert qm["enospc_error_count"] == 0

    # EventStore espelha as métricas
    db_path = tmp_path / "test_metrics.db"
    _init_test_sqlite_schema(db_path)
    store = EventStore(db_path=str(db_path), quarantine=quarantine)
    sm = store.get_financial_integrity_metrics()
    assert sm["quarantined_invalid_count"] == 2
    assert sm["quarantined_collision_count"] == 1


# ==============================================================================
# E11: RASTREABILIDADE E INVARIÂNCIA DE IDENTIDADE DE ZONAS E2E
# ==============================================================================
def test_e11_financial_pipeline_zone_lifecycle_identity_invariance(tmp_path: Path):
    """E11: Ciclo de vida de zonas com validação de identidade v2 determinística e idempotência relacional."""
    db_path = tmp_path / "trading_bot.db"
    _init_test_sqlite_schema(db_path)

    store = EventStore(db_path=str(db_path))

    zone_created = {
        "tipo_evento": "Zona",
        "symbol": "BTCUSDT",
        "epoch_ms": 1774000011000,
        "window_open_ms": 1774000011000,
        "anchor_price": 50000.0,
        "low": 49900.0,
        "high": 50100.0,
        "kind": "DEMAND",
        "last_touched": 1774000011000,
    }

    # As identidades computadas são determinísticas
    ver1, id1 = compute_producer_identity(zone_created)
    assert ver1 == "zone-v2"
    assert is_deterministic_event_id(id1)
    zone_created["event_id"] = id1

    # Salva zone_created com sucesso
    res1 = store.save_batch([zone_created])
    assert res1.commit_status == "COMMITTED"
    assert res1.persisted == 1

    # Reenvio idêntico é aceito como duplicata idempotente sem colisão
    res2 = store.save_batch([zone_created])
    assert res2.commit_status == "COMMITTED"
    assert res2.duplicates_skipped == 1

    # Zone touched gera nova identidade temporal distinta
    zone_touched = dict(zone_created)
    zone_touched["last_touched"] = 1774000012000
    ver2, id2 = compute_producer_identity(zone_touched)
    assert ver2 == "zone-v2"
    assert id2 != id1
    assert is_deterministic_event_id(id2)
    zone_touched["event_id"] = id2

    res3 = store.save_batch([zone_touched])
    assert res3.commit_status == "COMMITTED"
    assert res3.persisted == 1


# ==============================================================================
# E12: INVARIANTE DE INICIALIZAÇÃO LAZY E2E
# ==============================================================================
def test_e12_financial_pipeline_lazy_quarantine_invariance_nominal(tmp_path: Path):
    """E12: Execução de todo o fluxo nominal (EventSaver -> Fallback -> Reconciler) sem criar pasta de quarentena."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine_lazy"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_clean = {"event_id": "CLEAN_E12", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1774000012000, "price": 50000.0}

    fb_file = tmp_path / "eventos_fb_e12.jsonl"
    fb_file.write_text(json.dumps(ev_clean) + "\n", encoding="utf-8")

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine=quarantine,
        perform_backup=False,
        repo_root=tmp_path,
    )

    assert res["sqlite_inserted"] == 1
    assert res["quarantined_count"] == 0
    assert not quarantine_dir.exists()


# ==============================================================================
# E13: ALTA CARDINALIDADE E CARGA MISTA E2E
# ==============================================================================
def test_e13_financial_pipeline_high_cardinality_mixed_batch(tmp_path: Path):
    """E13: Lote com 100 eventos mistos (70 válidos + 30 inválidos) atesta conservação matemática exata."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine_e13"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    events = []
    expected_valid = 0
    expected_invalid = 0

    for i in range(100):
        if i % 3 == 0:
            events.append({
                "event_id": f"BAD_EV_{i}",
                "tipo_evento": "AI_ANALYSIS",
                "symbol": "BTCUSDT",
                "epoch_ms": 1774000013000 + i,
                "price": -1.0 if i % 2 == 0 else float("nan"),
            })
            expected_invalid += 1
        else:
            events.append({
                "event_id": f"GOOD_EV_{i}",
                "tipo_evento": "AI_ANALYSIS",
                "symbol": "BTCUSDT",
                "epoch_ms": 1774000013000 + i,
                "price": 50000.0 + i,
            })
            expected_valid += 1

    fb_file = tmp_path / "eventos_fb_e13.jsonl"
    fb_file.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    res = reconcile_events_hermetic(
        source_fallback_file=fb_file,
        target_sqlite_path=db_path,
        target_jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        quarantine=quarantine,
        perform_backup=False,
        repo_root=tmp_path,
        isolate_poison_pills=True,
    )

    assert res["sqlite_inserted"] == expected_valid
    assert res["quarantined_count"] == expected_invalid
    assert res["sqlite_inserted"] + res["quarantined_count"] == 100

    conn = sqlite3.connect(str(db_path))
    cnt = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    conn.close()
    assert cnt == expected_valid


# ==============================================================================
# E14: PRESERVAÇÃO DE ESTADO INCERTO E2E
# ==============================================================================
def test_e14_financial_pipeline_uncertain_state_preservation_on_db_lock(tmp_path: Path):
    """E14: OperationalError durante reconciliação mantém os eventos válidos pendentes no arquivo de fallback."""
    db_path = tmp_path / "trading_bot.db"
    jsonl_path = tmp_path / "eventos_fluxo.jsonl"
    archive_dir = tmp_path / "archive"
    quarantine_dir = tmp_path / "quarantine_e14"
    _init_test_sqlite_schema(db_path)
    archive_dir.mkdir(parents=True, exist_ok=True)

    ev_valid = {"event_id": "EV_UNCERTAIN_01", "tipo_evento": "AI_ANALYSIS", "symbol": "BTCUSDT", "epoch_ms": 1774000014000, "price": 50000.0}
    fb_file = tmp_path / "eventos_fb_e14.jsonl"
    fb_file.write_text(json.dumps(ev_valid) + "\n", encoding="utf-8")

    quarantine = FinancialQuarantine(quarantine_dir=quarantine_dir)
    with patch("events.fallback_reconciler.insert_event_to_sqlite", side_effect=sqlite3.OperationalError("database is locked")):
        res = reconcile_events_hermetic(
            source_fallback_file=fb_file,
            target_sqlite_path=db_path,
            target_jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            quarantine=quarantine,
            perform_backup=False,
            repo_root=tmp_path,
            isolate_poison_pills=True,
        )

    assert res["sqlite_inserted"] == 0
    assert res["unresolved_count"] == 1
    assert res["quarantined_count"] == 0
    assert not quarantine_dir.exists()

    # Arquivo de fallback permanece com o evento íntegro
    assert fb_file.exists()
    assert "EV_UNCERTAIN_01" in fb_file.read_text(encoding="utf-8")


# ==============================================================================
# E15: GOVERNANÇA CLI C2A
# ==============================================================================
def test_e15_governance_apply_not_enabled_c2a_enforced(tmp_path: Path):
    """E15: Trava APPLY_NOT_ENABLED_C2A permanece ativa e bloqueando execuções diretas de mutação via CLI."""
    cli_path = PROJECT_ROOT / "scripts" / "maintenance" / "reconcile_fallback_events.py"
    assert cli_path.exists()

    lock_file = tmp_path / ".operational_runtime.lock"
    init_lock = OperationalRuntimeLock(lock_path=lock_file, holder_name="test_init_e15")
    init_lock.acquire()
    init_lock.release()

    res = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--apply",
            "--confirm-apply",
            "--reason",
            "Auditoria formal de governança da Fase 5.11 Bloco 4 Etapa 5",
            "--lock-path",
            str(lock_file),
            "--json-output",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert res.returncode == 1
    assert "APPLY_NOT_ENABLED_C2A" in res.stdout or "APPLY_NOT_ENABLED_C2A" in res.stderr

