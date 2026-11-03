# -*- coding: utf-8 -*-
"""Coexistência base+Zona e colisões controladas em SQLite (H1-B3-D).

SQLite exclusivamente em tmp_path. Nenhum arquivo em dados/,
fallback_events/ ou memory/ é tocado. Sem operações financeiras.
"""

import sqlite3

from common.identity_contract import zone_v2_event_id
from database.event_store import EventStore

SYMBOL = "BTCUSDT"
WINDOW = "1700000060000"
EPOCH_MS = 1700000060000
BASE_ID = "b" * 64
LAST_TOUCHED = "2026-10-24T11:59:59.123+00:00"


def _base_event():
    return {
        "tipo_evento": "Exaustão",
        "ativo": SYMBOL,
        "event_id": BASE_ID,
        "epoch_ms": EPOCH_MS,
        "is_signal": True,
        "delta": 2.9,
        "window_id": WINDOW,
        "identity_version": "exhaustion-v2",
        "descricao": "Pico de compra",
    }


def _zone_event(descricao="Toque em Zona POC"):
    zone_id = zone_v2_event_id(
        SYMBOL, WINDOW, "POC", "daily", 100.0, 95.5, 104.5, LAST_TOUCHED
    )
    return {
        "tipo_evento": "Zona",
        "ativo": SYMBOL,
        "event_id": zone_id,
        "epoch_ms": EPOCH_MS,
        "is_signal": False,
        "descricao": descricao,
        "identity_version": "zone-v2",
        "zone_context": {
            "id": "zone-local-uuid",
            "kind": "POC",
            "timeframe": "daily",
            "anchor_price": 100.0,
            "low": 95.5,
            "high": 104.5,
            "last_touched": LAST_TOUCHED,
            "touch_count": 3,
        },
    }


def _physical_count(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()


def _physical_ids(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT CAST(json_extract(payload, '$.event_id') AS TEXT) FROM events"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def test_base_e_zona_coexistem_no_lote(tmp_path):
    db_path = tmp_path / "zone_identity.db"
    store = EventStore(db_path=str(db_path))

    result = store.save_batch([_base_event(), _zone_event()])

    assert result.commit_status == "COMMITTED"
    assert result.persisted == 2
    assert result.rows_inserted == 2
    assert result.rejected_pre_commit == 0
    assert _physical_count(db_path) == 2
    assert _physical_ids(db_path) == {BASE_ID, _zone_event()["event_id"]}


def test_duplicata_identica_e_deduplicada(tmp_path):
    db_path = tmp_path / "zone_identity_dup.db"
    store = EventStore(db_path=str(db_path))

    first = store.save_batch([_zone_event()])
    assert first.commit_status == "COMMITTED"
    assert first.rows_inserted == 1

    second = store.save_batch([_zone_event()])
    assert second.commit_status == "COMMITTED"
    assert second.duplicates_skipped == 1
    assert second.rows_inserted == 0
    assert second.persisted == 1
    assert _physical_count(db_path) == 1


def test_colisao_divergente_no_mesmo_batch_fail_closed(tmp_path):
    db_path = tmp_path / "zone_identity_collision_batch.db"
    store = EventStore(db_path=str(db_path))

    variant = _zone_event(descricao="payload divergente no mesmo batch")
    result = store.save_batch([_zone_event(), variant])

    assert result.commit_status == "FAILED"
    assert result.error_type == "IntegrityCollisionError"
    assert result.rows_inserted == 0
    assert _physical_count(db_path) == 0


def test_colisao_divergente_com_historico_fail_closed(tmp_path):
    db_path = tmp_path / "zone_identity_collision_db.db"
    store = EventStore(db_path=str(db_path))

    first = store.save_batch([_zone_event()])
    assert first.commit_status == "COMMITTED"

    variant = _zone_event(descricao="payload divergente vs banco")
    result = store.save_batch([variant])

    assert result.commit_status == "FAILED"
    assert result.error_type == "IntegrityCollisionError"
    assert result.persisted == 0
    assert _physical_count(db_path) == 1
