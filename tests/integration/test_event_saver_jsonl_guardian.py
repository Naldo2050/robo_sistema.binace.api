import json
import logging

import pytest

import event_saver


@pytest.mark.parametrize("max_bytes", [200, 500])
def test_jsonl_guardian_trims_analysis_trigger(tmp_path, monkeypatch, max_bytes):
    saver = event_saver.EventSaver.__new__(event_saver.EventSaver)
    saver.logger = logging.getLogger("tests.event_saver_jsonl_guardian")
    saver.write_jsonl = True
    saver.history_file = tmp_path / "events.jsonl"
    saver.max_jsonl_bytes = max_bytes

    huge_event = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000000000,
        "janela_numero": 1,
        "event_id": "deadbeef",
        "timestamp_utc": "2026-01-01 00:00:00 UTC",
        "huge": "x" * 10000,
    }

    saver._save_to_jsonl(huge_event)

    line = saver.history_file.read_text(encoding="utf-8").splitlines()[0]
    assert len(line.encode("utf-8", errors="replace")) <= max_bytes

    data = json.loads(line)
    assert data.get("tipo_evento") == "ANALYSIS_TRIGGER"
    assert data.get("note") in {"trimmed_by_guardian", "trimmed_by_guardian_hard"}
