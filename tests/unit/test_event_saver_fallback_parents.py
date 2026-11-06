import logging

from events.event_saver import EventSaver


def test_fallback_creates_missing_parent_directories(tmp_path):
    saver = EventSaver.__new__(EventSaver)
    saver.logger = logging.getLogger("tests.event_saver.fallback_parents")
    saver.enforce_financial_validation = False
    saver.history_file = tmp_path / "missing" / "events.jsonl"

    result = saver._save_fallback({"tipo_evento": "TEST", "valor": 1}, "json")

    assert result == "WRITTEN"
    fallback_files = list((tmp_path / "missing" / "fallback_events").glob("eventos_*.json"))
    assert len(fallback_files) == 1
