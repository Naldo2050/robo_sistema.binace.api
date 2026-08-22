import importlib
import sys
from unittest.mock import Mock

import trading.outcome_tracker


def test_event_memory_uses_packaged_outcome_tracker(monkeypatch):
    """Teste de regressão mínimo: confirma que event_memory usa trading.outcome_tracker."""
    mock_instance = Mock()
    mock_cls = Mock(return_value=mock_instance)

    monkeypatch.setattr(
        trading.outcome_tracker,
        "OutcomeTracker",
        mock_cls,
    )

    # Garante que events.event_memory não está em sys.modules
    old_module = sys.modules.pop("events.event_memory", None)

    try:
        module = importlib.import_module("events.event_memory")

        mock_cls.assert_called_once_with()
        assert module._TRACKER_OK is True
        assert module._outcome_tracker is mock_instance
    finally:
        sys.modules.pop("events.event_memory", None)
        if old_module is not None:
            sys.modules["events.event_memory"] = old_module