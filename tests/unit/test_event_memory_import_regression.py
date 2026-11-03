import importlib
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import trading.outcome_tracker

_UNSET = object()


@pytest.fixture
def load_event_memory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOAD_DOTENV", "0")

    original_events = {
        name: module
        for name, module in sys.modules.items()
        if name == "events" or name.startswith("events.")
    }

    def _drop_events_modules():
        for name in [
            name
            for name in sys.modules
            if name == "events" or name.startswith("events.")
        ]:
            sys.modules.pop(name, None)

    def _load(tracker_cls=_UNSET):
        _drop_events_modules()
        if tracker_cls is not _UNSET:
            monkeypatch.setattr(
                trading.outcome_tracker,
                "OutcomeTracker",
                tracker_cls,
            )
        return importlib.import_module("events.event_memory")

    yield _load

    _drop_events_modules()
    sys.modules.update(original_events)


def _signal_event(epoch_ms=60000):
    return {
        "timestamp": 1,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Compra",
        "preco_fechamento": 100.0,
        "epoch_ms": epoch_ms,
    }


def test_import_does_not_construct_tracker_or_create_data(
    load_event_memory,
    tmp_path,
):
    mock_instance = Mock()
    mock_cls = Mock(return_value=mock_instance)

    module = load_event_memory(mock_cls)

    assert mock_cls.call_count == 0
    assert module._outcome_tracker is None
    assert module._TRACKER_OK is False
    assert module._TRACKER_INITIALIZED is False
    assert not (tmp_path / "dados").exists()


def test_first_functional_use_initializes_once_and_reuses_instance(
    load_event_memory,
    tmp_path,
):
    mock_instance = Mock()
    mock_cls = Mock(return_value=mock_instance)

    module = load_event_memory(mock_cls)
    module.adicionar_memoria_evento({"timestamp": 1, "tipo_evento": "Outro"})
    assert mock_cls.call_count == 0

    module.adicionar_memoria_evento(_signal_event())
    assert mock_cls.call_count == 1
    assert module._outcome_tracker is mock_instance
    assert module._TRACKER_OK is True
    assert module._TRACKER_INITIALIZED is True
    assert (tmp_path / "dados").is_dir()

    module.adicionar_memoria_evento(_signal_event(120000))
    assert mock_cls.call_count == 1
    assert module._outcome_tracker is mock_instance


def test_probability_and_evaluation_reuse_same_instance(load_event_memory):
    tracker = Mock()
    tracker.get_confidence_for_event.return_value = {"has_data": False}
    tracker_cls = Mock(return_value=tracker)

    module = load_event_memory(tracker_cls)
    result = module.calcular_probabilidade_historica(_signal_event())
    assert tracker_cls.call_count == 1
    assert result["source"] == "fallback_base"

    module.avaliar_outcomes_pendentes(123.45, 60000)
    tracker.evaluate_pending_outcomes.assert_called_once_with(123.45, 60000)
    assert tracker_cls.call_count == 1


def test_concurrent_initialization_uses_single_instance(load_event_memory):
    constructions = []

    class SlowTracker:
        def __init__(self):
            time.sleep(0.05)
            constructions.append(1)

    module = load_event_memory(SlowTracker)
    with ThreadPoolExecutor(max_workers=16) as executor:
        instances = list(
            executor.map(lambda _: module._get_outcome_tracker(), range(32))
        )

    assert len(constructions) == 1
    assert all(instance is instances[0] for instance in instances)
    assert module._TRACKER_OK is True


def test_construction_failure_does_not_report_success(load_event_memory):
    class BoomTracker:
        def __init__(self):
            raise RuntimeError("boom")

    module = load_event_memory(BoomTracker)

    with pytest.raises(RuntimeError, match="boom"):
        module.adicionar_memoria_evento(_signal_event())

    assert module._TRACKER_OK is False
    assert module._outcome_tracker is None


def test_import_error_fallback_is_explicit(load_event_memory, monkeypatch):
    monkeypatch.setitem(sys.modules, "trading.outcome_tracker", None)

    module = load_event_memory()

    assert module.OutcomeTracker is None
    assert module._TRACKER_OK is False
    assert module._TRACKER_INITIALIZED is True
    assert isinstance(module._TRACKER_ERROR, ImportError)
    assert module._get_outcome_tracker() is None

    result = module.calcular_probabilidade_historica(_signal_event())
    assert result["source"] == "fallback_base"


def test_ensure_table_failure_is_known_false_positive(
    load_event_memory,
    monkeypatch,
    tmp_path,
):
    def failing_get_conn(self):
        raise sqlite3.OperationalError("synthetic ensure_table failure")

    monkeypatch.setattr(
        trading.outcome_tracker.OutcomeTracker,
        "_get_conn",
        failing_get_conn,
    )

    module = load_event_memory(trading.outcome_tracker.OutcomeTracker)
    tracker = module._get_outcome_tracker()

    assert tracker is not None
    assert not (tmp_path / "dados" / "trading_bot.db").exists()
    assert module._TRACKER_OK is True
    pytest.xfail(
        "Falso positivo confirmado: OutcomeTracker._ensure_table captura a "
        "falha, o objeto e construido, nenhum DB operacional existe e "
        "event_memory marca _TRACKER_OK=True. Contrato pendente em producao."
    )
