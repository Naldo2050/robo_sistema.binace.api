"""
Testes das metricas Prometheus do EventBus.

Cobrem o item 3 da auditoria (metrica de handlers registrados e eventos
entregues) e o item 4 (alerta/metrica de evento publicado sem handler),
com a distincao explicita entre:
  (a) nenhum handler registrado  -> trading_event_bus_events_without_handler_total
  (b) handler registrado que lanca excecao -> trading_event_bus_handler_errors_total

Protocolo de prova: cada teste referencia as metricas criadas no import de
events/event_bus.py; se o fix for revertido (sem metricas / sem incremento /
WARNING voltando a DEBUG), os testes falham.
"""
import time

import pytest

pytest.importorskip("prometheus_client")
from prometheus_client import REGISTRY

import events.event_bus as eb_mod
from events.event_bus import EventBus

HANDLERS_GAUGE = "trading_event_bus_handlers_registered"
DELIVERED_TOTAL = "trading_event_bus_events_delivered_total"
WITHOUT_HANDLER_TOTAL = "trading_event_bus_events_without_handler_total"
HANDLER_ERRORS_TOTAL = "trading_event_bus_handler_errors_total"


def _sample(metric_name: str, labels: dict) -> float:
    """Valor da serie com os labels (0.0 se a serie nunca foi emitida)."""
    value = REGISTRY.get_sample_value(metric_name, labels)
    return value if value is not None else 0.0


def _wait_processed(bus: EventBus, timeout: float = 2.0) -> None:
    """Polling ate a fila esvaziar (sem sleep fixo)."""
    deadline = time.monotonic() + timeout
    while bus._queue and time.monotonic() < deadline:
        time.sleep(0.01)


def test_metric_names_created_at_module_level():
    """As metricas existem e foram registradas no registry default."""
    for name in (
        HANDLERS_GAUGE,
        DELIVERED_TOTAL,
        WITHOUT_HANDLER_TOTAL,
        HANDLER_ERRORS_TOTAL,
    ):
        assert name in REGISTRY._names_to_collectors


def test_subscribe_sets_handlers_registered_gauge():
    """Gauge de handlers registrados reflete a contagem por event_type."""
    bus = EventBus()
    try:
        assert _sample(HANDLERS_GAUGE, {"event_type": "metrics_gauge"}) == 0.0

        bus.subscribe("metrics_gauge", lambda e: None)
        assert _sample(HANDLERS_GAUGE, {"event_type": "metrics_gauge"}) == 1

        bus.subscribe("metrics_gauge", lambda e: None)
        assert _sample(HANDLERS_GAUGE, {"event_type": "metrics_gauge"}) == 2
    finally:
        bus.shutdown()


def test_publish_with_handler_increments_delivered():
    """Evento entregue com sucesso incrementa delivered (item 3)."""
    bus = EventBus()
    try:
        received = []
        bus.subscribe("metrics_delivered", lambda e: received.append(e))
        before = _sample(DELIVERED_TOTAL, {"event_type": "metrics_delivered"}) or 0

        bus.publish("metrics_delivered", {"seq": 1, "timestamp": time.time()})
        _wait_processed(bus)

        assert len(received) == 1
        assert _sample(DELIVERED_TOTAL, {"event_type": "metrics_delivered"}) == before + 1
        assert _sample(WITHOUT_HANDLER_TOTAL, {"event_type": "metrics_delivered"}) == before
    finally:
        bus.shutdown()


def test_publish_without_handler_increments_counter_and_warns(caplog):
    """(a) Evento sem handler: contador proprio + WARNING (item 4)."""
    bus = EventBus()
    try:
        with caplog.at_level("WARNING", logger="EventBus"):
            bus.publish("metrics_void", {"msg": "hello", "timestamp": time.time()})
            _wait_processed(bus)

        assert _sample(WITHOUT_HANDLER_TOTAL, {"event_type": "metrics_void"}) == 1
        assert any("Nenhum handler" in r.message for r in caplog.records)

        bus.publish("metrics_void", {"msg": "hello2", "timestamp": time.time() + 1})
        _wait_processed(bus)
        assert _sample(WITHOUT_HANDLER_TOTAL, {"event_type": "metrics_void"}) == 2
    finally:
        bus.shutdown()


def test_handler_error_increments_errors_not_delivered():
    """(b) Handler registrado que lanca excecao: contador proprio, sem contar delivered."""
    bus = EventBus()
    try:
        def broken_handler(event):
            raise ValueError("boom")

        bus.subscribe("metrics_broken", broken_handler)
        before_err = _sample(HANDLER_ERRORS_TOTAL, {"event_type": "metrics_broken"}) or 0
        before_del = _sample(DELIVERED_TOTAL, {"event_type": "metrics_broken"}) or 0

        bus.publish("metrics_broken", {"seq": 1, "timestamp": time.time()})
        _wait_processed(bus)

        assert _sample(HANDLER_ERRORS_TOTAL, {"event_type": "metrics_broken"}) == before_err + 1
        assert _sample(DELIVERED_TOTAL, {"event_type": "metrics_broken"}) == before_del
    finally:
        bus.shutdown()
