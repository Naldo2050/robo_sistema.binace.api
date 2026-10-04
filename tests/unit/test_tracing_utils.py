# tests/unit/test_tracing_utils.py
# -*- coding: utf-8 -*-
"""
Testes unitarios hermeticos para TracerWrapper (orderbook_core/tracing_utils.py).

Valida:
1. Falha ao criar span falha aberto (corpo protegido ainda executa).
2. Excecao levantada no corpo protegido propaga intacta sem erro de generator.
3. Caminho nominal cria span com atributos e executa o corpo exatamente uma vez.
4. Modo no-op quando OpenTelemetry nao esta disponivel ou tracer e None.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock
import pytest

from orderbook_core.tracing_utils import TracerWrapper


def test_start_span_nominal_execution():
    """Valida que no caminho nominal o span e criado e o corpo executa uma unica vez."""
    mock_tracer = MagicMock()
    entered_span = False
    exited_span = False

    @contextmanager
    def fake_span_cm(name, kind=None, attributes=None):
        nonlocal entered_span, exited_span
        entered_span = True
        yield MagicMock()
        exited_span = True

    mock_tracer.start_as_current_span = fake_span_cm

    tracer = TracerWrapper("test_service", "test_component", "BTCUSDT")
    tracer._tracer = mock_tracer

    body_executed_count = 0
    with tracer.start_span("test_span", {"custom_attr": 123}):
        body_executed_count += 1

    assert body_executed_count == 1
    assert entered_span is True
    assert exited_span is True


def test_start_span_fail_open_on_span_creation_error():
    """Valida que se a criacao do span falhar, o tracing falha aberto e o corpo executa."""
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.side_effect = RuntimeError("OTel tracer initialization exploded")

    tracer = TracerWrapper("test_service", "test_component", "BTCUSDT")
    tracer._tracer = mock_tracer

    body_executed = False
    with tracer.start_span("failing_span"):
        body_executed = True

    assert body_executed is True


def test_start_span_propagates_body_exception_without_generator_error():
    """Valida que excecao levantada dentro do corpo protegido propaga limpa sem double-yield."""
    mock_tracer = MagicMock()

    @contextmanager
    def fake_span_cm(name, kind=None, attributes=None):
        yield MagicMock()

    mock_tracer.start_as_current_span = fake_span_cm

    tracer = TracerWrapper("test_service", "test_component", "BTCUSDT")
    tracer._tracer = mock_tracer

    body_count = 0
    with pytest.raises(RuntimeError, match="application body failure") as exc_info:
        with tracer.start_span("protected_span"):
            body_count += 1
            raise RuntimeError("application body failure")

    assert body_count == 1
    assert "generator didn't stop after throw" not in str(exc_info.value)
    assert str(exc_info.value) == "application body failure"


def test_start_span_noop_when_tracer_is_none():
    """Valida comportamento no-op estrito quando tracer e None."""
    tracer = TracerWrapper("test_service", "test_component", "BTCUSDT")
    tracer._tracer = None

    body_count = 0
    with tracer.start_span("noop_span"):
        body_count += 1

    assert body_count == 1
