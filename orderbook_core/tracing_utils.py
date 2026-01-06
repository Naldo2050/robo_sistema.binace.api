# pyright: reportMissingImports=false
# tracing_utils.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Dict, Any

import logging

try:
    # OpenTelemetry opcional
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
    _OTEL_AVAILABLE = True
except Exception:
    trace = None  # type: ignore
    SpanKind = None  # type: ignore
    _OTEL_AVAILABLE = False


class TracerWrapper:
    """
    Wrapper simples para OpenTelemetry.

    - Se opentelemetry NÃO estiver instalado, tudo é no-op.
    - Se estiver, cria spans com atributos básicos.

    Uso:
        tracer = TracerWrapper(service_name="orderbook", component="analyzer", symbol="BTCUSDT")

        with tracer.start_span("orderbook_analyze", {"window_id": "W0001"}):
            ... código ...
    """

    def __init__(self, service_name: str, component: str, symbol: str):
        self.service_name = service_name
        self.component = component
        self.symbol = symbol

        if _OTEL_AVAILABLE:
            try:
                self._tracer = trace.get_tracer(service_name)
            except Exception:
                self._tracer = None
        else:
            self._tracer = None

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager para criar span. Se OpenTelemetry não estiver disponível,
        apenas executa o bloco sem fazer nada.
        """
        if not _OTEL_AVAILABLE or self._tracer is None:
            # no-op
            yield
            return

        attrs = {
            "component": self.component,
            "symbol": self.symbol,
        }
        if attributes:
            attrs.update(attributes)

        try:
            with self._tracer.start_as_current_span(
                name,
                kind=SpanKind.INTERNAL if SpanKind else None,
                attributes=attrs,
            ):
                yield
        except Exception as e:
            # tracing não pode derrubar o fluxo principal
            logging.debug(f"Tracing span error ({name}): {e}", exc_info=True)
            yield