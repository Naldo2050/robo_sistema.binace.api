# orderbook_core/__init__.py
"""
Pacote núcleo do orderbook.

Contém: OrderBookConfig, OrderBookSnapshot, CircuitBreaker,
OrderBookFallback, StructuredLogger, EventFactory.
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig  # noqa: F401
from .constants import THRESHOLDS  # noqa: F401
from .event_factory import build_emergency_orderbook_event, build_invalid_orderbook_event  # noqa: F401
from .exceptions import InvalidUpdateError, OrderBookError  # noqa: F401
from .metrics import MetricsTracker, OrderBookMetrics  # noqa: F401
from .orderbook import OrderBookSnapshot  # noqa: F401
from .orderbook_config import OrderBookConfig  # noqa: F401
from .orderbook_fallback import (  # noqa: F401
    FallbackConfig,
    OrderBookFallback,
    fetch_with_fallback,
    get_fallback_instance,
)
from .protocols import BotProtocol, OrderBookAnalyzerProtocol, TimeManagerProtocol  # noqa: F401
from .structured_logging import StructuredLogger  # noqa: F401
from .tracing_utils import TracerWrapper  # noqa: F401

__all__ = [
    "BotProtocol",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "FallbackConfig",
    "InvalidUpdateError",
    "MetricsTracker",
    "OrderBookAnalyzerProtocol",
    "OrderBookConfig",
    "OrderBookError",
    "OrderBookFallback",
    "OrderBookMetrics",
    "OrderBookSnapshot",
    "StructuredLogger",
    "THRESHOLDS",
    "TimeManagerProtocol",
    "TracerWrapper",
    "build_emergency_orderbook_event",
    "build_invalid_orderbook_event",
    "fetch_with_fallback",
    "get_fallback_instance",
]
