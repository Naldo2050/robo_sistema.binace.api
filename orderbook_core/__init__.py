# orderbook_core/__init__.py
"""
Pacote nucleo do orderbook.

Contém: OrderBookConfig, CircuitBreaker, OrderBookFallback, StructuredLogger.
"""

from .orderbook_config import OrderBookConfig
from .metrics import OrderBookMetrics, MetricsTracker
from .structured_logging import StructuredLogger
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .tracing_utils import TracerWrapper
from .constants import THRESHOLDS
from .protocols import TimeManagerProtocol, OrderBookAnalyzerProtocol, BotProtocol
from .event_factory import build_invalid_orderbook_event, build_emergency_orderbook_event
from .exceptions import OrderBookError, InvalidUpdateError
from .orderbook_fallback import OrderBookFallback, FallbackConfig, get_fallback_instance, fetch_with_fallback

__all__ = [
    "OrderBookConfig",
    "OrderBookMetrics",
    "MetricsTracker",
    "StructuredLogger",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "TracerWrapper",
    "THRESHOLDS",
    "OrderBookError",
    "InvalidUpdateError",
    "OrderBookFallback",
    "FallbackConfig",
    "get_fallback_instance",
    "fetch_with_fallback",
]