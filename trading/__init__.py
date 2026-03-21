# trading/__init__.py
"""
Pacote de trading do sistema.

Contém: AsyncTradeBuffer, AlertEngine, AlertManager,
TradeValidator, ExportSignals, OutcomeTracker.
"""

from .alert_engine import generate_alerts, create_alert_summary  # noqa: F401
from .alert_manager import AlertManager, get_alert_manager  # noqa: F401
from .export_signals import create_chart_signal_from_event, export_signal_to_csv  # noqa: F401
from .trade_buffer import AsyncTradeBuffer, BufferStatus  # noqa: F401
from .trade_validator import validate_and_filter_trades, TradeLatencyMonitor  # noqa: F401

__all__ = [
    "AlertManager",
    "AsyncTradeBuffer",
    "BufferStatus",
    "TradeLatencyMonitor",
    "create_alert_summary",
    "create_chart_signal_from_event",
    "export_signal_to_csv",
    "generate_alerts",
    "get_alert_manager",
    "validate_and_filter_trades",
]
