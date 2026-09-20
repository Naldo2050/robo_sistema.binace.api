# paper_trading/adapters/__init__.py
"""
Adapters for translating runtime signals, market events, and risk rules into paper trading contracts.
"""

from paper_trading.adapters.risk_adapter import (
    RiskAdapter,
    RiskAdapterResult,
    RiskStatus,
)
from paper_trading.adapters.signal_adapter import (
    AdapterMode,
    AdapterResult,
    AdapterStatus,
    SignalDecisionAdapter,
)

__all__ = [
    "AdapterMode",
    "AdapterResult",
    "AdapterStatus",
    "RiskAdapter",
    "RiskAdapterResult",
    "RiskStatus",
    "SignalDecisionAdapter",
]
