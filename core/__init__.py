"""core — Fonte Única de Verdade para dados de janela."""

from core.window_state import (
    WindowState,
    PriceData,
    VolumeData,
    IndicatorData,
    OrderBookData,
    FlowData,
    MacroData,
    DerivativesData,
    OnChainData,
)
from core.state_manager import StateManager

__all__ = [
    "WindowState",
    "PriceData",
    "VolumeData",
    "IndicatorData",
    "OrderBookData",
    "FlowData",
    "MacroData",
    "DerivativesData",
    "OnChainData",
    "StateManager",
]
