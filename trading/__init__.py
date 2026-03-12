# trading/__init__.py
"""
Pacote de trading do sistema.

Contém: AsyncTradeBuffer, TradeValidator, AlertEngine, AlertManager.
"""

from .trade_buffer import AsyncTradeBuffer, BufferStatus

__all__ = [
    "AsyncTradeBuffer",
    "BufferStatus",
]
