# protocols.py
from __future__ import annotations

from typing import Protocol, Optional, Dict, Any
import asyncio
import threading
from _thread import LockType


class TimeManagerProtocol(Protocol):
    """Contrato mínimo para TimeManager (permite mock em testes)."""

    def now_ms(self) -> int: ...

    def build_time_index(
        self,
        epoch_ms: int,
        include_local: bool = False,
        timespec: str = "seconds",
    ) -> Dict[str, Any]: ...


class OrderBookAnalyzerProtocol(Protocol):
    """Contrato formal para qualquer analyzer compatível com o wrapper."""

    symbol: str

    async def analyze(
        self,
        current_snapshot: Optional[Dict[str, Any]] = None,
        *,
        event_epoch_ms: Optional[int] = None,
        window_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    async def close(self) -> None: ...

    def get_stats(self) -> Dict[str, Any]: ...

    def reset_stats(self) -> None: ...


class BotProtocol(Protocol):
    """
    Contrato mínimo para o bot que usa orderbook_wrapper.
    Mantém explícitos os atributos que o wrapper acessa/modifica.
    """

    # Identidade / execução
    symbol: str
    window_count: int
    should_stop: bool
    is_cleaning_up: bool

    # Analyzer
    orderbook_analyzer: OrderBookAnalyzerProtocol

    # Estado do orderbook (cache/fallback)
    last_valid_orderbook: Optional[Dict[str, Any]]
    last_valid_orderbook_time: float
    orderbook_fetch_failures: int
    orderbook_emergency_mode: bool

    # Async loop em thread dedicada
    _async_loop: asyncio.AbstractEventLoop
    _async_loop_thread: threading.Thread

    # Background refresh do orderbook
    _orderbook_refresh_lock: LockType
    _orderbook_background_refresh: bool
    _orderbook_bg_min_interval: float
    _last_async_ob_refresh: float
    _orderbook_refresh_thread: Optional[threading.Thread]

    # (Opcional no seu fallback)
    orderbook_top_n: int
    orderbook_limit: int
    market_symbol: str