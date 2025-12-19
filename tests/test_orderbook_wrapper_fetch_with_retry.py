# tests/test_orderbook_wrapper_fetch_with_retry.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

from market_orchestrator.orderbook import orderbook_wrapper as obw


@dataclass
class FakeBot:
    symbol: str = "BTCUSDT"
    market_symbol: str = "BTCUSDT"
    window_count: int = 7
    should_stop: bool = False
    is_cleaning_up: bool = False

    orderbook_analyzer: Any = None

    last_valid_orderbook: Optional[Dict[str, Any]] = None
    last_valid_orderbook_time: float = 0.0
    orderbook_fetch_failures: int = 5
    orderbook_emergency_mode: bool = True

    _orderbook_refresh_lock: Any = None
    _orderbook_background_refresh: bool = False
    _orderbook_bg_min_interval: float = 999.0
    _last_async_ob_refresh: float = 0.0
    _orderbook_refresh_thread: Optional[threading.Thread] = None

    orderbook_top_n: int = 20
    orderbook_limit: int = 100

    _async_loop: Any = None
    _async_loop_thread: Any = None

    def __post_init__(self):
        if self._orderbook_refresh_lock is None:
            self._orderbook_refresh_lock = threading.Lock()


def test_fetch_orderbook_with_retry_saves_success(monkeypatch):
    bot = FakeBot()

    fake_event = {
        "is_valid": True,
        "orderbook_data": {"bid_depth_usd": 9999.0, "ask_depth_usd": 9999.0},
    }

    # monkeypatch do run_orderbook_analyze para evitar loop/async real
    monkeypatch.setattr(obw, "run_orderbook_analyze", lambda _bot, _close_ms: fake_event)

    evt = obw.fetch_orderbook_with_retry(bot, close_ms=int(time.time() * 1000))
    assert evt["is_valid"] is True

    assert bot.last_valid_orderbook is not None
    assert bot.orderbook_fetch_failures == 0
    assert bot.last_valid_orderbook_time > 0

    # deepcopy: mutar retorno não altera o cache no bot
    evt["orderbook_data"]["bid_depth_usd"] = 1.0
    assert bot.last_valid_orderbook["orderbook_data"]["bid_depth_usd"] == 9999.0