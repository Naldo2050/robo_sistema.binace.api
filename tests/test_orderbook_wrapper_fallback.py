# tests/test_orderbook_wrapper_fallback.py
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
    window_count: int = 1
    should_stop: bool = False
    is_cleaning_up: bool = False

    # Analyzer (não usado diretamente no fallback)
    orderbook_analyzer: Any = None

    last_valid_orderbook: Optional[Dict[str, Any]] = None
    last_valid_orderbook_time: float = 0.0
    orderbook_fetch_failures: int = 0
    orderbook_emergency_mode: bool = True

    _orderbook_refresh_lock: Any = None
    _orderbook_background_refresh: bool = False
    _orderbook_bg_min_interval: float = 999.0
    _last_async_ob_refresh: float = 0.0
    _orderbook_refresh_thread: Optional[threading.Thread] = None

    # usados no wrapper
    orderbook_top_n: int = 20
    orderbook_limit: int = 100

    # loop (não usado no fallback)
    _async_loop: Any = None
    _async_loop_thread: Any = None

    def __post_init__(self):
        if self._orderbook_refresh_lock is None:
            self._orderbook_refresh_lock = threading.Lock()


def test_orderbook_fallback_cache_and_deepcopy(monkeypatch):
    bot = FakeBot()
    bot.last_valid_orderbook = {"schema_version": "2.1.0", "orderbook_data": {"bid_depth_usd": 123.0}}
    bot.last_valid_orderbook_time = time.time()  # fresh

    evt = obw.orderbook_fallback(bot)
    assert evt["data_quality"]["data_source"] == "cache"

    # muta retorno
    evt["orderbook_data"]["bid_depth_usd"] = 999.0

    # cache original não pode mudar (deepcopy)
    assert bot.last_valid_orderbook["orderbook_data"]["bid_depth_usd"] == 123.0


def test_orderbook_fallback_emergency_when_no_cache(monkeypatch):
    bot = FakeBot()
    bot.last_valid_orderbook = None
    bot.last_valid_orderbook_time = 0.0
    bot.orderbook_emergency_mode = True

    evt = obw.orderbook_fallback(bot)
    assert evt["is_valid"] is True
    assert evt["emergency_mode"] is True
    assert evt["data_quality"]["data_source"] == "emergency"


def test_orderbook_fallback_invalid_when_emergency_disabled(monkeypatch):
    bot = FakeBot()
    bot.last_valid_orderbook = None
    bot.last_valid_orderbook_time = 0.0
    bot.orderbook_emergency_mode = False

    evt = obw.orderbook_fallback(bot)
    assert evt["is_valid"] is False
    assert evt["should_skip"] is True
    assert evt["data_quality"]["data_source"] == "error"