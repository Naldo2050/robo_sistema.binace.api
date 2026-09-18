# tests/unit/test_orderbook_sync_snapshot.py
# -*- coding: utf-8 -*-
"""
Teste unitário para snapshot síncrono do OrderBook no fechamento da janela:
- Sem atraso -> source="live_sync" e snapshot_offset_ms < 500
- Com atraso > timeout -> source="cache_bg" usando cache background
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pytest
from market_orchestrator.orderbook import orderbook_wrapper as obw


@dataclass
class MockBot:
    symbol: str = "BTCUSDT"
    market_symbol: str = "BTCUSDT"
    window_count: int = 12
    should_stop: bool = False
    is_cleaning_up: bool = False

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

    orderbook_top_n: int = 20
    orderbook_limit: int = 100
    _async_loop: Any = None
    _async_loop_thread: Any = None

    def __post_init__(self):
        if self._orderbook_refresh_lock is None:
            self._orderbook_refresh_lock = threading.Lock()


def test_orderbook_sync_fast_produces_live_sync(monkeypatch):
    """Fetch sem atraso produz source='live_sync' e snapshot_offset_ms < 500."""
    bot = MockBot()
    close_ms = int(time.time() * 1000)
    exchange_ms = close_ms + 120  # 120ms após o fechamento da janela

    fake_live_event = {
        "is_valid": True,
        "timestamps": {
            "exchange_ms": exchange_ms,
            "received_ms": close_ms + 130,
        },
        "orderbook_data": {
            "bid_depth_usd": 15000.0,
            "ask_depth_usd": 16000.0,
        },
    }

    # run_orderbook_analyze responde imediatamente com o evento live
    monkeypatch.setattr(
        obw,
        "run_orderbook_analyze",
        lambda _bot, _close_ms, timeout_sec=None: fake_live_event,
    )

    evt = obw.fetch_orderbook_with_retry(bot, close_ms=close_ms, timeout_sec=1.5)

    assert evt is not None
    assert evt.get("is_valid") is True
    assert evt.get("source") == "live_sync"
    assert evt.get("orderbook_data", {}).get("source") == "live_sync"

    offset_ms = evt.get("snapshot_offset_ms")
    assert offset_ms is not None
    assert offset_ms == 120
    assert abs(offset_ms) < 500
    assert evt.get("orderbook_data", {}).get("snapshot_offset_ms") == 120
    assert "timestamps" in evt.get("orderbook_data", {})


def test_orderbook_sync_timeout_falls_back_to_cache_bg(monkeypatch):
    """Fetch com atraso > timeout recorre ao cache background com source='cache_bg'."""
    bot = MockBot()
    close_ms = int(time.time() * 1000)
    bg_cached_exchange_ms = close_ms - 2500  # Capturado 2.5s antes em background

    # Inicializa cache background prévio no bot
    bot.last_valid_orderbook = {
        "is_valid": True,
        "timestamps": {
            "exchange_ms": bg_cached_exchange_ms,
            "received_ms": bg_cached_exchange_ms + 10,
        },
        "orderbook_data": {
            "bid_depth_usd": 12000.0,
            "ask_depth_usd": 13000.0,
        },
    }
    bot.last_valid_orderbook_time = time.time() - 2.5

    # Simula timeout retornando None (ou simulando demora > timeout)
    def _mock_analyze_timeout(_bot, _close_ms, timeout_sec=None):
        return None

    monkeypatch.setattr(obw, "run_orderbook_analyze", _mock_analyze_timeout)

    evt = obw.fetch_orderbook_with_retry(bot, close_ms=close_ms, timeout_sec=1.5)

    assert evt is not None
    assert evt.get("is_valid") is True
    assert evt.get("source") == "cache_bg"
    assert evt.get("orderbook_data", {}).get("source") == "cache_bg"

    offset_ms = evt.get("snapshot_offset_ms")
    assert offset_ms is not None
    assert offset_ms == int(abs(close_ms - bg_cached_exchange_ms))
    assert evt.get("cache_age_ms") == offset_ms
    assert evt.get("orderbook_data", {}).get("snapshot_offset_ms") == offset_ms
    assert "timestamps" in evt.get("orderbook_data", {})


def test_orderbook_sync_excessive_offset_triggers_fallback(monkeypatch, caplog):
    """
    Testa a guarda de SLA estrita (Cenário a):
    Quando o fetch live retorna com sucesso mas com snapshot_offset_ms > 1500ms
    (ex: delay de trade ou latência de rede acumulada com offset = 2100ms):
    (a) O dado live é descartado e substituído pelo cache_bg
    (b) Nenhuma exceção ocorre
    (c) O log registra o fallback preventivo com mensagem de offset excessivo
    """
    import logging
    bot = MockBot()
    close_ms = int(time.time() * 1000)
    bg_cached_exchange_ms = close_ms - 3000  # Cache de 3s atrás

    # Cache background inicializado
    bot.last_valid_orderbook = {
        "is_valid": True,
        "timestamps": {
            "exchange_ms": bg_cached_exchange_ms,
            "received_ms": bg_cached_exchange_ms + 5,
        },
        "orderbook_data": {
            "bid_depth_usd": 18000.0,
            "ask_depth_usd": 19000.0,
            "source": "cache_bg",
        },
    }
    bot.last_valid_orderbook_time = time.time() - 3.0

    # Simula fetch live que RETORNOU COM SUCESSO, porém com offset excessivo (2100ms)
    live_delayed_exchange_ms = close_ms + 2100
    fake_live_delayed_event = {
        "is_valid": True,
        "timestamps": {
            "exchange_ms": live_delayed_exchange_ms,
            "received_ms": live_delayed_exchange_ms + 10,
        },
        "orderbook_data": {
            "bid_depth_usd": 50000.0,
            "ask_depth_usd": 55000.0,
        },
    }

    monkeypatch.setattr(
        obw,
        "run_orderbook_analyze",
        lambda _bot, _close_ms, timeout_sec=None: fake_live_delayed_event,
    )

    with caplog.at_level(logging.WARNING):
        evt = obw.fetch_orderbook_with_retry(bot, close_ms=close_ms, timeout_sec=1.5)

    # (a) O evento live atrasado foi descartado; o evento retornado é o cache_bg
    assert evt is not None
    assert evt.get("is_valid") is True
    assert evt.get("source") == "cache_bg"
    assert evt.get("source_type") == "cache_bg"
    assert evt.get("orderbook_data", {}).get("source") == "cache_bg"
    # O conteúdo veio do cache (18000), NÃO do live atrasado (50000)
    assert evt.get("orderbook_data", {}).get("bid_depth_usd") == 18000.0

    # (b) Nenhuma exceção e campos íntegros (positivo harmonizado)
    offset_ms = evt.get("snapshot_offset_ms")
    assert offset_ms == int(abs(close_ms - bg_cached_exchange_ms))
    assert offset_ms > 0
    assert evt.get("cache_age_ms") == 3000

    # (c) O log registrou o acionamento do fallback pela guarda de SLA
    assert any("Snapshot obtido com offset excessivo" in record.message for record in caplog.records)
    assert any("Recorrendo ao cache background" in record.message for record in caplog.records)

