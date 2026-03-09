from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from market_orchestrator.windows import window_processor as wp


@pytest.mark.asyncio
async def test_submit_window_processes_snapshot_in_background(monkeypatch):
    processed = []
    done = threading.Event()

    def _fake_process(bot, window_data, close_ms):
        processed.append(
            {
                "bot": bot,
                "window_data": window_data,
                "close_ms": close_ms,
            }
        )
        done.set()

    monkeypatch.setattr(wp, "process_window_snapshot", _fake_process)

    processor = wp.WindowProcessor(
        symbol="BTCUSDT",
        windows_minutes=[1, 5],
        event_bus=object(),
        time_manager=object(),
    )

    bot = SimpleNamespace(symbol="BTCUSDT")
    window_data = [{"p": 100.0, "q": 1.0, "T": 123}]

    await processor.start()
    try:
        assert processor.submit_window(bot, window_data, 456) is True
        assert await asyncio.to_thread(done.wait, 2.0) is True
    finally:
        await processor.stop()

    assert len(processed) == 1
    assert processed[0]["bot"] is bot
    assert processed[0]["close_ms"] == 456
    assert processed[0]["window_data"] == window_data
    assert processed[0]["window_data"] is not window_data

