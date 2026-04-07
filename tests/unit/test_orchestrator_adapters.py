from __future__ import annotations

from types import SimpleNamespace

import pytest

from market_orchestrator.adapters import (
    EnhancedMarketBotAdapter,
    MarketOrchestratorAdapter,
    adapt_orchestrator_runtime,
)
from market_orchestrator.orchestrator import MarketOrchestrator, OrchestratorConfig
from market_orchestrator.protocols import (
    OrchestratorControlProtocol,
    OrchestratorSnapshotProtocol,
)


@pytest.mark.asyncio
async def test_market_orchestrator_adapter_controls_and_snapshots() -> None:
    orchestrator = MarketOrchestrator(OrchestratorConfig(symbol="BTCUSDT"))
    adapter = MarketOrchestratorAdapter(orchestrator)

    assert isinstance(adapter, OrchestratorSnapshotProtocol)
    assert isinstance(adapter, OrchestratorControlProtocol)

    await adapter.start_runtime()
    snapshot_running = adapter.snapshot_state()

    assert snapshot_running["kind"] == "market_orchestrator"
    assert snapshot_running["symbol"] == "BTCUSDT"
    assert snapshot_running["is_running"] is True
    assert "health" in snapshot_running
    assert "metrics" in snapshot_running

    await adapter.stop_runtime()
    snapshot_stopped = adapter.snapshot_state()

    assert snapshot_stopped["is_running"] is False


@pytest.mark.asyncio
async def test_enhanced_market_bot_adapter_normalizes_runtime_state() -> None:
    calls: list[str] = []

    class FakeEnhancedMarketBot:
        symbol = "ETHUSDT"
        should_stop = False
        warming_up = True
        window_count = 7
        ai_analyzer = object()
        ai_test_passed = True
        ai_runner = object()
        connection_manager = object()
        health_monitor = object()
        event_bus = object()
        context_collector = object()
        ai_thread_pool = [object(), object()]
        max_ai_threads = 3
        window_ms = 300000

        async def run(self) -> None:
            calls.append("run")

        async def shutdown(self) -> None:
            self.should_stop = True
            calls.append("shutdown")

    bot = FakeEnhancedMarketBot()
    adapter = EnhancedMarketBotAdapter(bot)  # type: ignore[arg-type]

    assert isinstance(adapter, OrchestratorSnapshotProtocol)
    assert isinstance(adapter, OrchestratorControlProtocol)

    snapshot_before = adapter.snapshot_state()
    assert snapshot_before["kind"] == "enhanced_market_bot"
    assert snapshot_before["symbol"] == "ETHUSDT"
    assert snapshot_before["is_running"] is True
    assert snapshot_before["state"]["ai_ready"] is True
    assert snapshot_before["metrics"]["ai_threads"] == 2

    await adapter.start_runtime()
    await adapter.stop_runtime()

    snapshot_after = adapter.snapshot_state()
    assert snapshot_after["is_running"] is False
    assert calls == ["run", "shutdown"]


def test_adapter_factory_accepts_market_orchestrator() -> None:
    runtime = MarketOrchestrator(OrchestratorConfig(symbol="BTCUSDT"))

    adapter = adapt_orchestrator_runtime(runtime)

    assert isinstance(adapter, MarketOrchestratorAdapter)
    assert adapter.snapshot_state()["symbol"] == "BTCUSDT"


def test_adapter_factory_accepts_enhanced_market_bot_like_runtime() -> None:
    runtime = SimpleNamespace(
        symbol="BTCUSDT",
        should_stop=False,
        warming_up=False,
        window_count=0,
        ai_analyzer=None,
        ai_test_passed=False,
        ai_runner=None,
        connection_manager=None,
        health_monitor=None,
        event_bus=None,
        context_collector=None,
        ai_thread_pool=[],
        max_ai_threads=0,
        window_ms=60000,
        run=lambda: None,
        shutdown=lambda: None,
    )

    adapter = adapt_orchestrator_runtime(runtime)

    assert isinstance(adapter, EnhancedMarketBotAdapter)
    assert adapter.snapshot_state()["kind"] == "enhanced_market_bot"


def test_adapter_factory_rejects_unknown_runtime() -> None:
    with pytest.raises(TypeError):
        adapt_orchestrator_runtime(object())
