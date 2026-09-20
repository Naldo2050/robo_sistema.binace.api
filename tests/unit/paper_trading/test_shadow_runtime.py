# tests/unit/paper_trading/test_shadow_runtime.py
"""
Unit tests for hermetic ShadowPaperRuntime (Gate C3-C-B1).
Covers items 13 to 30.
"""

from __future__ import annotations

import builtins
import socket
from typing import Any, Dict
import pytest

from paper_trading.config import (
    ShadowPaperConfig,
    parse_shadow_config,
)
from paper_trading.shadow_runtime import (
    ShadowPaperRuntime,
    ShadowSignalResult,
    ShadowTradeResult,
    create_decision_provider,
)
from paper_trading.adapters.signal_adapter import SignalDecisionAdapter
from paper_trading.adapters.risk_adapter import RiskAdapter
from paper_trading.execution_sink import ExecutionSink
from paper_trading.executor import PaperExecutor
from paper_trading.cost_model import CostModel
from risk_management.risk_manager import RiskConfig, RiskManager


def make_valid_config(
    provider: str = "fixed_long",
    cohort_id: str = "CH_SHADOW_TEST_01",
    seed: int | None = None,
    notional: float = 1000.0,
    horizon_s: int = 300,
    ttl_ms: int = 5000,
) -> ShadowPaperConfig:
    env = {
        "PAPER_SHADOW_ENABLED": "1",
        "PAPER_COHORT_ID": cohort_id,
        "PAPER_PROVIDER": provider,
        "PAPER_SYMBOL": "BTCUSDT",
        "PAPER_TIMEFRAME": "1m",
        "PAPER_NOTIONAL_USDT": str(notional),
        "PAPER_HORIZON_S": str(horizon_s),
        "PAPER_ORDER_TTL_MS": str(ttl_ms),
        "PAPER_MAKER_FEE_BPS": "2.0",
        "PAPER_TAKER_FEE_BPS": "5.0",
        "PAPER_ENTRY_SLIPPAGE_BPS": "1.0",
        "PAPER_EXIT_SLIPPAGE_BPS": "1.0",
        "PAPER_COST_SOURCE": "test_source",
        "PAPER_COST_EFFECTIVE_AT": "2026-01-01T00:00:00Z",
        "PAPER_STRATEGY_VERSION": "c3_shadow_test",
    }
    if seed is not None:
        env["PAPER_RANDOM_SEED"] = str(seed)
    res = parse_shadow_config(env)
    assert res.is_valid is True
    assert res.config is not None
    return res.config


def test_13_fixed_long_creates_long_decision():
    """13. fixed LONG cria decisão LONG."""
    current_time = 1700000000000
    config = make_valid_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: current_time)

    signal = {
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    }
    res = runtime.on_signal(signal)
    assert res.status == "ORDER_SUBMITTED"
    assert res.order_id is not None

    # Verificar que a ordem no sink tem side LONG
    sink = runtime.execution_sink
    assert res.order_id in sink.executor.pending_orders
    order = sink.executor.pending_orders[res.order_id]
    assert order.side == "LONG"


def test_14_fixed_short_creates_short_decision():
    """14. fixed SHORT cria SHORT."""
    current_time = 1700000000000
    config = make_valid_config(provider="fixed_short")
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: current_time)

    signal = {
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    }
    res = runtime.on_signal(signal)
    assert res.status == "ORDER_SUBMITTED"

    sink = runtime.execution_sink
    order = sink.executor.pending_orders[res.order_id]
    assert order.side == "SHORT"


def test_15_seeded_random_deterministic():
    """15. seeded random determinístico."""
    config1 = make_valid_config(provider="seeded_random", seed=98765)
    config2 = make_valid_config(provider="seeded_random", seed=98765)

    current_ts = 1700000000000
    runtime1 = ShadowPaperRuntime(config=config1, clock_ms=lambda: current_ts)
    runtime2 = ShadowPaperRuntime(config=config2, clock_ms=lambda: current_ts)

    sides1 = []
    sides2 = []

    for i in range(5):
        current_ts = 1700000000000 + (i * 60000)
        sig = {
            "epoch_ms": current_ts,
            "price": 50000.0 + i,
            "symbol": "BTCUSDT",
        }
        res1 = runtime1.on_signal(sig)
        res2 = runtime2.on_signal(sig)

        assert res1.status == "ORDER_SUBMITTED"
        assert res2.status == "ORDER_SUBMITTED"

        order1 = runtime1.execution_sink.executor.pending_orders[res1.order_id]
        order2 = runtime2.execution_sink.executor.pending_orders[res2.order_id]

        sides1.append(order1.side)
        sides2.append(order2.side)

        # Clear pending order to allow next signal submission on same symbol
        runtime1.execution_sink.executor.pending_orders.clear()
        runtime1.execution_sink.executor._pending_symbol_map.clear()
        runtime2.execution_sink.executor.pending_orders.clear()
        runtime2.execution_sink.executor._pending_symbol_map.clear()

    assert sides1 == sides2
    assert len(sides1) == 5


def test_16_signal_to_risk_to_order_to_sink():
    """16. signal -> risk -> order -> sink."""
    current_time = 1700000000000
    config = make_valid_config()
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: current_time)

    signal = {
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    }
    res = runtime.on_signal(signal)
    assert res.status == "ORDER_SUBMITTED"
    assert res.order_id is not None

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 1
    assert counters["directional_decisions"] == 1
    assert counters["risk_approved"] == 1
    assert counters["orders_submitted"] == 1
    assert counters["runtime_errors"] == 0


def test_17_risk_rejection_does_not_submit_order():
    """17. risk rejection não submete ordem."""
    current_time = 1700000000000
    # Limitar o tamanho max_position_size no RiskManager para 500
    rm = RiskManager(RiskConfig(max_position_size=500.0))
    config = make_valid_config(notional=1000.0)  # Excede 500.0!

    runtime = ShadowPaperRuntime(config=config, risk_manager=rm, clock_ms=lambda: current_time)

    signal = {
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
    }
    res = runtime.on_signal(signal)
    assert res.status == "RISK_REJECTED"
    assert "position size limit exceeded" in (res.rejection_reason or "")

    # Garantir que nenhuma ordem chegou ao sink
    sink = runtime.execution_sink
    assert len(sink.executor.pending_orders) == 0

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 1
    assert counters["directional_decisions"] == 1
    assert counters["risk_rejected"] == 1
    assert counters["risk_approved"] == 0
    assert counters["orders_submitted"] == 0


def test_18_neutral_follow_signal_coverage_preserved():
    """18. NEUTRAL FOLLOW_SIGNAL coverage preservada conforme mode configurado."""
    config = make_valid_config()
    # Criar um SignalDecisionAdapter explícito em modo FOLLOW_SIGNAL
    custom_adapter = SignalDecisionAdapter(
        cohort_id=config.cohort_id,
        mode="FOLLOW_SIGNAL",
        timeframe=config.timeframe,
    )
    runtime = ShadowPaperRuntime(config=config, signal_adapter=custom_adapter)

    # Sinal neutro / sem direção
    neutral_signal = {
        "epoch_ms": 1700000000000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "direction": "NEUTRAL",
        "side": "NEUTRAL",
    }
    res = runtime.on_signal(neutral_signal)
    assert res.status == "SKIPPED"
    assert "non-directional" in (res.skip_reason or "").lower()

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 1
    assert counters["nondirectional_skips"] == 1
    assert counters["directional_decisions"] == 0
    assert counters["orders_submitted"] == 0


def test_19_on_market_trade_delegates_to_sink():
    """19. on_market_trade delega sink."""
    current_time = 1700000000000
    config = make_valid_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: current_time)

    # 1. Submete ordem a 50000
    sig_res = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert sig_res.status == "ORDER_SUBMITTED"

    # 2. Envia trade de mercado que preenche a ordem (LONG compra a 49990)
    trade_norm = {
        "p": 49990.0,
        "q": 1.0,
        "T": 1700000001000,
        "trade_id": 101,
        "symbol": "BTCUSDT",
        "m": False,
        "source": "fut_agg",
    }
    trade_res = runtime.on_market_trade(trade_norm)
    assert trade_res.status == "PROCESSED"
    assert trade_res.accepted is True
    assert trade_res.fills_count == 1

    counters = runtime.get_counters()
    assert counters["ticks_seen"] == 1
    assert counters["fills"] == 1


def test_20_inactive_ignores_signals():
    """20. inactive ignora sinais."""
    config = ShadowPaperConfig(enabled=False)
    runtime = ShadowPaperRuntime(config=config)

    res = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert res.status == "INACTIVE"
    assert runtime.get_counters()["signals_seen"] == 0


def test_21_inactive_ignores_ticks():
    """21. inactive ignora ticks."""
    config = ShadowPaperConfig(enabled=False)
    runtime = ShadowPaperRuntime(config=config)

    res = runtime.on_market_trade({"p": 50000.0, "q": 1.0, "T": 1700000001000, "trade_id": 101, "m": False, "source": "fut_agg"})
    assert res.status == "INACTIVE"
    assert runtime.get_counters()["ticks_seen"] == 0


def test_22_shutdown_deactivates_immediately():
    """22. shutdown desativa imediatamente."""
    config = make_valid_config()
    runtime = ShadowPaperRuntime(config=config)
    assert runtime.is_active is True

    runtime.shutdown()
    assert runtime.is_active is False

    res_sig = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0})
    assert res_sig.status == "INACTIVE"

    res_trade = runtime.on_market_trade({"p": 50000.0, "q": 1.0, "T": 1700000001000, "trade_id": 101, "m": False, "source": "fut_agg"})
    assert res_trade.status == "INACTIVE"


def test_23_signal_exception_does_not_escape():
    """23. exceção signal não escapa."""
    config = make_valid_config()
    runtime = ShadowPaperRuntime(config=config)

    # Monkeypatch no signal_adapter para lançar exceção
    def explosive_signal(event):
        raise RuntimeError("Catastrophic explosion inside signal adapter")

    runtime.signal_adapter.process_signal = explosive_signal

    res = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0})
    assert res.status == "ERROR"
    assert "Catastrophic explosion" in (res.error or "")

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 1
    assert counters["runtime_errors"] == 1


def test_24_tick_exception_does_not_escape():
    """24. exceção tick não escapa."""
    config = make_valid_config()
    runtime = ShadowPaperRuntime(config=config)

    # Monkeypatch no execution_sink para lançar exceção
    def explosive_trade(norm):
        raise ValueError("Catastrophic tick processing failure")

    runtime.execution_sink.on_market_trade = explosive_trade

    res = runtime.on_market_trade({"p": 50000.0, "q": 1.0, "T": 1700000001000, "trade_id": 101, "m": False, "source": "fut_agg"})
    assert res.status == "ERROR"
    assert "Catastrophic tick processing failure" in (res.error or "")

    counters = runtime.get_counters()
    assert counters["ticks_seen"] == 1
    assert counters["runtime_errors"] == 1


def test_25_counters_coherent():
    """25. counters coerentes."""
    config = make_valid_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: 1700000000000)

    # 1 sinal inválido
    runtime.on_signal("not_a_dict")
    # 1 sinal válido submetido
    runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    # 1 tick recebido que preenche a ordem
    runtime.on_market_trade({"p": 49000.0, "q": 1.0, "T": 1700000001000, "trade_id": 101, "symbol": "BTCUSDT", "m": False, "source": "fut_agg"})

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 2
    assert counters["invalid_signals"] == 1
    assert counters["directional_decisions"] == 1
    assert counters["risk_approved"] == 1
    assert counters["orders_submitted"] == 1
    assert counters["ticks_seen"] == 1
    assert counters["fills"] == 1
    assert counters["runtime_errors"] == 0

    status = runtime.get_status()
    assert status["active"] is True
    assert status["risk_position_limit_active"] is False
    assert status["risk_daily_loss_active"] is False


def test_26_cohort_preserved():
    """26. cohort preservada."""
    cohort_id = "CH_IMMUTABLE_COHORT_2026"
    config = make_valid_config(cohort_id=cohort_id)
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: 1700000000000)

    res = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert res.status == "ORDER_SUBMITTED"

    order = runtime.execution_sink.executor.pending_orders[res.order_id]
    assert order.cohort_id == cohort_id
    assert runtime.get_status()["cohort_id"] == cohort_id


def test_27_zero_network_access(monkeypatch):
    """27. nenhuma rede."""
    # Monkeypatch no socket.socket para garantir que nenhuma conexão ocorra
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("CRITICAL: Network socket call detected in hermetic shadow runtime!")

    monkeypatch.setattr(socket, "socket", forbidden_socket)

    config = make_valid_config()
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: 1700000000000)

    res_sig = runtime.on_signal({"epoch_ms": 1700000000000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert res_sig.status == "ORDER_SUBMITTED"

    res_trade = runtime.on_market_trade({"p": 49000.0, "q": 1.0, "T": 1700000001000, "trade_id": 101, "symbol": "BTCUSDT", "m": False, "source": "fut_agg"})
    assert res_trade.status == "PROCESSED"


def test_28_zero_ai_coupling():
    """28. nenhuma IA."""
    import paper_trading.shadow_runtime as srt_mod
    src = open(srt_mod.__file__, "r", encoding="utf-8").read()

    for forbidden in ("groq", "openai", "analyzer_qwen", "qwen", "xgboost", "llm"):
        assert forbidden not in src.lower(), f"Forbidden AI reference found in shadow_runtime: {forbidden}"


def test_29_zero_real_orders():
    """29. nenhuma ordem real."""
    import paper_trading.shadow_runtime as srt_mod
    src = open(srt_mod.__file__, "r", encoding="utf-8").read()

    for forbidden in ("binance_api", "create_order", "new_order", "order_test", "signed_request"):
        assert forbidden not in src.lower(), f"Forbidden trading API found in shadow_runtime: {forbidden}"


def test_30_zero_io_per_tick(monkeypatch):
    """30. nenhum I/O por tick."""
    original_open = builtins.open

    def forbidden_open(*args, **kwargs):
        raise AssertionError(f"CRITICAL: File I/O detected during tick processing! {args}")

    config = make_valid_config()
    runtime = ShadowPaperRuntime(config=config, clock_ms=lambda: 1700000000000)

    # Bloquear file I/O estritamente durante o hot path de ticks
    monkeypatch.setattr(builtins, "open", forbidden_open)

    res = runtime.on_market_trade({"p": 50000.0, "q": 1.0, "T": 1700000001000, "trade_id": 101, "symbol": "BTCUSDT", "m": False, "source": "fut_agg"})
    assert res.status == "PROCESSED"
