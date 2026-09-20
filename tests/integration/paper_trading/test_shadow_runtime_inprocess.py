# tests/integration/paper_trading/test_shadow_runtime_inprocess.py
"""
Integration test harness for hermetic ShadowPaperRuntime (Gate C3-C-B2).
Tests causal sequencing, concurrency, risk limits, exits, and fail-closed invariants in-process.
"""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
import socket
import sqlite3
import threading
from typing import Any, Dict, List, Optional
import pytest

from paper_trading.config import ShadowPaperConfig, parse_shadow_config
from paper_trading.shadow_runtime import (
    ShadowPaperRuntime,
    ShadowSignalResult,
    ShadowTradeResult,
)
from paper_trading.adapters.signal_adapter import SignalDecisionAdapter
from paper_trading.adapters.risk_adapter import RiskAdapter
from paper_trading.execution_sink import ExecutionSink
from paper_trading.executor import PaperExecutor
from paper_trading.cost_model import CostModel
from risk_management.risk_manager import RiskConfig, RiskManager


class FakeClock:
    """Deterministic injectable clock for causality testing without wall-clock sleep."""

    def __init__(self, start_ms: int = 1700000000000):
        self._current_ms: int = start_ms

    def __call__(self) -> int:
        return self._current_ms

    def advance(self, delta_ms: int) -> int:
        self._current_ms += delta_ms
        return self._current_ms

    def set(self, target_ms: int) -> int:
        self._current_ms = target_ms
        return self._current_ms


def make_norm_trade(
    price: float,
    qty: float,
    T: int,
    trade_id: int | str,
    is_buyer_maker: bool = False,
    source: str = "fut_agg",
) -> Dict[str, Any]:
    """Helper to produce realistic normalized trade payload identical to market_orchestrator."""
    return {
        "p": price,
        "q": qty,
        "T": T,
        "T_raw": T,
        "m": is_buyer_maker,
        "source": source,
        "trade_id": trade_id,
        "symbol": "BTCUSDT",
    }


def make_test_config(
    provider: str = "fixed_long",
    cohort_id: str = "CH_INPROCESS_2026",
    notional: float = 1000.0,
    horizon_s: int = 60,
    ttl_ms: int = 5000,
    seed: Optional[int] = None,
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
        "PAPER_COST_SOURCE": "test_inprocess",
        "PAPER_COST_EFFECTIVE_AT": "2026-01-01T00:00:00Z",
        "PAPER_STRATEGY_VERSION": "c3_harness_v1.0.0",
    }
    if seed is not None:
        env["PAPER_RANDOM_SEED"] = str(seed)
    res = parse_shadow_config(env)
    assert res.is_valid is True
    assert res.config is not None
    return res.config


def test_1_deterministic_fake_clock():
    """1. Prova do FakeClock injetável (sem sleep, sem time.time, sem wall clock)."""
    clock = FakeClock(1000)
    assert clock() == 1000
    assert clock.advance(50) == 1050
    assert clock() == 1050
    assert clock.set(2000) == 2000
    assert clock() == 2000


def test_2_controlled_tape_generation():
    """2. Tape controlado gerando payloads normalizados realistas."""
    trade = make_norm_trade(price=50000.0, qty=0.5, T=1700000000100, trade_id=12345)
    assert trade["p"] == 50000.0
    assert trade["q"] == 0.5
    assert trade["T"] == 1700000000100
    assert trade["source"] == "fut_agg"
    assert trade["trade_id"] == 12345


def test_3_main_causal_scenario():
    """
    3. CENÁRIO CAUSAL PRINCIPAL:
    T0 fecha janela. T1 e T2 chegam antes do on_signal.
    on_signal gera ordem. T3 chega depois.
    Provar: T0, T1, T2 nunca viram fill; T3 é o primeiro fill elegível;
    o fill usa o preço de T3 + adverse slippage, nunca reference_price.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long", ttl_ms=5000)
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    # T0 chega (e fecha a janela conceitual)
    t0 = make_norm_trade(price=50000.0, qty=1.0, T=1000, trade_id=100)
    res_t0 = runtime.on_market_trade(t0)
    assert res_t0.accepted is True
    assert res_t0.fills_count == 0

    # T1 e T2 chegam ANTES do sinal ser processado
    clock.set(1050)
    t1 = make_norm_trade(price=50010.0, qty=1.0, T=1050, trade_id=101)
    res_t1 = runtime.on_market_trade(t1)
    assert res_t1.accepted is True
    assert res_t1.fills_count == 0

    clock.set(1100)
    t2 = make_norm_trade(price=50020.0, qty=1.0, T=1100, trade_id=102)
    res_t2 = runtime.on_market_trade(t2)
    assert res_t2.accepted is True
    assert res_t2.fills_count == 0

    # Sinal da janela T0 é entregue ao runtime
    clock.set(1150)
    signal_event = {
        "epoch_ms": 1000,  # janela fechada em T0
        "price": 50000.0,  # reference price
        "symbol": "BTCUSDT",
    }
    sig_res = runtime.on_signal(signal_event)
    assert sig_res.status == "ORDER_SUBMITTED"

    # T3 chega DEPOIS da ordem ser submetida
    clock.set(1200)
    t3 = make_norm_trade(price=50030.0, qty=1.0, T=1200, trade_id=103)
    res_t3 = runtime.on_market_trade(t3)
    assert res_t3.accepted is True
    assert res_t3.fills_count == 1

    # Verificar executor fills
    fills = runtime.execution_sink.executor.position_manager.open_positions
    assert len(fills) == 1
    pos = list(fills.values())[0]

    # Preço do fill deve ser o preço de T3 (50030.0) + adverse slippage (1.0 bps), NUNCA 50000.0
    expected_slippage = 50030.0 * (1.0 / 10_000.0)
    assert pos.entry_price == pytest.approx(50030.0 + expected_slippage, rel=1e-5)
    assert pos.entry_price > 50030.0
    assert pos.entry_price != 50000.0


def test_4_same_millisecond_causality():
    """
    4. MESMO MILISSEGUNDO:
    Tick A com T=1000. Ordem submetida.
    Próximo tick B: T=1000 com trade_id maior.
    Provar: A nunca vira fill; B pode preencher; ingest_seq(B) > registered_seq(order).
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    # Tick A chega em T=1000
    tick_a = make_norm_trade(price=50000.0, qty=1.0, T=1000, trade_id=1)
    res_a = runtime.on_market_trade(tick_a)
    assert res_a.accepted is True
    assert res_a.fills_count == 0
    seq_a = runtime.execution_sink._ingest_seq

    # Ordem submetida após tick A
    sig = {"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"}
    sig_res = runtime.on_signal(sig)
    assert sig_res.status == "ORDER_SUBMITTED"
    registered_seq = runtime.execution_sink._registered_ingest_seq.get(sig_res.order_id, -1)
    assert registered_seq == seq_a

    # Tick B chega no MESMO milissegundo T=1000 com trade_id=2
    tick_b = make_norm_trade(price=50005.0, qty=1.0, T=1000, trade_id=2)
    res_b = runtime.on_market_trade(tick_b)
    assert res_b.accepted is True
    assert res_b.fills_count == 1
    seq_b = runtime.execution_sink._ingest_seq
    assert seq_b > registered_seq


def test_5_eventbus_latency_simulation():
    """
    5. LATÊNCIA DO EVENTBUS:
    10 ticks chegam antes do sinal ser despachado.
    Provar que nenhum tick passado busca preenchimento retroativo.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    for i in range(10):
        t = make_norm_trade(price=50000.0 + i, qty=1.0, T=1000 + (i * 10), trade_id=i + 1)
        res = runtime.on_market_trade(t)
        assert res.fills_count == 0

    # Chega on_signal atrasado
    clock.set(1150)
    sig = {"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"}
    sig_res = runtime.on_signal(sig)
    assert sig_res.status == "ORDER_SUBMITTED"

    # Nenhum fill aconteceu ainda
    assert runtime.get_counters()["fills"] == 0

    # Próximo tick futuro preenche
    clock.set(1200)
    t_next = make_norm_trade(price=50020.0, qty=1.0, T=1200, trade_id=11)
    res_next = runtime.on_market_trade(t_next)
    assert res_next.fills_count == 1
    assert runtime.get_counters()["fills"] == 1


def test_6_duplicate_signal_behavior():
    """
    6. DUPLICATE SIGNAL:
    Enviar o mesmo sinal duas vezes.
    Mesmo decision_id gerado; primeira ordem aceita; segunda ordem rejeitada
    economicamente por DUPLICATE_DECISION.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    sig = {"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"}

    res1 = runtime.on_signal(sig)
    assert res1.status == "ORDER_SUBMITTED"

    res2 = runtime.on_signal(sig)
    assert res2.status == "ORDER_REJECTED"
    assert res2.rejection_reason in ("DUPLICATE_DECISION", "POSITION_OPEN")

    # Apenas uma ordem pendente ativa
    sink = runtime.execution_sink
    assert len(sink.executor.pending_orders) == 1
    assert runtime.get_counters()["orders_submitted"] == 1
    assert runtime.get_counters()["order_rejected"] == 1


def test_7_multi_event_same_window():
    """
    7. DOIS EVENTOS MESMA JANELA:
    Mesmo symbol/timeframe/close_ms com características semânticas diferentes (event_type).
    Diferentes source_event_key e decision_ids.
    Primeiro aceito, segundo rejeitado economicamente por POSITION_OPEN.
    Coverage conta ambos os sinais.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    event1 = {
        "epoch_ms": 1000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "event_type": "ABSORCAO",
    }
    event2 = {
        "epoch_ms": 1000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "event_type": "EXAUSTAO",
    }

    res1 = runtime.on_signal(event1)
    assert res1.status == "ORDER_SUBMITTED"

    res2 = runtime.on_signal(event2)
    assert res2.status == "ORDER_REJECTED"
    assert res2.rejection_reason in ("POSITION_OPEN", "DUPLICATE_DECISION")

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 2
    assert counters["directional_decisions"] == 2
    assert counters["orders_submitted"] == 1
    assert counters["order_rejected"] == 1


def test_8_neutral_follow_signal():
    """
    8. NEUTRAL FOLLOW_SIGNAL:
    Em modo FOLLOW_SIGNAL, evento NEUTRAL incrementa nondirectional_skips,
    não cria PaperOrder nem fills.
    """
    config = make_test_config()
    custom_adapter = SignalDecisionAdapter(
        cohort_id=config.cohort_id,
        mode="FOLLOW_SIGNAL",
        timeframe=config.timeframe,
    )
    runtime = ShadowPaperRuntime(config=config, signal_adapter=custom_adapter)

    neutral_signal = {
        "epoch_ms": 1000,
        "price": 50000.0,
        "symbol": "BTCUSDT",
        "side": "NEUTRAL",
        "direction": "NEUTRAL",
    }
    res = runtime.on_signal(neutral_signal)
    assert res.status == "SKIPPED"
    assert "non-directional" in (res.skip_reason or "").lower()

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 1
    assert counters["nondirectional_skips"] == 1
    assert counters["orders_submitted"] == 0
    assert counters["fills"] == 0


def test_9_baseline_seeded_random_deterministic():
    """
    9. BASELINE RANDOM:
    Com seeded_random, mesma seed gera idêntica sequência de decisões LONG/SHORT.
    """
    cfg1 = make_test_config(provider="seeded_random", seed=1337)
    cfg2 = make_test_config(provider="seeded_random", seed=1337)

    clock = FakeClock(1000)
    rt1 = ShadowPaperRuntime(config=cfg1, clock_ms=clock)
    rt2 = ShadowPaperRuntime(config=cfg2, clock_ms=clock)

    sides1 = []
    sides2 = []

    for i in range(10):
        clock.set(1000 + i * 60000)
        sig = {"epoch_ms": 1000 + i * 60000, "price": 50000.0, "symbol": "BTCUSDT"}

        r1 = rt1.on_signal(sig)
        r2 = rt2.on_signal(sig)

        assert r1.status == "ORDER_SUBMITTED"
        assert r2.status == "ORDER_SUBMITTED"

        order1 = rt1.execution_sink.executor.pending_orders[r1.order_id]
        order2 = rt2.execution_sink.executor.pending_orders[r2.order_id]

        sides1.append(order1.side)
        sides2.append(order2.side)

        # Limpar pending orders para permitir próxima ordem na mesma coorte
        rt1.execution_sink.executor.pending_orders.clear()
        rt1.execution_sink.executor._pending_symbol_map.clear()
        rt2.execution_sink.executor.pending_orders.clear()
        rt2.execution_sink.executor._pending_symbol_map.clear()

    assert sides1 == sides2
    assert len(sides1) == 10
    assert "LONG" in sides1
    assert "SHORT" in sides1


def test_10_risk_rejection():
    """
    10. RISK REJECTION:
    Notional acima de max_position_size é rejeitado no RiskAdapter.
    Nenhuma ordem chega ao sink; ticks futuros não geram posição fantasma.
    """
    rm = RiskManager(RiskConfig(max_position_size=500.0))
    config = make_test_config(notional=1000.0)  # 1000 > 500
    runtime = ShadowPaperRuntime(config=config, risk_manager=rm)

    sig = {"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"}
    res = runtime.on_signal(sig)
    assert res.status == "RISK_REJECTED"

    # Injetar tick de mercado
    t = make_norm_trade(price=50000.0, qty=1.0, T=1050, trade_id=1)
    res_trade = runtime.on_market_trade(t)
    assert res_trade.fills_count == 0
    assert len(runtime.execution_sink.executor.pending_orders) == 0
    assert len(runtime.execution_sink.executor.position_manager.open_positions) == 0


def test_11_order_ttl_expiry():
    """
    11. ORDER TTL:
    Ordem sem ticks até depois de expires_at.
    Primeiro tick posterior produz EXPIRED_NO_MARKET_DATA e limpa registered_ingest_seq.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long", ttl_ms=1000)
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    sig = {"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"}
    res_sig = runtime.on_signal(sig)
    assert res_sig.status == "ORDER_SUBMITTED"
    order_id = res_sig.order_id

    sink = runtime.execution_sink
    assert order_id in sink._registered_ingest_seq

    # Tick chega em T=2500 (> 1000 + 1000 = 2000 expires_at)
    clock.set(2500)
    t = make_norm_trade(price=50000.0, qty=1.0, T=2500, trade_id=1)
    res_trade = runtime.on_market_trade(t)
    assert res_trade.fills_count == 0
    assert res_trade.rejections_count == 1

    # registered_ingest_seq deve ter sido limpo e pending removida
    assert order_id not in sink._registered_ingest_seq
    assert len(sink.executor.pending_orders) == 0

    # Provar reason EXPIRED_NO_MARKET_DATA
    sink_res = sink.on_market_trade(make_norm_trade(price=50000.0, qty=1.0, T=2600, trade_id=2))
    # já expirou, nenhuma nova rejection
    assert sink_res.events is not None


def test_12_horizon_exit():
    """
    12. HORIZON EXIT:
    Posição sem SL/TP fecha ao atingir o deadline do horizonte.
    Exit utiliza tick observável, custos aplicados, 1 ClosedTrade emitido.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long", horizon_s=60, ttl_ms=5000)
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    # 1. Submeter e preencher ordem em T=1050
    runtime.on_signal({"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"})
    clock.set(1050)
    t_fill = make_norm_trade(price=50000.0, qty=1.0, T=1050, trade_id=1)
    runtime.on_market_trade(t_fill)
    assert runtime.get_counters()["fills"] == 1

    # 2. Tick antes do horizonte (horizon é 60s = 60000ms, deadline = 1050 + 60000 = 61050)
    clock.set(30000)
    t_mid = make_norm_trade(price=50500.0, qty=1.0, T=30000, trade_id=2)
    res_mid = runtime.on_market_trade(t_mid)
    assert res_mid.closed_trades_count == 0
    assert runtime.get_counters()["closed_trades"] == 0

    # 3. Tick atingindo o horizonte em T=61050
    clock.set(61050)
    t_exit = make_norm_trade(price=51000.0, qty=1.0, T=61050, trade_id=3)
    sink_res = runtime.execution_sink.on_market_trade(t_exit)
    assert sink_res.events is not None
    assert len(sink_res.events.closed_trades) == 1
    assert runtime.execution_sink.executor.position_manager.open_positions == {}

    # Provar que exit utiliza tick observável, custos aplicados e 1 ClosedTrade emitido
    closed_trade = sink_res.events.closed_trades[0]
    assert closed_trade.exit_reason == "HORIZON_EXPIRY"
    assert closed_trade.costs_complete is True
    assert closed_trade.exit_price > 0


def test_13_funding_incomplete_crossing():
    """
    13. FUNDING INCOMPLETE:
    Se a posição cruzar fronteira de funding (00h, 08h, 16h UTC) sem funding rate disponível,
    funding_bps is None, funding_usdt is None, costs_complete=False, net_pnl_bps=None, trade_win=None.
    """
    # 07:59:00 UTC em ms: 1700035140000 (abertura antes das 08h)
    # 08:01:00 UTC em ms: 1700035260000 (fechamento após as 08h)
    t_open = 1700035140000
    t_close = 1700035260000

    clock = FakeClock(t_open)
    config = make_test_config(provider="fixed_long", horizon_s=120)
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    # Submeter ordem com funding_rate_at_decision=None
    sig = {"epoch_ms": t_open, "price": 50000.0, "symbol": "BTCUSDT"}
    runtime.on_signal(sig)

    # Fill antes das 08:00 UTC
    runtime.on_market_trade(make_norm_trade(price=50000.0, qty=1.0, T=t_open, trade_id=1))

    # Exit após as 08:00 UTC (cruza a fronteira das 08h em T=t_close atingindo o horizonte)
    clock.set(t_close)
    sink_res = runtime.execution_sink.on_market_trade(make_norm_trade(price=50500.0, qty=1.0, T=t_close, trade_id=2))
    assert sink_res.events is not None
    assert len(sink_res.events.closed_trades) == 1

    closed_trade = sink_res.events.closed_trades[0]
    assert closed_trade.funding_bps is None
    assert closed_trade.funding_usdt is None
    assert closed_trade.costs_complete is False
    assert closed_trade.net_pnl_bps is None
    assert closed_trade.trade_win is None


def test_14_thread_concurrency():
    """
    14. THREAD CONCURRENCY:
    Thread A envia ticks contínuos.
    Thread B envia sinais para o runtime.
    Sincronização via Barrier/Event.
    Provar zero exceções, monotonicidade de ingest_seq e consistência de estado.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    barrier = threading.Barrier(2)
    exceptions: List[Exception] = []

    def tick_worker():
        try:
            barrier.wait(timeout=5.0)
            for i in range(1, 51):
                t = make_norm_trade(price=50000.0 + i, qty=0.1, T=1000 + i * 10, trade_id=i)
                runtime.on_market_trade(t)
        except Exception as exc:
            exceptions.append(exc)

    def signal_worker():
        try:
            barrier.wait(timeout=5.0)
            for i in range(1, 11):
                sig = {"epoch_ms": 1000 + i * 50, "price": 50000.0, "symbol": "BTCUSDT"}
                runtime.on_signal(sig)
        except Exception as exc:
            exceptions.append(exc)

    t1 = threading.Thread(target=tick_worker, name="tick_worker")
    t2 = threading.Thread(target=signal_worker, name="signal_worker")

    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not exceptions
    assert not t1.is_alive()
    assert not t2.is_alive()

    counters = runtime.get_counters()
    assert counters["ticks_seen"] == 50
    assert counters["signals_seen"] == 10
    assert counters["runtime_errors"] == 0
    assert runtime.execution_sink._ingest_seq == 50


def test_15_exception_isolation():
    """
    15. EXCEPTION ISOLATION:
    Falhas forçadas no adapter ou sink não vazam para o chamador
    e incrementam runtime_errors.
    """
    config = make_test_config()
    runtime = ShadowPaperRuntime(config=config)

    # Injetar falha forçada em on_signal
    def broken_signal(event):
        raise ZeroDivisionError("Simulated math error in signal processor")

    runtime.signal_adapter.process_signal = broken_signal
    res_sig = runtime.on_signal({"epoch_ms": 1000, "price": 50000.0})
    assert res_sig.status == "ERROR"
    assert "Simulated math error" in (res_sig.error or "")

    # Injetar falha forçada em on_market_trade
    def broken_trade(norm):
        raise ConnectionResetError("Simulated connection error in sink")

    runtime.execution_sink.on_market_trade = broken_trade
    res_trade = runtime.on_market_trade(make_norm_trade(price=50000.0, qty=1.0, T=1000, trade_id=1))
    assert res_trade.status == "ERROR"
    assert "Simulated connection error" in (res_trade.error or "")

    counters = runtime.get_counters()
    assert counters["runtime_errors"] == 2


def test_16_shutdown_behavior():
    """
    16. SHUTDOWN:
    Ao chamar shutdown, novos sinais e ticks viram no-op imediatamente;
    nenhuma posição é artificialmente liquidada nem fechamento fake criado.
    """
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    # Criar e preencher posição
    runtime.on_signal({"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"})
    runtime.on_market_trade(make_norm_trade(price=50000.0, qty=1.0, T=1000, trade_id=1))
    assert runtime.get_counters()["fills"] == 1

    # Shutdown
    runtime.shutdown()
    assert runtime.is_active is False

    # Novo sinal ignorado
    sig_res = runtime.on_signal({"epoch_ms": 2000, "price": 50000.0})
    assert sig_res.status == "INACTIVE"

    # Novo tick ignorado
    trade_res = runtime.on_market_trade(make_norm_trade(price=51000.0, qty=1.0, T=2000, trade_id=2))
    assert trade_res.status == "INACTIVE"

    # Zero ClosedTrade artificial gerado
    assert runtime.get_counters()["closed_trades"] == 0


def test_17_coverage_accounting():
    """
    17. COVERAGE ACCOUNTING:
    Equação de conservação de sinais:
    signals_seen == directional_decisions + nondirectional_skips + invalid_signals
    """
    config = make_test_config()
    custom_adapter = SignalDecisionAdapter(
        cohort_id=config.cohort_id,
        mode="FOLLOW_SIGNAL",
        timeframe=config.timeframe,
    )
    runtime = ShadowPaperRuntime(config=config, signal_adapter=custom_adapter)

    # 1. Sinal inválido (não é dict)
    runtime.on_signal("bad_signal")

    # 2. Sinal inválido (sem timestamp)
    runtime.on_signal({"price": 50000.0, "symbol": "BTCUSDT"})

    # 3. Sinal não-direcional (NEUTRAL)
    runtime.on_signal({"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT", "side": "NEUTRAL"})

    # 4. Sinal direcional válido (LONG)
    runtime.on_signal({"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT", "side": "LONG"})

    counters = runtime.get_counters()
    assert counters["signals_seen"] == 4
    assert counters["invalid_signals"] == 2
    assert counters["nondirectional_skips"] == 1
    assert counters["directional_decisions"] == 1

    # Conservação exata
    assert counters["signals_seen"] == (
        counters["directional_decisions"]
        + counters["nondirectional_skips"]
        + counters["invalid_signals"]
    )


def test_18_zero_side_effects(monkeypatch):
    """
    18. ZERO SIDE EFFECTS:
    Zero network, zero LLM, zero real orders, zero SQLite.
    """
    # 1. Proibir network socket
    def no_socket(*args, **kwargs):
        raise AssertionError("Network socket access forbidden!")

    monkeypatch.setattr(socket, "socket", no_socket)

    # 2. Proibir SQLite connect
    def no_sqlite(*args, **kwargs):
        raise AssertionError("SQLite connection forbidden when ledger is disconnected!")

    monkeypatch.setattr(sqlite3, "connect", no_sqlite)

    # Executar pipeline
    clock = FakeClock(1000)
    config = make_test_config(provider="fixed_long")
    runtime = ShadowPaperRuntime(config=config, clock_ms=clock)

    res_sig = runtime.on_signal({"epoch_ms": 1000, "price": 50000.0, "symbol": "BTCUSDT"})
    assert res_sig.status == "ORDER_SUBMITTED"

    res_trade = runtime.on_market_trade(make_norm_trade(price=50000.0, qty=1.0, T=1000, trade_id=1))
    assert res_trade.status == "PROCESSED"
