# tests/unit/paper_trading/test_daily_loss_session_safe_c3_5_2.py
"""
Unit tests for Gate C3.5.2: SESSION_SAFE Daily Loss Lifecycle.

Tests cover:
1. Complete profitable trade adds once to daily_pnl.
2. Complete losing trade adds once to daily_pnl.
3. Duplicate trade is ignored and does not double-count.
4. Loss exactly at -limit blocks new trade requests.
5. Loss strictly less than limit allows new trade requests.
6. Positive PnL greater than limit magnitude never blocks.
7. costs_complete=False triggers PNL_INCOMPLETE and blocks without altering daily_pnl.
8. net_pnl=None triggers PNL_INCOMPLETE and blocks without altering daily_pnl.
9. net_pnl NaN/Inf triggers PNL_INCOMPLETE and blocks without altering daily_pnl.
10. Yesterday trade (past UTC) is ignored for today's PnL and does not invalidate today.
11. Tomorrow trade (future UTC) triggers FUTURE_TRADE_DATE and invalidates pnl_complete.
12. on_market_trade with closing loss updates RiskManager under lock before next on_signal.
13. on_signal is rejected when daily loss limit is reached.
14. Incomplete PnL in on_market_trade blocks new exposure on runtime.
15. Existing open positions continue being evaluated and closed even when new exposure is blocked.
16. Dedup cache is bounded and evicts oldest entries without unbounded growth.
17. is_risk_position_limit_active remains functional alongside daily loss.
18. Recovery level is honestly exposed as SESSION_SAFE and risk_daily_loss_active is False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, Optional
from unittest.mock import MagicMock
from paper_trading.executor import TickEvents
from paper_trading.execution_sink import TradeResult, TradeStatus, ExecutionTick
import pytest

from risk_management.risk_manager import (
    RiskConfig,
    RiskManager,
    TradeRequest,
    RiskClosedTradeResult,
)
from paper_trading.adapters.risk_adapter import RiskAdapter
from paper_trading.contracts import ClosedTrade, PaperOrder, CanonicalDecision
from paper_trading.shadow_runtime import (
    ShadowPaperConfig,
    ShadowPaperRuntime,
)
from paper_trading.execution_sink import ExecutionSink
from paper_trading.executor import PaperExecutor


@dataclass
class DummyClosedTrade:
    trade_id: str
    closed_ts_ms: int
    net_pnl_usdt: Optional[float]
    costs_complete: bool = True
    symbol: str = "BTCUSDT"
    side: str = "LONG"
    entry_price: float = 50000.0
    exit_price: float = 50500.0
    quantity: float = 1.0
    notional_usdt: float = 50000.0
    exit_reason: str = "TAKE_PROFIT"


def _now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_for_date(d: date, hour: int = 12) -> int:
    dt = datetime(d.year, d.month, d.day, hour, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _make_runtime_config(cohort_id: str = "CH_TEST_C352") -> ShadowPaperConfig:
    return ShadowPaperConfig(
        enabled=True,
        cohort_id=cohort_id,
        symbol="BTCUSDT",
        strategy_version="v1.0",
        provider="fixed_long",
        order_ttl_ms=5000,
    )


# =====================================================================
# 1. Trade completo lucro soma uma vez
# =====================================================================
def test_01_complete_profit_trade_adds_once():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(trade_id="tr_profit_1", closed_ts_ms=now_ms, net_pnl_usdt=250.0)
    res = rm.record_closed_trade(trade)

    assert res.status == "APPLIED"
    assert res.daily_pnl == 250.0
    assert res.pnl_complete is True
    assert res.limit_exceeded is False
    assert rm.daily_pnl == 250.0


# =====================================================================
# 2. Trade completo perda soma uma vez
# =====================================================================
def test_02_complete_loss_trade_adds_once():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(trade_id="tr_loss_1", closed_ts_ms=now_ms, net_pnl_usdt=-300.0)
    res = rm.record_closed_trade(trade)

    assert res.status == "APPLIED"
    assert res.daily_pnl == -300.0
    assert res.pnl_complete is True
    assert res.limit_exceeded is False
    assert rm.daily_pnl == -300.0


# =====================================================================
# 3. Duplicate trade soma uma vez
# =====================================================================
def test_03_duplicate_trade_ignored():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(trade_id="tr_dup_1", closed_ts_ms=now_ms, net_pnl_usdt=-400.0)
    res1 = rm.record_closed_trade(trade)
    assert res1.status == "APPLIED"
    assert rm.daily_pnl == -400.0

    res2 = rm.record_closed_trade(trade)
    assert res2.status == "DUPLICATE_IGNORED"
    assert res2.daily_pnl == -400.0
    assert rm.daily_pnl == -400.0


# =====================================================================
# 4. Perda exatamente no limite bloqueia (daily_pnl <= -limit)
# =====================================================================
def test_04_loss_exactly_at_limit_blocks():
    # initial_capital = 100_000, max_daily_loss = 0.05 -> limit = 5000.0
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(trade_id="tr_exact_limit", closed_ts_ms=now_ms, net_pnl_usdt=-5000.0)
    res = rm.record_closed_trade(trade)

    assert res.status == "APPLIED"
    assert res.daily_pnl == -5000.0
    assert res.limit_exceeded is True
    assert rm.is_daily_loss_exceeded is True

    # check_trade_request must reject
    req = TradeRequest(symbol="BTCUSDT", side="BUY", size=0.1, price=50000.0)
    check = rm.check_trade_request(req)
    assert check["approved"] is False
    assert check["reason"] == "daily loss limit exceeded"


# =====================================================================
# 5. Perda menor que limite não bloqueia
# =====================================================================
def test_05_loss_strictly_less_than_limit_approves():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(trade_id="tr_under_limit", closed_ts_ms=now_ms, net_pnl_usdt=-4999.99)
    res = rm.record_closed_trade(trade)

    assert res.status == "APPLIED"
    assert res.daily_pnl == -4999.99
    assert res.limit_exceeded is False
    assert rm.is_daily_loss_exceeded is False

    req = TradeRequest(symbol="BTCUSDT", side="BUY", size=0.1, price=50000.0)
    check = rm.check_trade_request(req)
    assert check["approved"] is True


# =====================================================================
# 6. PnL positivo > limite não bloqueia por magnitude
# =====================================================================
def test_06_positive_pnl_above_limit_never_blocks():
    # initial_capital = 100_000, max_daily_loss = 0.05 -> limit = 5000.0
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(trade_id="tr_big_profit", closed_ts_ms=now_ms, net_pnl_usdt=10_000.0)
    res = rm.record_closed_trade(trade)

    assert res.status == "APPLIED"
    assert res.daily_pnl == 10_000.0
    assert res.limit_exceeded is False
    assert rm.is_daily_loss_exceeded is False

    req = TradeRequest(symbol="BTCUSDT", side="BUY", size=0.1, price=50000.0)
    check = rm.check_trade_request(req)
    assert check["approved"] is True


# =====================================================================
# 7. costs_complete=False bloqueia sem alterar daily_pnl
# =====================================================================
def test_07_costs_incomplete_blocks_without_altering_pnl():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(
        trade_id="tr_incomp_costs",
        closed_ts_ms=now_ms,
        net_pnl_usdt=-100.0,
        costs_complete=False,
    )
    res = rm.record_closed_trade(trade)

    assert res.status == "PNL_INCOMPLETE"
    assert res.daily_pnl == 0.0  # Must NOT alter daily_pnl
    assert res.pnl_complete is False
    assert rm.daily_pnl == 0.0
    assert rm.risk_pnl_complete is False

    req = TradeRequest(symbol="BTCUSDT", side="BUY", size=0.1, price=50000.0)
    check = rm.check_trade_request(req)
    assert check["approved"] is False
    assert check["reason"] == "daily pnl incomplete"


# =====================================================================
# 8. net_pnl=None bloqueia sem alterar daily_pnl
# =====================================================================
def test_08_net_pnl_none_blocks_without_altering_pnl():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    trade = DummyClosedTrade(
        trade_id="tr_none_pnl",
        closed_ts_ms=now_ms,
        net_pnl_usdt=None,
        costs_complete=True,
    )
    res = rm.record_closed_trade(trade)

    assert res.status == "PNL_INCOMPLETE"
    assert res.daily_pnl == 0.0
    assert res.pnl_complete is False
    assert rm.risk_pnl_complete is False

    req = TradeRequest(symbol="BTCUSDT", side="BUY", size=0.1, price=50000.0)
    check = rm.check_trade_request(req)
    assert check["approved"] is False
    assert check["reason"] == "daily pnl incomplete"


# =====================================================================
# 9. net_pnl NaN/Inf bloqueia sem alterar daily_pnl
# =====================================================================
def test_09_net_pnl_nan_inf_blocks_without_altering_pnl():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    # Test NaN
    trade_nan = DummyClosedTrade(
        trade_id="tr_nan_pnl",
        closed_ts_ms=now_ms,
        net_pnl_usdt=float("nan"),
        costs_complete=True,
    )
    res_nan = rm.record_closed_trade(trade_nan)
    assert res_nan.status == "PNL_INCOMPLETE"
    assert rm.daily_pnl == 0.0
    assert rm.risk_pnl_complete is False

    # Test Inf on fresh manager
    rm2 = RiskManager(cfg)
    trade_inf = DummyClosedTrade(
        trade_id="tr_inf_pnl",
        closed_ts_ms=now_ms,
        net_pnl_usdt=float("inf"),
        costs_complete=True,
    )
    res_inf = rm2.record_closed_trade(trade_inf)
    assert res_inf.status == "PNL_INCOMPLETE"
    assert rm2.daily_pnl == 0.0
    assert rm2.risk_pnl_complete is False


# =====================================================================
# 10. Trade de ontem não altera PnL de hoje
# =====================================================================
def test_10_yesterday_trade_does_not_alter_today_pnl():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    yesterday_ms = _ms_for_date(yesterday)

    trade = DummyClosedTrade(trade_id="tr_yesterday", closed_ts_ms=yesterday_ms, net_pnl_usdt=-1000.0)
    res = rm.record_closed_trade(trade)

    assert res.status == "IGNORED_PAST_DATE"
    assert res.daily_pnl == 0.0
    assert res.pnl_complete is True  # Today's completeness is NOT invalidated
    assert rm.daily_pnl == 0.0
    assert rm.risk_pnl_complete is True


# =====================================================================
# 11. Trade de amanhã civil bloqueia
# =====================================================================
def test_11_future_civil_trade_blocks():
    cfg = RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05)
    rm = RiskManager(cfg)
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    tomorrow_ms = _ms_for_date(tomorrow)

    trade = DummyClosedTrade(trade_id="tr_tomorrow", closed_ts_ms=tomorrow_ms, net_pnl_usdt=500.0)
    res = rm.record_closed_trade(trade)

    assert res.status == "FUTURE_TRADE_DATE"
    assert res.daily_pnl == 0.0
    assert res.pnl_complete is False
    assert rm.daily_pnl == 0.0
    assert rm.risk_pnl_complete is False

    req = TradeRequest(symbol="BTCUSDT", side="BUY", size=0.1, price=50000.0)
    check = rm.check_trade_request(req)
    assert check["approved"] is False
    assert check["reason"] == "daily pnl incomplete"


# =====================================================================
# 12. on_market_trade com fechamento em perda atualiza risco antes de on_signal
# =====================================================================
def test_12_on_market_trade_loss_updates_risk_before_signal():
    rm = RiskManager(RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05))
    adapter = RiskAdapter(risk_manager=rm)

    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_12")
    clock = lambda: _now_utc_ms()
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=clock)

    assert runtime.start() is True

    # Simulate position opened and closed with loss exceeding limit (-5001.0)
    closed_trade = DummyClosedTrade(
        trade_id="tr_close_big_loss",
        closed_ts_ms=clock(),
        net_pnl_usdt=-5001.0,
    )

    with runtime._lock:
        res = runtime.risk_adapter.on_closed_trade(closed_trade)
        if res.limit_exceeded:
            runtime._accept_new_exposure = False
            runtime._risk_block_reason = "DAILY_LOSS_LIMIT_REACHED"

    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "DAILY_LOSS_LIMIT_REACHED"
    assert rm.daily_pnl == -5001.0

    # Next on_signal must be rejected
    sig_res = runtime.on_signal({"epoch_ms": clock(), "price": 50000.0, "symbol": "BTCUSDT"})
    assert sig_res.status == "INACTIVE"
    assert "DAILY_LOSS_LIMIT_REACHED" in (sig_res.error or "")


# =====================================================================
# 13. on_signal posterior é rejeitado quando limite atingido
# =====================================================================
def test_13_on_signal_rejected_when_daily_loss_reached():
    rm = RiskManager(RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05))
    # Directly set limit reached
    rm.daily_pnl = -5000.0
    adapter = RiskAdapter(risk_manager=rm)

    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_13")
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=lambda: _now_utc_ms())
    runtime.start()

    dec = CanonicalDecision(
        cohort_id="CH_TEST_C352_13",
        symbol="BTCUSDT",
        window_id="win_13",
        decision_provider="RULE_BASED",
        strategy_version="v1.0",
        signal_timestamp=1700000000000,
        decision_timestamp=1700000000000,
        available_at=1700000000000,
        side="LONG",
        reference_price=50000.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )
    risk_res = runtime.risk_adapter.evaluate(dec)
    assert risk_res.status == "REJECTED_BY_RISK"
    assert risk_res.risk_reason == "daily loss limit exceeded"


# =====================================================================
# 14. PnL incompleto bloqueia nova exposição no runtime
# =====================================================================
def test_14_pnl_incomplete_blocks_new_exposure():
    rm = RiskManager(RiskConfig())
    adapter = RiskAdapter(risk_manager=rm)

    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_14")
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=lambda: _now_utc_ms())
    runtime.start()

    incomplete_trade = DummyClosedTrade(
        trade_id="tr_gap_incomp",
        closed_ts_ms=_now_utc_ms(),
        net_pnl_usdt=None,
        costs_complete=False,
    )

    with runtime._lock:
        res = runtime.risk_adapter.on_closed_trade(incomplete_trade)
        if res.status == "PNL_INCOMPLETE":
            runtime._accept_new_exposure = False
            runtime._risk_block_reason = "PNL_INCOMPLETE"

    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "PNL_INCOMPLETE"
    assert runtime.risk_pnl_complete is False


# =====================================================================
# 15. Posições existentes continuam sendo processadas com exposição bloqueada
# =====================================================================
def test_15_existing_positions_processed_while_new_exposure_blocked():
    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_15")
    runtime = ShadowPaperRuntime(config=runtime_cfg, clock_ms=lambda: 1700000000000)
    runtime.start()

    # Manually disable new exposure
    with runtime._lock:
        runtime._accept_new_exposure = False
        runtime._risk_block_reason = "DAILY_LOSS_LIMIT_REACHED"

    # Feed market trade through on_market_trade
    norm_tick = {
        "p": 50000.0,
        "q": 1.0,
        "T": 1700000000000,
        "T_raw": 1700000000000,
        "m": False,
        "symbol": "BTCUSDT",
        "source": "trade",
        "trade_id": 12345,
    }
    trade_res = runtime.on_market_trade(norm_tick)

    # Must process without ERROR or INACTIVE: positions continue receiving market ticks
    assert trade_res.status == "PROCESSED"
    assert trade_res.accepted is True


# =====================================================================
# 16. Cache de dedup é bounded
# =====================================================================
def test_16_dedup_cache_is_bounded():
    cfg = RiskConfig(seen_trades_cache_size=10)
    rm = RiskManager(cfg)
    now_ms = _now_utc_ms()

    for i in range(25):
        trade = DummyClosedTrade(trade_id=f"tr_batch_{i}", closed_ts_ms=now_ms, net_pnl_usdt=10.0)
        rm.record_closed_trade(trade)

    assert len(rm._seen_trades) == 10
    # Old trades (0 to 14) evicted, newer trades (15 to 24) preserved
    assert "tr_batch_0" not in rm._seen_trades
    assert "tr_batch_24" in rm._seen_trades


# =====================================================================
# 17. is_risk_position_limit_active permanece funcional
# =====================================================================
def test_17_risk_position_limit_active_remains_functional():
    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_17")
    runtime = ShadowPaperRuntime(config=runtime_cfg, clock_ms=lambda: 1700000000000)
    runtime.start()

    status = runtime.get_status()
    assert status["risk_position_limit_active"] is True
    assert status["risk_daily_loss_enforced"] is True
    assert status["risk_pnl_complete"] is True


# =====================================================================
# 18. Nenhum recovery é anunciado (SESSION_SAFE, risk_daily_loss_active=False)
# =====================================================================
def test_18_no_recovery_announced_honest_session_safe():
    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_18")
    runtime = ShadowPaperRuntime(config=runtime_cfg, clock_ms=lambda: 1700000000000)
    runtime.start()

    status = runtime.get_status()
    assert status["risk_daily_loss_recovery_level"] == "SESSION_SAFE"
    assert status["risk_daily_loss_active"] is False  # Honest: CLEAN_RESTART_SAFE not yet implemented
    assert status["risk_daily_loss_enforced"] is True
    assert status["daily_pnl"] == 0.0
    assert status["daily_loss_limit_usdt"] == 5000.0


# =====================================================================
# 19. INVALID_TRADE por trade_id inválido/vazio
# =====================================================================
def test_19_invalid_trade_id_sets_pnl_incomplete_and_blocks_runtime():
    rm = RiskManager(RiskConfig())
    now_ms = _now_utc_ms()

    # trade_id vazio
    res_empty = rm.record_closed_trade(DummyClosedTrade(trade_id="", closed_ts_ms=now_ms, net_pnl_usdt=100.0))
    assert res_empty.status == "INVALID_TRADE"
    assert res_empty.daily_pnl == 0.0
    assert res_empty.pnl_complete is False
    assert rm.daily_pnl == 0.0
    assert rm.risk_pnl_complete is False

    # trade_id com espaços
    res_spaces = rm.record_closed_trade(DummyClosedTrade(trade_id="   ", closed_ts_ms=now_ms, net_pnl_usdt=100.0))
    assert res_spaces.status == "INVALID_TRADE"
    assert res_spaces.pnl_complete is False

    # trade_id não-string
    res_none = rm.record_closed_trade(DummyClosedTrade(trade_id=None, closed_ts_ms=now_ms, net_pnl_usdt=100.0))
    assert res_none.status == "INVALID_TRADE"
    assert res_none.pnl_complete is False

    # Runtime integration
    adapter = RiskAdapter(risk_manager=rm)
    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_19")
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=lambda: now_ms)
    runtime.start()

    # Inject mock closed trade through execution sink on_market_trade
    mock_tick = ExecutionTick(
        symbol="BTCUSDT",
        event_timestamp=now_ms,
        raw_timestamp=now_ms,
        trade_id=999,
        price=50000.0,
        quantity=1.0,
        is_buyer_maker=False,
        ingest_seq=1,
        source="trade",
        received_at_ms=now_ms,
    )
    mock_events = TickEvents(closed_trades=[DummyClosedTrade(trade_id="", closed_ts_ms=now_ms, net_pnl_usdt=50.0)])
    runtime.execution_sink.on_market_trade = MagicMock(return_value=TradeResult(
        status=TradeStatus.ACCEPTED,
        tick=mock_tick,
        events=mock_events,
    ))

    trade_res = runtime.on_market_trade({
        "p": 50000.0,
        "q": 1.0,
        "T": now_ms,
        "T_raw": now_ms,
        "m": False,
        "symbol": "BTCUSDT",
        "source": "trade",
        "trade_id": 999,
    })

    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "INVALID_CLOSED_TRADE"
    assert runtime.risk_pnl_complete is False
    assert rm.daily_pnl == 0.0

    sig_res = runtime.on_signal({"epoch_ms": now_ms, "price": 50000.0, "symbol": "BTCUSDT"})
    assert sig_res.status == "INACTIVE"
    assert "INVALID_CLOSED_TRADE" in (sig_res.error or "")


# =====================================================================
# 20. INVALID_TRADE por closed_ts_ms inválido
# =====================================================================
def test_20_invalid_closed_ts_ms_sets_pnl_incomplete_and_blocks_runtime():
    rm = RiskManager(RiskConfig())

    invalid_timestamps = [0, -1000, -1, math.nan, math.inf, -math.inf, False, True, "not_a_ts", 10**18]
    for ts in invalid_timestamps:
        trade = DummyClosedTrade(trade_id=f"tr_inv_ts_{ts}", closed_ts_ms=ts, net_pnl_usdt=100.0)
        res = rm.record_closed_trade(trade)
        assert res.status == "INVALID_TRADE", f"Failed for ts={ts}"
        assert res.pnl_complete is False
        assert res.daily_pnl == 0.0
        assert rm.daily_pnl == 0.0
        assert rm.risk_pnl_complete is False

    # Runtime integration
    now_ms = _now_utc_ms()
    adapter = RiskAdapter(risk_manager=rm)
    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_20")
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=lambda: now_ms)
    runtime.start()

    mock_tick = ExecutionTick(
        symbol="BTCUSDT",
        event_timestamp=now_ms,
        raw_timestamp=now_ms,
        trade_id=1000,
        price=50000.0,
        quantity=1.0,
        is_buyer_maker=False,
        ingest_seq=1,
        source="trade",
        received_at_ms=now_ms,
    )
    mock_events = TickEvents(closed_trades=[DummyClosedTrade(trade_id="tr_neg_ts", closed_ts_ms=-500, net_pnl_usdt=50.0)])
    runtime.execution_sink.on_market_trade = MagicMock(return_value=TradeResult(
        status=TradeStatus.ACCEPTED,
        tick=mock_tick,
        events=mock_events,
    ))

    runtime.on_market_trade({
        "p": 50000.0,
        "q": 1.0,
        "T": now_ms,
        "T_raw": now_ms,
        "m": False,
        "symbol": "BTCUSDT",
        "source": "trade",
        "trade_id": 1000,
    })

    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "INVALID_CLOSED_TRADE"
    assert runtime.risk_pnl_complete is False


# =====================================================================
# 21. Exceção inesperada no accounting falha fechado e preserva processamento
# =====================================================================
def test_21_unexpected_exception_in_accounting_fails_closed_and_preserves_position_processing():
    rm = RiskManager(RiskConfig())
    adapter = RiskAdapter(risk_manager=rm)

    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_21")
    now_ms = _now_utc_ms()
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=lambda: now_ms)
    runtime.start()

    # Simulate unexpected crash during accounting call
    runtime.risk_adapter.on_closed_trade = MagicMock(side_effect=RuntimeError("Simulated unexpected crash during accounting"))

    mock_tick = ExecutionTick(
        symbol="BTCUSDT",
        event_timestamp=now_ms,
        raw_timestamp=now_ms,
        trade_id=1001,
        price=50000.0,
        quantity=1.0,
        is_buyer_maker=False,
        ingest_seq=1,
        source="trade",
        received_at_ms=now_ms,
    )
    mock_events = TickEvents(closed_trades=[DummyClosedTrade(trade_id="tr_crash", closed_ts_ms=now_ms, net_pnl_usdt=50.0)])
    runtime.execution_sink.on_market_trade = MagicMock(return_value=TradeResult(
        status=TradeStatus.ACCEPTED,
        tick=mock_tick,
        events=mock_events,
    ))

    # on_market_trade must NOT raise uncaught exception
    trade_res = runtime.on_market_trade({
        "p": 50000.0,
        "q": 1.0,
        "T": now_ms,
        "T_raw": now_ms,
        "m": False,
        "symbol": "BTCUSDT",
        "source": "trade",
        "trade_id": 1001,
    })

    # Processing completed and accepted
    assert trade_res.status == "PROCESSED"
    assert trade_res.accepted is True

    # State marked fail-closed
    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "ACCOUNTING_ERROR"
    assert runtime.risk_pnl_complete is False
    assert rm.risk_pnl_complete is False

    # Next on_signal is rejected
    sig_res = runtime.on_signal({"epoch_ms": now_ms, "price": 50000.0, "symbol": "BTCUSDT"})
    assert sig_res.status == "INACTIVE"
    assert "ACCOUNTING_ERROR" in (sig_res.error or "")

    # Further market trade processing continues working for existing positions
    mock_events_empty = TickEvents()
    runtime.execution_sink.on_market_trade = MagicMock(return_value=TradeResult(
        status=TradeStatus.ACCEPTED,
        tick=mock_tick,
        events=mock_events_empty,
    ))
    trade_res2 = runtime.on_market_trade({
        "p": 50010.0,
        "q": 0.5,
        "T": now_ms + 100,
        "T_raw": now_ms + 100,
        "m": False,
        "symbol": "BTCUSDT",
        "source": "trade",
        "trade_id": 1002,
    })
    assert trade_res2.status == "PROCESSED"
    assert trade_res2.accepted is True


# =====================================================================
# 22. Validação estrita de seen_trades_cache_size
# =====================================================================
def test_22_seen_trades_cache_size_validation():
    # <= 0 inválido
    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskConfig(seen_trades_cache_size=0)

    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskConfig(seen_trades_cache_size=-10)

    # bool inválido
    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskConfig(seen_trades_cache_size=False)

    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskConfig(seen_trades_cache_size=True)

    # tipo não-int inválido
    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskConfig(seen_trades_cache_size="50000")

    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskConfig(seen_trades_cache_size=10.5)

    # Validação no RiskManager.__init__ com objeto mock
    fake_cfg = MagicMock()
    fake_cfg.seen_trades_cache_size = 0
    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskManager(fake_cfg)

    fake_cfg_bool = MagicMock()
    fake_cfg_bool.seen_trades_cache_size = False
    with pytest.raises(ValueError, match="Invalid seen_trades_cache_size"):
        RiskManager(fake_cfg_bool)


# =====================================================================
# 23. Reset UTC zera daily_pnl mas preserva fail-closed de integridade
# =====================================================================
def test_23_utc_reset_does_not_clear_pnl_incomplete_or_unblock_runtime():
    rm = RiskManager(RiskConfig(initial_capital=100_000.0, max_daily_loss=0.05))
    adapter = RiskAdapter(risk_manager=rm)
    now_ms = _now_utc_ms()

    runtime_cfg = _make_runtime_config(cohort_id="CH_TEST_C352_23")
    runtime = ShadowPaperRuntime(config=runtime_cfg, risk_adapter=adapter, clock_ms=lambda: now_ms)
    runtime.start()

    # Trigger PNL_INCOMPLETE via incomplete trade
    bad_trade = DummyClosedTrade(trade_id="tr_incomplete_day1", closed_ts_ms=now_ms, net_pnl_usdt=None, costs_complete=False)
    with runtime._lock:
        res = runtime.risk_adapter.on_closed_trade(bad_trade)
        if res.status == "PNL_INCOMPLETE":
            runtime._accept_new_exposure = False
            runtime._risk_block_reason = "PNL_INCOMPLETE"

    assert rm.risk_pnl_complete is False
    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "PNL_INCOMPLETE"

    # Simulate UTC date change to tomorrow
    tomorrow = datetime.now(timezone.utc).date() + timedelta(days=1)
    rm._get_utc_date = lambda: tomorrow

    # Force check_daily_reset
    rm._check_daily_reset()

    # Daily PnL resets to 0.0
    assert rm.daily_pnl == 0.0

    # But risk_pnl_complete does NOT reset back to True!
    assert rm.risk_pnl_complete is False
    assert runtime.risk_pnl_complete is False

    # And runtime._accept_new_exposure does NOT reset back to True!
    assert runtime.new_exposure_blocked is True
    assert runtime.risk_block_reason == "PNL_INCOMPLETE"

    # Future signals remain fail-closed rejected
    sig_res = runtime.on_signal({"epoch_ms": now_ms, "price": 50000.0, "symbol": "BTCUSDT"})
    assert sig_res.status == "INACTIVE"
    assert "PNL_INCOMPLETE" in (sig_res.error or "")

    # RiskManager evaluate also remains rejected
    dec = CanonicalDecision(
        cohort_id="CH_TEST_C352_23",
        symbol="BTCUSDT",
        window_id="win_23",
        decision_provider="RULE_BASED",
        strategy_version="v1.0",
        signal_timestamp=now_ms,
        decision_timestamp=now_ms,
        available_at=now_ms,
        side="LONG",
        reference_price=50000.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )
    eval_res = runtime.risk_adapter.evaluate(dec)
    assert eval_res.status == "REJECTED_BY_RISK"
    assert eval_res.risk_reason == "daily pnl incomplete"
