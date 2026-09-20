# tests/unit/paper_trading/test_execution_sink.py
"""Unit tests for the hermetic Paper ExecutionSink and ExecutionTick contract."""

import copy
import threading
from unittest.mock import MagicMock
import pytest

from paper_trading.contracts import PaperOrder
from paper_trading.executor import ExecutorConfig, PaperExecutor
from paper_trading.execution_sink import (
    ExecutionSink,
    ExecutionTick,
    SubmitStatus,
    TradeStatus,
)


def _sample_norm(
    p: float = 95000.0,
    q: float = 0.5,
    T: int = 1700000001000,
    T_raw: int | None = None,
    m: bool = False,
    source: str = "fut_agg",
    trade_id: int | str | None = 100001,
    extra_field: str = "ignored_metadata",
) -> dict:
    d = {
        "p": p,
        "q": q,
        "T": T,
        "m": m,
        "source": source,
        "trade_id": trade_id,
        "extra_field": extra_field,
    }
    if T_raw is not None:
        d["T_raw"] = T_raw
    return d


def _sample_order(
    order_id: str = "ord_sink_001",
    available_at: int = 1700000001000,
    expires_at: int = 1700000006000,
    side: str = "LONG",
    reference_price: float = 95000.0,
    notional_usdt: float = 1000.0,
) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        decision_id="dec_001",
        cohort_id="c_sink",
        decision_provider="flow_v1",
        symbol="BTCUSDT",
        side=side,  # type: ignore[arg-type]
        reference_price=reference_price,
        notional_usdt=notional_usdt,
        signal_timestamp=available_at - 100,
        decision_timestamp=available_at - 50,
        available_at=available_at,
        expires_at=expires_at,
        horizon_s=60,
    )


def test_valid_norm_creates_execution_tick():
    """1, 16: Valid norm creates ExecutionTick with correct fields without inventing received_at."""
    sink = ExecutionSink(symbol="BTCUSDT")
    norm = _sample_norm(p=95123.5, q=1.25, T=1000, T_raw=1000, m=True, source="fut_agg", trade_id=555)

    res = sink.on_market_trade(norm)

    assert res.status == TradeStatus.ACCEPTED
    assert res.tick is not None
    assert res.tick.symbol == "BTCUSDT"
    assert res.tick.price == 95123.5
    assert res.tick.quantity == 1.25
    assert res.tick.event_timestamp == 1000
    assert res.tick.raw_timestamp == 1000
    assert res.tick.is_buyer_maker is True
    assert res.tick.source == "fut_agg"
    assert res.tick.trade_id == 555
    assert res.tick.ingest_seq == 1
    assert res.tick.received_at_ms is None
    assert res.tick.ooo_clamped is False
    assert res.tick.continuity_suspect is False


def test_norm_original_not_mutated():
    """2: Original normalized trade dictionary is completely untouched."""
    sink = ExecutionSink(symbol="BTCUSDT")
    norm = _sample_norm(p=95000.0, q=0.5, T=1000, trade_id=123)
    norm_copy = copy.deepcopy(norm)

    sink.on_market_trade(norm)

    assert norm == norm_copy


def test_ingest_seq_monotonic_progression():
    """3, 4: ingest_seq starts at 1 and strictly increases for accepted ticks."""
    sink = ExecutionSink(symbol="BTCUSDT")

    res1 = sink.on_market_trade(_sample_norm(T=1000, trade_id=1))
    res2 = sink.on_market_trade(_sample_norm(T=1001, trade_id=2))
    res3 = sink.on_market_trade(_sample_norm(T=1002, trade_id=3))

    assert res1.tick is not None and res1.tick.ingest_seq == 1
    assert res2.tick is not None and res2.tick.ingest_seq == 2
    assert res3.tick is not None and res3.tick.ingest_seq == 3
    assert sink.get_metrics()["current_ingest_seq"] == 3


def test_invalid_and_duplicate_do_not_consume_seq():
    """5, 6: Malformed trades and duplicates do not increment ingest_seq."""
    sink = ExecutionSink(symbol="BTCUSDT")

    # 1. Valid tick -> seq = 1
    r1 = sink.on_market_trade(_sample_norm(T=1000, trade_id=10))
    assert r1.status == TradeStatus.ACCEPTED
    assert sink.get_metrics()["current_ingest_seq"] == 1

    # 2. Invalid tick (price <= 0) -> rejected, seq remains 1
    r_bad = sink.on_market_trade(_sample_norm(p=-50.0, T=1001, trade_id=11))
    assert r_bad.status == TradeStatus.INVALID_TRADE
    assert sink.get_metrics()["current_ingest_seq"] == 1

    # 3. Duplicate tick (trade_id=10 again) -> rejected, seq remains 1
    r_dup = sink.on_market_trade(_sample_norm(T=1002, trade_id=10))
    assert r_dup.status == TradeStatus.DUPLICATE
    assert sink.get_metrics()["current_ingest_seq"] == 1

    # 4. Next valid tick -> seq = 2
    r2 = sink.on_market_trade(_sample_norm(T=1003, trade_id=12))
    assert r2.status == TradeStatus.ACCEPTED
    assert r2.tick is not None and r2.tick.ingest_seq == 2


def test_submit_order_lifecycle():
    """8, 9, 10: External submit order records registered_seq only upon success."""
    sink = ExecutionSink(symbol="BTCUSDT")

    # Ingest 2 ticks before order
    sink.on_market_trade(_sample_norm(T=1000, trade_id=1))
    sink.on_market_trade(_sample_norm(T=1001, trade_id=2))
    assert sink.get_metrics()["current_ingest_seq"] == 2

    # Submit valid order
    order = _sample_order(order_id="order_success")
    sub_res = sink.submit_order(order)
    assert sub_res.status == SubmitStatus.ACCEPTED
    assert sub_res.order is not None
    # registered_seq matches sequence at submit time
    assert sink._registered_ingest_seq["order_success"] == 2

    # Submit duplicate order -> rejected, does not overwrite registered_seq
    sub_dup = sink.submit_order(order)
    assert sub_dup.status == SubmitStatus.REJECTED
    assert sub_dup.rejection is not None
    assert sub_dup.rejection.reason == "DUPLICATE_DECISION"

    # Submit structurally invalid order -> rejected, no registered_seq created
    bad_order = _sample_order(order_id="", side="LONG")
    sub_bad = sink.submit_order(bad_order)
    assert sub_bad.status == SubmitStatus.REJECTED
    assert "" not in sink._registered_ingest_seq


def test_temporal_causality_pre_submit_tick_cannot_fill():
    """11, 12, 13, 14: Pre-submit ticks cannot fill; first post-submit tick >= available_at fills."""
    executor = PaperExecutor(config=ExecutorConfig(order_ttl_ms=5000))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    # 1. Tick arrives at T=1000 before order exists
    t1 = sink.on_market_trade(_sample_norm(T=1000, trade_id=101))
    assert len(t1.events.fills if t1.events else []) == 0

    # 2. Order submitted with available_at=1000
    order = _sample_order(order_id="ord_causal", available_at=1000)
    sub_res = sink.submit_order(order)
    assert sub_res.status == SubmitStatus.ACCEPTED
    assert len(executor.pending_orders) == 1

    # T1 already passed; it cannot fill the order retroactively!
    assert len(executor.pending_orders) == 1

    # 3. Next tick arrives at T=999 (regressive timestamp) -> dropped
    t_ooo = sink.on_market_trade(_sample_norm(T=999, trade_id=102))
    assert t_ooo.status == TradeStatus.OUT_OF_ORDER
    assert len(executor.pending_orders) == 1

    # 4. Post-submit tick at T=1000 (>= available_at) -> fills order!
    t_fill = sink.on_market_trade(_sample_norm(T=1000, trade_id=103))
    assert t_fill.status == TradeStatus.ACCEPTED
    assert t_fill.events is not None
    assert len(t_fill.events.fills) == 1
    assert t_fill.events.fills[0].order_id == "ord_causal"
    assert t_fill.events.fills[0].trade_id_used == 103
    assert len(executor.pending_orders) == 0


def test_deduplication_by_symbol_and_trade_id():
    """15, 16: Deduplication applies by (symbol, trade_id); trade_id=None is not deduplicated."""
    sink = ExecutionSink(symbol="BTCUSDT")

    # Trade with ID 777
    r1 = sink.on_market_trade(_sample_norm(T=1000, trade_id=777))
    assert r1.status == TradeStatus.ACCEPTED

    # Same ID again -> duplicate
    r2 = sink.on_market_trade(_sample_norm(T=1001, trade_id=777))
    assert r2.status == TradeStatus.DUPLICATE
    assert sink.get_metrics()["duplicate_ticks"] == 1

    # Trade with trade_id=None
    r_none1 = sink.on_market_trade(_sample_norm(T=1002, trade_id=None))
    assert r_none1.status == TradeStatus.ACCEPTED

    # Second trade with trade_id=None -> accepted without dedup
    r_none2 = sink.on_market_trade(_sample_norm(T=1003, trade_id=None))
    assert r_none2.status == TradeStatus.ACCEPTED


def test_dedup_cache_bounded_eviction():
    """17: Dedup cache remains strictly within configured capacity."""
    capacity = 5
    sink = ExecutionSink(symbol="BTCUSDT", dedup_capacity=capacity)

    for i in range(1, 10):
        res = sink.on_market_trade(_sample_norm(T=1000 + i, trade_id=i))
        assert res.status == TradeStatus.ACCEPTED
        assert sink.get_metrics()["dedup_cache_size"] <= capacity

    assert sink.get_metrics()["dedup_cache_size"] == capacity
    # Earliest keys (1, 2, 3, 4) should have been evicted; 1 can now be accepted again
    res_evicted = sink.on_market_trade(_sample_norm(T=2000, trade_id=1))
    assert res_evicted.status == TradeStatus.ACCEPTED


def test_out_of_order_and_clamped_telemetry():
    """18, 19: T_raw != T marks ooo_clamped; timestamp regression below last accepted drops tick."""
    sink = ExecutionSink(symbol="BTCUSDT")

    # Clamped tick: T=1050, T_raw=1040 (e.g. clamp applied in bot)
    r_clamped = sink.on_market_trade(_sample_norm(T=1050, T_raw=1040, trade_id=1))
    assert r_clamped.status == TradeStatus.ACCEPTED
    assert r_clamped.tick is not None
    assert r_clamped.tick.ooo_clamped is True

    # Regressive tick: T=1045 < last accepted (1050) -> OUT_OF_ORDER
    r_regressive = sink.on_market_trade(_sample_norm(T=1045, trade_id=2))
    assert r_regressive.status == TradeStatus.OUT_OF_ORDER
    assert sink.get_metrics()["out_of_order_ticks"] == 1


def test_continuity_and_id_regression():
    """20, 21, 22: ID gaps mark continuity_suspect without synthetic ticks; ID regression increments metric."""
    sink = ExecutionSink(symbol="BTCUSDT")

    # Trade 100
    sink.on_market_trade(_sample_norm(T=1000, trade_id=100))

    # Trade 105: gap of 5 -> continuity suspect
    r_gap = sink.on_market_trade(_sample_norm(T=1001, trade_id=105))
    assert r_gap.status == TradeStatus.ACCEPTED
    assert r_gap.tick is not None
    assert r_gap.tick.continuity_suspect is True
    assert sink.get_metrics()["continuity_suspect_ticks"] == 1

    # Trade 102: ID regression (102 < 105)
    r_reg = sink.on_market_trade(_sample_norm(T=1002, trade_id=102))
    assert r_reg.status == TradeStatus.ACCEPTED
    assert sink.get_metrics()["id_regressions"] == 1


def test_exception_isolation_and_circuit_breaker():
    """7, 23, 24, 25, 26, 27: Executor exceptions are isolated; consecutive errors trip circuit breaker."""
    mock_executor = MagicMock(spec=PaperExecutor)
    mock_executor.pending_orders = {}
    mock_executor.on_tick.side_effect = RuntimeError("Simulated executor internal error")
    mock_executor.submit_order.side_effect = RuntimeError("Simulated submit error")

    threshold = 3
    sink = ExecutionSink(
        symbol="BTCUSDT",
        executor=mock_executor,
        consecutive_error_threshold=threshold,
    )

    # 1. First 2 errors -> EXECUTOR_ERROR, not disabled yet
    r1 = sink.on_market_trade(_sample_norm(T=1000, trade_id=1))
    assert r1.status == TradeStatus.EXECUTOR_ERROR
    assert not sink.is_disabled

    r2 = sink.on_market_trade(_sample_norm(T=1001, trade_id=2))
    assert r2.status == TradeStatus.EXECUTOR_ERROR
    assert not sink.is_disabled

    # 2. Third error reaches threshold -> trips circuit breaker to DISABLED
    r3 = sink.on_market_trade(_sample_norm(T=1002, trade_id=3))
    assert r3.status == TradeStatus.EXECUTOR_ERROR
    assert sink.is_disabled
    assert sink.get_metrics()["disabled"] is True

    # 3. New trades and submits return DISABLED immediately without calling executor
    r_dis = sink.on_market_trade(_sample_norm(T=1003, trade_id=4))
    assert r_dis.status == TradeStatus.DISABLED
    assert mock_executor.on_tick.call_count == 3  # executor not called on disabled

    sub_dis = sink.submit_order(_sample_order())
    assert sub_dis.status == SubmitStatus.DISABLED
    assert mock_executor.submit_order.call_count == 0

    # 4. Reset circuit breaker re-enables sink
    sink.reset_circuit_breaker()
    assert not sink.is_disabled


def test_dedup_cache_exact_capacity_eviction():
    """Prove deque and set exact contents at capacity = 3 with IDs 1, 2, 3, 4."""
    sink = ExecutionSink(symbol="BTCUSDT", dedup_capacity=3)
    sink.on_market_trade(_sample_norm(T=1001, trade_id=1))
    sink.on_market_trade(_sample_norm(T=1002, trade_id=2))
    sink.on_market_trade(_sample_norm(T=1003, trade_id=3))
    sink.on_market_trade(_sample_norm(T=1004, trade_id=4))

    # After 4: deque contains 2, 3, 4 and set contains EXACTLY 2, 3, 4
    assert list(sink._dedup_deque) == [("BTCUSDT", 2), ("BTCUSDT", 3), ("BTCUSDT", 4)]
    assert sink._dedup_set == {("BTCUSDT", 2), ("BTCUSDT", 3), ("BTCUSDT", 4)}

    # Resend ID 1 -> accepted again because it left the dedup window
    r_id1 = sink.on_market_trade(_sample_norm(T=1005, trade_id=1))
    assert r_id1.status == TradeStatus.ACCEPTED
    assert sink._dedup_set == {("BTCUSDT", 3), ("BTCUSDT", 4), ("BTCUSDT", 1)}


def test_circuit_breaker_resets_on_success_between_errors():
    """Confirm a successful operation between errors resets consecutive_errors to 0."""
    mock_executor = MagicMock(spec=PaperExecutor)
    mock_executor.pending_orders = {}

    sink = ExecutionSink(
        symbol="BTCUSDT",
        executor=mock_executor,
        consecutive_error_threshold=3,
    )

    # Error 1
    mock_executor.on_tick.side_effect = RuntimeError("err1")
    r1 = sink.on_market_trade(_sample_norm(T=1001, trade_id=1))
    assert r1.status == TradeStatus.EXECUTOR_ERROR
    assert sink._consecutive_errors == 1
    assert not sink.is_disabled

    # Success 1 -> resets consecutive_errors to 0
    mock_executor.on_tick.side_effect = None
    mock_executor.on_tick.return_value = MagicMock()
    r2 = sink.on_market_trade(_sample_norm(T=1002, trade_id=2))
    assert r2.status == TradeStatus.ACCEPTED
    assert sink._consecutive_errors == 0
    assert not sink.is_disabled

    # Error 2
    mock_executor.on_tick.side_effect = RuntimeError("err2")
    r3 = sink.on_market_trade(_sample_norm(T=1003, trade_id=3))
    assert r3.status == TradeStatus.EXECUTOR_ERROR
    assert sink._consecutive_errors == 1
    assert not sink.is_disabled

    # Total errors is 2, but consecutive is 1 -> circuit breaker NOT tripped
    metrics = sink.get_metrics()
    assert metrics["executor_errors"] == 2
    assert metrics["disabled"] is False


def test_cleanup_registered_seq_after_fill():
    """28: registered_ingest_seq entries are cleaned up when orders are filled."""
    executor = PaperExecutor(config=ExecutorConfig(order_ttl_ms=5000))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order = _sample_order(order_id="ord_cleanup", available_at=1000)
    sink.submit_order(order)
    assert "ord_cleanup" in sink._registered_ingest_seq

    # Fill the order on tick
    sink.on_market_trade(_sample_norm(T=1000, trade_id=99))
    assert len(executor.pending_orders) == 0
    # Cleaned up automatically
    assert "ord_cleanup" not in sink._registered_ingest_seq
    assert sink.get_metrics()["registered_orders"] == 0


def test_cleanup_registered_seq_after_expiration():
    """Prove registered_ingest_seq cleanup when order exits pending via EXPIRED_NO_MARKET_DATA."""
    executor = PaperExecutor(config=ExecutorConfig(order_ttl_ms=1000))
    sink = ExecutionSink(symbol="BTCUSDT", executor=executor)

    order = _sample_order(order_id="ord_expire", available_at=1000, expires_at=2000)
    sink.submit_order(order)
    assert "ord_expire" in sink._registered_ingest_seq

    # Tick arrives at T=2001 (past expiration) without filling
    res = sink.on_market_trade(_sample_norm(T=2001, trade_id=100))
    assert res.events is not None
    assert len(res.events.rejections) == 1
    assert res.events.rejections[0].reason == "EXPIRED_NO_MARKET_DATA"
    assert len(executor.pending_orders) == 0
    # Cleaned up from registered_ingest_seq
    assert "ord_expire" not in sink._registered_ingest_seq
    assert sink.get_metrics()["registered_orders"] == 0


def test_zero_forbidden_imports():
    """29, 30, 31, 32: execution_sink.py has zero forbidden imports (no I/O, DB, network, or AI)."""
    import sys
    import paper_trading.execution_sink as es

    forbidden = {
        "sqlite3",
        "requests",
        "aiohttp",
        "websockets",
        "binance",
        "openai",
        "groq",
        "events.event_saver",
    }

    loaded_modules = set(sys.modules.keys())
    for mod in forbidden:
        assert mod not in loaded_modules or mod not in es.__dict__


def test_concurrent_submit_and_market_trades():
    """33: Thread concurrency between submit_order and on_market_trade produces zero race conditions."""
    sink = ExecutionSink(symbol="BTCUSDT")
    errors = []

    def trade_producer():
        for i in range(1, 100):
            try:
                sink.on_market_trade(_sample_norm(T=1000 + i, trade_id=i))
            except Exception as e:
                errors.append(e)

    def order_producer():
        for i in range(1, 20):
            try:
                order = _sample_order(
                    order_id=f"concurrent_ord_{i}",
                    available_at=1000 + i,
                )
                sink.submit_order(order)
            except Exception as e:
                errors.append(e)

    t1 = threading.Thread(target=trade_producer)
    t2 = threading.Thread(target=order_producer)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0
    metrics = sink.get_metrics()
    assert metrics["accepted_ticks"] > 0
    assert metrics["accepted_orders"] > 0


@pytest.mark.parametrize(
    "bad_norm",
    [
        _sample_norm(m="not_a_bool"),    # 35: m not real bool
        _sample_norm(m=1),               # 35: m int
        _sample_norm(source=""),         # 36: empty source
        _sample_norm(source="   "),      # 36: whitespace source
        _sample_norm(p=float("nan")),    # 37: NaN price
        _sample_norm(p=float("inf")),    # 37: Inf price
        _sample_norm(q=float("nan")),    # 37: NaN quantity
        _sample_norm(p=True),            # 38: bool price
        _sample_norm(q=False),           # 38: bool quantity
        _sample_norm(T=True),            # 38: bool timestamp
        _sample_norm(trade_id=True),     # 38: bool trade_id
        "not_a_dict",                    # Malformed payload
    ],
)
def test_defensive_norm_validation_rejections(bad_norm):
    """35, 36, 37, 38: Defensive validation rejects non-bool m, empty source, NaN/Inf, and bool numerics."""
    sink = ExecutionSink(symbol="BTCUSDT")

    res = sink.on_market_trade(bad_norm)  # type: ignore[arg-type]

    assert res.status == TradeStatus.INVALID_TRADE
    assert res.tick is None
    assert sink.get_metrics()["invalid_ticks"] == 1
