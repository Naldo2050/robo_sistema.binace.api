# tests/unit/paper_trading/test_executor_submit_order.py
"""Unit tests for PaperExecutor.submit_order() contract, boundary hardening, and backward compatibility."""

import math
import pytest

from paper_trading.contracts import CanonicalDecision, PaperOrder
from paper_trading.executor import ExecutorConfig, PaperExecutor
from paper_trading.positions import PositionManager


def _make_external_order(
    order_id: str = "custom_ext_order_uuid_123",
    decision_id: str = "dec_ext_999",
    cohort_id: str = "cohort_alpha",
    decision_provider: str = "ml_flow",
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    reference_price: float = 95000.0,
    notional_usdt: float = 1000.0,
    signal_timestamp: int = 1700000000000,
    decision_timestamp: int = 1700000000050,
    available_at: int = 1700000000100,
    expires_at: int = 1700000005000,
    horizon_s: int = 120,
    stop_loss: float | None = 94000.0,
    take_profit: float | None = 97000.0,
    funding_rate_at_decision: float | None = 0.0001,
    funding_rate_source: str | None = "binance_futures",
    context: dict | None = None,
) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        decision_id=decision_id,
        cohort_id=cohort_id,
        decision_provider=decision_provider,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        reference_price=reference_price,
        notional_usdt=notional_usdt,
        signal_timestamp=signal_timestamp,
        decision_timestamp=decision_timestamp,
        available_at=available_at,
        expires_at=expires_at,
        horizon_s=horizon_s,
        stop_loss=stop_loss,
        take_profit=take_profit,
        funding_rate_at_decision=funding_rate_at_decision,
        funding_rate_source=funding_rate_source,
        context=context or {"approved_by": "risk_adapter_v2"},
    )


# ==============================================================================
# BASELINE FUNCTIONAL TESTS
# ==============================================================================

def test_submit_order_external_long_success():
    """External LONG PaperOrder enters pending with exact preserved fields."""
    executor = PaperExecutor()
    order = _make_external_order(side="LONG")

    reg_order, rej = executor.submit_order(order)

    assert rej is None
    assert reg_order is order
    assert reg_order.order_id == "custom_ext_order_uuid_123"
    assert reg_order.decision_id == "dec_ext_999"
    assert reg_order.expires_at == 1700000005000
    assert reg_order.stop_loss == 94000.0
    assert reg_order.take_profit == 97000.0
    assert reg_order.funding_rate_at_decision == 0.0001
    assert reg_order.funding_rate_source == "binance_futures"
    assert reg_order.context == {"approved_by": "risk_adapter_v2"}
    assert "custom_ext_order_uuid_123" in executor.pending_orders
    assert executor.pending_orders["custom_ext_order_uuid_123"] is order


def test_submit_order_external_short_success():
    """External SHORT PaperOrder enters pending successfully."""
    executor = PaperExecutor()
    order = _make_external_order(
        order_id="custom_ext_short_456",
        side="SHORT",
        stop_loss=96000.0,
        take_profit=93000.0,
    )

    reg_order, rej = executor.submit_order(order)

    assert rej is None
    assert reg_order is order
    assert reg_order.side == "SHORT"
    assert "custom_ext_short_456" in executor.pending_orders


def test_submit_order_rejects_neutral_side():
    """NEUTRAL side fails closed with reason NO_DIRECTION."""
    executor = PaperExecutor()
    order = _make_external_order(side="NEUTRAL")

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "NO_DIRECTION"
    assert "NEUTRAL" in rej.details
    assert len(executor.pending_orders) == 0


def test_submit_order_rejects_unknown_side():
    """UNKNOWN side fails closed with reason NO_DIRECTION."""
    executor = PaperExecutor()
    order = _make_external_order(side="UNKNOWN")

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "NO_DIRECTION"
    assert "UNKNOWN" in rej.details
    assert len(executor.pending_orders) == 0


def test_submit_order_rejects_conflict_with_pending_order():
    """Conflicting order for same (cohort, provider, symbol) fails with POSITION_OPEN."""
    executor = PaperExecutor()
    order1 = _make_external_order(order_id="order_1")
    reg1, rej1 = executor.submit_order(order1)
    assert reg1 is not None
    assert rej1 is None

    order2 = _make_external_order(order_id="order_2")
    reg2, rej2 = executor.submit_order(order2)

    assert reg2 is None
    assert rej2 is not None
    assert rej2.reason == "POSITION_OPEN"
    assert len(executor.pending_orders) == 1


def test_submit_order_rejects_conflict_with_open_position():
    """If a position is already open for (cohort, provider, symbol), reject with POSITION_OPEN."""
    pm = PositionManager()
    executor = PaperExecutor(position_manager=pm)

    order1 = _make_external_order(
        order_id="order_fillable",
        signal_timestamp=900,
        decision_timestamp=950,
        available_at=1000,
        expires_at=5000,
    )
    executor.submit_order(order1)
    ev = executor.on_tick(T=1000, p=95000.0, q=0.01, m=False)
    assert len(ev.fills) == 1
    assert pm.has_open_position("cohort_alpha", "ml_flow", "BTCUSDT") is True

    order2 = _make_external_order(
        order_id="order_after_open",
        signal_timestamp=900,
        decision_timestamp=950,
        available_at=1000,
        expires_at=5000,
    )
    reg2, rej2 = executor.submit_order(order2)

    assert reg2 is None
    assert rej2 is not None
    assert rej2.reason == "POSITION_OPEN"


def test_submit_order_duplicate_order_id_fails_closed():
    """30: Duplicate order details clearly identifies duplicate_order_id without overwriting."""
    executor = PaperExecutor()
    order1 = _make_external_order(
        order_id="same_exact_id",
        symbol="BTCUSDT",
        cohort_id="cohort_1",
    )
    reg1, rej1 = executor.submit_order(order1)
    assert reg1 is not None

    order2 = _make_external_order(
        order_id="same_exact_id",
        symbol="ETHUSDT",
        cohort_id="cohort_2",
    )
    reg2, rej2 = executor.submit_order(order2)

    assert reg2 is None
    assert rej2 is not None
    assert rej2.reason == "DUPLICATE_DECISION"
    assert "duplicate_order_id" in rej2.details
    assert len(executor.pending_orders) == 1


def test_submit_order_does_not_produce_fill_immediately():
    """submit_order only stages pending; only subsequent on_tick produces fills."""
    executor = PaperExecutor()
    order = _make_external_order(
        signal_timestamp=900,
        decision_timestamp=950,
        available_at=1000,
        expires_at=5000,
    )

    reg_order, rej = executor.submit_order(order)
    assert reg_order is not None
    assert rej is None
    assert len(executor.pending_orders) == 1
    assert len(executor.position_manager.open_positions) == 0

    # Tick before available_at: no fill
    ev_early = executor.on_tick(T=999, p=95000.0, q=0.01, m=False)
    assert len(ev_early.fills) == 0
    assert len(executor.pending_orders) == 1

    # Tick at available_at: fills
    ev_fill = executor.on_tick(T=1000, p=95000.0, q=0.01, m=False, trade_id=888)
    assert len(ev_fill.fills) == 1
    fill = ev_fill.fills[0]
    assert fill.order_id == "custom_ext_order_uuid_123"
    assert fill.trade_id_used == 888
    assert len(executor.pending_orders) == 0


def test_submit_decision_backwards_compatibility():
    """29: submit_decision maintains legacy order_id format, TTL derivation, and rejection semantics."""
    executor = PaperExecutor(config=ExecutorConfig(order_ttl_ms=3000))
    d = CanonicalDecision(
        cohort_id="c_legacy",
        symbol="BTCUSDT",
        window_id="w1",
        decision_provider="provider_legacy",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )

    order, rej = executor.submit_decision(d)
    assert rej is None
    assert order is not None
    assert order.order_id == f"ord_{d.decision_id[:12]}_1020"
    assert order.expires_at == 1050 + 3000

    d_conflict = CanonicalDecision(
        cohort_id="c_legacy",
        symbol="BTCUSDT",
        window_id="w2",
        decision_provider="provider_legacy",
        strategy_version="1.0",
        signal_timestamp=2000,
        decision_timestamp=2020,
        available_at=2050,
        side="LONG",
        reference_price=100.0,
        notional_usdt=1000.0,
        horizon_s=60,
    )
    order2, rej2 = executor.submit_decision(d_conflict)
    assert order2 is None
    assert rej2 is not None
    assert rej2.reason == "POSITION_OPEN"


def test_equivalence_between_decision_and_order_registration():
    """Equivalence test between direct decision route and pre-built order route."""
    executor_a = PaperExecutor(config=ExecutorConfig(order_ttl_ms=4000))
    executor_b = PaperExecutor(config=ExecutorConfig(order_ttl_ms=4000))

    decision = CanonicalDecision(
        cohort_id="c_equiv",
        symbol="BTCUSDT",
        window_id="w_eq",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=1000,
        decision_timestamp=1020,
        available_at=1050,
        side="SHORT",
        reference_price=90000.0,
        notional_usdt=500.0,
        horizon_s=30,
        stop_loss=91000.0,
        take_profit=88000.0,
        funding_rate_at_decision=0.0002,
        funding_rate_source="test_source",
        context={"tag": "equiv_test"},
    )

    order_a, rej_a = executor_a.submit_decision(decision)
    assert rej_a is None
    assert order_a is not None

    order_b_prebuilt = PaperOrder(
        order_id="arbitrary_risk_uuid_777",
        decision_id=decision.decision_id,
        cohort_id=decision.cohort_id,
        decision_provider=decision.decision_provider,
        symbol=decision.symbol,
        side=decision.side,
        reference_price=decision.reference_price,
        notional_usdt=decision.notional_usdt,
        signal_timestamp=decision.signal_timestamp,
        decision_timestamp=decision.decision_timestamp,
        available_at=decision.available_at,
        expires_at=decision.available_at + 4000,
        horizon_s=decision.horizon_s,
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        funding_rate_at_decision=decision.funding_rate_at_decision,
        funding_rate_source=decision.funding_rate_source,
        context=decision.context,
    )
    order_b, rej_b = executor_b.submit_order(order_b_prebuilt)
    assert rej_b is None
    assert order_b is not None

    assert order_a.cohort_id == order_b.cohort_id
    assert order_a.decision_provider == order_b.decision_provider
    assert order_a.symbol == order_b.symbol
    assert order_a.side == order_b.side
    assert order_a.reference_price == order_b.reference_price
    assert order_a.notional_usdt == order_b.notional_usdt
    assert order_a.available_at == order_b.available_at
    assert order_a.expires_at == order_b.expires_at
    assert order_a.horizon_s == order_b.horizon_s
    assert order_a.stop_loss == order_b.stop_loss
    assert order_a.take_profit == order_b.take_profit
    assert order_a.funding_rate_at_decision == order_b.funding_rate_at_decision
    assert order_a.funding_rate_source == order_b.funding_rate_source
    assert order_a.context == order_b.context


# ==============================================================================
# C3-B0.1 BOUNDARY HARDENING VALIDATION TESTS
# ==============================================================================

@pytest.mark.parametrize(
    "field_name,bad_val,expected_detail",
    [
        ("order_id", "", "invalid_order_id"),
        ("order_id", "   ", "invalid_order_id"),
        ("decision_id", "", "invalid_decision_id"),
        ("decision_id", "   ", "invalid_decision_id"),
        ("cohort_id", "", "invalid_cohort_id"),
        ("decision_provider", "", "invalid_decision_provider"),
        ("symbol", "", "invalid_symbol"),
        ("symbol", "   ", "invalid_symbol"),
    ],
)
def test_boundary_empty_string_fields_rejected(field_name, bad_val, expected_detail):
    """1, 2, 3: Empty or whitespace-only mandatory string fields fail closed."""
    executor = PaperExecutor()
    kwargs = {field_name: bad_val}
    order = _make_external_order(**kwargs)

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == expected_detail
    # 26, 27: Executor state remains unchanged
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


def test_boundary_invalid_runtime_side_rejected():
    """4: Invalid side outside typed contract fails closed with INVALID_DECISION."""
    executor = PaperExecutor()
    order = _make_external_order(side="INVALID_SIDE_STR")

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == "invalid_side"
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


@pytest.mark.parametrize(
    "bad_ref_price",
    [
        0.0,           # 5: zero
        -10.0,         # 6: negative
        float("nan"),  # 7: NaN
        float("inf"),  # 8: +Inf
        float("-inf"), # 8: -Inf
        True,          # 9: bool True
        False,         # 9: bool False
    ],
)
def test_boundary_reference_price_invalid_rejected(bad_ref_price):
    """5, 6, 7, 8, 9: reference_price must be finite real number > 0 and not bool."""
    executor = PaperExecutor()
    order = _make_external_order(reference_price=bad_ref_price)

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == "invalid_reference_price"
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


@pytest.mark.parametrize(
    "bad_notional",
    [
        0.0,           # 10: zero
        -500.0,        # 11: negative
        float("nan"),  # 12: NaN
        float("inf"),  # Inf
        True,          # 13: bool True
        False,         # bool False
    ],
)
def test_boundary_notional_invalid_rejected(bad_notional):
    """10, 11, 12, 13: notional_usdt must be finite real number > 0 and not bool."""
    executor = PaperExecutor()
    order = _make_external_order(notional_usdt=bad_notional)

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == "invalid_notional"
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


@pytest.mark.parametrize(
    "sig,dec,avail,exp",
    [
        (1000, 1020, 1050, 1050),  # 14: expires_at == available_at
        (1000, 1020, 1050, 1040),  # 15: expires_at < available_at
        (1020, 1000, 1050, 2000),  # 16: signal > decision
        (1000, 1060, 1050, 2000),  # 16: decision > available
        (True, 1020, 1050, 2000),  # bool in timestamp
        (1000, False, 1050, 2000), # bool in timestamp
    ],
)
def test_boundary_timestamps_invalid_rejected(sig, dec, avail, exp):
    """14, 15, 16: Timestamps must satisfy signal <= decision <= available < expires."""
    executor = PaperExecutor()
    order = _make_external_order(
        signal_timestamp=sig,
        decision_timestamp=dec,
        available_at=avail,
        expires_at=exp,
    )

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == "invalid_timestamp_order"
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


@pytest.mark.parametrize(
    "bad_horizon",
    [
        0,     # 17: zero
        -5,    # 18: negative
        True,  # 19: bool True
        False, # bool False
        12.5,  # float
    ],
)
def test_boundary_horizon_invalid_rejected(bad_horizon):
    """17, 18, 19: horizon_s must be strictly positive int and not bool."""
    executor = PaperExecutor()
    order = _make_external_order(horizon_s=bad_horizon)

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == "invalid_horizon"
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


def test_boundary_long_geometry_violations_rejected():
    """20, 21: LONG geometry violations (SL >= ref or TP <= ref) fail closed."""
    executor = PaperExecutor()

    # 20: LONG SL >= reference_price (95000)
    order_bad_sl = _make_external_order(side="LONG", reference_price=95000.0, stop_loss=95001.0, take_profit=98000.0)
    reg_sl, rej_sl = executor.submit_order(order_bad_sl)
    assert reg_sl is None
    assert rej_sl is not None
    assert rej_sl.reason == "INVALID_DECISION"
    assert rej_sl.details == "invalid_stop_loss"

    # 21: LONG TP <= reference_price (95000)
    order_bad_tp = _make_external_order(side="LONG", reference_price=95000.0, stop_loss=94000.0, take_profit=95000.0)
    reg_tp, rej_tp = executor.submit_order(order_bad_tp)
    assert reg_tp is None
    assert rej_tp is not None
    assert rej_tp.reason == "INVALID_DECISION"
    assert rej_tp.details == "invalid_take_profit"


def test_boundary_short_geometry_violations_rejected():
    """22, 23: SHORT geometry violations (SL <= ref or TP >= ref) fail closed."""
    executor = PaperExecutor()

    # 22: SHORT SL <= reference_price (95000)
    order_bad_sl = _make_external_order(side="SHORT", reference_price=95000.0, stop_loss=94999.0, take_profit=90000.0)
    reg_sl, rej_sl = executor.submit_order(order_bad_sl)
    assert reg_sl is None
    assert rej_sl is not None
    assert rej_sl.reason == "INVALID_DECISION"
    assert rej_sl.details == "invalid_stop_loss"

    # 23: SHORT TP >= reference_price (95000)
    order_bad_tp = _make_external_order(side="SHORT", reference_price=95000.0, stop_loss=96000.0, take_profit=95001.0)
    reg_tp, rej_tp = executor.submit_order(order_bad_tp)
    assert reg_tp is None
    assert rej_tp is not None
    assert rej_tp.reason == "INVALID_DECISION"
    assert rej_tp.details == "invalid_take_profit"


@pytest.mark.parametrize(
    "bad_val",
    [float("nan"), float("inf"), -1.0, True, False],
)
def test_boundary_nan_sl_tp_rejected(bad_val):
    """24: NaN/Inf/negative/bool SL or TP fail closed."""
    executor = PaperExecutor()

    order_nan_sl = _make_external_order(side="LONG", reference_price=95000.0, stop_loss=bad_val, take_profit=98000.0)
    reg_sl, rej_sl = executor.submit_order(order_nan_sl)
    assert reg_sl is None
    assert rej_sl is not None
    assert rej_sl.reason == "INVALID_DECISION"
    assert rej_sl.details == "invalid_stop_loss"

    order_nan_tp = _make_external_order(side="LONG", reference_price=95000.0, stop_loss=94000.0, take_profit=bad_val)
    reg_tp, rej_tp = executor.submit_order(order_nan_tp)
    assert reg_tp is None
    assert rej_tp is not None
    assert rej_tp.reason == "INVALID_DECISION"
    assert rej_tp.details == "invalid_take_profit"


@pytest.mark.parametrize(
    "bad_funding",
    [float("nan"), float("inf"), float("-inf"), True, False],
)
def test_boundary_funding_rate_invalid_rejected(bad_funding):
    """25: Non-finite or bool funding_rate_at_decision fails closed."""
    executor = PaperExecutor()
    order = _make_external_order(funding_rate_at_decision=bad_funding)

    reg_order, rej = executor.submit_order(order)

    assert reg_order is None
    assert rej is not None
    assert rej.reason == "INVALID_DECISION"
    assert rej.details == "invalid_funding_rate"
    assert len(executor.pending_orders) == 0
    assert len(executor._pending_symbol_map) == 0


def test_boundary_valid_order_remains_accepted():
    """28: Structurally valid order passes all boundary checks and enters pending."""
    executor = PaperExecutor()
    order = _make_external_order(
        order_id="valid_clean_order_1",
        reference_price=95000.0,
        notional_usdt=2500.0,
        signal_timestamp=1000,
        decision_timestamp=1010,
        available_at=1020,
        expires_at=6020,
        horizon_s=60,
        stop_loss=94000.0,
        take_profit=97000.0,
        funding_rate_at_decision=-0.00015,
    )

    reg_order, rej = executor.submit_order(order)

    assert rej is None
    assert reg_order is order
    assert len(executor.pending_orders) == 1
    assert len(executor._pending_symbol_map) == 1
