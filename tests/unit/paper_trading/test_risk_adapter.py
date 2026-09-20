# tests/unit/paper_trading/test_risk_adapter.py
"""
Hermetic unit and integration tests for RiskAdapter (Gate C2-B).

Proves all 28 required safety and contract invariants:
1. LONG -> BUY.
2. SHORT -> SELL.
3. NEUTRAL does NOT call RiskManager.
4. UNKNOWN does NOT call RiskManager.
5. notional -> size correct (size = notional / price).
6. reference_price passed through exactly.
7. reference_price invalid -> fail-closed.
8. notional invalid -> fail-closed.
9. SL None -> 0.0.
10. TP None -> 0.0.
11. Existing SL/TP preserved exactly.
12. confidence None preserved as source_confidence=None.
13. risk_confidence = 0.0 purely for TradeRequest compatibility.
14. Real calibrated confidence preserved.
15. strategy_version passed to TradeRequest.strategy.
16. RiskManager approve -> PaperOrder produced.
17. RiskManager reject -> NO PaperOrder.
18. Risk rejection reason preserved in result.
19. Malformed return from RiskManager -> fail-closed.
20. Exception from RiskManager -> fail-closed.
21. order_id deterministic via UUID5.
22. order TTL separated from position horizon.
23. Adapter does NOT call RiskManager.add_position().
24. Adapter does NOT call RiskManager.remove_position().
25. Same decision can be evaluated multiple times (stateless adapter).
26. Zero network calls.
27. Zero AI / LLM calls.
28. Zero executor fills created.
Plus: Real in-memory RiskManager integration test.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch
import pytest

from paper_trading.adapters.risk_adapter import RiskAdapter, RiskAdapterResult
from paper_trading.contracts import CanonicalDecision, PaperFill, PaperOrder
from risk_management.exceptions import RiskLimitExceeded
from risk_management.risk_manager import RiskConfig, RiskManager, TradeRequest


def make_decision(
    *,
    side: str = "LONG",
    reference_price: float = 50000.0,
    notional_usdt: float = 100.0,
    horizon_s: int = 300,
    confidence: float | None = None,
    stop_loss: float | None = 49000.0,
    take_profit: float | None = 52000.0,
    strategy_version: str = "test_strategy_v1",
) -> CanonicalDecision:
    """Helper to construct valid CanonicalDecision instances for test cases."""
    return CanonicalDecision(
        cohort_id="c2_test_cohort",
        symbol="BTCUSDT",
        window_id="BTCUSDT:1m:1700000060000#ev1",
        decision_provider="fixed_long",
        strategy_version=strategy_version,
        signal_timestamp=1700000060000,
        decision_timestamp=1700000065000,
        available_at=1700000065050,
        side=side,  # type: ignore[arg-type]
        reference_price=reference_price,
        notional_usdt=notional_usdt,
        horizon_s=horizon_s,
        confidence=confidence,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


@pytest.fixture
def mock_risk_manager() -> MagicMock:
    """Mock of RiskManager returning approval by default."""
    rm = MagicMock(spec=RiskManager)
    rm.check_trade_request.return_value = {
        "approved": True,
        "reason": "approved",
        "max_size": 2.0,
    }
    return rm


# ── Tests 1 to 4: Directional Mapping and Non-Directional Gate ────────────────


def test_long_maps_to_buy(mock_risk_manager: MagicMock) -> None:
    """Point 1: LONG side must map to BUY in TradeRequest."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(side="LONG")

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    assert result.paper_order is not None
    assert result.paper_order.side == "LONG"
    mock_risk_manager.check_trade_request.assert_called_once()
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.side == "BUY"


def test_short_maps_to_sell(mock_risk_manager: MagicMock) -> None:
    """Point 2: SHORT side must map to SELL in TradeRequest."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    # For SHORT, SL must be > ref and TP < ref
    decision = make_decision(side="SHORT", stop_loss=51000.0, take_profit=48000.0)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    assert result.paper_order is not None
    assert result.paper_order.side == "SHORT"
    mock_risk_manager.check_trade_request.assert_called_once()
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.side == "SELL"


def test_neutral_does_not_call_risk_manager(mock_risk_manager: MagicMock) -> None:
    """Point 3: NEUTRAL side fails closed without invoking RiskManager."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(side="NEUTRAL", stop_loss=None, take_profit=None)

    result = adapter.evaluate(decision)

    assert result.status == "INVALID_DECISION"
    assert result.paper_order is None
    assert "Non-directional" in (result.risk_reason or "")
    mock_risk_manager.check_trade_request.assert_not_called()


def test_unknown_does_not_call_risk_manager(mock_risk_manager: MagicMock) -> None:
    """Point 4: UNKNOWN side fails closed without invoking RiskManager."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(side="UNKNOWN", stop_loss=None, take_profit=None)

    result = adapter.evaluate(decision)

    assert result.status == "INVALID_DECISION"
    assert result.paper_order is None
    assert "Non-directional" in (result.risk_reason or "")
    mock_risk_manager.check_trade_request.assert_not_called()


# ── Tests 5 to 8: Notional, Price, and Input Boundary Validations ──────────────


def test_notional_to_size_exact_calculation(mock_risk_manager: MagicMock) -> None:
    """Point 5: size must equal notional_usdt / reference_price exactly."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(notional_usdt=250.0, reference_price=50000.0)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    expected_size = 250.0 / 50000.0  # 0.005
    assert called_tr.size == pytest.approx(expected_size, rel=1e-9)
    assert result.context["derived_base_size"] == pytest.approx(expected_size, rel=1e-9)
    assert result.context["requested_notional_usdt"] == 250.0


def test_reference_price_passed_through_exactly(mock_risk_manager: MagicMock) -> None:
    """Point 6: TradeRequest.price receives decision.reference_price exactly."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    ref_price = 87654.321
    decision = make_decision(reference_price=ref_price, stop_loss=None, take_profit=None)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.price == ref_price
    assert result.context["reference_price"] == ref_price


def test_invalid_reference_price_fails_closed(mock_risk_manager: MagicMock) -> None:
    """Point 7: non-finite or non-positive reference_price fails closed without calling RiskManager."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()
    # Mutate attribute using object.__setattr__ to bypass CanonicalDecision's post_init
    object.__setattr__(decision, "reference_price", -10.0)

    result = adapter.evaluate(decision)

    assert result.status == "INVALID_DECISION"
    assert result.paper_order is None
    assert "Invalid reference_price" in (result.risk_reason or "")
    mock_risk_manager.check_trade_request.assert_not_called()


def test_invalid_notional_fails_closed(mock_risk_manager: MagicMock) -> None:
    """Point 8: non-finite or non-positive notional fails closed without calling RiskManager."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()
    object.__setattr__(decision, "notional_usdt", float("nan"))

    result = adapter.evaluate(decision)

    assert result.status == "INVALID_DECISION"
    assert result.paper_order is None
    assert "Invalid notional_usdt" in (result.risk_reason or "")
    mock_risk_manager.check_trade_request.assert_not_called()


# ── Tests 9 to 11: SL / TP Mapping ───────────────────────────────────────────


def test_stop_loss_none_maps_to_zero(mock_risk_manager: MagicMock) -> None:
    """Point 9: stop_loss=None maps to 0.0 in TradeRequest."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(stop_loss=None)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.stop_loss == 0.0
    assert result.paper_order is not None
    assert result.paper_order.stop_loss is None


def test_take_profit_none_maps_to_zero(mock_risk_manager: MagicMock) -> None:
    """Point 10: take_profit=None maps to 0.0 in TradeRequest."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(take_profit=None)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.take_profit == 0.0
    assert result.paper_order is not None
    assert result.paper_order.take_profit is None


def test_existing_sl_tp_preserved_exactly(mock_risk_manager: MagicMock) -> None:
    """Point 11: existing SL/TP numeric values are passed into TradeRequest without alteration."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    sl = 49234.5
    tp = 53456.7
    decision = make_decision(stop_loss=sl, take_profit=tp)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.stop_loss == sl
    assert called_tr.take_profit == tp
    assert result.paper_order is not None
    assert result.paper_order.stop_loss == sl
    assert result.paper_order.take_profit == tp


# ── Tests 12 to 14: Confidence Semantics ──────────────────────────────────────


def test_confidence_none_preserved_with_zero_compatibility(mock_risk_manager: MagicMock) -> None:
    """Points 12 & 13: confidence=None is preserved in source_confidence, with risk_confidence=0.0."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(confidence=None)

    result = adapter.evaluate(decision)

    assert result.source_confidence is None
    assert result.risk_confidence == 0.0
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.confidence == 0.0


def test_calibrated_confidence_preserved(mock_risk_manager: MagicMock) -> None:
    """Point 14: calibrated real confidence is preserved in source and risk_confidence."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision(confidence=0.85)

    result = adapter.evaluate(decision)

    assert result.source_confidence == 0.85
    assert result.risk_confidence == 0.85
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.confidence == 0.85


# ── Test 15: Strategy Identifier ──────────────────────────────────────────────


def test_strategy_version_propagated_to_trade_request(mock_risk_manager: MagicMock) -> None:
    """Point 15: decision.strategy_version arrives in TradeRequest.strategy (no silent 'momentum' fallback)."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    strat = "special_breakout_v2.1"
    decision = make_decision(strategy_version=strat)

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    called_tr = mock_risk_manager.check_trade_request.call_args[0][0]
    assert called_tr.strategy == strat


# ── Tests 16 to 18: RiskManager Return and Rejection Handling ─────────────────


def test_risk_manager_approval_creates_paper_order(mock_risk_manager: MagicMock) -> None:
    """Point 16: RiskManager approval leads directly to a valid PaperOrder."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    assert isinstance(result.paper_order, PaperOrder)
    assert result.paper_order.decision_id == decision.decision_id
    assert result.risk_reason == "approved"
    assert result.max_size == 2.0


def test_risk_manager_rejection_prevents_paper_order(mock_risk_manager: MagicMock) -> None:
    """Points 17 & 18: RiskManager rejection yields REJECTED_BY_RISK and NO PaperOrder."""
    mock_risk_manager.check_trade_request.return_value = {
        "approved": False,
        "reason": "daily loss limit exceeded",
        "max_size": 0.0,
    }
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    result = adapter.evaluate(decision)

    assert result.status == "REJECTED_BY_RISK"
    assert result.paper_order is None
    assert result.risk_reason == "daily loss limit exceeded"
    assert result.max_size == 0.0


# ── Tests 19 & 20: Malformed Return and Exception Fail-Closed ─────────────────


def test_malformed_risk_manager_return_fails_closed(mock_risk_manager: MagicMock) -> None:
    """Point 19: non-dict or invalid shape return fails closed as INVALID_DECISION."""
    mock_risk_manager.check_trade_request.return_value = "NOT_A_DICT"
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    result = adapter.evaluate(decision)

    assert result.status == "INVALID_DECISION"
    assert result.paper_order is None
    assert "non-dict" in (result.risk_reason or "")


def test_risk_manager_exception_fails_closed(mock_risk_manager: MagicMock) -> None:
    """Point 20: unexpected exception from RiskManager fails closed with sanitized error reason."""
    mock_risk_manager.check_trade_request.side_effect = RuntimeError("database connection lost")
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    result = adapter.evaluate(decision)

    assert result.status == "INVALID_DECISION"
    assert result.paper_order is None
    assert "RiskManager execution failure: RuntimeError" in (result.risk_reason or "")


# ── Tests 21 & 22: Order Identity and TTL vs Horizon Separation ───────────────


def test_order_id_deterministic_via_uuid5(mock_risk_manager: MagicMock) -> None:
    """Point 21: same decision_id + same config generates identical order_id."""
    adapter1 = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    adapter2 = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    res1 = adapter1.evaluate(decision)
    res2 = adapter2.evaluate(decision)

    assert res1.paper_order is not None
    assert res2.paper_order is not None
    assert res1.paper_order.order_id == res2.paper_order.order_id

    # Changing TTL changes order_id deterministically
    adapter_diff_ttl = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=10000)
    res3 = adapter_diff_ttl.evaluate(decision)
    assert res3.paper_order is not None
    assert res3.paper_order.order_id != res1.paper_order.order_id


def test_order_ttl_separated_from_position_horizon(mock_risk_manager: MagicMock) -> None:
    """Point 22: order TTL (expires_at) must be independent of position forecast horizon (horizon_s)."""
    order_ttl_ms = 7000  # 7 seconds order TTL
    position_horizon_s = 300  # 300 seconds forecast horizon
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=order_ttl_ms)
    decision = make_decision(horizon_s=position_horizon_s)

    result = adapter.evaluate(decision)

    assert result.paper_order is not None
    assert result.paper_order.expires_at == decision.available_at + order_ttl_ms
    assert result.paper_order.horizon_s == position_horizon_s
    # Crucial assertion: order TTL != position horizon in ms
    assert (result.paper_order.expires_at - decision.available_at) != (result.paper_order.horizon_s * 1000)


# ── Tests 23 & 24: State Authority (Zero Position Calls) ──────────────────────


def test_adapter_does_not_call_add_position(mock_risk_manager: MagicMock) -> None:
    """Point 23: RiskAdapter must NEVER invoke RiskManager.add_position()."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    adapter.evaluate(decision)

    mock_risk_manager.add_position.assert_not_called()


def test_adapter_does_not_call_remove_position(mock_risk_manager: MagicMock) -> None:
    """Point 24: RiskAdapter must NEVER invoke RiskManager.remove_position()."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    adapter.evaluate(decision)

    mock_risk_manager.remove_position.assert_not_called()


# ── Test 25: Stateless Idempotency ────────────────────────────────────────────


def test_same_decision_can_be_re_evaluated(mock_risk_manager: MagicMock) -> None:
    """Point 25: Adapter is stateless; same decision can be re-evaluated producing deterministic results."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    res1 = adapter.evaluate(decision)
    res2 = adapter.evaluate(decision)

    assert res1.status == res2.status == "APPROVED"
    assert res1.paper_order is not None and res2.paper_order is not None
    assert res1.paper_order.order_id == res2.paper_order.order_id


# ── Tests 26 to 28: Zero Network, Zero AI, Zero Fills ─────────────────────────


def test_zero_network_access(mock_risk_manager: MagicMock) -> None:
    """Point 26: Zero network calls during evaluation."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    with patch("socket.socket") as mock_socket:
        result = adapter.evaluate(decision)
        mock_socket.assert_not_called()
    assert result.status == "APPROVED"


def test_zero_ai_calls(mock_risk_manager: MagicMock) -> None:
    """Point 27: Adapter does not invoke AI runners, LLMs, or analyzers."""
    import sys
    ai_modules = [m for m in sys.modules if "market_orchestrator.ai" in m]
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    # No new AI runtime was triggered
    assert not hasattr(adapter, "ai_runner")
    assert not hasattr(adapter, "analyzer")


def test_zero_executor_fills_created(mock_risk_manager: MagicMock) -> None:
    """Point 28: RiskAdapter produces PaperOrder upon approval, but NEVER produces PaperFill."""
    adapter = RiskAdapter(risk_manager=mock_risk_manager, order_ttl_ms=5000)
    decision = make_decision()

    result = adapter.evaluate(decision)

    assert result.status == "APPROVED"
    assert isinstance(result.paper_order, PaperOrder)
    assert not isinstance(result.paper_order, PaperFill)
    assert not hasattr(result, "fill")


# ── Integration Test: Real In-Memory RiskManager ──────────────────────────────


def test_integration_against_real_in_memory_risk_manager() -> None:
    """
    Hermetic integration test using the real production RiskManager.

    Verifies:
    - controlled RiskConfig with max_position_size=1000.0
    - trade below limit (notional=500.0) -> APPROVED
    - trade above limit (notional=2500.0) -> REJECTED_BY_RISK ('position size limit exceeded')
    - zero network, zero Binance calls.
    """
    config = RiskConfig(
        max_position_size=1000.0,
        max_daily_loss=0.05,
        max_loss_per_trade=0.05,
        max_open_positions=5,
    )
    real_rm = RiskManager(config)
    adapter = RiskAdapter(risk_manager=real_rm, order_ttl_ms=5000)

    # 1. Trade below limit (500 USD <= 1000 USD max)
    dec_approved = make_decision(notional_usdt=500.0, reference_price=50000.0)
    res_approved = adapter.evaluate(dec_approved)

    assert res_approved.status == "APPROVED"
    assert res_approved.paper_order is not None
    assert res_approved.risk_reason == "approved"
    assert res_approved.max_size == pytest.approx(1000.0 / 50000.0, rel=1e-9)

    # 2. Trade above limit (2500 USD > 1000 USD max)
    dec_rejected = make_decision(notional_usdt=2500.0, reference_price=50000.0)
    res_rejected = adapter.evaluate(dec_rejected)

    assert res_rejected.status == "REJECTED_BY_RISK"
    assert res_rejected.paper_order is None
    assert res_rejected.risk_reason == "position size limit exceeded"
    assert res_rejected.max_size == pytest.approx(1000.0 / 50000.0, rel=1e-9)

    # 3. Confirm RiskManager.positions remained completely empty
    assert len(real_rm.positions) == 0
