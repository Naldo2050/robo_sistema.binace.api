# paper_trading/adapters/risk_adapter.py
"""
Hermetic Risk Adapter bridging CanonicalDecision to RiskManager and PaperOrder.

ARCHITECTURAL STATE AUTHORITY NOTE (GATE C2-B):
------------------------------------------------
The `max_open_positions` rule based on `RiskManager.positions` is NOT integrated
with the actual simulated state of paper trading in this Gate, and therefore
MUST NOT be advertised as an active operational protection.
In this Gate:
- RiskManager acts solely as a pre-order rule validator via `check_trade_request()`.
- PaperExecutor and PositionManager remain the future authority over paper positions.
- Neither `RiskManager.add_position()` nor `RiskManager.remove_position()` are invoked by this adapter.
This separation will be resolved in Gate C3/C4 by explicit state coordination.

CONFIDENCE SEMANTIC NOTE:
-------------------------
TradeRequest requires `confidence: float`. If `decision.confidence` is None,
`risk_confidence` is mapped to 0.0 solely for compatibility with the existing
TradeRequest schema. This 0.0 MUST NOT be interpreted as observed or calibrated
confidence; the original uncalibrated status is preserved in `source_confidence=None`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
from typing import Any, Dict, Literal, Optional
import uuid

from paper_trading.contracts import CanonicalDecision, PaperOrder, SignalSide
from risk_management.risk_manager import RiskManager, TradeRequest

RiskStatus = Literal["APPROVED", "REJECTED_BY_RISK", "INVALID_DECISION"]


@dataclass(frozen=True)
class RiskAdapterResult:
    """
    Result of evaluating a CanonicalDecision through RiskManager rules.

    Preserves source confidence independently from TradeRequest compatibility coercion.
    """

    status: RiskStatus
    decision_id: str
    paper_order: Optional[PaperOrder]
    risk_reason: Optional[str]
    max_size: Optional[float]
    source_confidence: Optional[float]
    risk_confidence: float
    context: Dict[str, Any] = field(default_factory=dict)


class RiskAdapter:
    """
    Thread-safe, hermetic adapter bridging CanonicalDecision to RiskManager.check_trade_request().

    Converts valid directional decisions into simulated pending PaperOrders upon risk approval.
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        order_ttl_ms: int = 5_000,
    ) -> None:
        if not (isinstance(order_ttl_ms, int) and order_ttl_ms > 0):
            raise ValueError(f"order_ttl_ms must be a positive integer, got {order_ttl_ms}")
        self.risk_manager = risk_manager
        self.order_ttl_ms = order_ttl_ms
        self._lock = threading.RLock()

    def evaluate(self, decision: CanonicalDecision) -> RiskAdapterResult:
        """
        Evaluate a CanonicalDecision through RiskManager.check_trade_request().

        Fails closed on any contract violation, malformed response, or unexpected exception.
        """
        # Resolve confidence values up front
        source_confidence = decision.confidence
        risk_confidence = 0.0 if source_confidence is None else float(source_confidence)

        # 1. Direction check: only LONG or SHORT are eligible for order creation
        if decision.side not in ("LONG", "SHORT"):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=f"Non-directional side '{decision.side}' cannot be converted to TradeRequest",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={
                    "rejection_cause": "NON_ACTIONABLE_SIDE",
                    "side": decision.side,
                },
            )

        # 2. Input validation: reference_price, notional_usdt, horizon_s
        if not (math.isfinite(decision.reference_price) and decision.reference_price > 0):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=f"Invalid reference_price: {decision.reference_price}",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={"rejection_cause": "INVALID_REFERENCE_PRICE"},
            )

        if not (math.isfinite(decision.notional_usdt) and decision.notional_usdt > 0):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=f"Invalid notional_usdt: {decision.notional_usdt}",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={"rejection_cause": "INVALID_NOTIONAL_USDT"},
            )

        if not (isinstance(decision.horizon_s, int) and decision.horizon_s > 0):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=f"Invalid horizon_s: {decision.horizon_s}",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={"rejection_cause": "INVALID_HORIZON_S"},
            )

        # 3. Notional -> Base size derivation
        derived_base_size = decision.notional_usdt / decision.reference_price
        if not (math.isfinite(derived_base_size) and derived_base_size > 0):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=f"Derived base size is non-finite or non-positive: {derived_base_size}",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={"rejection_cause": "INVALID_DERIVED_SIZE"},
            )

        # 4. Strict Side mapping: LONG -> BUY, SHORT -> SELL
        trade_side = "BUY" if decision.side == "LONG" else "SELL"

        # 5. SL / TP mapping: None -> 0.0, float -> float
        stop_loss = 0.0 if decision.stop_loss is None else float(decision.stop_loss)
        take_profit = 0.0 if decision.take_profit is None else float(decision.take_profit)

        # 6. Build TradeRequest
        trade_request = TradeRequest(
            symbol=decision.symbol,
            side=trade_side,
            size=derived_base_size,
            price=decision.reference_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=decision.strategy_version,
            confidence=risk_confidence,
        )

        base_context: Dict[str, Any] = {
            "requested_notional_usdt": decision.notional_usdt,
            "derived_base_size": derived_base_size,
            "reference_price": decision.reference_price,
            "order_ttl_ms": self.order_ttl_ms,
            "position_horizon_s": decision.horizon_s,
        }

        # 7. Thread-safe call to RiskManager.check_trade_request()
        try:
            with self._lock:
                raw_result = self.risk_manager.check_trade_request(trade_request)
        except Exception as exc:
            # Fail closed on any unhandled exception in RiskManager
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=f"RiskManager execution failure: {type(exc).__name__}",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={**base_context, "rejection_cause": "RISK_MANAGER_EXCEPTION"},
            )

        # 8. Defensive shape validation of check_trade_request return
        if not isinstance(raw_result, dict):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason="RiskManager returned non-dict response",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={**base_context, "rejection_cause": "MALFORMED_RISK_RESPONSE"},
            )

        if "approved" not in raw_result or not isinstance(raw_result["approved"], bool):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason="RiskManager response missing boolean 'approved' field",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={**base_context, "rejection_cause": "MALFORMED_RISK_RESPONSE"},
            )

        if "reason" not in raw_result or not isinstance(raw_result["reason"], str):
            return RiskAdapterResult(
                status="INVALID_DECISION",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason="RiskManager response missing string 'reason' field",
                max_size=None,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={**base_context, "rejection_cause": "MALFORMED_RISK_RESPONSE"},
            )

        raw_max_size = raw_result.get("max_size")
        parsed_max_size: Optional[float] = None
        if raw_max_size is not None:
            if isinstance(raw_max_size, (int, float)) and math.isfinite(raw_max_size):
                parsed_max_size = float(raw_max_size)
            else:
                return RiskAdapterResult(
                    status="INVALID_DECISION",
                    decision_id=decision.decision_id,
                    paper_order=None,
                    risk_reason="RiskManager returned non-finite or invalid 'max_size'",
                    max_size=None,
                    source_confidence=source_confidence,
                    risk_confidence=risk_confidence,
                    context={**base_context, "rejection_cause": "MALFORMED_RISK_RESPONSE"},
                )

        # 9. Handle Risk Rejection
        if not raw_result["approved"]:
            return RiskAdapterResult(
                status="REJECTED_BY_RISK",
                decision_id=decision.decision_id,
                paper_order=None,
                risk_reason=raw_result["reason"],
                max_size=parsed_max_size,
                source_confidence=source_confidence,
                risk_confidence=risk_confidence,
                context={**base_context, "risk_rejection_reason": raw_result["reason"]},
            )

        # 10. Handle Risk Approval -> Construct deterministic PaperOrder
        expires_at = decision.available_at + self.order_ttl_ms
        order_identity = f"{decision.decision_id}:order:{self.order_ttl_ms}"
        order_id = str(uuid.uuid5(uuid.NAMESPACE_OID, order_identity))

        paper_order = PaperOrder(
            order_id=order_id,
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
            expires_at=expires_at,
            horizon_s=decision.horizon_s,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            funding_rate_at_decision=decision.funding_rate_at_decision,
            funding_rate_source=decision.funding_rate_source,
            context=dict(decision.context),
        )

        return RiskAdapterResult(
            status="APPROVED",
            decision_id=decision.decision_id,
            paper_order=paper_order,
            risk_reason=raw_result["reason"],
            max_size=parsed_max_size,
            source_confidence=source_confidence,
            risk_confidence=risk_confidence,
            context={**base_context, "order_id": order_id},
        )

    # Alias for flexibility
    evaluate_decision = evaluate
