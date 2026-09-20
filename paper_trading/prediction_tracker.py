# paper_trading/prediction_tracker.py
"""
Canonical Prediction Tracker (Gate D0-B).

In-memory causal tracker for directional predictions.
Completely independent of economic trade executions, order status, fees, or slippage.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from paper_trading.contracts import CanonicalDecision
from paper_trading.prediction import (
    PendingPrediction,
    PredictionOutcome,
    PredictionResult,
    PredictionTrackerConfig,
    make_prediction_id,
)


def _extract_tick_fields(tick: Any) -> Tuple[int, float, str]:
    """
    Extract (event_timestamp, price, symbol) from execution tick dict or object.
    """
    if isinstance(tick, dict):
        raw_ts = tick.get("T") or tick.get("event_timestamp") or tick.get("trade_time_ms") or 0
        raw_p = tick.get("p") or tick.get("price") or 0.0
        raw_s = tick.get("s") or tick.get("symbol") or ""
        return int(raw_ts), float(raw_p), str(raw_s)

    # Dataclass or object attributes
    raw_ts = getattr(tick, "event_timestamp", None)
    if raw_ts is None:
        raw_ts = getattr(tick, "timestamp_ms", getattr(tick, "T", 0))

    raw_p = getattr(tick, "price", None)
    if raw_p is None:
        raw_p = getattr(tick, "p", 0.0)

    raw_s = getattr(tick, "symbol", None)
    if raw_s is None:
        raw_s = getattr(tick, "s", "")

    return int(raw_ts or 0), float(raw_p or 0.0), str(raw_s or "")


class PredictionTracker:
    """
    In-memory tracking and resolution of canonical directional predictions.

    Features:
      - O(N_pending) resolution per tick, zero SQLite I/O on ordinary ticks.
      - Deterministic prediction IDs per policy version.
      - Idempotent registration and resolution.
      - Strictly causal: first eligible tick >= deadline within tolerance resolves.
      - Late ticks outside tolerance mark UNRESOLVED without adopting late price.
      - Graceful shutdown flushes remaining pendings as UNRESOLVED (PROCESS_SHUTDOWN).
    """

    def __init__(
        self,
        config: Optional[PredictionTrackerConfig] = None,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self.config = config or PredictionTrackerConfig()
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._pending: Dict[str, PendingPrediction] = {}
        self._resolved: Dict[str, PredictionOutcome] = {}
        self._lock = threading.Lock()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def resolved_count(self) -> int:
        with self._lock:
            return len(self._resolved)

    def register(self, decision: CanonicalDecision) -> Optional[str]:
        """
        Register a new directional prediction from a CanonicalDecision.

        Evaluates ONLY directional decisions (LONG or SHORT).
        Returns prediction_id, or None if signal is non-directional (NEUTRAL/UNKNOWN).
        Idempotent: returns existing prediction_id if already registered.
        """
        if decision.side not in ("LONG", "SHORT"):
            return None

        if not math.isfinite(decision.reference_price) or decision.reference_price <= 0:
            raise ValueError(
                f"decision.reference_price must be finite > 0, got {decision.reference_price}"
            )

        horizon_s = decision.horizon_s if decision.horizon_s > 0 else 300
        deadline_ms = decision.decision_timestamp + (horizon_s * 1000)
        pred_id = make_prediction_id(
            decision_id=decision.decision_id,
            horizon_s=horizon_s,
            policy_version=self.config.policy_version,
        )

        with self._lock:
            if pred_id in self._pending or pred_id in self._resolved:
                return pred_id

            pending = PendingPrediction(
                prediction_id=pred_id,
                decision_id=decision.decision_id,
                cohort_id=decision.cohort_id,
                symbol=decision.symbol,
                side=decision.side,
                reference_price=float(decision.reference_price),
                decision_timestamp=decision.decision_timestamp,
                horizon_s=horizon_s,
                deadline_ms=deadline_ms,
                flat_tolerance_bps=self.config.flat_tolerance_bps,
                resolution_tolerance_ms=self.config.resolution_tolerance_ms,
                policy_version=self.config.policy_version,
            )
            self._pending[pred_id] = pending

        return pred_id

    def on_tick(self, tick: Any) -> List[PredictionOutcome]:
        """
        Evaluate causal tick against all pending predictions for the tick's symbol.

        Rules:
          A) T < deadline:
             Keep pending.
          B) deadline <= T <= deadline + resolution_tolerance_ms:
             First tick resolves to CORRECT, INCORRECT, or FLAT.
          C) T > deadline + resolution_tolerance_ms:
             Late tick marks UNRESOLVED with reason NO_TICK_WITHIN_TOLERANCE.
             Does NOT adopt late tick price as resolution_price.
        """
        event_ts, price, symbol = _extract_tick_fields(tick)
        if event_ts <= 0 or not math.isfinite(price) or price <= 0:
            return []

        outcomes: List[PredictionOutcome] = []

        with self._lock:
            to_remove: List[str] = []

            for pred_id, pending in self._pending.items():
                if symbol and pending.symbol and symbol != pending.symbol:
                    continue

                # Case A: Tick before deadline -> not eligible for resolution
                if event_ts < pending.deadline_ms:
                    continue

                # Case B: First tick inside tolerance window [deadline, deadline + tolerance]
                if event_ts <= pending.deadline_ms + pending.resolution_tolerance_ms:
                    resolution_price = price
                    drift_ms = event_ts - pending.deadline_ms
                    raw_ret_bps = ((resolution_price - pending.reference_price) / pending.reference_price) * 10_000.0

                    if pending.side == "LONG":
                        dir_ret_bps = raw_ret_bps
                    else:
                        dir_ret_bps = -raw_ret_bps

                    # Flat check
                    if abs(raw_ret_bps) <= pending.flat_tolerance_bps:
                        res: PredictionResult = "FLAT"
                    elif dir_ret_bps > 0:
                        res = "CORRECT"
                    else:
                        res = "INCORRECT"

                    outcome = PredictionOutcome(
                        prediction_id=pending.prediction_id,
                        decision_id=pending.decision_id,
                        cohort_id=pending.cohort_id,
                        symbol=pending.symbol,
                        side=pending.side,
                        reference_price=pending.reference_price,
                        decision_timestamp=pending.decision_timestamp,
                        horizon_s=pending.horizon_s,
                        deadline_ms=pending.deadline_ms,
                        result=res,
                        reason="HORIZON_RESOLVED",
                        resolution_price=resolution_price,
                        raw_return_bps=raw_ret_bps,
                        directional_return_bps=dir_ret_bps,
                        resolved_timestamp_ms=event_ts,
                        resolution_drift_ms=drift_ms,
                        observed_at_ms=event_ts,
                        flat_tolerance_bps=pending.flat_tolerance_bps,
                        resolution_tolerance_ms=pending.resolution_tolerance_ms,
                        policy_version=pending.policy_version,
                        created_at_ms=self.clock_ms(),
                    )
                    outcomes.append(outcome)
                    to_remove.append(pred_id)

                # Case C: Tick arrived after tolerance deadline -> fail closed UNRESOLVED
                else:
                    outcome = PredictionOutcome(
                        prediction_id=pending.prediction_id,
                        decision_id=pending.decision_id,
                        cohort_id=pending.cohort_id,
                        symbol=pending.symbol,
                        side=pending.side,
                        reference_price=pending.reference_price,
                        decision_timestamp=pending.decision_timestamp,
                        horizon_s=pending.horizon_s,
                        deadline_ms=pending.deadline_ms,
                        result="UNRESOLVED",
                        reason="NO_TICK_WITHIN_TOLERANCE",
                        resolution_price=None,
                        raw_return_bps=None,
                        directional_return_bps=None,
                        resolved_timestamp_ms=None,
                        resolution_drift_ms=None,
                        observed_at_ms=event_ts,
                        flat_tolerance_bps=pending.flat_tolerance_bps,
                        resolution_tolerance_ms=pending.resolution_tolerance_ms,
                        policy_version=pending.policy_version,
                        created_at_ms=self.clock_ms(),
                    )
                    outcomes.append(outcome)
                    to_remove.append(pred_id)

            for pid in to_remove:
                del self._pending[pid]

            for out in outcomes:
                self._resolved[out.prediction_id] = out

        return outcomes

    def flush_unresolved(
        self,
        reason: str = "PROCESS_SHUTDOWN",
        observed_at_ms: Optional[int] = None,
    ) -> List[PredictionOutcome]:
        """
        Flush all remaining pending predictions as UNRESOLVED.

        Invoked on graceful shutdown or explicit pipeline termination.
        Never synthesizes artificial prices or returns.
        """
        now_ms = observed_at_ms if observed_at_ms is not None else self.clock_ms()
        outcomes: List[PredictionOutcome] = []

        with self._lock:
            for pred_id, pending in list(self._pending.items()):
                outcome = PredictionOutcome(
                    prediction_id=pending.prediction_id,
                    decision_id=pending.decision_id,
                    cohort_id=pending.cohort_id,
                    symbol=pending.symbol,
                    side=pending.side,
                    reference_price=pending.reference_price,
                    decision_timestamp=pending.decision_timestamp,
                    horizon_s=pending.horizon_s,
                    deadline_ms=pending.deadline_ms,
                    result="UNRESOLVED",
                    reason=reason,
                    resolution_price=None,
                    raw_return_bps=None,
                    directional_return_bps=None,
                    resolved_timestamp_ms=None,
                    resolution_drift_ms=None,
                    observed_at_ms=now_ms,
                    flat_tolerance_bps=pending.flat_tolerance_bps,
                    resolution_tolerance_ms=pending.resolution_tolerance_ms,
                    policy_version=pending.policy_version,
                    created_at_ms=self.clock_ms(),
                )
                outcomes.append(outcome)
                self._resolved[pred_id] = outcome

            self._pending.clear()

        return outcomes

    def get_pending_predictions(self) -> List[PendingPrediction]:
        with self._lock:
            return list(self._pending.values())

    def get_resolved_outcomes(self) -> List[PredictionOutcome]:
        with self._lock:
            return list(self._resolved.values())
