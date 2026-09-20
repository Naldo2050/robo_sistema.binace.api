# paper_trading/shadow_runtime.py
"""
Hermetic Shadow Paper Trading Runtime (Gate C3-C-B1).

Coordinates SignalDecisionAdapter, RiskAdapter, and ExecutionSink in-process.
Zero network access, zero AI coupling, zero real order capability, fail-closed boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Dict, Literal, Optional

from paper_trading.config import ShadowPaperConfig, ShadowProviderType
from paper_trading.cost_model import CostModel
from paper_trading.decision_providers import (
    DecisionProvider,
    FixedDecisionProvider,
    RandomDecisionProvider,
)
from paper_trading.adapters.signal_adapter import SignalDecisionAdapter
from paper_trading.adapters.risk_adapter import RiskAdapter
from paper_trading.execution_sink import ExecutionSink
from paper_trading.executor import PaperExecutor
from paper_trading.prediction import PredictionTrackerConfig
from paper_trading.prediction_tracker import PredictionTracker
from risk_management.risk_manager import RiskConfig, RiskManager

SignalResultStatus = Literal[
    "INACTIVE",
    "SKIPPED",
    "RISK_REJECTED",
    "ORDER_SUBMITTED",
    "ORDER_REJECTED",
    "ERROR",
]

TradeResultStatus = Literal[
    "INACTIVE",
    "PROCESSED",
    "ERROR",
]


@dataclass(frozen=True)
class ShadowSignalResult:
    """Explicit structured result of processing an incoming market signal in shadow mode."""

    status: SignalResultStatus
    skip_reason: Optional[str] = None
    rejection_reason: Optional[str] = None
    order_id: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ShadowTradeResult:
    """Explicit structured result of processing a market tick trade in shadow mode."""

    status: TradeResultStatus
    accepted: bool = False
    fills_count: int = 0
    closed_trades_count: int = 0
    rejections_count: int = 0
    error: Optional[str] = None


def create_decision_provider(
    provider_type: Optional[ShadowProviderType],
    random_seed: Optional[int] = None,
) -> DecisionProvider:
    """
    Hermetic provider factory using an explicit static allowlist.

    Prohibits dynamic imports, eval, or runtime code loading.
    """
    if provider_type == "fixed_long":
        return FixedDecisionProvider("LONG")
    if provider_type == "fixed_short":
        return FixedDecisionProvider("SHORT")
    if provider_type == "seeded_random":
        if random_seed is None or isinstance(random_seed, bool):
            raise ValueError(f"seeded_random provider requires an explicit integer seed, got: {random_seed!r}")
        return RandomDecisionProvider(random_seed)
    raise ValueError(f"Unsupported provider type: {provider_type!r}. Must be one of 'fixed_long', 'fixed_short', 'seeded_random'")


class ShadowPaperRuntime:
    """
    Self-contained, in-process runtime for shadow paper trading.

    Encapsulates:
      Signal Event
           ↓
      SignalDecisionAdapter
           ↓
      RiskAdapter (RiskManager)
           ↓
      ExecutionSink (PaperExecutor)
           ↓
      In-Memory State & Metrics & Optional Auditable PaperLedger
    """

    def __init__(
        self,
        config: ShadowPaperConfig,
        clock_ms: Optional[Callable[[], int]] = None,
        risk_manager: Optional[RiskManager] = None,
        risk_adapter: Optional[RiskAdapter] = None,
        execution_sink: Optional[ExecutionSink] = None,
        signal_adapter: Optional[SignalDecisionAdapter] = None,
        prediction_tracker: Optional[PredictionTracker] = None,
        ledger: Optional[Any] = None,
        git_sha: Optional[str] = None,
        ledger_is_owner: bool = False,
    ) -> None:
        self.config = config
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.ledger = ledger
        self._git_sha = git_sha or "HEAD"
        self._ledger_is_owner = ledger_is_owner
        self._lock = threading.RLock()

        # Lifecycle flags
        self._active: bool = bool(config.enabled)
        self._accept_new_exposure: bool = bool(config.enabled)
        self._persistence_failed: bool = False

        # In-memory operational counters
        self._counters: Dict[str, int] = {
            "signals_seen": 0,
            "directional_decisions": 0,
            "nondirectional_skips": 0,
            "invalid_signals": 0,
            "risk_approved": 0,
            "risk_rejected": 0,
            "orders_submitted": 0,
            "order_rejected": 0,
            "ticks_seen": 0,
            "fills": 0,
            "closed_trades": 0,
            "runtime_errors": 0,
            "persistence_blocks": 0,
            "persistence_errors": 0,
            "observations_enqueued": 0,
            "decisions_enqueued": 0,
            "risk_evaluations_enqueued": 0,
            "orders_enqueued": 0,
            "fills_enqueued": 0,
            "positions_enqueued": 0,
            "closed_trades_enqueued": 0,
            "predictions_registered": 0,
            "predictions_resolved": 0,
            "predictions_unresolved": 0,
            "prediction_outcomes_enqueued": 0,
        }

        # Build / wire components if enabled
        if config.enabled:
            # 1. Decision Provider
            self.provider = create_decision_provider(config.provider, config.random_seed)

            # 2. Signal Decision Adapter
            if signal_adapter is not None:
                self.signal_adapter = signal_adapter
            else:
                self.signal_adapter = SignalDecisionAdapter(
                    cohort_id=config.cohort_id or "CH_SHADOW_DEFAULT",
                    strategy_version=config.strategy_version,
                    mode="BASELINE",
                    provider=self.provider,
                    timeframe=config.timeframe,
                    default_notional_usdt=config.notional_usdt or 1000.0,
                    default_horizon_s=config.horizon_s or 300,
                    clock_ms=self.clock_ms,
                )

            # 3. Risk Adapter
            if risk_adapter is not None:
                self.risk_adapter = risk_adapter
            else:
                rm = risk_manager or RiskManager(RiskConfig())
                self.risk_adapter = RiskAdapter(
                    risk_manager=rm,
                    order_ttl_ms=config.order_ttl_ms or 5000,
                )

            # 4. Execution Sink
            if execution_sink is not None:
                self.execution_sink = execution_sink
            else:
                cost_model = CostModel(config.to_cost_config())
                executor = PaperExecutor(
                    cost_model=cost_model,
                )
                self.execution_sink = ExecutionSink(
                    symbol=config.symbol,
                    executor=executor,
                )

            # 5. Prediction Tracker
            if prediction_tracker is not None:
                self.prediction_tracker = prediction_tracker
            else:
                self.prediction_tracker = PredictionTracker(
                    config=PredictionTrackerConfig(
                        resolution_tolerance_ms=30_000,
                        flat_tolerance_bps=1.0,
                        policy_version="v1",
                    ),
                    clock_ms=self.clock_ms,
                )
        else:
            # Inactive stubs when disabled
            self.provider = None  # type: ignore[assignment]
            self.signal_adapter = None  # type: ignore[assignment]
            self.risk_adapter = None  # type: ignore[assignment]
            self.execution_sink = None  # type: ignore[assignment]
            self.prediction_tracker = None  # type: ignore[assignment]

    @property
    def is_active(self) -> bool:
        """Indicate whether the shadow runtime is actively processing events."""
        with self._lock:
            return self._active and self.config.enabled

    @property
    def accepts_new_exposure(self) -> bool:
        """Indicate whether the shadow runtime accepts new signals and risk evaluations."""
        with self._lock:
            return self._active and self._accept_new_exposure and self.config.enabled

    def start(self) -> bool:
        """
        Explicit lifecycle start for shadow runtime.

        When ledger is configured:
          1. Verify persistence health.
          2. Attempt strict cohort creation (fail-closed on duplicates).
          3. Record STARTED cohort event.
          4. Enable runtime operations for new exposures.
        """
        with self._lock:
            if not self.config.enabled:
                self._active = False
                self._accept_new_exposure = False
                return False

            if self.ledger is not None:
                # 1. Check health
                health = self.ledger.health_snapshot()
                if not health["healthy"]:
                    self._counters["persistence_blocks"] += 1
                    self._persistence_failed = True
                    self._accept_new_exposure = False
                    self._active = False
                    return False

                # 2. Strict cohort creation
                meta = {
                    "git_sha": self._git_sha,
                    "strategy_version": self.config.strategy_version,
                    "provider": self.config.provider,
                    "random_seed": self.config.random_seed,
                    "symbol": self.config.symbol,
                    "timeframe": self.config.timeframe,
                    "notional_usdt": self.config.notional_usdt,
                    "horizon_s": self.config.horizon_s,
                    "order_ttl_ms": self.config.order_ttl_ms,
                    "fees": self.config.taker_fee_bps,
                    "slippage": self.config.entry_slippage_bps,
                    "entry_slippage_bps": self.config.entry_slippage_bps,
                    "exit_slippage_bps": self.config.exit_slippage_bps,
                    "cost_source": self.config.cost_source,
                    "cost_effective_at": self.config.cost_effective_at,
                    "risk_config": {
                        "max_position_size": (
                            self.risk_adapter.risk_manager.config.max_position_size
                            if hasattr(self.risk_adapter, "risk_manager")
                            else None
                        )
                    },
                    "risk_position_limit_active": False,
                    "risk_daily_loss_active": False,
                }
                cohort_created = self.ledger.create_cohort(
                    cohort_id=self.config.cohort_id or "CH_SHADOW_DEFAULT",
                    created_at_ms=self.clock_ms(),
                    description="Shadow paper trading session",
                    metadata=meta,
                )
                if not cohort_created:
                    # Duplicate or failed cohort creation: fail closed
                    self._active = False
                    self._accept_new_exposure = False
                    return False

                # 3. Record STARTED cohort event
                self.ledger.record_cohort_event(
                    cohort_id=self.config.cohort_id or "CH_SHADOW_DEFAULT",
                    event_type="STARTED",
                    timestamp_ms=self.clock_ms(),
                    reason="Runtime started",
                    metadata={"git_sha": self._git_sha},
                )
                self.ledger.flush_status()

            self._active = True
            self._accept_new_exposure = True
            return True

    def on_signal(self, event: Dict[str, Any]) -> ShadowSignalResult:
        """
        Process a runtime market signal event into paper order execution.

        Fail-closed boundary: exceptions are trapped, incrementing runtime_errors
        without escaping into the orchestrator or EventBus.
        """
        with self._lock:
            if not self._active or not self._accept_new_exposure or not self.config.enabled:
                return ShadowSignalResult(status="INACTIVE")

            try:
                # 0. Health gate check
                if self.ledger is not None:
                    health = self.ledger.health_snapshot()
                    if not health["healthy"]:
                        self._counters["persistence_blocks"] += 1
                        self._persistence_failed = True
                        self._accept_new_exposure = False
                        return ShadowSignalResult(status="ERROR", error="Persistence health gate failure")

                self._counters["signals_seen"] += 1

                # 1. Adapt signal to canonical decision
                adapter_res = self.signal_adapter.process_signal(event)

                # Determine observation status & skip reason
                if adapter_res.status == "SKIPPED_INVALID_EVENT":
                    self._counters["invalid_signals"] += 1
                    obs_status = "INVALID_SIGNAL"
                    skip_reason = adapter_res.reason or "Invalid signal payload"
                elif adapter_res.status == "SKIPPED_NON_DIRECTIONAL":
                    self._counters["nondirectional_skips"] += 1
                    obs_status = "SKIPPED_NON_DIRECTIONAL"
                    skip_reason = adapter_res.reason or "Non-directional signal"
                elif adapter_res.decision is not None:
                    obs_status = "DECISION_CREATED"
                    skip_reason = None
                else:
                    self._counters["invalid_signals"] += 1
                    obs_status = "INVALID_SIGNAL"
                    skip_reason = adapter_res.reason or "No decision generated"

                # Persist signal observation
                if self.ledger is not None:
                    ok_obs = self.ledger.record_signal_observation(
                        cohort_id=self.config.cohort_id or "CH_SHADOW_DEFAULT",
                        symbol=self.config.symbol,
                        source_window_id=adapter_res.source_window_id or str(event.get("epoch_ms", 0)),
                        source_event_key=adapter_res.source_event_key or str(event.get("event_type", "default")),
                        signal_timestamp=int(event.get("epoch_ms", event.get("timestamp", self.clock_ms()))),
                        source_side=adapter_res.source_side or "UNKNOWN",
                        status=obs_status,
                        reason=skip_reason,
                        context={"event_type": str(event.get("event_type", "unknown"))},
                        created_at_ms=self.clock_ms(),
                    )
                    if ok_obs:
                        self._counters["observations_enqueued"] += 1
                    else:
                        self._counters["persistence_errors"] += 1
                        self._counters["persistence_blocks"] += 1
                        self._persistence_failed = True
                        self._accept_new_exposure = False
                        return ShadowSignalResult(status="ERROR", error="Failed to enqueue signal observation")

                if obs_status != "DECISION_CREATED":
                    return ShadowSignalResult(status="SKIPPED", skip_reason=skip_reason)

                decision = adapter_res.decision
                if decision is None:
                    return ShadowSignalResult(status="SKIPPED", skip_reason="No decision generated")

                self._counters["directional_decisions"] += 1

                # Persist decision BEFORE risk evaluation
                if self.ledger is not None:
                    ok_dec = self.ledger.record_decision(decision)
                    if ok_dec:
                        self._counters["decisions_enqueued"] += 1
                    else:
                        self._counters["persistence_errors"] += 1
                        self._counters["persistence_blocks"] += 1
                        self._persistence_failed = True
                        self._accept_new_exposure = False
                        return ShadowSignalResult(status="ERROR", error="Failed to enqueue decision")

                # Register decision with PredictionTracker BEFORE risk evaluation
                if self.prediction_tracker is not None:
                    pred_id = self.prediction_tracker.register(decision)
                    if pred_id:
                        self._counters["predictions_registered"] += 1

                # 2. Risk check
                risk_res = self.risk_adapter.evaluate(decision)
                risk_status = "APPROVED" if risk_res.status == "APPROVED" else ("REJECTED" if risk_res.status == "REJECTED_BY_RISK" else "ERROR")

                # Persist risk evaluation BEFORE order submission
                if self.ledger is not None:
                    ok_risk = self.ledger.record_risk_evaluation(
                        cohort_id=self.config.cohort_id or "CH_SHADOW_DEFAULT",
                        decision_id=decision.decision_id,
                        status=risk_status,
                        risk_confidence=float(risk_res.risk_confidence if risk_res.risk_confidence is not None else 0.0),
                        risk_reason=risk_res.risk_reason,
                        max_size=risk_res.max_size,
                        source_confidence=risk_res.source_confidence,
                        context=risk_res.context,
                        evaluated_at_ms=self.clock_ms(),
                    )
                    if ok_risk:
                        self._counters["risk_evaluations_enqueued"] += 1
                    else:
                        self._counters["persistence_errors"] += 1
                        self._counters["persistence_blocks"] += 1
                        self._persistence_failed = True
                        self._accept_new_exposure = False
                        return ShadowSignalResult(status="ERROR", error="Failed to enqueue risk evaluation")

                if risk_res.status != "APPROVED" or risk_res.paper_order is None:
                    self._counters["risk_rejected"] += 1
                    return ShadowSignalResult(
                        status="RISK_REJECTED",
                        rejection_reason=risk_res.risk_reason or "Rejected by risk manager",
                    )

                self._counters["risk_approved"] += 1

                # Persist order BEFORE submitting to sink
                if self.ledger is not None:
                    ok_ord = self.ledger.record_order(risk_res.paper_order)
                    if ok_ord:
                        self._counters["orders_enqueued"] += 1
                    else:
                        self._counters["persistence_errors"] += 1
                        self._counters["persistence_blocks"] += 1
                        self._persistence_failed = True
                        self._accept_new_exposure = False
                        return ShadowSignalResult(status="ERROR", error="Failed to enqueue order")

                # 3. Order submission to causal sink
                sink_res = self.execution_sink.submit_order(risk_res.paper_order)
                if sink_res.status == "ACCEPTED":
                    self._counters["orders_submitted"] += 1
                    return ShadowSignalResult(
                        status="ORDER_SUBMITTED",
                        order_id=risk_res.paper_order.order_id,
                    )
                else:
                    self._counters["order_rejected"] += 1
                    rejection_reason = (
                        sink_res.rejection.reason
                        if sink_res.rejection is not None
                        else sink_res.error_message or str(sink_res.status)
                    )
                    if self.ledger is not None and sink_res.rejection is not None:
                        self.ledger.record_rejection(sink_res.rejection)

                    return ShadowSignalResult(
                        status="ORDER_REJECTED",
                        rejection_reason=rejection_reason,
                    )

            except Exception as exc:
                self._counters["runtime_errors"] += 1
                return ShadowSignalResult(status="ERROR", error=str(exc))

    def on_market_trade(self, norm: Dict[str, Any]) -> ShadowTradeResult:
        """
        Process a normalized market trade tick through the causal ExecutionSink.

        Continues delegating to sink as long as runtime is active, even if
        ledger is unhealthy or new exposures are blocked, to avoid freezing
        existing positions.
        """
        with self._lock:
            if not self._active or not self.config.enabled:
                return ShadowTradeResult(status="INACTIVE")

            try:
                self._counters["ticks_seen"] += 1

                sink_res = self.execution_sink.on_market_trade(norm)
                accepted = (sink_res.status == "ACCEPTED")
                fills_count = 0
                closed_count = 0
                rejections_count = 0

                if sink_res.events:
                    fills_count = len(sink_res.events.fills)
                    closed_count = len(sink_res.events.closed_trades)
                    rejections_count = len(sink_res.events.rejections)

                    # Persist discrete events to ledger if configured
                    if self.ledger is not None:
                        for rej in sink_res.events.rejections:
                            self.ledger.record_rejection(rej)

                        for fill in sink_res.events.fills:
                            ok_f = self.ledger.record_fill(fill)
                            if ok_f:
                                self._counters["fills_enqueued"] += 1
                            else:
                                self._counters["persistence_errors"] += 1
                                self._persistence_failed = True
                                self._accept_new_exposure = False

                        for pos in sink_res.events.opened_positions:
                            ok_p = self.ledger.record_position(pos)
                            if ok_p:
                                self._counters["positions_enqueued"] += 1
                            else:
                                self._counters["persistence_errors"] += 1
                                self._persistence_failed = True
                                self._accept_new_exposure = False

                        for trade in sink_res.events.closed_trades:
                            ok_c = self.ledger.record_closed_trade(trade)
                            if ok_c:
                                self._counters["closed_trades_enqueued"] += 1
                            else:
                                self._counters["persistence_errors"] += 1
                                self._persistence_failed = True
                                self._accept_new_exposure = False

                self._counters["fills"] += fills_count
                self._counters["closed_trades"] += closed_count

                # Feed PredictionTracker on the same causal timeline
                if self.prediction_tracker is not None:
                    resolved_preds = self.prediction_tracker.on_tick(norm)
                    for pred_out in resolved_preds:
                        self._counters["predictions_resolved"] += 1
                        if self.ledger is not None:
                            ok_pred = self.ledger.record_prediction_outcome(pred_out)
                            if ok_pred:
                                self._counters["prediction_outcomes_enqueued"] += 1
                            else:
                                self._counters["persistence_errors"] += 1
                                self._counters["persistence_blocks"] += 1
                                self._persistence_failed = True
                                self._accept_new_exposure = False

                return ShadowTradeResult(
                    status="PROCESSED",
                    accepted=accepted,
                    fills_count=fills_count,
                    closed_trades_count=closed_count,
                    rejections_count=rejections_count,
                )

            except Exception as exc:
                self._counters["runtime_errors"] += 1
                return ShadowTradeResult(status="ERROR", error=str(exc))

    def shutdown(self) -> None:
        """
        Gracefully disable the runtime.

        Disallows new signals, new orders, and tick evaluations immediately.
        Records GRACEFUL_SHUTDOWN cohort event with open position counts.
        Flushes ledger and closes it only if runtime owns it.
        Does NOT synthesize artificial position liquidations.
        """
        with self._lock:
            self._active = False
            self._accept_new_exposure = False

            if self.prediction_tracker is not None:
                unresolved_preds = self.prediction_tracker.flush_unresolved(
                    reason="PROCESS_SHUTDOWN",
                    observed_at_ms=self.clock_ms(),
                )
                for pred_out in unresolved_preds:
                    self._counters["predictions_unresolved"] += 1
                    if self.ledger is not None:
                        ok_pred = self.ledger.record_prediction_outcome(pred_out)
                        if ok_pred:
                            self._counters["prediction_outcomes_enqueued"] += 1
                        else:
                            self._counters["persistence_errors"] += 1
                            self._counters["persistence_blocks"] += 1
                            self._persistence_failed = True
                            self._accept_new_exposure = False

            if self.ledger is not None:
                open_pos_count = 0
                pending_orders_count = 0
                if self.execution_sink is not None and hasattr(self.execution_sink, "executor"):
                    open_pos_count = len(self.execution_sink.executor.position_manager.open_positions)
                    pending_orders_count = len(self.execution_sink.executor.pending_orders)

                meta = {
                    "open_positions_count": open_pos_count,
                    "pending_orders_count": pending_orders_count,
                }
                self.ledger.record_cohort_event(
                    cohort_id=self.config.cohort_id or "CH_SHADOW_DEFAULT",
                    event_type="GRACEFUL_SHUTDOWN",
                    timestamp_ms=self.clock_ms(),
                    reason="Runtime shutdown",
                    metadata=meta,
                )
                self.ledger.flush_status()
                if self._ledger_is_owner:
                    self.ledger.close()

    def get_counters(self) -> Dict[str, int]:
        """Return a read-only snapshot copy of in-memory telemetry counters."""
        with self._lock:
            return dict(self._counters)

    def get_status(self) -> Dict[str, Any]:
        """
        Return runtime status metadata, explicitly documenting operational limitations.
        """
        with self._lock:
            return {
                "active": self._active,
                "enabled": self.config.enabled,
                "accept_new_exposure": self._accept_new_exposure,
                "persistence_failed": self._persistence_failed,
                "cohort_id": self.config.cohort_id,
                "provider": self.config.provider,
                "symbol": self.config.symbol,
                "timeframe": self.config.timeframe,
                "risk_position_limit_active": False,
                "risk_daily_loss_active": False,
                "counters": self.get_counters(),
            }
