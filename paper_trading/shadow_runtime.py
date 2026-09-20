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
      In-Memory State & Metrics
    """

    def __init__(
        self,
        config: ShadowPaperConfig,
        clock_ms: Optional[Callable[[], int]] = None,
        risk_manager: Optional[RiskManager] = None,
        risk_adapter: Optional[RiskAdapter] = None,
        execution_sink: Optional[ExecutionSink] = None,
        signal_adapter: Optional[SignalDecisionAdapter] = None,
        ledger: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.ledger = ledger
        self._lock = threading.RLock()

        # Lifecycle flag
        self._active: bool = bool(config.enabled)

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
        else:
            # Inactive stubs when disabled
            self.provider = None  # type: ignore[assignment]
            self.signal_adapter = None  # type: ignore[assignment]
            self.risk_adapter = None  # type: ignore[assignment]
            self.execution_sink = None  # type: ignore[assignment]

    @property
    def is_active(self) -> bool:
        """Indicate whether the shadow runtime is actively processing events."""
        with self._lock:
            return self._active and self.config.enabled

    def on_signal(self, event: Dict[str, Any]) -> ShadowSignalResult:
        """
        Process a runtime market signal event into paper order execution.

        Fail-closed boundary: exceptions are trapped, incrementing runtime_errors
        without escaping into the orchestrator or EventBus.
        """
        with self._lock:
            if not self._active or not self.config.enabled:
                return ShadowSignalResult(status="INACTIVE")

            try:
                self._counters["signals_seen"] += 1

                # 1. Adapt signal to canonical decision
                adapter_res = self.signal_adapter.process_signal(event)

                if adapter_res.status == "SKIPPED_INVALID_EVENT":
                    self._counters["invalid_signals"] += 1
                    return ShadowSignalResult(
                        status="SKIPPED",
                        skip_reason=adapter_res.reason or "Invalid signal payload",
                    )

                if adapter_res.status == "SKIPPED_NON_DIRECTIONAL":
                    self._counters["nondirectional_skips"] += 1
                    return ShadowSignalResult(
                        status="SKIPPED",
                        skip_reason=adapter_res.reason or "Non-directional signal",
                    )

                decision = adapter_res.decision
                if decision is None:
                    self._counters["invalid_signals"] += 1
                    return ShadowSignalResult(
                        status="SKIPPED",
                        skip_reason=adapter_res.reason or "No decision generated",
                    )

                self._counters["directional_decisions"] += 1

                # 2. Risk check
                risk_res = self.risk_adapter.evaluate(decision)
                if risk_res.status != "APPROVED" or risk_res.paper_order is None:
                    self._counters["risk_rejected"] += 1
                    return ShadowSignalResult(
                        status="RISK_REJECTED",
                        rejection_reason=risk_res.risk_reason or "Rejected by risk manager",
                    )

                self._counters["risk_approved"] += 1

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

        Fail-closed boundary: exceptions are trapped, incrementing runtime_errors
        without escaping into the WebSocket message loop.
        """
        with self._lock:
            if not self._active or not self.config.enabled:
                return ShadowTradeResult(status="INACTIVE")

            try:
                self._counters["ticks_seen"] += 1

                sink_res = self.execution_sink.on_market_trade(norm)
                accepted = (sink_res.status == "ACCEPTED")
                fills_count = len(sink_res.events.fills) if sink_res.events else 0
                closed_count = len(sink_res.events.closed_trades) if sink_res.events else 0
                rejections_count = len(sink_res.events.rejections) if sink_res.events else 0

                self._counters["fills"] += fills_count
                self._counters["closed_trades"] += closed_count

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
        Does NOT synthesize artificial position liquidations.
        """
        with self._lock:
            self._active = False

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
                "cohort_id": self.config.cohort_id,
                "provider": self.config.provider,
                "symbol": self.config.symbol,
                "timeframe": self.config.timeframe,
                "risk_position_limit_active": False,
                "risk_daily_loss_active": False,
                "counters": self.get_counters(),
            }
