# paper_trading/prediction.py
"""
Canonical Prediction Outcome Contracts & Domain Entities (Gate D0-B).

Defines data contracts and configuration for direction prediction evaluation,
completely isolated from economic execution (PaperOrder, PaperFill, ClosedTrade).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

# Canonical result states
PredictionResult = Literal[
    "PENDING",
    "CORRECT",
    "INCORRECT",
    "FLAT",
    "UNRESOLVED",
]

PREDICTION_RESULTS = {
    "PENDING",
    "CORRECT",
    "INCORRECT",
    "FLAT",
    "UNRESOLVED",
}

# Canonical outcome reasons
PredictionReason = Literal[
    "HORIZON_RESOLVED",
    "NO_TICK_WITHIN_TOLERANCE",
    "PROCESS_SHUTDOWN",
]

PREDICTION_REASONS = {
    "HORIZON_RESOLVED",
    "NO_TICK_WITHIN_TOLERANCE",
    "PROCESS_SHUTDOWN",
}


@dataclass(frozen=True)
class PredictionTrackerConfig:
    """Explicit configuration for canonical prediction evaluation."""

    resolution_tolerance_ms: int = 30_000
    flat_tolerance_bps: float = 1.0
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        if self.resolution_tolerance_ms < 0:
            raise ValueError("resolution_tolerance_ms must be non-negative")
        if not math.isfinite(self.flat_tolerance_bps) or self.flat_tolerance_bps < 0:
            raise ValueError("flat_tolerance_bps must be finite and non-negative")
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version must not be empty")


def make_prediction_id(decision_id: str, horizon_s: int, policy_version: str = "v1") -> str:
    """
    Generate a deterministic prediction_id from (decision_id, horizon_s, policy_version).

    Enables multi-horizon evaluation and prevents collisions across policies/versions.
    """
    key = f"{decision_id}:{horizon_s}:{policy_version}"
    return f"pred_{uuid.uuid5(uuid.NAMESPACE_DNS, key).hex[:20]}"


@dataclass(frozen=True)
class PredictionOutcome:
    """
    Canonical record of an evaluated directional prediction outcome.

    Independent of economic execution results (slippage, fees, fills, liquidation).
    """

    prediction_id: str
    decision_id: str
    cohort_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    reference_price: float
    decision_timestamp: int
    horizon_s: int
    deadline_ms: int
    result: PredictionResult
    reason: Optional[str]
    resolution_price: Optional[float]
    raw_return_bps: Optional[float]
    directional_return_bps: Optional[float]
    resolved_timestamp_ms: Optional[int]
    resolution_drift_ms: Optional[int]
    observed_at_ms: Optional[int]
    flat_tolerance_bps: float
    resolution_tolerance_ms: int
    policy_version: str
    created_at_ms: int

    def __post_init__(self) -> None:
        if self.result not in PREDICTION_RESULTS:
            raise ValueError(f"Invalid prediction result: {self.result}")
        if self.side not in ("LONG", "SHORT"):
            raise ValueError(f"PredictionOutcome side must be LONG or SHORT, got {self.side}")
        if not math.isfinite(self.reference_price) or self.reference_price <= 0:
            raise ValueError(f"reference_price must be finite > 0, got {self.reference_price}")


@dataclass(frozen=True)
class PendingPrediction:
    """In-memory representation of an active, unresolved prediction."""

    prediction_id: str
    decision_id: str
    cohort_id: str
    symbol: str
    side: str
    reference_price: float
    decision_timestamp: int
    horizon_s: int
    deadline_ms: int
    flat_tolerance_bps: float
    resolution_tolerance_ms: int
    policy_version: str
