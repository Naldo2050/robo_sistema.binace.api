# paper_trading/decision_providers.py
"""
Decision providers for paper trading simulations and baseline benchmarks.

Provides deterministic decision sources (Fixed and Random) to establish
null-hypothesis baselines without accessing external networks, real markets,
or AI/LLM components.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable

from common.signal_direction import SignalSide


@runtime_checkable
class DecisionProvider(Protocol):
    """Protocol for decision providers generating trade directions."""

    @property
    def provider_id(self) -> str:
        """Stable unique identifier for this provider configuration."""
        ...

    def get_decision_side(
        self,
        source_side: SignalSide,
        context: Optional[Dict[str, Any]] = None,
    ) -> SignalSide:
        """
        Generate decision side given the observed source side and optional context.

        Returns:
            "LONG" or "SHORT" for executable decisions.
        """
        ...


@dataclass(frozen=True)
class FixedDecisionProvider:
    """
    Deterministic provider emitting a single fixed directional side.

    Invariants:
    - side must be strictly "LONG" or "SHORT".
    - Zero temporal or external state.
    - Fully reproducible.
    """

    side: Literal["LONG", "SHORT"]
    _provider_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.side not in ("LONG", "SHORT"):
            raise ValueError(f"FixedDecisionProvider side must be 'LONG' or 'SHORT', got '{self.side}'")

    @property
    def provider_id(self) -> str:
        if self._provider_id:
            return self._provider_id
        return f"fixed_{self.side.lower()}"

    def get_decision_side(
        self,
        source_side: SignalSide,
        context: Optional[Dict[str, Any]] = None,
    ) -> SignalSide:
        return self.side


class RandomDecisionProvider:
    """
    Deterministic pseudo-random provider using an isolated random.Random instance.

    Invariants:
    - seed is mandatory (int).
    - NEVER touches Python's global random state.
    - NEVER uses system clock / time as seed.
    - Strictly produces "LONG" or "SHORT".
    - Replay-deterministic: identical seed generates identical sequence.
    """

    def __init__(
        self,
        seed: int,
        p_long: float = 0.5,
        provider_id: Optional[str] = None,
    ) -> None:
        if not isinstance(seed, int):
            raise TypeError(f"RandomDecisionProvider requires an explicit integer seed, got {type(seed).__name__}")
        if not (0.0 <= p_long <= 1.0):
            raise ValueError(f"p_long must be in [0.0, 1.0], got {p_long}")

        self.seed = seed
        self.p_long = p_long
        self._provider_id = provider_id or f"random_seed_{seed}"
        self._rng = random.Random(seed)

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def get_decision_side(
        self,
        source_side: SignalSide,
        context: Optional[Dict[str, Any]] = None,
    ) -> SignalSide:
        """Generate next pseudo-random side deterministically."""
        roll = self._rng.random()
        return "LONG" if roll < self.p_long else "SHORT"

    def reset(self) -> None:
        """Reset the internal generator to the initial seed for identical replay."""
        self._rng = random.Random(self.seed)
