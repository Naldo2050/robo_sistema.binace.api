# tests/unit/paper_trading/test_decision_providers.py
"""
Unit tests for hermetic DecisionProvider implementations (Fixed and Random).

Proves:
1. Fixed LONG determinism.
2. Fixed SHORT determinism.
3. Random with same seed generates identical sequence.
4. Random with different seeds generates different sequences.
5. Random does not contaminate or mutate Python's global random state.
"""

import random
import pytest

from paper_trading.decision_providers import (
    DecisionProvider,
    FixedDecisionProvider,
    RandomDecisionProvider,
)


def test_fixed_long_deterministic():
    """Point 1: Fixed LONG provider consistently returns LONG without external state."""
    provider = FixedDecisionProvider(side="LONG")
    assert isinstance(provider, DecisionProvider)
    assert provider.provider_id == "fixed_long"

    for _ in range(20):
        side = provider.get_decision_side("NEUTRAL")
        assert side == "LONG"
        assert provider.get_decision_side("SHORT") == "LONG"


def test_fixed_short_deterministic():
    """Point 2: Fixed SHORT provider consistently returns SHORT without external state."""
    provider = FixedDecisionProvider(side="SHORT")
    assert isinstance(provider, DecisionProvider)
    assert provider.provider_id == "fixed_short"

    for _ in range(20):
        side = provider.get_decision_side("NEUTRAL")
        assert side == "SHORT"
        assert provider.get_decision_side("LONG") == "SHORT"


def test_fixed_invalid_side():
    """FixedDecisionProvider fails fast if initialized with invalid side."""
    with pytest.raises(ValueError, match="side must be 'LONG' or 'SHORT'"):
        FixedDecisionProvider(side="NEUTRAL")  # type: ignore[arg-type]


def test_random_same_seed_produces_identical_sequence():
    """Point 3: Random providers initialized with the same seed generate identical sequences."""
    p1 = RandomDecisionProvider(seed=42, p_long=0.5)
    p2 = RandomDecisionProvider(seed=42, p_long=0.5)

    seq1 = [p1.get_decision_side("NEUTRAL") for _ in range(100)]
    seq2 = [p2.get_decision_side("NEUTRAL") for _ in range(100)]

    assert seq1 == seq2
    assert "LONG" in seq1
    assert "SHORT" in seq1


def test_random_reset_replays_exact_sequence():
    """Calling reset() on RandomDecisionProvider allows exact offline replay."""
    provider = RandomDecisionProvider(seed=123)
    seq_initial = [provider.get_decision_side("NEUTRAL") for _ in range(50)]

    provider.reset()
    seq_replayed = [provider.get_decision_side("NEUTRAL") for _ in range(50)]

    assert seq_initial == seq_replayed


def test_random_different_seeds_produce_divergent_sequences():
    """Point 4: Random providers initialized with different seeds generate distinct sequences."""
    p1 = RandomDecisionProvider(seed=1001)
    p2 = RandomDecisionProvider(seed=9999)

    seq1 = [p1.get_decision_side("NEUTRAL") for _ in range(100)]
    seq2 = [p2.get_decision_side("NEUTRAL") for _ in range(100)]

    assert seq1 != seq2


def test_random_does_not_affect_global_random_state():
    """Point 5: RandomDecisionProvider uses a private Random instance and never touches global random."""
    # Capture global state
    random.seed(777)
    global_before = random.getstate()

    # Interleave calls to RandomDecisionProvider
    local_provider = RandomDecisionProvider(seed=42)
    _ = [local_provider.get_decision_side("NEUTRAL") for _ in range(100)]

    # Global state must be completely unaffected
    global_after = random.getstate()
    assert global_before == global_after

    # Furthermore, verify global random output matches pure seed 777 progression
    expected_global = random.random()
    random.setstate(global_before)
    actual_global = random.random()
    assert expected_global == actual_global


def test_random_invalid_inputs():
    """RandomDecisionProvider validates seed and p_long boundaries."""
    with pytest.raises(TypeError, match="requires an explicit integer seed"):
        RandomDecisionProvider(seed="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="p_long must be in"):
        RandomDecisionProvider(seed=42, p_long=1.5)

    with pytest.raises(ValueError, match="p_long must be in"):
        RandomDecisionProvider(seed=42, p_long=-0.1)
