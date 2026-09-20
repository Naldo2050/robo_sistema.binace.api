# tests/unit/paper_trading/test_signal_adapter.py
"""
Unit tests for SignalDecisionAdapter (Gate C1-B).

Proves:
6. FOLLOW_SIGNAL LONG -> LONG.
7. FOLLOW_SIGNAL SHORT -> SHORT.
8. FOLLOW_SIGNAL NEUTRAL -> SKIPPED_NON_DIRECTIONAL.
9. FOLLOW_SIGNAL UNKNOWN -> SKIPPED_NON_DIRECTIONAL.
10. BASELINE preserves source_side NEUTRAL while allowing decision_side LONG/SHORT.
11. Retry of the same event produces identical source_event_key.
12. Retry of the same event produces identical decision_id.
13. Distinct events in the same window produce distinct source_event_keys.
14. Distinct events in the same window produce distinct decision_ids.
15. Canonical window identity is stable across calls.
16. reference_price is positive and finite.
17. reference_price does NOT create a PaperFill.
18. confidence remains strictly None.
19. Strict temporal causality (signal_ts <= decision_ts <= available_at).
20. Clock behind signal timestamp fails closed (SKIPPED_INVALID_EVENT).
21. Zero network access.
22. Zero order placement APIs.
23. Adapter does not depend on AIAnalyzer or AI runtime.
"""

import math
import pytest

from paper_trading.adapters.signal_adapter import (
    AdapterResult,
    SignalDecisionAdapter,
)
from paper_trading.contracts import CanonicalDecision, PaperFill
from paper_trading.decision_providers import (
    FixedDecisionProvider,
    RandomDecisionProvider,
)


@pytest.fixture
def base_adapter():
    """Default adapter with fixed simulated clock at 1700000065000 (decision_ts)."""
    return SignalDecisionAdapter(
        cohort_id="c1_audit_cohort",
        strategy_version="c1_v1.0.0",
        mode="FOLLOW_SIGNAL",
        timeframe="1m",
        clock_ms=lambda: 1700000065000,
        configured_latency_ms=50,
    )


def test_follow_signal_long(base_adapter):
    """Point 6: FOLLOW_SIGNAL converts a BULLISH / buy absorption signal into a LONG decision."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    result = base_adapter.process_signal(event)
    assert result.status == "DECISION_CREATED"
    assert result.source_side == "LONG"
    assert result.decision is not None
    assert result.decision.side == "LONG"
    assert result.decision.symbol == "BTCUSDT"
    assert result.decision.reference_price == 85000.0


def test_follow_signal_short(base_adapter):
    """Point 7: FOLLOW_SIGNAL converts a BEARISH / sell absorption signal into a SHORT decision."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",
        "preco_fechamento": 85200.0,
    }

    result = base_adapter.process_signal(event)
    assert result.status == "DECISION_CREATED"
    assert result.source_side == "SHORT"
    assert result.decision is not None
    assert result.decision.side == "SHORT"


def test_follow_signal_neutral_skipped(base_adapter):
    """Point 8: FOLLOW_SIGNAL skips NEUTRAL routine signals without creating decisions."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "resultado_da_batalha": "NEUTRAL",
        "preco_fechamento": 85100.0,
    }

    result = base_adapter.process_signal(event)
    assert result.status == "SKIPPED_NON_DIRECTIONAL"
    assert result.source_side == "NEUTRAL"
    assert result.decision is None
    assert "skips non-directional" in (result.reason or "")


def test_follow_signal_unknown_skipped(base_adapter):
    """Point 9: FOLLOW_SIGNAL skips unidentifiable / UNKNOWN side signals fail-closed."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "DESCONHECIDO_TESTE",
        "resultado_da_batalha": "SEM_RESULTADO",
        "preco_fechamento": 85100.0,
    }

    result = base_adapter.process_signal(event)
    assert result.status == "SKIPPED_NON_DIRECTIONAL"
    assert result.source_side == "UNKNOWN"
    assert result.decision is None


def test_baseline_preserves_source_side_neutral_with_decision_long():
    """Point 10: BASELINE mode preserves source_side=NEUTRAL while decision.side is LONG."""
    fixed_provider = FixedDecisionProvider(side="LONG")
    adapter = SignalDecisionAdapter(
        cohort_id="c1_baseline_cohort",
        mode="BASELINE",
        provider=fixed_provider,
        clock_ms=lambda: 1700000065000,
    )

    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "resultado_da_batalha": "EQUILIBRIO",
        "preco_fechamento": 85100.0,
    }

    result = adapter.process_signal(event)
    assert result.status == "DECISION_CREATED"
    # Source event direction remains unaltered
    assert result.source_side == "NEUTRAL"
    # Canonical decision receives benchmark provider direction
    assert result.decision is not None
    assert result.decision.side == "LONG"
    assert result.decision.decision_provider == "fixed_long"


def test_retry_same_event_produces_identical_key_and_decision_id(base_adapter):
    """Points 11 & 12: Retrying the identical event produces identical event key and decision_id."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    res1 = base_adapter.process_signal(event)
    res2 = base_adapter.process_signal(event)

    assert res1.source_event_key == res2.source_event_key
    assert res1.decision is not None and res2.decision is not None
    assert res1.decision.decision_id == res2.decision.decision_id


def test_distinct_events_in_same_window_produce_distinct_keys_and_decision_ids(base_adapter):
    """Points 13 & 14: Distinct events in the same time window produce distinct keys and decision_ids."""
    window_epoch = 1700000060000

    # Event A: Absorption
    event_a = {
        "symbol": "BTCUSDT",
        "epoch_ms": window_epoch,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    # Event B: Exhaustion
    event_b = {
        "symbol": "BTCUSDT",
        "epoch_ms": window_epoch,
        "tipo_evento": "Exaustão",
        "resultado_da_batalha": "Exaustão de Venda",
        "preco_fechamento": 85050.0,
    }

    res_a = base_adapter.process_signal(event_a)
    res_b = base_adapter.process_signal(event_b)

    # Same source window
    assert res_a.source_window_id == res_b.source_window_id == f"BTCUSDT:1m:{window_epoch}"

    # Distinct source event keys
    assert res_a.source_event_key != res_b.source_event_key

    # Distinct decision UUID5s
    assert res_a.decision is not None and res_b.decision is not None
    assert res_a.decision.decision_id != res_b.decision.decision_id


def test_window_identity_stable(base_adapter):
    """Point 15: Window identity matches deterministic specification '{symbol}:{timeframe}:{close_ms}'."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    result = base_adapter.process_signal(event)
    assert result.source_window_id == "BTCUSDT:1m:1700000060000"


def test_reference_price_positive_and_finite(base_adapter):
    """Point 16: reference_price must be finite and positive (>0)."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 88123.45,
    }

    result = base_adapter.process_signal(event)
    assert result.decision is not None
    ref_price = result.decision.reference_price
    assert math.isfinite(ref_price)
    assert ref_price > 0
    assert ref_price == 88123.45


def test_reference_price_does_not_create_fill(base_adapter):
    """Point 17: SignalDecisionAdapter creates CanonicalDecision without instantiating any PaperFill."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    result = base_adapter.process_signal(event)
    assert isinstance(result.decision, CanonicalDecision)
    # Proves no PaperFill is created or attached anywhere
    assert not isinstance(result.decision, PaperFill)
    assert not hasattr(result.decision, "fill_price")


def test_confidence_remains_strictly_none(base_adapter):
    """Point 18: confidence is kept as None (uncalibrated) and never coerced to 0.0 or heuristic score."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
        "historical_confidence": {"score": 0.85},  # Heuristic score present
    }

    result = base_adapter.process_signal(event)
    assert result.decision is not None
    assert result.decision.confidence is None


def test_temporal_causality(base_adapter):
    """Point 19: signal_timestamp <= decision_timestamp <= available_at."""
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    result = base_adapter.process_signal(event)
    d = result.decision
    assert d is not None
    assert d.signal_timestamp == 1700000060000
    assert d.decision_timestamp == 1700000065000
    assert d.available_at == 1700000065050  # 1700000065000 + 50ms latency
    assert d.signal_timestamp <= d.decision_timestamp <= d.available_at


def test_clock_behind_signal_timestamp_fails_closed():
    """Point 20: Clock behind signal timestamp triggers causal fail-closed skip."""
    adapter = SignalDecisionAdapter(
        cohort_id="c1_audit_cohort",
        clock_ms=lambda: 1700000050000,  # 10 seconds in the past!
    )

    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,  # Signal is at 60000
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    result = adapter.process_signal(event)
    assert result.status == "SKIPPED_INVALID_EVENT"
    assert result.decision is None
    assert "Causal violation" in (result.reason or "")


def test_zero_network_access(monkeypatch, base_adapter):
    """Point 21: SignalDecisionAdapter operates in complete network isolation."""
    import socket

    def _block_network(*args, **kwargs):
        raise AssertionError("Unexpected network socket connection attempted in hermetic adapter!")

    monkeypatch.setattr(socket, "socket", _block_network)

    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
    }

    result = base_adapter.process_signal(event)
    assert result.status == "DECISION_CREATED"


def test_zero_order_placement_apis(base_adapter):
    """Point 22: Adapter produces CanonicalDecision only and contains no order placement capabilities."""
    assert not hasattr(base_adapter, "place_order")
    assert not hasattr(base_adapter, "submit_order")
    assert not hasattr(base_adapter, "create_order")


def test_adapter_independent_of_ai_analyzer(base_adapter):
    """Point 23: Adapter operates identically when AI runtime is completely absent or disabled."""
    # Pass event with no ai_payload or AI references
    event = {
        "symbol": "BTCUSDT",
        "epoch_ms": 1700000060000,
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Venda",
        "preco_fechamento": 85000.0,
        "ai_analyzer": None,
        "ai_test_passed": False,
    }

    result = base_adapter.process_signal(event)
    assert result.status == "DECISION_CREATED"
    assert result.decision is not None
    assert result.decision.model_version is None
