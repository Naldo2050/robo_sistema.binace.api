"""
Tests for common/ai_throttler.py v3.

v3 changes vs v2:
- ALWAYS_PROCESS: full names ("Absorção", "Exaustão", ...) not short codes
- Significant change: delta >= 5.0 or volume_spike >= 2x (not imbalance/bsr)
- Stats via get_status() dict (no _calls_total/_calls_made/_calls_saved)
- record_call() separate from should_call_ai()
- No significant_imb_change or significant_bsr_change params
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common.ai_throttler import SmartAIThrottler, ALWAYS_PROCESS


def _make_payload(trigger="ANALYSIS_TRIGGER", delta=0.0, volume=0.0, avg_volume=10.0):
    return {
        "trigger": trigger,
        "delta": delta,
        "volume": volume,
        "avg_volume": avg_volume,
    }


class TestHardMinimum:
    def test_hard_min_blocks_all(self):
        """Nenhuma chamada antes de 60s, mesmo para eventos importantes."""
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t._last_call_ts = time.time()  # simula chamada agora

        # Evento importante dentro do hard min
        payload = _make_payload(trigger="Absorção", delta=10.0, volume=100.0, avg_volume=10.0)
        result = t.should_call_ai(payload)
        assert result is False, "Hard min should block even important events"

    def test_first_call_always_passes(self):
        t = SmartAIThrottler()
        payload = _make_payload(trigger="ANALYSIS_TRIGGER")
        assert t.should_call_ai(payload) is True


class TestSoftMinimum:
    def test_soft_min_blocks_analysis_trigger(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Última chamada 90s atrás (> hard_min, < soft_min)
        t._last_call_ts = time.time() - 90

        payload = _make_payload(trigger="ANALYSIS_TRIGGER", delta=1.0)
        result = t.should_call_ai(payload)
        assert result is False, "Soft min should block small-change trigger"

    def test_interval_ok_passes(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Última chamada 200s atrás (> soft_min)
        t._last_call_ts = time.time() - 200

        payload = _make_payload(trigger="ANALYSIS_TRIGGER")
        assert t.should_call_ai(payload) is True


class TestImportantEvents:
    def test_important_event_bypasses_soft_min(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Última chamada 90s atrás (> hard_min, < soft_min)
        t._last_call_ts = time.time() - 90

        payload = _make_payload(trigger="Absorção")
        result = t.should_call_ai(payload)
        assert result is True, "Important events should bypass soft min"

    def test_all_important_triggers(self):
        for trigger in ALWAYS_PROCESS:
            t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
            t._last_call_ts = time.time() - 90

            payload = _make_payload(trigger=trigger)
            assert t.should_call_ai(payload) is True, \
                f"Trigger '{trigger}' should bypass soft min"


class TestSignificantChange:
    def test_large_delta_bypasses_soft_min(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t._last_call_ts = time.time() - 90

        # delta >= 5.0 → significant
        payload = _make_payload(trigger="ANALYSIS_TRIGGER", delta=6.0)
        assert t.should_call_ai(payload) is True

    def test_small_delta_blocked(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t._last_call_ts = time.time() - 90

        # delta < 5.0 → not significant
        payload = _make_payload(trigger="ANALYSIS_TRIGGER", delta=2.0)
        assert t.should_call_ai(payload) is False

    def test_bsr_cross_with_large_change(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t._last_call_ts = time.time() - 90

        # volume spike: volume / avg_volume >= 2.0
        payload = _make_payload(trigger="ANALYSIS_TRIGGER", volume=30.0, avg_volume=10.0)
        assert t.should_call_ai(payload) is True

    def test_bsr_cross_small_change_blocked(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t._last_call_ts = time.time() - 90

        # volume spike < 2x → not significant
        payload = _make_payload(trigger="ANALYSIS_TRIGGER", volume=15.0, avg_volume=10.0)
        assert t.should_call_ai(payload) is False


class TestStatistics:
    def test_stats_tracking(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)

        # Primeira chamada aceita → depois chamar record_call()
        result = t.should_call_ai(_make_payload())
        assert result is True
        t.record_call()
        assert t._calls_this_hour == 1
        assert t._consecutive_skips == 0

        # Segunda chamada bloqueada (dentro do hard min)
        t.should_call_ai(_make_payload())
        assert t._consecutive_skips == 1

    def test_reset_stats(self):
        # v3 não tem reset_stats() — verificar que consecutive_skips reseta
        t = SmartAIThrottler()
        t.should_call_ai(_make_payload())
        t.record_call()
        assert t._consecutive_skips == 0

    def test_stats_property(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t.should_call_ai(_make_payload())
        status = t.get_status()
        assert "tokens_used" in status
        assert "calls_this_hour" in status
        assert "consecutive_skips" in status
        assert "is_rate_limited" in status


class TestThrottlerScenario:
    """Simula o cenário real descrito pelo usuário."""

    def test_realistic_5_windows(self):
        """
        W1: primeira chamada → passa
        W2: 55s depois, delta pequeno → bloqueado (soft min)
        W3: 90s depois, delta=6.0 → passa (significant delta)
        W4: Absorção 70s depois → passa (ALWAYS_PROCESS)
        W5: 30s depois de W4 → bloqueado (hard_min)
        """
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        now = time.time()

        # W1: primeira chamada
        t._last_call_ts = 0
        r1 = t.should_call_ai(_make_payload("ANALYSIS_TRIGGER", delta=1.0))
        assert r1 is True, "W1: First call should pass"
        t.record_call()

        # W2: 55s depois (dentro do hard min)
        t._last_call_ts = now - 55
        r2 = t.should_call_ai(_make_payload("ANALYSIS_TRIGGER", delta=1.0))
        assert r2 is False, "W2: Inside hard_min should be blocked"

        # W3: 90s depois (> hard min, large delta)
        t._last_call_ts = now - 90
        r3 = t.should_call_ai(_make_payload("ANALYSIS_TRIGGER", delta=6.0))
        assert r3 is True, "W3: Large delta should pass"
        t.record_call()

        # W4: Absorção 70s depois (ALWAYS_PROCESS)
        t._last_call_ts = now - 70
        r4 = t.should_call_ai(_make_payload("Absorção"))
        assert r4 is True, "W4: Important event should pass"
        t.record_call()

        # W5: 30s depois de W4 (dentro do hard min)
        t._last_call_ts = now - 30
        r5 = t.should_call_ai(_make_payload("ANALYSIS_TRIGGER"))
        assert r5 is False, "W5: Inside hard_min should be blocked"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
