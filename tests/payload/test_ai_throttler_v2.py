"""
Tests for common/ai_throttler.py v2.

Verifica:
1. Hard minimum (60s) bloqueia TODAS as chamadas
2. Soft minimum (180s) bloqueia ANALYSIS_TRIGGER
3. Eventos importantes bypass soft min mas respeitam hard min
4. Mudança significativa de imbalance (>0.5) bypass soft min
5. BSR cross 1.0 com mudança >50% bypass soft min
6. Estatísticas corretas
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common.ai_throttler import SmartAIThrottler


def _make_payload(trigger="AT", imb=0.0, bsr=0.5):
    return {
        "t": trigger,
        "p": {"c": 73250},
        "f": {"d1": "+280K", "imb": imb, "bsr": bsr},
    }


class TestHardMinimum:
    def test_hard_min_blocks_all(self):
        """Nenhuma chamada antes de 60s, mesmo para eventos importantes."""
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Simular primeira chamada
        t._last_call_ts = time.time()
        t._last_flow_state = {"imb": 0.0, "bsr": 0.5}

        # Evento importante dentro do hard min
        payload = _make_payload(trigger="ABS", imb=0.8, bsr=2.0)
        result = t.should_call_ai(payload)
        assert result is False, "Hard min should block even important events"

    def test_first_call_always_passes(self):
        t = SmartAIThrottler()
        payload = _make_payload(trigger="AT")
        assert t.should_call_ai(payload) is True


class TestSoftMinimum:
    def test_soft_min_blocks_analysis_trigger(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Simular última chamada 90s atrás (> hard_min, < soft_min)
        t._last_call_ts = time.time() - 90
        t._last_flow_state = {"imb": 0.0, "bsr": 0.5}

        payload = _make_payload(trigger="AT", imb=0.1, bsr=0.6)
        result = t.should_call_ai(payload)
        assert result is False, "Soft min should block AT without significant change"

    def test_interval_ok_passes(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Última chamada 200s atrás
        t._last_call_ts = time.time() - 200
        t._last_flow_state = {"imb": 0.0, "bsr": 0.5}

        payload = _make_payload(trigger="AT")
        assert t.should_call_ai(payload) is True


class TestImportantEvents:
    def test_important_event_bypasses_soft_min(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        # Última chamada 90s atrás (> hard_min, < soft_min)
        t._last_call_ts = time.time() - 90
        t._last_flow_state = {"imb": 0.0, "bsr": 0.5}

        payload = _make_payload(trigger="ABS")
        result = t.should_call_ai(payload)
        assert result is True, "Important events should bypass soft min"

    def test_all_important_triggers(self):
        for trigger in ["ABS", "EXH", "BRK", "WHL", "DIV", "REV", "VSPK", "MOM"]:
            t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
            t._last_call_ts = time.time() - 90
            t._last_flow_state = {"imb": 0.0, "bsr": 0.5}

            payload = _make_payload(trigger=trigger)
            assert t.should_call_ai(payload) is True, \
                f"Trigger {trigger} should be treated as important"


class TestSignificantChange:
    def test_large_imb_change_bypasses_soft_min(self):
        t = SmartAIThrottler(
            min_interval=180, hard_min_interval=60,
            significant_imb_change=0.5,
        )
        t._last_call_ts = time.time() - 90
        t._last_flow_state = {"imb": -0.3, "bsr": 0.5}

        # Mudança de -0.3 para +0.4 = 0.7 > 0.5
        payload = _make_payload(trigger="AT", imb=0.4)
        assert t.should_call_ai(payload) is True

    def test_small_imb_change_blocked(self):
        t = SmartAIThrottler(
            min_interval=180, hard_min_interval=60,
            significant_imb_change=0.5,
        )
        t._last_call_ts = time.time() - 90
        t._last_flow_state = {"imb": 0.3, "bsr": 0.5}

        # Mudança de 0.3 para 0.5 = 0.2 < 0.5
        payload = _make_payload(trigger="AT", imb=0.5)
        assert t.should_call_ai(payload) is False

    def test_bsr_cross_with_large_change(self):
        t = SmartAIThrottler(
            min_interval=180, hard_min_interval=60,
            significant_bsr_change=0.5,
        )
        t._last_call_ts = time.time() - 90
        t._last_flow_state = {"imb": 0.3, "bsr": 0.6}

        # BSR cruza 1.0 (0.6 → 2.0) = mudança de 233% > 50%
        payload = _make_payload(trigger="AT", imb=0.3, bsr=2.0)
        assert t.should_call_ai(payload) is True

    def test_bsr_cross_small_change_blocked(self):
        t = SmartAIThrottler(
            min_interval=180, hard_min_interval=60,
            significant_bsr_change=0.5,
        )
        t._last_call_ts = time.time() - 90
        t._last_flow_state = {"imb": 0.3, "bsr": 0.9}

        # BSR cruza 1.0 (0.9 → 1.1) = mudança de 22% < 50%
        payload = _make_payload(trigger="AT", imb=0.3, bsr=1.1)
        assert t.should_call_ai(payload) is False


class TestStatistics:
    def test_stats_tracking(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)

        # Primeira chamada aceita
        t.should_call_ai(_make_payload(trigger="AT"))
        assert t._calls_total == 1
        assert t._calls_made == 1
        assert t._calls_saved == 0

        # Segunda chamada bloqueada (dentro do hard min)
        t.should_call_ai(_make_payload(trigger="AT"))
        assert t._calls_total == 2
        assert t._calls_made == 1
        assert t._calls_saved == 1
        assert t._saving_pct == 50

    def test_reset_stats(self):
        t = SmartAIThrottler()
        t.should_call_ai(_make_payload())
        t.reset_stats()
        assert t._calls_total == 0
        assert t._calls_made == 0
        assert t._calls_saved == 0

    def test_stats_property(self):
        t = SmartAIThrottler(min_interval=180, hard_min_interval=60)
        t.should_call_ai(_make_payload())
        stats = t.stats
        assert "total_events" in stats
        assert "calls_made" in stats
        assert "calls_saved" in stats
        assert "saving_rate" in stats
        assert stats["min_interval"] == 180
        assert stats["hard_min_interval"] == 60
        assert stats["imb_threshold"] == 0.5


class TestThrottlerScenario:
    """Simula o cenário real descrito pelo usuário."""

    def test_realistic_5_windows(self):
        """
        W1: imb=+0.55, bsr=3.45 → Chamou (primeira)
        W2: imb=+0.18           → Pulou (dentro soft, small change)
        W3: imb=-0.16           → Δimb=0.71 > 0.5 → CHAMA
        W4: Absorção            → ALWAYS_PROCESS (> hard_min)
        W5: imb=+0.48           → Dentro hard_min de W4 → PULOU
        """
        t = SmartAIThrottler(
            min_interval=180, hard_min_interval=60,
            significant_imb_change=0.5,
        )

        # W1: primeira chamada
        now = time.time()
        t._last_call_ts = 0  # reset
        r1 = t.should_call_ai(_make_payload("AT", imb=0.55, bsr=3.45))
        assert r1 is True, "W1: First call should pass"

        # W2: 55s depois (dentro do soft min, small change)
        t._last_call_ts = now - 125  # simula que W1 foi há 125s
        r2 = t.should_call_ai(_make_payload("AT", imb=0.18, bsr=1.2))
        # Δimb = |0.18 - 0.55| = 0.37 < 0.5 → bloqueado
        assert r2 is False, "W2: Small change should be blocked"

        # W3: 90s depois (> hard min, Δimb = |−0.16−0.55|=0.71 > 0.5)
        t._last_call_ts = now - 90
        t._last_flow_state = {"imb": 0.55, "bsr": 3.45}  # estado original do W1
        r3 = t.should_call_ai(_make_payload("AT", imb=-0.16, bsr=0.73))
        assert r3 is True, "W3: Large imb change should pass"

        # W4: Absorção 70s depois (> hard_min)
        t._last_call_ts = now - 70
        r4 = t.should_call_ai(_make_payload("ABS", imb=-0.47, bsr=0.36))
        assert r4 is True, "W4: Important event should pass"

        # W5: 30s depois de W4 (dentro do hard min)
        t._last_call_ts = now - 30
        r5 = t.should_call_ai(_make_payload("AT", imb=0.48, bsr=2.81))
        assert r5 is False, "W5: Inside hard_min should be blocked"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
