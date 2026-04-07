# -*- coding: utf-8 -*-
"""
tests/integration/test_fix4_clock_sync_circuit_breaker_integration.py

Integration tests for ClockSyncCircuitBreaker with TimeManager

Tests cover:
- Integration with TimeManager's offset monitoring
- Real clock synchronization scenarios
- Circuit breaker blocking orders when TimeManager shows degraded status
"""

import pytest
import time
import logging
from monitoring.clock_sync_circuit_breaker import (
    ClockSyncCircuitBreaker,
    CircuitBreakerState,
)
from monitoring.time_manager import TimeManager


class TestClockSyncCircuitBreakerWithTimeManager:
    """Integration tests with TimeManager."""
    
    def test_circuit_breaker_monitors_timemanager_offset(self):
        """CB should monitor offset from TimeManager."""
        tm = TimeManager(max_acceptable_offset_ms=600)
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000, recovery_offset_ms=600)
        
        # TimeManager starts with offset = 0 (init)
        status_tm = tm.get_sync_stats()
        initial_offset = status_tm.get("server_time_offset_ms", 0)
        
        # Update CB with TimeManager's offset
        cb.update_offset(initial_offset)
        assert cb.can_operate() is True
        
        # Simulate critical offset
        cb.update_offset(1050)
        assert cb.can_operate() is False
        assert cb.get_state() == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_respects_binance_limit(self):
        """CB should enforce Binance's 1000ms limit."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Below limit: OK
        cb.update_offset(999)
        assert cb.can_operate() is True
        
        # At limit: still OK (boundary)
        cb.update_offset(1000)
        assert cb.can_operate() is True
        
        # Above limit: BLOCKED
        cb.update_offset(1001)
        assert cb.can_operate() is False
    
    def test_circuit_breaker_recovery_window(self):
        """CB should stay OPEN until offset recovers to threshold."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            recovery_offset_ms=600,
            half_open_timeout_seconds=0.05
        )
        
        # Trigger OPEN
        cb.update_offset(1100)
        assert cb.get_state() == CircuitBreakerState.OPEN
        
        # Intermediate recovery (above threshold) shouldn't close
        time.sleep(0.1)
        cb.try_half_open()
        cb.update_offset(700)
        assert cb.get_state() == CircuitBreakerState.HALF_OPEN
        
        # Further recovery should close
        cb.update_offset(500)
        assert cb.get_state() == CircuitBreakerState.CLOSED


class TestClockSyncCircuitBreakerOrderBlocking:
    """Test order blocking behavior."""
    
    def test_order_blocking_when_open(self):
        """Orders should be blocked when CB is OPEN."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Simulate order placement attempts
        can_place, reason = cb.can_place_order(500)
        assert can_place is True
        
        # Trigger OPEN
        cb.update_offset(1100)
        can_place, reason = cb.can_place_order(1100)
        assert can_place is False
        assert "OPEN" in reason
    
    def test_order_allowed_after_recovery(self):
        """Orders should be allowed after recovery."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            recovery_offset_ms=600,
            half_open_timeout_seconds=0.05
        )
        
        # Spike and recover
        cb.update_offset(1100)
        time.sleep(0.1)
        cb.try_half_open()
        cb.update_offset(400)  # Well below threshold
        
        # Orders should work again
        can_place, reason = cb.can_place_order(400)
        assert can_place is True
        assert "OK" in reason


class TestClockSyncCircuitBreakerMetrics:
    """Test metrics and monitoring."""
    
    def test_failure_counter_on_open(self):
        """CB should track failure count."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        status = cb.get_status()
        assert status["failure_count"] == 0
        
        # Trigger OPEN
        cb.update_offset(1100)
        status = cb.get_status()
        assert status["failure_count"] > 0
    
    def test_state_change_tracking(self):
        """CB should track elapsed time in state."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # CLOSED state
        status = cb.get_status()
        assert status["elapsed_in_state_seconds"] is not None
        
        # Trigger OPEN
        cb.update_offset(1100)
        initial_time = time.time()
        
        time.sleep(0.05)
        status = cb.get_status()
        elapsed = status["elapsed_in_state_seconds"]
        assert elapsed is not None
        assert elapsed >= 0.04  # At least ~50ms


class TestClockSyncCircuitBreakerStateTransitions:
    """Test complete state machine lifecycle."""
    
    def test_full_lifecycle_closed_open_half_open_closed(self):
        """Test complete state cycle."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            recovery_offset_ms=600,
            half_open_timeout_seconds=0.08
        )
        
        # Start: CLOSED
        assert cb.get_state() == CircuitBreakerState.CLOSED
        
        # Bad offset: OPEN
        cb.update_offset(1100)
        assert cb.get_state() == CircuitBreakerState.OPEN
        assert not cb.can_operate()
        
        # Wait & attempt recovery: HALF_OPEN
        time.sleep(0.1)
        cb.try_half_open()
        assert cb.get_state() == CircuitBreakerState.HALF_OPEN
        
        # Offset improves: CLOSED
        cb.update_offset(500)
        assert cb.get_state() == CircuitBreakerState.CLOSED
        assert cb.can_operate()


class TestClockSyncCircuitBreakerDiagnostics:
    """Test diagnostics and health reporting."""
    
    def test_diagnose_provides_completeness(self):
        """diagnose() should provide complete health picture."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Open the CB
        cb.update_offset(1100)
        
        diag = cb.diagnose()
        assert "timestamp_utc" in diag
        assert "circuit_breaker" in diag
        assert "health" in diag
        
        # Health should show degraded
        assert diag["health"]["is_healthy"] is False
        assert "DEGRADED" in diag["health"]["reason"]
    
    def test_status_boundary_values(self):
        """Status should reflect accurate boundary values."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            recovery_offset_ms=600
        )
        
        # Test exact critical value
        cb.update_offset(1000)
        status = cb.get_status()
        assert status["can_operate"] is True  # Boundary: == critical is OK
        
        # Test just above critical
        cb.update_offset(1001)
        status = cb.get_status()
        assert status["can_operate"] is False


class TestClockSyncCircuitBreakerReliability:
    """Test reliability under various conditions."""
    
    def test_continuous_offset_updates(self):
        """CB should handle continuous offset updates."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        for offset in [100, 200, 300, 500, 700, 900, 1100]:
            cb.update_offset(offset)
            status = cb.get_status()
            assert "state" in status
    
    def test_rapid_state_changes(self):
        """CB should handle rapid state transitions."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            half_open_timeout_seconds=0.01
        )
        
        # Rapid oscillation
        for _ in range(5):
            cb.update_offset(1100)  # OPEN
            assert cb.get_state() == CircuitBreakerState.OPEN
            
            time.sleep(0.02)
            cb.try_half_open()
            
            cb.update_offset(500)  # CLOSED
            assert cb.get_state() == CircuitBreakerState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
