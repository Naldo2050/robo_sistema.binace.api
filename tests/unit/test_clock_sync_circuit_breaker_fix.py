# -*- coding: utf-8 -*-
"""
tests/unit/test_clock_sync_circuit_breaker_fix.py

Unit tests for ClockSyncCircuitBreaker - FIX #4

Tests cover:
- State machine transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- Offset monitoring and blocking
- Half-open recovery mechanism
- Status reporting
"""

import pytest
import time
import logging
from monitoring.clock_sync_circuit_breaker import (
    ClockSyncCircuitBreaker,
    ClockSyncCircuitBreakerOpenError,
    CircuitBreakerState,
)


class TestClockSyncCircuitBreakerInitialization:
    """Test circuit breaker initialization."""
    
    def test_initialize_closed_state(self):
        """CB should initialize in CLOSED state."""
        cb = ClockSyncCircuitBreaker()
        assert cb.get_state() == CircuitBreakerState.CLOSED
        assert cb.can_operate() is True
    
    def test_custom_critical_offset(self):
        """CB should accept custom critical offset."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=800)
        status = cb.get_status()
        assert status["critical_offset_ms"] == 800
    
    def test_custom_recovery_offset(self):
        """CB should accept custom recovery offset."""
        cb = ClockSyncCircuitBreaker(recovery_offset_ms=400)
        status = cb.get_status()
        assert status["recovery_offset_ms"] == 400


class TestClockSyncCircuitBreakerClosedState:
    """Test CB behavior in CLOSED state."""
    
    def test_can_operate_when_closed(self):
        """CB should allow operations in CLOSED."""
        cb = ClockSyncCircuitBreaker()
        assert cb.can_operate() is True
    
    def test_good_offset_stays_closed(self):
        """CB should stay CLOSED with offset < critical."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Offset 500ms should keep CLOSED
        new_state, reason = cb.update_offset(500)
        assert new_state == CircuitBreakerState.CLOSED
        assert reason == "no_change"
    
    def test_open_transition_on_critical_offset(self):
        """CB should transition CLOSED → OPEN on critical offset."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Offset 1100ms should trigger OPEN
        new_state, reason = cb.update_offset(1100)
        assert new_state == CircuitBreakerState.OPEN
        assert "critical" in reason.lower()  # Changed to match actual message format
    
    def test_negative_offset_critical_check(self):
        """CB should handle negative offsets correctly."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Offset -1100ms should also trigger OPEN
        new_state, reason = cb.update_offset(-1100)
        assert new_state == CircuitBreakerState.OPEN


class TestClockSyncCircuitBreakerOpenState:
    """Test CB behavior in OPEN state."""
    
    def test_cannot_operate_when_open(self):
        """CB should block operations in OPEN."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        # Trigger OPEN
        cb.update_offset(1100)
        assert cb.can_operate() is False
    
    def test_can_place_order_blocked_in_open(self):
        """can_place_order() should return False in OPEN."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        
        cb.update_offset(1100)
        allowed, reason = cb.can_place_order(1100)
        assert allowed is False
        assert "OPEN" in reason
    
    def test_open_state_name(self):
        """get_state_name() should return readable state."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        cb.update_offset(1100)
        assert cb.get_state_name() == "OPEN"


class TestClockSyncCircuitBreakerHalfOpenState:
    """Test CB behavior in HALF_OPEN state."""
    
    def test_transition_to_half_open(self):
        """CB should transition OPEN → HALF_OPEN after timeout."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            half_open_timeout_seconds=0.1
        )
        
        cb.update_offset(1100)
        assert cb.get_state() == CircuitBreakerState.OPEN
        
        time.sleep(0.15)
        success = cb.try_half_open()
        assert success is True
        assert cb.get_state() == CircuitBreakerState.HALF_OPEN
    
    def test_half_open_recovery_to_closed(self):
        """CB should transition HALF_OPEN → CLOSED when offset recovers."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            recovery_offset_ms=600,
            half_open_timeout_seconds=0.05
        )
        
        cb.update_offset(1100)
        time.sleep(0.1)
        cb.try_half_open()
        assert cb.get_state() == CircuitBreakerState.HALF_OPEN
        
        new_state, reason = cb.update_offset(500)
        assert new_state == CircuitBreakerState.CLOSED
        assert "recovered" in reason.lower()
    
    def test_half_open_reopen_on_bad_offset(self):
        """CB should transition HALF_OPEN → OPEN if offset stays bad."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            half_open_timeout_seconds=0.05
        )
        
        cb.update_offset(1100)
        time.sleep(0.1)
        cb.try_half_open()
        
        new_state, reason = cb.update_offset(1200)
        assert new_state == CircuitBreakerState.OPEN
        assert "worsened" in reason.lower()


class TestClockSyncCircuitBreakerCanPlaceOrder:
    """Test can_place_order() method."""
    
    def test_can_place_order_in_closed(self):
        """can_place_order() should return True in CLOSED with good offset."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        allowed, reason = cb.can_place_order(500)
        assert allowed is True
        assert "OK" in reason
    
    def test_cannot_place_order_in_open(self):
        """can_place_order() should return False in OPEN."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        cb.update_offset(1100)
        
        allowed, reason = cb.can_place_order(1100)
        assert allowed is False
    
    def test_can_place_order_half_open_limited(self):
        """can_place_order() in HALF_OPEN should allow limited attempts."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            half_open_timeout_seconds=0.05,
            half_open_max_attempts=2
        )
        
        cb.update_offset(1100)
        time.sleep(0.1)
        cb.try_half_open()
        
        # First attempt OK
        allowed1, reason1 = cb.can_place_order(800)
        assert allowed1 is True
        assert "HALF_OPEN 1/2" in reason1
        
        # Second attempt OK
        allowed2, reason2 = cb.can_place_order(800)
        assert allowed2 is True
        assert "HALF_OPEN 2/2" in reason2
        
        # Third attempt blocked
        allowed3, reason3 = cb.can_place_order(800)
        assert allowed3 is False
        assert "max attempts" in reason3.lower()


class TestClockSyncCircuitBreakerReset:
    """Test reset functionality."""
    
    def test_reset_to_closed(self):
        """reset() should return CB to CLOSED."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        cb.update_offset(1100)
        assert cb.get_state() == CircuitBreakerState.OPEN
        
        cb.reset()
        assert cb.get_state() == CircuitBreakerState.CLOSED
        assert cb.can_operate() is True
    
    def test_reset_clears_counters(self):
        """reset() should clear failure counters."""
        cb = ClockSyncCircuitBreaker(critical_offset_ms=1000)
        cb.update_offset(1100)
        
        status_before = cb.get_status()
        assert status_before["failure_count"] > 0
        
        cb.reset()
        status_after = cb.get_status()
        assert status_after["failure_count"] == 0


class TestClockSyncCircuitBreakerStatus:
    """Test status and diagnostic methods."""
    
    def test_get_status_complete(self):
        """get_status() should return all required fields."""
        cb = ClockSyncCircuitBreaker()
        status = cb.get_status()
        
        required_fields = [
            "state", "state_name", "last_offset_ms", "can_operate",
            "failure_count", "half_open_attempts", "critical_offset_ms"
        ]
        for field in required_fields:
            assert field in status
    
    def test_diagnose_complete(self):
        """diagnose() should return health info."""
        cb = ClockSyncCircuitBreaker()
        cb.update_offset(1100)
        diag = cb.diagnose()
        
        assert "circuit_breaker" in diag
        assert "health" in diag
        assert "timestamp_utc" in diag
        assert diag["health"]["is_healthy"] is False


class TestClockSyncCircuitBreakerIntegration:
    """Integration tests for realistic scenarios."""
    
    def test_realistic_offset_spike_recovery(self):
        """Test realistic spike and recovery scenario."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            recovery_offset_ms=600,
            half_open_timeout_seconds=0.1
        )
        
        # Phase 1: Normal operation
        cb.update_offset(300)
        assert cb.can_operate() is True
        
        # Phase 2: Offset spike (network issue)
        cb.update_offset(1200)
        assert cb.can_operate() is False
        
        # Phase 3: Wait for HALF_OPEN
        time.sleep(0.15)
        cb.try_half_open()
        assert cb.get_state() == CircuitBreakerState.HALF_OPEN
        
        # Phase 4: Recovery
        cb.update_offset(500)
        assert cb.get_state() == CircuitBreakerState.CLOSED
        assert cb.can_operate() is True
    
    def test_multiple_spikes(self):
        """Test multiple offset spikes."""
        cb = ClockSyncCircuitBreaker(
            critical_offset_ms=1000,
            half_open_timeout_seconds=0.05
        )
        
        # First spike
        cb.update_offset(1100)
        assert cb.get_state() == CircuitBreakerState.OPEN
        
        # Recover
        time.sleep(0.1)
        cb.try_half_open()
        cb.update_offset(300)
        assert cb.get_state() == CircuitBreakerState.CLOSED
        
        # Second spike
        cb.update_offset(1150)
        assert cb.get_state() == CircuitBreakerState.OPEN


class TestClockSyncCircuitBreakerException:
    """Test exception class."""
    
    def test_exception_contains_offset(self):
        """Exception should contain offset information."""
        exc = ClockSyncCircuitBreakerOpenError(offset_ms=1100, critical_ms=1000)
        assert exc.offset_ms == 1100
        assert exc.critical_ms == 1000
        assert "1100" in str(exc)
        assert "1000" in str(exc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
