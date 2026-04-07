# -*- coding: utf-8 -*-
"""
clock_sync_circuit_breaker.py - v1.0.0
Clock Synchronization Circuit Breaker for Binance API Protection
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from threading import RLock  # Use RLock instead of Lock for reentrant support
from enum import Enum


class ClockSyncCircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is OPEN (critical offset)."""
    def __init__(self, offset_ms: int, critical_ms: int = 1000):
        self.offset_ms = offset_ms
        self.critical_ms = critical_ms
        msg = (
            f"ClockSync Circuit Breaker OPEN: "
            f"offset={abs(offset_ms)}ms exceeds critical limit {critical_ms}ms. "
            f"Orders are blocked until synchronization recovers."
        )
        super().__init__(msg)


class CircuitBreakerState(Enum):
    """Possible states of the circuit breaker."""
    CLOSED = "closed"        # Normal, allow operations
    OPEN = "open"            # Critical offset, block operations
    HALF_OPEN = "half_open"  # Attempting recovery


class ClockSyncCircuitBreaker:
    """
    Circuit Breaker for Clock Synchronization with Binance.
    
    Monitors offset and blocks operations when offset exceeds 1000ms
    (Binance technical limit for order placement).
    """
    
    # Configuration constants
    CRITICAL_OFFSET_MS = 1000
    RECOVERY_OFFSET_MS = 600
    HALF_OPEN_TIMEOUT_SECONDS = 10.0
    HALF_OPEN_MAX_ATTEMPTS = 3
    
    def __init__(self,
                 critical_offset_ms: int = 1000,
                 recovery_offset_ms: int = 600,
                 half_open_timeout_seconds: float = 10.0,
                 half_open_max_attempts: int = 3,
                 logger: Optional[logging.Logger] = None):
        """Initialize Circuit Breaker."""
        self._state = CircuitBreakerState.CLOSED
        self._critical_offset_ms = critical_offset_ms
        self._recovery_offset_ms = recovery_offset_ms
        self._half_open_timeout = half_open_timeout_seconds
        self._half_open_max_attempts = half_open_max_attempts
        
        self._last_open_time: Optional[float] = None
        self._half_open_attempts = 0
        self._failure_count = 0
        self._last_offset_ms = 0
        self._state_change_time: Optional[float] = None
        
        self._lock = RLock()  # Reentrant lock allows same thread to acquire multiple times
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(
            f"🔌 ClockSyncCircuitBreaker initialized. "
            f"Critical: {critical_offset_ms}ms, Recovery: {recovery_offset_ms}ms"
        )
    
    def can_operate(self) -> bool:
        """Check if operations (orders) are permitted."""
        with self._lock:
            return self._state != CircuitBreakerState.OPEN
    
    def update_offset(self, offset_ms: int) -> Tuple[CircuitBreakerState, str]:
        """
        Update offset and perform state transitions.
        
        Returns:
            (new_state, transition_reason)
        """
        abs_offset = abs(offset_ms)
        
        with self._lock:
            old_state = self._state
            self._last_offset_ms = offset_ms
            
            # CLOSED → OPEN on critical offset
            if old_state == CircuitBreakerState.CLOSED:
                if abs_offset > self._critical_offset_ms:
                    self._state = CircuitBreakerState.OPEN
                    self._last_open_time = time.time()
                    self._failure_count = 1
                    self._state_change_time = time.time()
                    reason = f"Offset {abs_offset}ms > critical {self._critical_offset_ms}ms"
                    self.logger.critical(
                        f"🔴 ClockSync Circuit Breaker OPEN! {reason}. Orders blocked."
                    )
                    return (self._state, reason)
            
            # HALF_OPEN → CLOSED when offset recovers
            elif old_state == CircuitBreakerState.HALF_OPEN:
                if abs_offset <= self._recovery_offset_ms:
                    self._state = CircuitBreakerState.CLOSED
                    self._state_change_time = time.time()
                    reason = f"Offset recovered to {abs_offset}ms"
                    self.logger.info(f"✅ ClockSync Circuit Breaker CLOSED. {reason}")
                    return (self._state, reason)
                
                # HALF_OPEN → OPEN if offset worsens
                if abs_offset > self._critical_offset_ms:
                    self._state = CircuitBreakerState.OPEN
                    self._last_open_time = time.time()
                    self._failure_count += 1
                    self._state_change_time = time.time()
                    reason = f"Offset worsened to {abs_offset}ms in HALF_OPEN"
                    self.logger.warning(f"🔴 ClockSync Circuit Breaker reopening. {reason}.")
                    return (self._state, reason)
            
            return (self._state, "no_change")
    
    def try_half_open(self) -> bool:
        """Attempt transition from OPEN to HALF_OPEN if timeout expired."""
        with self._lock:
            if self._state != CircuitBreakerState.OPEN:
                return False
            
            if self._last_open_time is None:
                return False
            
            elapsed = time.time() - self._last_open_time
            if elapsed < self._half_open_timeout:
                return False
            
            self._state = CircuitBreakerState.HALF_OPEN
            self._half_open_attempts = 0
            self._state_change_time = time.time()
            self.logger.warning(
                f"🟡 ClockSync Circuit Breaker HALF_OPEN. Attempting recovery."
            )
            return True
    
    def can_place_order(self, offset_ms: int) -> Tuple[bool, str]:
        """
        Check if an order can be placed given current offset.
        
        Returns:
            (allowed, reason)
        """
        abs_offset = abs(offset_ms)
        
        with self._lock:
            state = self._state
            
            if state == CircuitBreakerState.CLOSED:
                if abs_offset <= self._critical_offset_ms:
                    return (True, "OK (CLOSED)")
                else:
                    return (False, f"offset {abs_offset}ms exceeds critical limit")
            
            elif state == CircuitBreakerState.OPEN:
                return (False, f"CB OPEN (offset {abs_offset}ms > {self._critical_offset_ms}ms)")
            
            elif state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_attempts >= self._half_open_max_attempts:
                    return (False, "HALF_OPEN max attempts exceeded")
                
                if abs_offset <= self._critical_offset_ms:
                    self._half_open_attempts += 1
                    return (True, f"OK (HALF_OPEN {self._half_open_attempts}/{self._half_open_max_attempts})")
                else:
                    return (False, f"offset {abs_offset}ms exceeds limit in HALF_OPEN")
        
        return (False, "Unknown state")
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    def get_state_name(self) -> str:
        """Get human-readable state name."""
        return self.get_state().value.upper()
    
    def get_status(self) -> Dict:
        """Get complete circuit breaker status."""
        with self._lock:
            elapsed_in_state = None
            if self._state_change_time is not None:
                elapsed_in_state = time.time() - self._state_change_time
            
            time_to_half_open = None
            if (self._last_open_time and 
                self._state == CircuitBreakerState.OPEN):
                time_to_half_open = max(
                    0,
                    self._half_open_timeout - (time.time() - self._last_open_time)
                )
            
            return {
                "state": self._state.value,
                "state_name": self._state.value.upper(),
                "last_offset_ms": int(self._last_offset_ms),
                "critical_offset_ms": int(self._critical_offset_ms),
                "recovery_offset_ms": int(self._recovery_offset_ms),
                "can_operate": self._state != CircuitBreakerState.OPEN,  # Avoid recursion
                "failure_count": int(self._failure_count),
                "half_open_attempts": int(self._half_open_attempts),
                "half_open_max_attempts": int(self._half_open_max_attempts),
                "elapsed_in_state_seconds": elapsed_in_state,
                "time_to_half_open_seconds": time_to_half_open,
            }
    
    def reset(self):
        """Reset circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._half_open_attempts = 0
            self._last_open_time = None
            self._state_change_time = time.time()
            self.logger.info("🔄 ClockSync Circuit Breaker RESET to CLOSED")
    
    def diagnose(self) -> Dict:
        """Get diagnostic information for monitoring."""
        status = self.get_status()
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "circuit_breaker": status,
            "health": {
                "is_healthy": status["can_operate"],
                "reason": (
                    "HEALTHY (CLOSED)" if status["state"] == "closed"
                    else f"DEGRADED ({status['state'].upper()})"
                ),
            },
        }
