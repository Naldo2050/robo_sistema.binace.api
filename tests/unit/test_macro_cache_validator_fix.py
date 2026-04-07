# tests/unit/test_macro_cache_validator_fix.py
"""Unit tests for MacroCacheValidator - FIX #5"""

import pytest
import time
from datetime import datetime
from fetchers.macro_cache_validator import MacroCacheValidator


class TestMacroCacheValidatorInitialization:
    """Test validator initialization."""
    
    def test_init_creates_empty_state(self):
        validator = MacroCacheValidator()
        assert validator._failure_counters == {}
        assert validator._cache_invalid == {}
        assert validator._last_success_time == {}
    
    def test_init_constants_defined(self):
        validator = MacroCacheValidator()
        assert validator.MAX_CONSECUTIVE_FAILURES == 3
        assert validator.MAX_STALE_TIME_SECONDS["oil"] == 1800
        assert validator.MAX_STALE_TIME_SECONDS["sp500"] == 1800
        assert validator.MAX_STALE_TIME_SECONDS["vix"] == 300


class TestRecordSuccess:
    """Test recording successful fetches."""
    
    def test_record_success_resets_failures(self):
        validator = MacroCacheValidator()
        validator._failure_counters["oil"] = 2
        validator.record_success("oil")
        assert validator._failure_counters["oil"] == 0
    
    def test_record_success_clears_invalid_flag(self):
        validator = MacroCacheValidator()
        validator._cache_invalid["oil"] = True
        validator.record_success("oil")
        assert validator._cache_invalid["oil"] is False
    
    def test_record_success_sets_timestamps(self):
        validator = MacroCacheValidator()
        before = time.time()
        validator.record_success("oil")
        after = time.time()
        assert before <= validator._last_success_time["oil"] <= after
    
    def test_record_success_multiple_keys(self):
        validator = MacroCacheValidator()
        validator.record_success("oil")
        validator.record_success("sp500")
        assert validator._failure_counters["oil"] == 0
        assert validator._failure_counters["sp500"] == 0


class TestRecordFailure:
    """Test recording fetch failures."""
    
    def test_record_failure_increments_counter(self):
        validator = MacroCacheValidator()
        validator.record_failure("oil")
        assert validator._failure_counters["oil"] == 1
        validator.record_failure("oil")
        assert validator._failure_counters["oil"] == 2
    
    def test_record_failure_marks_invalid_after_threshold(self):
        validator = MacroCacheValidator()
        # First 2 failures: should not mark invalid
        validator.record_failure("oil")
        assert validator._cache_invalid.get("oil", False) is False
        validator.record_failure("oil")
        assert validator._cache_invalid.get("oil", False) is False
        # Third failure: should mark invalid
        validator.record_failure("oil")
        assert validator._cache_invalid["oil"] is True
    
    def test_record_failure_sets_fetch_time(self):
        validator = MacroCacheValidator()
        before = time.time()
        validator.record_failure("oil")
        after = time.time()
        assert before <= validator._last_fetch_time["oil"] <= after


class TestIsCacheValid:
    """Test cache validity checking."""
    
    def test_cache_valid_with_good_value_and_time(self):
        validator = MacroCacheValidator()
        current_time = time.time()
        is_valid, reason = validator.is_cache_valid("oil", 100.5, current_time)
        assert is_valid is True
        assert "valid" in reason
    
    def test_cache_invalid_with_none_value(self):
        validator = MacroCacheValidator()
        is_valid, reason = validator.is_cache_valid("oil", None, time.time())
        assert is_valid is False
        assert "None" in reason
    
    def test_cache_invalid_when_marked_invalid(self):
        validator = MacroCacheValidator()
        validator._cache_invalid["oil"] = True
        is_valid, reason = validator.is_cache_valid("oil", 100.5, time.time())
        assert is_valid is False
        assert "invalid" in reason
    
    def test_cache_invalid_when_too_stale(self):
        validator = MacroCacheValidator()
        # Oil has 1800s max stale, simulate 2000s old cache
        old_time = time.time() - 2000
        is_valid, reason = validator.is_cache_valid("oil", 100.5, old_time)
        assert is_valid is False
        assert "too old" in reason
    
    def test_cache_valid_within_stale_threshold(self):
        validator = MacroCacheValidator()
        # Oil has 1800s max stale, simulate 1000s old cache
        old_time = time.time() - 1000
        is_valid, reason = validator.is_cache_valid("oil", 100.5, old_time)
        assert is_valid is True
        assert "valid" in reason
    
    def test_cache_invalid_vix_faster_stale_threshold(self):
        """VIX has 300s stale threshold, oil has 1800s."""
        validator = MacroCacheValidator()
        # 500s old cache
        old_time = time.time() - 500
        # Oil should be valid (500 < 1800)
        is_valid_oil, _ = validator.is_cache_valid("oil", 100.5, old_time)
        assert is_valid_oil is True
        # VIX should be invalid (500 > 300)
        is_valid_vix, _ = validator.is_cache_valid("vix", 20.5, old_time)
        assert is_valid_vix is False


class TestShouldRefetch:
    """Test refetch decision logic."""
    
    def test_should_refetch_when_cache_invalid(self):
        validator = MacroCacheValidator()
        validator._cache_invalid["oil"] = True
        should_refetch, reason = validator.should_refetch("oil", time.time())
        assert should_refetch is True
        assert "invalid" in reason
    
    def test_should_refetch_when_failures_accumulating(self):
        validator = MacroCacheValidator()
        validator._failure_counters["oil"] = 2
        should_refetch, reason = validator.should_refetch("oil", time.time())
        assert should_refetch is True
        assert "recovering" in reason
    
    def test_no_refetch_when_healthy(self):
        validator = MacroCacheValidator()
        should_refetch, reason = validator.should_refetch("oil", time.time())
        assert should_refetch is False
        assert "no refetch" in reason


class TestGetCacheHealth:
    """Test cache health diagnostics."""
    
    def test_get_cache_health_returns_dict(self):
        validator = MacroCacheValidator()
        validator._failure_counters["oil"] = 1
        health = validator.get_cache_health("oil")
        assert isinstance(health, dict)
        assert "key" in health
        assert "is_invalid" in health
        assert "consecutive_failures" in health
    
    def test_get_cache_health_accurate_data(self):
        validator = MacroCacheValidator()
        validator.record_failure("oil")
        validator.record_failure("oil")
        health = validator.get_cache_health("oil")
        assert health["consecutive_failures"] == 2
        assert health["is_invalid"] is False
        assert health["max_failures_allowed"] == 3


class TestResetFailures:
    """Test failure counter reset."""
    
    def test_reset_failures_clears_counter(self):
        validator = MacroCacheValidator()
        validator._failure_counters["oil"] = 2
        validator.reset_failures("oil")
        assert validator._failure_counters["oil"] == 0
    
    def test_reset_failures_clears_invalid_flag(self):
        validator = MacroCacheValidator()
        validator._cache_invalid["oil"] = True
        validator.reset_failures("oil")
        assert validator._cache_invalid["oil"] is False


class TestInvalidateCache:
    """Test manual cache invalidation."""
    
    def test_invalidate_cache_sets_flags(self):
        validator = MacroCacheValidator()
        validator.invalidate_cache("oil")
        assert validator._cache_invalid["oil"] is True
        assert validator._failure_counters["oil"] == 3


class TestDiagnose:
    """Test diagnostic output."""
    
    def test_diagnose_returns_dict(self):
        validator = MacroCacheValidator()
        validator.record_failure("oil")
        diag = validator.diagnose()
        assert isinstance(diag, dict)
        assert "timestamp_utc" in diag
        assert "cache_health" in diag
    
    def test_diagnose_includes_timestamp(self):
        validator = MacroCacheValidator()
        validator.record_failure("oil")
        diag = validator.diagnose()
        # Should be valid ISO format
        assert "T" in diag["timestamp_utc"]
        assert "Z" in diag["timestamp_utc"] or "+" in diag["timestamp_utc"]


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def test_scenario_fetch_fails_then_succeeds(self):
        """Simulate: fetch fails, cache marked stale, then succeeds."""
        validator = MacroCacheValidator()
        
        # Initial failure
        validator.record_failure("oil")
        assert validator._failure_counters["oil"] == 1
        
        # Try again - still failing
        validator.record_failure("oil")
        assert validator._failure_counters["oil"] == 2
        
        # Finally succeeds
        validator.record_success("oil")
        assert validator._failure_counters["oil"] == 0
        assert validator._cache_invalid["oil"] is False
    
    def test_scenario_cache_becomes_stale_after_failures(self):
        """Simulate: failures trigger cache invalidation, cache becomes stale."""
        validator = MacroCacheValidator()
        old_time = time.time() - 2000  # 2000s old
        
        # 3 failures mark cache invalid
        validator.record_failure("oil")
        validator.record_failure("oil")
        validator.record_failure("oil")
        
        # Cache should now be invalid
        is_valid, reason = validator.is_cache_valid("oil", 100.5, old_time)
        assert is_valid is False
    
    def test_scenario_multiple_assets(self):
        """Simulate: tracking multiple assets simultaneously."""
        validator = MacroCacheValidator()
        
        # Oil: 1 failure
        validator.record_failure("oil")
        # SPX: 3 failures (invalid)
        validator.record_failure("sp500")
        validator.record_failure("sp500")
        validator.record_failure("sp500")
        # Gold: success
        validator.record_success("gold")
        
        assert validator._failure_counters["oil"] == 1
        assert validator._cache_invalid["sp500"] is True
        assert validator._failure_counters["gold"] == 0
