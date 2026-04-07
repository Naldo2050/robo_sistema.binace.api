# tests/integration/test_fix5_macro_data_integration.py (FIXED)
"""Integration tests for MacroCacheValidator with MacroDataProvider - FIX #5"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fetchers.macro_cache_validator import MacroCacheValidator


class TestMacroCacheValidatorIntegration:
    """Integration tests for cache validator with macro data provider."""
    
    @pytest.mark.asyncio
    async def test_validator_tracks_oil_fetch_failure_then_success(self):
        """Test: Oil fetch fails 3x, cache invalidated, then succeeds."""
        validator = MacroCacheValidator()
        
        # Simulate 3 consecutive failures
        validator.record_failure("oil")
        validator.record_failure("oil")
        assert validator._cache_invalid.get("oil", False) is False
        
        validator.record_failure("oil")  # Third failure marks invalid
        assert validator._cache_invalid["oil"] is True
        
        # Check cache validity
        is_valid, reason = validator.is_cache_valid("oil", 113.64, time.time())
        assert is_valid is False
        assert "invalid" in reason
        
        # Fetch succeeds
        validator.record_success("oil")
        assert validator._failure_counters["oil"] == 0
        assert validator._cache_invalid["oil"] is False
    
    @pytest.mark.asyncio
    async def test_validator_detects_stale_oil_cache(self):
        """Test: Oil cache > 30min should be rejected."""
        validator = MacroCacheValidator()
        validator.record_success("oil")
        
        # Simulate 31 minutes old cache
        old_time = time.time() - (31 * 60)
        is_valid, reason = validator.is_cache_valid("oil", 113.64, old_time)
        assert is_valid is False
        assert "too old" in reason
    
    @pytest.mark.asyncio
    async def test_validator_detects_stale_vix_cache_faster(self):
        """Test: VIX cache > 5min should be rejected (faster than oil/sp500)."""
        validator = MacroCacheValidator()
        
        # Simulate 6 minutes old VIX cache
        old_time = time.time() - (6 * 60)
        is_valid, reason = validator.is_cache_valid("vix", 20.5, old_time)
        assert is_valid is False
        assert "too old" in reason or "invalid" in reason
    
    @pytest.mark.asyncio
    async def test_validator_sp500_follows_same_pattern_as_oil(self):
        """Test: SP500 should follow same failure tracking as oil."""
        validator = MacroCacheValidator()
        
        # Simulate fetch failures
        validator.record_failure("sp500")
        validator.record_failure("sp500")
        validator.record_failure("sp500")
        
        # Cache should be invalid
        assert validator._cache_invalid["sp500"] is True
        is_valid, reason = validator.is_cache_valid("sp500", 500.0, time.time())
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validator_recovery_scenario(self):
        """Test: After failures, successful fetch resets state."""
        validator = MacroCacheValidator()
        
        # Accumulate failures
        validator.record_failure("oil")
        validator.record_failure("oil")
        validator.record_failure("oil")
        assert validator._cache_invalid["oil"] is True
        
        # Successful refetch
        validator.record_success("oil")
        assert validator._cache_invalid["oil"] is False
        assert validator._failure_counters["oil"] == 0
        
        # Cache should now be valid
        is_valid, reason = validator.is_cache_valid("oil", 100.5, time.time())
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validator_partial_failures_dont_mark_invalid(self):
        """Test: 1-2 failures don't mark cache invalid, only 3+ do."""
        validator = MacroCacheValidator()
        
        # First failure
        validator.record_failure("oil")
        assert validator._cache_invalid.get("oil", False) is False
        
        # Second failure
        validator.record_failure("oil")
        assert validator._cache_invalid.get("oil", False) is False
        
        # Cache should still be usable if value exists
        is_valid, reason = validator.is_cache_valid("oil", 100.5, time.time())
        assert is_valid is True  # Still valid despite 2 failures
        
        # Third failure marks invalid
        validator.record_failure("oil")
        assert validator._cache_invalid["oil"] is True
        is_valid, reason = validator.is_cache_valid("oil", 100.5, time.time())
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validator_diagnose_comprehensive(self):
        """Test: Diagnose produces comprehensive health report."""
        validator = MacroCacheValidator()
        
        # Setup different states for different assets
        validator.record_success("oil")  # Oil: healthy
        validator.record_failure("sp500")  # SPX: 1 failure
        validator.record_failure("sp500")
        validator.record_failure("sp500")  # SPX: 3 failures, invalid
        
        diag = validator.diagnose()
        
        # Check structure
        assert "timestamp_utc" in diag
        assert "cache_health" in diag
        
        # Oil should be healthy
        assert diag["cache_health"]["oil"]["is_invalid"] is False
        assert diag["cache_health"]["oil"]["consecutive_failures"] == 0
        
        # SPX should be invalid
        assert diag["cache_health"]["sp500"]["is_invalid"] is True
        assert diag["cache_health"]["sp500"]["consecutive_failures"] == 3
    
    @pytest.mark.asyncio
    async def test_validator_multiple_concurrent_assets(self):
        """Test: Validator can track multiple assets simultaneously."""
        validator = MacroCacheValidator()
        assets = ["oil", "sp500", "gold", "vix", "dxy"]
        
        # Simulate different states (0, 1, 2, 3, 4 failures respectively)
        for i, asset in enumerate(assets):
            for _ in range(i):  # oil: 0, sp500: 1, gold: 2, vix: 3, dxy: 4
                validator.record_failure(asset)
        
        # Verify each asset state
        assert validator._failure_counters.get("oil", 0) == 0
        assert validator._failure_counters.get("sp500", 0) == 1
        assert validator._failure_counters.get("gold", 0) == 2
        assert validator._failure_counters["vix"] == 3
        assert validator._cache_invalid["vix"] is True
        assert validator._failure_counters["dxy"] == 4
        assert validator._cache_invalid["dxy"] is True
    
    @pytest.mark.asyncio
    async def test_validator_reset_failures_all_keys(self):
        """Test: Can reset failures for all keys."""
        validator = MacroCacheValidator()
        
        # Accumulate failures across multiple keys
        validator.record_failure("oil")
        validator.record_failure("oil")
        validator.record_failure("sp500")
        validator.record_failure("sp500")
        validator.record_failure("sp500")
        
        # Mark one as invalid
        assert validator._cache_invalid.get("sp500", False) is True
        
        # Reset both
        validator.reset_failures("oil")
        validator.reset_failures("sp500")
        
        # Both should be clean
        assert validator._failure_counters["oil"] == 0
        assert validator._failure_counters["sp500"] == 0
        assert validator._cache_invalid["sp500"] is False
    
    @pytest.mark.asyncio
    async def test_validator_timing_age_measurement(self):
        """Test: Validator correctly measures and reports cache age."""
        validator = MacroCacheValidator()
        
        # Record success to establish cache
        validator.record_success("oil")
        current_time = time.time()
        
        # Test immediate age (should be ~0s)
        is_valid, reason = validator.is_cache_valid("oil", 100.5, current_time)
        assert is_valid is True
        assert "age: 0s" in reason  # Age should be approximately 0 seconds
        
        # Test with older timestamp
        old_time = current_time - 600  # 10 minutes old
        is_valid, reason = validator.is_cache_valid("oil", 100.5, old_time)
        assert is_valid is True  # Still valid (within 30 min limit)
        assert "600" in reason or "60" in reason  # Age should be around 600s
    
    @pytest.mark.asyncio
    async def test_validator_should_refetch_logic(self):
        """Test: should_refetch correctly identifies when to refetch."""
        validator = MacroCacheValidator()
        
        # Healthy cache
        should_fetch, reason = validator.should_refetch("oil", time.time())
        assert should_fetch is False
        
        # With failures
        validator.record_failure("oil")
        should_fetch, reason = validator.should_refetch("oil", time.time())
        assert should_fetch is True
        assert "recovering" in reason
        
        # Mark invalid
        validator.invalidate_cache("oil")
        should_fetch, reason = validator.should_refetch("oil", time.time())
        assert should_fetch is True
        assert "invalid" in reason
