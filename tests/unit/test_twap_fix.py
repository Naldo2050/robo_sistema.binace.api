# -*- coding: utf-8 -*-
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../common"))

from twap_validator import TWAPValidator


class TestTWAPValidator:
    """Tests for TWAP Fix #2 - Bounds Validation"""

    def test_twap_calculation_basic(self):
        closes = np.array([100.0, 101.0, 102.0, 103.0])
        twap = TWAPValidator.calculate_twap(closes)
        assert twap == pytest.approx(101.5, abs=0.01)

    def test_twap_with_nan_values(self):
        closes = np.array([100.0, np.nan, 102.0, 103.0])
        twap = TWAPValidator.calculate_twap(closes)
        assert twap == pytest.approx((100 + 102 + 103) / 3, abs=0.01)

    def test_twap_empty_array(self):
        closes = np.array([])
        twap = TWAPValidator.calculate_twap(closes)
        assert twap == 0.0

    def test_vwap_calculation_basic(self):
        closes = np.array([100.0, 102.0, 104.0])
        volumes = np.array([1.0, 2.0, 1.0])
        vwap = TWAPValidator.calculate_vwap(closes, volumes)
        assert vwap == pytest.approx(102.0, abs=0.01)

    def test_vwap_mismatched_lengths(self):
        closes = np.array([100.0, 102.0, 104.0, 105.0])
        volumes = np.array([1.0, 2.0, 1.0])
        vwap = TWAPValidator.calculate_vwap(closes, volumes)
        assert vwap > 0

    def test_validate_twap_bounds_valid(self):
        twap = 101.0
        low = 100.0
        high = 102.0
        
        twap_val, is_valid, reason = TWAPValidator.validate_twap_bounds(twap, low, high)
        assert is_valid is True
        assert reason == "VALID"

    def test_validate_twap_bounds_clearly_below_low(self):
        """Test TWAP clearly below low (beyond tolerance)"""
        twap = 98.0  # Well below low
        low = 100.0
        high = 102.0
        
        twap_val, is_valid, reason = TWAPValidator.validate_twap_bounds(twap, low, high)
        assert is_valid is False
        assert reason == "BELOW_LOW"

    def test_validate_twap_bounds_clearly_above_high(self):
        """Test TWAP clearly above high (beyond tolerance)"""
        twap = 104.0  # Well above high
        low = 100.0
        high = 102.0
        
        twap_val, is_valid, reason = TWAPValidator.validate_twap_bounds(twap, low, high)
        assert is_valid is False
        assert reason == "ABOVE_HIGH"

    def test_validate_twap_with_tolerance(self):
        """Test that tolerance is applied (1% buffer)"""
        twap = 102.005  # Slightly above high (within 1% tolerance)
        low = 100.0
        high = 102.0
        
        twap_val, is_valid, reason = TWAPValidator.validate_twap_bounds(twap, low, high)
        # upper_bound = 102 * 1.01 = 103.02
        assert is_valid is True

    def test_validate_twap_with_fallback_valid(self):
        """Test validate_twap_with_fallback when TWAP is valid"""
        closes = np.array([100.0, 101.0, 102.0])
        volumes = np.array([1.0, 2.0, 1.0])
        low = 100.0
        high = 102.0
        
        result = TWAPValidator.validate_twap_with_fallback(closes, volumes, low, high)
        
        assert result["is_valid"] is True
        assert result["used_fallback"] is False
        assert result["twap"] > 0
        assert result["final_value"] == result["twap"]

    def test_validate_twap_with_fallback_invalid_uses_vwap(self):
        """Test that invalid TWAP triggers VWAP fallback (FIX #2 core)"""
        closes = np.array([98.0, 98.5, 99.0])  # All below expected low
        volumes = np.array([1.0, 2.0, 1.0])
        low = 100.0  # TWAP will be ~98.5, clearly below
        high = 102.0
        
        result = TWAPValidator.validate_twap_with_fallback(closes, volumes, low, high)
        
        assert result["is_valid"] is False
        assert result["used_fallback"] is True
        assert result["reason"] == "BELOW_LOW"
        assert result["vwap_fallback"] is not None
        assert result["final_value"] == result["vwap_fallback"]

    def test_validate_twap_edge_case_equal_bounds(self):
        twap = 100.0
        low = 100.0
        high = 102.0
        
        twap_val, is_valid, reason = TWAPValidator.validate_twap_bounds(twap, low, high)
        assert is_valid is True

    def test_validate_twap_with_inverted_bounds(self):
        twap = 101.0
        low = 102.0
        high = 100.0
        
        twap_val, is_valid, reason = TWAPValidator.validate_twap_bounds(twap, low, high)
        assert is_valid is False
        assert reason == "INVERTED_BOUNDS"

    def test_bugfix_scenario_twap_fallback_to_vwap(self):
        """Test BUG #2 scenario: TWAP invalid, use VWAP"""
        closes = np.array([99.0, 99.5, 100.0, 100.5])
        volumes = np.array([1.0, 2.0, 1.0, 1.0])
        low = 100.5
        high = 102.0
        
        result = TWAPValidator.validate_twap_with_fallback(closes, volumes, low, high, "BTCUSDT")
        
        assert "final_value" in result
        assert result["final_value"] > 0
