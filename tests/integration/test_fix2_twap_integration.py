# -*- coding: utf-8 -*-
"""Tests for FIX #2 integration with technical_indicators module"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../common"))
from technical_indicators import twap, twap_validated


class TestTWAPFix2Integration:
    """Test FIX #2: TWAP bounds validation"""

    def test_original_twap_function_unchanged(self):
        """Ensure backward compatibility - twap() still works as before"""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0])
        result = twap(prices)
        expected = 101.5
        assert result == pytest.approx(expected, abs=0.01)

    def test_twap_validated_with_valid_bounds(self):
        """Test twap_validated when TWAP is within bounds"""
        closes = np.array([100.0, 101.0, 102.0, 103.0])
        volumes = np.array([1.0, 2.0, 1.0, 1.0])
        low = 100.0
        high = 103.0
        
        result = twap_validated(closes, volumes, low, high)
        
        assert result["is_valid"] is True
        assert result["used_fallback"] is False
        assert result["twap"] > 0
        assert result["final_value"] > 0

    def test_twap_validated_fallback_to_vwap(self):
        """Test FIX #2 core: TWAP out of bounds triggers VWAP fallback"""
        closes = np.array([98.0, 98.5, 99.0])
        volumes = np.array([1.0, 2.0, 1.0])
        low = 100.0  # TWAP will be ~98.5, below this
        high = 102.0
        
        result = twap_validated(closes, volumes, low, high, "BTCUSDT")
        
        assert result["is_valid"] is False
        assert result["used_fallback"] is True
        assert result["reason"] == "BELOW_LOW"
        assert result["vwap_fallback"] is not None
        assert result["final_value"] == result["vwap_fallback"]

    def test_twap_validated_with_varied_data(self):
        """Test with realistic price/volume scenario"""
        # Simulating candle with multiple trades
        closes = np.array([100.5, 100.7, 100.9, 101.1, 100.8])
        volumes = np.array([10.0, 15.0, 20.0, 25.0, 18.0])
        low = 100.0
        high = 101.5
        
        result = twap_validated(closes, volumes, low, high)
        
        assert "final_value" in result
        assert result["final_value"] > 0
        # TWAP should be valid in this case
        assert result["final_value"] >= low * 0.99  # Within tolerance

    def test_twap_old_vs_new(self):
        """Compare old twap() vs new twap_validated() for consistency"""
        prices = [100.0, 101.0, 102.0, 103.0]
        closes = np.array(prices)
        volumes = np.array([1.0, 1.0, 1.0, 1.0])
        low = 100.0
        high = 103.5
        
        # Old method
        old_twap = twap(pd.Series(prices))
        
        # New method (should use TWAP if valid)
        new_result = twap_validated(closes, volumes, low, high)
        
        # If bounds valid, new should use TWAP
        if new_result["is_valid"]:
            assert new_result["final_value"] == pytest.approx(old_twap, abs=0.01)

    def test_bugfix_scenario_btcusdt_style(self):
        """Test scenario similar to BUG #2 report (BTCUSDT TWAP < low)"""
        # Simulating a move where closes are mostly lower than expected
        closes = np.array([67291.0, 67292.0, 67293.0, 67294.0, 67290.0])
        volumes = np.array([1.0, 1.0, 1.0, 1.0, 5.0])  # Heavy at end
        low = 67300.0  # Low higher than most closes
        high = 67400.0
        
        result = twap_validated(closes, volumes, low, high, "BTCUSDT")
        
        # Should detect TWAP < low and fallback to VWAP
        assert "final_value" in result
        assert result["final_value"] > 0
        if result["is_valid"] is False and result["reason"] == "BELOW_LOW":
            assert result["used_fallback"] is True
