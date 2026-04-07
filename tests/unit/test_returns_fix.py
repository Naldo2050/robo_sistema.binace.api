# -*- coding: utf-8 -*-
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml"))

from returns_validator import ReturnsValidator


class TestReturnsValidator:
    def test_return_1_basic(self):
        prices = np.array([100.0, 101.0])
        ret, valid, reason = ReturnsValidator.validate_return(102.0, prices, 1)
        expected = 102.0 / 101.0 - 1.0
        assert ret == pytest.approx(expected, abs=0.001)
        assert valid is True
        assert reason == "VALID"

    def test_return_1_warmup_single_price(self):
        prices = np.array([100.0])
        ret, valid, reason = ReturnsValidator.validate_return(101.0, prices, 1)
        assert ret == 0.0
        assert valid is True
        assert reason == "WARMUP"

    def test_return_5_adaptive_window_short(self):
        prices = np.array([100.0, 100.5, 101.0])
        ret, valid, reason = ReturnsValidator.validate_return(101.5, prices, 5)
        expected = 101.5 / 100.5 - 1.0
        assert ret == pytest.approx(expected, abs=0.001)
        assert valid is True
        assert reason == "VALID"

    def test_return_5_full_window(self):
        prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        ret, valid, reason = ReturnsValidator.validate_return(103.0, prices, 5)
        expected = 103.0 / 100.5 - 1.0
        assert ret == pytest.approx(expected, abs=0.001)
        assert valid is True

    def test_return_10_adaptive_window_short(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
        ret, valid, reason = ReturnsValidator.validate_return(107.0, prices, 10)
        expected = 107.0 / 101.0 - 1.0
        assert ret == pytest.approx(expected, abs=0.001)
        assert valid is True

    def test_empty_prices(self):
        prices = np.array([])
        ret, valid, reason = ReturnsValidator.validate_return(100.0, prices, 1)
        assert ret == 0.0
        assert valid is True
        assert reason == "EMPTY"

    def test_invalid_current_price_nan(self):
        prices = np.array([100.0, 101.0])
        ret, valid, reason = ReturnsValidator.validate_return(np.nan, prices, 1)
        assert ret == 0.0
        assert valid is False
        assert reason == "INVALID_PRICE"

    def test_invalid_historical_price_nan(self):
        prices = np.array([100.0, np.nan])
        ret, valid, reason = ReturnsValidator.validate_return(102.0, prices, 1)
        assert ret == 0.0
        assert valid is False

    def test_zero_price(self):
        prices = np.array([100.0, 101.0])
        ret, valid, reason = ReturnsValidator.validate_return(102.0, prices, 1)
        assert valid is True
        assert ret != 0.0

    def test_extreme_return_clamped(self):
        prices = np.array([100.0, 101.0])
        ret, valid, reason = ReturnsValidator.validate_return(251.0, prices, 1)
        assert ret == pytest.approx(0.5, abs=0.01)
        assert valid is True

    def test_validate_all_returns(self):
        prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        result = ReturnsValidator.validate_all_returns(103.0, prices)
        
        assert "return_1" in result
        assert "return_5" in result
        assert "return_10" in result
        
        assert result["return_1"]["is_valid"] is True
        assert result["return_1"]["adaptive_window"] == 1
        assert result["return_1"]["value"] != 0.0
        
        assert result["return_5"]["is_valid"] is True
        assert result["return_5"]["value"] != 0.0
        
        assert result["return_10"]["is_valid"] is True
        assert result["return_10"]["adaptive_window"] == 5

    def test_returns_warmup_progression(self):
        prices_start = np.array([100.0])
        prices_2 = np.array([100.0, 101.0])
        prices_5 = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        
        result_1 = ReturnsValidator.validate_all_returns(100.5, prices_start)
        assert result_1["return_1"]["reason"] == "WARMUP"
        
        result_2 = ReturnsValidator.validate_all_returns(102.0, prices_2)
        assert result_2["return_1"]["is_valid"] is True
        assert result_2["return_1"]["value"] > 0.0  # Positive return (102 > 101)
        
        result_5 = ReturnsValidator.validate_all_returns(103.0, prices_5)
        assert result_5["return_1"]["is_valid"] is True
        assert result_5["return_5"]["is_valid"] is True

    def test_negative_return(self):
        prices = np.array([100.0, 101.0])
        ret, valid, reason = ReturnsValidator.validate_return(99.0, prices, 1)
        expected = 99.0 / 101.0 - 1.0
        assert ret == pytest.approx(expected, abs=0.001)
        assert valid is True
        assert ret < 0
