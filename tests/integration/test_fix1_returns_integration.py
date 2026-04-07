# -*- coding: utf-8 -*-
"""Tests for FIX #1 integration with LiveFeatureCalculator"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml"))
from feature_calculator import LiveFeatureCalculator


class TestReturnsFix1Integration:
    """Test FIX #1: Returns are calculated with adaptive window"""

    def test_feature_calculator_with_returns_validator(self):
        """Test that LiveFeatureCalculator works with ReturnsValidator integrated"""
        calc = LiveFeatureCalculator()
        
        # Update with some prices
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        for price in prices:
            calc.update(price)
        
        # Should not raise exception
        features = calc.compute()
        
        # Basic checks
        assert features is not None
        assert isinstance(features, dict)
        assert 'return_1' in features
        assert 'return_5' in features
        assert 'return_10' in features

    def test_returns_should_be_calculated_not_always_zero(self):
        """Test that at least some returns are non-zero with varied prices"""
        calc = LiveFeatureCalculator()
        
        # Increasing prices
        prices = [100.0 + i for i in range(15)]
        for price in prices:
            calc.update(price)
        
        features = calc.compute()
        
        # With actual price changes, returns should exist
        assert 'return_1' in features
        assert 'return_5' in features
        assert 'return_10' in features
        # At least some should be non-zero (price increased)
        total_returns = features['return_1'] + features['return_5'] + features['return_10']
        assert total_returns > 0

    def test_stable_prices_give_zero_returns(self):
        """Test that with identical prices, returns are zero"""
        calc = LiveFeatureCalculator()
        
        # Same price repeated
        for _ in range(10):
            calc.update(100.0)
        
        features = calc.compute()
        
        # All should be or near zero
        assert abs(features['return_1']) < 0.001
        assert abs(features['return_5']) < 0.001
        assert abs(features['return_10']) < 0.001

    def test_declining_prices_give_negative_returns(self):
        """Test that with declining prices, returns are negative"""
        calc = LiveFeatureCalculator()
        
        # Declining prices
        for i in range(15, 0, -1):
            calc.update(100.0 + i)
        
        features = calc.compute()
        
        # Most returns should be negative
        neg_count = sum([features['return_1'] < 0, 
                        features['return_5'] < 0, 
                        features['return_10'] < 0])
        assert neg_count >= 2  # At least 2 negative

    def test_warmup_state_metadata(self):
        """Test that warmup metadata is present"""
        calc = LiveFeatureCalculator()
        
        calc.update(100.0)
        calc.update(101.0)
        
        features = calc.compute()
        
        # Check metadata
        assert '_warmup_ready' in features
        assert '_features_real_count' in features
        assert '_features_default_list' in features
        assert features['_history_count'] == 2
