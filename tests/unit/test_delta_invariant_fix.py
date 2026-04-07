# -*- coding: utf-8 -*-
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_processing'))
from delta_validator import DeltaValidator

class TestDeltaInvariant:
    def test_delta_valid_match(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(100.0, 95.0, 5.0)
        assert delta == pytest.approx(5.0, abs=0.01)
        assert valid is True
        assert reason == 'VALID'

    def test_delta_zero_bug_detection(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(100.0, 90.0, 0.0)
        assert delta == pytest.approx(10.0, abs=0.01)
        assert valid is False
        assert reason == 'ZERO_BUG'

    def test_delta_negative_valid(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(80.0, 95.0, -15.0)
        assert delta == pytest.approx(-15.0, abs=0.01)
        assert valid is True

    def test_delta_small_rounding(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(100.0, 99.99, 0.009)
        assert valid is True

    def test_delta_drift_detection(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(1000.0, 500.0, 400.0)
        assert valid is False

    def test_build_delta_result_calc(self):
        result = DeltaValidator.build_delta_result(150.0, 140.0)
        assert result['delta'] == pytest.approx(10.0, abs=0.01)
        assert result['delta_verified'] is True

    def test_build_delta_fallback(self):
        result = DeltaValidator.build_delta_result(5.0, 3.0)
        assert result['delta'] == pytest.approx(2.0, abs=0.01)
        assert result['delta_verified'] is True

    def test_large_delta(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(500.0, 389.5, 110.5)
        assert valid is True

    def test_both_zero(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(0.0, 0.0, 0.0)
        assert delta == 0.0
        assert valid is True

    def test_mismatch_high(self):
        delta, valid, reason = DeltaValidator.validate_delta_invariant(100.0, 95.0, 500.0)
        assert valid is False
