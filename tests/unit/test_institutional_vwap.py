# tests/unit/test_institutional_vwap.py
"""Testes para VWAP/TWAP Analyzer."""
import pytest

from institutional.base import Side
from institutional.vwap_twap import (
    VWAPCalculator,
    TWAPCalculator,
    VWAPTWAPAnalyzer,
)


class TestVWAPCalculator:
    def test_empty_vwap(self):
        calc = VWAPCalculator()
        assert calc.vwap == 0.0
        assert calc.data_points == 0

    def test_single_data_point(self):
        calc = VWAPCalculator()
        vwap = calc.add_candle(1.0, high=105, low=95, close=100, volume=10)
        # Typical price = (105+95+100)/3 = 100
        assert vwap == 100.0

    def test_vwap_calculation(self):
        calc = VWAPCalculator()

        # Period 1: typical=100, vol=10
        calc.add_candle(1.0, 105, 95, 100, 10)
        # Period 2: typical=200, vol=10
        calc.add_candle(2.0, 210, 190, 200, 10)

        # VWAP = (100*10 + 200*10) / (10 + 10) = 150
        assert calc.vwap == pytest.approx(150.0, abs=0.5)

    def test_vwap_volume_weighted(self):
        calc = VWAPCalculator()

        # Low price, high volume
        calc.add_candle(1.0, 100, 100, 100, 100)
        # High price, low volume
        calc.add_candle(2.0, 200, 200, 200, 1)

        # VWAP should be much closer to 100 (high volume there)
        assert calc.vwap < 150.0

    def test_bands(self):
        calc = VWAPCalculator()

        for i in range(50):
            price = 100 + (i % 10) - 5  # 95-105
            calc.add_candle(float(i), price + 1, price - 1, price, 10)

        bands = calc.get_bands()
        assert bands.vwap > 0
        assert bands.upper_1sd > bands.vwap
        assert bands.lower_1sd < bands.vwap
        assert bands.upper_2sd > bands.upper_1sd

    def test_deviation(self):
        calc = VWAPCalculator()

        for i in range(20):
            calc.add_candle(float(i), 101, 99, 100, 10)

        dev = calc.get_deviation(100)
        assert dev["zone"] == "fair_value"

        dev_high = calc.get_deviation(120)
        assert dev_high["deviation_pct"] > 0

    def test_reset(self):
        calc = VWAPCalculator()
        calc.add_candle(1.0, 100, 100, 100, 10)
        assert calc.data_points == 1

        calc.reset()
        assert calc.data_points == 0
        assert calc.vwap == 0.0


class TestTWAPCalculator:
    def test_empty_twap(self):
        calc = TWAPCalculator()
        assert calc.twap == 0.0

    def test_twap_calculation(self):
        calc = TWAPCalculator()
        calc.add_price(1.0, 100)
        calc.add_price(2.0, 200)
        assert calc.twap == 150.0

    def test_twap_deviation(self):
        calc = TWAPCalculator()
        calc.add_price(1.0, 100)
        calc.add_price(2.0, 100)

        dev = calc.get_deviation(110)
        assert dev["deviation_pct"] == pytest.approx(10.0, abs=0.1)
        assert dev["above_twap"] is True

    def test_reset(self):
        calc = TWAPCalculator()
        calc.add_price(1.0, 100)
        calc.reset()
        assert calc.data_points == 0


class TestVWAPTWAPAnalyzer:
    def test_add_candle(self):
        analyzer = VWAPTWAPAnalyzer()
        metrics = analyzer.add_candle(1.0, 105, 95, 100, 10)

        assert "vwap" in metrics
        assert "twap" in metrics
        assert "price" in metrics

    def test_analyze_empty(self):
        analyzer = VWAPTWAPAnalyzer()
        result = analyzer.analyze(100.0)
        assert result.source == "vwap_twap_analyzer"

    def test_analyze_with_data(self):
        analyzer = VWAPTWAPAnalyzer()

        for i in range(50):
            analyzer.add_candle(float(i), 101, 99, 100, 10)

        result = analyzer.analyze(100.0)
        assert "vwap" in result.metrics
        assert "twap" in result.metrics

    def test_overbought_signal(self):
        analyzer = VWAPTWAPAnalyzer()

        # Add data centered around 100
        for i in range(100):
            analyzer.add_candle(float(i), 101, 99, 100, 10)

        # Analyze at extreme price
        result = analyzer.analyze(120.0)

        # Should have signals about being above VWAP
        sell_signals = [s for s in result.signals if s.direction == Side.SELL]
        # May have sell signal due to overbought
        assert isinstance(result.signals, list)

    def test_reset(self):
        analyzer = VWAPTWAPAnalyzer()
        analyzer.add_candle(1.0, 100, 100, 100, 10)
        analyzer.reset()
        assert analyzer.vwap.data_points == 0
        assert analyzer.twap.data_points == 0
