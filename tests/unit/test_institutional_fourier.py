"""Testes para Fourier Cycle Analyzer."""
import pytest
import math

from institutional.fourier_cycles import FourierCycleAnalyzer


class TestFourierInit:
    def test_default(self):
        analyzer = FourierCycleAnalyzer()
        assert analyzer.data_points == 0
        assert analyzer.last_result is None

    def test_invalid_period(self):
        with pytest.raises(Exception):
            FourierCycleAnalyzer(min_period=1)

    def test_invalid_window(self):
        with pytest.raises(Exception):
            FourierCycleAnalyzer(window_size=10)


class TestFourierCalculation:
    def test_insufficient_data(self):
        analyzer = FourierCycleAnalyzer(window_size=64)
        analyzer.add_prices([100.0] * 30)
        result = analyzer.calculate()
        assert result is None

    def test_sine_wave_detection(self):
        """A pure sine wave should have one dominant cycle."""
        analyzer = FourierCycleAnalyzer(
            window_size=128,
            min_period=5,
            max_period=64,
        )

        period = 20
        prices = [
            100 + 10 * math.sin(2 * math.pi * i / period)
            for i in range(128)
        ]
        analyzer.add_prices(prices)

        result = analyzer.calculate()
        assert result is not None
        assert len(result.dominant_cycles) > 0

        # The dominant cycle should be close to our period
        dominant = result.dominant_cycles[0]
        assert abs(dominant.period - period) < period * 0.3  # Within 30%

    def test_cycle_strength(self):
        analyzer = FourierCycleAnalyzer(window_size=64)

        # Purely cyclical data should have high strength
        prices = [100 + 5 * math.sin(2 * math.pi * i / 15) for i in range(64)]
        analyzer.add_prices(prices)

        result = analyzer.calculate()
        assert result is not None
        assert result.cycle_strength > 0


class TestFourierAnalysis:
    def test_analyze_empty(self):
        analyzer = FourierCycleAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_cycles(self):
        analyzer = FourierCycleAnalyzer(window_size=64, min_period=5)
        prices = [100 + 5 * math.sin(2 * math.pi * i / 20) for i in range(64)]
        analyzer.add_prices(prices)

        result = analyzer.analyze()
        assert result.source == "fourier_cycles"
        assert "spectral_entropy" in result.metrics
        assert "cycle_strength" in result.metrics


class TestFourierReset:
    def test_reset(self):
        analyzer = FourierCycleAnalyzer()
        analyzer.add_prices([100.0] * 100)
        assert analyzer.data_points == 100

        analyzer.reset()
        assert analyzer.data_points == 0
        assert analyzer.last_result is None
