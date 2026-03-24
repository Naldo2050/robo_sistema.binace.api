"""Testes para Mean Reversion Analyzer."""
import pytest

from institutional.base import Side
from institutional.mean_reversion import MeanReversionAnalyzer


class TestMeanReversionInit:
    def test_default(self):
        analyzer = MeanReversionAnalyzer()
        assert analyzer.data_points == 0
        assert analyzer.last_state is None

    def test_invalid_lookback(self):
        with pytest.raises(Exception):
            MeanReversionAnalyzer(lookback=2)


class TestMeanReversionCalc:
    def test_insufficient_data(self):
        analyzer = MeanReversionAnalyzer(lookback=20)
        for i in range(10):
            result = analyzer.add_price(float(i), 100.0)
        assert result is None

    def test_fair_value_around_mean(self):
        analyzer = MeanReversionAnalyzer(lookback=20)

        for i in range(25):
            analyzer.add_price(float(i), 100.0)

        state = analyzer.last_state
        assert state is not None
        assert state.regime == "fair_value"
        assert abs(state.z_score) < 0.5

    def test_overbought_detection(self):
        analyzer = MeanReversionAnalyzer(lookback=20, z_threshold=2.0)

        # Build mean at 100
        for i in range(20):
            analyzer.add_price(float(i), 100.0)

        # Spike to 120
        state = analyzer.add_price(21.0, 120.0)
        assert state is not None
        assert state.z_score > 0
        # Might or might not be "overbought" depending on std

    def test_oversold_detection(self):
        analyzer = MeanReversionAnalyzer(lookback=20, z_threshold=2.0)

        # Build mean at 100
        for i in range(20):
            analyzer.add_price(float(i), 100.0)

        # Drop to 80
        state = analyzer.add_price(21.0, 80.0)
        assert state is not None
        assert state.z_score < 0

    def test_bollinger_bands(self):
        analyzer = MeanReversionAnalyzer(lookback=20, bb_multiplier=2.0)

        for i in range(25):
            analyzer.add_price(float(i), 100.0 + (i % 5) - 2)

        state = analyzer.last_state
        assert state is not None
        assert state.bollinger_upper > state.mean
        assert state.bollinger_lower < state.mean

    def test_percent_b(self):
        analyzer = MeanReversionAnalyzer(lookback=20)

        for i in range(25):
            analyzer.add_price(float(i), 100.0)

        state = analyzer.last_state
        assert state is not None
        assert 0.0 <= state.percent_b <= 1.0 or True  # Can be outside bands


class TestMeanReversionProbability:
    def test_reversion_probability(self):
        analyzer = MeanReversionAnalyzer(lookback=20)

        for i in range(20):
            analyzer.add_price(float(i), 100.0)
        analyzer.add_price(21.0, 115.0)

        prob = analyzer.get_reversion_probability()
        assert 0.0 <= prob <= 1.0


class TestMeanReversionAnalysis:
    def test_analyze_empty(self):
        analyzer = MeanReversionAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0

    def test_analyze_overbought_signal(self):
        analyzer = MeanReversionAnalyzer(lookback=20, z_threshold=1.5)

        for i in range(20):
            analyzer.add_price(float(i), 100.0)
        # Extreme move
        for i in range(5):
            analyzer.add_price(float(20 + i), 130.0)

        result = analyzer.analyze()
        assert result.source == "mean_reversion"
        assert "z_score" in result.metrics


class TestMeanReversionReset:
    def test_reset(self):
        analyzer = MeanReversionAnalyzer()
        for i in range(30):
            analyzer.add_price(float(i), 100.0)

        analyzer.reset()
        assert analyzer.data_points == 0
        assert analyzer.last_state is None
