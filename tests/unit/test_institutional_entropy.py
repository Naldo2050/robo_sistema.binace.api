"""Testes para Shannon Entropy Analyzer."""
import pytest
import math

from institutional.entropy_analyzer import EntropyAnalyzer


class TestEntropyInit:
    def test_default(self):
        analyzer = EntropyAnalyzer()
        assert analyzer.current_entropy is None

    def test_invalid_bins(self):
        with pytest.raises(Exception):
            EntropyAnalyzer(n_bins=2)

    def test_invalid_window(self):
        with pytest.raises(Exception):
            EntropyAnalyzer(window_size=5)


class TestEntropyCalculation:
    def test_insufficient_data(self):
        analyzer = EntropyAnalyzer(window_size=50)
        for i in range(30):
            result = analyzer.add_price(float(i), 100.0 + i)
        assert result is None

    def test_constant_price_low_entropy(self):
        analyzer = EntropyAnalyzer(n_bins=10, window_size=50)

        for i in range(60):
            analyzer.add_price(float(i), 100.0)

        # Constant price → returns are 0 → low entropy
        state = analyzer.add_price(61.0, 100.0)
        if state:
            assert state.normalized_entropy < 0.5

    def test_varied_returns_higher_entropy(self):
        analyzer = EntropyAnalyzer(n_bins=10, window_size=50)

        for i in range(60):
            # Highly varied prices
            price = 100.0 + (i % 7) * 3 - 10
            analyzer.add_price(float(i), max(price, 1))

        if analyzer.current_entropy is not None:
            assert analyzer.current_entropy > 0


class TestEntropyTrend:
    def test_trend_unknown_few_data(self):
        analyzer = EntropyAnalyzer()
        trend = analyzer.get_entropy_trend()
        assert trend["trend"] == "unknown"

    def test_trend_with_data(self):
        analyzer = EntropyAnalyzer(n_bins=10, window_size=20)

        for i in range(100):
            analyzer.add_price(float(i), 100 + math.sin(i * 0.3) * 5)

        trend = analyzer.get_entropy_trend()
        assert trend["trend"] in ("increasing", "decreasing", "stable")


class TestEntropyAnalysis:
    def test_analyze_empty(self):
        analyzer = EntropyAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        analyzer = EntropyAnalyzer(n_bins=10, window_size=30)

        for i in range(50):
            analyzer.add_price(float(i), 100 + i * 0.5)

        result = analyzer.analyze()
        assert result.source == "entropy_analyzer"
        if result.metrics:
            assert "entropy" in result.metrics


class TestEntropyReset:
    def test_reset(self):
        analyzer = EntropyAnalyzer(window_size=20)
        for i in range(30):
            analyzer.add_price(float(i), 100 + i)

        analyzer.reset()
        assert analyzer.current_entropy is None
