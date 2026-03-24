# tests/unit/test_institutional_kalman.py
"""Testes para Kalman Trend Filter."""
import pytest
import math

from institutional.kalman_filter import KalmanTrendFilter, KalmanState


class TestKalmanInit:
    def test_default_init(self):
        kf = KalmanTrendFilter()
        assert not kf.is_initialized
        assert kf.filtered_price == 0.0

    def test_first_update_initializes(self):
        kf = KalmanTrendFilter()
        state = kf.update(1.0, 100.0)

        assert kf.is_initialized
        assert state.filtered_price == 100.0
        assert state.innovation == 0.0


class TestKalmanFiltering:
    def test_follows_trend(self):
        kf = KalmanTrendFilter(process_noise=0.1, measurement_noise=1.0)

        prices = [100 + i for i in range(20)]
        for i, price in enumerate(prices):
            state = kf.update(float(i), float(price))

        # Filtered price should be close to actual
        assert abs(state.filtered_price - prices[-1]) < 5.0

    def test_filters_noise(self):
        kf = KalmanTrendFilter(process_noise=0.01, measurement_noise=10.0)

        import random
        random.seed(42)

        states = []
        for i in range(100):
            # True trend: 100 + 0.1*i
            # With noise: ±5
            true_price = 100 + 0.1 * i
            noisy_price = true_price + random.gauss(0, 5)
            state = kf.update(float(i), noisy_price)
            states.append(state)

        # Filtered should be smoother than raw
        raw_changes = [
            abs(states[i].price - states[i - 1].price)
            for i in range(1, len(states))
        ]
        filtered_changes = [
            abs(states[i].filtered_price - states[i - 1].filtered_price)
            for i in range(1, len(states))
        ]

        avg_raw = sum(raw_changes) / len(raw_changes)
        avg_filtered = sum(filtered_changes) / len(filtered_changes)

        assert avg_filtered < avg_raw

    def test_velocity_positive_in_uptrend(self):
        kf = KalmanTrendFilter()

        for i in range(30):
            kf.update(float(i), 100.0 + i * 2.0)

        assert kf.velocity > 0

    def test_velocity_negative_in_downtrend(self):
        kf = KalmanTrendFilter()

        for i in range(30):
            kf.update(float(i), 100.0 - i * 2.0)

        assert kf.velocity < 0


class TestKalmanTrend:
    def test_trend_direction_up(self):
        kf = KalmanTrendFilter()
        for i in range(30):
            kf.update(float(i), 100.0 + i * 5.0)

        assert kf.get_trend_direction() == "up"

    def test_trend_direction_down(self):
        kf = KalmanTrendFilter()
        for i in range(30):
            kf.update(float(i), 200.0 - i * 5.0)

        assert kf.get_trend_direction() == "down"

    def test_trend_direction_flat(self):
        kf = KalmanTrendFilter()
        for i in range(30):
            kf.update(float(i), 100.0)

        assert kf.get_trend_direction() == "flat"

    def test_trend_strength(self):
        kf = KalmanTrendFilter()

        # Strong trend
        for i in range(50):
            kf.update(float(i), 100.0 + i * 10.0)

        strength = kf.get_trend_strength()
        assert 0.0 <= strength <= 1.0
        assert strength > 0.3  # Should be fairly strong


class TestKalmanPrediction:
    def test_predict_returns_list(self):
        kf = KalmanTrendFilter()
        for i in range(20):
            kf.update(float(i), 100.0 + i)

        predictions = kf.predict_price(5)
        assert len(predictions) == 5
        assert all(isinstance(p, float) for p in predictions)

    def test_predict_uninitialized(self):
        kf = KalmanTrendFilter()
        predictions = kf.predict_price(3)
        assert predictions == [0.0, 0.0, 0.0]

    def test_predict_uptrend_increasing(self):
        kf = KalmanTrendFilter()
        for i in range(30):
            kf.update(float(i), 100.0 + i * 2.0)

        predictions = kf.predict_price(5)
        # In uptrend, predictions should generally increase
        assert predictions[-1] > predictions[0]


class TestKalmanAnalysis:
    def test_analyze_not_initialized(self):
        kf = KalmanTrendFilter()
        result = kf.analyze(100.0)
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        kf = KalmanTrendFilter()
        for i in range(30):
            kf.update(float(i), 100.0 + i)

        result = kf.analyze(130.0)
        assert result.source == "kalman_filter"
        assert "filtered_price" in result.metrics
        assert "velocity" in result.metrics
        assert "trend_direction" in result.metrics
        assert "trend_strength" in result.metrics


class TestKalmanReset:
    def test_reset(self):
        kf = KalmanTrendFilter()
        for i in range(10):
            kf.update(float(i), 100.0)

        assert kf.is_initialized
        kf.reset()
        assert not kf.is_initialized
        assert kf.filtered_price == 0.0
        assert len(kf.history) == 0
