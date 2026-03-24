"""Testes para GARCH Volatility Model."""
import pytest
import math

from institutional.garch_volatility import GARCHModel, GARCHState


class TestGARCHInit:
    def test_default(self):
        model = GARCHModel()
        assert not model.is_initialized
        assert model.persistence < 1.0

    def test_invalid_alpha_beta(self):
        with pytest.raises(Exception):
            GARCHModel(alpha=0.5, beta=0.6)  # sum > 1

    def test_invalid_omega(self):
        with pytest.raises(Exception):
            GARCHModel(omega=0)

    def test_long_run_variance(self):
        model = GARCHModel(omega=0.00001, alpha=0.1, beta=0.85)
        expected = 0.00001 / (1 - 0.1 - 0.85)
        assert model.long_run_variance == pytest.approx(expected, rel=1e-6)


class TestGARCHUpdate:
    def test_first_price_returns_none(self):
        model = GARCHModel()
        result = model.update(1.0, 100.0)
        assert result is None

    def test_second_price_initializes(self):
        model = GARCHModel()
        model.update(1.0, 100.0)
        state = model.update(2.0, 101.0)
        assert state is not None
        assert model.is_initialized

    def test_variance_updates(self):
        model = GARCHModel()
        model.update(0, 100.0)
        s1 = model.update(1, 102.0)
        s2 = model.update(2, 98.0)

        assert s1 is not None
        assert s2 is not None
        # After a bigger move, variance should change
        assert s2.conditional_variance != s1.conditional_variance

    def test_stable_prices_low_volatility(self):
        model = GARCHModel()

        for i in range(50):
            model.update(float(i), 100.0 + (i % 2) * 0.01)

        assert model.current_volatility < 0.1


class TestGARCHForecast:
    def test_forecast(self):
        model = GARCHModel()
        for i in range(30):
            model.update(float(i), 100.0 + i * 0.5)

        forecast = model.forecast(5)
        assert forecast.steps_ahead == 5
        assert len(forecast.forecasted_variance) == 5
        assert len(forecast.forecasted_volatility) == 5
        assert all(v > 0 for v in forecast.forecasted_volatility)

    def test_forecast_converges_to_long_run(self):
        model = GARCHModel()
        for i in range(50):
            model.update(float(i), 100.0 + i)

        forecast = model.forecast(100)
        lr_var = model.long_run_variance

        # Last forecast should be close to long-run variance
        assert forecast.forecasted_variance[-1] == pytest.approx(lr_var, rel=0.1)


class TestGARCHRegime:
    def test_regime_unknown_before_init(self):
        model = GARCHModel()
        assert model.get_volatility_regime() == "unknown"

    def test_regime_classification(self):
        model = GARCHModel()
        for i in range(30):
            model.update(float(i), 100.0 + i * 0.1)

        regime = model.get_volatility_regime()
        assert regime in ("extreme_high", "high", "normal", "low", "extreme_low")


class TestGARCHAnalysis:
    def test_analyze_no_data(self):
        model = GARCHModel()
        result = model.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        model = GARCHModel()
        for i in range(50):
            model.update(float(i), 100.0 + math.sin(i * 0.5) * 5)

        result = model.analyze()
        assert result.source == "garch_model"
        assert "current_volatility" in result.metrics
        assert "volatility_regime" in result.metrics
        assert "persistence" in result.metrics


class TestGARCHReset:
    def test_reset(self):
        model = GARCHModel()
        for i in range(20):
            model.update(float(i), 100.0 + i)

        assert model.is_initialized
        model.reset()
        assert not model.is_initialized
