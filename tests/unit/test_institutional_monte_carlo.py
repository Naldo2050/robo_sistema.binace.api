"""Testes para Monte Carlo Simulator."""
import pytest

from institutional.monte_carlo import MonteCarloSimulator


class TestMonteCarloInit:
    def test_default(self):
        mc = MonteCarloSimulator()
        assert mc.data_points == 0

    def test_invalid_simulations(self):
        with pytest.raises(Exception):
            MonteCarloSimulator(n_simulations=10)


class TestMonteCarloSimulation:
    def test_insufficient_data(self):
        mc = MonteCarloSimulator()
        mc.add_prices([100.0] * 5)
        result = mc.simulate()
        assert result is None

    def test_basic_simulation(self):
        mc = MonteCarloSimulator(n_simulations=500, seed=42)

        prices = [100.0 + i * 0.5 for i in range(50)]
        mc.add_prices(prices)

        result = mc.simulate(steps=10)
        assert result is not None
        assert result.n_simulations == 500
        assert result.n_steps == 10
        assert result.current_price == prices[-1]

    def test_percentiles_ordered(self):
        mc = MonteCarloSimulator(n_simulations=1000, seed=42)
        mc.add_prices([100.0 + i * 0.1 for i in range(100)])

        result = mc.simulate(30)
        assert result is not None
        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.median_final_price
        assert result.median_final_price <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95

    def test_probabilities_valid(self):
        mc = MonteCarloSimulator(n_simulations=1000, seed=42)
        mc.add_prices([100.0 + i * 0.2 for i in range(100)])

        result = mc.simulate(20)
        assert result is not None
        assert 0.0 <= result.prob_above_current <= 1.0
        assert 0.0 <= result.prob_gain_5pct <= 1.0
        assert 0.0 <= result.prob_loss_5pct <= 1.0

    def test_var_ordering(self):
        mc = MonteCarloSimulator(n_simulations=500, seed=42)
        # Use volatile data so VaR values are meaningful
        import math
        prices = [100.0 + math.sin(i * 0.5) * 10 for i in range(80)]
        mc.add_prices(prices)

        result = mc.simulate(20)
        assert result is not None
        # 99% VaR should be >= 95% VaR (more extreme tail)
        assert result.var_99 >= result.var_95

    def test_drawdown(self):
        mc = MonteCarloSimulator(n_simulations=500, seed=42)
        mc.add_prices([100.0 + i * 0.1 for i in range(50)])

        result = mc.simulate(30)
        assert result is not None
        assert result.avg_max_drawdown >= 0
        assert result.worst_max_drawdown >= result.avg_max_drawdown


class TestMonteCarloAnalysis:
    def test_analyze_no_data(self):
        mc = MonteCarloSimulator()
        result = mc.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        mc = MonteCarloSimulator(n_simulations=200, seed=42)
        mc.add_prices([100.0 + i * 0.5 for i in range(50)])

        result = mc.analyze(steps=10)
        assert result.source == "monte_carlo"
        assert "prob_above_current" in result.metrics
        assert "var_95" in result.metrics


class TestMonteCarloReset:
    def test_reset(self):
        mc = MonteCarloSimulator()
        mc.add_prices([100.0] * 50)
        assert mc.data_points == 50

        mc.reset()
        assert mc.data_points == 0
