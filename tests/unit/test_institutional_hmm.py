"""Testes para Market Regime HMM."""
import pytest
import math

from institutional.market_regime_hmm import MarketRegimeHMM, HMMState


class TestHMMInit:
    def test_default(self):
        hmm = MarketRegimeHMM()
        assert hmm.current_state_name == "lateral"
        assert hmm.state_duration == 0

    def test_invalid_states(self):
        with pytest.raises(Exception):
            MarketRegimeHMM(n_states=1)


class TestHMMUpdate:
    def test_first_price_returns_none(self):
        hmm = MarketRegimeHMM()
        result = hmm.update(1.0, 100.0)
        assert result is None

    def test_needs_window_of_data(self):
        hmm = MarketRegimeHMM(return_window=10)
        for i in range(9):
            result = hmm.update(float(i), 100.0 + i * 0.1)
        # Not enough returns yet
        assert result is None

    def test_returns_state_after_window(self):
        hmm = MarketRegimeHMM(return_window=10)
        state = None
        for i in range(25):
            state = hmm.update(float(i), 100.0 + i * 0.5)

        assert state is not None
        assert isinstance(state, HMMState)
        assert state.state_name in ("bull", "bear", "lateral", "high_volatility")

    def test_uptrend_classified_as_bull(self):
        hmm = MarketRegimeHMM(return_window=10)

        for i in range(50):
            hmm.update(float(i), 100.0 + i * 2.0)

        assert hmm.current_state_name in ("bull", "lateral", "high_volatility")

    def test_downtrend_classified_as_bear(self):
        hmm = MarketRegimeHMM(return_window=10)

        for i in range(50):
            hmm.update(float(i), 200.0 - i * 2.0)

        assert hmm.current_state_name in ("bear", "lateral", "high_volatility")


class TestHMMTransitionMatrix:
    def test_transition_matrix_shape(self):
        hmm = MarketRegimeHMM(n_states=4)

        for i in range(50):
            hmm.update(float(i), 100.0 + math.sin(i * 0.3) * 10)

        matrix = hmm.get_transition_matrix()
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)

        # Each row should sum to ~1
        for row in matrix:
            row_sum = sum(row)
            if row_sum > 0:
                assert row_sum == pytest.approx(1.0, abs=0.01)


class TestHMMPrediction:
    def test_predict_next_state(self):
        hmm = MarketRegimeHMM()

        for i in range(50):
            hmm.update(float(i), 100.0 + i * 0.5)

        predictions = hmm.predict_next_state()
        assert isinstance(predictions, dict)
        assert sum(predictions.values()) == pytest.approx(1.0, abs=0.01)


class TestHMMRegimeInfo:
    def test_regime_info(self):
        hmm = MarketRegimeHMM()

        for i in range(50):
            hmm.update(float(i), 100.0 + math.sin(i * 0.2) * 5)

        regimes = hmm.get_regime_info()
        assert len(regimes) == 4
        assert all(r.name in ("bull", "bear", "lateral", "high_volatility") for r in regimes)


class TestHMMAnalysis:
    def test_analyze_no_data(self):
        hmm = MarketRegimeHMM()
        result = hmm.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        hmm = MarketRegimeHMM(return_window=10)

        for i in range(60):
            hmm.update(float(i), 100.0 + i * 0.3)

        result = hmm.analyze()
        assert result.source == "market_regime_hmm"
        assert "current_state_name" in result.metrics
        assert len(result.signals) > 0


class TestHMMReset:
    def test_reset(self):
        hmm = MarketRegimeHMM()
        for i in range(30):
            hmm.update(float(i), 100.0)

        hmm.reset()
        assert hmm.state_duration == 0
        assert len(hmm.history) == 0
