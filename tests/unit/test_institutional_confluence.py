"""Testes para Confluence Engine."""
import pytest
import time

from institutional.base import (
    AnalysisResult,
    Side,
    Signal,
    SignalStrength,
)
from institutional.confluence_engine import ConfluenceEngine, ConfluenceScore


def _make_result(
    source: str,
    signals: list[tuple[Side, float]] = None,
) -> AnalysisResult:
    """Helper para criar AnalysisResult com sinais."""
    result = AnalysisResult(source=source, timestamp=time.time())

    if signals:
        for direction, confidence in signals:
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type=f"{source}_signal",
                    direction=direction,
                    strength=SignalStrength.MODERATE,
                    price=100.0,
                    confidence=confidence,
                    source=source,
                )
            )

    return result


class TestConfluenceInit:
    def test_default(self):
        engine = ConfluenceEngine()
        assert engine.min_sources == 2

    def test_custom_weights(self):
        engine = ConfluenceEngine(custom_weights={"my_source": 5.0})
        assert engine._weights["my_source"] == 5.0


class TestConfluenceAddResult:
    def test_add_single(self):
        engine = ConfluenceEngine()
        result = _make_result("test_source", [(Side.BUY, 0.8)])
        engine.add_result(result)
        assert "test_source" in engine._latest_results

    def test_add_multiple(self):
        engine = ConfluenceEngine()
        results = [
            _make_result("source_1", [(Side.BUY, 0.7)]),
            _make_result("source_2", [(Side.BUY, 0.6)]),
        ]
        engine.add_results(results)
        assert len(engine._latest_results) == 2


class TestConfluenceCalculation:
    def test_empty(self):
        engine = ConfluenceEngine()
        score = engine.calculate_confluence()
        assert score.direction == Side.UNKNOWN
        assert score.total_signals == 0

    def test_all_buy(self):
        engine = ConfluenceEngine()
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.BUY, 0.8)]),
            _make_result("footprint_analyzer", [(Side.BUY, 0.7)]),
            _make_result("ofi_analyzer", [(Side.BUY, 0.9)]),
        ])

        score = engine.calculate_confluence()
        assert score.direction == Side.BUY
        assert score.buy_signals == 3
        assert score.sell_signals == 0
        assert score.sources_agreeing == 3
        assert score.total_score > 0

    def test_all_sell(self):
        engine = ConfluenceEngine()
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.SELL, 0.8)]),
            _make_result("smart_money_analyzer", [(Side.SELL, 0.6)]),
        ])

        score = engine.calculate_confluence()
        assert score.direction == Side.SELL
        assert score.total_score < 0

    def test_mixed_signals_neutral(self):
        engine = ConfluenceEngine()
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.BUY, 0.5)]),
            _make_result("footprint_analyzer", [(Side.SELL, 0.5)]),
        ])

        score = engine.calculate_confluence()
        # Should be roughly neutral
        assert abs(score.total_score) < 0.5

    def test_weighted_scoring(self):
        engine = ConfluenceEngine()

        # Order flow signal (weight 3.0) vs probabilistic (weight 0.5)
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.BUY, 0.8)]),     # weight 3
            _make_result("monte_carlo", [(Side.SELL, 0.8)]),      # weight 0.5
        ])

        score = engine.calculate_confluence()
        # CVD should dominate due to higher weight
        assert score.total_score > 0


class TestConfluenceStrength:
    def test_strong_confluence(self):
        engine = ConfluenceEngine()

        # Many sources agreeing strongly
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.BUY, 0.9)]),
            _make_result("footprint_analyzer", [(Side.BUY, 0.8)]),
            _make_result("ofi_analyzer", [(Side.BUY, 0.85)]),
            _make_result("smart_money_analyzer", [(Side.BUY, 0.7)]),
        ])

        score = engine.calculate_confluence()
        assert score.strength in (SignalStrength.STRONG, SignalStrength.MODERATE)
        assert score.confidence > 0.5


class TestConfluenceLayerSummary:
    def test_layer_summary(self):
        engine = ConfluenceEngine()
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.BUY, 0.8)]),
            _make_result("smart_money_analyzer", [(Side.SELL, 0.7)]),
            _make_result("kalman_filter", [(Side.BUY, 0.6)]),
        ])

        summary = engine.get_layer_summary()
        assert "order_flow" in summary
        assert "structure" in summary
        assert "statistical" in summary
        assert summary["order_flow"]["direction"] in ("bullish", "bearish", "neutral")


class TestConfluenceAnalysis:
    def test_analyze_empty(self):
        engine = ConfluenceEngine()
        result = engine.analyze()
        assert result.source == "confluence_engine"

    def test_analyze_with_data(self):
        engine = ConfluenceEngine(min_sources_for_signal=1)
        engine.add_results([
            _make_result("cvd_analyzer", [(Side.BUY, 0.8)]),
            _make_result("whale_detector", [(Side.BUY, 0.6)]),
        ])

        result = engine.analyze()
        assert "confluence_score" in result.metrics
        assert len(result.signals) > 0


class TestConfluenceReset:
    def test_reset(self):
        engine = ConfluenceEngine()
        engine.add_result(_make_result("test", [(Side.BUY, 0.5)]))

        engine.reset()
        assert len(engine._latest_results) == 0
