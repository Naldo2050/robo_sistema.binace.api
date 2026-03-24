# tests/unit/test_institutional_ofi.py
"""Testes para Order Flow Imbalance Analyzer."""
import pytest

from institutional.base import Side, Trade
from institutional.order_flow_imbalance import (
    OrderFlowImbalanceAnalyzer,
    OFIBar,
)


def _trade(price=100.0, qty=1.0, side=Side.BUY, ts=0.0):
    return Trade(timestamp=ts, price=price, quantity=qty, side=side)


class TestOFIInit:
    def test_default(self):
        analyzer = OrderFlowImbalanceAnalyzer()
        assert analyzer.window_seconds == 30.0

    def test_invalid_window(self):
        with pytest.raises(Exception):
            OrderFlowImbalanceAnalyzer(window_seconds=0)


class TestOFIProcessing:
    def test_single_trade_no_bar(self):
        analyzer = OrderFlowImbalanceAnalyzer(window_seconds=10)
        result = analyzer.process_trade(_trade(ts=1.0))
        assert result is None

    def test_bar_closes(self):
        analyzer = OrderFlowImbalanceAnalyzer(window_seconds=5)

        analyzer.process_trade(_trade(ts=0, side=Side.BUY, qty=10))
        analyzer.process_trade(_trade(ts=2, side=Side.SELL, qty=3))
        bar = analyzer.process_trade(_trade(ts=6, side=Side.BUY, qty=1))

        assert bar is not None
        assert bar.buy_volume == 10.0
        assert bar.sell_volume == 3.0
        assert bar.dominant_side == Side.BUY

    def test_extreme_imbalance_generates_alert(self):
        analyzer = OrderFlowImbalanceAnalyzer(
            window_seconds=5,
            alert_cooldown_seconds=0,
        )

        # Massive buy imbalance
        analyzer.process_trade(_trade(ts=0, side=Side.BUY, qty=100))
        analyzer.process_trade(_trade(ts=1, side=Side.SELL, qty=1))
        analyzer.process_trade(_trade(ts=6, side=Side.BUY, qty=0.01))

        assert len(analyzer.alerts) > 0
        assert analyzer.alerts[0].severity in ("extreme", "strong", "moderate")

    def test_no_alert_during_cooldown(self):
        analyzer = OrderFlowImbalanceAnalyzer(
            window_seconds=5,
            alert_cooldown_seconds=100,
        )

        # First imbalance
        analyzer.process_trade(_trade(ts=0, side=Side.BUY, qty=100))
        analyzer.process_trade(_trade(ts=6, side=Side.BUY, qty=0.01))

        # Second imbalance (within cooldown)
        analyzer.process_trade(_trade(ts=7, side=Side.BUY, qty=200))
        analyzer.process_trade(_trade(ts=13, side=Side.BUY, qty=0.01))

        assert len(analyzer.alerts) <= 1


class TestOFIImbalance:
    def test_current_imbalance(self):
        analyzer = OrderFlowImbalanceAnalyzer(window_seconds=60)

        analyzer.process_trade(_trade(ts=0, side=Side.BUY, qty=10))
        analyzer.process_trade(_trade(ts=1, side=Side.SELL, qty=5))

        current = analyzer.get_current_imbalance()
        assert current["buy_volume"] == 10.0
        assert current["sell_volume"] == 5.0
        assert current["ratio"] == 2.0
        assert current["dominant_side"] == "buy"


class TestOFITrend:
    def test_trend_strength_insufficient_data(self):
        analyzer = OrderFlowImbalanceAnalyzer(window_seconds=1)
        trend = analyzer.get_trend_strength(lookback=5)
        assert trend["score"] == 0.0

    def test_bullish_trend(self):
        analyzer = OrderFlowImbalanceAnalyzer(
            window_seconds=1,
            alert_cooldown_seconds=0,
        )

        for i in range(10):
            analyzer.process_trade(
                _trade(ts=i, side=Side.BUY, qty=20)
            )
            analyzer.process_trade(
                _trade(ts=i + 0.5, side=Side.SELL, qty=2)
            )
            analyzer.process_trade(
                _trade(ts=i + 1.01, side=Side.BUY, qty=0.01)
            )

        trend = analyzer.get_trend_strength(lookback=5)
        assert trend["direction"] == "bullish"
        assert trend["score"] > 0


class TestOFIAnalysis:
    def test_analyze_empty(self):
        analyzer = OrderFlowImbalanceAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        analyzer = OrderFlowImbalanceAnalyzer(window_seconds=1, alert_cooldown_seconds=0)

        for i in range(10):
            analyzer.process_trade(
                _trade(ts=i, price=100, side=Side.BUY, qty=15)
            )
            analyzer.process_trade(
                _trade(ts=i + 0.5, price=100, side=Side.SELL, qty=3)
            )
            analyzer.process_trade(
                _trade(ts=i + 1.01, price=100, side=Side.BUY, qty=0.01)
            )

        result = analyzer.analyze()
        assert result.source == "ofi_analyzer"
        assert "trend_score" in result.metrics


class TestOFIReset:
    def test_reset(self):
        analyzer = OrderFlowImbalanceAnalyzer(window_seconds=1)

        for i in range(5):
            analyzer.process_trade(_trade(ts=i, side=Side.BUY, qty=5))
            analyzer.process_trade(_trade(ts=i + 1.01, side=Side.BUY, qty=0.01))

        analyzer.reset()
        assert len(analyzer.bars) == 0
        assert len(analyzer.alerts) == 0
        assert analyzer.total_trades_processed == 0
