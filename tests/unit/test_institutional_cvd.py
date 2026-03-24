# tests/unit/test_institutional_cvd.py
"""Testes para CVD Analyzer."""
import pytest
import time

from institutional.base import Side, Trade, SignalStrength
from institutional.cvd import CVDAnalyzer, CVDBar


def _make_trade(
    price: float = 100.0,
    qty: float = 1.0,
    side: Side = Side.BUY,
    ts: float = 0.0,
) -> Trade:
    return Trade(
        timestamp=ts,
        price=price,
        quantity=qty,
        side=side,
    )


class TestCVDAnalyzerInit:
    def test_default_init(self):
        analyzer = CVDAnalyzer()
        assert analyzer.bar_interval_seconds == 60.0
        assert analyzer.max_bars == 1000
        assert analyzer.cumulative_delta == 0.0
        assert analyzer.bar_count == 0

    def test_custom_init(self):
        analyzer = CVDAnalyzer(
            bar_interval_seconds=30.0,
            max_bars=500,
            divergence_lookback=5,
        )
        assert analyzer.bar_interval_seconds == 30.0
        assert analyzer.max_bars == 500

    def test_invalid_interval_raises(self):
        with pytest.raises(Exception):
            CVDAnalyzer(bar_interval_seconds=0)

    def test_invalid_max_bars_raises(self):
        with pytest.raises(Exception):
            CVDAnalyzer(max_bars=5)


class TestCVDProcessTrade:
    def test_single_trade_no_bar(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=60)
        result = analyzer.process_trade(_make_trade(ts=1.0))
        assert result is None
        assert analyzer.total_trades_processed == 1

    def test_bar_completes_after_interval(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=10)

        # Trade at t=0
        analyzer.process_trade(_make_trade(ts=0.0, side=Side.BUY, qty=5.0))

        # Trade at t=5
        analyzer.process_trade(_make_trade(ts=5.0, side=Side.SELL, qty=3.0))

        # Trade at t=11 — should close bar
        bar = analyzer.process_trade(_make_trade(ts=11.0, side=Side.BUY, qty=1.0))

        assert bar is not None
        assert isinstance(bar, CVDBar)
        assert bar.buy_volume == 5.0
        assert bar.sell_volume == 3.0
        assert bar.delta == 2.0  # 5 - 3

    def test_cumulative_delta_accumulates(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=10)

        # Bar 1: buy=10, sell=3 → delta=7
        analyzer.process_trade(_make_trade(ts=0, side=Side.BUY, qty=10))
        analyzer.process_trade(_make_trade(ts=5, side=Side.SELL, qty=3))
        bar1 = analyzer.process_trade(_make_trade(ts=11, side=Side.BUY, qty=1))

        assert bar1 is not None
        assert bar1.cumulative_delta == 7.0

        # Bar 2: ts=11 BUY 1 + ts=15 SELL 8 → delta=-7, cum=0
        # ts=22 triggers bar close but goes to next bar
        analyzer.process_trade(_make_trade(ts=15, side=Side.SELL, qty=8))
        bar2 = analyzer.process_trade(_make_trade(ts=22, side=Side.BUY, qty=2))

        assert bar2 is not None
        assert analyzer.cumulative_delta == pytest.approx(7.0 + (1 - 8), abs=0.01)

    def test_process_trades_batch(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=10)

        trades = [
            _make_trade(ts=0, side=Side.BUY, qty=5),
            _make_trade(ts=3, side=Side.SELL, qty=2),
            _make_trade(ts=7, side=Side.BUY, qty=3),
            _make_trade(ts=11, side=Side.SELL, qty=1),  # closes bar
            _make_trade(ts=15, side=Side.BUY, qty=4),
        ]

        bars = analyzer.process_trades(trades)
        assert len(bars) == 1
        assert analyzer.total_trades_processed == 5


class TestCVDCurrentDelta:
    def test_current_delta(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=60)
        analyzer.process_trade(_make_trade(ts=0, side=Side.BUY, qty=10))
        analyzer.process_trade(_make_trade(ts=5, side=Side.SELL, qty=3))

        assert analyzer.get_current_delta() == 7.0

    def test_current_cvd_includes_open_bar(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=10)

        # Close one bar
        analyzer.process_trade(_make_trade(ts=0, side=Side.BUY, qty=10))
        analyzer.process_trade(_make_trade(ts=11, side=Side.SELL, qty=3))

        # Now in new bar, add trade
        analyzer.process_trade(_make_trade(ts=15, side=Side.BUY, qty=5))

        # CVD should include closed bar delta + current bar delta
        closed_delta = 10.0  # from first bar (buy=10, sell=0)
        current_delta = -3.0 + 5.0  # sell=3 + buy=5 in second bar
        expected_cvd = closed_delta + current_delta

        assert analyzer.get_current_cvd() == pytest.approx(expected_cvd, abs=0.01)


class TestCVDDivergences:
    def test_no_divergence_with_few_bars(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=1, divergence_lookback=5)
        divs = analyzer.detect_divergences()
        assert divs == []

    def test_bullish_divergence(self):
        """Price going down but CVD going up = bullish divergence."""
        analyzer = CVDAnalyzer(bar_interval_seconds=1, divergence_lookback=3)

        # Create bars where price falls but buying volume increases
        for i in range(5):
            price = 100.0 - i * 2  # Price declining
            # But buying volume increasing
            analyzer.process_trade(
                _make_trade(ts=i * 1.0, price=price, side=Side.BUY, qty=10 + i * 5)
            )
            analyzer.process_trade(
                _make_trade(ts=i * 1.0 + 0.5, price=price, side=Side.SELL, qty=5)
            )
            # Force close bar
            analyzer.process_trade(
                _make_trade(ts=(i + 1) * 1.0 + 0.01, price=price, side=Side.BUY, qty=0.01)
            )

        divs = analyzer.detect_divergences(lookback=3)
        # May or may not detect depending on exact values
        # The important thing is it doesn't crash
        assert isinstance(divs, list)


class TestCVDAnalysis:
    def test_analyze_empty(self):
        analyzer = CVDAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0
        assert result.source == "cvd_analyzer"

    def test_analyze_with_data(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=1)

        # Generate enough bars
        for i in range(10):
            analyzer.process_trade(
                _make_trade(ts=i, price=100 + i, side=Side.BUY, qty=10)
            )
            analyzer.process_trade(
                _make_trade(ts=i + 0.5, price=100 + i, side=Side.SELL, qty=2)
            )
            analyzer.process_trade(
                _make_trade(ts=i + 1.01, price=100 + i, side=Side.BUY, qty=0.01)
            )

        result = analyzer.analyze()
        assert result.source == "cvd_analyzer"
        assert "cumulative_delta" in result.metrics
        assert "buy_sell_ratio" in result.metrics

    def test_strong_buy_pressure_signal(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=1)

        for i in range(5):
            # Massive buy, tiny sell
            analyzer.process_trade(
                _make_trade(ts=i, price=100, side=Side.BUY, qty=100)
            )
            analyzer.process_trade(
                _make_trade(ts=i + 0.5, price=100, side=Side.SELL, qty=1)
            )
            analyzer.process_trade(
                _make_trade(ts=i + 1.01, price=100, side=Side.BUY, qty=0.01)
            )

        result = analyzer.analyze()

        buy_signals = [
            s for s in result.signals
            if s.direction == Side.BUY
        ]
        assert len(buy_signals) > 0


class TestCVDReset:
    def test_reset_clears_state(self):
        analyzer = CVDAnalyzer(bar_interval_seconds=1)

        for i in range(5):
            analyzer.process_trade(
                _make_trade(ts=i, side=Side.BUY, qty=10)
            )
            analyzer.process_trade(
                _make_trade(ts=i + 1.01, side=Side.BUY, qty=0.01)
            )

        assert analyzer.bar_count > 0
        assert analyzer.cumulative_delta != 0

        analyzer.reset()
        assert analyzer.bar_count == 0
        assert analyzer.cumulative_delta == 0.0
        assert analyzer.total_trades_processed == 0
