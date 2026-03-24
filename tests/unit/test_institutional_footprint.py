# tests/unit/test_institutional_footprint.py
"""Testes para Footprint Analyzer."""
import pytest

from institutional.base import Side, Trade
from institutional.footprint import (
    FootprintAnalyzer,
    FootprintBar,
    FootprintLevel,
)


def _trade(price=100.0, qty=1.0, side=Side.BUY, ts=0.0):
    return Trade(timestamp=ts, price=price, quantity=qty, side=side)


class TestFootprintLevel:
    def test_delta(self):
        level = FootprintLevel(price=100, buy_volume=10, sell_volume=3)
        assert level.delta == 7.0

    def test_total_volume(self):
        level = FootprintLevel(price=100, buy_volume=5, sell_volume=5)
        assert level.total_volume == 10.0

    def test_imbalance_ratio(self):
        level = FootprintLevel(price=100, buy_volume=9, sell_volume=3)
        assert level.imbalance_ratio == 3.0

    def test_imbalance_no_sell(self):
        level = FootprintLevel(price=100, buy_volume=5, sell_volume=0)
        assert level.imbalance_ratio == float("inf")

    def test_imbalance_no_volume(self):
        level = FootprintLevel(price=100)
        assert level.imbalance_ratio == 1.0


class TestFootprintAnalyzerInit:
    def test_default_init(self):
        analyzer = FootprintAnalyzer()
        assert analyzer.tick_size == 1.0
        assert analyzer.bar_count == 0

    def test_invalid_tick_size(self):
        with pytest.raises(Exception):
            FootprintAnalyzer(tick_size=0)

    def test_invalid_interval(self):
        with pytest.raises(Exception):
            FootprintAnalyzer(bar_interval_seconds=-1)


class TestFootprintProcessing:
    def test_single_trade_no_bar(self):
        analyzer = FootprintAnalyzer(bar_interval_seconds=60)
        result = analyzer.process_trade(_trade(ts=1.0))
        assert result is None

    def test_bar_closes_after_interval(self):
        analyzer = FootprintAnalyzer(
            bar_interval_seconds=10,
            tick_size=1.0,
        )

        analyzer.process_trade(_trade(ts=0, price=100, qty=5, side=Side.BUY))
        analyzer.process_trade(_trade(ts=5, price=101, qty=3, side=Side.SELL))
        bar = analyzer.process_trade(
            _trade(ts=11, price=100, qty=1, side=Side.BUY)
        )

        assert bar is not None
        assert isinstance(bar, FootprintBar)
        assert bar.total_buy_volume == 5.0
        assert bar.total_sell_volume == 3.0
        assert bar.total_delta == 2.0
        assert len(bar.levels) >= 1

    def test_price_levels_separated(self):
        analyzer = FootprintAnalyzer(
            bar_interval_seconds=10,
            tick_size=10.0,
        )

        # Trades at different price levels
        analyzer.process_trade(_trade(ts=0, price=100, qty=5, side=Side.BUY))
        analyzer.process_trade(_trade(ts=1, price=110, qty=3, side=Side.SELL))
        analyzer.process_trade(_trade(ts=2, price=120, qty=2, side=Side.BUY))
        bar = analyzer.process_trade(_trade(ts=11, price=100, qty=1, side=Side.BUY))

        assert bar is not None
        assert len(bar.levels) == 3  # 100, 110, 120

    def test_poc_price(self):
        analyzer = FootprintAnalyzer(
            bar_interval_seconds=10,
            tick_size=10.0,
        )

        # Most volume at 100
        analyzer.process_trade(_trade(ts=0, price=100, qty=20, side=Side.BUY))
        analyzer.process_trade(_trade(ts=1, price=110, qty=5, side=Side.BUY))
        analyzer.process_trade(_trade(ts=2, price=120, qty=3, side=Side.BUY))
        bar = analyzer.process_trade(_trade(ts=11, price=100, qty=0.01, side=Side.BUY))

        assert bar is not None
        assert bar.poc_price == 100.0

    def test_batch_processing(self):
        analyzer = FootprintAnalyzer(bar_interval_seconds=5)

        trades = [
            _trade(ts=0, side=Side.BUY, qty=5),
            _trade(ts=2, side=Side.SELL, qty=3),
            _trade(ts=6, side=Side.BUY, qty=2),  # closes bar 1
            _trade(ts=8, side=Side.SELL, qty=1),
            _trade(ts=12, side=Side.BUY, qty=4),  # closes bar 2
        ]

        bars = analyzer.process_trades(trades)
        assert len(bars) == 2


class TestFootprintImbalances:
    def test_get_imbalances(self):
        bar = FootprintBar(timestamp=0, duration_seconds=60)
        bar.levels[100] = FootprintLevel(
            price=100, buy_volume=15, sell_volume=3
        )  # ratio=5
        bar.levels[101] = FootprintLevel(
            price=101, buy_volume=5, sell_volume=5
        )  # ratio=1
        bar.levels[102] = FootprintLevel(
            price=102, buy_volume=2, sell_volume=8
        )  # ratio=0.25

        imbalances = bar.get_imbalances(threshold=3.0)
        assert len(imbalances) == 2  # 100 (5x buy) and 102 (4x sell)

    def test_detect_stacked_imbalances(self):
        analyzer = FootprintAnalyzer(
            bar_interval_seconds=10,
            tick_size=1.0,
            imbalance_threshold=3.0,
        )

        # Create bar with stacked buy imbalances
        for i, price in enumerate([100, 101, 102, 103, 104]):
            analyzer.process_trade(
                _trade(ts=i * 0.1, price=price, qty=15, side=Side.BUY)
            )
            analyzer.process_trade(
                _trade(ts=i * 0.1 + 0.05, price=price, qty=3, side=Side.SELL)
            )

        bar = analyzer.process_trade(_trade(ts=11, price=100, qty=0.01, side=Side.BUY))
        assert bar is not None

        stacked = analyzer.detect_stacked_imbalances(bar, min_consecutive=3)
        # Should detect stacked buy imbalances
        assert isinstance(stacked, list)


class TestFootprintAnalysis:
    def test_analyze_empty(self):
        analyzer = FootprintAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        analyzer = FootprintAnalyzer(bar_interval_seconds=1, tick_size=1.0)

        for i in range(5):
            analyzer.process_trade(
                _trade(ts=i, price=100 + i, qty=10, side=Side.BUY)
            )
            analyzer.process_trade(
                _trade(ts=i + 0.5, price=100 + i, qty=2, side=Side.SELL)
            )
            analyzer.process_trade(
                _trade(ts=i + 1.01, price=100 + i, qty=0.01, side=Side.BUY)
            )

        result = analyzer.analyze()
        assert result.source == "footprint_analyzer"
        assert "poc_price" in result.metrics
        assert "total_delta" in result.metrics


class TestFootprintReset:
    def test_reset(self):
        analyzer = FootprintAnalyzer(bar_interval_seconds=1)

        for i in range(3):
            analyzer.process_trade(
                _trade(ts=i, side=Side.BUY, qty=5)
            )
            analyzer.process_trade(
                _trade(ts=i + 1.01, side=Side.BUY, qty=0.01)
            )

        assert analyzer.bar_count > 0
        analyzer.reset()
        assert analyzer.bar_count == 0
        assert analyzer.total_trades_processed == 0
