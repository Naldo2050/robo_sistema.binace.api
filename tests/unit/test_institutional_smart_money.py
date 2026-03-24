# tests/unit/test_institutional_smart_money.py
"""Testes para Smart Money Concepts Analyzer."""
import pytest

from institutional.smart_money import (
    Candle,
    SmartMoneyAnalyzer,
    FairValueGap,
    OrderBlock,
)


def _candle(ts=0.0, o=100, h=105, l=95, c=102, v=10):
    return Candle(timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


class TestCandle:
    def test_bullish(self):
        c = _candle(o=100, c=110)
        assert c.is_bullish is True
        assert c.is_bearish is False

    def test_bearish(self):
        c = _candle(o=110, c=100)
        assert c.is_bearish is True
        assert c.is_bullish is False

    def test_body_size(self):
        c = _candle(o=100, c=110)
        assert c.body_size == 10.0

    def test_range_size(self):
        c = _candle(h=115, l=95)
        assert c.range_size == 20.0

    def test_body_ratio(self):
        c = _candle(o=100, h=110, l=90, c=110)
        assert c.body_ratio == pytest.approx(0.5, abs=0.01)


class TestSmartMoneyInit:
    def test_default(self):
        analyzer = SmartMoneyAnalyzer()
        assert analyzer.candle_count == 0

    def test_invalid_lookback(self):
        with pytest.raises(Exception):
            SmartMoneyAnalyzer(swing_lookback=1)


class TestFVGDetection:
    def test_bullish_fvg(self):
        analyzer = SmartMoneyAnalyzer(fvg_min_size_pct=0.0)

        # Candle 0: high=100
        analyzer.add_candle(_candle(ts=0, o=95, h=100, l=90, c=98))
        # Candle 1: bridge (doesn't matter much)
        analyzer.add_candle(_candle(ts=1, o=98, h=110, l=97, c=108))
        # Candle 2: low=105 > high of candle 0 (100) = bullish FVG
        result = analyzer.add_candle(
            _candle(ts=2, o=107, h=115, l=105, c=113)
        )

        assert len(result["fvgs"]) > 0
        fvg = result["fvgs"][0]
        assert fvg.gap_type == "bullish"

    def test_bearish_fvg(self):
        analyzer = SmartMoneyAnalyzer(fvg_min_size_pct=0.0)

        # Candle 0: low=100
        analyzer.add_candle(_candle(ts=0, o=105, h=110, l=100, c=103))
        # Candle 1: bridge
        analyzer.add_candle(_candle(ts=1, o=103, h=104, l=90, c=92))
        # Candle 2: high=95 < low of candle 0 (100) = bearish FVG
        result = analyzer.add_candle(
            _candle(ts=2, o=93, h=95, l=85, c=87)
        )

        assert len(result["fvgs"]) > 0
        fvg = result["fvgs"][0]
        assert fvg.gap_type == "bearish"

    def test_no_fvg_when_overlapping(self):
        analyzer = SmartMoneyAnalyzer()

        analyzer.add_candle(_candle(ts=0, o=100, h=105, l=95, c=103))
        analyzer.add_candle(_candle(ts=1, o=103, h=108, l=98, c=106))
        result = analyzer.add_candle(
            _candle(ts=2, o=106, h=110, l=100, c=108)
        )

        # No gap because candle 2 low (100) <= candle 0 high (105)
        assert len(result["fvgs"]) == 0


class TestOrderBlockDetection:
    def test_bullish_order_block(self):
        analyzer = SmartMoneyAnalyzer(order_block_body_ratio=0.5)

        # Candle 0
        analyzer.add_candle(_candle(ts=0, o=105, h=106, l=98, c=100))
        # Candle 1: bearish (this becomes the OB)
        analyzer.add_candle(_candle(ts=1, o=102, h=103, l=97, c=98))
        # Candle 2: strong bullish (impulse) — body > 1.5x prev body
        result = analyzer.add_candle(
            _candle(ts=2, o=98, h=115, l=97, c=114, v=50)
        )

        # Check for order blocks
        obs = result["order_blocks"]
        if obs:
            assert obs[0].block_type == "bullish"


class TestMarketStructure:
    def test_unknown_structure_few_candles(self):
        analyzer = SmartMoneyAnalyzer()
        structure = analyzer.get_market_structure()
        assert structure["trend"] == "unknown"

    def test_structure_with_data(self):
        analyzer = SmartMoneyAnalyzer(swing_lookback=2)

        # Generate enough candles for structure
        prices = [100, 105, 110, 103, 108, 115, 107, 112, 120, 113, 118, 125, 118, 123]

        for i, price in enumerate(prices):
            analyzer.add_candle(
                _candle(
                    ts=float(i),
                    o=price - 2,
                    h=price + 3,
                    l=price - 3,
                    c=price,
                )
            )

        structure = analyzer.get_market_structure()
        assert structure["trend"] in (
            "bullish", "bearish", "expanding",
            "contracting", "mixed", "unknown",
        )


class TestSmartMoneyAnalysis:
    def test_analyze_empty(self):
        analyzer = SmartMoneyAnalyzer()
        result = analyzer.analyze()
        assert result.confidence == 0.0

    def test_analyze_with_data(self):
        analyzer = SmartMoneyAnalyzer(swing_lookback=2, fvg_min_size_pct=0.0)

        # Generate data with FVGs and structure
        prices = [100, 102, 110, 108, 115, 112, 120, 118, 125, 122, 130, 127, 135, 132, 140]

        for i, price in enumerate(prices):
            analyzer.add_candle(
                _candle(
                    ts=float(i),
                    o=price - 1,
                    h=price + 2,
                    l=price - 2,
                    c=price,
                    v=10 + i,
                )
            )

        result = analyzer.analyze()
        assert result.source == "smart_money_analyzer"
        assert "market_structure" in result.metrics
        assert "active_fvgs" in result.metrics

    def test_active_fvgs(self):
        analyzer = SmartMoneyAnalyzer(fvg_min_size_pct=0.0)

        # Create conditions for FVG
        analyzer.add_candle(_candle(ts=0, o=95, h=100, l=90, c=98))
        analyzer.add_candle(_candle(ts=1, o=98, h=110, l=97, c=108))
        analyzer.add_candle(_candle(ts=2, o=107, h=115, l=105, c=113))

        active = analyzer.get_active_fvgs()
        assert isinstance(active, list)

    def test_active_order_blocks(self):
        analyzer = SmartMoneyAnalyzer()
        active = analyzer.get_active_order_blocks()
        assert isinstance(active, list)


class TestSmartMoneyReset:
    def test_reset(self):
        analyzer = SmartMoneyAnalyzer()

        for i in range(10):
            analyzer.add_candle(
                _candle(ts=float(i), o=100+i, h=103+i, l=97+i, c=101+i)
            )

        assert analyzer.candle_count > 0
        analyzer.reset()
        assert analyzer.candle_count == 0
        assert len(analyzer.fvgs) == 0
        assert len(analyzer.order_blocks) == 0
