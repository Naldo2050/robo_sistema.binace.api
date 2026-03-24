"""Testes para Whale Detector."""
import pytest
import time

from institutional.base import Side, Trade
from institutional.whale_detector import WhaleDetector, WhaleEvent


def _trade(price=50000.0, qty=1.0, side=Side.BUY, ts=None, value=None):
    ts = ts or time.time()
    t = Trade(timestamp=ts, price=price, quantity=qty, side=side)
    if value:
        t.value_usd = value
    return t


class TestWhaleInit:
    def test_default(self):
        detector = WhaleDetector()
        assert detector.threshold == 100_000
        assert len(detector.events) == 0

    def test_invalid_threshold(self):
        with pytest.raises(Exception):
            WhaleDetector(whale_threshold_usd=0)


class TestWhaleDetection:
    def test_small_trade_not_whale(self):
        detector = WhaleDetector(whale_threshold_usd=100_000)
        trade = _trade(price=50000, qty=0.5)  # $25k
        result = detector.process_trade(trade)
        assert result is None

    def test_large_trade_is_whale(self):
        detector = WhaleDetector(whale_threshold_usd=100_000, adaptive=False)
        trade = _trade(price=50000, qty=3.0)  # $150k
        result = detector.process_trade(trade)
        assert result is not None
        assert result.category == "whale"

    def test_mega_whale(self):
        detector = WhaleDetector(
            whale_threshold_usd=100_000,
            mega_whale_multiplier=5.0,
            adaptive=False,
        )
        trade = _trade(price=50000, qty=20.0)  # $1M
        result = detector.process_trade(trade)
        assert result is not None
        assert result.category in ("mega_whale", "institutional")

    def test_institutional(self):
        detector = WhaleDetector(
            whale_threshold_usd=100_000,
            institutional_multiplier=10.0,
            adaptive=False,
        )
        trade = _trade(price=50000, qty=40.0)  # $2M
        result = detector.process_trade(trade)
        assert result is not None
        assert result.category == "institutional"


class TestWhaleActivity:
    def test_empty_activity(self):
        detector = WhaleDetector()
        activity = detector.get_whale_activity()
        assert activity.total_whale_count == 0
        assert activity.total_whale_volume == 0

    def test_activity_with_events(self):
        detector = WhaleDetector(whale_threshold_usd=1000, adaptive=False)
        now = time.time()

        detector.process_trade(_trade(price=100, qty=20, side=Side.BUY, ts=now))
        detector.process_trade(_trade(price=100, qty=15, side=Side.SELL, ts=now))

        activity = detector.get_whale_activity(window_seconds=60)
        assert activity.total_whale_count == 2
        assert activity.buy_whale_count == 1
        assert activity.sell_whale_count == 1


class TestWhalePressure:
    def test_neutral_pressure(self):
        detector = WhaleDetector()
        pressure = detector.get_whale_pressure()
        assert pressure["direction"] == "neutral"

    def test_bullish_pressure(self):
        detector = WhaleDetector(whale_threshold_usd=1000, adaptive=False)
        now = time.time()

        for i in range(5):
            detector.process_trade(
                _trade(price=100, qty=20, side=Side.BUY, ts=now + i)
            )
        detector.process_trade(
            _trade(price=100, qty=10, side=Side.SELL, ts=now + 6)
        )

        pressure = detector.get_whale_pressure(window_seconds=60)
        assert pressure["pressure_score"] > 0


class TestWhaleAnalysis:
    def test_analyze_empty(self):
        detector = WhaleDetector()
        result = detector.analyze()
        assert result.source == "whale_detector"

    def test_analyze_with_data(self):
        detector = WhaleDetector(whale_threshold_usd=1000, adaptive=False)
        now = time.time()

        for i in range(10):
            side = Side.BUY if i % 3 != 0 else Side.SELL
            detector.process_trade(
                _trade(price=100, qty=20, side=side, ts=now + i)
            )

        result = detector.analyze()
        assert "whale_events_5min" in result.metrics


class TestWhaleReset:
    def test_reset(self):
        detector = WhaleDetector(whale_threshold_usd=1000, adaptive=False)
        detector.process_trade(_trade(price=100, qty=20))

        detector.reset()
        assert len(detector.events) == 0
        assert detector.total_trades == 0
