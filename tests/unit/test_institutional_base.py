# tests/unit/test_institutional_base.py
"""Testes para tipos base do arsenal institucional."""
import pytest

from institutional.base import (
    Trade,
    OrderBookLevel,
    OrderBookSnapshot,
    Signal,
    AnalysisResult,
    Side,
    MarketRegime,
    SignalStrength,
    InstitutionalError,
    InsufficientDataError,
    InvalidParameterError,
)


class TestTrade:
    def test_creation(self):
        trade = Trade(
            timestamp=1000.0,
            price=50000.0,
            quantity=0.5,
            side=Side.BUY,
        )
        assert trade.price == 50000.0
        assert trade.quantity == 0.5
        assert trade.value_usd == 25000.0

    def test_auto_value_usd(self):
        trade = Trade(timestamp=0, price=100, quantity=10, side=Side.SELL)
        assert trade.value_usd == 1000.0

    def test_custom_value_usd(self):
        trade = Trade(
            timestamp=0, price=100, quantity=10,
            side=Side.BUY, value_usd=999.0,
        )
        assert trade.value_usd == 999.0


class TestOrderBookSnapshot:
    def test_empty_snapshot(self):
        snap = OrderBookSnapshot(timestamp=1.0)
        assert snap.best_bid == 0.0
        assert snap.best_ask == 0.0
        assert snap.mid_price == 0.0
        assert snap.spread == 0.0

    def test_with_levels(self):
        snap = OrderBookSnapshot(
            timestamp=1.0,
            bids=[
                OrderBookLevel(100, 5, Side.BUY),
                OrderBookLevel(99, 10, Side.BUY),
            ],
            asks=[
                OrderBookLevel(101, 3, Side.SELL),
                OrderBookLevel(102, 7, Side.SELL),
            ],
        )
        assert snap.best_bid == 100.0
        assert snap.best_ask == 101.0
        assert snap.mid_price == 100.5
        assert snap.spread == 1.0
        assert snap.total_bid_depth == 15.0
        assert snap.total_ask_depth == 10.0
        assert snap.bid_ask_ratio == 1.5

    def test_spread_bps(self):
        snap = OrderBookSnapshot(
            timestamp=1.0,
            bids=[OrderBookLevel(100, 1, Side.BUY)],
            asks=[OrderBookLevel(101, 1, Side.SELL)],
        )
        # spread = 1, mid = 100.5
        # bps = (1/100.5) * 10000 ≈ 99.5
        assert snap.spread_bps == pytest.approx(99.5, abs=1.0)


class TestSignal:
    def test_to_dict(self):
        signal = Signal(
            timestamp=1.0,
            signal_type="test",
            direction=Side.BUY,
            strength=SignalStrength.STRONG,
            price=100.0,
            confidence=0.8,
            source="test_source",
        )

        d = signal.to_dict()
        assert d["direction"] == "buy"
        assert d["strength"] == "strong"
        assert d["confidence"] == 0.8


class TestAnalysisResult:
    def test_to_dict(self):
        result = AnalysisResult(
            source="test",
            regime=MarketRegime.TRENDING_UP,
            confidence=0.9,
        )

        d = result.to_dict()
        assert d["source"] == "test"
        assert d["regime"] == "trending_up"
        assert d["confidence"] == 0.9


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(InsufficientDataError, InstitutionalError)
        assert issubclass(InvalidParameterError, InstitutionalError)
        assert issubclass(InstitutionalError, Exception)

    def test_raise_catch(self):
        with pytest.raises(InstitutionalError):
            raise InsufficientDataError("not enough data")

    def test_error_message(self):
        try:
            raise InvalidParameterError("bad param")
        except InvalidParameterError as e:
            assert "bad param" in str(e)


class TestEnums:
    def test_side_values(self):
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    def test_regime_values(self):
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.RANGING.value == "ranging"

    def test_strength_values(self):
        assert SignalStrength.STRONG.value == "strong"
        assert SignalStrength.WEAK.value == "weak"
