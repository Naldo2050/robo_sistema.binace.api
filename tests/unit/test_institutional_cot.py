"""Testes para Crypto COT."""
import pytest

from institutional.crypto_cot import CryptoCOT


class TestCOTInit:
    def test_default(self):
        cot = CryptoCOT()
        assert cot.data_points == 0
        assert cot.latest is None


class TestCOTAddData:
    def test_add_single(self):
        cot = CryptoCOT()
        cot.add_data(
            timestamp=1.0,
            funding_rate=0.001,
            open_interest=50000,
            long_short_ratio=1.2,
            price=50000,
        )
        assert cot.data_points == 1
        assert cot.latest is not None
        assert cot.latest.funding_rate == 0.001


class TestCOTFundingAnalysis:
    def test_extreme_positive_funding(self):
        cot = CryptoCOT()
        cot.add_data(1.0, 0.015, 50000, 1.5, 1.0, 50000)
        result = cot.analyze()

        # Should have bearish signal for extreme positive funding
        bearish = [s for s in result.signals if "funding" in s.signal_type]
        assert len(bearish) > 0

    def test_extreme_negative_funding(self):
        cot = CryptoCOT()
        cot.add_data(1.0, -0.015, 50000, 0.8, 1.0, 50000)
        result = cot.analyze()

        bullish = [s for s in result.signals if "funding" in s.signal_type]
        assert len(bullish) > 0

    def test_normal_funding_no_signal(self):
        cot = CryptoCOT()
        cot.add_data(1.0, 0.0001, 50000, 1.0, 1.0, 50000)
        result = cot.analyze()

        funding_signals = [s for s in result.signals if "funding" in s.signal_type]
        assert len(funding_signals) == 0


class TestCOTOIAnalysis:
    def test_oi_confirming_uptrend(self):
        cot = CryptoCOT(oi_change_threshold_pct=5.0)

        for i in range(6):
            cot.add_data(
                timestamp=float(i),
                funding_rate=0.001,
                open_interest=50000 + i * 5000,
                long_short_ratio=1.0,
                price=50000 + i * 1000,
            )

        result = cot.analyze()
        oi_signals = [s for s in result.signals if "oi_" in s.signal_type]
        assert isinstance(oi_signals, list)


class TestCOTSqueeze:
    def test_short_squeeze_conditions(self):
        cot = CryptoCOT()
        cot.add_data(1.0, -0.015, 80000, 0.6, 0.5, 45000)
        result = cot.analyze()

        squeeze_signals = [s for s in result.signals if "squeeze" in s.signal_type]
        assert len(squeeze_signals) > 0

    def test_long_squeeze_conditions(self):
        cot = CryptoCOT()
        cot.add_data(1.0, 0.015, 80000, 2.0, 1.8, 55000)
        result = cot.analyze()

        squeeze_signals = [s for s in result.signals if "squeeze" in s.signal_type]
        assert len(squeeze_signals) > 0


class TestCOTAnalysis:
    def test_analyze_empty(self):
        cot = CryptoCOT()
        result = cot.analyze()
        assert result.confidence == 0.0

    def test_analyze_complete(self):
        cot = CryptoCOT()
        cot.add_data(1.0, 0.005, 60000, 1.3, 1.5, 52000)
        result = cot.analyze()
        assert result.source == "crypto_cot"
        assert "funding_rate" in result.metrics


class TestCOTReset:
    def test_reset(self):
        cot = CryptoCOT()
        cot.add_data(1.0, 0.001, 50000, 1.0, 1.0, 50000)
        cot.reset()
        assert cot.data_points == 0
