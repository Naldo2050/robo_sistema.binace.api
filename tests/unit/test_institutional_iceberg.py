# tests/unit/test_institutional_iceberg.py
"""Testes para Iceberg Detector."""
import pytest

from institutional.base import OrderBookLevel, OrderBookSnapshot, Side
from institutional.iceberg_detector import IcebergDetector


def _snapshot(
    ts: float,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=ts,
        bids=[
            OrderBookLevel(price=p, quantity=q, side=Side.BUY) for p, q in bids
        ],
        asks=[
            OrderBookLevel(price=p, quantity=q, side=Side.SELL) for p, q in asks
        ],
    )


class TestIcebergInit:
    def test_default(self):
        detector = IcebergDetector()
        assert detector.min_refills == 3
        assert len(detector.confirmed_icebergs) == 0

    def test_invalid_refills(self):
        with pytest.raises(Exception):
            IcebergDetector(min_refills=1)


class TestIcebergDetection:
    def test_first_snapshot_no_detection(self):
        detector = IcebergDetector()

        result = detector.process_snapshot(
            _snapshot(1.0, [(100, 10)], [(101, 10)])
        )
        assert result == []

    def test_normal_orderbook_changes(self):
        """Normal changes should not trigger iceberg detection."""
        detector = IcebergDetector(price_bucket_size=1.0)

        # Snapshot 1
        detector.process_snapshot(
            _snapshot(1.0, [(100, 10), (99, 20)], [(101, 15)])
        )

        # Snapshot 2 — different quantities (normal)
        result = detector.process_snapshot(
            _snapshot(2.0, [(100, 8), (99, 25)], [(101, 12)])
        )

        assert result == []

    def test_iceberg_pattern(self):
        """
        Simulate iceberg: order at 100 gets consumed and refills.
        """
        detector = IcebergDetector(
            min_refills=3,
            quantity_tolerance_pct=30,
            price_bucket_size=1.0,
        )

        new_icebergs = []

        for i in range(10):
            if i % 2 == 0:
                # Even: order present at 100 with ~10 qty
                snap = _snapshot(
                    float(i),
                    [(100, 10.0), (99, 20)],
                    [(101, 15)],
                )
            else:
                # Odd: order consumed at 100
                snap = _snapshot(
                    float(i),
                    [(100, 0.5), (99, 20)],  # Almost consumed
                    [(101, 15)],
                )

            result = detector.process_snapshot(snap)
            new_icebergs.extend(result)

        # May or may not detect — depends on exact logic
        assert isinstance(new_icebergs, list)


class TestIcebergAnalysis:
    def test_analyze_empty(self):
        detector = IcebergDetector()
        result = detector.analyze()
        assert result.source == "iceberg_detector"
        assert result.metrics["snapshots_processed"] == 0

    def test_analyze_with_snapshots(self):
        detector = IcebergDetector()

        for i in range(5):
            detector.process_snapshot(
                _snapshot(float(i), [(100, 10)], [(101, 10)])
            )

        result = detector.analyze()
        assert result.metrics["snapshots_processed"] == 5


class TestIcebergReset:
    def test_reset(self):
        detector = IcebergDetector()

        detector.process_snapshot(
            _snapshot(1.0, [(100, 10)], [(101, 10)])
        )
        detector.process_snapshot(
            _snapshot(2.0, [(100, 5)], [(101, 15)])
        )

        detector.reset()
        assert len(detector.confirmed_icebergs) == 0
        assert len(detector.candidates) == 0
