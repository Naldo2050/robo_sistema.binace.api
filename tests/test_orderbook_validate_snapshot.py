# tests/test_orderbook_validate_snapshot.py
from __future__ import annotations

from orderbook_analyzer import OrderBookAnalyzer, ORDERBOOK_MAX_AGE_MS
from tests.conftest import make_valid_snapshot


def test_validate_snapshot_ok(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    snap = make_valid_snapshot(tm.now_ms())
    ok, issues, converted = oba._validate_snapshot(snap)
    assert ok is True
    assert issues == [] or isinstance(issues, list)
    assert "bids" in converted and "asks" in converted
    assert isinstance(converted["bids"][0][0], float)


def test_validate_snapshot_missing_timestamp_fails(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    snap = make_valid_snapshot(tm.now_ms())
    snap.pop("E", None)  # remove timestamp
    ok, issues, _ = oba._validate_snapshot(snap)
    assert ok is False
    assert any("timestamp" in s for s in issues)


def test_validate_snapshot_too_old_fails(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    old_ts = tm.now_ms() - (ORDERBOOK_MAX_AGE_MS + 1_000)
    snap = make_valid_snapshot(old_ts)
    ok, issues, _ = oba._validate_snapshot(snap)
    assert ok is False
    assert any("muito antigos" in s for s in issues)


def test_validate_snapshot_bids_out_of_order_fails(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    snap = make_valid_snapshot(tm.now_ms())
    # invalida ordenação: bid2 maior que bid1
    snap["bids"][1] = (101.0, 1.0)
    ok, issues, _ = oba._validate_snapshot(snap)
    assert ok is False
    assert any("bids fora de ordem" in s for s in issues)


def test_validate_snapshot_asks_out_of_order_fails(tm):
    oba = OrderBookAnalyzer(symbol="BTCUSDT", time_manager=tm)
    snap = make_valid_snapshot(tm.now_ms())
    # invalida ordenação: ask2 menor que ask1
    snap["asks"][1] = (99.0, 1.0)
    ok, issues, _ = oba._validate_snapshot(snap)
    assert ok is False
    assert any("asks fora de ordem" in s for s in issues)