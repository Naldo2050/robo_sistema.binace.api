"""Diagnóstico reproduzível do limite temporal de ``FlowAnalyzer.flow_trades``.

Este teste reduz apenas a capacidade da deque na instância sob teste. Nenhuma
configuração ou implementação de produção é alterada.
"""

from collections import deque
from decimal import Decimal

import pytest

from flow_analyzer import RollingAggregate
import flow_analyzer.core as flow_core
from flow_analyzer.core import FlowAnalyzer


BASE_TS = 1_000_000_000_000
TRADES_PER_SECOND = 2
SIMULATED_SECONDS = 900
INJECTED_MAXLEN = 1_000


def _trade(ts: int) -> dict:
    return {
        "q": 1.0,
        "T": ts,
        "p": 100.0,
        "m": False,
    }


def _record(ts: int, price=100.0, qty=1.0, delta_btc=1.0, side="buy") -> dict:
    return {
        "ts": ts,
        "price": price,
        "qty": qty,
        "delta_btc": delta_btc,
        "delta_usd": delta_btc * price,
        "side": side,
        "sector": "retail",
    }


def _aggregate(records, window_min=1, max_trades=100):
    aggregate = RollingAggregate(window_min=window_min, max_trades=max_trades)
    for record in records:
        aggregate.add_trade(record, whale_threshold=999.0)
    return aggregate


def test_temporal_boundaries_are_closed_on_both_sides():
    start_ms = 1_000
    end_ms = 2_000
    records = [
        _record(start_ms - 1),
        _record(start_ms),
        _record(end_ms),
        _record(end_ms + 1),
    ]

    flow_delta, _, _ = FlowAnalyzer._calc_from_trades(None, records, start_ms, end_ms)
    aggregate = _aggregate(records[:3], window_min=1)
    aggregate.prune(start_ms)

    assert flow_delta == 200.0
    assert aggregate.get_metrics()["sum_delta_usd"] == 200.0
    assert [trade[0] for trade in aggregate.trades] == [start_ms, end_ms]

    future_trade_aggregate = _aggregate(records, window_min=1)
    future_trade_aggregate.prune(start_ms)
    assert future_trade_aggregate.get_metrics()["sum_delta_usd"] == 300.0


@pytest.mark.parametrize("maxlen", [1, 2, 5, 100])
def test_explicit_eviction_matches_deque_maxlen_for_normal_sequence(maxlen):
    analyzer = FlowAnalyzer()
    analyzer.flow_trades_maxlen = maxlen
    expected = deque(maxlen=maxlen)
    records = [_record(index, price=float(index)) for index in range(maxlen + 2)]

    for record in records:
        expected.append(record)
        analyzer._append_flow_trade(record, record["ts"])

    assert list(analyzer.flow_trades) == list(expected)
    assert analyzer._flow_trades_capacity_evictions_total == 2


@pytest.mark.parametrize("maxlen", [0, -1])
def test_flow_trades_maxlen_rejects_non_positive_configuration(monkeypatch, maxlen):
    monkeypatch.setattr(flow_core.config_module, "FLOW_TRADES_MAXLEN", maxlen, raising=False)

    with pytest.raises(ValueError, match="FLOW_TRADES_MAXLEN"):
        FlowAnalyzer()


def test_flow_trades_capacity_warning_is_rate_limited(caplog):
    analyzer = FlowAnalyzer()
    analyzer.flow_trades_maxlen = 2

    with caplog.at_level("WARNING"):
        for index, reference_ts in enumerate((0, 10_000, 30_001, 60_000, 60_001)):
            analyzer._append_flow_trade(_record(index, price=100.0), reference_ts)

    records = [record for record in caplog.records if "flow_trades_capacity_truncated" in record.message]
    assert len(records) == 2
    assert all("capacity=2" in record.message for record in records)


def test_flow_window_integrity_warmup_without_eviction():
    analyzer = FlowAnalyzer()
    now_ms = 100_000
    analyzer._flow_first_trade_ts = now_ms - 10_000
    snapshot = {"flow_trades": [_record(now_ms - 1_000)],
                "_flow_first_trade_ts": analyzer._flow_first_trade_ts,
                "_flow_trades_capacity_evictions_total": 0}

    integrity = analyzer._get_flow_window_integrity(snapshot, now_ms)

    assert integrity["1m"]["status"] == "WARMING_UP"
    assert integrity["1m"]["is_temporal_coverage_valid"] is False


def test_flow_window_integrity_full_at_99_percent():
    analyzer = FlowAnalyzer()
    now_ms = 100_000
    snapshot = {
        "flow_trades": [_record(now_ms - 60_000), _record(now_ms)],
        "_flow_first_trade_ts": now_ms - 60_000,
        "_flow_trades_capacity_evictions_total": 0,
    }

    integrity = analyzer._get_flow_window_integrity(snapshot, now_ms)

    assert integrity["1m"]["status"] == "FULL"
    assert integrity["1m"]["effective_coverage_pct"] == 100.0


def test_flow_window_integrity_uses_timestamp_extrema_for_ooo():
    analyzer = FlowAnalyzer()
    now_ms = 100_000
    snapshot = {
        "flow_trades": [_record(now_ms), _record(now_ms - 60_000)],
        "_flow_first_trade_ts": now_ms - 60_000,
        "_flow_trades_capacity_evictions_total": 0,
    }

    integrity = analyzer._get_flow_window_integrity(snapshot, now_ms)

    assert integrity["1m"]["status"] == "FULL"


@pytest.mark.parametrize(
    ("last_eviction_offset", "expected_status"),
    [(-1, "WARMING_UP"), (0, "CAPACITY_TRUNCATED"), (1, "CAPACITY_TRUNCATED")],
)
def test_capacity_eviction_relevance_is_temporal(last_eviction_offset, expected_status):
    analyzer = FlowAnalyzer()
    now_ms = 100_000
    window_start = now_ms - 60_000
    snapshot = {
        "flow_trades": [_record(window_start + 1_000), _record(now_ms)],
        "_flow_first_trade_ts": 1,
        "_flow_trades_capacity_evictions_total": 1,
        "_flow_trades_last_capacity_eviction_ts": window_start + last_eviction_offset,
    }

    integrity = analyzer._get_flow_window_integrity(snapshot, now_ms)

    assert integrity["1m"]["status"] == expected_status
    assert integrity["1m"]["is_temporal_coverage_valid"] is False


def test_flow_window_integrity_recovers_after_capacity_eviction():
    analyzer = FlowAnalyzer()
    analyzer.flow_trades_maxlen = 100
    now_ms = 100_000
    for index in range(200):
        analyzer._append_flow_trade(_record(1_000 + index), 1_000 + index)
    for index in range(61):
        analyzer._append_flow_trade(_record(40_000 + index * 1_000), 40_000 + index * 1_000)

    snapshot = analyzer._create_snapshot(now_ms)
    integrity = analyzer._get_flow_window_integrity(snapshot, now_ms)

    assert analyzer._flow_trades_capacity_evictions_total > 0
    assert integrity["1m"]["status"] == "FULL"


def test_flow_window_integrity_15m_recovers_after_capacity_eviction():
    analyzer = FlowAnalyzer()
    analyzer.flow_trades_maxlen = 1_000

    for index in range(1_500):
        analyzer._append_flow_trade(_record(1_000 + index), 1_000 + index)
    for index in range(901):
        timestamp = 2_000 + index * 1_000
        analyzer._append_flow_trade(_record(timestamp), timestamp)

    now_ms = 902_000
    snapshot = analyzer._create_snapshot(now_ms)
    integrity = analyzer._get_flow_window_integrity(snapshot, now_ms)

    assert analyzer._flow_trades_capacity_evictions_total > 0
    assert integrity["15m"]["status"] == "FULL"
    assert integrity["15m"]["is_temporal_coverage_valid"] is True


def test_out_of_order_is_rejected_by_rolling_aggregate_but_not_raw_calculation():
    records = [_record(1_000), _record(3_000), _record(2_000)]

    flow_delta, _, _ = FlowAnalyzer._calc_from_trades(None, records, 1_000, 3_000)
    aggregate = _aggregate(records, window_min=1)

    assert flow_delta == 300.0
    assert [trade[0] for trade in aggregate.trades] == [1_000, 3_000]
    assert aggregate.get_metrics()["sum_delta_usd"] == 200.0


def test_timestamp_delay_policy_preserves_raw_timestamps():
    analyzer = FlowAnalyzer()
    reference_ms = 100_000

    for delay_ms in (100, 5_000, 29_000, 30_001):
        adjusted, was_adjusted = analyzer._adjust_timestamp_if_needed(
            reference_ms - delay_ms, reference_ms
        )
        assert adjusted == reference_ms - delay_ms
        assert was_adjusted is False


@pytest.mark.parametrize(
    "records, expected_flow_ohlc, expected_aggregate_ohlc",
    [
        (
            [_record(1_000, 10.0), _record(2_000, 30.0), _record(3_000, 20.0)],
            (10.0, 30.0, 10.0, 20.0),
            (10.0, 30.0, 10.0, 20.0),
        ),
        (
            [_record(1_000, 10.0), _record(2_000, 10.0), _record(3_000, 20.0)],
            (10.0, 20.0, 10.0, 20.0),
            (10.0, 20.0, 10.0, 20.0),
        ),
        (
            [_record(1_000, 10.0), _record(3_000, 30.0), _record(2_000, 20.0)],
            (10.0, 30.0, 10.0, 20.0),
            (10.0, 30.0, 10.0, 30.0),
        ),
    ],
)
def test_ohlc_comparison(records, expected_flow_ohlc, expected_aggregate_ohlc):
    _, _, flow_ohlc = FlowAnalyzer._calc_from_trades(None, records, 1_000, 3_000)
    aggregate_ohlc = _aggregate(records, window_min=1).get_metrics()["ohlc"]

    assert flow_ohlc == expected_flow_ohlc
    assert aggregate_ohlc == expected_aggregate_ohlc


def test_decimal_precision_and_buy_sell_totals_match():
    records = []
    for index in range(1_000):
        qty = Decimal("0.00000001") * (index + 1)
        price = Decimal("0.12345678") + Decimal(index % 7) / Decimal("100000")
        side = "buy" if index % 2 == 0 else "sell"
        delta_btc = qty if side == "buy" else -qty
        records.append(_record(
            index + 1,
            price=price,
            qty=qty,
            delta_btc=delta_btc,
            side=side,
        ))

    flow_delta_usd, flow_delta_btc, _ = FlowAnalyzer._calc_from_trades(
        None, records, 1, 1_000
    )
    metrics = _aggregate(records, window_min=15, max_trades=2_000).get_metrics()

    assert float(flow_delta_usd) == pytest.approx(metrics["sum_delta_usd"])
    assert float(flow_delta_btc) == pytest.approx(
        metrics["sum_buy_btc"] - metrics["sum_sell_btc"]
    )
    assert metrics["sum_buy_usd"] - metrics["sum_sell_usd"] == pytest.approx(
        metrics["sum_delta_usd"]
    )


def test_flow_trades_truncates_15m_without_integrity_metadata(monkeypatch):
    analyzer = FlowAnalyzer()
    analyzer.flow_trades_maxlen = INJECTED_MAXLEN

    total_trades = SIMULATED_SECONDS * TRADES_PER_SECOND + 2
    newest_ts = BASE_TS + (total_trades - 1) * 500
    monkeypatch.setattr(analyzer, "_get_synced_timestamp_ms", lambda: newest_ts)

    records = []
    for index in range(total_trades):
        ts = BASE_TS + index * 500
        records.append({
            "ts": ts,
            "price": 100.0,
            "qty": 1.0,
            "delta_btc": 1.0,
            "delta_usd": 100.0,
            "side": "buy",
            "sector": "retail",
        })

    for record in records:
        analyzer._append_flow_trade(record, record["ts"])
    analyzer._last_price = 100.0
    for aggregate in analyzer._window_aggregates.values():
        for record in records:
            aggregate.add_trade(record, whale_threshold=float(analyzer.whale_threshold))

    metrics = analyzer.get_flow_metrics(reference_epoch_ms=newest_ts)
    order_flow = metrics["order_flow"]
    oldest_ts = analyzer.flow_trades[0]["ts"]
    effective_duration_sec = (newest_ts - oldest_ts) / 1000
    coverage_pct = effective_duration_sec / (SIMULATED_SECONDS * 60 / 60) * 100

    assert len(analyzer.flow_trades) == INJECTED_MAXLEN
    assert analyzer._flow_trades_capacity_evictions_total == total_trades - INJECTED_MAXLEN
    assert analyzer._flow_trades_last_capacity_eviction_ts == newest_ts
    assert effective_duration_sec < 900
    assert "net_flow_15m" in order_flow

    integrity = metrics["flow_window_integrity"]
    assert integrity["1m"]["status"] == "FULL"
    assert integrity["5m"]["status"] == "FULL"
    assert integrity["15m"]["status"] == "CAPACITY_TRUNCATED"
    assert integrity["15m"]["is_temporal_coverage_valid"] is False

    one_minute = analyzer._window_aggregates[1].get_metrics(100.0)
    five_minutes = analyzer._window_aggregates[5].get_metrics(100.0)
    fifteen_minutes = analyzer._window_aggregates[15].get_metrics(100.0)

    assert order_flow["net_flow_1m"] == pytest.approx(one_minute["sum_delta_usd"])
    assert order_flow["net_flow_5m"] == pytest.approx(five_minutes["sum_delta_usd"])
    assert order_flow["net_flow_15m"] != pytest.approx(fifteen_minutes["sum_delta_usd"])

    assert one_minute["window_status"] == "FULL"
    assert five_minutes["window_status"] == "FULL"
    assert fifteen_minutes["window_status"] == "FULL"
    assert fifteen_minutes["sum_delta_usd"] > order_flow["net_flow_15m"]

# End of diagnostic contracts.
