# tests/integration/paper_trading/test_replay_invariants.py
"""
Integration tests proving core simulation invariants on synthetic and recorded tapes:
- Invariant 2: Replay determinístico (identical ledger hash)
- Invariant 12: Placebo neutrality (directional accuracy converges to ~50% within Wilson bounds)
- Directional symmetry (LONG and SHORT on same tick never both win)
- Temporal causality (truncating tape past time T does not alter trades closed before T)
- Anti-lookahead protection (ticks before available_at cannot be filled)
- Conservation of attrition (every decision is strictly accounted for)
"""

import os
import random
from pathlib import Path
import pytest

from paper_trading.contracts import CanonicalDecision, PaperCostConfig
from paper_trading.executor import ExecutorConfig
from paper_trading.positions import PositionConfig
from paper_trading.replay import ReplayRunner
from paper_trading.scorecard import attrition_report, scorecard, wilson_interval
from paper_trading.tape import SyntheticTape, TapeReader


def generate_random_decisions(
    n_decisions=200,
    seed=42,
    tape=None,
    start_ts_ms=1_700_000_000_000,
    end_ts_ms=1_700_050_000_000,
    sl_dist=50.0,
    tp_dist=50.0,
    random_side=True,
):
    rng = random.Random(seed)
    decisions = []
    timestamps = sorted(rng.randint(start_ts_ms + 1000, end_ts_ms - 10_000) for _ in range(n_decisions))

    tape_lookup = {tick.T: tick.p for tick in tape} if tape else {}
    sorted_tape_ts = sorted(tape_lookup.keys()) if tape else []

    for i, ts in enumerate(timestamps):
        side = rng.choice(["LONG", "SHORT"]) if random_side else "LONG"
        if sorted_tape_ts:
            import bisect
            idx = bisect.bisect_right(sorted_tape_ts, ts) - 1
            ref_price = tape_lookup[sorted_tape_ts[max(0, idx)]]
        else:
            ref_price = 50_000.0

        if side == "LONG":
            sl = round(ref_price - sl_dist, 4)
            tp = round(ref_price + tp_dist, 4)
        else:
            sl = round(ref_price + sl_dist, 4)
            tp = round(ref_price - tp_dist, 4)

        d = CanonicalDecision(
            cohort_id="c_integ",
            symbol="BTCUSDT",
            window_id=f"win_{i}",
            decision_provider=f"prov_{i % 5}",
            strategy_version="1.0",
            signal_timestamp=ts - 50,
            decision_timestamp=ts,
            available_at=ts + 50,
            side=side,
            reference_price=ref_price,
            notional_usdt=100.0,
            horizon_s=30,
            stop_loss=sl,
            take_profit=tp,
        )
        decisions.append(d)
    return decisions


def test_deterministic_replay_runs():
    """Invariant 2: Two independent replay executions over identical inputs yield the exact same ledger hash."""
    tape1 = list(SyntheticTape(seed=123, n_ticks=1000, start_price=50_000.0, vol_bps=2.0))
    tape2 = list(SyntheticTape(seed=123, n_ticks=1000, start_price=50_000.0, vol_bps=2.0))

    decisions1 = generate_random_decisions(n_decisions=50, seed=999, end_ts_ms=tape1[-1].T)
    decisions2 = generate_random_decisions(n_decisions=50, seed=999, end_ts_ms=tape2[-1].T)

    runner1 = ReplayRunner(decisions1, tape1, db_path=":memory:")
    res1 = runner1.run()

    runner2 = ReplayRunner(decisions2, tape2, db_path=":memory:")
    res2 = runner2.run()

    assert res1.ledger_hash == res2.ledger_hash
    assert len(res1.closed_trades) == len(res2.closed_trades)
    assert len(res1.rejections) == len(res2.rejections)


def test_directional_symmetry_never_both_win():
    """LONG and SHORT on identical decision_timestamp and identical geometry NEVER both win."""
    tape = list(SyntheticTape(seed=777, n_ticks=2000, start_price=50_000.0, vol_bps=5.0))
    t0 = tape[100].T

    d_long = CanonicalDecision(
        cohort_id="c_sym_long",
        symbol="BTCUSDT",
        window_id="w_sym_1",
        decision_provider="prov_long",
        strategy_version="1.0",
        signal_timestamp=t0 - 50,
        decision_timestamp=t0,
        available_at=t0 + 20,
        side="LONG",
        reference_price=50_000.0,
        notional_usdt=100.0,
        horizon_s=30,
        stop_loss=49_900.0,
        take_profit=50_100.0,
    )
    d_short = CanonicalDecision(
        cohort_id="c_sym_short",
        symbol="BTCUSDT",
        window_id="w_sym_1",
        decision_provider="prov_short",
        strategy_version="1.0",
        signal_timestamp=t0 - 50,
        decision_timestamp=t0,
        available_at=t0 + 20,
        side="SHORT",
        reference_price=50_000.0,
        notional_usdt=100.0,
        horizon_s=30,
        stop_loss=50_100.0,
        take_profit=49_900.0,
    )

    runner = ReplayRunner([d_long, d_short], tape, db_path=":memory:")
    res = runner.run()

    closed_by_cohort = {t.cohort_id: t for t in res.closed_trades}
    t_long = closed_by_cohort.get("c_sym_long")
    t_short = closed_by_cohort.get("c_sym_short")

    if t_long and t_short and t_long.trade_win is not None and t_short.trade_win is not None:
        assert not (t_long.trade_win is True and t_short.trade_win is True)


def test_placebo_random_side_converges_to_fifty_percent():
    """
    Invariant 12: Directional accuracy on a symmetric zero-cost walk converges to ~50%
    within Wilson 99.7% confidence interval (z=3.0).
    """
    tape = list(SyntheticTape(seed=4242, n_ticks=6000, start_price=50_000.0, vol_bps=1.0, interval_ms=50))
    zero_cost = PaperCostConfig(
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        entry_slippage_bps=0.0,
        exit_slippage_bps=0.0,
        source="zero_cost",
        effective_at="2026-01-01T00:00:00Z",
    )

    decisions = generate_random_decisions(
        n_decisions=250,
        seed=101,
        tape=tape,
        start_ts_ms=tape[10].T,
        end_ts_ms=tape[-300].T,
        sl_dist=100.0,
        tp_dist=100.0,
        random_side=True,
    )

    runner = ReplayRunner(decisions, tape, cost_config=zero_cost, db_path=":memory:")
    res = runner.run()

    metrics = scorecard(res.closed_trades)
    decided = metrics.direction_correct_count + metrics.direction_incorrect_count
    assert decided > 50, f"Expected sufficient decided trades, got {decided}"

    ci_low, ci_high = wilson_interval(metrics.direction_correct_count, decided, z=3.0)
    assert ci_low <= 0.50 <= ci_high, (
        f"Placebo directional accuracy failed neutrality bound: correct={metrics.direction_correct_count}, "
        f"total={decided}, acc={metrics.directional_accuracy:.3f}, 99.7% CI=[{ci_low:.3f}, {ci_high:.3f}]"
    )


def test_temporal_causality_tape_truncation():
    """Truncating the tape at time T does not alter any trade closed before or at time T."""
    full_tape = list(SyntheticTape(seed=888, n_ticks=2000, start_price=50_000.0, vol_bps=3.0))
    decisions = generate_random_decisions(
        n_decisions=60,
        seed=77,
        start_ts_ms=full_tape[10].T,
        end_ts_ms=full_tape[1000].T,
        sl_dist=30.0,
        tp_dist=30.0,
    )

    full_runner = ReplayRunner(decisions, full_tape, db_path=":memory:")
    full_res = full_runner.run()

    closed_timestamps = sorted(t.closed_ts_ms for t in full_res.closed_trades)
    assert len(closed_timestamps) > 10
    cutoff_ts = closed_timestamps[len(closed_timestamps) // 2]

    full_pre_cutoff = [t for t in full_res.closed_trades if t.closed_ts_ms <= cutoff_ts]

    truncated_tape = [tick for tick in full_tape if tick.T <= cutoff_ts]
    trunc_runner = ReplayRunner(decisions, truncated_tape, db_path=":memory:")
    trunc_res = trunc_runner.run()

    trunc_pre_cutoff = [t for t in trunc_res.closed_trades if t.closed_ts_ms <= cutoff_ts]

    assert len(full_pre_cutoff) == len(trunc_pre_cutoff)
    full_map = {t.trade_id: (t.exit_reason, round(t.net_pnl_bps or 0.0, 4)) for t in full_pre_cutoff}
    trunc_map = {t.trade_id: (t.exit_reason, round(t.net_pnl_bps or 0.0, 4)) for t in trunc_pre_cutoff}
    assert full_map == trunc_map


def test_anti_lookahead_burst_at_decision_ts_cannot_be_filled():
    """Favorable burst at T == decision_timestamp cannot be exploited because fill requires T >= available_at."""
    t0 = 1_700_000_000_000
    burst_price = 51_000.0
    revert_price = 49_500.0

    from paper_trading.tape import TapeTick
    custom_tape = [
        TapeTick(T=t0 - 500, p=50_000.0, q=1.0, m=False),
        TapeTick(T=t0, p=burst_price, q=1.0, m=False),          # Burst exactly at decision_timestamp
        TapeTick(T=t0 + 200, p=burst_price, q=1.0, m=False),    # Inside latency window (available_at is t0 + 250)
        TapeTick(T=t0 + 260, p=revert_price, q=1.0, m=False),   # First tick at/after available_at (t0 + 260 >= t0 + 250)
        TapeTick(T=t0 + 10_000, p=revert_price, q=1.0, m=False),
    ]

    d = CanonicalDecision(
        cohort_id="c_lookahead",
        symbol="BTCUSDT",
        window_id="w_burst",
        decision_provider="prov_1",
        strategy_version="1.0",
        signal_timestamp=t0 - 100,
        decision_timestamp=t0,
        available_at=t0 + 250,
        side="LONG",
        reference_price=50_000.0,
        notional_usdt=100.0,
        horizon_s=10,
    )

    runner = ReplayRunner([d], custom_tape, db_path=":memory:")
    res = runner.run()

    # Fill MUST NOT occur at burst_price 51000; it must fill at revert_price 49500!
    assert len(res.fills) == 1
    assert res.fills[0].fill_timestamp == t0 + 260
    assert res.fills[0].raw_price == revert_price


def test_conservation_of_attrition():
    """Total decisions submitted must exactly equal rejections + closed trades."""
    tape = list(SyntheticTape(seed=333, n_ticks=1500, start_price=50_000.0, vol_bps=2.5))
    decisions = generate_random_decisions(
        n_decisions=50,
        seed=12,
        start_ts_ms=tape[10].T,
        end_ts_ms=tape[-50].T,
    )

    runner = ReplayRunner(decisions, tape, db_path=":memory:")
    res = runner.run()

    report = attrition_report(decisions, res.rejections, res.closed_trades)
    assert report.total_decisions == (report.rejections_count + report.closed_trades_count)
    assert report.balance_verified is True


def test_optional_real_dump_fixture():
    """Runs replay invariants against recorded raw trades fixture if present."""
    fixture_path = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "paper_trading" / "sample_trades_60s.jsonl"
    if not fixture_path.exists():
        pytest.skip("Fixture sample_trades_60s.jsonl not found.")

    tape = list(TapeReader(str(fixture_path)))
    assert len(tape) >= 5

    t0 = tape[0].T
    d = CanonicalDecision(
        cohort_id="c_dump_test",
        symbol="BTCUSDT",
        window_id="w_fixture",
        decision_provider="flow_v1",
        strategy_version="1.0",
        signal_timestamp=t0 - 20,
        decision_timestamp=t0 + 10,
        available_at=t0 + 20,
        side="LONG",
        reference_price=tape[0].p,
        notional_usdt=100.0,
        horizon_s=30,
    )

    runner = ReplayRunner([d], tape, db_path=":memory:")
    res = runner.run()
    assert len(res.closed_trades) + len(res.rejections) == 1
