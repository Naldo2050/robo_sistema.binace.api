# tests/unit/paper_trading/test_scorecard.py
"""Unit tests for the performance scorecard, Wilson intervals, net payoff break-even, and placebo validation."""

import pytest

from paper_trading.contracts import ClosedTrade
from paper_trading.scorecard import (
    breakeven_win_rate_gross_plus_costs,
    breakeven_win_rate_payoff,
    breakeven_win_rate_rr,
    calibration_table,
    group_by,
    scorecard,
    wilson_interval,
)


def make_dummy_trade(
    trade_id="tr_1",
    decision_provider="flow_v1",
    exit_reason="TAKE_PROFIT",
    gross_pnl_bps=100.0,
    net_pnl_bps=90.0,
    trade_direction_profitable=True,
    prediction_direction_correct=None,
    trade_win=True,
    confidence=None,
    regime="TRENDING",
):
    notional = 100.0
    gross_usdt = (gross_pnl_bps / 10_000.0) * notional if gross_pnl_bps is not None else None
    net_usdt = (net_pnl_bps / 10_000.0) * notional if net_pnl_bps is not None else None
    return ClosedTrade(
        trade_id=trade_id,
        decision_id=f"dec_{trade_id}",
        cohort_id="c1",
        decision_provider=decision_provider,
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        exit_price=110.0,
        quantity=1.0,
        notional_usdt=notional,
        opened_ts_ms=1000,
        closed_ts_ms=2000,
        exit_reason=exit_reason,
        trade_direction_profitable=trade_direction_profitable,
        prediction_direction_correct=prediction_direction_correct,
        trade_win=trade_win,
        gross_pnl_bps=gross_pnl_bps,
        net_pnl_bps=net_pnl_bps,
        fees_bps=8.0,
        slippage_bps=2.0,
        funding_bps=0.0,
        gross_pnl_usdt=gross_usdt,
        fees_usdt=0.08,
        slippage_usdt=0.02,
        funding_usdt=0.0,
        net_pnl_usdt=net_usdt,
        pnl_R=1.0 if net_usdt is not None else None,
        costs_complete=True if net_usdt is not None else False,
        data_gap=False,
        mae_bps=10.0,
        mfe_bps=50.0,
        ticks_count=20,
        direction_correct=trade_direction_profitable,
        context={"confidence": confidence, "regime": regime},
    )


def test_wilson_interval_formula():
    """wilson(60, 100) ≈ (0.502, 0.691) ±0.002."""
    low, high = wilson_interval(wins=60, n=100, z=1.96)
    assert abs(low - 0.502) <= 0.002
    assert abs(high - 0.691) <= 0.002


def test_breakeven_win_rate_payoff():
    """Item 6: Net payoff break-even win rate p_be = Lnet / (Wnet + Lnet)."""
    # Wnet = 100 bps, Lnet = 50 bps -> p_be = 50 / (100 + 50) = 50 / 150 = 0.3333
    net_wins = [100.0, 100.0]
    net_losses = [-50.0, -50.0]
    p_be = breakeven_win_rate_payoff(net_wins, net_losses)
    assert p_be is not None
    assert pytest.approx(p_be, rel=1e-4) == (50.0 / 150.0)

    # Incomplete data returns None
    assert breakeven_win_rate_payoff([], [-50.0]) is None
    assert breakeven_win_rate_payoff([100.0], []) is None

    # Alternative gross + costs formula
    p_be_gross = breakeven_win_rate_gross_plus_costs(average_win_gross=100.0, average_loss_gross=100.0, average_cost=10.0)
    assert p_be_gross is not None
    assert pytest.approx(p_be_gross, rel=1e-5) == 0.55


def test_placebo_directional_no_edge():
    """Invariant 12: Placebo ~50% test applies to directional profitability, and net expectancy shows no edge."""
    trades = []
    for i in range(100):
        if i < 50:
            # Direction correct: gross = +50 bps, net = +40 bps
            t = make_dummy_trade(
                trade_id=f"t_{i}",
                decision_provider="placebo",
                trade_direction_profitable=True,
                trade_win=True,
                gross_pnl_bps=50.0,
                net_pnl_bps=40.0,
            )
        else:
            # Direction incorrect: gross = -50 bps, net = -60 bps
            t = make_dummy_trade(
                trade_id=f"t_{i}",
                decision_provider="placebo",
                trade_direction_profitable=False,
                trade_win=False,
                gross_pnl_bps=-50.0,
                net_pnl_bps=-60.0,
            )
        trades.append(t)

    metrics = scorecard(trades)

    # 1. Trade direction profitability is exactly 50%
    assert metrics.trade_direction_profitability_rate == 0.50
    assert metrics.trade_direction_profitable_count == 50
    assert metrics.directional_accuracy == 0.50

    # 2. Net win rate is 50%, but economic advantage vanishes due to costs:
    # Expectancy in bps = (50*40 + 50*(-60)) / 100 = -10 bps
    assert metrics.expectancy_bps is not None
    assert metrics.expectancy_bps < 0.0
    assert pytest.approx(metrics.expectancy_bps, rel=1e-5) == -10.0


def test_flats_and_unknowns_excluded_from_win_rate():
    """Win rate denominator strictly excludes flats and unknowns."""
    trades = [
        make_dummy_trade("t1", gross_pnl_bps=100.0, net_pnl_bps=90.0, trade_win=True),
        make_dummy_trade("t2", gross_pnl_bps=150.0, net_pnl_bps=140.0, trade_win=True),
        make_dummy_trade("t3", gross_pnl_bps=-50.0, net_pnl_bps=-60.0, trade_win=False),
        make_dummy_trade("t4", gross_pnl_bps=10.0, net_pnl_bps=0.0, trade_win=False),  # FLAT net
        make_dummy_trade("t5", exit_reason="UNKNOWN_DATA_GAP", gross_pnl_bps=None, net_pnl_bps=None, trade_direction_profitable=None, trade_win=None),
    ]

    metrics = scorecard(trades)
    assert metrics.total_trades == 5
    assert metrics.wins == 2
    assert metrics.losses == 1
    assert metrics.flats == 1
    assert metrics.unknown == 1

    assert metrics.win_rate is not None
    assert pytest.approx(metrics.win_rate, rel=1e-3) == (2.0 / 3.0)


def test_group_by_categories():
    """group_by correctly segments trades by decision_provider, exit_reason, and context."""
    trades = [
        make_dummy_trade("t1", decision_provider="prov_A", exit_reason="TAKE_PROFIT", regime="BULL"),
        make_dummy_trade("t2", decision_provider="prov_A", exit_reason="STOP_LOSS", regime="BEAR"),
        make_dummy_trade("t3", decision_provider="prov_B", exit_reason="TAKE_PROFIT", regime="BULL"),
    ]

    by_prov = group_by(trades, "decision_provider")
    assert len(by_prov["prov_A"]) == 2
    assert len(by_prov["prov_B"]) == 1

    by_reason = group_by(trades, "exit_reason")
    assert len(by_reason["TAKE_PROFIT"]) == 2
    assert len(by_reason["STOP_LOSS"]) == 1


def test_calibration_table_with_none():
    """calibration_table ignores None confidence and isolates it without coercing to 0.0."""
    trades = [
        make_dummy_trade("t1", trade_win=True, confidence=0.75),
        make_dummy_trade("t2", trade_win=True, confidence=0.72),
        make_dummy_trade("t3", trade_win=False, confidence=0.55),
        make_dummy_trade("t4", trade_win=True, confidence=None),
    ]

    report = calibration_table(trades, bins=((0.5, 0.7), (0.7, 0.9)))
    assert report.none_count == 1
    assert pytest.approx(report.fraction_none, rel=1e-4) == 0.25

    bin_high = report.bins[1]
    assert bin_high.bin_range == (0.7, 0.9)
    assert bin_high.total_in_bin == 2
    assert bin_high.wins_in_bin == 2
    assert pytest.approx(bin_high.win_rate, rel=1e-4) == 1.0


def test_scorecard_denominators_and_metric_universes():
    """Prove exact mathematical isolation of trade direction profitability vs prediction accuracy vs net win rate."""
    # 5 trades matching the preflight scenario:
    # 4 gross profitable (80%), 1 gross loss (20%)
    # 2 net profitable (40%), 3 net losses (60%)
    # prediction_direction_correct is None (not evaluated yet)
    trades = [
        make_dummy_trade("t1", gross_pnl_bps=50.0, net_pnl_bps=30.0, trade_direction_profitable=True, trade_win=True, prediction_direction_correct=None),
        make_dummy_trade("t2", gross_pnl_bps=40.0, net_pnl_bps=20.0, trade_direction_profitable=True, trade_win=True, prediction_direction_correct=None),
        make_dummy_trade("t3", gross_pnl_bps=10.0, net_pnl_bps=-10.0, trade_direction_profitable=True, trade_win=False, prediction_direction_correct=None),
        make_dummy_trade("t4", gross_pnl_bps=15.0, net_pnl_bps=-5.0, trade_direction_profitable=True, trade_win=False, prediction_direction_correct=None),
        make_dummy_trade("t5", gross_pnl_bps=-30.0, net_pnl_bps=-50.0, trade_direction_profitable=False, trade_win=False, prediction_direction_correct=None),
    ]

    metrics = scorecard(trades)

    # 1. Trade Direction Profitability Universe (Gross PnL > 0)
    assert metrics.trade_direction_profitable_count == 4
    assert metrics.trade_direction_unprofitable_count == 1
    assert metrics.trade_direction_decided_count == 5
    assert metrics.direction_correct_count == 4
    assert metrics.direction_incorrect_count == 1
    # Prove that denominator is strictly 5 and NOT 7 (4 correct + 3 net losses)
    assert (metrics.direction_correct_count + metrics.direction_incorrect_count) == 5
    assert metrics.trade_direction_profitability_rate == 0.80
    assert metrics.trade_direction_profitability_rate != pytest.approx(4.0 / 7.0)
    assert pytest.approx(4.0 / 7.0, rel=1e-3) == 0.5714

    # 2. Prediction Accuracy Universe (Resolved market predictions)
    # None predictions do NOT enter denominator
    assert metrics.prediction_direction_correct_count == 0
    assert metrics.prediction_direction_decided_count == 0
    assert metrics.prediction_accuracy is None

    # 3. Net Win Rate Universe (Post-costs economic outcomes)
    assert metrics.wins == 2
    assert metrics.losses == 3
    assert metrics.flats == 0
    assert metrics.unknown == 0
    assert metrics.win_rate == 0.40  # 2 / 5


def test_scorecard_prediction_direction_distinct_from_trade_direction():
    """Prove trade_direction_profitable does not equal prediction_direction_correct."""
    # Trade with gross profit due to favourable market movement, but prediction was wrong or independent
    trades = [
        make_dummy_trade("t1", trade_direction_profitable=True, prediction_direction_correct=False),
        make_dummy_trade("t2", trade_direction_profitable=False, prediction_direction_correct=True),
        make_dummy_trade("t3", trade_direction_profitable=True, prediction_direction_correct=None),
    ]
    metrics = scorecard(trades)

    assert metrics.trade_direction_profitable_count == 2
    assert metrics.trade_direction_decided_count == 3
    assert metrics.trade_direction_profitability_rate == pytest.approx(2.0 / 3.0)

    assert metrics.prediction_direction_correct_count == 1
    assert metrics.prediction_direction_decided_count == 2  # t3 with None is excluded!
    assert metrics.prediction_accuracy == pytest.approx(1.0 / 2.0)


def test_scorecard_cost_incomplete_excluded_from_net_win_rate():
    """Cost-incomplete trades are recorded as unknown and excluded from net win rate denominator."""
    trades = [
        make_dummy_trade("t1", gross_pnl_bps=100.0, net_pnl_bps=90.0, trade_win=True),
        make_dummy_trade("t2", gross_pnl_bps=-50.0, net_pnl_bps=-60.0, trade_win=False),
        make_dummy_trade("t3", gross_pnl_bps=50.0, net_pnl_bps=None, trade_win=None),  # incomplete funding
    ]
    metrics = scorecard(trades)

    assert metrics.total_trades == 3
    assert metrics.wins == 1
    assert metrics.losses == 1
    assert metrics.unknown == 1
    # Win rate is strictly over decided outcomes: 1 / (1 + 1) = 0.50
    assert metrics.win_rate == 0.50


def test_canonical_prediction_scorecard_denominators():
    """
    Section 22 test:
      4 correct, 3 incorrect, 2 flat, 1 unresolved.
      Total directional decisions = 10.
      Prediction directional N = 7.
      Directional Accuracy = 4/7 = 57.142857%.
      Horizon observation coverage = 9/10 = 90%.
      Directional resolution coverage = 7/10 = 70%.
      Flat rate = 2/9.
      Unresolved rate = 1/10.
      No mixed denominators.
    """
    from paper_trading.prediction import PredictionOutcome
    from paper_trading.scorecard import prediction_scorecard

    def _make_pred(pred_id: str, result: str) -> PredictionOutcome:
        return PredictionOutcome(
            prediction_id=pred_id,
            decision_id=f"dec_{pred_id}",
            cohort_id="c1",
            symbol="BTCUSDT",
            side="LONG",
            reference_price=100.0,
            decision_timestamp=1000,
            horizon_s=300,
            deadline_ms=301000,
            result=result,  # type: ignore[arg-type]
            reason="HORIZON_RESOLVED" if result != "UNRESOLVED" else "NO_TICK_WITHIN_TOLERANCE",
            resolution_price=101.0 if result in ("CORRECT", "INCORRECT", "FLAT") else None,
            raw_return_bps=100.0 if result == "CORRECT" else (-100.0 if result == "INCORRECT" else 0.0),
            directional_return_bps=100.0 if result == "CORRECT" else (-100.0 if result == "INCORRECT" else 0.0),
            resolved_timestamp_ms=301000 if result != "UNRESOLVED" else None,
            resolution_drift_ms=0 if result != "UNRESOLVED" else None,
            observed_at_ms=301000,
            flat_tolerance_bps=1.0,
            resolution_tolerance_ms=30000,
            policy_version="v1",
            created_at_ms=301000,
        )

    outcomes = (
        [_make_pred(f"c_{i}", "CORRECT") for i in range(4)]
        + [_make_pred(f"i_{i}", "INCORRECT") for i in range(3)]
        + [_make_pred(f"f_{i}", "FLAT") for i in range(2)]
        + [_make_pred("u_0", "UNRESOLVED")]
    )

    metrics = prediction_scorecard(outcomes, total_directional_decisions=10)

    assert metrics.total_directional_decisions == 10
    assert metrics.correct == 4
    assert metrics.incorrect == 3
    assert metrics.flat == 2
    assert metrics.unresolved == 1

    # Directional prediction N (FLAT excluded)
    assert metrics.prediction_directional_n == 7
    assert metrics.directional_accuracy == pytest.approx(4.0 / 7.0)
    assert metrics.directional_accuracy == pytest.approx(0.57142857)

    # Observed N (C + I + F)
    assert metrics.observed_n == 9
    assert metrics.horizon_observation_coverage == pytest.approx(9.0 / 10.0)
    assert metrics.horizon_observation_coverage == 0.90

    # Directional Resolution Coverage
    assert metrics.directional_resolution_coverage == pytest.approx(7.0 / 10.0)
    assert metrics.directional_resolution_coverage == 0.70

    # Flat Rate = 2 / 9
    assert metrics.flat_rate == pytest.approx(2.0 / 9.0)

    # Unresolved Rate = 1 / 10
    assert metrics.unresolved_rate == pytest.approx(1.0 / 10.0)
    assert metrics.unresolved_rate == 0.10
