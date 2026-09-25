# scripts/analytics/run_paper_scorecard.py
"""
CLI tool for running analytics and generating scorecard reports from paper trading ledgers.

Usage:
    python scripts/analytics/run_paper_scorecard.py --db dados/paper_trading.db
    python scripts/analytics/run_paper_scorecard.py --db :memory:
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from paper_trading.contracts import ClosedTrade
from paper_trading.ledger import PaperLedger
from paper_trading.scorecard import group_by, prediction_scorecard, scorecard


def format_percent(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def format_float(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Paper Trading Scorecard Reporter")
    parser.add_argument("--db", default="dados/paper_trading.db", help="Path to paper trading SQLite database")
    parser.add_argument("--cohort", default=None, help="Filter by cohort ID")
    parser.add_argument("--provider", default=None, help="Filter by decision_provider name")
    args = parser.parse_args()

    db_path = args.db
    if db_path != ":memory:" and not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        print("Paper Trading Scorecard: Empty (0 trades recorded).")
        return 0

    try:
        ledger = PaperLedger(db_path=db_path, read_only=True)
        all_trades = ledger.get_closed_trades(cohort_id=args.cohort)
        rejections = ledger.get_rejections(cohort_id=args.cohort)
        decisions = ledger.get_decisions(cohort_id=args.cohort)
        prediction_outcomes = ledger.get_prediction_outcomes(cohort_id=args.cohort)
        cohort_events = ledger.get_cohort_events(cohort_id=args.cohort)
        ledger.close()
    except Exception as e:
        print(f"Error reading ledger from {db_path}: {e}")
        return 0

    if args.provider:
        all_trades = [t for t in all_trades if t.decision_provider == args.provider]
        dec_ids_for_prov = {d.decision_id for d in decisions if d.decision_provider == args.provider}
        prediction_outcomes = [p for p in prediction_outcomes if p.decision_id in dec_ids_for_prov]
        decisions = [d for d in decisions if d.decision_provider == args.provider]

    print("=" * 75)
    print(" PAPER TRADING FOUNDATION — SCORECARD REPORT")
    print("=" * 75)
    print(f"Database:        {db_path}")
    print(f"Cohort Filter:   {args.cohort or 'ALL'}")
    print(f"Provider Filter: {args.provider or 'ALL'}")
    print("-" * 75)

    # Compute prediction scorecard independently of trade execution
    dir_decisions = [d for d in decisions if d.side in ("LONG", "SHORT")]
    pred_metrics = prediction_scorecard(
        prediction_outcomes=prediction_outcomes,
        total_directional_decisions=len(dir_decisions),
        cohort_events=cohort_events,
        cohort_id=args.cohort,
    )

    p_ci_low, p_ci_high = pred_metrics.directional_accuracy_ci95
    p_ci_str = f"[{p_ci_low*100:.1f}%, {p_ci_high*100:.1f}%]" if pred_metrics.directional_accuracy is not None else "N/A"

    if pred_metrics.prediction_directional_n > 0:
        pred_acc_str = (
            f"{pred_metrics.correct} / {pred_metrics.prediction_directional_n} "
            f"({format_percent(pred_metrics.directional_accuracy)}) [95% CI: {p_ci_str}]"
        )
    else:
        pred_acc_str = "N/A (0 resolved directional predictions)"

    print("CANONICAL PREDICTION OUTCOMES:")
    print(f"  Cohort Lifecycle:             {pred_metrics.cohort_lifecycle}")
    print(f"  Prediction Accounting:        {pred_metrics.prediction_accounting}")
    print(f"  Prediction Integrity:         {pred_metrics.prediction_integrity}")
    print(f"  Total Directional Decisions:  {pred_metrics.total_directional_decisions}")
    print(f"  Terminal Prediction Outcomes: {pred_metrics.terminal_prediction_outcomes}")
    print(f"  Missing Prediction Outcomes:  {pred_metrics.missing_prediction_outcomes}")
    print(f"  Correct:                      {pred_metrics.correct}")
    print(f"  Incorrect:                    {pred_metrics.incorrect}")
    print(f"  Flat:                         {pred_metrics.flat}")
    print(f"  Unresolved:                   {pred_metrics.unresolved}")
    print(f"  Interrupted/Pending:          {pred_metrics.missing_prediction_outcomes}")
    print(
        f"  Conservation:                 {pred_metrics.total_directional_decisions} = "
        f"{pred_metrics.correct} + {pred_metrics.incorrect} + {pred_metrics.flat} + "
        f"{pred_metrics.unresolved} + {pred_metrics.missing_prediction_outcomes} "
        f"({'PASS' if pred_metrics.conservation_passed else 'FAIL'})"
    )
    print(f"  Directional Prediction Acc:   {pred_acc_str}")
    print(f"  Prediction Directional N:     {pred_metrics.prediction_directional_n}")
    print(f"  Observed Outcomes (C/I/F):    {pred_metrics.correct} / {pred_metrics.incorrect} / {pred_metrics.flat} (Observed N: {pred_metrics.observed_n})")
    print(f"  Horizon Observation Coverage: {format_percent(pred_metrics.horizon_observation_coverage)} ({pred_metrics.observed_n}/{pred_metrics.total_directional_decisions})")
    print(f"  Terminal Outcome Coverage:    {format_percent(pred_metrics.terminal_outcome_coverage)} ({pred_metrics.terminal_prediction_outcomes}/{pred_metrics.total_directional_decisions})")
    print(f"  Prediction Accounting Cov:    {format_percent(pred_metrics.prediction_accounting_coverage)} ({pred_metrics.terminal_prediction_outcomes}/{pred_metrics.total_directional_decisions})")
    print(f"  Directional Resolution Cov:   {format_percent(pred_metrics.directional_resolution_coverage)} ({pred_metrics.prediction_directional_n}/{pred_metrics.total_directional_decisions})")
    print(f"  Flat Rate:                    {format_percent(pred_metrics.flat_rate)}")
    print(f"  Unresolved Rate:              {format_percent(pred_metrics.unresolved_rate)}")
    print(f"  Interrupted Rate:             {format_percent(pred_metrics.interrupted_rate)}")
    print("-" * 75)

    if not all_trades:
        print("No closed trades recorded for the selected criteria.")
        print(f"Rejections recorded: {len(rejections)}")
        print("=" * 75)
        return 0

    metrics = scorecard(all_trades)

    ci_low, ci_high = metrics.win_rate_ci95
    ci_str = f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]" if metrics.win_rate is not None else "N/A"

    d_low, d_high = metrics.directional_accuracy_ci95
    d_ci_str = f"[{d_low*100:.1f}%, {d_high*100:.1f}%]" if metrics.directional_accuracy is not None else "N/A"

    if metrics.trade_direction_decided_count > 0:
        dir_profit_str = (
            f"{metrics.trade_direction_profitable_count} / {metrics.trade_direction_decided_count} "
            f"({format_percent(metrics.trade_direction_profitability_rate)}) [95% CI: {d_ci_str}]"
        )
    else:
        dir_profit_str = "N/A (0 trades with resolved direction)"

    decided_outcomes = metrics.wins + metrics.losses
    decided_str = f"{metrics.wins} / {metrics.losses} / {metrics.flats} (Decided: {decided_outcomes}, Incomplete: {metrics.unknown})"

    print("ECONOMIC EXECUTION METRICS (CLOSED TRADES):")
    print(f"  Total Trades:                 {metrics.total_trades}")
    print(f"  Trade Direction Profitability: {dir_profit_str}")
    print(f"  Wins / Loss / Flat:           {decided_str}")
    print(f"  Net Win Rate (Post-Costs):    {format_percent(metrics.win_rate)} (95% CI Wilson: {ci_str})")
    print(f"  Break-Even Win Rate:          {format_percent(metrics.breakeven_win_rate)}")
    print(f"  Expectancy:                   {format_float(metrics.expectancy_bps, 2)} bps | {format_float(metrics.expectancy_R, 2)} R")
    print(f"  Profit Factor:                {format_float(metrics.profit_factor, 2)}")
    print(f"  Avg Payoff:                   {format_float(metrics.avg_payoff, 2)}")
    print(f"  Max Drawdown:                 {format_float(metrics.max_drawdown_bps, 2)} bps")
    print(f"  Mean MAE / MFE:               {format_float(metrics.mae_bps_mean, 2)} bps / {format_float(metrics.mfe_bps_mean, 2)} bps")
    print(f"  Costs Complete Ratio:         {format_percent(metrics.costs_complete_ratio)}")
    print("-" * 75)

    # Breakdown by Provider
    by_provider = group_by(all_trades, "decision_provider")
    print("BREAKDOWN BY DECISION PROVIDER:")
    print(f"{'Provider':<20} | {'Trades':<7} | {'Gross Dir%':<10} | {'Net WR':<10} | {'Exp (bps)':<10} | {'Max DD':<10}")
    print("-" * 75)
    for prov, p_trades in sorted(by_provider.items()):
        p_met = scorecard(p_trades)
        d_acc_s = format_percent(p_met.trade_direction_profitability_rate)
        wr_s = format_percent(p_met.win_rate)
        exp_s = format_float(p_met.expectancy_bps, 1)
        dd_s = format_float(p_met.max_drawdown_bps, 1)
        print(f"{prov:<20} | {p_met.total_trades:<7} | {d_acc_s:<10} | {wr_s:<10} | {exp_s:<10} | {dd_s:<10}")

    print("-" * 75)
    # Breakdown by Exit Reason
    by_exit = group_by(all_trades, "exit_reason")
    print("BREAKDOWN BY EXIT REASON:")
    print(f"{'Exit Reason':<20} | {'Trades':<7} | {'Net WR':<10} | {'Exp (bps)':<10}")
    print("-" * 75)
    for reason, r_trades in sorted(by_exit.items()):
        r_met = scorecard(r_trades)
        wr_s = format_percent(r_met.win_rate)
        exp_s = format_float(r_met.expectancy_bps, 1)
        print(f"{reason:<20} | {r_met.total_trades:<7} | {wr_s:<10} | {exp_s:<10}")

    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
