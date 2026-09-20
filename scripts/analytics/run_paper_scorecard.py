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
from paper_trading.scorecard import group_by, scorecard


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
        ledger = PaperLedger(db_path=db_path)
        all_trades = ledger.get_closed_trades(cohort_id=args.cohort)
        rejections = ledger.get_rejections(cohort_id=args.cohort)
        ledger.close()
    except Exception as e:
        print(f"Error reading ledger from {db_path}: {e}")
        return 0

    if args.provider:
        all_trades = [t for t in all_trades if t.decision_provider == args.provider]

    print("=" * 75)
    print(" PAPER TRADING FOUNDATION — SCORECARD REPORT")
    print("=" * 75)
    print(f"Database:        {db_path}")
    print(f"Cohort Filter:   {args.cohort or 'ALL'}")
    print(f"Provider Filter: {args.provider or 'ALL'}")
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

    print("OVERALL METRICS:")
    print(f"  Total Trades:         {metrics.total_trades}")
    print(f"  Direction Correct:    {metrics.direction_correct_count} / {metrics.direction_correct_count + metrics.direction_incorrect_count} ({format_percent(metrics.directional_accuracy)}) [95% CI: {d_ci_str}]")
    print(f"  Wins / Loss / Flat:   {metrics.wins} / {metrics.losses} / {metrics.flats} (Unknown: {metrics.unknown})")
    print(f"  Net Win Rate:         {format_percent(metrics.win_rate)} (95% CI Wilson: {ci_str})")
    print(f"  Break-Even Win Rate:  {format_percent(metrics.breakeven_win_rate)}")
    print(f"  Expectancy:           {format_float(metrics.expectancy_bps, 2)} bps | {format_float(metrics.expectancy_R, 2)} R")
    print(f"  Profit Factor:        {format_float(metrics.profit_factor, 2)}")
    print(f"  Avg Payoff:           {format_float(metrics.avg_payoff, 2)}")
    print(f"  Max Drawdown:         {format_float(metrics.max_drawdown_bps, 2)} bps")
    print(f"  Mean MAE / MFE:       {format_float(metrics.mae_bps_mean, 2)} bps / {format_float(metrics.mfe_bps_mean, 2)} bps")
    print(f"  Costs Complete Ratio: {format_percent(metrics.costs_complete_ratio)}")
    print("-" * 75)

    # Breakdown by Provider
    by_provider = group_by(all_trades, "decision_provider")
    print("BREAKDOWN BY DECISION PROVIDER:")
    print(f"{'Provider':<20} | {'Trades':<7} | {'Dir Acc':<10} | {'Net WR':<10} | {'Exp (bps)':<10} | {'Max DD':<10}")
    print("-" * 75)
    for prov, p_trades in sorted(by_provider.items()):
        p_met = scorecard(p_trades)
        d_acc_s = format_percent(p_met.directional_accuracy)
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
