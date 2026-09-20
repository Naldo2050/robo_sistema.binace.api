# paper_trading/scorecard.py
"""
Analytics and performance scorecard for paper trading evaluation.

Calculates Wilson confidence intervals, net payoff break-even win rates,
trade profitability vs prediction accuracy, risk-adjusted expectancy,
MAE/MFE statistics, calibration tables, and attrition funnels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from paper_trading.contracts import CanonicalDecision, ClosedTrade, Rejection
from paper_trading.prediction import PredictionOutcome


@dataclass(frozen=True)
class ScorecardMetrics:
    """Consolidated performance metrics for evaluated trades."""

    total_trades: int
    wins: int
    losses: int
    flats: int
    unknown: int
    trade_direction_profitable_count: int
    trade_direction_unprofitable_count: int
    trade_direction_decided_count: int
    trade_direction_profitability_rate: Optional[float]
    prediction_direction_correct_count: int
    prediction_direction_decided_count: int
    prediction_accuracy: Optional[float]
    win_rate: Optional[float]
    win_rate_ci95: Tuple[float, float]
    expectancy_bps: Optional[float]
    expectancy_R: Optional[float]
    profit_factor: Optional[float]
    avg_payoff: Optional[float]
    breakeven_win_rate: Optional[float]
    max_drawdown_bps: float
    mae_bps_mean: float
    mfe_bps_mean: float
    costs_share: Optional[float]
    costs_complete_ratio: float
    directional_accuracy: Optional[float] = None
    directional_accuracy_ci95: Tuple[float, float] = (0.0, 0.0)

    @property
    def direction_correct_count(self) -> int:
        """Alias for trade_direction_profitable_count."""
        return self.trade_direction_profitable_count

    @property
    def direction_incorrect_count(self) -> int:
        """Count of trades where directional price movement was not profitable (gross PnL <= 0)."""
        return self.trade_direction_unprofitable_count


@dataclass(frozen=True)
class CalibrationBin:
    """Calibration bracket statistics."""

    bin_range: Tuple[float, float]
    total_in_bin: int
    wins_in_bin: int
    win_rate: Optional[float]


@dataclass(frozen=True)
class CalibrationReport:
    """Full confidence calibration analysis isolating uncalibrated / None values."""

    bins: List[CalibrationBin]
    none_count: int
    fraction_none: float


@dataclass(frozen=True)
class AttritionReport:
    """Lifecycle transition funnel from decisions to closed outcomes."""

    total_decisions: int
    rejections_count: int
    rejections_by_reason: Dict[str, int]
    closed_trades_count: int
    unknown_trades_count: int
    balance_verified: bool


@dataclass(frozen=True)
class PredictionScorecardMetrics:
    """
    Canonical directional prediction accuracy & coverage metrics (Gate D0-B).

    Completely isolated from economic execution, fees, slippage, and position limits.
    """

    total_directional_decisions: int
    correct: int
    incorrect: int
    flat: int
    unresolved: int

    # Denominators
    prediction_directional_n: int  # correct + incorrect (FLAT excluded)
    observed_n: int  # correct + incorrect + flat

    # Rates & Intervals
    directional_accuracy: Optional[float]  # correct / (correct + incorrect)
    directional_accuracy_ci95: Tuple[float, float]  # Wilson CI over (correct, correct+incorrect)
    horizon_observation_coverage: Optional[float]  # (correct + incorrect + flat) / total_directional_decisions
    directional_resolution_coverage: Optional[float]  # (correct + incorrect) / total_directional_decisions
    flat_rate: Optional[float]  # flat / (correct + incorrect + flat)
    unresolved_rate: Optional[float]  # unresolved / total_directional_decisions


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Compute Wilson score interval for binomial proportion with asymptotic bounds.

    Returns (ci_lower, ci_upper) in [0.0, 1.0].
    """
    if n <= 0:
        return 0.0, 0.0

    p_hat = wins / n
    z2 = z * z
    denominator = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def breakeven_win_rate_payoff(
    net_wins_bps: Sequence[float],
    net_losses_bps: Sequence[float],
) -> Optional[float]:
    """
    Item 6: Compute break-even win rate strictly over net PnL outcomes (without double-counting costs):
        W_net = mean(net_pnl_bps > 0)
        L_net = abs(mean(net_pnl_bps < 0))
        p_be  = L_net / (W_net + L_net)
    Returns None if data is incomplete or has no wins / no losses.
    """
    positive = [w for w in net_wins_bps if w > 0]
    negative = [abs(l) for l in net_losses_bps if l < 0]

    if not positive or not negative:
        return None

    w_net = sum(positive) / len(positive)
    l_net = sum(negative) / len(negative)

    denominator = w_net + l_net
    if denominator <= 0:
        return None

    p_be = l_net / denominator
    return min(1.0, max(0.0, p_be))


def breakeven_win_rate_gross_plus_costs(
    average_win_gross: float,
    average_loss_gross: float,
    average_cost: float = 0.0,
) -> Optional[float]:
    """
    Alternative separate break-even formula based on gross movement + costs:
        p_be = (average_loss_gross + average_cost) / (average_win_gross + average_loss_gross)
    """
    denominator = average_win_gross + average_loss_gross
    if denominator <= 0:
        return None
    p_be = (average_loss_gross + average_cost) / denominator
    return min(1.0, max(0.0, p_be))


def breakeven_win_rate_rr(rr: float, cost_r: float = 0.0) -> float:
    """
    Compute cost-adjusted breakeven win rate from risk-reward ratio:
        p = (1 + cost_r) / (rr + 1)
    """
    if rr <= 0:
        return 1.0
    return (1.0 + cost_r) / (rr + 1.0)


def scorecard(
    trades: Sequence[ClosedTrade],
) -> ScorecardMetrics:
    """
    Calculate full evaluation scorecard over closed trades.

    Separates trade direction profitability from prediction accuracy (Item 5),
    and computes break-even win rate on net PnL payoff (Item 6).
    """
    total = len(trades)
    if total == 0:
        return ScorecardMetrics(
            total_trades=0,
            wins=0,
            losses=0,
            flats=0,
            unknown=0,
            trade_direction_profitable_count=0,
            trade_direction_unprofitable_count=0,
            trade_direction_decided_count=0,
            trade_direction_profitability_rate=None,
            prediction_direction_correct_count=0,
            prediction_direction_decided_count=0,
            prediction_accuracy=None,
            win_rate=None,
            win_rate_ci95=(0.0, 0.0),
            expectancy_bps=None,
            expectancy_R=None,
            profit_factor=None,
            avg_payoff=None,
            breakeven_win_rate=None,
            max_drawdown_bps=0.0,
            mae_bps_mean=0.0,
            mfe_bps_mean=0.0,
            costs_share=None,
            costs_complete_ratio=0.0,
            directional_accuracy=None,
            directional_accuracy_ci95=(0.0, 0.0),
        )

    wins = 0
    losses = 0
    flats = 0
    unknown = 0

    trade_dir_profitable_cnt = 0
    trade_dir_decided_cnt = 0

    pred_dir_correct_cnt = 0
    pred_dir_decided_cnt = 0

    pnl_bps_list: List[float] = []
    pnl_R_list: List[float] = []
    win_pnls: List[float] = []
    loss_pnls: List[float] = []
    win_bps_list: List[float] = []
    loss_bps_list: List[float] = []
    mae_list: List[float] = []
    mfe_list: List[float] = []

    total_gross = 0.0
    total_costs = 0.0
    costs_complete_count = 0

    for tr in trades:
        mae_list.append(tr.mae_bps)
        mfe_list.append(tr.mfe_bps)

        if tr.costs_complete:
            costs_complete_count += 1

        # 1. Trade direction profitability (from gross PnL)
        if tr.trade_direction_profitable is not None:
            trade_dir_decided_cnt += 1
            if tr.trade_direction_profitable is True:
                trade_dir_profitable_cnt += 1

        # 2. Prediction accuracy (from prediction_direction_correct)
        if tr.prediction_direction_correct is not None:
            pred_dir_decided_cnt += 1
            if tr.prediction_direction_correct is True:
                pred_dir_correct_cnt += 1

        if tr.net_pnl_usdt is None or tr.trade_win is None or tr.net_pnl_bps is None:
            unknown += 1
            continue

        pnl = tr.net_pnl_usdt
        pnl_bps = tr.net_pnl_bps

        if tr.gross_pnl_usdt is not None:
            total_gross += abs(tr.gross_pnl_usdt)
        trade_cost = tr.fees_usdt + tr.slippage_usdt + (abs(tr.funding_usdt) if tr.funding_usdt else 0.0)
        total_costs += trade_cost

        if pnl > 0:
            wins += 1
            win_pnls.append(pnl)
            win_bps_list.append(pnl_bps)
        elif pnl < 0:
            losses += 1
            loss_pnls.append(pnl)
            loss_bps_list.append(pnl_bps)
        else:
            flats += 1

        pnl_bps_list.append(pnl_bps)

        if tr.pnl_R is not None:
            pnl_R_list.append(tr.pnl_R)

    # Rates
    trade_dir_rate = (trade_dir_profitable_cnt / trade_dir_decided_cnt) if trade_dir_decided_cnt > 0 else None
    pred_acc = (pred_dir_correct_cnt / pred_dir_decided_cnt) if pred_dir_decided_cnt > 0 else None

    # Net win rate (post-costs)
    decided_outcomes = wins + losses
    if decided_outcomes > 0:
        win_rate = wins / decided_outcomes
        ci95 = wilson_interval(wins, decided_outcomes, z=1.96)
    else:
        win_rate = None
        ci95 = (0.0, 0.0)

    dir_ci = wilson_interval(trade_dir_profitable_cnt, trade_dir_decided_cnt, z=1.96) if trade_dir_decided_cnt > 0 else (0.0, 0.0)

    # Expectancy
    expectancy_bps = (sum(pnl_bps_list) / len(pnl_bps_list)) if pnl_bps_list else None
    expectancy_R = (sum(pnl_R_list) / len(pnl_R_list)) if pnl_R_list else None

    # Profit Factor & Payoff
    sum_win = sum(win_pnls)
    abs_loss = abs(sum(loss_pnls))
    if abs_loss > 0:
        profit_factor = sum_win / abs_loss
    elif sum_win > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_win = (sum_win / len(win_pnls)) if win_pnls else 0.0
    avg_loss = (abs_loss / len(loss_pnls)) if loss_pnls else 0.0
    if avg_loss > 0:
        avg_payoff = avg_win / avg_loss
    elif avg_win > 0:
        avg_payoff = float("inf")
    else:
        avg_payoff = 0.0

    # Break-even win rate via net payoff (Item 6)
    p_be = breakeven_win_rate_payoff(net_wins_bps=win_bps_list, net_losses_bps=loss_bps_list)

    # Max Drawdown in bps
    cumulative_bps = 0.0
    peak_bps = 0.0
    max_dd_bps = 0.0
    for bps in pnl_bps_list:
        cumulative_bps += bps
        if cumulative_bps > peak_bps:
            peak_bps = cumulative_bps
        dd = peak_bps - cumulative_bps
        if dd > max_dd_bps:
            max_dd_bps = dd

    mae_mean = sum(mae_list) / len(mae_list) if mae_list else 0.0
    mfe_mean = sum(mfe_list) / len(mfe_list) if mfe_list else 0.0
    costs_share = (total_costs / total_gross) if total_gross > 0 else None
    costs_complete_ratio = (costs_complete_count / total) if total > 0 else 0.0

    return ScorecardMetrics(
        total_trades=total,
        wins=wins,
        losses=losses,
        flats=flats,
        unknown=unknown,
        trade_direction_profitable_count=trade_dir_profitable_cnt,
        trade_direction_unprofitable_count=trade_dir_decided_cnt - trade_dir_profitable_cnt,
        trade_direction_decided_count=trade_dir_decided_cnt,
        trade_direction_profitability_rate=trade_dir_rate,
        prediction_direction_correct_count=pred_dir_correct_cnt,
        prediction_direction_decided_count=pred_dir_decided_cnt,
        prediction_accuracy=pred_acc,
        win_rate=win_rate,
        win_rate_ci95=ci95,
        expectancy_bps=expectancy_bps,
        expectancy_R=expectancy_R,
        profit_factor=profit_factor,
        avg_payoff=avg_payoff,
        breakeven_win_rate=p_be,
        max_drawdown_bps=max_dd_bps,
        mae_bps_mean=mae_mean,
        mfe_bps_mean=mfe_mean,
        costs_share=costs_share,
        costs_complete_ratio=costs_complete_ratio,
        directional_accuracy=trade_dir_rate,
        directional_accuracy_ci95=dir_ci,
    )


def group_by(trades: Sequence[ClosedTrade], key: str) -> Dict[str, List[ClosedTrade]]:
    """Group closed trades by category."""
    grouped: Dict[str, List[ClosedTrade]] = {}
    for tr in trades:
        val: Optional[str] = None
        if key == "decision_provider":
            val = tr.decision_provider
        elif key == "exit_reason":
            val = tr.exit_reason
        elif key == "side":
            val = tr.side
        elif key in tr.context:
            val = str(tr.context[key])
        else:
            val = "UNKNOWN"

        group_key = val or "UNKNOWN"
        grouped.setdefault(group_key, []).append(tr)
    return grouped


def calibration_table(
    trades: Sequence[ClosedTrade],
    bins: Sequence[Tuple[float, float]] = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)),
) -> CalibrationReport:
    """
    Analyze accuracy per confidence bin, strictly isolating None / uncalibrated values.
    """
    total = len(trades)
    none_count = 0
    bin_trades: Dict[Tuple[float, float], List[ClosedTrade]] = {b: [] for b in bins}

    for tr in trades:
        conf = tr.context.get("confidence")
        if conf is None:
            none_count += 1
            continue

        try:
            c_val = float(conf)
            if not math.isfinite(c_val):
                none_count += 1
                continue
        except (ValueError, TypeError):
            none_count += 1
            continue

        placed = False
        for b_low, b_high in bins:
            if b_low <= c_val < b_high or (b_high == 1.0 and c_val == 1.0):
                bin_trades[(b_low, b_high)].append(tr)
                placed = True
                break
        if not placed:
            none_count += 1

    calibration_bins: List[CalibrationBin] = []
    for b_range in bins:
        b_list = bin_trades[b_range]
        b_wins = sum(1 for t in b_list if t.trade_win is True)
        b_decided = sum(1 for t in b_list if t.trade_win is not None)
        b_wr = (b_wins / b_decided) if b_decided > 0 else None
        calibration_bins.append(
            CalibrationBin(
                bin_range=b_range,
                total_in_bin=len(b_list),
                wins_in_bin=b_wins,
                win_rate=b_wr,
            )
        )

    fraction_none = (none_count / total) if total > 0 else 0.0
    return CalibrationReport(
        bins=calibration_bins,
        none_count=none_count,
        fraction_none=fraction_none,
    )


def attrition_report(
    decisions: Sequence[CanonicalDecision],
    rejections: Sequence[Rejection],
    trades: Sequence[ClosedTrade],
) -> AttritionReport:
    """Audit attrition from decisions through rejections, fills, and trade terminations."""
    total_decisions = len(decisions)
    rejections_count = len(rejections)
    closed_trades_count = len(trades)
    unknown_trades_count = sum(1 for t in trades if t.exit_reason == "UNKNOWN_DATA_GAP")

    by_reason: Dict[str, int] = {}
    for r in rejections:
        by_reason[r.reason] = by_reason.get(r.reason, 0) + 1

    balance_verified = (total_decisions == (rejections_count + closed_trades_count))

    return AttritionReport(
        total_decisions=total_decisions,
        rejections_count=rejections_count,
        rejections_by_reason=by_reason,
        closed_trades_count=closed_trades_count,
        unknown_trades_count=unknown_trades_count,
        balance_verified=balance_verified,
    )


def prediction_scorecard(
    prediction_outcomes: Sequence[PredictionOutcome],
    total_directional_decisions: Optional[int] = None,
) -> PredictionScorecardMetrics:
    """
    Compute canonical prediction performance over recorded PredictionOutcome instances.

    Evaluates:
      - Directional Prediction Accuracy = correct / (correct + incorrect), FLAT excluded.
      - Wilson CI 95% over (correct vs incorrect).
      - Horizon Observation Coverage = (correct + incorrect + flat) / total_directional_decisions.
      - Directional Resolution Coverage = (correct + incorrect) / total_directional_decisions.
      - Flat Rate = flat / (correct + incorrect + flat).
      - Unresolved Rate = unresolved / total_directional_decisions.
    """
    correct = sum(1 for p in prediction_outcomes if p.result == "CORRECT")
    incorrect = sum(1 for p in prediction_outcomes if p.result == "INCORRECT")
    flat = sum(1 for p in prediction_outcomes if p.result == "FLAT")
    unresolved = sum(1 for p in prediction_outcomes if p.result == "UNRESOLVED")

    if total_directional_decisions is not None:
        total_dec = total_directional_decisions
    else:
        total_dec = len(prediction_outcomes)

    pred_n = correct + incorrect
    obs_n = correct + incorrect + flat

    if pred_n > 0:
        dir_acc = correct / pred_n
        ci95 = wilson_interval(correct, pred_n, z=1.96)
    else:
        dir_acc = None
        ci95 = (0.0, 0.0)

    if obs_n > 0:
        flat_rate = flat / obs_n
    else:
        flat_rate = None

    if total_dec > 0:
        obs_cov = obs_n / total_dec
        res_cov = pred_n / total_dec
        unres_rate = unresolved / total_dec
    else:
        obs_cov = None
        res_cov = None
        unres_rate = None

    return PredictionScorecardMetrics(
        total_directional_decisions=total_dec,
        correct=correct,
        incorrect=incorrect,
        flat=flat,
        unresolved=unresolved,
        prediction_directional_n=pred_n,
        observed_n=obs_n,
        directional_accuracy=dir_acc,
        directional_accuracy_ci95=ci95,
        horizon_observation_coverage=obs_cov,
        directional_resolution_coverage=res_cov,
        flat_rate=flat_rate,
        unresolved_rate=unres_rate,
    )
