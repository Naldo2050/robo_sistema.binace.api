# paper_trading/__init__.py
"""
Hermetic paper trading foundation package.

Decoupled simulation components:
- contracts: CanonicalDecision, PaperOrder, PaperFill, PaperPosition, ClosedTrade, Rejection, PaperCostConfig
- cost_model: CostModel, PaperCostConfig, count_funding_crossings, PnLBreakdown
- executor: PaperExecutor, ExecutorConfig, TickEvents
- positions: PositionManager, PositionConfig
- ledger: PaperLedger
- tape: TapeReader, SyntheticTape, TapeTick
- replay: ReplayRunner, ReplayResult
- scorecard: scorecard, wilson_interval, breakeven_win_rate_payoff, breakeven_win_rate_rr, group_by, calibration_table, attrition_report
"""

from paper_trading.contracts import (
    CanonicalDecision,
    ClosedTrade,
    InvalidDecisionError,
    PaperCostConfig,
    PaperFill,
    PaperOrder,
    PaperPosition,
    PaperTradingError,
    Rejection,
)
from paper_trading.cost_model import (
    CostModel,
    DEFAULT_PAPER_COST_CONFIG,
    PnLBreakdown,
    count_funding_crossings,
)
from paper_trading.executor import ExecutorConfig, PaperExecutor, TickEvents
from paper_trading.ledger import PaperLedger
from paper_trading.positions import PositionConfig, PositionManager
from paper_trading.replay import ReplayResult, ReplayRunner
from paper_trading.scorecard import (
    AttritionReport,
    CalibrationBin,
    CalibrationReport,
    ScorecardMetrics,
    attrition_report,
    breakeven_win_rate_payoff,
    breakeven_win_rate_rr,
    calibration_table,
    group_by,
    scorecard,
    wilson_interval,
)
from paper_trading.tape import SyntheticTape, TapeReader, TapeTick

# Alias for backwards compatibility
CostConfig = PaperCostConfig

__all__ = [
    "CanonicalDecision",
    "PaperOrder",
    "PaperFill",
    "PaperPosition",
    "ClosedTrade",
    "Rejection",
    "PaperTradingError",
    "InvalidDecisionError",
    "PaperCostConfig",
    "CostConfig",
    "CostModel",
    "DEFAULT_PAPER_COST_CONFIG",
    "PnLBreakdown",
    "count_funding_crossings",
    "PaperExecutor",
    "ExecutorConfig",
    "TickEvents",
    "PositionManager",
    "PositionConfig",
    "PaperLedger",
    "TapeReader",
    "SyntheticTape",
    "TapeTick",
    "ReplayRunner",
    "ReplayResult",
    "scorecard",
    "wilson_interval",
    "breakeven_win_rate_payoff",
    "breakeven_win_rate_rr",
    "group_by",
    "calibration_table",
    "attrition_report",
    "ScorecardMetrics",
    "CalibrationBin",
    "CalibrationReport",
    "AttritionReport",
]
