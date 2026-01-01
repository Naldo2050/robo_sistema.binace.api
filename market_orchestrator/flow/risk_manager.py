# market_orchestrator/flow/risk_manager.py
from typing import Dict, Any, Optional
from datetime import datetime


class RiskManager:
    """Risk management component for the flow module."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.risk_limits = {
            'max_position_size': 100000,
            'max_daily_loss': 0.05,
            'max_open_positions': 10
        }
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.risk_violations = []

    def check_risk_limits(self, trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if trade request violates risk limits."""
        position_size = trade_request.get('size', 0) * trade_request.get('price', 0)
        
        # Check position size limit
        if position_size > self.risk_limits['max_position_size']:
            return {
                'approved': False,
                'reason': 'Position size limit exceeded',
                'max_allowed': self.risk_limits['max_position_size'],
                'requested': position_size
            }

        # Check number of open positions
        if len(self.current_positions) >= self.risk_limits['max_open_positions']:
            return {
                'approved': False,
                'reason': 'Maximum open positions limit exceeded',
                'max_allowed': self.risk_limits['max_open_positions'],
                'current': len(self.current_positions)
            }

        # Check daily loss limit
        if self.daily_pnl < 0 and abs(self.daily_pnl) > self.risk_limits['max_daily_loss']:
            return {
                'approved': False,
                'reason': 'Daily loss limit exceeded',
                'max_allowed': self.risk_limits['max_daily_loss'],
                'current': self.daily_pnl
            }

        return {'approved': True, 'reason': 'Risk checks passed'}

    def update_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """Update position data."""
        self.current_positions[symbol] = position_data

    def remove_position(self, symbol: str) -> None:
        """Remove a position."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]

    def update_daily_pnl(self, pnl_change: float) -> None:
        """Update daily P&L."""
        self.daily_pnl += pnl_change

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            'symbol': self.symbol,
            'current_positions': len(self.current_positions),
            'daily_pnl': self.daily_pnl,
            'risk_limits': self.risk_limits.copy(),
            'recent_violations': self.risk_violations[-5:],  # Last 5 violations
            'risk_utilization': {
                'position_count_pct': (len(self.current_positions) / self.risk_limits['max_open_positions']) * 100,
                'daily_loss_pct': (abs(self.daily_pnl) / self.risk_limits['max_daily_loss']) * 100 if self.daily_pnl < 0 else 0
            }
        }

    def add_risk_violation(self, violation: Dict[str, Any]) -> None:
        """Record a risk violation."""
        violation['timestamp'] = datetime.now().isoformat()
        self.risk_violations.append(violation)

    def reset_daily_metrics(self) -> None:
        """Reset daily risk metrics."""
        self.daily_pnl = 0.0
        self.risk_violations.clear()