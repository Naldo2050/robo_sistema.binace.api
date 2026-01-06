# risk_management/risk_manager.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from .exceptions import RiskLimitExceeded, PositionLimitError, DailyLossLimitError


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    max_position_size: float = 100000
    max_daily_loss: float = 0.05  # 5%
    max_loss_per_trade: float = 0.02  # 2%
    max_open_positions: int = 10
    max_correlation: float = 0.8
    var_confidence_level: float = 0.95
    enable_stress_testing: bool = True


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    timestamp: datetime = None
    notional_value: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.notional_value == 0.0:
            self.notional_value = self.size * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == 'BUY':
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * self.size)


@dataclass
class TradeRequest:
    """Represents a trade request."""
    symbol: str
    side: str
    size: float
    price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy: str = "momentum"
    confidence: float = 0.0


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio."""
    unrealized_pnl: float = 0.0
    pnl_percentage: float = 0.0
    distance_to_stop: float = 0.0
    distance_to_take: float = 0.0
    risk_reward_ratio: float = 0.0
    var: float = 0.0


class RiskManager:
    """Risk management system for trading operations."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0

    @property
    def max_position_size(self) -> float:
        return self.config.max_position_size

    @property
    def max_daily_loss(self) -> float:
        return self.config.max_daily_loss

    @property
    def max_loss_per_trade(self) -> float:
        return self.config.max_loss_per_trade

    @property
    def max_open_positions(self) -> int:
        return self.config.max_open_positions

    @property
    def max_correlation(self) -> float:
        return self.config.max_correlation

    @property
    def var_confidence_level(self) -> float:
        return self.config.var_confidence_level

    @property
    def enable_stress_testing(self) -> bool:
        return self.config.enable_stress_testing

    def add_position(self, position: Position) -> None:
        """Add a new position."""
        if position.symbol in self.positions:
            raise ValueError(f"Position for {position.symbol} already exists")
        self.positions[position.symbol] = position

    def update_position(self, symbol: str, current_price: float) -> None:
        """Update position with current market price."""
        if symbol not in self.positions:
            raise KeyError(f"Position for {symbol} not found")
        self.positions[symbol].current_price = current_price

    def remove_position(self, symbol: str, exit_price: float) -> float:
        """Remove position and return realized P&L."""
        if symbol not in self.positions:
            raise KeyError(f"Position for {symbol} not found")
        
        position = self.positions[symbol]
        old_price = position.current_price
        position.current_price = exit_price
        
        realized_pnl = position.unrealized_pnl
        self.daily_pnl += realized_pnl
        
        del self.positions[symbol]
        return realized_pnl

    def check_trade_request(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check if trade request is approved based on risk rules."""
        # Check position limit
        if len(self.positions) >= self.max_open_positions:
            return {
                'approved': False,
                'reason': 'position limit exceeded',
                'max_size': 0
            }

        # Check position size limit
        position_value = trade_request.size * trade_request.price
        if position_value > self.max_position_size:
            return {
                'approved': False,
                'reason': 'position size limit exceeded',
                'max_size': self.max_position_size / trade_request.price
            }

        # Check daily loss limit
        if self.daily_pnl < 0 and abs(self.daily_pnl) > self.max_daily_loss:
            return {
                'approved': False,
                'reason': 'daily loss limit exceeded',
                'max_size': 0
            }

        # Check per-trade loss limit
        if trade_request.stop_loss > 0:
            potential_loss = abs(trade_request.price - trade_request.stop_loss) * trade_request.size
            loss_percentage = potential_loss / position_value
            if loss_percentage > self.max_loss_per_trade:
                return {
                    'approved': False,
                    'reason': 'per trade loss limit exceeded',
                    'max_size': 0
                }

        return {
            'approved': True,
            'reason': 'approved',
            'max_size': self.max_position_size / trade_request.price
        }

    def calculate_position_risk(self, symbol: str) -> RiskMetrics:
        """Calculate risk metrics for a position."""
        if symbol not in self.positions:
            raise KeyError(f"Position for {symbol} not found")
        
        position = self.positions[symbol]
        
        # Basic metrics
        unrealized_pnl = position.unrealized_pnl
        pnl_percentage = position.pnl_percentage
        
        # Distance metrics
        distance_to_stop = 0.0
        distance_to_take = 0.0
        
        if position.stop_loss > 0:
            if position.side == 'BUY':
                distance_to_stop = position.current_price - position.stop_loss
            else:
                distance_to_stop = position.stop_loss - position.current_price
        
        if position.take_profit > 0:
            if position.side == 'BUY':
                distance_to_take = position.take_profit - position.current_price
            else:
                distance_to_take = position.current_price - position.take_profit
        
        # Risk-reward ratio
        risk_reward_ratio = 0.0
        if distance_to_stop > 0 and distance_to_take > 0:
            risk_reward_ratio = distance_to_take / distance_to_stop
        
        # Simple VaR calculation (placeholder)
        var = abs(unrealized_pnl) * 0.1  # Simplified VaR

        return RiskMetrics(
            unrealized_pnl=unrealized_pnl,
            pnl_percentage=pnl_percentage,
            distance_to_stop=distance_to_stop,
            distance_to_take=distance_to_take,
            risk_reward_ratio=risk_reward_ratio,
            var=var
        )

    def calculate_portfolio_risk(self) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        if not self.positions:
            return {
                'total_exposure': 0.0,
                'total_unrealized_pnl': 0.0,
                'portfolio_var': 0.0,
                'expected_shortfall': 0.0,
                'concentration_risk': 0.0,
                'correlation_matrix': {}
            }

        total_exposure = sum(pos.notional_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        # Simplified VaR calculation
        portfolio_var = abs(total_unrealized_pnl) * 0.15

        # Expected Shortfall (simplified)
        expected_shortfall = portfolio_var * 1.3

        # Concentration risk (Herfindahl Index)
        if total_exposure > 0:
            weights = [pos.notional_value / total_exposure for pos in self.positions.values()]
            concentration_risk = sum(w**2 for w in weights)
        else:
            concentration_risk = 0.0

        # Correlation matrix (placeholder - would need historical data)
        correlation_matrix = {}

        return {
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_unrealized_pnl,
            'portfolio_var': portfolio_var,
            'expected_shortfall': expected_shortfall,
            'concentration_risk': concentration_risk,
            'correlation_matrix': correlation_matrix
        }

    def run_stress_tests(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Run stress tests on the portfolio."""
        if not self.enable_stress_testing:
            return {}

        results = {}
        base_portfolio_value = sum(pos.notional_value for pos in self.positions.values())

        for scenario_name, price_changes in scenarios.items():
            portfolio_pnl = 0.0
            max_drawdown = 0.0
            liquidity_impact = 0.0

            for symbol, position in self.positions.items():
                if symbol in price_changes:
                    price_change = price_changes[symbol]
                    position_pnl = position.notional_value * price_change
                    portfolio_pnl += position_pnl

                    # Calculate drawdown impact
                    drawdown = abs(position_pnl) / base_portfolio_value if base_portfolio_value > 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)

            results[scenario_name] = {
                'portfolio_pnl': portfolio_pnl,
                'max_drawdown': max_drawdown,
                'liquidity_impact': liquidity_impact
            }

        return results

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)

    def analyze_correlations(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze correlations between assets."""
        return price_data.corr()

    def assess_liquidity_risk(self, orderbook_data: Dict[str, Any], position_size: float) -> Dict[str, Any]:
        """Assess liquidity risk for a given position size."""
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        if not bids or not asks:
            return {
                'slippage_estimate': 0.0,
                'market_impact': 0.0,
                'liquidation_time': 0.0,
                'spread_ratio': 0.0
            }

        # Calculate slippage estimate (simplified)
        best_bid = bids[0]['price'] if bids else 0
        best_ask = asks[0]['price'] if asks else 0
        
        if best_bid == 0 or best_ask == 0:
            slippage_estimate = 0.0
        else:
            spread = best_ask - best_bid
            slippage_estimate = (spread / ((best_bid + best_ask) / 2)) * position_size

        # Market impact estimate (simplified)
        market_impact = slippage_estimate * 1.5

        # Liquidation time estimate (placeholder)
        liquidation_time = 60.0  # seconds

        # Spread ratio
        spread_ratio = spread / ((best_bid + best_ask) / 2) if (best_bid + best_ask) > 0 else 0

        return {
            'slippage_estimate': slippage_estimate,
            'market_impact': market_impact,
            'liquidation_time': liquidation_time,
            'spread_ratio': spread_ratio
        }

    def calculate_margin_requirement(self, symbol: str) -> Dict[str, float]:
        """Calculate margin requirement for a position."""
        if symbol not in self.positions:
            raise KeyError(f"Position for {symbol} not found")
        
        position = self.positions[symbol]
        
        # Simplified margin calculation (would depend on exchange rules)
        initial_margin = position.notional_value * 0.1  # 10% margin
        maintenance_margin = initial_margin * 0.8  # 80% of initial
        margin_ratio = initial_margin / position.notional_value if position.notional_value > 0 else 0
        available_margin = initial_margin * 0.5  # Assume 50% available

        return {
            'initial_margin': initial_margin,
            'maintenance_margin': maintenance_margin,
            'margin_ratio': margin_ratio,
            'available_margin': available_margin
        }

    def calculate_concentration_risk(self) -> Dict[str, float]:
        """Calculate portfolio concentration risk metrics."""
        if not self.positions:
            return {
                'herfindahl_index': 0.0,
                'largest_position_pct': 0.0,
                'sector_concentration': 0.0,
                'diversification_score': 1.0
            }

        total_exposure = sum(pos.notional_value for pos in self.positions.values())
        
        if total_exposure == 0:
            return {
                'herfindahl_index': 0.0,
                'largest_position_pct': 0.0,
                'sector_concentration': 0.0,
                'diversification_score': 1.0
            }

        # Herfindahl Index
        weights = [pos.notional_value / total_exposure for pos in self.positions.values()]
        herfindahl_index = sum(w**2 for w in weights)

        # Largest position percentage
        largest_position_pct = max(weights) if weights else 0.0

        # Sector concentration (placeholder - would need sector data)
        sector_concentration = 0.5

        # Diversification score (inverse of Herfindahl)
        diversification_score = 1.0 - herfindahl_index

        return {
            'herfindahl_index': herfindahl_index,
            'largest_position_pct': largest_position_pct,
            'sector_concentration': sector_concentration,
            'diversification_score': diversification_score
        }

    def run_scenario_analysis(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Run scenario analysis on the portfolio."""
        results = {}
        base_portfolio_value = sum(pos.notional_value for pos in self.positions.values())

        for scenario_name, price_changes in scenarios.items():
            portfolio_pnl = 0.0
            var_impact = 0.0
            liquidity_impact = 0.0
            risk_adjusted_return = 0.0

            for symbol, position in self.positions.items():
                if symbol in price_changes:
                    price_change = price_changes[symbol]
                    position_pnl = position.notional_value * price_change
                    portfolio_pnl += position_pnl

            # Calculate impacts
            var_impact = abs(portfolio_pnl) * 0.1  # Simplified
            liquidity_impact = portfolio_pnl * 0.05  # 5% of P&L
            risk_adjusted_return = portfolio_pnl / base_portfolio_value if base_portfolio_value > 0 else 0

            results[scenario_name] = {
                'portfolio_pnl': portfolio_pnl,
                'var_impact': var_impact,
                'liquidity_impact': liquidity_impact,
                'risk_adjusted_return': risk_adjusted_return
            }

        return results

    def calculate_risk_adjusted_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        if len(returns) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'omega_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }

        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0

        # Sortino Ratio (using downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0.0

        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Calmar Ratio
        total_return = np.prod(1 + returns) - 1
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        # Omega Ratio (simplified)
        omega_ratio = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.5

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }

    def forecast_volatility(self, prices: pd.Series) -> Dict[str, Any]:
        """Forecast volatility using various methods."""
        returns = prices.pct_change().dropna()
        
        # Historical volatility
        historical_vol = returns.std()

        # GARCH-like volatility (simplified)
        garch_vol = historical_vol * 1.1

        # EWMA volatility
        ewma_vol = returns.ewm(span=30).std().iloc[-1]

        # Volatility ratio
        volatility_ratio = garch_vol / historical_vol if historical_vol > 0 else 1.0

        # Regime classification
        if volatility_ratio < 0.8:
            regime = 'LOW'
        elif volatility_ratio < 1.2:
            regime = 'MEDIUM'
        elif volatility_ratio < 2.0:
            regime = 'HIGH'
        else:
            regime = 'EXTREME'

        return {
            'historical_vol': historical_vol,
            'garch_vol': garch_vol,
            'ewma_vol': ewma_vol,
            'volatility_ratio': volatility_ratio,
            'regime': regime
        }

    def assess_counterparty_risk(self, counterparties: Dict[str, Dict[str, Any]], 
                               exposure: Dict[str, float]) -> Dict[str, Any]:
        """Assess counterparty risk."""
        total_exposure = sum(exposure.values())
        weighted_default_prob = 0.0
        expected_loss = 0.0
        concentration_risk = 0.0
        diversification_score = 0.0

        if total_exposure > 0:
            for cp_name, cp_data in counterparties.items():
                if cp_name in exposure:
                    cp_exposure = exposure[cp_name]
                    weight = cp_exposure / total_exposure
                    
                    default_prob = cp_data.get('default_probability', 0.01)
                    weighted_default_prob += weight * default_prob
                    expected_loss += cp_exposure * default_prob

            # Concentration risk (Herfindahl)
            weights = [exposure[cp] / total_exposure for cp in counterparties.keys() if cp in exposure]
            concentration_risk = sum(w**2 for w in weights)
            diversification_score = 1.0 - concentration_risk

        return {
            'total_exposure': total_exposure,
            'weighted_default_prob': weighted_default_prob,
            'expected_loss': expected_loss,
            'concentration_risk': concentration_risk,
            'diversification_score': diversification_score
        }

    def check_regulatory_compliance(self, regulatory_limits: Dict[str, Any], 
                                  current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance."""
        violations = []
        warnings = []
        required_actions = []

        # Check leverage
        current_leverage = current_state.get('leverage', 1)
        max_leverage = regulatory_limits.get('max_leverage', 20)
        if current_leverage > max_leverage:
            violations.append(f'Leverage {current_leverage} exceeds limit {max_leverage}')

        # Check liquidity ratio
        current_liquidity = current_state.get('liquidity_ratio', 1.0)
        min_liquidity = regulatory_limits.get('min_liquidity_ratio', 0.3)
        if current_liquidity < min_liquidity:
            violations.append(f'Liquidity ratio {current_liquidity} below minimum {min_liquidity}')

        # Check concentration
        largest_position_pct = current_state.get('largest_position_pct', 0.0)
        max_concentration = regulatory_limits.get('max_concentration', 0.25)
        if largest_position_pct > max_concentration:
            violations.append(f'Largest position {largest_position_pct:.1%} exceeds limit {max_concentration:.1%}')

        # Check stress test frequency
        last_stress_test = current_state.get('last_stress_test', datetime.min)
        required_frequency = regulatory_limits.get('stress_test_frequency', 'daily')
        
        all_compliant = len(violations) == 0

        return {
            'all_compliant': all_compliant,
            'violations': violations,
            'warnings': warnings,
            'required_actions': required_actions
        }

    def adjust_risk_limits(self, market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Dynamically adjust risk limits based on market conditions."""
        volatility = market_conditions.get('volatility', 0.02)
        volume = market_conditions.get('volume', 1000000000)
        spread = market_conditions.get('spread', 0.0002)
        vix = market_conditions.get('vix', 25.0)

        # Base adjustments
        position_size_multiplier = 1.0
        leverage_multiplier = 1.0
        stop_loss_adjustment = 1.0

        # Adjust based on volatility
        if volatility > 0.03:  # High volatility
            position_size_multiplier *= 0.8
            leverage_multiplier *= 0.9
            stop_loss_adjustment *= 1.1
        elif volatility < 0.01:  # Low volatility
            position_size_multiplier *= 1.2
            leverage_multiplier *= 1.1

        # Adjust based on VIX
        if vix > 30:
            position_size_multiplier *= 0.9
            recommended_action = 'REDUCE_POSITION_SIZE'
        elif vix < 15:
            position_size_multiplier *= 1.1
            recommended_action = 'INCREASE_POSITION_SIZE'
        else:
            recommended_action = 'MAINTAIN_CURRENT_POSITION'

        return {
            'position_size_multiplier': position_size_multiplier,
            'leverage_multiplier': leverage_multiplier,
            'stop_loss_adjustment': stop_loss_adjustment,
            'recommended_action': recommended_action
        }

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        portfolio_risk = self.calculate_portfolio_risk()
        concentration_risk = self.calculate_concentration_risk()

        return {
            'executive_summary': {
                'total_positions': len(self.positions),
                'daily_pnl': self.daily_pnl,
                'risk_level': 'HIGH' if len(self.positions) > 8 else 'MEDIUM' if len(self.positions) > 4 else 'LOW'
            },
            'portfolio_overview': {
                'total_positions': len(self.positions),
                'total_exposure': portfolio_risk['total_exposure'],
                'total_unrealized_pnl': portfolio_risk['total_unrealized_pnl'],
                'daily_pnl': self.daily_pnl
            },
            'risk_metrics': {
                'portfolio_var': portfolio_risk['portfolio_var'],
                'expected_shortfall': portfolio_risk['expected_shortfall'],
                'concentration_risk': concentration_risk['herfindahl_index'],
                'largest_position_pct': concentration_risk['largest_position_pct']
            },
            'stress_test_results': self.run_stress_tests({
                'market_crash': {'BTCUSDT': -0.20, 'ETHUSDT': -0.15},
                'volatility_spike': {'BTCUSDT': -0.10, 'ETHUSDT': 0.05}
            }),
            'recommendations': [
                'Monitor concentration risk',
                'Review stop-loss levels',
                'Consider position sizing adjustments'
            ],
            'timestamp': datetime.now().isoformat()
        }

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (typically called at start of new trading day)."""
        self.daily_pnl = 0.0