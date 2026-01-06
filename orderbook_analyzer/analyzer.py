# orderbook_analyzer/analyzer.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio


@dataclass
class OrderBookConfig:
    """Configuration for order book analyzer."""
    symbol: str
    depth_levels: int = 10
    update_interval_ms: int = 100
    imbalance_threshold: float = 0.7
    volume_threshold: float = 1000.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.depth_levels <= 0:
            raise ValueError("depth_levels must be positive")
        if self.update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")
        if not 0 <= self.imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold must be between 0 and 1")


class OrderBookAnalyzer:
    """Order book analyzer for market microstructure analysis."""

    def __init__(self, config: OrderBookConfig):
        self.config = config
        self.symbol = config.symbol
        self.depth_levels = config.depth_levels
        self.update_interval_ms = config.update_interval_ms
        self.imbalance_threshold = config.imbalance_threshold
        
        # Internal state
        self.orderbook = None
        self.order_flow = None
        self.ai_analyzer = None
        self.price_history = []
        self.max_history_size = 1000

    def process_orderbook_update(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an orderbook update."""
        try:
            # Validate orderbook data
            if not orderbook_data.get('bids') or not orderbook_data.get('asks'):
                return {'success': False, 'error': 'Invalid orderbook data'}

            # Calculate basic metrics
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            
            bid_volume = sum(float(qty) for _, qty in bids[:self.depth_levels])
            ask_volume = sum(float(qty) for _, qty in asks[:self.depth_levels])
            
            imbalance = 0
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume

            return {
                'success': True,
                'spread': spread,
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate basic orderbook metrics."""
        if not self.orderbook:
            return {}

        return {
            'spread': self.orderbook.get_spread(),
            'mid_price': self.orderbook.get_mid_price(),
            'bid_ask_imbalance': self.orderbook.get_imbalance(),
            'bid_volume': self.orderbook.get_total_bid_volume(),
            'ask_volume': self.orderbook.get_total_ask_volume(),
            'total_volume': self.orderbook.get_total_bid_volume() + self.orderbook.get_total_ask_volume()
        }

    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """Calculate advanced metrics."""
        if len(self.price_history) < 2:
            return {}

        prices = self.price_history[-10:]  # Last 10 prices
        price_mean = sum(prices) / len(prices)
        price_std = (sum((p - price_mean) ** 2 for p in prices) / len(prices)) ** 0.5
        
        volatility = price_std / price_mean if price_mean > 0 else 0
        
        # Simple trend calculation
        if len(prices) >= 3:
            recent_trend = 'UP' if prices[-1] > prices[-3] else 'DOWN'
        else:
            recent_trend = 'NEUTRAL'

        return {
            'price_std': price_std,
            'price_mean': price_mean,
            'volatility': volatility,
            'price_trend': recent_trend
        }

    async def analyze_with_ai(self, orderbook_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze orderbook with AI (placeholder implementation)."""
        try:
            if not self.ai_analyzer:
                return {'success': False, 'error': 'AI analyzer not available'}

            # Placeholder AI analysis
            analysis_result = {
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'reasoning': 'AI analysis not implemented',
                'predicted_price': None
            }

            return analysis_result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def detect_market_manipulation(self) -> Dict[str, Any]:
        """Detect potential market manipulation patterns."""
        # Placeholder implementation
        return {
            'is_spoofing': False,
            'is_layering': False,
            'confidence': 0.0,
            'patterns_detected': []
        }

    def analyze_volume_profile(self, volume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume profile."""
        price_levels = volume_data.get('price_levels', [])
        bid_volumes = volume_data.get('bid_volumes', [])
        ask_volumes = volume_data.get('ask_volumes', [])
        
        if not price_levels or not bid_volumes or not ask_volumes:
            return {}

        # Calculate volume weighted price
        total_volume = sum(bid_volumes) + sum(ask_volumes)
        if total_volume == 0:
            return {'volume_weighted_price': 0}

        vwap = sum(price * (bid + ask) for price, bid, ask in 
                  zip(price_levels, bid_volumes, ask_volumes)) / total_volume

        # Find high volume nodes
        avg_volume = total_volume / len(price_levels)
        high_volume_nodes = [price for price, bid, ask in 
                           zip(price_levels, bid_volumes, ask_volumes) 
                           if (bid + ask) > avg_volume * 2]

        return {
            'volume_weighted_price': vwap,
            'high_volume_nodes': high_volume_nodes,
            'support_levels': [],
            'resistance_levels': []
        }

    def analyze_order_flow(self, trade_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze order flow."""
        if not self.order_flow:
            return {'vpin': 0, 'trade_imbalance': 0, 'buy_pressure': 0, 'sell_pressure': 0}

        # Placeholder calculations
        total_buy_volume = sum(trade.get('quoteQty', 0) for trade in trade_data 
                              if not trade.get('isBuyerMaker', True))
        total_sell_volume = sum(trade.get('quoteQty', 0) for trade in trade_data 
                               if trade.get('isBuyerMaker', True))

        total_volume = total_buy_volume + total_sell_volume
        if total_volume == 0:
            return {'vpin': 0, 'trade_imbalance': 0, 'buy_pressure': 0, 'sell_pressure': 0}

        trade_imbalance = (total_buy_volume - total_sell_volume) / total_volume
        
        return {
            'vpin': abs(trade_imbalance),  # Volume Profile Imbalance
            'trade_imbalance': trade_imbalance,
            'buy_pressure': total_buy_volume / total_volume,
            'sell_pressure': total_sell_volume / total_volume
        }

    def analyze_market_depth(self, depth_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market depth."""
        bids = depth_data.get('bids', [])
        asks = depth_data.get('asks', [])
        
        if not bids or not asks:
            return {}

        bid_volumes = [level.get('volume', 0) for level in bids]
        ask_volumes = [level.get('volume', 0) for level in asks]
        
        total_bid_volume = sum(bid_volumes)
        total_ask_volume = sum(ask_volumes)
        total_volume = total_bid_volume + total_ask_volume
        
        depth_imbalance = 0
        if total_volume > 0:
            depth_imbalance = (total_bid_volume - total_ask_volume) / total_volume

        # Calculate average order size
        all_order_sizes = bid_volumes + ask_volumes
        avg_order_size = sum(all_order_sizes) / len(all_order_sizes) if all_order_sizes else 0

        return {
            'depth_imbalance': depth_imbalance,
            'average_order_size': avg_order_size,
            'liquidity_clusters': []
        }

    def calculate_technical_indicators(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate technical indicators."""
        if len(prices) < 5:
            return {}

        # Simple Moving Average
        sma_5 = sum(prices[-5:]) / 5 if len(prices) >= 5 else 0
        
        # Exponential Moving Average (simplified)
        alpha = 2 / (5 + 1)
        ema_5 = prices[0]
        for price in prices[1:]:
            ema_5 = alpha * price + (1 - alpha) * ema_5

        # RSI (simplified)
        if len(prices) >= 14:
            gains = []
            losses = []
            for i in range(1, min(len(prices), 15)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50

        # MACD (simplified)
        macd = sma_5 - ema_5
        macd_signal = macd * 0.9  # Simplified signal line
        macd_histogram = macd - macd_signal

        return {
            'sma': sma_5,
            'ema': ema_5,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }

    def calculate_position_risk(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for a position."""
        position_size = position_data.get('position_size', 0)
        entry_price = position_data.get('entry_price', 0)
        current_price = position_data.get('current_price', 0)
        stop_loss = position_data.get('stop_loss', 0)
        take_profit = position_data.get('take_profit', 0)

        unrealized_pnl = (current_price - entry_price) * position_size
        pnl_percentage = (unrealized_pnl / (entry_price * position_size)) if entry_price > 0 else 0

        distance_to_stop = abs(current_price - stop_loss) if stop_loss > 0 else 0
        distance_to_take = abs(take_profit - current_price) if take_profit > 0 else 0

        risk_reward_ratio = 0
        if distance_to_stop > 0:
            risk_reward_ratio = distance_to_take / distance_to_stop

        return {
            'unrealized_pnl': unrealized_pnl,
            'pnl_percentage': pnl_percentage,
            'risk_reward_ratio': risk_reward_ratio,
            'distance_to_stop': distance_to_stop,
            'distance_to_take': distance_to_take
        }

    def detect_market_regime(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Detect current market regime."""
        volatility = market_conditions.get('volatility', 0.02)
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        mean_reversion_score = market_conditions.get('mean_reversion_score', 0.5)

        # Simple regime classification
        if trend_strength > 0.7:
            regime = 'TRENDING'
            confidence = trend_strength
        elif mean_reversion_score > 0.7:
            regime = 'MEAN_REVERTING'
            confidence = mean_reversion_score
        elif volatility > 0.03:
            regime = 'VOLATILE'
            confidence = min(volatility * 20, 1.0)
        else:
            regime = 'SIDEWAYS'
            confidence = 0.6

        return {
            'regime': regime,
            'confidence': confidence
        }

    def generate_trading_signal(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal from analysis results."""
        components = analysis_results
        
        # Simple signal generation logic
        signals = []
        weights = []
        
        for component, data in components.items():
            if isinstance(data, dict) and 'signal' in data:
                signals.append(data['signal'])
                weights.append(data.get('strength', 0.5))
        
        if not signals:
            return {
                'final_signal': 'NEUTRAL',
                'confidence': 0.5,
                'components': components,
                'timestamp': datetime.now().isoformat()
            }

        # Weighted signal calculation
        signal_scores = {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}
        total_weight = sum(weights)
        
        for signal, weight in zip(signals, weights):
            if signal in signal_scores:
                signal_scores[signal] += weight

        # Normalize scores
        for signal in signal_scores:
            signal_scores[signal] = signal_scores[signal] / total_weight if total_weight > 0 else 0

        # Determine final signal
        final_signal = max(signal_scores, key=signal_scores.get)
        confidence = signal_scores[final_signal]

        return {
            'final_signal': final_signal,
            'confidence': confidence,
            'components': components,
            'signal_scores': signal_scores,
            'timestamp': datetime.now().isoformat()
        }

    def benchmark_performance(self, signal_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark signal performance."""
        if not signal_history:
            return {}

        total_return = 0
        winning_trades = 0
        total_trades = len(signal_history)

        for signal_data in signal_history:
            actual_return = signal_data.get('actual_return', 0)
            total_return += actual_return
            if actual_return > 0:
                winning_trades += 1

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Simplified Sharpe ratio calculation
        returns = [signal.get('actual_return', 0) for signal in signal_history]
        avg_return = sum(returns) / len(returns) if returns else 0
        return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0

        # Simplified max drawdown
        cumulative_returns = []
        cumulative = 0
        for ret in returns:
            cumulative += ret
            cumulative_returns.append(cumulative)
        
        running_max = []
        max_so_far = 0
        for ret in cumulative_returns:
            max_so_far = max(max_so_far, ret)
            running_max.append(max_so_far)
        
        drawdowns = [cumulative - max_val for cumulative, max_val in zip(cumulative_returns, running_max)]
        max_drawdown = min(drawdowns) if drawdowns else 0

        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }

    async def _async_analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Async analysis method."""
        await asyncio.sleep(0.01)  # Simulate async work
        return {'result': 'analysis_complete'}

    def reset_state(self) -> None:
        """Reset analyzer state."""
        self.price_history.clear()
        self.orderbook = None
        self.order_flow = None

    def cleanup_old_data(self) -> None:
        """Clean up old data to manage memory."""
        if len(self.price_history) > self.max_history_size:
            self.price_history = self.price_history[-self.max_history_size:]

    def serialize_state(self) -> Dict[str, Any]:
        """Serialize analyzer state."""
        return {
            'config': self.config.__dict__.copy(),
            'metrics': self.calculate_metrics(),
            'timestamp': datetime.now().isoformat()
        }

    def deserialize_state(self, state: Dict[str, Any]) -> None:
        """Deserialize analyzer state."""
        if 'config' in state:
            config_data = state['config']
            self.config = OrderBookConfig(**config_data)
            self.symbol = self.config.symbol
            self.depth_levels = self.config.depth_levels

    def generate_imbalance_signal(self) -> str:
        """Generate signal based on imbalance."""
        if not self.orderbook:
            return 'NEUTRAL'
        
        imbalance = self.orderbook.get_imbalance()
        
        if imbalance >= 0.8:
            return 'STRONG_BUY'
        elif imbalance >= 0.6:
            return 'BUY'
        elif imbalance <= -0.8:
            return 'STRONG_SELL'
        elif imbalance <= -0.6:
            return 'SELL'
        else:
            return 'NEUTRAL'