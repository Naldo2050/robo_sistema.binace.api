# orderbook_analyzer/analyzer.py
import collections
import threading
import inspect as _inspect
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio


@dataclass
class OrderBookConfig:
    """Configuration for order book analyzer (legacy — prefer settings.OrderBookConfig)."""
    symbol: str
    depth_levels: int = 10
    update_interval_ms: int = 100
    imbalance_threshold: float = 0.7
    volume_threshold: float = 1000.0
    max_history_size: Optional[int] = None

    def __post_init__(self):
        if self.depth_levels <= 0:
            raise ValueError("depth_levels must be positive")
        if self.update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")
        if not 0 <= self.imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold must be between 0 and 1")


# ── Internal orderbook proxy ────────────────────────────────────────────────

class _OBProxy:
    """Minimal proxy that holds last-computed orderbook state.
    Instance methods can be replaced by Mocks in tests."""

    def __init__(self):
        self._imbalance = 0.0
        self._spread = 0.0
        self._mid_price = 0.0
        self._bid_volume = 0.0
        self._ask_volume = 0.0

    def get_imbalance(self) -> float:
        return self._imbalance

    def get_spread(self) -> float:
        return self._spread

    def get_mid_price(self) -> float:
        return self._mid_price

    def get_total_bid_volume(self) -> float:
        return self._bid_volume

    def get_total_ask_volume(self) -> float:
        return self._ask_volume


# ── OrderBookAnalyzer ────────────────────────────────────────────────────────

class OrderBookAnalyzer:
    """Order book analyzer for market microstructure analysis."""

    def __init__(self, config):
        if config is None:
            raise TypeError("config cannot be None")
        if not getattr(config, 'symbol', None):
            raise ValueError("symbol cannot be empty")

        self.config = config
        self.symbol = config.symbol
        self.depth_levels = getattr(config, 'depth_levels', 10)
        self.update_interval_ms = getattr(config, 'update_interval_ms', 100)
        self.imbalance_threshold = getattr(config, 'imbalance_threshold', 0.7)
        max_hist = getattr(config, 'max_history_size', None)
        self.max_history_size: int = max_hist if max_hist is not None else 1000

        self.orderbook = _OBProxy()
        self.order_flow = None
        self.ai_analyzer = None
        self.price_history: collections.deque = collections.deque(maxlen=self.max_history_size)
        self.metrics_history: List[Dict[str, Any]] = []
        self._current_metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # ── Core update ──────────────────────────────────────────────────────

    def process_orderbook_update(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an orderbook update."""
        try:
            bids = orderbook_data.get('bids')
            asks = orderbook_data.get('asks')

            if not bids or not asks:
                return {'success': False, 'error': 'Invalid orderbook data',
                        'timestamp': datetime.now().isoformat()}

            if 'timestamp' not in orderbook_data:
                return {'success': False, 'error': 'Missing timestamp',
                        'timestamp': datetime.now().isoformat()}

            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            spread = best_ask - best_bid if best_bid and best_ask else 0.0
            mid_price = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0

            bid_volume = sum(float(qty) for _, qty in bids[:self.depth_levels])
            ask_volume = sum(float(qty) for _, qty in asks[:self.depth_levels])
            total_volume = bid_volume + ask_volume
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0.0

            with self._lock:
                # Update proxy
                self.orderbook._spread = spread
                self.orderbook._mid_price = mid_price
                self.orderbook._imbalance = imbalance
                self.orderbook._bid_volume = bid_volume
                self.orderbook._ask_volume = ask_volume

                # Record price
                if mid_price:
                    self.price_history.append(mid_price)

                # Store metrics snapshot
                self._current_metrics = {
                    'spread': spread,
                    'mid_price': mid_price,
                    'bid_ask_imbalance': imbalance,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'total_volume': total_volume,
                }
                metrics = dict(self._current_metrics)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

            return {
                'success': True,
                'spread': spread,
                'mid_price': mid_price,
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
            }

        except Exception as e:
            return {'success': False, 'error': str(e),
                    'timestamp': datetime.now().isoformat()}

    # ── Metrics ──────────────────────────────────────────────────────────

    def calculate_metrics(self) -> Dict[str, Any]:
        """Return last computed orderbook metrics."""
        with self._lock:
            return dict(self._current_metrics) if self._current_metrics else {}

    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """Calculate advanced metrics from price history."""
        prices = list(self.price_history)
        if len(prices) < 2:
            return {}

        import statistics as _stats
        price_mean = _stats.mean(prices)
        price_std = _stats.stdev(prices)
        volatility = float(price_std / price_mean) if price_mean != 0 else 0.0
        diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        price_trend = float(sum(diffs) / len(diffs)) if diffs else 0.0

        return {
            'price_mean': float(price_mean),
            'price_std': float(price_std),
            'volatility': volatility,
            'price_trend': price_trend,
            'min_price': float(min(prices)),
            'max_price': float(max(prices)),
            'price_range': float(max(prices) - min(prices)),
        }

    # ── AI analysis ──────────────────────────────────────────────────────

    async def analyze_with_ai(self, orderbook_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze orderbook with AI analyzer."""
        try:
            if not self.ai_analyzer:
                return {'success': False, 'error': 'AI analyzer not available'}
            result = self.ai_analyzer.analyze_orderbook(orderbook_snapshot)
            if _inspect.isawaitable(result):
                result = await result  # type: ignore[misc]
            if isinstance(result, dict):
                result['success'] = True
                return result
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e),
                    'timestamp': datetime.now().isoformat()}

    # ── Analysis methods ─────────────────────────────────────────────────

    def detect_market_manipulation(self) -> Dict[str, Any]:
        return {
            'is_spoofing': False,
            'is_layering': False,
            'confidence': 0.0,
            'patterns_detected': [],
            'indicators': {},
        }

    def analyze_volume_profile(self, volume_data: Dict[str, Any]) -> Dict[str, Any]:
        price_levels = volume_data.get('price_levels', [])
        bid_volumes = volume_data.get('bid_volumes', [])
        ask_volumes = volume_data.get('ask_volumes', [])

        if not price_levels or not bid_volumes or not ask_volumes:
            return {'volume_weighted_price': 0, 'total_volume': 0,
                    'high_volume_nodes': [], 'support_levels': [], 'resistance_levels': []}

        total_volume = sum(bid_volumes) + sum(ask_volumes)
        if total_volume == 0:
            return {'volume_weighted_price': 0, 'total_volume': 0,
                    'high_volume_nodes': [], 'support_levels': [], 'resistance_levels': []}

        vwap = sum(p * (b + a) for p, b, a in zip(price_levels, bid_volumes, ask_volumes)) / total_volume
        avg_vol = total_volume / len(price_levels)
        high_nodes = [p for p, b, a in zip(price_levels, bid_volumes, ask_volumes)
                      if (b + a) > avg_vol * 2]

        return {
            'volume_weighted_price': vwap,
            'total_volume': total_volume,
            'high_volume_nodes': high_nodes,
            'support_levels': [],
            'resistance_levels': [],
        }

    def analyze_order_flow(self, trade_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        buy_vol = sum(float(t.get('quoteQty', 0)) for t in trade_data if not t.get('isBuyerMaker', True))
        sell_vol = sum(float(t.get('quoteQty', 0)) for t in trade_data if t.get('isBuyerMaker', True))
        total = buy_vol + sell_vol

        if total == 0:
            return {'vpin': 0, 'trade_imbalance': 0, 'buy_pressure': 0,
                    'sell_pressure': 0, 'total_trades': len(trade_data)}

        imbalance = (buy_vol - sell_vol) / total
        return {
            'vpin': abs(imbalance),
            'trade_imbalance': imbalance,
            'buy_pressure': buy_vol / total,
            'sell_pressure': sell_vol / total,
            'total_trades': len(trade_data),
        }

    def analyze_market_depth(self, depth_data: Dict[str, Any]) -> Dict[str, Any]:
        bids = depth_data.get('bids', [])
        asks = depth_data.get('asks', [])

        if not bids or not asks:
            return {'depth_imbalance': 0, 'total_bid_volume': 0, 'total_ask_volume': 0,
                    'average_order_size': 0, 'liquidity_clusters': []}

        bid_vols = [float(lv.get('volume', 0)) for lv in bids]
        ask_vols = [float(lv.get('volume', 0)) for lv in asks]
        total_bid = sum(bid_vols)
        total_ask = sum(ask_vols)
        total = total_bid + total_ask
        depth_imbalance = (total_bid - total_ask) / total if total > 0 else 0.0
        all_sizes = bid_vols + ask_vols
        avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 0.0

        return {
            'depth_imbalance': depth_imbalance,
            'total_bid_volume': total_bid,
            'total_ask_volume': total_ask,
            'average_order_size': avg_size,
            'liquidity_clusters': [],
        }

    def calculate_technical_indicators(self, prices: List[float]) -> Dict[str, Any]:
        if len(prices) < 5:
            return {}

        sma_5 = sum(prices[-5:]) / 5
        alpha = 2 / (5 + 1)
        ema_5 = float(prices[0])
        for p in prices[1:]:
            ema_5 = alpha * float(p) + (1 - alpha) * ema_5

        if len(prices) >= 14:
            gains, losses = [], []
            for i in range(1, min(len(prices), 15)):
                change = float(prices[i]) - float(prices[i - 1])
                gains.append(change if change > 0 else 0.0)
                losses.append(-change if change < 0 else 0.0)
            avg_gain = sum(gains) / len(gains) if gains else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            rs = avg_gain / avg_loss if avg_loss > 0 else 0.0
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50.0

        macd = sma_5 - ema_5
        macd_signal = macd * 0.9
        macd_histogram = macd - macd_signal

        return {
            'sma': sma_5, 'ema': ema_5, 'rsi': rsi,
            'macd': macd, 'macd_signal': macd_signal, 'macd_histogram': macd_histogram,
        }

    def calculate_position_risk(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        size = position_data.get('position_size', 0)
        entry = position_data.get('entry_price', 0)
        current = position_data.get('current_price', 0)
        stop = position_data.get('stop_loss', 0)
        take = position_data.get('take_profit', 0)

        unrealized_pnl = (current - entry) * size
        position_value = entry * size
        pnl_pct = unrealized_pnl / position_value if position_value > 0 else 0.0
        d_stop = abs(current - stop) if stop > 0 else 0.0
        d_take = abs(take - current) if take > 0 else 0.0
        rr = d_take / d_stop if d_stop > 0 else 0.0

        return {
            'unrealized_pnl': unrealized_pnl,
            'pnl_percentage': pnl_pct,
            'position_value': position_value,
            'risk_reward_ratio': rr,
            'distance_to_stop': d_stop,
            'distance_to_take': d_take,
        }

    def detect_market_regime(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        mean_reversion = market_conditions.get('mean_reversion_score', 0.5)

        if trend_strength > 0.7:
            regime, confidence = 'TRENDING', trend_strength
        elif mean_reversion > 0.7:
            regime, confidence = 'MEAN_REVERTING', mean_reversion
        elif volatility > 0.03:
            regime, confidence = 'VOLATILE', min(volatility * 20, 1.0)
        else:
            regime, confidence = 'SIDEWAYS', 0.6

        return {
            'regime': regime,
            'confidence': confidence,
            'indicators': market_conditions,
        }

    def generate_trading_signal(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        signals, weights = [], []
        for data in analysis_results.values():
            if isinstance(data, dict) and 'signal' in data:
                signals.append(data['signal'])
                weights.append(data.get('strength', 0.5))

        if not signals:
            return {'final_signal': 'NEUTRAL', 'confidence': 0.5,
                    'components': analysis_results, 'timestamp': datetime.now().isoformat()}

        scores: Dict[str, float] = {'BUY': 0.0, 'SELL': 0.0, 'NEUTRAL': 0.0}
        total_w = sum(weights)
        for sig, w in zip(signals, weights):
            if sig in scores:
                scores[sig] += w
        for k in scores:
            scores[k] = scores[k] / total_w if total_w > 0 else 0.0

        final = max(scores, key=scores.get)  # type: ignore[arg-type]
        return {
            'final_signal': final,
            'confidence': scores[final],
            'components': analysis_results,
            'signal_scores': scores,
            'timestamp': datetime.now().isoformat(),
        }

    def benchmark_performance(self, signal_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not signal_history:
            return {'total_return': 0, 'win_rate': 0, 'sharpe_ratio': 0,
                    'max_drawdown': 0, 'total_trades': 0}

        pnls = [float(s.get('actual_return', 0)) for s in signal_history]
        total = sum(pnls)
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = total / len(pnls)
        std = (sum((p - avg) ** 2 for p in pnls) / len(pnls)) ** 0.5
        sharpe = avg / std if std > 0 else 0.0

        cumulative, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            max_dd = max(max_dd, peak - cumulative)

        return {
            'total_return': total,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': -max_dd,
            'total_trades': len(pnls),
        }

    # ── Signal generation ────────────────────────────────────────────────

    def generate_imbalance_signal(self) -> str:
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
        return 'NEUTRAL'

    # ── State management ─────────────────────────────────────────────────

    def reset_state(self) -> None:
        with self._lock:
            self.price_history = collections.deque(maxlen=self.max_history_size)
            self.metrics_history = []
            self._current_metrics = {}

    def cleanup_old_data(self) -> None:
        with self._lock:
            if len(self.price_history) > self.max_history_size:
                trimmed = list(self.price_history)[-self.max_history_size:]
                self.price_history = collections.deque(trimmed, maxlen=self.max_history_size)
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]

    def serialize_state(self) -> Dict[str, Any]:
        cfg = {k: v for k, v in vars(self.config).items()}
        return {
            'config': cfg,
            'price_history': list(self.price_history),
            'metrics': dict(self._current_metrics),
            'timestamp': datetime.now().isoformat(),
        }

    def deserialize_state(self, state: Dict[str, Any]) -> None:
        if 'config' in state:
            for k, v in state['config'].items():
                if hasattr(self.config, k):
                    try:
                        object.__setattr__(self.config, k, v)
                    except Exception:
                        pass
            self.symbol = getattr(self.config, 'symbol', self.symbol)
        if 'price_history' in state:
            self.price_history = collections.deque(
                state['price_history'], maxlen=self.max_history_size)

    # ── Async helper ─────────────────────────────────────────────────────

    async def _async_analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        return {'result': 'analysis_complete'}
