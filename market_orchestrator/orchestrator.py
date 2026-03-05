# market_orchestrator/orchestrator.py
import threading
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


# ── Position ──────────────────────────────────────────────────────────────────

class Position:
    """Represents a trading position."""

    def __init__(self, symbol: str, side: str, size: float, entry_price: float,
                 current_price: Optional[float] = None, stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None, timestamp: Optional[datetime] = None):
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.current_price = current_price if current_price is not None else entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.timestamp = timestamp or datetime.now()
        self.unrealized_pnl = self._calc_pnl()

    def _calc_pnl(self) -> float:
        if self.side.upper() == 'BUY':
            return self.size * (self.current_price - self.entry_price)
        return self.size * (self.entry_price - self.current_price)

    def update_price(self, new_price: float) -> float:
        self.current_price = new_price
        self.unrealized_pnl = self._calc_pnl()
        return self.unrealized_pnl

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime)
                         else str(self.timestamp),
        }


# ── OrchestratorConfig ────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    """Configuration for market orchestrator."""
    symbol: str
    stream_url: str = ""
    window_size_minutes: int = 5
    vol_factor_exh: float = 1.5
    history_size: int = 100
    delta_std_dev_factor: float = 2.0
    context_sma_period: int = 20
    liquidity_flow_alert_percentage: float = 0.4
    wall_std_dev_factor: float = 3.0
    # Compatibility fields used by tests / external callers
    max_position_size: Optional[float] = None
    max_daily_loss: Optional[float] = None
    trade_cooldown_seconds: int = 0
    enable_ai_analysis: bool = True
    max_open_positions: int = 10
    max_correlation: float = 0.8
    var_confidence_level: float = 0.95

    def __post_init__(self):
        if self.window_size_minutes <= 0:
            raise ValueError("window_size_minutes must be positive")
        if self.vol_factor_exh <= 0:
            raise ValueError("vol_factor_exh must be positive")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")
        if self.max_position_size is None:
            self.max_position_size = 100000
        if self.max_daily_loss is None:
            self.max_daily_loss = 0.05


# ── MarketOrchestrator ────────────────────────────────────────────────────────

class MarketOrchestrator:
    """Market orchestrator for managing trading operations."""

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        if config is None:
            config = OrchestratorConfig(symbol="BTCUSDT")
        self.config: OrchestratorConfig = config
        self.symbol = config.symbol
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        self.state: Dict[str, Any] = {}

        # Position / trade state
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Any] = []
        self.daily_pnl: float = 0.0
        self.last_trade_time: Optional[datetime] = None

        # Error state
        self.error_count: int = 0
        self.last_error_time: Optional[datetime] = None

        # Component handles (replaced by mocks in tests)
        self.orderbook_analyzer = None
        self.risk_manager = None
        self.trade_executor = None
        self.signal_processor = None
        self.ai_runner = None

        # Performance metrics
        self.performance_metrics: Dict[str, Any] = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_volume': 0.0,
            'avg_trade_size': 0.0,
            'start_time': datetime.now().isoformat(),
        }

        self._lock = threading.RLock()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        if self.is_running:
            return False
        self.is_running = True
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.state = {
            'status': 'running',
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat(),
        }
        return True

    def stop(self) -> bool:
        if not self.is_running:
            return False
        self.is_running = False
        self.state['status'] = 'stopped'
        self.state['stop_time'] = datetime.now().isoformat()
        return True

    # ── Market data processing ─────────────────────────────────────────────

    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_running:
            return {'success': False, 'error': 'Orchestrator is not running'}

        try:
            # Process orderbook
            analysis = {}
            if self.orderbook_analyzer:
                analysis = self.orderbook_analyzer.process_orderbook_update(
                    market_data.get('orderbook', market_data))

            # Generate signal
            signal = {}
            if self.signal_processor:
                signal = self.signal_processor.process(market_data)

            # AI analysis (optional)
            ai_analysis: Dict[str, Any] = {'success': False, 'skipped': True}
            if self.config.enable_ai_analysis and self.ai_runner:
                try:
                    import inspect as _inspect
                    result = self.ai_runner.analyze_orderbook(market_data)
                    if _inspect.isawaitable(result):
                        result = await result  # type: ignore[misc]
                    ai_analysis = result if isinstance(result, dict) else {'success': True, 'result': result}
                except Exception as e:
                    ai_analysis = {'success': False, 'error': str(e)}

            self.last_update = datetime.now()
            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'signal': signal,
                'ai_analysis': ai_analysis,
                'market_data': market_data,
            }
        except Exception as e:
            self.error_count += 1
            self.last_error_time = datetime.now()
            return {'success': False, 'error': str(e)}

    async def execute_trade_flow(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_running:
            return {'success': False, 'error': 'Orchestrator is not running'}

        # Cooldown check
        if not self.can_execute_trade():
            remaining = self.get_cooldown_remaining()
            return {'success': False, 'reason': 'cooldown_active',
                    'cooldown_remaining': remaining}

        try:
            # Risk check
            risk_result: Dict[str, Any] = {'approved': True}
            if self.risk_manager:
                risk_result = self.risk_manager.check_trade_request(signal)
                if not risk_result.get('approved', False):
                    return {'success': False, 'reason': 'risk_rejected',
                            'risk_check': risk_result}

            # Execute trade
            exec_result: Dict[str, Any] = {'success': False, 'error': 'No trade executor'}
            if self.trade_executor:
                exec_result = self.trade_executor.execute_trade(signal)

            self.performance_metrics['total_trades'] += 1
            self.last_trade_time = datetime.now()

            if exec_result.get('success'):
                self.performance_metrics['successful_trades'] += 1
                # Create position from signal
                sym = signal.get('symbol', self.config.symbol)
                pos = self.add_position({
                    'symbol': sym,
                    'side': signal.get('signal', 'BUY'),
                    'size': float(signal.get('size', exec_result.get('filled_size', 0))),
                    'entry_price': float(signal.get('price', exec_result.get('filled_price', 0))),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                })
                return {
                    'success': True,
                    'execution_result': exec_result,
                    'risk_check': risk_result,
                    'position': pos.to_dict(),
                    'timestamp': datetime.now().isoformat(),
                    'signal': signal,
                }

            return {'success': False, 'reason': 'execution_failed',
                    'execution_result': exec_result, 'signal': signal}

        except Exception as e:
            self.error_count += 1
            self.last_error_time = datetime.now()
            return {'success': False, 'error': str(e)}

    async def process_market_data_stream(self, data_stream=None):
        """Process a stream of market data updates."""
        if not self.is_running:
            return []
        if data_stream is None:
            return []
        results = []
        if isinstance(data_stream, (list, tuple)):
            for data in data_stream:
                if not self.is_running:
                    break
                result = await self.process_market_data(data)
                results.append(result)
        else:
            async for data in data_stream:
                if not self.is_running:
                    break
                result = await self.process_market_data(data)
                results.append(result)
        return results

    # ── Positions ──────────────────────────────────────────────────────────

    def add_position(self, position_data: Dict[str, Any]) -> Position:
        ts = position_data.get('timestamp')
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                ts = None
        entry_price = position_data.get('entry_price')
        if entry_price is None:
            raise ValueError("entry_price is required")
        pos = Position(
            symbol=position_data['symbol'],
            side=position_data['side'],
            size=float(position_data['size']),
            entry_price=float(entry_price),
            current_price=float(position_data.get('current_price', position_data['entry_price'])),
            stop_loss=position_data.get('stop_loss'),
            take_profit=position_data.get('take_profit'),
            timestamp=ts,
        )
        with self._lock:
            self.positions[pos.symbol] = pos
        return pos

    def update_position(self, symbol: str, new_price: float) -> Dict[str, Any]:
        with self._lock:
            if symbol not in self.positions:
                return {'success': False, 'error': f'Position {symbol} not found'}
            pos = self.positions[symbol]
            pnl = pos.update_price(new_price)
            # Also notify risk manager if available
            if self.risk_manager:
                try:
                    self.risk_manager.update_position(symbol, new_price)
                except Exception:
                    pass
            return {'success': True, 'unrealized_pnl': pnl, 'position': pos.to_dict()}

    def remove_position(self, symbol: str, exit_price: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            if symbol not in self.positions:
                return {'success': False, 'error': f'Position {symbol} not found'}
            pos = self.positions[symbol]
            ep = exit_price if exit_price is not None else pos.current_price
            if pos.side.upper() == 'BUY':
                realized = pos.size * (ep - pos.entry_price)
            else:
                realized = pos.size * (pos.entry_price - ep)
            self.daily_pnl += realized
            self.trade_history.append({
                'symbol': symbol,
                'side': pos.side,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'exit_price': ep,
                'realized_pnl': realized,
                'timestamp': datetime.now().isoformat(),
            })
            del self.positions[symbol]
        return {'success': True, 'realized_pnl': realized, 'position': pos.to_dict()}

    def get_portfolio_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_exposure = sum(p.size * p.current_price for p in self.positions.values())
            total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
            total_value = total_exposure + total_unrealized
            return {
                'total_positions': len(self.positions),
                'total_exposure': total_exposure,
                'total_unrealized_pnl': total_unrealized,
                'total_value': total_value,
                'daily_pnl': self.daily_pnl,
                'positions': {sym: pos.to_dict() for sym, pos in self.positions.items()},
                'risk_summary': {},
                'timestamp': datetime.now().isoformat(),
            }

    # ── Cooldown / limits ──────────────────────────────────────────────────

    def can_execute_trade(self) -> bool:
        if self.last_trade_time is None:
            return True
        cooldown = getattr(self.config, 'trade_cooldown_seconds', 0)
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= cooldown

    def get_cooldown_remaining(self) -> float:
        if self.last_trade_time is None:
            return 0
        cooldown = getattr(self.config, 'trade_cooldown_seconds', 0)
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        remaining = cooldown - elapsed
        return max(0.0, remaining)

    def check_position_limits(self, additional_size: float = 0.0) -> bool:
        current_exposure = sum(p.size * p.current_price for p in self.positions.values())
        max_size = getattr(self.config, 'max_position_size', 100000) or 100000
        return (current_exposure + additional_size) <= max_size

    def check_daily_loss_limit(self) -> bool:
        max_loss_pct = getattr(self.config, 'max_daily_loss', 0.05) or 0.05
        capital = 100000  # assumed capital
        max_loss_abs = max_loss_pct * capital
        return self.daily_pnl >= -max_loss_abs

    # ── Error state ────────────────────────────────────────────────────────

    def is_in_error_state(self) -> bool:
        if self.error_count < 10:
            return False
        if self.last_error_time is None:
            return False
        age = (datetime.now() - self.last_error_time).total_seconds()
        return age < 300  # errors within last 5 minutes

    def attempt_recovery(self) -> bool:
        if self.last_error_time is None:
            return True
        age = (datetime.now() - self.last_error_time).total_seconds()
        if age < 300:  # less than 5 minutes
            return False
        self.error_count = 0
        self.last_error_time = None
        return True

    # ── Performance metrics ────────────────────────────────────────────────

    def get_performance_metrics(self) -> Dict[str, Any]:
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        total = self.performance_metrics.get('total_trades', 0)
        successful = self.performance_metrics.get('successful_trades', 0)
        success_rate = successful / total if total > 0 else 0.0
        error_rate = self.error_count / total if total > 0 else 0.0
        return {
            'total_trades': total,
            'successful_trades': successful,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'is_in_error_state': self.is_in_error_state(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            **{k: v for k, v in self.performance_metrics.items()
               if k not in ('total_trades', 'successful_trades')},
        }

    # ── Configuration ──────────────────────────────────────────────────────

    def update_configuration(self, new_config: OrchestratorConfig) -> Dict[str, Any]:
        old = self.config
        old_dict = {k: getattr(old, k) for k in old.__dataclass_fields__}
        self.config = new_config
        self.symbol = new_config.symbol
        new_dict = {k: getattr(new_config, k) for k in new_config.__dataclass_fields__}
        return {'success': True, 'old_config': old_dict, 'new_config': new_dict}

    # ── Serialization ──────────────────────────────────────────────────────

    def serialize_state(self) -> Dict[str, Any]:
        cfg = {k: getattr(self.config, k) for k in self.config.__dataclass_fields__}
        return {
            'config': cfg,
            'positions': {sym: pos.to_dict() for sym, pos in self.positions.items()},
            'daily_pnl': self.daily_pnl,
            'performance_metrics': dict(self.performance_metrics),
            'error_count': self.error_count,
            'timestamp': datetime.now().isoformat(),
        }

    def deserialize_state(self, state: Dict[str, Any]) -> None:
        cfg_data = state.get('config', {})
        # Update config fields that exist
        for k, v in cfg_data.items():
            if hasattr(self.config, k):
                object.__setattr__(self.config, k, v)
        self.symbol = self.config.symbol

        # Restore positions
        self.positions = {}
        for sym, pos_data in state.get('positions', {}).items():
            self.positions[sym] = Position(
                symbol=pos_data['symbol'],
                side=pos_data['side'],
                size=float(pos_data['size']),
                entry_price=float(pos_data['entry_price']),
                current_price=float(pos_data.get('current_price', pos_data['entry_price'])),
                stop_loss=pos_data.get('stop_loss'),
                take_profit=pos_data.get('take_profit'),
            )

        self.daily_pnl = state.get('daily_pnl', 0.0)
        if 'performance_metrics' in state:
            self.performance_metrics.update(state['performance_metrics'])
        self.error_count = state.get('error_count', 0)

    # ── Emergency stop ─────────────────────────────────────────────────────

    def emergency_stop(self) -> Dict[str, Any]:
        if self.trade_executor is None:
            return {'success': False, 'error': 'Trade executor not available'}
        if not self.positions:
            return {'success': True, 'closed_positions': 0,
                    'message': 'No positions to close'}
        closed = 0
        symbols = list(self.positions.keys())
        for sym in symbols:
            try:
                self.remove_position(sym)
                closed += 1
            except Exception:
                pass
        self.is_running = False
        return {'success': True, 'closed_positions': closed}

    # ── Health check ───────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        critical = ['risk_manager', 'trade_executor']
        optional = ['orderbook_analyzer', 'signal_processor', 'ai_runner']

        components: Dict[str, str] = {}
        unhealthy: List[str] = []
        degraded: List[str] = []

        for name in critical + optional:
            val = getattr(self, name, None)
            if val is not None:
                components[name] = 'HEALTHY'
            elif name in critical:
                components[name] = 'UNHEALTHY'
                unhealthy.append(name)
            else:
                components[name] = 'MISSING'
                degraded.append(name)

        if unhealthy:
            status = 'UNHEALTHY'
        elif degraded:
            status = 'DEGRADED'
        else:
            status = 'HEALTHY'

        return {
            'status': status,
            'components': components,
            'unhealthy_components': unhealthy,
            'degraded_components': degraded,
            'is_running': self.is_running,
            'error_count': self.error_count,
            'last_update': self.last_update.isoformat() if self.last_update else None,
        }

    # ── Analytics report ───────────────────────────────────────────────────

    def generate_analytics_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=lookback_days)
        filtered = []
        for trade in self.trade_history:
            ts_raw = trade.get('timestamp', '')
            try:
                ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else ts_raw
                if ts >= cutoff:
                    filtered.append(trade)
            except Exception:
                filtered.append(trade)

        if not filtered:
            return {
                'total_trades': 0, 'total_pnl': 0.0, 'win_rate': 0.0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'period_days': lookback_days,
                'start_date': cutoff.isoformat(),
                'end_date': datetime.now().isoformat(),
                'message': 'No trades in period',
            }

        pnls = [float(t.get('realized_pnl', 0)) for t in filtered]
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0

        # Basic Sharpe (simplified)
        import statistics
        if len(pnls) > 1:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return {
            'total_trades': len(filtered),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'period_days': lookback_days,
            'start_date': cutoff.isoformat(),
            'end_date': datetime.now().isoformat(),
        }

    # ── Memory usage ───────────────────────────────────────────────────────

    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        import sys
        total = sys.getsizeof(self)
        total += sys.getsizeof(self.positions)
        for pos in self.positions.values():
            total += sys.getsizeof(pos)
        total += sys.getsizeof(self.trade_history)
        for t in self.trade_history:
            total += sys.getsizeof(t)
        return total

    # ── Legacy compat methods ──────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            'is_running': self.is_running,
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'state': self.state.copy(),
        }

    def update_state(self, updates: Dict[str, Any]) -> None:
        self.state.update(updates)
        self.last_update = datetime.now()

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_running:
            return {'status': 'error', 'message': 'Orchestrator is not running'}
        event_type = event.get('tipo_evento', 'UNKNOWN')
        self.update_state({'last_event': event_type,
                           'last_event_time': datetime.now().isoformat()})
        return {
            'status': 'processed',
            'event_type': event_type,
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        uptime = 0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            'uptime_seconds': uptime,
            'symbol': self.symbol,
            'is_running': self.is_running,
            'config': {k: getattr(self.config, k) for k in self.config.__dataclass_fields__},
            'state': self.state.copy(),
        }
