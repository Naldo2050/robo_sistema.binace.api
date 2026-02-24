# flow_analyzer/core.py
"""
FlowAnalyzer - Classe principal.

Analisador de fluxo institucional com:
- Thread safety com locks granulares
- Precisão Decimal para cálculos financeiros
- Agregação rolling O(1)
- Time budget cooperativo
- Observabilidade completa
"""

import logging
import time
from collections import deque
from decimal import Decimal
from threading import Lock, RLock
from typing import Dict, Any, Optional, List, Tuple

from .constants import (
    VERSION,
    DECIMAL_ZERO,
    DEFAULT_NET_FLOW_WINDOWS_MIN,
    DEFAULT_FLOW_TRADES_MAXLEN,
    DEFAULT_FLOW_TIME_BUDGET_MS,
    DEFAULT_FLOW_CACHE_ENABLED,
    DEFAULT_FLOW_LOG_PERF,
    DEFAULT_FLOW_LOG_DETAILED,
    DEFAULT_WHALE_TRADE_THRESHOLD,
    DEFAULT_CVD_RESET_INTERVAL_HOURS,
    DEFAULT_BURST_WINDOW_MS,
    DEFAULT_BURST_COOLDOWN_MS,
    BURST_END_THRESHOLD_RATIO,
    DEFAULT_ORDER_SIZE_BUCKETS,
    DEFAULT_LHM_WINDOW_SIZE,
    DEFAULT_LHM_CLUSTER_THRESHOLD_PCT,
    DEFAULT_LHM_MIN_TRADES_PER_CLUSTER,
    DEFAULT_LHM_UPDATE_INTERVAL_MS,
    DEFAULT_ABSORCAO_DELTA_EPS,
    DEFAULT_ABSORCAO_GUARD_MODE,
    TIMESTAMP_JITTER_TOLERANCE_MS,
    MAX_LATE_TRADE_MS,
    MAX_AGGREGATE_TRADES,
)
from .errors import FlowAnalyzerError, ConfigurationError
from .protocols import IFlowAnalyzer, ITimeProvider, IClockSync
from .utils import (
    lazy_log,
    to_decimal,
    decimal_round,
    ui_safe_round_usd,
    ui_safe_round_btc,
    BoundedErrorCounter,
    get_current_time_ms,
    elapsed_ms,
)
from .validation import (
    TradeSchema,
    validate_ohlc,
    fix_ohlc,
    guard_absorcao,
    FlowAnalyzerConfigValidator,
)
from .aggregates import RollingAggregate
from .metrics import PerformanceMonitor, CircuitBreaker, HealthChecker, calculate_buy_sell_ratios
from .absorption import (
    AbsorptionClassifier,
    AbsorptionAnalyzer,
    AbsorptionConfig,
)


# ==============================================================================
# IMPORTAÇÕES OPCIONAIS
# ==============================================================================

# TimeManager
try:
    from time_manager import TimeManager
    HAS_TIME_MANAGER = True
except ImportError:
    HAS_TIME_MANAGER = False
    TimeManager = None

# ClockSync
try:
    from clock_sync import get_clock_sync
    HAS_CLOCK_SYNC = True
except ImportError:
    HAS_CLOCK_SYNC = False

# LiquidityHeatmap
try:
    from liquidity_heatmap import LiquidityHeatmap
    HAS_HEATMAP = True
except ImportError:
    HAS_HEATMAP = False
    LiquidityHeatmap = None

# Config
try:
    import config as config_module
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    config_module = None


def _get_config(name: str, default: Any) -> Any:
    """Obtém configuração do módulo config ou usa default."""
    if HAS_CONFIG and hasattr(config_module, name):
        return getattr(config_module, name)
    return default


# ==============================================================================
# FLOW ANALYZER
# ==============================================================================

class FlowAnalyzer(IFlowAnalyzer):
    """
    Analisador de fluxo institucional.
    
    Features:
    - CVD (Cumulative Volume Delta) com reset periódico
    - Whale tracking por threshold configurável
    - Burst detection para identificar spikes de volume
    - Sector flow por tamanho de ordem
    - Rolling aggregates O(1) por janela temporal
    - Absorção contextual com OHLC
    - Heatmap de liquidez integrado
    - Métricas de performance e observabilidade
    
    Thread Safety:
    - RLock para estado principal (reentrant)
    - Lock dedicado para heatmap
    - Lock dedicado para contadores
    
    Example:
        >>> from time_manager import TimeManager
        >>> analyzer = FlowAnalyzer(TimeManager())
        >>> analyzer.process_trade({'q': 1.5, 'T': 1234567890, 'p': 50000, 'm': True})
        >>> metrics = analyzer.get_flow_metrics()
        >>> print(f"CVD: {metrics['cvd']}")
    """
    
    def __init__(self, time_manager: Optional[ITimeProvider] = None):
        """
        Inicializa FlowAnalyzer.
        
        Args:
            time_manager: Provedor de tempo (usa TimeManager padrão se None)
        """
        # Time Manager
        if time_manager is not None:
            self.time_manager = time_manager
        elif HAS_TIME_MANAGER and TimeManager is not None:
            self.time_manager = TimeManager()
        else:
            self.time_manager = None
        
        # Clock Sync
        self.clock_sync: Any = None
        if HAS_CLOCK_SYNC:
            try:
                self.clock_sync = get_clock_sync()
            except Exception as e:
                logging.warning(f"⚠️ Erro ao obter Clock Sync: {e}")
        
        # === CONFIGURAÇÕES ===
        self.log_perf = _get_config("FLOW_LOG_PERF", DEFAULT_FLOW_LOG_PERF)
        self.log_detailed = _get_config("FLOW_LOG_DETAILED", DEFAULT_FLOW_LOG_DETAILED)
        self.time_budget_ms = _get_config("FLOW_TIME_BUDGET_MS", DEFAULT_FLOW_TIME_BUDGET_MS)
        
        # === MÉTRICAS ACUMULADAS EM DECIMAL ===
        self.cvd = DECIMAL_ZERO
        self.whale_threshold = Decimal(str(
            _get_config("WHALE_TRADE_THRESHOLD", DEFAULT_WHALE_TRADE_THRESHOLD)
        ))
        self.whale_buy_volume = DECIMAL_ZERO
        self.whale_sell_volume = DECIMAL_ZERO
        self.whale_delta = DECIMAL_ZERO
        
        # Reset automático
        self.last_reset_ms = self._get_synced_timestamp_ms()
        self.reset_interval_ms = int(
            _get_config("CVD_RESET_INTERVAL_HOURS", DEFAULT_CVD_RESET_INTERVAL_HOURS) 
            * 3600 * 1000
        )
        
        # === THREAD SAFETY ===
        self._lock = RLock()
        self._heatmap_lock = Lock()
        self._counters_lock = Lock()
        
        # === BURST DETECTION ===
        self.recent_trades: deque = deque(maxlen=500)
        self.bursts = {"count": 0, "max_burst_volume": 0.0}
        self._in_burst = False
        self._last_burst_end_ms = 0
        
        self.burst_window_ms = int(_get_config("BURST_WINDOW_MS", DEFAULT_BURST_WINDOW_MS))
        self.burst_cooldown_ms = int(_get_config("BURST_COOLDOWN_MS", DEFAULT_BURST_COOLDOWN_MS))
        self.burst_volume_threshold = float(self.whale_threshold)
        
        # === SECTOR FLOW ===
        self._order_buckets = _get_config("ORDER_SIZE_BUCKETS", DEFAULT_ORDER_SIZE_BUCKETS)
        self.sector_flow = {
            name: {"buy": DECIMAL_ZERO, "sell": DECIMAL_ZERO, "delta": DECIMAL_ZERO}
            for name in self._order_buckets
        }
        
        # === LIQUIDITY HEATMAP ===
        self.liquidity_heatmap = None
        if HAS_HEATMAP and LiquidityHeatmap is not None:
            try:
                self.liquidity_heatmap = LiquidityHeatmap(
                    window_size=int(_get_config("LHM_WINDOW_SIZE", DEFAULT_LHM_WINDOW_SIZE)),
                    cluster_threshold_pct=float(_get_config(
                        "LHM_CLUSTER_THRESHOLD_PCT", DEFAULT_LHM_CLUSTER_THRESHOLD_PCT
                    )),
                    min_trades_per_cluster=int(_get_config(
                        "LHM_MIN_TRADES_PER_CLUSTER", DEFAULT_LHM_MIN_TRADES_PER_CLUSTER
                    )),
                    update_interval_ms=int(_get_config(
                        "LHM_UPDATE_INTERVAL_MS", DEFAULT_LHM_UPDATE_INTERVAL_MS
                    )),
                )
            except Exception as e:
                logging.warning(f"⚠️ Erro ao criar LiquidityHeatmap: {e}")
        
        # === FLOW TRACKING ===
        self.net_flow_windows_min: List[int] = list(
            _get_config("NET_FLOW_WINDOWS_MIN", DEFAULT_NET_FLOW_WINDOWS_MIN)
        )
        self.flow_trades_maxlen = _get_config("FLOW_TRADES_MAXLEN", DEFAULT_FLOW_TRADES_MAXLEN)
        self.flow_trades: deque = deque(maxlen=self.flow_trades_maxlen)
        self._max_ts_seen = 0
        self._out_of_order_seen = False
        self._cache_degraded_until_ms = 0
        
        # === CACHE DE AGREGAÇÃO ===
        self._cache_enabled = _get_config("FLOW_CACHE_ENABLED", DEFAULT_FLOW_CACHE_ENABLED)
        self._window_aggregates: Dict[int, RollingAggregate] = {}
        for window in self.net_flow_windows_min:
            max_trades = min(window * 60 * 10, MAX_AGGREGATE_TRADES)
            self._window_aggregates[window] = RollingAggregate(
                window_min=window,
                max_trades=max_trades
            )
        
        # === ABSORÇÃO ===
        self.absorcao_eps = float(_get_config("ABSORCAO_DELTA_EPS", DEFAULT_ABSORCAO_DELTA_EPS))
        self.absorcao_guard_mode = str(
            _get_config("ABSORCAO_GUARD_MODE", DEFAULT_ABSORCAO_GUARD_MODE)
        ).lower()
        
        self._absorption_config = AbsorptionConfig(eps=self.absorcao_eps)
        self._absorption_classifier = AbsorptionClassifier(self._absorption_config)
        self._absorption_analyzer = AbsorptionAnalyzer(self._absorption_config)
        
        # === CONTADORES ===
        self._total_trades_processed = 0
        self._invalid_trades = 0
        self._lock_contentions = 0
        self._last_price: Optional[float] = None
        
        self._whale_delta_corrections = 0
        self._is_buyer_maker_conversions = 0
        self._volume_discrepancies = 0
        
        self.negative_age_count = 0
        self.timestamp_adjustments = 0
        self.late_trades = 0

        # NOVO: Contador de out-of-order (nunca resetado automaticamente)
        self._out_of_order_count = 0

        # Error tracking com limite
        self._error_counts = BoundedErrorCounter(max_keys=100)
        
        # === PERFORMANCE ===
        self.perf_monitor = PerformanceMonitor()
        self._processing_times: deque = deque(maxlen=1000)
        
        # === OTIMIZAÇÕES PARA LATÊNCIA ===
        self._fast_path_enabled = True  # Ativa path otimizado para trades simples
        self._complex_analysis_threshold = 10  # Só faz análise complexa se > 10 trades
        self._last_complex_analysis = 0
        self._complex_analysis_interval = 1000  # ms entre análises complexas
        
        # === CIRCUIT BREAKER ===
        self._circuit_breaker = CircuitBreaker()
        
        # === HEALTH CHECKER ===
        self._health_checker = HealthChecker()
        
        # NOVO: Tracking de tempo de batch para verificação de timestamp
        self._batch_start_time_ms: Optional[int] = None
        
        # === CONFIGURAÇÃO DINÂMICA ===
        self._config_version = 1
        self._config_hash = hash(str(self._get_config_dict()))
        
        # === SHUTDOWN ===
        self._is_shutting_down = False
        
        logging.info(
            "✅ FlowAnalyzer v%s inicializado | "
            "Whale threshold: %.2f BTC | Windows: %s min | "
            "Reset interval: %.1fh | Max trades: %d | "
            "Time budget: %.1fms",
            VERSION,
            float(self.whale_threshold),
            self.net_flow_windows_min,
            self.reset_interval_ms / (3600 * 1000),
            self.flow_trades_maxlen,
            self.time_budget_ms
        )
    
    # ==========================================================================
    # CONFIGURAÇÃO
    # ==========================================================================
    
    def _get_config_dict(self) -> Dict[str, Any]:
        """Retorna dict com configurações atuais."""
        return {
            'whale_threshold': float(self.whale_threshold),
            'absorcao_eps': self.absorcao_eps,
            'absorcao_guard_mode': self.absorcao_guard_mode,
            'net_flow_windows_min': self.net_flow_windows_min,
            'burst_window_ms': self.burst_window_ms,
            'burst_cooldown_ms': self.burst_cooldown_ms,
            'reset_interval_ms': self.reset_interval_ms,
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Atualiza configuração dinamicamente.
        
        Args:
            new_config: Dict com novas configurações
            
        Returns:
            True se atualizado com sucesso
        """
        try:
            # Valida
            validated = FlowAnalyzerConfigValidator.validate_all(new_config)
            
            with self._lock:
                if 'whale_threshold' in validated:
                    self.whale_threshold = Decimal(str(validated['whale_threshold']))
                
                if 'absorcao_eps' in validated:
                    self.absorcao_eps = validated['absorcao_eps']
                    self._absorption_config.eps = self.absorcao_eps
                
                if 'absorcao_guard_mode' in validated:
                    self.absorcao_guard_mode = validated['absorcao_guard_mode']
                
                self._config_hash = hash(str(self._get_config_dict()))
                self._config_version += 1
                
                logging.info(f"✅ Configuração atualizada para v{self._config_version}")
                return True
                
        except ValueError as e:
            logging.error(f"❌ Erro ao atualizar configuração: {e}")
            return False
    
    # ==========================================================================
    # TIMESTAMP / CLOCK SYNC
    # ==========================================================================
    
    def _get_synced_timestamp_ms(self) -> int:
        """Obtém timestamp sincronizado."""
        if self.time_manager:
            try:
                return self.time_manager.now_ms()
            except Exception:
                pass
        
        if self.clock_sync:
            try:
                return self.clock_sync.get_server_time_ms()
            except Exception:
                pass
        
        return get_current_time_ms()
    
    def _adjust_timestamp_if_needed(
        self,
        trade_ts: int,
        reference_ts: int
    ) -> Tuple[int, bool]:
        """
        Ajusta timestamp com política clara.
        
        Returns:
            Tuple (adjusted_ts, was_adjusted)
        """
        # Usa tempo de início do batch se disponível, senão usa tempo atual
        effective_reference = self._batch_start_time_ms or reference_ts
        diff = trade_ts - effective_reference
        
        # Aceita atrasos até 30 segundos (tolerante para processamento em batch)
        if -30000 <= diff <= TIMESTAMP_JITTER_TOLERANCE_MS:
            if diff < 0:
                if diff < -1000:
                    with self._counters_lock:
                        self.late_trades += 1
                return trade_ts, False
            else:
                with self._counters_lock:
                    self.timestamp_adjustments += 1
                return effective_reference, True
        
        # Muito no futuro
        if diff > TIMESTAMP_JITTER_TOLERANCE_MS:
            with self._counters_lock:
                self.timestamp_adjustments += 1
            if lazy_log.should_log("timestamp_future"):
                logging.warning(f"⚠️ Timestamp futuro: diff={diff}ms")
            return effective_reference, True
        
        # Muito no passado (apenas loga, não incrementa contador excessivamente)
        if lazy_log.should_log("timestamp_past"):
            logging.warning(f"⚠️ Trade atrasado no batch: diff={diff}ms")
        return trade_ts, False
    
    def _calculate_age_ms(self, trade_ts: int, reference_ts: int) -> float:
        """Calcula idade do trade."""
        adjusted_ts, _ = self._adjust_timestamp_if_needed(trade_ts, reference_ts)
        age_ms = reference_ts - adjusted_ts
        
        if age_ms < 0:
            with self._counters_lock:
                self.negative_age_count += 1
            return 0.0
        
        return float(age_ms)
    
    # ==========================================================================
    # PROCESSAMENTO DE TRADES
    # ==========================================================================
    
    def start_batch(self) -> None:
        """
        Marca início do processamento do batch.
        
        Deve ser chamado antes de processar um batch de trades
        para que a verificação de timestamp use o tempo de início
        do batch, evitando falsos positivos de trades atrasados.
        """
        self._batch_start_time_ms = self._get_synced_timestamp_ms()
    
    def end_batch(self) -> None:
        """
        Limpa o tracking de tempo de batch.
        
        Deve ser chamado após processar um batch.
        """
        self._batch_start_time_ms = None
    
    def process_trade(self, trade: Dict[str, Any]) -> None:
        """
        Processa trade individual.
        
        Args:
            trade: Dict com 'q', 'T', 'p', 'm' (opcional)
        """
        if self._is_shutting_down:
            return
        
        start_time = time.perf_counter()
        heatmap_payload = None
        
        try:
            self._check_reset()
            
            # Validação
            valid, reason, processed = TradeSchema.validate_and_extract(trade)
            reference_ts = self._get_synced_timestamp_ms()
            
            if not valid:
                with self._lock:
                    self._total_trades_processed += 1
                    self._invalid_trades += 1
                self._error_counts.increment(reason)
                return
            
            # processed é garantido não-None aqui pois valid=True
            assert processed is not None, "processed should not be None when valid=True"
            
            # Extração
            qty = float(processed["qty"])
            ts_raw = int(processed["ts"])
            price = float(processed["price"])
            is_buyer_maker = bool(processed["is_buyer_maker"])
            conversion = bool(processed["conversion"])
            
            # Ajuste de timestamp
            ts, _ = self._adjust_timestamp_if_needed(ts_raw, reference_ts)
            
            # Cálculos em Decimal
            qty_dec = Decimal(str(qty))
            price_dec = Decimal(str(price))
            delta_btc_dec = (-qty_dec) if is_buyer_maker else qty_dec
            delta_usd_dec = delta_btc_dec * price_dec
            side = "sell" if is_buyer_maker else "buy"
            
            with self._lock:
                self._total_trades_processed += 1
                if conversion:
                    self._is_buyer_maker_conversions += 1
                
                # Out-of-order detection
                if ts < self._max_ts_seen:
                    self._out_of_order_seen = True
                    self._out_of_order_count += 1  # NOVO: Incrementa contador
                    max_window_ms = max(self.net_flow_windows_min) * 60_000 if self.net_flow_windows_min else 60_000
                    self._cache_degraded_until_ms = max(
                        self._cache_degraded_until_ms,
                        reference_ts + max_window_ms
                    )
                else:
                    self._max_ts_seen = ts
                
                # CVD
                self.cvd += delta_btc_dec
                
                # Whale
                if qty_dec >= self.whale_threshold:
                    if side == "buy":
                        self.whale_buy_volume += qty_dec
                    else:
                        self.whale_sell_volume += qty_dec
                    self.whale_delta = self.whale_buy_volume - self.whale_sell_volume
                
                # Sector
                sector_name = None
                for name, (minv, maxv) in self._order_buckets.items():
                    if minv <= qty < maxv:
                        sector_name = name
                        break
                
                if sector_name:
                    if side == "buy":
                        self.sector_flow[sector_name]["buy"] += qty_dec
                    else:
                        self.sector_flow[sector_name]["sell"] += qty_dec
                    self.sector_flow[sector_name]["delta"] += delta_btc_dec
                
                # Bursts
                self._update_bursts(ts, qty)
                
                # Trade record
                trade_record = {
                    "ts": ts,
                    "price": price,
                    "qty": qty,
                    "delta_btc": float(delta_btc_dec),
                    "delta_usd": float(delta_usd_dec),
                    "side": side,
                    "sector": sector_name,
                }
                
                self.flow_trades.append(trade_record)
                self._last_price = price
                
                # Prune
                self._prune_flow_history(reference_ts)
                
                # Rolling aggregates
                if self._cache_enabled:
                    degraded = reference_ts < self._cache_degraded_until_ms
                    
                    for _w, agg in self._window_aggregates.items():
                        cutoff = reference_ts - agg.window_ms
                        agg.prune(cutoff)
                        
                        if ts >= cutoff and not degraded:
                            if not (agg.last_update and ts < agg.last_update):
                                agg.add_trade(trade_record, float(self.whale_threshold))
                
                # Heatmap payload
                heatmap_payload = (price, qty, side, ts)
                
        except Exception as e:
            self._error_counts.increment("process_exception")
            self._circuit_breaker.record_failure()
            if lazy_log.should_log("process_trade_error"):
                logging.error("Erro ao processar trade: %s", e, exc_info=True)
        else:
            self._circuit_breaker.record_success()
        finally:
            duration = elapsed_ms(start_time)
            self._processing_times.append(duration)
            self.perf_monitor.record(duration)
            
            # Log apenas se muito lento (threshold ampliado)
            if self.log_perf and duration > 50.0:
                logging.warning("⚠️ process_trade lento: %.2fms", duration)
            
            # Alerta de latência crítica (> 200ms)
            if duration > 200.0:
                logging.error("🚨 LATÊNCIA CRÍTICA: process_trade took %.2fms", duration)
            
            # Heatmap fora do lock
            if heatmap_payload and self.liquidity_heatmap:
                p, q, s, tms = heatmap_payload
                try:
                    with self._heatmap_lock:
                        self.liquidity_heatmap.add_trade(
                            price=p, volume=q, side=s, timestamp_ms=tms
                        )
                except Exception as e:
                    if lazy_log.should_log("heatmap_error"):
                        logging.error("Erro heatmap: %s", e)
    
    def process_batch(self, trades: List[Dict[str, Any]]) -> int:
        """
        Processa batch de trades.
        
        Args:
            trades: Lista de trades
            
        Returns:
            Número de trades processados
        """
        if not trades:
            return 0
        
        # Marca início do batch para verificação correta de timestamp
        self.start_batch()
        
        count = 0
        try:
            for trade in trades:
                if self._is_shutting_down:
                    break
                self.process_trade(trade)
                count += 1
        finally:
            # Limpa tracking de batch
            self.end_batch()
        
        return count
    
    # ==========================================================================
    # HELPERS INTERNOS
    # ==========================================================================
    
    def _prune_flow_history(self, now_ms: int) -> None:
        """Remove trades antigos."""
        if not self.net_flow_windows_min:
            return
        
        max_window = max(self.net_flow_windows_min)
        cutoff_ms = now_ms - max_window * 60 * 1000
        
        if not self._out_of_order_seen:
            while self.flow_trades and self.flow_trades[0]['ts'] < cutoff_ms:
                self.flow_trades.popleft()
        else:
            self.flow_trades = deque(
                (t for t in self.flow_trades if t['ts'] >= cutoff_ms),
                maxlen=self.flow_trades_maxlen
            )
            self._out_of_order_seen = False
    
    def _update_bursts(self, ts_ms: int, qty: float) -> None:
        """Detecta bursts de volume."""
        try:
            self.recent_trades.append((ts_ms, qty))
            
            while self.recent_trades and \
                  (ts_ms - self.recent_trades[0][0] > self.burst_window_ms):
                self.recent_trades.popleft()
            
            burst_volume = sum(q for _, q in self.recent_trades)
            threshold = self.burst_volume_threshold
            
            if not self._in_burst:
                if (burst_volume >= threshold and 
                    (ts_ms - self._last_burst_end_ms) >= self.burst_cooldown_ms):
                    self.bursts["count"] += 1
                    self._in_burst = True
                    self.bursts["max_burst_volume"] = max(
                        self.bursts["max_burst_volume"], burst_volume
                    )
            else:
                self.bursts["max_burst_volume"] = max(
                    self.bursts["max_burst_volume"], burst_volume
                )
                if burst_volume < threshold * BURST_END_THRESHOLD_RATIO:
                    self._in_burst = False
                    self._last_burst_end_ms = ts_ms
                    
        except Exception:
            self._in_burst = False
    
    def _check_reset(self) -> None:
        """Verifica se deve resetar métricas."""
        now_ms = self._get_synced_timestamp_ms()
        if now_ms - self.last_reset_ms > self.reset_interval_ms:
            with self._lock:
                self._reset_metrics()
    
    def _reset_metrics(self) -> None:
        """Reseta todas as métricas acumuladas."""
        self.cvd = DECIMAL_ZERO
        self.whale_buy_volume = DECIMAL_ZERO
        self.whale_sell_volume = DECIMAL_ZERO
        self.whale_delta = DECIMAL_ZERO

        self.recent_trades.clear()
        self.bursts = {"count": 0, "max_burst_volume": 0.0}
        self._in_burst = False
        self._last_burst_end_ms = 0

        self.sector_flow = {
            name: {"buy": DECIMAL_ZERO, "sell": DECIMAL_ZERO, "delta": DECIMAL_ZERO}
            for name in self._order_buckets
        }

        self.flow_trades.clear()
        self._last_price = None

        # NOVO: Reseta contador de OOO
        self._out_of_order_count = 0
        self._out_of_order_seen = False

        if self._cache_enabled:
            for agg in self._window_aggregates.values():
                agg.reset()

        self.last_reset_ms = self._get_synced_timestamp_ms()

        if self.time_manager:
            reset_time = self.time_manager.format_timestamp(self.last_reset_ms)
        else:
            reset_time = str(self.last_reset_ms)

        logging.info(f"🔄 FlowAnalyzer resetado em {reset_time}")
    
    def _check_time_budget(self, start_time: float, operation: str = "") -> bool:
        """Verifica time budget."""
        duration = elapsed_ms(start_time)
        if duration > self.time_budget_ms:
            if lazy_log.should_log(f"time_budget_{operation}"):
                logging.warning(f"⚠️ Time budget excedido em {operation}: {duration:.1f}ms")
            return False
        return True
    
    # ==========================================================================
    # SNAPSHOT
    # ==========================================================================
    
    def _create_snapshot(self, now_ms: Optional[int] = None) -> Dict[str, Any]:
        """Cria snapshot para processamento fora do lock."""
        with self._lock:
            if now_ms is None:
                now_ms = self._get_synced_timestamp_ms()
            
            # Otimização: copia apenas trades necessários
            degraded = now_ms < self._cache_degraded_until_ms
            
            if self._cache_enabled and not degraded:
                window_min = min(self.net_flow_windows_min) if self.net_flow_windows_min else 1
            else:
                window_min = max(self.net_flow_windows_min) if self.net_flow_windows_min else 60
            
            cutoff = now_ms - window_min * 60 * 1000
            flow_trades_copy = [t for t in self.flow_trades if t['ts'] >= cutoff]
            
            snapshot = {
                'flow_trades': flow_trades_copy,
                'cvd': self.cvd,
                'whale_buy_volume': self.whale_buy_volume,
                'whale_sell_volume': self.whale_sell_volume,
                'whale_delta': self.whale_delta,
                'sector_flow': {
                    k: {'buy': v['buy'], 'sell': v['sell'], 'delta': v['delta']}
                    for k, v in self.sector_flow.items()
                },
                'bursts': self.bursts.copy(),
                '_in_burst': self._in_burst,
                '_last_price': self._last_price,
                'last_reset_ms': self.last_reset_ms,
                '_total_trades_processed': self._total_trades_processed,
                '_invalid_trades': self._invalid_trades,
                '_lock_contentions': self._lock_contentions,
                '_whale_delta_corrections': self._whale_delta_corrections,
                '_is_buyer_maker_conversions': self._is_buyer_maker_conversions,
                '_volume_discrepancies': self._volume_discrepancies,
                '_out_of_order_count': self._out_of_order_count,  # NOVO
            }
            
            # Contadores thread-safe
            with self._counters_lock:
                snapshot['negative_age_count'] = self.negative_age_count
                snapshot['timestamp_adjustments'] = self.timestamp_adjustments
                snapshot['late_trades'] = self.late_trades
                snapshot['error_counts'] = self._error_counts.get_all()
            
            # Cache
            if self._cache_enabled:
                wa = {}
                for w, agg in self._window_aggregates.items():
                    cutoff_agg = now_ms - agg.window_ms
                    agg.prune(cutoff_agg)
                    wa[w] = agg.get_metrics(self._last_price or 0.0)
                snapshot["window_aggregates"] = wa
            
            return snapshot
    
    # ==========================================================================
    # MÉTRICAS
    # ==========================================================================
    
    def get_flow_metrics(
        self,
        reference_epoch_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retorna métricas de fluxo completas.
        
        Args:
            reference_epoch_ms: Timestamp de referência (usa atual se None)
            
        Returns:
            Dict com todas as métricas
        """
        start_time = time.perf_counter()
        
        try:
            now_ms = reference_epoch_ms or self._get_synced_timestamp_ms()
            snapshot = self._create_snapshot(now_ms)
            
            # Time index
            if self.time_manager:
                time_index = self.time_manager.build_time_index(
                    now_ms, include_local=True, timespec="milliseconds"
                )
            else:
                time_index = {"timestamp_utc": str(now_ms), "epoch_ms": now_ms}
            
            # Métricas acumuladas
            metrics = self._compute_accumulated_metrics(snapshot, time_index)
            
            # Order flow
            if self._check_time_budget(start_time, "accumulated"):
                order_flow_result = self._compute_order_flow(snapshot, now_ms, start_time)
                if order_flow_result:
                    metrics.update(order_flow_result)
            
            # Participant analysis
            if self._check_time_budget(start_time, "order_flow"):
                participant = self._compute_participant_analysis(snapshot, now_ms)
                if participant:
                    metrics["participant_analysis"] = participant
            
            # Heatmap
            if self._check_time_budget(start_time, "participant"):
                heatmap = self._get_heatmap_data(now_ms)
                if heatmap:
                    metrics["liquidity_heatmap"] = heatmap
            
            # Absorption analysis
            if self._check_time_budget(start_time, "heatmap"):
                absorption = self._compute_absorption_analysis(metrics)
                if absorption:
                    metrics["absorption_analysis"] = absorption
            
            # Data quality
            metrics["data_quality"] = self._compute_data_quality(snapshot, start_time)
            
            # Observability
            metrics["observability"] = self._get_observability_metrics(start_time)
            
            # Invariants
            self._validate_invariants(metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Erro em get_flow_metrics: {e}", exc_info=True)
            return self._get_fallback_metrics(reference_epoch_ms, str(e))
    
    def _get_fallback_metrics(self, ts_ms: Optional[int], error: str) -> Dict[str, Any]:
        """Métricas de fallback em caso de erro."""
        return {
            "cvd": 0.0,
            "whale_delta": 0.0,
            "order_flow": {"flow_imbalance": 0.0},
            "timestamp": str(ts_ms or get_current_time_ms()),
            "data_quality": {"error": error, "is_valid": False},
        }
    
    def _compute_accumulated_metrics(
        self,
        snapshot: Dict[str, Any],
        time_index: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Computa métricas acumuladas."""
        cvd = decimal_round(float(snapshot['cvd']))
        whale_buy = decimal_round(float(snapshot['whale_buy_volume']))
        whale_sell = decimal_round(float(snapshot['whale_sell_volume']))
        whale_delta = decimal_round(float(snapshot['whale_delta']))
        
        # Validação
        expected = whale_buy - whale_sell
        if abs(whale_delta - expected) > 0.001:
            whale_delta = expected
        
        metrics = {
            "cvd": cvd,
            "whale_buy_volume": whale_buy,
            "whale_sell_volume": whale_sell,
            "whale_delta": whale_delta,
            "bursts": snapshot['bursts'],
            "sector_flow": {},
            "timestamp": time_index.get("timestamp_utc", ""),
            "time_index": time_index,
            "metadata": {
                "burst_window_ms": self.burst_window_ms,
                "in_burst": bool(snapshot['_in_burst']),
                "last_reset_ms": snapshot['last_reset_ms'],
                "config_version": self._config_version,
            },
        }
        
        # Sector flow
        for name, data in snapshot['sector_flow'].items():
            buy = decimal_round(float(data['buy']))
            sell = decimal_round(float(data['sell']))
            delta = decimal_round(float(data['delta']))
            
            expected = buy - sell
            if abs(delta - expected) > 0.001:
                delta = expected
            
            metrics["sector_flow"][name] = {"buy": buy, "sell": sell, "delta": delta}
        
        return metrics
    
    def _compute_order_flow(
        self,
        snapshot: Dict[str, Any],
        now_ms: int,
        start_time: float
    ) -> Dict[str, Any]:
         """Computa order flow por janela."""
         if not self.net_flow_windows_min:
             return {}
         
         order_flow = {}
         absorcao_por_janela = {}
         smallest_window = min(self.net_flow_windows_min)
         
         for window_min in self.net_flow_windows_min:
             if not self._check_time_budget(start_time, f"window_{window_min}"):
                 break
             
             window_ms = window_min * 60 * 1000
             start_ms = now_ms - window_ms
             
             # Cache ou cálculo
             degraded = now_ms < self._cache_degraded_until_ms
             
             if self._cache_enabled and not degraded and 'window_aggregates' in snapshot:
                 agg_data = snapshot['window_aggregates'].get(window_min)
                 if agg_data:
                     total_delta_usd = agg_data['sum_delta_usd']
                     total_delta_btc = agg_data['sum_delta_btc']
                     ohlc = agg_data['ohlc']
                 else:
                     total_delta_usd, total_delta_btc, ohlc = self._calc_from_trades(
                         snapshot['flow_trades'], start_ms, now_ms
                     )
             else:
                 total_delta_usd, total_delta_btc, ohlc = self._calc_from_trades(
                     snapshot['flow_trades'], start_ms, now_ms
                 )
             
             w_open, w_high, w_low, w_close = ohlc
             
             if not validate_ohlc(w_open, w_high, w_low, w_close):
                 w_open, w_high, w_low, w_close = fix_ohlc(
                     w_open, w_high, w_low, w_close, snapshot['_last_price']
                 )
             
             # Net flow
             key_net = f"net_flow_{window_min}m"
             order_flow[key_net] = decimal_round(total_delta_usd, 4)
             
             # Absorção
             rotulo = self._absorption_classifier.classify(
                 total_delta_btc, w_open, w_high, w_low, w_close, self.absorcao_eps
             )
             
             order_flow[f"absorcao_{window_min}m"] = rotulo
             absorcao_por_janela[window_min] = rotulo
             
             # Detalhes para menor janela
             if window_min == smallest_window:
                 self._compute_detailed_window(
                     order_flow, snapshot, window_min, start_ms, now_ms,
                     total_delta_btc, start_time
                 )
         
         order_flow["computation_window_min"] = smallest_window
         order_flow["available_windows_min"] = list(self.net_flow_windows_min)
         
         # Buy/Sell Ratio calculation
         if self._check_time_budget(start_time, "buy_sell_ratio"):
             flow_data = {
                 "buy_volume_btc": order_flow.get("buy_volume_btc", 0),
                 "sell_volume_btc": order_flow.get("sell_volume_btc", 0),
                 "total_volume": order_flow.get("total_volume", 0),
                 "sector_flow": {},
             }
             
             # Add net flow for different windows
             for window_min in self.net_flow_windows_min:
                 key_net = f"net_flow_{window_min}m"
                 flow_data[key_net] = order_flow.get(key_net, 0)
             
             # Add sector flow
             sector_flow = {}
             for name, data in snapshot['sector_flow'].items():
                 sector_flow[name] = {
                     "buy": float(data['buy']),
                     "sell": float(data['sell']),
                     "delta": float(data['delta'])
                 }
             flow_data["sector_flow"] = sector_flow
             
             # Calculate and add ratio
             ratio_result = calculate_buy_sell_ratios(flow_data)
             order_flow["buy_sell_ratio"] = ratio_result
         
         return {
             "order_flow": order_flow,
             "tipo_absorcao": absorcao_por_janela.get(smallest_window, "Neutra"),
         }
    
    def _calc_from_trades(
        self,
        trades: List[Dict[str, Any]],
        start_ms: int,
        end_ms: int
    ) -> Tuple[float, float, Tuple[float, float, float, float]]:
        """Calcula delta e OHLC de trades."""
        relevant = [t for t in trades if start_ms <= t['ts'] <= end_ms]
        
        if not relevant:
            return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        
        prices = [t['price'] for t in relevant]
        ohlc = (prices[0], max(prices), min(prices), prices[-1])
        delta_usd = sum(t['delta_usd'] for t in relevant)
        delta_btc = sum(t['delta_btc'] for t in relevant)
        
        return delta_usd, delta_btc, ohlc
    
    def _compute_detailed_window(
        self,
        order_flow: Dict[str, Any],
        snapshot: Dict[str, Any],
        window_min: int,
        start_ms: int,
        now_ms: int,
        total_delta_btc: float,
        start_time: float
    ) -> None:
        """Computa métricas detalhadas para uma janela."""
        relevant = [
            t for t in snapshot['flow_trades']
            if start_ms <= t['ts'] <= now_ms
        ]
        
        if not relevant:
            return
        
        # Cálculos
        total_buy_usd = Decimal('0')
        total_sell_usd = Decimal('0')
        total_buy_btc = Decimal('0')
        total_sell_btc = Decimal('0')
        
        for t in relevant:
            qty = Decimal(str(t['qty']))
            price = Decimal(str(t['price']))
            
            if t['side'] == 'buy':
                total_buy_usd += qty * price
                total_buy_btc += qty
            else:
                total_sell_usd += qty * price
                total_sell_btc += qty
        
        # USD
        buy_usd, sell_usd, total_usd, ui_ok, tol, gap = ui_safe_round_usd(
            total_buy_usd, total_sell_usd
        )
        
        order_flow["buy_volume"] = buy_usd
        order_flow["sell_volume"] = sell_usd
        order_flow["total_volume"] = total_usd
        order_flow["ui_sum_ok"] = ui_ok
        
        # BTC
        buy_btc, sell_btc, total_btc, diff = ui_safe_round_btc(total_buy_btc, total_sell_btc)
        
        order_flow["buy_volume_btc"] = buy_btc
        order_flow["sell_volume_btc"] = sell_btc
        order_flow["total_volume_btc"] = total_btc
        
        # Whale window
        whale_buy = sum(t['qty'] for t in relevant 
                       if t['qty'] >= float(self.whale_threshold) and t['side'] == 'buy')
        whale_sell = sum(t['qty'] for t in relevant 
                        if t['qty'] >= float(self.whale_threshold) and t['side'] == 'sell')
        
        order_flow["whale_buy_volume_window"] = decimal_round(whale_buy)
        order_flow["whale_sell_volume_window"] = decimal_round(whale_sell)
        order_flow["whale_delta_window"] = decimal_round(whale_buy - whale_sell)
        
        # Flow imbalance
        total_vol = float(total_buy_usd + total_sell_usd)
        if total_vol > 0:
            imbalance = float(total_buy_usd - total_sell_usd) / total_vol
            order_flow["flow_imbalance"] = decimal_round(imbalance, 4)
            
            buy_pct = float(total_buy_usd) / total_vol * 100
            sell_pct = float(total_sell_usd) / total_vol * 100
            order_flow["aggressive_buy_pct"] = decimal_round(buy_pct, 2)
            order_flow["aggressive_sell_pct"] = decimal_round(sell_pct, 2)
    
    def _compute_participant_analysis(
        self,
        snapshot: Dict[str, Any],
        now_ms: int
    ) -> Dict[str, Any]:
        """Computa participant analysis."""
        if not self.net_flow_windows_min:
            return {}
        
        window = min(self.net_flow_windows_min)
        start_ms = now_ms - window * 60 * 1000
        
        trades = [t for t in snapshot['flow_trades'] if start_ms <= t['ts'] <= now_ms]
        
        if not trades:
            return {}
        
        total_qty = sum(t['qty'] for t in trades)
        duration_sec = window * 60
        
        result = {}
        
        for sector in self._order_buckets.keys():
            sector_trades = [t for t in trades if t.get('sector') == sector]
            if not sector_trades:
                continue
            
            sector_qty = sum(t['qty'] for t in sector_trades)
            buy_qty = sum(t['qty'] for t in sector_trades if t['delta_btc'] > 0)
            sell_qty = sum(t['qty'] for t in sector_trades if t['delta_btc'] < 0)
            count = len(sector_trades)
            
            imbalance = (buy_qty - sell_qty) / max(sector_qty, 0.001)
            participation = sector_qty / max(total_qty, 0.001)
            frequency = count / max(duration_sec, 1)
            
            strength = 0.4 * abs(imbalance) + 0.4 * participation + 0.2 * min(1.0, frequency / 10)
            
            result[sector] = {
                "volume_pct": decimal_round(participation * 100, 2),
                "direction": "BUY" if imbalance > 0.1 else ("SELL" if imbalance < -0.1 else "NEUTRAL"),
                "sentiment": "BULLISH" if imbalance > 0.1 else ("BEARISH" if imbalance < -0.1 else "NEUTRAL"),
                "composite_score": decimal_round(strength * (1 if imbalance > 0 else -1), 3),
                "imbalance": decimal_round(imbalance, 3),
            }
        
        return result
    
    def _get_heatmap_data(self, now_ms: int) -> Dict[str, Any]:
        """Obtém dados do heatmap."""
        if not self.liquidity_heatmap:
            return {}
        
        try:
            with self._heatmap_lock:
                clusters = self.liquidity_heatmap.get_clusters(top_n=5)
                supports, resistances = self.liquidity_heatmap.get_support_resistance()
            
            return {
                "clusters": clusters or [],
                "supports": sorted(set(supports)) if supports else [],
                "resistances": sorted(set(resistances)) if resistances else [],
                "clusters_count": len(clusters) if clusters else 0,
            }
        except Exception as e:
            if lazy_log.should_log("heatmap_get_error"):
                logging.debug(f"Erro ao obter heatmap: {e}")
            return {}
    
    def _compute_absorption_analysis(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Computa análise de absorção avançada."""
        of = metrics.get("order_flow", {})
        window_min = of.get("computation_window_min")
        
        if window_min is None:
            return None
        
        net_key = f"net_flow_{window_min}m"
        delta_usd = float(of.get(net_key, 0) or 0)
        total_usd = float(of.get("total_volume", 0) or 0)
        flow_imb = float(of.get("flow_imbalance", 0) or 0)
        buy_pct = float(of.get("aggressive_buy_pct", 0) or 0)
        sell_pct = float(of.get("aggressive_sell_pct", 0) or 0)
        
        analysis = self._absorption_analyzer.analyze(
            delta_usd=delta_usd,
            total_volume_usd=total_usd,
            flow_imbalance=flow_imb,
            buy_pct=buy_pct,
            sell_pct=sell_pct,
            absorption_label=metrics.get("tipo_absorcao", "Neutra"),
            window_min=window_min,
        )
        
        if analysis:
            return {"current_absorption": analysis.to_dict()}
        return None
    
    def _compute_data_quality(
        self,
        snapshot: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Computa qualidade de dados."""
        total = snapshot['_total_trades_processed']
        invalid = snapshot['_invalid_trades']
        
        valid_rate = 100 * (1 - invalid / total) if total > 0 else 100.0
        
        return {
            "total_trades_processed": total,
            "invalid_trades": invalid,
            "valid_rate_pct": decimal_round(valid_rate, 2),
            "flow_trades_count": len(snapshot['flow_trades']),
            "error_counts": snapshot.get('error_counts', {}),
            "processing_time_ms": elapsed_ms(start_time),
        }
    
    def _get_observability_metrics(self, start_time: float) -> Dict[str, Any]:
        """Retorna métricas de observabilidade."""
        perf_stats = self.perf_monitor.get_stats()
        
        return {
            "processing_times_ms": perf_stats,
            "memory": {
                "flow_trades_size": len(self.flow_trades),
                "flow_trades_capacity": self.flow_trades_maxlen,
            },
            "circuit_breaker": self._circuit_breaker.get_stats(),
        }
    
    def _validate_invariants(self, metrics: Dict[str, Any]) -> None:
        """Valida invariantes matemáticos."""
        try:
            ok = True
            tol = 1e-6
            
            w_buy = float(metrics.get("whale_buy_volume", 0))
            w_sell = float(metrics.get("whale_sell_volume", 0))
            w_delta = float(metrics.get("whale_delta", 0))
            
            if abs((w_buy - w_sell) - w_delta) > tol:
                ok = False
            
            metrics["invariants_ok"] = ok
        except Exception:
            metrics["invariants_ok"] = False
    
    # ==========================================================================
    # VOLATILIDADE
    # ==========================================================================
    
    def update_volatility_context(
        self,
        atr_price: Optional[float] = None,
        price_volatility: Optional[float] = None
    ) -> None:
        """Atualiza contexto de volatilidade."""
        self._absorption_classifier.update_volatility(
            atr=atr_price, price_volatility=price_volatility
        )
    
    # ==========================================================================
    # ESTATÍSTICAS E SAÚDE
    # ==========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas completas."""
        with self._lock:
            total = self._total_trades_processed
            invalid = self._invalid_trades
            flow_count = len(self.flow_trades)
            cvd = self.cvd
            whale_delta = self.whale_delta
            
            with self._counters_lock:
                errors = self._error_counts.get_all()
        
        valid_rate = 100 * (1 - invalid / total) if total > 0 else 100.0
        
        return {
            "total_trades_processed": total,
            "invalid_trades": invalid,
            "valid_rate_pct": decimal_round(valid_rate, 2),
            "flow_trades_count": flow_count,
            "flow_trades_capacity": self.flow_trades_maxlen,
            "cvd": decimal_round(float(cvd)),
            "whale_delta": decimal_round(float(whale_delta)),
            "error_counts": errors,
            "config_version": self._config_version,
            "processing_performance_ms": self.perf_monitor.get_stats(),
            "circuit_breaker": self._circuit_breaker.get_stats(),
            "out_of_order_count": self._out_of_order_count,  # NOVO
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verificação de saúde."""
        stats = self.get_stats()
        perf_stats = stats.get("processing_performance_ms", {})
        
        formatter = None
        if self.time_manager:
            formatter = self.time_manager.format_timestamp
        
        status = self._health_checker.check(
            stats=stats,
            perf_stats=perf_stats,
            circuit_breaker=self._circuit_breaker,
            timestamp_formatter=formatter,
        )
        
        result = status.to_dict()
        result["stats"] = stats
        return result
    
    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================
    
    def shutdown(self) -> None:
        """Encerra graciosamente."""
        self._is_shutting_down = True
        logging.info("🛑 FlowAnalyzer shutdown iniciado")
    
    # ==========================================================================
    # COMPATIBILIDADE
    # ==========================================================================
    
    @staticmethod
    def map_absorcao_label(aggression_side: str) -> str:
        """Mapeia lado de agressão para rótulo."""
        return AbsorptionClassifier.map_aggression_to_label(aggression_side)
    
    @staticmethod
    def classificar_absorcao_por_delta(delta: float, eps: float = 1.0) -> str:
        """Classificador simples por delta."""
        from .absorption import classify_absorption_simple
        return classify_absorption_simple(delta, eps)
    
    def classificar_absorcao_contextual(
        self,
        delta_btc: float,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        eps: float = 1.0,
        atr: Optional[float] = None,
        price_volatility: Optional[float] = None,
    ) -> str:
        """Classificação contextual de absorção."""
        if atr or price_volatility:
            self._absorption_classifier.update_volatility(atr, price_volatility)
        return self._absorption_classifier.classify(
            delta_btc, open_p, high_p, low_p, close_p, eps
        )
