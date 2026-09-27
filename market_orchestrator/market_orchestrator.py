# market_orchestrator/market_orchestrator.py
# Otimização de eventos (auto-adicionado)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# -*- coding: utf-8 -*-
"""
Orquestrador de mercado (WebSocket + janelas + DataPipeline + IA) v2.3.2

Versão refatorada em módulos, preservando o comportamento do arquivo
original market_orchestrator.py. Toda a lógica continua igual, apenas
algumas partes foram extraídas para submódulos.
"""

import json
import time
import logging
import threading

import pandas as pd
from datetime import datetime, timezone

from collections import deque
import signal
import atexit
import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional, List, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import config
from trading.trade_buffer import AsyncTradeBuffer, BufferStatus

# ====== Contador Prometheus: trades com timestamp corrigido (clamp T<last_T) ======
# Exposto no endpoint /metrics (porta 8000) via REGISTRY padrão do prometheus_client
# (monitoring/pipeline_health.py usa generate_latest(REGISTRY)).
try:
    from prometheus_client import Counter as _PromCounter
    TRADES_CORRECTED_TOTAL = _PromCounter(
        "orchestrator_trades_timestamp_corrected_total",
        "Total de trades com timestamp corrigido (clamp T<last_T) no on_message",
    )
    PROMETHEUS_OK = True
except Exception:
    TRADES_CORRECTED_TOTAL = None
    PROMETHEUS_OK = False

# ====== Clock Sync (opcional) ======
# REMOVIDO: ClockSync duplicado - TimeManager já faz sincronização robusta com Binance
# Isso evita conflitos entre dois sistemas de sincronização de tempo

# ====== Utilitários de formatação ======
from common.format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific,
)

# ====== Módulos internos originais ======
from data_processing.data_handler import (
    NY_TZ,
)
from events.event_memory import (
    obter_memoria_eventos,
    adicionar_memoria_evento,
    calcular_probabilidade_historica,
    avaliar_outcomes_pendentes,
)

# Similarity search para eventos passados (memória longa)
try:
    from events.event_similarity import EventSimilaritySearch
    _similarity_search = EventSimilaritySearch()
    _SIMILARITY_OK = True
except ImportError:
    _similarity_search = None
    _SIMILARITY_OK = False

# Outcome tracker para confiança estatística
try:
    #     # from trading.outcome_tracker import OutcomeTracker  # Not used
    _OUTCOME_OK = True
except ImportError:
    _OUTCOME_OK = False
from orderbook_analyzer import OrderBookAnalyzer
from events.event_saver import EventSaver
from fetchers.context_collector import ContextCollector
from fetchers.onchain_updater import OnchainUpdater
from market_analysis.cross_asset_updater import CrossAssetUpdater

# ====== CFTC COT Updater (P6 — opcional, flag-gated, fora do hot path) ======
try:
    from fetchers.cftc_cot_updater import CftcCotUpdater
except Exception:
    CftcCotUpdater = None
from flow_analyzer import FlowAnalyzer
from market_analysis.levels_registry import LevelRegistry
from data_processing.data_validator import validator

from monitoring.time_manager import TimeManager
from monitoring.health_monitor import HealthMonitor
from events.event_bus import EventBus
from data_pipeline import DataPipeline
from data_processing.feature_store import FeatureStore

# ====== Structured Logging & Tracing ======
from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper

# ====== Alert engine (opcional) ======
try:
    from trading.alert_engine import generate_alerts
except Exception:
    generate_alerts = None

try:
    import support_resistance as _sr
    detect_support_resistance = getattr(_sr, "detect_support_resistance", None)
    defense_zones = getattr(_sr, "defense_zones", None)
except Exception:
    detect_support_resistance = None
    defense_zones = None

recognize_patterns = None  # Removed empty try block

# ====== Enriquecedor institucional (Onda 1 + 2) ======
try:
    from institutional.enricher import enrich_signal as _institutional_enrich
    _INSTITUTIONAL_ENRICHER_OK = True
except Exception as _ie_err:
    _institutional_enrich = None
    _INSTITUTIONAL_ENRICHER_OK = False
    logging.warning(f"institutional_enricher indisponível: {_ie_err}")

# ====== Submódulos refatorados ======
from .utils.logging_utils import configure_dedup_logs
from .connection.robust_connection import RobustConnectionManager, RateLimiter
from .flow.trade_flow_analyzer import TradeFlowAnalyzer
from .windows import WindowProcessor, process_window
from .signals.signal_processor import process_signals

if TYPE_CHECKING:
    from common.ai_protocols import (
        AIAnalyzerProtocol,
        FeatureCalculatorProtocol,
        PredictorProtocol,
    )
    from .ai.ai_runner import AIRunner

# ====== Institutional Analytics Engine ======
try:
    from .analysis.institutional_analytics import InstitutionalAnalyticsEngine
except Exception:
    InstitutionalAnalyticsEngine = None

# Ativa filtro anti-eco global
configure_dedup_logs()

from common.signal_direction import (
    infer_signal_side,
    get_directional_confidence,
    BULLISH_RESULTS,
)


def parse_trade_message(msg: Any) -> Optional[Dict[str, Any]]:
    """
    Parser normalizador de mensagens trade / aggTrade da Binance WebSocket.
    Suporta aggTrade (Futures) e trade (Spot legado).
    Retorna dict com (price, qty, ts_ms, is_buyer_maker, source, trade_id)
    ou None se a mensagem for inválida.
    """
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except Exception:
            return None
    if not isinstance(msg, dict):
        return None

    raw_trade = msg.get("data", msg)
    if not isinstance(raw_trade, dict):
        return None

    if raw_trade.get("e") == "aggTrade" or "f" in raw_trade:
        trade_id = raw_trade.get("a")        # agg trade id (futures)
        source = "fut_agg"
    else:
        trade_id = raw_trade.get("t")        # trade id (spot legado)
        source = "spot_trade"

    p = raw_trade.get("p") or raw_trade.get("P") or raw_trade.get("price")
    q = raw_trade.get("q") or raw_trade.get("Q") or raw_trade.get("quantity")
    T = raw_trade.get("T")
    m = raw_trade.get("m")

    # Fallback para mensagens de kline
    if (p is None or q is None or T is None) and isinstance(raw_trade.get("k"), dict):
        k = raw_trade["k"]
        if p is None:
            p = k.get("c")
        if q is None:
            q = k.get("v")
        if T is None:
            T = k.get("T")

    if p is None or q is None or T is None:
        return None

    try:
        price = float(p)
        qty = float(q)
        ts_ms = int(T)
    except (TypeError, ValueError):
        return None

    if price <= 0 or qty <= 0 or ts_ms <= 0:
        return None

    if trade_id is not None:
        try:
            trade_id = int(trade_id)
        except (TypeError, ValueError):
            pass

    is_buyer_maker = bool(m) if m is not None else None

    return {
        "price": price,
        "qty": qty,
        "ts_ms": ts_ms,
        "is_buyer_maker": is_buyer_maker,
        "source": source,
        "trade_id": trade_id,
        "p": price,
        "q": qty,
        "T": ts_ms,
        "m": is_buyer_maker if is_buyer_maker is not None else False,
    }


class EnhancedMarketBot:
    """Bot de análise de mercado com IA integrada (v2.3.2)."""

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        window_size_minutes: int,
        vol_factor_exh: float,
        history_size: int,
        delta_std_dev_factor: float,
        context_sma_period: int,
        liquidity_flow_alert_percentage: float,
        wall_std_dev_factor: float,
        dump_raw_trades: Optional[Union[str, bool, Path]] = None,
        shadow_runtime: Optional[Any] = None,
    ) -> None:
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False

        # Persistência opcional de trades brutos (dump JSONL para Item 8 / validação p99 whale)
        self.dump_raw_trades_path: Optional[Path] = None
        self._raw_trades_file = None
        self._raw_trades_lock = threading.Lock()
        if dump_raw_trades:
            if isinstance(dump_raw_trades, (str, Path)):
                self.dump_raw_trades_path = Path(dump_raw_trades)
            else:
                self.dump_raw_trades_path = Path("dados/trades_collect_2h.jsonl")
            try:
                self.dump_raw_trades_path.parent.mkdir(parents=True, exist_ok=True)
                self._raw_trades_file = open(self.dump_raw_trades_path, "a", encoding="utf-8", buffering=1)
                logging.info(f"💾 Persistência de trades brutos habilitada: {self.dump_raw_trades_path}")
            except Exception as e:
                logging.error(f"❌ Falha ao abrir arquivo de dump de trades: {e}")
                self._raw_trades_file = None

        # Locks e sinalização de shutdown/cleanup
        self._cleanup_lock = threading.Lock()
        self._cleanup_started = threading.Event()
        self._ai_init_lock = threading.Lock()
        self._is_shutdown = False
        self._shutdown_async_lock: Optional[asyncio.Lock] = None

        self.warming_up = False
        self._warmup_lock = threading.Lock()
        self.warmup_windows_remaining = 0
        self.warmup_windows_required = getattr(config, "WARMUP_WINDOWS", 3)

        # Buffer assíncrono de trades com backpressure
        self.trades_buffer = AsyncTradeBuffer(
            max_size=getattr(config, "TRADES_BUFFER_SIZE", 5000),
            backpressure_threshold=getattr(
                config, "TRADES_BUFFER_BACKPRESSURE", 0.6
            ),
            processing_batch_size=getattr(
                config, "TRADES_BUFFER_BATCH_SIZE", 200
            ),
            processing_interval_ms=getattr(
                config, "TRADES_BUFFER_PROCESSING_INTERVAL_MS", 5
            ),
            max_processing_time_ms=getattr(
                config, "TRADES_BUFFER_MAX_PROCESSING_MS", 500.0
            ),
            warning_callback=self._on_buffer_warning,
            heartbeat_callback=lambda module: (
                self.health_monitor.heartbeat(module)
                if getattr(self, "health_monitor", None) is not None
                else None
            )
        )
        self.min_trades_for_pipeline = getattr(
            config, "MIN_TRADES_FOR_PIPELINE", 10
        )

        # FASE B: bot possui exatamente 1 OnchainUpdater. O refresh onchain
        # roda fora do hot path; a janela só lê snapshot (DI explícita).
        self.onchain_updater = OnchainUpdater()

        # E3-B: bot possui exatamente 1 CrossAssetUpdater (mesmo padrão).
        self.cross_asset_updater = CrossAssetUpdater()

        # P6: CftcCotUpdater existe apenas com ENABLE_CFTC_COT_CONTEXT=1.
        # Ausência CFTC nunca afeta Binance positioning (fontes independentes).
        self.cftc_cot_updater = None
        try:
            if CftcCotUpdater is not None and bool(
                getattr(config, "ENABLE_CFTC_COT_CONTEXT", False)
            ):
                self.cftc_cot_updater = CftcCotUpdater(symbols=[self.symbol])
                logging.info("✅ CftcCotUpdater inicializado (flag P6 ativa)")
        except Exception as e:
            logging.warning(f"⚠️ CftcCotUpdater indisponível (não-crítico): {e}")
            self.cftc_cot_updater = None

        self._loop = None
        self._initialized = False

        # REMOVIDO: ClockSync duplicado - TimeManager é suficiente

        self.time_manager = TimeManager()

        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)
        self.feature_store = FeatureStore(base_dir="features")
        self.levels = LevelRegistry(self.symbol)

        # ====== Institutional Analytics Engine ======
        self.institutional_analytics = None
        if InstitutionalAnalyticsEngine is not None:
            try:
                self.institutional_analytics = InstitutionalAnalyticsEngine(
                    symbol=self.symbol
                )
                logging.info("✅ InstitutionalAnalyticsEngine inicializado")
            except Exception as e:
                logging.warning(f"⚠️ InstitutionalAnalyticsEngine falhou: {e}")

        self.health_monitor.heartbeat("main")

        self.trade_flow_analyzer = TradeFlowAnalyzer(
            vol_factor_exh, tz_output=self.ny_tz
        )

        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=self.symbol,
            liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
            wall_std_dev_factor=wall_std_dev_factor,
            time_manager=self.time_manager,
            cache_ttl_seconds=getattr(config, "ORDERBOOK_CACHE_TTL", 30.0),
            max_stale_seconds=getattr(config, "ORDERBOOK_MAX_STALE", 300.0),
            rate_limit_threshold=getattr(
                config, "ORDERBOOK_MAX_REQUESTS_PER_MIN", 5
            ),
        )

        # Estado do orderbook
        self.last_valid_orderbook: Optional[Dict[str, Any]] = None
        self.last_valid_orderbook_time: float = 0.0
        self.orderbook_fetch_failures = 0
        self.orderbook_emergency_mode = getattr(
            config, "ORDERBOOK_EMERGENCY_MODE", True
        )

        self._orderbook_refresh_lock = threading.Lock()
        self._orderbook_refresh_thread: Optional[threading.Thread] = None
        self._orderbook_background_refresh = getattr(
            config, "ORDERBOOK_BACKGROUND_REFRESH", True
        )
        self._orderbook_bg_min_interval = float(
            getattr(config, "ORDERBOOK_BG_MIN_INTERVAL", 5.0)
        )
        self._last_async_ob_refresh = 0.0

        # Executor para tarefas assíncronas auxiliares
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="orderbook_",
        )

        # Loop asyncio dedicado para o OrderBookAnalyzer
        self._async_loop = asyncio.new_event_loop()
        self._async_loop_thread = threading.Thread(
            target=self._run_async_loop,
            name="orderbook_async_loop",
            daemon=True,
        )
        self._async_loop_thread.start()

        self.last_valid_vp: Optional[Dict[str, Any]] = None
        self.last_valid_vp_time: float = 0.0

        # FASE E3-A: áudio opt-in via SOUND_ALERT (default OFF em server).
        self.event_saver = EventSaver(
            sound_alert=None, health_monitor=self.health_monitor
        )
        self.pattern_ohlc_history = deque(maxlen=200)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)

        # ===== IA =====
        self.ai_analyzer: Optional["AIAnalyzerProtocol"] = None
        self.ai_runner: Optional["AIRunner"] = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ml_engine: Optional["PredictorProtocol"] = None
        self.feature_calc: Optional["FeatureCalculatorProtocol"] = None
        self.ai_thread_pool: List[threading.Thread] = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self._ai_pool_lock = threading.Lock()
        self.ai_rate_limiter = RateLimiter(max_calls=10, period_seconds=60)

        # Inicializa IA em background (mesma lógica do original)
        self._initialize_ai_async()
        
        # Guardar referência da task do buffer
        self._buffer_task = None

        # ====== WindowProcessor ======
        self.window_processor = None
        self._window_task = None
        self._window_watchdog_task = None
        self._trade_count = 0

        # ====== Estado de Trades e Janelas ======
        self._last_trade_ts_ms = None  # ?ltimo timestamp de trade (monotonicity check)
        self._last_price = None  # ?ltimo pre?o (refer?ncia para agressor)
        self.window_count = 0  # Contador de janelas processadas
        self.window_data = []  # Trades na janela atual
        self.window_end_ms = None  # Timestamp de fechamento da janela

        # ====== Trades fora de ordem (clamp T<last_T) ======
        self._ooo_trades_count = 0
        self._last_ooo_log_ts = 0.0
        self._ooo_log_interval_sec = float(
            getattr(config, "OOO_LOG_INTERVAL_SEC", 60)
        )


        # ====== Hist?ricos de Volume e Delta ======
        self.delta_history = deque(maxlen=100)  # Hist?rico de deltas
        self.volume_history = deque(maxlen=100)  # Hist?rico de volumes
        self.close_price_history = deque(maxlen=100)  # Hist?rico de pre?os de fechamento
        self.volatility_history = deque(maxlen=100)  # Hist?rico de volatilidade
        self._history_lock = threading.Lock()  # Lock para acesso thread-safe aos hist?ricos
        self.delta_std_dev_factor = delta_std_dev_factor

        self._last_ai_analysis_ts = 0.0
        self._ai_min_interval_sec = getattr(
            config, "AI_MIN_INTERVAL_SEC", 60
        )

        self._sent_triggers = set()
        self._last_alert_ts = {}

        try:
            self._alert_cooldown_sec = getattr(
                config, "ALERT_COOLDOWN_SEC", 30
            )
        except Exception:
            self._alert_cooldown_sec = 30

        self._register_cleanup_handlers()

        # Logging estruturado e tracing para o bot
        self.slog = StructuredLogger("enhanced_market_bot", self.symbol)
        # ====== RobustConnectionManager ======
        self.connection_manager = RobustConnectionManager(
            stream_url=stream_url,
            symbol=symbol,
            max_reconnect_attempts=getattr(config, 'WS_MAX_RECONNECT_ATTEMPTS', 15),
            initial_delay=getattr(config, 'WS_INITIAL_RECONNECT_DELAY', 1.0),
            max_delay=getattr(config, 'WS_MAX_RECONNECT_DELAY', 60.0),
            backoff_factor=getattr(config, 'WS_BACKOFF_FACTOR', 1.5),
        )
        # Configure callbacks (methods exist in the class)
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_reconnect=self._on_reconnect,
        )
        # PRODUÇÃO: conectar o callback de recebimento de WS (hook já existia,
        # nunca conectado) ao HealthMonitor — sinal "ws" do health check do container.
        self.connection_manager.set_heartbeat_cb(
            lambda: self.health_monitor.heartbeat("ws")
        )

        self.tracer = TracerWrapper(
            service_name="enhanced_market_bot",
            component="orchestrator",
            symbol=self.symbol,
        )

        # ====== Paper Trading Shadow Runtime (Gate C3-C-B3-B) ======
        self.shadow_runtime: Optional[Any] = shadow_runtime
        self.paper_shadow_status: str = "DISABLED"
        self.paper_shadow_error: Optional[str] = None
        self.paper_shadow_hook_errors: int = 0
        self._shadow_subscribed: bool = False
        self._shadow_init_attempted: bool = False

        if shadow_runtime is not None:
            self.paper_shadow_status = "RUNNING"
        else:
            self._init_shadow_runtime()

    def _init_shadow_runtime(self) -> None:
        """Inicializa o ShadowPaperRuntime via factory hermética se habilitado."""
        if self._shadow_init_attempted:
            return
        self._shadow_init_attempted = True
        try:
            from paper_trading.factory import create_shadow_runtime

            res = create_shadow_runtime()
            self.paper_shadow_status = res.status
            self.paper_shadow_error = res.reason
            self.shadow_runtime = res.runtime
            if res.status == "RUNNING":
                logging.info("✅ ShadowPaperRuntime inicializado com sucesso (status=RUNNING)")
            elif res.status == "FAILED":
                logging.error(f"❌ Falha ao inicializar ShadowPaperRuntime: {res.reason}")
        except Exception as exc:
            self.paper_shadow_status = "FAILED"
            self.paper_shadow_error = str(exc)
            self.shadow_runtime = None
            logging.error(f"❌ Exceção inesperada na inicialização do ShadowPaperRuntime: {exc}")

    def _handle_shadow_runtime_error(self, exc: Exception) -> None:
        """Trata falha externa no hook de tick do shadow runtime sem stack trace por tick."""
        self.paper_shadow_hook_errors += 1
        if self.paper_shadow_status != "FAILED":
            self.paper_shadow_status = "FAILED"
            self.paper_shadow_error = str(exc)
            logging.error(f"❌ ShadowPaperRuntime hook error: {exc}")

    # ========================================
    # HANDLER DE RECONEXÃO
    # ========================================
    def _on_reconnect(self) -> None:
        logging.warning(
            "🔄 RECONEXÃO DETECTADA - Iniciando período de aquecimento..."
        )

        with self._warmup_lock:
            self.warming_up = True
            self.warmup_windows_remaining = self.warmup_windows_required
            self.window_data = []
            self.window_end_ms = None
            self._sent_triggers.clear()

        if getattr(self, "health_monitor", None) is not None:
            try:
                self.health_monitor.set_recovering(True, reason="ws_reconnect")
            except Exception as e:
                logging.debug(f"Erro ao ativar recovering no HealthMonitor: {e}")

        logging.info(
            f"⏳ Aguardando {self.warmup_windows_required} janelas "
            f"para estabilizar dados..."
        )

    # ========================================
    # CALLBACK DO BUFFER DE TRADES
    # ========================================
    def _on_buffer_warning(self, status: BufferStatus, buffer_size: int) -> None:
        """Callback para alertas do buffer de trades."""
        if status == BufferStatus.CRITICAL:
            logging.warning(
                f"🚨 Buffer de trades CRÍTICO: {buffer_size} trades no buffer"
            )
            self.health_monitor.heartbeat("buffer_critical")
        elif status == BufferStatus.OVERFLOW:
            logging.error(
                f"💀 Buffer de trades em OVERFLOW: {buffer_size} trades"
            )
            self.health_monitor.heartbeat("buffer_overflow")
        elif status == BufferStatus.WARNING:
            logging.info(
                f"⚠️ Buffer de trades.warning: {buffer_size} trades"
            )
    
    # ========================================
    # GERENCIAMENTO DE THREADS DE IA
    # ========================================
    def _wait_for_ai_threads(self, timeout_per_thread: float = 2.0) -> None:
        """Aguarda as threads de IA terminarem, com timeout por thread."""
        with self._ai_pool_lock:
            threads = list(self.ai_thread_pool)

        for t in threads:
            try:
                t.join(timeout=timeout_per_thread)
            except Exception:
                pass

    # ========================================
    # CLEANUP
    # ========================================
    def _cleanup_handler(self, signum=None, frame=None) -> None:
        with self._cleanup_lock:
            if self._cleanup_started.is_set():
                logging.debug("Cleanup já em andamento, ignorando chamada duplicada")
                return
            self._cleanup_started.set()
            self.is_cleaning_up = True

        logging.info("🧹 Iniciando limpeza dos recursos...")
        self.should_stop = True

        # Se o loop principal do bot estiver ativo, agenda o shutdown assíncrono
        # (evita warnings de corotina não aguardada e 'Event loop is closed' no exit).
        try:
            if self._loop is not None and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.shutdown(), self._loop)
                return
        except Exception:
            pass

        # Aguarda término das threads de IA
        try:
            self._wait_for_ai_threads(timeout_per_thread=2.0)
        except Exception as e:
            logging.debug(f"Falha ao aguardar threads de IA no cleanup: {e}")

        # Demais componentes
        try:
            if self.context_collector:
                self.context_collector.stop()
                logging.info("✅ Context Collector parado.")
        except Exception as e:
            logging.error(f"❌ Erro ao parar Context Collector: {e}")

        try:
            if self.ai_analyzer and hasattr(self.ai_analyzer, "close"):
                self.ai_analyzer.close()
                logging.info("✅ AI Analyzer fechado.")
        except Exception as e:
            logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")

        try:
            if self.connection_manager:
                # A conexão WebSocket já deve ter sido encerrada em run().
                # Aqui garantimos apenas que qualquer loop interno pare.
                try:
                    self.connection_manager.should_stop = True
                except Exception as e:
                    logging.warning(f"Erro ignorado: {e}")
                logging.info("✅ Connection Manager sinalizada para parada.")
        except Exception as e:
            logging.error(f"❌ Erro ao sinalizar parada da Connection Manager: {e}")

        try:
            if hasattr(self, "event_bus"):
                self.event_bus.shutdown()
                logging.info("✅ Event Bus encerrado.")
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar Event Bus: {e}")

        try:
            if hasattr(self, "health_monitor"):
                self.health_monitor.stop()
                logging.info("✅ Health Monitor parado.")
        except Exception as e:
            logging.error(f"❌ Erro ao parar Health Monitor: {e}")

        # REMOVIDO: ClockSync não é mais usado

        try:
            if hasattr(self, "_async_executor"):
                self._async_executor.shutdown(wait=True, cancel_futures=True)
                logging.info("✅ Async Executor encerrado.")
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar Async Executor: {e}")

        # Encerrar loop asyncio dedicado do OrderBookAnalyzer
        try:
            if hasattr(self, "_async_loop"):
                try:
                    if (
                        hasattr(self, "orderbook_analyzer")
                        and self.orderbook_analyzer
                        and hasattr(self.orderbook_analyzer, "close")
                    ):
                        fut = asyncio.run_coroutine_threadsafe(
                            self.orderbook_analyzer.close(),
                            self._async_loop,
                        )
                        try:
                            fut.result(timeout=2.0)
                        except FutureTimeoutError:
                            logging.debug(
                                "Timeout ao fechar OrderBookAnalyzer; cancelando tarefa"
                            )
                            fut.cancel()
                        except Exception:
                            pass
                except Exception as e:
                    logging.debug(f"Falha ao fechar OrderBookAnalyzer: {e}")

                try:
                    self._async_loop.call_soon_threadsafe(self._async_loop.stop)
                except Exception:
                    pass
                try:
                    if hasattr(self, "_async_loop_thread"):
                        self._async_loop_thread.join(timeout=2.0)
                except Exception:
                    pass

                logging.info("✅ Loop assíncrono do OrderBookAnalyzer encerrado.")
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar loop assíncrono: {e}")

        try:
            if hasattr(self, "feature_store") and self.feature_store is not None:
                self.feature_store.close()
        except Exception as e:
            logging.warning(f"Falha ao fechar FeatureStore: {e}")
        
        # WindowProcessor e trades_buffer devem ser finalizados via shutdown() (async).

        logging.info("✅ Bot encerrado com segurança.")

    def _register_cleanup_handlers(self) -> None:
        # Nota: main.py gerencia os sinais do event loop assíncrono cooperativamente (SIGTERM/SIGINT).
        # Mantemos apenas atexit como salvaguarda síncrona final para processos não orquestrados pelo main.py.
        try:
            atexit.register(self._cleanup_handler)
        except Exception:
            pass

    # ========================================
    # LOOP ASSÍNCRONO DEDICADO
    # ========================================
    def _run_async_loop(self) -> None:
        asyncio.set_event_loop(self._async_loop)
        try:
            self._async_loop.run_forever()
        finally:
            try:
                try:
                    pending = asyncio.all_tasks(loop=self._async_loop)
                except (TypeError, AttributeError):
                    pending = asyncio.all_tasks()
            except Exception:
                pending = []

            for task in pending:
                try:
                    task.cancel()
                except Exception:
                    pass

            if pending:
                try:
                    group = asyncio.gather(*pending, return_exceptions=True)
                    self._async_loop.run_until_complete(
                        asyncio.wait_for(group, timeout=2.0)
                    )
                except Exception:
                    pass

            try:
                shutdown_coro = self._async_loop.shutdown_asyncgens()
                self._async_loop.run_until_complete(
                    asyncio.wait_for(shutdown_coro, timeout=1.0)
                )
            except Exception:
                pass

            try:
                self._async_loop.close()
            except Exception:
                pass

    # ========================================
    # JANELA DE TEMPO
    # ========================================
    def _next_boundary_ms(self, ts_ms: int) -> int:
        return ((ts_ms // self.window_ms) + 1) * self.window_ms

    # ========================================
    # PROCESSAMENTO DE MENSAGENS
    # ========================================
    def on_message(self, ws: Any, message: str) -> None:
        if self.should_stop:
            return

        # 1) Decodificação de JSON
        try:
            raw = json.loads(message)
            # FORENSIC-AUDIT (observacional, sem efeito na lógica): captura RAW Binance
            # antes de normalização/clamp/drop. Ativo só se FORENSIC_CAPTURE=1.
            try:
                from audit_live.hooks import on_raw_message as _forensic_raw

                _forensic_raw(message, raw)
            except Exception as _fe:
                logging.warning("FORENSIC hook on_raw_message falhou: %s", _fe, exc_info=True)
        except json.JSONDecodeError as e:
            self._invalid_json_count += 1
            step = self._invalid_json_log_step or 100
            if step > 0 and self._invalid_json_count % step == 0:
                logging.error(
                    "Erro ao decodificar mensagem JSON (amostra %d, total=%d): %s",
                    step,
                    self._invalid_json_count,
                    e,
                )
            return
        except Exception as e:
            logging.error(
                f"Erro inesperado ao decodificar mensagem JSON: {e}",
                exc_info=True,
            )
            return

        # 2) Extração e normalização de campos
        try:
            trade = raw.get("data", raw)

            if trade.get("e") == "aggTrade" or "f" in trade:
                trade_id = trade.get("a")        # agg trade id (futures)
                source = "fut_agg"
            else:
                trade_id = trade.get("t")        # trade id (spot legado)
                source = "spot_trade"

            if trade_id is not None:
                try:
                    trade_id = int(trade_id)
                except (TypeError, ValueError):
                    pass

            p = trade.get("p") or trade.get("P") or trade.get("price")
            q = trade.get("q") or trade.get("Q") or trade.get("quantity")
            T = trade.get("T")
            m = trade.get("m")

            # Fallback para mensagens de kline
            if (p is None or q is None or T is None) and isinstance(
                trade.get("k"), dict
            ):
                k = trade["k"]
                if p is None:
                    p = k.get("c")
                if q is None:
                    q = k.get("v")
                if T is None:
                    T = k.get("T")

            # 3) Verificação de campos obrigatórios
            missing: List[str] = []
            if p is None:
                missing.append("p")
                self._missing_field_counts["p"] += 1
            if q is None:
                missing.append("q")
                self._missing_field_counts["q"] += 1
            if T is None:
                missing.append("T")
                self._missing_field_counts["T"] += 1

            if missing:
                # FORENSIC-AUDIT: registra descarte por campo ausente (não altera fluxo)
                try:
                    from audit_live.hooks import on_normalized as _forensic_norm

                    _forensic_norm(None, drop_reason="missing_" + ",".join(missing))
                except Exception as _fe:
                    logging.warning("FORENSIC hook on_normalized(missing) falhou: %s", _fe, exc_info=True)
                total_missing = sum(
                    self._missing_field_counts[k] for k in ("p", "q", "T")
                )
                if self._missing_field_log_step:
                    try:
                        step = int(self._missing_field_log_step)
                    except Exception:
                        step = None

                    if step and step > 0 and total_missing % step == 0:
                        logging.debug(
                            "Campos ausentes (amostra): p=%d q=%d T=%d",
                            self._missing_field_counts["p"],
                            self._missing_field_counts["q"],
                            self._missing_field_counts["T"],
                        )
                return

            # 4) Conversão de tipos
            try:
                p = float(p)
                q = float(q)
                T = int(T)
            except (TypeError, ValueError):
                self._invalid_trade_count += 1
                # FORENSIC-AUDIT: descarte por tipo inválido
                try:
                    from audit_live.hooks import on_normalized as _forensic_norm2

                    _forensic_norm2(None, drop_reason="invalid_type")
                except Exception as _fe:
                    logging.warning("FORENSIC hook on_normalized(type) falhou: %s", _fe, exc_info=True)
                step = self._invalid_trade_log_step or 100
                if step > 0 and self._invalid_trade_count % step == 0:
                    logging.error(
                        "Trade inválido (tipos) - amostra %d, total=%d: %s",
                        step,
                        self._invalid_trade_count,
                        trade,
                    )
                return

            # 5) Validação básica
            if p <= 0 or q <= 0 or T <= 0:
                self._invalid_trade_count += 1
                # FORENSIC-AUDIT: descarte por valor não positivo
                try:
                    from audit_live.hooks import on_normalized as _forensic_norm3

                    _forensic_norm3(None, drop_reason="non_positive")
                except Exception as _fe:
                    logging.warning("FORENSIC hook on_normalized(non_positive) falhou: %s", _fe, exc_info=True)
                step = self._invalid_trade_log_step or 100
                if step > 0 and self._invalid_trade_count % step == 0:
                    logging.warning(
                        "Trade descartado por valores não positivos (amostra %d, total=%d): p=%s q=%s T=%s",
                        step,
                        self._invalid_trade_count,
                        p,
                        q,
                        T,
                    )
                return

            # 5.1) Normalização de T para garantir monotonicidade
            # O timestamp ORIGINAL é preservado em T_raw (consumido por
            # flow_analyzer/core.py para a detecção de out-of-order); o clamp
            # vale apenas para ordenação de janela/agregação.
            T_original = T
            last_T = self._last_trade_ts_ms
            if last_T is not None and T < last_T:
                self._ooo_trades_count += 1
                if PROMETHEUS_OK and TRADES_CORRECTED_TOTAL is not None:
                    try:
                        TRADES_CORRECTED_TOTAL.inc()
                    except Exception:
                        pass
                now = time.time()
                if now - self._last_ooo_log_ts >= self._ooo_log_interval_sec:
                    self._last_ooo_log_ts = now
                    logging.warning(
                        "Timestamp de trade fora de ordem: T_atual=%d < T_ultimo=%d "
                        "(delta=%dms, total_corrigidos=%d). Clamp aplicado para "
                        "ordenação de janela; T original preservado em T_raw.",
                        T,
                        last_T,
                        last_T - T,
                        self._ooo_trades_count,
                    )
                T = last_T
            else:
                self._last_trade_ts_ms = T

            # 6) Inferência de agressor (m) se ausente
            if m is None:
                last_price = self._last_price
                m = (p <= last_price) if last_price is not None else False

            # 7) Atualiza estados compartilhados
            self._last_price = p

            norm = {
                "p": p,
                "q": q,
                "T": T,
                "T_raw": T_original,
                "m": bool(m),
                "source": source,
                "trade_id": trade_id,
            }
            # FORENSIC-AUDIT: captura trade normalizado (observacional)
            try:
                from audit_live.hooks import on_normalized as _forensic_norm_ok

                _forensic_norm_ok(norm)
            except Exception as _fe:
                logging.warning("FORENSIC hook on_normalized(ok) falhou: %s", _fe, exc_info=True)

            # Adiciona trade ao buffer assíncrono
            def process_trade_sync(trade):
                # Envia trade para FlowAnalyzer
                self.flow_analyzer.process_trade(trade)
            
            # Adiciona ao buffer com controle de backpressure (versão thread-safe)
            success = self.trades_buffer.add_trade_sync(norm, process_trade_sync)
            
            if not success:
                logging.warning(f"⚠️ Trade descartado por buffer overflow")
                # FORENSIC-AUDIT: contabiliza drop de buffer (observacional)
                try:
                    from audit_live.forensic_context import CTX as _fctx

                    _fctx.inc("trade_buffer_drops")
                except Exception as _fe:
                    logging.warning("FORENSIC buffer drop count falhou: %s", _fe, exc_info=True)

            if success:
                if not getattr(self, "_first_trade_logged", False):
                    self._first_trade_logged = True
                    logging.info(f"✅ Trade recebido via WebSocket: source={source}, trade_id={trade_id}, p={p}, q={q}")
                try:
                    self.health_monitor.heartbeat("trade_ingestion")
                except Exception:
                    pass

            # Grava trade bruto em JSONL se configurado
            if self._raw_trades_file is not None:
                try:
                    with self._raw_trades_lock:
                        trade_line = json.dumps({
                            "trade_id": trade_id,
                            "timestamp": T,
                            "price": p,
                            "quantity": q,
                            "is_buyer_maker": bool(m),
                            "source": source,
                            "p": p,
                            "q": q,
                            "T": T,
                        })
                        self._raw_trades_file.write(trade_line + "\n")
                except Exception as e_dump:
                    logging.debug(f"Erro ao escrever trade no dump JSONL: {e_dump}")

            # Shadow Paper Trading Hook (Gate C3-C-B3-B)
            if self.shadow_runtime is not None:
                try:
                    self.shadow_runtime.on_market_trade(norm)
                except Exception as _shadow_exc:
                    self._handle_shadow_runtime_error(_shadow_exc)

            # 8) Controle de janelas
            if self.window_end_ms is None:
                self.window_end_ms = self._next_boundary_ms(T)

            if T >= self.window_end_ms:
                self._process_window()
                self.window_end_ms = self._next_boundary_ms(T)
                self.window_data = [norm]
            else:
                self.window_data.append(norm)

        except Exception as e:
            logging.error(f"Erro ao processar mensagem: {e}", exc_info=True)

    # ========================================
    # PONTOS DE DELEGAÇÃO PARA SUBMÓDULOS
    # ========================================
    def _process_window(self) -> None:
        # FIX P2: Nunca bloquear o event loop com processamento síncrono.
        # Se a fila estiver cheia, descartamos a janela ao invés de travar.
        if not self.window_data:
            return

        window_snapshot = [dict(trade) for trade in self.window_data if isinstance(trade, dict)]
        close_ms = self.window_end_ms
        self.window_data = []

        if self.window_processor is not None:
            if self.window_processor.submit_window(self, window_snapshot, close_ms):
                return
            # Fila cheia — descartar janela protegendo o event loop
            logging.warning(
                "⚠️ Janela descartada (fila cheia, %d pendentes). "
                "Event loop protegido contra bloqueio.",
                self.window_processor._queue.qsize(),
            )
            return

        # Sem WindowProcessor — processar sync (só acontece se initialize() falhou)
        self.window_data = window_snapshot
        process_window(self)

    def _process_signals(
        self,
        signals,
        pipeline,
        flow_metrics,
        historical_profile,
        macro_context,
        ob_event,
        enriched,
        close_ms,
        total_buy_volume,
        total_sell_volume,
        valid_window_data,
    ):
        return process_signals(
            self,
            signals,
            pipeline,
            flow_metrics,
            historical_profile,
            macro_context,
            ob_event,
            enriched,
            close_ms,
            total_buy_volume,
            total_sell_volume,
            valid_window_data,
        )

    def _setup_ai(self) -> None:
        """Carrega o runner de IA sob demanda para evitar imports circulares."""
        if self.ai_runner is None:
            from .ai.ai_runner import AIRunner

            self.ai_runner = AIRunner.create()
            logging.info("AIRunner inicializado via factory")

    def _initialize_ai_async(self) -> None:
        """Inicializa a IA em background thread via ai_runner."""
        self._setup_ai()
        from .ai.ai_runner import initialize_ai_async
        initialize_ai_async(self)

    def _run_ai_analysis_threaded(self, event_data: Dict[str, Any]) -> None:
        """Executa análise da IA em thread separada via ai_runner."""
        self._setup_ai()
        from .ai.ai_runner import run_ai_analysis_threaded
        run_ai_analysis_threaded(self, event_data)

    # ========================================
    # LÓGICA DE IA (usa _run_ai_analysis_threaded)
    # ========================================
    def _is_important_event_for_ai(self, event_data: Dict[str, Any]) -> bool:
        tipo = (event_data.get("tipo_evento") or "").upper()
        resultado = (event_data.get("resultado_da_batalha") or "").upper()
        severity = (event_data.get("severity") or "").upper()
        
        if tipo in ("ABSORÇÃO", "ABSORCAO", "EXAUSTÃO", "EXAUSTAO"):
            return True
        
        if tipo == "ZONA" or "zone_context" in event_data:
            return True
        
        if tipo == "ORDERBOOK":
            crit = event_data.get("critical_flags", {}) or {}
            if crit.get("is_critical") or severity == "CRITICAL":
                return True
        
        if tipo == "ALERTA":
            alert_type = resultado
            important_alerts = {
                "SUPPLY_EXHAUSTION",
                "DEMAND_EXHAUSTION",
                "LIQUIDITY_BREAK",
                "VOLATILITY_EXPANSION",
                "VOLATILITY_SQUEEZE",
            }
            if alert_type in important_alerts:
                return True
        
        if tipo == "ANALYSIS_TRIGGER":
            now = time.time()
            if now - self._last_ai_analysis_ts < self._ai_min_interval_sec:
                return False
            # Além do intervalo mínimo, exigir pelo menos uma condição relevante
            flow = event_data.get("fluxo_continuo", {}) or {}
            ob = event_data.get("orderbook_data", {}) or {}
            sr = event_data.get("zone_context", {}) or {}
            volume_ratio = float(flow.get("volume_ratio", 0) or 0)
            absorption_score = abs(float(
                flow.get("absorption_analysis", {}).get(
                    "current_absorption", {}
                ).get("index", 0) or 0
            ))
            whale_activity = abs(float(flow.get("whale_delta", 0) or 0)) > 0.5
            near_sr = bool(sr) or bool(event_data.get("near_level"))
            ob_imbalance = abs(float(ob.get("imbalance", 0) or 0)) > 0.7
            if any([
                volume_ratio > 1.5,
                absorption_score > 0.6,
                whale_activity,
                near_sr,
                ob_imbalance,
            ]):
                return True
            logging.debug(
                "ANALYSIS_TRIGGER skipped: no relevant condition "
                "(vol_r=%.2f, abs=%.2f, whale=%s, sr=%s, ob_imb=%s)",
                volume_ratio, absorption_score, whale_activity, near_sr, ob_imbalance,
            )
            return False
        
        return False
    
    def _handle_signal_event(self, event_data: Dict[str, Any]) -> None:
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        
        if not self._is_important_event_for_ai(event_data):
            return
        
        severity = (event_data.get("severity") or "").upper()
        tipo = (event_data.get("tipo_evento") or "").upper()
        is_critical = (
            severity in ("CRITICAL", "HIGH")
            or tipo in ("ABSORÇÃO", "ABSORCAO", "ZONA")
        )
        
        now = time.time()
        
        if (
            not is_critical
            and now - self._last_ai_analysis_ts < self._ai_min_interval_sec
        ):
            logging.debug("⏱️ IA em cooldown, pulando evento não-crítico")
            return
        
        # Integração com RegimeBasedRules
        regime_analysis = event_data.get("ai_payload", {}).get("regime_analysis", {})
        if regime_analysis:
            try:
                from market_analysis.regime_rules import RegimeBasedRules
                regime_rules = RegimeBasedRules()
                
                # Verificar se deve operar baseado no regime.
                signal_side = infer_signal_side(
                    event_type=event_data.get("tipo_evento"),
                    battle_result=event_data.get("resultado_da_batalha"),
                    explicit_side=event_data.get("side", event_data.get("absorption_side")),
                )
                signal_direction = "long" if signal_side == "LONG" else ("short" if signal_side == "SHORT" else "neutral")
                signal_confidence = get_directional_confidence(
                    event_data.get("historical_confidence", {}),
                    signal_direction,
                    default_fallback=0.5,
                )
                
                should_trade, reason = regime_rules.should_trade(
                    regime_analysis=regime_analysis,
                    signal_direction=signal_direction,
                    signal_confidence=signal_confidence
                )
                
                if not should_trade:
                    logging.info(f"Trade bloqueado pelo regime: {reason}")
                    return
                
                # Log do regime
                logging.info(regime_rules.format_regime_summary(regime_analysis))
                
            except Exception as e:
                logging.error(f"Erro ao aplicar regras de regime: {e}")
        
        self._last_ai_analysis_ts = now
        self._run_ai_analysis_threaded(event_data.copy())

    def _handle_zone_touch_event(self, event_data: Dict[str, Any]) -> None:
        if not self.ai_analyzer or not self.ai_test_passed:
            return

        now = time.time()
        time_since_last = now - self._last_ai_analysis_ts

        if time_since_last < self._ai_min_interval_sec:
            logging.info(
                f"🎯 BYPASS DE COOLDOWN: Toque em zona processado "
                f"após apenas {time_since_last:.1f}s"
            )

        self._last_ai_analysis_ts = now
        self._run_ai_analysis_threaded(event_data.copy())

    # ========================================
    # PROCESSAMENTO DE VP FEATURES
    # ========================================
    def _process_vp_features(
        self,
        historical_profile: Dict[str, Any],
        preco_atual: float,
    ) -> Dict[str, Any]:
        try:
            if not preco_atual or preco_atual <= 0:
                return {"status": "no_data"}

            vp_daily = historical_profile.get("daily", {})
            hvns = vp_daily.get("hvns", [])
            lvns = vp_daily.get("lvns", [])
            sp = vp_daily.get("single_prints", [])
            poc = vp_daily.get("poc", 0)

            if not poc or (not hvns and not lvns):
                return {"status": "no_data"}

            dist_to_poc = preco_atual - poc

            nearest_hvn = (
                min(hvns, key=lambda x: abs(x - preco_atual)) if hvns else None
            )
            nearest_lvn = (
                min(lvns, key=lambda x: abs(x - preco_atual)) if lvns else None
            )

            dist_hvn = (preco_atual - nearest_hvn) if nearest_hvn else None
            dist_lvn = (preco_atual - nearest_lvn) if nearest_lvn else None

            faixa_lim = preco_atual * 0.005

            hvn_near = sum(
                1 for h in hvns if abs(h - preco_atual) <= faixa_lim
            )
            lvn_near = sum(
                1 for l in lvns if abs(l - preco_atual) <= faixa_lim
            )

            in_single = any(
                abs(px - preco_atual) <= faixa_lim for px in sp
            )

            return {
                "status": "ok",
                "distance_to_poc": round(dist_to_poc, 2),
                "nearest_hvn": nearest_hvn,
                "dist_to_nearest_hvn": (
                    round(dist_hvn, 2) if dist_hvn is not None else None
                ),
                "nearest_lvn": nearest_lvn,
                "dist_to_nearest_lvn": (
                    round(dist_lvn, 2) if dist_lvn is not None else None
                ),
                "hvns_within_0_5pct": hvn_near,
                "lvns_within_0_5pct": lvn_near,
                "in_single_print_zone": in_single,
            }

        except Exception as e:
            logging.error(f"Erro ao gerar vp_features: {e}")
            return {"status": "error"}

    # ========================================
    # LOG DE EVENTOS
    # ========================================
    def _format_memory_timestamp_ny(self, e: Dict[str, Any]) -> str:
        """
         Converte o timestamp de um evento da memória para horário de New York (string).

          Preferência:
          1) epoch_ms (ou metadata.timestamp_unix_ms)
          2) timestamp_ny
          3) timestamp_utc
          4) timestamp bruto (removendo 'Z' se existir)
          """
        epoch_ms = e.get("epoch_ms") or (e.get("metadata") or {}).get("timestamp_unix_ms")
        if epoch_ms is not None:
            try:
                epoch_ms_int = int(epoch_ms)
                dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=timezone.utc)
                dt_ny = dt_utc.astimezone(self.ny_tz)
                return dt_ny.strftime("%Y-%m-%d %H:%M:%S NY")
            except Exception:
                pass

        ts_candidates = [
            e.get("timestamp_ny"),
            e.get("timestamp_utc"),
            e.get("timestamp"),
        ]
        for ts in ts_candidates:
            if not ts:
                continue
            raw = str(ts)
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_ny = dt.astimezone(self.ny_tz)
                return dt_ny.strftime("%Y-%m-%d %H:%M:%S NY")
            except Exception:
                continue

        raw = str(e.get("timestamp", "N/A"))
        if raw.endswith("Z"):
            raw = raw[:-1]
        return raw

    def _log_event(self, event: Dict[str, Any]) -> None:
        ts_ny = event.get("timestamp_ny")

        if ts_ny:
            try:
                ny_time = datetime.fromisoformat(
                    ts_ny.replace("Z", "+00:00")
                ).astimezone(self.ny_tz)
            except Exception:
                ny_time = datetime.now(self.ny_tz)
        else:
            ny_time = datetime.now(self.ny_tz)

        resultado = event.get("resultado_da_batalha", "N/A").upper()
        tipo = event.get("tipo_evento", "EVENTO")
        descricao = event.get("descricao", "")
        conf = event.get("historical_confidence", {})

        print(
            f"\n🎯 {tipo}: {resultado} DETECTADO - "
            f"{ny_time.strftime('%H:%M:%S')} NY"
        )
        print(f" Símbolo: {self.symbol} | Janela #{self.window_count}")
        print(f" 📝 {descricao}")

        if conf:
            print(
                f" 📊 Probabilidades -> "
                f"Long={conf.get('long_prob')} | "
                f"Short={conf.get('short_prob')} | "
                f"Neutro={conf.get('neutral_prob')}"
            )

        ultimos = [
            e
            for e in obter_memoria_eventos(n=4)
            if e.get("tipo_evento") != "OrderBook"
        ]

        if ultimos:
            print(" 🕒 Últimos sinais:")
            for e in ultimos:
                delta_fmt = format_delta(e.get("delta", 0))
                vol_fmt = format_large_number(e.get("volume_total", 0))
                ts_display = self._format_memory_timestamp_ny(e)
                print(
                    f"  - {ts_display} | "
                    f"{e.get('tipo_evento', 'N/A')} "
                    f"{e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={delta_fmt}, Vol={vol_fmt})"
                )

    # ========================================
    # ENRIQUECIMENTO DE SINAL
    # ========================================
    def _enrich_signal(
        self,
        signal: Dict[str, Any],
        derivatives_context: Dict[str, Any],
        flow_metrics: Dict[str, Any],
        total_buy_volume: float,
        total_sell_volume: float,
        macro_context: Dict[str, Any],
        close_ms: int,
        ml_payload: Dict[str, Any],
        enriched_snapshot: Dict[str, Any],
        contextual_snapshot: Dict[str, Any],
        ob_event: Dict[str, Any],
        valid_window_data: List[Dict[str, Any]],
        support_resistance: Dict[str, Any],
        defense_zones_data: Dict[str, Any],
    ) -> None:
        """Enriquece sinal com dados adicionais e gera evento institucional."""

        signal.setdefault("janela_numero", self.window_count)

        if "epoch_ms" not in signal:
            signal["epoch_ms"] = close_ms

        dt_utc = self.time_manager.from_timestamp_ms(close_ms, tz=self.time_manager.tz_utc)
        dt_ny = self.time_manager.from_timestamp_ms(close_ms, tz=self.ny_tz)

        # Normaliza timestamps para evitar correções automáticas do DataValidator
        # (ele adiciona 'Z' quando o campo 'timestamp' não tem timezone explícito).
        ts_utc = signal.get("timestamp_utc")
        if not isinstance(ts_utc, str) or not ts_utc.strip():
            signal["timestamp_utc"] = dt_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        else:
            ts_utc_clean = ts_utc.strip()
            if ts_utc_clean.endswith("+00:00"):
                signal["timestamp_utc"] = ts_utc_clean.replace("+00:00", "Z")
            else:
                has_tz = ts_utc_clean.endswith("Z") or ("+" in ts_utc_clean[-6:] or "-" in ts_utc_clean[-6:])
                if not has_tz:
                    signal["timestamp_utc"] = dt_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        ts_local = signal.get("timestamp")
        if not isinstance(ts_local, str) or not ts_local.strip():
            signal["timestamp"] = dt_ny.isoformat(sep=" ", timespec="seconds")
        else:
            ts_local_clean = ts_local.strip()
            if ts_local_clean.endswith("+00:00") or "+00:00" in ts_local_clean:
                signal["timestamp"] = ts_local_clean.replace("+00:00", "Z")
            else:
                has_tz = ts_local_clean.endswith("Z") or ("+" in ts_local_clean[-6:] or "-" in ts_local_clean[-6:])
                if not has_tz:
                    signal["timestamp"] = dt_ny.isoformat(sep=" ", timespec="seconds")

        validated_signal = validator.validate_and_clean(signal)
        if not validated_signal:
            logging.warning(
                f"Evento {signal.get('tipo_evento')} / "
                f"{signal.get('resultado_da_batalha')} descartado pela validação."
            )
            return

        signal.update(validated_signal)

        if "derivatives" not in signal:
            signal["derivatives"] = derivatives_context

        should_validate_flow = (
            self.window_count % 10 == 0
            or self.orderbook_fetch_failures > 0
            or len(valid_window_data) < 5
        )

        if "fluxo_continuo" not in signal and flow_metrics:
            flow_valid = True
            if should_validate_flow:
                flow_valid = self._validate_flow_metrics(
                    flow_metrics, valid_window_data
                )
                if not flow_valid:
                    signal["flow_data_quality"] = "incomplete"
            signal["fluxo_continuo"] = flow_metrics
            # liquidity_heatmap já está dentro de fluxo_continuo — não duplicar na raiz

        if (
            signal.get("volume_compra", 0) == 0
            and signal.get("volume_venda", 0) == 0
        ):
            signal["volume_compra"] = total_buy_volume
            signal["volume_venda"] = total_sell_volume

        try:
            if "market_context" not in signal:
                signal["market_context"] = macro_context.get(
                    "market_context", {}
                )
            if "market_environment" not in signal:
                signal["market_environment"] = macro_context.get(
                    "market_environment", {}
                )
            if "external_markets" not in signal:
                signal["external_markets"] = macro_context.get(
                    "external", {}
                )
        except Exception:
            pass

        signal.setdefault("features_window_id", str(close_ms))
        signal["ml_features"] = ml_payload
        # Deduplica snapshots: remove chaves já presentes no signal root
        # (flow_metrics já está em fluxo_continuo, historical_vp/multi_tf/derivatives/
        #  market_context/market_environment/orderbook_data já estão no signal root)
        _enrich_dedup_keys = {
            "flow_metrics", "historical_vp", "orderbook_data",
            "multi_tf", "derivatives", "market_context", "market_environment",
        }
        _ctx_filtered = {
            k: v for k, v in contextual_snapshot.items() if k not in _enrich_dedup_keys
        }
        signal["contextual_snapshot"] = _ctx_filtered
        # enriched_snapshot é idêntico ao contextual_snapshot após dedup —
        # usar mesma referência para economizar ~5KB por evento
        signal["enriched_snapshot"] = _ctx_filtered

        if support_resistance:
            signal["support_resistance"] = support_resistance
        if defense_zones_data:
            signal["defense_zones"] = defense_zones_data

        EnhancedMarketBot._enrich_orderbook_metrics(signal, ob_event)

        # ====== Institutional Analytics ======
        if self.institutional_analytics is not None:
            try:
                # Extrair dados disponíveis
                # historical_vp: buscar no top-level (promovido), contextual, ou enriched (param)
                _hvp = (
                    signal.get("historical_vp")
                    or signal.get("contextual_snapshot", {}).get("historical_vp")
                    or (enriched_snapshot or {}).get("historical_vp")
                    or {}
                )
                _vp_daily = _hvp.get("daily", {})
                _weekly_vp = _hvp.get("weekly", {})
                _monthly_vp = _hvp.get("monthly", {})
                # multi_tf: buscar no top-level (promovido), contextual, ou enriched (param)
                _multi_tf = (
                    signal.get("multi_tf")
                    or signal.get("contextual_snapshot", {}).get("multi_tf")
                    or (enriched_snapshot or {}).get("multi_tf")
                    or {}
                )
                _derivatives = signal.get("derivatives", {})
                # FIX pivot_points (auditoria 2026-08-09): fonte canônica dos pivots
                # clássicos é macro_context["pivots"] (calculado por
                # context_collector._calculate_pivots via daily_pivot iloc[-2] =
                # período anterior COMPLETO, fix 75bd3ec). contextual_snapshot.pivots
                # NÃO existe em nenhum evento real — era um dead wire que causava
                # fallback silencioso para VP intraday parcial no enricher.
                _pivot_data = (
                    macro_context.get("pivots")
                    or signal.get("contextual_snapshot", {}).get("pivots")
                    or {}
                )
                if _pivot_data:
                    # Propaga para o sinal: fonte primária do institutional enricher
                    # (_build_pivot_points) e rastreável no evento final.
                    signal["pivots"] = _pivot_data

                # Extrair EMAs dos dados multi-TF
                _emas = {}
                for tf_name, tf_data in (_multi_tf or {}).items():
                    if isinstance(tf_data, dict):
                        mme = tf_data.get("mme_21")
                        if mme and mme > 0:
                            _emas[f"ema_21_{tf_name}"] = mme

                # Construir candles_df do histórico OHLC
                _candles_df = None
                if self.pattern_ohlc_history and len(self.pattern_ohlc_history) >= 5:
                    try:
                        _candles_df = pd.DataFrame(list(self.pattern_ohlc_history))
                    except Exception:
                        pass

                # Construir trades_df da janela
                _trades_df = None
                if valid_window_data and len(valid_window_data) > 20:
                    try:
                        _trades_df = pd.DataFrame(valid_window_data)
                    except Exception:
                        pass

                # Absorption data
                _absorption = None
                _flow = signal.get("fluxo_continuo", {})
                if isinstance(_flow, dict):
                    _absorption = _flow.get("absorption_analysis", {})

                # Positioning data (Binance Futures)
                _positioning = (
                    signal.get("positioning")
                    or signal.get("contextual_snapshot", {}).get("positioning")
                    or signal.get("sentiment", {}).get("positioning")
                )

                # Calcular tudo
                _t_inst_start = time.perf_counter()
                institutional_result = self.institutional_analytics.compute_all(
                    current_price=signal.get("preco_fechamento", 0) or (
                        signal.get("contextual_snapshot", {}).get("ohlc", {}).get("close", 0)
                    ),
                    flow_metrics=_flow if isinstance(_flow, dict) else None,
                    vp_data=_vp_daily,
                    orderbook_data=signal.get("orderbook_data", {}),
                    candles_df=_candles_df,
                    trades_df=_trades_df,
                    macro_context=signal.get("market_context", {}),
                    derivatives_data=_derivatives,
                    absorption_data=_absorption,
                    pivot_data=_pivot_data,
                    ema_values=_emas,
                    weekly_vp=_weekly_vp,
                    monthly_vp=_monthly_vp,
                    window_close_ms=close_ms,
                    time_manager=self.time_manager,
                    positioning_data=_positioning,
                )
                _t_inst_ms = (time.perf_counter() - _t_inst_start) * 1000

                signal["institutional_analytics"] = institutional_result

            except Exception as e:
                logging.debug(f"InstitutionalAnalytics error: {e}")
                signal["institutional_analytics"] = {"status": "error", "error": str(e)}
                _t_inst_ms = 0.0

        # P6: CFTC COT semanal (flag-gated; read_view µs, sem I/O).
        # Chave independente "cftc_cot"; ausência nunca altera "pos"/Binance.
        try:
            _cftc_updater = getattr(self, "cftc_cot_updater", None)
            if _cftc_updater is not None and bool(
                getattr(config, "ENABLE_CFTC_COT_CONTEXT", False)
            ):
                _cftc_view = _cftc_updater.read_view(getattr(self, "symbol", "BTCUSDT"))
                if isinstance(_cftc_view, dict) and _cftc_view:
                    signal["cftc_cot"] = _cftc_view
        except Exception as e:
            logging.debug(f"CftcCot read_view error (não-crítico): {e}")

        # ====== Promoção de campos para ANALYSIS_TRIGGER ======
        # Garante que multi_tf, historical_vp, flow_metrics estejam no top-level
        # (iguala a estrutura com eventos de Absorção)
        raw_evt = signal.get("raw_event", {}) or {}
        for _promote_key in ("multi_tf", "historical_vp"):
            if _promote_key not in signal and isinstance(raw_evt, dict):
                _val = raw_evt.get(_promote_key)
                if _val:
                    signal[_promote_key] = _val
        # flow_metrics do raw_event → fluxo_continuo (se ainda não existir)
        if "fluxo_continuo" not in signal and isinstance(raw_evt, dict):
            _fm = raw_evt.get("flow_metrics")
            if _fm and isinstance(_fm, dict):
                signal["fluxo_continuo"] = _fm

        # Limpar chaves duplicadas do raw_event após promoção ao top-level
        if isinstance(raw_evt, dict):
            for _dup_key in ("multi_tf", "historical_vp"):
                if _dup_key in raw_evt and _dup_key in signal:
                    raw_evt.pop(_dup_key, None)
            if "flow_metrics" in raw_evt and "fluxo_continuo" in signal:
                raw_evt.pop("flow_metrics", None)

        # ====== Enriquecimento Institucional (Onda 1 + 2) ======
        # Injeta campos faltantes: pivot_points, fibonacci, bid/ask, alertas,
        # volume_profile_advanced, volatility_metrics, whale_activity, etc.
        _t_enrich_ms = 0.0
        if _INSTITUTIONAL_ENRICHER_OK and _institutional_enrich is not None:
            try:
                _t_enrich_start = time.perf_counter()
                _institutional_enrich(signal, valid_window_data=valid_window_data)
                _t_enrich_ms = (time.perf_counter() - _t_enrich_start) * 1000
            except Exception as _enrich_err:
                logging.debug(f"institutional_enrich error (não crítico): {_enrich_err}")

        logging.info(
            "event=signal_enrichment_timings window_id=%s inst_analytics_ms=%.1f inst_enricher_ms=%.1f",
            f"{getattr(self, 'symbol', 'BTCUSDT')}_{close_ms}",
            locals().get("_t_inst_ms", 0.0),
            _t_enrich_ms,
        )

        if signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
            key = (
                signal.get("tipo_evento"),
                signal.get("features_window_id"),
            )
            if key in self._sent_triggers:
                logging.debug(
                    f"⏭️ ANALYSIS_TRIGGER duplicado ignorado (janela {close_ms})"
                )
                return
            self._sent_triggers.add(key)

        self.levels.add_from_event(signal)

        logging.debug(
            f"💾 Salvando: {signal.get('tipo_evento')} / "
            f"{signal.get('resultado_da_batalha')} | "
            f"epoch_ms={signal.get('epoch_ms')} | "
            f"janela_numero={signal.get('janela_numero')}"
        )

        self.event_bus.publish("signal", signal)

        institutional_event = self._build_institutional_event(signal)
        self.event_saver.save_event(institutional_event)

        if signal.get("tipo_evento") != "OrderBook":
            adicionar_memoria_evento(signal)

            # Avaliar outcomes pendentes de sinais anteriores
            current_price = signal.get("preco_fechamento", 0)
            current_epoch = signal.get("epoch_ms", 0)
            if current_price > 0 and current_epoch > 0:
                avaliar_outcomes_pendentes(current_price, current_epoch)

            # Enriquecer com similarity search (eventos passados similares)
            if _SIMILARITY_OK and _similarity_search:
                try:
                    similar = _similarity_search.find_similar(signal, top_k=5)
                    if similar.get("status") == "ok":
                        signal["similar_events"] = similar.get("summary", {})
                except Exception as e:
                    logging.debug(f"Similarity search falhou: {e}")

            # Enriquecer com confiança estatística real
            if False:  # _outcome_tracker disabled
                try:
                    confidence = _outcome_tracker.get_confidence_for_event(signal)
                    if confidence.get("has_data"):
                        signal["statistical_confidence"] = confidence
                except Exception as e:
                    logging.debug(f"Outcome confidence falhou: {e}")

        self._log_event(signal)

    # ========================================
    # ENRIQUECIMENTO DE ORDERBOOK E MARKET IMPACT
    # ========================================
    @staticmethod
    def _enrich_orderbook_metrics(signal: Dict[str, Any], ob_event: Dict[str, Any]) -> None:
        """Enriquece o sinal com métricas de orderbook e market impact de forma modular."""
        if not (ob_event and isinstance(ob_event, dict) and ob_event.get("is_valid", False)):
            return

        if "orderbook_data" in ob_event:
            signal["orderbook_data"] = ob_event["orderbook_data"]
        elif "orderbook_data" not in signal:
            signal["orderbook_data"] = ob_event

        if "order_book_depth" in ob_event:
            signal["order_book_depth"] = ob_event["order_book_depth"]

        # FIX provenance 2026-09: propaga walls observadas (price/qty/threshold
        # por lado, shape nativo {"bids":[...],"asks":[...]}) para dentro de
        # orderbook_data — único input lido pelo DefenseZoneDetector.
        # Sem isto, walls reais eram descartadas aqui e o detector projetava
        # current_price*0.999/1.001 sob o rótulo orderbook_*_wall.

        # FIX 3.4: Consolidar spread_analysis e orderbook_data_quality
        # dentro de orderbook_data (evita seções separadas duplicadas)
        if isinstance(signal.get("orderbook_data"), dict):
            signal["orderbook_data"] = signal["orderbook_data"].copy()
            # Garantir presença de timestamps, source e snapshot_offset_ms
            for k in ("timestamps", "source", "source_type", "snapshot_offset_ms"):
                val = ob_event.get(k)
                if val is not None and k not in signal["orderbook_data"]:
                    signal["orderbook_data"][k] = val
            # FIX provenance 2026-09 (cont.): walls observadas viajam junto.
            if isinstance(ob_event.get("walls"), dict) and "walls" not in signal["orderbook_data"]:
                signal["orderbook_data"]["walls"] = ob_event["walls"]
            # Mover spread_bps de spread_analysis para orderbook_data
            sa = ob_event.get("spread_analysis") or {}
            if sa.get("current_spread_bps"):
                signal["orderbook_data"]["spread_bps"] = sa["current_spread_bps"]
            # FIX 7B: depth_metrics NOT copied here — order_book_depth (L1-L25)
            # is the canonical source. Keeping both duplicates the data.

        if isinstance(signal.get("raw_event"), dict):
            signal["raw_event"]["orderbook_data"] = signal.get("orderbook_data")

        dq = ob_event.get("data_quality") or {}
        if dq:
            if isinstance(signal.get("orderbook_data"), dict):
                signal["orderbook_data"]["is_valid"] = dq.get("is_valid", True)
                signal["orderbook_data"]["data_source"] = dq.get("data_source", "unknown")
            src = dq.get("data_source") or "unknown"
            if src == "live":
                signal["orderbook_quality"] = "live"
            elif src == "emergency":
                signal["orderbook_quality"] = "emergency"
            else:
                signal["orderbook_quality"] = "unknown"

        try:
            mi_buy = ob_event.get("market_impact_buy", {}) or {}
            mi_sell = ob_event.get("market_impact_sell", {}) or {}

            def _get_slippage(mi_dict, key):
                d = mi_dict.get(key, {}) or {}
                if "execution_slippage_usd" in d:
                    return d.get("execution_slippage_usd")
                return d.get("move_usd")

            def _get_observed_slippage(mi_dict, key):
                d = mi_dict.get(key, {}) or {}
                if "observed_execution_slippage_usd" in d:
                    return d.get("observed_execution_slippage_usd")
                return d.get("observed_move_usd") if "observed_move_usd" in d else d.get("move_usd")

            def _get_terminal_move(mi_dict, key):
                d = mi_dict.get(key, {}) or {}
                if "terminal_move_usd" in d:
                    return d.get("terminal_move_usd")
                return d.get("move_usd")

            def _get_observed_terminal_move(mi_dict, key):
                d = mi_dict.get(key, {}) or {}
                if "observed_terminal_move_usd" in d:
                    return d.get("observed_terminal_move_usd")
                return d.get("observed_move_usd") if "observed_move_usd" in d else d.get("move_usd")

            def _get_fill_ratio(mi_dict, key):
                d = mi_dict.get(key, {}) or {}
                return d.get("fill_ratio", 1.0 if d else 0.0)

            def _get_insufficient(mi_dict, key):
                d = mi_dict.get(key, {}) or {}
                return d.get("insufficient_liquidity", False)

            slippage_matrix = {
                "1k_usd":   {"buy": _get_slippage(mi_buy,  "1k"),  "sell": _get_slippage(mi_sell,  "1k")},
                "10k_usd":  {"buy": _get_slippage(mi_buy, "10k"),  "sell": _get_slippage(mi_sell, "10k")},
                "100k_usd": {"buy": _get_slippage(mi_buy,"100k"),  "sell": _get_slippage(mi_sell,"100k")},
                "1m_usd":   {"buy": _get_slippage(mi_buy,  "1M"),  "sell": _get_slippage(mi_sell,  "1M")},
            }

            observed_partial_slippage_matrix = {
                "1k_usd":   {"buy": _get_observed_slippage(mi_buy,  "1k"),  "sell": _get_observed_slippage(mi_sell,  "1k")},
                "10k_usd":  {"buy": _get_observed_slippage(mi_buy, "10k"),  "sell": _get_observed_slippage(mi_sell, "10k")},
                "100k_usd": {"buy": _get_observed_slippage(mi_buy,"100k"),  "sell": _get_observed_slippage(mi_sell,"100k")},
                "1m_usd":   {"buy": _get_observed_slippage(mi_buy,  "1M"),  "sell": _get_observed_slippage(mi_sell,  "1M")},
            }

            terminal_move_matrix = {
                "1k_usd":   {"buy": _get_terminal_move(mi_buy,  "1k"),  "sell": _get_terminal_move(mi_sell,  "1k")},
                "10k_usd":  {"buy": _get_terminal_move(mi_buy, "10k"),  "sell": _get_terminal_move(mi_sell, "10k")},
                "100k_usd": {"buy": _get_terminal_move(mi_buy,"100k"),  "sell": _get_terminal_move(mi_sell,"100k")},
                "1m_usd":   {"buy": _get_terminal_move(mi_buy,  "1M"),  "sell": _get_terminal_move(mi_sell,  "1M")},
            }

            observed_terminal_move_matrix = {
                "1k_usd":   {"buy": _get_observed_terminal_move(mi_buy,  "1k"),  "sell": _get_observed_terminal_move(mi_sell,  "1k")},
                "10k_usd":  {"buy": _get_observed_terminal_move(mi_buy, "10k"),  "sell": _get_observed_terminal_move(mi_sell, "10k")},
                "100k_usd": {"buy": _get_observed_terminal_move(mi_buy,"100k"),  "sell": _get_observed_terminal_move(mi_sell,"100k")},
                "1m_usd":   {"buy": _get_observed_terminal_move(mi_buy,  "1M"),  "sell": _get_observed_terminal_move(mi_sell,  "1M")},
            }

            fill_ratio_matrix = {
                "100k_usd": {"buy": _get_fill_ratio(mi_buy, "100k"), "sell": _get_fill_ratio(mi_sell, "100k")},
                "1m_usd":   {"buy": _get_fill_ratio(mi_buy, "1M"),   "sell": _get_fill_ratio(mi_sell, "1M")},
            }

            insufficient_liquidity = {
                "100k_usd": {"buy": _get_insufficient(mi_buy, "100k"), "sell": _get_insufficient(mi_sell, "100k")},
                "1m_usd":   {"buy": _get_insufficient(mi_buy, "1M"),   "sell": _get_insufficient(mi_sell, "1M")},
            }

            insuf_100k_buy = _get_insufficient(mi_buy, "100k")
            insuf_100k_sell = _get_insufficient(mi_sell, "100k")

            if insuf_100k_buy or insuf_100k_sell:
                liquidity_score = None
            else:
                bps_100k_buy = (mi_buy.get("100k") or {}).get("bps")
                bps_100k_sell = (mi_sell.get("100k") or {}).get("bps")
                bps_list = [
                    v for v in (bps_100k_buy, bps_100k_sell)
                    if isinstance(v, (int, float))
                ]
                if bps_list:
                    avg_bps = float(sum(bps_list) / len(bps_list))
                    liquidity_score = max(0.0, min(10.0, 10.0 - avg_bps / 5.0))
                else:
                    liquidity_score = None

            insuf_1m = _get_insufficient(mi_buy, "1M") or _get_insufficient(mi_sell, "1M")
            if liquidity_score is not None:
                if liquidity_score >= 8:
                    execution_quality = "PARTIAL_1M" if insuf_1m else "EXCELLENT"
                elif liquidity_score >= 6:
                    execution_quality = "GOOD"
                elif liquidity_score >= 4:
                    execution_quality = "FAIR"
                else:
                    execution_quality = "POOR"
            else:
                execution_quality = "INSUFFICIENT" if (insuf_100k_buy or insuf_100k_sell) else None

            dir_liq = ob_event.get("directional_liquidity")
            if dir_liq is None:
                mid_val = (
                    (ob_event.get("spread_metrics") or {}).get("mid")
                    or (ob_event.get("orderbook_data") or {}).get("mid")
                )
                from orderbook_analyzer.directional_liquidity import build_directional_liquidity
                dir_liq = build_directional_liquidity(mi_buy, mi_sell, mid_val)

            signal["market_impact"] = {
                "slippage_matrix": slippage_matrix,
                "observed_partial_matrix": observed_partial_slippage_matrix,
                "observed_partial_slippage_matrix": observed_partial_slippage_matrix,
                "terminal_move_matrix": terminal_move_matrix,
                "observed_terminal_move_matrix": observed_terminal_move_matrix,
                "fill_ratio_matrix": fill_ratio_matrix,
                "insufficient_liquidity": insufficient_liquidity,
                "liquidity_score": liquidity_score,
                "execution_quality": execution_quality,
                "legacy_metadata": {
                    "status": "AGGREGATED_LEGACY",
                    "execution_gate": "NOT_DIRECTIONAL_EXECUTION_GATE",
                    "notes": (
                        "liquidity_score and execution_quality are aggregated legacy metrics "
                        "and do not represent directional execution gates for BUY or SELL."
                    ),
                },
                "directional_liquidity": dir_liq,
            }
            signal["directional_liquidity"] = dir_liq
        except Exception as e:
            logging.debug(f"Falha ao construir market_impact: {e}")

    # ========================================
    # BUILDER DE EVENTO INSTITUCIONAL
    # ========================================
    def _build_institutional_event(
        self, signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Retorna o signal diretamente sem re-embrulhar em outro raw_event
        # (antes criava raw_event.raw_event desnecessário, duplicando ~100% dos dados)
        return signal

    # ========================================
    # (Demais métodos auxiliares do original)
    # ========================================
    def _build_price_targets(
        self,
        pattern_recognition: Dict[str, Any],
        last_price: float,
    ) -> Dict[str, Any]:
        targets: List[Dict[str, Any]] = []
        try:
            patterns = pattern_recognition.get("active_patterns") or []
            for p in patterns:
                ptype = (p.get("type") or "").upper()
                target = p.get("target_price")
                stop = p.get("stop_loss")
                conf = float(p.get("confidence", 0.0) or 0.0)

                side = "UNKNOWN"
                if "ASCENDING" in ptype or "BULL" in ptype:
                    side = "BULLISH"
                elif "DESCENDING" in ptype or "BEAR" in ptype:
                    side = "BEARISH"

                if target is not None:
                    risk = None
                    rr = None
                    if stop is not None and last_price:
                        try:
                            risk = abs(last_price - float(stop))
                            reward = abs(float(target) - last_price)
                            rr = reward / risk if risk > 0 else None
                        except Exception:
                            rr = None

                    targets.append(
                        {
                            "pattern_type": ptype,
                            "side": side,
                            "target_price": float(target),
                            "stop_loss": float(stop) if stop is not None else None,
                            "confidence": conf,
                            "risk_reward": rr,
                        }
                    )
        except Exception as e:
            logging.debug(f"Erro ao construir price_targets: {e}")

        if not targets:
            return {}

        return {
            "targets": targets,
            "last_price": last_price,
        }

    def _validate_flow_metrics(
        self,
        flow_metrics: Dict[str, Any],
        valid_window_data: List[Dict[str, Any]],
    ) -> bool:
        try:
            trades_processed = 0
            if "data_quality" in flow_metrics:
                trades_processed = flow_metrics["data_quality"].get(
                    "flow_trades_count", 0
                )

            if trades_processed > 0:
                return True

            sector_flow = flow_metrics.get("sector_flow", {})
            for _, data in sector_flow.items():
                total_vol = abs(data.get("buy", 0)) + abs(data.get("sell", 0))
                if total_vol > 0.001:
                    return True

            order_flow = flow_metrics.get("order_flow", {})
            for key in ("net_flow_1m", "net_flow_5m", "net_flow_15m"):
                val = order_flow.get(key)
                if val is not None and val != 0:
                    return True

            buy_pct = order_flow.get("aggressive_buy_pct", 0.0)
            sell_pct = order_flow.get("aggressive_sell_pct", 0.0)
            if buy_pct > 0 or sell_pct > 0:
                return True

            whale_total = abs(flow_metrics.get("whale_buy_volume", 0.0)) + abs(
                flow_metrics.get("whale_sell_volume", 0.0)
            )
            if whale_total > 0.001:
                return True

            return False

        except Exception as e:
            logging.error(f"Erro ao validar flow_metrics: {e}")
            return False

    def _check_zone_touches(
        self, enriched: Dict[str, Any], signals: List[Dict[str, Any]]
    ) -> None:
        preco_atual = enriched.get("ohlc", {}).get("close", 0.0)

        if preco_atual > 0:
            try:
                touched = self.levels.check_price(float(preco_atual))

                for z in touched:
                    zone_event = signals[0].copy() if signals else {}

                    preco_fmt = format_price(preco_atual)
                    low_fmt = format_price(z.low)
                    high_fmt = format_price(z.high)

                    zone_event.update(
                        {
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque em Zona {z.kind}",
                            "descricao": (
                                f"Preço {preco_fmt} tocou {z.kind} "
                                f"{z.timeframe} [{low_fmt} ~ {high_fmt}]"
                            ),
                            "zone_context": z.to_dict(),
                            "preco_fechamento": preco_atual,
                            "timestamp": self.time_manager.now_utc_iso(
                                timespec="seconds"
                            ),
                        }
                    )

                    zone_event["janela_numero"] = self.window_count

                    if "historical_confidence" not in zone_event:
                        zone_event["historical_confidence"] = (
                            calcular_probabilidade_historica(zone_event)
                        )

                    self.event_bus.publish("zone_touch", zone_event)

                    institutional_zone_event = self._build_institutional_event(
                        zone_event
                    )
                    self.event_saver.save_event(institutional_zone_event)

                    adicionar_memoria_evento(
                        {
                            "timestamp": (
                                z.last_touched
                                or datetime.now(
                                    self.ny_tz
                                ).isoformat(timespec="seconds")
                            ),
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque {z.kind}",
                            "delta": zone_event.get("delta", 0.0),
                            "volume_total": zone_event.get(
                                "volume_total", 0.0
                            ),
                        }
                    )

            except Exception as e:
                logging.error(f"Erro ao verificar toques em zonas: {e}")

    def _update_histories(
        self, enriched: Dict[str, Any], ml_payload: Dict[str, Any]
    ) -> None:
        window_volume = enriched.get("volume_total", 0.0)
        window_delta = enriched.get("delta_fechamento", 0.0)
        window_close = enriched.get("ohlc", {}).get("close", 0.0)

        self.volume_history.append(window_volume)
        self.delta_history.append(window_delta)

        if window_close > 0:
            self.close_price_history.append(window_close)

        try:
            price_feats = ml_payload.get("price_features") or {}
            current_volatility = None

            if "volatility_5" in price_feats:
                current_volatility = price_feats["volatility_5"]
            elif "volatility_1" in price_feats:
                current_volatility = price_feats["volatility_1"]

            if current_volatility is not None:
                current_volatility = float(current_volatility)
                
                with self._history_lock:
                    self.volatility_history.append(current_volatility)

                try:
                    last_price = float(
                        enriched.get("ohlc", {}).get("close", 0.0) or window_close
                    )
                except Exception:
                    last_price = float(window_close or 0.0)

                if last_price > 0:
                    price_volatility_abs = current_volatility * last_price

                    try:
                        self.flow_analyzer.update_volatility_context(
                            atr_price=None,
                            price_volatility=price_volatility_abs,
                        )
                    except Exception as e:
                        logging.debug(
                            f"Falha ao atualizar contexto de volatilidade no FlowAnalyzer: {e}"
                        )

        except Exception as e:
            logging.debug(f"Falha ao atualizar histórico de volatilidade: {e}")

        try:
            ohlc = enriched.get("ohlc") or {}
            if ohlc:
                ts_open = int(ohlc.get("open_time") or ohlc.get("timestamp") or (time.time() * 1000))
                ts_close = int(ohlc.get("close_time") or (ts_open + 60000))
                # Volume canônico da candle: calculate_ohlc NÃO emite chave
                # "volume" (vive em volume_metrics.volume_total, base BTC).
                # Usar ohlc.get("volume", 0.0) aqui zerava todas as linhas e o
                # SessionVWAPTracker as rejeitava (volume<=0) — history parado.
                candle_volume = enriched.get("volume_total", 0.0)
                try:
                    candle_volume = float(candle_volume)
                except (TypeError, ValueError):
                    candle_volume = 0.0
                with self._history_lock:
                    self.pattern_ohlc_history.append(
                        {
                            "timestamp": ts_open,
                            "open_time": ts_open,
                            "close_time": ts_close,
                            "open": float(ohlc.get("open", ohlc.get("close", 0.0))),
                            "high": float(ohlc.get("high", 0.0)),
                            "low": float(ohlc.get("low", 0.0)),
                            "close": float(ohlc.get("close", 0.0)),
                            "volume": candle_volume,
                            "timeframe": "1m",
                            "is_closed": True,
                        }
                    )
        except Exception as e:
            logging.warning(
                f"Falha ao registrar OHLC em pattern_ohlc_history: {e!r}"
            )

    def _log_liquidity_heatmap(
        self, flow_metrics: Dict[str, Any]
    ) -> None:
        try:
            liquidity_data = flow_metrics.get("liquidity_heatmap", {})
            clusters = liquidity_data.get("clusters", [])

            if clusters:
                scope_size = liquidity_data.get("scope_size", "?")
                logging.info(
                    "📊 LIQUIDITY HEATMAP rolling(%s trades) @ Janela #%s:",
                    scope_size,
                    self.window_count,
                )

                for i, cluster in enumerate(clusters[:3]):
                    center_fmt = format_price(cluster.get("center", 0.0))
                    vol_fmt = format_large_number(
                        cluster.get("total_volume", 0.0)
                    )
                    imb_fmt = format_percent(
                        cluster.get("imbalance_ratio", 0.0) * 100.0
                    )
                    trades_fmt = format_quantity(
                        cluster.get("trades_count", 0)
                    )
                    age_fmt = format_time_seconds(cluster.get("age_ms", 0))

                    logging.info(
                        "  Cluster %d: $%s | Vol: %s | Imb: %s | Trades: %s | Age: %s",
                        i + 1,
                        center_fmt,
                        vol_fmt,
                        imb_fmt,
                        trades_fmt,
                        age_fmt,
                    )
        except Exception as e:
            logging.error(f"Erro ao logar liquidity heatmap: {e}")

    def _log_ml_features(self, ml_payload: Dict[str, Any]) -> None:
        try:
            pf = ml_payload.get("price_features", {}) if ml_payload else {}
            vf = ml_payload.get("volume_features", {}) if ml_payload else {}
            mf = ml_payload.get("microstructure", {}) if ml_payload else {}

            if pf or vf or mf:
                ret5_fmt = format_scientific(pf.get("returns_5", 0.0))
                vol5_fmt = format_scientific(
                    pf.get("volatility_5", 0.0), decimals=5
                )
                vsma_fmt = format_percent(
                    vf.get("volume_sma_ratio", 0.0) * 100.0
                )
                bs_fmt = format_delta(vf.get("buy_sell_pressure", 0.0))
                obs_fmt = format_scientific(
                    mf.get("order_book_slope", 0.0), decimals=3
                )
                flow_fmt = format_scientific(
                    mf.get("flow_imbalance", 0.0), decimals=3
                )

                logging.info(
                    "  ML: ret5=%s vol5=%s V/SMA=%s BSpress=%s OBslope=%s FlowImb=%s",
                    ret5_fmt,
                    vol5_fmt,
                    vsma_fmt,
                    bs_fmt,
                    obs_fmt,
                    flow_fmt,
                )
        except Exception:
            pass

    def _log_health_check(self) -> None:
        if self.window_count % 10 == 0:
            last_ob_age = (
                time.time() - self.last_valid_orderbook_time
                if self.last_valid_orderbook_time > 0
                else float("inf")
            )
            last_vp_age = (
                time.time() - self.last_valid_vp_time
                if self.last_valid_vp_time > 0
                else float("inf")
            )
            
            # Métricas do buffer de trades
            buffer_metrics = self.trades_buffer.get_stats()
            buffer_info = buffer_metrics.get('buffer', {})
            processing_info = buffer_metrics.get('processing', {})
            
            logging.info(
                f"\n📊 HEALTH CHECK - Janela #{self.window_count}:\n"
                f"  Orderbook: failures={self.orderbook_fetch_failures}, "
                f"last_valid={last_ob_age:.0f}s ago\n"
                f"  Value Area: last_valid={last_vp_age:.0f}s ago\n"
                f"  Trade Buffer: size={buffer_info.get('current_size', 0)}/"
                f"{buffer_info.get('capacity', 0)} "
                f"({buffer_info.get('fill_ratio', 0)*100:.1f}%) "
                f"status={buffer_info.get('status', 'unknown')}\n"
                f"  Processing: {processing_info.get('trades_per_second', 0):.1f} "
                f"trades/s, avg={processing_info.get('avg_time_ms', 0):.2f}ms"
            )

    def _log_window_summary(
        self,
        enriched: Dict[str, Any],
        historical_profile: Dict[str, Any],
        macro_context: Dict[str, Any],
    ) -> None:
        window_delta = enriched.get("delta_fechamento", 0.0)
        window_volume = enriched.get("volume_total", 0.0)

        delta_fmt = format_delta(window_delta)
        vol_fmt = format_large_number(window_volume)

        logging.info(
            "[%s NY] 🟡 Janela #%s | Delta: %s | Vol: %s",
            datetime.now(self.ny_tz).strftime("%H:%M:%S"),
            self.window_count,
            delta_fmt,
            vol_fmt,
        )

        if macro_context:
            trends = macro_context.get("mtf_trends", {})
            parts: List[str] = []
            for tf, data in trends.items():
                try:
                    parts.append(f"{tf.upper()}: {data['tendencia']}")
                except Exception:
                    parts.append(f"{tf.upper()}: {data}")
            trends_str = ", ".join(parts)
            if trends_str:
                logging.info("  Macro Context: %s", trends_str)

        if historical_profile and historical_profile.get("daily"):
            vp = historical_profile["daily"]
            poc_fmt = format_price(vp.get("poc", 0.0))
            val_fmt = format_price(vp.get("val", 0.0))
            vah_fmt = format_price(vp.get("vah", 0.0))

            logging.info(
                "  VP Diário: POC @ %s | VAL: %s | VAH: %s",
                poc_fmt,
                val_fmt,
                vah_fmt,
            )

        logging.info("─" * 80)

    def _process_institutional_alerts(
        self, enriched: Dict[str, Any], pipeline: DataPipeline
    ) -> None:
        if generate_alerts is None:
            return

        try:
            if detect_support_resistance is not None:
                try:
                    price_series = (
                        pipeline.df["p"]
                        if hasattr(pipeline, "df") and pipeline.df is not None
                        else None
                    )
                    if price_series is not None:
                        sr = detect_support_resistance(
                            price_series, num_levels=3
                        )
                    else:
                        sr = {
                            "immediate_support": [],
                            "immediate_resistance": [],
                        }
                except Exception:
                    sr = {
                        "immediate_support": [],
                        "immediate_resistance": [],
                    }
            else:
                sr = {
                    "immediate_support": [],
                    "immediate_resistance": [],
                }

            dz = None
            if defense_zones is not None:
                try:
                    dz = defense_zones(sr)
                except Exception:
                    dz = None

            window_close = enriched.get("ohlc", {}).get("close", 0.0)
            current_price_alert = window_close

            avg_vol = (
                sum(self.volume_history) / len(self.volume_history)
                if len(self.volume_history) > 0
                else enriched.get("volume_total", 0.0)
            )

            rec_vols = list(self.volatility_history)

            curr_vol = None
            try:
                if len(self.volatility_history) > 0:
                    curr_vol = self.volatility_history[-1]
            except Exception:
                curr_vol = None

            alerts_list = generate_alerts(
                price=current_price_alert,
                support_resistance=sr,
                current_volume=enriched.get("volume_total", 0.0),
                average_volume=avg_vol,
                current_volatility=curr_vol or 0.0,
                recent_volatilities=rec_vols,
                volume_threshold=3.0,
                tolerance_pct=0.001,
            )

            for alert in alerts_list or []:
                try:
                    atype = alert.get("type", "GENERIC")
                    now_s = time.time()
                    last_ts = self._last_alert_ts.get(atype, 0.0)

                    if now_s - last_ts < self._alert_cooldown_sec:
                        continue

                    self._last_alert_ts[atype] = now_s

                    desc_parts: List[str] = [f"Tipo: {alert.get('type')}"]

                    if "level" in alert:
                        desc_parts.append(
                            f"Nível: {format_price(alert['level'])}"
                        )

                    if "threshold_exceeded" in alert:
                        desc_parts.append(
                            f"Fator: {format_percent(alert['threshold_exceeded'] * 100.0)}"
                        )

                    descricao_alert = " | ".join(desc_parts)

                    print(f"🔔 ALERTA: {descricao_alert}")
                    logging.info(f"🔔 ALERTA: {descricao_alert}")

                    alert_event = {
                        "tipo_evento": "Alerta",
                        "resultado_da_batalha": alert.get("type"),
                        "descricao": descricao_alert,
                        "timestamp": self.time_manager.now_utc_iso(
                            timespec="seconds"
                        ),
                        "severity": alert.get("severity"),
                        "probability": alert.get("probability"),
                        "action": alert.get("action"),
                        "context": {
                            "price": current_price_alert,
                            "volume": enriched.get("volume_total", 0.0),
                            "average_volume": avg_vol,
                            "volatility": curr_vol or 0.0,
                        },
                        "data_context": "real_time",
                    }

                    alert_event["support_resistance"] = sr
                    if dz is not None:
                        alert_event["defense_zones"] = dz

                    alert_event["janela_numero"] = self.window_count
                    alert_event["epoch_ms"] = int(time.time() * 1000)

                    institutional_alert = self._build_institutional_event(
                        alert_event
                    )
                    self.event_saver.save_event(institutional_alert)

                except Exception as e:
                    logging.error(f"Erro ao processar alerta: {e}")

        except Exception as e:
            logging.error(f"Erro ao gerar alertas: {e}")

    # ========================================
    # CALLBACKS DO WEBSOCKET
    # ========================================
    def on_error(self, ws: Any, error: Exception) -> None:
        """Callback para erros de conexão WebSocket."""
        logging.error(
            f"❌ Erro na conexão WebSocket ({self.symbol}): {error}",
            exc_info=True,
        )
        try:
            if hasattr(self.health_monitor, "record_event"):
                self.health_monitor.record_event("ws_error")
            else:
                self.health_monitor.heartbeat("ws_error")
        except Exception as e:
            logging.debug(f"Erro ao registrar evento em on_error: {e}")

    def on_open(self, ws: Any) -> None:
        logging.info(
            f"🚀 Bot iniciado para {self.symbol} - "
            f"Fuso: New York (America/New_York)"
        )
        try:
            self.health_monitor.heartbeat("main")
        except Exception as e:
            logging.warning(f"Erro ignorado: {e}")

    def on_close(self, ws: Any, code: int, msg: str) -> None:
        if self.window_data and not self.should_stop:
            self._process_window()

    # ========================================
    # INITIALIZE (assíncrona)
    # ========================================
    async def initialize(self) -> None:
        """
        Inicialização assíncrona.
        Deve ser chamada APÓS o event loop estar rodando (ex.: dentro do asyncio.run(main())).
        """
        if self._initialized:
            return

        import asyncio
        self._loop = asyncio.get_running_loop()

        # Inicia o buffer de trades com loop já ativo.
        # O start() do AsyncTradeBuffer pode (internamente) criar tasks, mas agora isso é seguro.
        if self.trades_buffer is not None:
            await self.trades_buffer.start()

        # Inicia o WindowProcessor
        windows_min = [1, 5, 15]  # janelas desejadas
        self.window_processor = WindowProcessor(
            symbol=self.symbol,
            windows_minutes=windows_min,
            event_bus=self.event_bus,
            time_manager=self.time_manager,
            logger=logging.getLogger(__name__),
        )
        
        await self.window_processor.start()

        logging.info("✅ WindowProcessor iniciado | windows=%s", windows_min)

        # OnchainUpdater: refresh em background (não-bloqueante; a janela
        # nunca espera rede — lê snapshot, warming_up até o 1º fetch).
        try:
            if getattr(self, "onchain_updater", None) is not None:
                self.onchain_updater.start()
        except Exception as e:
            logging.warning(f"⚠️ Falha ao iniciar OnchainUpdater (não-crítico): {e}")

        # CrossAssetUpdater: mesmo padrão (E3-B).
        try:
            if getattr(self, "cross_asset_updater", None) is not None:
                self.cross_asset_updater.start()
        except Exception as e:
            logging.warning(f"⚠️ Falha ao iniciar CrossAssetUpdater (não-crítico): {e}")

        # P6: CFTC COT em background (não-bloqueante; janela lê read_view).
        try:
            if getattr(self, "cftc_cot_updater", None) is not None:
                self.cftc_cot_updater.start()
        except Exception as e:
            logging.warning(f"⚠️ Falha ao iniciar CftcCotUpdater (não-crítico): {e}")

        # Pre-popular histórico OHLC para habilitar indicadores avançados imediatamente
        await self._prefetch_ohlc_history()

        # Background task: re-sync periódico com Binance (non-blocking)
        self._periodic_sync_task = asyncio.create_task(
            self.time_manager.periodic_sync(600)
        )

        # Subscrição do ShadowPaperRuntime ao EventBus (Gate C3-C-B3-B)
        if (
            self.paper_shadow_status == "RUNNING"
            and self.shadow_runtime is not None
            and not self._shadow_subscribed
        ):
            self.event_bus.subscribe("signal", self.shadow_runtime.on_signal)
            self._shadow_subscribed = True
            logging.info("✅ ShadowPaperRuntime subscrito no EventBus para o evento 'signal'")

        self._initialized = True

    async def _prefetch_ohlc_history(self) -> None:
        """
        Pré-carrega pattern_ohlc_history com klines de 1m do Binance Futures
        desde 00:00 UTC (sessão completa, paginado) + últimos 200 no deque.
        Habilita Hurst, Kalman, Shannon, Regressão, Fourier, Fractal, Monte Carlo
        imediatamente (sem aguardar acumulação orgânica de 100-200 janelas).
        Também alimenta o SessionVWAPTracker com a sessão (dedup interno
        impede duplicação quando as janelas reenviarem as mesmas barras).
        Candle ainda aberto NUNCA entra como fechado. Falha parcial resulta
        em sessão PARTIAL (nunca FULL silencioso); falha total mantém o
        comportamento legado (warmup orgânico).
        """
        import aiohttp as _aiohttp
        from institutional.session_vwap import get_utc_session_start_ms
        url = "https://fapi.binance.com/fapi/v1/klines"
        try:
            now_ms = int(time.time() * 1000)
            session_start = get_utc_session_start_ms(now_ms)
            # Último minuto FECHADO (o minuto corrente ainda está formando).
            last_closed_open = (now_ms // 60000 - 1) * 60000
            all_klines: list = []
            async with _aiohttp.ClientSession() as _sess:
                cursor = session_start
                # Paginação limit=1000: 1 request até ~16h40, 2 requests/dia máx.
                while cursor <= last_closed_open:
                    params = {"symbol": self.symbol, "interval": "1m",
                              "startTime": cursor, "endTime": last_closed_open + 59999,
                              "limit": 1000}
                    async with _sess.get(url, params=params,
                                         timeout=_aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            logging.warning(
                                "⚠️  OHLC prefetch HTTP %s (parcial com %d barras)",
                                resp.status, len(all_klines),
                            )
                            break
                        data = await resp.json()
                    if not isinstance(data, list) or not data:
                        break
                    # Nunca incluir candle ainda aberto como fechado.
                    all_klines.extend([k for k in data if int(k[0]) <= last_closed_open])
                    last_ts = int(data[-1][0])
                    if last_ts <= cursor or len(data) < 1000:
                        break
                    cursor = last_ts + 60000
            for k in all_klines:
                self.pattern_ohlc_history.append({
                    "timestamp": int(k[0]),
                    "open_time": int(k[0]),
                    "close_time": int(k[6]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "timeframe": "1m",
                    "is_closed": True,
                })
            logging.info(
                "✅ OHLC history pré-carregado: %d barras 1m desde 00:00 UTC "
                "(deque: últimas %d; indicadores avançados e market structure ativos)",
                len(all_klines), len(self.pattern_ohlc_history),
            )
            # Prepara o tracker da sessão (idempotente via dedup por timestamp).
            try:
                tracker = getattr(
                    getattr(self, "institutional_analytics", None),
                    "session_vwap_tracker", None,
                )
                if tracker is not None and all_klines:
                    tracker.update_batch([
                        {"open_time": int(k[0]), "high": float(k[2]),
                         "low": float(k[3]), "close": float(k[4]),
                         "volume": float(k[5])}
                        for k in all_klines
                    ])
                    logging.info(
                        "✅ Session VWAP bootstrap: %d barras desde 00:00 UTC",
                        len(all_klines),
                    )
            except Exception as _e2:
                logging.warning("⚠️  Session VWAP bootstrap falhou (não-crítico): %s", _e2)
        except Exception as _e:
            logging.warning("⚠️  OHLC prefetch falhou (não-crítico): %s", _e)

    # ========================================
    # SHUTDOWN (assíncrono)
    # ========================================
    async def shutdown(self) -> None:
        """Shutdown limpo (chamar com await, dentro do loop). Idempotente."""
        if self._shutdown_async_lock is None:
            self._shutdown_async_lock = asyncio.Lock()

        async with self._shutdown_async_lock:
            if self._is_shutdown:
                logging.debug("Bot já foi encerrado (shutdown idempotente), ignorando chamada duplicada.")
                return
            self._is_shutdown = True

            # Marcar cleanup iniciado para evitar handler duplicado (atexit/signal)
            try:
                with self._cleanup_lock:
                    self._cleanup_started.set()
                    self.is_cleaning_up = True
            except Exception:
                pass

            self.should_stop = True

            # Cancela periodic_sync task
            if hasattr(self, "_periodic_sync_task") and self._periodic_sync_task:
                self._periodic_sync_task.cancel()

        # 1) Para geradores de trabalho primeiro
        try:
            if self.connection_manager:
                try:
                    self.connection_manager.should_stop = True
                except Exception:
                    pass
                try:
                    await self.connection_manager.disconnect()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.trades_buffer is not None:
                await self.trades_buffer.stop()
        except Exception as e:
            logging.warning(f"Falha ao parar trade buffer: {e}")

        try:
            if self.window_processor:
                await self.window_processor.stop()
        except Exception as e:
            logging.warning(f"Falha ao parar WindowProcessor: {e}")

        # 2) Fecha IA (async) antes do loop encerrar
        try:
            if self.ai_analyzer is not None:
                if hasattr(self.ai_analyzer, "aclose"):
                    await self.ai_analyzer.aclose()
                elif hasattr(self.ai_analyzer, "close"):
                    self.ai_analyzer.close()
        except Exception as e:
            logging.warning(f"Falha ao fechar AI Analyzer: {e}")

        # 3) Componentes síncronos
        try:
            if self.context_collector:
                self.context_collector.stop()
        except Exception:
            pass

        try:
            if getattr(self, "onchain_updater", None) is not None:
                self.onchain_updater.stop()
        except Exception:
            pass

        try:
            if getattr(self, "cross_asset_updater", None) is not None:
                self.cross_asset_updater.stop()
        except Exception:
            pass

        try:
            if getattr(self, "cftc_cot_updater", None) is not None:
                self.cftc_cot_updater.stop()
        except Exception:
            pass

        # PF-M3: ownership explícito das sessões MacroDataProvider.
        # Consumidores (onchain/cross/context) já parados acima; para o
        # MacroUpdateService e SÓ ENTÃO fecha as sessões. close_all_sessions
        # é idempotente, tolera sessão já fechada, esvazia o registry e
        # recria sob demanda (_get_session) se houver fetch tardio.
        try:
            from fetchers.macro_update_service import stop_macro_service
            await stop_macro_service()
        except Exception:
            pass

        try:
            from fetchers.macro_data_provider import get_macro_provider
            await get_macro_provider().close_all_sessions()
        except Exception:
            pass

        # Shadow Paper Trading Shutdown (Gate C3-C-B3-B)
        try:
            if getattr(self, "shadow_runtime", None) is not None:
                self.shadow_runtime.shutdown()
        except Exception as e:
            logging.warning(f"Falha ao encerrar shadow_runtime: {e}")

        try:
            if hasattr(self, "event_bus") and self.event_bus:
                self.event_bus.shutdown()
        except Exception:
            pass

        try:
            if hasattr(self, "health_monitor") and self.health_monitor:
                self.health_monitor.stop()
        except Exception:
            pass

        try:
            if (
                hasattr(self, "event_saver")
                and self.event_saver
                and hasattr(self.event_saver, "stop")
            ):
                self.event_saver.stop()
        except Exception:
            pass

        # 4) Encerrar loop asyncio dedicado do OrderBookAnalyzer + executor
        try:
            if (
                hasattr(self, "orderbook_analyzer")
                and self.orderbook_analyzer
                and hasattr(self.orderbook_analyzer, "close")
            ):
                # Aguarda a corrotina de fechamento do OrderBookAnalyzer
                await self.orderbook_analyzer.close()
        except Exception as e:
            logging.debug(f"Falha ao fechar OrderBookAnalyzer: {e}")

        try:
            if hasattr(self, "_async_loop"):
                try:
                    self._async_loop.call_soon_threadsafe(self._async_loop.stop)
                except Exception:
                    pass
                try:
                    if hasattr(self, "_async_loop_thread"):
                        self._async_loop_thread.join(timeout=2.0)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if hasattr(self, "_async_executor") and self._async_executor:
                self._async_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

        try:
            if hasattr(self, "feature_store") and self.feature_store is not None:
                self.feature_store.close()
        except Exception:
            pass

        try:
            if hasattr(self, "_raw_trades_file") and self._raw_trades_file is not None:
                with self._raw_trades_lock:
                    self._raw_trades_file.flush()
                    self._raw_trades_file.close()
                    self._raw_trades_file = None
                logging.info("💾 Arquivo de dump de trades brutos encerrado com sucesso.")
        except Exception:
            pass

    def shutdown_sync(self):
        """
        Shutdown síncrono (somente se for chamado fora do event loop / em outra thread).
        Retorna um Future do concurrent.futures.
        """
        import asyncio

        if self._loop is None:
            raise RuntimeError("Loop não definido: initialize() não foi executado")

        if self.trades_buffer is None:
            return None

        return asyncio.run_coroutine_threadsafe(self.trades_buffer.stop(), self._loop)

    # ========================================
    # RUN (versão assíncrona)
    # ========================================
    async def run(self) -> None:
        """
        Loop principal assíncrono do bot.

        Fica bloqueado enquanto o WebSocket estiver ativo.
        """
        try:
            self.context_collector.start()
            await self.initialize()
            
            logging.info(
                "🎯 Iniciando Enhanced Market Bot v2.3.2 "
                "(modo assíncrono, refatorado em módulos)..."
            )
            print("═" * 80)

            # RobustConnectionManager agora é totalmente assíncrono (aiohttp)
            await self.connection_manager.connect()

        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(
                f"❌ Erro crítico ao executar o bot: {e}",
                exc_info=True,
            )
        finally:
            # Garante que o gerenciador de conexão pare e feche o WS
            try:
                self.connection_manager.should_stop = True
            except Exception as e:
                logging.warning(f"Erro ignorado: {e}")

            try:
                # disconnect é async; aqui ainda estamos dentro do event loop
                await self.connection_manager.disconnect()
            except Exception as e:
                logging.error(
                    f"❌ Erro ao desconectar Connection Manager no run(): {e}",
                    exc_info=True,
                )

            await self.shutdown()
