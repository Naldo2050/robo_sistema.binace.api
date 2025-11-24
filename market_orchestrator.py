# market_orchestrator.py
# -*- coding: utf-8 -*-
"""
Orquestrador de mercado (WebSocket + janelas + DataPipeline + IA) v2.3.2

Extraído de main.py para separar:
- Entry point (main.py)
- Orquestração (este módulo)
"""

import json
import time
import logging
import threading
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import socket
import random
import websocket
from collections import deque
import signal
import atexit
import asyncio
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import config

# ✅ Clock Sync (opcional)
try:
    from clock_sync import get_clock_sync
    HAS_CLOCK_SYNC = True
except ImportError:
    HAS_CLOCK_SYNC = False
    logging.warning("⚠️ clock_sync.py não encontrado - timestamps usarão relógio local")

# Utilitários de formatação
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific,
)

# Importações internas
from data_handler import (
    create_absorption_event,
    create_exhaustion_event,
    NY_TZ,
)
from event_memory import (
    obter_memoria_eventos,
    adicionar_memoria_evento,
    calcular_probabilidade_historica,
)
from orderbook_analyzer import OrderBookAnalyzer
from event_saver import EventSaver
from context_collector import ContextCollector
from flow_analyzer import FlowAnalyzer
from ai_analyzer_qwen import AIAnalyzer
from report_generator import generate_ai_analysis_report  # pode ser usado futuramente
from levels_registry import LevelRegistry
from data_validator import validator

# Módulos avançados
from time_manager import TimeManager
from health_monitor import HealthMonitor
from event_bus import EventBus
from data_pipeline import DataPipeline
from feature_store import FeatureStore

# Alert engine
try:
    from alert_engine import generate_alerts
except Exception:
    generate_alerts = None

try:
    from support_resistance import detect_support_resistance, defense_zones
except Exception:
    detect_support_resistance = None
    defense_zones = None

try:
    from pattern_recognition import recognize_patterns
except Exception:
    recognize_patterns = None


# ===== FILTRO ANTI-ECO DE LOGS =====
class _DedupFilter(logging.Filter):
    """Filtro que suprime logs idênticos em janela de tempo."""

    def __init__(self, window: float = 1.0) -> None:
        super().__init__()
        self.window = float(window)
        self._last: Dict[str, float] = {}  # msg -> timestamp

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        now = time.time()
        ts = self._last.get(msg)

        if ts is not None and (now - ts) < self.window:
            return False

        self._last[msg] = now
        return True


logging.getLogger().addFilter(_DedupFilter(window=1.0))


# ===== RATE LIMITER =====
class RateLimiter:
    """Rate limiter thread-safe para controle de requisições."""

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Adquire permissão para fazer chamada (bloqueia se necessário)."""
        with self.lock:
            now = time.time()

            # Remove chamadas antigas
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()

            # Se atingiu limite, aguarda
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.period - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self.acquire()

            self.calls.append(now)


# ===== GESTOR DE CONEXÃO WEBSOCKET =====
class RobustConnectionManager:
    """Gerenciador robusto de conexão WebSocket com reconnect automático."""

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        max_reconnect_attempts: int = 10,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 1.5,
    ) -> None:
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False

        self.last_message_time: Optional[datetime] = None
        self.last_successful_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None

        self.heartbeat_thread: Optional[threading.Thread] = None
        self.should_stop = False

        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None

        self.total_messages_received = 0
        self.total_reconnects = 0

        self.external_heartbeat_cb = None

        self.on_reconnect_callback = None

    def set_callbacks(
        self,
        on_message=None,
        on_open=None,
        on_close=None,
        on_error=None,
        on_reconnect=None,
    ) -> None:
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def set_heartbeat_cb(self, cb) -> None:
        self.external_heartbeat_cb = cb

    def _test_connection(self) -> bool:
        """Testa conectividade TCP antes de tentar WebSocket."""
        try:
            parsed = urlparse(self.stream_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "wss" else 80)

            socket.getaddrinfo(host, port)

            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão TCP: {e}")
            return False

    def _on_message(self, ws, message: str) -> None:
        """Callback interno para mensagens recebidas."""
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1

            if self.on_message_callback:
                self.on_message_callback(ws, message)

            self.last_successful_message_time = self.last_message_time

            if self.current_delay > self.initial_delay:
                self.current_delay = max(
                    self.initial_delay,
                    self.current_delay * 0.9,
                )
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")

    def _on_open(self, ws) -> None:
        """Callback interno para conexão aberta."""
        self.is_connected = True

        if self.reconnect_count > 0 and self.on_reconnect_callback:
            try:
                self.on_reconnect_callback()
            except Exception as e:
                logging.error(f"Erro no callback de reconexão: {e}")

        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time
        self.last_successful_message_time = self.connection_start_time

        logging.info(f"✅ Conexão estabelecida com {self.symbol}")

        self._start_heartbeat()

        if self.on_open_callback:
            self.on_open_callback(ws)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.is_connected = False
        logging.warning(
            f"🔌 Conexão fechada - Código: {close_status_code}, Msg: {close_msg}"
        )

        self._stop_heartbeat()

        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)

    def _on_error(self, ws, error) -> None:
        logging.error(f"❌ Erro WebSocket: {error}")
        if self.on_error_callback:
            self.on_error_callback(ws, error)

    def _start_heartbeat(self) -> None:
        self.should_stop = False
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor,
            daemon=True,
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self.should_stop = True
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)

    def _heartbeat_monitor(self) -> None:
        while not self.should_stop and self.is_connected:
            time.sleep(20)

            if self.last_message_time:
                gap = (
                    datetime.now(timezone.utc) - self.last_message_time
                ).total_seconds()
                if gap > 60:
                    logging.warning(
                        f"⚠️ Sem mensagens há {gap:.0f}s. Forçando reconexão."
                    )
                    self.is_connected = False
                    break

            if self.last_successful_message_time:
                valid_gap = (
                    datetime.now(timezone.utc)
                    - self.last_successful_message_time
                ).total_seconds()
                if valid_gap > 120:
                    logging.critical(
                        f"💀 SEM MENSAGENS VÁLIDAS HÁ {valid_gap:.0f}s!"
                    )

    def _calculate_next_delay(self) -> float:
        delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        jitter = delay * 0.2 * (random.random() - 0.5)
        self.current_delay = max(self.initial_delay, delay + jitter)
        return self.current_delay

    def connect(self) -> None:
        ping_interval = getattr(config, "WS_PING_INTERVAL", 15)
        ping_timeout = getattr(config, "WS_PING_TIMEOUT", 8)

        while (
            self.reconnect_count < self.max_reconnect_attempts
            and not self.should_stop
        ):
            try:
                if self.external_heartbeat_cb:
                    try:
                        self.external_heartbeat_cb()
                    except Exception:
                        pass

                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")

                if self.reconnect_count > 0:
                    logging.info(
                        f"🔄 Tentativa de reconexão "
                        f"{self.reconnect_count + 1}/{self.max_reconnect_attempts}"
                    )

                ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                ws.run_forever(
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    skip_utf8_validation=True,
                )

                if self.should_stop:
                    break

            except KeyboardInterrupt:
                logging.info("⏹️ Interrompido pelo usuário")
                self.should_stop = True
                break

            except Exception as e:
                self.reconnect_count += 1
                self.total_reconnects += 1

                logging.error(
                    f"❌ Erro na conexão "
                    f"({self.reconnect_count}/{self.max_reconnect_attempts}): {e}"
                )

                if self.reconnect_count < self.max_reconnect_attempts:
                    delay = self._calculate_next_delay()
                    logging.info(f"⏳ Reconectando em {delay:.1f}s...")

                    t0 = time.time()
                    while time.time() - t0 < delay and not self.should_stop:
                        if self.external_heartbeat_cb:
                            try:
                                self.external_heartbeat_cb()
                            except Exception:
                                pass
                        time.sleep(1.0)
                else:
                    logging.error(
                        "💀 Máximo de tentativas atingido. Encerrando."
                    )
                    break

        self._stop_heartbeat()

    def disconnect(self) -> None:
        logging.info("🛑 Iniciando desconexão...")
        self.should_stop = True


# ===== ANALISADOR DE TRADE FLOW =====
class TradeFlowAnalyzer:
    """Analisador de fluxo de trades."""

    def __init__(self, vol_factor_exh: float, tz_output: ZoneInfo) -> None:
        self.vol_factor_exh = vol_factor_exh
        self.tz_output = tz_output

    def analyze_window(
        self,
        window_data: List[Dict[str, Any]],
        symbol: str,
        history_volumes: List[float],
        dynamic_delta_threshold: float,
        historical_profile: Optional[Dict[str, Any]] = None,
    ):
        if not window_data or len(window_data) < 2:
            return (
                {
                    "is_signal": False,
                    "delta": 0,
                    "volume_total": 0,
                    "preco_fechamento": 0,
                },
                {
                    "is_signal": False,
                    "delta": 0,
                    "volume_total": 0,
                    "preco_fechamento": 0,
                },
            )

        absorption_event = create_absorption_event(
            window_data,
            symbol,
            delta_threshold=dynamic_delta_threshold,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )

        exhaustion_event = create_exhaustion_event(
            window_data,
            symbol,
            history_volumes=list(history_volumes),
            volume_factor=self.vol_factor_exh,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )

        return absorption_event, exhaustion_event


# ===== GET CURRENT PRICE VIA REST =====
def get_current_price(symbol: str) -> float:
    """Obtém preço atual via REST API com retry."""
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            params = {"symbol": symbol}

            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()

            data = res.json()
            return float(data["price"])

        except requests.exceptions.RequestException as e:
            logging.error(
                f"Erro ao buscar preço via REST "
                f"(tentativa {attempt+1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))

        except Exception as e:
            logging.error(
                f"Erro inesperado ao buscar preço via REST "
                f"(tentativa {attempt+1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))

    logging.critical(
        "💀 FALHA CRÍTICA: Não foi possível obter preço via REST "
        "após todas as tentativas"
    )
    return 0.0


# ===== BOT PRINCIPAL =====
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
    ) -> None:
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False

        self._cleanup_lock = threading.Lock()
        self._ai_init_lock = threading.Lock()

        self.warming_up = False
        self.warmup_windows_remaining = 0
        self.warmup_windows_required = getattr(config, "WARMUP_WINDOWS", 3)

        self.trades_buffer: deque[Dict[str, Any]] = deque(
            maxlen=getattr(config, "TRADES_BUFFER_SIZE", 2000)
        )
        self.min_trades_for_pipeline = getattr(
            config, "MIN_TRADES_FOR_PIPELINE", 10
        )

        self.clock_sync = None
        if HAS_CLOCK_SYNC:
            try:
                self.clock_sync = get_clock_sync()
                logging.info("✅ Clock Sync inicializado")
            except Exception as e:
                logging.error(f"❌ Erro ao inicializar Clock Sync: {e}")

        self.time_manager = TimeManager()

        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.feature_store = FeatureStore()
        self.levels = LevelRegistry(self.symbol)

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

        # Executor para lidar com tarefas futuras CPU-bound sem bloquear o loop principal
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="orderbook_",
        )

        # Loop asyncio dedicado para o OrderBookAnalyzer (evita múltiplos asyncio.run)
        self._async_loop = asyncio.new_event_loop()
        self._async_loop_thread = threading.Thread(
            target=self._run_async_loop,
            name="orderbook_async_loop",
            daemon=True,
        )
        self._async_loop_thread.start()

        self.last_valid_vp: Optional[Dict[str, Any]] = None
        self.last_valid_vp_time: float = 0.0

        self.event_saver = EventSaver(sound_alert=True)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)

        self.ai_analyzer: Optional[AIAnalyzer] = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool: List[threading.Thread] = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self.ai_rate_limiter = RateLimiter(max_calls=10, period_seconds=60)

        self._initialize_ai_async()

        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)

        self.connection_manager = RobustConnectionManager(
            stream_url, symbol, max_reconnect_attempts=25
        )
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_reconnect=self._on_reconnect,
        )
        self.connection_manager.set_heartbeat_cb(
            lambda: self.health_monitor.heartbeat("main")
        )

        self.window_end_ms: Optional[int] = None
        self.window_data: List[Dict[str, Any]] = []
        self.window_count = 0

        self.history_size = history_size
        self.volume_history: deque[float] = deque(maxlen=history_size)
        self.delta_history: deque[float] = deque(maxlen=history_size)
        self.close_price_history: deque[float] = deque(
            maxlen=context_sma_period
        )
        self.delta_std_dev_factor = delta_std_dev_factor
        self.volatility_history: deque[float] = deque(maxlen=history_size)

        # Histórico de OHLC por janela para pattern recognition
        self.pattern_ohlc_history: deque[Dict[str, float]] = deque(
            maxlen=getattr(config, "PATTERN_LOOKBACK_BARS", 200)
        )

        self._missing_field_counts: Dict[str, int] = {
            "q": 0,
            "m": 0,
            "p": 0,
            "T": 0,
        }

        try:
            self._missing_field_log_step = getattr(
                config, "MISSING_FIELD_LOG_STEP", None
            )
        except Exception:
            self._missing_field_log_step = None

        self._last_price: Optional[float] = None
        self._last_alert_ts: Dict[str, float] = {}
        self._sent_triggers: set = set()

        self._last_ai_analysis_ts = 0.0
        self._ai_min_interval_sec = getattr(
            config, "AI_MIN_INTERVAL_SEC", 60
        )

        try:
            self._alert_cooldown_sec = getattr(
                config, "ALERT_COOLDOWN_SEC", 30
            )
        except Exception:
            self._alert_cooldown_sec = 30

        self._register_cleanup_handlers()

    # ========================================
    # HANDLER DE RECONEXÃO
    # ========================================
    def _on_reconnect(self) -> None:
        logging.warning(
            "🔄 RECONEXÃO DETECTADA - Iniciando período de aquecimento..."
        )

        self.warming_up = True
        self.warmup_windows_remaining = self.warmup_windows_required
        self.window_data = []
        self.window_end_ms = None

        logging.info(
            f"⏳ Aguardando {self.warmup_windows_required} janelas "
            f"para estabilizar dados..."
        )

    # ========================================
    # INICIALIZAÇÃO DA IA
    # ========================================
    def _initialize_ai_async(self) -> None:
        def ai_init_worker() -> None:
            try:
                with self._ai_init_lock:
                    if self.ai_initialization_attempted:
                        return
                    self.ai_initialization_attempted = True

                print("\n" + "=" * 30 + " INICIALIZANDO IA " + "=" * 30)
                logging.info("🧠 Tentando inicializar AI Analyzer...")

                self.ai_analyzer = AIAnalyzer()

                logging.info(
                    "✅ Módulo da IA carregado. Realizando teste de análise..."
                )

                current_price = get_current_price(self.symbol)

                test_event = {
                    "tipo_evento": "Teste de Conexão",
                    "ativo": self.symbol,
                    "descricao": (
                        "Teste inicial do sistema de análise "
                        "para garantir operacionalidade."
                    ),
                    "delta": 150.5,
                    "volume_total": 50000,
                    "preco_fechamento": current_price,
                    "orderbook_data": {
                        "bid_depth_usd": 1000000,
                        "ask_depth_usd": 800000,
                        "bid_usd": 1000000,
                        "ask_usd": 800000,
                        "imbalance": 0.2,
                        "mid": current_price,
                        "spread": 0.10,
                        "spread_percent": 0.0001,
                    },
                    "spread_metrics": {
                        "bid_depth_usd": 1000000,
                        "ask_depth_usd": 800000,
                    },
                }

                analysis = self.ai_analyzer.analyze(test_event)

                min_chars = getattr(config, "AI_TEST_MIN_CHARS", 10)

                if analysis and len(analysis.get("raw_response", "")) >= min_chars:
                    self.ai_test_passed = True
                    logging.info("✅ Teste da IA bem-sucedido!")

                    print(
                        "\n"
                        + "═" * 25
                        + " RESULTADO DO TESTE DA IA "
                        + "═" * 25
                    )
                    print(analysis.get("raw_response", ""))
                    print("═" * 75 + "\n")
                else:
                    self.ai_test_passed = True
                    logging.warning(
                        "⚠️ Teste da IA retornou resultado inesperado. "
                        "Prosseguindo em modo fallback."
                    )
                    print(f"Resultado recebido: {analysis}")
                    print("═" * 75 + "\n")

            except Exception as e:
                self.ai_analyzer = None
                self.ai_test_passed = False

                print("=" * 30 + " ERRO NA IA " + "=" * 30)
                logging.error(
                    f"❌ Falha crítica ao inicializar a IA: {e}",
                    exc_info=True,
                )
                print("═" * 75 + "\n")

        threading.Thread(target=ai_init_worker, daemon=True).start()

    # ========================================
    # CLEANUP
    # ========================================
    def _cleanup_handler(self, signum=None, frame=None) -> None:
        acquired = self._cleanup_lock.acquire(blocking=False)
        if not acquired:
            logging.debug("Cleanup já em andamento, ignorando chamada duplicada")
            return

        try:
            if self.is_cleaning_up:
                return
            self.is_cleaning_up = True
        finally:
            self._cleanup_lock.release()

        logging.info("🧹 Iniciando limpeza dos recursos...")
        self.should_stop = True

        cleanup_timeout = 5.0
        cleanup_threads: List[threading.Thread] = []

        def cleanup_context() -> None:
            try:
                if self.context_collector:
                    self.context_collector.stop()
                    logging.info("✅ Context Collector parado.")
            except Exception as e:
                logging.error(f"❌ Erro ao parar Context Collector: {e}")

        def cleanup_ai() -> None:
            try:
                if self.ai_analyzer and hasattr(self.ai_analyzer, "close"):
                    self.ai_analyzer.close()
                    logging.info("✅ AI Analyzer fechado.")
            except Exception as e:
                logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")

        def cleanup_connection() -> None:
            try:
                if self.connection_manager:
                    self.connection_manager.disconnect()
                    logging.info("✅ Connection Manager desconectado.")
            except Exception as e:
                logging.error(
                    f"❌ Erro ao desconectar Connection Manager: {e}"
                )

        def cleanup_event_bus() -> None:
            try:
                if hasattr(self, "event_bus"):
                    self.event_bus.shutdown()
                    logging.info("✅ Event Bus encerrado.")
            except Exception as e:
                logging.error(f"❌ Erro ao encerrar Event Bus: {e}")

        def cleanup_health_monitor() -> None:
            try:
                if hasattr(self, "health_monitor"):
                    self.health_monitor.stop()
                    logging.info("✅ Health Monitor parado.")
            except Exception as e:
                logging.error(f"❌ Erro ao parar Health Monitor: {e}")

        def cleanup_clock_sync() -> None:
            try:
                if self.clock_sync and hasattr(self.clock_sync, "stop"):
                    self.clock_sync.stop()
                    logging.info("✅ Clock Sync parado.")
            except Exception as e:
                logging.error(f"❌ Erro ao parar Clock Sync: {e}")

        def cleanup_executor() -> None:
            try:
                if hasattr(self, "_async_executor"):
                    self._async_executor.shutdown(wait=True)
                    logging.info("✅ Async Executor encerrado.")
            except Exception as e:
                logging.error(f"❌ Erro ao encerrar Async Executor: {e}")

        def cleanup_async_loop() -> None:
            """Encerra o loop asyncio dedicado do OrderBookAnalyzer com segurança."""
            try:
                if hasattr(self, "_async_loop"):
                    # Fecha o OrderBookAnalyzer dentro do loop
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
                                    "Timeout ao fechar OrderBookAnalyzer; "
                                    "cancelando tarefa pendente"
                                )
                                fut.cancel()
                            except Exception:
                                # Já estamos em shutdown, não queremos travar aqui
                                pass
                    except Exception as e:
                        logging.debug(f"Falha ao fechar OrderBookAnalyzer: {e}")

                    # Pede para o loop parar e aguarda a thread
                    try:
                        self._async_loop.call_soon_threadsafe(
                            self._async_loop.stop
                        )
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

        for cleanup_func in [
            cleanup_context,
            cleanup_ai,
            cleanup_connection,
            cleanup_event_bus,
            cleanup_health_monitor,
            cleanup_clock_sync,
            cleanup_executor,
            cleanup_async_loop,
        ]:
            t = threading.Thread(target=cleanup_func, daemon=True)
            t.start()
            cleanup_threads.append(t)

        for t in cleanup_threads:
            t.join(timeout=cleanup_timeout)

        logging.info("✅ Bot encerrado com segurança.")

    def _register_cleanup_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._cleanup_handler)
            signal.signal(signal.SIGTERM, self._cleanup_handler)
        except Exception:
            pass

        atexit.register(self._cleanup_handler)

    def _run_async_loop(self) -> None:
        """
        Loop de evento dedicado ao OrderBookAnalyzer.

        Roda em uma thread separada e permite que as corotinas do
        OrderBookAnalyzer sejam executadas via asyncio.run_coroutine_threadsafe
        sem recriar event loops.
        """
        asyncio.set_event_loop(self._async_loop)
        try:
            self._async_loop.run_forever()
        finally:
            # Tentativa de encerramento gracioso
            try:
                try:
                    pending = asyncio.all_tasks()
                except TypeError:
                    # Compatibilidade com versões mais antigas
                    pending = asyncio.Task.all_tasks()
            except Exception:
                pending = []

            for task in pending:
                task.cancel()

            if pending:
                try:
                    self._async_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass

            try:
                self._async_loop.run_until_complete(
                    self._async_loop.shutdown_asyncgens()
                )
            except Exception:
                pass

            self._async_loop.close()

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

        try:
            raw = json.loads(message)
            trade = raw.get("data", raw)

            p = trade.get("p") or trade.get("P") or trade.get("price")
            q = trade.get("q") or trade.get("Q") or trade.get("quantity")
            T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
            m = trade.get("m")

            if (p is None or q is None or T is None) and isinstance(
                trade.get("k"), dict
            ):
                k = trade["k"]
                p = p if p is not None else k.get("c")
                q = q if q is not None else k.get("v")
                T = T if T is not None else k.get("T") or raw.get("E")

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

            try:
                p = float(p)
                q = float(q)
                T = int(T)
            except (TypeError, ValueError):
                logging.error("Trade inválido (tipos): %s", trade)
                return

            if m is None:
                last_price = self._last_price
                m = (p <= last_price) if last_price is not None else False

            self._last_price = p

            norm = {"p": p, "q": q, "T": T, "m": bool(m)}

            self.trades_buffer.append(norm)

            try:
                self.health_monitor.heartbeat("main")
            except Exception as hb_err:
                logging.debug("Falha ao enviar heartbeat: %s", hb_err)

            self.flow_analyzer.process_trade(norm)

            if self.window_end_ms is None:
                self.window_end_ms = self._next_boundary_ms(T)

            if T >= self.window_end_ms:
                self._process_window()
                self.window_end_ms = self._next_boundary_ms(T)
                self.window_data = [norm]
            else:
                self.window_data.append(norm)

        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar mensagem JSON: {e}")
        except Exception as e:
            logging.error(f"Erro ao processar mensagem: {e}")

    # ========================================
    # HANDLERS DE EVENTOS / IA
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
            return False

        return False

    def _handle_signal_event(self, event_data: Dict[str, Any]) -> None:
        """Handler para eventos de sinal com bypass para críticos."""
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

        self._last_ai_analysis_ts = now
        self._run_ai_analysis_threaded(event_data.copy())

    def _handle_zone_touch_event(self, event_data: Dict[str, Any]) -> None:
        """Handler para eventos de toque em zona (considerados críticos)."""
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
    # ANÁLISE DA IA
    # ========================================
    def _run_ai_analysis_threaded(self, event_data: Dict[str, Any]) -> None:
        if not self.ai_analyzer or not self.ai_test_passed or self.should_stop:
            if self.ai_analyzer and not self.ai_test_passed:
                logging.warning(
                    "⚠️ Análise da IA ignorada: sistema não passou no teste inicial."
                )
            return

        logging.debug(
            "🔍 Evento recebido para análise da IA: %s",
            event_data.get("tipo_evento", "N/A"),
        )

        def _print_ai_report_clean(report_text: str) -> None:
            header = "ANÁLISE PROFISSIONAL DA IA"
            start = (report_text or "")[:200].upper()

            YELLOW = "\033[33m"
            RESET = "\033[0m"
            sep = "═" * 75

            if header in start:
                print("\n" + report_text.rstrip())
            else:
                print(
                    "\n" + "═" * 25 + " " + header + " " + "═" * 25
                )
                print(report_text)

            print(f"{YELLOW}{sep}{RESET}\n")

        def ai_worker() -> None:
            try:
                self.ai_rate_limiter.acquire()

                with self.ai_semaphore:
                    logging.info(
                        "🧠 IA iniciando análise para evento: %s",
                        event_data.get("resultado_da_batalha", "N/A"),
                    )

                    self.health_monitor.heartbeat("ai")

                    logging.debug(
                        "📊 Dados do evento para IA: %s",
                        {
                            "tipo": event_data.get("tipo_evento"),
                            "delta": format_delta(event_data.get("delta")),
                            "volume": format_large_number(
                                event_data.get("volume_total")
                            ),
                            "preco": format_price(
                                event_data.get("preco_fechamento")
                            ),
                        },
                    )

                    analysis_result = self.ai_analyzer.analyze(event_data)

                    if analysis_result and not self.should_stop:
                        try:
                            raw_response = analysis_result.get(
                                "raw_response", ""
                            )
                            _print_ai_report_clean(raw_response)
                            logging.info(
                                "✅ Análise da IA concluída com sucesso"
                            )
                        except Exception as e:
                            logging.error(
                                f"❌ Erro ao processar resposta da IA: {e}",
                                exc_info=True,
                            )

            except Exception as e:
                logging.error(
                    f"❌ Erro na thread de análise da IA: {e}",
                    exc_info=True,
                )
            finally:
                try:
                    self.ai_thread_pool = [
                        t for t in self.ai_thread_pool if t.is_alive()
                    ]
                except Exception:
                    pass

        logging.debug("🔧 Criando thread para análise da IA...")
        t = threading.Thread(target=ai_worker, daemon=True)
        self.ai_thread_pool.append(t)
        t.start()

    # ========================================
    # PROCESSAMENTO DE VP FEATURES
    # ========================================
    def _process_vp_features(
        self, historical_profile: Dict[str, Any], preco_atual: float
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
        # 1) Tentar via epoch_ms
        epoch_ms = e.get("epoch_ms") or (e.get("metadata") or {}).get("timestamp_unix_ms")
        if epoch_ms is not None:
            try:
                epoch_ms_int = int(epoch_ms)
                dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=timezone.utc)
                dt_ny = dt_utc.astimezone(self.ny_tz)
                return dt_ny.strftime("%Y-%m-%d %H:%M:%S NY")
            except Exception:
                pass

        # 2) Tentar via timestamp_ny / timestamp_utc / timestamp
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
                # Ajusta 'Z' para offset ISO (+00:00)
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    # se vier sem tz, assume UTC
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_ny = dt.astimezone(self.ny_tz)
                return dt_ny.strftime("%Y-%m-%d %H:%M:%S NY")
            except Exception:
                continue

        # 3) Fallback: timestamp bruto, sem 'Z'
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

        # pega últimos eventos da memória (exceto OrderBook)
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

                # Usa o helper para formatar o horário corretamente em NY
                ts_display = self._format_memory_timestamp_ny(e)

                print(
                    f"  - {ts_display} | "
                    f"{e.get('tipo_evento', 'N/A')} "
                    f"{e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={delta_fmt}, Vol={vol_fmt})"
                )

    # ========================================
    # PROCESSAMENTO DE JANELA
    # ========================================
    def _process_window(self) -> None:
        if not self.window_data or self.should_stop:
            self.window_data = []
            return

        if self.warming_up:
            self.warmup_windows_remaining -= 1

            logging.info(
                f"⏳ AQUECIMENTO: Janela processada "
                f"({self.warmup_windows_required - self.warmup_windows_remaining}/"
                f"{self.warmup_windows_required})"
            )

            if self.warmup_windows_remaining <= 0:
                self.warming_up = False
                logging.info("✅ AQUECIMENTO CONCLUÍDO - Sistema pronto!")

            self.window_data = []
            return

        valid_window_data: List[Dict[str, Any]] = []
        for trade in self.window_data:
            if "q" in trade and "p" in trade and "T" in trade:
                try:
                    trade["q"] = float(trade["q"])
                    trade["p"] = float(trade["p"])
                    trade["T"] = int(trade["T"])
                    valid_window_data.append(trade)
                except (ValueError, TypeError):
                    continue

        if len(valid_window_data) < self.min_trades_for_pipeline:
            logging.warning(
                f"⏳ Janela com apenas {len(valid_window_data)} trades "
                f"(mín: {self.min_trades_for_pipeline}). Aguardando mais dados..."
            )

            if len(self.trades_buffer) >= self.min_trades_for_pipeline:
                logging.info(
                    f"🔄 Recuperando {self.min_trades_for_pipeline} trades "
                    f"do buffer de emergência..."
                )
                valid_window_data = list(self.trades_buffer)[
                    -self.min_trades_for_pipeline :
                ]
            else:
                self.window_data = []
                return

        total_volume = sum(
            float(trade.get("q", 0)) for trade in valid_window_data
        )
        if total_volume == 0:
            self.window_data = []
            return

        self.window_count += 1

        try:
            quantities = np.array(
                [t.get("q", 0) for t in valid_window_data], dtype=float
            )
            is_sell = np.array(
                [t.get("m", False) for t in valid_window_data], dtype=bool
            )

            total_sell_volume = float(quantities[is_sell].sum())
            total_buy_volume = float(quantities[~is_sell].sum())
        except Exception as e:
            logging.error(
                f"Erro ao calcular volumes com NumPy, usando fallback: {e}"
            )
            total_buy_volume = 0.0
            total_sell_volume = 0.0
            for trade in valid_window_data:
                if trade.get("m"):
                    total_sell_volume += float(trade.get("q", 0))
                else:
                    total_buy_volume += float(trade.get("q", 0))

        try:
            self.health_monitor.heartbeat("main")

            dynamic_delta_threshold = 0.0
            if len(self.delta_history) > 10:
                mean_delta = float(np.mean(self.delta_history))
                std_delta = float(np.std(self.delta_history))
                dynamic_delta_threshold = abs(
                    mean_delta + self.delta_std_dev_factor * std_delta
                )

            macro_context = self.context_collector.get_context()
            historical_profile = macro_context.get("historical_vp", {})

            vp_daily = historical_profile.get("daily", {})
            val = vp_daily.get("val", 0)
            vah = vp_daily.get("vah", 0)
            poc = vp_daily.get("poc", 0)

            if val == 0 or vah == 0 or poc == 0:
                if self.last_valid_vp and (
                    time.time() - self.last_valid_vp_time < 3600
                ):
                    age = time.time() - self.last_valid_vp_time
                    logging.warning(
                        f"⚠️ Value Area zerada, usando cache (age={age:.0f}s)"
                    )
                    historical_profile = self.last_valid_vp.copy()
                else:
                    logging.warning(
                        "⚠️ Value Area indisponível e sem cache válido"
                    )
            else:
                self.last_valid_vp = historical_profile.copy()
                self.last_valid_vp_time = time.time()

            close_ms = self.window_end_ms

            self.levels.update_from_vp(historical_profile)

            try:
                pipeline = DataPipeline(
                    valid_window_data,
                    self.symbol,
                    time_manager=self.time_manager,
                )
            except ValueError as ve:
                logging.error(
                    f"❌ Erro ao criar pipeline (janela #{self.window_count}): {ve}"
                )
                self.window_data = []
                return

            flow_metrics = self.flow_analyzer.get_flow_metrics(
                reference_epoch_ms=close_ms
            )

            ob_event = self._fetch_orderbook_with_retry(close_ms)

            enriched = pipeline.enrich()

            pipeline.add_context(
                flow_metrics=flow_metrics,
                historical_vp=historical_profile,
                orderbook_data=ob_event,
                multi_tf=macro_context.get("mtf_trends", {}),
                derivatives=macro_context.get("derivatives", {}),
                market_context=macro_context.get("market_context", {}),
                market_environment=macro_context.get("market_environment", {}),
            )

            signals = pipeline.detect_signals(
                absorption_detector=lambda data, sym: create_absorption_event(
                    data,
                    sym,
                    delta_threshold=dynamic_delta_threshold,
                    tz_output=self.ny_tz,
                    flow_metrics=flow_metrics,
                    historical_profile=historical_profile,
                    time_manager=self.time_manager,
                    event_epoch_ms=close_ms,
                ),
                exhaustion_detector=lambda data, sym: create_exhaustion_event(
                    data,
                    sym,
                    history_volumes=list(self.volume_history),
                    volume_factor=config.VOL_FACTOR_EXH,
                    tz_output=self.ny_tz,
                    flow_metrics=flow_metrics,
                    historical_profile=historical_profile,
                    time_manager=self.time_manager,
                    event_epoch_ms=close_ms,
                ),
                orderbook_data=ob_event,
            )

            self._process_signals(
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

        except Exception as e:
            logging.error(
                f"Erro no processamento da janela #{self.window_count}: {e}",
                exc_info=True,
            )
        finally:
            try:
                if "pipeline" in locals() and hasattr(pipeline, "close"):
                    pipeline.close()
            except Exception as e:
                logging.debug(f"Falha ao fechar pipeline: {e}")
            self.window_data = []

    # ========================================
    # WRAPPER ORDERBOOK (SYNC/ASYNC + EXECUTOR)
    # ========================================
    def _run_orderbook_analyze(
        self, close_ms: int
    ) -> Optional[Dict[str, Any]]:
        """
        Executa OrderBookAnalyzer.analyze() no loop asyncio dedicado.

        Evita criar um novo event loop com asyncio.run() a cada chamada
        e reaproveita a ClientSession do OrderBookAnalyzer ligada a esse loop.
        Garante também que, em caso de timeout, a coroutine seja cancelada
        explicitamente para não ficar rodando como "zumbi" no loop.
        """
        # Se o bot está em processo de shutdown, não agendar novas corotinas
        if self.should_stop or getattr(self, "is_cleaning_up", False):
            return None

        now_ms = int(time.time() * 1000)
        if close_ms <= 0 or close_ms > now_ms + 60_000:
            logging.error(f"❌ close_ms inválido: {close_ms}")
            return None

        loop = getattr(self, "_async_loop", None)
        if loop is None:
            logging.error("❌ Loop assíncrono do OrderBookAnalyzer não inicializado")
            return None

        if loop.is_closed():
            logging.error("❌ Loop assíncrono do OrderBookAnalyzer já foi fechado")
            return None

        try:
            coro = self.orderbook_analyzer.analyze(
                current_snapshot=None,
                event_epoch_ms=close_ms,
                window_id=f"W{self.window_count:04d}",
            )

            # Envia a coroutine para o loop assíncrono dedicado (em outra thread)
            future = asyncio.run_coroutine_threadsafe(
                coro,
                loop,
            )

            try:
                # Espera o resultado com timeout (mesmo 5s que você já usava)
                return future.result(timeout=5.0)
            except FutureTimeoutError:
                # ⚠️ Ponto crítico: cancela explicitamente a tarefa no loop
                logging.error(
                    "⏱️ Timeout ao buscar orderbook (async loop) - "
                    "cancelando coroutine pendente"
                )
                future.cancel()
                return None

        except Exception as e:
            logging.error(
                f"❌ Erro ao buscar orderbook (async loop): {e}",
                exc_info=True,
            )
            return None

    # ========================================
    # FETCH ORDERBOOK NÃO-BLOQUEANTE
    # ========================================
    def _fetch_orderbook_with_retry(self, close_ms: int) -> Dict[str, Any]:
        try:
            ob_event = self._run_orderbook_analyze(close_ms)
            if ob_event and ob_event.get("is_valid", False):
                ob_data = ob_event.get("orderbook_data", {}) or {}
                bid_depth = float(ob_data.get("bid_depth_usd", 0.0))
                ask_depth = float(ob_data.get("ask_depth_usd", 0.0))
                min_depth = float(
                    getattr(config, "ORDERBOOK_MIN_DEPTH_USD", 500.0)
                )
                if bid_depth >= min_depth or ask_depth >= min_depth:
                    self.last_valid_orderbook = ob_event
                    self.last_valid_orderbook_time = time.time()
                    self.orderbook_fetch_failures = 0
                    logging.debug(
                        f"✅ Orderbook OK - Janela #{self.window_count}"
                    )
                    return ob_event
                else:
                    logging.warning(
                        "⚠️ Orderbook com liquidez baixa (best-effort)"
                    )
        except Exception as e:
            logging.error(
                f"❌ Erro ao buscar orderbook (best-effort): {e}"
            )

        fallback = self._orderbook_fallback()
        self._refresh_orderbook_async(close_ms)
        return fallback

    def _refresh_orderbook_async(self, close_ms: int) -> None:
        if not self._orderbook_background_refresh:
            return
        now = time.time()
        if now - self._last_async_ob_refresh < self._orderbook_bg_min_interval:
            return
        with self._orderbook_refresh_lock:
            if (
                self._orderbook_refresh_thread
                and self._orderbook_refresh_thread.is_alive()
            ):
                return
            self._last_async_ob_refresh = now

            def _worker() -> None:
                try:
                    evt = self._run_orderbook_analyze(close_ms)
                    if evt and evt.get("is_valid", False):
                        self.last_valid_orderbook = evt
                        self.last_valid_orderbook_time = time.time()
                        self.orderbook_fetch_failures = 0
                        logging.info(
                            "♻️ Orderbook cache atualizado em background"
                        )
                except Exception as e:
                    logging.debug(
                        f"Falha na atualização assíncrona do orderbook: {e}"
                    )

            self._orderbook_refresh_thread = threading.Thread(
                target=_worker, daemon=True
            )
            self._orderbook_refresh_thread.start()

    def _orderbook_fallback(self) -> Dict[str, Any]:
        self.orderbook_fetch_failures += 1
        fallback_max_age = getattr(
            config, "ORDERBOOK_FALLBACK_MAX_AGE", 600
        )

        if self.last_valid_orderbook and (
            time.time() - self.last_valid_orderbook_time < fallback_max_age
        ):
            age = time.time() - self.last_valid_orderbook_time
            logging.warning(
                f"⚠️ Usando orderbook em cache (age={age:.0f}s) "
                f"após {self.orderbook_fetch_failures} falhas"
            )

            ob_event = self.last_valid_orderbook.copy()
            ob_event["data_quality"] = {
                "is_valid": True,
                "data_source": "cache",
                "age_seconds": age,
            }
            return ob_event

        if self.orderbook_emergency_mode:
            logging.warning(
                f"🚨 MODO EMERGÊNCIA: Orderbook indisponível "
                f"(falhas: {self.orderbook_fetch_failures})"
            )

            return {
                "is_valid": True,
                "emergency_mode": True,
                "orderbook_data": {
                    "bid_depth_usd": 1000.0,
                    "ask_depth_usd": 1000.0,
                    "imbalance": 0.0,
                    "mid": 0.0,
                    "spread": 0.0,
                },
                "spread_metrics": {
                    "bid_depth_usd": 1000.0,
                    "ask_depth_usd": 1000.0,
                },
                "data_quality": {
                    "is_valid": True,
                    "data_source": "emergency",
                    "error": (
                        f"Emergency mode after "
                        f"{self.orderbook_fetch_failures} failures"
                    ),
                },
            }

        logging.error(
            f"❌ Orderbook totalmente indisponível "
            f"(falhas consecutivas: {self.orderbook_fetch_failures})"
        )

        return {
            "is_valid": False,
            "should_skip": False,
            "orderbook_data": {
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
                "imbalance": 0.0,
                "mid": 0.0,
                "spread": 0.0,
            },
            "spread_metrics": {
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
            },
            "data_quality": {
                "is_valid": False,
                "data_source": "error",
                "error": "Failed after max retries",
            },
        }

    # ========================================
    # PROCESSAMENTO DE SINAIS
    # ========================================
    def _process_signals(
        self,
        signals: List[Dict[str, Any]],
        pipeline: DataPipeline,
        flow_metrics: Dict[str, Any],
        historical_profile: Dict[str, Any],
        macro_context: Dict[str, Any],
        ob_event: Dict[str, Any],
        enriched: Dict[str, Any],
        close_ms: int,
        total_buy_volume: float,
        total_sell_volume: float,
        valid_window_data: List[Dict[str, Any]],
    ) -> None:
        has_real_signal = any(
            s.get("tipo_evento") not in ("ANALYSIS_TRIGGER", "OrderBook")
            for s in signals
        )

        if not signals or not has_real_signal:
            trigger_signal: Dict[str, Any] = {
                "is_signal": True,
                "tipo_evento": "ANALYSIS_TRIGGER",
                "resultado_da_batalha": "N/A",
                "descricao": "Evento automático para análise da IA",
                "timestamp": self.time_manager.now_utc_iso(
                    timespec="seconds"
                ),
                "delta": enriched.get("delta_fechamento", 0.0),
                "volume_total": enriched.get("volume_total", 0.0),
                "preco_fechamento": enriched.get("ohlc", {}).get("close", 0.0),
                "epoch_ms": close_ms,
                "ml_features": pipeline.get_final_features().get(
                    "ml_features", {}
                ),
                "orderbook_data": ob_event,
                "historical_vp": historical_profile,
                "multi_tf": macro_context.get("mtf_trends", {}),
                "data_context": "real_time",
            }

            if not signals:
                signals.append(trigger_signal)
            elif not has_real_signal:
                signals = [
                    s
                    for s in signals
                    if s.get("tipo_evento") != "ANALYSIS_TRIGGER"
                ]
                signals.append(trigger_signal)

        self._log_liquidity_heatmap(flow_metrics)

        features = pipeline.get_final_features()
        self.feature_store.save_features(
            window_id=str(close_ms),
            features=features,
        )

        ml_payload = features.get("ml_features", {}) or {}
        enriched_snapshot = features.get("enriched", {}) or {}
        contextual_snapshot = features.get("contextual", {}) or {}

        derivatives_context = macro_context.get("derivatives", {})

        # ----------------------------------------
        # PATTERN RECOGNITION & PRICE TARGETS
        # ----------------------------------------
        pattern_recognition_data: Dict[str, Any] = {}
        price_targets: Dict[str, Any] = {}

        if recognize_patterns is not None:
            try:
                current_ohlc = enriched_snapshot.get("ohlc") or {}
                bars: List[Dict[str, float]] = list(self.pattern_ohlc_history)

                if current_ohlc:
                    bars.append(
                        {
                            "high": float(current_ohlc.get("high", 0.0)),
                            "low": float(current_ohlc.get("low", 0.0)),
                            "close": float(current_ohlc.get("close", 0.0)),
                        }
                    )

                if len(bars) >= 3:
                    df_patterns = pd.DataFrame(bars)
                    pattern_recognition_data = recognize_patterns(df_patterns) or {}

                    last_price = float(current_ohlc.get("close", 0.0)) if current_ohlc else 0.0
                    price_targets = self._build_price_targets(
                        pattern_recognition_data,
                        last_price,
                    )
            except Exception as e:
                logging.debug(f"Falha em pattern_recognition: {e}")

        # ----------------------------------------
        # SUPORTE / RESISTÊNCIA E DEFENSE ZONES
        # ----------------------------------------
        support_resistance: Dict[str, Any] = {}
        defense_zones_data: Dict[str, Any] = {}

        if detect_support_resistance is not None:
            try:
                price_series = (
                    pipeline.df["p"]
                    if hasattr(pipeline, "df") and pipeline.df is not None
                    else None
                )
                if price_series is not None:
                    support_resistance = detect_support_resistance(
                        price_series, num_levels=3
                    )
                    if defense_zones is not None:
                        defense_zones_data = defense_zones(
                            support_resistance
                        )
            except Exception as e:
                logging.debug(
                    f"Falha ao calcular suporte/resistência: {e}"
                )

        logging.info(
            f"📊 JANELA #{self.window_count} - "
            f"Processando {len(signals)} sinal(is):"
        )
        for i, sig in enumerate(signals, 1):
            logging.info(
                f"  {i}. {sig.get('tipo_evento')} / "
                f"{sig.get('resultado_da_batalha', 'N/A')} | "
                f"delta={sig.get('delta', 0):.2f} | "
                f"volume={sig.get('volume_total', 0):.2f}"
            )

        for signal in signals:
            if signal.get("is_signal", False):
                # Anexa patterns/targets no próprio sinal
                if pattern_recognition_data:
                    signal["pattern_recognition"] = pattern_recognition_data
                if price_targets:
                    signal["price_targets"] = price_targets

                self._enrich_signal(
                    signal,
                    derivatives_context,
                    flow_metrics,
                    total_buy_volume,
                    total_sell_volume,
                    macro_context,
                    close_ms,
                    ml_payload,
                    enriched_snapshot,
                    contextual_snapshot,
                    ob_event,
                    valid_window_data,
                    support_resistance,
                    defense_zones_data,
                )

        self._check_zone_touches(enriched, signals)
        self._update_histories(enriched, ml_payload)
        self._log_ml_features(ml_payload)
        self._process_institutional_alerts(enriched, pipeline)
        self._log_health_check()
        self._log_window_summary(enriched, historical_profile, macro_context)

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

        # ✅ Marca a qual janela este sinal pertence (para EventSaver)
        signal.setdefault("janela_numero", self.window_count)

        if "epoch_ms" not in signal:
            signal["epoch_ms"] = close_ms

        if "timestamp_utc" not in signal:
            signal["timestamp_utc"] = self.time_manager.from_timestamp_ms(
                close_ms, tz=self.time_manager.tz_utc
            ).isoformat(timespec="milliseconds")

        if "timestamp" not in signal:
            signal["timestamp"] = self.time_manager.from_timestamp_ms(
                close_ms, tz=self.ny_tz
            ).strftime("%Y-%m-%d %H:%M:%S")

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
        except Exception:
            pass

        signal.setdefault("features_window_id", str(close_ms))
        signal["ml_features"] = ml_payload
        signal["enriched_snapshot"] = enriched_snapshot
        signal["contextual_snapshot"] = contextual_snapshot

        # Blocos explícitos de suporte/resistência e zonas de defesa
        if support_resistance:
            signal["support_resistance"] = support_resistance
        if defense_zones_data:
            signal["defense_zones"] = defense_zones_data

        # ------- Orderbook + métricas avançadas -------
        if ob_event and isinstance(ob_event, dict) and ob_event.get(
            "is_valid", False
        ):
            # L1 básico
            if "orderbook_data" in ob_event:
                signal["orderbook_data"] = ob_event["orderbook_data"]
            if "spread_metrics" in ob_event:
                signal["spread_metrics"] = ob_event["spread_metrics"]

            # Profundidade multi-level
            if "order_book_depth" in ob_event:
                signal["order_book_depth"] = ob_event["order_book_depth"]

            # Spread analysis completo
            if "spread_analysis" in ob_event:
                signal["spread_analysis"] = ob_event["spread_analysis"]

            # Market impact → market_impact (schema final)
            try:
                mi_buy = ob_event.get("market_impact_buy", {}) or {}
                mi_sell = ob_event.get("market_impact_sell", {}) or {}

                def _get_move(mi_dict, key):
                    d = mi_dict.get(key, {}) or {}
                    return d.get("move_usd")

                slippage_matrix = {
                    "1k_usd":   {"buy": _get_move(mi_buy,  "1k"),  "sell": _get_move(mi_sell,  "1k")},
                    "10k_usd":  {"buy": _get_move(mi_buy, "10k"),  "sell": _get_move(mi_sell, "10k")},
                    "100k_usd": {"buy": _get_move(mi_buy,"100k"),  "sell": _get_move(mi_sell,"100k")},
                    "1m_usd":   {"buy": _get_move(mi_buy,  "1M"),  "sell": _get_move(mi_sell,  "1M")},
                }

                # Heurística simples de liquidez com base no bps de 100k
                bps_100k_buy = (mi_buy.get("100k") or {}).get("bps")
                bps_100k_sell = (mi_sell.get("100k") or {}).get("bps")
                bps_list = [
                    v for v in (bps_100k_buy, bps_100k_sell)
                    if isinstance(v, (int, float))
                ]
                if bps_list:
                    avg_bps = float(sum(bps_list) / len(bps_list))
                    # Quanto menor o bps, maior o score (0–10)
                    liquidity_score = max(0.0, min(10.0, 10.0 - avg_bps / 5.0))
                else:
                    liquidity_score = None

                if liquidity_score is not None:
                    if liquidity_score >= 8:
                        execution_quality = "EXCELLENT"
                    elif liquidity_score >= 6:
                        execution_quality = "GOOD"
                    elif liquidity_score >= 4:
                        execution_quality = "FAIR"
                    else:
                        execution_quality = "POOR"
                else:
                    execution_quality = None

                signal["market_impact"] = {
                    "slippage_matrix": slippage_matrix,
                    "liquidity_score": liquidity_score,
                    "execution_quality": execution_quality,
                }
            except Exception as e:
                logging.debug(f"Falha ao construir market_impact: {e}")

        # ---------------------------------------------

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

        # ✅ EventBus continua recebendo o evento bruto (para IA / assinantes)
        self.event_bus.publish("signal", signal)

        # ✅ Evento institucional normalizado para persistência / análise
        institutional_event = self._build_institutional_event(signal)
        self.event_saver.save_event(institutional_event)

        if signal.get("tipo_evento") != "OrderBook":
            adicionar_memoria_evento(signal)

        self._log_event(signal)

    # ========================================
    # BUILDER DE EVENTO INSTITUCIONAL
    # ========================================
    def _build_institutional_event(
        self, signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Converte o 'signal' atual (formato legado) em um evento
        institucional organizado em blocos (metadata, price_data, etc.),
        preservando o original em 'raw_event'.
        """

        # --------- BASE / SAFETY ---------
        fluxo = signal.get("fluxo_continuo") or {}
        time_index = fluxo.get("time_index") or {}
        dq_flow = fluxo.get("data_quality") or {}
        es = signal.get("enriched_snapshot") or {}
        ohlc = es.get("ohlc") or {}
        orderbook = signal.get("orderbook_data") or {}
        spread_metrics = signal.get("spread_metrics") or {}
        order_book_depth_block = signal.get("order_book_depth") or {}
        spread_analysis_block = signal.get("spread_analysis") or {}
        market_impact_block = signal.get("market_impact") or {}
        hist_vp = signal.get("historical_vp") or {}
        vp_daily = hist_vp.get("daily") or {}
        participants = fluxo.get("participant_analysis") or {}
        volume_nodes_block = vp_daily.get("volume_nodes") or {}
        multi_tf = (
            signal.get("multi_tf")
            or (signal.get("contextual_snapshot") or {}).get("multi_tf")
            or {}
        )
        ml = signal.get("ml_features") or {}
        ml_pf = ml.get("price_features") or {}

        support_resistance_block = signal.get("support_resistance") or {}
        defense_zones_block = signal.get("defense_zones") or {}
        pattern_block = signal.get("pattern_recognition") or {}
        price_targets_block = signal.get("price_targets") or {}

        # --------- METADATA ---------
        epoch_ms = signal.get("epoch_ms") or time_index.get("epoch_ms")
        timestamp_utc = (
            signal.get("timestamp_utc")
            or time_index.get("timestamp_utc")
            or signal.get("timestamp")
        )

        exchange_epoch_ms = time_index.get("epoch_ms", epoch_ms)

        # Latência aproximada
        latency_ms = None
        try:
            if epoch_ms and exchange_epoch_ms:
                latency_ms = max(0, int(epoch_ms) - int(exchange_epoch_ms))
        except Exception:
            latency_ms = None

        valid_rate = dq_flow.get("valid_rate_pct")
        completeness_pct = float(valid_rate) if valid_rate is not None else None

        # Heurística simples 0–10 para confiabilidade
        reliability_score = None
        try:
            trades_cnt = dq_flow.get("flow_trades_count", 0) or 0
            if trades_cnt and valid_rate is not None:
                reliability_score = min(
                    10.0,
                    (trades_cnt / 500.0) * (valid_rate / 100.0),
                )
        except Exception:
            reliability_score = None

        symbol = (
            signal.get("ativo")
            or signal.get("symbol")
            or self.symbol
        )

        data_context_value = signal.get("data_context", "real_time")

        metadata = {
            "timestamp_utc": timestamp_utc,
            "timestamp_unix_ms": epoch_ms,
            "sequence_id": epoch_ms,
            "exchange_timestamp": (
                time_index.get("timestamp_utc")
                or time_index.get("timestamp_ny")
            ),
            "latency_ms": latency_ms,
            "data_quality_score": valid_rate,
            "completeness_pct": completeness_pct,
            "reliability_score": reliability_score,
            "symbol": symbol,
            "event_type": signal.get("tipo_evento"),
            "battle_outcome": signal.get("resultado_da_batalha"),
            "window_id": signal.get("features_window_id") or epoch_ms,
            "window_index": signal.get("janela_numero"),
            "data_context": data_context_value,
        }

        # --------- DATA SOURCE ---------
        src = signal.get("source") or {}
        data_source = {
            "primary_exchange": src.get("exchange", "binance_futures"),
            "data_feed_type": src.get("stream", "trades"),
            "validation_passed": True,
            "anomaly_detected": False,
        }

        # --------- MARKET CONTEXT / ENV ---------
        market_context = signal.get("market_context") or {}
        market_environment = signal.get("market_environment") or {}

        # --------- PRICE DATA ---------
        last_price = signal.get("preco_fechamento") or ohlc.get("close")
        volume_total = signal.get("volume_total", es.get("volume_total"))

        price_data = {
            "current": {
                "last": last_price,
                "mid": orderbook.get("mid"),
                "spread": orderbook.get("spread"),
                "volume": volume_total,
            },
            "session": {
                "open": ohlc.get("open"),
                "high": ohlc.get("high"),
                "low": ohlc.get("low"),
                "close": ohlc.get("close"),
                "vwap": ohlc.get("vwap"),
                "num_trades": es.get("num_trades"),
                "window_duration_s": signal.get("duration_s"),
            },
            "previous_periods": {
                "multi_tf": multi_tf,
            },
        }

        # --------- VOLUME PROFILE ---------
        volume_profile = {
            "poc_price": es.get("poc_price") or vp_daily.get("poc"),
            "poc_volume": es.get("poc_volume"),
            "poc_percentage": es.get("poc_percentage"),
            "vah": vp_daily.get("vah") or signal.get("vah"),
            "val": vp_daily.get("val") or signal.get("val"),
            "hvns": signal.get("hvns") or vp_daily.get("hvns"),
            "lvns": signal.get("lvns") or vp_daily.get("lvns"),
            "single_prints": vp_daily.get("single_prints"),
        }

        # --------- ORDER BOOK / SPREAD ---------
        order_book = {
            "L1": {
                "mid": orderbook.get("mid"),
                "spread": orderbook.get("spread"),
                "bid_depth_usd": orderbook.get("bid_depth_usd"),
                "ask_depth_usd": orderbook.get("ask_depth_usd"),
                "imbalance": orderbook.get("imbalance"),
                "volume_ratio": orderbook.get("volume_ratio"),
                "pressure": orderbook.get("pressure"),
            },
            "spread_analysis": {
                "mid": spread_metrics.get("mid", orderbook.get("mid")),
                "spread": spread_metrics.get("spread", orderbook.get("spread")),
                "bid_depth_usd": spread_metrics.get(
                    "bid_depth_usd", orderbook.get("bid_depth_usd")
                ),
                "ask_depth_usd": spread_metrics.get(
                    "ask_depth_usd", orderbook.get("ask_depth_usd")
                ),
            },
        }

        # --------- ORDER FLOW / PARTICIPANTES ---------
        order_flow = fluxo.get("order_flow") or {}
        participants_block = participants

        # --------- WHALE ACTIVITY ---------
        whale = participants.get("whale") or {}
        whale_activity = {
            "whale_buy_volume": fluxo.get("whale_buy_volume"),
            "whale_sell_volume": fluxo.get("whale_sell_volume"),
            "whale_delta": fluxo.get("whale_delta"),
            "whale_direction": whale.get("direction"),
            "whale_sentiment": whale.get("sentiment"),
            "whale_activity_level": whale.get("activity_level"),
        }

        # --------- TECHNICAL INDICATORS / VOL ---------
        technical_indicators = multi_tf

        volatility_metrics = {
            "returns_1": ml_pf.get("returns_1"),
            "returns_5": ml_pf.get("returns_5"),
            "returns_15": ml_pf.get("returns_15"),
            "volatility_5": ml_pf.get("volatility_5"),
            "volatility_15": ml_pf.get("volatility_15"),
            "realized_vol": {
                tf: data.get("realized_vol")
                for tf, data in multi_tf.items()
            },
            "atr": {
                tf: data.get("atr")
                for tf, data in multi_tf.items()
            },
        }

        # --------- ABSORPTION ANALYSIS ---------
        # 1) Preferir bloco avançado vindo do FlowAnalyzer (se existir)
        absorption_analysis = fluxo.get("absorption_analysis")

        # 2) Fallback simples para eventos de Absorção legados
        if not absorption_analysis:
            tipo_evt = (signal.get("tipo_evento") or "").lower()
            if "absor" in tipo_evt or fluxo.get("tipo_absorcao"):
                absorption_analysis = {
                    "current_absorption": {
                        "classification": (
                            fluxo.get("tipo_absorcao")
                            or signal.get("tipo_absorcao")
                            or "Neutra"
                        ),
                        "delta": signal.get("delta"),
                        "volume_total": signal.get("volume_total"),
                    },
                    "absorption_zones": [],
                }

        # --------- LIQUIDITY HEATMAP ---------
        liquidity_heatmap = fluxo.get("liquidity_heatmap") or {}

        # --------- REGIME ANALYSIS ---------
        regime_analysis = {
            "volatility_regime": market_environment.get("volatility_regime"),
            "trend_direction": market_environment.get("trend_direction"),
            "market_structure": market_environment.get("market_structure"),
            "liquidity_environment": market_environment.get(
                "liquidity_environment"
            ),
            "risk_sentiment": market_environment.get("risk_sentiment"),
        }

        # Timestamps legados para visualizadores externos
        timestamp_ny = time_index.get("timestamp_ny")
        timestamp_sp = time_index.get("timestamp_sp")

        institutional_event: Dict[str, Any] = {
            "tipo_evento": signal.get("tipo_evento"),
            "resultado_da_batalha": signal.get("resultado_da_batalha"),
            "descricao": signal.get("descricao"),
            "symbol": symbol,

            # Campos na raiz para compatibilidade com visualizadores
            "janela_numero": metadata.get("window_index"),
            "epoch_ms": metadata.get("timestamp_unix_ms"),
            "timestamp_utc": metadata.get("timestamp_utc"),
            "timestamp_ny": timestamp_ny,
            "timestamp_sp": timestamp_sp,
            "data_context": data_context_value,

            "metadata": metadata,
            "data_source": data_source,
            "market_context": market_context,
            "market_environment": market_environment,
            "price_data": price_data,
            "volume_profile": volume_profile,
            "volume_nodes": volume_nodes_block,
            "order_book": order_book,
            "order_flow": order_flow,
            "participants": participants_block,
            "whale_activity": whale_activity,
            "technical_indicators": technical_indicators,
            "volatility_metrics": volatility_metrics,
            "absorption_analysis": absorption_analysis,
            "liquidity_heatmap": liquidity_heatmap,
            "support_resistance": support_resistance_block,
            "defense_zones": defense_zones_block,
            "pattern_recognition": pattern_block,
            "price_targets": price_targets_block,
            "ml_features": ml,
            "regime_analysis": regime_analysis,

            # novos blocos
            "order_book_depth": order_book_depth_block,
            "spread_analysis": spread_analysis_block,
            "market_impact": market_impact_block,

            "raw_event": signal,
        }

        return institutional_event

    # ========================================
    # PRICE TARGETS A PARTIR DE PADRÕES
    # ========================================
    def _build_price_targets(
        self,
        pattern_recognition: Dict[str, Any],
        last_price: float,
    ) -> Dict[str, Any]:
        """
        Constrói um bloco de price_targets a partir dos padrões detectados.
        Usa target_price/stop_loss/confidence de cada padrão.
        """
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

    # ========================================
    # VALIDAÇÃO DE FLOW METRICS
    # ========================================
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

    # ========================================
    # VERIFICAÇÃO DE TOQUES EM ZONAS
    # ========================================
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

                    # ✅ Marca a janela do evento de zona
                    zone_event["janela_numero"] = self.window_count

                    if "historical_confidence" not in zone_event:
                        zone_event["historical_confidence"] = (
                            calcular_probabilidade_historica(zone_event)
                        )

                    self.event_bus.publish("zone_touch", zone_event)

                    # Salva também em formato institucional
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

    # ========================================
    # ATUALIZAÇÃO DE HISTÓRICOS
    # ========================================
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

               # Histórico de volatilidade (para alertas + FlowAnalyzer)
        try:
            price_feats = ml_payload.get("price_features") or {}
            current_volatility = None

            # Volatilidade recente vinda do pipeline (ex: std de retornos 5 janelas)
            if "volatility_5" in price_feats:
                current_volatility = price_feats["volatility_5"]
            elif "volatility_1" in price_feats:
                current_volatility = price_feats["volatility_1"]

            if current_volatility is not None:
                current_volatility = float(current_volatility)
                # Mantém histórico para o módulo de alertas
                self.volatility_history.append(current_volatility)

                # 🔹 Alimenta o FlowAnalyzer com a volatilidade recente
                try:
                    last_price = float(
                        enriched.get("ohlc", {}).get("close", 0.0) or window_close
                    )
                except Exception:
                    last_price = float(window_close or 0.0)

                if last_price > 0:
                    # Converte vol (std de retorno) em unidades de preço
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

        # Histórico de OHLC por janela (para pattern_recognition)
        try:
            ohlc = enriched.get("ohlc") or {}
            if ohlc:
                self.pattern_ohlc_history.append(
                    {
                        "high": float(ohlc.get("high", 0.0)),
                        "low": float(ohlc.get("low", 0.0)),
                        "close": float(ohlc.get("close", 0.0)),
                    }
                )
        except Exception:
            pass

    # ========================================
    # LOGS
    # ========================================
    def _log_liquidity_heatmap(
        self, flow_metrics: Dict[str, Any]
    ) -> None:
        try:
            liquidity_data = flow_metrics.get("liquidity_heatmap", {})
            clusters = liquidity_data.get("clusters", [])

            if clusters:
                print(
                    f"\n📊 LIQUIDITY HEATMAP - "
                    f"Janela #{self.window_count}:"
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

                    print(
                        f"  Cluster {i+1}: ${center_fmt} | "
                        f"Vol: {vol_fmt} | "
                        f"Imb: {imb_fmt} | "
                        f"Trades: {trades_fmt} | "
                        f"Age: {age_fmt}"
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

                print(
                    f"  ML: ret5={ret5_fmt} vol5={vol5_fmt} "
                    f"V/SMA={vsma_fmt} BSpress={bs_fmt} "
                    f"OBslope={obs_fmt} FlowImb={flow_fmt}"
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
            logging.info(
                f"\n📊 HEALTH CHECK - Janela #{self.window_count}:\n"
                f"  Orderbook: failures={self.orderbook_fetch_failures}, "
                f"last_valid={last_ob_age:.0f}s ago\n"
                f"  Value Area: last_valid={last_vp_age:.0f}s ago"
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

        print(
            f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] "
            f"🟡 Janela #{self.window_count} | Delta: {delta_fmt} | Vol: {vol_fmt}"
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
                print(f"  Macro Context: {trends_str}")

        if historical_profile and historical_profile.get("daily"):
            vp = historical_profile["daily"]
            poc_fmt = format_price(vp.get("poc", 0.0))
            val_fmt = format_price(vp.get("val", 0.0))
            vah_fmt = format_price(vp.get("vah", 0.0))

            print(
                f"  VP Diário: POC @ {poc_fmt} | "
                f"VAL: {val_fmt} | VAH: {vah_fmt}"
            )

        print("─" * 80)

    # ========================================
    # ALERTAS INSTITUCIONAIS
    # ========================================
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

                    # Anexa suporte/resistência e zonas de defesa ao alerta
                    alert_event["support_resistance"] = sr
                    if dz is not None:
                        alert_event["defense_zones"] = dz

                    # ✅ Marca janela também para alertas institucionais
                    alert_event["janela_numero"] = self.window_count
                    alert_event["epoch_ms"] = int(time.time() * 1000)

                    # Salva alerta em formato institucional
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
    def on_open(self, ws: Any) -> None:
        logging.info(
            f"🚀 Bot iniciado para {self.symbol} - "
            f"Fuso: New York (America/New_York)"
        )
        try:
            self.health_monitor.heartbeat("main")
        except Exception:
            pass

    def on_close(self, ws: Any, code: int, msg: str) -> None:
        if self.window_data and not self.should_stop:
            self._process_window()

    # ========================================
    # RUN
    # ========================================
    def run(self) -> None:
        try:
            self.context_collector.start()

            logging.info(
                "🎯 Iniciando Enhanced Market Bot v2.3.2 "
                "(CORREÇÃO COMPLETA + WARMUP + BUFFER + IA FILTRADA)..."
            )
            print("═" * 80)

            self.connection_manager.connect()

        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(
                f"❌ Erro crítico ao executar o bot: {e}",
                exc_info=True,
            )
        finally:
            self._cleanup_handler()